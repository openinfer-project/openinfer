//! Host-tier KV offload glue: the coordinator's two touch points with the
//! shared pegaflow pool. Restore runs at admission (a step boundary — every
//! rank is joined, so blocking on the load is safe and the loaded pages race
//! nothing); save runs on request release, fire-and-forget, with block
//! guards pinning the pages until the D2H copy lands.
//!
//! Both legs are cache maintenance, never a correctness dependency: every
//! failure degrades to a full prefill (or a forfeited future hit) with a
//! warn, in contrast to the pool-invariant breaks around them that fail the
//! step. The launch-time contract (`Glm52LaunchOptions` validation) already
//! guarantees offload implies the prefix cache is on.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use openinfer_core::engine::GenerateRequest;
use openinfer_kv_cache::{BlockPool, KvBlockGuard, PrefixProbe, RequestKv};
use openinfer_kv_offload::{OffloadEngine, QueryOutcome, VLLM_HASH_BYTES, VllmBlockHasher};

use super::PAGE;

/// Distinguishes concurrent queries inside pegaflow's bookkeeping; nothing
/// joins on it, so a process-local counter is enough.
static QUERY_SEQ: AtomicU64 = AtomicU64::new(0);

/// One rank's offload engine plus the pages its in-flight release saves
/// still pin. The save guards hold released blocks in the active pool
/// (unallocatable, un-evictable) until the D2H copy lands — pages the
/// admission full-lifetime math would otherwise promise to a new request,
/// turning a slow copy into a mid-request allocation failure and a
/// `fail_step` engine teardown. Admission subtracts [`Self::pinned_blocks`]
/// from the rank's usable count instead, degrading to "admit a few steps
/// later" — the same honor-or-reject posture as the rest of the scheduler.
pub(super) struct RankOffload {
    pub(super) engine: OffloadEngine,
    pinned: Arc<AtomicUsize>,
    /// `false` in vLLM-compat P/D mode: the content domain is keyed with
    /// vLLM's hash scheme, so this node's kvbm-keyed self-saves would be
    /// unfindable there (and multi-turn reuse doesn't need them — the peer
    /// re-registers the full history each turn).
    save_enabled: bool,
}

/// Keep-alive payload for one release save: the block guards plus the
/// pinned-page accounting. Dropped by the offload engine exactly when the
/// D2H copy lands (or on any early-error path), releasing both the pins and
/// the count together.
struct SavePin {
    _guards: Vec<KvBlockGuard>,
    pinned: Arc<AtomicUsize>,
    blocks: usize,
}

impl Drop for SavePin {
    fn drop(&mut self) {
        self.pinned.fetch_sub(self.blocks, Ordering::Release);
    }
}

impl RankOffload {
    pub(super) fn new(engine: OffloadEngine, save_enabled: bool) -> Self {
        Self {
            engine,
            pinned: Arc::new(AtomicUsize::new(0)),
            save_enabled,
        }
    }

    /// Pool pages currently pinned by in-flight release saves.
    pub(super) fn pinned_blocks(&self) -> usize {
        self.pinned.load(Ordering::Acquire)
    }

    /// Send the request's freshly-sealed blocks to the host tier before its
    /// pool pages release. Skips the prefix-matched head — those blocks were
    /// restored from the host tier or saved when their producing request
    /// released, so they are already resident there. Fire-and-forget: the
    /// [`SavePin`] keeps the pages pinned (and counted) until the D2H copy
    /// lands, and the last step that wrote them has already joined, so the
    /// bytes are final.
    pub(super) fn save_sealed_on_release(&self, kv: &RequestKv) {
        if !self.save_enabled {
            return;
        }
        let sealed = kv.assigned_block_hashes();
        let matched = kv.prefix_matched_blocks();
        if sealed.len() <= matched {
            return;
        }
        let fresh = &sealed[matched..];
        let block_ids: Vec<i32> = fresh.iter().map(|(id, _)| *id).collect();
        let block_hashes: Vec<Vec<u8>> = fresh.iter().map(|(_, hash)| hash.to_vec()).collect();
        let mut guards = kv.assigned_block_guards();
        let guards = guards.split_off(matched);
        self.pinned.fetch_add(guards.len(), Ordering::Release);
        let pin = SavePin {
            blocks: guards.len(),
            _guards: guards,
            pinned: Arc::clone(&self.pinned),
        };
        self.engine.save(&block_ids, &block_hashes, pin);
    }
}

/// Restore the prompt-prefix blocks the GPU cache no longer holds from the
/// host tier: probe → query → load into reserved pool pages → commit as
/// matchable prefix. Blocks on the load — admission is a step boundary, and
/// the request's first prefill chunk must not read half-restored pages.
///
/// Returns the probe, which holds the GPU-hit and freshly-committed blocks
/// alive; the caller keeps it across `match_and_add_prefix` so the restored
/// prefix cannot be evicted before it is re-matched.
/// vLLM-compat P/D miss breaker: after this many consecutive requests each
/// exhausted the whole zero-hit wait window, new requests skip the wait (the
/// prefill peer is evidently not publishing — misconfig or down) and reject
/// immediately, so the router fails over fast. Any complete restore re-arms.
const MISS_BREAKER_THRESHOLD: u32 = 3;

/// Hard ceiling on one request's remote-KV wait, covering an in-flight P2P
/// fetch (`QueryOutcome::Loading`). Well above pegaflow's own fetch timeout.
const REMOTE_FETCH_DEADLINE: Duration = Duration::from_secs(15);

/// Decode-node admission state for a vLLM prefill peer (one per coordinator;
/// see `crate::Glm52VllmCompatOptions` for the deployment contract). Tracks
/// each rank's parked front request — only the FIFO front can be waiting on
/// remote KV — and the cross-rank miss breaker.
pub(super) struct VllmPdState {
    hasher: VllmBlockHasher,
    miss_wait: Duration,
    allow_local_prefill: bool,
    /// Requests in a row that exhausted their whole wait window. At
    /// [`MISS_BREAKER_THRESHOLD`] new requests stop parking (first-shot
    /// queries still run; a complete restore resets this).
    consecutive_miss_windows: u32,
    parked: Vec<Option<ParkedFront>>,
}

/// The rank's front request currently waiting out the P/D handoff race.
struct ParkedFront {
    /// Re-identifies the front across admission retries (a rejected or
    /// disconnected front resets the deadlines for its successor).
    fingerprint: (Option<String>, usize),
    /// Stable pegaflow query id: an in-flight P2P fetch is polled by
    /// re-querying under the SAME id each retry.
    query_key: String,
    parked_at: Instant,
    /// Zero/partial-hit window: the producer's save + registration tail.
    miss_deadline: Instant,
    /// In-flight-fetch window (`Loading` seen): the transfer itself.
    hard_deadline: Instant,
    saw_loading: bool,
}

/// One admission attempt's verdict for the rank's front request.
pub(super) enum VllmAdmitOutcome {
    /// All peer-prefilled positions restored; exactly one token (the router-
    /// appended first generated token) remains to forward.
    Admit { kv: RequestKv, cached_tokens: usize },
    /// Remote KV not fully visible yet — leave the request at the queue
    /// front and retry at the next step boundary.
    Park,
    /// The wait window closed (or the local engine errored) and local
    /// prefill is forbidden: fail the request so the router retries it
    /// through the prefill peer.
    Reject { message: String },
    /// Same condition with `allow_local_prefill`: the caller runs the plain
    /// (non-compat) admission path for this request instead.
    LocalFallback,
}

/// How one restore attempt fell short of the full peer-prefilled prefix.
enum Shortfall {
    /// Registration race or in-flight fetch — worth waiting for.
    Racing,
    /// Local engine error (query/load RPC failed) — waiting won't heal it.
    Broken(String),
}

impl VllmPdState {
    pub(super) fn new(opts: &crate::Glm52VllmCompatOptions, ranks: usize) -> Self {
        let hasher = VllmBlockHasher::new(&opts.python_hash_seed, PAGE);
        // Cross-engine fingerprint: every P/D mismatch (seed, namespace,
        // block size, geometry) otherwise presents as nothing but rejected
        // requests — this line is what an operator diffs against the vLLM
        // peer's startup config.
        log::info!(
            "GLM5.2 vLLM-compat P/D active: seed={} namespace={} block_size={PAGE} \
             none_hash={} miss_wait={:?} allow_local_prefill={}",
            opts.python_hash_seed,
            opts.namespace,
            hasher
                .none_hash()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            opts.miss_wait,
            opts.allow_local_prefill,
        );
        Self {
            hasher,
            miss_wait: opts.miss_wait,
            allow_local_prefill: opts.allow_local_prefill,
            consecutive_miss_windows: 0,
            parked: (0..ranks).map(|_| None).collect(),
        }
    }

    /// The front request's parked state, resetting it when the front changed
    /// since the last attempt (rejection, disconnect, or first sighting).
    fn parked_front(&mut self, rank: usize, req: &GenerateRequest) -> &mut ParkedFront {
        let fingerprint = (req.request_id.clone(), req.prompt_tokens.len());
        let stale = self.parked[rank]
            .as_ref()
            .is_none_or(|parked| parked.fingerprint != fingerprint);
        if stale {
            let now = Instant::now();
            let query_key = req.request_id.clone().unwrap_or_else(|| {
                format!("glm52-pd-{}", QUERY_SEQ.fetch_add(1, Ordering::Relaxed))
            });
            self.parked[rank] = Some(ParkedFront {
                fingerprint,
                query_key,
                parked_at: now,
                miss_deadline: now + self.miss_wait,
                hard_deadline: now + REMOTE_FETCH_DEADLINE,
                saw_loading: false,
            });
        }
        self.parked[rank].as_mut().expect("just ensured")
    }

    pub(super) fn clear_parked(&mut self, rank: usize) {
        self.parked[rank] = None;
    }

    /// True while any rank's front request is parked on the handoff race —
    /// the idle coordinator loop throttles instead of spinning.
    pub(super) fn any_parked(&self) -> bool {
        self.parked.iter().any(Option::is_some)
    }
}

/// vLLM-compat P/D admission for one rank's front request. The router
/// appended the prefill peer's first generated token to the prompt, so the
/// peer's registered KV covers every prompt position except that last token:
/// all full 64-token pages under vLLM's own block hashes, plus the partial
/// tail page under the P-side connector extension's derived tail key. A
/// complete restore leaves a one-token forward — a decode-shaped step — and
/// zero prompt-position compute on this node.
///
/// `Err` is a kvbm invariant break (engine-fatal), mirroring the plain path.
pub(super) fn admit_vllm_pd(
    state: &mut VllmPdState,
    rank: usize,
    offload: &RankOffload,
    pool: &BlockPool,
    req: &GenerateRequest,
) -> anyhow::Result<VllmAdmitOutcome> {
    let prompt = &req.prompt_tokens;
    // Positions the peer prefilled: everything but the router-appended token.
    let prompt_kv = &prompt[..prompt.len() - 1];
    let full_blocks = prompt_kv.len() / PAGE;
    let tail_len = prompt_kv.len() % PAGE;
    let breaker_open = state.consecutive_miss_windows >= MISS_BREAKER_THRESHOLD;
    let query_key = state.parked_front(rank, req).query_key.clone();

    let chain = state.hasher.key_chain(prompt_kv);
    debug_assert_eq!(chain.len(), full_blocks);
    let mut kv = pool.new_request(prompt.clone(), req.max_tokens, None);
    let mut probe = pool.probe_prefix(prompt.clone(), None);
    let gpu_hit = probe.gpu_hit_blocks();
    let window = probe.cpu_query_window();
    // The one-token surplus makes the probe's reuse cap land exactly on the
    // peer-prefilled full blocks: cacheable = (len(prompt)-1)/PAGE = chain.
    debug_assert_eq!(gpu_hit + window, chain.len());

    let mut shortfall: Option<Shortfall> = None;
    let mut saw_loading = false;

    // Full pages: query the [gpu_hit .. chain) window under vLLM keys and
    // restore into pool pages as matchable prefix (same leg as the plain
    // host-tier restore, different key scheme).
    if window > 0 {
        let keys = &chain[gpu_hit..gpu_hit + window];
        match offload.engine.query(&query_key, keys) {
            Ok(QueryOutcome::Ready(hit)) => match hit.lease {
                Some(lease) if hit.num_blocks == window => {
                    match pool.reserve_loaded_blocks(hit.num_blocks) {
                        Some(reservation) => match offload.engine.load(lease, reservation.page_ids())
                        {
                            Ok(handle) => match handle.wait() {
                                Ok(()) => pool.commit_loaded_blocks(&mut probe, reservation),
                                Err(err) => {
                                    shortfall =
                                        Some(Shortfall::Broken(format!("remote KV load: {err}")));
                                }
                            },
                            Err(err) => {
                                offload.engine.release_query_lease(lease);
                                shortfall =
                                    Some(Shortfall::Broken(format!("remote KV load submit: {err}")));
                            }
                        },
                        None => {
                            // Pool pressure: in-flight release saves free pages
                            // within a few steps — a wait, not a failure.
                            offload.engine.release_query_lease(lease);
                            shortfall = Some(Shortfall::Racing);
                        }
                    }
                }
                Some(lease) => {
                    // Partial hit: the peer's registrations are still landing.
                    // GLM admits only on the complete prefix, so don't consume
                    // a partial lease — release and re-query.
                    offload.engine.release_query_lease(lease);
                    shortfall = Some(Shortfall::Racing);
                }
                None => shortfall = Some(Shortfall::Racing),
            },
            Ok(QueryOutcome::Loading) => {
                saw_loading = true;
                shortfall = Some(Shortfall::Racing);
            }
            Err(err) => shortfall = Some(Shortfall::Broken(format!("remote KV query: {err}"))),
        }
    }

    let mut cached_tokens = kv.match_and_add_prefix(pool)?;
    if shortfall.is_none() && cached_tokens < chain.len() * PAGE {
        // Committed blocks failed to re-match — an eviction race the probe
        // hold is supposed to prevent; retry rather than reject.
        shortfall = Some(Shortfall::Racing);
    }

    // Tail page: the peer-prefilled positions past the last full block,
    // saved by the P-side connector extension under a key both sides derive
    // (`hash_block(last_full_hash, tail_tokens)` — vLLM itself never hashes
    // partial blocks). Loaded into the request's OWN scheduled page — never
    // the radix: a partial page must not be matchable by other requests.
    if shortfall.is_none() && tail_len > 0 {
        let parent: Option<[u8; VLLM_HASH_BYTES]> = chain
            .last()
            .map(|key| key.as_slice().try_into().expect("vLLM keys are 16 bytes"));
        let tail_key = state
            .hasher
            .hash_block(parent.as_ref(), &prompt_kv[full_blocks * PAGE..])
            .to_vec();
        match offload
            .engine
            .query(&format!("{query_key}-tail"), &[tail_key])
        {
            Ok(QueryOutcome::Ready(hit)) => match hit.lease {
                Some(lease) => match kv.schedule_prefill(tail_len, pool) {
                    Ok(()) => {
                        // step_page_indices covers the whole sequence up to the
                        // step end; the restored full blocks occupy all but the
                        // last entry, and the tail page is that last entry (the
                        // restore left kv_position block-aligned, so tail_len
                        // tokens open exactly one fresh page).
                        let pages = kv.step_page_indices(tail_len);
                        let tail_page = *pages.last().expect("tail step has a page");
                        match offload.engine.load(lease, vec![tail_page]) {
                            Ok(handle) => match handle.wait() {
                                Ok(()) => {
                                    kv.apply_prefill_chunk(pool)?;
                                    cached_tokens += tail_len;
                                }
                                Err(err) => {
                                    kv.revert_schedule()?;
                                    shortfall =
                                        Some(Shortfall::Broken(format!("tail KV load: {err}")));
                                }
                            },
                            Err(err) => {
                                offload.engine.release_query_lease(lease);
                                kv.revert_schedule()?;
                                shortfall =
                                    Some(Shortfall::Broken(format!("tail KV load submit: {err}")));
                            }
                        }
                    }
                    Err(err) => {
                        offload.engine.release_query_lease(lease);
                        log::debug!("GLM5.2 P/D tail page allocation deferred: {err:?}");
                        shortfall = Some(Shortfall::Racing);
                    }
                },
                None => shortfall = Some(Shortfall::Racing),
            },
            Ok(QueryOutcome::Loading) => {
                saw_loading = true;
                shortfall = Some(Shortfall::Racing);
            }
            Err(err) => shortfall = Some(Shortfall::Broken(format!("tail KV query: {err}"))),
        }
    }

    let suffix = prompt.len() - kv.kv_position();
    if suffix == 1 {
        let parked_for = state.parked[rank]
            .as_ref()
            .map_or(Duration::ZERO, |parked| parked.parked_at.elapsed());
        state.clear_parked(rank);
        state.consecutive_miss_windows = 0;
        log::info!(
            "GLM5.2 P/D admit rank{rank}: prompt={} cached={cached_tokens} suffix=1 \
             (gpu_hit={gpu_hit} pulled={window} tail={tail_len}, parked {parked_for:?})",
            prompt.len(),
        );
        return Ok(VllmAdmitOutcome::Admit { kv, cached_tokens });
    }
    drop(kv); // release matched/loaded holdings before parking or rejecting

    let parked = state.parked[rank].as_mut().expect("front is parked");
    parked.saw_loading |= saw_loading;
    let (deadline, phase) = if parked.saw_loading {
        (parked.hard_deadline, "in-flight fetch")
    } else {
        (parked.miss_deadline, "registration")
    };
    match shortfall {
        Some(Shortfall::Broken(reason)) => {
            state.clear_parked(rank);
            Ok(fail_or_fallback(
                state,
                format!("GLM5.2 P/D remote KV unavailable ({reason}); retry via the prefill peer"),
            ))
        }
        _ if breaker_open || Instant::now() >= deadline => {
            let waited = parked.parked_at.elapsed();
            state.clear_parked(rank);
            state.consecutive_miss_windows = state.consecutive_miss_windows.saturating_add(1);
            if state.consecutive_miss_windows == MISS_BREAKER_THRESHOLD {
                log::warn!(
                    "GLM5.2 P/D miss breaker open: {MISS_BREAKER_THRESHOLD} consecutive requests \
                     exhausted the remote-KV wait window; new requests now reject immediately \
                     until a complete restore lands"
                );
            }
            Ok(fail_or_fallback(
                state,
                format!(
                    "GLM5.2 P/D remote KV incomplete after {waited:?} ({phase} window, \
                     cached {}/{} tokens); this decode node refuses local prefill — \
                     retry via the prefill peer (check P/D seed/namespace/block-size alignment)",
                    cached_tokens,
                    prompt.len() - 1,
                ),
            ))
        }
        _ => Ok(VllmAdmitOutcome::Park),
    }
}

/// Strict mode rejects (the router retries through the prefill peer); the
/// `allow_local_prefill` debug mode falls back to the plain admission path.
fn fail_or_fallback(state: &VllmPdState, message: String) -> VllmAdmitOutcome {
    if state.allow_local_prefill {
        log::warn!("{message} — admitting with LOCAL prompt compute (allow_local_prefill)");
        VllmAdmitOutcome::LocalFallback
    } else {
        log::warn!("{message}");
        VllmAdmitOutcome::Reject { message }
    }
}

pub(super) fn restore_host_prefix(
    engine: &OffloadEngine,
    pool: &BlockPool,
    prompt_tokens: &[u32],
) -> PrefixProbe {
    let mut probe = pool.probe_prefix(prompt_tokens.to_vec(), None);
    let hashes = probe.cpu_query_hashes();
    if hashes.is_empty() {
        return probe;
    }
    let req_key = format!("glm52-admit-{}", QUERY_SEQ.fetch_add(1, Ordering::Relaxed));
    let hit = match engine.query(&req_key, &hashes) {
        Ok(QueryOutcome::Ready(hit)) => hit,
        Ok(QueryOutcome::Loading) => {
            // Host-memory-only setup: pegaflow has no deeper tier to fetch
            // from, so an async outcome means a config drift worth seeing.
            log::warn!("GLM5.2 host-tier query went async in a host-only setup; skipping restore");
            return probe;
        }
        Err(err) => {
            log::warn!("GLM5.2 host-tier query failed (prefill from scratch): {err}");
            return probe;
        }
    };
    let Some(lease) = hit.lease else {
        return probe;
    };
    let Some(reservation) = pool.reserve_loaded_blocks(hit.num_blocks) else {
        // Block pressure: the pool cannot hold the restored prefix right
        // now. Prefill recomputes it — correct, just colder.
        engine.release_query_lease(lease);
        return probe;
    };
    let page_ids = reservation.page_ids();
    let restored = reservation.len();
    match engine.load(lease, page_ids) {
        Ok(handle) => match handle.wait() {
            Ok(()) => {
                pool.commit_loaded_blocks(&mut probe, reservation);
                // The only signal separating a host-tier restore from a plain
                // GPU prefix hit — the parity/eviction gates key on it.
                log::info!("GLM5.2 host-tier restore: {restored} blocks committed");
            }
            Err(err) => {
                log::warn!("GLM5.2 host-tier load failed (prefill from scratch): {err}");
            }
        },
        Err(err) => {
            log::warn!("GLM5.2 host-tier load submit failed (prefill from scratch): {err}");
            // `load` consumes the lease only past its early validation; a
            // submit error may leave it pinning the host blocks until the
            // lease TTL. Release explicitly (no-op if already consumed).
            engine.release_query_lease(lease);
        }
    }
    probe
}
