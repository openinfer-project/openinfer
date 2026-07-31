//! Host-tier KV offload glue: the engine's two touch points with the
//! shared pegaflow pool. Restore SUBMITS its host→GPU load at admission and
//! polls the [`LoadHandle`] at the following step boundaries — the engine
//! loop never blocks on the copy, because under free-running DP one rank
//! stalled in admission stalls every peer in the fixed-cadence collective
//! chain (#799). The waiting request stays at its queue front and the
//! in-flight load's page holds ride the parked state ([`HostRestoreState`],
//! [`NativePdState`]). Save runs on request release, fire-and-forget, with
//! block guards pinning the pages until the D2H copy lands.
//!
//! Both legs are cache maintenance, never a correctness dependency: every
//! failure degrades to a full prefill (or a forfeited future hit) with a
//! warn, in contrast to the pool-invariant breaks around them that fail the
//! step. The launch-time contract (`Glm52LaunchOptions` validation) already
//! guarantees offload implies the prefix cache is on.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context as _;
use openinfer_core::engine::GenerateRequest;
use openinfer_kv_cache::BlockPool;
use openinfer_kv_cache::KvBlockGuard;
use openinfer_kv_cache::LoadReservation;
use openinfer_kv_cache::PrefixProbe;
use openinfer_kv_cache::RequestKv;
use openinfer_kv_offload::LoadHandle;
use openinfer_kv_offload::OffloadEngine;
use openinfer_kv_offload::QueryOutcome;
use openinfer_kv_offload::VLLM_HASH_BYTES;
use openinfer_kv_offload::VllmBlockHasher;
use serde::Deserialize;

use super::PAGE;

/// Distinguishes concurrent queries inside pegaflow's bookkeeping; nothing
/// joins on it, so a process-local counter is enough.
static QUERY_SEQ: AtomicU64 = AtomicU64::new(0);

/// One rank's offload engine plus the pages its in-flight release saves
/// still pin. The save guards hold released blocks in the active pool
/// (unallocatable, un-evictable) until the D2H copy lands — pages the
/// admission full-lifetime math would otherwise promise to a new request,
/// turning a slow copy into a mid-request allocation failure and a fatal
/// engine exit. Admission subtracts [`Self::pinned_blocks`]
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

    /// Persist the mutable last page under the native-P/D handoff key.
    /// kvbm does not expose an immutable guard or lineage hash until a page
    /// seals, so capture this page synchronously while the request still owns
    /// it. The response may race cache visibility; D admission already parks
    /// and retries until PegaFlow publishes the key.
    pub(super) fn save_native_tail(&self, kv: &RequestKv, key: [u8; 16]) -> anyhow::Result<()> {
        let page_id = kv
            .current_page_indices()
            .last()
            .copied()
            .context("native P/D tail has no committed physical page")?;
        self.engine
            .save_blocking(&[page_id], &[key.to_vec()])
            .map_err(|err| anyhow::anyhow!("native P/D tail save: {err}"))
    }
}

/// vLLM-compat P/D miss breaker: after this many consecutive requests each
/// exhausted the whole zero-hit wait window, new requests park with the short
/// [`BREAKER_PROBE_WINDOW`] instead of the full miss window (the prefill peer
/// is evidently not publishing — misconfig or down), so the router fails over
/// fast. Any complete restore re-arms.
///
/// The probe window must stay wide enough for a healthy handoff to complete:
/// pegaflow's `query` only STARTS an async metaserver resolve + fetch and
/// reports a miss until it lands (~50 ms measured), so rejecting on the first
/// shot would starve every remote restore and the breaker could never close.
const MISS_BREAKER_THRESHOLD: u32 = 3;

/// Wait window while the breaker is open, replacing BOTH the miss and the
/// in-flight-fetch deadlines (a first-shot query already reports `Loading`,
/// so the miss window alone would never bind). Covers the P-side save
/// visibility pipeline (~46 ms measured) plus the async fetch of a healthy
/// peer, while a still-down peer drains its queue at probe cadence instead
/// of one full fetch window per request.
const BREAKER_PROBE_WINDOW: Duration = Duration::from_millis(500);

/// Hard ceiling on one request's remote-KV wait, covering an in-flight P2P
/// fetch (`QueryOutcome::Loading`). Well above pegaflow's own fetch timeout.
pub(crate) const REMOTE_FETCH_DEADLINE: Duration = Duration::from_secs(15);

/// Decode-node admission state for a vLLM prefill peer (one per engine;
/// see `crate::Glm52VllmCompatOptions` for the deployment contract). Tracks
/// each rank's parked front request — only the FIFO front can be waiting on
/// remote KV — and the cross-rank miss breaker.
pub(super) struct VllmPdState {
    hasher: VllmBlockHasher,
    miss_wait: Duration,
    allow_local_prefill: bool,
    /// Requests in a row that exhausted their whole wait window. At
    /// [`MISS_BREAKER_THRESHOLD`] new requests park with the short
    /// [`BREAKER_PROBE_WINDOW`] instead (a complete restore resets this).
    consecutive_miss_windows: u32,
    parked: Vec<Option<ParkedFront>>,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct NativeMtpHandoff {
    pub(super) draft_tokens: [u32; crate::mtp::GLM52_MTP_DRAFTS],
    pub(super) committed_len: usize,
    pub(super) arena_count: usize,
    pub(super) tail_len: usize,
    pub(super) tail_key: Option<String>,
    pub(super) anchor_token_id: u32,
    /// Whether the anchor is client-visible; false only when it is an EOS
    /// consumed by P but suppressed from the response, so D must not replay it.
    pub(super) anchor_emitted: bool,
}

#[derive(Deserialize)]
struct NativeMtpEnvelope {
    version: u32,
    native_mtp: NativeMtpHandoff,
}

#[derive(Deserialize)]
struct OpenInferPdEnvelope {
    openinfer_pd: NativeMtpEnvelope,
}

pub(super) fn native_mtp_handoff(
    req: &GenerateRequest,
) -> anyhow::Result<Option<NativeMtpHandoff>> {
    let Some(value) = req.kv_transfer_params.clone() else {
        return Ok(None);
    };
    if value.get("openinfer_pd").is_none() {
        return Ok(None);
    }
    let envelope: OpenInferPdEnvelope =
        serde_json::from_value(value).context("invalid openinfer native-MTP P/D metadata")?;
    let version = envelope.openinfer_pd.version;
    anyhow::ensure!(
        version == 2,
        "unsupported openinfer P/D metadata version {version}"
    );
    let handoff = envelope.openinfer_pd.native_mtp;
    anyhow::ensure!(
        handoff.arena_count == 101,
        "native-MTP P/D requires 101 arenas, got {}",
        handoff.arena_count
    );
    Ok(Some(handoff))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeAnchorPlan {
    pub(super) token: u32,
    pub(super) replay_to_client: bool,
    pub(super) emitted_by_prefill: bool,
}

/// Resolve the v2 router contract without mutating the queued request.
///
/// vLLM Router forwards the original request unchanged, so D must append and
/// replay P's anchor. A manual v2 harness may already append the anchor and
/// combine P+D output itself; keep accepting that shape without replay.
pub(super) fn native_anchor_plan(
    req: &GenerateRequest,
    handoff: &NativeMtpHandoff,
) -> anyhow::Result<NativeAnchorPlan> {
    let token = handoff.anchor_token_id;
    let emitted_by_prefill = handoff.anchor_emitted;
    if req.prompt_tokens.len() == handoff.committed_len {
        return Ok(NativeAnchorPlan {
            token,
            replay_to_client: true,
            emitted_by_prefill,
        });
    }
    anyhow::ensure!(
        req.prompt_tokens.len() == handoff.committed_len + 1
            && req.prompt_tokens.last() == Some(&token),
        "native-MTP P/D v2 expects the original prompt or committed KV + anchor"
    );
    Ok(NativeAnchorPlan {
        token,
        replay_to_client: false,
        emitted_by_prefill,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeKvShape {
    pub(super) input_tokens: usize,
    pub(super) max_output_tokens: usize,
}

/// Exact kvbm geometry of a native-MTP D request. Router handoffs append P's
/// anchor to the logical input; manual handoffs already carry it and instead
/// need one extra internal output position because that anchor did not consume
/// the D request's client-visible output budget.
pub(super) fn native_kv_shape(
    req: &GenerateRequest,
    anchor_plan: NativeAnchorPlan,
) -> anyhow::Result<NativeKvShape> {
    let input_tokens = req
        .prompt_tokens
        .len()
        .checked_add(usize::from(anchor_plan.replay_to_client))
        .context("native-MTP P/D KV input length overflow")?;
    let anchor_counts_against_client_budget =
        anchor_plan.replay_to_client && anchor_plan.emitted_by_prefill;
    let max_output_tokens = req
        .max_tokens
        .checked_add(usize::from(!anchor_counts_against_client_budget))
        .context("native-MTP P/D KV output budget overflow")?;
    Ok(NativeKvShape {
        input_tokens,
        max_output_tokens,
    })
}

/// A load whose admission owner is gone (disconnect, replaced front, or an
/// exhausted deadline). pegaflow's worker still writes the destination pages
/// after the handle is dropped, so the page holds ride here until the DMA
/// settles; [`reap`](HostRestoreState::reap) drops the settled ones.
struct AbandonedLoad {
    handle: LoadHandle,
    _hold: AbandonedHold,
}

/// What keeps an abandoned load's destination pages unallocatable. Never
/// read — held purely for its `Drop`; holding the value IS the contract.
#[allow(dead_code)]
enum AbandonedHold {
    /// Radix-bound restore: the reserved destination blocks.
    Pages(LoadReservation),
    /// Tail load: the whole request KV owns the scheduled destination page.
    Request(Box<RequestKv>),
}

impl AbandonedLoad {
    /// `true` while the DMA is still in flight (the hold must stay).
    fn live(&mut self) -> bool {
        self.handle.poll().is_none()
    }
}

pub(super) struct NativePdState {
    parked: Vec<Option<NativeParked>>,
    abandoned: Vec<AbandonedLoad>,
}

struct NativeParked {
    request_id: Option<String>,
    query_key: String,
    parked_at: Instant,
    deadline: Instant,
    /// This front's in-flight H2D load, polled on the next admission attempt.
    pending: Option<NativePendingLoad>,
}

/// An admission-submitted native-P/D load in flight across step boundaries.
/// Everything the commit needs — and everything pinning the DMA destination —
/// lives here while the front is parked.
enum NativePendingLoad {
    /// Full committed pages, destined for the radix on completion.
    Full {
        probe: PrefixProbe,
        reservation: LoadReservation,
        handle: LoadHandle,
    },
    /// The mutable tail page, scheduled on the request's own KV.
    Tail {
        kv: Box<RequestKv>,
        cached_tokens: usize,
        handle: LoadHandle,
    },
}

impl NativePendingLoad {
    /// Pool pages this load holds for the parked front. They become the
    /// request's blocks on admit, so the admission budget credits them back
    /// against the request's need (they are already out of
    /// `available_blocks`, and counting them on both sides would wedge the
    /// front forever).
    fn held_blocks(&self) -> usize {
        match self {
            Self::Full {
                probe, reservation, ..
            } => probe.held_blocks() + reservation.len(),
            Self::Tail { kv, .. } => kv.resident_blocks(),
        }
    }

    fn into_abandoned(self) -> AbandonedLoad {
        match self {
            Self::Full {
                probe,
                reservation,
                handle,
            } => {
                drop(probe); // source holds; the DMA only writes the reservation
                AbandonedLoad {
                    handle,
                    _hold: AbandonedHold::Pages(reservation),
                }
            }
            Self::Tail { kv, handle, .. } => AbandonedLoad {
                handle,
                _hold: AbandonedHold::Request(kv),
            },
        }
    }
}

/// Keep the final committed page out of the lineage-hashed full-page prefix.
/// KV block hashes require a dangling token after a sealed page, so an
/// exactly page-aligned context still has a PAGE-token explicit tail.
pub(super) fn native_pd_tail_len(committed_len: usize) -> usize {
    committed_len
        .checked_sub(1)
        .map_or(0, |last| last % PAGE + 1)
}

fn native_pd_needs_tail_load(cached_tokens: usize, committed_len: usize, tail_len: usize) -> bool {
    tail_len > 0 && cached_tokens < committed_len
}

impl NativePdState {
    pub(super) fn new(ranks: usize) -> Self {
        Self {
            parked: (0..ranks).map(|_| None).collect(),
            abandoned: Vec::new(),
        }
    }

    fn parked(&mut self, rank: usize, req: &GenerateRequest) -> &mut NativeParked {
        let stale = self.parked[rank]
            .as_ref()
            .is_none_or(|parked| parked.request_id != req.request_id);
        if stale {
            self.scrap_pending(rank);
            let now = Instant::now();
            self.parked[rank] = Some(NativeParked {
                request_id: req.request_id.clone(),
                query_key: format!(
                    "glm52-native-pd-{}",
                    QUERY_SEQ.fetch_add(1, Ordering::Relaxed)
                ),
                parked_at: now,
                deadline: now + REMOTE_FETCH_DEADLINE,
                pending: None,
            });
        }
        self.parked[rank].as_mut().expect("just initialized")
    }

    /// Take the front's in-flight load for this attempt to settle or repark.
    fn take_pending(&mut self, rank: usize) -> Option<NativePendingLoad> {
        self.parked[rank].as_mut().and_then(|p| p.pending.take())
    }

    /// Move the rank's in-flight load (if any) onto the abandoned list — the
    /// DMA still writes its destination pages, so they cannot release yet.
    fn scrap_pending(&mut self, rank: usize) {
        if let Some(pending) = self.take_pending(rank) {
            self.abandoned.push(pending.into_abandoned());
        }
    }

    /// Pool pages held by the rank's parked front for its in-flight load
    /// (zero when nothing is parked). See [`NativePendingLoad::held_blocks`].
    pub(super) fn front_held_blocks(&self, rank: usize) -> usize {
        self.parked[rank]
            .as_ref()
            .and_then(|p| p.pending.as_ref())
            .map_or(0, NativePendingLoad::held_blocks)
    }

    /// Drop abandoned loads whose DMA settled (their page holds release).
    pub(super) fn reap(&mut self) {
        self.abandoned.retain_mut(AbandonedLoad::live);
    }

    pub(super) fn clear(&mut self, rank: usize) {
        self.scrap_pending(rank);
        self.parked[rank] = None;
    }
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
    Admit {
        kv: Box<RequestKv>,
        cached_tokens: usize,
    },
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
             none_hash={:032x} miss_wait={:?} allow_local_prefill={}",
            opts.python_hash_seed,
            opts.namespace,
            u128::from_be_bytes(hasher.none_hash()),
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
            // Never the client-supplied request_id: pegaflow keys prefetch
            // state and its failed-remote blacklist by this id, so a
            // duplicate or reused external id would cross-consume another
            // request's fetch or inherit its 5-minute blacklist entry.
            let query_key = format!("glm52-pd-{}", QUERY_SEQ.fetch_add(1, Ordering::Relaxed));
            let (miss_wait, fetch_wait) = if self.consecutive_miss_windows >= MISS_BREAKER_THRESHOLD
            {
                (BREAKER_PROBE_WINDOW, BREAKER_PROBE_WINDOW)
            } else {
                (self.miss_wait, REMOTE_FETCH_DEADLINE)
            };
            self.parked[rank] = Some(ParkedFront {
                fingerprint,
                query_key,
                parked_at: now,
                miss_deadline: now + miss_wait,
                hard_deadline: now + fetch_wait,
                saw_loading: false,
            });
        }
        self.parked[rank].as_mut().expect("just ensured")
    }

    pub(super) fn clear_parked(&mut self, rank: usize) {
        self.parked[rank] = None;
    }
}

/// Deinterleave the RoPE dims of freshly-restored pages on their owning rank.
/// Blocking, but called only at step boundaries where its command queue is
/// idle. P/D is restricted to EP, so each arena has exactly one executor.
fn vllm_rope_fixup(worker: &crate::runner::Glm52Worker, pages: &[i32]) -> anyhow::Result<()> {
    worker
        .vllm_rope_fixup(pages.to_vec())
        .context("restored page rope fixup")
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
    fixup_worker: &crate::runner::Glm52Worker,
) -> anyhow::Result<VllmAdmitOutcome> {
    let prompt = &req.prompt_tokens;
    // Positions the peer prefilled: everything but the router-appended token.
    let prompt_kv = &prompt[..prompt.len() - 1];
    let full_blocks = prompt_kv.len() / PAGE;
    let tail_len = prompt_kv.len() % PAGE;
    let query_key = state.parked_front(rank, req).query_key.clone();

    let chain = state.hasher.key_chain(prompt_kv);
    debug_assert_eq!(chain.len(), full_blocks);
    let mut kv = pool.new_request(prompt.clone(), req.max_tokens, None);
    let mut probe = pool.probe_prefix(prompt.clone(), None);
    let gpu_hit = probe.gpu_hit_blocks();
    let window = probe.cpu_query_window();
    // The one-token surplus makes the probe's reuse cap land on the
    // peer-prefilled full blocks (cacheable = (len(prompt)-1)/PAGE = chain),
    // EXCEPT when the radix already holds the block containing the surplus
    // token (a retried block-aligned prompt): gpu_hit then overshoots
    // cacheable by one and the probe reports an empty query window — the
    // same tolerated state as the plain path's gpu_hit >= cacheable guard.

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
                    // A full-window metadata hit proves the peer IS
                    // publishing — close the breaker now, before the load,
                    // so a restore that outlives one probe window parks
                    // with full deadlines on its next attempt instead of
                    // feeding the breaker forever.
                    state.consecutive_miss_windows = 0;
                    if let Some(reservation) = pool.reserve_loaded_blocks(hit.num_blocks) {
                        match offload.engine.load(lease, reservation.page_ids()) {
                            Ok(handle) => {
                                // After the H2D lands, rewrite the pages'
                                // RoPE dims from the peer's interleaved
                                // placement to openinfer's block-out one —
                                // before they become matchable (exactly-once:
                                // radix hits skip this whole leg).
                                let landed = handle
                                    .wait()
                                    .map_err(|err| anyhow::anyhow!("remote KV load: {err}"))
                                    .and_then(|()| {
                                        vllm_rope_fixup(fixup_worker, &reservation.page_ids())
                                    });
                                match landed {
                                    Ok(()) => pool.commit_loaded_blocks(&mut probe, reservation),
                                    Err(err) => {
                                        shortfall = Some(Shortfall::Broken(format!("{err:#}")));
                                    }
                                }
                            }
                            Err(err) => {
                                offload.engine.release_query_lease(lease);
                                shortfall = Some(Shortfall::Broken(format!(
                                    "remote KV load submit: {err}"
                                )));
                            }
                        }
                    } else {
                        // Pool pressure: in-flight release saves free pages
                        // within a few steps — a wait, not a failure.
                        offload.engine.release_query_lease(lease);
                        shortfall = Some(Shortfall::Racing);
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
                Some(lease) => {
                    // Same publishing proof as the full-window hit — for a
                    // sub-block prompt this is the only query that can give it.
                    state.consecutive_miss_windows = 0;
                    match kv.schedule_prefill(tail_len, pool) {
                        Ok(()) => {
                            // step_page_indices covers the whole sequence up to
                            // the step end; the restored full blocks occupy all
                            // but the last entry, and the tail page is that last
                            // entry (the restore left kv_position block-aligned,
                            // so tail_len tokens open exactly one fresh page).
                            let pages = kv.step_page_indices(tail_len);
                            let tail_page = *pages.last().expect("tail step has a page");
                            match offload.engine.load(lease, vec![tail_page]) {
                                Ok(handle) => {
                                    let landed = handle
                                        .wait()
                                        .map_err(|err| anyhow::anyhow!("tail KV load: {err}"))
                                        .and_then(|()| vllm_rope_fixup(fixup_worker, &[tail_page]));
                                    match landed {
                                        Ok(()) => {
                                            kv.apply_prefill_chunk(pool)?;
                                            cached_tokens += tail_len;
                                        }
                                        Err(err) => {
                                            kv.revert_schedule()?;
                                            shortfall = Some(Shortfall::Broken(format!("{err:#}")));
                                        }
                                    }
                                }
                                Err(err) => {
                                    offload.engine.release_query_lease(lease);
                                    kv.revert_schedule()?;
                                    shortfall = Some(Shortfall::Broken(format!(
                                        "tail KV load submit: {err}"
                                    )));
                                }
                            }
                        }
                        Err(err) => {
                            offload.engine.release_query_lease(lease);
                            log::debug!("GLM5.2 P/D tail page allocation deferred: {err:?}");
                            shortfall = Some(Shortfall::Racing);
                        }
                    }
                }
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
        return Ok(VllmAdmitOutcome::Admit {
            kv: Box::new(kv),
            cached_tokens,
        });
    }
    drop(kv); // release matched/loaded holdings before parking or rejecting

    let parked = state.parked[rank].as_mut().expect("front is parked");
    // Phase reflects THIS attempt: pegaflow's first query always starts an
    // async fetch and reports `Loading`, so a sticky flag would pin every
    // request to the hard fetch deadline and the miss window would never
    // bind. Once the fetch resolves to a miss, the registration window
    // (still measured from parked_at) takes over.
    parked.saw_loading = saw_loading;
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
        _ if Instant::now() >= deadline => {
            let waited = parked.parked_at.elapsed();
            state.clear_parked(rank);
            state.consecutive_miss_windows = state.consecutive_miss_windows.saturating_add(1);
            if state.consecutive_miss_windows == MISS_BREAKER_THRESHOLD {
                log::warn!(
                    "GLM5.2 P/D miss breaker open: {MISS_BREAKER_THRESHOLD} consecutive requests \
                     exhausted the remote-KV wait window; new requests now park for \
                     {BREAKER_PROBE_WINDOW:?} instead of the full window until a complete \
                     restore lands"
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

/// Native OpenInfer P/D restore. Full pages use kvbm lineage hashes; the
/// producer includes the partial-tail hash in metadata because a mutable
/// page has no independently derivable cache key on the decode request.
///
/// Loads never block: the H2D copy is submitted, the request parks at its
/// queue front with the [`NativePendingLoad`], and the next admission
/// attempt polls the handle (#799 — a rank blocked in admission stalls
/// every peer in the fixed-cadence collective chain).
pub(super) fn admit_native_mtp_pd(
    state: &mut NativePdState,
    rank: usize,
    offload: &RankOffload,
    pool: &BlockPool,
    req: &GenerateRequest,
    handoff: &NativeMtpHandoff,
) -> anyhow::Result<VllmAdmitOutcome> {
    let anchor_plan = native_anchor_plan(req, handoff)?;
    anyhow::ensure!(
        handoff.tail_len == native_pd_tail_len(handoff.committed_len),
        "native-MTP P/D tail length {} disagrees with committed length {}",
        handoff.tail_len,
        handoff.committed_len
    );
    anyhow::ensure!(
        (handoff.tail_len == 0) == handoff.tail_key.is_none(),
        "native-MTP P/D tail key presence disagrees with tail length"
    );

    let query_key = state.parked(rank, req).query_key.clone();
    let prompt_kv = &req.prompt_tokens[..handoff.committed_len];
    let cache_salt = super::native_mtp_cache_salt(prompt_kv);
    let full_len = handoff.committed_len - handoff.tail_len;

    // Settle what this front parked for: its in-flight H2D load.
    let mut full_hold: Option<PrefixProbe> = None;
    let mut settled_tail: Option<(RequestKv, usize)> = None;
    match state.take_pending(rank) {
        None => {}
        Some(NativePendingLoad::Full {
            mut probe,
            reservation,
            mut handle,
        }) => match handle.poll() {
            None => {
                return native_pd_park_pending(
                    state,
                    rank,
                    req,
                    NativePendingLoad::Full {
                        probe,
                        reservation,
                        handle,
                    },
                    "full-page H2D copy",
                );
            }
            Some(Ok(())) => {
                pool.commit_loaded_blocks(&mut probe, reservation);
                // Hold the probe through the rematch below: the committed
                // blocks must not evict before the request re-pins them.
                full_hold = Some(probe);
            }
            Some(Err(err)) => {
                return native_pd_reject(state, rank, req, format!("full-page load: {err}"));
            }
        },
        Some(NativePendingLoad::Tail {
            mut kv,
            cached_tokens,
            mut handle,
        }) => match handle.poll() {
            None => {
                return native_pd_park_pending(
                    state,
                    rank,
                    req,
                    NativePendingLoad::Tail {
                        kv,
                        cached_tokens,
                        handle,
                    },
                    "tail H2D copy",
                );
            }
            Some(Ok(())) => {
                kv.apply_prefill_chunk(pool)?;
                settled_tail = Some((*kv, cached_tokens + handoff.tail_len));
            }
            Some(Err(err)) => {
                kv.revert_schedule()?;
                return native_pd_reject(state, rank, req, format!("tail load: {err}"));
            }
        },
    }

    let (mut kv, cached_tokens) = if let Some(done) = settled_tail {
        done
    } else {
        // Full pages: pull whatever the GPU cache no longer holds. A load
        // settled above already committed its pages, so the fresh probe here
        // reports them as GPU-hit and the query window is empty.
        if full_hold.is_none() {
            let probe =
                pool.probe_prefix_with_cache_salt(prompt_kv.to_vec(), Some(&cache_salt), None);
            let hashes = probe.cpu_query_hashes();
            if !hashes.is_empty() {
                match offload.engine.query(&format!("{query_key}-full"), &hashes) {
                    Ok(QueryOutcome::Ready(hit)) => match hit.lease {
                        Some(lease) if hit.num_blocks == hashes.len() => {
                            let Some(reservation) = pool.reserve_loaded_blocks(hit.num_blocks)
                            else {
                                offload.engine.release_query_lease(lease);
                                return native_pd_wait(state, rank, req, "GPU page reservation");
                            };
                            match offload.engine.load(lease, reservation.page_ids()) {
                                Ok(handle) => {
                                    return native_pd_park_pending(
                                        state,
                                        rank,
                                        req,
                                        NativePendingLoad::Full {
                                            probe,
                                            reservation,
                                            handle,
                                        },
                                        "full-page H2D copy",
                                    );
                                }
                                Err(err) => {
                                    offload.engine.release_query_lease(lease);
                                    return native_pd_reject(
                                        state,
                                        rank,
                                        req,
                                        format!("full-page load submit: {err}"),
                                    );
                                }
                            }
                        }
                        Some(lease) => {
                            offload.engine.release_query_lease(lease);
                            return native_pd_wait(
                                state,
                                rank,
                                req,
                                "partial full-page registration",
                            );
                        }
                        None => return native_pd_wait(state, rank, req, "full-page registration"),
                    },
                    Ok(QueryOutcome::Loading) => {
                        return native_pd_wait(state, rank, req, "full-page transfer");
                    }
                    Err(err) => {
                        return native_pd_reject(
                            state,
                            rank,
                            req,
                            format!("full-page query: {err}"),
                        );
                    }
                }
            }
        }

        let mut logical_prompt = req.prompt_tokens.clone();
        if anchor_plan.replay_to_client {
            logical_prompt.push(anchor_plan.token);
        }
        let logical_prompt_len = logical_prompt.len();
        let kv_shape = native_kv_shape(req, anchor_plan)?;
        anyhow::ensure!(
            logical_prompt_len == kv_shape.input_tokens,
            "native-MTP P/D KV input shape drift: built {logical_prompt_len}, planned {}",
            kv_shape.input_tokens
        );
        let mut kv = pool.new_request_with_cache_salt(
            logical_prompt,
            kv_shape.max_output_tokens,
            Some(&cache_salt),
            None,
        );
        let cached_tokens = kv.match_and_add_prefix(pool)?;
        drop(full_hold); // rematch done — the request itself holds the pages now
        if cached_tokens < full_len {
            return native_pd_wait(state, rank, req, "full-page rematch");
        }
        if native_pd_needs_tail_load(cached_tokens, handoff.committed_len, handoff.tail_len) {
            let key = hex::decode(handoff.tail_key.as_deref().expect("validated tail key"))
                .context("native-MTP P/D tail key is not hex")?;
            anyhow::ensure!(
                key.len() == 16,
                "native-MTP P/D tail key must be 16 bytes, got {}",
                key.len()
            );
            match offload.engine.query(&format!("{query_key}-tail"), &[key]) {
                Ok(QueryOutcome::Ready(hit)) => match hit.lease {
                    Some(lease) if hit.num_blocks == 1 => {
                        kv.schedule_prefill(handoff.tail_len, pool)
                            .map_err(|err| anyhow::anyhow!("native P/D tail schedule: {err}"))?;
                        let tail_page = *kv
                            .step_page_indices(handoff.tail_len)
                            .last()
                            .expect("tail schedule owns one page");
                        match offload.engine.load(lease, vec![tail_page]) {
                            Ok(handle) => {
                                return native_pd_park_pending(
                                    state,
                                    rank,
                                    req,
                                    NativePendingLoad::Tail {
                                        kv: Box::new(kv),
                                        cached_tokens,
                                        handle,
                                    },
                                    "tail H2D copy",
                                );
                            }
                            Err(err) => {
                                offload.engine.release_query_lease(lease);
                                kv.revert_schedule()?;
                                return native_pd_reject(
                                    state,
                                    rank,
                                    req,
                                    format!("tail load submit: {err}"),
                                );
                            }
                        }
                    }
                    Some(lease) => {
                        offload.engine.release_query_lease(lease);
                        return native_pd_wait(state, rank, req, "partial tail registration");
                    }
                    None => return native_pd_wait(state, rank, req, "tail registration"),
                },
                Ok(QueryOutcome::Loading) => {
                    return native_pd_wait(state, rank, req, "tail transfer");
                }
                Err(err) => {
                    return native_pd_reject(state, rank, req, format!("tail query: {err}"));
                }
            }
        }
        (kv, cached_tokens)
    };

    let logical_prompt_len = req.prompt_tokens.len() + usize::from(anchor_plan.replay_to_client);
    if cached_tokens != handoff.committed_len || logical_prompt_len - kv.kv_position() != 1 {
        return native_pd_wait(state, rank, req, "complete-prefix install");
    }
    kv.adopt_external_prefill_anchor()
        .context("native-MTP P/D anchor adoption")?;
    state.clear(rank);
    Ok(VllmAdmitOutcome::Admit {
        kv: Box::new(kv),
        cached_tokens,
    })
}

/// Park with an in-flight H2D load — or, past the deadline, scrap the load
/// onto the abandoned list (the DMA still writes its pages) and reject.
fn native_pd_park_pending(
    state: &mut NativePdState,
    rank: usize,
    req: &GenerateRequest,
    pending: NativePendingLoad,
    phase: &str,
) -> anyhow::Result<VllmAdmitOutcome> {
    let parked = state.parked(rank, req);
    if Instant::now() < parked.deadline {
        parked.pending = Some(pending);
        return Ok(VllmAdmitOutcome::Park);
    }
    let waited = parked.parked_at.elapsed();
    state.abandoned.push(pending.into_abandoned());
    state.parked[rank] = None;
    Ok(VllmAdmitOutcome::Reject {
        message: format!(
            "GLM5.2 native-MTP P/D handoff incomplete after {waited:?} ({phase}); retry via P"
        ),
    })
}

fn native_pd_wait(
    state: &mut NativePdState,
    rank: usize,
    req: &GenerateRequest,
    phase: &str,
) -> anyhow::Result<VllmAdmitOutcome> {
    let parked = state.parked(rank, req);
    if Instant::now() < parked.deadline {
        return Ok(VllmAdmitOutcome::Park);
    }
    let waited = parked.parked_at.elapsed();
    state.clear(rank);
    Ok(VllmAdmitOutcome::Reject {
        message: format!(
            "GLM5.2 native-MTP P/D handoff incomplete after {waited:?} ({phase}); retry via P"
        ),
    })
}

fn native_pd_reject(
    state: &mut NativePdState,
    rank: usize,
    _req: &GenerateRequest,
    reason: String,
) -> anyhow::Result<VllmAdmitOutcome> {
    state.clear(rank);
    Ok(VllmAdmitOutcome::Reject {
        message: format!("GLM5.2 native-MTP P/D restore failed ({reason}); retry via P"),
    })
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

/// One rank's plain host-tier restore (the non-P/D admission leg): probe →
/// query → load into reserved pool pages → commit as matchable prefix. The
/// load's H2D copy is submitted here and polled at the following step
/// boundaries instead of blocking the engine loop (#799); the front request
/// parks in its queue meanwhile.
///
/// Every shortfall degrades to "admit without the restore" — cache
/// maintenance, never a correctness dependency.
pub(super) struct HostRestoreState {
    pending: Option<PendingHostRestore>,
    /// See [`AbandonedLoad`].
    abandoned: Vec<AbandonedLoad>,
}

struct PendingHostRestore {
    /// Re-identifies the front across admission retries.
    fingerprint: (Option<String>, usize),
    probe: PrefixProbe,
    reservation: LoadReservation,
    handle: LoadHandle,
}

/// One admission attempt's verdict for the plain host-tier restore.
pub(super) enum HostRestoreOutcome {
    /// Nothing (left) to load: admit now. The probe, when present, holds the
    /// GPU-hit and freshly-committed blocks across the caller's
    /// `match_and_add_prefix` so the restored prefix cannot evict before it
    /// is re-matched.
    Ready(Option<PrefixProbe>),
    /// H2D copy in flight into reserved pages: leave the request at the
    /// queue front and re-poll at the next step boundary.
    Park,
}

impl HostRestoreState {
    pub(super) fn new() -> Self {
        Self {
            pending: None,
            abandoned: Vec::new(),
        }
    }

    /// Pool pages held for the parked front's in-flight restore. They become
    /// the request's cached prefix on admit, so the admission budget credits
    /// them back against the request's need (they are already out of
    /// `available_blocks`, and counting them on both sides would wedge the
    /// front forever).
    pub(super) fn front_held_blocks(&self) -> usize {
        self.pending
            .as_ref()
            .map_or(0, |p| p.probe.held_blocks() + p.reservation.len())
    }

    /// Drop abandoned loads whose DMA settled (their page holds release).
    pub(super) fn reap(&mut self) {
        self.abandoned.retain_mut(AbandonedLoad::live);
    }

    /// The FIFO front left the queue (disconnect) — its in-flight load keeps
    /// its page holds on the abandoned list until the DMA settles.
    pub(super) fn abandon_front(&mut self) {
        if let Some(p) = self.pending.take() {
            drop(p.probe); // source holds; the DMA only writes the reservation
            self.abandoned.push(AbandonedLoad {
                handle: p.handle,
                _hold: AbandonedHold::Pages(p.reservation),
            });
        }
    }

    /// Advance the front request's restore by one admission attempt: poll an
    /// in-flight load, or probe/query/submit a fresh one. `engine` is `None`
    /// only in offline contract tests; production callers always have the
    /// rank's offload engine.
    pub(super) fn poll_front(
        &mut self,
        engine: Option<&OffloadEngine>,
        pool: &BlockPool,
        req: &GenerateRequest,
    ) -> HostRestoreOutcome {
        let fingerprint = (req.request_id.clone(), req.prompt_tokens.len());
        if self
            .pending
            .as_ref()
            .is_some_and(|p| p.fingerprint != fingerprint)
        {
            // The front changed under an in-flight load (e.g. the previous
            // front was rejected at intake): scrap its load, start fresh.
            self.abandon_front();
        }
        if let Some(mut p) = self.pending.take() {
            return match p.handle.poll() {
                None => {
                    self.pending = Some(p);
                    HostRestoreOutcome::Park
                }
                Some(Ok(())) => {
                    let restored = p.reservation.len();
                    pool.commit_loaded_blocks(&mut p.probe, p.reservation);
                    // The only signal separating a host-tier restore from a
                    // plain GPU prefix hit — the parity/eviction gates key
                    // on it.
                    log::info!("GLM5.2 host-tier restore: {restored} blocks committed");
                    HostRestoreOutcome::Ready(Some(p.probe))
                }
                Some(Err(err)) => {
                    // The DMA settled, so the reservation can release; the
                    // prefix just prefills from scratch.
                    log::warn!("GLM5.2 host-tier load failed (prefill from scratch): {err}");
                    HostRestoreOutcome::Ready(Some(p.probe))
                }
            };
        }

        let probe = pool.probe_prefix(req.prompt_tokens.clone(), None);
        let hashes = probe.cpu_query_hashes();
        if hashes.is_empty() {
            return HostRestoreOutcome::Ready(Some(probe));
        }
        let Some(engine) = engine else {
            return HostRestoreOutcome::Ready(Some(probe));
        };
        let req_key = format!("glm52-admit-{}", QUERY_SEQ.fetch_add(1, Ordering::Relaxed));
        let hit = match engine.query(&req_key, &hashes) {
            Ok(QueryOutcome::Ready(hit)) => hit,
            Ok(QueryOutcome::Loading) => {
                // Host-memory-only setup: pegaflow has no deeper tier to
                // fetch from, so an async outcome means a config drift worth
                // seeing.
                log::warn!(
                    "GLM5.2 host-tier query went async in a host-only setup; skipping restore"
                );
                return HostRestoreOutcome::Ready(Some(probe));
            }
            Err(err) => {
                log::warn!("GLM5.2 host-tier query failed (prefill from scratch): {err}");
                return HostRestoreOutcome::Ready(Some(probe));
            }
        };
        let Some(lease) = hit.lease else {
            return HostRestoreOutcome::Ready(Some(probe));
        };
        let Some(reservation) = pool.reserve_loaded_blocks(hit.num_blocks) else {
            // Block pressure: the pool cannot hold the restored prefix right
            // now. Prefill recomputes it — correct, just colder.
            engine.release_query_lease(lease);
            return HostRestoreOutcome::Ready(Some(probe));
        };
        match engine.load(lease, reservation.page_ids()) {
            Ok(handle) => {
                self.pending = Some(PendingHostRestore {
                    fingerprint,
                    probe,
                    reservation,
                    handle,
                });
                HostRestoreOutcome::Park
            }
            Err(err) => {
                log::warn!("GLM5.2 host-tier load submit failed (prefill from scratch): {err}");
                // `load` consumes the lease only past its early validation; a
                // submit error may leave it pinning the host blocks until the
                // lease TTL. Release explicitly (no-op if already consumed).
                engine.release_query_lease(lease);
                HostRestoreOutcome::Ready(Some(probe))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use openinfer_core::engine::TokenEvent;
    use openinfer_core::engine::TokenSink;

    use super::super::RankSlots;
    use super::super::admission::admit_from_queue;
    use super::super::testkit;
    use super::*;

    /// An in-flight plain restore parks the front without blocking, holds its
    /// pages against the admission budget, and admits with the restored
    /// prefix once the DMA settles — the end-to-end contract of #799.
    #[test]
    fn parked_restore_defers_then_admits_with_the_restored_prefix() {
        let pool = BlockPool::new(PAGE, 8).expect("pool");
        let prompt = vec![7_u32; 5 * PAGE]; // needs 6 lifetime blocks with max_tokens=1
        let probe = pool.probe_prefix(prompt.clone(), None);
        assert_eq!(probe.cpu_query_hashes().len(), 4);
        let reservation = pool
            .reserve_loaded_blocks(4)
            .expect("4 destination pages fit");
        let (handle, settle) = LoadHandle::in_flight();
        let mut host_restore = Some(HostRestoreState {
            pending: Some(PendingHostRestore {
                fingerprint: (None, prompt.len()),
                probe,
                reservation,
                handle,
            }),
            abandoned: Vec::new(),
        });

        let mut slots: RankSlots = std::array::from_fn(|_| None);
        let mut pending = VecDeque::new();
        let mut req = testkit::request(prompt, testkit::sampled(0.0), 1);
        let (token_tx, mut token_rx) = TokenSink::standalone();
        req.token_tx = token_tx;
        pending.push_back(req);
        let mut pending_resets = Vec::new();
        let mut admit = |pending: &mut VecDeque<_>,
                         slots: &mut RankSlots,
                         host_restore: &mut Option<HostRestoreState>| {
            admit_from_queue(
                0,
                pending,
                slots,
                &pool,
                7,
                None,
                &mut None,
                &mut None,
                host_restore,
                None,
                false,
                true,
                false,
                false,
                &mut pending_resets,
            )
            .expect("admission");
        };

        // In flight: the front parks at the queue head. Without the
        // held-block credit the budget check would break here forever (the
        // reserved pages are out of `available_blocks` AND inside the
        // request's own need).
        admit(&mut pending, &mut slots, &mut host_restore);
        assert_eq!(pending.len(), 1, "front stays parked");
        assert!(slots.iter().all(Option::is_none));
        assert_eq!(
            host_restore.as_ref().unwrap().front_held_blocks(),
            4,
            "the in-flight load keeps its destination pages held"
        );

        settle.send(Ok(())).expect("handle alive");
        admit(&mut pending, &mut slots, &mut host_restore);
        assert!(pending.is_empty());
        assert!(slots[0].is_some(), "settled restore admits the front");
        assert_eq!(host_restore.as_ref().unwrap().front_held_blocks(), 0);
        match token_rx.try_recv().map(|(_, event)| event) {
            Ok(TokenEvent::Scheduled { cached_tokens, .. }) => {
                assert_eq!(cached_tokens, 4 * PAGE, "restored pages matched as prefix");
            }
            other => panic!("expected Scheduled, got {other:?}"),
        }
    }

    #[test]
    fn failed_restore_load_releases_pages_and_degrades_to_plain_admission() {
        let pool = BlockPool::new(PAGE, 8).expect("pool");
        let base = pool.available_blocks();
        let prompt = vec![7_u32; 3 * PAGE];
        let probe = pool.probe_prefix(prompt.clone(), None);
        let reservation = pool.reserve_loaded_blocks(2).expect("2 pages fit");
        let mut state = HostRestoreState {
            pending: Some(PendingHostRestore {
                fingerprint: (None, prompt.len()),
                probe,
                reservation,
                handle: LoadHandle::settled(Err(openinfer_kv_offload::EngineError::Storage(
                    "injected".into(),
                ))),
            }),
            abandoned: Vec::new(),
        };
        let req = testkit::request(prompt, testkit::sampled(0.0), 1);
        match state.poll_front(None, &pool, &req) {
            HostRestoreOutcome::Ready(probe) => {
                assert_eq!(probe.expect("probe survives").held_blocks(), 0);
            }
            HostRestoreOutcome::Park => panic!("a settled load must not park"),
        }
        assert_eq!(
            pool.available_blocks(),
            base,
            "the failed load's reservation released"
        );
    }

    #[test]
    fn abandoned_restore_holds_its_pages_until_the_dma_settles() {
        let pool = BlockPool::new(PAGE, 8).expect("pool");
        let base = pool.available_blocks();
        let prompt = vec![7_u32; 3 * PAGE];
        let probe = pool.probe_prefix(prompt.clone(), None);
        let reservation = pool.reserve_loaded_blocks(2).expect("2 pages fit");
        let (handle, settle) = LoadHandle::in_flight();
        let mut state = HostRestoreState {
            pending: Some(PendingHostRestore {
                fingerprint: (None, prompt.len()),
                probe,
                reservation,
                handle,
            }),
            abandoned: Vec::new(),
        };

        state.abandon_front();
        state.reap();
        assert_eq!(state.front_held_blocks(), 0);
        assert_eq!(
            pool.available_blocks(),
            base - 2,
            "the DMA still writes the reserved pages"
        );

        settle.send(Ok(())).expect("handle alive");
        state.reap();
        assert_eq!(pool.available_blocks(), base, "settled scrap releases");
    }

    #[test]
    fn native_pd_clear_scraps_the_in_flight_tail_load() {
        let pool = BlockPool::new(PAGE, 8).expect("pool");
        let base = pool.available_blocks();
        let mut kv = pool.new_request(vec![7_u32; PAGE], 1, None);
        kv.schedule_prefill(PAGE, &pool).expect("tail page");
        let held = kv.resident_blocks();
        assert!(held > 0);
        let (handle, settle) = LoadHandle::in_flight();

        let mut state = NativePdState::new(1);
        let req = testkit::request(vec![7_u32; PAGE], testkit::sampled(0.0), 1);
        state.parked(0, &req).pending = Some(NativePendingLoad::Tail {
            kv: Box::new(kv),
            cached_tokens: 0,
            handle,
        });
        assert_eq!(state.front_held_blocks(0), held);

        state.clear(0);
        state.reap();
        assert_eq!(state.front_held_blocks(0), 0);
        assert_eq!(
            pool.available_blocks(),
            base - held,
            "the DMA still writes the scheduled page"
        );

        settle.send(Ok(())).expect("handle alive");
        state.reap();
        assert_eq!(pool.available_blocks(), base);
    }

    #[test]
    fn a_new_native_front_scraps_the_predecessors_load() {
        let pool = BlockPool::new(PAGE, 8).expect("pool");
        let prompt = vec![7_u32; 3 * PAGE];
        let probe = pool.probe_prefix(prompt.clone(), None);
        let reservation = pool.reserve_loaded_blocks(2).expect("2 pages fit");
        let (handle, _settle) = LoadHandle::in_flight();

        let mut state = NativePdState::new(1);
        let mut first = testkit::request(prompt.clone(), testkit::sampled(0.0), 1);
        first.request_id = Some("first".to_string());
        state.parked(0, &first).pending = Some(NativePendingLoad::Full {
            probe,
            reservation,
            handle,
        });

        let mut second = testkit::request(prompt, testkit::sampled(0.0), 1);
        second.request_id = Some("second".to_string());
        state.parked(0, &second);
        assert_eq!(state.front_held_blocks(0), 0);
        assert_eq!(
            state.abandoned.len(),
            1,
            "the predecessor's load rides the abandoned list"
        );
    }

    fn pd_state(miss_wait: Duration) -> VllmPdState {
        VllmPdState::new(
            &crate::Glm52VllmCompatOptions {
                python_hash_seed: "0".to_string(),
                namespace: "deadbeef".to_string(),
                miss_wait,
                allow_local_prefill: false,
            },
            1,
        )
    }

    fn window_of(parked: &ParkedFront) -> (Duration, Duration) {
        (
            parked.miss_deadline - parked.parked_at,
            parked.hard_deadline - parked.parked_at,
        )
    }

    #[test]
    fn closed_breaker_parks_with_configured_windows() {
        let miss_wait = Duration::from_millis(3000);
        let mut state = pd_state(miss_wait);
        let req = testkit::request(vec![1, 2, 3], testkit::sampled(0.0), 8);
        let (miss, hard) = window_of(state.parked_front(0, &req));
        assert_eq!(miss, miss_wait);
        assert_eq!(hard, REMOTE_FETCH_DEADLINE);
    }

    #[test]
    fn open_breaker_parks_with_probe_window_on_both_deadlines() {
        // Zero-wait rejection would starve every remote restore: pegaflow's
        // first query only STARTS the async fetch, so the breaker could
        // never close (the deadlock failure injection found).
        let mut state = pd_state(Duration::from_millis(3000));
        state.consecutive_miss_windows = MISS_BREAKER_THRESHOLD;
        let req = testkit::request(vec![1, 2, 3], testkit::sampled(0.0), 8);
        let (miss, hard) = window_of(state.parked_front(0, &req));
        assert_eq!(miss, BREAKER_PROBE_WINDOW);
        assert_eq!(hard, BREAKER_PROBE_WINDOW);
    }

    #[test]
    fn reparking_the_same_front_keeps_its_deadlines_and_query_key() {
        let mut state = pd_state(Duration::from_millis(3000));
        let req = testkit::request(vec![1, 2, 3], testkit::sampled(0.0), 8);
        let (first_key, first_at) = {
            let parked = state.parked_front(0, &req);
            (parked.query_key.clone(), parked.parked_at)
        };
        let parked = state.parked_front(0, &req);
        assert_eq!(
            parked.query_key, first_key,
            "retries must poll the same fetch"
        );
        assert_eq!(
            parked.parked_at, first_at,
            "retries must not extend the window"
        );
    }

    #[test]
    fn a_new_front_resets_the_park() {
        let mut state = pd_state(Duration::from_millis(3000));
        let req = testkit::request(vec![1, 2, 3], testkit::sampled(0.0), 8);
        let first_key = state.parked_front(0, &req).query_key.clone();
        let other = testkit::request(vec![1, 2, 3, 4], testkit::sampled(0.0), 8);
        let parked = state.parked_front(0, &other);
        assert_ne!(parked.query_key, first_key);
    }

    #[test]
    fn query_key_never_reuses_the_client_request_id() {
        // pegaflow keys prefetch state and its failed-remote blacklist by
        // this id; a client-controlled value could cross-consume another
        // request's fetch or inherit a blacklist entry.
        let mut state = pd_state(Duration::from_millis(3000));
        let mut req = testkit::request(vec![1, 2, 3], testkit::sampled(0.0), 8);
        req.request_id = Some("client-controlled".to_string());
        let parked = state.parked_front(0, &req);
        assert!(parked.query_key.starts_with("glm52-pd-"));
    }

    #[test]
    fn native_pd_tail_keeps_the_last_aligned_page_explicit() {
        assert_eq!(native_pd_tail_len(0), 0);
        assert_eq!(native_pd_tail_len(1), 1);
        assert_eq!(native_pd_tail_len(PAGE - 1), PAGE - 1);
        assert_eq!(native_pd_tail_len(PAGE), PAGE);
        assert_eq!(native_pd_tail_len(PAGE + 1), 1);
        assert_eq!(native_pd_tail_len(2 * PAGE), PAGE);
    }

    #[test]
    fn native_pd_reuses_an_already_cached_aligned_tail() {
        assert!(!native_pd_needs_tail_load(PAGE, PAGE, PAGE));
        assert!(native_pd_needs_tail_load(0, PAGE, PAGE));
    }

    #[test]
    fn native_pd_v2_installs_and_replays_the_router_anchor() {
        let mut req = testkit::request(vec![1, 2, 3], testkit::sampled(0.0), 8);
        req.kv_transfer_params = Some(serde_json::json!({
            "openinfer_pd": {
                "version": 2,
                "native_mtp": {
                    "draft_tokens": [5, 6, 7, 8, 9],
                    "committed_len": 3,
                    "arena_count": 101,
                    "tail_len": 3,
                    "tail_key": "00000000000000000000000000000000",
                    "anchor_token_id": 4,
                    "anchor_emitted": true
                }
            }
        }));
        let handoff = native_mtp_handoff(&req)
            .expect("valid v2 envelope")
            .expect("native handoff");
        assert_eq!(
            native_anchor_plan(&req, &handoff).expect("router plan"),
            NativeAnchorPlan {
                token: 4,
                replay_to_client: true,
                emitted_by_prefill: true,
            }
        );

        req.prompt_tokens.push(4);
        assert_eq!(
            native_anchor_plan(&req, &handoff).expect("manual plan"),
            NativeAnchorPlan {
                token: 4,
                replay_to_client: false,
                emitted_by_prefill: true,
            }
        );
    }

    #[test]
    fn native_pd_rejects_v1_metadata() {
        let mut req = testkit::request(vec![1, 2, 3, 4], testkit::sampled(0.0), 8);
        req.kv_transfer_params = Some(serde_json::json!({
            "openinfer_pd": {
                "version": 1,
                "native_mtp": {
                    "draft_tokens": [5, 6, 7, 8, 9],
                    "committed_len": 3,
                    "arena_count": 101,
                    "tail_len": 3,
                    "tail_key": "00000000000000000000000000000000",
                    "anchor_token_id": 4,
                    "anchor_emitted": true
                }
            }
        }));
        let error = native_mtp_handoff(&req).expect_err("v1 must be rejected");
        assert!(
            error
                .to_string()
                .contains("unsupported openinfer P/D metadata version 1"),
            "{error:#}"
        );
    }

    #[test]
    fn native_pd_ignores_router_prefill_connector_metadata() {
        let mut req = testkit::request(vec![1, 2, 3], testkit::sampled(0.0), 1);
        req.kv_transfer_params = Some(serde_json::json!({
            "do_remote_prefill": true,
            "do_remote_decode": false
        }));
        assert!(
            native_mtp_handoff(&req)
                .expect("unrelated connector metadata is not malformed")
                .is_none()
        );
    }
}
