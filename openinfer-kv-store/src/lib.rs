//! `openinfer-kv-store`: the shared KV read/write orchestration layer.
//!
//! Design: `docs/subsystems/kv-cache/design.md`. This crate gives the per-model
//! offload glue (qwen3's prefetch state machine, glm52's `offload.rs`) one
//! home built on the same primitives they already use: the logical
//! [`BlockPool`] for GPU pages and a [`HostTier`] (pegaflow via
//! [`openinfer_kv_offload::OffloadEngine`]) below it.
//!
//! Three verbs:
//! - [`KvStore::resolve_prefix`] — the whole read path as one async fn:
//!   probe the GPU radix, query the host tier (re-query/deadline built in),
//!   reserve pages under the admission floor, load, and register into the
//!   radix. One terminal type: [`KvPrefix`] — degraded outcomes surface as a
//!   smaller hit plus a stats event, never a distinct variant.
//! - [`KvStore::seal`] — save freshly-sealed blocks at a checkpoint boundary.
//!   Guards pin the source pages across the async D2H (the reuse contract).
//! - [`KvStore::retire`] — final seal + release; parks the whole `RequestKv`
//!   with any must-complete saves instead of blocking anyone.
//!
//! The scheduler stays synchronous: it receives resolved requests from its
//! submit channel, reads [`KvStore::pinned_blocks`] during admission, and
//! maintains [`KvStore::set_admission_floor`]. Cancellation is the request's
//! existing abort state ([`CancelProbe`] over `TokenSink::is_closed`) observed
//! between operations; a submitted DMA is an uncancellable section.

mod stats;
pub mod testkit;
mod tier;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use openinfer_engine::engine::KvPrefix;
use openinfer_engine::engine::TokenSink;
use openinfer_kv_cache::BlockPool;
use openinfer_kv_cache::RequestKv;
use tokio::sync::oneshot;
use tokio::time::Instant;

pub use crate::stats::DegradeReason;
pub use crate::stats::KvStoreStats;
pub use crate::tier::HostTier;
pub use crate::tier::TierFuture;
pub use crate::tier::TierHit;
pub use crate::tier::TierQuery;

/// Whether the request wants the resolve abandoned. Implemented by the
/// engine's `TokenSink` (the abort atomic the frontend flips); the store
/// observes it between operations and short-circuits remaining I/O.
pub trait CancelProbe: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

impl CancelProbe for TokenSink {
    fn is_cancelled(&self) -> bool {
        self.is_closed()
    }
}

/// For callers without a request context (tests, warmup).
pub struct NeverCancelled;

impl CancelProbe for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Prefix-cache identity of a resolve, mirroring
/// [`BlockPool::probe_prefix_with_cache_salt`]: the producer request and the
/// resolve must derive identical block hashes or the query keys are unrelated.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheScope<'a> {
    pub cache_salt: Option<&'a str>,
    pub lora_name: Option<&'a str>,
}

/// Save quality-of-service, from the design doc's QoS split.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SaveClass {
    /// Fire-and-forget cacheability: a lost save forfeits a future hit, never
    /// correctness. Sheddable under pressure.
    Cacheable,
    /// Must-complete (P/D handoff): [`KvStore::retire`] parks the request's KV
    /// until these saves settle. Lease semantics against the consuming peer
    /// land with the glm52 P/D migration.
    Handoff,
}

/// Per-request save bookkeeping, owned by the scheduler next to the
/// `RequestKv` (no hidden map in the store). Starts past the prefix-cache
/// hit — those blocks were stored by whoever first sealed them.
#[derive(Default)]
pub struct SaveCursor {
    saved_blocks: usize,
    /// Completion outcomes of this request's `Handoff`-class saves, awaited
    /// by [`KvStore::retire`] before the KV releases.
    pending: Vec<oneshot::Receiver<Result<(), String>>>,
}

impl SaveCursor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug)]
pub struct KvStoreConfig {
    /// Pause between host-tier re-queries while a deeper tier is fetching.
    pub requery_interval: Duration,
    /// Ceiling on one resolve's host-tier wait — bounds every tier await
    /// (query and load alike), so a hung storage worker degrades the resolve
    /// instead of stranding the request outside the scheduler.
    pub resolve_deadline: Duration,
    /// Ceiling on the pool share `Cacheable` saves may pin, as a percent of
    /// the rank's total blocks. Past it, cacheable saves shed (a forfeited
    /// future hit) instead of pinning admission out of the pool. `Handoff`
    /// saves are exempt — their backpressure is admission reading
    /// [`KvStore::pinned_blocks`].
    pub cacheable_pin_percent: usize,
}

impl Default for KvStoreConfig {
    fn default() -> Self {
        Self {
            // The qwen3 remote-fetch cadence this generalizes (5ms re-query,
            // 15s handoff deadline).
            requery_interval: Duration::from_millis(5),
            resolve_deadline: Duration::from_secs(15),
            cacheable_pin_percent: 25,
        }
    }
}

/// One rank's KV surfaces. The pool is the same logical pool the rank's
/// scheduler allocates from — kvbm's `BlockManager` is internally
/// synchronized (save guards already cross threads today), so the resolve
/// task allocating from it is an arbitration question, answered by the floor.
struct RankState {
    pool: Arc<BlockPool>,
    tier: Option<Arc<dyn HostTier>>,
    /// Blocks admission has promised to admitted requests; resolve-side
    /// allocation yields to it (fail-soft to a smaller hit).
    floor: AtomicUsize,
    /// Blocks pinned by in-flight saves — physically unallocatable until
    /// their D2H lands. Admission subtracts this from its budget. `Arc`: the
    /// per-save watcher tasks decrement it after the store's borrow ends.
    pinned: Arc<AtomicUsize>,
    /// Pin ceiling for `Cacheable` saves (from
    /// [`KvStoreConfig::cacheable_pin_percent`] of the pool).
    cacheable_pin_budget: usize,
}

/// Builds a [`KvStore`]: every rank's pool and host tier is declared here,
/// and the rank table freezes at [`Self::build`] — the store's read paths
/// take no lock. Models whose pools are currently constructed inside engine
/// threads hoist that construction to before spawn (a `BlockPool` is a pure
/// CPU object with no thread affinity).
pub struct KvStoreBuilder {
    runtime: tokio::runtime::Handle,
    config: KvStoreConfig,
    ranks: HashMap<usize, RankState>,
}

impl KvStoreBuilder {
    #[must_use]
    pub fn new(runtime: tokio::runtime::Handle, config: KvStoreConfig) -> Self {
        Self {
            runtime,
            config,
            ranks: HashMap::new(),
        }
    }

    /// Declare one rank's pool and (optionally) its host tier.
    #[must_use]
    pub fn rank(
        mut self,
        rank: usize,
        pool: Arc<BlockPool>,
        tier: Option<Arc<dyn HostTier>>,
    ) -> Self {
        let cacheable_pin_budget = pool.total_blocks() * self.config.cacheable_pin_percent / 100;
        self.ranks.insert(
            rank,
            RankState {
                pool,
                tier,
                floor: AtomicUsize::new(0),
                pinned: Arc::new(AtomicUsize::new(0)),
                cacheable_pin_budget,
            },
        );
        self
    }

    #[must_use]
    pub fn build(self) -> KvStore {
        KvStore {
            runtime: self.runtime,
            config: self.config,
            ranks: self.ranks,
            stats: Arc::new(KvStoreStats::default()),
        }
    }
}

/// The process-wide KV store: one instance, `Arc`-shared. Knows token
/// prefixes, pools, and tiers — not engines, inboxes, or requests. Built by
/// [`KvStoreBuilder`]; the rank table is immutable after build.
pub struct KvStore {
    runtime: tokio::runtime::Handle,
    config: KvStoreConfig,
    ranks: HashMap<usize, RankState>,
    stats: Arc<KvStoreStats>,
}

impl KvStore {
    /// Blocks admission has promised on `rank`; resolve-side allocations
    /// yield to this watermark. The scheduler updates it as it admits and
    /// retires.
    pub fn set_admission_floor(&self, rank: usize, blocks: usize) {
        if let Some(state) = self.rank(rank) {
            state.floor.store(blocks, Ordering::Release);
        }
    }

    /// Blocks pinned by `rank`'s in-flight saves; admission subtracts this
    /// from its usable budget (the glm52 `pinned_blocks` discipline).
    pub fn pinned_blocks(&self, rank: usize) -> usize {
        self.rank(rank)
            .map_or(0, |state| state.pinned.load(Ordering::Acquire))
    }

    pub fn stats(&self) -> &KvStoreStats {
        &self.stats
    }

    /// The whole read path. Resolves `prompt_tokens`' cached prefix on
    /// `rank`: GPU radix probe, host-tier query (re-query while a deeper tier
    /// fetches, bounded by the resolve deadline), floor-gated page
    /// reservation, load, and registration into the radix — so the eventual
    /// `match_and_add_prefix` on the scheduler thread reuses the full prefix.
    ///
    /// One terminal type: a [`KvPrefix`] whose hold keeps the resolved blocks
    /// resident until the scheduler's match consumes them. Degraded outcomes
    /// (deadline, tier error, pool pressure) return the GPU hit alone and
    /// report to stats; a cancelled request returns [`KvPrefix::none`] — it
    /// dies at admission's existing closed-sink check.
    pub async fn resolve_prefix(
        &self,
        rank: usize,
        req_id: &str,
        prompt_tokens: &[u32],
        scope: CacheScope<'_>,
        cancel: &dyn CancelProbe,
    ) -> KvPrefix {
        self.stats.resolves.fetch_add(1, Ordering::Relaxed);
        let Some(state) = self.rank(rank) else {
            return KvPrefix::none();
        };
        let pool = &state.pool;
        let block_size = pool.block_size();
        let cacheable_blocks = prompt_tokens.len().saturating_sub(1) / block_size;

        let finish = |probe: openinfer_kv_cache::PrefixProbe| {
            let hit_blocks = probe.held_blocks().min(cacheable_blocks);
            if hit_blocks == 0 {
                return KvPrefix::none();
            }
            self.stats.resolve_hits.fetch_add(1, Ordering::Relaxed);
            KvPrefix::resolved(hit_blocks * block_size, Box::new(probe))
        };

        if cancel.is_cancelled() {
            self.stats.record_degrade(req_id, DegradeReason::Cancelled);
            return KvPrefix::none();
        }
        let mut probe = pool.probe_prefix_with_cache_salt(
            prompt_tokens.to_vec(),
            scope.cache_salt,
            scope.lora_name,
        );
        let host_hashes = probe.cpu_query_hashes();
        let tier = match state.tier.as_ref() {
            Some(tier) if !host_hashes.is_empty() => tier,
            _ => return finish(probe),
        };

        // Host-tier query, re-queried while a deeper tier fetches. The
        // deadline bounds every await — a query future that never settles (a
        // hung storage worker) degrades exactly like a slow one, instead of
        // stranding the request outside the scheduler forever.
        let deadline = Instant::now() + self.config.resolve_deadline;
        let hit = loop {
            if cancel.is_cancelled() {
                self.stats.record_degrade(req_id, DegradeReason::Cancelled);
                return KvPrefix::none();
            }
            let query = tokio::time::timeout_at(deadline, tier.query(req_id, host_hashes.clone()));
            match query.await {
                Err(_elapsed) => {
                    self.stats
                        .record_degrade(req_id, DegradeReason::DeadlineExceeded);
                    return finish(probe);
                }
                Ok(Err(err)) => {
                    log::warn!("kv-store resolve {req_id}: tier query failed: {err:#}");
                    self.stats.record_degrade(req_id, DegradeReason::TierError);
                    return finish(probe);
                }
                Ok(Ok(TierQuery::Miss)) => return finish(probe),
                Ok(Ok(TierQuery::Hit(hit))) => break hit,
                Ok(Ok(TierQuery::Loading)) => {
                    if Instant::now() >= deadline {
                        self.stats
                            .record_degrade(req_id, DegradeReason::DeadlineExceeded);
                        return finish(probe);
                    }
                    tokio::time::sleep(self.config.requery_interval).await;
                }
            }
        };

        if cancel.is_cancelled() {
            tier.release(hit);
            self.stats.record_degrade(req_id, DegradeReason::Cancelled);
            return KvPrefix::none();
        }

        // Floor gate: resolve-side allocation yields to admission's promises.
        // The tier lease is all-or-nothing, so a clamped hit is a declined
        // hit — release it now instead of waiting out its TTL.
        let available = pool.available_blocks();
        let floor = state.floor.load(Ordering::Acquire);
        if available.saturating_sub(floor) < hit.blocks {
            tier.release(hit);
            self.stats
                .record_degrade(req_id, DegradeReason::PoolPressure);
            return finish(probe);
        }
        let Some(reservation) = pool.reserve_loaded_blocks(hit.blocks) else {
            tier.release(hit);
            self.stats
                .record_degrade(req_id, DegradeReason::PoolPressure);
            return finish(probe);
        };
        if cancel.is_cancelled() {
            tier.release(hit);
            self.stats.record_degrade(req_id, DegradeReason::Cancelled);
            return KvPrefix::none();
        }

        // The DMA is an uncancellable section: the reservation must outlive
        // the copy. A spawned task owns both the load future and the
        // reservation, so the deadline below abandons only the *wait* — an
        // abandoned reservation is dropped by the task once the tier settles,
        // never while the DMA may still write into its blocks.
        let loaded = hit.blocks;
        let page_ids = reservation.page_ids();
        let load = tier.load(hit, page_ids);
        let join = self.runtime.spawn(async move {
            let result = load.await;
            (result, reservation)
        });
        match tokio::time::timeout_at(deadline, join).await {
            Ok(Ok((Ok(()), reservation))) => {
                pool.commit_loaded_blocks(&mut probe, reservation);
                self.stats
                    .resolve_loaded_blocks
                    .fetch_add(loaded as u64, Ordering::Relaxed);
                finish(probe)
            }
            Ok(Ok((Err(err), _reservation))) => {
                log::warn!("kv-store resolve {req_id}: tier load failed: {err:#}");
                self.stats.record_degrade(req_id, DegradeReason::TierError);
                // The load settled; dropping the reservation here returns the
                // destination blocks untouched by any registered hash.
                finish(probe)
            }
            Ok(Err(join_err)) => {
                log::warn!("kv-store resolve {req_id}: load task failed: {join_err}");
                self.stats.record_degrade(req_id, DegradeReason::TierError);
                finish(probe)
            }
            Err(_elapsed) => {
                self.stats
                    .record_degrade(req_id, DegradeReason::DeadlineExceeded);
                finish(probe)
            }
        }
    }

    /// Save the request's freshly-sealed blocks past the cursor. Call after
    /// the step that sealed them has synchronized (the tier reads the GPU
    /// asynchronously — the ordering contract passes through). Guards pin the
    /// source pages until the D2H lands; the pinned pages are visible to
    /// admission via [`Self::pinned_blocks`].
    pub fn seal(&self, rank: usize, kv: &RequestKv, cursor: &mut SaveCursor, class: SaveClass) {
        let Some(state) = self.rank(rank) else {
            return;
        };
        let Some(tier) = state.tier.as_ref() else {
            return;
        };
        if cursor.saved_blocks == 0 {
            // Prefix-hit blocks were stored by whoever first sealed them.
            cursor.saved_blocks = kv.prefix_matched_blocks();
        }
        let pairs = kv.assigned_block_hashes();
        if pairs.len() <= cursor.saved_blocks {
            return;
        }
        let count = pairs.len() - cursor.saved_blocks;

        // Cacheable saves shed under pin pressure instead of pinning
        // admission out of the pool — a shed save is a forfeited future hit,
        // never a correctness loss. The cursor does NOT advance, so a later
        // seal (or the retire) retries once pressure clears. Handoff saves
        // are exempt: their backpressure is admission via `pinned_blocks`.
        if class == SaveClass::Cacheable
            && state.pinned.load(Ordering::Acquire) + count > state.cacheable_pin_budget
        {
            self.stats.saves_shed.fetch_add(1, Ordering::Relaxed);
            log::debug!(
                "kv-store: shed cacheable save of {count} blocks (pin budget \
                 {} exceeded)",
                state.cacheable_pin_budget
            );
            return;
        }

        // Guards align 1:1 with `assigned_block_hashes`.
        let guards: Vec<_> = kv
            .assigned_block_guards()
            .into_iter()
            .skip(cursor.saved_blocks)
            .collect();
        let (ids, hashes): (Vec<i32>, Vec<Vec<u8>>) = pairs[cursor.saved_blocks..]
            .iter()
            .map(|(id, hash)| (*id, hash.to_vec()))
            .unzip();
        cursor.saved_blocks = pairs.len();

        state.pinned.fetch_add(count, Ordering::AcqRel);
        self.stats.saves_submitted.fetch_add(1, Ordering::Relaxed);
        let handle = tier.save(ids, hashes, Box::new(guards));

        let done_tx = if class == SaveClass::Handoff {
            let (tx, rx) = oneshot::channel();
            cursor.pending.push(rx);
            Some(tx)
        } else {
            None
        };
        let pinned = Arc::clone(&state.pinned);
        let stats = Arc::clone(&self.stats);
        self.runtime.spawn(async move {
            let result = handle.settle().await;
            pinned.fetch_sub(count, Ordering::AcqRel);
            let outcome = match result {
                Ok(()) => Ok(()),
                Err(err) => {
                    stats.saves_failed.fetch_add(1, Ordering::Relaxed);
                    log::warn!("kv-store save of {count} blocks failed: {err}");
                    Err(err.to_string())
                }
            };
            if let Some(tx) = done_tx {
                let _ = tx.send(outcome);
            }
        });
    }

    /// Final seal + release. With no must-complete saves outstanding the KV
    /// releases immediately (fire-and-forget D2H stays safe: the guards pin
    /// the pages independently of the release — the qwen3 save-then-drop
    /// pattern). With `Handoff` saves pending, the whole KV parks with them
    /// and releases when they settle — the glm52 `detach_tail_save` pattern,
    /// generalized. Never blocks the caller.
    pub fn retire(&self, rank: usize, mut kv: RequestKv, mut cursor: SaveCursor, class: SaveClass) {
        self.seal(rank, &kv, &mut cursor, class);
        if cursor.pending.is_empty() {
            release_logged(&mut kv);
            return;
        }
        self.stats.retires_parked.fetch_add(1, Ordering::Relaxed);
        let stats = Arc::clone(&self.stats);
        self.runtime.spawn(async move {
            let mut failed = false;
            for rx in cursor.pending {
                if !matches!(rx.await, Ok(Ok(()))) {
                    failed = true;
                }
            }
            if failed {
                // The checkpoint the consuming peer expects is missing; the
                // peer observes it as a short hit and rejects the handoff.
                // The producing scheduler withholds its KV-ready response
                // until these saves confirm (glm52 flush-on-finish) — that
                // wiring lands with the P/D migration; until then the failure
                // is counted and logged loudly.
                stats.handoff_failed.fetch_add(1, Ordering::Relaxed);
                log::error!("kv-store retire: handoff save failed; peer will miss this checkpoint");
            }
            release_logged(&mut kv);
        });
    }

    fn rank(&self, rank: usize) -> Option<&RankState> {
        self.ranks.get(&rank)
    }
}

fn release_logged(kv: &mut RequestKv) {
    if let Err(err) = kv.release() {
        // Blocks still return via assignment RAII when the KV drops; the
        // explicit release only failed to run from a clean state.
        log::warn!("kv-store retire: KV release failed (blocks return via RAII): {err:#}");
    }
}

// The store's whole concurrency story rests on these being thread-safe;
// fail at compile time, not in a review.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<KvStore>();
};
