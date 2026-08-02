//! [`KvStore`] itself: the resolve/seal/retire orchestration over the
//! per-rank surfaces frozen by [`crate::KvStoreBuilder`].

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use openinfer_engine::engine::KvPrefix;
use openinfer_kv_cache::BlockPool;
use openinfer_kv_cache::RequestKv;
use tokio::sync::oneshot;
use tokio::time::Instant;

use crate::CacheScope;
use crate::CancelProbe;
use crate::DegradeReason;
use crate::HostTier;
use crate::KvStoreStats;
use crate::ResolvePolicy;
use crate::SaveClass;
use crate::SaveCursor;
use crate::TierQuery;

/// One rank's KV surfaces. The pool is the same logical pool the rank's
/// scheduler allocates from — kvbm's `BlockManager` is internally
/// synchronized (save guards already cross threads today), so the resolve
/// task allocating from it is an arbitration question, answered by the floor.
pub(crate) struct RankState {
    pub(crate) pool: Arc<BlockPool>,
    pub(crate) tier: Option<Arc<dyn HostTier>>,
    /// Blocks admission has promised to admitted requests; resolve-side
    /// allocation yields to it (fail-soft to a smaller hit).
    pub(crate) floor: AtomicUsize,
    /// Blocks pinned by in-flight saves (both classes) — physically
    /// unallocatable until their D2H lands. Admission subtracts this from
    /// its budget. `Arc`: the per-save watcher tasks decrement it after the
    /// store's borrow ends.
    pub(crate) pinned: Arc<AtomicUsize>,
    /// The `Cacheable` subset of `pinned` — the shed budget gates on this
    /// alone, so Handoff traffic cannot starve cacheable saves.
    pub(crate) cacheable_pinned: Arc<AtomicUsize>,
    /// Pin ceiling for `Cacheable` saves (from
    /// [`crate::KvStoreBuilder::with_cacheable_pin_percent`] of the pool).
    pub(crate) cacheable_pin_budget: usize,
}

/// The process-wide KV store: one instance, `Arc`-shared. Knows token
/// prefixes, pools, and tiers — not engines, inboxes, or requests. Built by
/// [`crate::KvStoreBuilder`]; the rank table is immutable after build.
pub struct KvStore {
    pub(crate) runtime: tokio::runtime::Handle,
    pub(crate) requery_interval: Duration,
    pub(crate) resolve_deadline: Duration,
    pub(crate) ranks: HashMap<usize, RankState>,
    pub(crate) stats: Arc<KvStoreStats>,
}

impl KvStore {
    /// Blocks admission has promised on `rank`; resolve-side allocations
    /// yield to this watermark. The scheduler updates it as it admits and
    /// retires.
    pub fn set_admission_floor(&self, rank: usize, blocks: usize) {
        self.rank(rank).floor.store(blocks, Ordering::Release);
    }

    /// Blocks pinned by `rank`'s in-flight saves; admission subtracts this
    /// from its usable budget (the glm52 `pinned_blocks` discipline).
    pub fn pinned_blocks(&self, rank: usize) -> usize {
        self.rank(rank).pinned.load(Ordering::Acquire)
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
        policy: ResolvePolicy,
        cancel: &dyn CancelProbe,
    ) -> KvPrefix {
        self.stats.resolves.fetch_add(1, Ordering::Relaxed);
        let state = self.rank(rank);
        let pool = &state.pool;
        let block_size = pool.block_size();
        // The `-1`: matching always leaves at least one prompt token uncached
        // (the final chunk must forward to emit the first token), so a
        // full-prompt hit is never usable and the last partial/full block is
        // outside the reusable prefix.
        let cacheable_blocks = prompt_tokens.len().saturating_sub(1) / block_size;

        let finish = |probe: openinfer_kv_cache::PrefixProbe| {
            let hit_blocks = probe.held_blocks().min(cacheable_blocks);
            if hit_blocks == 0 {
                return KvPrefix::none();
            }
            self.stats.resolve_hits.fetch_add(1, Ordering::Relaxed);
            KvPrefix::resolved(hit_blocks * block_size, rank, Box::new(probe))
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

        // Host-tier query, re-queried until the hit is BOTH host-ready and
        // pool-admittable, bounded by one deadline. Waiting on pool pressure
        // (instead of instantly degrading) is deliberate: if the pool cannot
        // spare the prefix's blocks now, admission could not take the request
        // now either — it waits regardless, and waiting here buys the cheap
        // prefill while degrading would buy a full recompute for the same
        // wait. The lease is released before every pause (host pins + TTL
        // must not ride out a pool wait; a re-query re-establishes it
        // instantly from the host tier).
        let deadline = Instant::now() + self.resolve_deadline;
        let (hit, reservation) = loop {
            if cancel.is_cancelled() {
                self.stats.record_degrade(req_id, DegradeReason::Cancelled);
                return KvPrefix::none();
            }
            let query = tokio::time::timeout_at(
                deadline,
                tier.query(req_id, host_hashes.clone(), policy.wait_for_full_hit),
            );
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
                Ok(Ok(TierQuery::Miss)) => {
                    // Plain restore: a miss is a cold cache, conclude. Full-
                    // hit mode: the producer's registration may not have
                    // landed yet — keep waiting under the deadline.
                    if !policy.wait_for_full_hit {
                        return finish(probe);
                    }
                    if Instant::now() >= deadline {
                        self.stats
                            .record_degrade(req_id, DegradeReason::DeadlineExceeded);
                        return finish(probe);
                    }
                    tokio::time::sleep(self.requery_interval).await;
                }
                Ok(Ok(TierQuery::Loading)) => {
                    if Instant::now() >= deadline {
                        self.stats
                            .record_degrade(req_id, DegradeReason::DeadlineExceeded);
                        return finish(probe);
                    }
                    tokio::time::sleep(self.requery_interval).await;
                }
                Ok(Ok(TierQuery::Hit(hit))) => {
                    // Floor gate: resolve-side allocation yields to
                    // admission's promises. The lease is all-or-nothing, so
                    // an unplaceable hit is declined (release, not TTL-parked)
                    // and retried after a pause.
                    let available = pool.available_blocks();
                    let floor = state.floor.load(Ordering::Acquire);
                    if available.saturating_sub(floor) >= hit.blocks {
                        if let Some(reservation) = pool.reserve_loaded_blocks(hit.blocks) {
                            break (hit, reservation);
                        }
                    }
                    tier.release(hit);
                    if Instant::now() >= deadline {
                        self.stats
                            .record_degrade(req_id, DegradeReason::PoolPressure);
                        return finish(probe);
                    }
                    tokio::time::sleep(self.requery_interval).await;
                }
            }
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
                // The reservation now lives with the detached task until the
                // DMA settles; if it never does, those blocks are gone for
                // good — count it so pool-drain from hung DMAs is visible.
                self.stats.loads_abandoned.fetch_add(1, Ordering::Relaxed);
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
        let state = self.rank(rank);
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
        // seal (or the retire) retries once pressure clears. The gate reads
        // the cacheable-only counter: Handoff traffic neither sheds nor
        // consumes this budget (its backpressure is admission reading
        // `pinned_blocks`, which counts both classes).
        if class == SaveClass::Cacheable
            && state.cacheable_pinned.load(Ordering::Acquire) + count > state.cacheable_pin_budget
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
        if class == SaveClass::Cacheable {
            state.cacheable_pinned.fetch_add(count, Ordering::AcqRel);
        }
        self.stats.saves_submitted.fetch_add(1, Ordering::Relaxed);
        let save = tier.save(ids, hashes, Box::new(guards));

        let done_tx = if class == SaveClass::Handoff {
            let (tx, rx) = oneshot::channel();
            cursor.pending.push(rx);
            Some(tx)
        } else {
            None
        };
        let pinned = Arc::clone(&state.pinned);
        let cacheable_pinned =
            (class == SaveClass::Cacheable).then(|| Arc::clone(&state.cacheable_pinned));
        let stats = Arc::clone(&self.stats);
        self.runtime.spawn(async move {
            let result = save.await;
            pinned.fetch_sub(count, Ordering::AcqRel);
            if let Some(cacheable_pinned) = cacheable_pinned {
                cacheable_pinned.fetch_sub(count, Ordering::AcqRel);
            }
            let outcome = match result {
                Ok(()) => Ok(()),
                Err(err) => {
                    stats.saves_failed.fetch_add(1, Ordering::Relaxed);
                    log::warn!("kv-store save of {count} blocks failed: {err:#}");
                    Err(format!("{err:#}"))
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

    /// The rank table is frozen at build, so an unknown rank is a wiring
    /// bug — masking it (e.g. silently no-op'ing a Handoff seal) would
    /// violate the no-silent-drop contract. Fail fast instead.
    fn rank(&self, rank: usize) -> &RankState {
        self.ranks.get(&rank).unwrap_or_else(|| {
            panic!("kv-store: rank {rank} not registered (rank table is frozen at build)")
        })
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
