//! [`KvStore`] itself: the resolve/seal/retire orchestration over the
//! per-rank surfaces frozen by [`crate::KvStoreBuilder`].

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use openinfer_engine::engine::KvPrefix;
use tokio::sync::oneshot;
use tokio::time::Instant;

use crate::BlockPool;
use crate::CacheScope;
use crate::CancelProbe;
use crate::DegradeReason;
use crate::KvStoreStats;
use crate::PrefixProbe;
use crate::RequestKv;
use crate::ResolvePolicy;
use crate::SaveClass;
use crate::SaveCursor;
use crate::tier::HostTier;
use crate::tier::LeasedBlocks;
use crate::tier::TierQuery;

/// One rank's KV surfaces. `pool` is the same logical pool the rank's
/// scheduler allocates from; `BlockPool` is internally synchronized, so the
/// resolve task may allocate concurrently with the scheduler without a lock.
pub(crate) struct RankState {
    pub(crate) pool: Arc<BlockPool>,
    pub(crate) tier: Option<Arc<dyn HostTier>>,
    /// Blocks pinned by in-flight saves (both classes) — physically
    /// unallocatable until their D2H lands. Admission subtracts this from
    /// its budget. `Arc`: the per-save watcher tasks decrement it after the
    /// store's borrow ends.
    pub(crate) pinned: Arc<AtomicUsize>,
    /// Restore H2D tasks still in flight (including loads whose awaiter
    /// abandoned them at the deadline — the detached task owns the
    /// reservation until the DMA settles). Teardown must drain this to zero
    /// before the GPU arenas are freed.
    pub(crate) loads_pending: Arc<AtomicUsize>,
}

/// Terminal failure of [`KvStore::resolve_keyed_block`]. The variants differ
/// in exactly one contract: whether the H2D into the caller's destination
/// page may still be in flight.
pub enum KeyedFetchError {
    /// The tier settled (or the load was never submitted): no DMA targets
    /// the destination page, so the caller may free it immediately.
    Settled(anyhow::Error),
    /// The resolve deadline passed with the load unsettled: a detached task
    /// still owns a DMA targeting the caller's destination page. The caller
    /// must move whatever pins that page into `parking` before releasing the
    /// KV it belongs to.
    Abandoned {
        error: anyhow::Error,
        parking: KeyedLoadParking,
    },
}

/// One-shot slot for the destination pin of an abandoned keyed-tail load.
/// The detached load task holds the other end and drops whatever is parked
/// here only after the tier settles — the invariant this carries: an
/// abandoned keyed-tail DMA never writes a freed page.
pub struct KeyedLoadParking(oneshot::Sender<Box<dyn Any + Send>>);

impl KeyedLoadParking {
    /// Hand the destination pin to the detached load task. If the load
    /// settled between the timeout and this call, the pin comes straight
    /// back and drops here — safe either way, the DMA is over.
    pub fn park(self, pin: Box<dyn Any + Send>) {
        let _ = self.0.send(pin);
    }
}

/// A tier hit whose lease releases on drop unless the caller takes it. The
/// keyed-query task returns its hit through this guard, which carries the
/// invariant: a keyed query, once submitted, always settles and always
/// releases any lease the caller will never consume — a wait abandoned at
/// the deadline drops the task's output unread, and the drop releases.
struct LeaseGuard {
    tier: Arc<dyn HostTier>,
    hit: Option<LeasedBlocks>,
}

impl LeaseGuard {
    fn blocks(&self) -> usize {
        self.hit
            .as_ref()
            .expect("guard holds its hit until taken")
            .blocks
    }

    /// Consume the guard, taking over the lease (no release on drop).
    fn take(mut self) -> LeasedBlocks {
        self.hit.take().expect("guard holds its hit until taken")
    }
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        if let Some(hit) = self.hit.take() {
            self.tier.release(hit);
        }
    }
}

/// Decrements a counter when dropped — tied to the load task's future so the
/// count falls even for abandoned (timeout-detached) tasks.
struct DecOnDrop(Arc<AtomicUsize>);

impl Drop for DecOnDrop {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
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
    /// Blocks pinned by `rank`'s in-flight saves; admission subtracts this
    /// from its usable budget, so it never admits a request against pages a
    /// pending D2H copy still needs.
    pub fn pinned_blocks(&self, rank: usize) -> usize {
        self.rank(rank).pinned.load(Ordering::Acquire)
    }

    pub fn stats(&self) -> &KvStoreStats {
        &self.stats
    }

    /// The whole read path. Resolves `prompt_tokens`' cached prefix on
    /// `rank`: GPU radix probe, host-tier query (re-query while a deeper tier
    /// fetches, bounded by the resolve deadline), page reservation, load, and
    /// registration into the radix — so the eventual `match_and_add_prefix`
    /// on the scheduler thread reuses the full prefix.
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

        let finish = |mut probe: PrefixProbe| {
            let hit_blocks = probe.held_blocks().min(cacheable_blocks);
            if hit_blocks == 0 {
                return KvPrefix::none();
            }
            // Credit/pin parity: admission credits the hold with exactly
            // `hit_tokens / block_size` blocks, so the hold must pin exactly
            // that many. A page-aligned full match holds one block past the
            // cacheable cap; keeping it pinned would let a front at exact
            // pool capacity wait forever on the very block its own hold pins.
            probe.truncate_held(hit_blocks);
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
                    // The lease is all-or-nothing, so a hit that doesn't fit
                    // the pool right now is declined (release, not TTL-parked)
                    // and retried after a pause.
                    if let Some(reservation) = pool.reserve_loaded_blocks(hit.blocks) {
                        break (hit, reservation);
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
        state.loads_pending.fetch_add(1, Ordering::AcqRel);
        let dec = DecOnDrop(Arc::clone(&state.loads_pending));
        let join = self.runtime.spawn(async move {
            let _dec = dec;
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
        let save = tier.save(ids, hashes, guards);

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
            let result = save.await;
            pinned.fetch_sub(count, Ordering::AcqRel);
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
    /// releases immediately; fire-and-forget D2H stays safe because the save
    /// guards pin the source pages independently of the release. With
    /// `Handoff` saves pending, the KV instead stays parked — registered and
    /// resident — until they settle. Never blocks the caller.
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
                // The checkpoint the consuming peer expects is missing; it
                // sees this handoff as a short hit and rejects it. The
                // producing scheduler should eventually withhold its
                // KV-ready response until these saves confirm — that wiring
                // is not in place yet, so for now the failure is counted and
                // logged loudly.
                stats.handoff_failed.fetch_add(1, Ordering::Relaxed);
                log::error!("kv-store retire: handoff save failed; peer will miss this checkpoint");
            }
            release_logged(&mut kv);
        });
    }

    /// Write-pipeline visibility barrier for `rank`: on return, every
    /// [`Self::seal`]/[`Self::retire`] save submitted before this call is
    /// query-visible to this tier (and, with P2P, MetaServer-registered for
    /// peers). The `Handoff` save completion callback answers "data landed";
    /// this barrier answers "findable by hash". A rank without a tier has
    /// nothing to flush.
    pub async fn flush_saves(&self, rank: usize) -> anyhow::Result<()> {
        match self.rank(rank).tier.as_ref() {
            Some(tier) => tier.flush().await,
            None => Ok(()),
        }
    }

    /// Save the request's LAST currently-held page under an explicit content
    /// key — the disaggregated-prefill tail: a partial page that has no
    /// lineage hash (it never sealed), keyed instead by a digest the handoff
    /// envelope carries to the consuming peer.
    ///
    /// The save's completion joins `cursor.pending`, so the following
    /// [`Self::retire`] with [`SaveClass::Handoff`] parks the KV until the
    /// D2H lands — the parked KV itself is the reuse pin for the unsealed
    /// page (it has no `KvBlockGuard`). Contract: call this only between the
    /// step that wrote the tail and the retire; the KV must not release in
    /// between.
    pub fn seal_keyed(&self, rank: usize, kv: &RequestKv, key: [u8; 16], cursor: &mut SaveCursor) {
        let state = self.rank(rank);
        let Some(tier) = state.tier.as_ref() else {
            return;
        };
        let Some(&tail_page) = kv.current_page_indices().last() else {
            return;
        };
        self.stats.saves_submitted.fetch_add(1, Ordering::Relaxed);
        let save = tier.save(vec![tail_page], vec![key.to_vec()], Vec::new());
        let (tx, rx) = oneshot::channel();
        cursor.pending.push(rx);
        let stats = Arc::clone(&self.stats);
        self.runtime.spawn(async move {
            let outcome = match save.await {
                Ok(()) => Ok(()),
                Err(err) => {
                    stats.saves_failed.fetch_add(1, Ordering::Relaxed);
                    log::warn!("kv-store keyed save failed: {err:#}");
                    Err(format!("{err:#}"))
                }
            };
            let _ = tx.send(outcome);
        });
    }

    /// Fetch one block stored under an explicit content key into the GPU
    /// page `dst_page_id` — the decode side of the tail handoff. The
    /// destination page belongs to the caller's scheduled `RequestKv`, so it
    /// never enters the radix (no lineage hash to register under).
    ///
    /// The query phase re-queries under the resolve deadline (the producer's
    /// registration may not have landed); the H2D wait is bounded by the
    /// same deadline. Past it the wait is abandoned — the DMA is not — and
    /// the error carries a [`KeyedLoadParking`] the caller must move the
    /// destination's pin into, because the destination page belongs to the
    /// caller's scheduled `RequestKv` and the store holds no handle to it.
    pub async fn resolve_keyed_block(
        &self,
        rank: usize,
        req_id: &str,
        key: [u8; 16],
        dst_page_id: i32,
    ) -> Result<(), KeyedFetchError> {
        let state = self.rank(rank);
        let tier = state.tier.as_ref().ok_or_else(|| {
            KeyedFetchError::Settled(anyhow::anyhow!(
                "rank {rank} has no host tier for a keyed fetch"
            ))
        })?;
        let deadline = Instant::now() + self.resolve_deadline;
        let expired = || {
            KeyedFetchError::Settled(anyhow::anyhow!(
                "keyed fetch deadline: key never became resident"
            ))
        };
        // Dropping `tier.query`'s future detaches only the observer — the
        // tier's work is already submitted and may still settle into a
        // lease-carrying hit. So each query runs in a task owning the tier:
        // once submitted it always runs to settlement, and a hit comes back
        // wrapped in a [`LeaseGuard`], which releases the lease unless the
        // caller takes it — including a wait abandoned at the deadline,
        // where the task's output drops unread. No abandoned query can pin
        // host blocks until the lease TTL.
        let hit = loop {
            let query_tier = Arc::clone(tier);
            let query_req_id = req_id.to_string();
            let hashes = vec![key.to_vec()];
            let join = self.runtime.spawn(async move {
                match query_tier.query(&query_req_id, hashes, true).await {
                    Ok(TierQuery::Hit(hit)) => Ok(Some(LeaseGuard {
                        tier: query_tier,
                        hit: Some(hit),
                    })),
                    Ok(TierQuery::Miss | TierQuery::Loading) => Ok(None),
                    Err(err) => Err(err),
                }
            });
            match tokio::time::timeout_at(deadline, join).await {
                Err(_elapsed) => return Err(expired()),
                Ok(Err(join_err)) => {
                    return Err(KeyedFetchError::Settled(anyhow::anyhow!(
                        "keyed fetch query task failed: {join_err}"
                    )));
                }
                Ok(Ok(Err(err))) => {
                    return Err(KeyedFetchError::Settled(err.context("keyed fetch query")));
                }
                Ok(Ok(Ok(Some(guard)))) if guard.blocks() == 1 => break guard.take(),
                Ok(Ok(Ok(Some(guard)))) => {
                    let blocks = guard.blocks();
                    drop(guard);
                    return Err(KeyedFetchError::Settled(anyhow::anyhow!(
                        "keyed fetch resolved {blocks} blocks for one key"
                    )));
                }
                Ok(Ok(Ok(None))) => {
                    if Instant::now() >= deadline {
                        return Err(expired());
                    }
                    tokio::time::sleep(self.requery_interval).await;
                }
            }
        };

        // Same bounded-abandonment shape as `resolve_prefix`: a spawned task
        // owns the load future (counted in `loads_pending` for the teardown
        // barrier), and the deadline abandons only the wait. The task drops
        // whatever the abandoning caller parks only after the tier settles.
        let load = tier.load(hit, vec![dst_page_id]);
        state.loads_pending.fetch_add(1, Ordering::AcqRel);
        let dec = DecOnDrop(Arc::clone(&state.loads_pending));
        let (park_tx, park_rx) = oneshot::channel::<Box<dyn Any + Send>>();
        let join = self.runtime.spawn(async move {
            let _dec = dec;
            let result = load.await;
            // The DMA has settled: releasing the parking end now frees any
            // parked destination pin, and makes a not-yet-sent park bounce
            // back to its caller — either way the page outlived the copy.
            drop(park_rx);
            result
        });
        match tokio::time::timeout_at(deadline, join).await {
            Ok(Ok(Ok(()))) => Ok(()),
            Ok(Ok(Err(err))) => Err(KeyedFetchError::Settled(err.context("keyed fetch load"))),
            Ok(Err(join_err)) => Err(KeyedFetchError::Settled(anyhow::anyhow!(
                "keyed fetch load task failed: {join_err}"
            ))),
            Err(_elapsed) => {
                self.stats.loads_abandoned.fetch_add(1, Ordering::Relaxed);
                Err(KeyedFetchError::Abandoned {
                    error: anyhow::anyhow!("keyed fetch deadline: tail load did not settle"),
                    parking: KeyedLoadParking(park_tx),
                })
            }
        }
    }

    /// Wait until every restore H2D on `rank` has settled — including loads
    /// whose resolve abandoned the wait at its deadline (their detached
    /// tasks own the destination reservations). Call before freeing the GPU
    /// arenas at teardown; bound it with a caller-side timeout, since a load
    /// against a hung tier never settles.
    pub async fn flush_loads(&self, rank: usize) {
        let pending = Arc::clone(&self.rank(rank).loads_pending);
        while pending.load(Ordering::Acquire) > 0 {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
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

#[cfg(test)]
mod tests {
    use pegaflow_core::QueryLeaseId;
    use tokio::sync::Notify;

    use super::*;
    use crate::BlockPool;
    use crate::KvBlockGuard;
    use crate::tier::HostTier;
    use crate::tier::LeasedBlocks;
    use crate::tier::TierFuture;
    use crate::tier::TierQuery;

    /// A tier whose query always hits but whose load stalls until the test
    /// fires `release_load` — the hung-storage-worker shape the keyed-tail
    /// deadline exists for.
    struct StallLoadTier {
        release_load: Arc<Notify>,
    }

    impl HostTier for StallLoadTier {
        fn query(
            &self,
            _req_id: &str,
            block_hashes: Vec<Vec<u8>>,
            _wait_full: bool,
        ) -> TierFuture<anyhow::Result<TierQuery>> {
            let blocks = block_hashes.len();
            Box::pin(std::future::ready(Ok(TierQuery::Hit(LeasedBlocks {
                blocks,
                lease: QueryLeaseId::fresh(),
            }))))
        }

        fn load(
            &self,
            _hit: LeasedBlocks,
            _dst_page_ids: Vec<i32>,
        ) -> TierFuture<anyhow::Result<()>> {
            let release = Arc::clone(&self.release_load);
            Box::pin(async move {
                release.notified().await;
                Ok(())
            })
        }

        fn release(&self, _hit: LeasedBlocks) {}

        fn save(
            &self,
            _block_ids: Vec<i32>,
            _block_hashes: Vec<Vec<u8>>,
            _guards: Vec<KvBlockGuard>,
        ) -> TierFuture<anyhow::Result<()>> {
            Box::pin(std::future::ready(Ok(())))
        }

        fn flush(&self) -> TierFuture<anyhow::Result<()>> {
            Box::pin(std::future::ready(Ok(())))
        }
    }

    /// A tier whose query stalls until the test fires `release_query`, then
    /// answers with a lease-carrying hit — the shape where the caller's
    /// deadline passes while the tier still owes a settlement. `released`
    /// counts every lease handed back.
    struct StallQueryTier {
        release_query: Arc<Notify>,
        released: Arc<AtomicUsize>,
    }

    impl HostTier for StallQueryTier {
        fn query(
            &self,
            _req_id: &str,
            block_hashes: Vec<Vec<u8>>,
            _wait_full: bool,
        ) -> TierFuture<anyhow::Result<TierQuery>> {
            let release = Arc::clone(&self.release_query);
            let blocks = block_hashes.len();
            Box::pin(async move {
                release.notified().await;
                Ok(TierQuery::Hit(LeasedBlocks {
                    blocks,
                    lease: QueryLeaseId::fresh(),
                }))
            })
        }

        fn load(
            &self,
            _hit: LeasedBlocks,
            _dst_page_ids: Vec<i32>,
        ) -> TierFuture<anyhow::Result<()>> {
            Box::pin(std::future::ready(Ok(())))
        }

        fn release(&self, _hit: LeasedBlocks) {
            self.released.fetch_add(1, Ordering::AcqRel);
        }

        fn save(
            &self,
            _block_ids: Vec<i32>,
            _block_hashes: Vec<Vec<u8>>,
            _guards: Vec<KvBlockGuard>,
        ) -> TierFuture<anyhow::Result<()>> {
            Box::pin(std::future::ready(Ok(())))
        }

        fn flush(&self) -> TierFuture<anyhow::Result<()>> {
            Box::pin(std::future::ready(Ok(())))
        }
    }

    fn tier_store(runtime: tokio::runtime::Handle, tier: Arc<dyn HostTier>) -> KvStore {
        let mut ranks = HashMap::new();
        ranks.insert(
            0,
            RankState {
                pool: Arc::new(BlockPool::new(16, 8)),
                tier: Some(tier),
                pinned: Arc::new(AtomicUsize::new(0)),
                loads_pending: Arc::new(AtomicUsize::new(0)),
            },
        );
        KvStore {
            runtime,
            requery_interval: Duration::from_millis(1),
            resolve_deadline: Duration::from_millis(50),
            ranks,
            stats: Arc::new(KvStoreStats::default()),
        }
    }

    fn stalled_store(runtime: tokio::runtime::Handle, release_load: Arc<Notify>) -> KvStore {
        tier_store(runtime, Arc::new(StallLoadTier { release_load }))
    }

    /// The keyed-tail invariant: a stalled load abandons the WAIT at the
    /// deadline (the resolver task is never wedged), and a pin parked with
    /// the abandoned load is dropped only after the tier settles — an
    /// abandoned keyed-tail DMA never writes a freed page.
    #[test]
    fn keyed_tail_timeout_parks_the_destination_pin_until_the_load_settles() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let release_load = Arc::new(Notify::new());
        let store = stalled_store(rt.handle().clone(), Arc::clone(&release_load));

        let outcome = rt.block_on(store.resolve_keyed_block(0, "tail", [7u8; 16], 3));
        let Err(KeyedFetchError::Abandoned { parking, .. }) = outcome else {
            panic!("a stalled tail load must abandon at the deadline");
        };
        assert_eq!(
            store.rank(0).loads_pending.load(Ordering::Acquire),
            1,
            "the detached task still owns the load"
        );
        assert_eq!(store.stats.loads_abandoned.load(Ordering::Relaxed), 1);

        let pin = Arc::new(());
        let watch = Arc::clone(&pin);
        parking.park(Box::new(pin));
        assert_eq!(
            Arc::strong_count(&watch),
            2,
            "the parked pin stays alive while the load is unsettled"
        );

        release_load.notify_one();
        rt.block_on(async {
            tokio::time::timeout(Duration::from_secs(5), store.flush_loads(0)).await
        })
        .expect("the load settles once the tier releases it");
        assert_eq!(
            Arc::strong_count(&watch),
            1,
            "the settled load released the parked pin"
        );
    }

    /// The keyed-query invariant: a query stalled past the deadline abandons
    /// only the wait, and when the tier later settles with a hit, the
    /// detached task releases the lease the caller will never consume — a
    /// storage stall never pins host blocks until the lease TTL.
    #[test]
    fn keyed_query_timeout_releases_the_late_hits_lease() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let release_query = Arc::new(Notify::new());
        let released = Arc::new(AtomicUsize::new(0));
        let store = tier_store(
            rt.handle().clone(),
            Arc::new(StallQueryTier {
                release_query: Arc::clone(&release_query),
                released: Arc::clone(&released),
            }),
        );

        let outcome = rt.block_on(store.resolve_keyed_block(0, "tail", [7u8; 16], 3));
        let Err(KeyedFetchError::Settled(err)) = outcome else {
            panic!("a stalled query must fail at the deadline");
        };
        assert!(
            format!("{err:#}").contains("deadline"),
            "the caller sees the deadline error: {err:#}"
        );
        assert_eq!(
            released.load(Ordering::Acquire),
            0,
            "nothing to release while the query is unsettled"
        );

        release_query.notify_one();
        rt.block_on(async {
            tokio::time::timeout(Duration::from_secs(5), async {
                while released.load(Ordering::Acquire) == 0 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
        })
        .expect("the settled query releases the unconsumed lease");
        assert_eq!(released.load(Ordering::Acquire), 1);
    }
}
