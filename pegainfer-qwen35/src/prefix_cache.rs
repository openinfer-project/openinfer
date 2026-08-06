//! Joint full-attention KV and recurrent/conv prefix cache for Qwen3.5.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::Result;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_kv_cache::KvBlockGuard;
use pegainfer_kv_cache::KvBuffer;
use pegainfer_kv_cache::KvCacheManager;
use pegainfer_kv_cache::KvView;
use pegainfer_kv_cache::RequestKv;

use crate::config::Config35;
use crate::recurrent_state::RecurrentState;

/// Token interval at which a complete recurrent/conv snapshot may be cached.
pub(crate) const SNAPSHOT_STRIDE_TOKENS: usize = 256;

/// Content-addressed identity of one complete hybrid-model prefix boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PrefixBoundaryKey {
    /// Canonical full-attention KV lineage hash for this prefix.
    pub(crate) sequence_hash: [u8; 16],
    /// Exclusive token position represented by both KV and recurrent state.
    pub(crate) boundary_tokens: usize,
}

/// Cumulative counters for joint KV/recurrent prefix-cache activity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PrefixCacheStats {
    /// Requests that restored both KV and a recurrent snapshot.
    pub(crate) joint_hits: u64,
    /// Prompt tokens reused across all joint hits.
    pub(crate) joint_hit_tokens: u64,
    /// Requests with eligible resident KV but no matching snapshot.
    pub(crate) kv_only_fallbacks: u64,
    /// Individual eligible boundaries without a matching snapshot.
    pub(crate) snapshot_misses: u64,
    /// New recurrent snapshots published.
    pub(crate) inserts: u64,
    /// Published snapshots replaced by LRU insertion.
    pub(crate) evictions: u64,
    /// Successful joint restore time, including lookup, attach, copy, and checks.
    pub(crate) restore_ns: u64,
}

/// One reusable Qwen3.5 prefix entry.
///
/// The entry is the ownership boundary for the recurrent snapshot slot
/// and the leading KV blocks. Dropping an entry therefore drops its KV lease as well.
struct PrefixEntry {
    /// Identically numbered physical recurrent snapshot on every rank.
    recurrent_slot: usize,
    /// Strong pins for every KV block through the entry boundary.
    #[allow(dead_code)] // ownership is the use: dropping the entry drops the lease
    kv_lease: Vec<KvBlockGuard>,
    /// Active restore guards preventing this entry from being evicted.
    pin_count: Arc<AtomicUsize>,
    /// Logical timestamp used to select an unpinned LRU victim.
    last_used: u64,
}

/// RAII pin on one prefix cache entry while its recurrent state is being restored.
pub(crate) struct PrefixGuard {
    /// Token boundary represented by the pinned entry.
    boundary: usize,
    /// Rank-local physical snapshot slot selected by the directory.
    recurrent_slot: usize,
    /// Shared count consulted by insertion before choosing a victim.
    pin_count: Arc<AtomicUsize>,
    /// Start time used for restore latency accounting.
    started: Instant,
}

impl PrefixGuard {
    pub(crate) fn boundary(&self) -> usize {
        self.boundary
    }

    pub(crate) fn recurrent_slot(&self) -> usize {
        self.recurrent_slot
    }
}

impl Drop for PrefixGuard {
    fn drop(&mut self) {
        let previous = self.pin_count.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "prefix guard pin underflow");
    }
}

/// Mutable prefix-cache state shared by every execution rank.
struct PrefixCacheState {
    /// Published boundary key to its complete joint entry.
    entries: HashMap<PrefixBoundaryKey, PrefixEntry>,
    /// Unpublished slots available without eviction.
    free_slots: Vec<usize>,
    /// Total number of preallocated physical snapshot slots.
    slot_count: usize,
    /// Monotonic logical time for LRU ordering.
    clock: u64,
}

impl PrefixCacheState {
    fn new(slot_count: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(slot_count),
            free_slots: (0..slot_count).rev().collect(),
            slot_count,
            clock: 0,
        }
    }

    /// Number of currently published joint entries.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Number of preallocated recurrent-state slots.
    fn capacity(&self) -> usize {
        self.slot_count
    }

    /// Advance the non-zero logical clock used by the LRU policy.
    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }

    /// Look up `key`, refresh its LRU timestamp, and pin its entry.
    fn lookup(&mut self, key: PrefixBoundaryKey, started: Instant) -> Option<PrefixGuard> {
        let last_used = self.tick();
        let entry = self.entries.get_mut(&key)?;
        entry.last_used = last_used;
        entry.pin_count.fetch_add(1, Ordering::AcqRel);
        Some(PrefixGuard {
            boundary: key.boundary_tokens,
            recurrent_slot: entry.recurrent_slot,
            pin_count: Arc::clone(&entry.pin_count),
            started,
        })
    }

    /// Reserve a free or unpinned LRU slot without publishing the new entry.
    fn reserve(&mut self, key: PrefixBoundaryKey) -> Option<PrefixReservation> {
        let last_used = self.tick();
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = last_used;
            return None;
        }

        let (slot, evicted) = if let Some(slot) = self.free_slots.pop() {
            (slot, false)
        } else {
            let (&victim_key, slot) = self
                .entries
                .iter()
                .filter(|(_, entry)| entry.pin_count.load(Ordering::Acquire) == 0)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, entry)| (key, entry.recurrent_slot))?;
            let evicted = self
                .entries
                .remove(&victim_key)
                .expect("LRU victim must still be present");
            debug_assert_eq!(evicted.recurrent_slot, slot);
            (slot, true)
        };

        Some(PrefixReservation {
            recurrent_slot: slot,
            key,
            replaced: evicted,
        })
    }

    /// Publish one complete entry only after every rank has enqueued its
    /// physical recurrent-state copy.
    fn publish(&mut self, reservation: PrefixReservation, kv_lease: Vec<KvBlockGuard>) {
        let last_used = self.tick();
        let previous = self.entries.insert(
            reservation.key,
            PrefixEntry {
                recurrent_slot: reservation.recurrent_slot,
                kv_lease,
                pin_count: Arc::new(AtomicUsize::new(0)),
                last_used,
            },
        );
        debug_assert!(previous.is_none());
    }

    /// Return an unpublished slot after a physical copy failure.
    fn abort(&mut self, reservation: PrefixReservation) {
        debug_assert!(!self.entries.contains_key(&reservation.key));
        self.free_slots.push(reservation.recurrent_slot);
    }
}

/// One pending central-directory insertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PrefixReservation {
    recurrent_slot: usize,
    key: PrefixBoundaryKey,
    /// Whether this insertion replaced an unpinned joint entry.
    replaced: bool,
}

impl PrefixReservation {
    pub(crate) fn recurrent_slot(self) -> usize {
        self.recurrent_slot
    }

    fn was_replacement(self) -> bool {
        self.replaced
    }
}

/// Rank-local physical recurrent/conv snapshot allocations.
pub(crate) struct RecurrentStateStore {
    slots: Vec<RecurrentState>,
}

impl RecurrentStateStore {
    pub(crate) fn new(ctx: &DeviceContext, config: &Config35, slot_count: usize) -> Result<Self> {
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(RecurrentState::new(ctx, config)?);
        }
        Ok(Self { slots })
    }

    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn save(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        src: &RecurrentState,
    ) -> Result<()> {
        let dst = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("snapshot slot {slot} out of range"))?;
        dst.copy_from(ctx, src)
    }

    pub(crate) fn restore(
        &self,
        ctx: &DeviceContext,
        slot: usize,
        dst: &mut RecurrentState,
    ) -> Result<()> {
        let src = self
            .slots
            .get(slot)
            .ok_or_else(|| anyhow::anyhow!("snapshot slot {slot} out of range"))?;
        dst.copy_from(ctx, src)
    }
}

/// The only Qwen3.5 scheduler interface allowed to reconcile paged KV with
/// recurrent/conv state.
pub(crate) struct Qwen35PrefixCache {
    /// Logical block pool paired with the full-attention GPU KV buffer.
    kv: KvCacheManager,
    /// Prefix key, slot, pin, LRU, and KV-lease ownership for reusable entries.
    state: PrefixCacheState,
    /// Whether at least one joint snapshot can be retained.
    enabled: bool,
    /// Scheduler-thread-owned cumulative metrics.
    stats: PrefixCacheStats,
}

impl Qwen35PrefixCache {
    /// Build the joint coordinator for `snapshot_slots` per-rank allocations.
    pub(crate) fn new(kv: KvCacheManager, snapshot_slots: usize) -> Result<Self> {
        anyhow::ensure!(
            SNAPSHOT_STRIDE_TOKENS.is_multiple_of(kv.pool().block_size()),
            "Qwen3.5 snapshot stride {SNAPSHOT_STRIDE_TOKENS} must be a multiple of KV block size {}",
            kv.pool().block_size()
        );
        Ok(Self {
            kv,
            state: PrefixCacheState::new(snapshot_slots),
            enabled: snapshot_slots > 0,
            stats: PrefixCacheStats::default(),
        })
    }

    /// Logical KV block pool used for request allocation and admission.
    pub(crate) fn pool(&self) -> &pegainfer_kv_cache::BlockPool {
        self.kv.pool()
    }

    /// Rank-0 physical full-attention KV storage indexed by [`Self::pool`].
    pub(crate) fn buffer(&self) -> &KvBuffer {
        self.kv.buffer()
    }

    /// Whether joint prefix reuse and snapshot publication are enabled.
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Total number of preallocated recurrent snapshot slots.
    pub(crate) fn snapshot_slots(&self) -> usize {
        self.state.capacity()
    }

    /// Number of snapshot slots that currently have a published key.
    pub(crate) fn snapshot_occupancy(&self) -> usize {
        self.state.len()
    }

    /// Return a point-in-time copy of cumulative cache metrics.
    pub(crate) fn stats(&self) -> PrefixCacheStats {
        self.stats
    }

    /// Create request-local KV state and select the longest joint prefix.
    ///
    /// If a joint prefix is found, the KV blocks are attached to the request
    /// and a guard is returned to prevent eviction until the recurrent state
    /// is copied (see `Qwen35PrefixCache::finish_restore`).
    /// If no joint prefix is found, the request is still created and returned.
    pub(crate) fn begin_request(
        &mut self,
        prompt_tokens: &[u32],
        max_output_tokens: usize,
        lora_name: Option<&str>,
        allow_match: bool,
    ) -> Result<(RequestKv, Option<PrefixGuard>)> {
        let mut request =
            self.kv
                .pool()
                .new_request(prompt_tokens.to_vec(), max_output_tokens, lora_name);
        if !self.enabled || !allow_match {
            return Ok((request, None));
        }

        let probe = self
            .kv
            .pool()
            .probe_prefix(prompt_tokens.to_vec(), lora_name);
        let resident_tokens = probe.reusable_blocks() * self.kv.pool().block_size();
        let mut saw_eligible_kv = false;
        let started = Instant::now();
        for boundary in eligible_boundaries(resident_tokens, SNAPSHOT_STRIDE_TOKENS) {
            saw_eligible_kv = true;
            let Some(sequence_hash) = probe.boundary_hash(boundary) else {
                continue;
            };
            let key = PrefixBoundaryKey {
                sequence_hash,
                boundary_tokens: boundary,
            };
            let Some(guard) = self.state.lookup(key, started) else {
                self.stats.snapshot_misses += 1;
                continue;
            };

            let max_blocks = boundary / self.kv.pool().block_size();
            let attached = match request.match_and_add_prefix_up_to(self.kv.pool(), max_blocks) {
                Ok(attached) => attached,
                Err(error) => {
                    let _ = request.release();
                    return Err(error);
                }
            };
            if attached != boundary {
                let _ = request.release();
                anyhow::bail!(
                    "Qwen3.5 joint prefix attach selected {boundary} tokens but attached {attached}"
                );
            }
            return Ok((request, Some(guard)));
        }

        if saw_eligible_kv {
            self.stats.kv_only_fallbacks += 1;
        }
        Ok((request, None))
    }

    /// Finish a restore after the caller has copied the recurrent state.
    ///
    /// `begin_request` performs the lookup and KV blocks' attachment, but the physical
    /// recurrent-state copy is executor-specific: single-GPU execution copies
    /// from the local store, while TP execution coordinates the copy across
    /// workers. This method therefore only be called after the recurrent state
    /// is copied. It validates the boundary and releases the guard.
    pub(crate) fn finish_restore(
        &mut self,
        request: &RequestKv,
        guard: PrefixGuard,
        recurrent_positions: &[usize],
    ) -> Result<usize> {
        let boundary = guard.boundary();
        anyhow::ensure!(
            request.kv_position() == boundary
                && !recurrent_positions.is_empty()
                && recurrent_positions
                    .iter()
                    .all(|&position| position == boundary),
            "Qwen3.5 joint prefix restore position mismatch: kv={}, recurrent={recurrent_positions:?}, boundary={}",
            request.kv_position(),
            boundary,
        );
        self.stats.joint_hits += 1;
        self.stats.joint_hit_tokens += boundary as u64;
        self.stats.restore_ns = self
            .stats
            .restore_ns
            .saturating_add(guard.started.elapsed().as_nanos() as u64);
        // End the cache pin only after every physical restore was checked.
        drop(guard);
        Ok(boundary)
    }

    /// Reserve the KV pages required by the next prefill forward.
    pub(crate) fn schedule_prefill(&self, request: &mut RequestKv, tokens: usize) -> Result<()> {
        request
            .schedule_prefill(tokens, self.kv.pool())
            .map_err(|e| anyhow::anyhow!("Qwen3.5 prefill KV schedule failed: {e}"))
    }

    /// Build the exact, immutable KV page-table view for prefill kernels.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn prefill_view(&self, request: &RequestKv, tokens: usize) -> KvView {
        request.prefill_view(tokens)
    }

    /// Apply one successful whole-model prefill window.
    ///
    /// KV is applied for every window. Snapshot publication is attempted
    /// only at a non-zero multiple of [`SNAPSHOT_STRIDE_TOKENS`]. A non-aligned
    /// prompt tail still applies successfully but does not create a snapshot.
    pub(crate) fn apply_prefill(
        &self,
        request: &mut RequestKv,
        first_token: Option<u32>,
    ) -> Result<usize> {
        if let Some(first_token) = first_token {
            request.apply_prefill(first_token, self.kv.pool())?;
        } else {
            request.apply_prefill_chunk(self.kv.pool())?;
        }
        let boundary = request.kv_position();
        Ok(boundary)
    }

    /// Reserve a recurrent-state slot for an eligible applied boundary.
    ///
    /// Return a reservation handle when the boundary is alighed to [`SNAPSHOT_STRIDE_TOKENS`].
    /// The reservation is later performed locally at every rank.
    /// Return `None` when the boundary is not aligned or when the cache is disabled.
    pub(crate) fn reserve_prefix(
        &mut self,
        request: &RequestKv,
        boundary: usize,
    ) -> Result<Option<PrefixReservation>> {
        if !self.enabled {
            return Ok(None);
        }
        if boundary == 0 || !boundary.is_multiple_of(SNAPSHOT_STRIDE_TOKENS) {
            return Ok(None);
        }
        let sequence_hash = request
            .registered_boundary_hash(boundary)
            .ok_or_else(|| anyhow::anyhow!("no registered KV hash at boundary {boundary}"))?;
        let key = PrefixBoundaryKey {
            sequence_hash,
            boundary_tokens: boundary,
        };
        Ok(self.state.reserve(key))
    }

    /// Insert a prefix entry to the cache.
    ///
    /// The caller must have already attached the KV blocks and copied the recurrent state to the reserved slot.
    pub(crate) fn publish_prefix(&mut self, request: &RequestKv, reservation: PrefixReservation) {
        let block_count = reservation.key.boundary_tokens / self.kv.pool().block_size();
        let mut kv_guards = request.assigned_block_guards();
        assert!(
            kv_guards.len() >= block_count,
            "Qwen3.5 snapshot boundary {} requires {block_count} KV blocks, request has {}",
            reservation.key.boundary_tokens,
            kv_guards.len()
        );
        kv_guards.truncate(block_count);
        self.stats.inserts += 1;
        if reservation.was_replacement() {
            self.stats.evictions += 1;
        }
        self.state.publish(reservation, kv_guards);
    }

    /// Abort a prefix reservation after any rank-local copy failure.
    pub(crate) fn abort_prefix(&mut self, reservation: PrefixReservation) {
        self.state.abort(reservation);
    }

    /// Reserve KV capacity for the next one-token decode forward.
    pub(crate) fn schedule_decode(&self, request: &mut RequestKv) -> Result<()> {
        request
            .schedule_decode(self.kv.pool())
            .map_err(|e| anyhow::anyhow!("Qwen3.5 decode KV schedule failed: {e}"))
    }

    /// Build the exact, immutable KV page-table view for decode kernels.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn decode_view(&self, request: &RequestKv) -> KvView {
        request.decode_view()
    }

    /// Apply the KV written by decode and record the newly sampled token.
    pub(crate) fn apply_decode(&self, request: &mut RequestKv, token: u32) -> Result<()> {
        request.apply_decode(token, self.kv.pool())?;
        Ok(())
    }

    /// Roll back pages reserved by a scheduled step that did not apply.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn revert_schedule(&self, request: &mut RequestKv) -> Result<()> {
        request.revert_schedule()
    }

    /// Release all request KV.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn release_request(&self, request: &mut RequestKv) -> Result<()> {
        request.mark_blocks_reset_on_release();
        request.release()
    }
}

/// Yield reusable snapshot boundaries from longest to shortest.
fn eligible_boundaries(resident_tokens: usize, stride: usize) -> impl Iterator<Item = usize> {
    let highest = resident_tokens / stride * stride;
    (1..=highest / stride).rev().map(move |n| n * stride)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use pegainfer_kv_cache::BlockPool;

    use super::PrefixBoundaryKey;
    use super::PrefixCacheState;
    use super::SNAPSHOT_STRIDE_TOKENS;
    use super::eligible_boundaries;

    fn key(tag: u8) -> PrefixBoundaryKey {
        PrefixBoundaryKey {
            sequence_hash: [tag; 16],
            boundary_tokens: 256,
        }
    }

    #[test]
    fn joint_boundaries_descend_on_snapshot_stride() {
        assert_eq!(
            eligible_boundaries(255, 256).collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
        assert_eq!(eligible_boundaries(256, 256).collect::<Vec<_>>(), [256]);
        assert_eq!(
            eligible_boundaries(900, 256).collect::<Vec<_>>(),
            [768, 512, 256]
        );
    }

    #[test]
    fn state_publishes_only_after_explicit_commit() {
        let mut state = PrefixCacheState::new(1);
        let reservation = state
            .reserve(key(1))
            .expect("empty state must reserve a write");
        assert!(state.lookup(key(1), Instant::now()).is_none());
        state.publish(reservation, Vec::new());
        assert!(state.lookup(key(1), Instant::now()).is_some());
    }

    #[test]
    fn aborted_write_does_not_expose_partial_snapshot() {
        let mut state = PrefixCacheState::new(1);
        let reservation = state
            .reserve(key(1))
            .expect("empty state must reserve a write");
        state.abort(reservation);
        assert!(state.lookup(key(1), Instant::now()).is_none());
        assert!(state.reserve(key(2)).is_some());
    }

    #[test]
    fn duplicate_refreshes_lru_without_allocating_a_slot() {
        let mut state = PrefixCacheState::new(1);
        let reservation = state
            .reserve(key(1))
            .expect("empty directory must reserve a write");
        state.publish(reservation, Vec::new());
        assert!(state.reserve(key(1)).is_none());
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn lru_evicts_untouched_snapshot_but_preserves_touched_snapshot() {
        let mut state = PrefixCacheState::new(2);
        for tag in [1, 2] {
            let reservation = state
                .reserve(key(tag))
                .expect("state should have a free slot");
            state.publish(reservation, Vec::new());
        }
        drop(
            state
                .lookup(key(1), Instant::now())
                .expect("key 1 should be present"),
        );
        let reservation = state
            .reserve(key(3))
            .expect("an unpinned LRU victim should be available");
        state.publish(reservation, Vec::new());
        assert!(state.lookup(key(1), Instant::now()).is_some());
        assert!(state.lookup(key(2), Instant::now()).is_none());
        assert!(state.lookup(key(3), Instant::now()).is_some());
    }

    #[test]
    fn all_pinned_snapshots_make_insertion_a_soft_miss() {
        let mut state = PrefixCacheState::new(1);
        let reservation = state
            .reserve(key(1))
            .expect("empty state must reserve a write");
        state.publish(reservation, Vec::new());
        let guard = state
            .lookup(key(1), Instant::now())
            .expect("key 1 should be present");
        assert!(state.reserve(key(2)).is_none());
        drop(guard);
        assert!(state.reserve(key(2)).is_some());
    }

    #[test]
    fn joint_cache_entry_releases_kv_blocks_on_eviction() {
        let pool = BlockPool::new(16, 32).expect("block pool");
        let baseline = pool.available_blocks();
        let mut request = pool.new_request(vec![7; SNAPSHOT_STRIDE_TOKENS], 0, None);
        request
            .schedule_prefill(SNAPSHOT_STRIDE_TOKENS, &pool)
            .expect("schedule prefill");
        request.apply_prefill_chunk(&pool).expect("apply prefill");

        let mut state = PrefixCacheState::new(1);
        let reservation = state
            .reserve(key(1))
            .expect("empty state must reserve a write");
        state.publish(reservation, request.assigned_block_guards());
        request.mark_blocks_reset_on_release();
        request.release().expect("release request");

        let retained_blocks = SNAPSHOT_STRIDE_TOKENS / pool.block_size();
        assert_eq!(pool.available_blocks(), baseline - retained_blocks);

        let reservation = state
            .reserve(key(2))
            .expect("full directory must evict its unpinned snapshot");
        assert_eq!(pool.available_blocks(), baseline);
        state.abort(reservation);
    }
}
