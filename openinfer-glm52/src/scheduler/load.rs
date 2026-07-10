//! Rank-local scheduler load snapshots exported to the frontend.

use std::collections::VecDeque;

use openinfer_core::engine::{GenerateRequest, LoadSnapshot};
use openinfer_kv_cache::BlockPool;
use tokio::sync::watch;

use super::RankSlots;

pub(super) fn running_counts(slots: &[RankSlots]) -> Vec<usize> {
    slots
        .iter()
        .map(|rank_slots| rank_slots.iter().flatten().count())
        .collect()
}

pub(super) fn pending_is_empty(pending: &[VecDeque<GenerateRequest>]) -> bool {
    pending.iter().all(VecDeque::is_empty)
}

/// Publish one truthful scheduler snapshot per logical DP rank. Cached pages
/// that kvbm can evict count as available; the reserved padding page is
/// excluded from both used and total.
pub(super) fn publish_load(
    load_txs: &[watch::Sender<LoadSnapshot>],
    pools: &[BlockPool],
    slots: &[RankSlots],
    pending: &[VecDeque<GenerateRequest>],
) {
    for (rank, load_tx) in load_txs.iter().enumerate() {
        let kv_total_blocks = pools[rank].total_blocks() - 1;
        load_tx.send_replace(LoadSnapshot {
            kv_used_blocks: kv_total_blocks.saturating_sub(pools[rank].available_blocks()) as u64,
            kv_total_blocks: kv_total_blocks as u64,
            num_running_reqs: slots[rank].iter().flatten().count() as u64,
            num_waiting_reqs: pending[rank].len() as u64,
        });
    }
}
