//! Rank-local scheduler load snapshot exported to the frontend.

use std::collections::VecDeque;

use pegainfer_core::engine::GenerateRequest;
use pegainfer_core::engine::LoadSnapshot;
use pegainfer_kv_cache::BlockPool;
use tokio::sync::watch;

use super::RankSlots;

/// Publish this rank's truthful scheduler snapshot. Cached pages that kvbm
/// can evict count as available; the reserved padding page is excluded from
/// both used and total. The frontend scores this rank's engine by it, and an
/// unbound request's least-load placement (in `EngineHandle::submit`) reads
/// the same numbers.
pub(super) fn publish_load(
    load_tx: &watch::Sender<LoadSnapshot>,
    pool: &BlockPool,
    slots: &RankSlots,
    pending: &VecDeque<GenerateRequest>,
) {
    let kv_total_blocks = pool.total_blocks() - 1;
    load_tx.send_replace(LoadSnapshot {
        kv_used_blocks: kv_total_blocks.saturating_sub(pool.available_blocks()) as u64,
        kv_total_blocks: kv_total_blocks as u64,
        num_running_reqs: slots.iter().flatten().count() as u64,
        num_waiting_reqs: pending.len() as u64,
    });
}
