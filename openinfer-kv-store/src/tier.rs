//! The host-tier seam: what [`crate::KvStore`] needs from the storage layer
//! below it (pegaflow today), expressed as a dyn-compatible trait.
//!
//! This is the anti-corruption boundary from
//! `docs/subsystems/kv-cache/design.md`: pegaflow's interface quirks (the
//! re-query-to-poll `Loading` outcome, lease lifetimes) are absorbed here,
//! and the contract tests run against [`crate::testkit::MockTier`] with no
//! GPU or pegaflow at all.

use std::any::Any;
use std::future::Future;
use std::pin::Pin;

use openinfer_kv_offload::OffloadEngine;
use openinfer_kv_offload::QueryLeaseId;
use openinfer_kv_offload::QueryOutcome;
use openinfer_kv_offload::SaveHandle;

pub type TierFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// A host-resident prefix hit: `blocks` leading blocks are pinned behind the
/// tier's lease until [`HostTier::load`] consumes it or [`HostTier::release`]
/// declines it. The lease is erased behind `Any` so mock tiers need no
/// pegaflow types; each impl downcasts only its own token.
pub struct TierHit {
    pub blocks: usize,
    pub(crate) token: Box<dyn Any + Send>,
}

/// One query's outcome against the host tier.
pub enum TierQuery {
    /// Nothing resident and nothing in flight.
    Miss,
    /// A deeper tier (SSD / remote peer) is fetching; re-query after a pause
    /// (pegaflow's poll-by-requery contract, absorbed here).
    Loading,
    Hit(TierHit),
}

/// The storage layer below the store. All returned futures observe operations
/// already submitted to the tier's own runtime — dropping one detaches the
/// observer, never the I/O (the [`OffloadEngine`] handle contract).
pub trait HostTier: Send + Sync {
    /// How long a prefix of `block_hashes` is host-resident. `req_id` scopes
    /// an in-flight deeper-tier fetch across re-queries.
    fn query(
        &self,
        req_id: &str,
        block_hashes: Vec<Vec<u8>>,
    ) -> TierFuture<anyhow::Result<TierQuery>>;

    /// Copy the hit's blocks into the GPU pages `dst_page_ids` (one
    /// destination per hit block, across all registered layers). Consumes the
    /// lease whether it succeeds or fails.
    fn load(&self, hit: TierHit, dst_page_ids: Vec<i32>) -> TierFuture<anyhow::Result<()>>;

    /// Decline a hit without loading it, releasing the host-side pins now
    /// instead of waiting out the lease TTL.
    fn release(&self, hit: TierHit);

    /// Submit an async GPU→host save of sealed blocks. Returns an
    /// already-in-flight handle; `keep_alive` is dropped only once the D2H
    /// lands (the reuse contract — pass the source blocks' guards).
    fn save(
        &self,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
        keep_alive: Box<dyn Any + Send>,
    ) -> SaveHandle;
}

impl HostTier for OffloadEngine {
    fn query(
        &self,
        req_id: &str,
        block_hashes: Vec<Vec<u8>>,
    ) -> TierFuture<anyhow::Result<TierQuery>> {
        let handle = self.submit_query(req_id, &block_hashes, false);
        Box::pin(async move {
            match handle.settle().await.map_err(anyhow::Error::msg)? {
                QueryOutcome::Loading => Ok(TierQuery::Loading),
                QueryOutcome::Ready(hit) => match hit.lease {
                    Some(lease) if hit.num_blocks > 0 => Ok(TierQuery::Hit(TierHit {
                        blocks: hit.num_blocks,
                        token: Box::new(lease),
                    })),
                    _ => Ok(TierQuery::Miss),
                },
            }
        })
    }

    fn load(&self, hit: TierHit, dst_page_ids: Vec<i32>) -> TierFuture<anyhow::Result<()>> {
        let lease: QueryLeaseId = *hit
            .token
            .downcast()
            .expect("OffloadEngine tier consumed a foreign lease token");
        match OffloadEngine::load(self, lease, dst_page_ids) {
            Ok(handle) => {
                Box::pin(async move { handle.settle().await.map_err(anyhow::Error::msg) })
            }
            Err(err) => Box::pin(std::future::ready(Err(anyhow::Error::msg(err)))),
        }
    }

    fn release(&self, hit: TierHit) {
        let lease: QueryLeaseId = *hit
            .token
            .downcast()
            .expect("OffloadEngine tier released a foreign lease token");
        self.release_query_lease(lease);
    }

    fn save(
        &self,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
        keep_alive: Box<dyn Any + Send>,
    ) -> SaveHandle {
        self.submit_save(&block_ids, &block_hashes, keep_alive)
    }
}
