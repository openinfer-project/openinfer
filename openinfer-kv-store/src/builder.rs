//! Construction: chainable knobs, rank declaration, and the freeze at
//! [`KvStoreBuilder::build`].

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use openinfer_kv_cache::BlockPool;

use crate::HostTier;
use crate::KvStore;
use crate::KvStoreStats;
use crate::store::RankState;

/// One declared rank, held until [`KvStoreBuilder::build`] resolves the
/// derived values.
type RankDecl = (usize, Arc<BlockPool>, Option<Arc<dyn HostTier>>);

/// The qwen3 remote-fetch re-query cadence this store generalizes.
const DEFAULT_REQUERY_INTERVAL: Duration = Duration::from_millis(5);
/// The qwen3 handoff deadline this store generalizes.
const DEFAULT_RESOLVE_DEADLINE: Duration = Duration::from_secs(15);
/// Pending bench calibration (#824).
const DEFAULT_CACHEABLE_PIN_PERCENT: usize = 25;

/// Builds a [`KvStore`]. The rank table freezes at [`Self::build`] — the
/// store's read paths take no lock — and derived values (per-rank pin
/// budgets) are computed there, never in setters. A `BlockPool` is a pure
/// CPU object with no thread affinity: construct pools before engine
/// threads spawn.
pub struct KvStoreBuilder {
    runtime: tokio::runtime::Handle,
    requery_interval: Duration,
    resolve_deadline: Duration,
    cacheable_pin_percent: usize,
    ranks: Vec<RankDecl>,
}

impl KvStoreBuilder {
    #[must_use]
    pub fn new(runtime: tokio::runtime::Handle) -> Self {
        Self {
            runtime,
            requery_interval: DEFAULT_REQUERY_INTERVAL,
            resolve_deadline: DEFAULT_RESOLVE_DEADLINE,
            cacheable_pin_percent: DEFAULT_CACHEABLE_PIN_PERCENT,
            ranks: Vec::new(),
        }
    }

    /// Pause between host-tier re-queries while a deeper tier is fetching —
    /// also the retry cadence while waiting out pool pressure.
    #[must_use]
    pub fn with_requery_interval(mut self, interval: Duration) -> Self {
        self.requery_interval = interval;
        self
    }

    /// Ceiling on one resolve's host-tier wait — bounds every tier await
    /// (query and load alike) and the pool-pressure wait, so a hung storage
    /// worker degrades the resolve instead of stranding the request outside
    /// the scheduler.
    #[must_use]
    pub fn with_resolve_deadline(mut self, deadline: Duration) -> Self {
        self.resolve_deadline = deadline;
        self
    }

    /// Ceiling on the pool share `Cacheable` saves may pin, as a percent of
    /// each rank's total blocks. Past it, cacheable saves shed (a forfeited
    /// future hit) instead of pinning admission out of the pool. `Handoff`
    /// saves are exempt — their backpressure is admission reading
    /// [`KvStore::pinned_blocks`].
    #[must_use]
    pub fn with_cacheable_pin_percent(mut self, percent: usize) -> Self {
        self.cacheable_pin_percent = percent;
        self
    }

    /// Declare a rank with only its GPU pool: resolves serve radix hits and
    /// seals are no-ops — the tier-less mode behind the same API.
    #[must_use]
    pub fn rank(mut self, rank: usize, pool: Arc<BlockPool>) -> Self {
        self.ranks.push((rank, pool, None));
        self
    }

    /// Declare a rank with its GPU pool and host tier.
    #[must_use]
    pub fn rank_with_tier(
        mut self,
        rank: usize,
        pool: Arc<BlockPool>,
        tier: Arc<dyn HostTier>,
    ) -> Self {
        self.ranks.push((rank, pool, Some(tier)));
        self
    }

    #[must_use]
    pub fn build(self) -> KvStore {
        let ranks = self
            .ranks
            .into_iter()
            .map(|(rank, pool, tier)| {
                let cacheable_pin_budget = pool.total_blocks() * self.cacheable_pin_percent / 100;
                (
                    rank,
                    RankState {
                        pool,
                        tier,
                        floor: AtomicUsize::new(0),
                        pinned: Arc::new(AtomicUsize::new(0)),
                        cacheable_pinned: Arc::new(AtomicUsize::new(0)),
                        cacheable_pin_budget,
                    },
                )
            })
            .collect();
        KvStore {
            runtime: self.runtime,
            requery_interval: self.requery_interval,
            resolve_deadline: self.resolve_deadline,
            ranks,
            stats: Arc::new(KvStoreStats::default()),
        }
    }
}
