//! The per-tick decision for a CPU-tier KV prefetch query.
//!
//! `begin_kv_prefetch` (first query) and `poll_remote_fetch` (re-query while
//! parked in [`RemoteFetch`](super::PrefetchPhase::RemoteFetch)) share one
//! cascade: still loading → wait; timeout / zero-hit / query error → prefill
//! from scratch; leased hit past the reservation budget → release the lease
//! and prefill from scratch; leased hit within budget → start the H2D load.
//! This module owns that cascade so the two call sites cannot drift, and so
//! the timeout / zero-hit / budget branches are testable without a GPU, an
//! RDMA fabric, or a MetaServer (only *injecting* a real `Loading` needs
//! those; deciding on one does not).
//!
//! The lease type is generic: every branch that receives a lease must route
//! it into the returned action, so a silently dropped (never-released) lease
//! is unrepresentable. Production monomorphizes `L` to pegaflow's opaque
//! `QueryLeaseId`; tests use plain integers.

/// A lease-carrying view of [`QueryOutcome`](openinfer_kv_offload::QueryOutcome),
/// decoupled from pegaflow types so the decision stays constructible in tests.
///
/// `lease: None` only occurs together with `num_blocks == 0` (the engine
/// builds a lease for every non-empty hit); the pair is kept as-is rather
/// than re-encoded because the decision treats every `None` as a miss.
pub(super) enum QueryView<L> {
    Loading,
    Ready { lease: Option<L>, num_blocks: usize },
}

impl From<openinfer_kv_offload::QueryOutcome> for QueryView<openinfer_kv_offload::QueryLeaseId> {
    fn from(outcome: openinfer_kv_offload::QueryOutcome) -> Self {
        match outcome {
            openinfer_kv_offload::QueryOutcome::Loading => QueryView::Loading,
            openinfer_kv_offload::QueryOutcome::Ready(hit) => QueryView::Ready {
                lease: hit.lease,
                num_blocks: hit.num_blocks,
            },
        }
    }
}

/// What to do with the request after one prefetch query tick.
#[derive(Debug, Eq, PartialEq)]
pub(super) enum RemoteFetchAction<L> {
    /// Remote fetch still in flight: park (first tick) or stay parked.
    Wait,
    /// Give up the prefetch — deadline passed, zero-hit, or the query
    /// errored — and prefill from scratch. No lease was taken.
    Scratch,
    /// The hit is leased but reserving `num_blocks` would eat into blocks
    /// promised to admitted requests: release the lease, then prefill from
    /// scratch.
    Release(L),
    /// Leased hit within budget: reserve blocks and submit the H2D load.
    Load(L, usize),
}

/// Decide one prefetch query tick.
///
/// `query` runs only when the deadline has not passed — a parked request
/// whose deadline expired must not issue another query RPC — which is why
/// this takes a closure rather than a pre-computed outcome. A query `Err` is
/// already logged at the call site's level of context, so it folds into
/// [`RemoteFetchAction::Scratch`] here.
pub(super) fn remote_fetch_action<L, E>(
    timed_out: bool,
    query: impl FnOnce() -> Result<QueryView<L>, E>,
    available_blocks: usize,
    reserve_floor: usize,
) -> RemoteFetchAction<L> {
    if timed_out {
        return RemoteFetchAction::Scratch;
    }
    let (lease, num_blocks) = match query() {
        Ok(QueryView::Loading) => return RemoteFetchAction::Wait,
        Ok(QueryView::Ready { lease, num_blocks }) => (lease, num_blocks),
        Err(_) => return RemoteFetchAction::Scratch,
    };
    let Some(lease) = lease else {
        return RemoteFetchAction::Scratch; // miss
    };
    // Blocks promised to admitted requests are off-limits: reserving into
    // them makes a later prefill chunk or decode growth fail allocation.
    if available_blocks.saturating_sub(reserve_floor) < num_blocks {
        return RemoteFetchAction::Release(lease);
    }
    RemoteFetchAction::Load(lease, num_blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    type Action = RemoteFetchAction<u32>;

    fn ready(lease: Option<u32>, num_blocks: usize) -> Result<QueryView<u32>, ()> {
        Ok(QueryView::Ready { lease, num_blocks })
    }

    /// Past the deadline the request prefills from scratch and, critically,
    /// no further query RPC is issued.
    #[test]
    fn timeout_gives_up_without_querying() {
        let action = remote_fetch_action::<u32, ()>(
            true,
            || unreachable!("post-deadline tick must not query"),
            usize::MAX,
            0,
        );
        assert_eq!(action, Action::Scratch);
    }

    /// A remote fetch still in flight keeps the request parked.
    #[test]
    fn loading_waits() {
        let action = remote_fetch_action::<u32, ()>(false, || Ok(QueryView::Loading), 0, 0);
        assert_eq!(action, Action::Wait);
    }

    /// Zero-hit (peer evicted the blocks, or no owner): no lease exists, so
    /// the request just prefills from scratch.
    #[test]
    fn zero_hit_prefills_from_scratch() {
        let action = remote_fetch_action(false, || ready(None, 0), usize::MAX, 0);
        assert_eq!(action, Action::Scratch);
    }

    /// A query error folds into prefill-from-scratch.
    #[test]
    fn query_error_prefills_from_scratch() {
        let action = remote_fetch_action::<u32, &str>(false, || Err("rpc failed"), usize::MAX, 0);
        assert_eq!(action, Action::Scratch);
    }

    /// A leased hit that would eat into promised blocks routes the lease out
    /// for release — the type makes dropping it silently unrepresentable.
    #[test]
    fn budget_guard_releases_the_lease() {
        let action = remote_fetch_action(false, || ready(Some(7), 10), 12, 3);
        assert_eq!(action, Action::Release(7));
    }

    /// `reserve_floor > available_blocks` must saturate, not underflow.
    #[test]
    fn budget_guard_saturates_when_floor_exceeds_available() {
        let action = remote_fetch_action(false, || ready(Some(7), 1), 2, 5);
        assert_eq!(action, Action::Release(7));
    }

    /// Exactly-at-budget reserves and loads: the guard is strict-less-than.
    #[test]
    fn exact_budget_boundary_loads() {
        let action = remote_fetch_action(false, || ready(Some(7), 9), 12, 3);
        assert_eq!(action, Action::Load(7, 9));
    }

    /// The normal path: leased hit within budget starts the H2D load.
    #[test]
    fn leased_hit_within_budget_loads() {
        let action = remote_fetch_action(false, || ready(Some(42), 4), 64, 8);
        assert_eq!(action, Action::Load(42, 4));
    }
}
