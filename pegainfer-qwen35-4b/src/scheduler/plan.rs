pub(super) enum ExecutionPlan<T> {
    Prefill { pending: Vec<T> },
    Decode,
    Unified { pending: Vec<T> },
}

pub(super) struct AdmissionOutcome<T> {
    pub(super) pending: Vec<T>,
    pub(super) deferred: Vec<T>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SlotCompaction {
    pub(super) moved_from: usize,
    pub(super) moved_to: usize,
}

pub(super) fn build_next_plan<T>(have_active: bool, pending: Vec<T>) -> Option<ExecutionPlan<T>> {
    if !pending.is_empty() && have_active {
        Some(ExecutionPlan::Unified { pending })
    } else if !pending.is_empty() {
        Some(ExecutionPlan::Prefill { pending })
    } else if have_active {
        Some(ExecutionPlan::Decode)
    } else {
        None
    }
}

pub(super) fn admit_pending_requests<T>(
    mut pending: Vec<T>,
    active_count: usize,
    max_batch: usize,
    page_size: usize,
    available_pages: usize,
    mut prompt_len: impl FnMut(&T) -> usize,
) -> AdmissionOutcome<T> {
    assert!(page_size > 0, "Qwen3.5 KV page size must be non-zero");

    // Preserve the current Qwen3.5 scheduler contract exactly: each active
    // decode request reserves one page for the next token, and pending requests
    // are admitted FCFS on prompt-only page demand.
    let mut page_budget = available_pages.saturating_sub(active_count);
    let slot_budget = max_batch.saturating_sub(active_count);
    let mut admitted = 0;

    for req in &pending {
        if admitted >= slot_budget {
            break;
        }

        let needed_pages = prompt_len(req).div_ceil(page_size);
        if needed_pages > page_budget {
            break;
        }

        page_budget -= needed_pages;
        admitted += 1;
    }

    let deferred = if admitted < pending.len() {
        pending.split_off(admitted)
    } else {
        Vec::new()
    };

    AdmissionOutcome { pending, deferred }
}

pub(super) fn slot_for_new_request(active_count: usize, max_batch: usize) -> Option<usize> {
    (active_count < max_batch).then_some(active_count)
}

pub(super) fn compaction_after_retire(
    active_len_before: usize,
    retired_idx: usize,
) -> Option<SlotCompaction> {
    assert!(
        retired_idx < active_len_before,
        "retired Qwen3.5 slot index must be active"
    );

    let last = active_len_before - 1;
    (retired_idx < last).then_some(SlotCompaction {
        moved_from: last,
        moved_to: retired_idx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct Pending {
        id: u32,
        prompt_len: usize,
    }

    fn pending(id: u32, prompt_len: usize) -> Pending {
        Pending { id, prompt_len }
    }

    fn ids(reqs: &[Pending]) -> Vec<u32> {
        reqs.iter().map(|req| req.id).collect()
    }

    #[test]
    fn plan_selection_follows_active_and_pending_state() {
        assert!(
            build_next_plan::<Pending>(false, vec![]).is_none(),
            "idle scheduler produces no execution plan"
        );
        assert!(
            matches!(
                build_next_plan::<Pending>(true, vec![]),
                Some(ExecutionPlan::Decode)
            ),
            "active-only scheduler tick decodes the active batch"
        );
        assert!(
            matches!(
                build_next_plan(false, vec![pending(1, 8)]),
                Some(ExecutionPlan::Prefill { pending }) if ids(&pending) == vec![1]
            ),
            "pending-only scheduler tick prefills new requests"
        );
        assert!(
            matches!(
                build_next_plan(true, vec![pending(1, 8)]),
                Some(ExecutionPlan::Unified { pending }) if ids(&pending) == vec![1]
            ),
            "active + pending scheduler tick runs the unified path"
        );
    }

    #[test]
    fn admission_respects_slot_capacity_and_active_decode_reserve() {
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 16), pending(3, 16)],
            2,
            4,
            16,
            4,
            |req| req.prompt_len,
        );

        assert_eq!(
            ids(&outcome.pending),
            vec![1, 2],
            "two active decode requests reserve two pages, leaving two prompt pages"
        );
        assert_eq!(
            ids(&outcome.deferred),
            vec![3],
            "requests beyond the remaining slot/page budget stay deferred"
        );
    }

    #[test]
    fn admission_is_fcfs_and_keeps_later_requests_deferred_after_first_miss() {
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 33), pending(3, 16)],
            0,
            8,
            16,
            3,
            |req| req.prompt_len,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert_eq!(
            ids(&outcome.deferred),
            vec![2, 3],
            "a later smaller request must not jump ahead of an earlier budget miss"
        );
    }

    #[test]
    fn admission_keeps_order_when_first_pending_request_misses_budget() {
        let outcome =
            admit_pending_requests(vec![pending(1, 33), pending(2, 16)], 0, 8, 16, 2, |req| {
                req.prompt_len
            });

        assert!(outcome.pending.is_empty());
        assert_eq!(
            ids(&outcome.deferred),
            vec![1, 2],
            "a later smaller request must not bypass the first deferred request"
        );
    }

    #[test]
    fn admission_uses_ceil_div_at_page_boundaries() {
        let outcome = admit_pending_requests(
            vec![pending(1, 15), pending(2, 16), pending(3, 17)],
            0,
            8,
            16,
            3,
            |req| req.prompt_len,
        );

        assert_eq!(
            ids(&outcome.pending),
            vec![1, 2],
            "15 and 16 tokens each use one page"
        );
        assert_eq!(
            ids(&outcome.deferred),
            vec![3],
            "17 tokens needs two pages and waits when only one page remains"
        );
    }

    #[test]
    fn admission_returns_all_pending_when_active_batch_is_at_slot_capacity() {
        let outcome =
            admit_pending_requests(vec![pending(1, 1), pending(2, 1)], 4, 4, 16, 10, |req| {
                req.prompt_len
            });

        assert!(outcome.pending.is_empty());
        assert_eq!(ids(&outcome.deferred), vec![1, 2]);
    }

    #[test]
    fn active_scheduler_decodes_when_no_pending_request_is_admitted() {
        let outcome =
            admit_pending_requests(vec![pending(1, 16)], 1, 4, 16, 1, |req| req.prompt_len);

        assert!(outcome.pending.is_empty());
        assert_eq!(ids(&outcome.deferred), vec![1]);
        assert!(
            matches!(
                build_next_plan(true, outcome.pending),
                Some(ExecutionPlan::Decode)
            ),
            "active requests should keep decoding when pending requests are all deferred"
        );
    }

    #[test]
    fn graph_slot_assignment_stays_dense_after_retirement() {
        assert_eq!(slot_for_new_request(0, 4), Some(0));
        assert_eq!(slot_for_new_request(3, 4), Some(3));
        assert_eq!(slot_for_new_request(4, 4), None);

        assert_eq!(
            compaction_after_retire(4, 1),
            Some(SlotCompaction {
                moved_from: 3,
                moved_to: 1
            }),
            "retiring a middle slot moves the last dense slot into the hole"
        );
        assert_eq!(
            compaction_after_retire(4, 0),
            Some(SlotCompaction {
                moved_from: 3,
                moved_to: 0
            }),
            "retiring the first slot also moves the last dense slot into the hole"
        );
        assert_eq!(
            compaction_after_retire(4, 3),
            None,
            "retiring the last slot does not need a recurrent-state copy"
        );
        assert_eq!(
            slot_for_new_request(3, 4),
            Some(3),
            "after compaction, the next request reuses the next dense slot"
        );
    }
}
