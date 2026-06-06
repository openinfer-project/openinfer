pub(super) enum ExecutionPlan<T> {
    Prefill { pending: Vec<T> },
    Decode,
    Unified { pending: Vec<T> },
}

pub(super) struct AdmissionOutcome<T> {
    pub(super) pending: Vec<T>,
    pub(super) deferred: Vec<T>,
    pub(super) rejected: Vec<T>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ActiveKvBudget {
    pub(super) prompt_len: usize,
    pub(super) generated_count: usize,
    pub(super) max_tokens: usize,
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
    pending: Vec<T>,
    active: &[ActiveKvBudget],
    max_batch: usize,
    page_size: usize,
    available_pages: usize,
    max_request_pages: usize,
    mut prompt_len: impl FnMut(&T) -> usize,
    mut max_tokens: impl FnMut(&T) -> usize,
) -> AdmissionOutcome<T> {
    assert!(page_size > 0, "Qwen3.5 KV page size must be non-zero");

    let mut page_budget = available_pages.saturating_sub(active_future_pages(active, page_size));
    let slot_budget = max_batch.saturating_sub(active.len());
    let mut admitted = Vec::new();
    let mut still_deferred = Vec::new();
    let mut rejected = Vec::new();
    let mut blocked = false;

    for req in pending {
        let request_pages =
            pages_needed(max_kv_tokens(prompt_len(&req), max_tokens(&req)), page_size);
        if request_pages > max_request_pages {
            rejected.push(req);
            continue;
        }

        if blocked || admitted.len() >= slot_budget || request_pages > page_budget {
            blocked = true;
            still_deferred.push(req);
            continue;
        }

        page_budget -= request_pages;
        admitted.push(req);
    }

    AdmissionOutcome {
        pending: admitted,
        deferred: still_deferred,
        rejected,
    }
}

fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

// Prefill samples the first output token but does not append it to KV. A
// generated token occupies KV only when it is fed as the next decode input.
// Therefore N returned completion tokens occupy at most N - 1 generated-token
// KV slots.
pub(super) fn max_kv_tokens(prompt_len: usize, max_tokens: usize) -> usize {
    prompt_len.saturating_add(max_tokens.saturating_sub(1))
}

fn current_active_tokens(req: ActiveKvBudget) -> usize {
    req.prompt_len
        .saturating_add(req.generated_count.saturating_sub(1))
}

fn active_future_pages(active: &[ActiveKvBudget], page_size: usize) -> usize {
    active
        .iter()
        .map(|req| {
            let max_pages = pages_needed(max_kv_tokens(req.prompt_len, req.max_tokens), page_size);
            let current_pages = pages_needed(current_active_tokens(*req), page_size);
            assert!(
                current_pages <= max_pages,
                "active Qwen3.5 request exceeded its admitted KV lifetime budget"
            );
            max_pages.saturating_sub(current_pages)
        })
        .sum()
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
        max_tokens: usize,
    }

    fn pending(id: u32, prompt_len: usize) -> Pending {
        Pending {
            id,
            prompt_len,
            max_tokens: 1,
        }
    }

    fn pending_with_max(id: u32, prompt_len: usize, max_tokens: usize) -> Pending {
        Pending {
            id,
            prompt_len,
            max_tokens,
        }
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
        let active = [
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1,
                max_tokens: 18,
            },
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1,
                max_tokens: 1,
            },
        ];
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 16), pending(3, 16)],
            &active,
            4,
            16,
            6,
            6,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(
            ids(&outcome.pending),
            vec![1, 2],
            "active future KV growth is reserved before admitting new requests"
        );
        assert_eq!(
            ids(&outcome.deferred),
            vec![3],
            "requests beyond the remaining slot/page budget stay deferred"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_is_fcfs_and_keeps_later_requests_deferred_after_first_miss() {
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 33), pending(3, 16)],
            &[],
            8,
            16,
            3,
            8,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert_eq!(
            ids(&outcome.deferred),
            vec![2, 3],
            "a later smaller request must not jump ahead of an earlier budget miss"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_keeps_order_when_first_pending_request_misses_budget() {
        let outcome = admit_pending_requests(
            vec![pending(1, 33), pending(2, 16)],
            &[],
            8,
            16,
            2,
            8,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert!(outcome.pending.is_empty());
        assert_eq!(
            ids(&outcome.deferred),
            vec![1, 2],
            "a later smaller request must not bypass the first deferred request"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_uses_ceil_div_at_page_boundaries() {
        let outcome = admit_pending_requests(
            vec![pending(1, 15), pending(2, 16), pending(3, 17)],
            &[],
            8,
            16,
            3,
            8,
            |req| req.prompt_len,
            |req| req.max_tokens,
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
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_returns_all_pending_when_active_batch_is_at_slot_capacity() {
        let outcome = admit_pending_requests(
            vec![pending(1, 1), pending(2, 1)],
            &[
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
            ],
            4,
            16,
            10,
            10,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert!(outcome.pending.is_empty());
        assert_eq!(ids(&outcome.deferred), vec![1, 2]);
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn active_scheduler_decodes_when_no_pending_request_is_admitted() {
        let outcome = admit_pending_requests(
            vec![pending(1, 16)],
            &[ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1,
                max_tokens: 17,
            }],
            4,
            16,
            1,
            4,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert!(outcome.pending.is_empty());
        assert_eq!(ids(&outcome.deferred), vec![1]);
        assert!(outcome.rejected.is_empty());
        assert!(
            matches!(
                build_next_plan(true, outcome.pending),
                Some(ExecutionPlan::Decode)
            ),
            "active requests should keep decoding when pending requests are all deferred"
        );
    }

    #[test]
    fn admission_counts_pending_generation_budget() {
        let outcome = admit_pending_requests(
            vec![
                pending_with_max(1, 16, 17), // 32 KV tokens -> 2 pages
                pending(2, 16),              // 16 KV tokens -> 1 page
            ],
            &[],
            8,
            16,
            2,
            8,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert_eq!(
            ids(&outcome.deferred),
            vec![2],
            "pending request 1 reserves its future decode KV page"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn active_future_pages_counts_only_remaining_growth() {
        let active = [
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1, // current 16 tokens -> 1 page
                max_tokens: 33,     // max 48 KV tokens -> 3 pages
            },
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 17, // current 32 tokens -> 2 pages
                max_tokens: 17,      // max 32 KV tokens -> 2 pages
            },
            ActiveKvBudget {
                prompt_len: 9,
                generated_count: 8, // current 16 tokens -> 1 page
                max_tokens: 24,     // max 32 KV tokens -> 2 pages
            },
        ];

        assert_eq!(
            active_future_pages(&active, 16),
            3,
            "active admission reserves only future page growth, not pages already held"
        );
    }

    #[test]
    fn active_future_reservation_can_defer_pending_by_page_budget() {
        let active = [ActiveKvBudget {
            prompt_len: 16,
            generated_count: 1, // current 16 tokens -> 1 page
            max_tokens: 49,     // max 64 KV tokens -> 4 pages; future growth = 3 pages
        }];
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 16)],
            &active,
            8,
            16,
            4,
            8,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert_eq!(
            ids(&outcome.deferred),
            vec![2],
            "slot budget is available, but active future KV growth leaves only one page"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_rejects_impossible_request_without_blocking_later_fit() {
        let outcome = admit_pending_requests(
            vec![
                pending_with_max(1, 16, 65), // 80 KV tokens -> 5 pages
                pending(2, 16),
            ],
            &[],
            8,
            16,
            4,
            4,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![2]);
        assert!(outcome.deferred.is_empty());
        assert_eq!(ids(&outcome.rejected), vec![1]);
    }

    #[test]
    fn admission_allows_request_at_single_request_page_cap() {
        let outcome = admit_pending_requests(
            vec![pending_with_max(1, 16, 49)], // 64 KV tokens -> 4 pages
            &[],
            1,
            16,
            4,
            4,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert!(outcome.deferred.is_empty());
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn one_token_completion_on_page_boundary_uses_only_prompt_page() {
        assert_eq!(max_kv_tokens(16, 1), 16);
        let outcome = admit_pending_requests(
            vec![pending_with_max(1, 16, 1)],
            &[],
            1,
            16,
            1,
            1,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert!(outcome.deferred.is_empty());
        assert!(outcome.rejected.is_empty());
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
