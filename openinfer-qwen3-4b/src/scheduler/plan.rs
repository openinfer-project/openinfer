use anyhow::Result;
use rand::rngs::StdRng;

use crate::executor::{
    DecodePlan, DecodeResult, DecodeStepItem, ModelExecutor, PrefillPlan, PrefillResult,
    PrefillStepItem, UnifiedPlan, UnifiedResult,
};
use crate::speculative::{
    DraftPlan as SpeculativeDraftPlan, DraftRequestResult as SpeculativeDraftRequestResult,
    DraftStepItem as SpeculativeDraftStepItem, VerifyPlan as SpeculativeVerifyPlan,
    VerifyResult as SpeculativeVerifyResult, VerifyStepItem as SpeculativeVerifyStepItem,
};

use super::{ActiveRequestState, PendingRequest};

pub(super) enum ExecutionPlan {
    Prefill { pending: Vec<PendingRequest> },
    Decode,
    SpeculativeDecode,
    Unified { pending: Vec<PendingRequest> },
}

pub(super) enum ExecutionArtifacts {
    Prefill {
        pending: Vec<PendingRequest>,
        result: PrefillResult,
        /// Stamped before the forward pass ran — downstream metrics split
        /// queue time (queued→scheduled) from prefill time (scheduled→first
        /// token), so stamping after execution would fold prefill into queue.
        scheduled_at_unix_s: f64,
    },
    Decode {
        result: DecodeResult,
    },
    SpeculativeDecode {
        verify: SpeculativeVerifyResult,
    },
    Unified {
        pending: Vec<PendingRequest>,
        result: UnifiedResult,
        scheduled_at_unix_s: f64,
    },
}

pub(super) fn build_next_plan(
    have_active: bool,
    pending: Vec<PendingRequest>,
) -> Option<ExecutionPlan> {
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

pub(super) fn execute_plan(
    executor: &mut impl ModelExecutor,
    active: &mut [ActiveRequestState],
    plan: ExecutionPlan,
    rng: &mut StdRng,
) -> Result<ExecutionArtifacts> {
    match plan {
        ExecutionPlan::Prefill { pending } => {
            let scheduled_at_unix_s = openinfer_core::engine::unix_now_s();
            let indices: Vec<usize> = (0..pending.len()).collect();
            let requests = build_prefill_items(&pending, &indices, rng);
            let any_echo = pending.iter().any(|req| req.echo);
            let mut result = executor.execute_prefill(PrefillPlan {
                requests: &requests,
                echo: any_echo,
            })?;
            sort_prefill_results(&mut result.requests);
            Ok(ExecutionArtifacts::Prefill {
                pending,
                result,
                scheduled_at_unix_s,
            })
        }
        ExecutionPlan::Decode => {
            let indices: Vec<usize> = (0..active.len()).collect();
            let requests = build_decode_items(active, &indices, rng);
            let mut result = executor.execute_decode(DecodePlan {
                requests: &requests,
            })?;
            sort_decode_results(&mut result.requests);
            Ok(ExecutionArtifacts::Decode { result })
        }
        ExecutionPlan::SpeculativeDecode => {
            let draft_requests = build_speculative_draft_items(active);
            let mut draft = executor.execute_speculative_draft(SpeculativeDraftPlan {
                requests: &draft_requests,
            })?;
            draft.requests.sort_by_key(|result| result.request_id);
            let verify_requests = build_speculative_verify_items(active, &draft.requests);
            let mut verify = executor.execute_speculative_verify(SpeculativeVerifyPlan {
                requests: &verify_requests,
            })?;
            verify.requests.sort_by_key(|result| result.request_id);
            Ok(ExecutionArtifacts::SpeculativeDecode { verify })
        }
        ExecutionPlan::Unified { pending } => {
            let scheduled_at_unix_s = openinfer_core::engine::unix_now_s();
            let pending_indices: Vec<usize> = (0..pending.len()).collect();
            let active_indices: Vec<usize> = (0..active.len()).collect();
            let prefill_requests = build_prefill_items(&pending, &pending_indices, rng);
            let decode_requests = build_decode_items(active, &active_indices, rng);
            let mut result = executor.execute_unified(UnifiedPlan {
                prefill_requests: &prefill_requests,
                decode_requests: &decode_requests,
            })?;
            sort_prefill_results(&mut result.prefill_requests);
            sort_decode_results(&mut result.decode_requests);
            Ok(ExecutionArtifacts::Unified {
                pending,
                result,
                scheduled_at_unix_s,
            })
        }
    }
}

pub(super) fn should_speculative_decode(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
) -> bool {
    executor.speculative_enabled()
        && !active.is_empty()
        && active.iter().all(|req| {
            executor.speculative_request_ready(req.request_id)
                && req.lora_adapter.is_none()
                && req.logprobs == 0
                && req.params.is_greedy()
        })
}

fn build_prefill_items(
    pending: &[PendingRequest],
    indices: &[usize],
    rng: &mut StdRng,
) -> Vec<PrefillStepItem> {
    indices
        .iter()
        .map(|&index| {
            let r = &pending[index];
            PrefillStepItem {
                request_id: r.request_id,
                prompt_tokens: r.prompt_tokens.clone(),
                max_output_tokens: r.max_tokens,
                params: r.params,
                logprobs: r.logprobs,
                echo: r.echo,
                lora_adapter: r.lora_adapter.clone(),
                random_val: rand::RngExt::random(rng),
                cached_tokens: r.cached_tokens,
                chunk_budget: r.step_chunk,
                chunk_start: 0,
                chunk_tokens: 0,
            }
        })
        .collect()
}

fn build_decode_items(
    active: &[ActiveRequestState],
    indices: &[usize],
    rng: &mut StdRng,
) -> Vec<DecodeStepItem> {
    indices
        .iter()
        .map(|&index| {
            let r = &active[index];
            DecodeStepItem {
                request_id: r.request_id,
                token_id: r.last_token,
                params: r.params,
                logprobs: r.logprobs,
                lora_adapter: r.lora_adapter.clone(),
                random_val: rand::RngExt::random(rng),
            }
        })
        .collect()
}

fn build_speculative_draft_items(active: &[ActiveRequestState]) -> Vec<SpeculativeDraftStepItem> {
    active
        .iter()
        .map(|r| SpeculativeDraftStepItem {
            request_id: r.request_id,
            current_token: r.last_token,
            params: r.params,
        })
        .collect()
}

fn build_speculative_verify_items(
    active: &[ActiveRequestState],
    draft_results: &[SpeculativeDraftRequestResult],
) -> Vec<SpeculativeVerifyStepItem> {
    draft_results
        .iter()
        .map(|draft| {
            let active = active
                .iter()
                .find(|req| req.request_id == draft.request_id)
                .expect("draft request_id must exist in active set");
            SpeculativeVerifyStepItem {
                request_id: draft.request_id,
                token_ids: {
                    let remaining = active.max_tokens.saturating_sub(active.generated_count);
                    assert!(remaining > 0, "active request must have output budget");
                    let mut token_ids = draft.token_ids.clone();
                    token_ids.truncate(remaining);
                    token_ids
                },
                params: active.params,
                lora_adapter: None,
            }
        })
        .collect()
}

fn sort_prefill_results(results: &mut [crate::executor::PrefillRequestResult]) {
    results.sort_by_key(|result| result.request_id);
}

fn sort_decode_results(results: &mut [crate::executor::DecodeRequestResult]) {
    results.sort_by_key(|result| result.request_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::RequestId;
    use openinfer_core::sampler::SamplingParams;

    fn pending() -> PendingRequest {
        let (token_tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        PendingRequest {
            request_id: RequestId::new(0),
            lora_adapter: None,
            prompt_tokens: vec![1, 2, 3],
            params: SamplingParams::default(),
            max_tokens: 8,
            token_tx,
            logprobs: 0,
            echo: false,
            queued_at_unix_s: None,
            prefetch_offered: false,
            prefill_pos: 0,
            step_chunk: 3,
            cached_tokens: 0,
        }
    }

    fn active(generated_count: usize, max_tokens: usize) -> ActiveRequestState {
        let (token_tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        ActiveRequestState {
            request_id: RequestId::new(7),
            lora_adapter: None,
            token_tx,
            last_token: 42,
            generated_count,
            max_tokens,
            prompt_len: 10,
            params: SamplingParams::default(),
            logprobs: 0,
        }
    }

    // The plan selector is the whole batch-formation policy: what the scheduler
    // does each tick is fully determined by (have_active, has_pending). Pin the
    // 2×2 truth table so a policy regression can't slip through silently.
    #[test]
    fn plan_selection_follows_active_and_pending_state() {
        assert!(
            build_next_plan(false, vec![]).is_none(),
            "idle scheduler (no active, no pending) produces no plan"
        );
        assert!(
            matches!(build_next_plan(true, vec![]), Some(ExecutionPlan::Decode)),
            "active-only ticks decode the running batch"
        );
        assert!(
            matches!(
                build_next_plan(false, vec![pending()]),
                Some(ExecutionPlan::Prefill { pending }) if pending.len() == 1
            ),
            "pending-only prefills the new arrivals"
        );
        assert!(
            matches!(
                build_next_plan(true, vec![pending()]),
                Some(ExecutionPlan::Unified { pending }) if pending.len() == 1
            ),
            "active + pending fuses prefill and decode into one unified step"
        );
    }

    #[test]
    fn speculative_verify_items_clamp_to_remaining_output_budget() {
        let active = [active(24, 32)];
        let draft = SpeculativeDraftRequestResult {
            request_id: RequestId::new(7),
            token_ids: (0..16).collect(),
        };

        let verify = build_speculative_verify_items(&active, &[draft]);

        assert_eq!(verify.len(), 1);
        assert_eq!(verify[0].token_ids.len(), 8);
        assert_eq!(verify[0].token_ids, (0..8).collect::<Vec<_>>());
    }
}
