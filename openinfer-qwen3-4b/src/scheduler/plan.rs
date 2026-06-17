use anyhow::Result;
use rand::rngs::StdRng;

use crate::executor::{
    DecodePlan, DecodeResult, DecodeStepItem, ModelExecutor, PrefillPlan, PrefillResult,
    PrefillStepItem, UnifiedPlan, UnifiedResult,
};

use super::{ActiveRequestState, PendingRequest};

pub(super) enum ExecutionPlan {
    Prefill { pending: Vec<PendingRequest> },
    Decode,
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
    Unified {
        pending: Vec<PendingRequest>,
        result: UnifiedResult,
        scheduled_at_unix_s: f64,
    },
}

/// Whether the batch needs all-position prompt logprobs, i.e. it has an echo
/// request that also asked for logprobs. Echo alone only echoes ids back.
fn batch_needs_prompt_logprobs(pending: &[PendingRequest]) -> bool {
    pending.iter().any(|req| req.echo && req.logprobs > 0)
}

pub(super) fn build_next_plan(
    have_active: bool,
    pending: Vec<PendingRequest>,
) -> Option<ExecutionPlan> {
    // echo+logprobs needs a dedicated Prefill: the unified forward can't produce
    // all-position logits. Active decodes wait those ticks; it's rare.
    let needs_dedicated_prefill = batch_needs_prompt_logprobs(&pending);
    if !pending.is_empty() && have_active && !needs_dedicated_prefill {
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
            // `echo` here = "compute all-position logits"; only echo+logprobs
            // needs them.
            let mut result = executor.execute_prefill(PrefillPlan {
                requests: &requests,
                echo: batch_needs_prompt_logprobs(&pending),
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

    // Regression for #372: an echo+logprobs request must not be fused into a
    // Unified step (the unified forward can't produce all-position prompt
    // logprobs), even when decodes are active. It takes a dedicated Prefill.
    #[test]
    fn echo_logprobs_pending_forces_dedicated_prefill() {
        let echo_logprobs = || {
            let mut p = pending();
            p.echo = true;
            p.logprobs = 5;
            p
        };
        assert!(
            matches!(
                build_next_plan(true, vec![echo_logprobs()]),
                Some(ExecutionPlan::Prefill { pending }) if pending.len() == 1
            ),
            "echo+logprobs must take a dedicated prefill, not a unified step"
        );
        // A mixed batch with any echo+logprobs request also takes prefill.
        assert!(
            matches!(
                build_next_plan(true, vec![pending(), echo_logprobs()]),
                Some(ExecutionPlan::Prefill { .. })
            ),
            "a batch containing an echo+logprobs request takes prefill"
        );
        // echo without logprobs still fuses (no all-position logits needed).
        let echo_only = || {
            let mut p = pending();
            p.echo = true;
            p
        };
        assert!(
            matches!(
                build_next_plan(true, vec![echo_only()]),
                Some(ExecutionPlan::Unified { .. })
            ),
            "echo without logprobs does not need a dedicated prefill"
        );
    }

    // All-position logits fire only for echo+logprobs: plain, echo-only, and
    // logprobs-only batches must not, a mixed batch with one such request must.
    #[test]
    fn all_position_logits_gated_on_echo_plus_logprobs() {
        let echo_logprobs = || {
            let mut p = pending();
            p.echo = true;
            p.logprobs = 5;
            p
        };
        let echo_only = || {
            let mut p = pending();
            p.echo = true;
            p
        };
        let logprobs_only = || {
            let mut p = pending();
            p.logprobs = 5;
            p
        };

        assert!(
            !batch_needs_prompt_logprobs(&[pending()]),
            "a plain prompt needs no all-position logits"
        );
        assert!(
            !batch_needs_prompt_logprobs(&[echo_only()]),
            "echo without logprobs only echoes ids back — no all-position logits"
        );
        assert!(
            !batch_needs_prompt_logprobs(&[logprobs_only()]),
            "logprobs without echo only needs the sampled token's logprob"
        );
        assert!(
            batch_needs_prompt_logprobs(&[echo_logprobs()]),
            "echo+logprobs needs all-position logits"
        );
        assert!(
            batch_needs_prompt_logprobs(&[echo_only(), echo_logprobs()]),
            "one echo+logprobs request in the batch turns the scratch on"
        );
    }
}
