use crate::executor::{DecodeRequestResult, ModelExecutor, PrefillRequestResult};
use openinfer_core::engine::{ExecutionError, ExecutionResult, FinishReason};

use super::effects::{DecodeEffect, PendingEffect, PromptEchoEffect, ScheduledEffect, StepEffects};
use super::plan::ExecutionArtifacts;
use super::{ActiveRequestState, PendingRequest};

pub(super) fn resolve_step(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
    artifacts: ExecutionArtifacts,
) -> ExecutionResult<StepEffects> {
    match artifacts {
        ExecutionArtifacts::Prefill {
            pending,
            result,
            scheduled_at_unix_s,
        } => resolve_prefill_outputs(executor, pending, result.requests, scheduled_at_unix_s),
        ExecutionArtifacts::Decode { result } => Ok(StepEffects {
            scheduled: Vec::new(),
            prompt_echoes: Vec::new(),
            pending: Vec::new(),
            decode: resolve_decode_outputs(executor, active, &result.requests)?,
        }),
        ExecutionArtifacts::Unified {
            pending,
            result,
            scheduled_at_unix_s,
        } => {
            let mut effects = resolve_prefill_outputs(
                executor,
                pending,
                result.prefill_requests,
                scheduled_at_unix_s,
            )?;
            effects.decode = resolve_decode_outputs(executor, active, &result.decode_requests)?;
            Ok(effects)
        }
    }
}

fn resolve_prefill_outputs(
    executor: &impl ModelExecutor,
    pending: Vec<PendingRequest>,
    request_results: Vec<PrefillRequestResult>,
    scheduled_at_unix_s: f64,
) -> ExecutionResult<StepEffects> {
    if pending.len() != request_results.len() {
        return Err(ExecutionError::unexpected_worker_response(
            "prefill resolve",
            format!(
                "result count {} does not match request count {}",
                request_results.len(),
                pending.len()
            ),
        ));
    }

    let mut effects = StepEffects::empty();
    for (mut req, result) in pending.into_iter().zip(request_results) {
        // Results are matched to requests positionally; a misalignment here
        // would deliver request A's tokens to request B, so fail loudly in
        // release builds too.
        if req.request_id != result.request_id {
            return Err(ExecutionError::unexpected_worker_response(
                "prefill resolve",
                format!(
                    "result request id {:?} does not match pending {:?}",
                    result.request_id, req.request_id
                ),
            ));
        }
        let prompt_len = req.prompt_tokens.len();

        // Fire Scheduled on the request's first chunk only: queue time ends
        // when prompt work first reaches the GPU, and the prefix-cache hit
        // count is determined there. Later chunks must not re-send the event.
        if req.prefill_pos == 0 {
            effects.scheduled.push(ScheduledEffect {
                token_tx: req.token_tx.clone(),
                queued_at_unix_s: req.queued_at_unix_s,
                scheduled_at_unix_s,
                prompt_tokens: prompt_len,
                cached_tokens: result.cached_tokens,
            });
        }

        if !result.completed {
            req.prefill_pos = result.prefill_pos;
            req.cached_tokens = req.cached_tokens.max(result.cached_tokens);
            effects.pending.push(PendingEffect::ContinuePrefill { req });
            continue;
        }

        if req.echo {
            effects.prompt_echoes.push(PromptEchoEffect {
                token_tx: req.token_tx.clone(),
                ids: req.prompt_tokens.clone(),
                logprobs: result
                    .prompt_logprobs
                    .unwrap_or_else(|| vec![None; req.prompt_tokens.len()]),
            });
        }

        if !req.params.ignore_eos && executor.is_stop_token(result.first_token) {
            effects.pending.push(PendingEffect::Finish {
                request_id: req.request_id,
                token_tx: req.token_tx,
                finish_reason: FinishReason::Stop,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            continue;
        }

        if req.max_tokens <= 1 {
            effects.pending.push(PendingEffect::EmitAndFinish {
                request_id: req.request_id,
                token_tx: req.token_tx,
                token: result.first_token,
                logprob: result.first_token_logprob,
                finish_reason: FinishReason::Length,
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            });
            continue;
        }

        effects.pending.push(PendingEffect::Promote {
            state: ActiveRequestState {
                request_id: req.request_id,
                lora_adapter: req.lora_adapter,
                token_tx: req.token_tx,
                last_token: result.first_token,
                generated_count: 1,
                max_tokens: req.max_tokens,
                prompt_len,
                params: req.params,
                logprobs: req.logprobs,
            },
            first_token: result.first_token,
            logprob: result.first_token_logprob,
        });
    }

    Ok(effects)
}

fn resolve_decode_outputs(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
    request_results: &[DecodeRequestResult],
) -> ExecutionResult<Vec<DecodeEffect>> {
    let mut effects = Vec::with_capacity(request_results.len());
    for result in request_results {
        let req = active
            .iter()
            .find(|req| req.request_id == result.request_id)
            .ok_or_else(|| {
                ExecutionError::unexpected_worker_response(
                    "decode resolve",
                    format!("unknown request id {:?}", result.request_id),
                )
            })?;
        let completion_tokens = req.generated_count + 1;
        let is_eos = !req.params.ignore_eos && executor.is_stop_token(result.token);
        let at_limit = completion_tokens >= req.max_tokens;
        effects.push(if is_eos {
            DecodeEffect::Finish {
                request_id: result.request_id,
                finish_reason: FinishReason::Stop,
                completion_tokens,
            }
        } else if at_limit {
            DecodeEffect::EmitAndFinish {
                request_id: result.request_id,
                token: result.token,
                logprob: result.logprob.clone(),
                finish_reason: FinishReason::Length,
                completion_tokens,
            }
        } else {
            DecodeEffect::EmitAndContinue {
                request_id: result.request_id,
                token: result.token,
                logprob: result.logprob.clone(),
                completion_tokens,
            }
        });
    }
    Ok(effects)
}
