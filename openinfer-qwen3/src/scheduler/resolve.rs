use openinfer_core::engine::FinishReason;

use super::ActiveRequestState;
use super::PendingRequest;
use super::effects::DecodeEffect;
use super::effects::PendingEffect;
use super::effects::PromptEchoEffect;
use super::effects::ScheduledEffect;
use super::effects::StepEffects;
use super::plan::ExecutionArtifacts;
use crate::executor::DecodeRequestResult;
use crate::executor::ModelExecutor;
use crate::executor::PrefillRequestResult;
use crate::speculative::VerifyRequestResult;

pub(super) fn resolve_step(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
    artifacts: ExecutionArtifacts,
) -> StepEffects {
    match artifacts {
        ExecutionArtifacts::Prefill {
            pending,
            result,
            scheduled_at_unix_s,
        } => resolve_prefill_outputs(executor, pending, result.requests, scheduled_at_unix_s),
        ExecutionArtifacts::Decode { result } => StepEffects {
            scheduled: Vec::new(),
            prompt_echoes: Vec::new(),
            pending: Vec::new(),
            decode: resolve_decode_outputs(executor, active, &result.requests),
        },
        ExecutionArtifacts::SpeculativeDecode { verify } => StepEffects {
            scheduled: Vec::new(),
            prompt_echoes: Vec::new(),
            pending: Vec::new(),
            decode: resolve_speculative_outputs(executor, active, &verify.requests),
        },
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
            );
            effects.decode = resolve_decode_outputs(executor, active, &result.decode_requests);
            effects
        }
    }
}

/// Turn each request's accepted speculative span into a decode effect. A span
/// commits 1..=K+1 tokens at once; we walk it in order so a stop token or the
/// max-output budget truncates exactly where it lands (the executor already
/// suppressed nothing — stop handling lives here, mirroring single-token decode).
pub(super) fn resolve_speculative_outputs(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
    request_results: &[VerifyRequestResult],
) -> Vec<DecodeEffect> {
    request_results
        .iter()
        .map(|result| {
            let req = active
                .iter()
                .find(|req| req.request_id == result.request_id)
                .expect("speculative request_id must exist in active set");
            let mut emitted = Vec::new();
            let mut completion_tokens = req.generated_count;
            for &token in &result.accepted_tokens {
                completion_tokens += 1;
                let is_eos = !req.params.ignore_eos && executor.is_stop_token(token);
                if is_eos {
                    return DecodeEffect::EmitManyAndFinish {
                        request_id: result.request_id,
                        tokens: emitted,
                        finish_reason: FinishReason::Stop,
                        completion_tokens,
                    };
                }
                emitted.push(token);
                if completion_tokens >= req.max_tokens {
                    return DecodeEffect::EmitManyAndFinish {
                        request_id: result.request_id,
                        tokens: emitted,
                        finish_reason: FinishReason::Length,
                        completion_tokens,
                    };
                }
            }
            DecodeEffect::EmitManyAndContinue {
                request_id: result.request_id,
                tokens: emitted,
                completion_tokens,
            }
        })
        .collect()
}

fn resolve_prefill_outputs(
    executor: &impl ModelExecutor,
    pending: Vec<PendingRequest>,
    request_results: Vec<PrefillRequestResult>,
    scheduled_at_unix_s: f64,
) -> StepEffects {
    let mut effects = StepEffects::empty();
    for (mut req, result) in pending.into_iter().zip(request_results) {
        // Results are matched to requests positionally; a misalignment here
        // would deliver request A's tokens to request B, so fail loudly in
        // release builds too.
        assert_eq!(req.request_id, result.request_id);
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

    effects
}

fn resolve_decode_outputs(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
    request_results: &[DecodeRequestResult],
) -> Vec<DecodeEffect> {
    request_results
        .iter()
        .map(|result| {
            let req = active
                .iter()
                .find(|req| req.request_id == result.request_id)
                .expect("decode request_id must exist in active set");
            let completion_tokens = req.generated_count + 1;
            let is_eos = !req.params.ignore_eos && executor.is_stop_token(result.token);
            let at_limit = completion_tokens >= req.max_tokens;
            if is_eos {
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
            }
        })
        .collect()
}
