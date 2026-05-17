use crate::executor::{DecodeRequestResult, ModelExecutor, PrefillRequestResult};
use pegainfer_core::engine::FinishReason;

use super::effects::{DecodeEffect, PendingEffect, PromptEchoEffect, StepEffects};
use super::plan::ExecutionArtifacts;
use super::{ActiveRequestState, PendingRequest};

pub(super) fn resolve_step(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
    artifacts: ExecutionArtifacts,
) -> StepEffects {
    match artifacts {
        ExecutionArtifacts::Prefill { pending, result } => {
            resolve_prefill_outputs(executor, pending, result.requests)
        }
        ExecutionArtifacts::Decode { result } => StepEffects {
            prompt_echoes: Vec::new(),
            pending: Vec::new(),
            decode: resolve_decode_outputs(executor, active, &result.requests),
        },
        ExecutionArtifacts::Unified { pending, result } => {
            let mut effects = resolve_prefill_outputs(executor, pending, result.prefill_requests);
            effects.decode = resolve_decode_outputs(executor, active, &result.decode_requests);
            effects
        }
    }
}

fn resolve_prefill_outputs(
    executor: &impl ModelExecutor,
    pending: Vec<PendingRequest>,
    request_results: Vec<PrefillRequestResult>,
) -> StepEffects {
    let mut effects = StepEffects::empty();
    for (req, result) in pending.into_iter().zip(request_results) {
        debug_assert_eq!(req.request_id, result.request_id);
        let prompt_len = req.prompt_tokens.len();

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
