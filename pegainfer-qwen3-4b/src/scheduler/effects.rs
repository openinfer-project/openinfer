use tokio::sync::mpsc;

use crate::executor::RequestId;
use pegainfer_core::engine::{FinishReason, TokenLogprob};

use super::{ActiveRequestState, TokenEvent};

pub(super) struct PromptEchoEffect {
    pub(super) token_tx: mpsc::UnboundedSender<TokenEvent>,
    pub(super) ids: Vec<u32>,
    pub(super) logprobs: Vec<Option<TokenLogprob>>,
}

pub(super) enum PendingEffect {
    Finish {
        request_id: RequestId,
        token_tx: mpsc::UnboundedSender<TokenEvent>,
        finish_reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    EmitAndFinish {
        request_id: RequestId,
        token_tx: mpsc::UnboundedSender<TokenEvent>,
        token: u32,
        logprob: Option<TokenLogprob>,
        finish_reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Promote {
        state: ActiveRequestState,
        first_token: u32,
        logprob: Option<TokenLogprob>,
    },
}

pub(super) enum DecodeEffect {
    Finish {
        request_id: RequestId,
        finish_reason: FinishReason,
        completion_tokens: usize,
    },
    EmitAndFinish {
        request_id: RequestId,
        token: u32,
        logprob: Option<TokenLogprob>,
        finish_reason: FinishReason,
        completion_tokens: usize,
    },
    EmitAndContinue {
        request_id: RequestId,
        token: u32,
        logprob: Option<TokenLogprob>,
        completion_tokens: usize,
    },
}

pub(super) struct StepEffects {
    pub(super) prompt_echoes: Vec<PromptEchoEffect>,
    pub(super) pending: Vec<PendingEffect>,
    pub(super) decode: Vec<DecodeEffect>,
}

impl StepEffects {
    pub(super) fn empty() -> Self {
        Self {
            prompt_echoes: Vec::new(),
            pending: Vec::new(),
            decode: Vec::new(),
        }
    }
}

pub(super) fn apply_effects(
    executor: &mut impl crate::executor::ModelExecutor,
    active: &mut Vec<ActiveRequestState>,
    effects: StepEffects,
) {
    for echo in effects.prompt_echoes {
        let _ = echo.token_tx.send(TokenEvent::PromptTokens {
            ids: echo.ids,
            logprobs: echo.logprobs,
        });
    }

    let mut to_retire = Vec::new();
    for effect in effects.decode {
        match effect {
            DecodeEffect::Finish {
                request_id,
                finish_reason,
                completion_tokens,
            } => {
                let Some(index) = active.iter().position(|req| req.request_id == request_id) else {
                    continue;
                };
                let req = &active[index];
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason,
                    prompt_tokens: req.prompt_len,
                    completion_tokens,
                });
                let _ = executor.drop_request(request_id);
                to_retire.push(index);
            }
            DecodeEffect::EmitAndFinish {
                request_id,
                token,
                logprob,
                finish_reason,
                completion_tokens,
            } => {
                let Some(index) = active.iter().position(|req| req.request_id == request_id) else {
                    continue;
                };
                let req = &active[index];
                if req
                    .token_tx
                    .send(TokenEvent::Token { id: token, logprob })
                    .is_ok()
                {
                    let _ = req.token_tx.send(TokenEvent::Finished {
                        finish_reason,
                        prompt_tokens: req.prompt_len,
                        completion_tokens,
                    });
                }
                let _ = executor.drop_request(request_id);
                to_retire.push(index);
            }
            DecodeEffect::EmitAndContinue {
                request_id,
                token,
                logprob,
                completion_tokens,
            } => {
                let Some(index) = active.iter().position(|req| req.request_id == request_id) else {
                    continue;
                };
                let req = &mut active[index];
                if req
                    .token_tx
                    .send(TokenEvent::Token { id: token, logprob })
                    .is_err()
                {
                    let _ = executor.drop_request(request_id);
                    to_retire.push(index);
                } else {
                    req.last_token = token;
                    req.generated_count = completion_tokens;
                }
            }
        }
    }
    for &i in to_retire.iter().rev() {
        active.swap_remove(i);
    }

    for effect in effects.pending {
        match effect {
            PendingEffect::Finish {
                request_id,
                token_tx,
                finish_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                let _ = token_tx.send(TokenEvent::Finished {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                });
                let _ = executor.drop_request(request_id);
            }
            PendingEffect::EmitAndFinish {
                request_id,
                token_tx,
                token,
                logprob,
                finish_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                if token_tx
                    .send(TokenEvent::Token { id: token, logprob })
                    .is_ok()
                {
                    let _ = token_tx.send(TokenEvent::Finished {
                        finish_reason,
                        prompt_tokens,
                        completion_tokens,
                    });
                }
                let _ = executor.drop_request(request_id);
            }
            PendingEffect::Promote {
                state,
                first_token,
                logprob,
            } => {
                if state
                    .token_tx
                    .send(TokenEvent::Token {
                        id: first_token,
                        logprob,
                    })
                    .is_ok()
                {
                    active.push(state);
                } else {
                    let _ = executor.drop_request(state.request_id);
                }
            }
        }
    }
}
