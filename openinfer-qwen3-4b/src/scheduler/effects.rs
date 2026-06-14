use tokio::sync::mpsc;

use log::debug;

use crate::executor::RequestId;
use openinfer_core::engine::{FinishReason, TokenLogprob};

use super::{ActiveRequestState, PendingRequest, TokenEvent};

pub(super) struct PromptEchoEffect {
    pub(super) token_tx: mpsc::UnboundedSender<TokenEvent>,
    pub(super) ids: Vec<u32>,
    pub(super) logprobs: Vec<Option<TokenLogprob>>,
}

/// Emitted once per request when its prefill result lands — carries the
/// prefix-cache hit count the frontend reports in usage (#246). The
/// scheduled timestamp was stamped when the batch was formed, not when the
/// event is sent, so queue-time metrics exclude prefill execution.
pub(super) struct ScheduledEffect {
    pub(super) token_tx: mpsc::UnboundedSender<TokenEvent>,
    pub(super) queued_at_unix_s: Option<f64>,
    pub(super) scheduled_at_unix_s: f64,
    pub(super) prompt_tokens: usize,
    pub(super) cached_tokens: usize,
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
    /// A non-final prefill chunk ran; the request goes back to the front of
    /// the prefilling queue with its progress updated.
    ContinuePrefill { req: PendingRequest },
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
    pub(super) scheduled: Vec<ScheduledEffect>,
    pub(super) prompt_echoes: Vec<PromptEchoEffect>,
    pub(super) pending: Vec<PendingEffect>,
    pub(super) decode: Vec<DecodeEffect>,
}

impl StepEffects {
    pub(super) fn empty() -> Self {
        Self {
            scheduled: Vec::new(),
            prompt_echoes: Vec::new(),
            pending: Vec::new(),
            decode: Vec::new(),
        }
    }
}

pub(super) fn apply_effects(
    executor: &mut impl crate::executor::ModelExecutor,
    active: &mut Vec<ActiveRequestState>,
    prefilling: &mut Vec<PendingRequest>,
    effects: StepEffects,
) {
    for scheduled in effects.scheduled {
        let _ = scheduled.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s: scheduled
                .queued_at_unix_s
                .unwrap_or(scheduled.scheduled_at_unix_s),
            scheduled_at_unix_s: scheduled.scheduled_at_unix_s,
            prompt_tokens: scheduled.prompt_tokens,
            cached_tokens: scheduled.cached_tokens,
        });
    }

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
                debug!(
                    "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                    request_id, req.prompt_len, completion_tokens, finish_reason
                );
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
                debug!(
                    "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                    request_id, req.prompt_len, completion_tokens, finish_reason
                );
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
                    debug!(
                        "request dropped: client disconnected: request_id={:?} tokens_generated={}",
                        request_id, completion_tokens
                    );
                    let _ = executor.drop_request(request_id);
                    to_retire.push(index);
                } else {
                    req.last_token = token;
                    req.generated_count = completion_tokens;
                }
            }
        }
    }
    to_retire.sort_unstable();
    to_retire.dedup();
    for &i in to_retire.iter().rev() {
        active.swap_remove(i);
    }

    // Requests that ran a non-final chunk this step came off the front of the
    // prefilling queue; splicing them back at the front (in step order, which
    // is request-id order) keeps the queue FIFO so chunked prompts finish
    // before newer arrivals start.
    let mut continued: Vec<PendingRequest> = Vec::new();
    for effect in effects.pending {
        match effect {
            PendingEffect::ContinuePrefill { req } => {
                if req.token_tx.is_closed() {
                    let _ = executor.drop_request(req.request_id);
                } else {
                    continued.push(req);
                }
            }
            PendingEffect::Finish {
                request_id,
                token_tx,
                finish_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                debug!(
                    "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                    request_id, prompt_tokens, completion_tokens, finish_reason
                );
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
                debug!(
                    "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                    request_id, prompt_tokens, completion_tokens, finish_reason
                );
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
    prefilling.splice(0..0, continued);
}
