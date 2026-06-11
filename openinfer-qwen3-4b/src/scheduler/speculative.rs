//! Self-contained n-gram speculative decode step for the scheduler.
//!
//! When speculation is enabled this replaces the batched single-token decode
//! for the active set. Per request: propose drafts from the request's token
//! history, run one speculative verify (or fall back to a single decode when
//! there is no draft), then stream the committed tokens — applying stop /
//! max-token handling to each.
//!
//! Greedy speculation is lossless, so this emits the same tokens as the normal
//! decode path, just (often) several per step. It is kept isolated from the
//! generic plan/resolve/effects pipeline, which assumes exactly one token per
//! request per step.

use log::warn;
use openinfer_core::engine::FinishReason;

use crate::executor::{DecodePlan, DecodeStepItem, ModelExecutor, SpeculativeStepItem};
use crate::speculative::SpeculativeProposer;

use super::{ActiveRequestState, TokenEvent};

/// Advance every active request by one speculative step. Finished requests are
/// dropped from `active`. The proposer is method-agnostic ([`SpeculativeProposer`]);
/// everything below it (verify, KV rollback, acceptance, streaming) is reused
/// unchanged regardless of which proposer is plugged in.
pub(super) fn speculative_decode_step(
    executor: &mut impl ModelExecutor,
    active: &mut Vec<ActiveRequestState>,
    proposer: &dyn SpeculativeProposer,
) {
    let mut to_retire = Vec::new();
    for idx in 0..active.len() {
        let request_id = active[idx].request_id;
        let drafts = proposer.propose(&active[idx].token_history);

        let committed = match commit_tokens(executor, &active[idx], drafts) {
            Ok(tokens) => tokens,
            Err(e) => {
                warn!("speculative decode failed for {request_id:?}: {e}");
                let req = &active[idx];
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
                let _ = executor.drop_request(request_id);
                to_retire.push(idx);
                continue;
            }
        };

        if apply_committed(executor, &mut active[idx], &committed) {
            to_retire.push(idx);
        }
    }
    to_retire.sort_unstable();
    to_retire.dedup();
    for &i in to_retire.iter().rev() {
        active.swap_remove(i);
    }
}

/// Verify the drafts (or fall back to a single decode) and return the committed
/// tokens. Always at least one token.
fn commit_tokens(
    executor: &mut impl ModelExecutor,
    req: &ActiveRequestState,
    drafts: Vec<u32>,
) -> anyhow::Result<Vec<u32>> {
    if drafts.is_empty() {
        let result = executor.execute_decode(DecodePlan {
            requests: &[DecodeStepItem {
                request_id: req.request_id,
                token_id: req.last_token,
                params: req.params,
                logprobs: 0,
                lora_adapter: req.lora_adapter.clone(),
                random_val: 0.0,
            }],
        })?;
        Ok(result.requests.into_iter().map(|r| r.token).collect())
    } else {
        let mut item = SpeculativeStepItem::new(req.request_id, req.last_token, drafts);
        item.lora_adapter = req.lora_adapter.clone();
        executor.execute_speculative(&item)
    }
}

/// Stream `committed`, updating request state and applying stop / max-token
/// handling. Returns `true` when the request finished (caller retires it).
fn apply_committed(
    executor: &mut impl ModelExecutor,
    req: &mut ActiveRequestState,
    committed: &[u32],
) -> bool {
    for &token in committed {
        let completion = req.generated_count + 1;
        let is_eos = !req.params.ignore_eos && executor.is_stop_token(token);
        if req
            .token_tx
            .send(TokenEvent::Token {
                id: token,
                logprob: None,
            })
            .is_err()
        {
            // Client hung up: drop without a Finished event.
            let _ = executor.drop_request(req.request_id);
            return true;
        }
        req.generated_count = completion;
        req.last_token = token;
        req.token_history.push(token);

        let finish = if is_eos {
            Some(FinishReason::Stop)
        } else if completion >= req.max_tokens {
            Some(FinishReason::Length)
        } else {
            None
        };
        if let Some(finish_reason) = finish {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
            let _ = executor.drop_request(req.request_id);
            return true;
        }
    }
    false
}
