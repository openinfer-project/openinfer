//! Self-contained n-gram speculative decode step for the scheduler.
//!
//! When speculation is enabled this replaces the batched single-token decode
//! for the active set. Per request, IF the request is spec-eligible (greedy and
//! not asking for decode logprobs): propose drafts from its token history, run
//! one speculative verify (or fall back to a single decode when there is no
//! draft), then stream the committed tokens.
//!
//! Speculation verifies with **argmax** and drops per-token logprobs, so it is
//! only valid for greedy, no-logprobs requests. Any other request takes a
//! normal sampled single-token decode this tick (with its own sampling params,
//! logprobs, and a fresh `random_val`), so enabling speculation never silently
//! changes a sampled request's output or strips its requested logprobs.
//!
//! Greedy speculation is lossless, so eligible requests emit the same tokens as
//! the normal decode path, just (often) several per step. This is kept isolated
//! from the generic plan/resolve/effects pipeline, which assumes exactly one
//! token per request per step.

use log::warn;
use openinfer_core::engine::{FinishReason, TokenLogprob};
use rand::rngs::StdRng;

use crate::executor::{DecodePlan, DecodeStepItem, ModelExecutor, SpeculativeStepItem};
use crate::speculative::SpeculativeProposer;

use super::{ActiveRequestState, TokenEvent};

/// Advance every active request by one step. Spec-eligible requests take a
/// speculative step; the rest take a normal sampled decode. Finished requests
/// are dropped from `active`. The proposer is method-agnostic
/// ([`SpeculativeProposer`]).
pub(super) fn speculative_decode_step(
    executor: &mut impl ModelExecutor,
    active: &mut Vec<ActiveRequestState>,
    proposer: &dyn SpeculativeProposer,
    rng: &mut StdRng,
) {
    let mut to_retire = Vec::new();
    for idx in 0..active.len() {
        let request_id = active[idx].request_id;

        // Speculation is argmax-based and logprob-free: only greedy requests
        // that don't ask for decode logprobs may use it. Everyone else takes a
        // normal sampled decode so their semantics are preserved.
        let spec_eligible = active[idx].params.is_greedy() && active[idx].logprobs == 0;
        let outcome = if spec_eligible {
            speculative_one(executor, &mut active[idx], proposer)
        } else {
            sampled_decode_one(executor, &mut active[idx], rng)
        };

        match outcome {
            Ok(true) => to_retire.push(idx),
            Ok(false) => {}
            Err(e) => {
                warn!("decode step failed for {request_id:?}: {e}");
                let req = &active[idx];
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
                let _ = executor.drop_request(request_id);
                to_retire.push(idx);
            }
        }
    }
    to_retire.sort_unstable();
    to_retire.dedup();
    for &i in to_retire.iter().rev() {
        active.swap_remove(i);
    }
}

/// Greedy speculative step for one eligible request: propose, verify, commit,
/// stream. Returns `true` if the request finished.
fn speculative_one(
    executor: &mut impl ModelExecutor,
    req: &mut ActiveRequestState,
    proposer: &dyn SpeculativeProposer,
) -> anyhow::Result<bool> {
    let mut drafts = proposer.propose(&req.token_history);

    // A verify step commits at most `drafts.len() + 1` tokens. Cap the draft
    // count to the request's remaining token budget so the commit can never
    // exceed it — otherwise the executor's `schedule_speculative` clamps to the
    // budget and the larger accepted run is rejected. Tokens past the budget
    // would be truncated by `apply_committed` anyway.
    let remaining = req.max_tokens.saturating_sub(req.generated_count);
    let max_drafts = remaining.saturating_sub(1);
    if drafts.len() > max_drafts {
        drafts.truncate(max_drafts);
    }

    let committed = commit_tokens(executor, req, drafts)?;
    Ok(apply_committed(executor, req, &committed))
}

/// Normal sampled single-token decode for a request that isn't spec-eligible.
/// Uses the request's own sampling params, logprobs, and a fresh `random_val`,
/// then streams the one token (with its logprob). Returns `true` if finished.
fn sampled_decode_one(
    executor: &mut impl ModelExecutor,
    req: &mut ActiveRequestState,
    rng: &mut StdRng,
) -> anyhow::Result<bool> {
    let result = executor.execute_decode(DecodePlan {
        requests: &[DecodeStepItem {
            request_id: req.request_id,
            token_id: req.last_token,
            params: req.params,
            logprobs: req.logprobs,
            lora_adapter: req.lora_adapter.clone(),
            random_val: rand::RngExt::random(rng),
        }],
    })?;
    let out = result
        .requests
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("decode returned no result for {:?}", req.request_id))?;
    Ok(emit_one(executor, req, out.token, out.logprob))
}

/// Verify the drafts (or fall back to a single greedy decode) and return the
/// committed tokens. Always at least one token. Only called for spec-eligible
/// (greedy, no-logprobs) requests, so the fallback's `logprobs: 0` /
/// `random_val: 0.0` are correct.
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

/// Stream `committed` (no per-token logprobs — speculation is greedy), updating
/// request state and applying stop / max-token handling. Returns `true` when
/// the request finished (caller retires it).
fn apply_committed(
    executor: &mut impl ModelExecutor,
    req: &mut ActiveRequestState,
    committed: &[u32],
) -> bool {
    for &token in committed {
        if emit_one(executor, req, token, None) {
            return true;
        }
    }
    false
}

/// Emit one token, update request state, and apply stop / max-token handling.
/// Returns `true` when the request finished (caller retires it).
fn emit_one(
    executor: &mut impl ModelExecutor,
    req: &mut ActiveRequestState,
    token: u32,
    logprob: Option<TokenLogprob>,
) -> bool {
    let completion = req.generated_count + 1;
    let is_eos = !req.params.ignore_eos && executor.is_stop_token(token);
    if req
        .token_tx
        .send(TokenEvent::Token { id: token, logprob })
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
    false
}
