//! Worker-side DFlash draft lane: the draft model plus per-request draft state.
//!
//! This lives on the worker thread next to the target model because the draft
//! rollout reads the target's embeddings/head and its captured hidden states.
//! The draft/verify boundary stays a pure token span — the hidden states are
//! private to this lane (`pending_context`), never crossing to the scheduler.

use std::collections::HashMap;

use anyhow::Result;
use openinfer_core::sampler::SamplingParams;
use openinfer_core::tensor::HiddenStates;

use super::dflash_prefill::{dflash_prefill_can_capture, should_capture_dflash_prefill_context};
use super::{LocalQwen3Lane, PrefillStepItem, RequestId};
use crate::dflash::{DFlashDraftModel, DFlashRequestState};
use crate::speculative::{
    DraftRequestResult, DraftResult, DraftStepItem, VerifyRequestResult, VerifyStepItem,
};

pub(super) struct DFlashLaneState {
    pub(super) model: DFlashDraftModel,
    pub(super) requests: HashMap<RequestId, DFlashRequestState>,
    verified_draft_tokens: usize,
    accepted_draft_tokens: usize,
}

impl DFlashLaneState {
    pub(super) fn new(model: DFlashDraftModel) -> Self {
        Self {
            model,
            requests: HashMap::new(),
            verified_draft_tokens: 0,
            accepted_draft_tokens: 0,
        }
    }
}

impl LocalQwen3Lane {
    /// Target layers whose hidden states the draft model consumes (None when
    /// DFlash is not loaded).
    pub(super) fn dflash_capture_layer_ids(&self) -> Option<Vec<usize>> {
        self.dflash
            .as_ref()
            .map(|dflash| dflash.model.target_layer_ids().to_vec())
    }

    pub(super) fn should_capture_dflash_prefill_context(
        &self,
        requests: &[PrefillStepItem],
    ) -> bool {
        let Some(dflash) = self.dflash.as_ref() else {
            return false;
        };
        should_capture_dflash_prefill_context(requests, |request_id| {
            dflash.requests.contains_key(&request_id)
        })
    }

    /// Fold target hidden states captured during prefill into each eligible
    /// request's pending context. Returns the requests that now have context.
    pub(super) fn record_prefill_dflash_context(
        &mut self,
        requests: &[PrefillStepItem],
        capture_requested: bool,
        captured_hidden: Option<&HiddenStates>,
    ) -> Result<Vec<RequestId>> {
        let Some(captured_hidden) = captured_hidden else {
            anyhow::ensure!(
                !capture_requested,
                "DFlash prefill context capture was requested but no hidden states were returned"
            );
            return Ok(Vec::new());
        };
        anyhow::ensure!(
            capture_requested,
            "DFlash prefill hidden states were returned without a capture request"
        );
        let Some(dflash) = self.dflash.as_mut() else {
            anyhow::bail!("DFlash prefill context record requested without DFlash");
        };
        let expected_tokens: usize = requests.iter().map(|req| req.chunk_tokens).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "DFlash prefill captured {} hidden rows for {} scheduled tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        let ctx = self.model.device_ctx().clone();
        let mut captured_requests = Vec::new();
        let mut token_offset = 0usize;
        for req in requests {
            let pending_exists = dflash.requests.contains_key(&req.request_id);
            if dflash_prefill_can_capture(req, pending_exists) {
                // Admission already caps the request at `draft.max_pos - block_size`
                // (see `max_context_tokens`), so this `.min` is a defensive floor:
                // it keeps the draft KV alloc within the draft's max positions even
                // if a caller bypasses admission.
                let max_cache_len =
                    (req.prompt_tokens.len() + req.max_output_tokens + dflash.model.block_size())
                        .min(dflash.model.max_position_embeddings());
                let mut state = match dflash.requests.remove(&req.request_id) {
                    Some(state) => state,
                    None => dflash.model.new_request_state(&ctx, max_cache_len)?,
                };
                let pending_len = state.pending_context_len().unwrap_or(0);
                anyhow::ensure!(
                    pending_len == req.chunk_start,
                    "DFlash prefill context for {:?} is discontinuous: pending={}, chunk_start={}",
                    req.request_id,
                    pending_len,
                    req.chunk_start
                );
                dflash.model.append_pending_context(
                    &ctx,
                    &mut state,
                    captured_hidden,
                    token_offset,
                    req.chunk_tokens,
                )?;
                dflash.requests.insert(req.request_id, state);
                captured_requests.push(req.request_id);
            } else {
                dflash.requests.remove(&req.request_id);
            }
            token_offset += req.chunk_tokens;
        }
        Ok(captured_requests)
    }

    /// Seed the next draft round from a verify step: append the target hidden
    /// states for the *accepted* span positions to each request's pending
    /// context, and log the cumulative acceptance rate.
    pub(super) fn record_verify_dflash_context(
        &mut self,
        requests: &[VerifyStepItem],
        results: &[VerifyRequestResult],
        captured_hidden: Option<&HiddenStates>,
    ) -> Result<()> {
        let Some(captured_hidden) = captured_hidden else {
            anyhow::bail!("DFlash verify context capture requested but no hidden states returned");
        };
        let Some(dflash) = self.dflash.as_mut() else {
            anyhow::bail!("DFlash verify context record requested without DFlash");
        };
        anyhow::ensure!(
            requests.len() == results.len(),
            "DFlash verify result count {} does not match request count {}",
            results.len(),
            requests.len()
        );
        let expected_tokens: usize = requests.iter().map(|req| req.token_ids.len()).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "DFlash verify captured {} hidden rows for {} scheduled tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        let ctx = self.model.device_ctx().clone();
        let mut token_offset = 0usize;
        for (req, result) in requests.iter().zip(results) {
            anyhow::ensure!(
                req.request_id == result.request_id,
                "DFlash verify result {:?} does not match request {:?}",
                result.request_id,
                req.request_id
            );
            let mut state = dflash.requests.remove(&req.request_id).ok_or_else(|| {
                anyhow::anyhow!("missing DFlash state after verify for {:?}", req.request_id)
            })?;
            // Only the accepted prefix's target hidden states are valid context
            // for the next draft; rejected drafts had the wrong continuation.
            dflash.model.append_pending_context(
                &ctx,
                &mut state,
                captured_hidden,
                token_offset,
                result.accepted_tokens.len(),
            )?;
            dflash.requests.insert(req.request_id, state);
            dflash.verified_draft_tokens += req.token_ids.len().saturating_sub(1);
            dflash.accepted_draft_tokens += result.matched_draft_tokens;
            let rate = if dflash.verified_draft_tokens == 0 {
                0.0
            } else {
                dflash.accepted_draft_tokens as f64 / dflash.verified_draft_tokens as f64
            };
            log::debug!(
                "Qwen3 DFlash request={} accepted_draft={} committed_tokens={} cumulative_accept_rate={:.3}",
                req.request_id.get(),
                result.matched_draft_tokens,
                result.accepted_tokens.len(),
                rate,
            );
            token_offset += req.token_ids.len();
        }
        Ok(())
    }

    /// Roll out one draft span per request: draft forward + greedy argmax over
    /// the block. Returns the verify span `[current_token, draft_1, …]`.
    pub(super) fn execute_dflash_draft(
        &mut self,
        requests: &[DraftStepItem],
    ) -> Result<DraftResult> {
        anyhow::ensure!(
            !requests.is_empty(),
            "DFlash draft requested without active requests"
        );
        for req in requests {
            anyhow::ensure!(
                req.params.is_greedy(),
                "DFlash draft currently supports greedy sampling only"
            );
        }

        // Take the lane out of `self` so the draft forward (which borrows
        // `self.model`) and the argmax (which borrows `self.sample_scratch`)
        // don't collide on a `self` borrow.
        let Some(mut dflash) = self.dflash.take() else {
            anyhow::bail!("DFlash draft requested but DFlash is not loaded");
        };
        let result = (|| -> Result<Vec<DraftRequestResult>> {
            let mut outputs = Vec::with_capacity(requests.len());
            for req in requests {
                let mut state = dflash.requests.remove(&req.request_id).ok_or_else(|| {
                    anyhow::anyhow!("missing DFlash state for {:?}", req.request_id)
                })?;
                let draft_logits =
                    dflash
                        .model
                        .draft_logits(&self.model, &mut state, req.current_token)?;
                let draft_len = draft_logits.seq_len;
                let greedy = SamplingParams::default();
                let params: Vec<&SamplingParams> = vec![&greedy; draft_len];
                let sampled = self.select_step_tokens(draft_logits, &params, 0)?;
                dflash.requests.insert(req.request_id, state);
                anyhow::ensure!(
                    sampled.len() == draft_len && sampled.len() >= 2,
                    "DFlash draft sampled {} tokens from {} logits columns",
                    sampled.len(),
                    draft_len
                );
                // Verify span = [current dangling token, draft_1, …, draft_{K}].
                let mut token_ids = Vec::with_capacity(sampled.len());
                token_ids.push(req.current_token);
                token_ids.extend(sampled.into_iter().skip(1));
                outputs.push(DraftRequestResult {
                    request_id: req.request_id,
                    token_ids,
                });
            }
            Ok(outputs)
        })();
        self.dflash = Some(dflash);
        Ok(DraftResult { requests: result? })
    }

    pub(super) fn drop_dflash_request(&mut self, request_id: RequestId) {
        if let Some(dflash) = self.dflash.as_mut() {
            dflash.requests.remove(&request_id);
        }
    }
}
