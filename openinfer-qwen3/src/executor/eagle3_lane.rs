//! Worker-side EAGLE-3 draft lane: the drafter plus per-request draft state.
//!
//! Parallels [`super::dflash_lane::DFlashLaneState`] but for the EAGLE-3 chain
//! drafter. Lives on the worker thread next to the target model because the draft
//! rollout reuses the target's `embed_tokens` and reads its captured low/mid/high
//! hidden states; the draft/verify boundary stays a pure token span, with the
//! captured features private to this lane.
//!
//! Flow: the prefill capture hook seeds each eligible [`Eagle3RequestState`],
//! `execute_eagle3_draft` rolls the top-1 chain forward, `record_verify_eagle3_context`
//! re-seeds from verify. Verify/accept/commit is the shared, drafter-agnostic path —
//! losslessness comes from there, not here.

use std::collections::HashMap;

use anyhow::Result;
use openinfer_core::tensor::{DeviceContext, HiddenStates};

// Prefill-cleanliness predicates are spec-agnostic (greedy, no LoRA, no prefix
// hit, …), so EAGLE-3 reuses DFlash's.
use super::dflash_prefill::{dflash_prefill_can_capture, should_capture_dflash_prefill_context};
use super::{LocalQwen3Lane, PrefillStepItem, RequestId};
use crate::eagle3::{Eagle3DraftModel, Eagle3RequestState, Eagle3Scratch};
use crate::speculative::{
    DraftRequestResult, DraftResult, DraftStepItem, VerifyRequestResult, VerifyStepItem,
};

pub(super) struct Eagle3LaneState {
    pub(super) model: Eagle3DraftModel,
    /// Per-request single-layer draft KV + carried residual state. Populated by
    /// the prefill capture hook once a prompt finishes prefilling.
    pub(super) requests: HashMap<RequestId, Eagle3RequestState>,
    /// Single-token draft scratch, reused across steps. v1 drafts one request at
    /// a time (`seq_len == 1`); batching the chain is a follow-up.
    pub(super) scratch: Eagle3Scratch,
    /// Target layers whose hidden states feed the drafter (low/mid/high).
    pub(super) aux_layer_ids: [usize; 3],
    /// Cumulative drafted tokens offered to verify (denominator of the acceptance
    /// rate) — a diagnostic for chain quality / speedup headroom.
    verified_draft_tokens: usize,
    /// Cumulative drafts accepted (matched the target argmax). The acceptance rate
    /// `accepted / verified` bounds the achievable speculative speedup.
    accepted_draft_tokens: usize,
}

impl Eagle3LaneState {
    pub(super) fn new(
        ctx: &DeviceContext,
        model: Eagle3DraftModel,
        aux_layer_ids: [usize; 3],
    ) -> Result<Self> {
        let scratch = model.new_scratch(ctx)?;
        Ok(Self {
            model,
            requests: HashMap::new(),
            scratch,
            aux_layer_ids,
            verified_draft_tokens: 0,
            accepted_draft_tokens: 0,
        })
    }
}

impl LocalQwen3Lane {
    /// Target layers whose hidden states the drafter consumes
    pub(super) fn eagle3_capture_layer_ids(&self) -> Option<Vec<usize>> {
        self.eagle3
            .as_ref()
            .map(|eagle3| eagle3.aux_layer_ids.to_vec())
    }

    pub(super) fn should_capture_eagle3_prefill_context(
        &self,
        requests: &[PrefillStepItem],
    ) -> bool {
        let Some(eagle3) = self.eagle3.as_ref() else {
            return false;
        };
        should_capture_dflash_prefill_context(requests, |request_id| {
            eagle3.requests.contains_key(&request_id)
        })
    }

    /// Fold target hidden states captured during prefill into each eligible
    /// request's draft state: run the teacher-forced draft prefill over the
    /// chunk (appending draft KV) and record the chain seed. Returns the requests
    /// that captured context this step.
    pub(super) fn record_prefill_eagle3_context(
        &mut self,
        requests: &[PrefillStepItem],
        capture_requested: bool,
        captured_hidden: Option<&HiddenStates>,
    ) -> Result<Vec<RequestId>> {
        let Some(captured_hidden) = captured_hidden else {
            anyhow::ensure!(
                !capture_requested,
                "EAGLE-3 prefill context capture was requested but no hidden states were returned"
            );
            return Ok(Vec::new());
        };
        anyhow::ensure!(
            capture_requested,
            "EAGLE-3 prefill hidden states were returned without a capture request"
        );
        let expected_tokens: usize = requests.iter().map(|req| req.chunk_tokens).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "EAGLE-3 prefill captured {} hidden rows for {} scheduled tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        // Split-borrow: the draft prefill borrows `&self.model` (target) while it
        // mutates `&mut self.eagle3`; the two fields are disjoint.
        let LocalQwen3Lane { model, eagle3, .. } = self;
        let Some(eagle3) = eagle3.as_mut() else {
            anyhow::bail!("EAGLE-3 prefill context record requested without EAGLE-3");
        };
        let max_pos = eagle3.model.config.max_position_embeddings;
        let mut captured_requests = Vec::new();
        let mut token_offset = 0usize;
        for req in requests {
            // v1 captures only single-chunk prompts: the EAGLE feature↔token shift
            // must not cross a prefill-chunk boundary. A prompt that was chunked
            // (or too short to draft) simply isn't captured and falls back to plain
            // decode — still correct, just not accelerated.
            let single_chunk = req.chunk_start == 0 && req.chunk_tokens == req.prompt_tokens.len();
            // The chain writes EAGLE3_CHAIN_LENGTH transient draft positions past the
            // committed length each round (rewound afterwards); those must stay under
            // the drafter's position limit `max_pos`. A request that could generate to
            // within a chain of `max_pos` can't be drafted for its final positions —
            // `draft_step` would bail and turn a valid length-capped generation into an
            // error — so skip EAGLE for it entirely (plain decode) rather than admit it
            // with the clamp silently eating the chain headroom.
            let fits_position_budget = req.prompt_tokens.len()
                + req.max_output_tokens
                + crate::eagle3::EAGLE3_CHAIN_LENGTH
                <= max_pos;
            if dflash_prefill_can_capture(req, false)
                && single_chunk
                && req.prompt_tokens.len() >= 2
                && fits_position_budget
            {
                // Prompt + decode + one chain's worth of in-flight draft KV
                let max_cache_len = (req.prompt_tokens.len()
                    + req.max_output_tokens
                    + crate::eagle3::EAGLE3_CHAIN_LENGTH)
                    .min(max_pos)
                    .max(1);
                let mut state = eagle3
                    .model
                    .new_request_state(model.device_ctx(), max_cache_len)?;
                eagle3.model.prefill_prompt(
                    model,
                    &mut state,
                    captured_hidden,
                    token_offset,
                    &req.prompt_tokens,
                )?;
                eagle3.requests.insert(req.request_id, state);
                captured_requests.push(req.request_id);
            } else {
                eagle3.requests.remove(&req.request_id);
            }
            token_offset += req.chunk_tokens;
        }
        Ok(captured_requests)
    }

    /// One EAGLE-3 speculative draft round per request: roll the top-1 chain
    /// forward `EAGLE3_CHAIN_LENGTH` tokens from each request's carried seed.
    /// Returns the verify span `[current_token, draft_1, …, draft_k]`. The chain's
    /// speculative draft KV is rewound inside `chain_round`; the accepted prefix
    /// is rebuilt teacher-forced by the verify step's re-seed.
    pub(super) fn execute_eagle3_draft(
        &mut self,
        requests: &[DraftStepItem],
    ) -> Result<DraftResult> {
        anyhow::ensure!(
            !requests.is_empty(),
            "EAGLE-3 draft requested without active requests"
        );
        for req in requests {
            anyhow::ensure!(
                req.params.is_greedy(),
                "EAGLE-3 draft currently supports greedy sampling only"
            );
        }
        let k = crate::eagle3::EAGLE3_CHAIN_LENGTH;
        // Split-borrow: the chain forward reads `&self.model` (target embeddings)
        // while mutating the draft lane.
        let LocalQwen3Lane { model, eagle3, .. } = self;
        let Some(eagle3) = eagle3.as_mut() else {
            anyhow::bail!("EAGLE-3 draft requested but EAGLE-3 is not loaded");
        };
        let Eagle3LaneState {
            model: draft,
            requests: state_map,
            scratch,
            ..
        } = eagle3;
        let mut out = Vec::with_capacity(requests.len());
        for req in requests {
            let state = state_map
                .get_mut(&req.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing EAGLE-3 state for {:?}", req.request_id))?;
            let drafts = draft.chain_round(model, state, scratch, req.current_token, k)?;
            let mut token_ids = Vec::with_capacity(k + 1);
            token_ids.push(req.current_token);
            token_ids.extend(drafts);
            out.push(DraftRequestResult {
                request_id: req.request_id,
                token_ids,
            });
        }
        Ok(DraftResult { requests: out })
    }

    /// Re-seed each request's chain from the verify step's captured target hidden:
    /// teacher-force the accepted prefix (rebuilding its draft KV) and store the
    /// next chain seed. The EAGLE counterpart to `record_verify_dflash_context`,
    /// but it re-injects the target hidden rather than appending to a pending buffer.
    pub(super) fn record_verify_eagle3_context(
        &mut self,
        requests: &[VerifyStepItem],
        results: &[VerifyRequestResult],
        captured_hidden: &HiddenStates,
    ) -> Result<()> {
        anyhow::ensure!(
            requests.len() == results.len(),
            "EAGLE-3 verify result count {} does not match request count {}",
            results.len(),
            requests.len()
        );
        let expected_tokens: usize = requests.iter().map(|req| req.token_ids.len()).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "EAGLE-3 verify captured {} hidden rows for {} span tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        let LocalQwen3Lane { model, eagle3, .. } = self;
        let Some(eagle3) = eagle3.as_mut() else {
            anyhow::bail!("EAGLE-3 verify context record requested without EAGLE-3");
        };
        let Eagle3LaneState {
            model: draft,
            requests: state_map,
            verified_draft_tokens,
            accepted_draft_tokens,
            ..
        } = eagle3;
        let mut token_offset = 0usize;
        for (req, result) in requests.iter().zip(results) {
            anyhow::ensure!(
                req.request_id == result.request_id,
                "EAGLE-3 verify result {:?} does not match request {:?}",
                result.request_id,
                req.request_id
            );
            let state = state_map.get_mut(&req.request_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing EAGLE-3 state after verify for {:?}",
                    req.request_id
                )
            })?;
            draft.reseed_after_verify(
                model,
                state,
                captured_hidden,
                token_offset,
                &req.token_ids,
                result.matched_draft_tokens,
            )?;
            // Acceptance bookkeeping (greedy argmax match): drafts offered this
            // round = span len - 1 (span[0] is `current_token`, the confirmed last
            // token, not a draft); accepted = the matched-draft prefix. The extra
            // committed token is the target's bonus argmax, not a draft, so it stays
            // out of the rate. The cumulative rate bounds the achievable speculative speedup.
            *verified_draft_tokens += req.token_ids.len().saturating_sub(1);
            *accepted_draft_tokens += result.matched_draft_tokens;
            let rate = if *verified_draft_tokens == 0 {
                0.0
            } else {
                *accepted_draft_tokens as f64 / *verified_draft_tokens as f64
            };
            log::debug!(
                "Qwen3 EAGLE-3 request={} accepted_draft={} committed_tokens={} cumulative_accept_rate={:.3}",
                req.request_id.get(),
                result.matched_draft_tokens,
                result.accepted_tokens.len(),
                rate,
            );
            token_offset += req.token_ids.len();
        }
        Ok(())
    }

    /// Drop a request's EAGLE-3 draft state (request retired, or a plain decode
    /// advanced the sequence outside the speculative path). No-op when EAGLE-3 is
    /// not loaded or the request was never captured.
    pub(super) fn drop_eagle3_request(&mut self, request_id: RequestId) {
        if let Some(eagle3) = self.eagle3.as_mut() {
            eagle3.requests.remove(&request_id);
        }
    }
}
