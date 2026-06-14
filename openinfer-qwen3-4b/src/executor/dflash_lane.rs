use std::collections::HashMap;
use std::env;
use std::time::Instant;

use anyhow::Result;

use crate::dflash::{DFlashDraftModel, DFlashRequestState};
use crate::speculative::{
    DraftRequestResult as SpeculativeDraftRequestResult, DraftResult as SpeculativeDraftResult,
    DraftStepItem as SpeculativeDraftStepItem,
    VerifyRequestResult as SpeculativeVerifyRequestResult,
    VerifyStepItem as SpeculativeVerifyStepItem,
};
use openinfer_core::tensor::HiddenStates;

use super::dflash_prefill::{
    dflash_prefill_can_capture,
    should_capture_dflash_prefill_context as should_capture_dflash_prefill_context_with_state,
};
use super::worker::LocalQwen3Lane;
use super::{PrefillStepItem, RequestId};

pub(super) struct DFlashLaneState {
    pub(super) model: DFlashDraftModel,
    pub(super) requests: HashMap<RequestId, DFlashRequestState>,
    verified_tokens: usize,
    accepted_tokens: usize,
    draft_steps: usize,
    draft_ms_total: f64,
    verify_steps: usize,
    verify_ms_total: f64,
    nvtx_enabled: bool,
    last_draft_ms: HashMap<RequestId, f64>,
    last_draft_context_tokens: HashMap<RequestId, usize>,
    last_draft_committed_context: HashMap<RequestId, usize>,
}

impl DFlashLaneState {
    pub(super) fn new(model: DFlashDraftModel) -> Self {
        let nvtx_enabled = dflash_nvtx_enabled_from_env();
        if nvtx_enabled {
            log::info!("Qwen3 DFlash NVTX ranges enabled");
        }
        Self {
            model,
            requests: HashMap::new(),
            verified_tokens: 0,
            accepted_tokens: 0,
            draft_steps: 0,
            draft_ms_total: 0.0,
            verify_steps: 0,
            verify_ms_total: 0.0,
            nvtx_enabled,
            last_draft_ms: HashMap::new(),
            last_draft_context_tokens: HashMap::new(),
            last_draft_committed_context: HashMap::new(),
        }
    }
}

impl LocalQwen3Lane {
    pub(super) fn dflash_capture_layer_ids(&self) -> Option<Vec<usize>> {
        self.dflash
            .as_ref()
            .map(|dflash| dflash.model.target_layer_ids().to_vec())
    }

    pub(super) fn dflash_nvtx_enabled(&self) -> bool {
        self.dflash
            .as_ref()
            .is_some_and(|dflash| dflash.nvtx_enabled)
    }

    pub(super) fn should_capture_dflash_prefill_context(
        &self,
        requests: &[PrefillStepItem],
    ) -> bool {
        let Some(dflash) = self.dflash.as_ref() else {
            return false;
        };
        should_capture_dflash_prefill_context_with_state(requests, |request_id| {
            dflash.requests.contains_key(&request_id)
        })
    }

    pub(super) fn start_dflash_timing(&self) -> Result<Option<Instant>> {
        if !self.dflash_nvtx_enabled() {
            return Ok(None);
        }
        self.model.device_ctx().sync()?;
        Ok(Some(Instant::now()))
    }

    pub(super) fn finish_dflash_timing(&self, start: Option<Instant>) -> Result<Option<f64>> {
        let Some(start) = start else {
            return Ok(None);
        };
        self.model.device_ctx().sync()?;
        Ok(Some(start.elapsed().as_secs_f64() * 1000.0))
    }

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
                let max_cache_len =
                    req.prompt_tokens.len() + req.max_output_tokens + dflash.model.block_size();
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

    pub(super) fn record_verify_dflash_context(
        &mut self,
        requests: &[SpeculativeVerifyStepItem],
        results: &[SpeculativeVerifyRequestResult],
        captured_hidden: Option<&HiddenStates>,
        verify_ms: Option<f64>,
    ) -> Result<()> {
        let Some(captured_hidden) = captured_hidden else {
            anyhow::bail!(
                "DFlash speculative verify context capture was requested but no hidden states were returned"
            );
        };
        let Some(dflash) = self.dflash.as_mut() else {
            anyhow::bail!("DFlash speculative verify context record requested without DFlash");
        };
        anyhow::ensure!(
            requests.len() == results.len(),
            "DFlash speculative verify result count {} does not match request count {}",
            results.len(),
            requests.len()
        );
        let expected_tokens: usize = requests.iter().map(|req| req.token_ids.len()).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "DFlash speculative verify captured {} hidden rows for {} scheduled tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        for (req, result) in requests.iter().zip(results) {
            anyhow::ensure!(
                req.request_id == result.request_id,
                "DFlash speculative verify result {:?} does not match request {:?}",
                result.request_id,
                req.request_id
            );
            anyhow::ensure!(
                dflash.requests.contains_key(&req.request_id),
                "missing DFlash state after speculative verify for {:?}",
                req.request_id
            );
        }
        if let Some(ms) = verify_ms {
            dflash.verify_steps += 1;
            dflash.verify_ms_total += ms;
        }
        let ctx = self.model.device_ctx().clone();
        let mut token_offset = 0usize;
        for (req, result) in requests.iter().zip(results) {
            let Some(mut state) = dflash.requests.remove(&req.request_id) else {
                dflash.last_draft_ms.remove(&req.request_id);
                dflash.last_draft_context_tokens.remove(&req.request_id);
                dflash.last_draft_committed_context.remove(&req.request_id);
                anyhow::bail!(
                    "missing DFlash state after speculative verify for {:?}",
                    req.request_id
                );
            };
            dflash.model.append_pending_context(
                &ctx,
                &mut state,
                captured_hidden,
                token_offset,
                result.accepted_tokens.len(),
            )?;
            dflash.requests.insert(req.request_id, state);
            dflash.verified_tokens += req.token_ids.len().saturating_sub(1);
            dflash.accepted_tokens += result.matched_draft_tokens;
            let rate = if dflash.verified_tokens == 0 {
                0.0
            } else {
                dflash.accepted_tokens as f64 / dflash.verified_tokens as f64
            };
            let draft_ms = dflash.last_draft_ms.remove(&req.request_id).unwrap_or(-1.0);
            let draft_context_tokens = dflash
                .last_draft_context_tokens
                .remove(&req.request_id)
                .unwrap_or(0);
            let draft_committed_context = dflash
                .last_draft_committed_context
                .remove(&req.request_id)
                .unwrap_or(0);
            let avg_draft_ms = if dflash.draft_steps == 0 {
                -1.0
            } else {
                dflash.draft_ms_total / dflash.draft_steps as f64
            };
            let avg_verify_ms = if dflash.verify_steps == 0 {
                -1.0
            } else {
                dflash.verify_ms_total / dflash.verify_steps as f64
            };
            log::info!(
                "Qwen3 DFlash request={} accepted_draft={} verified_draft={} committed_tokens={} cumulative_accept_rate={:.3} draft_ms={:.3} verify_ms={:.3} avg_draft_ms={:.3} avg_verify_ms={:.3} draft_context_tokens={} draft_committed_context={}",
                req.request_id.get(),
                result.matched_draft_tokens,
                req.token_ids.len().saturating_sub(1),
                result.accepted_tokens.len(),
                rate,
                draft_ms,
                verify_ms.unwrap_or(-1.0),
                avg_draft_ms,
                avg_verify_ms,
                draft_context_tokens,
                draft_committed_context,
            );
            token_offset += req.token_ids.len();
        }
        Ok(())
    }

    pub(super) fn execute_dflash_draft(
        &mut self,
        requests: &[SpeculativeDraftStepItem],
    ) -> Result<SpeculativeDraftResult> {
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

        let ctx = self.model.device_ctx().clone();
        let profiling_enabled = self.dflash_nvtx_enabled();
        let draft_nvtx_range = if self.dflash_nvtx_enabled() {
            Some(nvtx::range!("qwen3.dflash.draft"))
        } else {
            None
        };
        let Some(mut dflash) = self.dflash.take() else {
            anyhow::bail!("DFlash draft requested but DFlash is not loaded");
        };
        let draft_results =
            (|| -> Result<Vec<(SpeculativeDraftRequestResult, usize, usize, Option<f64>)>> {
                let mut outputs = Vec::with_capacity(requests.len());
                for req in requests {
                    if profiling_enabled {
                        ctx.sync()?;
                    }
                    let draft_start = profiling_enabled.then(Instant::now);
                    let mut state = dflash.requests.remove(&req.request_id).ok_or_else(|| {
                        anyhow::anyhow!("missing DFlash state for {:?}", req.request_id)
                    })?;
                    state.pending_context_len().ok_or_else(|| {
                        anyhow::anyhow!(
                            "DFlash draft requested before target hidden context is available"
                        )
                    })?;
                    let draft =
                        dflash
                            .model
                            .draft_logits(&self.model, &mut state, req.current_token)?;
                    let draft_len = draft.logits.seq_len;
                    let sampled = self.select_greedy_contiguous_tokens(draft.logits)?;
                    let context_tokens = draft.context_len;
                    let committed_context = draft.committed_len;
                    dflash.requests.insert(req.request_id, state);
                    if profiling_enabled {
                        ctx.sync()?;
                    }
                    let draft_ms = draft_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);
                    anyhow::ensure!(
                        sampled.len() == draft_len && sampled.len() >= 2,
                        "DFlash draft sampled {} tokens from {} logits columns",
                        sampled.len(),
                        draft_len
                    );
                    let mut token_ids = Vec::with_capacity(sampled.len());
                    token_ids.push(req.current_token);
                    token_ids.extend(sampled.into_iter().skip(1));
                    outputs.push((
                        SpeculativeDraftRequestResult {
                            request_id: req.request_id,
                            token_ids,
                        },
                        context_tokens,
                        committed_context,
                        draft_ms,
                    ));
                }
                Ok(outputs)
            })();
        self.dflash = Some(dflash);
        let draft_results = draft_results?;
        drop(draft_nvtx_range);
        if let Some(dflash) = self.dflash.as_mut() {
            for (draft, context_tokens, committed_context, draft_ms) in &draft_results {
                if let Some(draft_ms) = *draft_ms {
                    dflash.draft_steps += 1;
                    dflash.draft_ms_total += draft_ms;
                    dflash.last_draft_ms.insert(draft.request_id, draft_ms);
                }
                dflash
                    .last_draft_context_tokens
                    .insert(draft.request_id, *context_tokens);
                dflash
                    .last_draft_committed_context
                    .insert(draft.request_id, *committed_context);
            }
        }
        Ok(SpeculativeDraftResult {
            requests: draft_results
                .into_iter()
                .map(|(draft, _, _, _)| draft)
                .collect(),
        })
    }

    pub(super) fn drop_dflash_request(&mut self, request_id: RequestId) {
        if let Some(dflash) = self.dflash.as_mut() {
            dflash.requests.remove(&request_id);
            dflash.last_draft_ms.remove(&request_id);
            dflash.last_draft_context_tokens.remove(&request_id);
            dflash.last_draft_committed_context.remove(&request_id);
        }
    }
}

fn dflash_nvtx_enabled_from_env() -> bool {
    matches!(
        env::var("OPENINFER_QWEN3_DFLASH_NVTX").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}
