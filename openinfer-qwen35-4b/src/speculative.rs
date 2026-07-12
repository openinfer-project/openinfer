use anyhow::Result;
use openinfer_core::engine::TokenLogprob;
use openinfer_core::kv_pool::KvState;
use openinfer_core::sampler::SamplingParams;

use crate::batch_decode_graph::BatchDecodeGraphState;
use crate::executor::{Qwen35Executor, RequestId};
use crate::logprobs::snapshot_requested_logprobs;
use crate::prefill::PREFILL_CHUNK_LEN;
use crate::recurrent_state::RecurrentState;
use crate::verify_buffers::VerifyBuffers35;
use crate::weights::Qwen35Model;

#[derive(Clone, Debug)]
pub struct VerifyStepItem {
    pub request_id: RequestId,
    pub token_ids: Vec<u32>,
    pub logprobs: usize,
}

impl VerifyStepItem {
    pub fn new(request_id: RequestId, token_ids: Vec<u32>, logprobs: usize) -> Self {
        Self {
            request_id,
            token_ids,
            logprobs,
        }
    }
}

#[derive(Clone, Copy)]
pub struct VerifyPlan<'a> {
    pub requests: &'a [VerifyStepItem],
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedToken {
    pub token: u32,
    pub logprob: Option<TokenLogprob>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifyRequestResult {
    pub request_id: RequestId,
    pub matched_draft_tokens: usize,
    pub accepted_tokens: Vec<VerifiedToken>,
}

pub struct VerifyResult {
    pub requests: Vec<VerifyRequestResult>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VerifySpanResult {
    pub(crate) matched_draft_tokens: usize,
    pub(crate) accepted_tokens: Vec<VerifiedToken>,
}

#[must_use]
pub(crate) fn accept_greedy(proposed: &[u32], target_argmax: &[u32]) -> (usize, Vec<u32>) {
    debug_assert_eq!(
        target_argmax.len(),
        proposed.len() + 1,
        "verify must produce one posterior token per draft plus one bonus"
    );
    let mut matched = 0usize;
    while matched < proposed.len() && proposed[matched] == target_argmax[matched] {
        matched += 1;
    }
    let mut accepted = Vec::with_capacity(matched + 1);
    accepted.extend_from_slice(&proposed[..matched]);
    accepted.push(target_argmax[matched]);
    (matched, accepted)
}

pub(crate) fn capture_hybrid_states(
    model: &Qwen35Model,
    graph_state: &BatchDecodeGraphState,
    graph_slot_indices: &[usize],
    backup_states: &mut [RecurrentState],
) -> Result<()> {
    anyhow::ensure!(
        graph_slot_indices.len() == backup_states.len(),
        "Qwen3.5 speculative backup batch size mismatch"
    );
    for (&graph_slot_idx, backup_state) in graph_slot_indices.iter().zip(backup_states.iter_mut()) {
        graph_state.copy_slot_to_state(model.device_ctx(), graph_slot_idx, backup_state)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_hybrid_verify(
    model: &Qwen35Model,
    kv_states: &mut [&mut KvState],
    spans: &[&[u32]],
    capture_layer_ids: &[usize],
    verify_bufs: &mut VerifyBuffers35,
    backup_states: &[RecurrentState],
    verify_states: &mut [RecurrentState],
) -> Result<()> {
    let batch = spans.len();
    anyhow::ensure!(batch > 0, "Qwen3.5 speculative verify batch is empty");
    anyhow::ensure!(
        kv_states.len() == batch && backup_states.len() == batch && verify_states.len() == batch,
        "Qwen3.5 speculative verify batch size mismatch"
    );
    for span in spans {
        anyhow::ensure!(
            !span.is_empty(),
            "Qwen3.5 speculative verify requires a non-empty token span"
        );
    }

    for (verify_state, backup_state) in verify_states.iter_mut().zip(backup_states.iter()) {
        verify_state.copy_from(model.device_ctx(), backup_state)?;
    }
    let mut verify_refs: Vec<&mut RecurrentState> = verify_states.iter_mut().collect();
    model.prefill_verify_into(
        spans,
        kv_states,
        &mut verify_refs,
        capture_layer_ids,
        verify_bufs,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_hybrid_spans(
    model: &Qwen35Model,
    kv_states: &mut [&mut KvState],
    spans: &[&[u32]],
    capture_layer_ids: &[usize],
    requested_logprobs: &[usize],
    verify_bufs: &mut VerifyBuffers35,
    backup_states: &[RecurrentState],
    verify_states: &mut [RecurrentState],
) -> Result<Vec<VerifySpanResult>> {
    let batch = spans.len();
    anyhow::ensure!(
        batch == requested_logprobs.len(),
        "Qwen3.5 speculative verify logprobs batch size mismatch"
    );
    for span in spans {
        anyhow::ensure!(
            span.len() >= 2,
            "Qwen3.5 speculative verify needs [current, draft...]"
        );
    }
    run_hybrid_verify(
        model,
        kv_states,
        spans,
        capture_layer_ids,
        verify_bufs,
        backup_states,
        verify_states,
    )?;

    let row_logprobs: Vec<usize> = spans
        .iter()
        .zip(requested_logprobs.iter())
        .flat_map(|(span, &logprobs)| std::iter::repeat_n(logprobs, span.len()))
        .collect();
    let cpu_logits =
        snapshot_requested_logprobs(model.device_ctx(), &verify_bufs.logits, &row_logprobs)?;
    let greedy = SamplingParams::default();
    let params = vec![&greedy; verify_bufs.logits.seq_len];
    let steps = vec![0_u64; verify_bufs.logits.seq_len];
    let target_tokens = openinfer_sample::select_batch(
        model.device_ctx(),
        &verify_bufs.logits,
        &params,
        &steps,
        0,
        &mut verify_bufs.sample,
    )?;

    let mut outputs = Vec::with_capacity(batch);
    let mut row_offset = 0usize;
    for span in spans {
        let row_end = row_offset + span.len();
        let target_slice = &target_tokens[row_offset..row_end];
        let (matched, accepted_ids) = accept_greedy(&span[1..], target_slice);
        let accepted_tokens = accepted_ids
            .into_iter()
            .enumerate()
            .map(|(i, token)| VerifiedToken {
                token,
                logprob: cpu_logits[row_offset + i].as_ref().and_then(|row| {
                    openinfer_sample::token_logprob_from_row(
                        row,
                        token,
                        requested_logprobs[outputs.len()],
                    )
                }),
            })
            .collect();
        outputs.push(VerifySpanResult {
            matched_draft_tokens: matched,
            accepted_tokens,
        });
        row_offset = row_end;
    }
    Ok(outputs)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_hybrid_states(
    model: &Qwen35Model,
    kv_states: &mut [&mut KvState],
    graph_state: &mut BatchDecodeGraphState,
    graph_slot_indices: &[usize],
    spans: &[&[u32]],
    backup_states: &[RecurrentState],
    verify_states: &mut [RecurrentState],
    verify_bufs: &mut VerifyBuffers35,
    original_seq_lens: &[usize],
    results: &[VerifySpanResult],
) -> Result<()> {
    let batch = spans.len();
    anyhow::ensure!(
        kv_states.len() == batch
            && graph_slot_indices.len() == batch
            && backup_states.len() == batch
            && verify_states.len() == batch
            && original_seq_lens.len() == batch
            && results.len() == batch,
        "Qwen3.5 speculative commit batch size mismatch"
    );

    for slot_idx in 0..batch {
        let accepted_len = results[slot_idx].accepted_tokens.len();
        anyhow::ensure!(
            accepted_len > 0 && accepted_len <= spans[slot_idx].len(),
            "Qwen3.5 speculative accepted span {} is invalid for verify span {}",
            accepted_len,
            spans[slot_idx].len()
        );
        if accepted_len == spans[slot_idx].len() {
            graph_state.copy_state_to_slot(
                model.device_ctx(),
                &verify_states[slot_idx],
                graph_slot_indices[slot_idx],
            )?;
            continue;
        }

        kv_states[slot_idx].truncate_to(original_seq_lens[slot_idx])?;
        verify_states[slot_idx].copy_from(model.device_ctx(), &backup_states[slot_idx])?;
        let mut replay_tokens = Vec::with_capacity(accepted_len);
        replay_tokens.push(spans[slot_idx][0]);
        replay_tokens.extend(
            results[slot_idx]
                .accepted_tokens
                .iter()
                .take(accepted_len - 1)
                .map(|token| token.token),
        );
        let replay_spans = [replay_tokens.as_slice()];
        let mut one_kv = [&mut *kv_states[slot_idx]];
        let mut one_state = [&mut verify_states[slot_idx]];
        openinfer_kernels::ops::with_gemm_lt_disabled(|| {
            model.prefill_verify_into(&replay_spans, &mut one_kv, &mut one_state, &[], verify_bufs)
        })?;
        graph_state.copy_state_to_slot(
            model.device_ctx(),
            &verify_states[slot_idx],
            graph_slot_indices[slot_idx],
        )?;
    }
    Ok(())
}

pub(crate) fn restore_hybrid_states(
    model: &Qwen35Model,
    kv_states: &mut [&mut KvState],
    graph_state: &mut BatchDecodeGraphState,
    graph_slot_indices: &[usize],
    backup_states: &[RecurrentState],
    original_seq_lens: &[usize],
) -> Result<()> {
    anyhow::ensure!(
        kv_states.len() == graph_slot_indices.len()
            && kv_states.len() == backup_states.len()
            && kv_states.len() == original_seq_lens.len(),
        "Qwen3.5 speculative restore batch size mismatch"
    );
    let mut errors = Vec::new();
    for slot_idx in 0..kv_states.len() {
        if let Err(err) = kv_states[slot_idx].truncate_to(original_seq_lens[slot_idx]) {
            errors.push(format!(
                "truncate slot {slot_idx} to {} failed: {err}",
                original_seq_lens[slot_idx]
            ));
        }
        if let Err(err) = graph_state.copy_state_to_slot(
            model.device_ctx(),
            &backup_states[slot_idx],
            graph_slot_indices[slot_idx],
        ) {
            errors.push(format!("restore recurrent slot {slot_idx} failed: {err}"));
        }
    }
    anyhow::ensure!(
        errors.is_empty(),
        "Qwen3.5 speculative rollback failed: {}",
        errors.join("; ")
    );
    Ok(())
}

impl Qwen35Executor {
    pub fn execute_speculative_verify(&mut self, plan: VerifyPlan<'_>) -> Result<VerifyResult> {
        self.validate_speculative_verify(plan)?;
        let batch = self.active.len();
        let graph_slot_indices: Vec<usize> = self
            .active
            .iter()
            .map(|active| active.graph_slot_idx)
            .collect();
        let original_seq_lens: Vec<usize> = self
            .active
            .iter()
            .map(|active| active.kv.seq_len())
            .collect();
        let mut backup_states = Vec::with_capacity(batch);
        let mut verify_states = Vec::with_capacity(batch);
        for _ in 0..batch {
            backup_states.push(RecurrentState::new(
                self.model.device_ctx(),
                self.model.config(),
            )?);
            verify_states.push(RecurrentState::new(
                self.model.device_ctx(),
                self.model.config(),
            )?);
        }
        capture_hybrid_states(
            &self.model,
            &self.graph_state,
            &graph_slot_indices,
            &mut backup_states,
        )?;

        let max_span = plan
            .requests
            .iter()
            .map(|req| req.token_ids.len())
            .max()
            .unwrap_or(1);
        let mut verify_bufs = VerifyBuffers35::new(
            self.model.device_ctx(),
            self.model.config(),
            batch,
            max_span,
            0,
            self.model.kv_pool().capacity_pages(),
        )?;
        let spans: Vec<&[u32]> = plan
            .requests
            .iter()
            .map(|req| req.token_ids.as_slice())
            .collect();
        let requested_logprobs: Vec<usize> = plan.requests.iter().map(|req| req.logprobs).collect();

        let transaction = (|| -> Result<Vec<VerifySpanResult>> {
            let mut kv_states: Vec<&mut KvState> = self
                .active
                .iter_mut()
                .map(|active| &mut active.kv)
                .collect();
            let results = verify_hybrid_spans(
                &self.model,
                &mut kv_states,
                &spans,
                &[],
                &requested_logprobs,
                &mut verify_bufs,
                &backup_states,
                &mut verify_states,
            )?;
            commit_hybrid_states(
                &self.model,
                &mut kv_states,
                &mut self.graph_state,
                &graph_slot_indices,
                &spans,
                &backup_states,
                &mut verify_states,
                &mut verify_bufs,
                &original_seq_lens,
                &results,
            )?;
            Ok(results)
        })();

        let results = match transaction {
            Ok(results) => results,
            Err(err) => {
                let mut kv_states: Vec<&mut KvState> = self
                    .active
                    .iter_mut()
                    .map(|active| &mut active.kv)
                    .collect();
                if let Err(rollback_err) = restore_hybrid_states(
                    &self.model,
                    &mut kv_states,
                    &mut self.graph_state,
                    &graph_slot_indices,
                    &backup_states,
                    &original_seq_lens,
                ) {
                    anyhow::bail!("{err}; additionally failed to roll back: {rollback_err}");
                }
                return Err(err);
            }
        };

        Ok(VerifyResult {
            requests: plan
                .requests
                .iter()
                .zip(results)
                .map(|(request, result)| VerifyRequestResult {
                    request_id: request.request_id,
                    matched_draft_tokens: result.matched_draft_tokens,
                    accepted_tokens: result.accepted_tokens,
                })
                .collect(),
        })
    }

    fn validate_speculative_verify(&self, plan: VerifyPlan<'_>) -> Result<()> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 speculative verify plan requires at least one request"
        );
        anyhow::ensure!(
            plan.requests.len() == self.active.len(),
            "Qwen3.5 speculative verify must include all active requests in slot order"
        );
        for (slot_idx, req) in plan.requests.iter().enumerate() {
            anyhow::ensure!(
                self.active[slot_idx].request_id == req.request_id,
                "Qwen3.5 speculative verify request order differs from active slot order"
            );
            anyhow::ensure!(
                req.token_ids.len() >= 2,
                "Qwen3.5 speculative verify request {} needs [current, draft...]",
                req.request_id.get()
            );
            anyhow::ensure!(
                req.token_ids.len() <= PREFILL_CHUNK_LEN,
                "Qwen3.5 speculative verify request {} span len {} exceeds max chunk {PREFILL_CHUNK_LEN}",
                req.request_id.get(),
                req.token_ids.len()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_full_run_plus_bonus() {
        let (matched, accepted) = accept_greedy(&[10, 11, 12], &[10, 11, 12, 13]);
        assert_eq!(matched, 3);
        assert_eq!(accepted, vec![10, 11, 12, 13]);
    }

    #[test]
    fn accepts_prefix_then_correction() {
        let (matched, accepted) = accept_greedy(&[10, 11, 99], &[10, 11, 22, 33]);
        assert_eq!(matched, 2);
        assert_eq!(accepted, vec![10, 11, 22]);
    }

    #[test]
    fn rejects_first_candidate_commits_one() {
        let (matched, accepted) = accept_greedy(&[10, 11, 12], &[7, 8, 9, 10]);
        assert_eq!(matched, 0);
        assert_eq!(accepted, vec![7]);
    }
}
