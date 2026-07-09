use anyhow::Result;
use openinfer_core::engine::TokenLogprob;
use openinfer_core::sampler::SamplingParams;

use crate::executor::{Qwen35Executor, RequestId};
use crate::logprobs::snapshot_requested_logprobs;
use crate::prefill::PREFILL_CHUNK_LEN;
use crate::recurrent_state::RecurrentState;
use crate::verify_buffers::VerifyBuffers35;

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

impl Qwen35Executor {
    pub fn execute_speculative_verify(&mut self, plan: VerifyPlan<'_>) -> Result<VerifyResult> {
        self.validate_speculative_verify(plan)?;
        let original_seq_lens: Vec<usize> = self
            .active
            .iter()
            .map(|active| active.kv.seq_len())
            .collect();
        match self.execute_speculative_verify_inner(plan) {
            Ok(result) => Ok(result),
            Err(err) => {
                if let Err(rollback_err) = self.rollback_kv_states(&original_seq_lens) {
                    anyhow::bail!(
                        "{err}; additionally failed to roll back Qwen3.5 KV state: {rollback_err}"
                    );
                }
                Err(err)
            }
        }
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

    fn execute_speculative_verify_inner(&mut self, plan: VerifyPlan<'_>) -> Result<VerifyResult> {
        let backup_states = self.copy_canonical_recurrent_states()?;
        let original_seq_lens: Vec<usize> = self
            .active
            .iter()
            .map(|active| active.kv.seq_len())
            .collect();

        let mut verify_states = Vec::with_capacity(self.active.len());
        for backup in &backup_states {
            let mut state = RecurrentState::new(self.model.device_ctx(), self.model.config())?;
            state.copy_from(self.model.device_ctx(), backup)?;
            verify_states.push(state);
        }

        let max_span = plan
            .requests
            .iter()
            .map(|req| req.token_ids.len())
            .max()
            .unwrap_or(1);
        let mut verify_bufs = VerifyBuffers35::new(
            self.model.device_ctx(),
            self.model.config(),
            plan.requests.len(),
            max_span,
            0,
            self.model.kv_pool().capacity_pages(),
        )?;
        let spans: Vec<&[u32]> = plan
            .requests
            .iter()
            .map(|req| req.token_ids.as_slice())
            .collect();
        {
            let mut kv_refs: Vec<_> = self
                .active
                .iter_mut()
                .map(|active| &mut active.kv)
                .collect();
            let mut rec_refs: Vec<_> = verify_states.iter_mut().collect();
            self.model.prefill_verify_into(
                &spans,
                &mut kv_refs,
                &mut rec_refs,
                &[],
                &mut verify_bufs,
            )?;
        }

        let requested_logprobs: Vec<usize> = plan
            .requests
            .iter()
            .flat_map(|req| std::iter::repeat_n(req.logprobs, req.token_ids.len()))
            .collect();
        let cpu_logits = snapshot_requested_logprobs(
            self.model.device_ctx(),
            &verify_bufs.logits,
            &requested_logprobs,
        )?;
        let params = vec![SamplingParams::default(); verify_bufs.logits.seq_len];
        let params_refs: Vec<&SamplingParams> = params.iter().collect();
        let steps = vec![0_u64; verify_bufs.logits.seq_len];
        let target_tokens = openinfer_sample::select_batch(
            self.model.device_ctx(),
            &verify_bufs.logits,
            &params_refs,
            &steps,
            0,
            &mut verify_bufs.sample,
        )?;

        let mut request_outputs = Vec::with_capacity(plan.requests.len());
        let mut row_offset = 0usize;
        for req in plan.requests {
            let row_end = row_offset + req.token_ids.len();
            let target_slice = &target_tokens[row_offset..row_end];
            let target_logprobs = target_slice
                .iter()
                .enumerate()
                .map(|(i, &token)| {
                    cpu_logits[row_offset + i].as_ref().and_then(|row| {
                        openinfer_sample::token_logprob_from_row(row, token, req.logprobs)
                    })
                })
                .collect::<Vec<_>>();
            let (matched, accepted_ids) = accept_greedy(&req.token_ids[1..], target_slice);
            let accepted_tokens: Vec<VerifiedToken> = accepted_ids
                .iter()
                .enumerate()
                .map(|(i, &token)| VerifiedToken {
                    token,
                    logprob: target_logprobs.get(i).cloned().unwrap_or(None),
                })
                .collect();
            request_outputs.push(VerifyRequestResult {
                request_id: req.request_id,
                matched_draft_tokens: matched,
                accepted_tokens,
            });
            row_offset = row_end;
        }

        if let Err(err) = self.commit_speculative_states(
            plan.requests,
            &request_outputs,
            &backup_states,
            &original_seq_lens,
        ) {
            if let Err(restore_err) =
                self.restore_canonical_states(&backup_states, &original_seq_lens)
            {
                anyhow::bail!(
                    "{err}; additionally failed to restore Qwen3.5 recurrent/conv state after speculative commit failure: {restore_err}"
                );
            }
            return Err(err);
        }

        Ok(VerifyResult {
            requests: request_outputs,
        })
    }

    fn copy_canonical_recurrent_states(&self) -> Result<Vec<RecurrentState>> {
        let mut scratch_states = Vec::with_capacity(self.active.len());
        for active in &self.active {
            let mut state = RecurrentState::new(self.model.device_ctx(), self.model.config())?;
            self.graph_state.copy_slot_to_state(
                self.model.device_ctx(),
                active.graph_slot_idx,
                &mut state,
            )?;
            scratch_states.push(state);
        }
        Ok(scratch_states)
    }

    fn commit_speculative_states(
        &mut self,
        requests: &[VerifyStepItem],
        results: &[VerifyRequestResult],
        backup_states: &[RecurrentState],
        original_seq_lens: &[usize],
    ) -> Result<()> {
        anyhow::ensure!(
            requests.len() == self.active.len()
                && results.len() == self.active.len()
                && backup_states.len() == self.active.len()
                && original_seq_lens.len() == self.active.len(),
            "Qwen3.5 speculative commit batch size mismatch"
        );

        for (slot_idx, ((req, result), backup_state)) in requests
            .iter()
            .zip(results.iter())
            .zip(backup_states.iter())
            .enumerate()
        {
            let accepted_len = result.accepted_tokens.len();
            anyhow::ensure!(
                accepted_len <= req.token_ids.len(),
                "Qwen3.5 speculative accepted span {} exceeds verify span {}",
                accepted_len,
                req.token_ids.len()
            );
            if accepted_len == 0 {
                continue;
            }
            self.active[slot_idx]
                .kv
                .truncate_to(original_seq_lens[slot_idx])?;
            let mut replay_state =
                RecurrentState::new(self.model.device_ctx(), self.model.config())?;
            replay_state.copy_from(self.model.device_ctx(), backup_state)?;
            let mut replay_tokens = Vec::with_capacity(accepted_len);
            replay_tokens.push(req.token_ids[0]);
            replay_tokens.extend(
                result
                    .accepted_tokens
                    .iter()
                    .take(accepted_len.saturating_sub(1))
                    .map(|token| token.token),
            );
            let _ = self.model.prefill_logits_all(
                replay_tokens.as_slice(),
                &mut self.active[slot_idx].kv,
                &mut replay_state,
            )?;
            let graph_slot_idx = self.active[slot_idx].graph_slot_idx;
            self.graph_state.copy_state_to_slot(
                self.model.device_ctx(),
                &replay_state,
                graph_slot_idx,
            )?;
        }
        Ok(())
    }

    fn restore_canonical_states(
        &mut self,
        backup_states: &[RecurrentState],
        original_seq_lens: &[usize],
    ) -> Result<()> {
        for ((active, backup_state), &seq_len) in self
            .active
            .iter_mut()
            .zip(backup_states.iter())
            .zip(original_seq_lens.iter())
        {
            active.kv.truncate_to(seq_len)?;
            self.graph_state.copy_state_to_slot(
                self.model.device_ctx(),
                backup_state,
                active.graph_slot_idx,
            )?;
        }
        Ok(())
    }

    fn rollback_kv_states(&mut self, original_seq_lens: &[usize]) -> Result<()> {
        for (active, &seq_len) in self.active.iter_mut().zip(original_seq_lens.iter()) {
            active.kv.truncate_to(seq_len)?;
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
