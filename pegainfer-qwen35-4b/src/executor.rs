//! Minimal Qwen3.5 logits executor for model-local accuracy gates.
//!
//! The production server uses the scheduler. This executor exists so tests can
//! teacher-force fixed token sequences through prefill + decode and inspect
//! logits without widening the northbound engine API.

use std::collections::HashSet;

use anyhow::Result;
use pegainfer_core::engine::TokenLogprob;
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::tensor::DeviceVec;

use crate::batch_decode_graph::{BatchDecodeGraphState, MAX_BATCH};
use crate::recurrent_state::RecurrentState;
use crate::weights::Qwen35Model;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RequestId(u64);

impl RequestId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone)]
pub struct PrefillStepItem {
    request_id: RequestId,
    prompt_tokens: Vec<u32>,
    logprobs: usize,
}

impl PrefillStepItem {
    pub fn new(request_id: RequestId, prompt_tokens: Vec<u32>, logprobs: usize) -> Self {
        Self {
            request_id,
            prompt_tokens,
            logprobs,
        }
    }
}

#[derive(Clone)]
pub struct DecodeStepItem {
    request_id: RequestId,
    token_id: u32,
    logprobs: usize,
}

impl DecodeStepItem {
    pub fn new(request_id: RequestId, token_id: u32, logprobs: usize) -> Self {
        Self {
            request_id,
            token_id,
            logprobs,
        }
    }
}

pub struct PrefillPlan<'a> {
    pub requests: &'a [PrefillStepItem],
}

pub struct DecodePlan<'a> {
    pub requests: &'a [DecodeStepItem],
}

#[derive(Clone, Debug)]
pub struct PrefillRequestResult {
    pub request_id: RequestId,
    pub first_token: u32,
    pub first_token_logprob: Option<TokenLogprob>,
}

#[derive(Clone, Debug)]
pub struct DecodeRequestResult {
    pub request_id: RequestId,
    pub token: u32,
    pub logprob: Option<TokenLogprob>,
}

pub struct PrefillResult {
    pub requests: Vec<PrefillRequestResult>,
}

pub struct DecodeResult {
    pub requests: Vec<DecodeRequestResult>,
}

struct ActiveRequest {
    request_id: RequestId,
    kv: KvState,
    graph_slot_idx: usize,
}

pub struct Qwen35Executor {
    model: Qwen35Model,
    graph_state: BatchDecodeGraphState,
    active: Vec<ActiveRequest>,
}

impl Qwen35Executor {
    pub fn from_runtime(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        Self::from_runtime_with_capacity(model_path, enable_cuda_graph, device_ordinals, MAX_BATCH)
    }

    pub fn from_runtime_with_capacity(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
        max_batch: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            device_ordinals.len() == 1,
            "Qwen3.5 logits executor supports exactly one CUDA device"
        );
        let model = Qwen35Model::from_safetensors_with_device_options(
            model_path,
            enable_cuda_graph,
            device_ordinals[0],
        )?;
        let graph_state = model.create_batch_decode_graph_state_with_capacity(max_batch)?;
        Ok(Self {
            model,
            graph_state,
            active: Vec::new(),
        })
    }

    pub fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 prefill plan requires at least one request"
        );
        anyhow::ensure!(
            self.active.len() + plan.requests.len() <= self.graph_state.slot_states.len(),
            "Qwen3.5 prefill would exceed logits executor capacity"
        );
        let mut seen = HashSet::with_capacity(plan.requests.len());
        for req in plan.requests {
            anyhow::ensure!(
                !req.prompt_tokens.is_empty(),
                "Qwen3.5 logits executor prefill request {} has an empty prompt",
                req.request_id.get()
            );
            anyhow::ensure!(
                seen.insert(req.request_id),
                "duplicate Qwen3.5 request id {} in prefill plan",
                req.request_id.get()
            );
            anyhow::ensure!(
                !self
                    .active
                    .iter()
                    .any(|active| active.request_id == req.request_id),
                "duplicate Qwen3.5 request id {}",
                req.request_id.get()
            );
        }

        let prompts: Vec<&[u32]> = plan
            .requests
            .iter()
            .map(|req| req.prompt_tokens.as_slice())
            .collect();
        let mut kv_states: Vec<KvState> = plan
            .requests
            .iter()
            .map(|_| self.model.alloc_kv())
            .collect();
        let mut recurrent_states: Vec<RecurrentState> = plan
            .requests
            .iter()
            .map(|_| RecurrentState::new(self.model.device_ctx(), self.model.config()))
            .collect::<Result<_>>()?;
        let mut recurrent_refs: Vec<&mut RecurrentState> = recurrent_states.iter_mut().collect();
        let logits = self
            .model
            .batch_prefill(&prompts, &mut kv_states, &mut recurrent_refs)?;

        let mut results = Vec::with_capacity(plan.requests.len());
        for (i, (req, kv)) in plan.requests.iter().zip(kv_states.into_iter()).enumerate() {
            let (first_token, first_token_logprob) =
                self.token_and_logprob(&logits[i], req.logprobs)?;
            let slot_idx = self.active.len();
            self.graph_state.copy_state_to_slot(
                self.model.device_ctx(),
                &recurrent_states[i],
                slot_idx,
            )?;
            self.active.push(ActiveRequest {
                request_id: req.request_id,
                kv,
                graph_slot_idx: slot_idx,
            });
            results.push(PrefillRequestResult {
                request_id: req.request_id,
                first_token,
                first_token_logprob,
            });
        }

        Ok(PrefillResult { requests: results })
    }

    pub fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 decode plan requires at least one request"
        );
        anyhow::ensure!(
            plan.requests.len() == self.active.len(),
            "Qwen3.5 logits executor decode must include all active requests in slot order"
        );
        for (i, req) in plan.requests.iter().enumerate() {
            anyhow::ensure!(
                self.active[i].request_id == req.request_id,
                "Qwen3.5 decode request order differs from active slot order"
            );
        }

        let token_ids: Vec<u32> = plan.requests.iter().map(|req| req.token_id).collect();
        let mut kv_refs: Vec<&mut KvState> =
            self.active.iter_mut().map(|req| &mut req.kv).collect();
        self.model
            .batch_decode_graph(&token_ids, &mut kv_refs, &mut self.graph_state)?;

        let mut results = Vec::with_capacity(plan.requests.len());
        for (i, req) in plan.requests.iter().enumerate() {
            let logits = crate::ops::extract_vec(
                self.model.device_ctx(),
                &self.graph_state.buffers.logits,
                i,
            )?;
            let (token, logprob) = self.token_and_logprob(&logits, req.logprobs)?;
            results.push(DecodeRequestResult {
                request_id: req.request_id,
                token,
                logprob,
            });
        }
        Ok(DecodeResult { requests: results })
    }

    pub fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        let Some(idx) = self
            .active
            .iter()
            .position(|active| active.request_id == request_id)
        else {
            return Ok(());
        };
        self.compact_slot(idx)
    }

    fn compact_slot(&mut self, idx: usize) -> Result<()> {
        let last = self.active.len() - 1;
        self.active.swap_remove(idx);

        if idx < self.active.len() {
            anyhow::ensure!(
                self.active[idx].graph_slot_idx == last,
                "Qwen3.5 logits executor slot invariant broken: active slot {} moved from graph slot {}, expected {}",
                idx,
                self.active[idx].graph_slot_idx,
                last
            );
            for layer_idx in 0..self.graph_state.slot_states[last].layers.len() {
                let (src_part, dst_part) = if idx < last {
                    let (left, right) = self.graph_state.slot_states.split_at_mut(last);
                    (
                        &right[0].layers[layer_idx],
                        &mut left[idx].layers[layer_idx],
                    )
                } else {
                    unreachable!("idx < active.len() <= last");
                };
                self.model
                    .device_ctx()
                    .stream
                    .memcpy_dtod(&src_part.state, &mut dst_part.state)
                    .map_err(|e| {
                        anyhow::anyhow!("compact Qwen3.5 logits executor state copy failed: {e}")
                    })?;
                self.model
                    .device_ctx()
                    .stream
                    .memcpy_dtod(&src_part.conv_state.data, &mut dst_part.conv_state.data)
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "compact Qwen3.5 logits executor conv_state copy failed: {e}"
                        )
                    })?;
            }
            self.graph_state.slot_states[idx].seq_len = self.graph_state.slot_states[last].seq_len;
            self.active[idx].graph_slot_idx = idx;
        }
        Ok(())
    }

    fn token_and_logprob(
        &self,
        logits: &DeviceVec,
        requested_top_k: usize,
    ) -> Result<(u32, Option<TokenLogprob>)> {
        let logits_f32 = logits.to_host(self.model.device_ctx())?;
        let Some((token, logprob, top_logprobs)) =
            compute_logprobs_from_cpu(&logits_f32, requested_top_k.max(1))
        else {
            anyhow::bail!("Qwen3.5 logits were empty");
        };
        let logprob = (requested_top_k > 0).then_some(TokenLogprob {
            logprob,
            top_logprobs,
        });
        Ok((token, logprob))
    }
}

fn compute_logprobs_from_cpu(
    logits_f32: &[f32],
    top_k: usize,
) -> Option<(u32, f32, Vec<(u32, f32)>)> {
    if logits_f32.is_empty() {
        return None;
    }

    let max_val = logits_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits_f32.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();

    let k = top_k.min(logits_f32.len());
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (idx, &val) in logits_f32.iter().enumerate() {
        if best.len() < k || val > best.last().unwrap().1 {
            let pos = best.partition_point(|&(_, v)| v > val);
            best.insert(pos, (idx as u32, val));
            if best.len() > k {
                best.pop();
            }
        }
    }

    let token = best[0].0;
    let logprob = best[0].1 - log_sum_exp;
    let top_logprobs = best
        .into_iter()
        .map(|(idx, val)| (idx, val - log_sum_exp))
        .collect();
    Some((token, logprob, top_logprobs))
}
