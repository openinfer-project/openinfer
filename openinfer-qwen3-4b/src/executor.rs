use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};

use crate::batch_decode_buffers::BATCH_BUCKETS;
use crate::config::{Config, DFlashConfig};
use crate::speculative::{
    DraftPlan as SpeculativeDraftPlan, DraftResult as SpeculativeDraftResult,
    VerifyPlan as SpeculativeVerifyPlan, VerifyResult as SpeculativeVerifyResult,
    VerifyStepItem as SpeculativeVerifyStepItem, build_verify_results,
};
use crate::weights::Qwen3Model;
use crate::{Qwen3LoraOptions, Qwen3SpeculativeOptions};
use openinfer_core::engine::{LoadLoraAdapterRequest, TokenLogprob, UnloadLoraAdapterRequest};
use openinfer_core::ops;
use openinfer_core::sampler::SamplingParams;
use openinfer_core::tensor::{DeviceContext, HiddenStates};
use openinfer_kv_cache::{KvCacheManager, LoadReservation, PrefixProbe};
use openinfer_kv_offload::{LoadHandle, OffloadEngine};

mod dflash_lane;
mod dflash_prefill;
mod lifecycle;
mod model_executor;
mod speculative_exec;
mod worker;

use self::worker::{LocalQwen3Lane, RankWorker, StepCommand, WorkerStepOutcome};

const BF16_BYTES: usize = 2;
const DFLASH_MIN_EXTRA_MEMORY_RESERVE_BYTES: usize = 1024 * 1024 * 1024;
const DFLASH_ALLOCATOR_MARGIN_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct RequestId(pub(crate) u64);

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
    pub(crate) request_id: RequestId,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) max_output_tokens: usize,
    pub(crate) params: SamplingParams,
    pub(crate) logprobs: usize,
    pub(crate) echo: bool,
    pub(crate) lora_adapter: Option<String>,
    pub(crate) random_val: f32,
    /// Leading prompt tokens whose KV came from the prefix cache.
    /// Set by the executor after matching; the forward pass only computes
    /// the remaining suffix.
    pub(crate) cached_tokens: usize,
    /// Scheduler-set cap on prompt tokens forwarded this step (chunked
    /// prefill). The executor clamps it to the tokens actually remaining.
    pub(crate) chunk_budget: usize,
    /// First prompt position forwarded this step. Set by the executor from
    /// the request's KV position (covers both prefix-cache hits and chunks
    /// applied in earlier steps).
    pub(crate) chunk_start: usize,
    /// Prompt tokens forwarded this step. Set by the executor.
    pub(crate) chunk_tokens: usize,
}

impl PrefillStepItem {
    pub fn new(
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
        max_output_tokens: usize,
        params: SamplingParams,
        logprobs: usize,
        echo: bool,
        random_val: f32,
    ) -> Self {
        let chunk_tokens = prompt_tokens.len();
        Self {
            request_id,
            prompt_tokens,
            max_output_tokens,
            params,
            logprobs,
            echo,
            lora_adapter: None,
            random_val,
            cached_tokens: 0,
            chunk_budget: usize::MAX,
            chunk_start: 0,
            chunk_tokens,
        }
    }

    pub fn with_chunk_budget(mut self, chunk_budget: usize) -> Self {
        self.chunk_budget = chunk_budget;
        self
    }

    /// Prompt tokens forwarded this step.
    fn as_slice(&self) -> &[u32] {
        &self.prompt_tokens[self.chunk_start..self.chunk_start + self.chunk_tokens]
    }

    /// Whether this step's chunk reaches the end of the prompt (and so
    /// produces the first generated token).
    fn is_final_chunk(&self) -> bool {
        self.chunk_start + self.chunk_tokens == self.prompt_tokens.len()
    }
}

#[derive(Clone)]
pub struct DecodeStepItem {
    pub(crate) request_id: RequestId,
    pub(crate) token_id: u32,
    pub(crate) params: SamplingParams,
    pub(crate) logprobs: usize,
    pub(crate) lora_adapter: Option<String>,
    pub(crate) random_val: f32,
}

impl DecodeStepItem {
    pub fn new(
        request_id: RequestId,
        token_id: u32,
        params: SamplingParams,
        logprobs: usize,
        random_val: f32,
    ) -> Self {
        Self {
            request_id,
            token_id,
            params,
            logprobs,
            lora_adapter: None,
            random_val,
        }
    }
}

fn build_prefill_request_results(
    lane: &mut LocalQwen3Lane,
    requests: &[PrefillStepItem],
    logits: &HiddenStates,
    tokens: &[u32],
    all_position_logits: Option<&HiddenStates>,
    compute_prompt_logprobs: bool,
) -> Result<Vec<PrefillRequestResult>> {
    let mut token_offset = 0usize;
    let mut outputs = Vec::with_capacity(requests.len());
    for (i, req) in requests.iter().enumerate() {
        let completed = req.is_final_chunk();
        let first_token = tokens[i];
        let first_token_logprob = if completed && req.logprobs > 0 {
            let logits_i = ops::extract_vec(lane.model.device_ctx(), logits, i)?;
            Some(lane.extract_logprobs(&logits_i, first_token, req.logprobs)?)
        } else {
            None
        };
        let prompt_logprobs = if req.echo {
            if compute_prompt_logprobs {
                let mut echo_logprobs = Vec::with_capacity(req.prompt_tokens.len());
                echo_logprobs.push(None);
                if let Some(all_logits) = all_position_logits {
                    for j in 1..req.prompt_tokens.len() {
                        let prev_pos = token_offset + j - 1;
                        let target_token = req.prompt_tokens[j];
                        echo_logprobs.push(lane.extract_prompt_logprobs(
                            all_logits,
                            prev_pos,
                            target_token,
                            req.logprobs,
                        ));
                    }
                } else {
                    for _ in 1..req.prompt_tokens.len() {
                        echo_logprobs.push(None);
                    }
                }
                Some(echo_logprobs)
            } else {
                Some(vec![None; req.prompt_tokens.len()])
            }
        } else {
            None
        };
        token_offset += req.chunk_tokens;
        outputs.push(PrefillRequestResult {
            request_id: req.request_id,
            first_token,
            first_token_logprob,
            prompt_logprobs,
            cached_tokens: req.cached_tokens,
            completed,
            prefill_pos: req.chunk_start + req.chunk_tokens,
        });
    }
    Ok(outputs)
}

fn build_decode_request_results(
    lane: &mut LocalQwen3Lane,
    requests: &[DecodeStepItem],
    logits: &HiddenStates,
    row_offset: usize,
    tokens: &[u32],
) -> Result<Vec<DecodeRequestResult>> {
    let mut outputs = Vec::with_capacity(requests.len());
    for (i, req) in requests.iter().enumerate() {
        let token = tokens[row_offset + i];
        let logprob = if req.logprobs > 0 {
            let logits_i = ops::extract_vec(lane.model.device_ctx(), logits, row_offset + i)?;
            Some(lane.extract_logprobs(&logits_i, token, req.logprobs)?)
        } else {
            None
        };
        outputs.push(DecodeRequestResult {
            request_id: req.request_id,
            token,
            logprob,
        });
    }
    Ok(outputs)
}

fn build_batch_decode_request_results(
    lane: &mut LocalQwen3Lane,
    requests: &[DecodeStepItem],
) -> Result<Vec<DecodeRequestResult>> {
    let params: Vec<&SamplingParams> = requests.iter().map(|req| &req.params).collect();
    let random_vals: Vec<f32> = requests.iter().map(|req| req.random_val).collect();
    let tokens = openinfer_core::ops::select_batch_tokens_into(
        lane.model.device_ctx(),
        &lane.bufs.logits,
        &params,
        &random_vals,
        &mut lane.sample_scratch.row_indices,
        &mut lane.sample_scratch.argmax_partial_values,
        &mut lane.sample_scratch.argmax_partial_indices,
        &mut lane.sample_scratch.probs,
        &mut lane.sample_scratch.top1_values,
        &mut lane.sample_scratch.row_states,
        &mut lane.sample_scratch.valid,
        &mut lane.sample_scratch.out,
    )?;

    let mut outputs = Vec::with_capacity(requests.len());
    for (i, req) in requests.iter().enumerate() {
        let token = tokens[i];
        let logprob = if req.logprobs > 0 {
            let logits_i = ops::extract_vec(lane.model.device_ctx(), &lane.bufs.logits, i)?;
            Some(lane.extract_logprobs(&logits_i, token, req.logprobs)?)
        } else {
            None
        };
        outputs.push(DecodeRequestResult {
            request_id: req.request_id,
            token,
            logprob,
        });
    }
    Ok(outputs)
}

pub(super) struct DFlashMemoryReserve {
    total_bytes: usize,
    request_state_budget_bytes: usize,
}

fn dflash_memory_reserve(options: &Qwen3SpeculativeOptions) -> Result<DFlashMemoryReserve> {
    let Some(dflash) = options.dflash.as_ref() else {
        return Ok(DFlashMemoryReserve {
            total_bytes: 0,
            request_state_budget_bytes: 0,
        });
    };
    let path = dflash
        .model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("DFlash model path must be valid UTF-8"))?;
    let config = DFlashConfig::from_file(path)?;
    let (shard_paths, _) = openinfer_core::weight_loader::load_shard_info(path)?;
    let weight_bytes = shard_paths.iter().try_fold(0usize, |acc, path| {
        let len = std::fs::metadata(path)
            .with_context(|| format!("stat DFlash shard {path}"))?
            .len() as usize;
        Ok::<usize, anyhow::Error>(acc + len)
    })?;
    let extra_bytes = dflash_request_state_reserve_bytes(&config)?;
    let state_budget_bytes = extra_bytes.saturating_sub(DFLASH_ALLOCATOR_MARGIN_BYTES);
    let reserved = weight_bytes
        .checked_add(extra_bytes)
        .context("DFlash memory reserve overflow")?;
    log::info!(
        "Qwen3 DFlash memory reserve: weights={:.1} MB, extra={:.1} MB, state_budget={:.1} MB, total={:.1} MB",
        weight_bytes as f64 / 1e6,
        extra_bytes as f64 / 1e6,
        state_budget_bytes as f64 / 1e6,
        reserved as f64 / 1e6,
    );
    Ok(DFlashMemoryReserve {
        total_bytes: reserved,
        request_state_budget_bytes: state_budget_bytes,
    })
}

fn checked_product(values: &[usize], context: &'static str) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value).context(context)
    })
}

fn checked_sum(values: &[usize], context: &'static str) -> Result<usize> {
    values.iter().try_fold(0usize, |acc, &value| {
        acc.checked_add(value).context(context)
    })
}

fn bf16_tensor_bytes(rows: usize, cols: usize, context: &'static str) -> Result<usize> {
    checked_product(&[rows, cols, BF16_BYTES], context)
}

fn dflash_request_state_reserve_bytes(config: &DFlashConfig) -> Result<usize> {
    let footprint =
        dflash_request_state_footprint_bytes_for_len(config, config.max_position_embeddings)?;
    let with_margin = footprint
        .checked_add(DFLASH_ALLOCATOR_MARGIN_BYTES)
        .context("DFlash request-state reserve overflow")?;
    Ok(with_margin.max(DFLASH_MIN_EXTRA_MEMORY_RESERVE_BYTES))
}

pub(super) fn dflash_request_state_footprint_bytes_for_len(
    config: &DFlashConfig,
    max_cache_len: usize,
) -> Result<usize> {
    let max_tokens = max_cache_len;
    let block = config.block_size;
    let hidden = config.hidden_size;
    let q_dim = checked_product(
        &[config.num_attention_heads, config.head_dim],
        "DFlash q dimension overflow",
    )?;
    let kv_dim = checked_product(
        &[config.num_key_value_heads, config.head_dim],
        "DFlash KV dimension overflow",
    )?;
    let inter = config.intermediate_size;
    let context_feature_dim = hidden
        .checked_mul(config.dflash_config.target_layer_ids.len())
        .context("DFlash context feature dimension overflow")?;

    let kv_cache = checked_product(
        &[config.num_hidden_layers, 2, max_tokens, kv_dim, BF16_BYTES],
        "DFlash KV cache reserve overflow",
    )?;
    let pending_context = bf16_tensor_bytes(
        max_tokens,
        context_feature_dim,
        "DFlash pending-context reserve overflow",
    )?;
    let context_projected = bf16_tensor_bytes(
        max_tokens,
        hidden,
        "DFlash projected-context reserve overflow",
    )?;
    let context_hidden =
        bf16_tensor_bytes(max_tokens, hidden, "DFlash context-hidden reserve overflow")?;
    let tail_input = bf16_tensor_bytes(max_tokens, hidden, "DFlash tail-input reserve overflow")?;
    let k_tail = bf16_tensor_bytes(max_tokens, kv_dim, "DFlash K-tail reserve overflow")?;
    let v_tail = bf16_tensor_bytes(max_tokens, kv_dim, "DFlash V-tail reserve overflow")?;

    let block_hidden_count = checked_product(
        &[5, block, hidden, BF16_BYTES],
        "DFlash block hidden scratch reserve overflow",
    )?;
    let block_q_count = checked_product(
        &[2, block, q_dim, BF16_BYTES],
        "DFlash block attention scratch reserve overflow",
    )?;
    let block_mlp_count = checked_product(
        &[3, block, inter, BF16_BYTES],
        "DFlash block MLP scratch reserve overflow",
    )?;
    let block_logits = bf16_tensor_bytes(
        block,
        config.vocab_size,
        "DFlash block logits reserve overflow",
    )?;
    let block_token_ids = checked_product(
        &[block, std::mem::size_of::<u32>()],
        "DFlash token-id scratch reserve overflow",
    )?;

    checked_sum(
        &[
            kv_cache,
            pending_context,
            context_projected,
            context_hidden,
            tail_input,
            k_tail,
            v_tail,
            block_hidden_count,
            block_q_count,
            block_mlp_count,
            block_logits,
            block_token_ids,
        ],
        "DFlash request-state reserve overflow",
    )
}

fn execute_step_on_lane(
    lane: &mut LocalQwen3Lane,
    step: &StepCommand,
    collect_result: bool,
) -> Result<WorkerStepOutcome> {
    match step {
        StepCommand::Prefill {
            requests,
            kv_views,
            echo,
        } => {
            let prompts: Vec<&[u32]> = requests.iter().map(PrefillStepItem::as_slice).collect();
            let lora_adapters: Vec<Option<&str>> = requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            let capture_dflash_context = lane.should_capture_dflash_prefill_context(requests);
            let capture_layer_ids = if capture_dflash_context {
                lane.dflash_capture_layer_ids()
            } else {
                None
            };
            let (logits, all_position_logits, _captured_hidden) = lane.execute_prefill(
                &prompts,
                kv_views,
                &lora_adapters,
                *echo,
                capture_layer_ids.as_deref(),
            )?;
            let dflash_context_captured_requests = lane.record_prefill_dflash_context(
                requests,
                capture_dflash_context,
                _captured_hidden.as_ref(),
            )?;
            if collect_result {
                let params: Vec<&SamplingParams> = requests.iter().map(|r| &r.params).collect();
                let random_vals: Vec<f32> = requests.iter().map(|r| r.random_val).collect();
                let tokens = lane.select_step_tokens(&logits, &params, &random_vals)?;
                Ok(WorkerStepOutcome::Prefill(PrefillResult {
                    requests: build_prefill_request_results(
                        lane,
                        requests,
                        &logits,
                        &tokens,
                        all_position_logits.as_ref(),
                        *echo,
                    )?,
                    dflash_context_captured_requests,
                }))
            } else {
                Ok(WorkerStepOutcome::Ack)
            }
        }
        StepCommand::SpeculativeVerify { requests, kv_views } => {
            for req in requests {
                anyhow::ensure!(
                    req.params.is_greedy(),
                    "speculative verification currently supports greedy sampling only"
                );
            }
            let prompts: Vec<&[u32]> = requests
                .iter()
                .map(SpeculativeVerifyStepItem::as_slice)
                .collect();
            let lora_adapters: Vec<Option<&str>> = requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            let capture_layer_ids = lane.dflash_capture_layer_ids();
            let verify_start = lane.start_dflash_timing()?;
            let verify_nvtx_range = if lane.dflash_nvtx_enabled() {
                Some(nvtx::range!("qwen3.dflash.verify"))
            } else {
                None
            };
            let (_last_logits, all_position_logits, captured_hidden) = lane.execute_prefill(
                &prompts,
                kv_views,
                &lora_adapters,
                true,
                capture_layer_ids.as_deref(),
            )?;
            if collect_result {
                let all_position_logits = all_position_logits.ok_or_else(|| {
                    anyhow::anyhow!("speculative verification did not return all-position logits")
                })?;
                let params: Vec<&SamplingParams> = requests
                    .iter()
                    .flat_map(|req| std::iter::repeat_n(&req.params, req.token_ids.len()))
                    .collect();
                let random_vals = vec![0.0; params.len()];
                let target_tokens =
                    lane.select_step_tokens(&all_position_logits, &params, &random_vals)?;
                let verify_ms = lane.finish_dflash_timing(verify_start)?;
                drop(verify_nvtx_range);
                let results = build_verify_results(requests, &target_tokens)?;
                lane.record_verify_dflash_context(
                    requests,
                    &results,
                    captured_hidden.as_ref(),
                    verify_ms,
                )?;
                Ok(WorkerStepOutcome::SpeculativeVerify(
                    SpeculativeVerifyResult { requests: results },
                ))
            } else {
                Ok(WorkerStepOutcome::Ack)
            }
        }
        StepCommand::SpeculativeDraft { requests } => {
            if collect_result {
                Ok(WorkerStepOutcome::SpeculativeDraft(
                    lane.execute_dflash_draft(requests)?,
                ))
            } else {
                Ok(WorkerStepOutcome::Ack)
            }
        }
        StepCommand::Decode { requests, kv_views } => {
            let token_ids: Vec<u32> = requests.iter().map(|req| req.token_id).collect();
            let lora_adapters: Vec<Option<&str>> = requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            lane.execute_decode(&token_ids, kv_views, &lora_adapters)?;
            if collect_result {
                Ok(WorkerStepOutcome::Decode(DecodeResult {
                    requests: build_batch_decode_request_results(lane, requests)?,
                }))
            } else {
                Ok(WorkerStepOutcome::Ack)
            }
        }
        StepCommand::Unified {
            prefill_requests,
            prefill_kv_views,
            decode_requests,
            decode_kv_views,
        } => {
            let prefill_prompts: Vec<&[u32]> = prefill_requests
                .iter()
                .map(PrefillStepItem::as_slice)
                .collect();
            let decode_tokens: Vec<u32> = decode_requests.iter().map(|req| req.token_id).collect();
            let prefill_lora_adapters: Vec<Option<&str>> = prefill_requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            let decode_lora_adapters: Vec<Option<&str>> = decode_requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            let logits = lane.execute_unified(
                &prefill_prompts,
                prefill_kv_views,
                &prefill_lora_adapters,
                &decode_tokens,
                decode_kv_views,
                &decode_lora_adapters,
            )?;
            if collect_result {
                // Logits columns: prefill requests first, then decode rows.
                let params: Vec<&SamplingParams> = prefill_requests
                    .iter()
                    .map(|r| &r.params)
                    .chain(decode_requests.iter().map(|r| &r.params))
                    .collect();
                let random_vals: Vec<f32> = prefill_requests
                    .iter()
                    .map(|r| r.random_val)
                    .chain(decode_requests.iter().map(|r| r.random_val))
                    .collect();
                let tokens = lane.select_step_tokens(&logits, &params, &random_vals)?;
                Ok(WorkerStepOutcome::Unified(UnifiedResult {
                    prefill_requests: build_prefill_request_results(
                        lane,
                        prefill_requests,
                        &logits,
                        &tokens,
                        None,
                        false,
                    )?,
                    decode_requests: build_decode_request_results(
                        lane,
                        decode_requests,
                        &logits,
                        prefill_requests.len(),
                        &tokens,
                    )?,
                }))
            } else {
                Ok(WorkerStepOutcome::Ack)
            }
        }
    }
}

struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            openinfer_core::ffi::cublas_destroy();
        }
    }
}

struct SamplingScratch {
    row_indices: cudarc::driver::CudaSlice<i32>,
    argmax_partial_values: cudarc::driver::CudaSlice<f32>,
    argmax_partial_indices: cudarc::driver::CudaSlice<i32>,
    probs: cudarc::driver::CudaSlice<f32>,
    top1_values: cudarc::driver::CudaSlice<half::bf16>,
    row_states: cudarc::driver::CudaSlice<u8>,
    valid: cudarc::driver::CudaSlice<u8>,
    out: cudarc::driver::CudaSlice<i32>,
    out_host: Vec<i32>,
}

impl SamplingScratch {
    fn new(ctx: &DeviceContext, vocab_size: usize, max_batch_bucket: usize) -> Result<Self> {
        let partials =
            openinfer_core::ops::argmax_batch_bf16_split_partials_len(max_batch_bucket, vocab_size);
        Ok(Self {
            row_indices: ctx.stream.alloc_zeros(max_batch_bucket)?,
            argmax_partial_values: ctx.stream.alloc_zeros(partials)?,
            argmax_partial_indices: ctx.stream.alloc_zeros(partials)?,
            probs: ctx.stream.alloc_zeros(vocab_size)?,
            top1_values: ctx.stream.alloc_zeros(max_batch_bucket)?,
            row_states: ctx
                .stream
                .alloc_zeros(openinfer_core::ops::flashinfer_topk_row_states_bytes())?,
            valid: ctx.stream.alloc_zeros(1)?,
            out: ctx.stream.alloc_zeros(max_batch_bucket)?,
            out_host: vec![0; max_batch_bucket],
        })
    }
}

fn compute_logprobs_from_cpu(
    logits_f32: &[f32],
    sampled_token: u32,
    top_k: usize,
) -> Option<TokenLogprob> {
    if logits_f32.is_empty() {
        return None;
    }

    let max_val = logits_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits_f32.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();
    let sampled_logprob = logits_f32[sampled_token as usize] - log_sum_exp;

    let k = top_k.min(logits_f32.len());
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
    if k > 0 {
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
        for (idx, val) in best {
            top.push((idx, val - log_sum_exp));
        }
    }

    Some(TokenLogprob {
        logprob: sampled_logprob,
        top_logprobs: top,
    })
}

fn bind_model_thread(model: &Qwen3Model) -> Result<()> {
    unsafe {
        let err = openinfer_core::ffi::cuda_set_device(model.device_ctx().device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on worker thread: cudaError={}",
                model.device_ctx().device_ordinal,
                err
            ));
        }
    }
    model
        .device_ctx()
        .ctx
        .bind_to_thread()
        .map_err(|e| anyhow::anyhow!("Failed to bind CUDA context to thread: {e}"))?;
    unsafe {
        openinfer_core::ffi::cublas_init();
    }
    Ok(())
}

/// Pick the fastest cublasLt algo for every decode GEMM shape (buckets up to
/// `GEMM_LT_MAX_N`) before the first step, so CUDA-Graph capture bakes in the
/// tuned kernels; adds a few seconds of startup per model thread. Every
/// layer's weights enter the timing rotation to keep the loop L2-cold, the
/// regime steady-state decode runs in.
fn tune_decode_gemm_algos(model: &Qwen3Model) -> Result<()> {
    let ctx = model.device_ctx();
    let hidden = model.config().hidden_size;
    let vocab = model.config().vocab_size;
    let q_dim = model.local_q_dim();
    let kv_dim = model.local_kv_dim();
    let intermediate = model.local_intermediate_size();
    let layers = &model.layers;

    let q_samples: Vec<_> = layers.iter().map(|l| (&l.attention.qkv_proj, 0)).collect();
    let kv_samples: Vec<_> = layers
        .iter()
        .flat_map(|l| {
            [
                (&l.attention.qkv_proj, q_dim),
                (&l.attention.qkv_proj, q_dim + kv_dim),
            ]
        })
        .collect();
    let o_samples: Vec<_> = layers.iter().map(|l| (&l.attention.o_proj, 0)).collect();
    let gate_up_samples: Vec<_> = layers
        .iter()
        .flat_map(|l| {
            [
                (&l.mlp.gate_up_proj, 0),
                (&l.mlp.gate_up_proj, intermediate),
            ]
        })
        .collect();
    let down_samples: Vec<_> = layers.iter().map(|l| (&l.mlp.down_proj, 0)).collect();
    let lm_head_samples = [(model.output_projection(), 0)];

    for &n in BATCH_BUCKETS.iter().filter(|&&b| b <= ops::GEMM_LT_MAX_N) {
        ops::gemm_lt_tune(ctx, &q_samples, q_dim, n)?;
        ops::gemm_lt_tune(ctx, &kv_samples, kv_dim, n)?;
        ops::gemm_lt_tune(ctx, &o_samples, hidden, n)?;
        ops::gemm_lt_tune(ctx, &gate_up_samples, intermediate, n)?;
        ops::gemm_lt_tune(ctx, &down_samples, hidden, n)?;
        ops::gemm_lt_tune(ctx, &lm_head_samples, vocab, n)?;
    }
    Ok(())
}

pub struct PrefillPlan<'a> {
    pub requests: &'a [PrefillStepItem],
    pub echo: bool,
}

pub struct DecodePlan<'a> {
    pub requests: &'a [DecodeStepItem],
}

pub struct UnifiedPlan<'a> {
    pub prefill_requests: &'a [PrefillStepItem],
    pub decode_requests: &'a [DecodeStepItem],
}

#[derive(Clone, Debug)]
pub struct PrefillRequestResult {
    pub request_id: RequestId,
    pub first_token: u32,
    pub first_token_logprob: Option<TokenLogprob>,
    pub prompt_logprobs: Option<Vec<Option<TokenLogprob>>>,
    /// Prompt tokens served from the prefix cache (KV reused, not recomputed).
    pub cached_tokens: usize,
    /// Whether the prompt is fully prefilled. When false this step ran a
    /// non-final chunk and `first_token` is meaningless.
    pub completed: bool,
    /// Prompt tokens with KV computed after this step (authoritative —
    /// includes prefix-cache hits the scheduler can't see).
    pub prefill_pos: usize,
}

#[derive(Clone, Debug)]
pub struct DecodeRequestResult {
    pub request_id: RequestId,
    pub token: u32,
    pub logprob: Option<TokenLogprob>,
}

pub struct PrefillResult {
    pub requests: Vec<PrefillRequestResult>,
    pub dflash_context_captured_requests: Vec<RequestId>,
}

pub struct DecodeResult {
    pub requests: Vec<DecodeRequestResult>,
}

pub struct UnifiedResult {
    pub prefill_requests: Vec<PrefillRequestResult>,
    pub decode_requests: Vec<DecodeRequestResult>,
}

pub(crate) trait ModelExecutor: Send {
    fn block_size(&self) -> usize;
    fn max_request_blocks(&self) -> usize;
    fn max_context_tokens(&self) -> usize;
    fn max_decode_batch_size(&self) -> usize;
    fn available_blocks(&self) -> usize;
    fn is_stop_token(&self, token_id: u32) -> bool;
    fn drop_request(&mut self, request_id: RequestId) -> Result<()>;

    fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult>;
    fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult>;
    fn execute_speculative_verify(
        &mut self,
        _plan: SpeculativeVerifyPlan<'_>,
    ) -> Result<SpeculativeVerifyResult> {
        anyhow::bail!("speculative verification is not implemented for this executor")
    }
    fn execute_speculative_draft(
        &mut self,
        _plan: SpeculativeDraftPlan<'_>,
    ) -> Result<SpeculativeDraftResult> {
        anyhow::bail!("speculative draft is not implemented for this executor")
    }
    fn speculative_enabled(&self) -> bool {
        false
    }
    fn speculative_request_ready(&self, _request_id: RequestId) -> bool {
        self.speculative_enabled()
    }
    fn speculative_state_budget_bytes(&self) -> Option<usize> {
        None
    }
    fn speculative_request_state_bytes(&self, _prompt_len: usize, _max_tokens: usize) -> usize {
        0
    }
    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult>;

    fn load_lora_adapter(&mut self, request: &LoadLoraAdapterRequest) -> Result<()> {
        anyhow::bail!(
            "Qwen3 LoRA adapter loading is not implemented yet: name={}, path={}",
            request.lora_name,
            request.lora_path.display()
        )
    }

    fn unload_lora_adapter(&mut self, request: &UnloadLoraAdapterRequest) -> Result<()> {
        anyhow::bail!(
            "Qwen3 LoRA adapter unloading is not implemented yet: name={}",
            request.lora_name
        )
    }

    fn list_lora_adapters(&self) -> Vec<String> {
        Vec::new()
    }

    // ── KV-offload prefetch hooks (no-op unless offload is enabled) ─────

    /// Offer a freshly-submitted request for async CPU-tier KV prefetch.
    /// Returns `true` if a load is now in flight and the scheduler must park
    /// the request until [`Self::drain_ready_prefetch`] reports it ready.
    ///
    /// `reserve_floor` is the number of free blocks already promised to
    /// admitted requests (active decode growth + remaining prefill chunks);
    /// the prefetch must not reserve into it, or a mid-prefill request's next
    /// chunk fails allocation and the whole step errors out.
    fn begin_kv_prefetch(
        &mut self,
        _request_id: RequestId,
        _prompt_tokens: &[u32],
        _lora_adapter: Option<&str>,
        _reserve_floor: usize,
    ) -> bool {
        false
    }

    /// Non-blocking sweep: request ids whose prefetch just settled (now
    /// prefill-eligible).
    fn drain_ready_prefetch(&mut self) -> Vec<RequestId> {
        Vec::new()
    }

    /// Block until at least one in-flight prefetch settles (idle-only), then
    /// sweep the rest.
    fn wait_ready_prefetch(&mut self) -> Vec<RequestId> {
        Vec::new()
    }

    /// Blocks `request_id` already holds via a settled prefetch (its restored
    /// prefix). These were taken out of the free pool for this request and
    /// become its cached prefill prefix, so admission credits them against the
    /// request's block need to avoid double-counting. Zero unless a prefetch
    /// has committed for `request_id`.
    fn prefetched_blocks(&self, _request_id: RequestId) -> usize {
        0
    }
}

struct Qwen3ExecutorMetadata {
    block_size: usize,
    stop_token_ids: Vec<u32>,
    config: Config,
    max_context_tokens: usize,
    dflash_config: Option<DFlashConfig>,
    dflash_state_budget_bytes: usize,
}

pub struct Qwen3Executor {
    metadata: Qwen3ExecutorMetadata,
    kv_mgr: KvCacheManager,
    request_kvs: HashMap<RequestId, openinfer_kv_cache::RequestKv>,
    primary: RankWorker,
    workers: Vec<RankWorker>,
    loaded_lora_adapters: HashSet<String>,
    prefix_cache_enabled: bool,
    lora_options: Qwen3LoraOptions,
    /// pegaflow KV-offload bridge; `None` unless offload is opted in on the
    /// single-GPU path. Drives both the SAVE hook and the async LOAD prefetch.
    offload: Option<OffloadEngine>,
    /// Per-request count of sealed blocks already saved to the host tier, so
    /// each step only saves blocks that newly sealed. Initialized to the
    /// GPU-hit prefix (already resident) on first save.
    saved_cursor: HashMap<RequestId, usize>,
    /// In-flight CPU→GPU prefetches keyed by request, parked until their load
    /// settles and the blocks register into the prefix cache.
    prefetch: HashMap<RequestId, PrefetchState>,
    /// Offload pure-L2 mode. When set, completed blocks are not kept for
    /// cross-request HBM reuse: the prefetch probe drains the inactive pool
    /// first, so every probe sees `gpu_hit == 0` and the whole cacheable prefix
    /// is restored from the host tier. This is what `--no-prefix-cache` means
    /// once offload is on (the L2 restore still rides on `match_and_add_prefix`,
    /// so prefix matching itself stays enabled). Set via
    /// [`Self::set_no_prefix_cache`].
    l1_retention_disabled: bool,
    speculative_enabled: bool,
    dflash_ready_requests: HashSet<RequestId>,
}

/// One request's in-flight CPU-tier KV prefetch.
///
/// Holds the destination blocks (via `probe`/`reservation`) and the load handle
/// so the scheduler can poll completion non-blockingly. Once the load settles,
/// the reservation is committed (blocks staged + registered) and only `probe`
/// remains, holding the GPU+CPU prefix resident until the request prefills.
struct PrefetchState {
    probe: PrefixProbe,
    /// `Some` until the load lands and the blocks are committed.
    reservation: Option<LoadReservation>,
    /// `Some` while the DMA is in flight; `None` once it has settled.
    handle: Option<LoadHandle>,
}

impl Drop for Qwen3Executor {
    fn drop(&mut self) {
        self.primary.shutdown();
        for worker in &mut self.workers {
            worker.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DFLASH_MIN_EXTRA_MEMORY_RESERVE_BYTES, dflash_request_state_footprint_bytes_for_len,
        dflash_request_state_reserve_bytes,
    };
    use crate::config::{DFlashConfig, DFlashInnerConfig};

    #[test]
    fn dflash_request_state_reserve_covers_long_context_scratch() {
        let config = DFlashConfig {
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 5,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_target_layers: 5,
            head_dim: 128,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 40960,
            block_size: 16,
            dflash_config: DFlashInnerConfig {
                mask_token_id: 151669,
                target_layer_ids: vec![1, 9, 17, 25, 33],
            },
        };

        let reserve = dflash_request_state_reserve_bytes(&config).unwrap();

        assert!(
            reserve > DFLASH_MIN_EXTRA_MEMORY_RESERVE_BYTES * 2,
            "Qwen3-4B DFlash reserve should account for long-context request state, got {reserve}"
        );
    }

    #[test]
    fn dflash_short_request_footprint_does_not_include_global_allocator_floor() {
        let config = DFlashConfig {
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 5,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_target_layers: 5,
            head_dim: 128,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 40960,
            block_size: 16,
            dflash_config: DFlashInnerConfig {
                mask_token_id: 151669,
                target_layer_ids: vec![1, 9, 17, 25, 33],
            },
        };

        let footprint = dflash_request_state_footprint_bytes_for_len(&config, 80).unwrap();

        assert!(
            footprint < DFLASH_MIN_EXTRA_MEMORY_RESERVE_BYTES / 8,
            "short-request admission must charge true DFlash state footprint, not the global floor: {footprint}"
        );
    }
}
