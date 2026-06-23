use std::collections::{HashMap, HashSet};
use std::thread;

use anyhow::Result;
use crossbeam_channel as channel;

use crate::batch_decode_buffers::{BATCH_BUCKETS, BatchDecodeBuffers};
use crate::config::{Config, TensorParallelConfig};
use crate::weights::{KvBudget, ModelRuntimeConfig, Qwen3MemoryOptions, Qwen3Model};
use crate::{Qwen3LoraOptions, Qwen3OffloadOptions};
use openinfer_core::engine::{LoadLoraAdapterRequest, TokenLogprob, UnloadLoraAdapterRequest};
use openinfer_core::kv_pool::KvLayout;
use openinfer_core::ops;
use openinfer_core::sampler::SamplingParams;
use openinfer_core::tensor::{DeviceContext, DeviceVec, HiddenStates};
use openinfer_kv_cache::{
    KvBlockGuard, KvBuffer, KvCacheManager, KvView, LoadReservation, PrefixProbe,
};
use openinfer_kv_offload::{LoadHandle, OffloadConfig, OffloadEngine};

mod dflash_lane;
mod dflash_prefill;
mod spec;

use crate::dflash::DFlashDraftModel;
use crate::speculative::{
    DraftPlan, DraftResult, DraftStepItem, VerifyPlan, VerifyResult, VerifyStepItem,
    build_verify_results,
};
use dflash_lane::DFlashLaneState;
use dflash_prefill::{DFlashPrefillAction, dflash_prefill_action};

use crate::verify_graph::VerifyGraphBuffers;

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
            cached_tokens: 0,
            chunk_budget: usize::MAX,
            chunk_start: 0,
            chunk_tokens,
        }
    }

    #[must_use]
    pub fn with_lora_adapter(mut self, lora_adapter: Option<String>) -> Self {
        self.lora_adapter = lora_adapter;
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
}

impl DecodeStepItem {
    pub fn new(
        request_id: RequestId,
        token_id: u32,
        params: SamplingParams,
        logprobs: usize,
    ) -> Self {
        Self {
            request_id,
            token_id,
            params,
            logprobs,
            lora_adapter: None,
        }
    }

    #[must_use]
    pub fn with_lora_adapter(mut self, lora_adapter: Option<String>) -> Self {
        self.lora_adapter = lora_adapter;
        self
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
    sample_seed: u64,
) -> Result<Vec<DecodeRequestResult>> {
    let params: Vec<&SamplingParams> = requests.iter().map(|req| &req.params).collect();
    let tokens = openinfer_sample::select_batch(
        lane.model.device_ctx(),
        &lane.bufs.logits,
        &params,
        sample_seed,
        &mut lane.sample_scratch,
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
            sample_seed,
        } => {
            let prompts: Vec<&[u32]> = requests.iter().map(PrefillStepItem::as_slice).collect();
            let lora_adapters: Vec<Option<&str>> = requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            // When DFlash is loaded, capture target hidden states for eligible
            // requests so they can seed the draft model after prefill finishes.
            let capture_requested = lane.should_capture_dflash_prefill_context(requests);
            let capture_layer_ids = if capture_requested {
                lane.dflash_capture_layer_ids()
            } else {
                None
            };
            let (logits, all_position_logits, captured_hidden) = lane.execute_prefill(
                &prompts,
                kv_views,
                &lora_adapters,
                *echo,
                capture_layer_ids.as_deref(),
            )?;
            let dflash_context_captured_requests = lane.record_prefill_dflash_context(
                requests,
                capture_requested,
                captured_hidden.as_ref(),
            )?;
            if collect_result {
                let params: Vec<&SamplingParams> = requests.iter().map(|r| &r.params).collect();
                let tokens = lane.select_step_tokens(&logits, &params, *sample_seed)?;
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
        StepCommand::Decode {
            requests,
            kv_views,
            sample_seed,
        } => {
            let token_ids: Vec<u32> = requests.iter().map(|req| req.token_id).collect();
            let lora_adapters: Vec<Option<&str>> = requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            lane.execute_decode(&token_ids, kv_views, &lora_adapters)?;
            if collect_result {
                Ok(WorkerStepOutcome::Decode(DecodeResult {
                    requests: build_batch_decode_request_results(lane, requests, *sample_seed)?,
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
            sample_seed,
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
                let tokens = lane.select_step_tokens(&logits, &params, *sample_seed)?;
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
        StepCommand::SplitConcurrent {
            prefill_requests,
            prefill_kv_views,
            decode_requests,
            decode_kv_views,
            prefill_stream,
            decode_stream,
            sample_seed,
        } => {
            use openinfer_kernels::tensor::{clear_stream_override, set_stream_override};

            // If there's still an inflight prefill from a previous step, sync it
            // now (shouldn't happen normally, but safety).
            if lane.inflight_prefill.is_some() {
                lane.resolve_inflight_prefill()?;
            }

            let prefill_prompts: Vec<&[u32]> = prefill_requests
                .iter()
                .map(PrefillStepItem::as_slice)
                .collect();
            let prefill_lora_adapters: Vec<Option<&str>> = prefill_requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            let decode_tokens: Vec<u32> = decode_requests.iter().map(|req| req.token_id).collect();
            let decode_lora_adapters: Vec<Option<&str>> = decode_requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();

            // Sync ctx.stream: ensures all prior stream-ordered allocs and
            // H2D copies are complete before green streams touch them.
            lane.model.device_ctx().sync()?;

            // Launch prefill on prefill partition stream.
            unsafe { set_stream_override(prefill_stream.0) };
            let (prefill_logits, _, _) = lane.execute_prefill(
                &prefill_prompts,
                prefill_kv_views,
                &prefill_lora_adapters,
                false,
                None,
            )?;
            clear_stream_override();

            // Launch decode on the decode partition stream. CUDA graph stays
            // enabled: batch_decode captures into the split graph cache keyed on
            // the active stream override, so the replayed kernel nodes stay
            // pinned to the decode SM partition (CUDA PG §4.6.5 — capture stream
            // determines a node's execution context).
            unsafe { set_stream_override(decode_stream.0) };
            lane.execute_decode(&decode_tokens, decode_kv_views, &decode_lora_adapters)?;
            clear_stream_override();

            // Only sync decode stream — decode result is ready for sampling.
            // Prefill continues async on GPU; polled later via event.
            let r = unsafe { cudarc::driver::sys::cuStreamSynchronize(decode_stream.0) };
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                anyhow::bail!("cuStreamSynchronize(decode) failed: {r:?}");
            }

            if collect_result {
                // Sample decode tokens immediately.
                let decode_result =
                    build_batch_decode_request_results(lane, decode_requests, *sample_seed)?;

                // Record event on prefill stream for non-blocking poll.
                let mut event: cudarc::driver::sys::CUevent = std::ptr::null_mut();
                unsafe {
                    cudarc::driver::sys::cuEventCreate(
                        &mut event,
                        cudarc::driver::sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32,
                    );
                    cudarc::driver::sys::cuEventRecord(event, prefill_stream.0);
                }

                // Store prefill state for deferred sync+sample.
                lane.inflight_prefill = Some(InflightPrefillState {
                    prefill_stream: prefill_stream.0,
                    prefill_logits,
                    prefill_requests: prefill_requests.clone(),
                    sample_seed: *sample_seed,
                });

                Ok(WorkerStepOutcome::SplitDecodeReady {
                    decode: DecodeResult {
                        requests: decode_result,
                    },
                    prefill_event: SendEvent(event),
                })
            } else {
                // Non-primary worker: still need to sync prefill before returning.
                let r = unsafe { cudarc::driver::sys::cuStreamSynchronize(prefill_stream.0) };
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    anyhow::bail!("cuStreamSynchronize(prefill) failed: {r:?}");
                }
                Ok(WorkerStepOutcome::Ack)
            }
        }
        StepCommand::SpeculativeVerify { requests, kv_views } => {
            // One target forward over each request's K+1 draft span with a
            // speculative KV view. The fixed-buffer verify path computes all-
            // position logits (accept_greedy needs the target's posterior at
            // each span position) and captures the target hidden states (at the
            // DFlash layers) to seed the next draft — all into reused,
            // pointer-stable scratch (`VerifyGraphBuffers`).
            let result = lane.execute_dflash_verify(requests, kv_views)?;
            Ok(WorkerStepOutcome::SpeculativeVerify(result))
        }
        StepCommand::SpeculativeDraft { requests } => Ok(WorkerStepOutcome::SpeculativeDraft(
            lane.execute_dflash_draft(requests)?,
        )),
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

/// Prepare decode GEMM algos before capture, per the active numeric policy: under `Pin`, eagerly pin
/// one algo per projection {M,K} (reused for all N); otherwise tune the fastest cublasLt algo per
/// decode shape (buckets up to `GEMM_LT_MAX_N`, every layer's weights in the L2-cold timing rotation).
/// Adds a few seconds of startup per model thread.
fn tune_decode_gemm_algos(model: &Qwen3Model) -> Result<()> {
    let ctx = model.device_ctx();
    let hidden = model.config().hidden_size;
    let vocab = model.config().vocab_size;
    let q_dim = model.local_q_dim();
    let kv_dim = model.local_kv_dim();
    let intermediate = model.local_intermediate_size();

    use openinfer_kernels::ops::{NumericPolicy, gemm_lt_pin_warmup, numeric_policy};
    if numeric_policy() == NumericPolicy::Pin {
        // Eager pin before capture: the lazy pin-workspace alloc is illegal mid-capture.
        gemm_lt_pin_warmup(q_dim, hidden)?;
        gemm_lt_pin_warmup(kv_dim, hidden)?;
        gemm_lt_pin_warmup(hidden, q_dim)?;
        gemm_lt_pin_warmup(intermediate, hidden)?;
        gemm_lt_pin_warmup(hidden, intermediate)?;
        gemm_lt_pin_warmup(vocab, hidden)?;
        return Ok(());
    }

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
    pub sample_seed: u64,
}

pub struct DecodePlan<'a> {
    pub requests: &'a [DecodeStepItem],
    pub sample_seed: u64,
}

pub struct UnifiedPlan<'a> {
    pub prefill_requests: &'a [PrefillStepItem],
    pub decode_requests: &'a [DecodeStepItem],
    pub sample_seed: u64,
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
    /// Requests whose DFlash target context was captured this prefill step.
    /// Empty unless speculative decoding is enabled. The executor folds these
    /// into its `dflash_ready_requests` set once the prompt is fully prefilled.
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
    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult>;

    /// Run one speculative draft round (propose `K` tokens per request). Only
    /// meaningful when [`Self::speculative_enabled`] is true.
    fn execute_speculative_draft(&mut self, _plan: DraftPlan<'_>) -> Result<DraftResult> {
        anyhow::bail!("speculative draft is not implemented for this executor")
    }

    /// Verify a draft span with one target forward and accept the greedy prefix.
    fn execute_speculative_verify(&mut self, _plan: VerifyPlan<'_>) -> Result<VerifyResult> {
        anyhow::bail!("speculative verification is not implemented for this executor")
    }

    /// Whether a draft model is loaded and speculative decoding is active.
    fn speculative_enabled(&self) -> bool {
        false
    }

    /// Whether `request_id` has captured draft context and can be drafted.
    fn speculative_request_ready(&self, _request_id: RequestId) -> bool {
        false
    }

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

    // ── Decode-overlap async prefill ─────────────────────────────────────

    /// Whether prefill/decode overlap is enabled (async prefill supported).
    fn has_decode_overlap(&self) -> bool {
        false
    }

    /// Poll whether the async prefill has completed. Returns `Some(result)` if
    /// done, `None` if still in-flight.
    fn poll_async_prefill(&mut self) -> Option<PrefillResult> {
        None
    }
}

struct Qwen3ExecutorMetadata {
    block_size: usize,
    stop_token_ids: Vec<u32>,
    config: Config,
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
    /// Green Context SM partition for concurrent prefill/decode. `None` when
    /// disabled (default) or when the GPU does not support Green Contexts.
    overlap: Option<crate::green_ctx::OverlapStreams>,
    /// In-flight async prefill state. Populated by the SplitConcurrent step,
    /// consumed by `poll_async_prefill`.
    async_prefill: Option<AsyncPrefillState>,
    /// DFlash draft metadata; `Some` once a draft model is loaded into the
    /// primary lane. Speculative decoding is enabled iff this is set.
    speculative: Option<DFlashMeta>,
    /// Requests whose DFlash context is captured and ready to draft. A request
    /// enters this set when its prompt finishes prefilling with captured target
    /// context, and leaves on retire or a plain (non-speculative) decode.
    dflash_ready_requests: HashSet<RequestId>,
}

/// State for an in-flight async prefill on the prefill overlap stream.
struct AsyncPrefillState {
    event: cudarc::driver::sys::CUevent,
}

// SAFETY: AsyncPrefillState is only accessed from the single executor/scheduler
// thread that owns the GPU context. The raw CUevent pointer is not shared.
unsafe impl Send for AsyncPrefillState {}

/// Wrapper to send CUevent across the worker→executor channel boundary.
/// SAFETY: The event is created on the worker thread's GPU context and consumed
/// on the executor thread (same device, sequential access).
struct SendEvent(cudarc::driver::sys::CUevent);
unsafe impl Send for SendEvent {}

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

impl Qwen3Executor {
    pub(crate) fn single(
        model: Qwen3Model,
        offload_opts: &Qwen3OffloadOptions,
        max_prefill_tokens: usize,
        dflash_kv_bytes_per_token: usize,
        memory_options: Qwen3MemoryOptions,
    ) -> Result<Self> {
        let (model, budget) = profile_kv_budget_on_worker(
            model,
            max_prefill_tokens,
            dflash_kv_bytes_per_token,
            memory_options,
        )?;
        let kv_mgr = KvCacheManager::new(
            &model.device_ctx().stream,
            budget.num_layers,
            budget.num_kv_heads,
            budget.head_dim,
            budget.block_size,
            budget.num_blocks,
        )?;
        let metadata = Qwen3ExecutorMetadata {
            block_size: budget.block_size,
            stop_token_ids: model.config().stop_token_ids.clone(),
            config: model.config().clone(),
        };
        let kv_buffer = kv_mgr.buffer().clone();
        // Build the offload engine while the model's stream is still in hand
        // (it moves into the RankWorker below). Registers the fused KV buffer.
        let offload = build_offload(offload_opts, &kv_mgr, model.device_ctx())?;
        let total_blocks = kv_mgr.pool().total_blocks();
        let padding_block_id = kv_mgr.pool().padding_block_id();
        Ok(Self {
            metadata,
            kv_mgr,
            request_kvs: HashMap::new(),
            primary: RankWorker::spawn(
                0,
                LocalQwen3Lane::new(model, kv_buffer, total_blocks, padding_block_id)?,
            )?,
            workers: Vec::new(),
            loaded_lora_adapters: HashSet::new(),
            prefix_cache_enabled: true,
            lora_options: Qwen3LoraOptions::default(),
            offload,
            saved_cursor: HashMap::new(),
            prefetch: HashMap::new(),
            l1_retention_disabled: false,
            overlap: None,
            async_prefill: None,
            speculative: None,
            dflash_ready_requests: HashSet::new(),
        })
    }

    pub fn from_runtime(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        Self::from_runtime_with_lora_options(
            model_path,
            enable_cuda_graph,
            device_ordinals,
            Qwen3LoraOptions::default(),
            Qwen3OffloadOptions::disabled(),
            crate::scheduler::DEFAULT_MAX_PREFILL_TOKENS,
            None,
            Qwen3MemoryOptions::default(),
        )
    }

    pub fn from_runtime_with_lora_options(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
        lora_options: Qwen3LoraOptions,
        offload_options: Qwen3OffloadOptions,
        max_prefill_tokens: usize,
        dflash_draft_path: Option<&str>,
        memory_options: Qwen3MemoryOptions,
    ) -> Result<Self> {
        let mut memory_options = memory_options.validate()?;
        let lora_options = lora_options.validate()?;
        anyhow::ensure!(
            !device_ordinals.is_empty(),
            "Qwen3 executor requires at least one device"
        );
        anyhow::ensure!(
            !offload_options.enabled || device_ordinals.len() == 1,
            "KV offload is only supported on the single-GPU path (tensor parallel \
             shards KV per rank); got {} devices",
            device_ordinals.len()
        );
        if device_ordinals.len() == 1 {
            let model = Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: None,
                    device_ordinal: device_ordinals[0],
                    max_loras: lora_options.max_loras,
                    max_lora_rank: lora_options.max_lora_rank,
                },
            )?;
            // The DFlash draft model loads after profiling but lives outside the
            // paged KV pool, so reserve its footprint up front from the draft
            // config: fixed bytes (weights + block scratch) via the margin, and
            // pool-scaling per-token bytes folded into the block budget.
            let dflash_kv_bytes_per_token = match dflash_draft_path {
                Some(path) => {
                    let reservation = crate::dflash::DFlashMemoryReservation::from_path(
                        path,
                        *BATCH_BUCKETS.last().unwrap(),
                    )?;
                    memory_options.kv_cache_memory_margin_bytes += reservation.fixed_bytes;
                    reservation.kv_bytes_per_token
                }
                None => 0,
            };
            let mut executor = Self::single(
                model,
                &offload_options,
                max_prefill_tokens,
                dflash_kv_bytes_per_token,
                memory_options,
            )?;
            executor.lora_options = lora_options;
            return Ok(executor);
        }
        anyhow::ensure!(
            dflash_draft_path.is_none(),
            "speculative decoding requires the single-GPU path (got {} devices)",
            device_ordinals.len()
        );

        let world_size = device_ordinals.len();
        let mut models = Vec::with_capacity(world_size);
        for (rank, &device_ordinal) in device_ordinals.iter().enumerate() {
            models.push(Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: Some(TensorParallelConfig { rank, world_size }),
                    device_ordinal,
                    max_loras: lora_options.max_loras,
                    max_lora_rank: lora_options.max_lora_rank,
                },
            )?);
        }

        // Profile each rank independently and use the minimum shared block
        // count. The logical scheduler uses one block budget for all ranks, but
        // free memory and worker-thread runtime allocations are per device.
        let mut profiled_models = Vec::with_capacity(world_size);
        let mut budgets = Vec::with_capacity(world_size);
        for model in models {
            // DFlash is single-GPU only, so the TP path reserves nothing for it.
            let (model, budget) =
                profile_kv_budget_on_worker(model, max_prefill_tokens, 0, memory_options)?;
            profiled_models.push(model);
            budgets.push(budget);
        }
        let mut models = profiled_models;
        let mut budget = budgets[0];
        budget.num_blocks = budgets
            .iter()
            .map(|budget| budget.num_blocks)
            .min()
            .expect("at least one TP rank");
        log::info!(
            "TP KV budget: using {} blocks (minimum across {} ranks)",
            budget.num_blocks,
            world_size
        );

        // Create the centralized KvCacheManager on rank 0's stream.
        let kv_mgr = KvCacheManager::new(
            &models[0].device_ctx().stream,
            budget.num_layers,
            budget.num_kv_heads,
            budget.head_dim,
            budget.block_size,
            budget.num_blocks,
        )?;

        let metadata = Qwen3ExecutorMetadata {
            block_size: budget.block_size,
            stop_token_ids: models[0].config().stop_token_ids.clone(),
            config: models[0].config().clone(),
        };

        // Create extra KvBuffers for ranks 1+ on their respective streams.
        let mut extra_kv_buffers = Vec::with_capacity(world_size - 1);
        for model in &models[1..] {
            extra_kv_buffers.push(KvBuffer::new(
                &model.device_ctx().stream,
                budget.num_layers,
                budget.num_kv_heads,
                budget.head_dim,
                budget.block_size,
                budget.num_blocks,
            )?);
        }

        let streams = models
            .iter()
            .map(|m| m.device_ctx().stream.clone())
            .collect();
        let comms = cudarc::nccl::safe::Comm::from_devices(streams)
            .map_err(|e| anyhow::anyhow!("failed to initialize NCCL comms: {e:?}"))?;
        for (model, comm) in models.iter_mut().zip(comms) {
            model.attach_tp_comm(comm);
        }

        let total_blocks = kv_mgr.pool().total_blocks();
        let padding_block_id = kv_mgr.pool().padding_block_id();

        // Primary rank gets the KvBuffer from the centralized manager.
        let primary_buffer = kv_mgr.buffer().clone();
        let mut models_iter = models.into_iter();
        let primary_model = models_iter.next().unwrap();
        let primary = RankWorker::spawn(
            0,
            LocalQwen3Lane::new(
                primary_model,
                primary_buffer,
                total_blocks,
                padding_block_id,
            )?,
        )?;

        // Worker ranks get their own extra KvBuffers.
        let workers = models_iter
            .zip(extra_kv_buffers)
            .enumerate()
            .map(|(index, (model, buffer))| {
                let lane = LocalQwen3Lane::new(model, buffer, total_blocks, padding_block_id)?;
                RankWorker::spawn(index + 1, lane)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            metadata,
            kv_mgr,
            request_kvs: HashMap::new(),
            primary,
            workers,
            loaded_lora_adapters: HashSet::new(),
            prefix_cache_enabled: true,
            lora_options,
            // Offload is single-GPU only (asserted above); never built here.
            offload: None,
            saved_cursor: HashMap::new(),
            prefetch: HashMap::new(),
            l1_retention_disabled: false,
            overlap: None,
            async_prefill: None,
            speculative: None,
            dflash_ready_requests: HashSet::new(),
        })
    }

    pub fn block_size(&self) -> usize {
        <Self as ModelExecutor>::block_size(self)
    }

    pub fn max_request_blocks(&self) -> usize {
        <Self as ModelExecutor>::max_request_blocks(self)
    }

    pub fn available_blocks(&self) -> usize {
        <Self as ModelExecutor>::available_blocks(self)
    }

    pub fn is_stop_token(&self, token_id: u32) -> bool {
        <Self as ModelExecutor>::is_stop_token(self, token_id)
    }

    pub fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        <Self as ModelExecutor>::drop_request(self, request_id)
    }

    pub fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        <Self as ModelExecutor>::execute_prefill(self, plan)
    }

    pub fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        <Self as ModelExecutor>::execute_decode(self, plan)
    }

    pub fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult> {
        <Self as ModelExecutor>::execute_unified(self, plan)
    }

    pub fn load_lora_adapter(&mut self, request: &LoadLoraAdapterRequest) -> Result<()> {
        <Self as ModelExecutor>::load_lora_adapter(self, request)
    }

    /// Prefix caching is on by default; tests that assert bit-identical
    /// replay disable it (a cache hit changes prefill GEMM shapes, which
    /// drifts logits by bf16 ULPs).
    pub fn set_prefix_cache_enabled(&mut self, enabled: bool) {
        self.prefix_cache_enabled = enabled;
    }

    /// Configure two-stream prefill/decode overlap (see [`crate::DecodeOverlap`]).
    /// A no-op for [`crate::DecodeOverlap::Off`]; otherwise sets up the streams,
    /// returning an error if the GPU/driver cannot honor the requested mode.
    pub fn enable_decode_overlap(&mut self, overlap: crate::DecodeOverlap) -> Result<()> {
        let device_ordinal = 0; // single-GPU path
        self.overlap = crate::green_ctx::OverlapStreams::create(device_ordinal, overlap)?;
        Ok(())
    }

    /// Whether prefill/decode overlap (two-stream or SM partition) is active.
    pub fn decode_overlap_enabled(&self) -> bool {
        self.overlap.is_some()
    }

    /// vLLM-style `--no-prefix-cache`. Behaviour depends on whether offload is
    /// active:
    ///   * **No offload** — classic: disable prefix matching outright, so every
    ///     prefill recomputes the full prompt.
    ///   * **With offload** — pure-L2 mode: keep matching on (the host-tier
    ///     restore registers blocks and relies on `match_and_add_prefix` to pick
    ///     them up) but stop retaining completed blocks in HBM, so no request
    ///     ever serves its prefix from a cross-request L1 hit. Every reuse then
    ///     comes from the host tier, which is the point of the L2 benchmark.
    ///
    /// A resident HBM block and its host-tier copy share one content hash, so
    /// the cache cannot be told to prefer L2 for a block still in HBM — the only
    /// way to force the bytes from L2 is to not keep the HBM copy around.
    pub fn set_no_prefix_cache(&mut self, on: bool) {
        if self.offload.is_some() {
            self.l1_retention_disabled = on;
        } else {
            self.prefix_cache_enabled = !on;
        }
    }

    /// Enable speculative decoding by loading a DFlash draft model into the
    /// primary lane.
    ///
    /// Requires the single-GPU topology (tensor parallel shards KV per rank) and
    /// is incompatible with KV offload. Disables the prefix cache: speculative
    /// capture needs clean, uncached target hidden states for every prompt
    /// token, and a prefix-cache hit skips the forward that would produce them.
    pub fn load_dflash_draft_model(&mut self, draft_path: &str) -> Result<()> {
        anyhow::ensure!(
            self.workers.is_empty(),
            "speculative decoding requires the single-GPU path (got {} extra ranks)",
            self.workers.len()
        );
        anyhow::ensure!(
            self.offload.is_none(),
            "speculative decoding is not supported together with KV offload"
        );
        let meta = self.primary.load_dflash(draft_path.to_string())?;
        log::info!(
            "Qwen3 DFlash speculative decoding enabled: draft block size {}",
            meta.block_size
        );
        self.prefix_cache_enabled = false;
        self.speculative = Some(meta);
        Ok(())
    }

    /// Whether KV offload is active on this executor.
    pub fn offload_enabled(&self) -> bool {
        self.offload.is_some()
    }

    /// Flush pending offload saves into the host read cache so a following
    /// query can see them. A persistence barrier for handoff and tests; no-op
    /// without offload.
    pub fn flush_offload_saves(&self) {
        if let Some(offload) = &self.offload {
            offload.flush_saves();
        }
    }

    /// Drop every cached-but-unused GPU prefix block. With offload on, this
    /// forces a cold prefix to be restored from the host tier on its next
    /// request (rather than served from HBM).
    pub fn evict_cached_blocks(&self) {
        self.kv_mgr.pool().evict_inactive();
    }

    /// Begin an async CPU-tier KV prefetch for `request_id`; see the
    /// [`ModelExecutor`] hook. Public so admission drivers and tests can park a
    /// request on its load. Returns `true` when a load is in flight.
    pub fn begin_kv_prefetch(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
        lora_adapter: Option<&str>,
        reserve_floor: usize,
    ) -> bool {
        <Self as ModelExecutor>::begin_kv_prefetch(
            self,
            request_id,
            prompt_tokens,
            lora_adapter,
            reserve_floor,
        )
    }

    /// Block until at least one in-flight prefetch settles, then sweep the
    /// rest; returns the settled request ids (now prefill-eligible).
    pub fn wait_ready_prefetch(&mut self) -> Vec<RequestId> {
        <Self as ModelExecutor>::wait_ready_prefetch(self)
    }

    // ── KV-offload SAVE ────────────────────────────────────────────────

    /// Save every block that sealed since this request's last save to the host
    /// tier (fire-and-forget). Safe to call right after `apply_prefill`/
    /// `apply_decode`: the producing step's token read-back has already
    /// synchronized the compute stream, so the sealed KV is fully written.
    fn save_sealed_blocks(&mut self, request_id: RequestId) {
        if self.offload.is_none() {
            return;
        }
        let Some(rkv) = self.request_kvs.get(&request_id) else {
            return;
        };
        // `assigned_block_hashes` lists only sealed (registered) blocks; the
        // partial tail block has no hash and never appears here.
        let assigned = rkv.assigned_block_hashes();
        let prefix_matched = rkv.prefix_matched_blocks();
        let cursor = self
            .saved_cursor
            .entry(request_id)
            .or_insert(prefix_matched);
        if assigned.len() <= *cursor {
            return;
        }
        let fresh = &assigned[*cursor..];
        let block_ids: Vec<i32> = fresh.iter().map(|(id, _)| *id).collect();
        let block_hashes: Vec<Vec<u8>> = fresh.iter().map(|(_, h)| h.to_vec()).collect();
        // Pin exactly the blocks being saved (aligned 1:1 with `assigned`) for
        // the duration of the async D2H, so a finished request can't hand the
        // slot to a new request that overwrites it before the copy lands.
        let pins: Vec<KvBlockGuard> = rkv
            .assigned_block_guards()
            .into_iter()
            .skip(*cursor)
            .collect();
        *cursor = assigned.len();
        self.offload
            .as_ref()
            .expect("offload present")
            .save(&block_ids, &block_hashes, pins);
    }

    // ── Chunked prefill ────────────────────────────────────────────────

    /// Prepare one prefill step for `req`: create its `RequestKv` on the
    /// first chunk (matching the prefix cache), then clamp the scheduler's
    /// chunk budget to the prompt tokens actually remaining and allocate KV
    /// for them. Sets `chunk_start`/`chunk_tokens` on the item.
    fn schedule_prefill_chunk(&mut self, req: &mut PrefillStepItem) -> Result<()> {
        if !self.request_kvs.contains_key(&req.request_id) {
            let mut rkv = self.kv_mgr.pool().new_request(
                req.prompt_tokens.clone(),
                req.max_output_tokens,
                req.lora_adapter.as_deref(),
            );
            // Echo needs logits for every prompt position; cached positions
            // are never forwarded, so echo requests prefill from scratch.
            if self.prefix_cache_enabled && !req.echo {
                req.cached_tokens = rkv.match_and_add_prefix(self.kv_mgr.pool())?;
            }
            self.request_kvs.insert(req.request_id, rkv);
            // match_and_add_prefix above already absorbed any CPU-prefetched
            // blocks (now held by the request's sequence), so release the
            // prefetch's separate hold.
            self.prefetch.remove(&req.request_id);
        }
        let rkv = self
            .request_kvs
            .get_mut(&req.request_id)
            .expect("inserted above");
        req.chunk_start = rkv.kv_position();
        let remaining = req.prompt_tokens.len() - req.chunk_start;
        // Echo must produce all-position logits in a single forward, so it is
        // exempt from chunking (the scheduler never splits echo requests).
        req.chunk_tokens = if req.echo {
            remaining
        } else {
            remaining.min(req.chunk_budget)
        };
        assert!(
            req.chunk_tokens > 0,
            "zero-token prefill chunk for {:?} (budget {})",
            req.request_id,
            req.chunk_budget
        );
        rkv.schedule_prefill(req.chunk_tokens, self.kv_mgr.pool())
            .map_err(|e| anyhow::anyhow!("schedule_prefill failed for {:?}: {e}", req.request_id))
    }

    /// Register a finished prefill step on the request's KV: the final chunk
    /// carries the first generated token, non-final chunks only advance the
    /// KV position.
    fn apply_prefill_result(&mut self, result: &PrefillRequestResult) -> Result<()> {
        let rkv = self
            .request_kvs
            .get_mut(&result.request_id)
            .expect("request must exist after prefill");
        if result.completed {
            rkv.apply_prefill(result.first_token, self.kv_mgr.pool())
        } else {
            rkv.apply_prefill_chunk(self.kv_mgr.pool())
        }
    }

    // ── KV-offload LOAD (async CPU-tier prefetch) ──────────────────────
    // The trait-facing prefetch hooks (`begin_kv_prefetch`,
    // `drain_ready_prefetch`, `wait_ready_prefetch`, `has_pending_prefetch`)
    // live in the `ModelExecutor` impl below; `settle_prefetch` is their shared
    // helper.

    /// Finalize one prefetch whose load returned `result`. On success the
    /// reserved blocks are staged + registered (held by the probe until the
    /// request prefills); on failure the state is dropped so the request
    /// prefills from scratch.
    fn settle_prefetch(
        &mut self,
        id: RequestId,
        result: Result<(), openinfer_kv_offload::EngineError>,
    ) {
        if let Some(st) = self.prefetch.get_mut(&id) {
            st.handle = None;
        }
        match result {
            Ok(()) => {
                let reservation = self
                    .prefetch
                    .get_mut(&id)
                    .and_then(|st| st.reservation.take())
                    .expect("reservation present until commit");
                let st = self.prefetch.get_mut(&id).expect("prefetch present");
                self.kv_mgr
                    .pool()
                    .commit_loaded_blocks(&mut st.probe, reservation);
            }
            Err(e) => {
                log::warn!("KV offload load failed for {id:?} (prefill from scratch): {e}");
                self.prefetch.remove(&id);
            }
        }
    }

    fn wait_for_step_ack(
        pending: Vec<channel::Receiver<Result<WorkerStepOutcome>>>,
        op_name: &'static str,
    ) -> Result<()> {
        for recv in pending {
            match recv
                .recv()
                .map_err(|_| anyhow::anyhow!("tensor-parallel {op_name} worker dropped"))??
            {
                WorkerStepOutcome::Ack => {}
                other => {
                    return Err(anyhow::anyhow!(
                        "tensor-parallel {op_name} worker returned unexpected payload: {}",
                        other.kind()
                    ));
                }
            }
        }
        Ok(())
    }

    fn run_step(&self, step: &StepCommand) -> Result<WorkerStepOutcome> {
        let primary = self.primary.run_step(step.clone(), true)?;
        let mut pending = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            pending.push(worker.run_step(step.clone(), false)?);
        }
        let primary_result = primary
            .recv()
            .map_err(|_| anyhow::anyhow!("primary worker dropped step response"))??;
        Self::wait_for_step_ack(pending, step.kind())?;
        Ok(primary_result)
    }
}

fn profile_kv_budget_on_worker(
    model: Qwen3Model,
    max_prefill_tokens: usize,
    dflash_kv_bytes_per_token: usize,
    memory_options: Qwen3MemoryOptions,
) -> Result<(Qwen3Model, KvBudget)> {
    let handle = thread::Builder::new()
        .name(format!(
            "qwen3-memory-profile-dev{}",
            model.device_ctx().device_ordinal
        ))
        .spawn(move || -> Result<(Qwen3Model, KvBudget)> {
            let _guard = {
                bind_model_thread(&model)?;
                tune_decode_gemm_algos(&model)?;
                CublasThreadGuard
            };
            let budget = model.profiled_kv_budget(
                max_prefill_tokens,
                *BATCH_BUCKETS.last().unwrap(),
                dflash_kv_bytes_per_token,
                memory_options,
            )?;
            Ok((model, budget))
        })
        .map_err(|e| anyhow::anyhow!("failed to spawn Qwen3 memory profile worker: {e}"))?;
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("Qwen3 memory profile worker panicked"))?
}

/// Build the KV-offload engine for the single-GPU path, or `None` when offload
/// is disabled. Registers the fused KV buffer with pegaflow against the model's
/// device/stream — must be called while that stream is still owned by the model
/// (before it moves into the `RankWorker`).
fn build_offload(
    opts: &Qwen3OffloadOptions,
    kv_mgr: &KvCacheManager,
    ctx: &DeviceContext,
) -> Result<Option<OffloadEngine>> {
    if !opts.enabled {
        return Ok(None);
    }
    let device_id = ctx.device_ordinal as i32;
    let config = OffloadConfig::new(
        format!("qwen3-4b-dev{device_id}"),
        device_id,
        opts.pinned_pool_bytes,
    );
    let engine = OffloadEngine::new(config, kv_mgr.buffer(), &ctx.stream)
        .map_err(|e| anyhow::anyhow!("KV offload engine init failed: {e}"))?;
    log::info!(
        "KV offload enabled on device {device_id} ({} MiB host tier)",
        opts.pinned_pool_bytes >> 20
    );
    Ok(Some(engine))
}

fn ensure_lora_capacity(
    loaded_lora_adapters: &HashSet<String>,
    lora_name: &str,
    max_loras: usize,
    load_inplace: bool,
) -> Result<()> {
    if loaded_lora_adapters.contains(lora_name) {
        anyhow::ensure!(
            load_inplace,
            "Qwen3 LoRA adapter {lora_name} is already loaded"
        );
        return Ok(());
    }
    anyhow::ensure!(
        loaded_lora_adapters.len() < max_loras,
        "Qwen3 LoRA adapter capacity exceeded: max_loras={}, loaded_adapters={}, requested={}",
        max_loras,
        loaded_lora_adapters.len(),
        lora_name
    );
    Ok(())
}

impl ModelExecutor for Qwen3Executor {
    fn block_size(&self) -> usize {
        self.metadata.block_size
    }

    fn max_request_blocks(&self) -> usize {
        self.kv_mgr.pool().max_request_blocks()
    }

    fn max_context_tokens(&self) -> usize {
        let target = self.metadata.config.max_position_embeddings;
        match &self.speculative {
            // The draft's fixed-width in-fill block writes `block_size` positions
            // past the committed length each step, so a request may use at most
            // `draft.max_pos - block_size` tokens before the draft cache would
            // overflow. Reject the rest at admission instead of crashing mid-prefill.
            Some(meta) => target.min(meta.max_position_embeddings.saturating_sub(meta.block_size)),
            None => target,
        }
    }

    fn max_decode_batch_size(&self) -> usize {
        *BATCH_BUCKETS.last().unwrap()
    }

    fn available_blocks(&self) -> usize {
        self.kv_mgr.pool().available_blocks()
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.metadata.stop_token_ids.contains(&token_id)
    }

    fn prefetched_blocks(&self, request_id: RequestId) -> usize {
        self.prefetch
            .get(&request_id)
            .map_or(0, |st| st.probe.held_blocks())
    }

    fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        // Remove and drop — RAII on SchedulableSequence's block guards
        // returns all allocated blocks regardless of lifecycle state. The same
        // RAII frees any parked prefetch's reserved/held blocks.
        self.request_kvs.remove(&request_id);
        // A parked prefetch may still have a load in flight: pegaflow's worker
        // is writing the reserved GPU blocks (H2D). Dropping the reservation now
        // frees those physical pages for immediate reuse while the DMA keeps
        // landing on them — silent KV corruption, the load-side mirror of the
        // SAVE keep-alive pin. Block until the copy finishes before the
        // reservation drops. The scheduler is a dedicated synchronous thread, so
        // this brief wait costs nothing it could spend elsewhere.
        if let Some(mut state) = self.prefetch.remove(&request_id) {
            if let Some(handle) = state.handle.take() {
                let _ = handle.wait();
            }
        }
        self.saved_cursor.remove(&request_id);
        if self.speculative.is_some() {
            self.dflash_ready_requests.remove(&request_id);
            self.primary.drop_dflash_request(request_id)?;
        }
        Ok(())
    }

    fn begin_kv_prefetch(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
        lora_adapter: Option<&str>,
        reserve_floor: usize,
    ) -> bool {
        let Some(offload) = self.offload.as_ref() else {
            return false;
        };
        if !self.prefix_cache_enabled {
            return false;
        }
        if self.l1_retention_disabled {
            // Pure-L2 mode: drop any cross-request HBM retention so the probe
            // sees gpu_hit == 0 and queries the whole cacheable prefix from the
            // host tier. Only inactive (completed, unheld) blocks are drained —
            // the current request holds nothing yet, and in-flight prefetches
            // keep their reserved blocks, so this never touches live KV.
            self.kv_mgr.pool().evict_inactive();
        }
        let probe = self
            .kv_mgr
            .pool()
            .probe_prefix(prompt_tokens.to_vec(), lora_adapter);
        let query_hashes = probe.cpu_query_hashes();
        if query_hashes.is_empty() {
            return false;
        }
        let hit = match offload.query(&request_id.0.to_string(), &query_hashes) {
            Ok(hit) => hit,
            Err(e) => {
                log::warn!("KV offload query failed for {request_id:?} (skipping): {e}");
                return false;
            }
        };
        let (Some(lease), num_blocks) = (hit.lease, hit.num_blocks) else {
            return false; // miss
        };
        // Blocks promised to admitted requests are off-limits: reserving into
        // them makes a later prefill chunk or decode growth fail allocation.
        if self
            .kv_mgr
            .pool()
            .available_blocks()
            .saturating_sub(reserve_floor)
            < num_blocks
        {
            offload.release_query_lease(lease);
            return false;
        }
        let Some(reservation) = self.kv_mgr.pool().reserve_loaded_blocks(num_blocks) else {
            // Block pressure: release the lease so its pinned host blocks aren't
            // held for the full lease TTL, and prefill from scratch rather than
            // stall.
            offload.release_query_lease(lease);
            return false;
        };
        let page_ids = reservation.page_ids();
        let handle = match offload.load(lease, page_ids) {
            Ok(handle) => handle,
            Err(e) => {
                log::warn!("KV offload load submit failed for {request_id:?} (skipping): {e}");
                // `load` consumes the lease only past its early validation; a
                // submit error may leave it pinned, so release it (no-op if it
                // was already consumed).
                offload.release_query_lease(lease);
                return false;
            }
        };
        self.prefetch.insert(
            request_id,
            PrefetchState {
                probe,
                reservation: Some(reservation),
                handle: Some(handle),
            },
        );
        true
    }

    fn drain_ready_prefetch(&mut self) -> Vec<RequestId> {
        let ids: Vec<RequestId> = self.prefetch.keys().copied().collect();
        let mut done = Vec::new();
        for id in ids {
            let poll = match self.prefetch.get_mut(&id).and_then(|st| st.handle.as_mut()) {
                Some(handle) => handle.poll(),
                None => continue, // already settled, awaiting prefill
            };
            if let Some(result) = poll {
                self.settle_prefetch(id, result);
                done.push(id);
            }
        }
        done
    }

    fn wait_ready_prefetch(&mut self) -> Vec<RequestId> {
        let mut done = Vec::new();
        if let Some(id) = self
            .prefetch
            .iter()
            .find(|(_, st)| st.handle.is_some())
            .map(|(id, _)| *id)
        {
            let handle = self
                .prefetch
                .get_mut(&id)
                .and_then(|st| st.handle.take())
                .expect("in-flight handle present");
            let result = handle.wait();
            self.settle_prefetch(id, result);
            // `settle_prefetch` clears the handle, so the drain below skips it;
            // record it here as the one we blocked on.
            done.push(id);
        }
        // Sweep any others that completed concurrently.
        for id in self.drain_ready_prefetch() {
            if !done.contains(&id) {
                done.push(id);
            }
        }
        done
    }

    fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        // 1. Create RequestKvs (first chunk only), clamp chunk budgets,
        // schedule KV for this step's tokens
        let mut requests = plan.requests.to_vec();
        for req in &mut requests {
            self.schedule_prefill_chunk(req)?;
        }

        // 2. Build KvViews (seq_len = chunk_start + this chunk)
        let kv_views: Vec<KvView> = requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].prefill_view(req.chunk_tokens))
            .collect();

        // 3. Execute forward
        let step = StepCommand::Prefill {
            requests,
            kv_views,
            echo: plan.echo,
            sample_seed: plan.sample_seed,
        };
        let outcome = self.run_step(&step)?;

        // 4. Apply prefill
        let result = match outcome {
            WorkerStepOutcome::Prefill(result) => result,
            other => {
                return Err(anyhow::anyhow!(
                    "prefill returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        for req_result in &result.requests {
            self.apply_prefill_result(req_result)?;
        }
        // A request becomes draft-ready once its prompt is fully prefilled with
        // captured target context. Partial chunks stay pending; ineligible
        // requests drop any stale worker state.
        if self.speculative.is_some() {
            for req_result in &result.requests {
                let captured = result
                    .dflash_context_captured_requests
                    .contains(&req_result.request_id);
                match dflash_prefill_action(captured, req_result.completed) {
                    DFlashPrefillAction::MarkReady => {
                        self.dflash_ready_requests.insert(req_result.request_id);
                    }
                    DFlashPrefillAction::KeepPending => {
                        self.dflash_ready_requests.remove(&req_result.request_id);
                    }
                    DFlashPrefillAction::Drop => {
                        self.dflash_ready_requests.remove(&req_result.request_id);
                        self.primary.drop_dflash_request(req_result.request_id)?;
                    }
                }
            }
        }
        // 5. Offload the blocks this prefill just sealed (post-step-sync).
        for req_result in &result.requests {
            self.save_sealed_blocks(req_result.request_id);
        }

        Ok(result)
    }

    fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        // 1. Schedule decode for all active requests
        for req in plan.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing RequestKv for {:?}", req.request_id))?;
            rkv.schedule_decode(self.kv_mgr.pool()).map_err(|e| {
                anyhow::anyhow!("schedule_decode failed for {:?}: {e}", req.request_id)
            })?;
        }

        // 2. Build KvViews
        let kv_views: Vec<KvView> = plan
            .requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].decode_view())
            .collect();

        // 3. Execute forward
        let step = StepCommand::Decode {
            requests: plan.requests.to_vec(),
            kv_views,
            sample_seed: plan.sample_seed,
        };
        let outcome = self.run_step(&step)?;

        // 4. Apply decode
        let result = match outcome {
            WorkerStepOutcome::Decode(result) => result,
            other => {
                return Err(anyhow::anyhow!(
                    "decode returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        for req_result in &result.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req_result.request_id)
                .expect("request must exist after decode");
            rkv.apply_decode(req_result.token, self.kv_mgr.pool())?;
        }
        // A plain decode advances the sequence outside the speculative path, so
        // any captured draft context is now stale — drop it.
        if self.speculative.is_some() {
            for req_result in &result.requests {
                if self.dflash_ready_requests.remove(&req_result.request_id) {
                    self.primary.drop_dflash_request(req_result.request_id)?;
                }
            }
        }
        // 5. Offload any block this decode step just sealed (post-step-sync).
        for req_result in &result.requests {
            self.save_sealed_blocks(req_result.request_id);
        }

        Ok(result)
    }

    fn execute_speculative_draft(&mut self, plan: DraftPlan<'_>) -> Result<DraftResult> {
        self.execute_speculative_draft_impl(plan)
    }

    fn execute_speculative_verify(&mut self, plan: VerifyPlan<'_>) -> Result<VerifyResult> {
        self.execute_speculative_verify_impl(plan)
    }

    fn speculative_enabled(&self) -> bool {
        self.speculative.is_some()
    }

    fn speculative_request_ready(&self, request_id: RequestId) -> bool {
        self.dflash_ready_requests.contains(&request_id)
    }

    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult> {
        // 1. Create RequestKvs for prefill requests (first chunk only), clamp
        // chunk budgets, schedule KV for this step's tokens
        let mut prefill_requests = plan.prefill_requests.to_vec();
        for req in &mut prefill_requests {
            self.schedule_prefill_chunk(req)?;
        }

        // Schedule decode for active requests
        for req in plan.decode_requests {
            let rkv = self
                .request_kvs
                .get_mut(&req.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing RequestKv for {:?}", req.request_id))?;
            rkv.schedule_decode(self.kv_mgr.pool()).map_err(|e| {
                anyhow::anyhow!("schedule_decode failed for {:?}: {e}", req.request_id)
            })?;
        }

        // 2. Build KvViews
        let prefill_kv_views: Vec<KvView> = prefill_requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].prefill_view(req.chunk_tokens))
            .collect();
        let decode_kv_views: Vec<KvView> = plan
            .decode_requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].decode_view())
            .collect();

        // 3. Execute forward — use split-concurrent if overlap streams are active
        let step = if let Some(ref overlap) = self.overlap {
            StepCommand::SplitConcurrent {
                prefill_requests,
                prefill_kv_views,
                decode_requests: plan.decode_requests.to_vec(),
                decode_kv_views,
                prefill_stream: overlap.prefill_stream,
                decode_stream: overlap.decode_stream,
                sample_seed: plan.sample_seed,
            }
        } else {
            StepCommand::Unified {
                prefill_requests,
                prefill_kv_views,
                decode_requests: plan.decode_requests.to_vec(),
                decode_kv_views,
                sample_seed: plan.sample_seed,
            }
        };
        let outcome = self.run_step(&step)?;

        // 4. Apply results
        match outcome {
            WorkerStepOutcome::Unified(result) => {
                // Normal unified path: both results ready
                for req_result in &result.prefill_requests {
                    self.apply_prefill_result(req_result)?;
                }
                for req_result in &result.decode_requests {
                    let rkv = self
                        .request_kvs
                        .get_mut(&req_result.request_id)
                        .expect("request must exist after unified decode");
                    rkv.apply_decode(req_result.token, self.kv_mgr.pool())?;
                }
                // A plain decode via the fused unified step advances the sequence
                // outside the speculative path, so any captured draft context is
                // now stale — drop it, mirroring execute_decode. (Eligible pending
                // are routed to a dedicated prefill step, so unified prefills never
                // need DFlash mark-ready here.)
                if self.speculative.is_some() {
                    for req_result in &result.decode_requests {
                        if self.dflash_ready_requests.remove(&req_result.request_id) {
                            self.primary.drop_dflash_request(req_result.request_id)?;
                        }
                    }
                }
                for req_result in &result.prefill_requests {
                    self.save_sealed_blocks(req_result.request_id);
                }
                for req_result in &result.decode_requests {
                    self.save_sealed_blocks(req_result.request_id);
                }
                Ok(result)
            }
            WorkerStepOutcome::SplitDecodeReady {
                decode: decode_result,
                prefill_event: SendEvent(event),
            } => {
                // SM-partition path: decode done, prefill still in-flight.
                // Apply decode immediately.
                for req_result in &decode_result.requests {
                    let rkv = self
                        .request_kvs
                        .get_mut(&req_result.request_id)
                        .expect("request must exist after split decode");
                    rkv.apply_decode(req_result.token, self.kv_mgr.pool())?;
                }
                for req_result in &decode_result.requests {
                    self.save_sealed_blocks(req_result.request_id);
                }
                // Store event for non-blocking poll by scheduler.
                // If a previous async prefill wasn't consumed, wait for it now.
                if let Some(old) = self.async_prefill.take() {
                    unsafe {
                        cudarc::driver::sys::cuEventSynchronize(old.event);
                    }
                    unsafe {
                        cudarc::driver::sys::cuEventDestroy_v2(old.event);
                    }
                    // Force resolve the worker's inflight prefill too
                    if let Ok(rx) = self.primary.resolve_prefill() {
                        if let Ok(Ok(result)) = rx.recv() {
                            for req_result in &result.requests {
                                let _ = self.apply_prefill_result(req_result);
                            }
                            for req_result in &result.requests {
                                self.save_sealed_blocks(req_result.request_id);
                            }
                        }
                    }
                }
                self.async_prefill = Some(AsyncPrefillState { event });
                // Return a UnifiedResult with empty prefill — scheduler will
                // get prefill results via poll_async_prefill.
                Ok(UnifiedResult {
                    prefill_requests: Vec::new(),
                    decode_requests: decode_result.requests,
                })
            }
            other => Err(anyhow::anyhow!(
                "unified returned unexpected: {}",
                other.kind()
            )),
        }
    }

    fn load_lora_adapter(&mut self, request: &LoadLoraAdapterRequest) -> Result<()> {
        ensure_lora_capacity(
            &self.loaded_lora_adapters,
            &request.lora_name,
            self.lora_options.max_loras,
            request.load_inplace,
        )?;
        let adapter = crate::lora::load_lora_adapter(
            &request.lora_path,
            &self.metadata.config,
            self.lora_options.max_lora_rank,
        )?;
        let world_size = self.workers.len() + 1;
        let projection_count: usize = adapter
            .layers
            .iter()
            .map(|layer| layer.projections.len())
            .sum();
        let element_count: usize = adapter
            .layers
            .iter()
            .flat_map(|layer| layer.projections.values())
            .map(|projection| projection.a.data.len() + projection.b.data.len())
            .sum();
        let shape_elems: usize = adapter
            .layers
            .iter()
            .flat_map(|layer| layer.projections.values())
            .map(|projection| {
                projection.a.rows * projection.a.cols + projection.b.rows * projection.b.cols
            })
            .sum();
        debug_assert_eq!(element_count, shape_elems);
        let rank = adapter.manifest.rank;
        let targets = adapter.manifest.target_modules.join(", ");
        let path = adapter.manifest.path.display().to_string();
        let mut sharded_adapters = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            sharded_adapters.push(adapter.shard_for_tensor_parallel(
                &self.metadata.config,
                TensorParallelConfig { rank, world_size },
            )?);
        }

        let mut sharded_adapters = sharded_adapters.into_iter();
        let primary_adapter = sharded_adapters
            .next()
            .expect("rank 0 adapter must exist for nonzero world_size");
        let primary_response = self.primary.load_lora_adapter(
            request.lora_name.clone(),
            primary_adapter,
            request.load_inplace,
        )?;
        let mut pending = Vec::with_capacity(self.workers.len());
        let mut errors = Vec::new();
        for (index, worker) in self.workers.iter().enumerate() {
            let rank = index + 1;
            let rank_adapter = sharded_adapters
                .next()
                .expect("worker adapter must exist for every tensor-parallel rank");
            match worker.load_lora_adapter(
                request.lora_name.clone(),
                rank_adapter,
                request.load_inplace,
            ) {
                Ok(response) => pending.push((rank, response)),
                Err(err) => errors.push(format!("rank {rank} dispatch: {err:#}")),
            }
        }

        match primary_response.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => errors.push(format!("rank 0: {err:#}")),
            Err(_) => errors.push("rank 0: dropped LoRA load response".to_string()),
        }
        for (rank, response) in pending {
            match response.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => errors.push(format!("rank {rank}: {err:#}")),
                Err(_) => errors.push(format!("rank {rank}: dropped LoRA load response")),
            }
        }
        if !errors.is_empty() {
            let mut cleanup_errors = Vec::new();
            match self.primary.discard_lora_adapter(request.lora_name.clone()) {
                Ok(response) => match response.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => cleanup_errors.push(format!("rank 0 cleanup: {err:#}")),
                    Err(_) => cleanup_errors
                        .push("rank 0 cleanup: dropped LoRA discard response".to_string()),
                },
                Err(err) => cleanup_errors.push(format!("rank 0 cleanup dispatch: {err:#}")),
            }
            for (index, worker) in self.workers.iter().enumerate() {
                let rank = index + 1;
                match worker.discard_lora_adapter(request.lora_name.clone()) {
                    Ok(response) => match response.recv() {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            cleanup_errors.push(format!("rank {rank} cleanup: {err:#}"));
                        }
                        Err(_) => cleanup_errors.push(format!(
                            "rank {rank} cleanup: dropped LoRA discard response"
                        )),
                    },
                    Err(err) => {
                        cleanup_errors.push(format!("rank {rank} cleanup dispatch: {err:#}"));
                    }
                }
            }
            if cleanup_errors.is_empty() {
                self.loaded_lora_adapters.remove(&request.lora_name);
            }
            let cleanup_suffix = if cleanup_errors.is_empty() {
                String::new()
            } else {
                format!("; cleanup errors: {}", cleanup_errors.join("; "))
            };
            anyhow::bail!(
                "failed to load Qwen3 LoRA adapter {} on tensor-parallel ranks: {}{}",
                request.lora_name,
                errors.join("; "),
                cleanup_suffix
            );
        }

        log::info!(
            "Loaded Qwen3 LoRA adapter {} from {} (rank={}, targets={}, projections={}, bf16_elements={}, tp_world_size={}, load_inplace={})",
            request.lora_name,
            path,
            rank,
            targets,
            projection_count,
            element_count,
            world_size,
            request.load_inplace
        );
        self.loaded_lora_adapters.insert(request.lora_name.clone());
        Ok(())
    }

    fn unload_lora_adapter(&mut self, request: &UnloadLoraAdapterRequest) -> Result<()> {
        let primary_response = self
            .primary
            .unload_lora_adapter(request.lora_name.clone())?;
        let mut pending = Vec::with_capacity(self.workers.len());
        for (index, worker) in self.workers.iter().enumerate() {
            pending.push((
                index + 1,
                worker.unload_lora_adapter(request.lora_name.clone())?,
            ));
        }

        let mut errors = Vec::new();
        match primary_response.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => errors.push(format!("rank 0: {err:#}")),
            Err(_) => errors.push("rank 0: dropped LoRA unload response".to_string()),
        }
        for (rank, response) in pending {
            match response.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => errors.push(format!("rank {rank}: {err:#}")),
                Err(_) => errors.push(format!("rank {rank}: dropped LoRA unload response")),
            }
        }
        if !errors.is_empty() {
            anyhow::bail!(
                "failed to unload Qwen3 LoRA adapter {} on tensor-parallel ranks: {}",
                request.lora_name,
                errors.join("; ")
            );
        }

        log::info!("Unloaded Qwen3 LoRA adapter {}", request.lora_name);
        self.loaded_lora_adapters.remove(&request.lora_name);
        Ok(())
    }

    fn list_lora_adapters(&self) -> Vec<String> {
        let mut names: Vec<_> = self.loaded_lora_adapters.iter().cloned().collect();
        names.sort();
        names
    }

    fn has_decode_overlap(&self) -> bool {
        self.overlap.is_some()
    }

    fn poll_async_prefill(&mut self) -> Option<PrefillResult> {
        let state = self.async_prefill.as_ref()?;
        // Non-blocking check: is the prefill stream done?
        let status = unsafe { cudarc::driver::sys::cuEventQuery(state.event) };
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            // Not ready yet (CUDA_ERROR_NOT_READY)
            return None;
        }
        // Prefill is done — resolve it via the worker.
        let event = self.async_prefill.take().unwrap().event;
        unsafe {
            cudarc::driver::sys::cuEventDestroy_v2(event);
        }

        // Ask worker to sync + sample the prefill result.
        let rx = match self.primary.resolve_prefill() {
            Ok(rx) => rx,
            Err(_) => return None,
        };
        let result = match rx.recv() {
            Ok(Ok(r)) => r,
            _ => return None,
        };

        // Apply prefill results (KV commit)
        for req_result in &result.requests {
            let _ = self.apply_prefill_result(req_result);
        }
        for req_result in &result.requests {
            self.save_sealed_blocks(req_result.request_id);
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::ensure_lora_capacity;
    use std::collections::HashSet;

    #[test]
    fn lora_capacity_rejects_new_adapter_at_limit() {
        let loaded = HashSet::from(["adapter-a".to_string()]);

        let error = ensure_lora_capacity(&loaded, "adapter-b", 1, false)
            .expect_err("new adapter should exceed capacity")
            .to_string();

        assert!(error.contains("max_loras=1"));
        assert!(error.contains("requested=adapter-b"));
    }

    #[test]
    fn lora_capacity_allows_existing_adapter_replacement_at_limit_with_load_inplace() {
        let loaded = HashSet::from(["adapter-a".to_string()]);

        ensure_lora_capacity(&loaded, "adapter-a", 1, true)
            .expect("existing adapter should fit with load_inplace");
    }

    #[test]
    fn lora_capacity_rejects_duplicate_without_load_inplace() {
        let loaded = HashSet::from(["adapter-a".to_string()]);

        let error = ensure_lora_capacity(&loaded, "adapter-a", 1, false)
            .expect_err("duplicate without load_inplace should fail")
            .to_string();

        assert!(error.contains("already loaded"));
    }
}

impl Drop for Qwen3Executor {
    fn drop(&mut self) {
        self.primary.shutdown();
        for worker in &mut self.workers {
            worker.shutdown();
        }
    }
}

/// What the executor learns about the draft model after loading it on the
/// worker: the draft block size (`K` candidates per round) and which target
/// layers feed the draft (the worker captures these; kept for diagnostics).
#[derive(Clone, Debug)]
struct DFlashMeta {
    block_size: usize,
    /// Draft's max cacheable position; with the `block_size` in-fill headroom
    /// this caps the DFlash-effective context to `max_position_embeddings - block_size`.
    max_position_embeddings: usize,
    #[allow(dead_code)]
    target_layer_ids: Vec<usize>,
}

struct LocalQwen3Lane {
    model: Qwen3Model,
    kv_buffer: KvBuffer,
    layout: KvLayout,
    bufs: BatchDecodeBuffers,
    sample_scratch: openinfer_sample::SampleScratch,
    /// In-flight prefill from a previous SplitConcurrent step (not yet synced).
    inflight_prefill: Option<InflightPrefillState>,
    /// DFlash draft lane (the draft model + per-request draft state). `None`
    /// unless speculative decoding is enabled; only the primary rank carries it.
    dflash: Option<DFlashLaneState>,
    /// Fixed, pre-allocated scratch for the DFlash verify forward. Lazily built
    /// on the first verify step (its shape depends on the loaded draft model's
    /// block size and the target's capture layers). Pointer-stable for the
    /// upcoming verify CUDA Graph.
    verify_bufs: Option<VerifyGraphBuffers>,
    /// KV pool block count — the worst-case page-list bound for `verify_bufs`.
    total_blocks: usize,
}

/// Stored state for an async prefill that was launched but not yet synced.
struct InflightPrefillState {
    prefill_stream: cudarc::driver::sys::CUstream,
    prefill_logits: HiddenStates,
    prefill_requests: Vec<PrefillStepItem>,
    /// Per-step sampling seed captured when the prefill was launched, replayed
    /// when its tokens are sampled after the deferred sync.
    sample_seed: u64,
}

// SAFETY: InflightPrefillState lives entirely within the worker thread that
// owns the GPU context. It is never shared across threads.
unsafe impl Send for InflightPrefillState {}

impl LocalQwen3Lane {
    fn new(
        model: Qwen3Model,
        kv_buffer: KvBuffer,
        total_blocks: usize,
        padding_block_id: i32,
    ) -> Result<Self> {
        let buf_layout = kv_buffer.layout();
        let layout = KvLayout::new(
            buf_layout.num_layers,
            buf_layout.num_kv_heads,
            buf_layout.head_dim,
            buf_layout.page_size,
        );
        let max_bucket = *BATCH_BUCKETS.last().unwrap();
        let bufs = BatchDecodeBuffers::new(
            model.device_ctx(),
            model.config().hidden_size,
            model.local_q_dim(),
            model.local_kv_dim(),
            model.local_intermediate_size(),
            model.config().vocab_size,
            max_bucket,
            total_blocks,
            padding_block_id,
            model.local_num_attention_heads(),
        )?;
        let sample_scratch = openinfer_sample::SampleScratch::new(
            model.device_ctx(),
            model.config().vocab_size,
            max_bucket,
        )?;
        Ok(Self {
            model,
            kv_buffer,
            layout,
            bufs,
            sample_scratch,
            inflight_prefill: None,
            dflash: None,
            verify_bufs: None,
            total_blocks,
        })
    }

    /// Load the DFlash draft model into this lane (primary rank only). The draft
    /// model is built here on the worker thread because it reads the co-located
    /// target model's embeddings and head.
    fn load_dflash(&mut self, draft_path: &str) -> Result<DFlashMeta> {
        let model = DFlashDraftModel::from_safetensors_for_target(
            self.model.device_ctx(),
            draft_path,
            &self.model,
        )?;
        model.tune_gemm_algos(&self.model)?;
        let meta = DFlashMeta {
            block_size: model.block_size(),
            max_position_embeddings: model.max_position_embeddings(),
            target_layer_ids: model.target_layer_ids().to_vec(),
        };
        let max_decode_batch_size = *BATCH_BUCKETS.last().unwrap();
        self.dflash = Some(DFlashLaneState::new(
            self.model.device_ctx(),
            model,
            max_decode_batch_size,
        )?);
        Ok(meta)
    }

    fn bind(&self) -> Result<CublasThreadGuard> {
        bind_model_thread(&self.model)?;
        tune_decode_gemm_algos(&self.model)?;
        Ok(CublasThreadGuard)
    }

    /// Sync the in-flight prefill stream and sample prefill tokens.
    /// Returns the prefill result. Panics if no inflight prefill exists.
    fn resolve_inflight_prefill(&mut self) -> Result<PrefillResult> {
        let state = self
            .inflight_prefill
            .take()
            .ok_or_else(|| anyhow::anyhow!("no inflight prefill to resolve"))?;

        // Sync prefill stream
        let r = unsafe { cudarc::driver::sys::cuStreamSynchronize(state.prefill_stream) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuStreamSynchronize(prefill) failed: {r:?}");
        }

        // Now safe to drop deferred GPU buffers (prefill kernels are done).
        crate::prefill::drain_deferred_drops();

        // Sample prefill tokens
        let params: Vec<&SamplingParams> =
            state.prefill_requests.iter().map(|r| &r.params).collect();
        let tokens = self.select_step_tokens(&state.prefill_logits, &params, state.sample_seed)?;

        // Build prefill result
        let results = build_prefill_request_results(
            self,
            &state.prefill_requests,
            &state.prefill_logits,
            &tokens,
            None,
            false,
        )?;

        // Split-concurrent prefill never runs with DFlash (capture needs the
        // synchronous result), so no context is captured here.
        Ok(PrefillResult {
            requests: results,
            dflash_context_captured_requests: Vec::new(),
        })
    }

    /// Pick one token per logits column (batched argmax for greedy rows,
    /// one batched sampler call for non-greedy rows). Grows the sampling
    /// scratch when a step is wider than the decode bucket it was sized for.
    fn select_step_tokens(
        &mut self,
        logits: &HiddenStates,
        params: &[&SamplingParams],
        sample_seed: u64,
    ) -> Result<Vec<u32>> {
        if params.len() > self.sample_scratch.max_rows() {
            self.sample_scratch = openinfer_sample::SampleScratch::new(
                self.model.device_ctx(),
                self.model.config().vocab_size,
                params.len(),
            )?;
        }
        openinfer_sample::select_batch(
            self.model.device_ctx(),
            logits,
            params,
            sample_seed,
            &mut self.sample_scratch,
        )
    }

    fn extract_logprobs(
        &self,
        logits: &DeviceVec,
        sampled_token: u32,
        top_k: usize,
    ) -> Result<TokenLogprob> {
        let logits_f32 = logits.to_host(self.model.device_ctx())?;
        openinfer_sample::token_logprob_from_row(&logits_f32, sampled_token, top_k)
            .ok_or_else(|| anyhow::anyhow!("logprobs computation failed"))
    }

    fn extract_prompt_logprobs(
        &self,
        all_logits: &HiddenStates,
        prev_pos: usize,
        target_token: u32,
        top_k: usize,
    ) -> Option<TokenLogprob> {
        openinfer_core::ops::extract_vec(self.model.device_ctx(), all_logits, prev_pos)
            .ok()
            .and_then(|logits_vec| {
                let logits_f32 = logits_vec.to_host(self.model.device_ctx()).ok()?;
                openinfer_sample::token_logprob_from_row(&logits_f32, target_token, top_k)
            })
    }

    fn execute_prefill(
        &mut self,
        prompts: &[&[u32]],
        kv_views: &[KvView],
        lora_adapters: &[Option<&str>],
        echo: bool,
        capture_layer_ids: Option<&[usize]>,
    ) -> Result<(HiddenStates, Option<HiddenStates>, Option<HiddenStates>)> {
        self.model.batch_prefill(
            prompts,
            kv_views,
            lora_adapters,
            self.kv_buffer.buffer(),
            &self.layout,
            echo,
            capture_layer_ids,
        )
    }

    /// DFlash verify forward over each request's `block_size`-token span, using
    /// the fixed pre-allocated [`VerifyGraphBuffers`] (no per-step allocation).
    /// Numerically equivalent to the `batch_prefill(echo=true)` verify path it
    /// replaces; the buffers are lazily built on first use.
    fn execute_dflash_verify(
        &mut self,
        requests: &[VerifyStepItem],
        kv_views: &[KvView],
    ) -> Result<VerifyResult> {
        let capture_layer_ids = self.dflash_capture_layer_ids().ok_or_else(|| {
            anyhow::anyhow!("DFlash verify requested but no draft model is loaded")
        })?;
        let block_size = self
            .dflash
            .as_ref()
            .expect("DFlash present when capture layers exist")
            .model
            .block_size();

        if self.verify_bufs.is_none() {
            let max_batch = *BATCH_BUCKETS.last().unwrap();
            self.verify_bufs = Some(VerifyGraphBuffers::new(
                &self.model,
                max_batch,
                block_size,
                capture_layer_ids.len(),
                self.total_blocks,
            )?);
        }

        // Take the buffers out of `self` so the forward (borrows `&self.model`,
        // `&mut bufs`) and the subsequent sampling (`&mut self.sample_scratch`)
        // and context record (`&mut self.dflash`) don't alias a `self` borrow.
        let mut bufs = self.verify_bufs.take().expect("verify buffers just set");
        let result = (|| -> Result<VerifyResult> {
            let spans: Vec<&[u32]> = requests.iter().map(VerifyStepItem::as_slice).collect();
            self.model.batch_prefill_into(
                &spans,
                kv_views,
                self.kv_buffer.buffer(),
                &self.layout,
                &capture_layer_ids,
                &mut bufs,
            )?;

            let total_tokens: usize = requests.iter().map(|req| req.as_slice().len()).sum();
            let greedy = SamplingParams::default();
            let params: Vec<&SamplingParams> = vec![&greedy; total_tokens];
            let target_tokens = self.select_step_tokens(bufs.all_logits(), &params, 0)?;
            let request_results = build_verify_results(requests, &target_tokens)?;
            self.record_verify_dflash_context(
                requests,
                &request_results,
                Some(bufs.captured_hidden()),
            )?;
            Ok(VerifyResult {
                requests: request_results,
            })
        })();
        self.verify_bufs = Some(bufs);
        result
    }

    fn execute_decode(
        &mut self,
        token_ids: &[u32],
        kv_views: &[KvView],
        lora_adapters: &[Option<&str>],
    ) -> Result<()> {
        self.model.batch_decode(
            token_ids,
            kv_views,
            lora_adapters,
            self.kv_buffer.buffer(),
            &self.layout,
            &mut self.bufs,
        )
    }

    fn execute_unified(
        &mut self,
        prefill_prompts: &[&[u32]],
        prefill_views: &[KvView],
        prefill_lora_adapters: &[Option<&str>],
        decode_tokens: &[u32],
        decode_views: &[KvView],
        decode_lora_adapters: &[Option<&str>],
    ) -> Result<HiddenStates> {
        self.model.unified_step(
            prefill_prompts,
            prefill_views,
            prefill_lora_adapters,
            decode_tokens,
            decode_views,
            decode_lora_adapters,
            self.kv_buffer.buffer(),
            &self.layout,
        )
    }

    fn load_lora_adapter(
        &mut self,
        name: String,
        adapter: crate::lora::LoraAdapter,
        load_inplace: bool,
    ) -> Result<()> {
        let device_adapter =
            crate::lora::load_device_lora_adapter(self.model.device_ctx(), name, adapter)?;
        self.model
            .install_lora_adapter(device_adapter, load_inplace)
    }

    fn unload_lora_adapter(&mut self, name: &str) -> Result<()> {
        self.model.uninstall_lora_adapter(name)
    }

    fn discard_lora_adapter(&mut self, name: &str) -> Result<()> {
        self.model.discard_lora_adapter(name)
    }
}

#[derive(Clone)]
enum StepCommand {
    Prefill {
        requests: Vec<PrefillStepItem>,
        kv_views: Vec<KvView>,
        echo: bool,
        sample_seed: u64,
    },
    Decode {
        requests: Vec<DecodeStepItem>,
        kv_views: Vec<KvView>,
        sample_seed: u64,
    },
    Unified {
        prefill_requests: Vec<PrefillStepItem>,
        prefill_kv_views: Vec<KvView>,
        decode_requests: Vec<DecodeStepItem>,
        decode_kv_views: Vec<KvView>,
        sample_seed: u64,
    },
    /// Split-concurrent: prefill and decode launch on separate Green Context
    /// streams (different SM partitions) for true GPU-level parallelism.
    SplitConcurrent {
        prefill_requests: Vec<PrefillStepItem>,
        prefill_kv_views: Vec<KvView>,
        decode_requests: Vec<DecodeStepItem>,
        decode_kv_views: Vec<KvView>,
        prefill_stream: crate::green_ctx::SendStream,
        decode_stream: crate::green_ctx::SendStream,
        sample_seed: u64,
    },
    /// Speculative verify: one target forward over each request's `K + 1` draft
    /// span (with a speculative KV view), capturing target hidden states for the
    /// next draft round. Greedy argmax per position drives [`accept_greedy`].
    SpeculativeVerify {
        requests: Vec<VerifyStepItem>,
        kv_views: Vec<KvView>,
    },
    /// Speculative draft: roll the DFlash draft model forward one block per
    /// request. Uses the draft's own KV — no target KV views.
    SpeculativeDraft { requests: Vec<DraftStepItem> },
}

impl StepCommand {
    fn kind(&self) -> &'static str {
        match self {
            Self::Prefill { .. } => "prefill",
            Self::Decode { .. } => "decode",
            Self::Unified { .. } => "unified",
            Self::SplitConcurrent { .. } => "split_concurrent",
            Self::SpeculativeVerify { .. } => "speculative_verify",
            Self::SpeculativeDraft { .. } => "speculative_draft",
        }
    }
}

enum WorkerCommand {
    RunStep {
        step: StepCommand,
        collect_result: bool,
        resp: channel::Sender<Result<WorkerStepOutcome>>,
    },
    LoadLoraAdapter {
        name: String,
        adapter: crate::lora::LoraAdapter,
        load_inplace: bool,
        resp: channel::Sender<Result<()>>,
    },
    UnloadLoraAdapter {
        name: String,
        resp: channel::Sender<Result<()>>,
    },
    DiscardLoraAdapter {
        name: String,
        resp: channel::Sender<Result<()>>,
    },
    /// Sync the in-flight prefill from a previous SplitConcurrent step and
    /// return the sampled prefill result.
    ResolvePrefill {
        resp: channel::Sender<Result<PrefillResult>>,
    },
    /// Load the DFlash draft model into the primary lane (built on the worker
    /// thread because it reads the co-located target model).
    LoadDflash {
        draft_path: String,
        resp: channel::Sender<Result<DFlashMeta>>,
    },
    /// Drop a request's DFlash draft state (request retired, or it fell back to
    /// a plain decode that advanced the sequence outside the speculative path).
    DropDflash {
        request_id: RequestId,
        resp: channel::Sender<Result<()>>,
    },
    Shutdown,
}

enum WorkerStepOutcome {
    Ack,
    Prefill(PrefillResult),
    Decode(DecodeResult),
    Unified(UnifiedResult),
    /// SM-partition split: decode result is ready; prefill is still in-flight
    /// on the prefill stream. The executor must call a follow-up to sync+sample
    /// prefill before using prefill scratch buffers again.
    SplitDecodeReady {
        decode: DecodeResult,
        /// Event recorded on prefill stream after all prefill kernels;
        /// query this to check if prefill is done without blocking.
        prefill_event: SendEvent,
    },
    SpeculativeVerify(VerifyResult),
    SpeculativeDraft(DraftResult),
}

impl WorkerStepOutcome {
    fn kind(&self) -> &'static str {
        match self {
            Self::Ack => "ack",
            Self::Prefill(_) => "prefill",
            Self::Decode(_) => "decode",
            Self::Unified(_) => "unified",
            Self::SplitDecodeReady { .. } => "split_decode_ready",
            Self::SpeculativeVerify(_) => "speculative_verify",
            Self::SpeculativeDraft(_) => "speculative_draft",
        }
    }
}

struct RankWorker {
    tx: channel::Sender<WorkerCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RankWorker {
    fn spawn(rank: usize, mut lane: LocalQwen3Lane) -> Result<Self> {
        let (tx, rx) = channel::unbounded();
        let (startup_tx, startup_rx) = channel::bounded(1);
        let handle = thread::Builder::new()
            .name(format!("qwen3-tp-rank-{rank}"))
            .spawn(move || {
                let startup = lane.bind();
                match startup {
                    Ok(_guard) => {
                        let _ = startup_tx.send(Ok(()));
                        while let Ok(cmd) = rx.recv() {
                            match cmd {
                                WorkerCommand::RunStep {
                                    step,
                                    collect_result,
                                    resp,
                                } => {
                                    let result =
                                        execute_step_on_lane(&mut lane, &step, collect_result);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::LoadLoraAdapter {
                                    name,
                                    adapter,
                                    load_inplace,
                                    resp,
                                } => {
                                    let result =
                                        lane.load_lora_adapter(name, adapter, load_inplace);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::UnloadLoraAdapter { name, resp } => {
                                    let result = lane.unload_lora_adapter(&name);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::DiscardLoraAdapter { name, resp } => {
                                    let result = lane.discard_lora_adapter(&name);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::ResolvePrefill { resp } => {
                                    let result = lane.resolve_inflight_prefill();
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::LoadDflash { draft_path, resp } => {
                                    let result = lane.load_dflash(&draft_path);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::DropDflash { request_id, resp } => {
                                    lane.drop_dflash_request(request_id);
                                    let _ = resp.send(Ok(()));
                                }
                                WorkerCommand::Shutdown => break,
                            }
                        }
                    }
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                    }
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn tensor-parallel worker {rank}: {e}"))?;
        startup_rx.recv().map_err(|_| {
            anyhow::anyhow!("tensor-parallel worker {rank} exited during startup")
        })??;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    fn run_step(
        &self,
        step: StepCommand,
        collect_result: bool,
    ) -> Result<channel::Receiver<Result<WorkerStepOutcome>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::RunStep {
                step,
                collect_result,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker step channel closed"))?;
        Ok(resp_rx)
    }

    /// Ask the worker to sync its in-flight prefill and return the result.
    fn resolve_prefill(&self) -> Result<channel::Receiver<Result<PrefillResult>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::ResolvePrefill { resp: resp_tx })
            .map_err(|_| anyhow::anyhow!("worker channel closed on resolve_prefill"))?;
        Ok(resp_rx)
    }

    /// Load the DFlash draft model into this worker's lane and return its
    /// metadata. Blocks until the worker finishes loading.
    fn load_dflash(&self, draft_path: String) -> Result<DFlashMeta> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::LoadDflash {
                draft_path,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("worker channel closed on load_dflash"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("worker dropped load_dflash response"))?
    }

    /// Drop a request's DFlash state. Blocks until the worker acknowledges.
    fn drop_dflash_request(&self, request_id: RequestId) -> Result<()> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::DropDflash {
                request_id,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("worker channel closed on drop_dflash"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("worker dropped drop_dflash response"))?
    }

    fn load_lora_adapter(
        &self,
        name: String,
        adapter: crate::lora::LoraAdapter,
        load_inplace: bool,
    ) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::LoadLoraAdapter {
                name,
                adapter,
                load_inplace,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker channel closed on LoRA load"))?;
        Ok(resp_rx)
    }

    fn unload_lora_adapter(&self, name: String) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::UnloadLoraAdapter {
                name,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker channel closed on LoRA unload"))?;
        Ok(resp_rx)
    }

    fn discard_lora_adapter(&self, name: String) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::DiscardLoraAdapter {
                name,
                resp: resp_tx,
            })
            .map_err(|_| {
                anyhow::anyhow!("tensor-parallel worker channel closed on LoRA discard")
            })?;
        Ok(resp_rx)
    }

    fn shutdown(&mut self) {
        let _ = self.tx.send(WorkerCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
