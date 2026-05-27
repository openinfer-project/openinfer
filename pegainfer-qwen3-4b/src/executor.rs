use std::collections::HashMap;
use std::thread;

use anyhow::Result;
use crossbeam_channel as channel;

use crate::batch_decode_buffers::{BATCH_BUCKETS, BatchDecodeBuffers};
use crate::config::{Config, TensorParallelConfig};
use crate::weights::{ModelRuntimeConfig, Qwen3Model};
use pegainfer_core::engine::{LoadLoraAdapterRequest, TokenLogprob};
use pegainfer_core::kv_pool::KvLayout;
use pegainfer_core::ops;
use pegainfer_core::sampler::SamplingParams;
use pegainfer_core::tensor::{DeviceContext, DeviceVec, HiddenStates};
use pegainfer_kv_cache::{KvBuffer, KvCacheManager, KvView};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
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
    pub(crate) random_val: f32,
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
        Self {
            request_id,
            prompt_tokens,
            max_output_tokens,
            params,
            logprobs,
            echo,
            random_val,
        }
    }

    fn as_slice(&self) -> &[u32] {
        &self.prompt_tokens
    }
}

#[derive(Clone, Copy)]
pub struct DecodeStepItem {
    pub(crate) request_id: RequestId,
    pub(crate) token_id: u32,
    pub(crate) params: SamplingParams,
    pub(crate) logprobs: usize,
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
            random_val,
        }
    }
}

fn build_prefill_request_results(
    lane: &mut LocalQwen3Lane,
    requests: &[PrefillStepItem],
    logits_vec: &[DeviceVec],
    all_position_logits: Option<&HiddenStates>,
    compute_prompt_logprobs: bool,
) -> Result<Vec<PrefillRequestResult>> {
    let mut token_offset = 0usize;
    let mut outputs = Vec::with_capacity(requests.len());
    for (i, req) in requests.iter().enumerate() {
        let first_token = lane.sample_from_logits(&logits_vec[i], &req.params, req.random_val)?;
        let first_token_logprob = if req.logprobs > 0 {
            Some(lane.extract_logprobs(&logits_vec[i], first_token, req.logprobs)?)
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
        token_offset += req.prompt_tokens.len();
        outputs.push(PrefillRequestResult {
            request_id: req.request_id,
            first_token,
            first_token_logprob,
            prompt_logprobs,
        });
    }
    Ok(outputs)
}

fn build_decode_request_results(
    lane: &mut LocalQwen3Lane,
    requests: &[DecodeStepItem],
    logits: &[DeviceVec],
) -> Result<Vec<DecodeRequestResult>> {
    let mut outputs = Vec::with_capacity(requests.len());
    for (i, req) in requests.iter().enumerate() {
        let token = lane.sample_from_logits(&logits[i], &req.params, req.random_val)?;
        let logprob = if req.logprobs > 0 {
            Some(lane.extract_logprobs(&logits[i], token, req.logprobs)?)
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
        } => {
            let prompts: Vec<&[u32]> = requests.iter().map(PrefillStepItem::as_slice).collect();
            let (logits, all_position_logits) = lane.execute_prefill(&prompts, kv_views, *echo)?;
            if collect_result {
                Ok(WorkerStepOutcome::Prefill(PrefillResult {
                    requests: build_prefill_request_results(
                        lane,
                        requests,
                        &logits,
                        all_position_logits.as_ref(),
                        *echo,
                    )?,
                }))
            } else {
                Ok(WorkerStepOutcome::Ack)
            }
        }
        StepCommand::Decode { requests, kv_views } => {
            let token_ids: Vec<u32> = requests.iter().map(|req| req.token_id).collect();
            lane.execute_decode(&token_ids, kv_views)?;
            if collect_result {
                let logits: Vec<DeviceVec> = (0..requests.len())
                    .map(|i| ops::extract_vec(lane.model.device_ctx(), &lane.bufs.logits, i))
                    .collect::<Result<Vec<_>>>()?;
                Ok(WorkerStepOutcome::Decode(DecodeResult {
                    requests: build_decode_request_results(lane, requests, &logits)?,
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
            let (prefill_logits, decode_logits) = lane.execute_unified(
                &prefill_prompts,
                prefill_kv_views,
                &decode_tokens,
                decode_kv_views,
            )?;
            if collect_result {
                Ok(WorkerStepOutcome::Unified(UnifiedResult {
                    prefill_requests: build_prefill_request_results(
                        lane,
                        prefill_requests,
                        &prefill_logits,
                        None,
                        false,
                    )?,
                    decode_requests: build_decode_request_results(
                        lane,
                        decode_requests,
                        &decode_logits,
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
            pegainfer_core::ffi::cublas_destroy();
        }
    }
}

struct SamplingScratch {
    probs: cudarc::driver::CudaSlice<f32>,
    top1_value: cudarc::driver::CudaSlice<half::bf16>,
    row_states: cudarc::driver::CudaSlice<u8>,
    valid: cudarc::driver::CudaSlice<u8>,
    out: cudarc::driver::CudaSlice<i32>,
}

impl SamplingScratch {
    fn new(ctx: &DeviceContext, vocab_size: usize) -> Result<Self> {
        Ok(Self {
            probs: ctx.stream.alloc_zeros(vocab_size)?,
            top1_value: ctx.stream.alloc_zeros(1)?,
            row_states: ctx
                .stream
                .alloc_zeros(pegainfer_core::ops::flashinfer_topk_row_states_bytes())?,
            valid: ctx.stream.alloc_zeros(1)?,
            out: ctx.stream.alloc_zeros(1)?,
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
        let err = pegainfer_core::ffi::cuda_set_device(model.device_ctx().device_ordinal as i32);
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
        pegainfer_core::ffi::cublas_init();
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

pub struct UnifiedResult {
    pub prefill_requests: Vec<PrefillRequestResult>,
    pub decode_requests: Vec<DecodeRequestResult>,
}

pub(crate) trait ModelExecutor: Send {
    fn block_size(&self) -> usize;
    fn max_request_blocks(&self) -> usize;
    fn available_blocks(&self) -> usize;
    fn is_stop_token(&self, token_id: u32) -> bool;
    fn drop_request(&mut self, request_id: RequestId) -> Result<()>;

    fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult>;
    fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult>;
    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult>;

    fn load_lora_adapter(&mut self, request: &LoadLoraAdapterRequest) -> Result<()> {
        anyhow::bail!(
            "Qwen3 LoRA adapter loading is not implemented yet: name={}, path={}",
            request.lora_name,
            request.lora_path.display()
        )
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
    request_kvs: HashMap<RequestId, pegainfer_kv_cache::RequestKv>,
    primary: RankWorker,
    workers: Vec<RankWorker>,
}

impl Qwen3Executor {
    pub(crate) fn single(model: Qwen3Model) -> Result<Self> {
        let budget = model.kv_budget();
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
        let total_blocks = kv_mgr.total_blocks();
        let padding_block_id = kv_mgr.padding_block_id();
        Ok(Self {
            metadata,
            kv_mgr,
            request_kvs: HashMap::new(),
            primary: RankWorker::spawn(
                0,
                LocalQwen3Lane::new(model, kv_buffer, total_blocks, padding_block_id)?,
            )?,
            workers: Vec::new(),
        })
    }

    pub fn from_runtime(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        anyhow::ensure!(
            !device_ordinals.is_empty(),
            "Qwen3 executor requires at least one device"
        );
        if device_ordinals.len() == 1 {
            let model = Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: None,
                    device_ordinal: device_ordinals[0],
                },
            )?;
            return Self::single(model);
        }

        let world_size = device_ordinals.len();
        let mut models = Vec::with_capacity(world_size);
        for (rank, &device_ordinal) in device_ordinals.iter().enumerate() {
            models.push(Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: Some(TensorParallelConfig { rank, world_size }),
                    device_ordinal,
                },
            )?);
        }

        // Compute budget from first model (all ranks share geometry).
        let budget = models[0].kv_budget();

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

        let total_blocks = kv_mgr.total_blocks();
        let padding_block_id = kv_mgr.padding_block_id();

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

impl ModelExecutor for Qwen3Executor {
    fn block_size(&self) -> usize {
        self.metadata.block_size
    }

    fn max_request_blocks(&self) -> usize {
        self.kv_mgr.max_request_blocks()
    }

    fn available_blocks(&self) -> usize {
        self.kv_mgr.available_blocks()
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.metadata.stop_token_ids.contains(&token_id)
    }

    fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        // Remove and drop — RAII on SchedulableSequence's block guards
        // returns all allocated blocks regardless of lifecycle state.
        self.request_kvs.remove(&request_id);
        Ok(())
    }

    fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        // 1. Create RequestKvs and schedule prefill
        for req in plan.requests {
            let mut rkv = self
                .kv_mgr
                .new_request(req.prompt_tokens.clone(), req.max_output_tokens);
            rkv.schedule_prefill(req.prompt_tokens.len(), &self.kv_mgr)
                .map_err(|e| {
                    anyhow::anyhow!("schedule_prefill failed for {:?}: {e}", req.request_id)
                })?;
            self.request_kvs.insert(req.request_id, rkv);
        }

        // 2. Build KvViews
        let kv_views: Vec<KvView> = plan
            .requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].prefill_view(req.prompt_tokens.len()))
            .collect();

        // 3. Execute forward
        let step = StepCommand::Prefill {
            requests: plan.requests.to_vec(),
            kv_views,
            echo: plan.echo,
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
            let rkv = self
                .request_kvs
                .get_mut(&req_result.request_id)
                .expect("request must exist after prefill");
            rkv.apply_prefill(req_result.first_token, &self.kv_mgr)?;
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
            rkv.schedule_decode(&self.kv_mgr).map_err(|e| {
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
            rkv.apply_decode(req_result.token, &self.kv_mgr)?;
        }

        Ok(result)
    }

    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult> {
        // 1. Create RequestKvs for prefill requests and schedule
        for req in plan.prefill_requests {
            let mut rkv = self
                .kv_mgr
                .new_request(req.prompt_tokens.clone(), req.max_output_tokens);
            rkv.schedule_prefill(req.prompt_tokens.len(), &self.kv_mgr)
                .map_err(|e| {
                    anyhow::anyhow!("schedule_prefill failed for {:?}: {e}", req.request_id)
                })?;
            self.request_kvs.insert(req.request_id, rkv);
        }

        // Schedule decode for active requests
        for req in plan.decode_requests {
            let rkv = self
                .request_kvs
                .get_mut(&req.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing RequestKv for {:?}", req.request_id))?;
            rkv.schedule_decode(&self.kv_mgr).map_err(|e| {
                anyhow::anyhow!("schedule_decode failed for {:?}: {e}", req.request_id)
            })?;
        }

        // 2. Build KvViews
        let prefill_kv_views: Vec<KvView> = plan
            .prefill_requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].prefill_view(req.prompt_tokens.len()))
            .collect();
        let decode_kv_views: Vec<KvView> = plan
            .decode_requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].decode_view())
            .collect();

        // 3. Execute forward
        let step = StepCommand::Unified {
            prefill_requests: plan.prefill_requests.to_vec(),
            prefill_kv_views,
            decode_requests: plan.decode_requests.to_vec(),
            decode_kv_views,
        };
        let outcome = self.run_step(&step)?;

        // 4. Apply both prefill and decode
        let result = match outcome {
            WorkerStepOutcome::Unified(result) => result,
            other => {
                return Err(anyhow::anyhow!(
                    "unified returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        for req_result in &result.prefill_requests {
            let rkv = self
                .request_kvs
                .get_mut(&req_result.request_id)
                .expect("request must exist after unified prefill");
            rkv.apply_prefill(req_result.first_token, &self.kv_mgr)?;
        }
        for req_result in &result.decode_requests {
            let rkv = self
                .request_kvs
                .get_mut(&req_result.request_id)
                .expect("request must exist after unified decode");
            rkv.apply_decode(req_result.token, &self.kv_mgr)?;
        }

        Ok(result)
    }

    fn load_lora_adapter(&mut self, request: &LoadLoraAdapterRequest) -> Result<()> {
        let adapter = crate::lora::load_lora_adapter(&request.lora_path, &self.metadata.config)?;
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
        let primary_response = self
            .primary
            .load_lora_adapter(request.lora_name.clone(), primary_adapter)?;
        let mut pending = Vec::with_capacity(self.workers.len());
        for (index, worker) in self.workers.iter().enumerate() {
            let rank = index + 1;
            let rank_adapter = sharded_adapters
                .next()
                .expect("worker adapter must exist for every tensor-parallel rank");
            pending.push((
                rank,
                worker.load_lora_adapter(request.lora_name.clone(), rank_adapter)?,
            ));
        }

        let mut errors = Vec::new();
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
            anyhow::bail!(
                "failed to load Qwen3 LoRA adapter {} on tensor-parallel ranks: {}",
                request.lora_name,
                errors.join("; ")
            );
        }

        log::info!(
            "Loaded Qwen3 LoRA adapter {} from {} (rank={}, targets={}, projections={}, bf16_elements={}, tp_world_size={})",
            request.lora_name,
            path,
            rank,
            targets,
            projection_count,
            element_count,
            world_size
        );
        Ok(())
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

struct LocalQwen3Lane {
    model: Qwen3Model,
    kv_buffer: KvBuffer,
    layout: KvLayout,
    bufs: BatchDecodeBuffers,
    sample_scratch: SamplingScratch,
}

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
        let sample_scratch = SamplingScratch::new(model.device_ctx(), model.config().vocab_size)?;
        Ok(Self {
            model,
            kv_buffer,
            layout,
            bufs,
            sample_scratch,
        })
    }

    fn bind(&self) -> Result<CublasThreadGuard> {
        bind_model_thread(&self.model)?;
        Ok(CublasThreadGuard)
    }

    fn sample_from_logits(
        &mut self,
        logits: &DeviceVec,
        params: &SamplingParams,
        random_val: f32,
    ) -> Result<u32> {
        pegainfer_core::ops::gpu_sample_into(
            self.model.device_ctx(),
            logits,
            &mut self.sample_scratch.probs,
            &mut self.sample_scratch.top1_value,
            &mut self.sample_scratch.row_states,
            &mut self.sample_scratch.valid,
            &mut self.sample_scratch.out,
            params,
            random_val,
        )
    }

    fn extract_logprobs(
        &self,
        logits: &DeviceVec,
        sampled_token: u32,
        top_k: usize,
    ) -> Result<TokenLogprob> {
        let logits_f32 = logits.to_host(self.model.device_ctx())?;
        compute_logprobs_from_cpu(&logits_f32, sampled_token, top_k)
            .ok_or_else(|| anyhow::anyhow!("logprobs computation failed"))
    }

    fn extract_prompt_logprobs(
        &self,
        all_logits: &HiddenStates,
        prev_pos: usize,
        target_token: u32,
        top_k: usize,
    ) -> Option<TokenLogprob> {
        pegainfer_core::ops::extract_vec(self.model.device_ctx(), all_logits, prev_pos)
            .ok()
            .and_then(|logits_vec| {
                let logits_f32 = logits_vec.to_host(self.model.device_ctx()).ok()?;
                compute_logprobs_from_cpu(&logits_f32, target_token, top_k)
            })
    }

    fn execute_prefill(
        &mut self,
        prompts: &[&[u32]],
        kv_views: &[KvView],
        echo: bool,
    ) -> Result<(Vec<DeviceVec>, Option<HiddenStates>)> {
        self.model.batch_prefill(
            prompts,
            kv_views,
            self.kv_buffer.buffer(),
            &self.layout,
            echo,
        )
    }

    fn execute_decode(&mut self, token_ids: &[u32], kv_views: &[KvView]) -> Result<()> {
        self.model.batch_decode(
            token_ids,
            kv_views,
            self.kv_buffer.buffer(),
            &self.layout,
            &mut self.bufs,
        )
    }

    fn execute_unified(
        &mut self,
        prefill_prompts: &[&[u32]],
        prefill_views: &[KvView],
        decode_tokens: &[u32],
        decode_views: &[KvView],
    ) -> Result<(Vec<DeviceVec>, Vec<DeviceVec>)> {
        self.model.unified_step(
            prefill_prompts,
            prefill_views,
            decode_tokens,
            decode_views,
            self.kv_buffer.buffer(),
            &self.layout,
        )
    }

    fn load_lora_adapter(&mut self, name: String, adapter: crate::lora::LoraAdapter) -> Result<()> {
        let device_adapter =
            crate::lora::load_device_lora_adapter(self.model.device_ctx(), name, adapter)?;
        self.model.set_lora_adapter(device_adapter);
        Ok(())
    }
}

#[derive(Clone)]
enum StepCommand {
    Prefill {
        requests: Vec<PrefillStepItem>,
        kv_views: Vec<KvView>,
        echo: bool,
    },
    Decode {
        requests: Vec<DecodeStepItem>,
        kv_views: Vec<KvView>,
    },
    Unified {
        prefill_requests: Vec<PrefillStepItem>,
        prefill_kv_views: Vec<KvView>,
        decode_requests: Vec<DecodeStepItem>,
        decode_kv_views: Vec<KvView>,
    },
}

impl StepCommand {
    fn kind(&self) -> &'static str {
        match self {
            Self::Prefill { .. } => "prefill",
            Self::Decode { .. } => "decode",
            Self::Unified { .. } => "unified",
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
        resp: channel::Sender<Result<()>>,
    },
    Shutdown,
}

enum WorkerStepOutcome {
    Ack,
    Prefill(PrefillResult),
    Decode(DecodeResult),
    Unified(UnifiedResult),
}

impl WorkerStepOutcome {
    fn kind(&self) -> &'static str {
        match self {
            Self::Ack => "ack",
            Self::Prefill(_) => "prefill",
            Self::Decode(_) => "decode",
            Self::Unified(_) => "unified",
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
                                    resp,
                                } => {
                                    let result = lane.load_lora_adapter(name, adapter);
                                    let _ = resp.send(result);
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

    fn load_lora_adapter(
        &self,
        name: String,
        adapter: crate::lora::LoraAdapter,
    ) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::LoadLoraAdapter {
                name,
                adapter,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker channel closed on LoRA load"))?;
        Ok(resp_rx)
    }

    fn shutdown(&mut self) {
        let _ = self.tx.send(WorkerCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
