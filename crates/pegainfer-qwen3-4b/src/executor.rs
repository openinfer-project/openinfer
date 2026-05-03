use std::collections::HashMap;
use std::thread;

use anyhow::Result;
use crossbeam_channel as channel;

use crate::batch_decode_buffers::{BATCH_BUCKETS, BatchDecodeBuffers};
use crate::config::TensorParallelConfig;
use crate::weights::{ModelRuntimeConfig, Qwen3Model};
use pegainfer_core::engine::TokenLogprob;
use pegainfer_core::kv_pool::{KvPool, KvState};
use pegainfer_core::ops;
use pegainfer_core::sampler::SamplingParams;
use pegainfer_core::tensor::{DeviceContext, DeviceVec, HiddenStates};

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
    pub(crate) params: SamplingParams,
    pub(crate) logprobs: usize,
    pub(crate) echo: bool,
    pub(crate) random_val: f32,
}

impl PrefillStepItem {
    pub fn new(
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        logprobs: usize,
        echo: bool,
        random_val: f32,
    ) -> Self {
        Self {
            request_id,
            prompt_tokens,
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

type RequestStateBatch = Vec<(RequestId, KvState)>;

struct RequestStateStore {
    states: HashMap<RequestId, KvState>,
}

impl RequestStateStore {
    fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    fn ensure_with<F>(&mut self, request_ids: &[RequestId], mut alloc: F)
    where
        F: FnMut() -> KvState,
    {
        for &request_id in request_ids {
            self.states.entry(request_id).or_insert_with(&mut alloc);
        }
    }

    fn drop_request(&mut self, request_id: RequestId) {
        self.states.remove(&request_id);
    }

    fn take_batch(
        &mut self,
        request_ids: &[RequestId],
        missing_context: &'static str,
    ) -> Result<RequestStateBatch> {
        request_ids
            .iter()
            .map(|request_id| {
                self.states
                    .remove(request_id)
                    .ok_or_else(|| anyhow::anyhow!("{missing_context} for {:?}", request_id))
                    .map(|kv| (*request_id, kv))
            })
            .collect()
    }

    fn restore_batch(&mut self, batch: RequestStateBatch) {
        for (request_id, kv_state) in batch {
            let replaced = self.states.insert(request_id, kv_state);
            debug_assert!(replaced.is_none(), "request state restored twice");
        }
    }
}

fn kv_state_refs(batch: &mut RequestStateBatch) -> Vec<&mut KvState> {
    batch.iter_mut().map(|(_, kv_state)| kv_state).collect()
}

fn execute_prefill_on_lane(
    lane: &mut LocalQwen3Lane,
    request_states: &mut RequestStateStore,
    requests: &[PrefillStepItem],
    echo: bool,
    missing_context: &'static str,
) -> Result<(Vec<DeviceVec>, Option<HiddenStates>)> {
    let request_ids: Vec<RequestId> = requests.iter().map(|req| req.request_id).collect();
    request_states.ensure_with(&request_ids, || lane.alloc_kv());
    let prompts: Vec<&[u32]> = requests.iter().map(PrefillStepItem::as_slice).collect();
    let mut request_state_batch = request_states.take_batch(&request_ids, missing_context)?;
    let mut kv_states = kv_state_refs(&mut request_state_batch);
    let result = lane.execute_prefill(&prompts, &mut kv_states, echo);
    request_states.restore_batch(request_state_batch);
    result
}

fn execute_decode_on_lane(
    lane: &mut LocalQwen3Lane,
    request_states: &mut RequestStateStore,
    requests: &[DecodeStepItem],
    missing_context: &'static str,
) -> Result<()> {
    let request_ids: Vec<RequestId> = requests.iter().map(|req| req.request_id).collect();
    let token_ids: Vec<u32> = requests.iter().map(|req| req.token_id).collect();
    let mut request_state_batch = request_states.take_batch(&request_ids, missing_context)?;
    let mut kv_states = kv_state_refs(&mut request_state_batch);
    let result = lane.execute_decode(&token_ids, &mut kv_states);
    request_states.restore_batch(request_state_batch);
    result
}

fn execute_unified_on_lane(
    lane: &mut LocalQwen3Lane,
    request_states: &mut RequestStateStore,
    prefill_requests: &[PrefillStepItem],
    decode_requests: &[DecodeStepItem],
    prefill_missing_context: &'static str,
    decode_missing_context: &'static str,
) -> Result<(Vec<DeviceVec>, Vec<DeviceVec>)> {
    let prefill_request_ids: Vec<RequestId> =
        prefill_requests.iter().map(|req| req.request_id).collect();
    let decode_request_ids: Vec<RequestId> =
        decode_requests.iter().map(|req| req.request_id).collect();
    request_states.ensure_with(&prefill_request_ids, || lane.alloc_kv());
    let prefill_prompts: Vec<&[u32]> = prefill_requests
        .iter()
        .map(PrefillStepItem::as_slice)
        .collect();
    let decode_tokens: Vec<u32> = decode_requests.iter().map(|req| req.token_id).collect();
    let mut prefill_request_state_batch =
        request_states.take_batch(&prefill_request_ids, prefill_missing_context)?;
    let mut decode_request_state_batch =
        request_states.take_batch(&decode_request_ids, decode_missing_context)?;
    let mut prefill_kv_states = kv_state_refs(&mut prefill_request_state_batch);
    let mut decode_kv_states = kv_state_refs(&mut decode_request_state_batch);
    let result = lane.execute_unified(
        &prefill_prompts,
        &mut prefill_kv_states,
        &decode_tokens,
        &mut decode_kv_states,
    );
    request_states.restore_batch(prefill_request_state_batch);
    request_states.restore_batch(decode_request_state_batch);
    result
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
    request_states: &mut RequestStateStore,
    step: &StepCommand,
    collect_result: bool,
) -> Result<WorkerStepOutcome> {
    match step {
        StepCommand::Prefill { requests, echo } => {
            let (logits, all_position_logits) = execute_prefill_on_lane(
                lane,
                request_states,
                requests,
                *echo,
                "missing local request state",
            )?;
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
        StepCommand::Decode { requests } => {
            execute_decode_on_lane(
                lane,
                request_states,
                requests,
                "missing local decode request state",
            )?;
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
            decode_requests,
        } => {
            let (prefill_logits, decode_logits) = execute_unified_on_lane(
                lane,
                request_states,
                prefill_requests,
                decode_requests,
                "missing local unified prefill request state",
                "missing local unified decode request state",
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
    fn page_size(&self) -> usize;
    fn available_pages(&self) -> usize;
    fn is_stop_token(&self, token_id: u32) -> bool;
    fn drop_request(&mut self, request_id: RequestId) -> Result<()>;

    fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult>;
    fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult>;
    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult>;
}

struct Qwen3ExecutorMetadata {
    page_size: usize,
    stop_token_ids: Vec<u32>,
}

pub struct Qwen3Executor {
    metadata: Qwen3ExecutorMetadata,
    kv_pools: Vec<KvPool>,
    primary: RankWorker,
    workers: Vec<RankWorker>,
}

impl Qwen3Executor {
    pub(crate) fn single(model: Qwen3Model) -> Result<Self> {
        let metadata = Qwen3ExecutorMetadata {
            page_size: model.kv_pool().layout().page_size,
            stop_token_ids: model.config().stop_token_ids.clone(),
        };
        let kv_pool = model.kv_pool().clone();
        Ok(Self {
            metadata,
            kv_pools: vec![kv_pool],
            primary: RankWorker::spawn(0, LocalQwen3Lane::new(model)?)?,
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

        let metadata = Qwen3ExecutorMetadata {
            page_size: models[0].kv_pool().layout().page_size,
            stop_token_ids: models[0].config().stop_token_ids.clone(),
        };

        let streams = models
            .iter()
            .map(|m| m.device_ctx().stream.clone())
            .collect();
        let comms = cudarc::nccl::safe::Comm::from_devices(streams)
            .map_err(|e| anyhow::anyhow!("failed to initialize NCCL comms: {e:?}"))?;
        for (model, comm) in models.iter_mut().zip(comms) {
            model.attach_tp_comm(comm);
        }

        let kv_pools = models.iter().map(|model| model.kv_pool().clone()).collect();
        let mut lanes = models
            .into_iter()
            .map(LocalQwen3Lane::new)
            .collect::<Result<Vec<_>>>()?;
        let primary = RankWorker::spawn(0, lanes.remove(0))?;
        let workers = lanes
            .into_iter()
            .enumerate()
            .map(|(index, lane)| RankWorker::spawn(index + 1, lane))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            metadata,
            kv_pools,
            primary,
            workers,
        })
    }

    pub fn page_size(&self) -> usize {
        <Self as ModelExecutor>::page_size(self)
    }

    pub fn available_pages(&self) -> usize {
        <Self as ModelExecutor>::available_pages(self)
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
    fn page_size(&self) -> usize {
        self.metadata.page_size
    }

    fn available_pages(&self) -> usize {
        self.kv_pools
            .iter()
            .map(KvPool::available_pages)
            .min()
            .unwrap_or(0)
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.metadata.stop_token_ids.contains(&token_id)
    }

    fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        self.primary.drop_request(request_id)?;
        for worker in &self.workers {
            worker.drop_request(request_id)?;
        }
        Ok(())
    }

    fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        let step = StepCommand::Prefill {
            requests: plan.requests.to_vec(),
            echo: plan.echo,
        };
        match self.run_step(&step)? {
            WorkerStepOutcome::Prefill(result) => Ok(result),
            other => Err(anyhow::anyhow!(
                "prefill step returned unexpected payload: {}",
                other.kind()
            )),
        }
    }

    fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        let step = StepCommand::Decode {
            requests: plan.requests.to_vec(),
        };
        match self.run_step(&step)? {
            WorkerStepOutcome::Decode(result) => Ok(result),
            other => Err(anyhow::anyhow!(
                "decode step returned unexpected payload: {}",
                other.kind()
            )),
        }
    }

    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult> {
        let step = StepCommand::Unified {
            prefill_requests: plan.prefill_requests.to_vec(),
            decode_requests: plan.decode_requests.to_vec(),
        };
        match self.run_step(&step)? {
            WorkerStepOutcome::Unified(result) => Ok(result),
            other => Err(anyhow::anyhow!(
                "unified step returned unexpected payload: {}",
                other.kind()
            )),
        }
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
    bufs: BatchDecodeBuffers,
    sample_scratch: SamplingScratch,
}

impl LocalQwen3Lane {
    fn new(model: Qwen3Model) -> Result<Self> {
        let max_bucket = *BATCH_BUCKETS.last().unwrap();
        let bufs = model.create_batch_decode_bufs(max_bucket)?;
        let sample_scratch = SamplingScratch::new(model.device_ctx(), model.config().vocab_size)?;
        Ok(Self {
            model,
            bufs,
            sample_scratch,
        })
    }

    fn bind(&self) -> Result<CublasThreadGuard> {
        bind_model_thread(&self.model)?;
        Ok(CublasThreadGuard)
    }

    fn alloc_kv(&self) -> KvState {
        self.model.alloc_kv()
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
        kv_states: &mut [&mut KvState],
        echo: bool,
    ) -> Result<(Vec<DeviceVec>, Option<HiddenStates>)> {
        self.model.batch_prefill(prompts, kv_states, echo)
    }

    fn execute_decode(&mut self, token_ids: &[u32], kv_states: &mut [&mut KvState]) -> Result<()> {
        self.model
            .batch_decode(token_ids, kv_states, &mut self.bufs)
    }

    fn execute_unified(
        &mut self,
        prefill_prompts: &[&[u32]],
        prefill_kv_states: &mut [&mut KvState],
        decode_tokens: &[u32],
        decode_kv_states: &mut [&mut KvState],
    ) -> Result<(Vec<DeviceVec>, Vec<DeviceVec>)> {
        self.model.unified_step(
            prefill_prompts,
            prefill_kv_states,
            decode_tokens,
            decode_kv_states,
        )
    }
}

#[derive(Clone)]
enum StepCommand {
    Prefill {
        requests: Vec<PrefillStepItem>,
        echo: bool,
    },
    Decode {
        requests: Vec<DecodeStepItem>,
    },
    Unified {
        prefill_requests: Vec<PrefillStepItem>,
        decode_requests: Vec<DecodeStepItem>,
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
    DropRequest {
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
                let mut request_states = RequestStateStore::new();
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
                                    let result = execute_step_on_lane(
                                        &mut lane,
                                        &mut request_states,
                                        &step,
                                        collect_result,
                                    );
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::DropRequest { request_id, resp } => {
                                    request_states.drop_request(request_id);
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

    fn drop_request(&self, request_id: RequestId) -> Result<()> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::DropRequest {
                request_id,
                resp: resp_tx,
            })
            .map_err(|_| {
                anyhow::anyhow!("tensor-parallel worker channel closed on drop_request")
            })?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker dropped drop_request response"))?
    }

    fn shutdown(&mut self) {
        let _ = self.tx.send(WorkerCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
