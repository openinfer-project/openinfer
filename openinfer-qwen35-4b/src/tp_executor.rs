//! Tensor-parallel worker runtime for Qwen3.5.
//!
//! Phase 1 supports eager dense TP prefill and decode. Unified execution still
//! fails closed until the scheduler path can drive ordered eager decode.

use std::collections::HashSet;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use anyhow::Result;

use crate::batch_decode_graph::MAX_BATCH;
use crate::config::TensorParallelConfig;
use crate::decode_buffers::BatchDecodeBuffers35;
use crate::executor::{
    DecodePlan, DecodeRequestResult, DecodeResult, DecodeStepItem, PrefillPlan,
    PrefillRequestResult, PrefillResult, PrefillStepItem, RequestId,
};
use crate::logprobs::snapshot_requested_logprobs;
use crate::recurrent_state::RecurrentState;
use crate::weights::{ModelRuntimeConfig, Qwen35Model};
use openinfer_core::kv_pool::KvState;
use openinfer_core::sampler::SamplingParams;

#[allow(dead_code)]
enum TpWorkerCommand {
    Ping {
        resp: mpsc::Sender<Result<TpWorkerReply>>,
    },
    RunPrefillChunks {
        chunks: Vec<TpPrefillChunkItem>,
        sample_seed: u64,
        resp: mpsc::Sender<Result<TpWorkerReply>>,
    },
    RunDecodeStep {
        requests: Vec<TpDecodeStepItem>,
        sample_seed: u64,
        resp: mpsc::Sender<Result<TpWorkerReply>>,
    },
    RunUnifiedStep {
        resp: mpsc::Sender<Result<TpWorkerReply>>,
    },
    DropRequest {
        request_id: RequestId,
        resp: mpsc::Sender<Result<TpWorkerReply>>,
    },
    Shutdown,
}

#[derive(Debug)]
enum TpWorkerReply {
    Ack,
    Prefill(PrefillResult),
    Decode(DecodeResult),
}

/// TP executor. Rank 0 is the primary worker and returns scheduler-visible
/// artifacts; every rank runs the same ordered state-mutating commands.
pub struct Qwen35TpExecutor {
    workers: Vec<TpWorker>,
    world_size: usize,
    max_batch: usize,
    page_size: usize,
    capacity_pages_for_requests: usize,
    max_position_embeddings: usize,
    eos_token_id: u32,
}

#[derive(Clone)]
pub struct TpPrefillChunkItem {
    request_id: RequestId,
    prompt_tokens: Vec<u32>,
    logprobs: usize,
    sampling_params: SamplingParams,
    finish_prefill: bool,
}

impl TpPrefillChunkItem {
    pub fn new(
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
        logprobs: usize,
        finish_prefill: bool,
    ) -> Self {
        Self {
            request_id,
            prompt_tokens,
            logprobs,
            sampling_params: SamplingParams::default(),
            finish_prefill,
        }
    }

    pub fn new_with_sampling(
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
        logprobs: usize,
        sampling_params: SamplingParams,
        finish_prefill: bool,
    ) -> Self {
        Self {
            request_id,
            prompt_tokens,
            logprobs,
            sampling_params,
            finish_prefill,
        }
    }
}

#[derive(Clone)]
pub struct TpDecodeStepItem {
    request_id: RequestId,
    token_id: u32,
    logprobs: usize,
    sampling_params: SamplingParams,
}

impl TpDecodeStepItem {
    pub fn new(
        request_id: RequestId,
        token_id: u32,
        logprobs: usize,
        sampling_params: SamplingParams,
    ) -> Self {
        Self {
            request_id,
            token_id,
            logprobs,
            sampling_params,
        }
    }
}

impl Qwen35TpExecutor {
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
            device_ordinals.len() > 1,
            "Qwen3.5 TP executor requires at least two CUDA devices, got {}",
            device_ordinals.len()
        );
        anyhow::ensure!(
            !enable_cuda_graph,
            "Qwen3.5 TP Phase 1 supports eager execution only; disable CUDA Graph"
        );

        let world_size = device_ordinals.len();
        let mut models = Vec::with_capacity(world_size);
        for (rank, &device_ordinal) in device_ordinals.iter().enumerate() {
            models.push(Qwen35Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph: false,
                    tensor_parallel: Some(TensorParallelConfig { rank, world_size }),
                    device_ordinal,
                },
            )?);
        }
        let first = models
            .first()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 TP executor loaded no models"))?;
        let page_size = first.kv_pool().layout().page_size;
        let mut min_capacity_pages = usize::MAX;
        for (rank, model) in models.iter().enumerate() {
            let rank_page_size = model.kv_pool().layout().page_size;
            anyhow::ensure!(
                rank_page_size == page_size,
                "Qwen3.5 TP rank {rank} KV page size {rank_page_size} does not match rank 0 page size {page_size}"
            );
            min_capacity_pages = min_capacity_pages.min(model.kv_pool().capacity_pages());
        }
        let capacity_pages_for_requests = min_capacity_pages.saturating_sub(1);
        let max_position_embeddings = first.config().max_position_embeddings;
        let eos_token_id = first.config().eos_token_id;

        let nccl_id = cudarc::nccl::safe::Id::new()
            .map_err(|e| anyhow::anyhow!("failed to create Qwen3.5 TP NCCL id: {e:?}"))?;
        let mut workers = Vec::with_capacity(world_size);
        let mut startups = Vec::with_capacity(world_size);
        for (rank, model) in models.into_iter().enumerate() {
            let (worker, startup) = TpWorker::spawn(rank, world_size, model, max_batch, nccl_id)?;
            workers.push(worker);
            startups.push(startup);
        }
        for (rank, startup) in startups.into_iter().enumerate() {
            startup
                .recv()
                .map_err(|_| anyhow::anyhow!("Qwen3.5 TP worker {rank} exited during startup"))??;
        }

        Ok(Self {
            workers,
            world_size,
            max_batch,
            page_size,
            capacity_pages_for_requests,
            max_position_embeddings,
            eos_token_id,
        })
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn capacity_pages_for_requests(&self) -> usize {
        self.capacity_pages_for_requests
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    pub fn is_stop_token(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id
    }

    pub fn ping_all(&self) -> Result<()> {
        self.broadcast_ack(TpWorkerCommandKind::Ping)
    }

    pub fn execute_prefill(&self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 TP prefill plan requires at least one request"
        );
        let chunks: Vec<TpPrefillChunkItem> = plan
            .requests
            .iter()
            .cloned()
            .map(TpPrefillChunkItem::from)
            .collect();
        self.execute_prefill_chunks(&chunks)
    }

    pub fn execute_prefill_chunks(&self, chunks: &[TpPrefillChunkItem]) -> Result<PrefillResult> {
        self.execute_prefill_chunks_with_seed(chunks, 0)
    }

    pub fn execute_prefill_chunks_with_seed(
        &self,
        chunks: &[TpPrefillChunkItem],
        sample_seed: u64,
    ) -> Result<PrefillResult> {
        anyhow::ensure!(
            !chunks.is_empty(),
            "Qwen3.5 TP prefill chunk command requires at least one chunk"
        );
        let chunks = chunks.to_vec();
        let mut pending = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            let (resp_tx, resp_rx) = mpsc::channel();
            worker.send(TpWorkerCommand::RunPrefillChunks {
                chunks: chunks.clone(),
                sample_seed,
                resp: resp_tx,
            })?;
            pending.push(resp_rx);
        }
        wait_for_prefill(pending)
    }

    pub fn execute_decode(&self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 TP decode plan requires at least one request"
        );
        let requests: Vec<TpDecodeStepItem> = plan
            .requests
            .iter()
            .map(|request| {
                TpDecodeStepItem::new(
                    request.request_id,
                    request.token_id,
                    request.logprobs,
                    SamplingParams::default(),
                )
            })
            .collect();
        self.execute_decode_items(&requests, 0)
    }

    pub fn execute_decode_items(
        &self,
        requests: &[TpDecodeStepItem],
        sample_seed: u64,
    ) -> Result<DecodeResult> {
        anyhow::ensure!(
            !requests.is_empty(),
            "Qwen3.5 TP decode plan requires at least one request"
        );
        let requests = requests.to_vec();
        let mut pending = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            let (resp_tx, resp_rx) = mpsc::channel();
            worker.send(TpWorkerCommand::RunDecodeStep {
                requests: requests.clone(),
                sample_seed,
                resp: resp_tx,
            })?;
            pending.push(resp_rx);
        }
        wait_for_decode(pending)
    }

    pub fn drop_request(&self, request_id: RequestId) -> Result<()> {
        let mut pending = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            let (resp_tx, resp_rx) = mpsc::channel();
            worker.send(TpWorkerCommand::DropRequest {
                request_id,
                resp: resp_tx,
            })?;
            pending.push(resp_rx);
        }
        wait_for_acks(pending, "drop request")
    }

    fn broadcast_ack(&self, kind: TpWorkerCommandKind) -> Result<()> {
        let mut pending = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            let (resp_tx, resp_rx) = mpsc::channel();
            let command = match kind {
                TpWorkerCommandKind::Ping => TpWorkerCommand::Ping { resp: resp_tx },
                TpWorkerCommandKind::RunPrefillChunks => TpWorkerCommand::RunPrefillChunks {
                    chunks: Vec::new(),
                    sample_seed: 0,
                    resp: resp_tx,
                },
                TpWorkerCommandKind::RunDecodeStep => TpWorkerCommand::RunDecodeStep {
                    requests: Vec::new(),
                    sample_seed: 0,
                    resp: resp_tx,
                },
                TpWorkerCommandKind::RunUnifiedStep => {
                    TpWorkerCommand::RunUnifiedStep { resp: resp_tx }
                }
            };
            worker.send(command)?;
            pending.push(resp_rx);
        }
        wait_for_acks(pending, kind.name())
    }
}

impl Drop for Qwen35TpExecutor {
    fn drop(&mut self) {
        for worker in &self.workers {
            let _ = worker.tx.send(TpWorkerCommand::Shutdown);
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum TpWorkerCommandKind {
    Ping,
    RunPrefillChunks,
    RunDecodeStep,
    RunUnifiedStep,
}

impl TpWorkerCommandKind {
    fn name(self) -> &'static str {
        match self {
            Self::Ping => "ping",
            Self::RunPrefillChunks => "prefill chunks",
            Self::RunDecodeStep => "decode step",
            Self::RunUnifiedStep => "unified step",
        }
    }
}

struct TpWorker {
    tx: mpsc::Sender<TpWorkerCommand>,
    handle: Option<JoinHandle<()>>,
}

impl TpWorker {
    fn spawn(
        rank: usize,
        world_size: usize,
        model: Qwen35Model,
        max_batch: usize,
        nccl_id: cudarc::nccl::safe::Id,
    ) -> Result<(Self, mpsc::Receiver<Result<()>>)> {
        let (tx, rx) = mpsc::channel();
        let (startup_tx, startup_rx) = mpsc::channel();
        let handle = thread::Builder::new()
            .name(format!("qwen35-tp-rank-{rank}"))
            .spawn(move || {
                let startup = TpWorkerState::new(rank, world_size, model, max_batch, nccl_id);
                match startup {
                    Ok(mut state) => {
                        let _ = startup_tx.send(Ok(()));
                        state.run(rx);
                    }
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                    }
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn Qwen3.5 TP worker {rank}: {e}"))?;

        Ok((
            Self {
                tx,
                handle: Some(handle),
            },
            startup_rx,
        ))
    }

    fn send(&self, command: TpWorkerCommand) -> Result<()> {
        self.tx
            .send(command)
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP worker channel closed"))
    }
}

impl Drop for TpWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(TpWorkerCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct TpWorkerState {
    rank: usize,
    _world_size: usize,
    max_batch: usize,
    model: Qwen35Model,
    requests: Vec<TpRequestState>,
    decode_buffers: BatchDecodeBuffers35,
    sample_scratch: openinfer_sample::SampleScratch,
    _cublas_guard: CublasThreadGuard,
}

struct TpRequestState {
    request_id: RequestId,
    phase: TpRequestPhase,
    kv: KvState,
    recurrent: RecurrentState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TpRequestPhase {
    Prefilling,
    Decoding,
}

impl TpWorkerState {
    fn new(
        rank: usize,
        world_size: usize,
        mut model: Qwen35Model,
        max_batch: usize,
        nccl_id: cudarc::nccl::safe::Id,
    ) -> Result<Self> {
        let cublas_guard = bind_worker_thread(&model)?;
        let comm = cudarc::nccl::safe::Comm::from_rank(
            model.device_ctx().stream.clone(),
            rank,
            world_size,
            nccl_id,
        )
        .map_err(|e| anyhow::anyhow!("failed to initialize Qwen3.5 TP NCCL rank {rank}: {e:?}"))?;
        model.attach_tp_comm(comm);
        let decode_buffers = model.create_batch_decode_buffers_with_capacity(max_batch)?;
        let sample_scratch = openinfer_sample::SampleScratch::new(
            model.device_ctx(),
            model.config().vocab_size,
            max_batch,
        )?;
        Ok(Self {
            rank,
            _world_size: world_size,
            max_batch,
            model,
            requests: Vec::new(),
            decode_buffers,
            sample_scratch,
            _cublas_guard: cublas_guard,
        })
    }

    fn run(&mut self, rx: mpsc::Receiver<TpWorkerCommand>) {
        while let Ok(command) = rx.recv() {
            match command {
                TpWorkerCommand::Ping { resp } => {
                    let _ = resp.send(Ok(TpWorkerReply::Ack));
                }
                TpWorkerCommand::RunPrefillChunks {
                    chunks,
                    sample_seed,
                    resp,
                } => {
                    let result = self.execute_prefill_chunks(&chunks, sample_seed);
                    let _ = resp.send(result);
                }
                TpWorkerCommand::RunDecodeStep {
                    requests,
                    sample_seed,
                    resp,
                } => {
                    let result = self.execute_decode(&requests, sample_seed);
                    let _ = resp.send(result);
                }
                TpWorkerCommand::RunUnifiedStep { resp } => {
                    let _ = resp.send(Err(anyhow::anyhow!(
                        "Qwen3.5 TP worker rank {} has no TP unified implementation yet",
                        self.rank
                    )));
                }
                TpWorkerCommand::DropRequest { request_id, resp } => {
                    self.drop_request(request_id);
                    let _ = resp.send(Ok(TpWorkerReply::Ack));
                }
                TpWorkerCommand::Shutdown => break,
            }
        }
    }

    fn execute_prefill_chunks(
        &mut self,
        chunks: &[TpPrefillChunkItem],
        sample_seed: u64,
    ) -> Result<TpWorkerReply> {
        anyhow::ensure!(
            !chunks.is_empty(),
            "Qwen3.5 TP prefill chunk command requires at least one chunk"
        );
        validate_prefill_chunks(chunks)?;
        let new_requests = chunks
            .iter()
            .filter(|chunk| self.request_index(chunk.request_id).is_none())
            .count();
        anyhow::ensure!(
            self.requests.len() + new_requests <= self.max_batch,
            "Qwen3.5 TP prefill chunks would exceed worker capacity {}",
            self.max_batch
        );

        let mut primary_results = Vec::new();
        let mut final_row_idx = 0usize;
        for chunk in chunks {
            let state_idx = self.ensure_prefill_state(chunk.request_id)?;
            let state = &mut self.requests[state_idx];
            anyhow::ensure!(
                state.phase == TpRequestPhase::Prefilling,
                "Qwen3.5 TP request {} is already in decode state",
                chunk.request_id.get()
            );

            let prompt = [chunk.prompt_tokens.as_slice()];
            let mut recurrent_refs = vec![&mut state.recurrent];
            let logits = self.model.batch_prefill_logits(
                &prompt,
                std::slice::from_mut(&mut state.kv),
                &mut recurrent_refs,
            )?;

            if chunk.finish_prefill {
                if self.rank == 0 {
                    // TP prefill samples final chunks one row at a time. Offset
                    // by the final-row index so rows from the same command do
                    // not reuse the same sampling stream.
                    let row_seed = sample_seed.wrapping_add(final_row_idx as u64);
                    let result = self.sample_final_prefill_chunk(chunk, &logits, row_seed)?;
                    primary_results.push(result);
                }
                final_row_idx += 1;
                self.requests[state_idx].phase = TpRequestPhase::Decoding;
            }
        }

        if self.rank == 0 {
            Ok(TpWorkerReply::Prefill(PrefillResult {
                requests: primary_results,
            }))
        } else {
            Ok(TpWorkerReply::Ack)
        }
    }

    fn sample_final_prefill_chunk(
        &mut self,
        chunk: &TpPrefillChunkItem,
        logits: &openinfer_core::tensor::HiddenStates,
        sample_seed: u64,
    ) -> Result<PrefillRequestResult> {
        let cpu_logits =
            snapshot_requested_logprobs(self.model.device_ctx(), logits, &[chunk.logprobs])?;
        let params_refs = [&chunk.sampling_params];
        let tokens = openinfer_sample::select_batch(
            self.model.device_ctx(),
            logits,
            &params_refs,
            &[0],
            sample_seed,
            &mut self.sample_scratch,
        )?;
        let first_token = tokens[0];
        let first_token_logprob = cpu_logits[0].as_ref().and_then(|row| {
            openinfer_sample::token_logprob_from_row(row, first_token, chunk.logprobs)
        });
        Ok(PrefillRequestResult {
            request_id: chunk.request_id,
            first_token,
            first_token_logprob,
        })
    }

    fn execute_decode(
        &mut self,
        requests: &[TpDecodeStepItem],
        sample_seed: u64,
    ) -> Result<TpWorkerReply> {
        anyhow::ensure!(
            !requests.is_empty(),
            "Qwen3.5 TP decode command requires at least one request"
        );
        validate_decode_requests(requests)?;
        anyhow::ensure!(
            requests.len() <= self.max_batch,
            "Qwen3.5 TP decode batch {} exceeds worker capacity {}",
            requests.len(),
            self.max_batch
        );

        let mut primary_results =
            Vec::with_capacity(if self.rank == 0 { requests.len() } else { 0 });
        for (row_idx, request) in requests.iter().enumerate() {
            let state_idx = self.request_index(request.request_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.5 TP decode request {} has no worker state",
                    request.request_id.get()
                )
            })?;
            anyhow::ensure!(
                self.requests[state_idx].phase == TpRequestPhase::Decoding,
                "Qwen3.5 TP request {} is not ready for decode",
                request.request_id.get()
            );

            {
                let state = &mut self.requests[state_idx];
                let mut kv_refs = vec![&mut state.kv];
                let mut recurrent_refs = vec![&mut state.recurrent];
                self.model.batch_decode_eager_logits(
                    &[request.token_id],
                    &mut kv_refs,
                    &mut recurrent_refs,
                    &mut self.decode_buffers,
                )?;
            }

            if self.rank == 0 {
                let cpu_logits = snapshot_requested_logprobs(
                    self.model.device_ctx(),
                    &self.decode_buffers.logits,
                    &[request.logprobs],
                )?;
                let params_refs = [&request.sampling_params];
                let tokens = openinfer_sample::select_batch(
                    self.model.device_ctx(),
                    &self.decode_buffers.logits,
                    &params_refs,
                    &[0],
                    sample_seed.wrapping_add(row_idx as u64),
                    &mut self.sample_scratch,
                )?;
                let token = tokens[0];
                let logprob = cpu_logits[0].as_ref().and_then(|row| {
                    openinfer_sample::token_logprob_from_row(row, token, request.logprobs)
                });
                primary_results.push(DecodeRequestResult {
                    request_id: request.request_id,
                    token,
                    logprob,
                });
            }
        }

        if self.rank == 0 {
            Ok(TpWorkerReply::Decode(DecodeResult {
                requests: primary_results,
            }))
        } else {
            Ok(TpWorkerReply::Ack)
        }
    }

    fn ensure_prefill_state(&mut self, request_id: RequestId) -> Result<usize> {
        if let Some(idx) = self.request_index(request_id) {
            return Ok(idx);
        }
        let state = TpRequestState {
            request_id,
            phase: TpRequestPhase::Prefilling,
            kv: self.model.alloc_kv(),
            recurrent: RecurrentState::new(self.model.device_ctx(), self.model.config())?,
        };
        self.requests.push(state);
        Ok(self.requests.len() - 1)
    }

    fn request_index(&self, request_id: RequestId) -> Option<usize> {
        self.requests
            .iter()
            .position(|state| state.request_id == request_id)
    }

    fn drop_request(&mut self, request_id: RequestId) {
        if let Some(idx) = self.request_index(request_id) {
            self.requests.swap_remove(idx);
        }
    }
}

fn validate_prefill_chunks(chunks: &[TpPrefillChunkItem]) -> Result<()> {
    let mut seen = HashSet::with_capacity(chunks.len());
    for chunk in chunks {
        anyhow::ensure!(
            !chunk.prompt_tokens.is_empty(),
            "Qwen3.5 TP prefill chunk for request {} is empty",
            chunk.request_id.get()
        );
        anyhow::ensure!(
            seen.insert(chunk.request_id),
            "duplicate Qwen3.5 TP request id {} in one prefill chunk command",
            chunk.request_id.get()
        );
    }
    Ok(())
}

fn validate_decode_requests(requests: &[TpDecodeStepItem]) -> Result<()> {
    let mut seen = HashSet::with_capacity(requests.len());
    for request in requests {
        anyhow::ensure!(
            seen.insert(request.request_id),
            "duplicate Qwen3.5 TP request id {} in one decode command",
            request.request_id.get()
        );
    }
    Ok(())
}

impl From<PrefillStepItem> for TpPrefillChunkItem {
    fn from(request: PrefillStepItem) -> Self {
        Self::new(
            request.request_id,
            request.prompt_tokens,
            request.logprobs,
            true,
        )
    }
}

impl From<DecodeStepItem> for TpDecodeStepItem {
    fn from(request: DecodeStepItem) -> Self {
        Self::new(
            request.request_id,
            request.token_id,
            request.logprobs,
            SamplingParams::default(),
        )
    }
}

fn wait_for_acks(
    pending: Vec<mpsc::Receiver<Result<TpWorkerReply>>>,
    op_name: &'static str,
) -> Result<()> {
    for recv in pending {
        match recv
            .recv()
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP {op_name} worker dropped"))??
        {
            TpWorkerReply::Ack => {}
            TpWorkerReply::Prefill(_) => {
                anyhow::bail!("Qwen3.5 TP {op_name} unexpectedly returned prefill result")
            }
            TpWorkerReply::Decode(_) => {
                anyhow::bail!("Qwen3.5 TP {op_name} unexpectedly returned decode result")
            }
        }
    }
    Ok(())
}

fn wait_for_prefill(pending: Vec<mpsc::Receiver<Result<TpWorkerReply>>>) -> Result<PrefillResult> {
    let mut result = None;
    for (rank, recv) in pending.into_iter().enumerate() {
        match recv
            .recv()
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP prefill worker {rank} dropped"))??
        {
            TpWorkerReply::Ack => {}
            TpWorkerReply::Prefill(prefill) => {
                anyhow::ensure!(
                    result.is_none(),
                    "Qwen3.5 TP prefill returned multiple primary results"
                );
                result = Some(prefill);
            }
            TpWorkerReply::Decode(_) => {
                anyhow::bail!("Qwen3.5 TP prefill unexpectedly returned decode result")
            }
        }
    }
    result.ok_or_else(|| anyhow::anyhow!("Qwen3.5 TP prefill returned no primary result"))
}

fn wait_for_decode(pending: Vec<mpsc::Receiver<Result<TpWorkerReply>>>) -> Result<DecodeResult> {
    let mut result = None;
    for (rank, recv) in pending.into_iter().enumerate() {
        match recv
            .recv()
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP decode worker {rank} dropped"))??
        {
            TpWorkerReply::Ack => {}
            TpWorkerReply::Decode(decode) => {
                anyhow::ensure!(
                    result.is_none(),
                    "Qwen3.5 TP decode returned multiple primary results"
                );
                result = Some(decode);
            }
            TpWorkerReply::Prefill(_) => {
                anyhow::bail!("Qwen3.5 TP decode unexpectedly returned prefill result")
            }
        }
    }
    result.ok_or_else(|| anyhow::anyhow!("Qwen3.5 TP decode returned no primary result"))
}

struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            crate::ffi::cublas_destroy();
        }
    }
}

fn bind_worker_thread(model: &Qwen35Model) -> Result<CublasThreadGuard> {
    let ctx = model.device_ctx();
    unsafe {
        let err = crate::ffi::cuda_set_device(ctx.device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on Qwen3.5 TP worker thread: cudaError={}",
                ctx.device_ordinal,
                err
            ));
        }
    }
    ctx.ctx.bind_to_thread().map_err(|e| {
        anyhow::anyhow!("Failed to bind CUDA context to Qwen3.5 TP worker thread: {e}")
    })?;
    unsafe {
        crate::ffi::cublas_init();
    }
    Ok(CublasThreadGuard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_single_device_topology() {
        let err = match Qwen35TpExecutor::from_runtime_with_capacity("unused", false, &[0], 1) {
            Ok(_) => panic!("single-device TP topology should fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires at least two CUDA devices"));
    }

    #[test]
    fn rejects_tensor_parallel_cuda_graph() {
        let err = match Qwen35TpExecutor::from_runtime_with_capacity("unused", true, &[0, 1], 1) {
            Ok(_) => panic!("TP CUDA Graph should fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("eager execution only"));
    }

    #[test]
    fn validates_prefill_chunk_shape() {
        let empty = [TpPrefillChunkItem::new(RequestId::new(1), vec![], 0, false)];
        let err = validate_prefill_chunks(&empty).unwrap_err().to_string();
        assert!(err.contains("is empty"));

        let duplicate = [
            TpPrefillChunkItem::new(RequestId::new(1), vec![151_646], 0, false),
            TpPrefillChunkItem::new(RequestId::new(1), vec![9707], 0, true),
        ];
        let err = validate_prefill_chunks(&duplicate).unwrap_err().to_string();
        assert!(err.contains("duplicate"));
    }

    #[test]
    fn validates_decode_request_shape() {
        validate_decode_requests(&[TpDecodeStepItem::new(
            RequestId::new(1),
            9707,
            0,
            SamplingParams::default(),
        )])
        .expect("single decode request is valid");

        let duplicate = [
            TpDecodeStepItem::new(RequestId::new(1), 9707, 0, SamplingParams::default()),
            TpDecodeStepItem::new(RequestId::new(1), 560, 0, SamplingParams::default()),
        ];
        let err = validate_decode_requests(&duplicate)
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate"));
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn starts_tp2_workers_and_broadcasts_lifecycle_commands() {
        let model_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        assert_eq!(executor.world_size(), 2);
        assert_eq!(executor.max_batch(), 1);
        executor.ping_all().expect("ping all workers");
        executor
            .drop_request(RequestId::new(7))
            .expect("drop request");
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_prefill_runs_and_returns_primary_result() {
        let model_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        let request_id = RequestId::new(11);
        let request = PrefillStepItem::new(request_id, vec![151_646, 9707], 0);
        let result = executor
            .execute_prefill(PrefillPlan {
                requests: &[request],
            })
            .expect("run TP2 prefill");
        assert_eq!(result.requests.len(), 1);
        assert_eq!(result.requests[0].request_id, request_id);
        executor
            .drop_request(request_id)
            .expect("drop prefetched request");
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_chunked_prefill_advances_existing_request_state() {
        let model_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        let request_id = RequestId::new(13);
        let first = TpPrefillChunkItem::new(request_id, vec![151_646], 0, false);
        let first_result = executor
            .execute_prefill_chunks(&[first])
            .expect("run non-final TP2 prefill chunk");
        assert!(first_result.requests.is_empty());

        let final_chunk = TpPrefillChunkItem::new(request_id, vec![9707], 0, true);
        let final_result = executor
            .execute_prefill_chunks(&[final_chunk])
            .expect("run final TP2 prefill chunk");
        assert_eq!(final_result.requests.len(), 1);
        assert_eq!(final_result.requests[0].request_id, request_id);

        executor
            .drop_request(request_id)
            .expect("drop chunk-prefilled request");
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_decode_runs_after_prefill() {
        let model_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        let request_id = RequestId::new(17);
        let request = PrefillStepItem::new(request_id, vec![151_646, 9707], 0);
        let prefill = executor
            .execute_prefill(PrefillPlan {
                requests: &[request],
            })
            .expect("run TP2 prefill");
        assert_eq!(prefill.requests.len(), 1);
        assert_eq!(prefill.requests[0].request_id, request_id);

        let decode_request = DecodeStepItem::new(request_id, prefill.requests[0].first_token, 0);
        let decode = executor
            .execute_decode(DecodePlan {
                requests: &[decode_request],
            })
            .expect("run TP2 eager decode");
        assert_eq!(decode.requests.len(), 1);
        assert_eq!(decode.requests[0].request_id, request_id);

        executor
            .drop_request(request_id)
            .expect("drop decoded request");
    }
}
