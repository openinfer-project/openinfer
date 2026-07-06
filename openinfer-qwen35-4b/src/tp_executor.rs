//! Tensor-parallel worker runtime for Qwen3.5.
//!
//! Phase 1 starts with eager dense TP prefill. Decode/unified execution still
//! fails closed until the scheduler path can drive ordered worker commands.

use std::collections::HashSet;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use anyhow::Result;

use crate::batch_decode_graph::MAX_BATCH;
use crate::config::TensorParallelConfig;
use crate::executor::{
    PrefillPlan, PrefillRequestResult, PrefillResult, PrefillStepItem, RequestId,
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
    RunPrefillStep {
        requests: Vec<PrefillStepItem>,
        resp: mpsc::Sender<Result<TpWorkerReply>>,
    },
    RunDecodeStep {
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
}

/// TP executor. Rank 0 is the primary worker and returns scheduler-visible
/// artifacts; every rank runs the same ordered state-mutating commands.
pub struct Qwen35TpExecutor {
    workers: Vec<TpWorker>,
    world_size: usize,
    max_batch: usize,
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
        })
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub fn ping_all(&self) -> Result<()> {
        self.broadcast_ack(TpWorkerCommandKind::Ping)
    }

    pub fn execute_prefill(&self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 TP prefill plan requires at least one request"
        );
        let requests = plan.requests.to_vec();
        let mut pending = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            let (resp_tx, resp_rx) = mpsc::channel();
            worker.send(TpWorkerCommand::RunPrefillStep {
                requests: requests.clone(),
                resp: resp_tx,
            })?;
            pending.push(resp_rx);
        }
        wait_for_prefill(pending)
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
                TpWorkerCommandKind::RunPrefillStep => TpWorkerCommand::RunPrefillStep {
                    requests: Vec::new(),
                    resp: resp_tx,
                },
                TpWorkerCommandKind::RunDecodeStep => {
                    TpWorkerCommand::RunDecodeStep { resp: resp_tx }
                }
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
    RunPrefillStep,
    RunDecodeStep,
    RunUnifiedStep,
}

impl TpWorkerCommandKind {
    fn name(self) -> &'static str {
        match self {
            Self::Ping => "ping",
            Self::RunPrefillStep => "prefill step",
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
    active: Vec<TpActiveRequest>,
    sample_scratch: openinfer_sample::SampleScratch,
    _cublas_guard: CublasThreadGuard,
}

struct TpActiveRequest {
    request_id: RequestId,
    #[allow(dead_code)]
    kv: KvState,
    #[allow(dead_code)]
    recurrent: RecurrentState,
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
            active: Vec::new(),
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
                TpWorkerCommand::RunPrefillStep { requests, resp } => {
                    let result = self.execute_prefill(&requests);
                    let _ = resp.send(result);
                }
                TpWorkerCommand::RunDecodeStep { resp }
                | TpWorkerCommand::RunUnifiedStep { resp } => {
                    let _ = resp.send(Err(anyhow::anyhow!(
                        "Qwen3.5 TP worker rank {} has no TP decode/unified implementation yet",
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

    fn execute_prefill(&mut self, requests: &[PrefillStepItem]) -> Result<TpWorkerReply> {
        anyhow::ensure!(
            !requests.is_empty(),
            "Qwen3.5 TP prefill plan requires at least one request"
        );
        anyhow::ensure!(
            self.active.len() + requests.len() <= self.max_batch,
            "Qwen3.5 TP prefill would exceed worker capacity {}",
            self.max_batch
        );
        let mut seen = HashSet::with_capacity(requests.len());
        for req in requests {
            anyhow::ensure!(
                !req.prompt_tokens.is_empty(),
                "Qwen3.5 TP prefill request {} has an empty prompt",
                req.request_id.get()
            );
            anyhow::ensure!(
                seen.insert(req.request_id),
                "duplicate Qwen3.5 TP request id {} in prefill plan",
                req.request_id.get()
            );
            anyhow::ensure!(
                !self
                    .active
                    .iter()
                    .any(|active| active.request_id == req.request_id),
                "duplicate Qwen3.5 TP request id {}",
                req.request_id.get()
            );
        }

        let prompts: Vec<&[u32]> = requests
            .iter()
            .map(|req| req.prompt_tokens.as_slice())
            .collect();
        let mut kv_states: Vec<KvState> = requests.iter().map(|_| self.model.alloc_kv()).collect();
        let mut recurrent_states: Vec<RecurrentState> = requests
            .iter()
            .map(|_| RecurrentState::new(self.model.device_ctx(), self.model.config()))
            .collect::<Result<_>>()?;
        let mut recurrent_refs: Vec<&mut RecurrentState> = recurrent_states.iter_mut().collect();
        let logits =
            self.model
                .batch_prefill_logits(&prompts, &mut kv_states, &mut recurrent_refs)?;

        if self.rank == 0 {
            let requested_logprobs: Vec<usize> = requests.iter().map(|req| req.logprobs).collect();
            let cpu_logits =
                snapshot_requested_logprobs(self.model.device_ctx(), &logits, &requested_logprobs)?;
            let params = vec![SamplingParams::default(); requests.len()];
            let params_refs: Vec<&SamplingParams> = params.iter().collect();
            let tokens = openinfer_sample::select_batch(
                self.model.device_ctx(),
                &logits,
                &params_refs,
                0,
                &mut self.sample_scratch,
            )?;

            let mut results = Vec::with_capacity(requests.len());
            for (i, req) in requests.iter().enumerate() {
                let first_token = tokens[i];
                let first_token_logprob = cpu_logits[i].as_ref().and_then(|row| {
                    openinfer_sample::token_logprob_from_row(row, first_token, req.logprobs)
                });
                results.push(PrefillRequestResult {
                    request_id: req.request_id,
                    first_token,
                    first_token_logprob,
                });
            }
            self.install_active_requests(requests, kv_states, recurrent_states);
            Ok(TpWorkerReply::Prefill(PrefillResult { requests: results }))
        } else {
            self.install_active_requests(requests, kv_states, recurrent_states);
            Ok(TpWorkerReply::Ack)
        }
    }

    fn install_active_requests(
        &mut self,
        requests: &[PrefillStepItem],
        kv_states: Vec<KvState>,
        recurrent_states: Vec<RecurrentState>,
    ) {
        for ((req, kv), recurrent) in requests.iter().zip(kv_states).zip(recurrent_states) {
            self.active.push(TpActiveRequest {
                request_id: req.request_id,
                kv,
                recurrent,
            });
        }
    }

    fn drop_request(&mut self, request_id: RequestId) {
        if let Some(idx) = self
            .active
            .iter()
            .position(|active| active.request_id == request_id)
        {
            self.active.swap_remove(idx);
        }
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
        }
    }
    result.ok_or_else(|| anyhow::anyhow!("Qwen3.5 TP prefill returned no primary result"))
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
}
