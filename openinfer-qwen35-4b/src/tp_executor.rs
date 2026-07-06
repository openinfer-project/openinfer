//! Tensor-parallel worker skeleton for Qwen3.5.
//!
//! This module intentionally stops at lifecycle and command plumbing. TP
//! prefill/decode math is added after the worker ownership model is verified.

use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use anyhow::Result;

use crate::batch_decode_graph::MAX_BATCH;
use crate::config::TensorParallelConfig;
use crate::executor::RequestId;
use crate::weights::{ModelRuntimeConfig, Qwen35Model};

#[allow(dead_code)]
#[derive(Debug)]
enum TpWorkerCommand {
    Ping {
        resp: mpsc::Sender<Result<TpWorkerAck>>,
    },
    RunPrefillStep {
        resp: mpsc::Sender<Result<TpWorkerAck>>,
    },
    RunDecodeStep {
        resp: mpsc::Sender<Result<TpWorkerAck>>,
    },
    RunUnifiedStep {
        resp: mpsc::Sender<Result<TpWorkerAck>>,
    },
    DropRequest {
        request_id: RequestId,
        resp: mpsc::Sender<Result<TpWorkerAck>>,
    },
    Shutdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TpWorkerAck {
    Ack,
}

/// TP executor scaffold. Rank 0 is the primary worker; other ranks currently
/// only acknowledge lifecycle commands.
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
        let mut workers = Vec::with_capacity(world_size);
        for (rank, &device_ordinal) in device_ordinals.iter().enumerate() {
            workers.push(TpWorker::spawn(
                rank,
                world_size,
                device_ordinal,
                model_path.to_string(),
                max_batch,
            )?);
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
                TpWorkerCommandKind::RunPrefillStep => {
                    TpWorkerCommand::RunPrefillStep { resp: resp_tx }
                }
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
        device_ordinal: usize,
        model_path: String,
        max_batch: usize,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        let (startup_tx, startup_rx) = mpsc::channel();
        let handle = thread::Builder::new()
            .name(format!("qwen35-tp-rank-{rank}"))
            .spawn(move || {
                let startup =
                    TpWorkerState::new(rank, world_size, device_ordinal, &model_path, max_batch);
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

        startup_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP worker {rank} exited during startup"))??;

        Ok(Self {
            tx,
            handle: Some(handle),
        })
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
    _max_batch: usize,
    _model: Qwen35Model,
}

impl TpWorkerState {
    fn new(
        rank: usize,
        world_size: usize,
        device_ordinal: usize,
        model_path: &str,
        max_batch: usize,
    ) -> Result<Self> {
        let model = Qwen35Model::from_safetensors_with_runtime(
            model_path,
            ModelRuntimeConfig {
                enable_cuda_graph: false,
                tensor_parallel: Some(TensorParallelConfig { rank, world_size }),
                device_ordinal,
            },
        )?;
        model.tune_decode_gemm_algos()?;
        Ok(Self {
            rank,
            _world_size: world_size,
            _max_batch: max_batch,
            _model: model,
        })
    }

    fn run(&mut self, rx: mpsc::Receiver<TpWorkerCommand>) {
        while let Ok(command) = rx.recv() {
            match command {
                TpWorkerCommand::Ping { resp } => {
                    let _ = resp.send(Ok(TpWorkerAck::Ack));
                }
                TpWorkerCommand::RunPrefillStep { resp }
                | TpWorkerCommand::RunDecodeStep { resp }
                | TpWorkerCommand::RunUnifiedStep { resp } => {
                    let _ = resp.send(Err(anyhow::anyhow!(
                        "Qwen3.5 TP worker rank {} has no TP forward implementation yet",
                        self.rank
                    )));
                }
                TpWorkerCommand::DropRequest { request_id, resp } => {
                    log::debug!(
                        "Qwen3.5 TP worker rank {} dropping request {:?}",
                        self.rank,
                        request_id
                    );
                    let _ = resp.send(Ok(TpWorkerAck::Ack));
                }
                TpWorkerCommand::Shutdown => break,
            }
        }
    }
}

fn wait_for_acks(
    pending: Vec<mpsc::Receiver<Result<TpWorkerAck>>>,
    op_name: &'static str,
) -> Result<()> {
    for recv in pending {
        match recv
            .recv()
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP {op_name} worker dropped"))??
        {
            TpWorkerAck::Ack => {}
        }
    }
    Ok(())
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
}
