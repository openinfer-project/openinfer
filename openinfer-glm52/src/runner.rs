use std::{
    path::{Path, PathBuf},
    thread,
};

use anyhow::{Result, ensure};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use openinfer_core::engine::{GenerateRequest, TokenEvent, unix_now_s};
use tokio::sync::mpsc;

use crate::weights::{
    Glm52NonExpertWeightContractReport, Glm52RankExpertFp8Weights, Glm52RankGpuContext,
    Glm52RankGpuWeights, Glm52RankLoadBundle, load_rank_sliced_weights_to_gpu,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankPlacement {
    pub(crate) rank: usize,
    pub(crate) device_ordinal: usize,
}

impl Glm52RankPlacement {
    pub(crate) fn new(rank: usize, device_ordinal: usize) -> Result<Self> {
        ensure!(rank < 8, "GLM5.2 rank must be < 8, got {rank}");
        Ok(Self {
            rank,
            device_ordinal,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankWeightLoadReport {
    pub(crate) rank: usize,
    pub(crate) tensor_count: usize,
    pub(crate) total_bytes: usize,
    pub(crate) non_expert_weight_contract: Glm52NonExpertWeightContractReport,
    pub(crate) loaded_to_gpu: bool,
}

/// One command per GPU-owning stage thread. The PP8 forward (graph replay,
/// stage handoff) will add its own variants; today the thread only loads its
/// sliced weights and shuts down — the coordinator still rejects every request.
enum Glm52RankCommand {
    LoadSlicedWeights {
        model_path: PathBuf,
        resp: Sender<Result<Glm52RankWeightLoadReport>>,
    },
    Shutdown,
}

pub(crate) struct Glm52RankWorker {
    tx: Sender<Glm52RankCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Glm52RankWorker {
    pub(crate) fn spawn(
        placement: Glm52RankPlacement,
        bundle: Glm52RankLoadBundle,
    ) -> Result<Self> {
        ensure!(
            bundle.load_plan.rank == placement.rank,
            "GLM5.2 rank load plan {} does not match placement {}",
            bundle.load_plan.rank,
            placement.rank
        );
        let (tx, rx) = unbounded();
        let (startup_tx, startup_rx) = bounded::<Result<()>>(1);
        let handle = thread::Builder::new()
            .name(format!("glm52-rank-{}", placement.rank))
            .spawn(move || {
                let ctx = match Glm52RankGpuContext::new(placement.device_ordinal) {
                    Ok(ctx) => ctx,
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                        return;
                    }
                };
                let _ = startup_tx.send(Ok(()));
                rank_worker_loop(rx, Glm52RankThreadState::new(placement, ctx, bundle));
            })
            .map_err(|err| anyhow::anyhow!("failed to spawn GLM5.2 rank worker: {err}"))?;
        startup_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker exited during startup"))??;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    pub(crate) fn load_sliced_weights_async(
        &self,
        model_path: &Path,
    ) -> Result<Receiver<Result<Glm52RankWeightLoadReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::LoadSlicedWeights {
                model_path: model_path.to_path_buf(),
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn shutdown(&mut self) -> Result<()> {
        self.request_shutdown()?;
        self.join()
    }

    fn request_shutdown(&self) -> Result<()> {
        self.tx
            .send(Glm52RankCommand::Shutdown)
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(())
    }

    fn join(&mut self) -> Result<()> {
        if self.handle.is_none() {
            return Ok(());
        }
        let handle = self.handle.take().expect("GLM5.2 rank handle must exist");
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker panicked"))?;
        Ok(())
    }
}

impl Drop for Glm52RankWorker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

struct Glm52RankThreadState {
    placement: Glm52RankPlacement,
    ctx: Glm52RankGpuContext,
    bundle: Glm52RankLoadBundle,
    loaded: Option<Glm52RankLoadedState>,
}

struct Glm52RankLoadedState {
    weights: Glm52RankGpuWeights,
    expert_weights: Glm52RankExpertFp8Weights,
}

impl Glm52RankLoadedState {
    fn total_bytes(&self) -> usize {
        self.weights.total_bytes + self.expert_weights.total_bytes
    }
}

impl Glm52RankThreadState {
    fn new(
        placement: Glm52RankPlacement,
        ctx: Glm52RankGpuContext,
        bundle: Glm52RankLoadBundle,
    ) -> Self {
        Self {
            placement,
            ctx,
            bundle,
            loaded: None,
        }
    }

    fn load_sliced_weights(&mut self, model_path: &Path) -> Result<Glm52RankWeightLoadReport> {
        let loaded = load_rank_sliced_weights_to_gpu(&self.ctx, model_path, &self.bundle)?;
        ensure!(
            loaded.loaded_total_bytes
                == loaded.weights.total_bytes + loaded.expert_kernel_weights.total_bytes,
            "GLM5.2 rank {} loaded bytes {} differ from resident raw {} + expert package {}",
            self.placement.rank,
            loaded.loaded_total_bytes,
            loaded.weights.total_bytes,
            loaded.expert_kernel_weights.total_bytes
        );
        let loaded_state = Glm52RankLoadedState {
            weights: loaded.weights,
            expert_weights: loaded.expert_kernel_weights,
        };
        let total_bytes = loaded_state.total_bytes();
        let report = Glm52RankWeightLoadReport {
            rank: self.placement.rank,
            tensor_count: loaded.loaded_tensor_count,
            total_bytes,
            non_expert_weight_contract: loaded.non_expert_weight_contract,
            loaded_to_gpu: true,
        };
        self.loaded = Some(loaded_state);
        Ok(report)
    }
}

fn rank_worker_loop(rx: Receiver<Glm52RankCommand>, mut state: Glm52RankThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Glm52RankCommand::LoadSlicedWeights { model_path, resp } => {
                let _ = resp.send(state.load_sliced_weights(&model_path));
            }
            Glm52RankCommand::Shutdown => break,
        }
    }
}

pub(crate) fn run_rejecting_dp_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    mut workers: Vec<Glm52RankWorker>,
) {
    while let Some(req) = submit_rx.blocking_recv() {
        send_scheduled(&req);
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message: "GLM5.2 PP8 decode forward runtime is not implemented yet: the PP runtime spine, MLA/indexer/KV decode, stage-local MoE, and the full PP8 graph are tracked in docs/models/glm52/pp-decode.md".to_string(),
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
    }
    if let Err(err) = shutdown_rank_workers(&mut workers) {
        log::error!("GLM5.2 rank worker shutdown failed: {err:?}");
    }
}

fn send_scheduled(req: &GenerateRequest) {
    let scheduled_at = unix_now_s();
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s: req.queued_at_unix_s.unwrap_or(scheduled_at),
        scheduled_at_unix_s: scheduled_at,
        prompt_tokens: req.prompt_tokens.len(),
        cached_tokens: 0,
    });
}

fn shutdown_rank_workers(workers: &mut [Glm52RankWorker]) -> Result<()> {
    for worker in workers.iter() {
        worker.request_shutdown()?;
    }
    for worker in workers.iter_mut() {
        worker.join()?;
    }
    Ok(())
}
