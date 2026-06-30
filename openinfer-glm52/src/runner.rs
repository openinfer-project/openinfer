use std::{
    path::{Path, PathBuf},
    thread,
};

use anyhow::{Result, ensure};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use openinfer_core::engine::{GenerateRequest, TokenEvent, unix_now_s};
use tokio::sync::mpsc;

use crate::weights::{
    GLM52_EP_RANKS, Glm52RankGpuContext, Glm52RankGpuWeights, Glm52RankLoadBundle,
    load_rank_weights_to_gpu,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankPlacement {
    pub(crate) rank: usize,
    pub(crate) device_ordinal: usize,
}

impl Glm52RankPlacement {
    pub(crate) fn new(rank: usize, device_ordinal: usize) -> Result<Self> {
        ensure!(
            rank < GLM52_EP_RANKS,
            "GLM5.2 rank must be < {GLM52_EP_RANKS}, got {rank}"
        );
        Ok(Self {
            rank,
            device_ordinal,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankWeightLoadReport {
    pub(crate) rank: usize,
    pub(crate) loaded_tensor_count: usize,
    pub(crate) loaded_total_bytes: usize,
    pub(crate) resident_raw_bytes: usize,
    pub(crate) loaded_to_gpu: bool,
}

enum Glm52RankCommand {
    LoadWeights {
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
            bundle.plan.rank == placement.rank,
            "GLM5.2 rank load plan {} does not match placement {}",
            bundle.plan.rank,
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

    pub(crate) fn load_weights_async(
        &self,
        model_path: &Path,
    ) -> Result<Receiver<Result<Glm52RankWeightLoadReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::LoadWeights {
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
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
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

#[allow(dead_code)]
struct Glm52RankLoadedState {
    weights: Glm52RankGpuWeights,
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

    fn load_weights(&mut self, model_path: &Path) -> Result<Glm52RankWeightLoadReport> {
        let loaded = load_rank_weights_to_gpu(&self.ctx, model_path, &self.bundle)?;
        ensure!(
            loaded.loaded_total_bytes == loaded.weights.total_bytes,
            "GLM5.2 rank {} loaded bytes {} differ from resident raw bytes {}",
            self.placement.rank,
            loaded.loaded_total_bytes,
            loaded.weights.total_bytes
        );
        let report = Glm52RankWeightLoadReport {
            rank: self.placement.rank,
            loaded_tensor_count: loaded.loaded_tensor_count,
            loaded_total_bytes: loaded.loaded_total_bytes,
            resident_raw_bytes: loaded.weights.total_bytes,
            loaded_to_gpu: true,
        };
        self.loaded = Some(Glm52RankLoadedState {
            weights: loaded.weights,
        });
        Ok(report)
    }
}

fn rank_worker_loop(rx: Receiver<Glm52RankCommand>, mut state: Glm52RankThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Glm52RankCommand::LoadWeights { model_path, resp } => {
                let _ = resp.send(state.load_weights(&model_path));
            }
            Glm52RankCommand::Shutdown => break,
        }
    }
}

pub(crate) fn run_rejecting_load_only_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    workers: Vec<Glm52RankWorker>,
) {
    let _workers = workers;
    while let Some(req) = submit_rx.blocking_recv() {
        let prompt_tokens = req.prompt_tokens.len();
        let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
        let scheduled_at_unix_s = unix_now_s();
        let _ = req.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s,
            scheduled_at_unix_s,
            prompt_tokens,
            cached_tokens: 0,
        });
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message: "GLM5.2 load-weight-only branch loaded resident GPU weights, but forward/decode is not implemented".to_owned(),
            prompt_tokens,
            completion_tokens: 0,
        });
    }
}
