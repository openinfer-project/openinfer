use std::{
    path::{Path, PathBuf},
    thread,
};

use anyhow::{Context as _, Result, ensure};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};

use crate::model::{GLM52_MAX_BATCH_PER_RANK, Glm52RankModel, Glm52StepShape};
use crate::moe_ep8::Glm52MoeEp8State;
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
    /// Non-collective: adopt the resident weights into the rank's model.
    /// Every rank must report success BEFORE anyone enters SetupComm — a
    /// build failure on one rank must never strand the others in NCCL init.
    BuildModel {
        resp: Sender<Result<()>>,
    },
    /// Collective: create the DeepEP context (barriers across ranks). Issued
    /// to every rank concurrently, only after all builds succeeded.
    SetupComm {
        unique_id: Box<[u8; 128]>,
        resp: Sender<Result<()>>,
    },
    /// One lock-step full-model step (75 MoE collectives inside): feed each
    /// active slot's `(token, position)` row, reply with the greedy next
    /// token per slot. The coordinator sends this to every rank each global
    /// step with the SAME batch bucket in `shape` (the collectives require
    /// every rank to agree on the step's global row count) — unoccupied
    /// slots carry the padding row and their outputs are discarded.
    Step {
        inputs: Box<[(u32, usize); GLM52_MAX_BATCH_PER_RANK]>,
        shape: Glm52StepShape,
        resp: Sender<Result<[u32; GLM52_MAX_BATCH_PER_RANK]>>,
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

    pub(crate) fn build_model_async(&self) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::BuildModel { resp: resp_tx })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn setup_comm_async(&self, unique_id: [u8; 128]) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::SetupComm {
                unique_id: Box::new(unique_id),
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn step_async(
        &self,
        inputs: [(u32, usize); GLM52_MAX_BATCH_PER_RANK],
        shape: Glm52StepShape,
    ) -> Result<Receiver<Result<[u32; GLM52_MAX_BATCH_PER_RANK]>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::Step {
                inputs: Box::new(inputs),
                shape,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn shutdown(&mut self) -> Result<()> {
        self.request_shutdown()?;
        self.join()
    }

    pub(crate) fn request_shutdown(&self) -> Result<()> {
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

struct Glm52RankRuntime {
    model: Box<Glm52RankModel>,
    /// Second stream on the same device: the shared expert overlaps the MoE
    /// collectives on it (fork/join via events inside the decode graph).
    aux_ctx: openinfer_kernels::tensor::DeviceContext,
    /// Populated by SetupComm (collective), after every rank's build succeeded.
    ep8: Option<Glm52MoeEp8State>,
}

struct Glm52RankThreadState {
    placement: Glm52RankPlacement,
    ctx: Glm52RankGpuContext,
    bundle: Glm52RankLoadBundle,
    loaded: Option<Glm52RankGpuWeights>,
    runtime: Option<Glm52RankRuntime>,
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
            runtime: None,
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
        self.loaded = Some(loaded.weights);
        Ok(report)
    }

    fn build_model(&mut self) -> Result<()> {
        let mut weights = self
            .loaded
            .take()
            .context("GLM5.2 build_model called before weights were loaded")?;
        let dev_ctx = self.ctx.device_context()?;
        let model = Box::new(Glm52RankModel::build(&dev_ctx, &mut weights)?);
        ensure!(
            weights.expert_layers.is_empty(),
            "GLM5.2 rank {} left {} expert layers unconsumed after model build",
            self.placement.rank,
            weights.expert_layers.len()
        );
        // Non-expert leftovers are the MTP-layer tensors (out of scope) —
        // dropped with `weights` here.
        drop(weights);
        let aux_ctx = self.ctx.auxiliary_device_context("decode aux")?;
        self.runtime = Some(Glm52RankRuntime {
            model,
            aux_ctx,
            ep8: None,
        });
        Ok(())
    }

    fn setup_comm(&mut self, unique_id: &[u8; 128]) -> Result<()> {
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 setup_comm before build_model")?;
        ensure!(
            runtime.ep8.is_none(),
            "GLM5.2 rank {} DeepEP context already created",
            self.placement.rank
        );
        // Collective: every rank calls this concurrently.
        runtime.ep8 = Some(Glm52MoeEp8State::new(
            &dev_ctx,
            unique_id,
            GLM52_EP_RANKS,
            self.placement.rank,
        )?);
        Ok(())
    }

    fn step(
        &mut self,
        inputs: &[(u32, usize); GLM52_MAX_BATCH_PER_RANK],
        shape: Glm52StepShape,
    ) -> Result<[u32; GLM52_MAX_BATCH_PER_RANK]> {
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 step before build_model")?;
        let ep8 = runtime
            .ep8
            .as_mut()
            .context("GLM5.2 step before setup_comm")?;
        runtime
            .model
            .decode_step(&dev_ctx, &runtime.aux_ctx, ep8, inputs, shape)
    }
}

fn rank_worker_loop(rx: Receiver<Glm52RankCommand>, mut state: Glm52RankThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Glm52RankCommand::LoadWeights { model_path, resp } => {
                let _ = resp.send(state.load_weights(&model_path));
            }
            Glm52RankCommand::BuildModel { resp } => {
                let _ = resp.send(state.build_model());
            }
            Glm52RankCommand::SetupComm { unique_id, resp } => {
                let _ = resp.send(state.setup_comm(&unique_id));
            }
            Glm52RankCommand::Step {
                inputs,
                shape,
                resp,
            } => {
                let _ = resp.send(state.step(&inputs, shape));
            }
            Glm52RankCommand::Shutdown => break,
        }
    }
    // The DeepEP context drop is collective — it runs here as every rank's
    // worker exits its loop after the coordinator broadcast Shutdown.
    drop(state);
}
