use std::{
    path::{Path, PathBuf},
    thread,
};

use anyhow::{Context as _, Result, ensure};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};

use crate::dspark::{
    GLM52_DSPARK_DRAFTS, Glm52DsparkModel, Glm52DsparkScratch, Glm52DsparkSlotState,
};

use crate::model::{GLM52_MAX_BATCH_PER_RANK, Glm52RankModel, Glm52StepKv, Glm52StepShape};
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
    /// This rank's device free VRAM right after the weights landed — the
    /// launch-time context-cap probe takes the fleet minimum.
    pub(crate) free_vram_bytes: usize,
}

/// The coordinator's launch-ahead directives for one step — both are GLOBAL
/// claims (a speculative replay is a full set of collectives, so ranks must
/// act on them together or not at all).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Glm52StepFlags {
    /// This step IS the speculative replay every rank enqueued last step.
    pub(crate) consume: bool,
    /// The next step is guaranteed to repeat this shape with each row
    /// advanced by its own argmax — every rank MUST enqueue that next
    /// replay launch-ahead (see `Glm52RankModel::decode_step`).
    pub(crate) lease: bool,
}

impl Glm52StepFlags {
    /// No speculation in either direction (warm pre-capture, tests).
    pub(crate) fn plain() -> Self {
        Self {
            consume: false,
            lease: false,
        }
    }
}

/// One step row whose committed token is sampled instead of taking the fused
/// greedy argmax: a non-greedy request's plain decode row, or the last row of
/// its prompt-completing span. `step` is the request-local decode step
/// (tokens generated so far) — a seeded request's philox seed mixes it so its
/// tokens replay independently of batch composition.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Glm52RowSample {
    pub(crate) row: usize,
    pub(crate) params: openinfer_sample::SamplingParams,
    pub(crate) step: u64,
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
        max_model_len: usize,
        resp: Sender<Result<()>>,
    },
    /// Collective: create the DeepEP context (barriers across ranks). Issued
    /// to every rank concurrently, only after all builds succeeded.
    SetupComm {
        unique_id: Box<[u8; 128]>,
        resp: Sender<Result<()>>,
    },
    /// One lock-step full-model step (75 MoE collectives inside): feed
    /// `inputs[row]` per forwarded row (a slot's span rows walk consecutive
    /// positions), reply with the next token per ROW (greedy argmax, or a
    /// sampling pass for the rows in `sampling`). The coordinator
    /// sends this to every rank each global step with the SAME batch bucket
    /// in `shape` (the collectives require every rank to agree on the step's
    /// global row count) — padding rows ride free slots and their outputs
    /// are discarded.
    Step {
        inputs: Box<[(u32, usize); GLM52_MAX_BATCH_PER_RANK]>,
        shape: Glm52StepShape,
        kv: Box<Glm52StepKv>,
        flags: Glm52StepFlags,
        /// Rows sampled instead of argmaxed (non-greedy requests; empty on
        /// the all-greedy fast path), and the step's philox seed for their
        /// unseeded members.
        sampling: Vec<Glm52RowSample>,
        seed: u64,
        resp: Sender<Result<[u32; GLM52_MAX_BATCH_PER_RANK]>>,
    },
    /// Non-collective: load the DSpark draft model onto this rank. Issued to
    /// every rank after BuildModel (the draft reuses the target's
    /// embed/lm_head at forward time).
    LoadDspark {
        path: PathBuf,
        resp: Sender<Result<()>>,
    },
    /// Non-collective: report this rank's current free device VRAM (the
    /// post-build headroom check).
    FreeVram {
        resp: Sender<Result<usize>>,
    },
    /// Rank-local draft round (no collectives; runs between global steps).
    /// `resets` clear slot draft states (request left / new admission),
    /// `appends` feed step rows of the LAST Step's `bucket` capture buffer to
    /// slot pending contexts, `proposals` ask for a 7-token draft span per
    /// slot from `(slot, anchor_token, anchor_pos)`. Ordered: resets, then
    /// appends, then proposals.
    Draft {
        bucket: usize,
        resets: Vec<usize>,
        appends: Vec<(usize, usize)>,
        proposals: Vec<(usize, u32, usize)>,
        resp: Sender<Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>>>,
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
                rank_worker_loop(&rx, Glm52RankThreadState::new(placement, ctx, bundle));
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

    pub(crate) fn build_model_async(&self, max_model_len: usize) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::BuildModel {
                max_model_len,
                resp: resp_tx,
            })
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
        kv: Glm52StepKv,
        flags: Glm52StepFlags,
        sampling: Vec<Glm52RowSample>,
        seed: u64,
    ) -> Result<Receiver<Result<[u32; GLM52_MAX_BATCH_PER_RANK]>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::Step {
                inputs: Box::new(inputs),
                shape,
                kv: Box::new(kv),
                flags,
                sampling,
                seed,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn load_dspark_async(&self, path: &Path) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::LoadDspark {
                path: path.to_path_buf(),
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn free_vram_async(&self) -> Result<Receiver<Result<usize>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::FreeVram { resp: resp_tx })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn draft_async(
        &self,
        bucket: usize,
        resets: Vec<usize>,
        appends: Vec<(usize, usize)>,
        proposals: Vec<(usize, u32, usize)>,
    ) -> Result<Receiver<Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::Draft {
                bucket,
                resets,
                appends,
                proposals,
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
    /// Populated by LoadDspark when the drafter is enabled.
    dspark: Option<Glm52DsparkRank>,
}

/// This rank's DSpark lane: the replicated draft model, the shared forward
/// scratch, and one draft state per slot — every buffer preallocated to the
/// launch-time cap at load (the VRAM probe's ledger charged exactly this
/// footprint), because a mid-serving draft round must never hit the
/// allocator: a transient OOM there would tear the whole engine down.
struct Glm52DsparkRank {
    model: Glm52DsparkModel,
    scratch: Glm52DsparkScratch,
    slots: [Glm52DsparkSlotState; GLM52_MAX_BATCH_PER_RANK],
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
        let free_vram_bytes = self.ctx.free_vram_bytes()?;
        let report = Glm52RankWeightLoadReport {
            rank: self.placement.rank,
            loaded_tensor_count: loaded.loaded_tensor_count,
            loaded_total_bytes: loaded.loaded_total_bytes,
            resident_raw_bytes: loaded.weights.total_bytes,
            loaded_to_gpu: true,
            free_vram_bytes,
        };
        self.loaded = Some(loaded.weights);
        Ok(report)
    }

    fn build_model(&mut self, max_model_len: usize) -> Result<()> {
        let mut weights = self
            .loaded
            .take()
            .context("GLM5.2 build_model called before weights were loaded")?;
        let dev_ctx = self.ctx.device_context()?;
        let model = Box::new(Glm52RankModel::build(
            &dev_ctx,
            &mut weights,
            max_model_len,
        )?);
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
            dspark: None,
        });
        Ok(())
    }

    fn load_dspark(&mut self, path: &Path) -> Result<()> {
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 load_dspark before build_model")?;
        ensure!(
            runtime.dspark.is_none(),
            "GLM5.2 rank {} DSpark drafter already loaded",
            self.placement.rank
        );
        let model = Glm52DsparkModel::load(&dev_ctx, path, runtime.model.max_model_len())?;
        let scratch = Glm52DsparkScratch::new(&dev_ctx, model.cache_len())?;
        let mut slots = Vec::with_capacity(GLM52_MAX_BATCH_PER_RANK);
        for _ in 0..GLM52_MAX_BATCH_PER_RANK {
            slots.push(Glm52DsparkSlotState::new(&dev_ctx, model.cache_len())?);
        }
        runtime.dspark = Some(Glm52DsparkRank {
            model,
            scratch,
            slots: slots
                .try_into()
                .map_err(|_| anyhow::anyhow!("GLM5.2 dspark slot state count drifted"))?,
        });
        Ok(())
    }

    fn draft(
        &mut self,
        bucket: usize,
        resets: &[usize],
        appends: &[(usize, usize)],
        proposals: &[(usize, u32, usize)],
    ) -> Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>> {
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 draft before build_model")?;
        let Glm52RankRuntime { model, dspark, .. } = runtime;
        let dspark = dspark
            .as_mut()
            .context("GLM5.2 draft command without a loaded DSpark drafter")?;

        for &slot in resets {
            ensure!(slot < GLM52_MAX_BATCH_PER_RANK, "dspark reset slot {slot}");
            dspark.slots[slot].reset();
        }
        if !appends.is_empty() {
            let captured = model.captured(bucket)?;
            for &(row, slot) in appends {
                ensure!(
                    row < bucket && slot < GLM52_MAX_BATCH_PER_RANK,
                    "dspark append row {row} (bucket {bucket}) slot {slot}"
                );
                dspark.slots[slot].append_captured_row(&dev_ctx, captured, row)?;
            }
        }
        if proposals.is_empty() {
            return Ok(Vec::new());
        }
        // Proposals must arrive slot-ascending so the reply order is the
        // request order (and slots are trivially unique).
        ensure!(
            proposals.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "dspark proposals must be sorted by slot: {proposals:?}"
        );
        let mut states = Vec::with_capacity(proposals.len());
        let mut anchors = Vec::with_capacity(proposals.len());
        let mut wanted = proposals.iter().peekable();
        for (slot, state) in dspark.slots.iter_mut().enumerate() {
            let Some(&&(want_slot, anchor, anchor_pos)) = wanted.peek() else {
                break;
            };
            if want_slot != slot {
                continue;
            }
            wanted.next();
            states.push(state);
            anchors.push((anchor, anchor_pos));
        }
        ensure!(
            wanted.peek().is_none(),
            "dspark propose slot out of range in {proposals:?}"
        );
        dspark.model.propose(
            &dev_ctx,
            model.embed(),
            model.lm_head(),
            &mut states,
            &anchors,
            &mut dspark.scratch,
        )
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
        kv: &Glm52StepKv,
        flags: Glm52StepFlags,
        sampling: &[Glm52RowSample],
        seed: u64,
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
        runtime.model.decode_step(
            &dev_ctx,
            &runtime.aux_ctx,
            ep8,
            inputs,
            shape,
            kv,
            flags,
            sampling,
            seed,
        )
    }
}

fn rank_worker_loop(rx: &Receiver<Glm52RankCommand>, mut state: Glm52RankThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Glm52RankCommand::LoadWeights { model_path, resp } => {
                let _ = resp.send(state.load_weights(&model_path));
            }
            Glm52RankCommand::BuildModel {
                max_model_len,
                resp,
            } => {
                let _ = resp.send(state.build_model(max_model_len));
            }
            Glm52RankCommand::SetupComm { unique_id, resp } => {
                let _ = resp.send(state.setup_comm(&unique_id));
            }
            Glm52RankCommand::Step {
                inputs,
                shape,
                kv,
                flags,
                sampling,
                seed,
                resp,
            } => {
                let _ = resp.send(state.step(&inputs, shape, &kv, flags, &sampling, seed));
            }
            Glm52RankCommand::LoadDspark { path, resp } => {
                let _ = resp.send(state.load_dspark(&path));
            }
            Glm52RankCommand::FreeVram { resp } => {
                let _ = resp.send(state.ctx.free_vram_bytes());
            }
            Glm52RankCommand::Draft {
                bucket,
                resets,
                appends,
                proposals,
                resp,
            } => {
                let _ = resp.send(state.draft(bucket, &resets, &appends, &proposals));
            }
            Glm52RankCommand::Shutdown => break,
        }
    }
    // The DeepEP context drop is collective — it runs here as every rank's
    // worker exits its loop after the coordinator broadcast Shutdown.
    drop(state);
}
