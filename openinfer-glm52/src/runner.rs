use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::bounded;
use crossbeam_channel::unbounded;
use openinfer_core::cuda_graph::CudaGraphDumpSummary;
use openinfer_kv_offload::KvArena;

use crate::dspark::GLM52_DSPARK_DRAFTS;
use crate::dspark::Glm52DsparkModel;
use crate::dspark::Glm52DsparkScratch;
use crate::dspark::Glm52DsparkSlotState;
use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::model::Glm52RankModel;
use crate::model::Glm52StepKv;
use crate::model::Glm52StepShape;
use crate::moe_ep_wo::Glm52MoeEpState;
use crate::moe_ep_wo::Glm52MoeEpWoState;
use crate::moe_ep8::Glm52MoeEp8State;
use crate::moe_tp::Glm52MoeTpRank;
use crate::moe_tp::Glm52MoeTpSliceBank;
use crate::moe_tp::Glm52MoeTpState;
use crate::moe_tp::Glm52TpExchange;
use crate::moe_tp::load_tp_slice_layer;
use crate::weights::Glm52RankGpuContext;
use crate::weights::Glm52RankGpuWeights;
use crate::weights::Glm52RankLoadBundle;
use crate::weights::Glm52WeightManifest;
use crate::weights::load_rank_weights_to_gpu;

/// Global rank + local CUDA device of one worker. Rank bounds are enforced
/// where placements are built, against the launch topology's real width —
/// there is no per-width invariant here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankPlacement {
    pub(crate) rank: usize,
    pub(crate) device_ordinal: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Glm52RankWeightLoadReport {
    pub(crate) rank: usize,
    pub(crate) loaded_tensor_count: usize,
    pub(crate) loaded_total_bytes: usize,
    resident_raw_bytes: usize,
    pub(crate) loaded_to_gpu: bool,
    /// This rank's device free VRAM right after the weights landed — the
    /// launch-time context-cap probe takes the fleet minimum.
    pub(crate) free_vram_bytes: usize,
}

/// The coordinator's launch-ahead directives for one step — both are GLOBAL
/// claims (a speculative replay is a full set of collectives, so ranks must
/// act on them together or not at all).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
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
/// greedy argmax: a non-greedy request's decode span row (every row of a
/// verify span — anchor and draft prefix alike), or the last row of its
/// prompt-completing span. `step` is the request-local decode step the row's
/// token lands at — a seeded request's philox seed mixes it, so its tokens
/// replay independently of batch composition AND of how many rows rode each
/// speculative round (spec and plain produce the same seeded stream).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Glm52RowSample {
    pub(crate) row: usize,
    pub(crate) params: openinfer_sample::SamplingParams,
    pub(crate) step: u64,
}

enum Glm52RankCommand {
    LoadWeights {
        model_path: PathBuf,
        moe_topo: crate::Glm52MoeTopo,
        resp: Sender<Result<Glm52RankWeightLoadReport>>,
    },
    /// Non-collective: adopt the resident weights into the rank's model.
    /// Every rank must report success BEFORE anyone enters SetupComm — a
    /// build failure on one rank must never strand the others in NCCL init.
    /// Replies with the rank's cache arena descriptors so launch can
    /// register them with the shared KV offload host.
    BuildModel {
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        dspark_enabled: bool,
        resp: Sender<Result<Vec<KvArena>>>,
    },
    /// Collective: create the DeepEP context (barriers across ranks). Issued
    /// to every rank concurrently, only after all builds succeeded. Under the
    /// TP8 topology, also allocates the LL buffers and rendezvouses peer
    /// pointers through `tp_exchange`.
    SetupComm {
        unique_id: Box<[u8; 128]>,
        moe_topo: crate::Glm52MoeTopo,
        tp_exchange: Option<Arc<Glm52TpExchange>>,
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
    /// Rank-local, vLLM-compat P/D only: deinterleave the RoPE dims of pages
    /// just restored from a vLLM-written namespace (see
    /// glm52_vllm_rope_fixup.cu). Sent after the pegaflow H2D completed and
    /// before the pages become readable; command-queue FIFO plus same-stream
    /// launch order the rewrite before any subsequent Step kernels.
    VllmRopeFixup {
        pages: Vec<i32>,
        resp: Sender<Result<()>>,
    },
    /// Rank-local inspection of an already pre-captured whole-step graph.
    /// This command stays on the worker so CUDA graph handles never cross the
    /// thread/context ownership boundary.
    DumpDecodeGraph {
        bucket: usize,
        png_path: PathBuf,
        title: String,
        resp: Sender<Result<CudaGraphDumpSummary>>,
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

    /// Deinterleave vLLM-restored RoPE dims at an idle step boundary.
    pub(crate) fn vllm_rope_fixup(&self, pages: Vec<i32>) -> Result<()> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::VllmRopeFixup {
                pages,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker dropped its fixup response"))?
    }

    pub(crate) fn load_weights_async(
        &self,
        model_path: &Path,
        moe_topo: crate::Glm52MoeTopo,
    ) -> Result<Receiver<Result<Glm52RankWeightLoadReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::LoadWeights {
                model_path: model_path.to_path_buf(),
                moe_topo,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn build_model_async(
        &self,
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        dspark_enabled: bool,
    ) -> Result<Receiver<Result<Vec<KvArena>>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::BuildModel {
                max_model_len,
                moe_topo,
                dspark_enabled,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn setup_comm_async(
        &self,
        unique_id: [u8; 128],
        moe_topo: crate::Glm52MoeTopo,
        tp_exchange: Option<Arc<Glm52TpExchange>>,
    ) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::SetupComm {
                unique_id: Box::new(unique_id),
                moe_topo,
                tp_exchange,
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

    fn dump_decode_graph_async(
        &self,
        bucket: usize,
        png_path: PathBuf,
        title: String,
    ) -> Result<Receiver<Result<CudaGraphDumpSummary>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::DumpDecodeGraph {
                bucket,
                png_path,
                title,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    fn shutdown(&mut self) -> Result<()> {
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

/// One rank executor as the engine sees it: an in-process worker thread or a
/// rank-host-side worker behind the wire. Same typed surface either way (the
/// remote twin mirrors [`Glm52RankWorker`] method-for-method), so the
/// coordinator and the load/build/probe paths never branch on locality.
pub(crate) enum Glm52Worker {
    Local(Glm52RankWorker),
    Remote(crate::remote::Glm52RemoteRankWorker),
}

impl Glm52Worker {
    pub(crate) fn load_weights_async(
        &self,
        model_path: &Path,
        moe_topo: crate::Glm52MoeTopo,
    ) -> Result<Receiver<Result<Glm52RankWeightLoadReport>>> {
        match self {
            Self::Local(worker) => worker.load_weights_async(model_path, moe_topo),
            Self::Remote(worker) => worker.load_weights_async(model_path, moe_topo),
        }
    }

    pub(crate) fn build_model_async(
        &self,
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        dspark_enabled: bool,
    ) -> Result<Receiver<Result<Vec<KvArena>>>> {
        match self {
            Self::Local(worker) => {
                worker.build_model_async(max_model_len, moe_topo, dspark_enabled)
            }
            Self::Remote(worker) => {
                worker.build_model_async(max_model_len, moe_topo, dspark_enabled)
            }
        }
    }

    pub(crate) fn setup_comm_async(
        &self,
        unique_id: [u8; 128],
        moe_topo: crate::Glm52MoeTopo,
        tp_exchange: Option<Arc<Glm52TpExchange>>,
    ) -> Result<Receiver<Result<()>>> {
        match self {
            Self::Local(worker) => worker.setup_comm_async(unique_id, moe_topo, tp_exchange),
            Self::Remote(worker) => {
                ensure!(
                    tp_exchange.is_none(),
                    "GLM5.2 tensor-replicated topologies are single-node"
                );
                worker.setup_comm_async(unique_id, moe_topo)
            }
        }
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
        match self {
            Self::Local(worker) => worker.step_async(inputs, shape, kv, flags, sampling, seed),
            Self::Remote(worker) => worker.step_async(inputs, shape, kv, flags, sampling, seed),
        }
    }

    pub(crate) fn load_dspark_async(&self, path: &Path) -> Result<Receiver<Result<()>>> {
        match self {
            Self::Local(worker) => worker.load_dspark_async(path),
            Self::Remote(worker) => worker.load_dspark_async(path),
        }
    }

    pub(crate) fn free_vram_async(&self) -> Result<Receiver<Result<usize>>> {
        match self {
            Self::Local(worker) => worker.free_vram_async(),
            Self::Remote(worker) => worker.free_vram_async(),
        }
    }

    pub(crate) fn draft_async(
        &self,
        bucket: usize,
        resets: Vec<usize>,
        appends: Vec<(usize, usize)>,
        proposals: Vec<(usize, u32, usize)>,
    ) -> Result<Receiver<Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>>>> {
        match self {
            Self::Local(worker) => worker.draft_async(bucket, resets, appends, proposals),
            Self::Remote(worker) => worker.draft_async(bucket, resets, appends, proposals),
        }
    }

    pub(crate) fn dump_decode_graph_async(
        &self,
        bucket: usize,
        png_path: PathBuf,
        title: String,
    ) -> Result<Receiver<Result<CudaGraphDumpSummary>>> {
        match self {
            Self::Local(worker) => worker.dump_decode_graph_async(bucket, png_path, title),
            Self::Remote(worker) => anyhow::bail!(
                "GLM5.2 rank {} is remote; the decode-graph dump is a local dev tool",
                worker.rank()
            ),
        }
    }

    pub(crate) fn request_shutdown(&self) -> Result<()> {
        match self {
            Self::Local(worker) => worker.request_shutdown(),
            Self::Remote(worker) => worker.request_shutdown(),
        }
    }
}

struct Glm52RankRuntime {
    model: Box<Glm52RankModel>,
    /// Second stream on the same device: the shared expert overlaps the MoE
    /// collectives on it (fork/join via events inside the decode graph).
    aux_ctx: openinfer_kernels::tensor::DeviceContext,
    /// Populated by SetupComm (collective), after every rank's build succeeded.
    ep8: Option<Glm52MoeEpState>,
    /// Populated by SetupComm when the TP8 pilot is on (LL rendezvous is
    /// collective too): the runtime state plus the slice banks loaded in
    /// LoadWeights.
    tp: Option<Glm52MoeTpRank>,
    /// Kept from SetupComm for the shutdown-side teardown rendezvous: the
    /// TP8 LL buffers are peer-mapped on every device, so no rank may unmap
    /// (drop `tp`) until every rank has retired its GPU work — see
    /// `teardown_tp`. Stored on ALL ranks (even if this rank's TP8 setup
    /// failed) so the barrier's participant count is always the full fleet.
    tp_exchange: Option<Arc<Glm52TpExchange>>,
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
    /// TP8-topology slice banks (loaded with the weights), waiting for the
    /// SetupComm rendezvous to assemble the runtime `Glm52MoeTpRank`.
    tp_slices: BTreeMap<usize, Glm52MoeTpSliceBank>,
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
            tp_slices: BTreeMap::new(),
            runtime: None,
        }
    }

    fn load_weights(
        &mut self,
        model_path: &Path,
        moe_topo: crate::Glm52MoeTopo,
    ) -> Result<Glm52RankWeightLoadReport> {
        let loaded = load_rank_weights_to_gpu(&self.ctx, model_path, &self.bundle)?;
        if moe_topo.uses_tensor_replicated_moe() {
            // Tensor-replicated topology: the bundle carried no routed experts;
            // gather this rank's 1/TP intermediate slice of ALL experts for
            // every MoE layer instead.
            let manifest = Glm52WeightManifest::from_model_dir(model_path)?;
            let dev_ctx = self.ctx.device_context()?;
            for layer in crate::config::GLM52_DENSE_LAYERS..crate::config::GLM52_LAYERS {
                let bank = load_tp_slice_layer(
                    &dev_ctx,
                    model_path,
                    &manifest,
                    self.placement.rank,
                    moe_topo.device_count(),
                    layer,
                )?;
                self.tp_slices.insert(layer, bank);
            }
        }
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

    fn build_model(
        &mut self,
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        dspark_enabled: bool,
    ) -> Result<Vec<KvArena>> {
        let mut weights = self
            .loaded
            .take()
            .context("GLM5.2 build_model called before weights were loaded")?;
        let dev_ctx = self.ctx.device_context()?;
        let model = Box::new(Glm52RankModel::build(
            &dev_ctx,
            &mut weights,
            max_model_len,
            moe_topo,
            moe_topo
                .uses_tensor_replicated_moe()
                .then_some(self.placement.rank),
            dspark_enabled,
        )?);
        let arenas = model.kv_arenas(&dev_ctx.stream)?;
        let aux_ctx = self.ctx.auxiliary_device_context("decode aux")?;
        self.runtime = Some(Glm52RankRuntime {
            model,
            aux_ctx,
            ep8: None,
            tp: None,
            tp_exchange: None,
            dspark: None,
        });
        Ok(arenas)
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

    fn setup_comm(
        &mut self,
        unique_id: &[u8; 128],
        moe_topo: crate::Glm52MoeTopo,
        tp_exchange: Option<&Arc<Glm52TpExchange>>,
    ) -> Result<()> {
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
        // Before anything fallible: every rank must hold the exchange so the
        // shutdown teardown rendezvous counts the full fleet even when this
        // rank's TP setup fails below.
        runtime.tp_exchange = tp_exchange.cloned();
        if moe_topo.uses_ep_expert_bundles() {
            // Collective: every EP rank calls this concurrently. The topology
            // selects the shim instantiation; the device generation selects
            // the routed-expert GEMM chain (the DeepGEMM masked chain is
            // sm_90a-only; everything else runs the weight-only mma chain).
            let sm_major = dev_ctx.ctx.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )?;
            let ranks = moe_topo.expected_ep_size();
            let rank = self.placement.rank;
            runtime.ep8 = Some(match (moe_topo, sm_major) {
                (crate::Glm52MoeTopo::Ep8, 9) => Glm52MoeEpState::MaskedFp8(Box::new(
                    Glm52MoeEp8State::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                (crate::Glm52MoeTopo::Ep8, _) => Glm52MoeEpState::WeightOnlyEp8(Box::new(
                    Glm52MoeEpWoState::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                (crate::Glm52MoeTopo::Ep4, _) => Glm52MoeEpState::WeightOnlyEp4(Box::new(
                    Glm52MoeEpWoState::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                (crate::Glm52MoeTopo::Ep16, _) => Glm52MoeEpState::WeightOnlyEp16(Box::new(
                    Glm52MoeEpWoState::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                (crate::Glm52MoeTopo::Ep32, _) => Glm52MoeEpState::WeightOnlyEp32(Box::new(
                    Glm52MoeEpWoState::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                (crate::Glm52MoeTopo::Ep64, _) => Glm52MoeEpState::WeightOnlyEp64(Box::new(
                    Glm52MoeEpWoState::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                (other, _) => anyhow::bail!("GLM5.2 {other:?} is not an expert-bundle topology"),
            });
        }
        if let Some(exchange) = tp_exchange {
            ensure!(
                !self.tp_slices.is_empty(),
                "GLM5.2 rank {} TP rendezvous without slice banks — load/setup drifted",
                self.placement.rank
            );
            // Collective: the LL pointer rendezvous blocks until all topology
            // ranks publish.
            let topology = match moe_topo {
                crate::Glm52MoeTopo::Tp8 => openinfer_kernels::ops::Glm52TpTopology::Tp8,
                crate::Glm52MoeTopo::Tp4 => openinfer_kernels::ops::Glm52TpTopology::Tp4,
                _ => anyhow::bail!("GLM5.2 {moe_topo:?} setup received a TP exchange"),
            };
            let state = Glm52MoeTpState::new(
                &dev_ctx,
                topology,
                self.placement.rank,
                self.placement.device_ordinal,
                exchange,
                self.tp_slices.len(),
                // One tail slot gathers rank-local vocabulary argmax
                // candidates after the 78 attention slots.
                crate::config::GLM52_LAYERS + 1,
            )?;
            runtime.tp = Some(Glm52MoeTpRank {
                state,
                slices: std::mem::take(&mut self.tp_slices),
            });
        }
        Ok(())
    }

    /// Shutdown path only: retire this rank's in-flight GPU work, then block
    /// in the fleet-wide teardown rendezvous before dropping the TP state —
    /// unmapping an LL buffer pulls the mapping from under EVERY device, so
    /// it is only safe once no rank can still be replaying a step.
    fn teardown_tp(&mut self) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let Some(exchange) = runtime.tp_exchange.take() else {
            return;
        };
        match self.ctx.device_context() {
            Ok(dev_ctx) => {
                if let Err(err) = dev_ctx.stream.synchronize() {
                    log::warn!(
                        "GLM5.2 rank {} TP8 teardown stream sync failed: {err:#}",
                        self.placement.rank
                    );
                }
            }
            Err(err) => log::warn!(
                "GLM5.2 rank {} TP8 teardown could not reach its device: {err:#}",
                self.placement.rank
            ),
        }
        exchange.teardown_rendezvous(self.placement.rank);
        runtime.tp = None;
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
        runtime.model.decode_step(
            &dev_ctx,
            &runtime.aux_ctx,
            runtime.ep8.as_mut(),
            runtime.tp.as_mut(),
            inputs,
            shape,
            kv,
            flags,
            sampling,
            seed,
        )
    }

    fn vllm_rope_fixup(&mut self, pages: &[i32]) -> Result<()> {
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 vLLM rope fixup before build_model")?;
        runtime.model.vllm_rope_fixup(&dev_ctx, pages)
    }

    fn dump_decode_graph(
        &self,
        bucket: usize,
        png_path: &Path,
        title: &str,
    ) -> Result<CudaGraphDumpSummary> {
        let runtime = self
            .runtime
            .as_ref()
            .context("GLM5.2 graph dump before build_model")?;
        runtime.model.dump_decode_graph_png(bucket, png_path, title)
    }
}

fn rank_worker_loop(rx: &Receiver<Glm52RankCommand>, mut state: Glm52RankThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Glm52RankCommand::LoadWeights {
                model_path,
                moe_topo,
                resp,
            } => {
                let _ = resp.send(state.load_weights(&model_path, moe_topo));
            }
            Glm52RankCommand::BuildModel {
                max_model_len,
                moe_topo,
                dspark_enabled,
                resp,
            } => {
                let _ = resp.send(state.build_model(max_model_len, moe_topo, dspark_enabled));
            }
            Glm52RankCommand::SetupComm {
                unique_id,
                moe_topo,
                tp_exchange,
                resp,
            } => {
                let _ = resp.send(state.setup_comm(&unique_id, moe_topo, tp_exchange.as_ref()));
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
            Glm52RankCommand::VllmRopeFixup { pages, resp } => {
                let _ = resp.send(state.vllm_rope_fixup(&pages));
            }
            Glm52RankCommand::DumpDecodeGraph {
                bucket,
                png_path,
                title,
                resp,
            } => {
                let _ = resp.send(state.dump_decode_graph(bucket, &png_path, &title));
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
    // Worker exit is load-bearing for the fleet (a silently gone rank hangs
    // every peer in the next collective) — always say why the loop ended.
    log::info!(
        "GLM5.2 rank {} worker loop exiting ({})",
        state.placement.rank,
        if rx.is_empty() {
            "shutdown or channel closed"
        } else {
            "channel closed with queued commands"
        }
    );
    // TP8 LL buffers are peer-mapped on every device, and a speculative
    // launch-ahead replay can still be in flight when Shutdown lands (harvest
    // only blocks on the argmax D2H). Quiesce this rank, then rendezvous with
    // the fleet before any rank unmaps — must run before the collective
    // DeepEP drop below, and keeps LL teardown independent of
    // `Glm52RankRuntime` field order.
    state.teardown_tp();
    // The DeepEP context drop is collective — it runs here as every rank's
    // worker exits its loop after the coordinator broadcast Shutdown.
    drop(state);
}
