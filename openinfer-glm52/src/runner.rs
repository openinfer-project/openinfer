use std::{
    path::{Path, PathBuf},
    thread,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use openinfer_core::engine::{GenerateRequest, TokenEvent, unix_now_s};
use openinfer_kernels::ops::GLM52_TRTLLM_GROUPED_OFFSET_ROWS;
use tokio::sync::mpsc;

use crate::arena::Glm52DecodeArena;
use crate::linear::{Glm52LinearSmokeReport, Glm52ProjectionSmokeReport};
use crate::moe_deepep::{
    Glm52DecodeGraphSmokeReport, Glm52DeepEpEnableReport, Glm52DeepEpSmokeReport,
    Glm52MoeGemmSmokeReport, Glm52MoeQuantSmokeReport,
};
use crate::moe_gemm::Glm52MoeGemmContractReport;
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

enum Glm52RankCommand {
    LoadSlicedWeights {
        model_path: PathBuf,
        resp: Sender<Result<Glm52RankWeightLoadReport>>,
    },
    EnableDeepEp {
        unique_id: [u8; 128],
        num_ranks: usize,
        resp: Sender<Result<Glm52DeepEpEnableReport>>,
    },
    SmokeDeepEpDecode {
        num_tokens: usize,
        resp: Sender<Result<Glm52DeepEpSmokeReport>>,
    },
    SmokeMoeQuantDecode {
        rows: usize,
        resp: Sender<Result<Glm52MoeQuantSmokeReport>>,
    },
    SmokeMoeGemmDecode {
        num_tokens: usize,
        resp: Sender<Result<Glm52MoeGemmSmokeReport>>,
    },
    SmokeNonExpertLinear {
        rows: usize,
        resp: Sender<Result<Glm52LinearSmokeReport>>,
    },
    SmokeDecodeGraph {
        num_tokens: usize,
        resp: Sender<Result<Glm52DecodeGraphSmokeReport>>,
    },
    ValidateMoeGemmContract {
        resp: Sender<Result<Glm52MoeGemmContractReport>>,
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

    pub(crate) fn enable_deepep_async(
        &self,
        unique_id: [u8; 128],
        num_ranks: usize,
    ) -> Result<Receiver<Result<Glm52DeepEpEnableReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::EnableDeepEp {
                unique_id,
                num_ranks,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn smoke_deepep_decode_async(
        &self,
        num_tokens: usize,
    ) -> Result<Receiver<Result<Glm52DeepEpSmokeReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::SmokeDeepEpDecode {
                num_tokens,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn smoke_moe_quant_decode_async(
        &self,
        rows: usize,
    ) -> Result<Receiver<Result<Glm52MoeQuantSmokeReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::SmokeMoeQuantDecode {
                rows,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn smoke_moe_gemm_decode_async(
        &self,
        num_tokens: usize,
    ) -> Result<Receiver<Result<Glm52MoeGemmSmokeReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::SmokeMoeGemmDecode {
                num_tokens,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn smoke_non_expert_linear_async(
        &self,
        rows: usize,
    ) -> Result<Receiver<Result<Glm52LinearSmokeReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::SmokeNonExpertLinear {
                rows,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn smoke_decode_graph_async(
        &self,
        num_tokens: usize,
    ) -> Result<Receiver<Result<Glm52DecodeGraphSmokeReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::SmokeDecodeGraph {
                num_tokens,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn validate_moe_gemm_contract_async(
        &self,
    ) -> Result<Receiver<Result<Glm52MoeGemmContractReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::ValidateMoeGemmContract { resp: resp_tx })
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
    deepep: Option<crate::moe_deepep::Glm52MoeDeepEpState>,
}

struct Glm52RankLoadedState {
    weights: Glm52RankGpuWeights,
    expert_weights: Glm52RankExpertFp8Weights,
    decode_arena: Glm52DecodeArena,
}

impl Glm52RankLoadedState {
    fn total_bytes(&self) -> usize {
        self.weights.total_bytes + self.expert_weights.total_bytes + self.decode_arena.total_bytes()
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
            deepep: None,
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
            decode_arena: Glm52DecodeArena::new(&self.ctx)?,
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

    fn enable_deepep(
        &mut self,
        unique_id: &[u8; 128],
        num_ranks: usize,
    ) -> Result<Glm52DeepEpEnableReport> {
        ensure!(
            self.loaded.is_some(),
            "GLM5.2 rank {} must load weights before enabling DeepEP",
            self.placement.rank
        );
        ensure!(
            self.deepep.is_none(),
            "GLM5.2 rank {} DeepEP already enabled",
            self.placement.rank
        );
        self.ctx.set_current()?;
        let state = crate::moe_deepep::Glm52MoeDeepEpState::new(
            &self.ctx.as_device_context(),
            unique_id,
            num_ranks,
            self.placement.rank,
        )?;
        let report = state.report();
        self.deepep = Some(state);
        Ok(report)
    }

    fn smoke_deepep_decode(&mut self, num_tokens: usize) -> Result<Glm52DeepEpSmokeReport> {
        let loaded = self.loaded.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} must load weights before DeepEP smoke",
                self.placement.rank
            )
        })?;
        let deepep = self.deepep.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} must enable DeepEP before DeepEP smoke",
                self.placement.rank
            )
        })?;
        deepep.decode_smoke_roundtrip(&self.ctx, &mut loaded.decode_arena, num_tokens)?;
        let router = loaded.weights.first_moe_router(&self.bundle.names)?;
        deepep.decode_router_smoke_roundtrip(
            &self.ctx,
            &router,
            &mut loaded.decode_arena,
            num_tokens,
        )
    }

    fn smoke_moe_quant_decode(&mut self, rows: usize) -> Result<Glm52MoeQuantSmokeReport> {
        let loaded = self.loaded.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} must load weights before MoE quant smoke",
                self.placement.rank
            )
        })?;
        crate::moe_deepep::decode_moe_quant_smoke(
            &self.ctx,
            self.placement.rank,
            &mut loaded.decode_arena,
            rows,
        )
    }

    fn smoke_moe_gemm_decode(&mut self, num_tokens: usize) -> Result<Glm52MoeGemmSmokeReport> {
        let loaded = self.loaded.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} must load weights before MoE GEMM smoke",
                self.placement.rank
            )
        })?;
        let deepep = self.deepep.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} must enable DeepEP before MoE GEMM smoke",
                self.placement.rank
            )
        })?;
        let router = loaded.weights.first_moe_router(&self.bundle.names)?;
        let layer = loaded.expert_weights.layers.first().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} has no MoE expert package",
                self.placement.rank
            )
        })?;
        deepep.decode_moe_gemm_smoke_roundtrip(
            &self.ctx,
            &router,
            layer,
            &mut loaded.decode_arena,
            num_tokens,
        )
    }

    fn smoke_non_expert_linear(&mut self, rows: usize) -> Result<Glm52LinearSmokeReport> {
        let loaded = self.loaded.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} must load weights before non-expert linear smoke",
                self.placement.rank
            )
        })?;
        let attention = loaded.weights.first_attention(&self.bundle.names)?;
        crate::linear::decode_attention_projection_smoke(
            &self.ctx,
            self.placement.rank,
            &attention,
            &mut loaded.decode_arena,
            rows,
        )
    }

    fn smoke_decode_graph(&mut self, num_tokens: usize) -> Result<Glm52DecodeGraphSmokeReport> {
        let loaded = self.loaded.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} must load weights before decode graph smoke",
                self.placement.rank
            )
        })?;
        let deepep = self.deepep.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} must enable DeepEP before decode graph smoke",
                self.placement.rank
            )
        })?;
        let router = loaded.weights.first_moe_router(&self.bundle.names)?;
        let layer = loaded.expert_weights.layers.first().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} has no MoE expert package",
                self.placement.rank
            )
        })?;
        deepep.decode_graph_smoke_roundtrip(
            &self.ctx,
            &router,
            layer,
            &mut loaded.decode_arena,
            num_tokens,
        )
    }

    fn validate_moe_gemm_contract(&mut self) -> Result<Glm52MoeGemmContractReport> {
        let loaded = self.loaded.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM5.2 rank {} must load weights before MoE GEMM contract validation",
                self.placement.rank
            )
        })?;
        crate::moe_gemm::validate_moe_gemm_contract(
            self.placement.rank,
            &loaded.expert_weights,
            &loaded.decode_arena,
        )
    }
}

fn rank_worker_loop(rx: Receiver<Glm52RankCommand>, mut state: Glm52RankThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Glm52RankCommand::LoadSlicedWeights { model_path, resp } => {
                let _ = resp.send(state.load_sliced_weights(&model_path));
            }
            Glm52RankCommand::EnableDeepEp {
                unique_id,
                num_ranks,
                resp,
            } => {
                let _ = resp.send(state.enable_deepep(&unique_id, num_ranks));
            }
            Glm52RankCommand::SmokeDeepEpDecode { num_tokens, resp } => {
                let _ = resp.send(state.smoke_deepep_decode(num_tokens));
            }
            Glm52RankCommand::SmokeMoeQuantDecode { rows, resp } => {
                let _ = resp.send(state.smoke_moe_quant_decode(rows));
            }
            Glm52RankCommand::SmokeMoeGemmDecode { num_tokens, resp } => {
                let _ = resp.send(state.smoke_moe_gemm_decode(num_tokens));
            }
            Glm52RankCommand::SmokeNonExpertLinear { rows, resp } => {
                let _ = resp.send(state.smoke_non_expert_linear(rows));
            }
            Glm52RankCommand::SmokeDecodeGraph { num_tokens, resp } => {
                let _ = resp.send(state.smoke_decode_graph(num_tokens));
            }
            Glm52RankCommand::ValidateMoeGemmContract { resp } => {
                let _ = resp.send(state.validate_moe_gemm_contract());
            }
            Glm52RankCommand::Shutdown => break,
        }
    }
}

pub(crate) fn install_deepep_backends(workers: &[Glm52RankWorker]) -> Result<()> {
    let started = Instant::now();
    log::info!(
        "start install GLM5.2 DeepEP backend: ranks={}",
        workers.len()
    );
    let unique_id = crate::moe_deepep::unique_id()?;
    let receivers = workers
        .iter()
        .map(|worker| worker.enable_deepep_async(unique_id, workers.len()))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(receivers.len());
    for (rank, receiver) in receivers.into_iter().enumerate() {
        let report = receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped DeepEP enable response"))?
            .with_context(|| format!("GLM5.2 rank {rank} DeepEP enable"))?;
        ensure!(
            report.rank == rank && report.num_ranks == workers.len(),
            "GLM5.2 rank {rank} invalid DeepEP report: {report:?}"
        );
        reports.push(report);
    }
    log::info!(
        "GLM5.2 DeepEP backend install cost {:.2}s: ranks={}, decode_caps={:?}",
        started.elapsed().as_secs_f64(),
        workers.len(),
        reports
            .iter()
            .map(|report| report.decode_max_tokens_per_rank)
            .collect::<Vec<_>>()
    );
    Ok(())
}

pub(crate) fn smoke_deepep_decode_roundtrip(
    workers: &[Glm52RankWorker],
    num_tokens: usize,
) -> Result<()> {
    let started = Instant::now();
    log::info!(
        "start GLM5.2 DeepEP decode smoke: ranks={}, tokens_per_rank={num_tokens}",
        workers.len()
    );
    let receivers = workers
        .iter()
        .map(|worker| worker.smoke_deepep_decode_async(num_tokens))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(receivers.len());
    for (rank, receiver) in receivers.into_iter().enumerate() {
        let report = receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped DeepEP smoke response"))?
            .with_context(|| format!("GLM5.2 rank {rank} DeepEP decode smoke"))?;
        ensure!(
            report.rank == rank
                && report.num_tokens == num_tokens
                && report.topk == crate::config::GLM52_TOPK
                && report.hidden == crate::config::GLM52_HIDDEN
                && report.router_routes_valid
                && report.router_weights_normalized
                && report.grouped_layout.grouped_layout_valid
                && report.grouped_layout.local_experts == crate::deepep::GLM52_LOCAL_EXPERTS
                && report.grouped_layout.expert_alignment
                    == crate::deepep::GLM52_DEEPEP_EXPERT_ALIGNMENT
                && report.recv_quant.is_some()
                && report
                    .recv_quant
                    .as_ref()
                    .is_some_and(|quant| quant.rank == rank
                        && quant.rows == report.grouped_layout.expanded_rows
                        && (quant.quant_ran || quant.rows == 0)
                        && (quant.quant_ran || report.grouped_layout.empty_rank)
                        && (quant.route_weights_applied || quant.rows == 0)
                        && quant.hidden_quant_valid
                        && quant.swiglu_quant_valid
                        && quant.swiglu_weighted_scale_valid
                        && quant.hidden_scale_layout_valid
                        && quant.swiglu_scale_layout_valid
                        && quant.scale_layout_aligned_rows
                            == openinfer_kernels::ops::glm52_deepgemm_tma_aligned_rows(quant.rows))
                && report
                    .gemm_metadata
                    .is_some_and(|metadata| metadata.rank == rank
                        && metadata.local_experts == crate::deepep::GLM52_LOCAL_EXPERTS
                        && metadata.active_experts == report.grouped_layout.active_experts
                        && metadata.expanded_rows == report.grouped_layout.expanded_rows
                        && metadata.offsets_valid
                        && metadata.w13_problem_sizes_valid
                        && metadata.w2_problem_sizes_valid)
                && report.combined_zero,
            "GLM5.2 rank {rank} invalid DeepEP smoke report: {report:?}"
        );
        reports.push(report);
    }
    log::info!(
        "GLM5.2 DeepEP decode smoke cost {:.2}s: ranks={}, tokens_per_rank={}, reports={:?}",
        started.elapsed().as_secs_f64(),
        workers.len(),
        num_tokens,
        reports
    );
    Ok(())
}

pub(crate) fn smoke_moe_quant_decode(workers: &[Glm52RankWorker], rows: usize) -> Result<()> {
    let started = Instant::now();
    log::info!(
        "start GLM5.2 MoE quant decode smoke: ranks={}, rows_per_rank={rows}",
        workers.len()
    );
    let receivers = workers
        .iter()
        .map(|worker| worker.smoke_moe_quant_decode_async(rows))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(receivers.len());
    for (rank, receiver) in receivers.into_iter().enumerate() {
        let report = receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped MoE quant smoke response"))?
            .with_context(|| format!("GLM5.2 rank {rank} MoE quant decode smoke"))?;
        ensure!(
            report.rank == rank
                && report.rows == rows
                && report.group_size == openinfer_kernels::ops::GLM52_MOE_QUANT_GROUP_SIZE
                && report.quant_ran
                && !report.route_weights_applied
                && report.hidden_quant_valid
                && report.swiglu_quant_valid
                && report.swiglu_weighted_scale_valid
                && report.hidden_scale_layout_valid
                && report.swiglu_scale_layout_valid
                && report.scale_layout_aligned_rows
                    == openinfer_kernels::ops::glm52_deepgemm_tma_aligned_rows(rows),
            "GLM5.2 rank {rank} invalid MoE quant smoke report: {report:?}"
        );
        reports.push(report);
    }
    log::info!(
        "GLM5.2 MoE quant decode smoke cost {:.2}s: ranks={}, rows_per_rank={}, reports={:?}",
        started.elapsed().as_secs_f64(),
        workers.len(),
        rows,
        reports
    );
    Ok(())
}

pub(crate) fn smoke_non_expert_linear(workers: &[Glm52RankWorker], rows: usize) -> Result<()> {
    let started = Instant::now();
    log::info!(
        "start GLM5.2 non-expert FP8 linear smoke: ranks={}, rows_per_rank={rows}",
        workers.len()
    );
    let receivers = workers
        .iter()
        .map(|worker| worker.smoke_non_expert_linear_async(rows))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(receivers.len());
    for (rank, receiver) in receivers.into_iter().enumerate() {
        let report = receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped FP8 linear smoke response"))?
            .with_context(|| format!("GLM5.2 rank {rank} non-expert FP8 linear smoke"))?;
        let expected = [
            (
                "q_a",
                crate::config::GLM52_Q_LORA_RANK,
                crate::config::GLM52_HIDDEN,
            ),
            (
                "q_b",
                crate::config::GLM52_Q_B_OUT,
                crate::config::GLM52_Q_LORA_RANK,
            ),
            (
                "kv_a",
                crate::config::GLM52_KV_A_OUT,
                crate::config::GLM52_HIDDEN,
            ),
            (
                "kv_b",
                crate::config::GLM52_KV_B_OUT,
                crate::config::GLM52_KV_LORA_RANK,
            ),
            (
                "o_proj",
                crate::config::GLM52_HIDDEN,
                crate::config::GLM52_O_PROJ_IN,
            ),
            (
                "indexer_wk",
                crate::config::GLM52_INDEX_HEAD_DIM,
                crate::config::GLM52_HIDDEN,
            ),
            (
                "indexer_wq_b",
                crate::config::GLM52_INDEX_HEADS * crate::config::GLM52_INDEX_HEAD_DIM,
                crate::config::GLM52_Q_LORA_RANK,
            ),
        ];
        ensure!(
            report.rank == rank
                && report.rows == rows
                && report.projections.len() == expected.len(),
            "GLM5.2 rank {rank} invalid non-expert FP8 linear smoke report: {report:?}"
        );
        for ((name, n, k), projection) in expected.iter().zip(&report.projections) {
            validate_projection_smoke_report(rank, projection, name, *n, *k)?;
        }
        reports.push(report);
    }
    log::info!(
        "GLM5.2 non-expert FP8 linear smoke cost {:.2}s: ranks={}, rows_per_rank={}, reports={:?}",
        started.elapsed().as_secs_f64(),
        workers.len(),
        rows,
        reports
    );
    Ok(())
}

fn validate_projection_smoke_report(
    rank: usize,
    report: &Glm52ProjectionSmokeReport,
    name: &str,
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!(
        report.name == name
            && report.n == n
            && report.k == k
            && report.weight_scale_rows == n.div_ceil(128)
            && report.weight_scale_cols == k.div_ceil(128)
            && report.activation_scale_cols == k.div_ceil(128)
            && report.workspace_bytes == 0
            && report.activation_quant_valid
            && report.output_nonzero,
        "GLM5.2 rank {rank} invalid {name} FP8 projection smoke report: {report:?}"
    );
    Ok(())
}

pub(crate) fn smoke_moe_gemm_decode_roundtrip(
    workers: &[Glm52RankWorker],
    num_tokens: usize,
) -> Result<()> {
    let started = Instant::now();
    log::info!(
        "start GLM5.2 MoE GEMM decode smoke: ranks={}, tokens_per_rank={num_tokens}",
        workers.len()
    );
    let receivers = workers
        .iter()
        .map(|worker| worker.smoke_moe_gemm_decode_async(num_tokens))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(receivers.len());
    for (rank, receiver) in receivers.into_iter().enumerate() {
        let report = receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped MoE GEMM smoke response"))?
            .with_context(|| format!("GLM5.2 rank {rank} MoE GEMM decode smoke"))?;
        ensure!(
            report.rank == rank
                && report.num_tokens == num_tokens
                && report.layer_idx == crate::config::GLM52_DENSE_LAYERS
                && report.router_routes_valid
                && report.router_weights_normalized
                && report.grouped_layout.grouped_layout_valid
                && report.gemm_metadata.offsets_valid
                && report.gemm_metadata.w13_problem_sizes_valid
                && report.gemm_metadata.w2_problem_sizes_valid
                && report.w13_output_nonzero
                && report.w2_output_nonzero
                && report.combined_nonzero,
            "GLM5.2 rank {rank} invalid MoE GEMM smoke report: {report:?}"
        );
        reports.push(report);
    }
    log::info!(
        "GLM5.2 MoE GEMM decode smoke cost {:.2}s: ranks={}, tokens_per_rank={}, reports={:?}",
        started.elapsed().as_secs_f64(),
        workers.len(),
        num_tokens,
        reports
    );
    Ok(())
}

pub(crate) fn smoke_decode_graph_roundtrip(
    workers: &[Glm52RankWorker],
    num_tokens: usize,
) -> Result<()> {
    let started = Instant::now();
    log::info!(
        "start GLM5.2 decode CUDA Graph smoke: ranks={}, tokens_per_rank={num_tokens}",
        workers.len()
    );
    let receivers = workers
        .iter()
        .map(|worker| worker.smoke_decode_graph_async(num_tokens))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(receivers.len());
    for (rank, receiver) in receivers.into_iter().enumerate() {
        let report = receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped decode graph smoke response"))?
            .with_context(|| format!("GLM5.2 rank {rank} decode CUDA Graph smoke"))?;
        ensure!(
            report.rank == rank
                && report.num_tokens == num_tokens
                && report.fixed_bucket_tokens == crate::deepep::GLM52_DEEPEP_DECODE_BATCH_CAP
                && report.worst_expanded_rows
                    == crate::deepep::Glm52DeepEpShape::tp1_dp8_h200()
                        .decode_capacity()?
                        .worst_expanded_tokens
                && report.router_routes_valid
                && report.router_weights_normalized
                && report.route_weights_applied
                && report.swiglu_weighted_scale_valid
                && report.moe_gemm_metadata_valid
                && report.grouped_layout_valid
                && report.w13_output_nonzero
                && report.w2_output_nonzero
                && report.combined_nonzero
                && report.capture_and_first_launch_ok
                && report.replay_ok,
            "GLM5.2 rank {rank} invalid decode graph smoke report: {report:?}"
        );
        reports.push(report);
    }
    log::info!(
        "GLM5.2 decode CUDA Graph smoke cost {:.2}s: ranks={}, tokens_per_rank={}, reports={:?}",
        started.elapsed().as_secs_f64(),
        workers.len(),
        num_tokens,
        reports
    );
    Ok(())
}

pub(crate) fn validate_moe_gemm_contracts(workers: &[Glm52RankWorker]) -> Result<()> {
    let started = Instant::now();
    log::info!(
        "start GLM5.2 MoE GEMM contract validation: ranks={}",
        workers.len()
    );
    let receivers = workers
        .iter()
        .map(Glm52RankWorker::validate_moe_gemm_contract_async)
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(receivers.len());
    for (rank, receiver) in receivers.into_iter().enumerate() {
        let report = receiver.recv().map_err(|_| {
            anyhow::anyhow!("GLM5.2 rank {rank} dropped MoE GEMM contract response")
        })??;
        ensure!(
            report.rank == rank
                && report.layer_count == crate::config::GLM52_MOE_LAYERS
                && report.first_layer_idx == crate::config::GLM52_DENSE_LAYERS
                && report.last_layer_idx == crate::config::GLM52_LAYERS - 1
                && report.w13.groups == crate::deepep::GLM52_LOCAL_EXPERTS
                && report.w13.m_capacity
                    == crate::deepep::Glm52DeepEpShape::tp1_dp8_h200()
                        .decode_capacity()?
                        .worst_expanded_tokens
                && report.w13.n == crate::config::GLM52_EXPERT_INTERMEDIATE * 2
                && report.w13.k == crate::config::GLM52_HIDDEN
                && report.w13.activation_scale_trtllm_rows == GLM52_TRTLLM_GROUPED_OFFSET_ROWS
                && report.w13.trtllm_workspace_bytes == 0
                && report.w2.groups == crate::deepep::GLM52_LOCAL_EXPERTS
                && report.w2.n == crate::config::GLM52_HIDDEN
                && report.w2.k == crate::config::GLM52_EXPERT_INTERMEDIATE
                && report.w2.activation_scale_trtllm_rows == GLM52_TRTLLM_GROUPED_OFFSET_ROWS
                && report.w2.trtllm_workspace_bytes == 0
                && report.psum_layout_entries == crate::deepep::GLM52_LOCAL_EXPERTS
                && report.expert_alignment == crate::deepep::GLM52_DEEPEP_EXPERT_ALIGNMENT
                && report.graph_stable_arena,
            "GLM5.2 rank {rank} invalid MoE GEMM contract report: {report:?}"
        );
        reports.push(report);
    }
    log::info!(
        "GLM5.2 MoE GEMM contract validation cost {:.2}s: ranks={}, reports={:?}",
        started.elapsed().as_secs_f64(),
        workers.len(),
        reports
    );
    Ok(())
}

pub(crate) fn run_rejecting_dp_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    mut workers: Vec<Glm52RankWorker>,
) {
    while let Some(req) = submit_rx.blocking_recv() {
        send_scheduled(&req);
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message: "GLM5.2 decode-only forward runtime is not implemented yet: prefilled KV handoff, batched decode bs>1, DeepEP MoE, and decode CUDA Graph are still tracked in docs/models/glm52/support.md".to_string(),
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
