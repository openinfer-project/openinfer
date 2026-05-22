use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Barrier,
        mpsc::{self, Receiver, Sender, SyncSender},
    },
    thread,
};

use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::nccl::{
    ReduceOp,
    safe::{Comm, Id},
};
use pegainfer_core::cuda_graph::CudaGraphState;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_trace;
use pegainfer_kernels::{
    ffi,
    ops::{
        KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_A_OUT, KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
        KIMI_K2_MLA_KV_LORA_RANK, KIMI_K2_MLA_LOCAL_HEADS_TP8, KIMI_K2_MLA_NOPE_DIM,
        KIMI_K2_MLA_O_LOCAL_IN_TP8, KIMI_K2_MLA_Q_HEAD_DIM, KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
        KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8, KIMI_K2_MLA_QKV_A_OUT, KIMI_K2_MLA_ROPE_DIM,
        KIMI_K2_MLA_V_HEAD_DIM, KIMI_K2_ROUTER_SCALE, KimiMarlinRouteWorkspace,
        KimiMarlinWna16Workspace, KimiMlaPagedKvLayout, KimiRouterBatch, KimiRouterConfig,
        KimiRouterOutput, KimiRouterScratch, add_batch_into, bf16_hidden_to_f32_into,
        embedding_batch_vocab_shard, f32_to_bf16_hidden_into, flashinfer_topk_row_states_bytes,
        gemm_graphsafe_into_checked, gemm_into_checked, kimi_flashinfer_batch_decode_mla,
        kimi_flashinfer_single_prefill_mla, kimi_marlin_sum_topk_rows_f32, kimi_marlin_w13_swiglu,
        kimi_marlin_wna16_w2_gemm, kimi_marlin_wna16_w13_gemm, kimi_mla_absorb_q_nope,
        kimi_mla_paged_kv_append, kimi_mla_rope_apply_kpe, kimi_mla_rope_assemble_prefill,
        kimi_mla_rope_split_decode, kimi_mla_split_qkv_a, kimi_mla_v_up,
        kimi_moe_marlin_align_block_size, kimi_router_noaux_tc_launch,
        kimi_scaled_add_f32_bf16_to_bf16, repeat_f32_for_reduce_scatter_into, rms_norm_batch_into,
        scale_f32_in_place, silu_mul_fused_batch_into,
    },
    tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates},
};

use crate::{
    config::{
        KIMI_K2_DENSE_INTERMEDIATE, KIMI_K2_DENSE_LAYERS, KIMI_K2_EXPERT_INTERMEDIATE,
        KIMI_K2_HIDDEN, KIMI_K2_LAYERS, KIMI_K2_MOE_LAYERS, KIMI_K2_Q_LORA_RANK,
        KIMI_K2_QK_ROPE_HEAD_DIM, KIMI_K2_RMS_NORM_EPS, KIMI_K2_ROPE_THETA, KIMI_K2_ROUTED_EXPERTS,
        KIMI_K2_TOPK, KIMI_K2_YARN_BETA_FAST, KIMI_K2_YARN_BETA_SLOW, KIMI_K2_YARN_FACTOR,
        KIMI_K2_YARN_ORIGINAL_MAX_POS,
    },
    layers::experts::{KIMI_K2_EP_WORLD, KIMI_K2_EP8_LOCAL_EXPERTS},
    runner::affinity::{KimiRankThreadPlacement, pin_rank_worker_thread},
    weights::{
        KimiGpuRawTensor, KimiLayerWeightKindNames, KimiLayerWeightNames,
        KimiRankExpertMarlinWeights, KimiRankGpuContext, KimiRankGpuWeights, KimiRankShardPlan,
        KimiRankSlicedLoadPlan, KimiRankWeightNames, KimiRankWeightPlan, KimiRouterDeviceWeights,
        KimiRouterGpuWeights, load_rank_sliced_weights_to_gpu,
    },
};

const KIMI_MARLIN_MAX_BLOCK_SIZE: usize = 64;
const KIMI_DECODE_MAX_BATCH: usize = 4;
const KIMI_DECODE_PAGE_SIZE: usize = 16;
const KIMI_DECODE_PAGES_PER_REQUEST: usize = 128;
const KIMI_DECODE_ROPE_CACHE_TOKENS: usize = KIMI_DECODE_PAGE_SIZE * KIMI_DECODE_PAGES_PER_REQUEST;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiK2RankPlacement {
    pub rank: usize,
    pub device_ordinal: usize,
}

impl KimiK2RankPlacement {
    pub fn new(rank: usize, device_ordinal: usize) -> Result<Self> {
        ensure!(rank < 8, "Kimi-K2 rank must be < 8, got {rank}");
        Ok(Self {
            rank,
            device_ordinal,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct KimiRankWeightLoadReport {
    pub rank: usize,
    pub tensor_count: usize,
    pub total_bytes: usize,
    pub expert_kernel_layers: usize,
    pub expert_kernel_total_bytes: usize,
    pub loaded_to_gpu: bool,
    pub typed_view_validated: bool,
    pub expert_kernel_weights_packaged: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct KimiOneTokenForwardReport {
    pub rank: usize,
    pub batch_slot: usize,
    pub input_token_id: u32,
    pub local_next_token_id: u32,
    pub local_next_token_global_id: u32,
    pub local_top_logit_f32: f32,
    pub vocab_start: usize,
    pub vocab_rows: usize,
    pub dense_layers_executed: usize,
    pub moe_layers_executed: usize,
}

enum KimiRankCommand {
    LoadSlicedWeights {
        model_path: PathBuf,
        resp: SyncSender<Result<KimiRankWeightLoadReport>>,
    },
    InitTpComm {
        id: Id,
        world_size: usize,
        resp: SyncSender<Result<()>>,
    },
    ForwardPromptNextToken {
        slot: usize,
        decode_batch_size: usize,
        input_ids: Vec<u32>,
        resp: SyncSender<Result<KimiOneTokenForwardReport>>,
    },
    ForwardDecodeBatchNextTokens {
        token_ids: Vec<u32>,
        append_positions: Vec<usize>,
        slots: Vec<usize>,
        decode_batch_size: usize,
        resp: SyncSender<Result<Vec<KimiOneTokenForwardReport>>>,
    },
    Shutdown,
}

pub(super) struct KimiRankWorker {
    placement: KimiK2RankPlacement,
    weight_plan: KimiRankWeightPlan,
    weight_names: KimiRankWeightNames,
    shard_plan: KimiRankShardPlan,
    sliced_load_plan: KimiRankSlicedLoadPlan,
    thread_placement: KimiRankThreadPlacement,
    tx: Sender<KimiRankCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl KimiRankWorker {
    pub(super) fn spawn(
        placement: KimiK2RankPlacement,
        weight_plan: KimiRankWeightPlan,
        weight_names: KimiRankWeightNames,
        shard_plan: KimiRankShardPlan,
        sliced_load_plan: KimiRankSlicedLoadPlan,
        thread_placement: KimiRankThreadPlacement,
        ctx: KimiRankGpuContext,
        collective_barrier: Arc<Barrier>,
        enable_cuda_graph: bool,
    ) -> Result<Self> {
        ensure!(
            placement.rank == weight_plan.rank,
            "Kimi rank placement {} does not match weight plan {}",
            placement.rank,
            weight_plan.rank
        );
        ensure!(
            weight_names.rank == weight_plan.rank,
            "Kimi rank weight names {} do not match weight plan {}",
            weight_names.rank,
            weight_plan.rank
        );
        ensure!(
            shard_plan.rank == weight_plan.rank,
            "Kimi rank shard plan {} does not match weight plan {}",
            shard_plan.rank,
            weight_plan.rank
        );
        ensure!(
            sliced_load_plan.rank == weight_plan.rank,
            "Kimi rank sliced load plan {} does not match weight plan {}",
            sliced_load_plan.rank,
            weight_plan.rank
        );
        ensure!(
            thread_placement.rank == weight_plan.rank,
            "Kimi rank thread placement {} does not match weight plan {}",
            thread_placement.rank,
            weight_plan.rank
        );
        let (tx, rx) = mpsc::channel();
        let (startup_tx, startup_rx) = mpsc::sync_channel::<Result<()>>(1);
        let worker_thread_placement = thread_placement.clone();
        let worker_weight_names = weight_names.clone();
        let worker_sliced_load_plan = sliced_load_plan.clone();
        let worker_ctx = ctx.clone();
        let worker_collective_barrier = Arc::clone(&collective_barrier);
        let handle = thread::Builder::new()
            .name(format!("kimi-k2-rank-{}", placement.rank))
            .spawn(move || {
                pin_rank_worker_thread(&worker_thread_placement);
                match bind_rank_thread(
                    worker_ctx,
                    worker_weight_names,
                    worker_sliced_load_plan,
                    worker_collective_barrier,
                    enable_cuda_graph,
                ) {
                    Ok(state) => {
                        let _ = startup_tx.send(Ok(()));
                        rank_worker_loop(rx, state);
                    }
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                    }
                }
            })
            .map_err(|err| anyhow::anyhow!("failed to spawn Kimi-K2 rank worker: {err}"))?;
        startup_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker exited during startup"))??;
        Ok(Self {
            placement,
            weight_plan,
            weight_names,
            shard_plan,
            sliced_load_plan,
            thread_placement,
            tx,
            handle: Some(handle),
        })
    }

    pub(super) fn placement(&self) -> KimiK2RankPlacement {
        self.placement
    }

    pub(super) fn weight_plan(&self) -> &KimiRankWeightPlan {
        &self.weight_plan
    }

    pub(super) fn weight_names(&self) -> &KimiRankWeightNames {
        &self.weight_names
    }

    pub(super) fn shard_plan(&self) -> &KimiRankShardPlan {
        &self.shard_plan
    }

    pub(super) fn sliced_load_plan(&self) -> &KimiRankSlicedLoadPlan {
        &self.sliced_load_plan
    }

    pub(super) fn thread_placement(&self) -> &KimiRankThreadPlacement {
        &self.thread_placement
    }

    pub(super) fn load_sliced_weights_async(
        &self,
        model_path: &Path,
    ) -> Result<Receiver<Result<KimiRankWeightLoadReport>>> {
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        self.tx
            .send(KimiRankCommand::LoadSlicedWeights {
                model_path: model_path.to_path_buf(),
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(super) fn init_tp_comm_async(
        &self,
        id: Id,
        world_size: usize,
    ) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        self.tx
            .send(KimiRankCommand::InitTpComm {
                id,
                world_size,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(super) fn forward_prompt_next_token_async(
        &self,
        input_ids: Vec<u32>,
        slot: usize,
        decode_batch_size: usize,
    ) -> Result<Receiver<Result<KimiOneTokenForwardReport>>> {
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        self.tx
            .send(KimiRankCommand::ForwardPromptNextToken {
                slot,
                decode_batch_size,
                input_ids,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(super) fn forward_decode_batch_next_tokens_async(
        &self,
        token_ids: Vec<u32>,
        append_positions: Vec<usize>,
        slots: Vec<usize>,
        decode_batch_size: usize,
    ) -> Result<Receiver<Result<Vec<KimiOneTokenForwardReport>>>> {
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        self.tx
            .send(KimiRankCommand::ForwardDecodeBatchNextTokens {
                token_ids,
                append_positions,
                slots,
                decode_batch_size,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(super) fn shutdown(&mut self) -> Result<()> {
        if self.handle.is_none() {
            return Ok(());
        }
        self.tx
            .send(KimiRankCommand::Shutdown)
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        let handle = self.handle.take().expect("Kimi rank handle must exist");
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker panicked"))?;
        Ok(())
    }
}

impl Drop for KimiRankWorker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

struct KimiRankThreadState {
    ctx: KimiRankGpuContext,
    decode_aux_ctx: DeviceContext,
    _cublas: KimiCublasThreadGuard,
    tp_comm: Option<OwnedRankComm>,
    weight_names: KimiRankWeightNames,
    sliced_load_plan: KimiRankSlicedLoadPlan,
    collective_barrier: Arc<Barrier>,
    enable_cuda_graph: bool,
    weight_report: Option<KimiRankWeightLoadReport>,
    loaded: Option<KimiRankLoadedWeights>,
}

struct OwnedRankComm(Comm);

// SAFETY: each NCCL communicator is moved into exactly one persistent Kimi rank
// worker and is only used from that worker thread on its owning CUDA stream.
unsafe impl Send for OwnedRankComm {}

impl OwnedRankComm {
    fn get(&self) -> &Comm {
        &self.0
    }
}

struct KimiRankLoadedWeights {
    gpu: KimiRankGpuWeights,
    expert_kernels: KimiRankExpertMarlinWeights,
    one_token_cache: KimiOneTokenForwardCache,
    decode_arenas: KimiWorkerDecodeArenas,
}

struct KimiWorkerDecodeArenas {
    arenas: Vec<KimiWorkerDecodeArena>,
}

impl KimiWorkerDecodeArenas {
    fn new(ctx: &DeviceContext, vocab_rows: usize) -> Result<Self> {
        let mut arenas = Vec::with_capacity(KIMI_DECODE_MAX_BATCH);
        for batch_size in 1..=KIMI_DECODE_MAX_BATCH {
            arenas.push(
                KimiWorkerDecodeArena::new(
                    ctx,
                    KIMI_K2_LAYERS,
                    batch_size,
                    KIMI_DECODE_PAGE_SIZE,
                    vocab_rows,
                )
                .with_context(|| format!("failed to allocate Kimi bs{batch_size} decode arena"))?,
            );
        }
        Ok(Self { arenas })
    }

    fn get_mut(&mut self, decode_batch_size: usize) -> Result<&mut KimiWorkerDecodeArena> {
        ensure!(
            (1..=KIMI_DECODE_MAX_BATCH).contains(&decode_batch_size),
            "Kimi decode batch size {decode_batch_size} must be in 1..={KIMI_DECODE_MAX_BATCH}"
        );
        Ok(&mut self.arenas[decode_batch_size - 1])
    }
}

struct KimiOneTokenForwardCache {
    vocab_start: usize,
    vocab_rows: usize,
    token_embedding: DeviceMatrix,
    final_norm: DeviceVec,
    lm_head: DeviceMatrix,
    layers: Vec<KimiLayerForwardCache>,
}

struct KimiLayerForwardCache {
    layer_idx: usize,
    attention: KimiAttentionForwardCache,
    kind: KimiLayerForwardKindCache,
}

struct KimiAttentionForwardCache {
    input_norm: DeviceVec,
    fused_qkv_a_proj: DeviceMatrix,
    q_a_norm: DeviceVec,
    q_b_proj: DeviceMatrix,
    kv_a_norm: DeviceVec,
    kv_b_proj: DeviceMatrix,
    o_proj: DeviceMatrix,
    post_attention_norm: DeviceVec,
}

enum KimiLayerForwardKindCache {
    Dense(KimiDenseForwardCache),
    Moe(KimiMoeForwardCache),
}

struct KimiDenseForwardCache {
    gate_up_proj: DeviceMatrix,
    down_proj: DeviceMatrix,
}

struct KimiMoeForwardCache {
    router: KimiRouterDeviceWeights,
    shared_gate_up_proj: DeviceMatrix,
    shared_down_proj: DeviceMatrix,
}

struct KimiWorkerDecodeArena {
    batch_size: usize,
    page_size: usize,
    max_pages: usize,
    append_capacity: usize,
    layout: KimiMlaPagedKvLayout,
    page_indices_d: CudaSlice<i32>,
    page_indptr_d: CudaSlice<i32>,
    last_page_len_d: CudaSlice<i32>,
    batch_indices_d: CudaSlice<i32>,
    positions_d: CudaSlice<i32>,
    request_indices_d: CudaSlice<i32>,
    kv_tile_indices_d: CudaSlice<i32>,
    kv_chunk_size_d: CudaSlice<i32>,
    token_ids_d: CudaSlice<u32>,
    cos_d: CudaSlice<half::bf16>,
    sin_d: CudaSlice<half::bf16>,
    layer_caches: Vec<KimiWorkerMlaLayerCache>,
    scratch: KimiWorkerDecodeScratch,
    logits: HiddenStates,
    graph: CudaGraphState,
}

struct KimiWorkerMlaLayerCache {
    ckv_cache: CudaSlice<half::bf16>,
    kpe_cache: CudaSlice<half::bf16>,
}

struct KimiWorkerDecodeScratch {
    hidden: HiddenStates,
    normed: HiddenStates,
    dense_gate_up: HiddenStates,
    dense_activated: HiddenStates,
    shared_gate_up: HiddenStates,
    shared_activated: HiddenStates,
    qkv_a: HiddenStates,
    q_a: HiddenStates,
    q_a_normed: HiddenStates,
    q_proj: HiddenStates,
    compressed_kv: HiddenStates,
    k_rope: HiddenStates,
    compressed_normed: HiddenStates,
    q_nope: HiddenStates,
    q_pe: HiddenStates,
    append_kpe: HiddenStates,
    q_abs_nope: HiddenStates,
    latent: HiddenStates,
    attn_out: HiddenStates,
    projected: HiddenStates,
    router_logits: CudaSlice<f32>,
    router_scores: CudaSlice<f32>,
    router_choice_scores: CudaSlice<f32>,
    router_topk_weight: CudaSlice<f32>,
    router_topk_idx: CudaSlice<i32>,
    marlin_route_workspace: KimiMarlinRouteWorkspace,
    marlin_workspace: KimiMarlinWna16Workspace,
    marlin_w13_out: HiddenStates,
    marlin_activated: HiddenStates,
    marlin_expert_output: HiddenStates,
    routed_out_f32: CudaSlice<f32>,
    routed_reduce_scatter_send_f32: CudaSlice<f32>,
    top1_value_scratch: CudaSlice<half::bf16>,
    top1_out: CudaSlice<i32>,
    hidden_allreduce_f32: CudaSlice<f32>,
}

struct KimiCublasThreadGuard;

impl Drop for KimiCublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            pegainfer_kernels::ffi::cublas_destroy();
        }
    }
}

fn bind_rank_thread(
    ctx: KimiRankGpuContext,
    weight_names: KimiRankWeightNames,
    sliced_load_plan: KimiRankSlicedLoadPlan,
    collective_barrier: Arc<Barrier>,
    enable_cuda_graph: bool,
) -> Result<KimiRankThreadState> {
    ctx.set_current()?;
    let decode_aux_stream = ctx.ctx.new_stream().with_context(|| {
        format!(
            "failed to create Kimi decode aux stream for device {}",
            ctx.device_ordinal
        )
    })?;
    let decode_aux_ctx = DeviceContext {
        ctx: Arc::clone(&ctx.ctx),
        stream: decode_aux_stream,
        device_ordinal: ctx.device_ordinal,
    };
    unsafe {
        pegainfer_kernels::ffi::cublas_init();
    }
    Ok(KimiRankThreadState {
        ctx,
        decode_aux_ctx,
        _cublas: KimiCublasThreadGuard,
        tp_comm: None,
        weight_names,
        sliced_load_plan,
        collective_barrier,
        enable_cuda_graph,
        weight_report: None,
        loaded: None,
    })
}

fn rank_worker_loop(rx: Receiver<KimiRankCommand>, mut state: KimiRankThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            KimiRankCommand::LoadSlicedWeights { model_path, resp } => {
                let result = state.load_sliced_weights(&model_path);
                let _ = resp.send(result);
            }
            KimiRankCommand::InitTpComm {
                id,
                world_size,
                resp,
            } => {
                let result = state.init_tp_comm(id, world_size);
                let _ = resp.send(result);
            }
            KimiRankCommand::ForwardPromptNextToken {
                slot,
                decode_batch_size,
                input_ids,
                resp,
            } => {
                let result = state.forward_prompt_next_token(slot, decode_batch_size, &input_ids);
                let _ = resp.send(result);
            }
            KimiRankCommand::ForwardDecodeBatchNextTokens {
                token_ids,
                append_positions,
                slots,
                decode_batch_size,
                resp,
            } => {
                let result = state.forward_decode_batch_next_tokens(
                    &token_ids,
                    &append_positions,
                    &slots,
                    decode_batch_size,
                );
                let _ = resp.send(result);
            }
            KimiRankCommand::Shutdown => break,
        }
    }
}

impl KimiRankThreadState {
    fn init_tp_comm(&mut self, id: Id, world_size: usize) -> Result<()> {
        ensure!(
            self.tp_comm.is_none(),
            "Kimi rank {} TP comm already attached",
            self.sliced_load_plan.rank
        );
        self.ctx.set_current()?;
        let rank = self.sliced_load_plan.rank;
        let comm = Comm::from_rank(self.ctx.stream.clone(), rank, world_size, id)
            .map_err(|err| anyhow::anyhow!("Kimi rank {rank} NCCL init failed: {err:?}"))?;
        self.tp_comm = Some(OwnedRankComm(comm));
        Ok(())
    }

    fn load_sliced_weights(&mut self, model_path: &Path) -> Result<KimiRankWeightLoadReport> {
        let mut weights =
            load_rank_sliced_weights_to_gpu(&self.ctx, model_path, &self.sliced_load_plan)
                .with_context(|| {
                    format!(
                        "failed to load Kimi rank {} sliced weights to GPU",
                        self.sliced_load_plan.rank
                    )
                })?;
        weights.typed_view(&self.weight_names).with_context(|| {
            format!(
                "failed to validate Kimi rank {} typed GPU weight view",
                self.sliced_load_plan.rank
            )
        })?;
        let tensor_count = weights.tensors.len();
        let total_bytes = weights.total_bytes;
        let expert_kernel_weights = weights
            .pack_rank_expert_marlin_weights(&self.ctx, &self.weight_names)
            .with_context(|| {
                format!(
                    "failed to package Kimi rank {} expert Marlin weights",
                    self.sliced_load_plan.rank
                )
            })?;
        let one_token_cache =
            KimiOneTokenForwardCache::from_gpu_weights(&self.ctx, &weights, &self.weight_names)
                .with_context(|| {
                    format!(
                        "failed to build Kimi rank {} one-token forward cache",
                        self.sliced_load_plan.rank
                    )
                })?;
        let decode_arenas =
            KimiWorkerDecodeArenas::new(&self.ctx.as_device_context(), one_token_cache.vocab_rows)
                .with_context(|| {
                    format!(
                        "failed to allocate Kimi rank {} decode arenas",
                        self.sliced_load_plan.rank
                    )
                })?;
        let report = KimiRankWeightLoadReport::from_loaded_weights(
            tensor_count,
            total_bytes,
            &expert_kernel_weights,
        );
        let loaded = KimiRankLoadedWeights {
            gpu: weights,
            expert_kernels: expert_kernel_weights,
            one_token_cache,
            decode_arenas,
        };
        debug_assert_eq!(loaded.gpu.rank, report.rank);
        debug_assert_eq!(
            loaded.expert_kernels.layers.len(),
            report.expert_kernel_layers
        );
        self.loaded = Some(loaded);
        self.weight_report = Some(report.clone());
        Ok(report)
    }

    fn forward_prompt_next_token(
        &mut self,
        slot: usize,
        decode_batch_size: usize,
        input_ids: &[u32],
    ) -> Result<KimiOneTokenForwardReport> {
        self.forward_prompt_next_token_inner(slot, decode_batch_size, input_ids)
    }

    fn forward_decode_batch_next_tokens(
        &mut self,
        token_ids: &[u32],
        append_positions: &[usize],
        slots: &[usize],
        decode_batch_size: usize,
    ) -> Result<Vec<KimiOneTokenForwardReport>> {
        ensure!(!token_ids.is_empty(), "Kimi batch decode requires tokens");
        ensure!(
            token_ids.len() == append_positions.len() && token_ids.len() == slots.len(),
            "Kimi batch decode input length mismatch: tokens={}, positions={}, slots={}",
            token_ids.len(),
            append_positions.len(),
            slots.len()
        );
        self.ctx.set_current()?;
        let loaded = self.loaded.as_mut().ok_or_else(|| {
            anyhow::anyhow!("Kimi rank weights must be loaded before batch decode")
        })?;
        let tp_comm = self.tp_comm.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Kimi rank {} TP comm must be attached before batch decode",
                loaded.gpu.rank
            )
        })?;
        let device_ctx = self.ctx.as_device_context();
        let decode_aux_ctx = DeviceContext {
            ctx: Arc::clone(&self.decode_aux_ctx.ctx),
            stream: Arc::clone(&self.decode_aux_ctx.stream),
            device_ordinal: self.decode_aux_ctx.device_ordinal,
        };
        let KimiRankLoadedWeights {
            gpu,
            expert_kernels,
            one_token_cache: cache,
            decode_arenas,
        } = loaded;
        let rank = gpu.rank;
        let active_len = token_ids.len();
        #[cfg(feature = "kernel-call-trace")]
        if rank == 0 && call_trace::is_enabled() {
            let kv_len = append_positions
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            for call in
                crate::batch_decode_trace::trace_decode_kernel_calls("", decode_batch_size, kv_len)?
            {
                call_trace::record_call(call);
            }
        }
        ensure!(
            (1..=KIMI_DECODE_MAX_BATCH).contains(&decode_batch_size),
            "Kimi decode batch size {decode_batch_size} must be in 1..={KIMI_DECODE_MAX_BATCH}"
        );
        ensure!(
            active_len <= decode_batch_size,
            "Kimi active decode rows {active_len} exceed decode batch size {decode_batch_size}"
        );
        let decode_arena = decode_arenas.get_mut(decode_batch_size)?;
        decode_arena
            .configure_batch_decode(&device_ctx, slots, append_positions)
            .with_context(|| format!("Kimi rank {rank} configure batch decode KV page table"))?;
        decode_arena
            .upload_batch_tokens(&device_ctx, token_ids)
            .with_context(|| format!("Kimi rank {rank} upload batch decode tokens"))?;

        if self.enable_cuda_graph {
            let mut graph = std::mem::take(&mut decode_arena.graph);
            let graph_barrier = Arc::clone(&self.collective_barrier);
            let result = graph.run_or_capture_synchronized(
                &device_ctx,
                |_| {
                    graph_barrier.wait();
                },
                || {
                    forward_decode_batch_next_token_kernels(
                        &device_ctx,
                        &decode_aux_ctx,
                        tp_comm.get(),
                        cache,
                        expert_kernels,
                        decode_arena,
                    )
                },
            );
            decode_arena.graph = graph;
            result?;
        } else {
            forward_decode_batch_next_token_kernels(
                &device_ctx,
                &decode_aux_ctx,
                tp_comm.get(),
                cache,
                expert_kernels,
                decode_arena,
            )?;
        }

        let local_top1 = read_local_top1_batch_values(
            &device_ctx,
            &decode_arena.logits,
            active_len,
            &mut decode_arena.scratch.top1_value_scratch,
            &mut decode_arena.scratch.top1_out,
        )?;
        let mut reports = Vec::with_capacity(active_len);
        for (row, (local_next, local_top_logit_f32)) in local_top1.into_iter().enumerate() {
            reports.push(KimiOneTokenForwardReport {
                rank,
                batch_slot: slots[row],
                input_token_id: token_ids[row],
                local_next_token_id: local_next,
                local_next_token_global_id: cache.vocab_start as u32 + local_next,
                local_top_logit_f32,
                vocab_start: cache.vocab_start,
                vocab_rows: cache.vocab_rows,
                dense_layers_executed: KIMI_K2_DENSE_LAYERS,
                moe_layers_executed: KIMI_K2_MOE_LAYERS,
            });
        }
        Ok(reports)
    }

    fn forward_prompt_next_token_inner(
        &mut self,
        slot: usize,
        decode_batch_size: usize,
        input_ids: &[u32],
    ) -> Result<KimiOneTokenForwardReport> {
        ensure!(!input_ids.is_empty(), "Kimi prompt forward requires tokens");
        self.ctx.set_current()?;
        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Kimi rank weights must be loaded before forward"))?;
        let tp_comm = self.tp_comm.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Kimi rank {} TP comm must be attached before forward",
                loaded.gpu.rank
            )
        })?;
        let device_ctx = self.ctx.as_device_context();
        let KimiRankLoadedWeights {
            gpu,
            expert_kernels,
            one_token_cache: cache,
            decode_arenas,
        } = loaded;
        let rank = gpu.rank;
        let seq_len = input_ids.len();
        let input_token_id = *input_ids
            .last()
            .ok_or_else(|| anyhow::anyhow!("Kimi prompt ids unexpectedly empty"))?;
        let decode_arena = decode_arenas.get_mut(decode_batch_size)?;
        decode_arena
            .configure_slot_prefill(&device_ctx, slot, seq_len)
            .with_context(|| {
                format!("Kimi rank {rank} configure slot {slot} prefill KV page table")
            })?;

        let mut hidden = HiddenStates::zeros(&device_ctx, KIMI_K2_HIDDEN, seq_len)?;
        let token_ids = device_ctx.stream.clone_htod(input_ids)?;
        embedding_batch_vocab_shard(
            &device_ctx,
            &cache.token_embedding,
            &token_ids,
            &mut hidden,
            cache.vocab_start as u32,
            cache.vocab_rows as u32,
        )?;
        self.collective_barrier.wait();
        // H20 NCCL returns ncclUnhandledCudaError if the first TP collective
        // follows the vocab-shard embedding launch without an explicit stream
        // drain. This is part of the temporary NCCL-sum correctness bridge.
        device_ctx
            .sync()
            .with_context(|| format!("Kimi rank {} sync before first TP all-reduce", rank))?;
        all_reduce_hidden_in_place(&mut hidden, tp_comm.get())?;

        let (cos_host, sin_host) = build_yarn_rope_cache(seq_len);
        let cos = device_ctx.stream.clone_htod(&cos_host)?;
        let sin = device_ctx.stream.clone_htod(&sin_host)?;
        let mut normed = HiddenStates::zeros(&device_ctx, KIMI_K2_HIDDEN, seq_len)?;
        let mut next_hidden = HiddenStates::zeros(&device_ctx, KIMI_K2_HIDDEN, seq_len)?;

        let mut dense_layers_executed = 0usize;
        let mut moe_layers_executed = 0usize;
        for layer in &cache.layers {
            Self::forward_mla_prefill(
                &device_ctx,
                tp_comm.get(),
                layer.layer_idx,
                &layer.attention,
                &cos,
                &sin,
                decode_arena,
                &mut hidden,
                &mut normed,
                &mut next_hidden,
            )
            .with_context(|| format!("Kimi MLA prefill layer {}", layer.layer_idx))?;
            match &layer.kind {
                KimiLayerForwardKindCache::Dense(dense) => {
                    Self::forward_dense_mlp(
                        &device_ctx,
                        tp_comm.get(),
                        dense,
                        &layer.attention.post_attention_norm,
                        &mut hidden,
                        &mut normed,
                        &mut next_hidden,
                    )
                    .with_context(|| format!("Kimi dense MLP layer {}", layer.layer_idx))?;
                    dense_layers_executed += 1;
                }
                KimiLayerForwardKindCache::Moe(moe) => {
                    Self::forward_moe_layer(
                        &device_ctx,
                        tp_comm.get(),
                        layer.layer_idx,
                        moe,
                        &layer.attention.post_attention_norm,
                        expert_kernels,
                        &mut hidden,
                        &mut normed,
                        &mut next_hidden,
                    )
                    .with_context(|| format!("Kimi MoE layer {}", layer.layer_idx))?;
                    moe_layers_executed += 1;
                }
            }
        }

        rms_norm_batch_into(
            &device_ctx,
            &hidden,
            &cache.final_norm,
            KIMI_K2_RMS_NORM_EPS,
            &mut normed,
        );
        let mut logits_hidden = HiddenStates::zeros(&device_ctx, cache.vocab_rows, seq_len)?;
        gemm_into_checked(&device_ctx, &cache.lm_head, &normed, &mut logits_hidden)?;
        let logits_offset = (seq_len - 1) * cache.vocab_rows;
        let logits_last = logits_hidden
            .data
            .slice(logits_offset..logits_offset + cache.vocab_rows);
        let mut logits_data = device_ctx.stream.alloc_zeros(cache.vocab_rows)?;
        device_ctx
            .stream
            .memcpy_dtod(&logits_last, &mut logits_data)?;
        let logits = DeviceVec {
            data: logits_data,
            len: cache.vocab_rows,
        };
        let (local_next, local_top_logit_f32) = sample_local_top1_with_value(&device_ctx, &logits)?;

        Ok(KimiOneTokenForwardReport {
            rank,
            batch_slot: slot,
            input_token_id,
            local_next_token_id: local_next,
            local_next_token_global_id: cache.vocab_start as u32 + local_next,
            local_top_logit_f32,
            vocab_start: cache.vocab_start,
            vocab_rows: cache.vocab_rows,
            dense_layers_executed,
            moe_layers_executed,
        })
    }

    fn forward_mla_prefill(
        ctx: &DeviceContext,
        comm: &Comm,
        layer_idx: usize,
        attention: &KimiAttentionForwardCache,
        cos: &CudaSlice<half::bf16>,
        sin: &CudaSlice<half::bf16>,
        decode_arena: &mut KimiWorkerDecodeArena,
        hidden: &mut HiddenStates,
        normed: &mut HiddenStates,
        next_hidden: &mut HiddenStates,
    ) -> Result<()> {
        let seq_len = hidden.seq_len;
        rms_norm_batch_into(
            ctx,
            hidden,
            &attention.input_norm,
            KIMI_K2_RMS_NORM_EPS,
            normed,
        );
        let mut qkv_a = HiddenStates::zeros(ctx, KIMI_K2_MLA_QKV_A_OUT, seq_len)?;
        let mut q_a = HiddenStates::zeros(ctx, KIMI_K2_Q_LORA_RANK, seq_len)?;
        let mut q_a_normed = HiddenStates::zeros(ctx, KIMI_K2_Q_LORA_RANK, seq_len)?;
        let mut q_proj = HiddenStates::zeros(ctx, KIMI_K2_MLA_Q_LOCAL_OUT_TP8, seq_len)?;
        let mut compressed_kv = HiddenStates::zeros(ctx, KIMI_K2_MLA_KV_LORA_RANK, seq_len)?;
        let mut k_rope = HiddenStates::zeros(ctx, KIMI_K2_QK_ROPE_HEAD_DIM, seq_len)?;
        gemm_into_checked(ctx, &attention.fused_qkv_a_proj, normed, &mut qkv_a)?;
        kimi_mla_split_qkv_a(ctx, &qkv_a, &mut q_a, &mut compressed_kv, &mut k_rope)?;
        rms_norm_batch_into(
            ctx,
            &q_a,
            &attention.q_a_norm,
            KIMI_K2_RMS_NORM_EPS,
            &mut q_a_normed,
        );
        gemm_into_checked(ctx, &attention.q_b_proj, &q_a_normed, &mut q_proj)?;

        let mut compressed_normed = HiddenStates::zeros(ctx, KIMI_K2_MLA_KV_LORA_RANK, seq_len)?;
        let mut kv_b = HiddenStates::zeros(ctx, KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8, seq_len)?;
        rms_norm_batch_into(
            ctx,
            &compressed_kv,
            &attention.kv_a_norm,
            KIMI_K2_RMS_NORM_EPS,
            &mut compressed_normed,
        );
        let mut append_kpe = HiddenStates::zeros(ctx, KIMI_K2_MLA_ROPE_DIM, seq_len)?;
        kimi_mla_rope_apply_kpe(
            ctx,
            &k_rope,
            cos,
            sin,
            &decode_arena.positions_d,
            &mut append_kpe,
        )?;
        decode_arena.append_prefill_layer_kv(ctx, layer_idx, &compressed_normed, &append_kpe)?;
        gemm_into_checked(ctx, &attention.kv_b_proj, &compressed_normed, &mut kv_b)?;

        let mut q_attn = HiddenStates::zeros(ctx, KIMI_K2_MLA_Q_LOCAL_OUT_TP8, seq_len)?;
        let mut k_cache = ctx
            .stream
            .alloc_zeros(seq_len * KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_Q_HEAD_DIM)?;
        let mut v_cache = ctx
            .stream
            .alloc_zeros(seq_len * KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_V_HEAD_DIM)?;
        kimi_mla_rope_assemble_prefill(
            ctx,
            &q_proj,
            &k_rope,
            &kv_b,
            cos,
            sin,
            &mut q_attn,
            &mut k_cache,
            &mut v_cache,
        )?;

        let mut attn_out = HiddenStates::zeros(ctx, KIMI_K2_MLA_O_LOCAL_IN_TP8, seq_len)?;
        kimi_flashinfer_single_prefill_mla(
            ctx,
            &q_attn,
            &k_cache,
            &v_cache,
            &mut attn_out,
            kimi_mla_softmax_scale(),
        )?;
        let mut projected = HiddenStates::zeros(ctx, KIMI_K2_HIDDEN, seq_len)?;
        gemm_into_checked(ctx, &attention.o_proj, &attn_out, &mut projected)?;
        all_reduce_hidden_in_place(&mut projected, comm)?;
        add_batch_into(ctx, hidden, &projected, next_hidden)?;
        std::mem::swap(hidden, next_hidden);
        Ok(())
    }

    fn forward_dense_mlp(
        ctx: &DeviceContext,
        comm: &Comm,
        dense: &KimiDenseForwardCache,
        post_attention_norm: &DeviceVec,
        hidden: &mut HiddenStates,
        normed: &mut HiddenStates,
        next_hidden: &mut HiddenStates,
    ) -> Result<()> {
        forward_dense_mlp_batch_into(
            ctx,
            comm,
            dense,
            post_attention_norm,
            hidden,
            normed,
            next_hidden,
        )
    }

    fn forward_moe_layer(
        ctx: &DeviceContext,
        comm: &Comm,
        layer_idx: usize,
        moe: &KimiMoeForwardCache,
        post_attention_norm: &DeviceVec,
        expert_kernels: &KimiRankExpertMarlinWeights,
        hidden: &mut HiddenStates,
        normed: &mut HiddenStates,
        next_hidden: &mut HiddenStates,
    ) -> Result<()> {
        forward_moe_layer_batch_into(
            ctx,
            comm,
            layer_idx,
            moe,
            post_attention_norm,
            expert_kernels,
            hidden,
            normed,
            next_hidden,
        )
    }
}

fn forward_decode_batch_next_token_kernels(
    device_ctx: &DeviceContext,
    decode_aux_ctx: &DeviceContext,
    comm: &Comm,
    cache: &KimiOneTokenForwardCache,
    expert_kernels: &KimiRankExpertMarlinWeights,
    decode_arena: &mut KimiWorkerDecodeArena,
) -> Result<()> {
    embedding_batch_vocab_shard(
        device_ctx,
        &cache.token_embedding,
        &decode_arena.token_ids_d,
        &mut decode_arena.scratch.hidden,
        cache.vocab_start as u32,
        cache.vocab_rows as u32,
    )?;
    all_reduce_hidden_via_f32_in_place(
        device_ctx,
        &mut decode_arena.scratch.hidden,
        &mut decode_arena.scratch.hidden_allreduce_f32,
        comm,
    )?;

    for layer in &cache.layers {
        forward_mla_decode_layer_into(device_ctx, &layer.attention, decode_arena, layer.layer_idx)
            .with_context(|| format!("Kimi MLA batch decode layer {}", layer.layer_idx))?;
        all_reduce_hidden_via_f32_in_place(
            device_ctx,
            &mut decode_arena.scratch.projected,
            &mut decode_arena.scratch.hidden_allreduce_f32,
            comm,
        )?;
        add_batch_into(
            device_ctx,
            &decode_arena.scratch.hidden,
            &decode_arena.scratch.projected,
            &mut decode_arena.scratch.normed,
        )?;
        std::mem::swap(
            &mut decode_arena.scratch.hidden,
            &mut decode_arena.scratch.normed,
        );
        match &layer.kind {
            KimiLayerForwardKindCache::Dense(dense) => {
                forward_dense_mlp_decode_into(
                    device_ctx,
                    comm,
                    dense,
                    &layer.attention.post_attention_norm,
                    &mut decode_arena.scratch,
                )
                .with_context(|| {
                    format!("Kimi dense batch decode MLP layer {}", layer.layer_idx)
                })?;
            }
            KimiLayerForwardKindCache::Moe(moe) => {
                forward_moe_layer_decode_into(
                    device_ctx,
                    decode_aux_ctx,
                    comm,
                    layer.layer_idx,
                    moe,
                    &layer.attention.post_attention_norm,
                    expert_kernels,
                    &mut decode_arena.scratch,
                )
                .with_context(|| format!("Kimi MoE batch decode layer {}", layer.layer_idx))?;
            }
        }
    }

    let active_len = decode_arena.scratch.hidden.seq_len;
    rms_norm_batch_into(
        device_ctx,
        &decode_arena.scratch.hidden,
        &cache.final_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut decode_arena.scratch.normed,
    );
    gemm_graphsafe_into_checked(
        device_ctx,
        &cache.lm_head,
        &decode_arena.scratch.normed,
        &mut decode_arena.logits,
    )?;
    launch_local_top1_batch(
        device_ctx,
        &decode_arena.logits,
        active_len,
        &mut decode_arena.scratch.top1_value_scratch,
        &mut decode_arena.scratch.top1_out,
    )
}

impl KimiOneTokenForwardCache {
    fn from_gpu_weights(
        ctx: &KimiRankGpuContext,
        weights: &KimiRankGpuWeights,
        names: &KimiRankWeightNames,
    ) -> Result<Self> {
        ensure!(
            weights.rank == names.rank,
            "Kimi forward cache rank mismatch: weights={}, names={}",
            weights.rank,
            names.rank
        );
        ensure!(
            names.layers.len() == KIMI_K2_LAYERS,
            "Kimi forward cache needs {} layers, got {}",
            KIMI_K2_LAYERS,
            names.layers.len()
        );

        let vocab_rows = names.plan.vocab_range.len();
        let token_embedding = raw_tensor(weights, &names.top.token_embedding)?.copy_bf16_matrix(
            ctx,
            vocab_rows,
            KIMI_K2_HIDDEN,
            "token_embedding",
        )?;
        let final_norm = raw_tensor(weights, &names.top.final_norm)?.copy_bf16_vec(
            ctx,
            KIMI_K2_HIDDEN,
            "final_norm",
        )?;
        let lm_head = raw_tensor(weights, &names.top.lm_head)?.copy_bf16_matrix(
            ctx,
            vocab_rows,
            KIMI_K2_HIDDEN,
            "lm_head",
        )?;
        let layers = names
            .layers
            .iter()
            .map(|layer| load_layer_forward_cache(ctx, weights, layer))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            vocab_start: names.plan.vocab_range.start,
            vocab_rows,
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }
}

impl KimiWorkerDecodeArena {
    fn new(
        ctx: &DeviceContext,
        num_layers: usize,
        batch_size: usize,
        page_size: usize,
        vocab_rows: usize,
    ) -> Result<Self> {
        ensure!(num_layers > 0, "Kimi decode arena needs layers");
        ensure!(
            batch_size > 0,
            "Kimi decode arena batch_size must be positive"
        );
        ensure!(
            KIMI_DECODE_PAGES_PER_REQUEST > 0,
            "Kimi decode arena pages per request must be positive"
        );
        let max_pages = batch_size
            .checked_mul(KIMI_DECODE_PAGES_PER_REQUEST)
            .ok_or_else(|| anyhow::anyhow!("Kimi decode arena max_pages overflow"))?;
        let append_capacity = batch_size
            .checked_mul(KIMI_DECODE_ROPE_CACHE_TOKENS)
            .ok_or_else(|| anyhow::anyhow!("Kimi decode arena append metadata overflow"))?;
        let layout = KimiMlaPagedKvLayout::separate_contiguous(max_pages, page_size, batch_size);
        let initial_positions = vec![0usize; batch_size];
        let (page_indices, page_indptr, last_page_len) = build_decode_append_page_metadata(
            batch_size,
            page_size,
            KIMI_DECODE_PAGES_PER_REQUEST,
            &initial_positions,
        )?;
        let batch_indices = (0..batch_size).map(|idx| idx as i32).collect::<Vec<_>>();
        let positions = initial_positions
            .iter()
            .map(|position| *position as i32)
            .collect::<Vec<_>>();
        let mut batch_indices_padded = vec![0i32; append_capacity];
        let mut positions_padded = vec![0i32; append_capacity];
        batch_indices_padded[..batch_size].copy_from_slice(&batch_indices);
        positions_padded[..batch_size].copy_from_slice(&positions);
        let request_indices = (0..batch_size).map(|idx| idx as i32).collect::<Vec<_>>();
        let kv_tile_indices = vec![0i32; batch_size];
        let kv_chunk_size = vec![1i32; batch_size];
        let token_ids = vec![0u32; batch_size];
        let (cos_host, sin_host) = build_yarn_rope_cache(KIMI_DECODE_ROPE_CACHE_TOKENS);

        let mut layer_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layer_caches.push(KimiWorkerMlaLayerCache {
                ckv_cache: ctx
                    .stream
                    .alloc_zeros::<half::bf16>(layout.required_ckv_len()?)?,
                kpe_cache: ctx
                    .stream
                    .alloc_zeros::<half::bf16>(layout.required_kpe_len()?)?,
            });
        }

        Ok(Self {
            batch_size,
            page_size,
            max_pages,
            append_capacity,
            layout,
            page_indices_d: ctx.stream.clone_htod(&page_indices)?,
            page_indptr_d: ctx.stream.clone_htod(&page_indptr)?,
            last_page_len_d: ctx.stream.clone_htod(&last_page_len)?,
            batch_indices_d: ctx.stream.clone_htod(&batch_indices_padded)?,
            positions_d: ctx.stream.clone_htod(&positions_padded)?,
            request_indices_d: ctx.stream.clone_htod(&request_indices)?,
            kv_tile_indices_d: ctx.stream.clone_htod(&kv_tile_indices)?,
            kv_chunk_size_d: ctx.stream.clone_htod(&kv_chunk_size)?,
            token_ids_d: ctx.stream.clone_htod(&token_ids)?,
            cos_d: ctx.stream.clone_htod(&cos_host)?,
            sin_d: ctx.stream.clone_htod(&sin_host)?,
            layer_caches,
            scratch: KimiWorkerDecodeScratch::new(ctx, batch_size)?,
            logits: HiddenStates::zeros(ctx, vocab_rows, batch_size)?,
            graph: CudaGraphState::new(),
        })
    }

    fn configure_slot_prefill(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        seq_len: usize,
    ) -> Result<()> {
        ensure!(seq_len > 0, "Kimi prefill KV write requires tokens");
        ensure!(
            slot < self.batch_size,
            "Kimi prefill slot {slot} exceeds batch_size {}",
            self.batch_size
        );
        ensure!(
            seq_len <= KIMI_DECODE_ROPE_CACHE_TOKENS,
            "Kimi prefill seq_len {seq_len} exceeds per-request decode KV capacity {}",
            KIMI_DECODE_ROPE_CACHE_TOKENS
        );
        ensure!(
            seq_len <= self.append_capacity,
            "Kimi prefill seq_len {seq_len} exceeds append metadata capacity {}",
            self.append_capacity
        );
        let mut append_positions = vec![0usize; self.batch_size];
        append_positions[slot] = seq_len - 1;
        let (page_indices, page_indptr, last_page_len) = build_decode_append_page_metadata(
            self.batch_size,
            self.page_size,
            KIMI_DECODE_PAGES_PER_REQUEST,
            &append_positions,
        )?;
        ensure!(
            page_indices.len() == self.max_pages,
            "Kimi prefill page table length {} must match arena max_pages {}",
            page_indices.len(),
            self.max_pages
        );
        let batch_indices = vec![slot as i32; seq_len];
        let positions = (0..seq_len as i32).collect::<Vec<_>>();
        ctx.stream
            .memcpy_htod(&page_indices, &mut self.page_indices_d)?;
        ctx.stream
            .memcpy_htod(&page_indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(&last_page_len, &mut self.last_page_len_d)?;
        {
            let mut batch_indices_d = self.batch_indices_d.slice_mut(0..seq_len);
            ctx.stream
                .memcpy_htod(&batch_indices, &mut batch_indices_d)?;
        }
        {
            let mut positions_d = self.positions_d.slice_mut(0..seq_len);
            ctx.stream.memcpy_htod(&positions, &mut positions_d)?;
        }
        Ok(())
    }

    fn configure_batch_decode(
        &mut self,
        ctx: &DeviceContext,
        slots: &[usize],
        append_positions: &[usize],
    ) -> Result<()> {
        ensure!(!slots.is_empty(), "Kimi batch decode requires active slots");
        ensure!(
            slots.len() == append_positions.len(),
            "Kimi batch decode slots/positions mismatch: slots={}, positions={}",
            slots.len(),
            append_positions.len()
        );
        ensure!(
            slots.len() <= self.batch_size,
            "Kimi batch decode active slots {} exceeds batch_size {}",
            slots.len(),
            self.batch_size
        );
        let mut slot_positions = vec![0usize; self.batch_size];
        let mut batch_indices = vec![0i32; self.batch_size];
        let mut row_positions = vec![0i32; self.batch_size];
        let mut request_indices = vec![0i32; self.batch_size];
        let kv_tile_indices = vec![0i32; self.batch_size];
        let mut kv_chunk_size = vec![1i32; self.batch_size];
        let mut occupied_slots = vec![false; self.batch_size];

        for (row, (&slot, &position)) in slots.iter().zip(append_positions.iter()).enumerate() {
            ensure!(
                slot < self.batch_size,
                "Kimi batch decode slot {slot} exceeds batch_size {}",
                self.batch_size
            );
            ensure!(
                !occupied_slots[slot],
                "Kimi batch decode slot {slot} appears more than once"
            );
            ensure!(
                position < KIMI_DECODE_ROPE_CACHE_TOKENS,
                "Kimi decode append_position {position} exceeds per-request KV capacity {}",
                KIMI_DECODE_ROPE_CACHE_TOKENS
            );
            occupied_slots[slot] = true;
            slot_positions[slot] = position;
            batch_indices[row] = slot as i32;
            row_positions[row] = position as i32;
            request_indices[row] = slot as i32;
            kv_chunk_size[row] = (position + 1) as i32;
        }
        let free_slots = occupied_slots
            .iter()
            .enumerate()
            .filter_map(|(slot, occupied)| (!occupied).then_some(slot))
            .collect::<Vec<_>>();
        for row in slots.len()..self.batch_size {
            let padding_slot = free_slots[row - slots.len()];
            batch_indices[row] = padding_slot as i32;
            request_indices[row] = padding_slot as i32;
        }

        let (page_indices, page_indptr, last_page_len) = build_decode_append_page_metadata(
            self.batch_size,
            self.page_size,
            KIMI_DECODE_PAGES_PER_REQUEST,
            &slot_positions,
        )?;
        ensure!(
            page_indices.len() == self.max_pages,
            "Kimi batch decode page table length {} must match arena max_pages {}",
            page_indices.len(),
            self.max_pages
        );
        ctx.stream
            .memcpy_htod(&page_indices, &mut self.page_indices_d)?;
        ctx.stream
            .memcpy_htod(&page_indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(&last_page_len, &mut self.last_page_len_d)?;
        {
            let mut batch_indices_d = self.batch_indices_d.slice_mut(0..self.batch_size);
            ctx.stream
                .memcpy_htod(&batch_indices, &mut batch_indices_d)?;
        }
        {
            let mut positions_d = self.positions_d.slice_mut(0..self.batch_size);
            ctx.stream.memcpy_htod(&row_positions, &mut positions_d)?;
        }
        ctx.stream
            .memcpy_htod(&request_indices, &mut self.request_indices_d)?;
        ctx.stream
            .memcpy_htod(&kv_tile_indices, &mut self.kv_tile_indices_d)?;
        ctx.stream
            .memcpy_htod(&kv_chunk_size, &mut self.kv_chunk_size_d)?;
        Ok(())
    }

    fn upload_batch_tokens(&mut self, ctx: &DeviceContext, token_ids: &[u32]) -> Result<()> {
        ensure!(
            token_ids.len() <= self.batch_size,
            "Kimi batch token upload length {} exceeds batch_size {}",
            token_ids.len(),
            self.batch_size
        );
        let mut tokens = vec![0u32; self.batch_size];
        tokens[..token_ids.len()].copy_from_slice(token_ids);
        ctx.stream.memcpy_htod(&tokens, &mut self.token_ids_d)?;
        Ok(())
    }

    fn append_prefill_layer_kv(
        &mut self,
        ctx: &DeviceContext,
        layer_idx: usize,
        compressed_normed: &HiddenStates,
        append_kpe: &HiddenStates,
    ) -> Result<()> {
        ensure!(
            compressed_normed.seq_len <= self.append_capacity,
            "Kimi prefill append seq_len {} exceeds metadata capacity {}",
            compressed_normed.seq_len,
            self.append_capacity
        );
        let layer_cache = self.layer_caches.get_mut(layer_idx).ok_or_else(|| {
            anyhow::anyhow!("Kimi prefill KV layer cache {layer_idx} out of range")
        })?;
        kimi_mla_paged_kv_append(
            ctx,
            &mut layer_cache.ckv_cache,
            &mut layer_cache.kpe_cache,
            self.layout,
            &self.page_indices_d,
            &self.page_indptr_d,
            &self.last_page_len_d,
            compressed_normed,
            append_kpe,
            &self.batch_indices_d,
            &self.positions_d,
        )
    }
}

fn build_decode_append_page_metadata(
    batch_size: usize,
    page_size: usize,
    pages_per_request: usize,
    append_positions: &[usize],
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    ensure!(
        append_positions.len() == batch_size,
        "Kimi decode append positions length {} must match batch_size {}",
        append_positions.len(),
        batch_size
    );
    ensure!(page_size > 0, "Kimi decode page_size must be positive");
    ensure!(
        pages_per_request > 0,
        "Kimi decode pages_per_request must be positive"
    );
    let max_pages = batch_size
        .checked_mul(pages_per_request)
        .ok_or_else(|| anyhow::anyhow!("Kimi decode page table max_pages overflow"))?;
    let mut page_indices = vec![0i32; max_pages];
    let mut page_indptr = Vec::with_capacity(batch_size + 1);
    let mut last_page_len = Vec::with_capacity(batch_size);
    let mut cursor = 0usize;

    for (request_idx, position) in append_positions.iter().copied().enumerate() {
        let cached_tokens_after_append = position
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Kimi decode append position overflow"))?;
        let pages = cached_tokens_after_append.div_ceil(page_size);
        ensure!(
            pages <= pages_per_request,
            "Kimi decode request {request_idx} needs {pages} pages for position {position}, capacity is {pages_per_request}"
        );
        page_indptr.push(cursor as i32);
        let request_page_base = request_idx
            .checked_mul(pages_per_request)
            .ok_or_else(|| anyhow::anyhow!("Kimi decode request page base overflow"))?;
        for local_page in 0..pages {
            page_indices[cursor] = (request_page_base + local_page) as i32;
            cursor += 1;
        }
        let last_len = ((cached_tokens_after_append - 1) % page_size) + 1;
        last_page_len.push(last_len as i32);
    }
    page_indptr.push(cursor as i32);
    Ok((page_indices, page_indptr, last_page_len))
}

impl KimiWorkerDecodeScratch {
    fn new(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        let marlin_block_size = kimi_marlin_block_size(batch_size);
        let marlin_route_workspace =
            KimiMarlinRouteWorkspace::new(ctx, batch_size, marlin_block_size)?;
        let marlin_workspace = KimiMarlinWna16Workspace::new(
            ctx,
            marlin_route_workspace.max_m_blocks,
            KIMI_K2_HIDDEN,
            marlin_block_size,
        )?;
        let route_elems = batch_size * KIMI_K2_TOPK;
        let reduce_scatter_send_rows = batch_size * KIMI_K2_EP_WORLD;
        Ok(Self {
            hidden: HiddenStates::zeros(ctx, KIMI_K2_HIDDEN, batch_size)?,
            normed: HiddenStates::zeros(ctx, KIMI_K2_HIDDEN, batch_size)?,
            dense_gate_up: HiddenStates::zeros(ctx, KIMI_K2_DENSE_INTERMEDIATE / 4, batch_size)?,
            dense_activated: HiddenStates::zeros(ctx, KIMI_K2_DENSE_INTERMEDIATE / 8, batch_size)?,
            shared_gate_up: HiddenStates::zeros(ctx, KIMI_K2_EXPERT_INTERMEDIATE / 4, batch_size)?,
            shared_activated: HiddenStates::zeros(
                ctx,
                KIMI_K2_EXPERT_INTERMEDIATE / 8,
                batch_size,
            )?,
            qkv_a: HiddenStates::zeros(ctx, KIMI_K2_MLA_QKV_A_OUT, batch_size)?,
            q_a: HiddenStates::zeros(ctx, KIMI_K2_Q_LORA_RANK, batch_size)?,
            q_a_normed: HiddenStates::zeros(ctx, KIMI_K2_Q_LORA_RANK, batch_size)?,
            q_proj: HiddenStates::zeros(ctx, KIMI_K2_MLA_Q_LOCAL_OUT_TP8, batch_size)?,
            compressed_kv: HiddenStates::zeros(ctx, KIMI_K2_MLA_KV_LORA_RANK, batch_size)?,
            k_rope: HiddenStates::zeros(ctx, KIMI_K2_MLA_ROPE_DIM, batch_size)?,
            compressed_normed: HiddenStates::zeros(ctx, KIMI_K2_MLA_KV_LORA_RANK, batch_size)?,
            q_nope: HiddenStates::zeros(
                ctx,
                KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_NOPE_DIM,
                batch_size,
            )?,
            q_pe: HiddenStates::zeros(ctx, KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8, batch_size)?,
            append_kpe: HiddenStates::zeros(ctx, KIMI_K2_MLA_ROPE_DIM, batch_size)?,
            q_abs_nope: HiddenStates::zeros(ctx, KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, batch_size)?,
            latent: HiddenStates::zeros(ctx, KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, batch_size)?,
            attn_out: HiddenStates::zeros(ctx, KIMI_K2_MLA_O_LOCAL_IN_TP8, batch_size)?,
            projected: HiddenStates::zeros(ctx, KIMI_K2_HIDDEN, batch_size)?,
            router_logits: ctx
                .stream
                .alloc_zeros(batch_size * KIMI_K2_ROUTED_EXPERTS)?,
            router_scores: ctx
                .stream
                .alloc_zeros(batch_size * KIMI_K2_ROUTED_EXPERTS)?,
            router_choice_scores: ctx
                .stream
                .alloc_zeros(batch_size * KIMI_K2_ROUTED_EXPERTS)?,
            router_topk_weight: ctx.stream.alloc_zeros(batch_size * KIMI_K2_TOPK)?,
            router_topk_idx: ctx.stream.alloc_zeros(batch_size * KIMI_K2_TOPK)?,
            marlin_route_workspace,
            marlin_workspace,
            marlin_w13_out: HiddenStates::zeros(ctx, 2 * KIMI_K2_EXPERT_INTERMEDIATE, route_elems)?,
            marlin_activated: HiddenStates::zeros(ctx, KIMI_K2_EXPERT_INTERMEDIATE, route_elems)?,
            marlin_expert_output: HiddenStates::zeros(ctx, KIMI_K2_HIDDEN, route_elems)?,
            routed_out_f32: ctx.stream.alloc_zeros(batch_size * KIMI_K2_HIDDEN)?,
            routed_reduce_scatter_send_f32: ctx
                .stream
                .alloc_zeros(reduce_scatter_send_rows * KIMI_K2_HIDDEN)?,
            top1_value_scratch: ctx.stream.alloc_zeros(batch_size)?,
            top1_out: ctx.stream.alloc_zeros(batch_size)?,
            hidden_allreduce_f32: ctx.stream.alloc_zeros(batch_size * KIMI_K2_HIDDEN)?,
        })
    }
}

fn forward_mla_decode_layer_into(
    ctx: &DeviceContext,
    attention: &KimiAttentionForwardCache,
    arena: &mut KimiWorkerDecodeArena,
    layer_idx: usize,
) -> Result<()> {
    let KimiWorkerDecodeArena {
        batch_size,
        layout,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        batch_indices_d,
        positions_d,
        request_indices_d,
        kv_tile_indices_d,
        kv_chunk_size_d,
        cos_d,
        sin_d,
        layer_caches,
        scratch,
        ..
    } = arena;
    let layer_cache = layer_caches
        .get_mut(layer_idx)
        .ok_or_else(|| anyhow::anyhow!("Kimi decode layer cache {layer_idx} out of range"))?;

    rms_norm_batch_into(
        ctx,
        &scratch.hidden,
        &attention.input_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.normed,
    );
    gemm_graphsafe_into_checked(
        ctx,
        &attention.fused_qkv_a_proj,
        &scratch.normed,
        &mut scratch.qkv_a,
    )?;
    kimi_mla_split_qkv_a(
        ctx,
        &scratch.qkv_a,
        &mut scratch.q_a,
        &mut scratch.compressed_kv,
        &mut scratch.k_rope,
    )?;
    rms_norm_batch_into(
        ctx,
        &scratch.q_a,
        &attention.q_a_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.q_a_normed,
    );
    gemm_graphsafe_into_checked(
        ctx,
        &attention.q_b_proj,
        &scratch.q_a_normed,
        &mut scratch.q_proj,
    )?;
    rms_norm_batch_into(
        ctx,
        &scratch.compressed_kv,
        &attention.kv_a_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.compressed_normed,
    );
    kimi_mla_rope_split_decode(
        ctx,
        &scratch.q_proj,
        &scratch.k_rope,
        cos_d,
        sin_d,
        positions_d,
        &mut scratch.q_nope,
        &mut scratch.q_pe,
        &mut scratch.append_kpe,
    )?;
    kimi_mla_absorb_q_nope(
        ctx,
        &attention.kv_b_proj,
        &scratch.q_nope,
        &mut scratch.q_abs_nope,
    )?;
    kimi_mla_paged_kv_append(
        ctx,
        &mut layer_cache.ckv_cache,
        &mut layer_cache.kpe_cache,
        *layout,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        &scratch.compressed_normed,
        &scratch.append_kpe,
        batch_indices_d,
        positions_d,
    )?;
    kimi_flashinfer_batch_decode_mla(
        ctx,
        &scratch.q_abs_nope,
        &scratch.q_pe,
        &mut scratch.latent,
        &layer_cache.ckv_cache,
        &layer_cache.kpe_cache,
        *layout,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        request_indices_d,
        kv_tile_indices_d,
        kv_chunk_size_d,
        kimi_mla_softmax_scale(),
    )?;
    kimi_mla_v_up(
        ctx,
        &attention.kv_b_proj,
        &scratch.latent,
        &mut scratch.attn_out,
    )?;
    gemm_graphsafe_into_checked(
        ctx,
        &attention.o_proj,
        &scratch.attn_out,
        &mut scratch.projected,
    )?;
    ensure!(
        scratch.projected.seq_len == *batch_size,
        "Kimi decode projected batch mismatch"
    );
    Ok(())
}

fn forward_dense_mlp_batch_into(
    ctx: &DeviceContext,
    comm: &Comm,
    dense: &KimiDenseForwardCache,
    post_attention_norm: &DeviceVec,
    hidden: &mut HiddenStates,
    normed: &mut HiddenStates,
    next_hidden: &mut HiddenStates,
) -> Result<()> {
    rms_norm_batch_into(
        ctx,
        hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        normed,
    );
    let local_intermediate = dense.gate_up_proj.rows / 2;
    let mut gate_up = HiddenStates::zeros(ctx, dense.gate_up_proj.rows, hidden.seq_len)?;
    let mut activated = HiddenStates::zeros(ctx, local_intermediate, hidden.seq_len)?;
    let mut mlp_out = HiddenStates::zeros(ctx, KIMI_K2_HIDDEN, hidden.seq_len)?;
    gemm_into_checked(ctx, &dense.gate_up_proj, normed, &mut gate_up)?;
    silu_mul_fused_batch_into(ctx, &gate_up, &mut activated);
    gemm_into_checked(ctx, &dense.down_proj, &activated, &mut mlp_out)?;
    all_reduce_hidden_in_place(&mut mlp_out, comm)?;
    add_batch_into(ctx, hidden, &mlp_out, next_hidden)?;
    std::mem::swap(hidden, next_hidden);
    Ok(())
}

fn forward_dense_mlp_decode_into(
    ctx: &DeviceContext,
    comm: &Comm,
    dense: &KimiDenseForwardCache,
    post_attention_norm: &DeviceVec,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    ensure!(
        (1..=KIMI_DECODE_MAX_BATCH).contains(&scratch.hidden.seq_len),
        "Kimi dense decode scratch seq_len {} outside supported range 1..={}",
        scratch.hidden.seq_len,
        KIMI_DECODE_MAX_BATCH
    );
    rms_norm_batch_into(
        ctx,
        &scratch.hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.normed,
    );
    gemm_graphsafe_into_checked(
        ctx,
        &dense.gate_up_proj,
        &scratch.normed,
        &mut scratch.dense_gate_up,
    )?;
    silu_mul_fused_batch_into(ctx, &scratch.dense_gate_up, &mut scratch.dense_activated);
    gemm_graphsafe_into_checked(
        ctx,
        &dense.down_proj,
        &scratch.dense_activated,
        &mut scratch.projected,
    )?;
    all_reduce_hidden_via_f32_in_place(
        ctx,
        &mut scratch.projected,
        &mut scratch.hidden_allreduce_f32,
        comm,
    )?;
    add_batch_into(
        ctx,
        &scratch.hidden,
        &scratch.projected,
        &mut scratch.normed,
    )?;
    std::mem::swap(&mut scratch.hidden, &mut scratch.normed);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_moe_layer_batch_into(
    ctx: &DeviceContext,
    comm: &Comm,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    post_attention_norm: &DeviceVec,
    expert_kernels: &KimiRankExpertMarlinWeights,
    hidden: &mut HiddenStates,
    normed: &mut HiddenStates,
    next_hidden: &mut HiddenStates,
) -> Result<()> {
    let seq_len = hidden.seq_len;
    rms_norm_batch_into(
        ctx,
        hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        normed,
    );

    let mut shared_gate_up = HiddenStates::zeros(ctx, moe.shared_gate_up_proj.rows, seq_len)?;
    let mut shared_activated = HiddenStates::zeros(ctx, moe.shared_gate_up_proj.rows / 2, seq_len)?;
    let mut shared_out = HiddenStates::zeros(ctx, KIMI_K2_HIDDEN, seq_len)?;
    gemm_into_checked(ctx, &moe.shared_gate_up_proj, normed, &mut shared_gate_up)?;
    silu_mul_fused_batch_into(ctx, &shared_gate_up, &mut shared_activated);
    gemm_into_checked(
        ctx,
        &moe.shared_down_proj,
        &shared_activated,
        &mut shared_out,
    )?;
    all_reduce_hidden_in_place(&mut shared_out, comm)?;

    let mut router_logits = ctx.stream.alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_scores = ctx.stream.alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_choice_scores = ctx.stream.alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_topk_weight = ctx.stream.alloc_zeros(seq_len * KIMI_K2_TOPK)?;
    let mut router_topk_idx = ctx.stream.alloc_zeros(seq_len * KIMI_K2_TOPK)?;
    {
        let mut scratch = KimiRouterScratch {
            logits: &mut router_logits,
            scores: &mut router_scores,
            choice_scores: &mut router_choice_scores,
        };
        let mut output = KimiRouterOutput {
            topk_weight: &mut router_topk_weight,
            topk_idx: &mut router_topk_idx,
        };
        kimi_router_noaux_tc_launch(
            ctx,
            KimiRouterConfig::kimi_k2(),
            KimiRouterBatch {
                batch_size: seq_len,
                active_tokens: seq_len,
                padded_tokens: seq_len,
            },
            normed,
            &moe.router.gate_weight,
            &moe.router.e_score_correction_bias,
            &mut scratch,
            &mut output,
        )?;
    }

    let marlin_block_size = kimi_marlin_block_size(seq_len);
    let mut route_workspace = KimiMarlinRouteWorkspace::new(ctx, seq_len, marlin_block_size)?;
    let routing = kimi_moe_marlin_align_block_size(
        ctx,
        &mut route_workspace,
        &router_topk_idx,
        seq_len,
        seq_len,
        expert_kernels.local_expert_range.start,
    )?;
    let layer_weights = expert_kernels
        .layers
        .iter()
        .find(|layer| layer.layer_idx == layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!("Kimi rank expert Marlin package missing layer {layer_idx}")
        })?
        .as_marlin_weights();

    let mut marlin_workspace = KimiMarlinWna16Workspace::new(
        ctx,
        routing.max_m_blocks,
        KIMI_K2_HIDDEN,
        marlin_block_size,
    )?;
    let mut w13_out =
        HiddenStates::zeros(ctx, 2 * KIMI_K2_EXPERT_INTERMEDIATE, routing.route_elems)?;
    kimi_marlin_wna16_w13_gemm(
        ctx,
        &mut marlin_workspace,
        &routing,
        normed,
        &layer_weights.w13,
        &router_topk_weight,
        &mut w13_out,
    )?;
    let mut activated = HiddenStates::zeros(ctx, KIMI_K2_EXPERT_INTERMEDIATE, routing.route_elems)?;
    kimi_marlin_w13_swiglu(ctx, &w13_out, &mut activated)?;

    let mut expert_output = HiddenStates::zeros(ctx, KIMI_K2_HIDDEN, routing.route_elems)?;
    kimi_marlin_wna16_w2_gemm(
        ctx,
        &mut marlin_workspace,
        &routing,
        &activated,
        &layer_weights.w2_down,
        &router_topk_weight,
        &mut expert_output,
    )?;

    let mut routed_out_f32 = ctx.stream.alloc_zeros(seq_len * KIMI_K2_HIDDEN)?;
    kimi_marlin_sum_topk_rows_f32(ctx, &expert_output, seq_len, &mut routed_out_f32)?;
    all_reduce_f32_in_place(&mut routed_out_f32, comm)?;
    scale_f32_in_place(
        ctx,
        &mut routed_out_f32,
        seq_len * KIMI_K2_HIDDEN,
        KIMI_K2_ROUTER_SCALE,
    )?;
    add_batch_into(ctx, hidden, &shared_out, next_hidden)?;
    add_f32_bf16_to_bf16_hidden_into(ctx, &routed_out_f32, next_hidden, hidden)?;
    Ok(())
}

fn forward_moe_layer_decode_into(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    comm: &Comm,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    post_attention_norm: &DeviceVec,
    expert_kernels: &KimiRankExpertMarlinWeights,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    let seq_len = scratch.hidden.seq_len;
    ensure!(
        (1..=KIMI_DECODE_MAX_BATCH).contains(&seq_len),
        "Kimi MoE decode scratch seq_len {seq_len} outside supported range 1..={KIMI_DECODE_MAX_BATCH}"
    );
    rms_norm_batch_into(
        ctx,
        &scratch.hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.normed,
    );
    let norm_ready = ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE layer {layer_idx} record norm_ready"))?;
    aux_ctx
        .stream
        .wait(&norm_ready)
        .with_context(|| format!("Kimi MoE layer {layer_idx} aux wait norm_ready"))?;

    gemm_graphsafe_into_checked(
        ctx,
        &moe.shared_gate_up_proj,
        &scratch.normed,
        &mut scratch.shared_gate_up,
    )?;
    silu_mul_fused_batch_into(ctx, &scratch.shared_gate_up, &mut scratch.shared_activated);
    gemm_graphsafe_into_checked(
        ctx,
        &moe.shared_down_proj,
        &scratch.shared_activated,
        &mut scratch.projected,
    )?;
    all_reduce_hidden_via_f32_in_place(
        ctx,
        &mut scratch.projected,
        &mut scratch.hidden_allreduce_f32,
        comm,
    )?;

    {
        let mut router_scratch = KimiRouterScratch {
            logits: &mut scratch.router_logits,
            scores: &mut scratch.router_scores,
            choice_scores: &mut scratch.router_choice_scores,
        };
        let mut router_output = KimiRouterOutput {
            topk_weight: &mut scratch.router_topk_weight,
            topk_idx: &mut scratch.router_topk_idx,
        };
        kimi_router_noaux_tc_launch(
            aux_ctx,
            KimiRouterConfig::kimi_k2(),
            KimiRouterBatch {
                batch_size: seq_len,
                active_tokens: seq_len,
                padded_tokens: seq_len,
            },
            &scratch.normed,
            &moe.router.gate_weight,
            &moe.router.e_score_correction_bias,
            &mut router_scratch,
            &mut router_output,
        )?;
    }

    let routing = kimi_moe_marlin_align_block_size(
        aux_ctx,
        &mut scratch.marlin_route_workspace,
        &scratch.router_topk_idx,
        seq_len,
        seq_len,
        expert_kernels.local_expert_range.start,
    )?;
    let layer_weights = expert_kernels
        .layers
        .iter()
        .find(|layer| layer.layer_idx == layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!("Kimi rank expert Marlin package missing layer {layer_idx}")
        })?
        .as_marlin_weights();

    aux_ctx
        .stream
        .memset_zeros(&mut scratch.marlin_w13_out.data)?;
    kimi_marlin_wna16_w13_gemm(
        aux_ctx,
        &mut scratch.marlin_workspace,
        &routing,
        &scratch.normed,
        &layer_weights.w13,
        &scratch.router_topk_weight,
        &mut scratch.marlin_w13_out,
    )?;
    kimi_marlin_w13_swiglu(
        aux_ctx,
        &scratch.marlin_w13_out,
        &mut scratch.marlin_activated,
    )?;
    aux_ctx
        .stream
        .memset_zeros(&mut scratch.marlin_expert_output.data)?;
    kimi_marlin_wna16_w2_gemm(
        aux_ctx,
        &mut scratch.marlin_workspace,
        &routing,
        &scratch.marlin_activated,
        &layer_weights.w2_down,
        &scratch.router_topk_weight,
        &mut scratch.marlin_expert_output,
    )?;
    kimi_marlin_sum_topk_rows_f32(
        aux_ctx,
        &scratch.marlin_expert_output,
        seq_len,
        &mut scratch.routed_out_f32,
    )?;
    repeat_f32_for_reduce_scatter_into(
        aux_ctx,
        &scratch.routed_out_f32,
        &mut scratch.routed_reduce_scatter_send_f32,
        seq_len * KIMI_K2_HIDDEN,
        KIMI_K2_EP_WORLD,
    )?;
    let routed_local_done = aux_ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE layer {layer_idx} record routed_local_done"))?;
    ctx.stream
        .wait(&routed_local_done)
        .with_context(|| format!("Kimi MoE layer {layer_idx} main wait routed_local_done"))?;
    reduce_scatter_f32_hidden_into(
        &scratch.routed_reduce_scatter_send_f32,
        seq_len * KIMI_K2_EP_WORLD,
        KIMI_K2_HIDDEN,
        &mut scratch.routed_out_f32,
        seq_len,
        KIMI_K2_EP_WORLD,
        comm,
    )?;
    add_batch_into(
        ctx,
        &scratch.hidden,
        &scratch.projected,
        &mut scratch.normed,
    )?;
    scaled_add_f32_bf16_to_bf16_hidden_into(
        ctx,
        &scratch.routed_out_f32,
        KIMI_K2_ROUTER_SCALE,
        &scratch.normed,
        &mut scratch.hidden,
    )?;
    Ok(())
}

fn load_layer_forward_cache(
    ctx: &KimiRankGpuContext,
    weights: &KimiRankGpuWeights,
    layer: &KimiLayerWeightNames,
) -> Result<KimiLayerForwardCache> {
    let q_a_proj = raw_tensor(weights, &layer.attention.q_a_proj)?
        .copy_bf16_matrix_from_shape(ctx, "attention_q_a_proj")?;
    ensure!(
        q_a_proj.rows == KIMI_K2_Q_LORA_RANK && q_a_proj.cols == KIMI_K2_HIDDEN,
        "layer {} q_a_proj shape must be [{}, {}], got [{}, {}]",
        layer.layer_idx,
        KIMI_K2_Q_LORA_RANK,
        KIMI_K2_HIDDEN,
        q_a_proj.rows,
        q_a_proj.cols
    );
    let kv_a_proj_with_mqa = raw_tensor(weights, &layer.attention.kv_a_proj_with_mqa)?
        .copy_bf16_matrix_from_shape(ctx, "attention_kv_a_proj_with_mqa")?;
    ensure!(
        kv_a_proj_with_mqa.rows == KIMI_K2_MLA_KV_A_OUT
            && kv_a_proj_with_mqa.cols == KIMI_K2_HIDDEN,
        "layer {} kv_a_proj_with_mqa shape must be [{}, {}], got [{}, {}]",
        layer.layer_idx,
        KIMI_K2_MLA_KV_A_OUT,
        KIMI_K2_HIDDEN,
        kv_a_proj_with_mqa.rows,
        kv_a_proj_with_mqa.cols
    );
    let device_ctx = ctx.as_device_context();
    let fused_qkv_a_proj = DeviceMatrix::vstack(&device_ctx, &[&q_a_proj, &kv_a_proj_with_mqa])?;
    let attention = KimiAttentionForwardCache {
        input_norm: raw_tensor(weights, &layer.attention.input_layernorm)?.copy_bf16_vec(
            ctx,
            KIMI_K2_HIDDEN,
            "attention_input_norm",
        )?,
        fused_qkv_a_proj,
        q_a_norm: raw_tensor(weights, &layer.attention.q_a_layernorm)?.copy_bf16_vec(
            ctx,
            KIMI_K2_Q_LORA_RANK,
            "attention_q_a_norm",
        )?,
        q_b_proj: raw_tensor(weights, &layer.attention.q_b_proj)?
            .copy_bf16_matrix_from_shape(ctx, "attention_q_b_proj")?,
        kv_a_norm: raw_tensor(weights, &layer.attention.kv_a_layernorm)?.copy_bf16_vec(
            ctx,
            KIMI_K2_MLA_KV_LORA_RANK,
            "attention_kv_a_norm",
        )?,
        kv_b_proj: raw_tensor(weights, &layer.attention.kv_b_proj)?
            .copy_bf16_matrix_from_shape(ctx, "attention_kv_b_proj")?,
        o_proj: raw_tensor(weights, &layer.attention.o_proj)?
            .copy_bf16_matrix_from_shape(ctx, "attention_o_proj")?,
        post_attention_norm: raw_tensor(weights, &layer.attention.post_attention_layernorm)?
            .copy_bf16_vec(ctx, KIMI_K2_HIDDEN, "post_attention_norm")?,
    };
    ensure_attention_shapes(layer.layer_idx, &attention)?;

    let kind = match &layer.kind {
        KimiLayerWeightKindNames::Dense(mlp) => {
            let gate_proj = raw_tensor(weights, &mlp.gate_proj)?
                .copy_bf16_matrix_from_shape(ctx, "dense_gate_proj")?;
            let up_proj = raw_tensor(weights, &mlp.up_proj)?
                .copy_bf16_matrix_from_shape(ctx, "dense_up_proj")?;
            let down_proj = raw_tensor(weights, &mlp.down_proj)?
                .copy_bf16_matrix_from_shape(ctx, "dense_down_proj")?;
            ensure_dense_mlp_shapes("dense_mlp", &gate_proj, &up_proj, &down_proj)?;
            ensure!(
                gate_proj.rows == KIMI_K2_DENSE_INTERMEDIATE / 8,
                "dense gate local rows must be {}, got {}",
                KIMI_K2_DENSE_INTERMEDIATE / 8,
                gate_proj.rows
            );
            let gate_up_proj = DeviceMatrix::vstack(&device_ctx, &[&gate_proj, &up_proj])?;
            KimiLayerForwardKindCache::Dense(KimiDenseForwardCache {
                gate_up_proj,
                down_proj,
            })
        }
        KimiLayerWeightKindNames::Moe(moe) => {
            let router = KimiRouterGpuWeights {
                gate_weight: raw_tensor(weights, &moe.router.gate_weight)?,
                e_score_correction_bias: raw_tensor(weights, &moe.router.e_score_correction_bias)?,
            }
            .copy_to_device_weights(ctx)?;
            let shared_gate_proj = raw_tensor(weights, &moe.shared_experts.gate_proj)?
                .copy_bf16_matrix_from_shape(ctx, "shared_gate_proj")?;
            let shared_up_proj = raw_tensor(weights, &moe.shared_experts.up_proj)?
                .copy_bf16_matrix_from_shape(ctx, "shared_up_proj")?;
            let shared_down_proj = raw_tensor(weights, &moe.shared_experts.down_proj)?
                .copy_bf16_matrix_from_shape(ctx, "shared_down_proj")?;
            ensure_dense_mlp_shapes(
                "shared_expert",
                &shared_gate_proj,
                &shared_up_proj,
                &shared_down_proj,
            )?;
            let shared_gate_up_proj =
                DeviceMatrix::vstack(&device_ctx, &[&shared_gate_proj, &shared_up_proj])?;
            KimiLayerForwardKindCache::Moe(KimiMoeForwardCache {
                router,
                shared_gate_up_proj,
                shared_down_proj,
            })
        }
    };

    Ok(KimiLayerForwardCache {
        layer_idx: layer.layer_idx,
        attention,
        kind,
    })
}

fn raw_tensor<'a>(weights: &'a KimiRankGpuWeights, name: &str) -> Result<&'a KimiGpuRawTensor> {
    weights
        .tensors
        .get(name)
        .with_context(|| format!("missing Kimi forward tensor {name}"))
}

fn ensure_dense_mlp_shapes(
    label: &str,
    gate: &DeviceMatrix,
    up: &DeviceMatrix,
    down: &DeviceMatrix,
) -> Result<()> {
    ensure!(
        gate.cols == KIMI_K2_HIDDEN,
        "{label} gate input dim must be {}, got {}",
        KIMI_K2_HIDDEN,
        gate.cols
    );
    ensure!(
        up.rows == gate.rows && up.cols == gate.cols,
        "{label} up shape [{}, {}] must match gate [{}, {}]",
        up.rows,
        up.cols,
        gate.rows,
        gate.cols
    );
    ensure!(
        down.rows == KIMI_K2_HIDDEN && down.cols == gate.rows,
        "{label} down shape [{}, {}] must be [{}, {}]",
        down.rows,
        down.cols,
        KIMI_K2_HIDDEN,
        gate.rows
    );
    Ok(())
}

fn ensure_attention_shapes(layer_idx: usize, attention: &KimiAttentionForwardCache) -> Result<()> {
    ensure!(
        attention.fused_qkv_a_proj.rows == KIMI_K2_MLA_QKV_A_OUT
            && attention.fused_qkv_a_proj.cols == KIMI_K2_HIDDEN,
        "layer {layer_idx} fused_qkv_a_proj shape must be [{}, {}], got [{}, {}]",
        KIMI_K2_MLA_QKV_A_OUT,
        KIMI_K2_HIDDEN,
        attention.fused_qkv_a_proj.rows,
        attention.fused_qkv_a_proj.cols
    );
    ensure!(
        attention.q_b_proj.rows == KIMI_K2_MLA_Q_LOCAL_OUT_TP8
            && attention.q_b_proj.cols == KIMI_K2_Q_LORA_RANK,
        "layer {layer_idx} q_b_proj shape must be [{}, {}], got [{}, {}]",
        KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
        KIMI_K2_Q_LORA_RANK,
        attention.q_b_proj.rows,
        attention.q_b_proj.cols
    );
    ensure!(
        attention.kv_b_proj.rows == KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8
            && attention.kv_b_proj.cols == KIMI_K2_MLA_KV_LORA_RANK,
        "layer {layer_idx} kv_b_proj shape must be [{}, {}], got [{}, {}]",
        KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
        KIMI_K2_MLA_KV_LORA_RANK,
        attention.kv_b_proj.rows,
        attention.kv_b_proj.cols
    );
    ensure!(
        attention.o_proj.rows == KIMI_K2_HIDDEN
            && attention.o_proj.cols == KIMI_K2_MLA_O_LOCAL_IN_TP8,
        "layer {layer_idx} o_proj shape must be [{}, {}], got [{}, {}]",
        KIMI_K2_HIDDEN,
        KIMI_K2_MLA_O_LOCAL_IN_TP8,
        attention.o_proj.rows,
        attention.o_proj.cols
    );
    Ok(())
}

fn all_reduce_hidden_in_place(hidden: &mut HiddenStates, comm: &Comm) -> Result<()> {
    comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum)
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0))
}

fn all_reduce_hidden_via_f32_in_place(
    ctx: &DeviceContext,
    hidden: &mut HiddenStates,
    f32_scratch: &mut CudaSlice<f32>,
    comm: &Comm,
) -> Result<()> {
    ensure!(
        f32_scratch.len() >= hidden.data.len(),
        "Kimi deterministic all-reduce scratch len {} < hidden len {}",
        f32_scratch.len(),
        hidden.data.len()
    );
    bf16_hidden_to_f32_into(ctx, hidden, f32_scratch)?;
    all_reduce_f32_bulk_in_place(f32_scratch, hidden.seq_len, hidden.hidden_dim, comm)?;
    f32_to_bf16_hidden_into(ctx, f32_scratch, hidden)
}

fn all_reduce_f32_in_place(values: &mut CudaSlice<f32>, comm: &Comm) -> Result<()> {
    comm.all_reduce_in_place(values, &ReduceOp::Sum)
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("Kimi TP/EP f32 sum failed: status={:?}", err.0))
}

fn all_reduce_f32_bulk_in_place(
    values: &mut CudaSlice<f32>,
    rows: usize,
    row_len: usize,
    comm: &Comm,
) -> Result<()> {
    ensure!(
        values.len() >= rows * row_len,
        "Kimi bulk f32 all-reduce len {} < rows {} * row_len {}",
        values.len(),
        rows,
        row_len
    );
    let mut view = values.slice_mut(0..rows * row_len);
    comm.all_reduce_in_place(&mut view, &ReduceOp::Sum)
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("Kimi bulk f32 all-reduce failed: status={:?}", err.0))
}

fn reduce_scatter_f32_hidden_into(
    global: &CudaSlice<f32>,
    global_rows: usize,
    row_len: usize,
    local: &mut CudaSlice<f32>,
    local_rows: usize,
    world_size: usize,
    comm: &Comm,
) -> Result<()> {
    ensure!(world_size > 0, "Kimi RS world size must be positive");
    ensure!(
        global_rows == local_rows * world_size,
        "Kimi RS rows mismatch: global_rows={global_rows}, local_rows={local_rows}, world_size={world_size}"
    );
    ensure!(
        global.len() >= global_rows * row_len,
        "Kimi RS global buffer too small: have {}, need {}",
        global.len(),
        global_rows * row_len
    );
    ensure!(
        local.len() >= local_rows * row_len,
        "Kimi RS local buffer too small: have {}, need {}",
        local.len(),
        local_rows * row_len
    );
    let global_view = global.slice(0..global_rows * row_len);
    let mut local_view = local.slice_mut(0..local_rows * row_len);
    comm.reduce_scatter(&global_view, &mut local_view, &ReduceOp::Sum)
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("Kimi routed RS f32 hidden failed: status={:?}", err.0))
}

fn kimi_mla_softmax_scale() -> f32 {
    let base = (KIMI_K2_MLA_Q_HEAD_DIM as f32).sqrt().recip();
    let mscale = yarn_get_mscale(KIMI_K2_YARN_FACTOR, 1.0);
    base * mscale * mscale
}

fn kimi_marlin_block_size(active_tokens: usize) -> usize {
    for block_size in [8usize, 16, 32, 48, KIMI_MARLIN_MAX_BLOCK_SIZE] {
        let routes_per_expert_block = active_tokens as f32 * KIMI_K2_TOPK as f32
            / KIMI_K2_EP8_LOCAL_EXPERTS as f32
            / block_size as f32;
        if routes_per_expert_block < 0.9 {
            return block_size;
        }
    }
    KIMI_MARLIN_MAX_BLOCK_SIZE
}

fn build_yarn_rope_cache(seq_len: usize) -> (Vec<half::bf16>, Vec<half::bf16>) {
    let dim = KIMI_K2_QK_ROPE_HEAD_DIM;
    let half_dim = dim / 2;
    let (low, high) = yarn_find_correction_range(
        KIMI_K2_YARN_BETA_FAST,
        KIMI_K2_YARN_BETA_SLOW,
        dim,
        KIMI_K2_ROPE_THETA,
        KIMI_K2_YARN_ORIGINAL_MAX_POS,
    );
    let rope_mscale =
        yarn_get_mscale(KIMI_K2_YARN_FACTOR, 1.0) / yarn_get_mscale(KIMI_K2_YARN_FACTOR, 1.0);
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let exponent = (2 * i) as f32 / dim as f32;
        let freq_extra = 1.0 / KIMI_K2_ROPE_THETA.powf(exponent);
        let freq_inter = 1.0 / (KIMI_K2_YARN_FACTOR * KIMI_K2_ROPE_THETA.powf(exponent));
        let ramp = yarn_linear_ramp_mask(i as f32, low as f32, high as f32);
        inv_freq.push(freq_inter * ramp + freq_extra * (1.0 - ramp));
    }

    let mut cos = vec![half::bf16::from_f32(0.0); seq_len * dim];
    let mut sin = vec![half::bf16::from_f32(0.0); seq_len * dim];
    for token in 0..seq_len {
        for i in 0..half_dim {
            let freq = token as f32 * inv_freq[i];
            let c = half::bf16::from_f32(freq.cos() * rope_mscale);
            let s = half::bf16::from_f32(freq.sin() * rope_mscale);
            cos[token * dim + i] = c;
            sin[token * dim + i] = s;
            cos[token * dim + half_dim + i] = c;
            sin[token * dim + half_dim + i] = s;
        }
    }
    (cos, sin)
}

fn yarn_find_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> (usize, usize) {
    let low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
    let high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
    (
        low.max(0.0) as usize,
        (high as usize).min(dim.saturating_sub(1)),
    )
}

fn yarn_find_correction_dim(
    num_rotations: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> f32 {
    (dim as f32
        * (max_position_embeddings as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * base.ln())
}

fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

fn yarn_linear_ramp_mask(value: f32, min: f32, max: f32) -> f32 {
    let denom = if (max - min).abs() < f32::EPSILON {
        0.001
    } else {
        max - min
    };
    ((value - min) / denom).clamp(0.0, 1.0)
}

fn add_f32_bf16_to_bf16_hidden_into(
    ctx: &DeviceContext,
    a: &CudaSlice<f32>,
    b: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    let elems = b.hidden_dim * b.seq_len;
    ensure!(
        a.len() >= elems,
        "Kimi f32 add input too small: have {}, need {}",
        a.len(),
        elems
    );
    ensure!(
        out.hidden_dim == b.hidden_dim && out.seq_len == b.seq_len,
        "Kimi f32 add output shape mismatch: out=[{}, {}], b=[{}, {}]",
        out.hidden_dim,
        out.seq_len,
        b.hidden_dim,
        b.seq_len
    );
    let (a_ptr, _a_guard) = a.device_ptr(&ctx.stream);
    let (b_ptr, _b_guard) = b.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_add_f32_bf16_to_bf16_cuda(
            a_ptr as *const f32,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            elems as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn scaled_add_f32_bf16_to_bf16_hidden_into(
    ctx: &DeviceContext,
    a: &CudaSlice<f32>,
    scale: f32,
    b: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    kimi_scaled_add_f32_bf16_to_bf16(ctx, a, scale, b, out)
}

fn sample_local_top1_with_value(ctx: &DeviceContext, logits: &DeviceVec) -> Result<(u32, f32)> {
    let mut top1_value_scratch = ctx.stream.alloc_zeros::<half::bf16>(1)?;
    let mut row_states_scratch = ctx
        .stream
        .alloc_zeros::<u8>(flashinfer_topk_row_states_bytes())?;
    let mut out = ctx.stream.alloc_zeros::<i32>(1)?;
    sample_local_top1_with_value_reuse(
        ctx,
        logits,
        &mut top1_value_scratch,
        &mut row_states_scratch,
        &mut out,
    )
}

fn sample_local_top1_with_value_reuse(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
) -> Result<(u32, f32)> {
    ensure!(
        top1_value_scratch.len() >= 1 && out.len() >= 1,
        "Kimi top1 scratch must have scalar value/id outputs"
    );
    ensure!(
        row_states_scratch.len() >= flashinfer_topk_row_states_bytes(),
        "Kimi top1 row-states scratch too small: have {}, need {}",
        row_states_scratch.len(),
        flashinfer_topk_row_states_bytes()
    );
    {
        let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
        let (value_ptr, _value_guard) = top1_value_scratch.device_ptr_mut(&ctx.stream);
        let (row_states_ptr, _row_states_guard) = row_states_scratch.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::flashinfer_top1_cuda(
                logits_ptr as *const ffi::Half,
                value_ptr as *mut ffi::Half,
                row_states_ptr as *mut u8,
                out_ptr as *mut i32,
                logits.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    }
    ctx.sync()?;
    let top_id = ctx
        .stream
        .clone_dtoh(&*out)
        .map_err(|err| anyhow::anyhow!("D2H Kimi top1 id read failed: {err}"))?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Kimi top1 id output was empty"))?;
    let top_value = ctx
        .stream
        .clone_dtoh(&*top1_value_scratch)
        .map_err(|err| anyhow::anyhow!("D2H Kimi top1 value read failed: {err}"))?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Kimi top1 value output was empty"))?
        .to_f32();
    ensure!(
        top_id >= 0 && (top_id as usize) < logits.len,
        "Kimi local top1 id {} out of logits range {}",
        top_id,
        logits.len
    );
    Ok((top_id as u32, top_value))
}

fn launch_local_top1_batch(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    active_rows: usize,
    top1_values: &mut CudaSlice<half::bf16>,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        active_rows > 0 && active_rows <= logits.seq_len,
        "Kimi batched top1 active_rows {active_rows} must be in 1..={}",
        logits.seq_len
    );
    ensure!(
        top1_values.len() >= active_rows && out.len() >= active_rows,
        "Kimi batched top1 scratch too small: values={}, out={}, active_rows={}",
        top1_values.len(),
        out.len(),
        active_rows
    );
    {
        let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
        let (value_ptr, _value_guard) = top1_values.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_batch_bf16_cuda(
                logits_ptr as *const ffi::Half,
                value_ptr as *mut ffi::Half,
                out_ptr as *mut i32,
                active_rows as i32,
                logits.hidden_dim as i32,
                ctx.stream.cu_stream(),
            );
        }
    }
    Ok(())
}

fn read_local_top1_batch_values(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    active_rows: usize,
    top1_values: &mut CudaSlice<half::bf16>,
    out: &mut CudaSlice<i32>,
) -> Result<Vec<(u32, f32)>> {
    ctx.sync()?;
    let top_ids = ctx
        .stream
        .clone_dtoh(&*out)
        .map_err(|err| anyhow::anyhow!("D2H Kimi batched top1 ids read failed: {err}"))?;
    let top_values = ctx
        .stream
        .clone_dtoh(&*top1_values)
        .map_err(|err| anyhow::anyhow!("D2H Kimi batched top1 values read failed: {err}"))?;
    let mut rows = Vec::with_capacity(active_rows);
    for row in 0..active_rows {
        let top_id = top_ids[row];
        ensure!(
            top_id >= 0 && (top_id as usize) < logits.hidden_dim,
            "Kimi batched local top1 id {} at row {} out of logits range {}",
            top_id,
            row,
            logits.hidden_dim
        );
        rows.push((top_id as u32, top_values[row].to_f32()));
    }
    Ok(rows)
}

impl KimiRankWeightLoadReport {
    fn from_loaded_weights(
        tensor_count: usize,
        total_bytes: usize,
        expert_kernel_weights: &KimiRankExpertMarlinWeights,
    ) -> Self {
        Self {
            rank: expert_kernel_weights.rank,
            tensor_count,
            total_bytes,
            expert_kernel_layers: expert_kernel_weights.layers.len(),
            expert_kernel_total_bytes: expert_kernel_weights.total_bytes,
            loaded_to_gpu: true,
            typed_view_validated: true,
            expert_kernel_weights_packaged: true,
        }
    }
}

pub(super) fn build_tp8_ep8_placements(
    device_ordinals: &[usize],
) -> Result<Vec<KimiK2RankPlacement>> {
    if device_ordinals.len() != 8 {
        bail!(
            "Kimi-K2 TP8/EP8 requires exactly 8 device ordinals, got {:?}",
            device_ordinals
        );
    }
    device_ordinals
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, device_ordinal)| KimiK2RankPlacement::new(rank, device_ordinal))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_page_metadata_uses_multiple_pages_per_request() {
        let (page_indices, page_indptr, last_page_len) =
            build_decode_append_page_metadata(4, 16, 128, &[26, 0, 0, 0]).unwrap();
        assert_eq!(page_indptr, vec![0, 2, 3, 4, 5]);
        assert_eq!(&page_indices[..5], &[0, 1, 128, 256, 384]);
        assert_eq!(last_page_len, vec![11, 1, 1, 1]);

        let (_, page_indptr, last_page_len) =
            build_decode_append_page_metadata(4, 16, 128, &[27, 0, 0, 0]).unwrap();
        assert_eq!(page_indptr, vec![0, 2, 3, 4, 5]);
        assert_eq!(last_page_len[0], 12);
    }
}
