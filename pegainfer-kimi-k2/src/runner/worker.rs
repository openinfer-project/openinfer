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
    ops::{
        KIMI_K2_MLA_KV_A_OUT, KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_LORA_RANK,
        KIMI_K2_MLA_LOCAL_HEADS_TP8, KIMI_K2_MLA_O_LOCAL_IN_TP8, KIMI_K2_MLA_Q_HEAD_DIM,
        KIMI_K2_MLA_Q_LOCAL_OUT_TP8, KIMI_K2_MLA_QKV_A_OUT, KIMI_K2_MLA_ROPE_DIM,
        KIMI_K2_MLA_V_HEAD_DIM, KIMI_K2_ROUTER_SCALE, KimiMarlinRouteWorkspace,
        KimiMarlinWna16Workspace, KimiMlaPagedKvLayout, KimiRouterBatch, KimiRouterConfig,
        KimiRouterOutput, KimiRouterScratch, flashinfer_topk_row_states_bytes,
        kimi_add_f32_bf16_to_bf16, kimi_flashinfer_batch_decode_mla,
        kimi_flashinfer_single_prefill_mla, kimi_marlin_sum_topk_rows_f32, kimi_marlin_w13_swiglu,
        kimi_marlin_wna16_w2_gemm, kimi_marlin_wna16_w13_gemm, kimi_mla_absorb_q_nope,
        kimi_mla_paged_kv_append, kimi_mla_rope_apply_kpe, kimi_mla_rope_assemble_prefill,
        kimi_mla_rope_split_decode, kimi_mla_split_qkv_a, kimi_mla_v_up,
        kimi_moe_marlin_align_block_size, kimi_router_noaux_tc_launch,
        kimi_scaled_add_f32_bf16_to_bf16, repeat_f32_for_reduce_scatter_into, scale_f32_in_place,
    },
    tensor::{
        DeviceContext, DeviceMatrix, DeviceVec, GpuTensor, GpuWeight, HiddenStates, NormWeight,
    },
    typed_ops,
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

pub(super) use crate::typed_scratch::{
    DENSE_ACTIVATED_DIM, DENSE_GATE_UP_DIM, KimiWorkerDecodeScratch, MARLIN_W13_OUT_DIM,
    SHARED_ACTIVATED_DIM, SHARED_GATE_UP_DIM,
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
    #[cfg(feature = "pplx-ep")]
    EnablePplx {
        ep_backend: pegainfer_comm::EpBackend,
        resp: SyncSender<Result<()>>,
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

    #[cfg(feature = "pplx-ep")]
    pub(super) fn enable_pplx_async(
        &self,
        ep_backend: pegainfer_comm::EpBackend,
    ) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        self.tx
            .send(KimiRankCommand::EnablePplx {
                ep_backend,
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
    #[cfg(feature = "pplx-ep")]
    ep_backend: Option<pegainfer_comm::EpBackend>,
    #[cfg(feature = "pplx-ep")]
    moe_pplx_scratch: Option<super::moe_pplx::KimiMoePplxScratch>,
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
    token_embedding: GpuTensor<KIMI_K2_HIDDEN>,
    final_norm: NormWeight<KIMI_K2_HIDDEN>,
    lm_head: GpuTensor<KIMI_K2_HIDDEN>,
    layers: Vec<KimiLayerForwardCache>,
}

struct KimiLayerForwardCache {
    layer_idx: usize,
    attention: KimiAttentionForwardCache,
    kind: KimiLayerForwardKindCache,
}

struct KimiAttentionForwardCache {
    input_norm: NormWeight<KIMI_K2_HIDDEN>,
    fused_qkv_a_proj: GpuWeight<KIMI_K2_MLA_QKV_A_OUT, KIMI_K2_HIDDEN>,
    q_a_norm: NormWeight<KIMI_K2_Q_LORA_RANK>,
    q_b_proj: GpuWeight<KIMI_K2_MLA_Q_LOCAL_OUT_TP8, KIMI_K2_Q_LORA_RANK>,
    kv_a_norm: NormWeight<KIMI_K2_MLA_KV_LORA_RANK>,
    kv_b_proj: GpuWeight<KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_LORA_RANK>,
    o_proj: GpuWeight<KIMI_K2_HIDDEN, KIMI_K2_MLA_O_LOCAL_IN_TP8>,
    post_attention_norm: NormWeight<KIMI_K2_HIDDEN>,
}

enum KimiLayerForwardKindCache {
    Dense(KimiDenseForwardCache),
    Moe(KimiMoeForwardCache),
}

struct KimiDenseForwardCache {
    gate_up_proj: GpuWeight<DENSE_GATE_UP_DIM, KIMI_K2_HIDDEN>,
    down_proj: GpuWeight<KIMI_K2_HIDDEN, DENSE_ACTIVATED_DIM>,
}

pub(super) struct KimiMoeForwardCache {
    pub(super) router: KimiRouterDeviceWeights,
    pub(super) shared_gate_up_proj: GpuWeight<SHARED_GATE_UP_DIM, KIMI_K2_HIDDEN>,
    pub(super) shared_down_proj: GpuWeight<KIMI_K2_HIDDEN, SHARED_ACTIVATED_DIM>,
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
        #[cfg(feature = "pplx-ep")]
        ep_backend: None,
        #[cfg(feature = "pplx-ep")]
        moe_pplx_scratch: None,
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
            #[cfg(feature = "pplx-ep")]
            KimiRankCommand::EnablePplx { ep_backend, resp } => {
                let result = state.enable_pplx(ep_backend);
                let _ = resp.send(result);
            }
            KimiRankCommand::Shutdown => break,
        }
    }
}

mod cache;
mod load;
mod runtime;
#[cfg(feature = "pplx-ep")]
pub(super) use runtime::all_reduce_hidden_via_f32_in_place;
mod state;

#[cfg(feature = "pplx-ep")]
struct PplxDecodeContext<'a> {
    ep: &'a mut pegainfer_comm::EpBackend,
    scratch: &'a mut super::moe_pplx::KimiMoePplxScratch,
}

mod forward;

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
    use super::cache::build_decode_append_page_metadata;

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
