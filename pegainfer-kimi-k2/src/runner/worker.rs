use std::{
    path::{Path, PathBuf},
    sync::{Arc, Barrier},
    thread,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use bytesize::ByteSize;
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::nccl::{
    ReduceOp,
    safe::{Comm, Id},
};
use log::debug;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::engine::TokenLogprob;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_trace;
use pegainfer_kernels::{
    ops::{
        KIMI_K2_LOCAL_EXPERTS, KIMI_K2_MLA_KV_A_OUT, KIMI_K2_MLA_KV_LORA_RANK,
        KIMI_K2_MLA_Q_HEAD_DIM, KIMI_K2_MLA_QKV_A_OUT, KIMI_K2_MLA_ROPE_DIM,
        KIMI_K2_MLA_V_HEAD_DIM, KIMI_O_PROJ_CUBLASLT_INPUT, KimiMarlinRouteWorkspace,
        KimiMarlinWna16Workspace, KimiMlaPagedKvLayout, flashinfer_topk_row_states_bytes,
        kimi_flashinfer_batch_decode_mla_rt, kimi_flashinfer_single_prefill_mla_rt,
        kimi_mla_absorb_q_nope_rt, kimi_mla_paged_kv_append, kimi_mla_rope_apply_kpe,
        kimi_mla_rope_assemble_prefill_rt, kimi_mla_rope_split_decode_rt, kimi_mla_split_qkv_a,
        kimi_mla_split_qkv_a_norm, kimi_mla_v_up_rt, kimi_o_proj_cublaslt_into,
        kimi_o_proj_cublaslt_supports_batch_size,
    },
    tensor::{
        DeviceContext, DeviceMatrix, DeviceVec, GpuTensor, GpuWeight, HiddenStates, NormWeight,
    },
    typed_ops,
};

use crate::{
    config::{
        KIMI_K2_DENSE_LAYERS, KIMI_K2_HIDDEN, KIMI_K2_LAYERS, KIMI_K2_MOE_LAYERS,
        KIMI_K2_Q_LORA_RANK, KIMI_K2_QK_ROPE_HEAD_DIM, KIMI_K2_RMS_NORM_EPS, KIMI_K2_ROPE_THETA,
        KIMI_K2_TOPK, KIMI_K2_YARN_BETA_FAST, KIMI_K2_YARN_BETA_SLOW, KIMI_K2_YARN_FACTOR,
        KIMI_K2_YARN_ORIGINAL_MAX_POS,
    },
    runner::affinity::{KimiRankThreadPlacement, pin_rank_worker_thread},
    weights::{
        KimiGpuRawTensor, KimiLayerWeightKindNames, KimiLayerWeightNames,
        KimiRankExpertMarlinWeights, KimiRankGpuContext, KimiRankGpuWeights,
        KimiRankSlicedLoadPlan, KimiRankWeightNames, KimiRouterDeviceWeights, KimiRouterGpuWeights,
        load_rank_sliced_weights_to_gpu,
    },
};

pub(super) use crate::typed_scratch::{KimiWorkerDecodeScratch, MARLIN_W13_OUT_DIM};

const KIMI_MARLIN_MAX_BLOCK_SIZE: usize = 64;
const KIMI_DECODE_MAX_BATCH: usize = 64;
const KIMI_DECODE_BATCH_BUCKETS: [usize; 7] = [1, 2, 4, 8, 16, 32, KIMI_DECODE_MAX_BATCH];
const KIMI_DECODE_PAGE_SIZE: usize = 16;
const KIMI_DECODE_PAGES_PER_REQUEST: usize = 128;
const KIMI_DECODE_ROPE_CACHE_TOKENS: usize = KIMI_DECODE_PAGE_SIZE * KIMI_DECODE_PAGES_PER_REQUEST;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct KimiK2RankPlacement {
    pub rank: usize,
    pub device_ordinal: usize,
}

impl KimiK2RankPlacement {
    pub(crate) fn new(rank: usize, device_ordinal: usize) -> Result<Self> {
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
    /// Exact log-softmax of the picked token plus the top-K, computed on the
    /// host from the full-vocab logits row. `Some` only when the request
    /// asked for logprobs (`GenerateRequest::logprobs > 0`); the serving
    /// path never pays for it.
    pub logprob: Option<TokenLogprob>,
}

enum KimiRankCommand {
    LoadSlicedWeights {
        model_path: PathBuf,
        resp: Sender<Result<KimiRankWeightLoadReport>>,
    },
    InitTpComm {
        id: Id,
        world_size: usize,
        resp: Sender<Result<()>>,
    },
    EnsureDecodeArena {
        decode_batch_size: usize,
        resp: Sender<Result<()>>,
    },
    ForwardPromptNextToken {
        slot: usize,
        decode_batch_size: usize,
        input_ids: Vec<u32>,
        ep_max_seq_len: usize,
        logprobs: usize,
        resp: Sender<Result<KimiOneTokenForwardReport>>,
    },
    ForwardDecodeBatchNextTokens {
        token_ids: Vec<u32>,
        append_positions: Vec<usize>,
        slots: Vec<usize>,
        decode_batch_size: usize,
        logprobs: Vec<usize>,
        resp: Sender<Result<Vec<KimiOneTokenForwardReport>>>,
    },
    EnablePplx {
        ep_backend: pegainfer_comm::EpBackend,
        resp: Sender<Result<()>>,
    },
    Shutdown,
}

pub(super) struct KimiRankWorker {
    placement: KimiK2RankPlacement,
    tx: Sender<KimiRankCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl KimiRankWorker {
    pub(super) fn spawn(
        placement: KimiK2RankPlacement,
        weight_names: KimiRankWeightNames,
        sliced_load_plan: KimiRankSlicedLoadPlan,
        thread_placement: KimiRankThreadPlacement,
        local_dims: crate::config::KimiLocalDims,
        ctx: KimiRankGpuContext,
        collective_barrier: Arc<Barrier>,
        enable_cuda_graph: bool,
    ) -> Result<Self> {
        ensure!(
            weight_names.rank == placement.rank,
            "Kimi rank weight names {} do not match placement {}",
            weight_names.rank,
            placement.rank
        );
        ensure!(
            sliced_load_plan.rank == placement.rank,
            "Kimi rank sliced load plan {} does not match placement {}",
            sliced_load_plan.rank,
            placement.rank
        );
        ensure!(
            thread_placement.rank == placement.rank,
            "Kimi rank thread placement {} does not match placement {}",
            thread_placement.rank,
            placement.rank
        );
        let (tx, rx) = unbounded();
        let (startup_tx, startup_rx) = bounded::<Result<()>>(1);
        let handle = thread::Builder::new()
            .name(format!("kimi-k2-rank-{}", placement.rank))
            .spawn(move || {
                pin_rank_worker_thread(&thread_placement);
                match bind_rank_thread(
                    ctx,
                    weight_names,
                    sliced_load_plan,
                    local_dims,
                    collective_barrier,
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
            tx,
            handle: Some(handle),
        })
    }

    pub(super) fn placement(&self) -> KimiK2RankPlacement {
        self.placement
    }

    pub(super) fn load_sliced_weights_async(
        &self,
        model_path: &Path,
    ) -> Result<Receiver<Result<KimiRankWeightLoadReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
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
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(KimiRankCommand::InitTpComm {
                id,
                world_size,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(super) fn ensure_decode_arena_async(
        &self,
        decode_batch_size: usize,
    ) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(KimiRankCommand::EnsureDecodeArena {
                decode_batch_size,
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
        ep_max_seq_len: usize,
        logprobs: usize,
    ) -> Result<Receiver<Result<KimiOneTokenForwardReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(KimiRankCommand::ForwardPromptNextToken {
                slot,
                decode_batch_size,
                input_ids,
                ep_max_seq_len,
                logprobs,
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
        logprobs: Vec<usize>,
    ) -> Result<Receiver<Result<Vec<KimiOneTokenForwardReport>>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(KimiRankCommand::ForwardDecodeBatchNextTokens {
                token_ids,
                append_positions,
                slots,
                decode_batch_size,
                logprobs,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(super) fn enable_pplx_async(
        &self,
        ep_backend: pegainfer_comm::EpBackend,
    ) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
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
    local_dims: crate::config::KimiLocalDims,
    collective_barrier: Arc<Barrier>,
    enable_cuda_graph: bool,
    loaded: Option<KimiRankLoadedWeights>,
    ep_backend: Option<pegainfer_comm::EpBackend>,
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
    arenas: Vec<Option<KimiWorkerDecodeArena>>,
    vocab_rows: usize,
    dims: crate::config::KimiLocalDims,
}

impl KimiWorkerDecodeArenas {
    fn new(vocab_rows: usize, dims: &crate::config::KimiLocalDims) -> Self {
        let arenas = KIMI_DECODE_BATCH_BUCKETS.iter().map(|_| None).collect();
        Self {
            arenas,
            vocab_rows,
            dims: *dims,
        }
    }

    fn get_mut(
        &mut self,
        ctx: &DeviceContext,
        decode_batch_size: usize,
    ) -> Result<&mut KimiWorkerDecodeArena> {
        ensure!(
            (1..=KIMI_DECODE_MAX_BATCH).contains(&decode_batch_size),
            "Kimi decode batch size {decode_batch_size} must be in 1..={KIMI_DECODE_MAX_BATCH}"
        );
        let (idx, arena_batch_size) = decode_batch_bucket(decode_batch_size)?;
        if self.arenas[idx].is_none() {
            self.arenas[idx] = Some(
                KimiWorkerDecodeArena::new(
                    ctx,
                    KIMI_K2_LAYERS,
                    arena_batch_size,
                    KIMI_DECODE_PAGE_SIZE,
                    self.vocab_rows,
                    &self.dims,
                )
                .with_context(|| {
                    format!(
                        "failed to allocate Kimi bs{arena_batch_size} decode arena for requested bs{decode_batch_size}"
                    )
                })?,
            );
        }
        self.arenas[idx]
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Kimi bs{arena_batch_size} decode arena missing"))
    }
}

fn decode_batch_bucket(decode_batch_size: usize) -> Result<(usize, usize)> {
    ensure!(
        (1..=KIMI_DECODE_MAX_BATCH).contains(&decode_batch_size),
        "Kimi decode batch size {decode_batch_size} must be in 1..={KIMI_DECODE_MAX_BATCH}"
    );
    KIMI_DECODE_BATCH_BUCKETS
        .iter()
        .copied()
        .enumerate()
        .find(|(_, bucket)| decode_batch_size <= *bucket)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Kimi decode batch size {decode_batch_size} has no arena bucket up to {KIMI_DECODE_MAX_BATCH}"
            )
        })
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
    q_b_proj: DeviceMatrix,
    kv_a_norm: NormWeight<KIMI_K2_MLA_KV_LORA_RANK>,
    kv_b_proj: DeviceMatrix,
    o_proj: DeviceMatrix,
    post_attention_norm: NormWeight<KIMI_K2_HIDDEN>,
}

enum KimiLayerForwardKindCache {
    Dense(KimiDenseForwardCache),
    Moe(KimiMoeForwardCache),
}

struct KimiDenseForwardCache {
    gate_up_proj: DeviceMatrix,
    down_proj: DeviceMatrix,
}

pub(super) struct KimiMoeForwardCache {
    pub(super) router: KimiRouterDeviceWeights,
    pub(super) shared_gate_up_proj: DeviceMatrix,
    pub(super) shared_down_proj: DeviceMatrix,
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
            pegainfer_kernels::ffi::kimi_mla_cublaslt_destroy_cuda();
            pegainfer_kernels::ffi::kimi_o_proj_cublaslt_destroy_cuda();
            pegainfer_kernels::ffi::kimi_shared_gate_up_cublaslt_destroy_cuda();
            pegainfer_kernels::ffi::cublas_destroy();
        }
    }
}

fn bind_rank_thread(
    ctx: KimiRankGpuContext,
    weight_names: KimiRankWeightNames,
    sliced_load_plan: KimiRankSlicedLoadPlan,
    local_dims: crate::config::KimiLocalDims,
    collective_barrier: Arc<Barrier>,
    enable_cuda_graph: bool,
) -> Result<KimiRankThreadState> {
    ctx.set_current()?;
    let decode_aux_ctx = ctx.auxiliary_device_context("decode aux")?;
    unsafe {
        pegainfer_kernels::ffi::cublas_init();
        let status = pegainfer_kernels::ffi::kimi_shared_gate_up_cublaslt_init_cuda();
        if status != 0 {
            if status >= 100_000 {
                anyhow::bail!(
                    "Kimi shared_gate_up cuBLASLt init failed: cublas_status={}",
                    status - 100_000
                );
            }
            anyhow::bail!(
                "Kimi shared_gate_up cuBLASLt init failed: cuda_status={}",
                status
            );
        }
        let status = pegainfer_kernels::ffi::kimi_mla_cublaslt_init_cuda();
        if status != 0 {
            if status >= 100_000 {
                anyhow::bail!(
                    "Kimi MLA cuBLASLt init failed: cublas_status={}",
                    status - 100_000
                );
            }
            anyhow::bail!("Kimi MLA cuBLASLt init failed: cuda_status={}", status);
        }
        if local_dims.o_proj_in == KIMI_O_PROJ_CUBLASLT_INPUT {
            let status = pegainfer_kernels::ffi::kimi_o_proj_cublaslt_init_cuda();
            if status != 0 {
                if status >= 100_000 {
                    anyhow::bail!(
                        "Kimi o_proj cuBLASLt init failed: cublas_status={}",
                        status - 100_000
                    );
                }
                anyhow::bail!("Kimi o_proj cuBLASLt init failed: cuda_status={}", status);
            }
        }
    }
    Ok(KimiRankThreadState {
        ctx,
        decode_aux_ctx,
        _cublas: KimiCublasThreadGuard,
        tp_comm: None,
        weight_names,
        sliced_load_plan,
        local_dims,
        collective_barrier,
        enable_cuda_graph,
        loaded: None,
        ep_backend: None,
        moe_pplx_scratch: None,
    })
}

// The worker owns its command channel for the lifetime of the loop: taking
// `&Receiver` would leave the channel alive in the caller and break the
// "senders dropped → loop exits" shutdown signal.
#[allow(clippy::needless_pass_by_value)]
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
            KimiRankCommand::EnsureDecodeArena {
                decode_batch_size,
                resp,
            } => {
                let result = state.ensure_decode_arena(decode_batch_size);
                let _ = resp.send(result);
            }
            KimiRankCommand::ForwardPromptNextToken {
                slot,
                decode_batch_size,
                input_ids,
                ep_max_seq_len,
                logprobs,
                resp,
            } => {
                let result = state.forward_prompt_next_token(
                    slot,
                    decode_batch_size,
                    &input_ids,
                    ep_max_seq_len,
                    logprobs,
                );
                let _ = resp.send(result);
            }
            KimiRankCommand::ForwardDecodeBatchNextTokens {
                token_ids,
                append_positions,
                slots,
                decode_batch_size,
                logprobs,
                resp,
            } => {
                let result = state.forward_decode_batch_next_tokens(
                    &token_ids,
                    &append_positions,
                    &slots,
                    decode_batch_size,
                    &logprobs,
                );
                let _ = resp.send(result);
            }
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
// Collective + Marlin helpers shared with the sibling `moe_nccl` backend.
pub(super) use runtime::{
    all_reduce_f32_in_place, kimi_marlin_block_size, maybe_all_reduce_hidden_via_f32_in_place,
    reduce_scatter_f32_hidden_into,
};
mod state;
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

pub(super) fn build_placements(device_ordinals: &[usize]) -> Result<Vec<KimiK2RankPlacement>> {
    ensure!(
        !device_ordinals.is_empty(),
        "Kimi-K2 requires at least one device ordinal"
    );
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
    use super::decode_batch_bucket;

    #[test]
    fn decode_batch_bucket_rounds_up_to_power_of_two_buckets() {
        let cases = [
            (1, (0, 1)),
            (2, (1, 2)),
            (3, (2, 4)),
            (4, (2, 4)),
            (5, (3, 8)),
            (17, (5, 32)),
            (33, (6, 64)),
            (64, (6, 64)),
        ];
        for (requested, expected) in cases {
            assert_eq!(decode_batch_bucket(requested).unwrap(), expected);
        }
    }

    #[test]
    fn decode_batch_bucket_rejects_out_of_range_sizes() {
        assert!(decode_batch_bucket(0).is_err());
        assert!(decode_batch_bucket(65).is_err());
    }

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
