//! pplx-garden NVLink + RDMA MoE all-to-all decode path.
//!
//! Drop-in replacement for the NCCL AG/RS backend in [`super::moe_nccl`]:
//! same shared-expert + routed-expert flow, but cross-rank token movement
//! uses the four-step pipeline (`dispatch_send → dispatch_recv →
//! combine_send → combine_recv`) wrapped by [`pegainfer_comm::EpBackend`].
//!
//! # Expert-major layout alignment
//!
//! PPLX `dispatch_recv` writes tokens in expert-major padded layout, where
//! each expert occupies `ceil(count, expert_padding)` rows.  Because
//! `expert_padding` (8) equals the Marlin `block_size` (8), the Marlin GEMM
//! kernel can read/write the PPLX buffer directly using identity
//! `sorted_token_ids`.  No gather/scatter copies are needed.
//!
//! # Router scale
//!
//! `combine_recv` runs with `accumulate=false` so the routed contribution
//! is written to a separate buffer. The KIMI_K2_ROUTER_SCALE is applied
//! only to the routed part before adding to the residual + shared expert.

use std::{ffi::c_void, ptr};

use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::nccl::safe::Comm;
use pegainfer_comm::{EpBackend, ScalarType};
use pegainfer_kernels::{
    ops::{
        KIMI_K2_EP_WORLD, KIMI_K2_LOCAL_EXPERTS, KIMI_K2_ROUTER_SCALE, KIMI_K2_SHARED_GATE_UP,
        KimiMarlinRouteWorkspace, KimiMarlinWna16Workspace, KimiRouterBatch, KimiRouterConfig,
        KimiRouterOutput, KimiRouterScratch, kimi_marlin_w13_swiglu, kimi_marlin_w13_swiglu_pplx,
        kimi_marlin_wna16_pplx_w2_gemm, kimi_marlin_wna16_pplx_w13_gemm, kimi_marlin_wna16_w2_gemm,
        kimi_marlin_wna16_w13_gemm, kimi_moe_marlin_align_block_size,
        kimi_pplx_build_marlin_routing_on_stream, kimi_residual_add_scaled_f32,
        kimi_router_noaux_tc_launch, kimi_scatter_marlin_routes_to_compact,
        kimi_shared_gate_up_cublaslt_into, kimi_shared_gate_up_cublaslt_supports_batch_size,
    },
    tensor::{DeviceContext, GpuTensor, NormWeight},
    typed_ops,
};

use pegainfer_kernels::tensor::HiddenStates;

use crate::{
    config::{
        KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_RMS_NORM_EPS, KIMI_K2_ROUTED_EXPERTS,
        KIMI_K2_TOPK,
    },
    weights::KimiRankExpertMarlinWeights,
};

use super::worker::{KimiMoeForwardCache, KimiWorkerDecodeScratch, MARLIN_W13_OUT_DIM};

pub(super) const PPLX_EXPERT_PADDING: usize = 8;

pub(super) struct KimiMoePplxScratch {
    pub(super) max_local_output_tokens: usize,
    pub(super) expert_padding: usize,
    pub(super) pplx_recv_capacity: usize,
    pub(super) recv_tokens_per_expert: CudaSlice<i32>,
    pub(super) pplx_recv_hidden: GpuTensor<KIMI_K2_HIDDEN>,
    pub(super) pplx_expert_output: GpuTensor<KIMI_K2_HIDDEN>,
    pub(super) pplx_w13_out: GpuTensor<MARLIN_W13_OUT_DIM>,
    pub(super) pplx_activated: GpuTensor<KIMI_K2_EXPERT_INTERMEDIATE>,
    pub(super) pplx_recv_topk_weight: CudaSlice<f32>,
    pub(super) pplx_route_workspace: KimiMarlinRouteWorkspace,
    pub(super) pplx_marlin_workspace: KimiMarlinWna16Workspace,
    pub(super) pplx_dummy_topk_weight: CudaSlice<f32>,
    pub(super) pplx_routed_f32: CudaSlice<f32>,
}

impl KimiMoePplxScratch {
    pub(super) fn new_decode(ctx: &DeviceContext, max_batch_size: usize) -> Result<Self> {
        let max_total_tokens = max_batch_size
            .checked_mul(KIMI_K2_EP_WORLD)
            .ok_or_else(|| anyhow::anyhow!("Kimi PPLX decode total token capacity overflow"))?;
        Self::new_for_total_dispatch_tokens(ctx, max_batch_size, max_total_tokens)
    }

    pub(super) fn new_prefill(ctx: &DeviceContext, max_prompt_tokens: usize) -> Result<Self> {
        let padding_ranks = KIMI_K2_EP_WORLD.saturating_sub(1);
        let max_total_tokens = max_prompt_tokens
            .checked_add(padding_ranks)
            .ok_or_else(|| anyhow::anyhow!("Kimi PPLX prefill total token capacity overflow"))?;
        Self::new_for_total_dispatch_tokens(ctx, max_prompt_tokens, max_total_tokens)
    }

    fn new_for_total_dispatch_tokens(
        ctx: &DeviceContext,
        max_local_output_tokens: usize,
        max_total_tokens: usize,
    ) -> Result<Self> {
        let pplx_recv_capacity = pplx_recv_capacity(max_total_tokens)?;

        let marlin_block_size = 8;
        let route_workspace =
            KimiMarlinRouteWorkspace::new(ctx, pplx_recv_capacity, marlin_block_size)?;
        let marlin_workspace = KimiMarlinWna16Workspace::new(
            ctx,
            route_workspace.max_m_blocks,
            KIMI_K2_HIDDEN,
            marlin_block_size,
        )?;

        let dummy_weights = vec![1.0f32; pplx_recv_capacity];
        let pplx_dummy_topk_weight = ctx.stream.clone_htod(&dummy_weights)?;

        Ok(Self {
            max_local_output_tokens,
            expert_padding: PPLX_EXPERT_PADDING,
            pplx_recv_capacity,
            recv_tokens_per_expert: ctx.stream.alloc_zeros(KIMI_K2_LOCAL_EXPERTS)?,
            pplx_recv_hidden: GpuTensor::zeros(ctx, pplx_recv_capacity)?,
            pplx_expert_output: GpuTensor::zeros(ctx, pplx_recv_capacity)?,
            pplx_w13_out: GpuTensor::zeros(ctx, pplx_recv_capacity)?,
            pplx_activated: GpuTensor::zeros(ctx, pplx_recv_capacity)?,
            pplx_recv_topk_weight: ctx.stream.alloc_zeros(pplx_recv_capacity)?,
            pplx_route_workspace: route_workspace,
            pplx_marlin_workspace: marlin_workspace,
            pplx_dummy_topk_weight,
            pplx_routed_f32: ctx
                .stream
                .alloc_zeros(max_local_output_tokens * KIMI_K2_HIDDEN)?,
        })
    }
}

fn pplx_recv_capacity(max_total_tokens: usize) -> Result<usize> {
    let max_routes = max_total_tokens
        .checked_mul(KIMI_K2_TOPK)
        .ok_or_else(|| anyhow::anyhow!("Kimi PPLX routed-token capacity overflow"))?;
    let active_experts = max_routes.min(KIMI_K2_LOCAL_EXPERTS);
    let padding_rows = active_experts
        .checked_mul(PPLX_EXPERT_PADDING - 1)
        .ok_or_else(|| anyhow::anyhow!("Kimi PPLX expert padding capacity overflow"))?;
    max_routes
        .checked_add(padding_rows)
        .ok_or_else(|| anyhow::anyhow!("Kimi PPLX recv capacity overflow"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_moe_layer_decode_pplx_normed(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    comm: Option<&Comm>,
    ep: &mut EpBackend,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    expert_kernels: &KimiRankExpertMarlinWeights,
    scratch: &mut KimiWorkerDecodeScratch,
    pplx: &mut KimiMoePplxScratch,
) -> Result<()> {
    let batch_size = scratch.mla.hidden.seq_len;
    let stream_raw = ctx.stream.cu_stream() as u64;
    let use_tp8_dp1_duplicate_routes = comm.is_some()
        && ep.world_size() == KIMI_K2_EP_WORLD
        && ep.dp_size() == 1
        && ep.node_size() == ep.world_size()
        && ep.canonicalize_duplicate_sources()
        && KIMI_K2_LOCAL_EXPERTS * KIMI_K2_EP_WORLD == KIMI_K2_ROUTED_EXPERTS;
    if comm.is_some() {
        anyhow::ensure!(
            use_tp8_dp1_duplicate_routes,
            "Kimi PPLX route-only decode requires TP8/DP1 intra-node duplicate-source canonicalization"
        );
    }

    // Shared expert (main stream) + router (aux stream) both consume the
    // post-attention normed hidden state, so start router as soon as norm is
    // ready instead of waiting for shared expert/all-reduce to finish.
    let norm_ready = ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE PPLX layer {layer_idx} record norm_ready"))?;
    aux_ctx
        .stream
        .wait(&norm_ready)
        .with_context(|| format!("Kimi MoE PPLX layer {layer_idx} aux wait norm_ready"))?;
    {
        let mut router_scratch = KimiRouterScratch {
            logits: &mut scratch.router.router_logits.data,
            scores: &mut scratch.router.router_scores.data,
            choice_scores: &mut scratch.router.router_choice_scores.data,
        };
        let mut router_output = KimiRouterOutput {
            topk_weight: &mut scratch.router.router_topk_weight.data,
            topk_idx: &mut scratch.router.router_topk_idx.data,
        };
        kimi_router_noaux_tc_launch(
            aux_ctx,
            KimiRouterConfig::kimi_k2(),
            KimiRouterBatch {
                batch_size,
                active_tokens: batch_size,
                padded_tokens: batch_size,
            },
            &scratch.mla.normed,
            &moe.router.gate_weight,
            &moe.router.e_score_correction_bias,
            &mut router_scratch,
            &mut router_output,
        )?;
    }
    let route_ready = aux_ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE PPLX layer {layer_idx} record route_ready"))?;

    if moe.shared_gate_up_proj.rows == KIMI_K2_SHARED_GATE_UP
        && kimi_shared_gate_up_cublaslt_supports_batch_size(batch_size)
    {
        kimi_shared_gate_up_cublaslt_into(
            ctx,
            &moe.shared_gate_up_proj,
            &scratch.mla.normed,
            &mut scratch.shared_expert.gate_up,
        )?;
    } else {
        typed_ops::gemm_dm_typed_to_hs_graphsafe(
            ctx,
            &moe.shared_gate_up_proj,
            &scratch.mla.normed,
            &mut scratch.shared_expert.gate_up,
        )?;
    }
    typed_ops::silu_mul_hs_fused_into(
        ctx,
        &scratch.shared_expert.gate_up,
        &mut scratch.shared_expert.activated,
    )?;
    typed_ops::gemm_dm_hs_to_typed_graphsafe(
        ctx,
        &moe.shared_down_proj,
        &scratch.shared_expert.activated,
        &mut scratch.mla.projected,
    )?;
    super::worker::maybe_all_reduce_hidden_via_f32_in_place(
        ctx,
        &mut scratch.mla.projected,
        &mut scratch.comm.hidden_allreduce_f32,
        comm,
    )?;

    ctx.stream
        .wait(&route_ready)
        .with_context(|| format!("Kimi MoE PPLX layer {layer_idx} main wait route_ready"))?;
    // ---- 4. dispatch_send ----
    if use_tp8_dp1_duplicate_routes {
        let (idx_ptr, _idx_guard) = scratch.router.router_topk_idx.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = scratch
            .router
            .router_topk_weight
            .data
            .device_ptr(&ctx.stream);
        ep.dispatch_send_route_only(
            batch_size,
            idx_ptr as *const i32,
            KIMI_K2_TOPK,
            w_ptr as *const f32,
            KIMI_K2_TOPK,
            ptr::null(),
            stream_raw,
        )
        .with_context(|| format!("pplx dispatch_send_route_only layer {layer_idx}"))?;
    } else {
        let (x_ptr, _x_guard) = scratch.mla.normed.data.device_ptr(&ctx.stream);
        let (idx_ptr, _idx_guard) = scratch.router.router_topk_idx.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = scratch
            .router
            .router_topk_weight
            .data
            .device_ptr(&ctx.stream);
        let x_stride = KIMI_K2_HIDDEN * std::mem::size_of::<u16>();
        ep.dispatch_send(
            batch_size,
            x_ptr as *const c_void,
            x_stride,
            ptr::null(),
            0,
            0,
            idx_ptr as *const i32,
            KIMI_K2_TOPK,
            w_ptr as *const f32,
            KIMI_K2_TOPK,
            ptr::null(),
            stream_raw,
        )
        .with_context(|| format!("pplx dispatch_send layer {layer_idx}"))?;
    }

    let layer_weights = expert_kernels
        .layers
        .iter()
        .find(|layer| layer.layer_idx == layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!("Kimi rank expert Marlin package missing layer {layer_idx}")
        })?
        .as_marlin_weights();

    if use_tp8_dp1_duplicate_routes {
        // TP8/DP1 keeps the post-collective hidden state identical on every
        // rank, so compute local experts with the NCCL Marlin routing and use
        // PPLX only for the combine transfer. This preserves NCCL's route-slot
        // layout and BF16 rounding behavior.
        {
            let (out_num_ptr, _g0) = pplx.recv_tokens_per_expert.device_ptr_mut(&ctx.stream);
            ep.dispatch_recv_counts(out_num_ptr as *mut i32, stream_raw)
                .with_context(|| format!("pplx dispatch_recv_counts layer {layer_idx}"))?;
        }

        let routing = kimi_moe_marlin_align_block_size(
            ctx,
            &mut scratch.marlin_route_workspace,
            &scratch.router.router_topk_idx.data,
            batch_size,
            batch_size,
            expert_kernels.local_expert_range.start,
        )
        .with_context(|| format!("pplx tp8 build NCCL-layout routing layer {layer_idx}"))?;

        scratch.marlin.w13_out.seq_len = routing.route_elems;
        scratch.marlin.activated.seq_len = routing.route_elems;
        scratch.marlin.expert_output.seq_len = routing.route_elems;
        ctx.stream.memset_zeros(&mut scratch.marlin.w13_out.data)?;
        ctx.stream
            .memset_zeros(&mut scratch.marlin.expert_output.data)?;
        kimi_marlin_wna16_w13_gemm(
            ctx,
            &mut scratch.marlin_workspace,
            &routing,
            &scratch.mla.normed,
            &layer_weights.w13,
            &scratch.router.router_topk_weight.data,
            &mut scratch.marlin.w13_out,
        )?;
        kimi_marlin_w13_swiglu(ctx, &scratch.marlin.w13_out, &mut scratch.marlin.activated)?;
        kimi_marlin_wna16_w2_gemm(
            ctx,
            &mut scratch.marlin_workspace,
            &routing,
            &scratch.marlin.activated,
            &layer_weights.w2_down,
            &scratch.router.router_topk_weight.data,
            &mut scratch.marlin.expert_output,
        )?;
        pplx.pplx_expert_output.seq_len = routing.max_padded_tokens;
        kimi_scatter_marlin_routes_to_compact(
            ctx,
            &scratch.marlin.expert_output,
            &routing,
            &mut pplx.pplx_expert_output,
        )?;
    } else {
        // ---- 5. dispatch_recv ----
        {
            let (out_num_ptr, _g0) = pplx.recv_tokens_per_expert.device_ptr_mut(&ctx.stream);
            let (out_x_ptr, _g1) = pplx.pplx_recv_hidden.data.device_ptr_mut(&ctx.stream);
            let (out_w_ptr, _g2) = pplx.pplx_recv_topk_weight.device_ptr_mut(&ctx.stream);
            ep.dispatch_recv(
                out_num_ptr as *mut i32,
                out_x_ptr as *mut c_void,
                KIMI_K2_HIDDEN * std::mem::size_of::<u16>(),
                out_w_ptr as *mut c_void,
                1,
                1,
                stream_raw,
            )
            .with_context(|| format!("pplx dispatch_recv layer {layer_idx}"))?;
        }

        // ---- 6. Build Marlin routing ----
        let routing = kimi_pplx_build_marlin_routing_on_stream(
            ctx,
            &mut pplx.pplx_route_workspace,
            &pplx.recv_tokens_per_expert,
            pplx.expert_padding,
            pplx.pplx_recv_capacity,
        )
        .with_context(|| format!("pplx build Marlin routing layer {layer_idx}"))?;

        // ---- 7. Marlin W13 (gate+up) GEMM ----
        pplx.pplx_recv_hidden.seq_len = routing.route_elems;
        pplx.pplx_w13_out.seq_len = routing.route_elems;
        kimi_marlin_wna16_pplx_w13_gemm(
            ctx,
            &mut pplx.pplx_marlin_workspace,
            &routing,
            &pplx.pplx_recv_hidden,
            &layer_weights.w13,
            &pplx.pplx_dummy_topk_weight,
            &mut pplx.pplx_w13_out,
        )?;

        // ---- 8. SwiGLU activation (GPU reads actual row count, no D2H) ----
        pplx.pplx_activated.seq_len = routing.route_elems;
        kimi_marlin_w13_swiglu_pplx(
            ctx,
            &pplx.pplx_w13_out,
            routing.num_tokens_post_padded,
            &mut pplx.pplx_activated,
        )?;

        // ---- 9. Marlin W2 (down) GEMM ----
        pplx.pplx_expert_output.seq_len = routing.route_elems;
        kimi_marlin_wna16_pplx_w2_gemm(
            ctx,
            &mut pplx.pplx_marlin_workspace,
            &routing,
            &pplx.pplx_activated,
            &layer_weights.w2_down,
            &pplx.pplx_recv_topk_weight,
            &mut pplx.pplx_expert_output,
        )?;
    }

    // ---- 10. combine_send ----
    {
        let (exp_ptr, _g) = pplx.pplx_expert_output.data.device_ptr(&ctx.stream);
        ep.combine_send(
            exp_ptr as *const c_void,
            KIMI_K2_HIDDEN * std::mem::size_of::<u16>(),
            stream_raw,
        )
        .with_context(|| format!("pplx combine_send layer {layer_idx}"))?;
    }

    // ---- 11. combine_recv: gather weighted expert rows into F32 ----
    {
        let (out_ptr, _g0) = pplx.pplx_routed_f32.device_ptr_mut(&ctx.stream);
        let (idx_ptr, _g1) = scratch.router.router_topk_idx.data.device_ptr(&ctx.stream);
        let (w_ptr, _g2) = pplx.pplx_dummy_topk_weight.device_ptr(&ctx.stream);
        ep.combine_recv(
            batch_size,
            0,
            ScalarType::BF16,
            out_ptr as *mut c_void,
            KIMI_K2_HIDDEN,
            idx_ptr as *const i32,
            KIMI_K2_TOPK,
            w_ptr as *const f32,
            KIMI_K2_TOPK,
            ptr::null(),
            false,
            stream_raw,
        )
        .with_context(|| format!("pplx combine_recv layer {layer_idx}"))?;
    }
    kimi_residual_add_scaled_f32(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &pplx.pplx_routed_f32,
        KIMI_K2_ROUTER_SCALE,
        &mut scratch.mla.normed,
    )?;
    std::mem::swap(&mut scratch.mla.hidden, &mut scratch.mla.normed);
    Ok(())
}

/// Batch PPLX MoE for prefill: all seq_len tokens dispatched in a single
/// PPLX collective per layer. All EP ranks must call this simultaneously.
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_moe_layer_prefill_pplx(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    ep: &mut EpBackend,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    expert_kernels: &KimiRankExpertMarlinWeights,
    hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
    next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    pplx: &mut KimiMoePplxScratch,
) -> Result<()> {
    let seq_len = hidden.seq_len;
    let stream_raw = ctx.stream.cu_stream() as u64;

    // ---- 1. RMS norm ----
    typed_ops::rms_norm_into(
        ctx,
        hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        normed,
    )?;

    // ---- 2. Shared expert (main stream, TP1 so no all-reduce) ----
    let mut shared_gate_up = HiddenStates::zeros(ctx, moe.shared_gate_up_proj.rows, seq_len)?;
    typed_ops::gemm_dm_typed_to_hs(ctx, &moe.shared_gate_up_proj, normed, &mut shared_gate_up)?;
    let mut shared_activated = HiddenStates::zeros(ctx, moe.shared_down_proj.cols, seq_len)?;
    typed_ops::silu_mul_hs_fused_into(ctx, &shared_gate_up, &mut shared_activated)?;
    let mut shared_out = GpuTensor::<KIMI_K2_HIDDEN>::zeros(ctx, seq_len)?;
    typed_ops::gemm_dm_hs_to_typed(
        ctx,
        &moe.shared_down_proj,
        &shared_activated,
        &mut shared_out,
    )?;

    // ---- 3. Router on aux stream (overlap with shared expert) ----
    let norm_ready = ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE PPLX prefill layer {layer_idx} record norm_ready"))?;
    aux_ctx
        .stream
        .wait(&norm_ready)
        .with_context(|| format!("Kimi MoE PPLX prefill layer {layer_idx} aux wait norm_ready"))?;

    let mut router_logits: CudaSlice<f32> = aux_ctx
        .stream
        .alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_scores: CudaSlice<f32> = aux_ctx
        .stream
        .alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_choice_scores: CudaSlice<f32> = aux_ctx
        .stream
        .alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_topk_weight: CudaSlice<f32> =
        aux_ctx.stream.alloc_zeros(seq_len * KIMI_K2_TOPK)?;
    let mut router_topk_idx: CudaSlice<i32> = aux_ctx.stream.alloc_zeros(seq_len * KIMI_K2_TOPK)?;
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
            aux_ctx,
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
    let route_ready = aux_ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE PPLX prefill layer {layer_idx} record route_ready"))?;
    ctx.stream.wait(&route_ready).with_context(|| {
        format!("Kimi MoE PPLX prefill layer {layer_idx} main wait route_ready")
    })?;

    // ---- 4. dispatch_send ----
    {
        let (x_ptr, _x_guard) = normed.data.device_ptr(&ctx.stream);
        let (idx_ptr, _idx_guard) = router_topk_idx.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = router_topk_weight.device_ptr(&ctx.stream);
        let x_stride = KIMI_K2_HIDDEN * std::mem::size_of::<u16>();
        ep.dispatch_send(
            seq_len,
            x_ptr as *const c_void,
            x_stride,
            ptr::null(),
            0,
            0,
            idx_ptr as *const i32,
            KIMI_K2_TOPK,
            w_ptr as *const f32,
            KIMI_K2_TOPK,
            ptr::null(),
            stream_raw,
        )
        .with_context(|| format!("pplx prefill dispatch_send layer {layer_idx}"))?;
    }

    // ---- 5. dispatch_recv ----
    {
        let (out_num_ptr, _g0) = pplx.recv_tokens_per_expert.device_ptr_mut(&ctx.stream);
        let (out_x_ptr, _g1) = pplx.pplx_recv_hidden.data.device_ptr_mut(&ctx.stream);
        let (out_w_ptr, _g2) = pplx.pplx_recv_topk_weight.device_ptr_mut(&ctx.stream);
        ep.dispatch_recv(
            out_num_ptr as *mut i32,
            out_x_ptr as *mut c_void,
            KIMI_K2_HIDDEN * std::mem::size_of::<u16>(),
            out_w_ptr as *mut c_void,
            1,
            1,
            stream_raw,
        )
        .with_context(|| format!("pplx prefill dispatch_recv layer {layer_idx}"))?;
    }

    // ---- 6. Build Marlin routing ----
    let routing = kimi_pplx_build_marlin_routing_on_stream(
        ctx,
        &mut pplx.pplx_route_workspace,
        &pplx.recv_tokens_per_expert,
        pplx.expert_padding,
        pplx.pplx_recv_capacity,
    )
    .with_context(|| format!("pplx prefill build Marlin routing layer {layer_idx}"))?;

    let layer_weights = expert_kernels
        .layers
        .iter()
        .find(|layer| layer.layer_idx == layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!("Kimi rank expert Marlin package missing layer {layer_idx}")
        })?
        .as_marlin_weights();

    // ---- 7. Marlin W13 (gate+up) GEMM ----
    pplx.pplx_recv_hidden.seq_len = routing.route_elems;
    pplx.pplx_w13_out.seq_len = routing.route_elems;
    kimi_marlin_wna16_pplx_w13_gemm(
        ctx,
        &mut pplx.pplx_marlin_workspace,
        &routing,
        &pplx.pplx_recv_hidden,
        &layer_weights.w13,
        &pplx.pplx_dummy_topk_weight,
        &mut pplx.pplx_w13_out,
    )?;

    // ---- 8. SwiGLU activation ----
    pplx.pplx_activated.seq_len = routing.route_elems;
    kimi_marlin_w13_swiglu_pplx(
        ctx,
        &pplx.pplx_w13_out,
        routing.num_tokens_post_padded,
        &mut pplx.pplx_activated,
    )?;

    // ---- 9. Marlin W2 (down) GEMM ----
    pplx.pplx_expert_output.seq_len = routing.route_elems;
    kimi_marlin_wna16_pplx_w2_gemm(
        ctx,
        &mut pplx.pplx_marlin_workspace,
        &routing,
        &pplx.pplx_activated,
        &layer_weights.w2_down,
        &pplx.pplx_recv_topk_weight,
        &mut pplx.pplx_expert_output,
    )?;

    // ---- 10. combine_send ----
    {
        let (exp_ptr, _g) = pplx.pplx_expert_output.data.device_ptr(&ctx.stream);
        ep.combine_send(
            exp_ptr as *const c_void,
            KIMI_K2_HIDDEN * std::mem::size_of::<u16>(),
            stream_raw,
        )
        .with_context(|| format!("pplx prefill combine_send layer {layer_idx}"))?;
    }

    // ---- 11. combine_recv: gather weighted expert rows into F32 ----
    {
        let (out_ptr, _g0) = pplx.pplx_routed_f32.device_ptr_mut(&ctx.stream);
        let (idx_ptr, _g1) = router_topk_idx.device_ptr(&ctx.stream);
        let (w_ptr, _g2) = pplx.pplx_dummy_topk_weight.device_ptr(&ctx.stream);
        ep.combine_recv(
            seq_len,
            0,
            ScalarType::BF16,
            out_ptr as *mut c_void,
            KIMI_K2_HIDDEN,
            idx_ptr as *const i32,
            KIMI_K2_TOPK,
            w_ptr as *const f32,
            KIMI_K2_TOPK,
            ptr::null(),
            false,
            stream_raw,
        )
        .with_context(|| format!("pplx prefill combine_recv layer {layer_idx}"))?;
    }

    kimi_residual_add_scaled_f32(
        ctx,
        hidden,
        &shared_out,
        &pplx.pplx_routed_f32,
        KIMI_K2_ROUTER_SCALE,
        next_hidden,
    )?;
    std::mem::swap(hidden, next_hidden);
    Ok(())
}
