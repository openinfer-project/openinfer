//! pplx-garden NVLink + RDMA MoE all-to-all decode path (feature `pplx-ep`).
//!
//! Drop-in replacement for the NCCL AG/RS `forward_moe_layer_decode_into`:
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

use std::ffi::c_void;
use std::ptr;

use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::nccl::safe::Comm;
use pegainfer_comm::{EpBackend, ScalarType};
use pegainfer_kernels::{
    ops::{
        KIMI_K2_ROUTER_SCALE, KimiMarlinRouteWorkspace, KimiMarlinWna16Workspace, KimiRouterBatch,
        KimiRouterConfig, KimiRouterOutput, KimiRouterScratch, kimi_marlin_w13_swiglu_pplx,
        kimi_marlin_wna16_pplx_w2_gemm, kimi_marlin_wna16_pplx_w13_gemm,
        kimi_pplx_build_marlin_routing_on_stream, kimi_router_noaux_tc_launch,
        kimi_scaled_add_f32_bf16_to_bf16,
    },
    tensor::{DeviceContext, GpuTensor, NormWeight},
    typed_ops,
};

use crate::{
    config::{KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_RMS_NORM_EPS, KIMI_K2_TOPK},
    layers::experts::{KIMI_K2_EP_WORLD, KIMI_K2_EP8_LOCAL_EXPERTS},
    weights::KimiRankExpertMarlinWeights,
};

use super::worker::{
    KimiMoeForwardCache, KimiWorkerDecodeScratch, MARLIN_W13_OUT_DIM, SHARED_ACTIVATED_DIM,
    SHARED_GATE_UP_DIM,
};

pub(super) const PPLX_EXPERT_PADDING: usize = 8;

pub(super) struct KimiMoePplxScratch {
    pub(super) expert_padding: usize,
    pub(super) pplx_recv_capacity: usize,
    pub(super) recv_tokens_per_expert: CudaSlice<i32>,
    pub(super) pplx_recv_hidden: GpuTensor<KIMI_K2_HIDDEN>,
    pub(super) pplx_expert_output: GpuTensor<KIMI_K2_HIDDEN>,
    pub(super) pplx_w13_out: GpuTensor<MARLIN_W13_OUT_DIM>,
    pub(super) pplx_activated: GpuTensor<KIMI_K2_EXPERT_INTERMEDIATE>,
    pub(super) pplx_route_workspace: KimiMarlinRouteWorkspace,
    pub(super) pplx_marlin_workspace: KimiMarlinWna16Workspace,
    pub(super) pplx_dummy_topk_weight: CudaSlice<f32>,
    pub(super) pplx_routed_out: GpuTensor<KIMI_K2_HIDDEN>,
    pub(super) pplx_routed_f32: CudaSlice<f32>,
}

impl KimiMoePplxScratch {
    pub(super) fn new(ctx: &DeviceContext, max_batch_size: usize) -> Result<Self> {
        let max_recv_per_expert =
            max_batch_size * KIMI_K2_TOPK * KIMI_K2_EP_WORLD / KIMI_K2_EP8_LOCAL_EXPERTS;
        let max_recv_padded =
            max_recv_per_expert.div_ceil(PPLX_EXPERT_PADDING) * PPLX_EXPERT_PADDING;
        let pplx_recv_capacity = max_recv_padded * KIMI_K2_EP8_LOCAL_EXPERTS;

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
            expert_padding: PPLX_EXPERT_PADDING,
            pplx_recv_capacity,
            recv_tokens_per_expert: ctx.stream.alloc_zeros(KIMI_K2_EP8_LOCAL_EXPERTS)?,
            pplx_recv_hidden: GpuTensor::zeros(ctx, pplx_recv_capacity)?,
            pplx_expert_output: GpuTensor::zeros(ctx, pplx_recv_capacity)?,
            pplx_w13_out: GpuTensor::zeros(ctx, pplx_recv_capacity)?,
            pplx_activated: GpuTensor::zeros(ctx, pplx_recv_capacity)?,
            pplx_route_workspace: route_workspace,
            pplx_marlin_workspace: marlin_workspace,
            pplx_dummy_topk_weight,
            pplx_routed_out: GpuTensor::zeros(ctx, max_batch_size)?,
            pplx_routed_f32: ctx.stream.alloc_zeros(max_batch_size * KIMI_K2_HIDDEN)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_moe_layer_decode_pplx(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    comm: &Comm,
    ep: &mut EpBackend,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    expert_kernels: &KimiRankExpertMarlinWeights,
    scratch: &mut KimiWorkerDecodeScratch,
    pplx: &mut KimiMoePplxScratch,
) -> Result<()> {
    let seq_len = scratch.mla.hidden.seq_len;
    let stream_raw = ctx.stream.cu_stream() as u64;

    // Shared expert (main stream) + RMS norm
    typed_ops::rms_norm_into(
        ctx,
        &scratch.mla.hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.mla.normed,
    )?;

    pegainfer_kernels::typed_pipeline! {
        ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS;
        gemm     (&scratch.mla.normed           => &mut scratch.shared_expert.gate_up,  moe.shared_gate_up_proj);
        silu_mul<SHARED_ACTIVATED_DIM> (&scratch.shared_expert.gate_up => &mut scratch.shared_expert.activated);
        gemm     (&scratch.shared_expert.activated => &mut scratch.mla.projected,       moe.shared_down_proj);
    }
    super::worker::all_reduce_hidden_via_f32_in_place(
        ctx,
        &mut scratch.mla.projected,
        &mut scratch.comm.hidden_allreduce_f32,
        comm,
    )?;

    // ---- 3. Router on aux stream (overlap with shared expert) ----
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
                batch_size: seq_len,
                active_tokens: seq_len,
                padded_tokens: seq_len,
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
    ctx.stream
        .wait(&route_ready)
        .with_context(|| format!("Kimi MoE PPLX layer {layer_idx} main wait route_ready"))?;

    // ---- 4. dispatch_send ----
    {
        let (x_ptr, _x_guard) = scratch.mla.normed.data.device_ptr(&ctx.stream);
        let (idx_ptr, _idx_guard) = scratch.router.router_topk_idx.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = scratch
            .router
            .router_topk_weight
            .data
            .device_ptr(&ctx.stream);
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
        .with_context(|| format!("pplx dispatch_send layer {layer_idx}"))?;
    }

    // ---- 5. dispatch_recv ----
    {
        let (out_num_ptr, _g0) = pplx.recv_tokens_per_expert.device_ptr_mut(&ctx.stream);
        let (out_x_ptr, _g1) = pplx.pplx_recv_hidden.data.device_ptr_mut(&ctx.stream);
        ep.dispatch_recv(
            out_num_ptr as *mut i32,
            out_x_ptr as *mut c_void,
            KIMI_K2_HIDDEN * std::mem::size_of::<u16>(),
            ptr::null_mut(),
            0,
            0,
            stream_raw,
        )
        .with_context(|| format!("pplx dispatch_recv layer {layer_idx}"))?;
    }

    // ---- 6. Build Marlin routing (tight host-side bound, zero D2H) ----
    let routing = kimi_pplx_build_marlin_routing_on_stream(
        ctx,
        &mut pplx.pplx_route_workspace,
        &pplx.recv_tokens_per_expert,
        pplx.expert_padding,
        pplx.pplx_recv_capacity,
        seq_len,
    )
    .with_context(|| format!("pplx build Marlin routing layer {layer_idx}"))?;

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
        &pplx.pplx_dummy_topk_weight,
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
        .with_context(|| format!("pplx combine_send layer {layer_idx}"))?;
    }

    // ---- 11. combine_recv: get weighted routed output separately ----
    pplx.pplx_routed_out.seq_len = seq_len;
    {
        let (out_ptr, _g0) = pplx.pplx_routed_out.data.device_ptr_mut(&ctx.stream);
        let (idx_ptr, _g1) = scratch.router.router_topk_idx.data.device_ptr(&ctx.stream);
        let (w_ptr, _g2) = scratch
            .router
            .router_topk_weight
            .data
            .device_ptr(&ctx.stream);
        ep.combine_recv(
            seq_len,
            0,
            ScalarType::BF16,
            out_ptr as *mut c_void,
            KIMI_K2_HIDDEN * std::mem::size_of::<u16>(),
            idx_ptr as *const i32,
            KIMI_K2_TOPK,
            w_ptr as *const f32,
            KIMI_K2_TOPK,
            ptr::null(),
            false, // don't accumulate — we need to scale routed separately
            stream_raw,
        )
        .with_context(|| format!("pplx combine_recv layer {layer_idx}"))?;
    }

    // Combine: hidden = hidden + shared + routed * scale
    typed_ops::add_into(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &mut scratch.mla.normed,
    )?;
    typed_ops::bf16_to_f32_into(ctx, &pplx.pplx_routed_out, &mut pplx.pplx_routed_f32)?;
    kimi_scaled_add_f32_bf16_to_bf16(
        ctx,
        &pplx.pplx_routed_f32,
        KIMI_K2_ROUTER_SCALE,
        &scratch.mla.normed,
        &mut scratch.mla.hidden,
    )?;

    Ok(())
}
