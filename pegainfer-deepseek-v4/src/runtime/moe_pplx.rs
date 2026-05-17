//! pplx-garden NVLink + RDMA MoE all-to-all decode path (feature `pplx-ep`).
//!
//! Drop-in replacement for [`super::moe::decode_moe_ag_rs_bf16_hidden_with_scratch`]:
//! same call-site contract, same routing / shared-expert / grouped-FP4
//! GEMM helpers, but cross-rank token movement uses the upstream
//! four-step pipeline (`dispatch_send → dispatch_recv → combine_send →
//! combine_recv`) wrapped by [`pegainfer_comm::EpBackend`].
//!
//! # Stream layout
//!
//! PPLX dispatch/combine and local MoE compute all run on `ctx.stream`.
//! Earlier two-stream overlap did not move H200 `output_len=64` p50, so the
//! integration keeps one stream and avoids explicit cross-stream handoff
//! events in the decode hot path.
//!
//! # Shared expert + combine
//!
//! Shared-expert output is staged into `moe_scratch.out` first; the
//! final `combine_recv` runs with `accumulate=true` so the routed
//! contribution is added in-place. This trades the AG/RS path's F32
//! intermediate accumulation for BF16 accumulation — numerics will
//! drift slightly vs NCCL AG/RS and need a tolerance bump in the
//! exact-E2E gate (see `docs/models/deepseek-v4/pplx-ep-integration.md`).
//!
//! # expert_indptr build
//!
//! `dispatch_recv` writes per-local-expert received-token counts into a
//! caller-provided device buffer. A tiny device kernel converts those counts
//! into pplx's padded expert-major `expert_indptr`, keeping decode on-stream.

use std::ffi::c_void;
use std::ptr;

use cudarc::driver::CudaStream;
use pegainfer_comm::{EpBackend, ScalarType};
use pegainfer_kernels::ffi;

use super::core::shared_expert_forward_bf16_hidden_scratch;
use super::moe::{
    hash_route_bf16_hidden_into, local_experts_forward_packed_bf16_hidden_scratch,
    score_route_bf16_hidden_into,
};
use super::*;

/// Decode-MoE driver routed through the pplx backend.
///
/// Signature mirrors `decode_moe_ag_rs_bf16_hidden_with_scratch` so the
/// rank worker can swap call sites with one `if let Some(ep)` branch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_moe_pplx_bf16_hidden_with_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    ptr_cache: &MoeGroupedPtrCache,
    ep: &mut EpBackend,
    layer: usize,
    input: &Bf16HiddenStates,
    token_ids: &CudaSlice<u32>,
    shared_scratch: &mut SharedExpertScratch,
    moe_scratch: &'a mut MoePplxScratch,
) -> Result<&'a Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        token_ids.len() == input.seq_len,
        "decode MoE pplx token count mismatch: tokens={}, hidden seq_len={}",
        token_ids.len(),
        input.seq_len
    );
    let ffn = weights.ffn(layer)?;
    let num_tokens = input.seq_len;
    let world_size = weights.world_size();
    ensure!(
        config.n_routed_experts.is_multiple_of(world_size),
        "n_routed_experts={} must be divisible by world_size={world_size}",
        config.n_routed_experts
    );
    let local_experts = config.n_routed_experts / world_size;
    let topk = config.n_activated_experts;
    let comm_stream = ctx.stream.as_ref();
    let stream_raw = comm_stream.cu_stream() as u64;

    // ---- 1. Local route -> (indices, weights) on ctx.stream ----
    {
        if layer < config.n_hash_layers {
            let _ = hash_route_bf16_hidden_into(
                ctx,
                config,
                input,
                &ffn,
                token_ids,
                &mut moe_scratch.route_weights,
                &mut moe_scratch.route_indices,
            )?;
        } else {
            let _ = score_route_bf16_hidden_into(
                ctx,
                config,
                input,
                &ffn,
                &mut moe_scratch.route_weights,
                &mut moe_scratch.route_indices,
            )?;
        }
    }
    // ---- 2. Shared expert on ctx.stream ----
    {
        let shared = shared_expert_forward_bf16_hidden_scratch(
            ctx,
            input,
            &ffn,
            config.swiglu_limit,
            shared_scratch,
        )?;
        ensure!(shared.hidden_dim == moe_scratch.out.hidden_dim);
        ensure!(shared.seq_len == num_tokens);
        moe_scratch.out.seq_len = num_tokens;
        // Stage shared expert into `out`; `combine_recv` accumulates routed
        // into it (D2D same-stream copy).
        let copy_len = num_tokens * shared.hidden_dim;
        ctx.stream.memcpy_dtod(
            &shared.data.slice(0..copy_len),
            &mut moe_scratch.out.data.slice_mut(0..copy_len),
        )?;
    }

    // ---- 3. dispatch_send on ctx.stream ----
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(comm_stream);
        let (idx_ptr, _idx_guard) = moe_scratch.route_indices.device_ptr(comm_stream);
        let (w_ptr, _w_guard) = moe_scratch.route_weights.device_ptr(comm_stream);
        // Upstream a2a kernels address `(uint4*)(x_ptr + token * x_stride)`,
        // so x_stride is in BYTES (sizeof(bf16) * hidden_dim).
        let x_stride = input.hidden_dim * std::mem::size_of::<u16>();
        ep.dispatch_send(
            num_tokens,
            x_ptr as *const c_void,
            x_stride,
            ptr::null(),
            0,
            0,
            idx_ptr as *const i32,
            topk, // indices_stride: ELEMENTS
            w_ptr as *const f32,
            topk,        // weights_stride: ELEMENTS
            ptr::null(), // num_tokens known on host
            stream_raw,
        )
        .with_context(|| format!("pplx dispatch_send layer {layer}"))?;
    }

    // ---- 4. dispatch_recv -> expert-packed activations ----
    {
        let (out_num_ptr, _g0) = moe_scratch
            .recv_tokens_per_expert
            .device_ptr_mut(comm_stream);
        let (out_x_ptr, _g1) = moe_scratch.expanded_input.data.device_ptr_mut(comm_stream);
        ep.dispatch_recv(
            out_num_ptr as *mut i32,
            out_x_ptr as *mut c_void,
            moe_scratch.expanded_input.hidden_dim * std::mem::size_of::<u16>(),
            ptr::null_mut(),
            0,
            0,
            stream_raw,
        )
        .with_context(|| format!("pplx dispatch_recv layer {layer}"))?;
    }

    // ---- 5. Device prefix: recv counts -> padded expert_indptr ----
    // pplx writes payload rows at `padded_index`, so grouped GEMM must see
    // the same padded expert-major layout.
    ensure!(
        moe_scratch.expert_padding > 0,
        "pplx expert_padding must be positive"
    );
    {
        build_padded_expert_indptr_on_stream(ctx, comm_stream, local_experts, moe_scratch)?;
    }

    // ---- 6. Local grouped FP4 experts over the received activations ----
    // The grouped GEMM helpers take a host `rows` launch bound. Use the
    // preallocated scratch capacity so the hot path no longer needs to know
    // the dynamic padded token count on host; actual expert ranges still come
    // from device `expert_indptr`. The routing-side fields on the plan view
    // (`pos_to_token`, `token_topk_to_pos`, `num_expanded`,
    // `routed.weights/indices`) are not touched by the grouped GEMM path
    // — they point at the local route slices only to satisfy the borrow.
    let num_padded_recv_tokens = moe_scratch.expanded_input.seq_capacity();
    ensure!(
        num_padded_recv_tokens <= moe_scratch.expanded_input.seq_capacity(),
        "pplx grouped rows={} exceed expanded_input capacity {}",
        num_padded_recv_tokens,
        moe_scratch.expanded_input.seq_capacity()
    );
    moe_scratch.expanded_input.seq_len = num_padded_recv_tokens;
    moe_scratch.expert_gate.seq_len = num_padded_recv_tokens;
    moe_scratch.expert_up.seq_len = num_padded_recv_tokens;
    moe_scratch.expert_out.seq_len = num_padded_recv_tokens;
    let plan = MoeFusedRoutePlanView {
        routed: RoutedExpertsView {
            weights: &moe_scratch.route_weights,
            indices: &moe_scratch.route_indices,
            topk,
            seq_len: num_tokens,
        },
        pos_to_token: &moe_scratch.route_indices,
        token_topk_to_pos: &moe_scratch.route_indices,
        expert_indptr: &moe_scratch.expert_indptr,
        local_experts,
        num_expanded: num_padded_recv_tokens,
    };
    {
        let _ = local_experts_forward_packed_bf16_hidden_scratch(
            ctx,
            config,
            ptr_cache,
            layer,
            &moe_scratch.expanded_input,
            &plan,
            &mut moe_scratch.expert_gate,
            &mut moe_scratch.expert_up,
            &mut moe_scratch.expert_out,
            &mut moe_scratch.fp4_act_workspace,
            &mut moe_scratch.fp4_act_scale_workspace,
        )?;
    }
    drop(plan);

    // ---- 7. combine_send on ctx.stream ----
    {
        let (exp_ptr, _g) = moe_scratch.expert_out.data.device_ptr(comm_stream);
        ep.combine_send(
            exp_ptr as *const c_void,
            moe_scratch.expert_out.hidden_dim * std::mem::size_of::<u16>(), // expert_x_stride: BF16 BYTES
            stream_raw,
        )
        .with_context(|| format!("pplx combine_send layer {layer}"))?;
    }

    // ---- 8. combine_recv: routed += into `out` (already holds shared) ----
    {
        let (out_ptr, _g0) = moe_scratch.out.data.device_ptr_mut(comm_stream);
        let (idx_ptr, _g1) = moe_scratch.route_indices.device_ptr(comm_stream);
        let (w_ptr, _g2) = moe_scratch.route_weights.device_ptr(comm_stream);
        ep.combine_recv(
            num_tokens,
            0, // Currently ignored by the a2a combine_recv kernel.
            ScalarType::BF16,
            out_ptr as *mut c_void,
            moe_scratch.out.hidden_dim, // out_tokens_stride: BF16 ELEMENTS
            idx_ptr as *const i32,
            topk, // indices_stride
            w_ptr as *const f32,
            topk, // weights_stride
            ptr::null(),
            true, // accumulate routed into shared-expert output
            stream_raw,
        )
        .with_context(|| format!("pplx combine_recv layer {layer}"))?;
    }

    Ok(&moe_scratch.out)
}

fn build_padded_expert_indptr_on_stream(
    ctx: &RankGpuContext,
    stream: &CudaStream,
    local_experts: usize,
    moe_scratch: &mut MoePplxScratch,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        local_experts <= i32::MAX as usize,
        "pplx local_experts exceeds i32: {local_experts}"
    );
    ensure!(
        moe_scratch.expert_padding <= i32::MAX as usize,
        "pplx expert_padding exceeds i32: {}",
        moe_scratch.expert_padding
    );
    let (counts_ptr, _counts_guard) = moe_scratch.recv_tokens_per_expert.device_ptr(stream);
    let (indptr_ptr, _indptr_guard) = moe_scratch.expert_indptr.device_ptr_mut(stream);
    let result = unsafe {
        ffi::deepseek_pplx_padded_expert_indptr_cuda(
            counts_ptr as *const i32,
            indptr_ptr as *mut i32,
            local_experts as i32,
            moe_scratch.expert_padding as i32,
            stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}
