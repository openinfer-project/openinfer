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
//! All four `EpBackend` methods run on `moe_stream` (the stream the
//! upstream worker thread fences against). Local compute (routing,
//! shared expert, grouped FP4 GEMMs) runs on `ctx.stream`. Cross-stream
//! handoffs are explicit CUDA events.
//!
//! # Shared expert + combine
//!
//! Shared-expert output is staged into `moe_scratch.out` first; the
//! final `combine_recv` runs with `accumulate=true` so the routed
//! contribution is added in-place. This trades the AG/RS path's F32
//! intermediate accumulation for BF16 accumulation — numerics will
//! drift slightly vs NCCL AG/RS and need a tolerance bump in the
//! exact-E2E gate (see `docs/projects/pplx-ep-integration.md`).
//!
//! # expert_indptr build
//!
//! `dispatch_recv` writes the per-local-expert received-token counter
//! into the upstream worker's `tokens_per_expert` buffer (exposed via
//! [`EpBackend::tokens_per_expert_ptr`]). We pull it back to host,
//! exclusive-prefix-sum on host (≤256 / `world_size` u32 entries —
//! trivial), and H2D into [`MoePplxScratch::expert_indptr`]. Adding a
//! device-side prefix-sum kernel is a future optimization once the
//! basic flow lands.

use std::ffi::c_void;
use std::ptr;

use cudarc::driver::CudaStream;
use pegainfer_comm::{EpBackend, ScalarType};

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
    moe_stream: &CudaStream,
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

    // ---- 1. Local route -> (indices, weights) on ctx.stream ----
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

    // moe_stream waits for ctx.stream so dispatch sees route output.
    let route_done = ctx.stream.record_event(Some(
        cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
    ))?;
    moe_stream.wait(&route_done)?;

    // ---- 2. dispatch_send on moe_stream ----
    let stream_raw = moe_stream.cu_stream() as u64;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(moe_stream);
        let (idx_ptr, _idx_guard) = moe_scratch.route_indices.device_ptr(moe_stream);
        let (w_ptr, _w_guard) = moe_scratch.route_weights.device_ptr(moe_stream);
        // Upstream a2a_dispatch_send.cu uses `(uint4*)(x_ptr + token * x_stride)`,
        // so x_stride is in BYTES (sizeof(bf16) * hidden_dim).
        ep.dispatch_send(
            num_tokens,
            x_ptr as *const c_void,
            input.hidden_dim * std::mem::size_of::<u16>(), // x_stride: BF16 BYTES per row
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

    // ---- 3. Shared expert on ctx.stream (overlaps with dispatch_send) ----
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

    // ---- 4. dispatch_recv on moe_stream -> expert-packed activations ----
    {
        let (out_num_ptr, _g0) = moe_scratch.num_recv_tokens.device_ptr_mut(moe_stream);
        let (out_x_ptr, _g1) = moe_scratch.expanded_input.data.device_ptr_mut(moe_stream);
        ep.dispatch_recv(
            out_num_ptr as *mut i32,
            out_x_ptr as *mut c_void,
            moe_scratch.expanded_input.hidden_dim, // out_x_stride
            ptr::null_mut(),
            0,
            0,
            stream_raw,
        )
        .with_context(|| format!("pplx dispatch_recv layer {layer}"))?;
    }

    // ---- 5. D2H per-expert recv counts, prefix-sum on host ----
    // dispatch_recv writes `out_num_tokens_ptr[expert]` for each local
    // expert on moe_stream; sync once so the host readback is safe.
    moe_stream.synchronize()?;
    let mut per_expert_recv = vec![0i32; local_experts];
    ctx.stream
        .memcpy_dtoh(&moe_scratch.num_recv_tokens, &mut per_expert_recv)?;
    ctx.sync()?;
    // Exclusive prefix sum on host -> expert_indptr.
    let mut indptr_host = vec![0i32; local_experts + 1];
    let mut acc: i32 = 0;
    for i in 0..local_experts {
        indptr_host[i] = acc;
        acc = acc
            .checked_add(per_expert_recv[i])
            .ok_or_else(|| anyhow::anyhow!("pplx expert_indptr overflow"))?;
    }
    indptr_host[local_experts] = acc;
    let num_recv_tokens = acc as usize;
    ensure!(
        num_recv_tokens <= moe_scratch.expanded_input.seq_capacity(),
        "pplx num_recv_tokens={num_recv_tokens} exceeds expanded_input capacity {}",
        moe_scratch.expanded_input.seq_capacity()
    );
    ctx.stream
        .memcpy_htod(&indptr_host, &mut moe_scratch.expert_indptr)?;

    // ---- 6. Local grouped FP4 experts over the received activations ----
    // The grouped GEMM helpers read `expanded_input.seq_len` AND
    // `expert_indptr[..local_experts + 1]`; align both with the dynamic
    // `num_recv_tokens`. The routing-side fields on the plan view
    // (`pos_to_token`, `token_topk_to_pos`, `num_expanded`,
    // `routed.weights/indices`) are not touched by the grouped GEMM path
    // — they point at the local route slices only to satisfy the borrow.
    moe_scratch.expanded_input.seq_len = num_recv_tokens;
    moe_scratch.expert_gate.seq_len = num_recv_tokens;
    moe_scratch.expert_up.seq_len = num_recv_tokens;
    moe_scratch.expert_out.seq_len = num_recv_tokens;
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
        num_expanded: num_recv_tokens,
    };
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
    drop(plan);

    // ---- 7. combine_send on moe_stream ----
    let experts_done = ctx.stream.record_event(Some(
        cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
    ))?;
    moe_stream.wait(&experts_done)?;
    {
        let (exp_ptr, _g) = moe_scratch.expert_out.data.device_ptr(moe_stream);
        ep.combine_send(
            exp_ptr as *const c_void,
            moe_scratch.expert_out.hidden_dim * std::mem::size_of::<u16>(), // expert_x_stride: BF16 BYTES
            stream_raw,
        )
        .with_context(|| format!("pplx combine_send layer {layer}"))?;
    }

    // ---- 8. combine_recv: routed += into `out` (already holds shared) ----
    {
        let (out_ptr, _g0) = moe_scratch.out.data.device_ptr_mut(moe_stream);
        let (idx_ptr, _g1) = moe_scratch.route_indices.device_ptr(moe_stream);
        let (w_ptr, _g2) = moe_scratch.route_weights.device_ptr(moe_stream);
        ep.combine_recv(
            num_tokens,
            num_recv_tokens,
            ScalarType::BF16,
            out_ptr as *mut c_void,
            moe_scratch.out.hidden_dim * std::mem::size_of::<u16>(), // out_tokens_stride: BF16 BYTES
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
    let combine_done = moe_stream.record_event(Some(
        cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
    ))?;
    ctx.stream.wait(&combine_done)?;

    Ok(&moe_scratch.out)
}
