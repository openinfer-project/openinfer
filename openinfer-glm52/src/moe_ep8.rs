//! GLM5.2 EP8 routed-MoE decode: the PR3 grouped FP8 chain with DeepEP
//! dispatch/combine substituted for the local scatter/combine stand-ins.
//!
//! Every rank runs the same collective per MoE layer: a rank with an active
//! request dispatches its token (under the DP8 coordinator that is every
//! rank, padding tokens included) and computes its 32 local experts over
//! whatever the all-to-all delivered:
//!
//! ```text
//! dispatch(x bf16, global topk)          # collective; recv = expert-major
//!   → recv IS fp8+scales (source-quantized) #   aligned segments + psum_expert
//!   → metadata: psum i32 → offsets i64
//!   → TRTLLM grouped W13 (32 groups) → weighted SiLU·quant (recv weights)
//!   → TRTLLM grouped W2 → combine        # collective; sums slots per token
//! ```
//!
//! The dispatch payload is bf16 (the shim's wire format); each rank re-quants
//! its received rows before the grouped GEMMs. Re-quant / SiLU / grouped GEMM
//! cover `bound_rows` — the host-derived aligned-segment bound for the step's
//! global token count (2080 at the DP8 protocol's 8 tokens/step vs the
//! 10240-row capacity); the metadata
//! kernel device-traps if a real segment ends past it. Rows past the bound
//! hold stale bytes, but every kernel is row-independent and combine only
//! reads slots addressed by the dispatch metadata — the PR3 row-isolation
//! invariant. The whole layer is host-quiet (no D2H, shapes fixed per batch
//! size) — CUDA-graph capturable, same bar as PR3.

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ffi::DeepEpInfo;
use openinfer_kernels::ops::{
    DeepEpDispatchScratch, GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT, Glm52DeepEp,
    Glm52MoeQuantShape, Glm52TrtllmGroupedFp8Kind, Glm52TrtllmGroupedOffsetScaleLayout,
    glm52_deepep_info, glm52_deepgemm_grouped_fp8_metadata_launch,
    glm52_fp8_per_token_group_quant_bf16_launch,
    glm52_silu_and_mul_weighted_per_token_group_quant_bf16_bounded_launch,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::moe_decode::{
    EXPERTS, Glm52MoeExpertBank, Glm52MoeRouterWeights, Glm52MoeSharedExpert, HIDDEN,
    HIDDEN_SCALE_COLS, QUANT_GROUP, RoutedTopk, TOPK, W2_K, W2_N, W2_SCALE_COLS, W2_SCALE_ROWS,
    W13_K, W13_N, W13_SCALE_ROWS, grouped_gemm_into,
};

/// One rank's weights for one EP8 MoE layer: router and shared expert run
/// where the token lives (every rank, under the DP8 coordinator); the bank
/// holds this rank's 32 local experts.
pub(crate) struct Glm52MoeEp8LayerWeights {
    pub(crate) router: Glm52MoeRouterWeights,
    pub(crate) shared: Glm52MoeSharedExpert,
    pub(crate) bank: Glm52MoeExpertBank,
}

/// Per-rank DeepEP context plus every buffer the MoE chain touches, allocated
/// once at startup at worst-case capacity (crash early on OOM; pointer-stable
/// for future graph capture; no per-layer allocator traffic — the per-call
/// alloc/free/memset churn was ~35% of decode CUDA API time at bs=1).
pub(crate) struct Glm52MoeEp8State {
    ep: Glm52DeepEp,
    scratch: DeepEpDispatchScratch,
    info: DeepEpInfo,
    /// FP8-dispatch send operands for this rank's (single) token: the
    /// per-128-group quantized activation + scales, produced on the SOURCE
    /// rank (bit-identical to the retired recv-side re-quant — dispatch is
    /// byte-preserving, so quant commutes).
    x_fp8: CudaSlice<u8>,
    x_sf: CudaSlice<f32>,
    recv_topk_weight: CudaSlice<f32>,
    recv_src_metadata: CudaSlice<i32>,
    /// Dispatch inputs for token-less expert ranks (num_tokens = 0 still
    /// requires valid pointers).
    zero_x_fp8: CudaSlice<u8>,
    zero_x_sf: CudaSlice<f32>,
    zero_topk_idx: CudaSlice<i32>,
    zero_topk_weight: CudaSlice<f32>,
    /// Grouped-GEMM chain workspace. Rows past a launch's `bound_rows` hold
    /// stale bytes from earlier layers — row-isolated by construction (every
    /// consumer addresses rows through the dispatch metadata / expert
    /// offsets, never past the last aligned segment end).
    act_fp8: CudaSlice<u8>,
    act_scale: CudaSlice<f32>,
    expert_offsets: CudaSlice<i64>,
    w13_scale_tma: CudaSlice<f32>,
    w13_out: CudaSlice<bf16>,
    w2_act: CudaSlice<u8>,
    w2_act_scale: CudaSlice<f32>,
    w2_scale_tma: CudaSlice<f32>,
    expert_out: CudaSlice<bf16>,
    /// The combined routed output for this rank's (single) source token,
    /// rewritten by every `glm52_moe_ep8_routed_forward` call. Persistent so
    /// the whole chain is allocation-free (pointer-stable for graph capture).
    combined: CudaSlice<bf16>,
}

impl Glm52MoeEp8State {
    /// Collective: all ranks' worker threads must call concurrently with the
    /// same unique id, device set.
    pub(crate) fn new(
        ctx: &DeviceContext,
        unique_id: &[u8; 128],
        num_ranks: usize,
        rank_idx: usize,
    ) -> Result<Self> {
        let info = glm52_deepep_info();
        ensure!(
            info.num_experts as usize == EXPERTS
                && info.num_topk as usize == TOPK
                && info.hidden as usize == HIDDEN
                && info.num_local_experts as usize == crate::weights::GLM52_LOCAL_EXPERTS
                && info.expert_alignment as usize == GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT,
            "GLM5.2 DeepEP shim config does not match the model: {info:?}"
        );
        ensure!(
            num_ranks == info.num_ranks as usize,
            "GLM5.2 DeepEP requires {} ranks, got {num_ranks}",
            info.num_ranks
        );
        let ep = Glm52DeepEp::new(unique_id, num_ranks, rank_idx)
            .with_context(|| format!("GLM5.2 rank {rank_idx} DeepEP context create"))?;
        let expanded = info.decode_worst_expanded_tokens as usize;
        let recv_tokens = info.decode_worst_recv_tokens as usize;
        let n_local = info.num_local_experts as usize;
        let w13_scale_tma_len =
            Glm52TrtllmGroupedOffsetScaleLayout::f32(expanded, HIDDEN_SCALE_COLS, n_local)
                .output_len()?;
        let w2_scale_tma_len =
            Glm52TrtllmGroupedOffsetScaleLayout::f32(expanded, W2_SCALE_COLS, n_local)
                .output_len()?;
        // Padding-rank dispatch operands: the quant of a zero token, produced
        // by the same kernel so the bytes match what the retired recv-side
        // re-quant emitted for dispatched zero rows (fp8 zeros + eps-floored
        // scales).
        let zero_x_fp8 = {
            let zeros = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
            let mut z_fp8 = ctx.stream.alloc_zeros::<u8>(HIDDEN)?;
            let mut z_sf = ctx.stream.alloc_zeros::<f32>(HIDDEN_SCALE_COLS)?;
            glm52_fp8_per_token_group_quant_bf16_launch(
                ctx,
                Glm52MoeQuantShape {
                    rows: 1,
                    width: HIDDEN,
                    group_size: QUANT_GROUP,
                },
                &zeros,
                &mut z_fp8,
                &mut z_sf,
            )?;
            (z_fp8, z_sf)
        };
        Ok(Self {
            ep,
            scratch: DeepEpDispatchScratch::new_decode_with(ctx, &info)?,
            info,
            x_fp8: ctx.stream.alloc_zeros(HIDDEN)?,
            x_sf: ctx.stream.alloc_zeros(HIDDEN_SCALE_COLS)?,
            recv_topk_weight: ctx.stream.alloc_zeros(expanded)?,
            recv_src_metadata: ctx.stream.alloc_zeros(recv_tokens * (TOPK + 2))?,
            zero_x_fp8: zero_x_fp8.0,
            zero_x_sf: zero_x_fp8.1,
            zero_topk_idx: ctx.stream.alloc_zeros(TOPK)?,
            zero_topk_weight: ctx.stream.alloc_zeros(TOPK)?,
            act_fp8: ctx.stream.alloc_zeros(expanded * W13_K)?,
            act_scale: ctx.stream.alloc_zeros(expanded * HIDDEN_SCALE_COLS)?,
            expert_offsets: ctx.stream.alloc_zeros(n_local + 1)?,
            w13_scale_tma: ctx.stream.alloc_zeros(w13_scale_tma_len)?,
            w13_out: ctx.stream.alloc_zeros(expanded * W13_N)?,
            w2_act: ctx.stream.alloc_zeros(expanded * W2_K)?,
            w2_act_scale: ctx.stream.alloc_zeros(expanded * W2_SCALE_COLS)?,
            w2_scale_tma: ctx.stream.alloc_zeros(w2_scale_tma_len)?,
            expert_out: ctx.stream.alloc_zeros(expanded * W2_N)?,
            combined: ctx.stream.alloc_zeros(HIDDEN)?,
        })
    }

    /// The routed output written by the last `glm52_moe_ep8_routed_forward`
    /// call that dispatched a token (valid only when that call returned
    /// `true`).
    pub(crate) fn combined(&self) -> &CudaSlice<bf16> {
        &self.combined
    }
}

/// One EP8 MoE layer's routed contribution — a collective every rank must
/// enter simultaneously per layer. A rank with a token passes its
/// post-attention normed hidden + router output; on `Ok(true)` the routed
/// output `[HIDDEN]` (route weight and ×2.5 scaling already folded) is in
/// `state.combined()`. A token-less rank passes `None` and gets `Ok(false)`.
/// The DP8 production path always passes `Some` (idle ranks step a padding
/// token); `None` survives for the EP8 layer oracle gate's single-dispatcher
/// replay.
///
/// `global_tokens` is the total token count dispatched across ALL ranks this
/// step — every rank must pass the same value (a protocol constant of the
/// coordinator, not derived from device data, so the chain stays host-quiet).
/// It bounds how many recv rows the re-quant/SiLU kernels must cover instead
/// of the fixed worst case: each source token expands to at most `TOPK` rows
/// on this rank, and each non-empty local expert segment pads to the
/// `GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT` boundary, so the last aligned
/// segment end is at most `min(expanded, g*TOPK + (ALIGN-1)*min(g*TOPK,
/// n_local))`. At the DP8 protocol's 8 tokens/step that is 2080 rows vs the
/// 10240-row capacity.
pub(crate) fn glm52_moe_ep8_routed_forward(
    ctx: &DeviceContext,
    state: &mut Glm52MoeEp8State,
    bank: &Glm52MoeExpertBank,
    token: Option<(&CudaSlice<bf16>, &RoutedTopk)>,
    global_tokens: usize,
) -> Result<bool> {
    let n_local = state.info.num_local_experts as usize;
    ensure!(
        bank.n_experts() == n_local,
        "GLM5.2 EP8 MoE needs the {n_local}-expert rank-local bank, got {}",
        bank.n_experts()
    );
    let expanded = state.info.decode_worst_expanded_tokens as usize;
    let num_tokens = usize::from(token.is_some());
    ensure!(
        global_tokens >= num_tokens,
        "GLM5.2 EP8 MoE global_tokens {global_tokens} < local tokens {num_tokens}"
    );
    let expanded_rows = global_tokens * TOPK;
    let bound_rows = expanded.min(
        expanded_rows + (GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT - 1) * expanded_rows.min(n_local),
    );

    // Collective FP8 dispatch: quantize this rank's token on the source
    // side (identical math to the retired recv-side re-quant — dispatch is
    // byte-preserving, so the recv bytes are bit-identical), then send fp8 +
    // per-group scales; the copy epilogue emits the grouped-GEMM operands
    // (`act_fp8` + plain `[rows, 48]` `act_scale`) directly.
    {
        if let Some((normed, _)) = token {
            glm52_fp8_per_token_group_quant_bf16_launch(
                ctx,
                Glm52MoeQuantShape {
                    rows: 1,
                    width: HIDDEN,
                    group_size: QUANT_GROUP,
                },
                normed,
                &mut state.x_fp8,
                &mut state.x_sf,
            )?;
        }
        let (x_fp8, x_sf, topk_idx, topk_weight) = match token {
            Some((_, route)) => (
                &state.x_fp8,
                &state.x_sf,
                &route.topk_idx,
                &route.topk_weight,
            ),
            None => (
                &state.zero_x_fp8,
                &state.zero_x_sf,
                &state.zero_topk_idx,
                &state.zero_topk_weight,
            ),
        };
        state.ep.decode_dispatch_fp8(
            ctx,
            x_fp8,
            x_sf,
            topk_idx,
            topk_weight,
            num_tokens,
            &mut state.scratch,
            &mut state.act_fp8,
            &mut state.act_scale,
            &mut state.recv_topk_weight,
            &mut state.recv_src_metadata,
        )?;
    }

    // psum_expert (i32 aligned running ends) → expert_offsets (i64 segment
    // starts) for the grouped GEMMs. Passing `bound_rows` as the capacity
    // makes the kernel device-trap if the ranks disagreed about
    // `global_tokens` — the only place in the chain that sees the real psum.
    // Runs BEFORE the re-quant so `expert_offsets[n_local]` (the real aligned
    // end) can bound the row-proportional kernels below on-device.
    glm52_deepgemm_grouped_fp8_metadata_launch(
        ctx,
        n_local,
        bound_rows,
        &state.scratch.psum_expert,
        &mut state.expert_offsets,
    )?;

    // W13 grouped FP8 GEMM (gate|up) over the local expert segments. The
    // GEMM's row capacity is `bound_rows` too: the scale relayout and grid
    // are capacity-proportional, and every real segment ends below the bound.
    grouped_gemm_into(
        ctx,
        Glm52TrtllmGroupedFp8Kind::W13,
        n_local,
        bound_rows,
        W13_N,
        W13_K,
        HIDDEN_SCALE_COLS,
        W13_SCALE_ROWS,
        &state.act_fp8,
        &state.act_scale,
        &bank.w13_weight,
        &bank.w13_scale,
        &state.expert_offsets,
        &mut state.w13_scale_tma,
        &mut state.w13_out,
    )?;

    // Weighted SwiGLU quant: silu(gate)*up*route_weight → fp8 W2 input. The
    // per-slot weight is exactly what dispatch delivered per expanded row.
    glm52_silu_and_mul_weighted_per_token_group_quant_bf16_bounded_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: bound_rows,
            width: W2_K,
            group_size: QUANT_GROUP,
        },
        &state.w13_out,
        &state.recv_topk_weight,
        &mut state.w2_act,
        &mut state.w2_act_scale,
        &state.expert_offsets,
        n_local,
    )?;

    // W2 grouped FP8 GEMM (down).
    grouped_gemm_into(
        ctx,
        Glm52TrtllmGroupedFp8Kind::W2,
        n_local,
        bound_rows,
        W2_N,
        W2_K,
        W2_SCALE_COLS,
        W2_SCALE_ROWS,
        &state.w2_act,
        &state.w2_act_scale,
        &bank.w2_weight,
        &bank.w2_scale,
        &state.expert_offsets,
        &mut state.w2_scale_tma,
        &mut state.expert_out,
    )?;

    // Collective combine: weighted expert outputs → per-source-token sums,
    // into the persistent `combined` buffer (`num_tokens` is 0 or 1, and the
    // buffer holds one HIDDEN row).
    let topk_idx = match token {
        Some((_, route)) => &route.topk_idx,
        None => &state.zero_topk_idx,
    };
    state.ep.decode_combine(
        ctx,
        &state.expert_out,
        &state.scratch,
        &state.recv_src_metadata,
        topk_idx,
        num_tokens,
        &mut state.combined,
    )?;

    Ok(token.is_some())
}
