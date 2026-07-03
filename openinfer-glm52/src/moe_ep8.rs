//! GLM5.2 EP8 routed-MoE decode: the PR3 grouped FP8 chain with DeepEP
//! dispatch/combine substituted for the local scatter/combine stand-ins.
//!
//! Every rank runs the same collective per MoE layer. Rank 0 (the only DP
//! rank) dispatches its token; ranks 1..7 dispatch zero tokens and only
//! compute their 32 local experts over whatever the all-to-all delivered:
//!
//! ```text
//! dispatch(x bf16, global topk)          # collective; recv = expert-major
//!   → re-quant recv rows to fp8          #   aligned segments + psum_expert
//!   → metadata: psum i32 → offsets i64
//!   → TRTLLM grouped W13 (32 groups) → weighted SiLU·quant (recv weights)
//!   → TRTLLM grouped W2 → combine        # collective; sums slots per token
//! ```
//!
//! The dispatch payload is bf16 (the shim's wire format); each rank re-quants
//! its received rows before the grouped GEMMs. Re-quant / SiLU launch at the
//! fixed worst-case row capacity: pad rows hold stale bytes, but every kernel
//! is row-independent and combine only reads slots addressed by the dispatch
//! metadata — the PR3 row-isolation invariant. The whole layer is host-quiet
//! (no D2H, fixed shapes) — CUDA-graph capturable, same bar as PR3.

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ffi::DeepEpInfo;
use openinfer_kernels::ops::{
    DeepEpDispatchScratch, GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT, Glm52DeepEp,
    Glm52MoeQuantShape, Glm52TrtllmGroupedFp8Kind, glm52_deepep_info,
    glm52_deepgemm_grouped_fp8_metadata_launch, glm52_fp8_per_token_group_quant_bf16_launch,
    glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::moe_decode::{
    Glm52MoeExpertBank, Glm52MoeRouterWeights, Glm52MoeSharedExpert, HIDDEN_SCALE_COLS, RoutedTopk,
    W2_K, W2_N, W2_SCALE_COLS, W2_SCALE_ROWS, W13_K, W13_N, W13_SCALE_ROWS, grouped_gemm,
};

const HIDDEN: usize = 6144;
const EXPERTS: usize = 256;
const TOPK: usize = 8;
const QUANT_GROUP: usize = 128;

/// Rank-0's weights for one EP8 MoE layer: the router and shared expert run
/// only where the token lives; the bank holds this rank's 32 local experts.
/// Expert ranks (1..7) hold only a bank per layer.
pub(crate) struct Glm52MoeEp8LayerWeights {
    pub(crate) router: Glm52MoeRouterWeights,
    pub(crate) shared: Glm52MoeSharedExpert,
    pub(crate) bank: Glm52MoeExpertBank,
}

/// Per-rank DeepEP context plus the dispatch-side buffers, allocated once at
/// startup (fixed worst case → crash early on OOM; pointer-stable for future
/// graph capture). Compute scratch downstream of dispatch allocates per call
/// through the stream-ordered pool (PR3-proven capture-safe).
pub(crate) struct Glm52MoeEp8State {
    ep: Glm52DeepEp,
    scratch: DeepEpDispatchScratch,
    info: DeepEpInfo,
    recv_x: CudaSlice<bf16>,
    recv_topk_weight: CudaSlice<f32>,
    recv_src_metadata: CudaSlice<i32>,
    /// Dispatch inputs for token-less expert ranks (num_tokens = 0 still
    /// requires valid pointers).
    zero_x: CudaSlice<bf16>,
    zero_topk_idx: CudaSlice<i32>,
    zero_topk_weight: CudaSlice<f32>,
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
        Ok(Self {
            ep,
            scratch: DeepEpDispatchScratch::new_decode_with(ctx, &info)?,
            info,
            recv_x: ctx.stream.alloc_zeros(expanded * HIDDEN)?,
            recv_topk_weight: ctx.stream.alloc_zeros(expanded)?,
            recv_src_metadata: ctx.stream.alloc_zeros(recv_tokens * (TOPK + 2))?,
            zero_x: ctx.stream.alloc_zeros(HIDDEN)?,
            zero_topk_idx: ctx.stream.alloc_zeros(TOPK)?,
            zero_topk_weight: ctx.stream.alloc_zeros(TOPK)?,
        })
    }
}

/// One EP8 MoE layer's routed contribution — a collective every rank must
/// enter simultaneously per layer. Rank 0 passes its post-attention normed
/// hidden + router output and gets back `Some(routed[HIDDEN])` (route weight
/// and ×2.5 scaling already folded); expert ranks pass `None` and get `None`.
pub(crate) fn glm52_moe_ep8_routed_forward(
    ctx: &DeviceContext,
    state: &mut Glm52MoeEp8State,
    bank: &Glm52MoeExpertBank,
    token: Option<(&CudaSlice<bf16>, &RoutedTopk)>,
) -> Result<Option<CudaSlice<bf16>>> {
    let n_local = state.info.num_local_experts as usize;
    ensure!(
        bank.n_experts() == n_local,
        "GLM5.2 EP8 MoE needs the {n_local}-expert rank-local bank, got {}",
        bank.n_experts()
    );
    let stream = &ctx.stream;
    let expanded = state.info.decode_worst_expanded_tokens as usize;
    let num_tokens = usize::from(token.is_some());

    // Collective dispatch: bf16 token rows → expert-major aligned recv slots.
    {
        let (x, topk_idx, topk_weight) = match token {
            Some((normed, route)) => (normed, &route.topk_idx, &route.topk_weight),
            None => (&state.zero_x, &state.zero_topk_idx, &state.zero_topk_weight),
        };
        state.ep.decode_dispatch(
            ctx,
            x,
            topk_idx,
            topk_weight,
            num_tokens,
            &mut state.scratch,
            &mut state.recv_x,
            &mut state.recv_topk_weight,
            &mut state.recv_src_metadata,
        )?;
    }

    // Re-quant the received bf16 rows to fp8 (worst-case rows; pad rows are
    // garbage but row-isolated).
    let mut act_fp8 = stream.alloc_zeros::<u8>(expanded * W13_K)?;
    let mut act_scale = stream.alloc_zeros::<f32>(expanded * HIDDEN_SCALE_COLS)?;
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: expanded,
            width: W13_K,
            group_size: QUANT_GROUP,
        },
        &state.recv_x,
        &mut act_fp8,
        &mut act_scale,
    )?;

    // psum_expert (i32 aligned running ends) → expert_offsets (i64 segment
    // starts) for the grouped GEMMs.
    let mut expert_offsets = stream.alloc_zeros::<i64>(n_local + 1)?;
    let mut w13_problem_sizes = stream.alloc_zeros::<i32>(n_local * 3)?;
    let mut w2_problem_sizes = stream.alloc_zeros::<i32>(n_local * 3)?;
    glm52_deepgemm_grouped_fp8_metadata_launch(
        ctx,
        n_local,
        expanded,
        &state.scratch.psum_expert,
        &mut expert_offsets,
        &mut w13_problem_sizes,
        &mut w2_problem_sizes,
    )?;

    // W13 grouped FP8 GEMM (gate|up) over the local expert segments.
    let w13_out = grouped_gemm(
        ctx,
        Glm52TrtllmGroupedFp8Kind::W13,
        n_local,
        expanded,
        W13_N,
        W13_K,
        HIDDEN_SCALE_COLS,
        W13_SCALE_ROWS,
        &act_fp8,
        &act_scale,
        &bank.w13_weight,
        &bank.w13_scale,
        &expert_offsets,
    )?;

    // Weighted SwiGLU quant: silu(gate)*up*route_weight → fp8 W2 input. The
    // per-slot weight is exactly what dispatch delivered per expanded row.
    let mut w2_act = stream.alloc_zeros::<u8>(expanded * W2_K)?;
    let mut w2_act_scale = stream.alloc_zeros::<f32>(expanded * W2_SCALE_COLS)?;
    glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: expanded,
            width: W2_K,
            group_size: QUANT_GROUP,
        },
        &w13_out,
        &state.recv_topk_weight,
        &mut w2_act,
        &mut w2_act_scale,
    )?;

    // W2 grouped FP8 GEMM (down).
    let expert_out = grouped_gemm(
        ctx,
        Glm52TrtllmGroupedFp8Kind::W2,
        n_local,
        expanded,
        W2_N,
        W2_K,
        W2_SCALE_COLS,
        W2_SCALE_ROWS,
        &w2_act,
        &w2_act_scale,
        &bank.w2_weight,
        &bank.w2_scale,
        &expert_offsets,
    )?;

    // Collective combine: weighted expert outputs → per-source-token sums.
    let mut combined = stream.alloc_zeros::<bf16>(HIDDEN.max(num_tokens * HIDDEN))?;
    let topk_idx = match token {
        Some((_, route)) => &route.topk_idx,
        None => &state.zero_topk_idx,
    };
    state.ep.decode_combine(
        ctx,
        &expert_out,
        &state.scratch,
        &state.recv_src_metadata,
        topk_idx,
        num_tokens,
        &mut combined,
    )?;

    Ok(token.map(|_| combined))
}
