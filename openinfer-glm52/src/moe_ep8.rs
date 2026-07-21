//! GLM5.2 EP8 routed-MoE decode: the DeepEP dispatch/combine collectives
//! around the DeepGEMM masked grouped fp8 GEMMs.
//!
//! Every rank runs the same collective per MoE layer: a rank with an active
//! request dispatches its token (under the DP8 coordinator that is every
//! rank, padding tokens included) and computes its 32 local experts over
//! whatever the all-to-all delivered:
//!
//! ```text
//! dispatch(x bf16, global topk)          # collective; recv = expert-major
//!   → metadata: psum i32 → offsets i64 + masked_m + row_map
//!   → re-quant recv rows → masked [32, 64, k] fp8 + mn-major scales
//!   → DeepGEMM masked W13 (32 groups) → weighted SiLU·quant (recv weights)
//!   → DeepGEMM masked W2 → remap masked→aligned slots
//!   → combine                            # collective; sums slots per token
//! ```
//!
//! The dispatch payload is bf16 (the shim's wire format); each rank re-quants
//! its received rows before the grouped GEMMs. The masked layout gives each
//! local expert a fixed 64-row slab (`[32, 64, k]` — the DP8 protocol's
//! worst case is all 64 global tokens routing one row each to one expert), so
//! the GEMM reads no offset indirection and the activation scales go straight
//! into the mn-major TMA layout the kernel wants — no relayout kernel. The
//! replaced TRTLLM grouped GEMM measured 1.5-1.9× slower on the same
//! data/distribution (64-row M-tile against ~1-8 real rows per expert).
//!
//! Re-quant / SiLU cover `bound_rows` — the host-derived aligned-segment
//! bound for the step's global token count (2528 at the DP8 protocol's 64
//! tokens/step vs the 10240-row capacity); the metadata kernel device-traps
//! if a real segment ends past it. Masked rows past an expert's `masked_m`
//! hold stale bytes, but every kernel is row-independent and combine only
//! reads slots addressed by the dispatch metadata — the PR3 row-isolation
//! invariant. The whole layer is host-quiet (no D2H, shapes fixed per batch
//! size) — CUDA-graph capturable, same bar as PR3.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ffi::DeepEpInfo;
use openinfer_kernels::ops::DeepEpDispatchScratch;
use openinfer_kernels::ops::GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT;
use openinfer_kernels::ops::GLM52_DEEPGEMM_MASKED_CAP;
use openinfer_kernels::ops::GLM52_DEEPGEMM_MASKED_GROUPS;
use openinfer_kernels::ops::Glm52DeepEp;
use openinfer_kernels::ops::Glm52DeepGemmGroupedFp8Kind;
use openinfer_kernels::ops::Glm52MoeQuantShape;
use openinfer_kernels::ops::glm52_deepep_info;
use openinfer_kernels::ops::glm52_deepgemm_grouped_fp8_metadata_launch;
use openinfer_kernels::ops::glm52_deepgemm_masked_grouped_fp8_launch;
use openinfer_kernels::ops::glm52_deepgemm_masked_out_to_aligned_launch;
use openinfer_kernels::ops::glm52_fp8_per_token_group_quant_bf16_masked_launch;
use openinfer_kernels::ops::glm52_silu_and_mul_weighted_per_token_group_quant_bf16_masked_launch;
use openinfer_kernels::tensor::DeviceContext;

use crate::moe_decode::EXPERTS;
use crate::moe_decode::Glm52MoeExpertBank;
use crate::moe_decode::Glm52MoeRouterWeights;
use crate::moe_decode::Glm52MoeSharedExpert;
use crate::moe_decode::HIDDEN;
use crate::moe_decode::HIDDEN_SCALE_COLS;
use crate::moe_decode::QUANT_GROUP;
use crate::moe_decode::RoutedTopk;
use crate::moe_decode::TOPK;
use crate::moe_decode::W2_K;
use crate::moe_decode::W2_N;
use crate::moe_decode::W2_SCALE_COLS;
use crate::moe_decode::W13_K;
use crate::moe_decode::W13_N;

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
    recv_x: CudaSlice<bf16>,
    recv_topk_weight: CudaSlice<f32>,
    recv_src_metadata: CudaSlice<i32>,
    /// Dispatch inputs for token-less expert ranks (num_tokens = 0 still
    /// requires valid pointers).
    zero_x: CudaSlice<bf16>,
    zero_topk_idx: CudaSlice<i32>,
    zero_topk_weight: CudaSlice<f32>,
    /// Aligned-segment metadata (starts + aligned end) and its masked-layout
    /// bridge: per-expert real row counts and the aligned-row → masked-slot
    /// map. Rewritten by the metadata kernel every layer.
    expert_offsets: CudaSlice<i64>,
    masked_m: CudaSlice<i32>,
    row_map: CudaSlice<i32>,
    /// Masked grouped-GEMM chain workspace (`[32, 64, ·]` slabs). Rows past
    /// an expert's `masked_m` hold stale bytes from earlier layers —
    /// row-isolated by construction (every consumer addresses rows through
    /// `row_map` / `masked_m`, never past an expert's real count).
    act_masked: CudaSlice<u8>,
    act_scale_masked: CudaSlice<f32>,
    w13_out_masked: CudaSlice<bf16>,
    w2_act_masked: CudaSlice<u8>,
    w2_act_scale_masked: CudaSlice<f32>,
    expert_out_masked: CudaSlice<bf16>,
    /// W2 output remapped into the aligned recv slots `decode_combine`
    /// addresses (sized at the recv capacity).
    expert_out: CudaSlice<bf16>,
    /// The combined routed output for this rank's source tokens (row-major
    /// `[tokens, HIDDEN]`), rewritten by every `glm52_moe_ep8_routed_forward`
    /// call. Persistent so the whole chain is allocation-free (pointer-stable
    /// for graph capture); sized at the shim's per-rank decode cap.
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
        ensure!(
            info.num_local_experts as usize == GLM52_DEEPGEMM_MASKED_GROUPS,
            "GLM5.2 masked grouped GEMM instantiation does not cover the DeepEP shim: {info:?}"
        );
        let ep = Glm52DeepEp::new(unique_id, num_ranks, rank_idx)
            .with_context(|| format!("GLM5.2 rank {rank_idx} DeepEP context create"))?;
        let expanded = info.decode_worst_expanded_tokens as usize;
        let recv_tokens = info.decode_worst_recv_tokens as usize;
        let n_local = info.num_local_experts as usize;
        let masked_rows = GLM52_DEEPGEMM_MASKED_GROUPS * GLM52_DEEPGEMM_MASKED_CAP;
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
            expert_offsets: ctx.stream.alloc_zeros(n_local + 1)?,
            masked_m: ctx.stream.alloc_zeros(n_local)?,
            row_map: ctx.stream.alloc_zeros(expanded)?,
            act_masked: ctx.stream.alloc_zeros(masked_rows * W13_K)?,
            act_scale_masked: ctx.stream.alloc_zeros(masked_rows * HIDDEN_SCALE_COLS)?,
            w13_out_masked: ctx.stream.alloc_zeros(masked_rows * W13_N)?,
            w2_act_masked: ctx.stream.alloc_zeros(masked_rows * W2_K)?,
            w2_act_scale_masked: ctx.stream.alloc_zeros(masked_rows * W2_SCALE_COLS)?,
            expert_out_masked: ctx.stream.alloc_zeros(masked_rows * W2_N)?,
            expert_out: ctx.stream.alloc_zeros(expanded * W2_N)?,
            combined: ctx
                .stream
                .alloc_zeros(info.decode_max_tokens_per_rank as usize * HIDDEN)?,
        })
    }

    /// The routed output rows (`[tokens, HIDDEN]`) written by the last
    /// `glm52_moe_ep8_routed_forward` call that dispatched tokens (valid only
    /// when that call returned `true`).
    pub(crate) fn combined(&self) -> &CudaSlice<bf16> {
        &self.combined
    }
}

/// One EP8 MoE layer's routed contribution — a collective every rank must
/// enter simultaneously per layer. A rank with tokens passes its
/// post-attention normed hidden rows + router output (`[T, HIDDEN]` /
/// `[T, 8]`) and the row count; on `Ok(true)` the routed output
/// `[T, HIDDEN]` (route weight and ×2.5 scaling already folded) is in
/// `state.combined()`. A token-less rank passes `None` and gets `Ok(false)`.
/// The DP8 production path always passes `Some` (pad rows are dispatched like
/// real ones); `None` survives for the EP8 layer oracle gate's
/// single-dispatcher replay.
///
/// `global_tokens` is the total token count dispatched across ALL ranks this
/// step — every rank must pass the same value (a protocol constant of the
/// coordinator, not derived from device data, so the chain stays host-quiet).
/// It bounds how many recv rows the re-quant/SiLU kernels must cover instead
/// of the fixed worst case: each source token expands to at most `TOPK` rows
/// on this rank, and each non-empty local expert segment pads to the
/// `GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT` boundary, so the last aligned
/// segment end is at most `min(expanded, g*TOPK + (ALIGN-1)*min(g*TOPK,
/// n_local))`. At the DP8 protocol's 64 tokens/step that is 2528 rows vs the
/// 10240-row capacity.
pub(crate) fn glm52_moe_ep8_routed_forward(
    ctx: &DeviceContext,
    state: &mut Glm52MoeEp8State,
    bank: &Glm52MoeExpertBank,
    token: Option<(&CudaSlice<bf16>, &RoutedTopk, usize)>,
    global_tokens: usize,
) -> Result<bool> {
    let n_local = state.info.num_local_experts as usize;
    ensure!(
        bank.n_experts() == n_local,
        "GLM5.2 EP8 MoE needs the {n_local}-expert rank-local bank, got {}",
        bank.n_experts()
    );
    let expanded = state.info.decode_worst_expanded_tokens as usize;
    let num_tokens = token.map_or(0, |(_, _, t)| t);
    ensure!(
        token.is_none() || num_tokens > 0,
        "GLM5.2 EP8 MoE dispatching rank must pass a positive token count"
    );
    ensure!(
        global_tokens >= num_tokens && global_tokens > 0,
        "GLM5.2 EP8 MoE global_tokens {global_tokens} must be positive and >= local tokens {num_tokens}"
    );
    // Each source token contributes at most one row per expert, so the
    // masked per-expert capacity covers the step iff it covers the global
    // token count (a protocol constant — the DeepEP shim's own capacity is
    // larger; the metadata kernel device-traps as the backstop).
    ensure!(
        global_tokens <= GLM52_DEEPGEMM_MASKED_CAP,
        "GLM5.2 EP8 MoE global_tokens {global_tokens} exceeds the masked grouped GEMM per-expert capacity {GLM52_DEEPGEMM_MASKED_CAP}"
    );
    let expanded_rows = global_tokens * TOPK;
    let bound_rows = expanded.min(
        expanded_rows + (GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT - 1) * expanded_rows.min(n_local),
    );

    // Collective dispatch: bf16 token rows → expert-major aligned recv slots.
    {
        let (x, topk_idx, topk_weight) = match token {
            Some((normed, route, _)) => (normed, &route.topk_idx, &route.topk_weight),
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

    // psum_expert (i32 aligned running ends) → expert_offsets (i64 segment
    // starts), per-expert real row counts (masked_m — the GEMM's grouped
    // layout), and the aligned-row → masked-slot map. Passing `bound_rows`
    // as the capacity makes the kernel device-trap if the ranks disagreed
    // about `global_tokens` — the only place in the chain that sees the real
    // psum. Runs BEFORE the re-quant so `expert_offsets[n_local]` (the real
    // aligned end) can bound the row-proportional kernels below on-device.
    glm52_deepgemm_grouped_fp8_metadata_launch(
        ctx,
        n_local,
        bound_rows,
        &state.scratch.psum_expert,
        &mut state.expert_offsets,
        &mut state.masked_m,
        &mut state.row_map,
    )?;

    // Re-quant the received bf16 rows to fp8 in the masked layout. The grid
    // covers `bound_rows` (the host-known worst case — CUDA-graph shape
    // stability), but blocks at or past the device-side aligned end retire
    // immediately, and alignment-gap rows (row_map < 0) are skipped. The
    // per-row scales go straight into the mn-major TMA layout the masked
    // GEMM reads.
    glm52_fp8_per_token_group_quant_bf16_masked_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: bound_rows,
            width: W13_K,
            group_size: QUANT_GROUP,
        },
        GLM52_DEEPGEMM_MASKED_GROUPS,
        GLM52_DEEPGEMM_MASKED_CAP,
        &state.recv_x,
        &mut state.act_masked,
        &mut state.act_scale_masked,
        &state.expert_offsets,
        n_local,
        &state.row_map,
    )?;

    // W13 masked grouped FP8 GEMM (gate|up) over the local experts.
    glm52_deepgemm_masked_grouped_fp8_launch(
        ctx,
        Glm52DeepGemmGroupedFp8Kind::W13,
        &state.act_masked,
        &state.act_scale_masked,
        &bank.w13_weight,
        &bank.w13_scale,
        &state.masked_m,
        &mut state.w13_out_masked,
    )?;

    // Weighted SwiGLU quant: silu(gate)*up*route_weight → fp8 W2 input. The
    // gate|up rows are already masked (the W13 GEMM wrote them there); the
    // per-slot weight is exactly what dispatch delivered per expanded row.
    glm52_silu_and_mul_weighted_per_token_group_quant_bf16_masked_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: bound_rows,
            width: W2_K,
            group_size: QUANT_GROUP,
        },
        GLM52_DEEPGEMM_MASKED_GROUPS,
        GLM52_DEEPGEMM_MASKED_CAP,
        &state.w13_out_masked,
        &state.recv_topk_weight,
        &mut state.w2_act_masked,
        &mut state.w2_act_scale_masked,
        &state.expert_offsets,
        n_local,
        &state.row_map,
    )?;

    // W2 masked grouped FP8 GEMM (down).
    glm52_deepgemm_masked_grouped_fp8_launch(
        ctx,
        Glm52DeepGemmGroupedFp8Kind::W2,
        &state.w2_act_masked,
        &state.w2_act_scale_masked,
        &bank.w2_weight,
        &bank.w2_scale,
        &state.masked_m,
        &mut state.expert_out_masked,
    )?;

    // Masked GEMM output → the aligned recv slots decode_combine addresses.
    glm52_deepgemm_masked_out_to_aligned_launch(
        ctx,
        W2_N,
        &state.expert_out_masked,
        &state.masked_m,
        &state.expert_offsets,
        &mut state.expert_out,
    )?;

    // Collective combine: weighted expert outputs → per-source-token sums,
    // into the persistent `combined` buffer (`[num_tokens, HIDDEN]` rows of
    // the per-rank-cap-sized buffer).
    let topk_idx = match token {
        Some((_, route, _)) => &route.topk_idx,
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
