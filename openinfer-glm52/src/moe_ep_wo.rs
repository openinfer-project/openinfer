//! GLM5.2 weight-only routed-MoE decode: the DeepEP dispatch/combine
//! collectives around the weight-only masked grouped mma GEMMs, generic over
//! the shim instantiation (EP4 = 4 ranks × 64 local experts, EP8 = 8 × 32 —
//! every shape below comes from the shim's `DeepEpInfo`, so a new EP width
//! is a new instantiation, not new code).
//!
//! Same DP-coordinator protocol as the masked EP8 chain (every rank enters
//! the collective per MoE layer with the agreed `global_tokens`), but the
//! expert GEMMs run the arch-portable weight-only chain
//! (`glm52_moe_ep_wo.cu`) instead of the sm_90a-only DeepGEMM masked chain:
//!
//! ```text
//! dispatch(x bf16, global topk)        # collective; recv = expert-major
//!   → psum → compact tile list (expert, aligned row base, rows ≤ 8)
//!   → W13 weight-only mma over tiles (bf16 recv rows × deq fp8 bank)
//!   → silu(gate)·up (bf16)
//!   → W2 weight-only mma × route_weight → aligned slots
//!   → combine                          # collective; sums slots per token
//! ```
//!
//! The chain never leaves the DeepEP aligned receive layout: expert segments
//! are 64-aligned and 8 | 64, so an 8-row tile never straddles an expert —
//! no fp8 activation re-quant, no masked relayout, no output remap. Shapes
//! are host-quiet at fixed worst-case tile counts (CUDA-graph capturable,
//! same bar as the EP8 chain); the tile kernel device-traps on a cross-rank
//! token-count disagreement.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ffi::DeepEpInfo;
use openinfer_kernels::ops::DeepEpAbi;
use openinfer_kernels::ops::DeepEpBase;
use openinfer_kernels::ops::DeepEpDispatchScratch;
use openinfer_kernels::ops::GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT;
use openinfer_kernels::ops::Glm52DeepEpAbi;
use openinfer_kernels::ops::Glm52DeepGemmGroupedFp8Kind;
use openinfer_kernels::ops::Glm52Ep4DeepEpAbi;
use openinfer_kernels::ops::Glm52Ep16DeepEpAbi;
use openinfer_kernels::ops::Glm52Ep32DeepEpAbi;
use openinfer_kernels::ops::Glm52Ep64DeepEpAbi;
use openinfer_kernels::ops::glm52_moe_ep_wo_masked_mma_launch;
use openinfer_kernels::ops::glm52_moe_ep_wo_max_tiles;
use openinfer_kernels::ops::glm52_moe_ep_wo_silu_launch;
use openinfer_kernels::ops::glm52_moe_ep_wo_tiles_launch;
use openinfer_kernels::tensor::DeviceContext;

use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::moe_decode::EXPERTS;
use crate::moe_decode::Glm52MoeExpertBank;
use crate::moe_decode::HIDDEN;
use crate::moe_decode::RoutedTopk;
use crate::moe_decode::TOPK;
use crate::moe_decode::W2_K;
use crate::moe_decode::W2_N;
use crate::moe_decode::W13_N;
use crate::moe_ep8::Glm52MoeEp8State;
use crate::moe_ep8::glm52_moe_ep8_routed_forward;

/// Per-rank DeepEP context plus every buffer the weight-only chain touches,
/// allocated once at startup at worst-case capacity (pointer-stable for
/// graph capture, no per-layer allocator traffic — the EP8 discipline).
pub(crate) struct Glm52MoeEpWoState<A: DeepEpAbi> {
    ep: DeepEpBase<A>,
    scratch: DeepEpDispatchScratch,
    info: DeepEpInfo,
    /// Fixed worst-case tile budget: the protocol's max global tokens
    /// (`num_ranks × GLM52_MAX_BATCH_PER_RANK`) at launch — every bucket's
    /// GEMM grid uses this one shape.
    max_tiles: usize,
    recv_x: CudaSlice<bf16>,
    recv_topk_weight: CudaSlice<f32>,
    recv_src_metadata: CudaSlice<i32>,
    /// Dispatch inputs for token-less expert ranks (num_tokens = 0 still
    /// requires valid pointers).
    zero_x: CudaSlice<bf16>,
    zero_topk_idx: CudaSlice<i32>,
    zero_topk_weight: CudaSlice<f32>,
    /// Compact tile work list (int2 entries as an i32 pair buffer) + device
    /// tile count, rewritten by the tiles kernel every layer.
    tiles: CudaSlice<i32>,
    tile_count: CudaSlice<i32>,
    /// W13 gate|up output rows in the aligned receive layout. Rows past a
    /// segment's real count hold stale bytes — row-isolated by construction
    /// (every consumer walks the tile list; combine only reads slots the
    /// dispatch metadata addresses).
    w13_out: CudaSlice<bf16>,
    /// SiLU output = W2 activation rows (bf16, aligned layout).
    w2_act: CudaSlice<bf16>,
    /// W2 output in the aligned recv slots `decode_combine` addresses.
    expert_out: CudaSlice<bf16>,
    /// The combined routed output for this rank's source tokens (row-major
    /// `[tokens, HIDDEN]`), sized at the shim's per-rank decode cap.
    combined: CudaSlice<bf16>,
}

impl<A: DeepEpAbi> Glm52MoeEpWoState<A> {
    /// The routed output rows written by the last dispatching
    /// [`glm52_moe_ep_wo_routed_forward`] call (valid only when that call
    /// returned `true`).
    pub(crate) fn combined(&self) -> &CudaSlice<bf16> {
        &self.combined
    }

    /// Collective: all ranks' worker threads must call concurrently with the
    /// same unique id, device set.
    pub(crate) fn new(
        ctx: &DeviceContext,
        unique_id: &[u8; 128],
        num_ranks: usize,
        rank_idx: usize,
    ) -> Result<Self> {
        let info = A::info();
        ensure!(
            info.num_experts as usize == EXPERTS
                && info.num_topk as usize == TOPK
                && info.hidden as usize == HIDDEN
                && info.expert_alignment as usize == GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT,
            "GLM5.2 EP-wo DeepEP shim config does not match the model: {info:?}"
        );
        ensure!(
            num_ranks == info.num_ranks as usize,
            "GLM5.2 EP-wo DeepEP requires {} ranks, got {num_ranks}",
            info.num_ranks
        );
        ensure!(
            info.num_local_experts as usize * info.num_ranks as usize == EXPERTS,
            "GLM5.2 EP-wo shim local experts do not partition the routed set: {info:?}"
        );
        let ep = DeepEpBase::<A>::new(unique_id, num_ranks, rank_idx)
            .with_context(|| format!("GLM5.2 rank {rank_idx} EP-wo DeepEP context create"))?;
        let expanded = info.decode_worst_expanded_tokens as usize;
        let recv_tokens = info.decode_worst_recv_tokens as usize;
        let n_local = info.num_local_experts as usize;
        let max_global_tokens = num_ranks * GLM52_MAX_BATCH_PER_RANK;
        let max_tiles = glm52_moe_ep_wo_max_tiles(n_local, max_global_tokens, TOPK);
        Ok(Self {
            ep,
            scratch: DeepEpDispatchScratch::new_decode_with(ctx, &info)?,
            info,
            max_tiles,
            recv_x: ctx.stream.alloc_zeros(expanded * HIDDEN)?,
            recv_topk_weight: ctx.stream.alloc_zeros(expanded)?,
            recv_src_metadata: ctx.stream.alloc_zeros(recv_tokens * (TOPK + 2))?,
            zero_x: ctx.stream.alloc_zeros(HIDDEN)?,
            zero_topk_idx: ctx.stream.alloc_zeros(TOPK)?,
            zero_topk_weight: ctx.stream.alloc_zeros(TOPK)?,
            tiles: ctx.stream.alloc_zeros(2 * max_tiles)?,
            tile_count: ctx.stream.alloc_zeros(1)?,
            w13_out: ctx.stream.alloc_zeros(expanded * W13_N)?,
            w2_act: ctx.stream.alloc_zeros(expanded * W2_K)?,
            expert_out: ctx.stream.alloc_zeros(expanded * W2_N)?,
            combined: ctx
                .stream
                .alloc_zeros(info.decode_max_tokens_per_rank as usize * HIDDEN)?,
        })
    }
}

/// One weight-only MoE layer's routed contribution — a collective every rank
/// must enter simultaneously per layer, same contract as
/// [`glm52_moe_ep8_routed_forward`] (see there for the `token` /
/// `global_tokens` semantics).
pub(crate) fn glm52_moe_ep_wo_routed_forward<A: DeepEpAbi>(
    ctx: &DeviceContext,
    state: &mut Glm52MoeEpWoState<A>,
    bank: &Glm52MoeExpertBank,
    token: Option<(&CudaSlice<bf16>, &RoutedTopk, usize)>,
    global_tokens: usize,
) -> Result<bool> {
    let n_local = state.info.num_local_experts as usize;
    ensure!(
        bank.n_experts() == n_local,
        "GLM5.2 EP-wo MoE needs the {n_local}-expert rank-local bank, got {}",
        bank.n_experts()
    );
    let expanded = state.info.decode_worst_expanded_tokens as usize;
    let num_tokens = token.map_or(0, |(_, _, t)| t);
    ensure!(
        token.is_none() || num_tokens > 0,
        "GLM5.2 EP-wo MoE dispatching rank must pass a positive token count"
    );
    ensure!(
        global_tokens >= num_tokens && global_tokens > 0,
        "GLM5.2 EP-wo MoE global_tokens {global_tokens} must be positive and >= local tokens {num_tokens}"
    );
    // The startup tile budget covers the protocol's max global token count;
    // a larger step would overflow the fixed tile buffer.
    let max_global_tokens = state.info.num_ranks as usize * GLM52_MAX_BATCH_PER_RANK;
    ensure!(
        global_tokens <= max_global_tokens,
        "GLM5.2 EP-wo MoE global_tokens {global_tokens} exceeds the protocol cap {max_global_tokens}"
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

    // psum_expert (i32 aligned running ends) → compact tile list. Device-
    // traps if any segment ends past `bound_rows` or holds more than
    // `global_tokens` rows — the only place in the chain that sees the real
    // psum, so a cross-rank disagreement crashes here instead of multiplying
    // stale rows into real outputs.
    glm52_moe_ep_wo_tiles_launch(
        ctx,
        n_local,
        bound_rows,
        global_tokens,
        state.max_tiles,
        &state.scratch.psum_expert,
        &mut state.tiles,
        &mut state.tile_count,
    )?;

    // W13 (gate|up) weight-only masked mma over the tile list.
    glm52_moe_ep_wo_masked_mma_launch(
        ctx,
        Glm52DeepGemmGroupedFp8Kind::W13,
        n_local,
        state.max_tiles,
        &state.recv_x,
        &bank.w13_weight,
        &bank.w13_scale,
        &state.tiles,
        &state.tile_count,
        None,
        &mut state.w13_out,
    )?;

    // silu(gate)·up → bf16 W2 activation rows (route weight applies below).
    glm52_moe_ep_wo_silu_launch(
        ctx,
        W2_K,
        state.max_tiles,
        &state.w13_out,
        &state.tiles,
        &state.tile_count,
        &mut state.w2_act,
    )?;

    // W2 (down) weight-only masked mma; the dispatch route weight scales the
    // f32 output before the bf16 store, and the rows land straight in the
    // aligned slots `decode_combine` addresses.
    glm52_moe_ep_wo_masked_mma_launch(
        ctx,
        Glm52DeepGemmGroupedFp8Kind::W2,
        n_local,
        state.max_tiles,
        &state.w2_act,
        &bank.w2_weight,
        &bank.w2_scale,
        &state.tiles,
        &state.tile_count,
        Some(&state.recv_topk_weight),
        &mut state.expert_out,
    )?;

    // Collective combine: weighted expert outputs → per-source-token sums.
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

/// One rank's EP MoE state: the topology decides the routed-expert chain at
/// launch. Both chains share the DeepEP protocol and the
/// [`glm52_moe_ep8_routed_forward`] calling contract.
pub(crate) enum Glm52MoeEpState {
    /// EP8: sm_90a DeepGEMM masked grouped fp8 chain (8×H200 production).
    MaskedFp8(Box<Glm52MoeEp8State>),
    /// Arch-portable weight-only mma chain, one variant per shim
    /// instantiation (EP8 uses it on Blackwell, where the masked DeepGEMM
    /// chain's sm_90a kernels don't run).
    WeightOnlyEp4(Box<Glm52MoeEpWoState<Glm52Ep4DeepEpAbi>>),
    WeightOnlyEp8(Box<Glm52MoeEpWoState<Glm52DeepEpAbi>>),
    WeightOnlyEp16(Box<Glm52MoeEpWoState<Glm52Ep16DeepEpAbi>>),
    WeightOnlyEp32(Box<Glm52MoeEpWoState<Glm52Ep32DeepEpAbi>>),
    WeightOnlyEp64(Box<Glm52MoeEpWoState<Glm52Ep64DeepEpAbi>>),
}

impl Glm52MoeEpState {
    pub(crate) fn routed_forward(
        &mut self,
        ctx: &DeviceContext,
        bank: &Glm52MoeExpertBank,
        token: Option<(&CudaSlice<bf16>, &RoutedTopk, usize)>,
        global_tokens: usize,
    ) -> Result<bool> {
        match self {
            Self::MaskedFp8(state) => {
                glm52_moe_ep8_routed_forward(ctx, state, bank, token, global_tokens)
            }
            Self::WeightOnlyEp4(state) => {
                glm52_moe_ep_wo_routed_forward(ctx, state, bank, token, global_tokens)
            }
            Self::WeightOnlyEp8(state) => {
                glm52_moe_ep_wo_routed_forward(ctx, state, bank, token, global_tokens)
            }
            Self::WeightOnlyEp16(state) => {
                glm52_moe_ep_wo_routed_forward(ctx, state, bank, token, global_tokens)
            }
            Self::WeightOnlyEp32(state) => {
                glm52_moe_ep_wo_routed_forward(ctx, state, bank, token, global_tokens)
            }
            Self::WeightOnlyEp64(state) => {
                glm52_moe_ep_wo_routed_forward(ctx, state, bank, token, global_tokens)
            }
        }
    }

    /// The routed output rows written by the last dispatching
    /// `routed_forward` call (valid only when that call returned `true`).
    pub(crate) fn combined(&self) -> &CudaSlice<bf16> {
        match self {
            Self::MaskedFp8(state) => state.combined(),
            Self::WeightOnlyEp4(state) => state.combined(),
            Self::WeightOnlyEp8(state) => state.combined(),
            Self::WeightOnlyEp16(state) => state.combined(),
            Self::WeightOnlyEp32(state) => state.combined(),
            Self::WeightOnlyEp64(state) => state.combined(),
        }
    }
}
