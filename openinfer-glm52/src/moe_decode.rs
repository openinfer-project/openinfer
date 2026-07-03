//! GLM5.2 single-layer routed-MoE decode forward for bs=1 (EP1: all 256 routed
//! experts local; no all-to-all).
//!
//! Two expert paths behind one signature, sharing the router, the expert-major
//! fp8 weights, and the weighted-SwiGLU fold:
//!
//! - `Grouped` — the DeepEP-shaped spine PR4 swaps its scatter/combine stand-ins
//!   out of: quant hidden -> `route_offsets` (device-side `expert_offsets[E+1]`)
//!   -> `scatter` into expert-major aligned slots -> TRTLLM grouped FP8 GEMM
//!   (gate|up) -> weighted SwiGLU re-quant -> grouped GEMM (down) -> combine sum.
//! - `Gemv` — the measured bs=1 winner from the PP8 campaign: broadcast the bf16
//!   hidden across the top-k experts, dequant the fp8 weight on the fly
//!   (weight-memory-bound, no 64-row M-tile padding), weighted SwiGLU in bf16,
//!   per-slot down GEMV, plain slot sum.
//!
//! The route weight (already normalized and x2.5-scaled by the router) is folded
//! into the W2 input by the weighted SwiGLU, so both combines are plain sums. The
//! shared expert is a plain fp8 MLP added unscaled.
//!
//! Buffers are allocated per call through the stream-ordered cudarc pool — the
//! PP8 branch's `graph_alloc_probe` proved pool allocation inside CUDA-graph
//! capture replays correctly, so this stays graph-capturable without an arena.
//! Grouped-path pad rows may hold stale data from a previous token; every kernel
//! after `scatter` is row-independent and `combine` reads only the top-k experts'
//! slots via `expert_offsets`, so stale pad rows never reach the output.

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT, GLM52_GEMV_KIND_W2, GLM52_GEMV_KIND_W13,
    Glm52MoeQuantShape, Glm52RouterBatch, Glm52RouterConfig, Glm52RouterOutput,
    Glm52TrtllmGroupedFp8Contract, Glm52TrtllmGroupedFp8Kind, Glm52TrtllmGroupedOffsetScaleLayout,
    add_batch, glm52_deepgemm_grouped_offset_tma_aligned_f32_launch,
    glm52_fp8_per_token_group_quant_bf16_launch, glm52_moe_combine_launch,
    glm52_moe_combine_slots_launch, glm52_moe_fp8_weight_only_gemv_launch,
    glm52_moe_route_offsets_launch, glm52_moe_route_scatter_launch, glm52_router_noaux_tc_launch,
    glm52_silu_and_mul_weighted_bf16_launch,
    glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch, glm52_trtllm_grouped_fp8_launch,
};
use openinfer_kernels::tensor::{DeviceContext, HiddenStates};

use crate::fp8::{Glm52ProjBytes, ProjWeight, bytes_to_f32, fp8_mlp};

const HIDDEN: usize = 6144;
const EXPERTS: usize = 256;
const TOPK: usize = 8;
const INTERMEDIATE: usize = 2048;
const QUANT_GROUP: usize = 128;

const W13_N: usize = 2 * INTERMEDIATE; // 4096 (gate|up)
const W13_K: usize = HIDDEN; // 6144
const W2_N: usize = HIDDEN; // 6144
const W2_K: usize = INTERMEDIATE; // 2048

const HIDDEN_SCALE_COLS: usize = HIDDEN / QUANT_GROUP; // 48
const W13_SCALE_ROWS: usize = W13_N / QUANT_GROUP; // 32
const W2_SCALE_COLS: usize = W2_K / QUANT_GROUP; // 16
const W2_SCALE_ROWS: usize = W2_N / QUANT_GROUP; // 48

/// bs=1 expert-major row capacity for the grouped path: each of the top-k distinct
/// experts owns at most one row, padded to the 64-row alignment, so
/// `TOPK * ALIGNMENT` is a tight upper bound (`route_offsets` emits
/// `expert_offsets[E] <=` it).
const M_CAPACITY: usize = TOPK * GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT; // 512

/// Which expert-GEMM implementation runs the routed contribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Glm52MoeExpertPath {
    /// TRTLLM grouped FP8 GEMM over expert-major aligned slots — the layout the
    /// DeepEP dispatch delivers (PR4 swaps scatter/combine for dispatch/combine).
    Grouped,
    /// Weight-only fp8 GEMV (bf16 activation, on-the-fly dequant) — the measured
    /// bs=1 decode winner; no activation quant, no M-tile padding.
    Gemv,
}

/// Raw per-expert checkpoint bytes for one routed expert (fp8 block-scaled).
pub(crate) struct Glm52MoeRoutedExpertBytes<'a> {
    pub(crate) gate: Glm52ProjBytes<'a>, // [INTERMEDIATE, HIDDEN]
    pub(crate) up: Glm52ProjBytes<'a>,   // [INTERMEDIATE, HIDDEN]
    pub(crate) down: Glm52ProjBytes<'a>, // [HIDDEN, INTERMEDIATE]
}

/// All weights for one MoE layer: the 256 routed experts as expert-major fp8
/// `[EXPERTS, n, k]` + f32 block scales `[EXPERTS, n/128, k/128]` (the layout both
/// the grouped GEMM and the GEMV index directly by expert id), plus the router
/// gate/bias and the single shared expert. Built once on device; borrowed by every
/// decode step.
pub(crate) struct Glm52MoeLayerWeights {
    gate_weight: CudaSlice<u8>,  // bf16 [EXPERTS, HIDDEN]
    e_score_bias: CudaSlice<u8>, // f32  [EXPERTS]
    w13_weight: CudaSlice<u8>,   // fp8  [EXPERTS, W13_N, W13_K]
    w13_scale: CudaSlice<f32>,   // f32  [EXPERTS, W13_SCALE_ROWS, HIDDEN_SCALE_COLS]
    w2_weight: CudaSlice<u8>,    // fp8  [EXPERTS, W2_N, W2_K]
    w2_scale: CudaSlice<f32>,    // f32  [EXPERTS, W2_SCALE_ROWS, W2_SCALE_COLS]
    shared_gate: ProjWeight,     // fp8  [INTERMEDIATE, HIDDEN]
    shared_up: ProjWeight,       // fp8  [INTERMEDIATE, HIDDEN]
    shared_down: ProjWeight,     // fp8  [HIDDEN, INTERMEDIATE]
}

impl Glm52MoeLayerWeights {
    /// Pack the 256 per-expert checkpoint tensors into the expert-major grouped
    /// buffers and upload everything (the oracle/test path). W13 = per-expert
    /// `[gate; up]` rows with scales concatenated likewise; W2 = down. The
    /// production `from_device` path against the resident EP8 slab lands in PR4
    /// and must reuse this exact layout.
    pub(crate) fn from_host(
        ctx: &DeviceContext,
        gate_weight: &[u8],
        e_score_bias: &[u8],
        experts: &[Glm52MoeRoutedExpertBytes<'_>],
        shared_gate: &Glm52ProjBytes<'_>,
        shared_up: &Glm52ProjBytes<'_>,
        shared_down: &Glm52ProjBytes<'_>,
    ) -> Result<Self> {
        ensure!(
            experts.len() == EXPERTS,
            "GLM5.2 MoE from_host expects {EXPERTS} routed experts, got {}",
            experts.len()
        );
        let mut w13_host: Vec<u8> = Vec::with_capacity(EXPERTS * W13_N * W13_K);
        let mut w13_scale_host: Vec<f32> =
            Vec::with_capacity(EXPERTS * W13_SCALE_ROWS * HIDDEN_SCALE_COLS);
        let mut w2_host: Vec<u8> = Vec::with_capacity(EXPERTS * W2_N * W2_K);
        let mut w2_scale_host: Vec<f32> =
            Vec::with_capacity(EXPERTS * W2_SCALE_ROWS * W2_SCALE_COLS);
        for (idx, expert) in experts.iter().enumerate() {
            let proj = |label: &str, p: &Glm52ProjBytes<'_>, n: usize, k: usize| -> Result<()> {
                ensure!(
                    p.n == n && p.k == k && p.weight.len() == n * k,
                    "GLM5.2 MoE expert {idx} {label} shape [{},{}] != [{n},{k}]",
                    p.n,
                    p.k
                );
                ensure!(
                    p.scale.len() == (n / QUANT_GROUP) * (k / QUANT_GROUP) * 4,
                    "GLM5.2 MoE expert {idx} {label} scale bytes {} unexpected",
                    p.scale.len()
                );
                Ok(())
            };
            proj("gate_proj", &expert.gate, INTERMEDIATE, HIDDEN)?;
            proj("up_proj", &expert.up, INTERMEDIATE, HIDDEN)?;
            proj("down_proj", &expert.down, HIDDEN, INTERMEDIATE)?;
            w13_host.extend_from_slice(expert.gate.weight);
            w13_host.extend_from_slice(expert.up.weight);
            w13_scale_host.extend_from_slice(&bytes_to_f32(expert.gate.scale));
            w13_scale_host.extend_from_slice(&bytes_to_f32(expert.up.scale));
            w2_host.extend_from_slice(expert.down.weight);
            w2_scale_host.extend_from_slice(&bytes_to_f32(expert.down.scale));
        }
        let stream = &ctx.stream;
        let upload_u8 = |host: &[u8]| -> Result<CudaSlice<u8>> {
            let mut dev = stream.alloc_zeros::<u8>(host.len())?;
            stream.memcpy_htod(host, &mut dev)?;
            Ok(dev)
        };
        let upload_f32 = |host: &[f32]| -> Result<CudaSlice<f32>> {
            let mut dev = stream.alloc_zeros::<f32>(host.len())?;
            stream.memcpy_htod(host, &mut dev)?;
            Ok(dev)
        };
        Self::new(
            upload_u8(gate_weight)?,
            upload_u8(e_score_bias)?,
            upload_u8(&w13_host)?,
            upload_f32(&w13_scale_host)?,
            upload_u8(&w2_host)?,
            upload_f32(&w2_scale_host)?,
            ProjWeight::upload(ctx, shared_gate)?,
            ProjWeight::upload(ctx, shared_up)?,
            ProjWeight::upload(ctx, shared_down)?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        gate_weight: CudaSlice<u8>,
        e_score_bias: CudaSlice<u8>,
        w13_weight: CudaSlice<u8>,
        w13_scale: CudaSlice<f32>,
        w2_weight: CudaSlice<u8>,
        w2_scale: CudaSlice<f32>,
        shared_gate: ProjWeight,
        shared_up: ProjWeight,
        shared_down: ProjWeight,
    ) -> Result<Self> {
        let check = |name: &str, have: usize, want: usize| -> Result<()> {
            ensure!(
                have == want,
                "GLM5.2 MoE layer weight {name} length {have} != expected {want}"
            );
            Ok(())
        };
        check("gate_weight", gate_weight.len(), EXPERTS * HIDDEN * 2)?;
        check("e_score_bias", e_score_bias.len(), EXPERTS * 4)?;
        check("w13_weight", w13_weight.len(), EXPERTS * W13_N * W13_K)?;
        check(
            "w13_scale",
            w13_scale.len(),
            EXPERTS * W13_SCALE_ROWS * HIDDEN_SCALE_COLS,
        )?;
        check("w2_weight", w2_weight.len(), EXPERTS * W2_N * W2_K)?;
        check(
            "w2_scale",
            w2_scale.len(),
            EXPERTS * W2_SCALE_ROWS * W2_SCALE_COLS,
        )?;
        let shape = |label: &str, p: &ProjWeight, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 MoE shared {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        shape("gate_proj", &shared_gate, INTERMEDIATE, HIDDEN)?;
        shape("up_proj", &shared_up, INTERMEDIATE, HIDDEN)?;
        shape("down_proj", &shared_down, HIDDEN, INTERMEDIATE)?;
        Ok(Self {
            gate_weight,
            e_score_bias,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            shared_gate,
            shared_up,
            shared_down,
        })
    }
}

/// Router output for one token: the top-8 expert ids and their normalized,
/// x2.5-scaled weights, both device-resident (never read back to host).
struct RoutedTopk {
    topk_idx: CudaSlice<i32>,
    topk_weight: CudaSlice<f32>,
}

fn run_router(
    ctx: &DeviceContext,
    weights: &Glm52MoeLayerWeights,
    normed_hidden: &CudaSlice<bf16>,
) -> Result<RoutedTopk> {
    let stream = &ctx.stream;
    let mut logits = stream.alloc_zeros::<f32>(EXPERTS)?;
    let mut topk_idx = stream.alloc_zeros::<i32>(TOPK)?;
    let mut topk_weight = stream.alloc_zeros::<f32>(TOPK)?;
    let mut router_out = Glm52RouterOutput {
        topk_weight: &mut topk_weight,
        topk_idx: &mut topk_idx,
    };
    glm52_router_noaux_tc_launch(
        ctx,
        Glm52RouterConfig::glm52(),
        Glm52RouterBatch {
            active_tokens: 1,
            padded_tokens: 1,
        },
        normed_hidden,
        &weights.gate_weight,
        &weights.e_score_bias,
        &mut logits,
        &mut router_out,
    )?;
    Ok(RoutedTopk {
        topk_idx,
        topk_weight,
    })
}

/// Routed contribution via the DeepEP-shaped grouped FP8 GEMM chain.
fn routed_forward_grouped(
    ctx: &DeviceContext,
    weights: &Glm52MoeLayerWeights,
    normed_hidden: &CudaSlice<bf16>,
    route: &RoutedTopk,
) -> Result<CudaSlice<bf16>> {
    let stream = &ctx.stream;

    // Quantize the single hidden row -> fp8 + per-group scale.
    let mut hidden_fp8 = stream.alloc_zeros::<u8>(HIDDEN)?;
    let mut hidden_scale = stream.alloc_zeros::<f32>(HIDDEN_SCALE_COLS)?;
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: 1,
            width: HIDDEN,
            group_size: QUANT_GROUP,
        },
        normed_hidden,
        &mut hidden_fp8,
        &mut hidden_scale,
    )?;

    // Device-side grouped expert_offsets from the top-k ids.
    let mut expert_offsets = stream.alloc_zeros::<i64>(EXPERTS + 1)?;
    glm52_moe_route_offsets_launch(
        ctx,
        EXPERTS,
        TOPK,
        GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT,
        &route.topk_idx,
        &mut expert_offsets,
    )?;

    // Scatter the fp8 row + per-row route weight into the expert-major slots.
    let mut w13_act = stream.alloc_zeros::<u8>(M_CAPACITY * W13_K)?;
    let mut w13_act_scale = stream.alloc_zeros::<f32>(M_CAPACITY * HIDDEN_SCALE_COLS)?;
    let mut row_weight = stream.alloc_zeros::<f32>(M_CAPACITY)?;
    glm52_moe_route_scatter_launch(
        ctx,
        M_CAPACITY,
        TOPK,
        W13_K,
        HIDDEN_SCALE_COLS,
        &hidden_fp8,
        &hidden_scale,
        &route.topk_idx,
        &route.topk_weight,
        &expert_offsets,
        &mut w13_act,
        &mut w13_act_scale,
        &mut row_weight,
    )?;

    // W13 grouped FP8 GEMM (gate|up).
    let w13_out = grouped_gemm(
        ctx,
        Glm52TrtllmGroupedFp8Kind::W13,
        W13_N,
        W13_K,
        HIDDEN_SCALE_COLS,
        W13_SCALE_ROWS,
        &w13_act,
        &w13_act_scale,
        &weights.w13_weight,
        &weights.w13_scale,
        &expert_offsets,
    )?;

    // Weighted SwiGLU quant: silu(gate)*up*route_weight -> fp8 W2 input.
    let mut w2_act = stream.alloc_zeros::<u8>(M_CAPACITY * W2_K)?;
    let mut w2_act_scale = stream.alloc_zeros::<f32>(M_CAPACITY * W2_SCALE_COLS)?;
    glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: M_CAPACITY,
            width: W2_K,
            group_size: QUANT_GROUP,
        },
        &w13_out,
        &row_weight,
        &mut w2_act,
        &mut w2_act_scale,
    )?;

    // W2 grouped FP8 GEMM (down).
    let w2_out = grouped_gemm(
        ctx,
        Glm52TrtllmGroupedFp8Kind::W2,
        W2_N,
        W2_K,
        W2_SCALE_COLS,
        W2_SCALE_ROWS,
        &w2_act,
        &w2_act_scale,
        &weights.w2_weight,
        &weights.w2_scale,
        &expert_offsets,
    )?;

    // Combine: sum the selected experts' rows -> routed[HIDDEN].
    let mut routed = stream.alloc_zeros::<bf16>(HIDDEN)?;
    glm52_moe_combine_launch(
        ctx,
        M_CAPACITY,
        HIDDEN,
        TOPK,
        &w2_out,
        &route.topk_idx,
        &expert_offsets,
        &mut routed,
    )?;
    Ok(routed)
}

/// Routed contribution via the bs=1 weight-only fp8 GEMV chain.
fn routed_forward_gemv(
    ctx: &DeviceContext,
    weights: &Glm52MoeLayerWeights,
    normed_hidden: &CudaSlice<bf16>,
    route: &RoutedTopk,
) -> Result<CudaSlice<bf16>> {
    let stream = &ctx.stream;

    // W13 grouped GEMV: broadcast the bf16 hidden across the top-k experts,
    // dequant the fp8 weight on the fly -> [TOPK, W13_N] bf16 (gate|up).
    let mut w13_out = stream.alloc_zeros::<bf16>(TOPK * W13_N)?;
    glm52_moe_fp8_weight_only_gemv_launch(
        ctx,
        GLM52_GEMV_KIND_W13,
        W13_N,
        W13_K,
        TOPK,
        0, // broadcast one activation row across every slot
        normed_hidden,
        &route.topk_idx,
        &weights.w13_weight,
        &weights.w13_scale,
        &mut w13_out,
    )?;

    // Weighted SiLU(gate)*up -> bf16 W2 input (route weight folded per slot).
    let mut w2_act = stream.alloc_zeros::<bf16>(TOPK * W2_K)?;
    glm52_silu_and_mul_weighted_bf16_launch(
        ctx,
        TOPK,
        W2_K,
        &w13_out,
        Some(&route.topk_weight),
        &mut w2_act,
    )?;

    // W2 grouped GEMV: per-slot down projection -> [TOPK, W2_N] bf16.
    let mut w2_out = stream.alloc_zeros::<bf16>(TOPK * W2_N)?;
    glm52_moe_fp8_weight_only_gemv_launch(
        ctx,
        GLM52_GEMV_KIND_W2,
        W2_N,
        W2_K,
        TOPK,
        W2_K, // per-slot activation row stride
        &w2_act,
        &route.topk_idx,
        &weights.w2_weight,
        &weights.w2_scale,
        &mut w2_out,
    )?;

    // Combine: sum the top-k slot rows (route weight already folded).
    let mut routed = stream.alloc_zeros::<bf16>(HIDDEN)?;
    glm52_moe_combine_slots_launch(ctx, HIDDEN, TOPK, &w2_out, &mut routed)?;
    Ok(routed)
}

/// Run the routed-MoE contribution for a single token. `normed_hidden` is the
/// post-attention-layernorm hidden `[HIDDEN]`; returns the routed output
/// `[HIDDEN]` (route weight + routed_scaling already folded in). The shared
/// expert is added by `glm52_moe_forward`.
pub(crate) fn glm52_moe_routed_forward(
    ctx: &DeviceContext,
    weights: &Glm52MoeLayerWeights,
    normed_hidden: &CudaSlice<bf16>,
    path: Glm52MoeExpertPath,
) -> Result<CudaSlice<bf16>> {
    ensure!(
        normed_hidden.len() >= HIDDEN,
        "GLM5.2 MoE routed forward hidden too small: have {}, need {HIDDEN}",
        normed_hidden.len()
    );
    let route = run_router(ctx, weights, normed_hidden)?;
    match path {
        Glm52MoeExpertPath::Grouped => routed_forward_grouped(ctx, weights, normed_hidden, &route),
        Glm52MoeExpertPath::Gemv => routed_forward_gemv(ctx, weights, normed_hidden, &route),
    }
}

/// Shared-expert contribution for a single token: a plain fp8 SwiGLU MLP
/// (intermediate 2048). Returns `[HIDDEN]` bf16.
pub(crate) fn glm52_moe_shared_forward(
    ctx: &DeviceContext,
    weights: &Glm52MoeLayerWeights,
    normed_hidden: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    fp8_mlp(
        ctx,
        &weights.shared_gate,
        &weights.shared_up,
        &weights.shared_down,
        normed_hidden,
    )
}

/// Full MoE contribution for a single token: routed experts + shared expert. The
/// caller adds this to the post-attention residual. Returns `[HIDDEN]` bf16.
pub(crate) fn glm52_moe_forward(
    ctx: &DeviceContext,
    weights: &Glm52MoeLayerWeights,
    normed_hidden: &CudaSlice<bf16>,
    path: Glm52MoeExpertPath,
) -> Result<CudaSlice<bf16>> {
    let routed = glm52_moe_routed_forward(ctx, weights, normed_hidden, path)?;
    let shared = glm52_moe_shared_forward(ctx, weights, normed_hidden)?;
    let routed_hs = HiddenStates {
        data: routed,
        hidden_dim: HIDDEN,
        seq_len: 1,
    };
    let shared_hs = HiddenStates {
        data: shared,
        hidden_dim: HIDDEN,
        seq_len: 1,
    };
    Ok(add_batch(ctx, &routed_hs, &shared_hs)?.data)
}

/// Relayout the plain per-row activation scale into the offset-major TMA layout,
/// then run one grouped FP8 GEMM. Returns the bf16 output `[M_CAPACITY, n]`.
#[allow(clippy::too_many_arguments)]
fn grouped_gemm(
    ctx: &DeviceContext,
    kind: Glm52TrtllmGroupedFp8Kind,
    n: usize,
    k: usize,
    scale_cols: usize,
    weight_scale_rows: usize,
    activation: &CudaSlice<u8>,
    activation_scale_plain: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    expert_offsets: &CudaSlice<i64>,
) -> Result<CudaSlice<bf16>> {
    let stream = &ctx.stream;
    let scale_layout = Glm52TrtllmGroupedOffsetScaleLayout::f32(M_CAPACITY, scale_cols, EXPERTS);
    let mut activation_scale_tma = stream.alloc_zeros::<f32>(scale_layout.output_len()?)?;
    glm52_deepgemm_grouped_offset_tma_aligned_f32_launch(
        ctx,
        scale_layout,
        activation_scale_plain,
        expert_offsets,
        &mut activation_scale_tma,
    )?;

    let contract = Glm52TrtllmGroupedFp8Contract {
        groups: EXPERTS,
        m_capacity: M_CAPACITY,
        n,
        k,
        weight_scale_rows,
        weight_scale_cols: scale_cols,
        activation_scale_cols: scale_cols,
        activation_scale_trtllm_rows: scale_layout.padded_rows,
    };
    let mut out = stream.alloc_zeros::<bf16>(M_CAPACITY * n)?;
    glm52_trtllm_grouped_fp8_launch(
        ctx,
        kind,
        contract,
        activation,
        &activation_scale_tma,
        weight,
        weight_scale,
        expert_offsets,
        &mut out,
    )?;
    Ok(out)
}
