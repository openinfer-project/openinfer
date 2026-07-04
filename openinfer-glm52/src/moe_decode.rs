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
#[cfg(test)]
use openinfer_kernels::ops::add_batch;
#[cfg(test)]
use openinfer_kernels::ops::{
    GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT, GLM52_GEMV_KIND_W2, GLM52_GEMV_KIND_W13,
    Glm52MoeQuantShape, glm52_fp8_per_token_group_quant_bf16_launch, glm52_moe_combine_launch,
    glm52_moe_combine_slots_launch, glm52_moe_fp8_weight_only_gemv_launch,
    glm52_moe_route_offsets_launch, glm52_moe_route_scatter_launch,
    glm52_silu_and_mul_weighted_bf16_launch,
    glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch,
};
use openinfer_kernels::ops::{
    Glm52RouterBatch, Glm52RouterConfig, Glm52RouterOutput, Glm52TrtllmGroupedFp8Contract,
    Glm52TrtllmGroupedFp8Kind, Glm52TrtllmGroupedOffsetScaleLayout,
    glm52_deepgemm_grouped_offset_tma_aligned_f32_launch, glm52_router_noaux_tc_launch,
    glm52_trtllm_grouped_fp8_launch,
};
use openinfer_kernels::tensor::DeviceContext;
#[cfg(test)]
use openinfer_kernels::tensor::HiddenStates;

#[cfg(test)]
use crate::fp8::fp8_mlp;
use crate::fp8::{Glm52MlpScratch, ProjWeight, fp8_mlp_into, pack_proj_pair};
#[cfg(test)]
use crate::fp8::{Glm52ProjBytes, bytes_to_f32};

pub(crate) const HIDDEN: usize = 6144;
pub(crate) const EXPERTS: usize = 256;
pub(crate) const TOPK: usize = 8;
const INTERMEDIATE: usize = 2048;
pub(crate) const QUANT_GROUP: usize = 128;

pub(crate) const W13_N: usize = 2 * INTERMEDIATE; // 4096 (gate|up)
pub(crate) const W13_K: usize = HIDDEN; // 6144
pub(crate) const W2_N: usize = HIDDEN; // 6144
pub(crate) const W2_K: usize = INTERMEDIATE; // 2048

pub(crate) const HIDDEN_SCALE_COLS: usize = HIDDEN / QUANT_GROUP; // 48
pub(crate) const W13_SCALE_ROWS: usize = W13_N / QUANT_GROUP; // 32
pub(crate) const W2_SCALE_COLS: usize = W2_K / QUANT_GROUP; // 16
pub(crate) const W2_SCALE_ROWS: usize = W2_N / QUANT_GROUP; // 48

/// bs=1 expert-major row capacity for the grouped path: each of the top-k distinct
/// experts owns at most one row, padded to the 64-row alignment, so
/// `TOPK * ALIGNMENT` is a tight upper bound (`route_offsets` emits
/// `expert_offsets[E] <=` it).
#[cfg(test)]
const M_CAPACITY: usize = TOPK * GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT; // 512

/// Which expert-GEMM implementation runs the routed contribution.
#[cfg(test)]
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
#[cfg(test)]
pub(crate) struct Glm52MoeRoutedExpertBytes<'a> {
    pub(crate) gate: Glm52ProjBytes<'a>, // [INTERMEDIATE, HIDDEN]
    pub(crate) up: Glm52ProjBytes<'a>,   // [INTERMEDIATE, HIDDEN]
    pub(crate) down: Glm52ProjBytes<'a>, // [HIDDEN, INTERMEDIATE]
}

/// Router weights: the bf16 gate GEMM and the f32 selection-bias.
pub(crate) struct Glm52MoeRouterWeights {
    gate_weight: CudaSlice<u8>,  // bf16 [EXPERTS, HIDDEN]
    e_score_bias: CudaSlice<u8>, // f32  [EXPERTS]
}

impl Glm52MoeRouterWeights {
    pub(crate) fn new(gate_weight: CudaSlice<u8>, e_score_bias: CudaSlice<u8>) -> Result<Self> {
        ensure!(
            gate_weight.len() == EXPERTS * HIDDEN * 2 && e_score_bias.len() == EXPERTS * 4,
            "GLM5.2 MoE router weight bytes unexpected: gate {}, bias {}",
            gate_weight.len(),
            e_score_bias.len()
        );
        Ok(Self {
            gate_weight,
            e_score_bias,
        })
    }
}

/// The single shared expert (a plain fp8 SwiGLU MLP at intermediate 2048).
/// gate|up are packed into one `[2*INTERMEDIATE, HIDDEN]` projection at build
/// so the decode path is one GEMV + SwiGLU + one GEMV.
pub(crate) struct Glm52MoeSharedExpert {
    gate_up: ProjWeight, // fp8 [2*INTERMEDIATE, HIDDEN] (gate | up)
    down: ProjWeight,    // fp8 [HIDDEN, INTERMEDIATE]
}

impl Glm52MoeSharedExpert {
    pub(crate) fn new(
        ctx: &DeviceContext,
        gate: ProjWeight,
        up: ProjWeight,
        down: ProjWeight,
    ) -> Result<Self> {
        let shape = |label: &str, p: &ProjWeight, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 MoE shared {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        shape("gate_proj", &gate, INTERMEDIATE, HIDDEN)?;
        shape("up_proj", &up, INTERMEDIATE, HIDDEN)?;
        shape("down_proj", &down, HIDDEN, INTERMEDIATE)?;
        Ok(Self {
            gate_up: pack_proj_pair(ctx, &gate, &up)?,
            down,
        })
    }

    /// Shared-expert contribution for a single token: a plain fp8 SwiGLU MLP.
    #[cfg(test)]
    pub(crate) fn forward(
        &self,
        ctx: &DeviceContext,
        normed_hidden: &CudaSlice<bf16>,
    ) -> Result<CudaSlice<bf16>> {
        fp8_mlp(ctx, &self.gate_up, &self.down, normed_hidden)
    }

    /// [`Self::forward`] into a pre-allocated output through persistent
    /// scratch (the decode path). `mlp` must be sized for the shared-expert
    /// intermediate (2048).
    pub(crate) fn forward_into(
        &self,
        ctx: &DeviceContext,
        normed_hidden: &CudaSlice<bf16>,
        mlp: &mut Glm52MlpScratch,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        fp8_mlp_into(ctx, &self.gate_up, &self.down, normed_hidden, mlp, out)
    }
}

/// The shared-expert intermediate width (sizes the decode mlp scratch).
pub(crate) const GLM52_SHARED_EXPERT_INTERMEDIATE: usize = INTERMEDIATE;

/// A bank of routed experts in the expert-major packed layout: fp8 weights
/// `[n_experts, n, k]` + f32 block scales `[n_experts, n/128, k/128]`, per
/// expert W13 = `[gate; up]`. EP1 holds all 256; an EP8 rank holds its 32
/// local experts (indexed by LOCAL id — the dispatch delivers local segments).
pub(crate) struct Glm52MoeExpertBank {
    n_experts: usize,
    pub(crate) w13_weight: CudaSlice<u8>, // fp8  [n_experts, W13_N, W13_K]
    pub(crate) w13_scale: CudaSlice<f32>, // f32  [n_experts, W13_SCALE_ROWS, HIDDEN_SCALE_COLS]
    pub(crate) w2_weight: CudaSlice<u8>,  // fp8  [n_experts, W2_N, W2_K]
    pub(crate) w2_scale: CudaSlice<f32>,  // f32  [n_experts, W2_SCALE_ROWS, W2_SCALE_COLS]
}

impl Glm52MoeExpertBank {
    pub(crate) fn new(
        n_experts: usize,
        w13_weight: CudaSlice<u8>,
        w13_scale: CudaSlice<f32>,
        w2_weight: CudaSlice<u8>,
        w2_scale: CudaSlice<f32>,
    ) -> Result<Self> {
        let check = |name: &str, have: usize, want: usize| -> Result<()> {
            ensure!(
                have == want,
                "GLM5.2 MoE expert bank {name} length {have} != expected {want}"
            );
            Ok(())
        };
        check("w13_weight", w13_weight.len(), n_experts * W13_N * W13_K)?;
        check(
            "w13_scale",
            w13_scale.len(),
            n_experts * W13_SCALE_ROWS * HIDDEN_SCALE_COLS,
        )?;
        check("w2_weight", w2_weight.len(), n_experts * W2_N * W2_K)?;
        check(
            "w2_scale",
            w2_scale.len(),
            n_experts * W2_SCALE_ROWS * W2_SCALE_COLS,
        )?;
        Ok(Self {
            n_experts,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
        })
    }

    /// Pack per-expert checkpoint bytes expert-major and upload (test path;
    /// works for any expert count — 256 at EP1, a 32-expert rank slice for
    /// the EP8 gate).
    #[cfg(test)]
    pub(crate) fn pack_from_host(
        ctx: &DeviceContext,
        experts: &[Glm52MoeRoutedExpertBytes<'_>],
    ) -> Result<Self> {
        let n_experts = experts.len();
        let mut w13_host: Vec<u8> = Vec::with_capacity(n_experts * W13_N * W13_K);
        let mut w13_scale_host: Vec<f32> =
            Vec::with_capacity(n_experts * W13_SCALE_ROWS * HIDDEN_SCALE_COLS);
        let mut w2_host: Vec<u8> = Vec::with_capacity(n_experts * W2_N * W2_K);
        let mut w2_scale_host: Vec<f32> =
            Vec::with_capacity(n_experts * W2_SCALE_ROWS * W2_SCALE_COLS);
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
            n_experts,
            upload_u8(&w13_host)?,
            upload_f32(&w13_scale_host)?,
            upload_u8(&w2_host)?,
            upload_f32(&w2_scale_host)?,
        )
    }

    /// Adopt one layer's loader-packed regions (this rank's 32 local experts)
    /// — a pure retype, no copies. The loader wrote the regions in exactly
    /// the `from_host` packing (proven by `expert_placement_matches_from_host_packing`).
    pub(crate) fn from_regions(
        ctx: &DeviceContext,
        regions: crate::weights::Glm52ExpertLayerRegions,
    ) -> Result<Self> {
        Self::new(
            crate::weights::GLM52_LOCAL_EXPERTS,
            regions.w13_weight,
            crate::weights::retype_owned::<f32>(&ctx.stream, regions.w13_scale)?,
            regions.w2_weight,
            crate::weights::retype_owned::<f32>(&ctx.stream, regions.w2_scale)?,
        )
    }

    pub(crate) fn n_experts(&self) -> usize {
        self.n_experts
    }
}

/// All weights for one MoE layer at EP1 (all 256 routed experts local), plus
/// the router and the single shared expert. Built once on device; borrowed by
/// every decode step.
#[cfg(test)]
pub(crate) struct Glm52MoeLayerWeights {
    pub(crate) router: Glm52MoeRouterWeights,
    pub(crate) bank: Glm52MoeExpertBank,
    pub(crate) shared: Glm52MoeSharedExpert,
}

#[cfg(test)]
impl Glm52MoeLayerWeights {
    /// Pack per-expert checkpoint tensors into the expert-major grouped
    /// buffers and upload everything (the oracle/test path). W13 = per-expert
    /// `[gate; up]` rows with scales concatenated likewise; W2 = down. The
    /// production path adopts loader-packed regions with the same layout
    /// ([`Glm52MoeExpertBank::from_regions`]).
    #[cfg(test)]
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
        let stream = &ctx.stream;
        let upload_u8 = |host: &[u8]| -> Result<CudaSlice<u8>> {
            let mut dev = stream.alloc_zeros::<u8>(host.len())?;
            stream.memcpy_htod(host, &mut dev)?;
            Ok(dev)
        };
        Ok(Self {
            router: Glm52MoeRouterWeights::new(upload_u8(gate_weight)?, upload_u8(e_score_bias)?)?,
            bank: Glm52MoeExpertBank::pack_from_host(ctx, experts)?,
            shared: Glm52MoeSharedExpert::new(
                ctx,
                ProjWeight::upload(ctx, shared_gate)?,
                ProjWeight::upload(ctx, shared_up)?,
                ProjWeight::upload(ctx, shared_down)?,
            )?,
        })
    }
}

/// Router output for one token: the top-8 GLOBAL expert ids and their
/// normalized, x2.5-scaled weights, both device-resident (never read back to
/// host).
pub(crate) struct RoutedTopk {
    pub(crate) topk_idx: CudaSlice<i32>,
    pub(crate) topk_weight: CudaSlice<f32>,
}

/// Persistent router scratch for the decode path: the expert logits plus the
/// top-k output the MoE dispatch consumes, written in place every MoE layer.
/// Sized for `tokens` rows.
pub(crate) struct Glm52RouterScratch {
    tokens: usize,
    logits: CudaSlice<f32>,
    pub(crate) route: RoutedTopk,
}

impl Glm52RouterScratch {
    pub(crate) fn new(ctx: &DeviceContext, tokens: usize) -> Result<Self> {
        ensure!(tokens > 0, "GLM5.2 router scratch needs positive tokens");
        Ok(Self {
            tokens,
            logits: ctx.stream.alloc_zeros::<f32>(tokens * EXPERTS)?,
            route: RoutedTopk {
                topk_idx: ctx.stream.alloc_zeros::<i32>(tokens * TOPK)?,
                topk_weight: ctx.stream.alloc_zeros::<f32>(tokens * TOPK)?,
            },
        })
    }
}

/// Router over the scratch's `tokens` rows into the persistent scratch
/// (`s.route` holds the per-row top-k, `[T, 8]`).
pub(crate) fn run_router_into(
    ctx: &DeviceContext,
    router: &Glm52MoeRouterWeights,
    normed_hidden: &CudaSlice<bf16>,
    s: &mut Glm52RouterScratch,
) -> Result<()> {
    let mut router_out = Glm52RouterOutput {
        topk_weight: &mut s.route.topk_weight,
        topk_idx: &mut s.route.topk_idx,
    };
    glm52_router_noaux_tc_launch(
        ctx,
        Glm52RouterConfig::glm52(),
        Glm52RouterBatch {
            active_tokens: s.tokens,
            padded_tokens: s.tokens,
        },
        normed_hidden,
        &router.gate_weight,
        &router.e_score_bias,
        &mut s.logits,
        &mut router_out,
    )?;
    Ok(())
}

/// Allocating convenience over [`run_router_into`] for the oracle-gate/test
/// paths.
#[cfg(test)]
pub(crate) fn run_router(
    ctx: &DeviceContext,
    router: &Glm52MoeRouterWeights,
    normed_hidden: &CudaSlice<bf16>,
) -> Result<RoutedTopk> {
    let mut s = Glm52RouterScratch::new(ctx, 1)?;
    run_router_into(ctx, router, normed_hidden, &mut s)?;
    Ok(s.route)
}

/// Routed contribution via the DeepEP-shaped grouped FP8 GEMM chain.
#[cfg(test)]
fn routed_forward_grouped(
    ctx: &DeviceContext,
    bank: &Glm52MoeExpertBank,
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
        EXPERTS,
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
        EXPERTS,
        M_CAPACITY,
        W13_N,
        W13_K,
        HIDDEN_SCALE_COLS,
        W13_SCALE_ROWS,
        &w13_act,
        &w13_act_scale,
        &bank.w13_weight,
        &bank.w13_scale,
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
        EXPERTS,
        M_CAPACITY,
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

    // Combine: sum the selected experts' rows -> routed[HIDDEN].
    let mut routed = stream.alloc_zeros::<bf16>(HIDDEN)?;
    glm52_moe_combine_launch(
        ctx,
        M_CAPACITY,
        EXPERTS,
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
#[cfg(test)]
fn routed_forward_gemv(
    ctx: &DeviceContext,
    bank: &Glm52MoeExpertBank,
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
        &bank.w13_weight,
        &bank.w13_scale,
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
        &bank.w2_weight,
        &bank.w2_scale,
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
#[cfg(test)]
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
    ensure!(
        weights.bank.n_experts == EXPERTS,
        "GLM5.2 EP1 MoE forward needs the full {EXPERTS}-expert bank (topk ids are global), got {}",
        weights.bank.n_experts
    );
    let route = run_router(ctx, &weights.router, normed_hidden)?;
    match path {
        Glm52MoeExpertPath::Grouped => {
            routed_forward_grouped(ctx, &weights.bank, normed_hidden, &route)
        }
        Glm52MoeExpertPath::Gemv => routed_forward_gemv(ctx, &weights.bank, normed_hidden, &route),
    }
}

/// Shared-expert contribution for a single token: a plain fp8 SwiGLU MLP
/// (intermediate 2048). Returns `[HIDDEN]` bf16.
#[cfg(test)]
pub(crate) fn glm52_moe_shared_forward(
    ctx: &DeviceContext,
    weights: &Glm52MoeLayerWeights,
    normed_hidden: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    weights.shared.forward(ctx, normed_hidden)
}

/// Full MoE contribution for a single token: routed experts + shared expert. The
/// caller adds this to the post-attention residual. Returns `[HIDDEN]` bf16.
#[cfg(test)]
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
/// then run one grouped FP8 GEMM. Returns the bf16 output `[m_capacity, n]`.
/// `groups`/`m_capacity` are runtime: 256/512 at EP1 (bs=1), 32/row-bound at
/// EP8 (moe_ep8 drives this over the DeepEP recv layout).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn grouped_gemm(
    ctx: &DeviceContext,
    kind: Glm52TrtllmGroupedFp8Kind,
    groups: usize,
    m_capacity: usize,
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
    let scale_layout = Glm52TrtllmGroupedOffsetScaleLayout::f32(m_capacity, scale_cols, groups);
    let mut activation_scale_tma = ctx.stream.alloc_zeros::<f32>(scale_layout.output_len()?)?;
    let mut out = ctx.stream.alloc_zeros::<bf16>(m_capacity * n)?;
    grouped_gemm_into(
        ctx,
        kind,
        groups,
        m_capacity,
        n,
        k,
        scale_cols,
        weight_scale_rows,
        activation,
        activation_scale_plain,
        weight,
        weight_scale,
        expert_offsets,
        &mut activation_scale_tma,
        &mut out,
    )?;
    Ok(out)
}

/// `grouped_gemm` writing into caller-owned buffers (EP8's persistent
/// workspace): `scale_tma` and `out` must hold at least the layout /
/// `m_capacity * n` for the largest `m_capacity` ever passed — launches may
/// cover fewer rows than the buffers, never more.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grouped_gemm_into(
    ctx: &DeviceContext,
    kind: Glm52TrtllmGroupedFp8Kind,
    groups: usize,
    m_capacity: usize,
    n: usize,
    k: usize,
    scale_cols: usize,
    weight_scale_rows: usize,
    activation: &CudaSlice<u8>,
    activation_scale_plain: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    expert_offsets: &CudaSlice<i64>,
    scale_tma: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let scale_layout = Glm52TrtllmGroupedOffsetScaleLayout::f32(m_capacity, scale_cols, groups);
    let scale_tma_len = scale_layout.output_len()?;
    ensure!(
        scale_tma.len() >= scale_tma_len && out.len() >= m_capacity * n,
        "GLM5.2 grouped GEMM workspace too small: scale_tma {} < {scale_tma_len} or out {} < {}",
        scale_tma.len(),
        out.len(),
        m_capacity * n
    );
    glm52_deepgemm_grouped_offset_tma_aligned_f32_launch(
        ctx,
        scale_layout,
        activation_scale_plain,
        expert_offsets,
        scale_tma,
    )?;

    let contract = Glm52TrtllmGroupedFp8Contract {
        groups,
        m_capacity,
        n,
        k,
        weight_scale_rows,
        weight_scale_cols: scale_cols,
        activation_scale_cols: scale_cols,
        activation_scale_trtllm_rows: scale_layout.padded_rows,
    };
    glm52_trtllm_grouped_fp8_launch(
        ctx,
        kind,
        contract,
        activation,
        scale_tma,
        weight,
        weight_scale,
        expert_offsets,
        out,
    )?;
    Ok(())
}
