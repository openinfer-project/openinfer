//! Shared GLM5.2 MoE weights and router used by the EP8 and TP8 production
//! paths.

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::{
    Glm52RouterBatch, Glm52RouterConfig, Glm52RouterOutput, glm52_router_noaux_tc_launch,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::fp8::{Glm52MlpScratch, ProjWeight, fp8_mlp_into, pack_proj_pair};
#[cfg(test)]
use crate::fp8::{Glm52ProjBytes, bytes_to_f32};

pub(crate) const HIDDEN: usize = crate::config::GLM52_HIDDEN;
pub(crate) const EXPERTS: usize = crate::config::GLM52_ROUTED_EXPERTS;
pub(crate) const TOPK: usize = crate::config::GLM52_TOPK;
const INTERMEDIATE: usize = crate::config::GLM52_EXPERT_INTERMEDIATE;
pub(crate) const QUANT_GROUP: usize = 128;

pub(crate) const W13_N: usize = 2 * INTERMEDIATE; // 4096 (gate|up)
pub(crate) const W13_K: usize = HIDDEN; // 6144
pub(crate) const W2_N: usize = HIDDEN; // 6144
pub(crate) const W2_K: usize = INTERMEDIATE; // 2048

pub(crate) const HIDDEN_SCALE_COLS: usize = HIDDEN / QUANT_GROUP; // 48
pub(crate) const W13_SCALE_ROWS: usize = W13_N / QUANT_GROUP; // 32
pub(crate) const W2_SCALE_COLS: usize = W2_K / QUANT_GROUP; // 16
pub(crate) const W2_SCALE_ROWS: usize = W2_N / QUANT_GROUP; // 48

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
        gate: &ProjWeight,
        up: &ProjWeight,
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
        shape("gate_proj", gate, INTERMEDIATE, HIDDEN)?;
        shape("up_proj", up, INTERMEDIATE, HIDDEN)?;
        shape("down_proj", &down, HIDDEN, INTERMEDIATE)?;
        Ok(Self {
            gate_up: pack_proj_pair(ctx, gate, up)?,
            down,
        })
    }

    /// Shared-expert contribution into pre-allocated decode scratch.
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

/// One EP8 rank's routed experts in expert-major packed layout.
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

    /// Pack one rank's checkpoint experts for the EP8 oracle gate.
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
