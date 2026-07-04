//! GLM5.2 dense-MLP decode forward for bs=1 (layers 0..first_k_dense_replace).
//!
//! The dense layers replace the MoE block with a plain fp8 SwiGLU MLP
//! `down(silu(gate(x)) * up(x))` -- the same shape as the MoE shared expert, only
//! the intermediate is wider (12288 vs 2048). It reuses the shared `fp8_mlp`
//! helper, so this module is just the weight bundle + a thin forward.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::tensor::DeviceContext;

#[cfg(test)]
use crate::fp8::Glm52ProjBytes;
use crate::fp8::{Glm52MlpScratch, ProjWeight, fp8_mlp_into, pack_proj_pair};

const HIDDEN: usize = 6144;
const INTERMEDIATE: usize = 12288;

/// The dense-layer intermediate width (sizes the shared decode mlp scratch).
pub(crate) const GLM52_DENSE_INTERMEDIATE: usize = INTERMEDIATE;

/// The fp8 projections of one dense MLP layer, resident on device. gate|up
/// are packed into one `[2*INTERMEDIATE, HIDDEN]` projection at build (one
/// GEMV + SwiGLU + one GEMV at decode).
pub(crate) struct Glm52DenseMlpWeights {
    gate_up: ProjWeight, // fp8 [2*INTERMEDIATE, HIDDEN] (gate | up)
    down: ProjWeight,    // fp8 [HIDDEN, INTERMEDIATE]
}

impl Glm52DenseMlpWeights {
    /// Upload the dense MLP projections, validating every extent against the
    /// GLM5.2 dense-layer architecture (crash-early on a packaging drift).
    #[cfg(test)]
    pub(crate) fn from_host(
        ctx: &DeviceContext,
        gate: &Glm52ProjBytes<'_>,
        up: &Glm52ProjBytes<'_>,
        down: &Glm52ProjBytes<'_>,
    ) -> Result<Self> {
        let shape = |label: &str, p: &Glm52ProjBytes<'_>, n: usize, k: usize| -> Result<()> {
            anyhow::ensure!(
                p.n == n && p.k == k,
                "GLM5.2 dense MLP {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        shape("gate_proj", gate, INTERMEDIATE, HIDDEN)?;
        shape("up_proj", up, INTERMEDIATE, HIDDEN)?;
        shape("down_proj", down, HIDDEN, INTERMEDIATE)?;
        Ok(Self {
            gate_up: pack_proj_pair(
                ctx,
                &ProjWeight::upload(ctx, gate)?,
                &ProjWeight::upload(ctx, up)?,
            )?,
            down: ProjWeight::upload(ctx, down)?,
        })
    }

    /// Build from already-resident projections (the production loader path).
    pub(crate) fn from_device(
        ctx: &DeviceContext,
        gate: ProjWeight,
        up: ProjWeight,
        down: ProjWeight,
    ) -> Result<Self> {
        let shape = |label: &str, p: &ProjWeight, n: usize, k: usize| -> Result<()> {
            anyhow::ensure!(
                p.n == n && p.k == k,
                "GLM5.2 dense MLP {label} shape [{},{}] != [{n},{k}]",
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
}

/// Dense MLP contribution for a single token into a pre-allocated output.
/// `normed_hidden` is the post-attention-layernorm hidden `[HIDDEN]`; `out`
/// gets the MLP output `[HIDDEN]` (the caller adds it to the post-attention
/// residual). `mlp` must be sized for the dense intermediate (12288).
pub(crate) fn glm52_dense_mlp_forward_into(
    ctx: &DeviceContext,
    weights: &Glm52DenseMlpWeights,
    normed_hidden: &CudaSlice<bf16>,
    mlp: &mut Glm52MlpScratch,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    fp8_mlp_into(
        ctx,
        &weights.gate_up,
        &weights.down,
        normed_hidden,
        mlp,
        out,
    )
}
