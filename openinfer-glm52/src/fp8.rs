//! Shared GLM5.2 fp8 block-scaled projection primitives (bs=1 decode).
//!
//! Every dense/attention/expert projection in GLM5.2 is fp8 e4m3 with a per-128
//! block `weight_scale_inv`. The decode-time recipe is the same everywhere: quant
//! the bf16 activation per 128-group, relay that activation scale into the TRTLLM
//! col-major TMA layout (the documented footgun — every projection must do it),
//! then run the blockscale linear. MLA, the dense MLP, and the MoE shared expert
//! all share these helpers.

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;

use openinfer_kernels::ops::{
    Glm52DeepGemmScaleLayout, Glm52MoeQuantShape, Glm52TrtllmFp8LinearContract,
    glm52_deepgemm_mn_major_tma_aligned_f32_launch, glm52_fp8_per_token_group_quant_bf16_launch,
    glm52_silu_and_mul_per_token_group_quant_bf16_launch, glm52_trtllm_fp8_linear_launch,
};
use openinfer_kernels::tensor::DeviceContext;

pub(crate) const FP8_BLOCK: usize = 128;

/// OCP `float8_e4m3fn` decode (bias 7, no inf; subnormals supported). Used by the
/// host-side dequant paths (kv_b absorb factors), not the GPU kernels.
pub(crate) fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if (b >> 7) & 1 == 1 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    let mag = if e == 0 {
        2f32.powi(-6) * (m / 8.0)
    } else {
        2f32.powi(e - 7) * (1.0 + m / 8.0)
    };
    sign * mag
}

pub(crate) fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Raw fp8 block-scaled projection bytes (row-major weight `[n,k]` + per-128-block
/// `weight_scale_inv` `[n/128, k/128]` f32).
pub(crate) struct Glm52ProjBytes<'a> {
    pub(crate) weight: &'a [u8],
    pub(crate) scale: &'a [u8],
    pub(crate) n: usize,
    pub(crate) k: usize,
}

/// One fp8 projection resident on device.
pub(crate) struct ProjWeight {
    pub(crate) weight: CudaSlice<u8>,
    pub(crate) scale: CudaSlice<u8>,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl ProjWeight {
    pub(crate) fn upload(ctx: &DeviceContext, b: &Glm52ProjBytes) -> Result<Self> {
        ensure!(
            b.weight.len() == b.n * b.k,
            "GLM5.2 proj weight bytes {} != n*k {}",
            b.weight.len(),
            b.n * b.k
        );
        ensure!(
            b.scale.len() == b.n.div_ceil(FP8_BLOCK) * b.k.div_ceil(FP8_BLOCK) * 4,
            "GLM5.2 proj scale bytes {} unexpected for [{},{}]",
            b.scale.len(),
            b.n,
            b.k
        );
        let mut weight = ctx.stream.alloc_zeros::<u8>(b.weight.len())?;
        let mut scale = ctx.stream.alloc_zeros::<u8>(b.scale.len())?;
        ctx.stream.memcpy_htod(b.weight, &mut weight)?;
        ctx.stream.memcpy_htod(b.scale, &mut scale)?;
        Ok(Self {
            weight,
            scale,
            n: b.n,
            k: b.k,
        })
    }

    /// Wrap already-resident GPU buffers (the production loader path), moving them
    /// in with no copy. `weight` is the fp8 `[n,k]` e4m3 bytes, `scale` the f32
    /// `weight_scale_inv` (`[n/128, k/128]`) kept as raw `u8`. Same validation as
    /// `upload`, so a packaging drift crashes here, not in the kernel.
    pub(crate) fn from_device(
        weight: CudaSlice<u8>,
        scale: CudaSlice<u8>,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        ensure!(
            weight.len() == n * k,
            "GLM5.2 proj weight bytes {} != n*k {}",
            weight.len(),
            n * k
        );
        ensure!(
            scale.len() == n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK) * 4,
            "GLM5.2 proj scale bytes {} unexpected for [{n},{k}]",
            scale.len()
        );
        Ok(Self {
            weight,
            scale,
            n,
            k,
        })
    }
}

/// Persistent per-projection scratch (bs=1): the fp8 activation, its plain
/// per-group scale, and the TRTLLM col-major relaid scale. One instance is
/// shared by EVERY `fp8_linear_into` call site in a decode step — safe because
/// each consumer reads exactly the prefix its own producer wrote in the same
/// call, and the relayout pad rows (never written by any call) keep their
/// zero initialization.
pub(crate) struct Glm52ProjScratch {
    a_fp8: CudaSlice<u8>,
    a_scale_plain: CudaSlice<f32>,
    a_scale: CudaSlice<f32>,
    max_k: usize,
}

impl Glm52ProjScratch {
    pub(crate) fn new(ctx: &DeviceContext, max_k: usize) -> Result<Self> {
        ensure!(
            max_k.is_multiple_of(FP8_BLOCK),
            "GLM5.2 proj scratch max_k {max_k} not a multiple of {FP8_BLOCK}"
        );
        let scale_cols = max_k / FP8_BLOCK;
        let layout = Glm52DeepGemmScaleLayout::f32(1, scale_cols);
        Ok(Self {
            a_fp8: ctx.stream.alloc_zeros::<u8>(max_k)?,
            a_scale_plain: ctx.stream.alloc_zeros::<f32>(scale_cols)?,
            a_scale: ctx.stream.alloc_zeros::<f32>(layout.output_len()?)?,
            max_k,
        })
    }
}

/// One fp8 projection (bs=1) into a pre-allocated output: quant the bf16
/// activation into the scratch, then the prequant linear.
pub(crate) fn fp8_linear_into(
    ctx: &DeviceContext,
    w: &ProjWeight,
    input: &CudaSlice<bf16>,
    s: &mut Glm52ProjScratch,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        input.len() >= w.k,
        "GLM5.2 fp8_linear input {} < k {}",
        input.len(),
        w.k
    );
    ensure!(
        w.k <= s.max_k,
        "GLM5.2 proj scratch sized for k <= {} but projection has k {}",
        s.max_k,
        w.k
    );
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: 1,
            width: w.k,
            group_size: FP8_BLOCK,
        },
        input,
        &mut s.a_fp8,
        &mut s.a_scale_plain,
    )?;
    fp8_linear_prequant_into(ctx, w, &s.a_fp8, &s.a_scale_plain, &mut s.a_scale, out)
}

/// One fp8 projection (bs=1) from a pre-quantized activation: relay the plain
/// per-group activation scale into the TRTLLM col-major TMA layout (via the
/// `a_scale` scratch), then the blockscale linear. Used when the input is
/// already fp8 (e.g. the SwiGLU output feeding a down projection).
pub(crate) fn fp8_linear_prequant_into(
    ctx: &DeviceContext,
    w: &ProjWeight,
    a_fp8: &CudaSlice<u8>,
    a_scale_plain: &CudaSlice<f32>,
    a_scale: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let scale_cols = w.k / FP8_BLOCK;
    ensure!(
        a_fp8.len() >= w.k && a_scale_plain.len() >= scale_cols,
        "GLM5.2 fp8_linear_prequant input too small: fp8 {} (need {}), scale {} (need {scale_cols})",
        a_fp8.len(),
        w.k,
        a_scale_plain.len()
    );
    ensure!(
        out.len() >= w.n,
        "GLM5.2 fp8_linear output {} < n {}",
        out.len(),
        w.n
    );
    let layout = Glm52DeepGemmScaleLayout::f32(1, scale_cols);
    ensure!(
        a_scale.len() >= layout.output_len()?,
        "GLM5.2 fp8_linear a_scale scratch {} < relaid layout {}",
        a_scale.len(),
        layout.output_len()?
    );
    glm52_deepgemm_mn_major_tma_aligned_f32_launch(ctx, layout, a_scale_plain, a_scale)?;
    glm52_trtllm_fp8_linear_launch(
        ctx,
        Glm52TrtllmFp8LinearContract {
            m: 1,
            n: w.n,
            k: w.k,
            weight_scale_rows: w.n.div_ceil(FP8_BLOCK),
            weight_scale_cols: scale_cols,
            activation_scale_cols: scale_cols,
        },
        a_fp8,
        a_scale,
        &w.weight,
        &w.scale,
        out,
    )?;
    Ok(())
}

/// Allocating convenience over [`fp8_linear_into`] for the oracle-gate/test
/// paths. Returns `[n]` bf16.
#[cfg(test)]
pub(crate) fn fp8_linear(
    ctx: &DeviceContext,
    w: &ProjWeight,
    input: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    let mut s = Glm52ProjScratch::new(ctx, w.k)?;
    let mut out = ctx.stream.alloc_zeros::<bf16>(w.n)?;
    fp8_linear_into(ctx, w, input, &mut s, &mut out)?;
    Ok(out)
}

/// Persistent scratch for one fp8 SwiGLU MLP shape. Sized to an exact
/// `intermediate` (the dense MLP and the MoE shared expert differ — 12288 vs
/// 2048 — and each gets its own instance so a cross-wiring crashes here).
pub(crate) struct Glm52MlpScratch {
    intermediate: usize,
    gate_out: CudaSlice<bf16>,
    up_out: CudaSlice<bf16>,
    gate_up: CudaSlice<bf16>,
    w_act: CudaSlice<u8>,
    w_act_scale: CudaSlice<f32>,
}

impl Glm52MlpScratch {
    pub(crate) fn new(ctx: &DeviceContext, intermediate: usize) -> Result<Self> {
        ensure!(
            intermediate.is_multiple_of(FP8_BLOCK),
            "GLM5.2 fp8_mlp intermediate {intermediate} not a multiple of {FP8_BLOCK}"
        );
        Ok(Self {
            intermediate,
            gate_out: ctx.stream.alloc_zeros::<bf16>(intermediate)?,
            up_out: ctx.stream.alloc_zeros::<bf16>(intermediate)?,
            gate_up: ctx.stream.alloc_zeros::<bf16>(2 * intermediate)?,
            w_act: ctx.stream.alloc_zeros::<u8>(intermediate)?,
            w_act_scale: ctx.stream.alloc_zeros::<f32>(intermediate / FP8_BLOCK)?,
        })
    }
}

/// A plain fp8 SwiGLU MLP for one token (bs=1) into a pre-allocated output:
/// `down(silu(gate(h)) * up(h))`, with SEPARATE gate/up projections (the
/// GLM5.2 dense layer and the MoE shared expert both use this shape -- only
/// the intermediate size differs). `out` is `[down.n]` bf16 (= `[HIDDEN]`).
pub(crate) fn fp8_mlp_into(
    ctx: &DeviceContext,
    gate: &ProjWeight,
    up: &ProjWeight,
    down: &ProjWeight,
    input: &CudaSlice<bf16>,
    proj: &mut Glm52ProjScratch,
    s: &mut Glm52MlpScratch,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let intermediate = gate.n;
    ensure!(
        up.n == intermediate && down.k == intermediate && gate.k == up.k && gate.k == down.n,
        "GLM5.2 fp8_mlp shape mismatch: gate [{},{}], up [{},{}], down [{},{}]",
        gate.n,
        gate.k,
        up.n,
        up.k,
        down.n,
        down.k
    );
    ensure!(
        s.intermediate == intermediate,
        "GLM5.2 fp8_mlp scratch sized for intermediate {} but weights have {intermediate}",
        s.intermediate
    );
    fp8_linear_into(ctx, gate, input, proj, &mut s.gate_out)?;
    fp8_linear_into(ctx, up, input, proj, &mut s.up_out)?;

    // Concatenate gate|up (gate first half) for the fused SwiGLU.
    ctx.stream
        .memcpy_dtod(&s.gate_out, &mut s.gate_up.slice_mut(0..intermediate))?;
    ctx.stream.memcpy_dtod(
        &s.up_out,
        &mut s.gate_up.slice_mut(intermediate..2 * intermediate),
    )?;

    glm52_silu_and_mul_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: 1,
            width: intermediate,
            group_size: FP8_BLOCK,
        },
        &s.gate_up,
        &mut s.w_act,
        &mut s.w_act_scale,
    )?;
    fp8_linear_prequant_into(ctx, down, &s.w_act, &s.w_act_scale, &mut proj.a_scale, out)
}

/// Allocating convenience over [`fp8_mlp_into`] for the oracle-gate/test
/// paths. Returns `[down.n]` bf16 (= `[HIDDEN]`).
#[cfg(test)]
pub(crate) fn fp8_mlp(
    ctx: &DeviceContext,
    gate: &ProjWeight,
    up: &ProjWeight,
    down: &ProjWeight,
    input: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    let mut proj = Glm52ProjScratch::new(ctx, gate.k.max(down.k))?;
    let mut s = Glm52MlpScratch::new(ctx, gate.n)?;
    let mut out = ctx.stream.alloc_zeros::<bf16>(down.n)?;
    fp8_mlp_into(ctx, gate, up, down, input, &mut proj, &mut s, &mut out)?;
    Ok(out)
}
