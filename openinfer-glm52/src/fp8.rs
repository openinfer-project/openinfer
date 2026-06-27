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
}

/// One fp8 projection (bs=1): quant the bf16 activation, then the prequant linear.
/// Returns `[n]` bf16.
pub(crate) fn fp8_linear(
    ctx: &DeviceContext,
    w: &ProjWeight,
    input: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    ensure!(
        input.len() >= w.k,
        "GLM5.2 fp8_linear input {} < k {}",
        input.len(),
        w.k
    );
    let scale_cols = w.k / FP8_BLOCK;
    let mut a_fp8 = ctx.stream.alloc_zeros::<u8>(w.k)?;
    let mut a_scale_plain = ctx.stream.alloc_zeros::<f32>(scale_cols)?;
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: 1,
            width: w.k,
            group_size: FP8_BLOCK,
        },
        input,
        &mut a_fp8,
        &mut a_scale_plain,
    )?;
    fp8_linear_prequant(ctx, w, &a_fp8, &a_scale_plain)
}

/// One fp8 projection (bs=1) from a pre-quantized activation: relay the plain
/// per-group activation scale into the TRTLLM col-major TMA layout, then the
/// blockscale linear. Used when the input is already fp8 (e.g. the SwiGLU output
/// feeding a down projection). Returns `[n]` bf16.
pub(crate) fn fp8_linear_prequant(
    ctx: &DeviceContext,
    w: &ProjWeight,
    a_fp8: &CudaSlice<u8>,
    a_scale_plain: &CudaSlice<f32>,
) -> Result<CudaSlice<bf16>> {
    let scale_cols = w.k / FP8_BLOCK;
    ensure!(
        a_fp8.len() >= w.k && a_scale_plain.len() >= scale_cols,
        "GLM5.2 fp8_linear_prequant input too small: fp8 {} (need {}), scale {} (need {scale_cols})",
        a_fp8.len(),
        w.k,
        a_scale_plain.len()
    );
    let layout = Glm52DeepGemmScaleLayout::f32(1, scale_cols);
    let mut a_scale = ctx.stream.alloc_zeros::<f32>(layout.output_len()?)?;
    glm52_deepgemm_mn_major_tma_aligned_f32_launch(ctx, layout, a_scale_plain, &mut a_scale)?;
    let mut out = ctx.stream.alloc_zeros::<bf16>(w.n)?;
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
        &a_scale,
        &w.weight,
        &w.scale,
        &mut out,
    )?;
    Ok(out)
}

/// A plain fp8 SwiGLU MLP for one token (bs=1): `down(silu(gate(h)) * up(h))`, with
/// SEPARATE gate/up projections (the GLM5.2 dense layer and the MoE shared expert
/// both use this shape -- only the intermediate size differs, derived here from the
/// weights). Returns `[down.n]` bf16 (= `[HIDDEN]`).
pub(crate) fn fp8_mlp(
    ctx: &DeviceContext,
    gate: &ProjWeight,
    up: &ProjWeight,
    down: &ProjWeight,
    input: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
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
        intermediate.is_multiple_of(FP8_BLOCK),
        "GLM5.2 fp8_mlp intermediate {intermediate} not a multiple of {FP8_BLOCK}"
    );
    let gate_out = fp8_linear(ctx, gate, input)?; // [intermediate]
    let up_out = fp8_linear(ctx, up, input)?; // [intermediate]

    // Concatenate gate|up (gate first half) for the fused SwiGLU.
    let mut gate_up = ctx.stream.alloc_zeros::<bf16>(2 * intermediate)?;
    ctx.stream
        .memcpy_dtod(&gate_out, &mut gate_up.slice_mut(0..intermediate))?;
    ctx.stream.memcpy_dtod(
        &up_out,
        &mut gate_up.slice_mut(intermediate..2 * intermediate),
    )?;

    let mut w_act = ctx.stream.alloc_zeros::<u8>(intermediate)?;
    let mut w_act_scale = ctx.stream.alloc_zeros::<f32>(intermediate / FP8_BLOCK)?;
    glm52_silu_and_mul_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: 1,
            width: intermediate,
            group_size: FP8_BLOCK,
        },
        &gate_up,
        &mut w_act,
        &mut w_act_scale,
    )?;
    fp8_linear_prequant(ctx, down, &w_act, &w_act_scale)
}
