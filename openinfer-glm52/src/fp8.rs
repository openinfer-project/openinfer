//! Shared GLM5.2 fp8 block-scaled projection primitives (decode, row-batched).
//!
//! Every dense/attention/expert projection in GLM5.2 is fp8 e4m3 with a per-128
//! block `weight_scale_inv`. At m=1 the projection is a weight-only GEMV: the
//! bf16 activation is read directly and the fp8 weight is block-scale dequanted
//! on the fly (`csrc/glm52/glm52_moe_gemv.cu`) — no activation quant, no TMA
//! scale relayout, one kernel instead of three, and weight-memory-bound instead
//! of the TRTLLM CUTLASS M-tile padding 1→64. MLA, the dense MLP, and the MoE
//! shared expert all share these helpers; only the EP8 routed-expert chain
//! (multi-row) stays on the grouped CUTLASS path.

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;

use openinfer_kernels::ops::{
    glm52_fp8_weight_only_gemv_launch, glm52_silu_and_mul_weighted_bf16_launch,
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
    #[cfg(test)]
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

/// Pack two fp8 projections that share the same input into one `[a.n + b.n, k]`
/// projection (weight bytes and per-128-block scale rows concatenated along n).
/// Requires `a.n` to be a multiple of 128 so `b`'s scale rows stay aligned to
/// the packed row/128 grid. Used to fuse gate|up into a single GEMV whose
/// output is exactly the `[gate | up]` layout the SwiGLU consumes.
pub(crate) fn pack_proj_pair(
    ctx: &DeviceContext,
    a: &ProjWeight,
    b: &ProjWeight,
) -> Result<ProjWeight> {
    ensure!(
        a.k == b.k,
        "GLM5.2 proj pack k mismatch: {} vs {}",
        a.k,
        b.k
    );
    ensure!(
        a.n.is_multiple_of(FP8_BLOCK),
        "GLM5.2 proj pack first n {} not a multiple of {FP8_BLOCK}",
        a.n
    );
    let n = a.n + b.n;
    let mut weight = ctx.stream.alloc_zeros::<u8>(n * a.k)?;
    ctx.stream
        .memcpy_dtod(&a.weight, &mut weight.slice_mut(0..a.n * a.k))?;
    ctx.stream
        .memcpy_dtod(&b.weight, &mut weight.slice_mut(a.n * a.k..n * a.k))?;
    let scale_cols = a.k.div_ceil(FP8_BLOCK);
    let a_scale_len = (a.n / FP8_BLOCK) * scale_cols * 4;
    let b_scale_len = b.n.div_ceil(FP8_BLOCK) * scale_cols * 4;
    let mut scale = ctx.stream.alloc_zeros::<u8>(a_scale_len + b_scale_len)?;
    ctx.stream
        .memcpy_dtod(&a.scale, &mut scale.slice_mut(0..a_scale_len))?;
    ctx.stream.memcpy_dtod(
        &b.scale,
        &mut scale.slice_mut(a_scale_len..a_scale_len + b_scale_len),
    )?;
    ProjWeight::from_device(weight, scale, n, a.k)
}

/// One fp8 projection into a pre-allocated output: weight-only GEMV of `rows`
/// bf16 activation rows against the fp8 `[n,k]` weight, block scale dequanted
/// on the fly. rows > 1 runs the weight-stationary batched kernel — the weight
/// is still read once, and every row is bit-identical to the rows=1 kernel.
pub(crate) fn fp8_linear_into(
    ctx: &DeviceContext,
    w: &ProjWeight,
    rows: usize,
    input: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        input.len() >= rows * w.k,
        "GLM5.2 fp8_linear input {} < rows {rows} * k {}",
        input.len(),
        w.k
    );
    glm52_fp8_weight_only_gemv_launch(ctx, rows, w.n, w.k, input, &w.weight, &w.scale, out)
}

/// Allocating convenience over [`fp8_linear_into`] for the oracle-gate/test
/// paths. Returns `[n]` bf16.
#[cfg(test)]
pub(crate) fn fp8_linear(
    ctx: &DeviceContext,
    w: &ProjWeight,
    input: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    let mut out = ctx.stream.alloc_zeros::<bf16>(w.n)?;
    fp8_linear_into(ctx, w, 1, input, &mut out)?;
    Ok(out)
}

/// Persistent scratch for one fp8 SwiGLU MLP shape. Sized to an exact
/// `intermediate` (the dense MLP and the MoE shared expert differ — 12288 vs
/// 2048 — and each gets its own instance so a cross-wiring crashes here).
pub(crate) struct Glm52MlpScratch {
    intermediate: usize,
    rows: usize,
    gate_up: CudaSlice<bf16>,
    silu_out: CudaSlice<bf16>,
}

impl Glm52MlpScratch {
    pub(crate) fn new(ctx: &DeviceContext, intermediate: usize, rows: usize) -> Result<Self> {
        ensure!(
            intermediate.is_multiple_of(FP8_BLOCK),
            "GLM5.2 fp8_mlp intermediate {intermediate} not a multiple of {FP8_BLOCK}"
        );
        ensure!(rows > 0, "GLM5.2 fp8_mlp scratch needs positive rows");
        Ok(Self {
            intermediate,
            rows,
            gate_up: ctx.stream.alloc_zeros::<bf16>(rows * 2 * intermediate)?,
            silu_out: ctx.stream.alloc_zeros::<bf16>(rows * intermediate)?,
        })
    }
}

/// A plain fp8 SwiGLU MLP over the scratch's `rows` tokens into a
/// pre-allocated output: `down(silu(gate(h)) * up(h))` with the gate|up
/// projections PACKED into one `[2*intermediate, k]` weight (see
/// [`pack_proj_pair`]) — a single GEMV writes the `[gate | up]` layout the
/// SwiGLU consumes, no concat copies. `out` is `[rows, down.n]` bf16.
pub(crate) fn fp8_mlp_into(
    ctx: &DeviceContext,
    gate_up: &ProjWeight,
    down: &ProjWeight,
    input: &CudaSlice<bf16>,
    s: &mut Glm52MlpScratch,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        gate_up.n.is_multiple_of(2),
        "GLM5.2 fp8_mlp packed gate|up n {} is odd",
        gate_up.n
    );
    let intermediate = gate_up.n / 2;
    ensure!(
        down.k == intermediate && gate_up.k == down.n,
        "GLM5.2 fp8_mlp shape mismatch: gate_up [{},{}], down [{},{}]",
        gate_up.n,
        gate_up.k,
        down.n,
        down.k
    );
    ensure!(
        s.intermediate == intermediate,
        "GLM5.2 fp8_mlp scratch sized for intermediate {} but weights have {intermediate}",
        s.intermediate
    );
    fp8_linear_into(ctx, gate_up, s.rows, input, &mut s.gate_up)?;
    // bf16 SwiGLU (no route weight, no activation quant) -> bf16 down input.
    glm52_silu_and_mul_weighted_bf16_launch(
        ctx,
        s.rows,
        intermediate,
        &s.gate_up,
        None,
        &mut s.silu_out,
    )?;
    fp8_linear_into(ctx, down, s.rows, &s.silu_out, out)
}

/// Allocating convenience over [`fp8_mlp_into`] for the oracle-gate/test
/// paths. Returns `[down.n]` bf16 (= `[HIDDEN]`).
#[cfg(test)]
pub(crate) fn fp8_mlp(
    ctx: &DeviceContext,
    gate_up: &ProjWeight,
    down: &ProjWeight,
    input: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    let mut s = Glm52MlpScratch::new(ctx, gate_up.n / 2, 1)?;
    let mut out = ctx.stream.alloc_zeros::<bf16>(down.n)?;
    fp8_mlp_into(ctx, gate_up, down, input, &mut s, &mut out)?;
    Ok(out)
}
