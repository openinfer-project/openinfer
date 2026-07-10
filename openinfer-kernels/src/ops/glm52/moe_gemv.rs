//! GLM5.2 weight-only FP8 GEMV and bf16 SwiGLU helpers used by dense,
//! attention, indexer, and shared-expert projections.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

const FP8_BLOCK: usize = 128;

/// Plain `SiLU(gate) * up` over `[rows, 2*inter]` into `[rows, inter]`.
pub fn glm52_silu_and_mul_bf16_launch(
    ctx: &DeviceContext,
    rows: usize,
    inter: usize,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        rows > 0 && inter > 0,
        "GLM5.2 SiLU needs positive rows/inter, got {rows}/{inter}"
    );
    ensure!(
        input.len() >= rows * 2 * inter && output.len() >= rows * inter,
        "GLM5.2 SiLU buffers too small: in {}, out {}",
        input.len(),
        output.len()
    );
    let (in_ptr, _i) = input.device_ptr(&ctx.stream);
    let (out_ptr, _o) = output.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::glm52_silu_and_mul_bf16_cuda(
            in_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            rows as i32,
            inter as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 SiLU launch failed: {err}"))
}

/// f32 partial-scratch floats PER ROW the tensor-core batched GEMV path can
/// need: max ksplit (16) × max whitelisted n (dense gate|up 24576). Callers
/// size their buffer as `rows * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW` — one
/// constant instead of mirroring the CUDA-side per-shape config table; the
/// launcher still guards the exact requirement and rejects a short buffer.
pub const GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW: usize = 16 * 24576;

/// Plain weight-only fp8 GEMV (bs=1): `out[n] = deq(weight[n,k]) @ activation[k]`
/// with the bf16 activation read directly (no activation quant, no scale
/// relayout) and the e4m3 block-scale weight dequanted on the fly. Replaces
/// the TRTLLM CUTLASS block-scale GEMM at m=1, where the M-tile pads 1->64 and
/// runs compute-bound. `scale_bytes` is the checkpoint `weight_scale_inv`
/// (f32 `[ceil(n/128), k/128]`) kept as raw bytes.
///
/// `scratch` is the caller-owned f32 partial buffer for the rows-4/8
/// tensor-core path. Ownership contract: the layer forward overlaps the ctx
/// and aux streams, so a scratch buffer must belong to exactly one launch
/// stream — every scratch struct owns its own. Pass `None` only on rows ≤ 2
/// paths; an mma-routed launch without a buffer fails loudly.
pub fn glm52_fp8_weight_only_gemv_launch(
    ctx: &DeviceContext,
    rows: usize,
    n: usize,
    k: usize,
    activation: &CudaSlice<bf16>,
    weight: &CudaSlice<u8>,
    scale_bytes: &CudaSlice<u8>,
    scratch: Option<&mut CudaSlice<f32>>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        rows > 0 && n > 0 && k > 0,
        "GLM5.2 linear GEMV needs positive rows/n/k, got {rows}/{n}/{k}"
    );
    let scale_len = n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK) * 4;
    ensure!(
        weight.len() >= n * k
            && scale_bytes.len() >= scale_len
            && activation.len() >= rows * k
            && out.len() >= rows * n,
        "GLM5.2 linear GEMV buffers too small: w {} (need {}), scale {} (need {scale_len}), act {} (need {}), out {} (need {})",
        weight.len(),
        n * k,
        scale_bytes.len(),
        activation.len(),
        rows * k,
        out.len(),
        rows * n
    );
    let (act_ptr, _a) = activation.device_ptr(&ctx.stream);
    let (w_ptr, _w) = weight.device_ptr(&ctx.stream);
    let (s_ptr, _s) = scale_bytes.device_ptr(&ctx.stream);
    let (out_ptr, _o) = out.device_ptr_mut(&ctx.stream);
    let (scr_ptr, scr_floats, _scr) = match scratch {
        Some(buf) => {
            let len = buf.len();
            let (ptr, guard) = buf.device_ptr_mut(&ctx.stream);
            (ptr as *mut f32, len, Some(guard))
        }
        None => (std::ptr::null_mut(), 0, None),
    };
    // rows == 1 runs the m=1 kernel; rows 2 the bit-parity register tile;
    // rows 4/8 the tensor-core mma path on winning shapes (deterministic per
    // bucket, not bit-identical to m=1). The CUDA side whitelists the
    // supported batches — a drifted GLM52_DECODE_BUCKETS crashes here.
    unsafe {
        ffi::glm52_fp8_weight_only_gemv_batched_cuda(
            act_ptr as *const ffi::Half,
            w_ptr as *const u8,
            s_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            scr_ptr,
            scr_floats,
            rows as i32,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 linear GEMV launch failed: {err}"))
}
