//! GLM5.2 bs=1 weight-only FP8 GEMV path: bf16 activation x fp8 e4m3 block-scale
//! weight (dequant on the fly), plus the bf16 SiLU + slot-combine companions that
//! bracket the routed MoE FFN. Replaces the TRTLLM tile-GEMM + activation quant +
//! scale relayout at M=1. See `csrc/glm52/glm52_moe_gemv.cu`.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

const FP8_BLOCK: usize = 128;

/// W13 (gate|up) grouped-GEMV operand kind.
pub const GLM52_GEMV_KIND_W13: i32 = 1;
/// W2 (down) grouped-GEMV operand kind.
pub const GLM52_GEMV_KIND_W2: i32 = 2;

/// Routed MoE grouped GEMV. For each of `topk` slots, dequant the selected expert
/// (`topk_idx[slot]`) fp8 weight on the fly and multiply by the bf16 activation.
/// `act_row_stride == 0` broadcasts one activation row (W13 input); `== k` feeds a
/// per-slot row (W2 input). Output is slot-major `[topk, n]` bf16.
#[allow(clippy::too_many_arguments)]
pub fn glm52_moe_fp8_weight_only_gemv_launch(
    ctx: &DeviceContext,
    operand_kind: i32,
    n: usize,
    k: usize,
    topk: usize,
    act_row_stride: usize,
    activation: &CudaSlice<bf16>,
    topk_idx: &CudaSlice<i32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        n > 0 && k > 0 && topk > 0,
        "GLM5.2 GEMV needs positive n/k/topk, got {n}/{k}/{topk}"
    );
    ensure!(
        weight.len().is_multiple_of(n * k) && weight.len() >= n * k,
        "GLM5.2 GEMV weight {} is not a whole number of {n}x{k} experts",
        weight.len()
    );
    let experts = weight.len() / (n * k);
    let scale_per_expert = (n / FP8_BLOCK) * (k / FP8_BLOCK);
    ensure!(
        weight_scale.len() >= experts * scale_per_expert,
        "GLM5.2 GEMV weight_scale too small: have {}, need {}",
        weight_scale.len(),
        experts * scale_per_expert
    );
    let act_needed = if act_row_stride == 0 { k } else { topk * k };
    ensure!(
        topk_idx.len() >= topk && activation.len() >= act_needed && out.len() >= topk * n,
        "GLM5.2 GEMV buffers too small: idx {}, act {} (need {act_needed}), out {} (need {})",
        topk_idx.len(),
        activation.len(),
        out.len(),
        topk * n
    );

    let (act_ptr, _a) = activation.device_ptr(&ctx.stream);
    let (idx_ptr, _i) = topk_idx.device_ptr(&ctx.stream);
    let (w_ptr, _w) = weight.device_ptr(&ctx.stream);
    let (s_ptr, _s) = weight_scale.device_ptr(&ctx.stream);
    let (out_ptr, _o) = out.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::glm52_moe_fp8_weight_only_gemv_cuda(
            operand_kind,
            act_ptr as *const ffi::Half,
            act_row_stride as i32,
            idx_ptr as *const i32,
            w_ptr as *const u8,
            s_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            topk as i32,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 grouped GEMV launch failed: {err}"))
}

/// Plain weight-only FP8 GEMV: bf16 activation x one fp8 linear/expert weight `[n,k]`,
/// scale dequant on the fly. `scale_bytes` is the `ProjWeight` `weight_scale_inv` kept
/// as raw `u8` f32-bytes (reinterpreted here as f32). `n` need not be a multiple of 128
/// (e.g. MLA kv_a n=576): the scale buffer is sized `div_ceil(n,128) * div_ceil(k,128)`
/// f32 and the kernel's 32-row blocks never straddle a /128 scale boundary.
pub fn glm52_fp8_weight_only_gemv_launch(
    ctx: &DeviceContext,
    n: usize,
    k: usize,
    activation: &CudaSlice<bf16>,
    weight: &CudaSlice<u8>,
    scale_bytes: &CudaSlice<u8>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        n > 0 && k > 0,
        "GLM5.2 linear GEMV needs positive n/k, got {n}/{k}"
    );
    let scale_len = n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK) * 4;
    ensure!(
        weight.len() >= n * k
            && scale_bytes.len() >= scale_len
            && activation.len() >= k
            && out.len() >= n,
        "GLM5.2 linear GEMV buffers too small: w {} (need {}), scale {} (need {scale_len}), act {} (need {k}), out {} (need {n})",
        weight.len(),
        n * k,
        scale_bytes.len(),
        activation.len(),
        out.len()
    );
    let (act_ptr, _a) = activation.device_ptr(&ctx.stream);
    let (w_ptr, _w) = weight.device_ptr(&ctx.stream);
    let (s_ptr, _s) = scale_bytes.device_ptr(&ctx.stream);
    let (out_ptr, _o) = out.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::glm52_fp8_weight_only_gemv_cuda(
            act_ptr as *const ffi::Half,
            w_ptr as *const u8,
            s_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 linear GEMV launch failed: {err}"))
}

/// SiLU(gate)*up -> bf16. `input` is `[rows, 2*inter]` (gate|up); `output` is
/// `[rows, inter]`. `topk_weights` (per-row route weight) is folded in when `Some`;
/// `None` is the plain MLP SwiGLU (no route weight).
pub fn glm52_silu_and_mul_weighted_bf16_launch(
    ctx: &DeviceContext,
    rows: usize,
    inter: usize,
    input: &CudaSlice<bf16>,
    topk_weights: Option<&CudaSlice<f32>>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        rows > 0 && inter > 0,
        "GLM5.2 weighted SiLU needs positive rows/inter, got {rows}/{inter}"
    );
    ensure!(
        input.len() >= rows * 2 * inter && output.len() >= rows * inter,
        "GLM5.2 weighted SiLU buffers too small: in {}, out {}",
        input.len(),
        output.len()
    );
    let (in_ptr, _i) = input.device_ptr(&ctx.stream);
    let (out_ptr, _o) = output.device_ptr_mut(&ctx.stream);
    let mut w_guard = None;
    let mut w_ptr: *const f32 = core::ptr::null();
    if let Some(w) = topk_weights {
        ensure!(
            w.len() >= rows,
            "GLM5.2 weighted SiLU route weights too small: {} < {rows}",
            w.len()
        );
        let (p, g) = w.device_ptr(&ctx.stream);
        w_guard = Some(g);
        w_ptr = p as *const f32;
    }
    let result = unsafe {
        ffi::glm52_silu_and_mul_weighted_bf16_cuda(
            in_ptr as *const ffi::Half,
            w_ptr,
            out_ptr as *mut ffi::Half,
            rows as i32,
            inter as i32,
            ctx.stream.cu_stream(),
        )
    };
    drop(w_guard);
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 weighted SiLU launch failed: {err}"))
}

/// Combine the topk slot rows of `w2_out` `[topk, n]` into `routed[n]` (plain sum;
/// the route weight is already folded by the weighted SiLU).
pub fn glm52_moe_combine_slots_launch(
    ctx: &DeviceContext,
    n: usize,
    topk: usize,
    w2_out: &CudaSlice<bf16>,
    routed: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        n > 0 && topk > 0,
        "GLM5.2 combine-slots needs positive n/topk, got {n}/{topk}"
    );
    ensure!(
        w2_out.len() >= topk * n && routed.len() >= n,
        "GLM5.2 combine-slots buffers too small: w2 {}, routed {}",
        w2_out.len(),
        routed.len()
    );
    let (w_ptr, _w) = w2_out.device_ptr(&ctx.stream);
    let (r_ptr, _r) = routed.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::glm52_moe_combine_slots_cuda(
            w_ptr as *const ffi::Half,
            r_ptr as *mut ffi::Half,
            n as i32,
            topk as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 combine-slots launch failed: {err}"))
}
