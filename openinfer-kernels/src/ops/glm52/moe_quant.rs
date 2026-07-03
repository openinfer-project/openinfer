use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_MOE_QUANT_GROUP_SIZE: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52MoeQuantShape {
    pub rows: usize,
    pub width: usize,
    pub group_size: usize,
}

impl Glm52MoeQuantShape {
    pub fn scale_cols(self) -> Result<usize> {
        self.validate()?;
        Ok(self.width / self.group_size)
    }

    pub fn validate(self) -> Result<()> {
        ensure!(self.rows > 0, "GLM5.2 MoE quant rows must be positive");
        ensure!(self.width > 0, "GLM5.2 MoE quant width must be positive");
        ensure!(
            self.group_size == GLM52_MOE_QUANT_GROUP_SIZE,
            "GLM5.2 MoE quant group_size must be {GLM52_MOE_QUANT_GROUP_SIZE}, got {}",
            self.group_size
        );
        ensure!(
            self.width.is_multiple_of(self.group_size),
            "GLM5.2 MoE quant width {} is not divisible by group_size {}",
            self.width,
            self.group_size
        );
        Ok(())
    }
}

pub fn glm52_fp8_per_token_group_quant_bf16_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
) -> Result<()> {
    validate_quant_buffers(shape, input, output, scales)?;
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _scale_guard) = scales.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_fp8_per_token_group_quant_bf16_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FP8 per-token-group quant launch failed: {err}"))
}

pub fn glm52_silu_and_mul_per_token_group_quant_bf16_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
) -> Result<()> {
    shape.validate()?;
    ensure!(
        input.len() >= shape.rows * shape.width * 2,
        "GLM5.2 SiLU quant input too small: have {}, need {}",
        input.len(),
        shape.rows * shape.width * 2
    );
    ensure!(
        output.len() >= shape.rows * shape.width,
        "GLM5.2 SiLU quant output too small: have {}, need {}",
        output.len(),
        shape.rows * shape.width
    );
    ensure!(
        scales.len() >= shape.rows * shape.scale_cols()?,
        "GLM5.2 SiLU quant scales too small: have {}, need {}",
        scales.len(),
        shape.rows * shape.scale_cols()?
    );
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _scale_guard) = scales.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_silu_and_mul_per_token_group_quant_bf16_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 SiLU+mul FP8 group quant launch failed: {err}"))
}

pub fn glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    input: &CudaSlice<bf16>,
    topk_weights: &CudaSlice<f32>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
) -> Result<()> {
    shape.validate()?;
    ensure!(
        input.len() >= shape.rows * shape.width * 2,
        "GLM5.2 weighted SiLU quant input too small: have {}, need {}",
        input.len(),
        shape.rows * shape.width * 2
    );
    ensure!(
        topk_weights.len() >= shape.rows,
        "GLM5.2 weighted SiLU quant topk_weights too small: have {}, need {}",
        topk_weights.len(),
        shape.rows
    );
    ensure!(
        output.len() >= shape.rows * shape.width,
        "GLM5.2 weighted SiLU quant output too small: have {}, need {}",
        output.len(),
        shape.rows * shape.width
    );
    ensure!(
        scales.len() >= shape.rows * shape.scale_cols()?,
        "GLM5.2 weighted SiLU quant scales too small: have {}, need {}",
        scales.len(),
        shape.rows * shape.scale_cols()?
    );
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = topk_weights.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _scale_guard) = scales.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_silu_and_mul_weighted_per_token_group_quant_bf16_cuda(
            input_ptr as *const ffi::Half,
            weight_ptr as *const f32,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 weighted SiLU+mul FP8 group quant launch failed: {err}"))
}

fn validate_quant_buffers(
    shape: Glm52MoeQuantShape,
    input: &CudaSlice<bf16>,
    output: &CudaSlice<u8>,
    scales: &CudaSlice<f32>,
) -> Result<()> {
    shape.validate()?;
    let scale_elems = shape.rows * shape.scale_cols()?;
    ensure!(
        input.len() >= shape.rows * shape.width,
        "GLM5.2 MoE quant input too small: have {}, need {}",
        input.len(),
        shape.rows * shape.width
    );
    ensure!(
        output.len() >= shape.rows * shape.width,
        "GLM5.2 MoE quant output too small: have {}, need {}",
        output.len(),
        shape.rows * shape.width
    );
    ensure!(
        scales.len() >= scale_elems,
        "GLM5.2 MoE quant scales too small: have {}, need {scale_elems}",
        scales.len()
    );
    Ok(())
}

/// [`glm52_fp8_per_token_group_quant_bf16_launch`] with a device-side row
/// bound: the grid covers `shape.rows` (capacity — CUDA-graph shape
/// stability), but blocks whose row is at or past `row_bound[bound_index]`
/// retire immediately. Used by the EP8 MoE recv re-quant, where the bound is
/// the grouped-GEMM aligned segment end (`expert_offsets[n_local]`) and the
/// capacity is the worst-case `bound_rows`.
#[allow(clippy::too_many_arguments)]
pub fn glm52_fp8_per_token_group_quant_bf16_bounded_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
    row_bound: &CudaSlice<i64>,
    bound_index: usize,
) -> Result<()> {
    shape.validate()?;
    ensure!(
        input.len() >= shape.rows * shape.width,
        "GLM5.2 FP8 quant input too small: have {}, need {}",
        input.len(),
        shape.rows * shape.width
    );
    ensure!(
        output.len() >= shape.rows * shape.width
            && scales.len() >= shape.rows * shape.scale_cols()?,
        "GLM5.2 FP8 quant outputs too small"
    );
    ensure!(
        row_bound.len() > bound_index,
        "GLM5.2 FP8 quant row_bound index {bound_index} outside buffer of {}",
        row_bound.len()
    );
    let (input_ptr, _g0) = input.device_ptr(&ctx.stream);
    let (output_ptr, _g1) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _g2) = scales.device_ptr_mut(&ctx.stream);
    let (bound_ptr, _g3) = row_bound.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_fp8_per_token_group_quant_bf16_bounded_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            (bound_ptr as *const i64).wrapping_add(bound_index),
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FP8 bounded group quant launch failed: {err}"))
}

/// [`glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch`] with a
/// device-side row bound (see the quant twin above).
#[allow(clippy::too_many_arguments)]
pub fn glm52_silu_and_mul_weighted_per_token_group_quant_bf16_bounded_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    input: &CudaSlice<bf16>,
    topk_weights: &CudaSlice<f32>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
    row_bound: &CudaSlice<i64>,
    bound_index: usize,
) -> Result<()> {
    shape.validate()?;
    ensure!(
        input.len() >= shape.rows * shape.width * 2
            && topk_weights.len() >= shape.rows
            && output.len() >= shape.rows * shape.width
            && scales.len() >= shape.rows * shape.scale_cols()?,
        "GLM5.2 weighted SiLU bounded quant buffers too small"
    );
    ensure!(
        row_bound.len() > bound_index,
        "GLM5.2 weighted SiLU quant row_bound index {bound_index} outside buffer of {}",
        row_bound.len()
    );
    let (input_ptr, _g0) = input.device_ptr(&ctx.stream);
    let (weight_ptr, _g1) = topk_weights.device_ptr(&ctx.stream);
    let (output_ptr, _g2) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _g3) = scales.device_ptr_mut(&ctx.stream);
    let (bound_ptr, _g4) = row_bound.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_silu_and_mul_weighted_per_token_group_quant_bf16_bounded_cuda(
            input_ptr as *const ffi::Half,
            weight_ptr as *const f32,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            (bound_ptr as *const i64).wrapping_add(bound_index),
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 weighted SiLU bounded quant launch failed: {err}"))
}
