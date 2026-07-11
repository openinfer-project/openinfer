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

/// Bounded re-quant writing the DeepGEMM masked grouped layout: the loop
/// space stays the aligned recv rows (`shape.rows` capacity, device bound),
/// `row_map` redirects each row to its masked slot (skipping alignment
/// gaps), values land in `[groups, masked_cap, width]` and scales in the
/// mn-major `[groups, width/128, masked_cap]` layout the masked GEMM's SFA
/// descriptor reads. Per-row math is bit-identical to the bounded twin.
#[allow(clippy::too_many_arguments)]
pub fn glm52_fp8_per_token_group_quant_bf16_masked_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    masked_groups: usize,
    masked_cap: usize,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
    row_bound: &CudaSlice<i64>,
    bound_index: usize,
    row_map: &CudaSlice<i32>,
) -> Result<()> {
    shape.validate()?;
    let masked_rows = masked_groups * masked_cap;
    ensure!(
        input.len() >= shape.rows * shape.width
            && output.len() >= masked_rows * shape.width
            && scales.len() >= masked_rows * shape.scale_cols()?
            && row_map.len() >= shape.rows,
        "GLM5.2 FP8 masked quant buffers too small"
    );
    ensure!(
        row_bound.len() > bound_index,
        "GLM5.2 FP8 masked quant row_bound index {bound_index} outside buffer of {}",
        row_bound.len()
    );
    let (input_ptr, _g0) = input.device_ptr(&ctx.stream);
    let (output_ptr, _g1) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _g2) = scales.device_ptr_mut(&ctx.stream);
    let (bound_ptr, _g3) = row_bound.device_ptr(&ctx.stream);
    let (map_ptr, _g4) = row_map.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_fp8_per_token_group_quant_bf16_masked_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            (bound_ptr as *const i64).wrapping_add(bound_index),
            map_ptr as *const i32,
            masked_cap as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FP8 masked group quant launch failed: {err}"))
}

/// Bounded weighted SwiGLU quant for the masked layout: the gate|up input
/// rows are already masked (the W13 masked GEMM wrote them), the route
/// weight stays indexed by the aligned recv row, output/scales land in the
/// masked layouts (see the quant twin above).
#[allow(clippy::too_many_arguments)]
pub fn glm52_silu_and_mul_weighted_per_token_group_quant_bf16_masked_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    masked_groups: usize,
    masked_cap: usize,
    input: &CudaSlice<bf16>,
    topk_weights: &CudaSlice<f32>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
    row_bound: &CudaSlice<i64>,
    bound_index: usize,
    row_map: &CudaSlice<i32>,
) -> Result<()> {
    shape.validate()?;
    let masked_rows = masked_groups * masked_cap;
    ensure!(
        input.len() >= masked_rows * shape.width * 2
            && topk_weights.len() >= shape.rows
            && output.len() >= masked_rows * shape.width
            && scales.len() >= masked_rows * shape.scale_cols()?
            && row_map.len() >= shape.rows,
        "GLM5.2 weighted SiLU masked quant buffers too small"
    );
    ensure!(
        row_bound.len() > bound_index,
        "GLM5.2 weighted SiLU masked quant row_bound index {bound_index} outside buffer of {}",
        row_bound.len()
    );
    let (input_ptr, _g0) = input.device_ptr(&ctx.stream);
    let (weight_ptr, _g1) = topk_weights.device_ptr(&ctx.stream);
    let (output_ptr, _g2) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _g3) = scales.device_ptr_mut(&ctx.stream);
    let (bound_ptr, _g4) = row_bound.device_ptr(&ctx.stream);
    let (map_ptr, _g5) = row_map.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_silu_and_mul_weighted_per_token_group_quant_bf16_masked_cuda(
            input_ptr as *const ffi::Half,
            weight_ptr as *const f32,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            (bound_ptr as *const i64).wrapping_add(bound_index),
            map_ptr as *const i32,
            masked_cap as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 weighted SiLU masked quant launch failed: {err}"))
}
