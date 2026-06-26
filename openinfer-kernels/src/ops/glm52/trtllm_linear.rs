use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_TRTLLM_LINEAR_BATCH_CAPACITY: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52TrtllmFp8LinearContract {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub weight_scale_rows: usize,
    pub weight_scale_cols: usize,
    pub activation_scale_cols: usize,
}

impl Glm52TrtllmFp8LinearContract {
    pub fn validate(self) -> Result<()> {
        ensure!(
            (1..=GLM52_TRTLLM_LINEAR_BATCH_CAPACITY).contains(&self.m),
            "GLM5.2 TRTLLM FP8 linear m {} out of 1..={}",
            self.m,
            GLM52_TRTLLM_LINEAR_BATCH_CAPACITY
        );
        ensure!(
            glm52_trtllm_fp8_linear_shape_supported(self.n, self.k),
            "GLM5.2 TRTLLM FP8 linear unsupported projection shape: n={}, k={}",
            self.n,
            self.k
        );
        ensure!(
            self.weight_scale_rows == self.n.div_ceil(128)
                && self.weight_scale_cols == self.k.div_ceil(128)
                && self.activation_scale_cols == self.k.div_ceil(128),
            "GLM5.2 TRTLLM FP8 linear scale grid drifted: {self:?}"
        );
        Ok(())
    }
}

pub fn glm52_trtllm_fp8_linear_shape_supported(n: usize, k: usize) -> bool {
    if k == 0 || !k.is_multiple_of(128) {
        return false;
    }
    matches!(
        (n, k),
        (2048, 6144)
            | (65536, 2048)
            | (576, 6144)
            | (114688, 512)
            | (6144, 16384)
            | (128, 6144)
            | (4096, 2048)
            | (12288, 6144)
            | (6144, 12288)
            | (6144, 2048)
    )
}

pub fn glm52_trtllm_fp8_linear_contract_validate(
    contract: Glm52TrtllmFp8LinearContract,
) -> Result<()> {
    contract.validate()?;
    let result = unsafe {
        ffi::glm52_trtllm_fp8_linear_contract_cuda(
            contract.m as i32,
            contract.n as i32,
            contract.k as i32,
            contract.weight_scale_rows as i32,
            contract.weight_scale_cols as i32,
            contract.activation_scale_cols as i32,
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 TRTLLM FP8 linear ABI contract check failed: {err}"))
}

pub fn glm52_trtllm_fp8_linear_workspace_size(
    contract: Glm52TrtllmFp8LinearContract,
) -> Result<usize> {
    contract.validate()?;
    let mut workspace_bytes = 0usize;
    let result = unsafe {
        ffi::glm52_trtllm_fp8_linear_workspace_size_cuda(
            contract.m as i32,
            contract.n as i32,
            contract.k as i32,
            &mut workspace_bytes as *mut usize,
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 TRTLLM FP8 linear workspace query failed: {err}"))?;
    Ok(workspace_bytes)
}

pub fn glm52_trtllm_fp8_linear_launch(
    ctx: &DeviceContext,
    contract: Glm52TrtllmFp8LinearContract,
    activation: &CudaSlice<u8>,
    activation_scale: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale_bytes: &CudaSlice<u8>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    validate_launch_buffers(
        contract,
        activation,
        activation_scale,
        weight,
        weight_scale_bytes,
        output,
    )?;
    let workspace_bytes = glm52_trtllm_fp8_linear_workspace_size(contract)?;
    ensure!(
        workspace_bytes == 0,
        "GLM5.2 TRTLLM FP8 linear unexpected workspace requirement: {workspace_bytes} bytes"
    );

    let (activation_ptr, _activation_guard) = activation.device_ptr(&ctx.stream);
    let (activation_scale_ptr, _activation_scale_guard) = activation_scale.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (weight_scale_ptr, _weight_scale_guard) = weight_scale_bytes.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_trtllm_fp8_linear_launch_cuda(
            activation_ptr as *const u8,
            activation_scale_ptr as *const f32,
            weight_ptr as *const u8,
            weight_scale_ptr as *const f32,
            output_ptr as *mut ffi::Half,
            std::ptr::null_mut(),
            0,
            contract.m as i32,
            contract.n as i32,
            contract.k as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 TRTLLM FP8 linear launch failed: {err}"))
}

fn validate_launch_buffers(
    contract: Glm52TrtllmFp8LinearContract,
    activation: &CudaSlice<u8>,
    activation_scale: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale_bytes: &CudaSlice<u8>,
    output: &CudaSlice<bf16>,
) -> Result<()> {
    contract.validate()?;
    let activation_len = contract.m * contract.k;
    ensure!(
        activation.len() >= activation_len,
        "GLM5.2 TRTLLM FP8 linear activation buffer too small: have {}, need {activation_len}",
        activation.len()
    );
    let activation_scale_len = contract.m * contract.activation_scale_cols;
    ensure!(
        activation_scale.len() >= activation_scale_len,
        "GLM5.2 TRTLLM FP8 linear activation scales too small: have {}, need {activation_scale_len}",
        activation_scale.len()
    );
    let weight_len = contract.n * contract.k;
    ensure!(
        weight.len() >= weight_len,
        "GLM5.2 TRTLLM FP8 linear weight buffer too small: have {}, need {weight_len}",
        weight.len()
    );
    let weight_scale_bytes_len =
        contract.weight_scale_rows * contract.weight_scale_cols * std::mem::size_of::<f32>();
    ensure!(
        weight_scale_bytes.len() >= weight_scale_bytes_len,
        "GLM5.2 TRTLLM FP8 linear weight scale buffer too small: have {}, need {weight_scale_bytes_len}",
        weight_scale_bytes.len()
    );
    let output_len = contract.m * contract.n;
    ensure!(
        output.len() >= output_len,
        "GLM5.2 TRTLLM FP8 linear output buffer too small: have {}, need {output_len}",
        output.len()
    );
    Ok(())
}
