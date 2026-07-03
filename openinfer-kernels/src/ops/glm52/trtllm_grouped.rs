use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use super::deepgemm_layout::glm52_trtllm_grouped_offset_padded_rows;
use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_TRTLLM_GROUPED_W13_KIND: i32 = 1;
pub const GLM52_TRTLLM_GROUPED_W2_KIND: i32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm52TrtllmGroupedFp8Kind {
    W13,
    W2,
}

impl Glm52TrtllmGroupedFp8Kind {
    fn abi(self) -> i32 {
        match self {
            Self::W13 => GLM52_TRTLLM_GROUPED_W13_KIND,
            Self::W2 => GLM52_TRTLLM_GROUPED_W2_KIND,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52TrtllmGroupedFp8Contract {
    pub groups: usize,
    pub m_capacity: usize,
    pub n: usize,
    pub k: usize,
    pub weight_scale_rows: usize,
    pub weight_scale_cols: usize,
    pub activation_scale_cols: usize,
    pub activation_scale_trtllm_rows: usize,
}

impl Glm52TrtllmGroupedFp8Contract {
    pub fn validate(self, kind: Glm52TrtllmGroupedFp8Kind) -> Result<()> {
        ensure!(
            self.groups > 0,
            "GLM5.2 TRTLLM grouped FP8 needs groups>0, got {}",
            self.groups
        );
        let offset_rows = glm52_trtllm_grouped_offset_padded_rows(self.m_capacity, self.groups);
        ensure!(
            self.m_capacity > 0 && self.activation_scale_trtllm_rows == offset_rows,
            "GLM5.2 TRTLLM grouped FP8 needs m_capacity>0 and activation scale rows={offset_rows} (offset-padded for m_capacity={}, groups={}), got m_capacity={}, activation_scale_trtllm_rows={}",
            self.m_capacity,
            self.groups,
            self.m_capacity,
            self.activation_scale_trtllm_rows
        );
        match kind {
            Glm52TrtllmGroupedFp8Kind::W13 => self.validate_w13(),
            Glm52TrtllmGroupedFp8Kind::W2 => self.validate_w2(),
        }
    }

    fn validate_w13(self) -> Result<()> {
        ensure!(
            self.n == 4096
                && self.k == 6144
                && self.weight_scale_rows == 32
                && self.weight_scale_cols == 48
                && self.activation_scale_cols == 48,
            "GLM5.2 TRTLLM W13 contract drifted: {self:?}"
        );
        Ok(())
    }

    fn validate_w2(self) -> Result<()> {
        ensure!(
            self.n == 6144
                && self.k == 2048
                && self.weight_scale_rows == 48
                && self.weight_scale_cols == 16
                && self.activation_scale_cols == 16,
            "GLM5.2 TRTLLM W2 contract drifted: {self:?}"
        );
        Ok(())
    }
}

pub fn glm52_trtllm_grouped_fp8_contract_validate(
    kind: Glm52TrtllmGroupedFp8Kind,
    contract: Glm52TrtllmGroupedFp8Contract,
) -> Result<()> {
    contract.validate(kind)?;
    let result = unsafe {
        ffi::glm52_trtllm_grouped_fp8_contract_cuda(
            kind.abi(),
            contract.groups as i32,
            contract.m_capacity as i32,
            contract.n as i32,
            contract.k as i32,
            contract.weight_scale_rows as i32,
            contract.weight_scale_cols as i32,
            contract.activation_scale_cols as i32,
            contract.activation_scale_trtllm_rows as i32,
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 TRTLLM grouped FP8 ABI contract check failed: {err}"))
}

pub fn glm52_trtllm_grouped_fp8_workspace_size(
    kind: Glm52TrtllmGroupedFp8Kind,
    contract: Glm52TrtllmGroupedFp8Contract,
) -> Result<usize> {
    contract.validate(kind)?;
    let mut workspace_bytes = 0usize;
    let result = unsafe {
        ffi::glm52_trtllm_grouped_fp8_workspace_size_cuda(
            kind.abi(),
            contract.groups as i32,
            contract.m_capacity as i32,
            contract.n as i32,
            contract.k as i32,
            &raw mut workspace_bytes,
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 TRTLLM grouped FP8 workspace query failed: {err}"))?;
    Ok(workspace_bytes)
}

pub fn glm52_trtllm_grouped_fp8_launch(
    ctx: &DeviceContext,
    kind: Glm52TrtllmGroupedFp8Kind,
    contract: Glm52TrtllmGroupedFp8Contract,
    activation: &CudaSlice<u8>,
    activation_scale_trtllm: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    expert_offsets: &CudaSlice<i64>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    validate_launch_buffers(
        kind,
        contract,
        activation,
        activation_scale_trtllm,
        weight,
        weight_scale,
        expert_offsets,
        output,
    )?;
    let workspace_bytes = glm52_trtllm_grouped_fp8_workspace_size(kind, contract)?;
    ensure!(
        workspace_bytes == 0,
        "GLM5.2 TRTLLM grouped FP8 unexpected workspace requirement: {workspace_bytes} bytes"
    );

    let (activation_ptr, _activation_guard) = activation.device_ptr(&ctx.stream);
    let (activation_scale_ptr, _activation_scale_guard) =
        activation_scale_trtllm.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (weight_scale_ptr, _weight_scale_guard) = weight_scale.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = expert_offsets.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_trtllm_grouped_fp8_launch_cuda(
            kind.abi(),
            activation_ptr as *const u8,
            activation_scale_ptr as *const f32,
            weight_ptr as *const u8,
            weight_scale_ptr as *const f32,
            offsets_ptr as *const i64,
            output_ptr as *mut ffi::Half,
            std::ptr::null_mut(),
            0,
            contract.groups as i32,
            contract.m_capacity as i32,
            contract.n as i32,
            contract.k as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 TRTLLM grouped FP8 launch failed: {err}"))
}

fn validate_launch_buffers(
    kind: Glm52TrtllmGroupedFp8Kind,
    contract: Glm52TrtllmGroupedFp8Contract,
    activation: &CudaSlice<u8>,
    activation_scale_trtllm: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    expert_offsets: &CudaSlice<i64>,
    output: &CudaSlice<bf16>,
) -> Result<()> {
    contract.validate(kind)?;
    let activation_len = contract.m_capacity * contract.k;
    ensure!(
        activation.len() >= activation_len,
        "GLM5.2 TRTLLM grouped FP8 activation buffer too small: have {}, need {activation_len}",
        activation.len()
    );
    let activation_scale_len =
        contract.activation_scale_trtllm_rows * contract.activation_scale_cols;
    ensure!(
        activation_scale_trtllm.len() >= activation_scale_len,
        "GLM5.2 TRTLLM grouped FP8 activation scale buffer too small: have {}, need {activation_scale_len}",
        activation_scale_trtllm.len()
    );
    let weight_len = contract.groups * contract.n * contract.k;
    ensure!(
        weight.len() >= weight_len,
        "GLM5.2 TRTLLM grouped FP8 weight buffer too small: have {}, need {weight_len}",
        weight.len()
    );
    let weight_scale_len =
        contract.groups * contract.weight_scale_rows * contract.weight_scale_cols;
    ensure!(
        weight_scale.len() >= weight_scale_len,
        "GLM5.2 TRTLLM grouped FP8 weight scale buffer too small: have {}, need {weight_scale_len}",
        weight_scale.len()
    );
    ensure!(
        expert_offsets.len() > contract.groups,
        "GLM5.2 TRTLLM grouped FP8 expert_offsets buffer too small: have {}, need {}",
        expert_offsets.len(),
        contract.groups + 1
    );
    let output_len = contract.m_capacity * contract.n;
    ensure!(
        output.len() >= output_len,
        "GLM5.2 TRTLLM grouped FP8 output buffer too small: have {}, need {output_len}",
        output.len()
    );
    Ok(())
}
