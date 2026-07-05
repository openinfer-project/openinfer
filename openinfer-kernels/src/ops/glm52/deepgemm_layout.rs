use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_TRTLLM_GROUPED_OFFSET_ALIGNMENT: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52TrtllmGroupedOffsetScaleLayout {
    pub m_capacity: usize,
    pub scale_cols: usize,
    pub groups: usize,
    pub padded_rows: usize,
}

impl Glm52TrtllmGroupedOffsetScaleLayout {
    pub fn f32(m_capacity: usize, scale_cols: usize, groups: usize) -> Self {
        Self {
            m_capacity,
            scale_cols,
            groups,
            padded_rows: glm52_trtllm_grouped_offset_padded_rows(m_capacity, groups),
        }
    }

    pub fn input_len(self) -> Result<usize> {
        self.validate()?;
        Ok(self.m_capacity * self.scale_cols)
    }

    pub fn output_len(self) -> Result<usize> {
        self.validate()?;
        Ok(self.padded_rows * self.scale_cols)
    }

    pub fn offsets_len(self) -> Result<usize> {
        self.validate()?;
        Ok(self.groups + 1)
    }

    pub fn validate(self) -> Result<()> {
        ensure!(
            self.m_capacity > 0,
            "GLM5.2 TRTLLM grouped-offset scale layout m_capacity must be positive"
        );
        ensure!(
            self.scale_cols > 0,
            "GLM5.2 TRTLLM grouped-offset scale layout scale_cols must be positive"
        );
        ensure!(
            self.groups > 0,
            "GLM5.2 TRTLLM grouped-offset scale layout groups must be positive"
        );
        let expected = glm52_trtllm_grouped_offset_padded_rows(self.m_capacity, self.groups);
        ensure!(
            self.padded_rows == expected,
            "GLM5.2 TRTLLM grouped-offset scale layout padded_rows must be {expected}, got {}",
            self.padded_rows
        );
        Ok(())
    }
}

pub fn glm52_trtllm_grouped_offset_padded_rows(rows: usize, groups: usize) -> usize {
    (rows + groups * (GLM52_TRTLLM_GROUPED_OFFSET_ALIGNMENT - 1))
        / GLM52_TRTLLM_GROUPED_OFFSET_ALIGNMENT
        * GLM52_TRTLLM_GROUPED_OFFSET_ALIGNMENT
}

pub fn glm52_deepgemm_grouped_offset_tma_aligned_f32_launch(
    ctx: &DeviceContext,
    layout: Glm52TrtllmGroupedOffsetScaleLayout,
    input: &CudaSlice<f32>,
    expert_offsets: &CudaSlice<i64>,
    output: &mut CudaSlice<f32>,
) -> Result<()> {
    validate_grouped_offset_scale_layout_buffers(layout, input, expert_offsets, output)?;
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = expert_offsets.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_deepgemm_grouped_offset_tma_aligned_f32_cuda(
            input_ptr as *const f32,
            offsets_ptr as *const i64,
            output_ptr as *mut f32,
            layout.m_capacity as i32,
            layout.scale_cols as i32,
            layout.groups as i32,
            layout.padded_rows as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 TRTLLM grouped-offset scale layout launch failed: {err}"))
}

fn validate_grouped_offset_scale_layout_buffers(
    layout: Glm52TrtllmGroupedOffsetScaleLayout,
    input: &CudaSlice<f32>,
    expert_offsets: &CudaSlice<i64>,
    output: &CudaSlice<f32>,
) -> Result<()> {
    layout.validate()?;
    let input_len = layout.input_len()?;
    ensure!(
        input.len() >= input_len,
        "GLM5.2 TRTLLM grouped-offset scale layout input too small: have {}, need {input_len}",
        input.len()
    );
    let offsets_len = layout.offsets_len()?;
    ensure!(
        expert_offsets.len() >= offsets_len,
        "GLM5.2 TRTLLM grouped-offset scale layout offsets too small: have {}, need {offsets_len}",
        expert_offsets.len()
    );
    let output_len = layout.output_len()?;
    ensure!(
        output.len() >= output_len,
        "GLM5.2 TRTLLM grouped-offset scale layout output too small: have {}, need {output_len}",
        output.len()
    );
    Ok(())
}
