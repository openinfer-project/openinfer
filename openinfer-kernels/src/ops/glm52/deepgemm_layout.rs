use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_DEEPGEMM_TMA_ALIGNMENT_BYTES: usize = 16;
pub const GLM52_DEEPGEMM_F32_TMA_ROW_ALIGNMENT: usize =
    GLM52_DEEPGEMM_TMA_ALIGNMENT_BYTES / std::mem::size_of::<f32>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52DeepGemmScaleLayout {
    pub rows: usize,
    pub scale_cols: usize,
    pub aligned_rows: usize,
}

impl Glm52DeepGemmScaleLayout {
    pub fn f32(rows: usize, scale_cols: usize) -> Self {
        Self {
            rows,
            scale_cols,
            aligned_rows: glm52_deepgemm_tma_aligned_rows(rows),
        }
    }

    pub fn output_len(self) -> Result<usize> {
        self.validate()?;
        Ok(self.aligned_rows * self.scale_cols)
    }

    pub fn input_len(self) -> Result<usize> {
        self.validate()?;
        Ok(self.rows * self.scale_cols)
    }

    pub fn validate(self) -> Result<()> {
        ensure!(
            self.rows > 0,
            "GLM5.2 DeepGEMM scale layout rows must be positive"
        );
        ensure!(
            self.scale_cols > 0,
            "GLM5.2 DeepGEMM scale layout scale_cols must be positive"
        );
        let expected = glm52_deepgemm_tma_aligned_rows(self.rows);
        ensure!(
            self.aligned_rows == expected,
            "GLM5.2 DeepGEMM scale layout aligned_rows must be {expected}, got {}",
            self.aligned_rows
        );
        Ok(())
    }
}

pub fn glm52_deepgemm_tma_aligned_rows(rows: usize) -> usize {
    rows.div_ceil(GLM52_DEEPGEMM_F32_TMA_ROW_ALIGNMENT) * GLM52_DEEPGEMM_F32_TMA_ROW_ALIGNMENT
}

pub fn glm52_deepgemm_mn_major_tma_aligned_f32_launch(
    ctx: &DeviceContext,
    layout: Glm52DeepGemmScaleLayout,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
) -> Result<()> {
    validate_scale_layout_buffers(layout, input, output)?;
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_deepgemm_mn_major_tma_aligned_f32_cuda(
            input_ptr as *const f32,
            output_ptr as *mut f32,
            layout.rows as i32,
            layout.scale_cols as i32,
            layout.aligned_rows as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 DeepGEMM scale layout launch failed: {err}"))
}

fn validate_scale_layout_buffers(
    layout: Glm52DeepGemmScaleLayout,
    input: &CudaSlice<f32>,
    output: &CudaSlice<f32>,
) -> Result<()> {
    layout.validate()?;
    let input_len = layout.input_len()?;
    ensure!(
        input.len() >= input_len,
        "GLM5.2 DeepGEMM scale layout input too small: have {}, need {input_len}",
        input.len()
    );
    let output_len = layout.output_len()?;
    ensure!(
        output.len() >= output_len,
        "GLM5.2 DeepGEMM scale layout output too small: have {}, need {output_len}",
        output.len()
    );
    Ok(())
}
