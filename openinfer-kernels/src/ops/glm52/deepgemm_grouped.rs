use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_DEEPGEMM_GROUPED_W13_KIND: i32 = 1;
pub const GLM52_DEEPGEMM_GROUPED_W2_KIND: i32 = 2;
/// Per-expert row alignment of the grouped layout (a fixed design constant; the
/// group count and m_capacity are runtime — PP8 EP1 owns all 256 experts and
/// bs=1 decode sizes m_capacity to top_k*alignment).
pub const GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm52DeepGemmGroupedFp8Kind {
    W13,
    W2,
}

impl Glm52DeepGemmGroupedFp8Kind {
    fn abi(self) -> i32 {
        match self {
            Self::W13 => GLM52_DEEPGEMM_GROUPED_W13_KIND,
            Self::W2 => GLM52_DEEPGEMM_GROUPED_W2_KIND,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52DeepGemmGroupedFp8Contract {
    pub groups: usize,
    pub m_capacity: usize,
    pub n: usize,
    pub k: usize,
    pub weight_scale_rows: usize,
    pub weight_scale_cols: usize,
    pub activation_scale_cols: usize,
    pub activation_scale_tma_rows: usize,
    pub psum_entries: usize,
    pub expert_alignment: usize,
}

impl Glm52DeepGemmGroupedFp8Contract {
    pub fn validate(self, kind: Glm52DeepGemmGroupedFp8Kind) -> Result<()> {
        ensure!(
            self.groups > 0 && self.psum_entries == self.groups,
            "GLM5.2 DeepGEMM grouped FP8 needs groups>0 with one psum entry per group, got groups={}, psum_entries={}",
            self.groups,
            self.psum_entries
        );
        ensure!(
            self.m_capacity > 0 && self.activation_scale_tma_rows == self.m_capacity,
            "GLM5.2 DeepGEMM grouped FP8 needs m_capacity>0 == activation scale rows, got m_capacity={}, activation_scale_tma_rows={}",
            self.m_capacity,
            self.activation_scale_tma_rows
        );
        ensure!(
            self.expert_alignment == GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT,
            "GLM5.2 DeepGEMM grouped FP8 expert alignment must be {}, got {}",
            GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT,
            self.expert_alignment
        );
        match kind {
            Glm52DeepGemmGroupedFp8Kind::W13 => self.validate_w13(),
            Glm52DeepGemmGroupedFp8Kind::W2 => self.validate_w2(),
        }
    }

    fn validate_w13(self) -> Result<()> {
        ensure!(
            self.n == 4096
                && self.k == 6144
                && self.weight_scale_rows == 32
                && self.weight_scale_cols == 48
                && self.activation_scale_cols == 48,
            "GLM5.2 DeepGEMM W13 contract drifted: {self:?}"
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
            "GLM5.2 DeepGEMM W2 contract drifted: {self:?}"
        );
        Ok(())
    }
}

pub fn glm52_deepgemm_grouped_fp8_contract_validate(
    kind: Glm52DeepGemmGroupedFp8Kind,
    contract: Glm52DeepGemmGroupedFp8Contract,
) -> Result<()> {
    contract.validate(kind)?;
    let result = unsafe {
        ffi::glm52_deepgemm_grouped_fp8_contract_cuda(
            kind.abi(),
            contract.groups as i32,
            contract.m_capacity as i32,
            contract.n as i32,
            contract.k as i32,
            contract.weight_scale_rows as i32,
            contract.weight_scale_cols as i32,
            contract.activation_scale_cols as i32,
            contract.activation_scale_tma_rows as i32,
            contract.psum_entries as i32,
            contract.expert_alignment as i32,
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 DeepGEMM grouped FP8 ABI contract check failed: {err}"))
}

pub fn glm52_deepgemm_grouped_fp8_metadata_launch(
    ctx: &DeviceContext,
    groups: usize,
    m_capacity: usize,
    psum_expert: &CudaSlice<i32>,
    expert_offsets: &mut CudaSlice<i64>,
    w13_problem_sizes: &mut CudaSlice<i32>,
    w2_problem_sizes: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        groups > 0 && m_capacity > 0,
        "GLM5.2 DeepGEMM grouped FP8 metadata needs groups>0 and m_capacity>0, got groups={groups}, m_capacity={m_capacity}"
    );
    ensure!(
        psum_expert.len() >= groups
            && expert_offsets.len() > groups
            && w13_problem_sizes.len() >= groups * 3
            && w2_problem_sizes.len() >= groups * 3,
        "GLM5.2 DeepGEMM grouped FP8 metadata buffers too small for {groups} groups: psum={}, offsets={}, w13={}, w2={}",
        psum_expert.len(),
        expert_offsets.len(),
        w13_problem_sizes.len(),
        w2_problem_sizes.len()
    );
    let (psum_ptr, _psum_guard) = psum_expert.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = expert_offsets.device_ptr_mut(&ctx.stream);
    let (w13_ptr, _w13_guard) = w13_problem_sizes.device_ptr_mut(&ctx.stream);
    let (w2_ptr, _w2_guard) = w2_problem_sizes.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_deepgemm_grouped_fp8_metadata_cuda(
            psum_ptr as *const i32,
            offsets_ptr as *mut i64,
            w13_ptr as *mut i32,
            w2_ptr as *mut i32,
            groups as i32,
            m_capacity as i32,
            GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 DeepGEMM grouped FP8 metadata launch failed: {err}"))
}

pub fn glm52_deepgemm_grouped_fp8_launch(
    ctx: &DeviceContext,
    kind: Glm52DeepGemmGroupedFp8Kind,
    contract: Glm52DeepGemmGroupedFp8Contract,
    activation: &CudaSlice<u8>,
    activation_scale_tma: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    psum_expert: &CudaSlice<i32>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    validate_launch_buffers(
        kind,
        contract,
        activation,
        activation_scale_tma,
        weight,
        weight_scale,
        psum_expert,
        output,
    )?;
    let (activation_ptr, _activation_guard) = activation.device_ptr(&ctx.stream);
    let (activation_scale_ptr, _activation_scale_guard) =
        activation_scale_tma.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (weight_scale_ptr, _weight_scale_guard) = weight_scale.device_ptr(&ctx.stream);
    let (psum_ptr, _psum_guard) = psum_expert.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_deepgemm_grouped_fp8_launch_cuda(
            kind.abi(),
            activation_ptr as *const u8,
            activation_scale_ptr as *const f32,
            weight_ptr as *const u8,
            weight_scale_ptr as *const f32,
            psum_ptr as *const i32,
            output_ptr as *mut ffi::Half,
            contract.groups as i32,
            contract.m_capacity as i32,
            contract.n as i32,
            contract.k as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 DeepGEMM grouped FP8 launch failed: {err}"))
}

fn validate_launch_buffers(
    kind: Glm52DeepGemmGroupedFp8Kind,
    contract: Glm52DeepGemmGroupedFp8Contract,
    activation: &CudaSlice<u8>,
    activation_scale_tma: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    psum_expert: &CudaSlice<i32>,
    output: &CudaSlice<bf16>,
) -> Result<()> {
    contract.validate(kind)?;
    let activation_len = contract.m_capacity * contract.k;
    ensure!(
        activation.len() >= activation_len,
        "GLM5.2 DeepGEMM grouped FP8 activation buffer too small: have {}, need {activation_len}",
        activation.len()
    );
    let activation_scale_len = contract.activation_scale_tma_rows * contract.activation_scale_cols;
    ensure!(
        activation_scale_tma.len() >= activation_scale_len,
        "GLM5.2 DeepGEMM grouped FP8 activation scale buffer too small: have {}, need {activation_scale_len}",
        activation_scale_tma.len()
    );
    let weight_len = contract.groups * contract.n * contract.k;
    ensure!(
        weight.len() >= weight_len,
        "GLM5.2 DeepGEMM grouped FP8 weight buffer too small: have {}, need {weight_len}",
        weight.len()
    );
    let weight_scale_len =
        contract.groups * contract.weight_scale_rows * contract.weight_scale_cols;
    ensure!(
        weight_scale.len() >= weight_scale_len,
        "GLM5.2 DeepGEMM grouped FP8 weight scale buffer too small: have {}, need {weight_scale_len}",
        weight_scale.len()
    );
    ensure!(
        psum_expert.len() >= contract.psum_entries,
        "GLM5.2 DeepGEMM grouped FP8 psum_expert buffer too small: have {}, need {}",
        psum_expert.len(),
        contract.psum_entries
    );
    let output_len = contract.m_capacity * contract.n;
    ensure!(
        output.len() >= output_len,
        "GLM5.2 DeepGEMM grouped FP8 output buffer too small: have {}, need {output_len}",
        output.len()
    );
    Ok(())
}
