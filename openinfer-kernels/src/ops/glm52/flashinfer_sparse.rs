use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_FLASHINFER_SPARSE_HEADS: usize = 16;
pub const GLM52_FLASHINFER_SPARSE_QK_HEAD_DIM: usize = 576;
pub const GLM52_FLASHINFER_SPARSE_V_HEAD_DIM: usize = 512;
pub const GLM52_FLASHINFER_SPARSE_PAGE_SIZE: usize = 64;
pub const GLM52_FLASHINFER_SPARSE_BYTES_PER_TOKEN: usize = 576;
pub const GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm52FlashInferSparseDecode {
    pub batch_size: usize,
    pub heads: usize,
    pub num_blocks: usize,
    pub topk: usize,
    pub sm_scale: f32,
}

impl Glm52FlashInferSparseDecode {
    pub fn validate(self) -> Result<()> {
        ensure!(
            matches!(self.batch_size, 1 | 2 | 4 | 8),
            "GLM5.2 FlashInfer sparse batch {} is not a decode bucket",
            self.batch_size
        );
        ensure!(
            self.heads == GLM52_FLASHINFER_SPARSE_HEADS,
            "GLM5.2 FlashInfer sparse requires {} heads, got {}",
            GLM52_FLASHINFER_SPARSE_HEADS,
            self.heads
        );
        ensure!(
            self.num_blocks > 0,
            "GLM5.2 FlashInfer sparse needs cache blocks"
        );
        ensure!(
            matches!(self.topk, 256 | 2048),
            "GLM5.2 FlashInfer sparse topk must be 256 or 2048, got {}",
            self.topk
        );
        ensure!(
            self.sm_scale.is_finite() && self.sm_scale > 0.0,
            "GLM5.2 FlashInfer sparse scale must be finite and positive"
        );
        Ok(())
    }

    pub fn query_len(self) -> usize {
        self.batch_size * self.heads * GLM52_FLASHINFER_SPARSE_QK_HEAD_DIM
    }

    pub fn cache_len(self) -> usize {
        self.num_blocks
            * GLM52_FLASHINFER_SPARSE_PAGE_SIZE
            * GLM52_FLASHINFER_SPARSE_BYTES_PER_TOKEN
    }

    pub fn topk_len(self) -> usize {
        self.batch_size * self.topk
    }

    pub fn output_len(self) -> usize {
        self.batch_size * self.heads * GLM52_FLASHINFER_SPARSE_V_HEAD_DIM
    }
}

pub fn glm52_flashinfer_sparse_mla_supported(heads: usize) -> Result<bool> {
    let mut supported = 0;
    let result = unsafe {
        ffi::glm52_flashinfer_sparse_mla_supported_cuda(heads as i32, &raw mut supported)
    };
    result.result()?;
    Ok(supported != 0)
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_flashinfer_sparse_mla_fp8_launch(
    ctx: &DeviceContext,
    contract: Glm52FlashInferSparseDecode,
    query: &CudaSlice<u8>,
    cache: &CudaSlice<u8>,
    topk_indices: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    contract.validate()?;
    ensure!(
        query.len() >= contract.query_len(),
        "GLM5.2 FlashInfer query too small"
    );
    ensure!(
        cache.len() >= contract.cache_len(),
        "GLM5.2 FlashInfer cache too small"
    );
    ensure!(
        topk_indices.len() >= contract.topk_len(),
        "GLM5.2 FlashInfer sparse indices too small"
    );
    ensure!(
        seq_lens.len() >= contract.batch_size,
        "GLM5.2 FlashInfer sequence lengths too small"
    );
    ensure!(
        out.len() >= contract.output_len(),
        "GLM5.2 FlashInfer output too small"
    );
    ensure!(
        workspace.len() >= GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES,
        "GLM5.2 FlashInfer workspace too small: have {}, need {}",
        workspace.len(),
        GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES
    );

    let workspace_bytes = workspace.len();
    let (query_ptr, _g0) = query.device_ptr(&ctx.stream);
    let (cache_ptr, _g1) = cache.device_ptr(&ctx.stream);
    let (indices_ptr, _g2) = topk_indices.device_ptr(&ctx.stream);
    let (seq_lens_ptr, _g3) = seq_lens.device_ptr(&ctx.stream);
    let (out_ptr, _g4) = out.device_ptr_mut(&ctx.stream);
    let (workspace_ptr, _g5) = workspace.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_flashinfer_sparse_mla_fp8_cuda(
            query_ptr as *const u8,
            cache_ptr as *const u8,
            indices_ptr as *const i32,
            seq_lens_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            workspace_ptr as *mut u8,
            workspace_bytes,
            contract.batch_size as i32,
            contract.heads as i32,
            contract.num_blocks as i32,
            contract.topk as i32,
            contract.sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    ensure!(
        result == 0,
        "GLM5.2 FlashInfer sparse MLA launch failed with error {result}{}",
        crate::ops::ffi_exception_message(result)
    );
    Ok(())
}
