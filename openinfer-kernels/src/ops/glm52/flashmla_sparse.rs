use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_FLASHMLA_SPARSE_BATCH_CAPACITY: usize = 128;
pub const GLM52_FLASHMLA_SPARSE_HEADS: usize = 64;
pub const GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM: usize = 576;
pub const GLM52_FLASHMLA_SPARSE_V_HEAD_DIM: usize = 512;
pub const GLM52_FLASHMLA_SPARSE_PAGE_SIZE: usize = 64;
pub const GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN: usize = 656;
pub const GLM52_FLASHMLA_SPARSE_TOPK: usize = 2048;
pub const GLM52_FLASHMLA_SPARSE_SCHED_META_INTS: usize = 8;
pub const GLM52_FLASHMLA_SPARSE_MAX_SM_PARTS: usize = 160;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm52FlashMlaSparseDecode {
    pub batch_size: usize,
    pub num_blocks: usize,
    pub topk: usize,
    pub num_sm_parts: usize,
    pub sm_scale: f32,
}

impl Glm52FlashMlaSparseDecode {
    pub fn validate(self) -> Result<()> {
        ensure!(
            (1..=GLM52_FLASHMLA_SPARSE_BATCH_CAPACITY).contains(&self.batch_size),
            "GLM5.2 FlashMLA sparse decode batch_size {} out of 1..={}",
            self.batch_size,
            GLM52_FLASHMLA_SPARSE_BATCH_CAPACITY
        );
        ensure!(
            self.num_blocks > 0,
            "GLM5.2 FlashMLA sparse decode num_blocks must be positive"
        );
        ensure!(
            self.topk == GLM52_FLASHMLA_SPARSE_TOPK,
            "GLM5.2 FlashMLA sparse decode topk must be {}, got {}",
            GLM52_FLASHMLA_SPARSE_TOPK,
            self.topk
        );
        ensure!(
            (1..=GLM52_FLASHMLA_SPARSE_MAX_SM_PARTS).contains(&self.num_sm_parts),
            "GLM5.2 FlashMLA sparse decode num_sm_parts {} out of 1..={}",
            self.num_sm_parts,
            GLM52_FLASHMLA_SPARSE_MAX_SM_PARTS
        );
        ensure!(
            self.sm_scale.is_finite() && self.sm_scale > 0.0,
            "GLM5.2 FlashMLA sparse decode sm_scale must be finite and positive"
        );
        Ok(())
    }

    pub fn q_len(self) -> usize {
        self.batch_size * GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM
    }

    pub fn packed_kv_cache_len(self) -> usize {
        self.num_blocks * GLM52_FLASHMLA_SPARSE_PAGE_SIZE * GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN
    }

    pub fn topk_indices_len(self) -> usize {
        self.batch_size * self.topk
    }

    pub fn tile_scheduler_metadata_len(self) -> usize {
        self.num_sm_parts * GLM52_FLASHMLA_SPARSE_SCHED_META_INTS
    }

    pub fn num_splits_len(self) -> usize {
        self.batch_size + 1
    }

    pub fn lse_len(self) -> usize {
        self.batch_size * GLM52_FLASHMLA_SPARSE_HEADS
    }

    pub fn latent_len(self) -> usize {
        self.batch_size * GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_V_HEAD_DIM
    }

    pub fn split_count(self) -> usize {
        self.batch_size + self.num_sm_parts
    }

    pub fn lse_accum_len(self) -> usize {
        self.split_count() * GLM52_FLASHMLA_SPARSE_HEADS
    }

    pub fn o_accum_len(self) -> usize {
        self.split_count() * GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_V_HEAD_DIM
    }
}

pub fn glm52_flashmla_sparse_decode_num_sm_parts() -> Result<usize> {
    let mut num_sm_parts = 0i32;
    let result = unsafe { ffi::glm52_flashmla_sparse_decode_num_sm_parts_cuda(&mut num_sm_parts) };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FlashMLA sparse num_sm_parts query failed: {err}"))?;
    ensure!(
        (1..=GLM52_FLASHMLA_SPARSE_MAX_SM_PARTS).contains(&(num_sm_parts as usize)),
        "GLM5.2 FlashMLA sparse num_sm_parts query returned {num_sm_parts}; supported range is 1..={}",
        GLM52_FLASHMLA_SPARSE_MAX_SM_PARTS
    );
    Ok(num_sm_parts as usize)
}

pub fn glm52_flashmla_sparse_decode_metadata_launch(
    ctx: &DeviceContext,
    batch_size: usize,
    num_sm_parts: usize,
    tile_scheduler_metadata: &mut CudaSlice<i32>,
    num_splits: &mut CudaSlice<i32>,
) -> Result<()> {
    let contract = Glm52FlashMlaSparseDecode {
        batch_size,
        num_blocks: 1,
        topk: GLM52_FLASHMLA_SPARSE_TOPK,
        num_sm_parts,
        sm_scale: 1.0,
    };
    contract.validate()?;
    validate_metadata_buffers(contract, tile_scheduler_metadata, num_splits)?;

    let (sched_ptr, _sched_guard) = tile_scheduler_metadata.device_ptr_mut(&ctx.stream);
    let (splits_ptr, _splits_guard) = num_splits.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_flashmla_sparse_decode_metadata_cuda(
            sched_ptr as *mut i32,
            splits_ptr as *mut i32,
            batch_size as i32,
            GLM52_FLASHMLA_SPARSE_TOPK as i32,
            num_sm_parts as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FlashMLA sparse metadata launch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_flashmla_sparse_decode_launch(
    ctx: &DeviceContext,
    contract: Glm52FlashMlaSparseDecode,
    q: &CudaSlice<bf16>,
    packed_kv_cache: &CudaSlice<u8>,
    topk_indices: &CudaSlice<i32>,
    tile_scheduler_metadata: &CudaSlice<i32>,
    num_splits: &CudaSlice<i32>,
    out_latent: &mut CudaSlice<bf16>,
    lse: &mut CudaSlice<f32>,
    lse_accum: &mut CudaSlice<f32>,
    o_accum: &mut CudaSlice<f32>,
) -> Result<()> {
    validate_decode_buffers(
        contract,
        q,
        packed_kv_cache,
        topk_indices,
        tile_scheduler_metadata,
        num_splits,
        out_latent,
        lse,
        lse_accum,
        o_accum,
    )?;

    let (q_ptr, _q_guard) = q.device_ptr(&ctx.stream);
    let (kv_ptr, _kv_guard) = packed_kv_cache.device_ptr(&ctx.stream);
    let (indices_ptr, _indices_guard) = topk_indices.device_ptr(&ctx.stream);
    let (sched_ptr, _sched_guard) = tile_scheduler_metadata.device_ptr(&ctx.stream);
    let (splits_ptr, _splits_guard) = num_splits.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out_latent.device_ptr_mut(&ctx.stream);
    let (lse_ptr, _lse_guard) = lse.device_ptr_mut(&ctx.stream);
    let (lse_accum_ptr, _lse_accum_guard) = lse_accum.device_ptr_mut(&ctx.stream);
    let (o_accum_ptr, _o_accum_guard) = o_accum.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_flashmla_sparse_decode_launch_cuda(
            q_ptr as *const ffi::Half,
            kv_ptr as *const u8,
            indices_ptr as *const i32,
            sched_ptr as *const i32,
            splits_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            lse_ptr as *mut f32,
            lse_accum_ptr as *mut f32,
            o_accum_ptr as *mut f32,
            contract.batch_size as i32,
            contract.num_blocks as i32,
            contract.topk as i32,
            contract.num_sm_parts as i32,
            contract.sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FlashMLA sparse decode launch failed: {err}"))
}

fn validate_metadata_buffers(
    contract: Glm52FlashMlaSparseDecode,
    tile_scheduler_metadata: &CudaSlice<i32>,
    num_splits: &CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        tile_scheduler_metadata.len() >= contract.tile_scheduler_metadata_len(),
        "GLM5.2 FlashMLA sparse scheduler metadata too small: have {}, need {}",
        tile_scheduler_metadata.len(),
        contract.tile_scheduler_metadata_len()
    );
    ensure!(
        num_splits.len() >= contract.num_splits_len(),
        "GLM5.2 FlashMLA sparse num_splits too small: have {}, need {}",
        num_splits.len(),
        contract.num_splits_len()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_decode_buffers(
    contract: Glm52FlashMlaSparseDecode,
    q: &CudaSlice<bf16>,
    packed_kv_cache: &CudaSlice<u8>,
    topk_indices: &CudaSlice<i32>,
    tile_scheduler_metadata: &CudaSlice<i32>,
    num_splits: &CudaSlice<i32>,
    out_latent: &CudaSlice<bf16>,
    lse: &CudaSlice<f32>,
    lse_accum: &CudaSlice<f32>,
    o_accum: &CudaSlice<f32>,
) -> Result<()> {
    contract.validate()?;
    ensure!(
        q.len() >= contract.q_len(),
        "GLM5.2 FlashMLA sparse q too small: have {}, need {}",
        q.len(),
        contract.q_len()
    );
    ensure!(
        packed_kv_cache.len() >= contract.packed_kv_cache_len(),
        "GLM5.2 FlashMLA sparse packed kv cache too small: have {}, need {}",
        packed_kv_cache.len(),
        contract.packed_kv_cache_len()
    );
    ensure!(
        topk_indices.len() >= contract.topk_indices_len(),
        "GLM5.2 FlashMLA sparse topk_indices too small: have {}, need {}",
        topk_indices.len(),
        contract.topk_indices_len()
    );
    validate_metadata_buffers(contract, tile_scheduler_metadata, num_splits)?;
    ensure!(
        out_latent.len() >= contract.latent_len(),
        "GLM5.2 FlashMLA sparse latent output too small: have {}, need {}",
        out_latent.len(),
        contract.latent_len()
    );
    ensure!(
        lse.len() >= contract.lse_len(),
        "GLM5.2 FlashMLA sparse lse too small: have {}, need {}",
        lse.len(),
        contract.lse_len()
    );
    ensure!(
        lse_accum.len() >= contract.lse_accum_len(),
        "GLM5.2 FlashMLA sparse lse_accum too small: have {}, need {}",
        lse_accum.len(),
        contract.lse_accum_len()
    );
    ensure!(
        o_accum.len() >= contract.o_accum_len(),
        "GLM5.2 FlashMLA sparse o_accum too small: have {}, need {}",
        o_accum.len(),
        contract.o_accum_len()
    );
    Ok(())
}
