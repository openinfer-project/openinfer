use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_INDEXER_HEAD_DIM: usize = 128;
pub const GLM52_INDEXER_QUANT_BLOCK_SIZE: usize = 128;
pub const GLM52_INDEXER_SCALE_BYTES_PER_TOKEN: usize = 4;
pub const GLM52_INDEXER_TOPK: usize = 2048;
pub const GLM52_INDEXER_TOPK_WORKSPACE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52IndexerCacheLayout {
    pub cache_blocks: usize,
    pub cache_block_size: usize,
    pub cache_block_stride_bytes: usize,
}

impl Glm52IndexerCacheLayout {
    pub fn min_block_stride_bytes(self) -> usize {
        self.cache_block_size * (GLM52_INDEXER_HEAD_DIM + GLM52_INDEXER_SCALE_BYTES_PER_TOKEN)
    }

    pub fn validate(self) -> Result<()> {
        ensure!(
            self.cache_blocks > 0,
            "GLM5.2 indexer cache_blocks must be positive"
        );
        ensure!(
            self.cache_block_size > 0,
            "GLM5.2 indexer cache_block_size must be positive"
        );
        ensure!(
            self.cache_block_stride_bytes >= self.min_block_stride_bytes(),
            "GLM5.2 indexer cache block stride too small: have {} bytes, need at least {}",
            self.cache_block_stride_bytes,
            self.min_block_stride_bytes()
        );
        Ok(())
    }

    pub fn min_cache_bytes(self) -> Result<usize> {
        self.validate()?;
        self.cache_blocks
            .checked_mul(self.cache_block_stride_bytes)
            .ok_or_else(|| anyhow!("GLM5.2 indexer cache byte size overflow: {self:?}"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm52IndexerScaleFormat {
    F32,
    Ue8m0RoundedF32,
}

impl Glm52IndexerScaleFormat {
    fn as_ffi(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::Ue8m0RoundedF32 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52IndexerCacheInsert {
    pub tokens: usize,
    pub layout: Glm52IndexerCacheLayout,
    pub scale_format: Glm52IndexerScaleFormat,
}

impl Glm52IndexerCacheInsert {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.tokens > 0,
            "GLM5.2 indexer cache insert tokens must be positive"
        );
        self.layout.validate()
    }
}

pub fn glm52_indexer_k_quant_and_cache_launch(
    ctx: &DeviceContext,
    contract: Glm52IndexerCacheInsert,
    k: &CudaSlice<bf16>,
    indexer_cache: &mut CudaSlice<u8>,
    slot_mapping: &CudaSlice<i64>,
) -> Result<()> {
    contract.validate()?;
    ensure!(
        k.len() >= contract.tokens * GLM52_INDEXER_HEAD_DIM,
        "GLM5.2 indexer K buffer too small: have {}, need {}",
        k.len(),
        contract.tokens * GLM52_INDEXER_HEAD_DIM
    );
    ensure!(
        slot_mapping.len() >= contract.tokens,
        "GLM5.2 indexer slot_mapping too small: have {}, need {}",
        slot_mapping.len(),
        contract.tokens
    );
    let min_cache_bytes = contract.layout.min_cache_bytes()?;
    ensure!(
        indexer_cache.len() >= min_cache_bytes,
        "GLM5.2 indexer cache buffer too small: have {}, need {}",
        indexer_cache.len(),
        min_cache_bytes
    );

    let (k_ptr, _k_guard) = k.device_ptr(&ctx.stream);
    let (cache_ptr, _cache_guard) = indexer_cache.device_ptr_mut(&ctx.stream);
    let (slot_ptr, _slot_guard) = slot_mapping.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_k_quant_and_cache_cuda(
            k_ptr as *const ffi::Half,
            cache_ptr as *mut u8,
            slot_ptr as *const i64,
            contract.tokens as i32,
            GLM52_INDEXER_HEAD_DIM as i32,
            GLM52_INDEXER_QUANT_BLOCK_SIZE as i32,
            contract.layout.cache_block_size as i32,
            contract.layout.cache_block_stride_bytes as i64,
            contract.scale_format.as_ffi(),
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer K quant/cache launch failed: {err}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52IndexerCacheGather {
    pub tokens: usize,
    pub batch_size: usize,
    pub num_blocks_per_seq: usize,
    pub layout: Glm52IndexerCacheLayout,
}

impl Glm52IndexerCacheGather {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.tokens > 0,
            "GLM5.2 indexer gather tokens must be positive"
        );
        ensure!(
            self.batch_size > 0,
            "GLM5.2 indexer gather batch_size must be positive"
        );
        ensure!(
            self.num_blocks_per_seq > 0,
            "GLM5.2 indexer gather num_blocks_per_seq must be positive"
        );
        self.layout.validate()
    }
}

pub fn glm52_indexer_k_gather_quant_cache_launch(
    ctx: &DeviceContext,
    contract: Glm52IndexerCacheGather,
    indexer_cache: &CudaSlice<u8>,
    dst_k: &mut CudaSlice<u8>,
    dst_scale: &mut CudaSlice<u8>,
    block_table: &CudaSlice<i32>,
    cu_seq_lens: &CudaSlice<i32>,
) -> Result<()> {
    contract.validate()?;
    let min_cache_bytes = contract.layout.min_cache_bytes()?;
    ensure!(
        indexer_cache.len() >= min_cache_bytes,
        "GLM5.2 indexer cache buffer too small: have {}, need {}",
        indexer_cache.len(),
        min_cache_bytes
    );
    ensure!(
        dst_k.len() >= contract.tokens * GLM52_INDEXER_HEAD_DIM,
        "GLM5.2 indexer gather dst_k too small: have {}, need {}",
        dst_k.len(),
        contract.tokens * GLM52_INDEXER_HEAD_DIM
    );
    ensure!(
        dst_scale.len() >= contract.tokens * GLM52_INDEXER_SCALE_BYTES_PER_TOKEN,
        "GLM5.2 indexer gather dst_scale too small: have {}, need {}",
        dst_scale.len(),
        contract.tokens * GLM52_INDEXER_SCALE_BYTES_PER_TOKEN
    );
    ensure!(
        block_table.len() >= contract.batch_size * contract.num_blocks_per_seq,
        "GLM5.2 indexer gather block_table too small: have {}, need {}",
        block_table.len(),
        contract.batch_size * contract.num_blocks_per_seq
    );
    ensure!(
        cu_seq_lens.len() >= contract.batch_size + 1,
        "GLM5.2 indexer gather cu_seq_lens too small: have {}, need {}",
        cu_seq_lens.len(),
        contract.batch_size + 1
    );

    let (cache_ptr, _cache_guard) = indexer_cache.device_ptr(&ctx.stream);
    let (dst_k_ptr, _dst_k_guard) = dst_k.device_ptr_mut(&ctx.stream);
    let (dst_scale_ptr, _dst_scale_guard) = dst_scale.device_ptr_mut(&ctx.stream);
    let (block_table_ptr, _block_table_guard) = block_table.device_ptr(&ctx.stream);
    let (cu_seq_lens_ptr, _cu_seq_lens_guard) = cu_seq_lens.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_k_gather_quant_cache_cuda(
            cache_ptr as *const u8,
            dst_k_ptr as *mut u8,
            dst_scale_ptr as *mut u8,
            block_table_ptr as *const i32,
            cu_seq_lens_ptr as *const i32,
            contract.batch_size as i32,
            contract.num_blocks_per_seq as i32,
            contract.tokens as i32,
            GLM52_INDEXER_HEAD_DIM as i32,
            GLM52_INDEXER_QUANT_BLOCK_SIZE as i32,
            contract.layout.cache_block_size as i32,
            contract.layout.cache_block_stride_bytes as i64,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer K gather launch failed: {err}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52IndexerTopK2048 {
    pub rows: usize,
    pub stride: usize,
    pub max_seq_len: usize,
    pub next_n: usize,
    pub seq_lens_is_2d: bool,
}

impl Glm52IndexerTopK2048 {
    pub fn validate(self) -> Result<()> {
        ensure!(self.rows > 0, "GLM5.2 indexer top-k rows must be positive");
        ensure!(
            self.stride >= GLM52_INDEXER_TOPK,
            "GLM5.2 indexer top-k stride {} is smaller than top-k {}",
            self.stride,
            GLM52_INDEXER_TOPK
        );
        ensure!(
            self.max_seq_len <= self.stride,
            "GLM5.2 indexer top-k max_seq_len {} exceeds stride {}",
            self.max_seq_len,
            self.stride
        );
        ensure!(
            self.next_n > 0,
            "GLM5.2 indexer top-k next_n must be positive"
        );
        Ok(())
    }
}

pub fn glm52_indexer_topk_2048_workspace_size(contract: Glm52IndexerTopK2048) -> Result<usize> {
    contract.validate()?;
    let mut workspace_bytes = 0usize;
    let result = unsafe {
        ffi::glm52_indexer_topk_2048_contract_cuda(
            contract.rows as i32,
            contract.stride as i32,
            contract.max_seq_len as i32,
            &mut workspace_bytes as *mut usize,
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer top-k 2048 ABI contract check failed: {err}"))?;
    Ok(workspace_bytes)
}

pub fn glm52_indexer_topk_2048_launch(
    ctx: &DeviceContext,
    contract: Glm52IndexerTopK2048,
    logits: &CudaSlice<f32>,
    seq_lens: &CudaSlice<i32>,
    indices: &mut CudaSlice<i32>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    contract.validate()?;
    ensure!(
        logits.len() >= contract.rows * contract.stride,
        "GLM5.2 indexer top-k logits too small: have {}, need {}",
        logits.len(),
        contract.rows * contract.stride
    );
    ensure!(
        seq_lens.len() >= contract.rows,
        "GLM5.2 indexer top-k seq_lens too small: have {}, need {}",
        seq_lens.len(),
        contract.rows
    );
    ensure!(
        indices.len() >= contract.rows * GLM52_INDEXER_TOPK,
        "GLM5.2 indexer top-k indices too small: have {}, need {}",
        indices.len(),
        contract.rows * GLM52_INDEXER_TOPK
    );
    let required_workspace = glm52_indexer_topk_2048_workspace_size(contract)?;
    ensure!(
        workspace.len() >= required_workspace,
        "GLM5.2 indexer top-k workspace too small: have {}, need {}",
        workspace.len(),
        required_workspace
    );
    let workspace_len = workspace.len();

    let (logits_ptr, _logits_guard) = logits.device_ptr(&ctx.stream);
    let (seq_lens_ptr, _seq_lens_guard) = seq_lens.device_ptr(&ctx.stream);
    let (indices_ptr, _indices_guard) = indices.device_ptr_mut(&ctx.stream);
    let (workspace_ptr, _workspace_guard) = workspace.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_topk_2048_cuda(
            logits_ptr as *const f32,
            seq_lens_ptr as *const i32,
            indices_ptr as *mut i32,
            workspace_ptr as *mut u8,
            workspace_len,
            contract.rows as i32,
            contract.stride as i32,
            contract.max_seq_len as i32,
            contract.next_n as i32,
            if contract.seq_lens_is_2d { 1 } else { 0 },
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer top-k 2048 launch failed: {err}"))
}
