use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_INDEXER_HEAD_DIM: usize = 128;
const GLM52_INDEXER_QUANT_BLOCK_SIZE: usize = 128;
const GLM52_INDEXER_SCALE_BYTES_PER_TOKEN: usize = 4;
pub const GLM52_INDEXER_TOPK: usize = 2048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52IndexerCacheLayout {
    pub cache_blocks: usize,
    pub cache_block_size: usize,
    pub cache_block_stride_bytes: usize,
}

impl Glm52IndexerCacheLayout {
    fn min_block_stride_bytes(self) -> usize {
        self.cache_block_size * (GLM52_INDEXER_HEAD_DIM + GLM52_INDEXER_SCALE_BYTES_PER_TOKEN)
    }

    fn validate(self) -> Result<()> {
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
pub struct Glm52IndexerCacheInsert {
    pub tokens: usize,
    pub layout: Glm52IndexerCacheLayout,
}

impl Glm52IndexerCacheInsert {
    fn validate(self) -> Result<()> {
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
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer K quant/cache launch failed: {err}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52IndexerLocalTopKToSlots {
    pub num_tokens: usize,
    pub topk: usize,
    pub block_size: usize,
    pub block_table_cols: usize,
}

impl Glm52IndexerLocalTopKToSlots {
    fn validate(self) -> Result<()> {
        ensure!(
            self.num_tokens > 0,
            "GLM5.2 indexer local_topk_to_slots num_tokens must be positive"
        );
        ensure!(
            self.topk > 0,
            "GLM5.2 indexer local_topk_to_slots topk must be positive"
        );
        ensure!(
            self.block_size > 0,
            "GLM5.2 indexer local_topk_to_slots block_size must be positive"
        );
        ensure!(
            self.block_table_cols > 0,
            "GLM5.2 indexer local_topk_to_slots block_table_cols must be positive"
        );
        Ok(())
    }
}

/// Convert local top-k offsets (within a sequence's KV cache) to global KV
/// slot indices via the block table. Also writes `topk_lens` (valid slot
/// count per token). Ported from TokenSpeed Triton `local_topk_to_global_slots`.
///
/// - `local_topk_offsets`: `[num_tokens, topk]` int32, row-major.
/// - `seq_lens`: `[num_tokens]` int32, valid KV length per token (required,
///   matches vLLM `sparse_attn_indexer` which always passes `seq_lens` as
///   the top-k `lengths`).
/// - `block_table`: `[num_tokens, block_table_cols]` int32, row-major.
/// - `global_slots`: `[num_tokens, topk]` int32 output, `-1` for invalid slots.
/// - `topk_lens`: `[num_tokens]` int32 output, valid slot count per token.
pub fn glm52_indexer_local_topk_to_slots_launch(
    ctx: &DeviceContext,
    contract: Glm52IndexerLocalTopKToSlots,
    local_topk_offsets: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    block_table: &CudaSlice<i32>,
    global_slots: &mut CudaSlice<i32>,
    topk_lens: &mut CudaSlice<i32>,
) -> Result<()> {
    contract.validate()?;
    ensure!(
        local_topk_offsets.len() >= contract.num_tokens * contract.topk,
        "GLM5.2 indexer local_topk_to_slots offsets too small: have {}, need {}",
        local_topk_offsets.len(),
        contract.num_tokens * contract.topk
    );
    ensure!(
        seq_lens.len() >= contract.num_tokens,
        "GLM5.2 indexer local_topk_to_slots seq_lens too small: have {}, need {}",
        seq_lens.len(),
        contract.num_tokens
    );
    ensure!(
        block_table.len() >= contract.num_tokens * contract.block_table_cols,
        "GLM5.2 indexer local_topk_to_slots block_table too small: have {}, need {}",
        block_table.len(),
        contract.num_tokens * contract.block_table_cols
    );
    ensure!(
        global_slots.len() >= contract.num_tokens * contract.topk,
        "GLM5.2 indexer local_topk_to_slots global_slots too small: have {}, need {}",
        global_slots.len(),
        contract.num_tokens * contract.topk
    );
    ensure!(
        topk_lens.len() >= contract.num_tokens,
        "GLM5.2 indexer local_topk_to_slots topk_lens too small: have {}, need {}",
        topk_lens.len(),
        contract.num_tokens
    );

    let (offsets_ptr, _offsets_guard) = local_topk_offsets.device_ptr(&ctx.stream);
    let (seq_lens_ptr, _seq_lens_guard) = seq_lens.device_ptr(&ctx.stream);
    let (block_table_ptr, _block_table_guard) = block_table.device_ptr(&ctx.stream);
    let (global_slots_ptr, _global_slots_guard) = global_slots.device_ptr_mut(&ctx.stream);
    let (topk_lens_ptr, _topk_lens_guard) = topk_lens.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_local_topk_to_slots_cuda(
            global_slots_ptr as *mut i32,
            topk_lens_ptr as *mut i32,
            offsets_ptr as *const i32,
            contract.topk as i32,
            seq_lens_ptr as *const i32,
            block_table_ptr as *const i32,
            contract.block_table_cols as i32,
            contract.block_table_cols as i32,
            contract.block_size as i32,
            contract.topk as i32,
            contract.num_tokens as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer local_topk_to_slots launch failed: {err}"))
}

/// Token bound baked into `glm52_min_gemv.cuh`'s `launch_tokens` switch.
pub const GLM52_MIN_GEMV_MAX_TOKENS: usize = 8;

/// weights_proj min-latency GEMV: `out[t, h] = dot(hidden[t], weights[h])`,
/// bf16 in/out with fixed-order f32 accumulation. Replaces the cublas splitK
/// plan (GEMM + splitKreduce + workspace alloc/free per call).
pub fn glm52_indexer_weights_proj_launch(
    ctx: &DeviceContext,
    hidden: &CudaSlice<bf16>,
    weights_proj: &CudaSlice<bf16>,
    tokens: usize,
    heads: usize,
    hidden_dim: usize,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        (1..=GLM52_MIN_GEMV_MAX_TOKENS).contains(&tokens),
        "GLM5.2 indexer weights_proj tokens {tokens} outside 1..={GLM52_MIN_GEMV_MAX_TOKENS}"
    );
    ensure!(
        hidden.len() >= tokens * hidden_dim,
        "GLM5.2 indexer weights_proj hidden too small: have {}, need {}",
        hidden.len(),
        tokens * hidden_dim
    );
    ensure!(
        weights_proj.len() >= heads * hidden_dim,
        "GLM5.2 indexer weights_proj weights too small: have {}, need {}",
        weights_proj.len(),
        heads * hidden_dim
    );
    ensure!(
        out.len() >= tokens * heads,
        "GLM5.2 indexer weights_proj out too small: have {}, need {}",
        out.len(),
        tokens * heads
    );
    let (h_ptr, _g0) = hidden.device_ptr(&ctx.stream);
    let (w_ptr, _g1) = weights_proj.device_ptr(&ctx.stream);
    let (o_ptr, _g2) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_weights_proj_cuda(
            h_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            tokens as i32,
            heads as i32,
            hidden_dim as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer weights_proj launch failed: {err}"))
}

/// Fold the per-head `weights_proj` output (bf16) with the per-head q quant
/// scale and the two attention scale constants into f32 weights for the
/// DeepGEMM MQA logits kernel: `out[h] = weights[h] * q_scale[h] *
/// softmax_scale * n_heads_scale` (left-to-right f32, bit-identical to the
/// retired host-side fold). Replaces two mid-step D2H readbacks + an H2D —
/// the DSA indexer chain stays on-device (CUDA-graph capturable).
pub fn glm52_indexer_weights_fold_launch(
    ctx: &DeviceContext,
    weights: &CudaSlice<bf16>,
    q_scale: &CudaSlice<f32>,
    softmax_scale: f32,
    n_heads_scale: f32,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    let heads = out.len();
    ensure!(
        heads > 0 && heads <= 1024,
        "GLM5.2 indexer weights fold heads {heads} outside 1..=1024"
    );
    ensure!(
        weights.len() >= heads && q_scale.len() >= heads,
        "GLM5.2 indexer weights fold inputs too small: weights {}, q_scale {} (need {heads})",
        weights.len(),
        q_scale.len()
    );
    let (w_ptr, _g0) = weights.device_ptr(&ctx.stream);
    let (q_ptr, _g1) = q_scale.device_ptr(&ctx.stream);
    let (out_ptr, _g2) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_weights_fold_cuda(
            w_ptr as *const ffi::Half,
            q_ptr as *const f32,
            softmax_scale,
            n_heads_scale,
            out_ptr as *mut f32,
            heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer weights fold launch failed: {err}"))
}
