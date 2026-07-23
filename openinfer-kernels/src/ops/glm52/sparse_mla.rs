//! Right-sized sparse MLA decode (M5b): the attention-TP replacement for the
//! FlashMLA sparse decode launch. TileLang-generated 16-split main kernel
//! (sm_90a only, AOT-instantiated per topk) plus a deterministic fixed-order
//! combine. The EP8 (64-head) path stays on FlashMLA (`flashmla_sparse.rs`).

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use super::flashmla_sparse::GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN;
use super::flashmla_sparse::GLM52_FLASHMLA_SPARSE_HEADS;
use super::flashmla_sparse::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use super::flashmla_sparse::GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM;
use super::flashmla_sparse::GLM52_FLASHMLA_SPARSE_V_HEAD_DIM;
use crate::ffi;
use crate::tensor::DeviceContext;

/// Split count and partial store width of the kernel pair. Also baked into
/// the TileLang generator (`NUM_SPLITS`/`HEAD_SLOTS_OUT`) and the CUDA
/// combine (`kNumSplits`/`kHeadSlots`); the launch passes them down and both
/// layers validate, so a lone edit fails at the first launch instead of
/// writing `o_part` out of bounds.
const GLM52_SPARSE_MLA_NUM_SPLITS: usize = 16;
pub const GLM52_SPARSE_MLA_HEAD_SLOTS: usize = 16;
/// GLM5.2's softmax scale, baked into the TileLang kernel at generation
/// time. Validated here by name — the CUDA entry would only report a bare
/// INVALID_VALUE.
const GLM52_SPARSE_MLA_SM_SCALE: f32 = 0.0625;
/// The TileLang main kernel is AOT-instantiated per topk
/// (`tools/tilelang/glm52/generate.py` `TOPKS`); production runs only the
/// full DSA topk (the 256 short tier was dropped — see the note there for
/// how to build it again). Keep in sync with the generator and the
/// launcher's `supported_topk`.
const GLM52_SPARSE_MLA_TOPKS: [usize; 1] = [2048];

/// Launch contract for the right-sized sparse MLA decode. `heads` is the real
/// head count (attention-TP shard, <= 16); the query stays full-width
/// `[batch, 64, 576]` with pad slots zero-filled, and only slots `0..heads`
/// of the `[batch, 64, 512]` latent are written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm52SparseMlaDecode {
    pub batch_size: usize,
    pub num_blocks: usize,
    pub topk: usize,
    pub heads: usize,
    pub sm_scale: f32,
}

impl Glm52SparseMlaDecode {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.batch_size > 0,
            "GLM5.2 sparse MLA decode batch_size must be positive"
        );
        ensure!(
            self.num_blocks > 0,
            "GLM5.2 sparse MLA decode num_blocks must be positive"
        );
        ensure!(
            GLM52_SPARSE_MLA_TOPKS.contains(&self.topk),
            "GLM5.2 sparse MLA decode topk must be one of {GLM52_SPARSE_MLA_TOPKS:?}, got {}",
            self.topk
        );
        ensure!(
            (1..=GLM52_SPARSE_MLA_HEAD_SLOTS).contains(&self.heads),
            "GLM5.2 sparse MLA decode heads {} out of 1..={GLM52_SPARSE_MLA_HEAD_SLOTS} (the EP8 64-head path stays on FlashMLA)",
            self.heads
        );
        ensure!(
            self.sm_scale == GLM52_SPARSE_MLA_SM_SCALE,
            "GLM5.2 sparse MLA decode sm_scale {} != {GLM52_SPARSE_MLA_SM_SCALE} \
             (the TileLang kernel bakes the softmax scale at generation time)",
            self.sm_scale
        );
        Ok(())
    }

    fn max_slots(self) -> usize {
        self.num_blocks * GLM52_FLASHMLA_SPARSE_PAGE_SIZE
    }

    fn q_len(self) -> usize {
        self.batch_size * GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM
    }

    fn packed_kv_cache_len(self) -> usize {
        self.max_slots() * GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN
    }

    fn topk_indices_len(self) -> usize {
        self.batch_size * self.topk
    }

    pub fn latent_len(self) -> usize {
        self.batch_size * GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_V_HEAD_DIM
    }

    pub fn o_part_len(self) -> usize {
        GLM52_SPARSE_MLA_NUM_SPLITS
            * self.batch_size
            * GLM52_SPARSE_MLA_HEAD_SLOTS
            * GLM52_FLASHMLA_SPARSE_V_HEAD_DIM
    }

    pub fn ml_part_len(self) -> usize {
        GLM52_SPARSE_MLA_NUM_SPLITS * self.batch_size * GLM52_SPARSE_MLA_HEAD_SLOTS * 2
    }
}

pub fn glm52_sparse_mla_decode_launch(
    ctx: &DeviceContext,
    contract: Glm52SparseMlaDecode,
    q: &CudaSlice<bf16>,
    packed_kv_cache: &CudaSlice<u8>,
    topk_indices: &CudaSlice<i32>,
    o_part: &mut CudaSlice<f32>,
    ml_part: &mut CudaSlice<f32>,
    out_latent: &mut CudaSlice<bf16>,
) -> Result<()> {
    validate_buffers(contract, q, packed_kv_cache, topk_indices, out_latent)?;
    ensure!(
        o_part.len() >= contract.o_part_len(),
        "GLM5.2 sparse MLA o_part too small: have {}, need {}",
        o_part.len(),
        contract.o_part_len()
    );
    ensure!(
        ml_part.len() >= contract.ml_part_len(),
        "GLM5.2 sparse MLA ml_part too small: have {}, need {}",
        ml_part.len(),
        contract.ml_part_len()
    );

    let (q_ptr, _q_guard) = q.device_ptr(&ctx.stream);
    let (kv_ptr, _kv_guard) = packed_kv_cache.device_ptr(&ctx.stream);
    let (indices_ptr, _indices_guard) = topk_indices.device_ptr(&ctx.stream);
    let (o_part_ptr, _o_part_guard) = o_part.device_ptr_mut(&ctx.stream);
    let (ml_part_ptr, _ml_part_guard) = ml_part.device_ptr_mut(&ctx.stream);
    let (out_ptr, _out_guard) = out_latent.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_sparse_mla_decode_cuda(
            q_ptr as *const ffi::Half,
            kv_ptr as *const u8,
            indices_ptr as *const i32,
            o_part_ptr as *mut f32,
            ml_part_ptr as *mut f32,
            out_ptr as *mut ffi::Half,
            contract.batch_size as i32,
            contract.max_slots() as i64,
            contract.topk as i32,
            contract.heads as i32,
            GLM52_SPARSE_MLA_NUM_SPLITS as i32,
            GLM52_SPARSE_MLA_HEAD_SLOTS as i32,
            contract.sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 sparse MLA decode launch failed: {err}"))
}

/// Naive f64 attention over the same packed cache: the parity gate's ground
/// truth (runs on any SM; the FlashMLA reference is sm90-only). Test-tier.
pub fn glm52_sparse_mla_reference_launch(
    ctx: &DeviceContext,
    contract: Glm52SparseMlaDecode,
    q: &CudaSlice<bf16>,
    packed_kv_cache: &CudaSlice<u8>,
    topk_indices: &CudaSlice<i32>,
    out_latent: &mut CudaSlice<bf16>,
) -> Result<()> {
    validate_buffers(contract, q, packed_kv_cache, topk_indices, out_latent)?;
    let (q_ptr, _q_guard) = q.device_ptr(&ctx.stream);
    let (kv_ptr, _kv_guard) = packed_kv_cache.device_ptr(&ctx.stream);
    let (indices_ptr, _indices_guard) = topk_indices.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out_latent.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_sparse_mla_reference_cuda(
            q_ptr as *const ffi::Half,
            kv_ptr as *const u8,
            indices_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            contract.batch_size as i32,
            contract.max_slots() as i64,
            contract.topk as i32,
            contract.heads as i32,
            contract.sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 sparse MLA reference launch failed: {err}"))
}

fn validate_buffers(
    contract: Glm52SparseMlaDecode,
    q: &CudaSlice<bf16>,
    packed_kv_cache: &CudaSlice<u8>,
    topk_indices: &CudaSlice<i32>,
    out_latent: &CudaSlice<bf16>,
) -> Result<()> {
    contract.validate()?;
    ensure!(
        q.len() >= contract.q_len(),
        "GLM5.2 sparse MLA q too small: have {}, need {}",
        q.len(),
        contract.q_len()
    );
    ensure!(
        packed_kv_cache.len() >= contract.packed_kv_cache_len(),
        "GLM5.2 sparse MLA packed kv cache too small: have {}, need {}",
        packed_kv_cache.len(),
        contract.packed_kv_cache_len()
    );
    ensure!(
        topk_indices.len() >= contract.topk_indices_len(),
        "GLM5.2 sparse MLA topk_indices too small: have {}, need {}",
        topk_indices.len(),
        contract.topk_indices_len()
    );
    ensure!(
        out_latent.len() >= contract.latent_len(),
        "GLM5.2 sparse MLA latent output too small: have {}, need {}",
        out_latent.len(),
        contract.latent_len()
    );
    Ok(())
}
