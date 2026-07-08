use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_MLA_HEADS: usize = 64;
pub const GLM52_MLA_QK_NOPE: usize = 512;
pub const GLM52_MLA_ROPE_DIM: usize = 64;
/// cos/sin length used by interleave RoPE (rope_dim / 2).
pub const GLM52_MLA_ROPE_HALF: usize = 32;
pub const GLM52_MLA_QUERY_DIM: usize = GLM52_MLA_QK_NOPE + GLM52_MLA_ROPE_DIM; // 576
pub const GLM52_MLA_KV_LORA: usize = 512;
pub const GLM52_MLA_SCALE_GROUPS: usize = GLM52_MLA_KV_LORA / 128; // 4
/// fp8_ds_mla token: 512 e4m3 ckv + 4 f32 scales + 64 bf16 rope-key.
pub const GLM52_MLA_CACHE_BYTES: usize = 656;

/// Assemble the FlashMLA sparse decode query `[GLM52_MLA_HEADS, 576]` =
/// `[ql_nope(512) | rope(q_pe)(64)]` per head (bs=1 decode). `num_q_heads`
/// may be a head-parallel shard (attention-TP: 8 of 64): `ql_nope`/`q_pe`
/// are COMPACT `[T, num_q_heads, .]`, while `query` stays full-width
/// `[T, GLM52_MLA_HEADS, 576]` (the FlashMLA kernel only runs h_q=64; shard
/// heads land in slots 0..num_q_heads, pad slots keep their zero fill).
/// `cos`/`sin` are the first `GLM52_MLA_ROPE_HALF` (=32) entries of the
/// position's rotary table; RoPE is interleave-in / block-out. q_pe is read
/// at `q_pe_base[q_pe_offset + h*q_pe_head_stride]`: pass `(0, 64)` for a
/// contiguous `[num_q_heads,64]` q_pe, or `(192, 256)` to read it in place
/// from the `[num_q_heads,256]` q_b output.
#[allow(clippy::too_many_arguments)]
pub fn glm52_mla_query_assemble_launch(
    ctx: &DeviceContext,
    tokens: usize,
    num_q_heads: usize,
    ql_nope: &CudaSlice<bf16>,
    q_pe_base: &CudaSlice<bf16>,
    q_pe_offset: usize,
    q_pe_head_stride: usize,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    query: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(tokens > 0, "GLM5.2 MLA assemble tokens must be positive");
    ensure!(
        num_q_heads >= 1 && num_q_heads <= GLM52_MLA_HEADS,
        "GLM5.2 MLA assemble num_q_heads {num_q_heads} out of 1..={GLM52_MLA_HEADS}"
    );
    ensure!(
        ql_nope.len() >= tokens * num_q_heads * GLM52_MLA_QK_NOPE,
        "GLM5.2 MLA assemble ql_nope too small: have {}, need {}",
        ql_nope.len(),
        tokens * num_q_heads * GLM52_MLA_QK_NOPE
    );
    ensure!(
        q_pe_base.len()
            >= q_pe_offset + (tokens * num_q_heads - 1) * q_pe_head_stride + GLM52_MLA_ROPE_DIM,
        "GLM5.2 MLA assemble q_pe (offset {q_pe_offset}, stride {q_pe_head_stride}) overruns buffer of {}",
        q_pe_base.len()
    );
    ensure!(
        cos.len() >= tokens * GLM52_MLA_ROPE_HALF && sin.len() >= tokens * GLM52_MLA_ROPE_HALF,
        "GLM5.2 MLA assemble cos/sin must be >= tokens * {GLM52_MLA_ROPE_HALF}"
    );
    ensure!(
        query.len() >= tokens * GLM52_MLA_HEADS * GLM52_MLA_QUERY_DIM,
        "GLM5.2 MLA assemble query too small: have {}, need {}",
        query.len(),
        tokens * GLM52_MLA_HEADS * GLM52_MLA_QUERY_DIM
    );
    let (ql_ptr, _g0) = ql_nope.device_ptr(&ctx.stream);
    let (qpe_ptr, _g1) = q_pe_base.device_ptr(&ctx.stream);
    let (cos_ptr, _g2) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _g3) = sin.device_ptr(&ctx.stream);
    let (query_ptr, _g4) = query.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_mla_query_assemble_cuda(
            ql_ptr as *const ffi::Half,
            qpe_ptr as *const ffi::Half,
            q_pe_offset as i32,
            q_pe_head_stride as i32,
            num_q_heads as i32,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            query_ptr as *mut ffi::Half,
            tokens as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 MLA query assemble launch failed: {err}"))
}

/// Pack one fp8_ds_mla 656-byte cache token = `[512 e4m3 ckv | 4 f32 group scales
/// | 64 bf16 rope(k_pe)]` into `cache` at token `slot` (paged cache, stride 656).
/// `ckv_fp8` + `ckv_scales` come straight from `glm52_fp8_per_token_group_quant`
/// (amax/448, the cache's own scale convention); `k_pe` is the pre-rope shared
/// rope-key. Slot starts are 4-byte aligned since 656 % 4 == 0.
#[allow(clippy::too_many_arguments)]
pub fn glm52_mla_cache_pack_launch(
    ctx: &DeviceContext,
    tokens: usize,
    ckv_fp8: &CudaSlice<u8>,
    ckv_scales: &CudaSlice<f32>,
    k_pe: &CudaSlice<bf16>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    slot_mapping: &CudaSlice<i64>,
) -> Result<()> {
    ensure!(tokens > 0, "GLM5.2 MLA cache pack tokens must be positive");
    ensure!(
        ckv_fp8.len() >= tokens * GLM52_MLA_KV_LORA,
        "GLM5.2 MLA cache pack ckv_fp8 too small: have {}, need {}",
        ckv_fp8.len(),
        tokens * GLM52_MLA_KV_LORA
    );
    ensure!(
        ckv_scales.len() >= tokens * GLM52_MLA_SCALE_GROUPS,
        "GLM5.2 MLA cache pack ckv_scales too small: have {}, need {}",
        ckv_scales.len(),
        tokens * GLM52_MLA_SCALE_GROUPS
    );
    ensure!(
        k_pe.len() >= tokens * GLM52_MLA_ROPE_DIM,
        "GLM5.2 MLA cache pack k_pe too small: have {}, need {}",
        k_pe.len(),
        tokens * GLM52_MLA_ROPE_DIM
    );
    ensure!(
        cos.len() >= tokens * GLM52_MLA_ROPE_HALF && sin.len() >= tokens * GLM52_MLA_ROPE_HALF,
        "GLM5.2 MLA cache pack cos/sin must be >= tokens * {GLM52_MLA_ROPE_HALF}"
    );
    ensure!(
        slot_mapping.len() >= tokens,
        "GLM5.2 MLA cache pack slot_mapping too small: have {}, need {tokens}",
        slot_mapping.len()
    );
    // The slot itself is device data (graph-replayable); the kernel traps on
    // an out-of-window slot. Host-side we can only pin the window size.
    let max_slots = cache.len() / GLM52_MLA_CACHE_BYTES;
    ensure!(
        max_slots > 0,
        "GLM5.2 MLA cache pack cache smaller than one {GLM52_MLA_CACHE_BYTES}-byte token"
    );
    let (fp8_ptr, _g0) = ckv_fp8.device_ptr(&ctx.stream);
    let (scale_ptr, _g1) = ckv_scales.device_ptr(&ctx.stream);
    let (kpe_ptr, _g2) = k_pe.device_ptr(&ctx.stream);
    let (cos_ptr, _g3) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _g4) = sin.device_ptr(&ctx.stream);
    let (cache_ptr, _g5) = cache.device_ptr_mut(&ctx.stream);
    let (slot_ptr, _g6) = slot_mapping.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_mla_cache_pack_cuda(
            fp8_ptr as *const u8,
            scale_ptr as *const f32,
            kpe_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            cache_ptr as *mut u8,
            slot_ptr as *const i64,
            max_slots as i64,
            tokens as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 MLA cache pack launch failed: {err}"))
}

/// Split the kv_a projection output `[T, 576]` into contiguous kv_c `[T, 512]`
/// (pre-norm compressed kv) and k_pe `[T, 64]` (pre-rope shared key). Replaces
/// the per-token dtod slice copies that don't batch.
pub fn glm52_mla_ckv_split_launch(
    ctx: &DeviceContext,
    tokens: usize,
    ckv: &CudaSlice<bf16>,
    kv_c: &mut CudaSlice<bf16>,
    k_pe: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(tokens > 0, "GLM5.2 MLA ckv split tokens must be positive");
    let width = GLM52_MLA_KV_LORA + GLM52_MLA_ROPE_DIM;
    ensure!(
        ckv.len() >= tokens * width
            && kv_c.len() >= tokens * GLM52_MLA_KV_LORA
            && k_pe.len() >= tokens * GLM52_MLA_ROPE_DIM,
        "GLM5.2 MLA ckv split buffers too small: ckv {}, kv_c {}, k_pe {} for {tokens} tokens",
        ckv.len(),
        kv_c.len(),
        k_pe.len()
    );
    let (ckv_ptr, _g0) = ckv.device_ptr(&ctx.stream);
    let (kv_c_ptr, _g1) = kv_c.device_ptr_mut(&ctx.stream);
    let (k_pe_ptr, _g2) = k_pe.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_mla_ckv_split_cuda(
            ckv_ptr as *const ffi::Half,
            kv_c_ptr as *mut ffi::Half,
            k_pe_ptr as *mut ffi::Half,
            tokens as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 MLA ckv split launch failed: {err}"))
}
