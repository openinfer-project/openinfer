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

/// Assemble the FlashMLA sparse decode query `[H, 576]` = `[ql_nope(512) |
/// rope(q_pe)(64)]` per head (bs=1 decode). `cos`/`sin` are the first
/// `GLM52_MLA_ROPE_HALF` (=32) entries of the position's rotary table; RoPE is
/// interleave-in / block-out. q_pe is read at `q_pe_base[q_pe_offset +
/// h*q_pe_head_stride]`: pass `(0, 64)` for a contiguous `[H,64]` q_pe, or
/// `(192, 256)` to read it in place from the `[H,256]` q_b output.
pub fn glm52_mla_query_assemble_launch(
    ctx: &DeviceContext,
    ql_nope: &CudaSlice<bf16>,
    q_pe_base: &CudaSlice<bf16>,
    q_pe_offset: usize,
    q_pe_head_stride: usize,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    query: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        ql_nope.len() >= GLM52_MLA_HEADS * GLM52_MLA_QK_NOPE,
        "GLM5.2 MLA assemble ql_nope too small: have {}, need {}",
        ql_nope.len(),
        GLM52_MLA_HEADS * GLM52_MLA_QK_NOPE
    );
    ensure!(
        q_pe_base.len()
            >= q_pe_offset + (GLM52_MLA_HEADS - 1) * q_pe_head_stride + GLM52_MLA_ROPE_DIM,
        "GLM5.2 MLA assemble q_pe (offset {q_pe_offset}, stride {q_pe_head_stride}) overruns buffer of {}",
        q_pe_base.len()
    );
    ensure!(
        cos.len() >= GLM52_MLA_ROPE_HALF && sin.len() >= GLM52_MLA_ROPE_HALF,
        "GLM5.2 MLA assemble cos/sin must be >= {GLM52_MLA_ROPE_HALF}"
    );
    ensure!(
        query.len() >= GLM52_MLA_HEADS * GLM52_MLA_QUERY_DIM,
        "GLM5.2 MLA assemble query too small: have {}, need {}",
        query.len(),
        GLM52_MLA_HEADS * GLM52_MLA_QUERY_DIM
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
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            query_ptr as *mut ffi::Half,
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
pub fn glm52_mla_cache_pack_launch(
    ctx: &DeviceContext,
    ckv_fp8: &CudaSlice<u8>,
    ckv_scales: &CudaSlice<f32>,
    k_pe: &CudaSlice<bf16>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    slot: usize,
) -> Result<()> {
    ensure!(
        ckv_fp8.len() >= GLM52_MLA_KV_LORA,
        "GLM5.2 MLA cache pack ckv_fp8 too small: have {}, need {GLM52_MLA_KV_LORA}",
        ckv_fp8.len()
    );
    ensure!(
        ckv_scales.len() >= GLM52_MLA_SCALE_GROUPS,
        "GLM5.2 MLA cache pack ckv_scales too small: have {}, need {GLM52_MLA_SCALE_GROUPS}",
        ckv_scales.len()
    );
    ensure!(
        k_pe.len() >= GLM52_MLA_ROPE_DIM,
        "GLM5.2 MLA cache pack k_pe too small: have {}, need {GLM52_MLA_ROPE_DIM}",
        k_pe.len()
    );
    ensure!(
        cos.len() >= GLM52_MLA_ROPE_HALF && sin.len() >= GLM52_MLA_ROPE_HALF,
        "GLM5.2 MLA cache pack cos/sin must be >= {GLM52_MLA_ROPE_HALF}"
    );
    ensure!(
        cache.len() >= (slot + 1) * GLM52_MLA_CACHE_BYTES,
        "GLM5.2 MLA cache pack slot {slot} overruns cache of {} bytes",
        cache.len()
    );
    let (fp8_ptr, _g0) = ckv_fp8.device_ptr(&ctx.stream);
    let (scale_ptr, _g1) = ckv_scales.device_ptr(&ctx.stream);
    let (kpe_ptr, _g2) = k_pe.device_ptr(&ctx.stream);
    let (cos_ptr, _g3) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _g4) = sin.device_ptr(&ctx.stream);
    let (cache_ptr, _g5) = cache.device_ptr_mut(&ctx.stream);
    let slot_ptr = cache_ptr + (slot * GLM52_MLA_CACHE_BYTES) as u64;
    let result = unsafe {
        ffi::glm52_mla_cache_pack_cuda(
            fp8_ptr as *const u8,
            scale_ptr as *const f32,
            kpe_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            slot_ptr as *mut u8,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 MLA cache pack launch failed: {err}"))
}
