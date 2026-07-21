use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;

/// Which GLM5.2 cache family a restored arena holds — selects the byte range
/// the RoPE deinterleave rewrites per token row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm52VllmFixupKind {
    /// fp8_ds_mla arena (656 B/token): permute the 64 bf16 rope-key dims at
    /// byte 528.
    Mla,
    /// index-K arena (`[64x128 fp8][64x4 f32 scale]` per block): permute the
    /// first 64 of each token's 128 fp8 key bytes.
    IndexK,
}

/// Deinterleave the RoPE dims of vLLM-written cache pages, in place.
///
/// vLLM (`is_neox_style=False`) stores rotated RoPE pairs interleaved at
/// (2i, 2i+1); openinfer's kernels expect the interleave-in / block-out
/// placement `[i, i + rope/2]`. Same values, permuted dims — this pass
/// rewrites each restored page once so it reads like a locally-written page.
///
/// NOT idempotent (deinterleave's inverse is interleave): the caller must run
/// it exactly once per restored page, after the pegaflow H2D has completed
/// and before the page becomes readable. Launched on `ctx.stream`, so
/// ordering against subsequent compute on the same stream is free.
pub fn glm52_vllm_rope_fixup_launch(
    ctx: &DeviceContext,
    arena: &mut CudaSlice<u8>,
    block_stride_bytes: usize,
    kind: Glm52VllmFixupKind,
    pages: &CudaSlice<i32>,
    num_pages: usize,
) -> Result<()> {
    ensure!(num_pages > 0, "GLM5.2 vLLM rope fixup needs pages");
    ensure!(
        pages.len() >= num_pages,
        "GLM5.2 vLLM rope fixup pages buffer too small: have {}, need {num_pages}",
        pages.len()
    );
    let (arena_ptr, _arena_guard) = arena.device_ptr_mut(&ctx.stream);
    let (pages_ptr, _pages_guard) = pages.device_ptr(&ctx.stream);
    let kind = match kind {
        Glm52VllmFixupKind::Mla => 0,
        Glm52VllmFixupKind::IndexK => 1,
    };
    let result = unsafe {
        ffi::glm52_vllm_rope_fixup_cuda(
            arena_ptr as *mut u8,
            block_stride_bytes as i64,
            kind,
            pages_ptr as *const i32,
            num_pages as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 vLLM rope fixup launch failed: {err}"))
}
