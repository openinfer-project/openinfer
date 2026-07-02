use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

use super::indexer::GLM52_INDEXER_HEAD_DIM;

pub const GLM52_INDEXER_ROPE_DIM: usize = 64;
pub const GLM52_INDEXER_ROPE_HALF: usize = 32;

/// Interleaved RoPE for the DSA indexer q `[n_heads, head_dim]` and k
/// `[head_dim]` (in-place). Applies RoPE to the first `GLM52_INDEXER_ROPE_DIM`
/// (=64) elements of each q head and of k; the remaining 64 pass-through
/// dimensions are left unchanged. `cos`/`sin` are `[32]` (rope_dim / 2).
///
/// Reuses the `rope_block` device function from `glm52_mla_assembly.cu` —
/// identical RoPE convention (interleave-in / block-out), oracle-validated in
/// PR1 (#477). Aligned to vllm `DeepseekV32Indexer` which uses
/// `is_neox_style=not indexer_rope_interleave`; GLM5.2 config has
/// `indexer_rope_interleave=true` → interleaved.
pub fn glm52_indexer_rope_launch(
    ctx: &DeviceContext,
    q: &mut CudaSlice<bf16>,
    k: &mut CudaSlice<bf16>,
    n_heads: usize,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
) -> Result<()> {
    ensure!(n_heads > 0, "GLM5.2 indexer RoPE n_heads must be positive");
    ensure!(
        q.len() >= n_heads * GLM52_INDEXER_HEAD_DIM,
        "GLM5.2 indexer RoPE q too small: have {}, need {}",
        q.len(),
        n_heads * GLM52_INDEXER_HEAD_DIM
    );
    ensure!(
        k.len() >= GLM52_INDEXER_HEAD_DIM,
        "GLM5.2 indexer RoPE k too small: have {}, need {}",
        k.len(),
        GLM52_INDEXER_HEAD_DIM
    );
    ensure!(
        cos.len() >= GLM52_INDEXER_ROPE_HALF && sin.len() >= GLM52_INDEXER_ROPE_HALF,
        "GLM5.2 indexer RoPE cos/sin must be >= {GLM52_INDEXER_ROPE_HALF}"
    );

    let (q_ptr, _q_guard) = q.device_ptr_mut(&ctx.stream);
    let (k_ptr, _k_guard) = k.device_ptr_mut(&ctx.stream);
    let (cos_ptr, _cos_guard) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _sin_guard) = sin.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_rope_cuda(
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            n_heads as i32,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer RoPE launch failed: {err}"))
}
