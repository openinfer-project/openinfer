use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_INDEXER_HADAMARD_HEAD_DIM: usize = 128;

/// Naive in-place radix Hadamard rotate for the DSA indexer (head_dim=128).
/// Applies scale = head_dim^-0.5. NOT the Dao-AILab fast-hadamard-transform
/// port — correct but not tuned; first ncu candidate if decode TPOT is measured.
///
/// `input` and `output` are `[tokens, head_dim]` bf16, row-major.
/// `output` may alias `input` for in-place operation.
pub fn glm52_indexer_hadamard_bf16_launch(
    ctx: &DeviceContext,
    tokens: usize,
    head_dim: usize,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        tokens > 0,
        "GLM5.2 indexer Hadamard tokens must be positive"
    );
    ensure!(
        head_dim == GLM52_INDEXER_HADAMARD_HEAD_DIM,
        "GLM5.2 indexer Hadamard head_dim must be {}, got {}",
        GLM52_INDEXER_HADAMARD_HEAD_DIM,
        head_dim
    );
    ensure!(
        input.len() >= tokens * head_dim,
        "GLM5.2 indexer Hadamard input too small: have {}, need {}",
        input.len(),
        tokens * head_dim
    );
    ensure!(
        output.len() >= tokens * head_dim,
        "GLM5.2 indexer Hadamard output too small: have {}, need {}",
        output.len(),
        tokens * head_dim
    );

    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_hadamard_bf16_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut ffi::Half,
            tokens as i32,
            head_dim as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer Hadamard launch failed: {err}"))
}
