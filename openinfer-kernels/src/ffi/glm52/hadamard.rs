use crate::ffi::Half;
use cudarc::driver::sys::{CUresult, CUstream};

unsafe extern "C" {
    pub fn glm52_indexer_hadamard_bf16_cuda(
        input: *const Half,
        output: *mut Half,
        tokens: i32,
        head_dim: i32,
        stream: CUstream,
    ) -> CUresult;
}
