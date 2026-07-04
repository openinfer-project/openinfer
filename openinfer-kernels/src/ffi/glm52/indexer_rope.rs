use crate::ffi::Half;
use cudarc::driver::sys::{CUresult, CUstream};

unsafe extern "C" {
    pub fn glm52_indexer_rope_cuda(
        q: *mut Half,
        k: *mut Half,
        n_heads: i32,
        tokens: i32,
        cos: *const Half,
        sin: *const Half,
        stream: CUstream,
    ) -> CUresult;
}
