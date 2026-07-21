use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use crate::ffi::Half;

unsafe extern "C" {
    pub fn glm52_sparse_mla_decode_cuda(
        q: *const Half,
        cache: *const u8,
        indices: *const i32,
        o_part: *mut f32,
        ml_part: *mut f32,
        latent: *mut Half,
        batch: i32,
        max_slots: i64,
        topk: i32,
        heads: i32,
        num_splits: i32,
        head_slots: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_sparse_mla_reference_cuda(
        q: *const Half,
        cache: *const u8,
        indices: *const i32,
        latent: *mut Half,
        batch: i32,
        max_slots: i64,
        topk: i32,
        heads: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;
}
