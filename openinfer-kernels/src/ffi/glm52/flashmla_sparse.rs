use cudarc::driver::sys::{CUresult, CUstream};

use crate::ffi::Half;

unsafe extern "C" {
    pub fn glm52_flashmla_sparse_decode_num_sm_parts_cuda(num_sm_parts: *mut i32) -> CUresult;

    pub fn glm52_flashmla_sparse_decode_metadata_cuda(
        tile_scheduler_metadata: *mut i32,
        num_splits: *mut i32,
        batch_size: i32,
        topk: i32,
        num_sm_parts: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_flashmla_sparse_decode_launch_cuda(
        q: *const Half,
        packed_kv_cache: *const u8,
        topk_indices: *const i32,
        tile_scheduler_metadata: *const i32,
        num_splits: *const i32,
        out_latent: *mut Half,
        lse: *mut f32,
        lse_accum: *mut f32,
        o_accum: *mut f32,
        batch_size: i32,
        num_blocks: i32,
        topk: i32,
        num_sm_parts: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;
}
