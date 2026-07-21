use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use crate::ffi::Half;

unsafe extern "C" {
    pub fn glm52_indexer_k_quant_and_cache_cuda(
        k: *const Half,
        indexer_cache: *mut u8,
        slot_mapping: *const i64,
        tokens: i32,
        head_dim: i32,
        quant_block_size: i32,
        cache_block_size: i32,
        cache_block_stride_bytes: i64,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_indexer_weights_proj_cuda(
        hidden: *const Half,
        weights_proj: *const Half,
        out: *mut Half,
        tokens: i32,
        heads: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_indexer_local_topk_to_slots_cuda(
        global_slots: *mut i32,
        topk_lens: *mut i32,
        local_topk_offsets: *const i32,
        local_topk_stride: i32,
        seq_lens: *const i32,
        block_table: *const i32,
        block_table_stride: i32,
        block_table_cols: i32,
        block_size: i32,
        topk: i32,
        num_tokens: i32,
        stream: CUstream,
    ) -> CUresult;
}
