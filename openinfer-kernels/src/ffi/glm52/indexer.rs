use crate::ffi::Half;
use cudarc::driver::sys::{CUresult, CUstream};

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
        use_ue8m0_scale: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_indexer_k_gather_quant_cache_cuda(
        indexer_cache: *const u8,
        dst_k: *mut u8,
        dst_scale: *mut u8,
        block_table: *const i32,
        cu_seq_lens: *const i32,
        batch_size: i32,
        num_blocks_per_seq: i32,
        tokens: i32,
        head_dim: i32,
        quant_block_size: i32,
        cache_block_size: i32,
        cache_block_stride_bytes: i64,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_indexer_topk_2048_contract_cuda(
        rows: i32,
        stride: i32,
        max_seq_len: i32,
        workspace_bytes: *mut usize,
    ) -> CUresult;

    pub fn glm52_indexer_topk_2048_cuda(
        logits: *const f32,
        seq_lens: *const i32,
        indices: *mut i32,
        workspace: *mut u8,
        workspace_bytes: usize,
        rows: i32,
        stride: i32,
        max_seq_len: i32,
        next_n: i32,
        seq_lens_is_2d: i32,
        stream: CUstream,
    ) -> CUresult;
}
