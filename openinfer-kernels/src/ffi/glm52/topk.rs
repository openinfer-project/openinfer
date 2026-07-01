use cudarc::driver::sys::{CUresult, CUstream};

unsafe extern "C" {
    pub fn glm52_flashinfer_topk_2048_row_states_bytes_cuda() -> usize;

    pub fn glm52_flashinfer_topk_2048_cuda(
        logits: *const f32,
        output_indices: *mut i32,
        output_values: *mut f32,
        lengths: *const i32,
        num_rows: i32,
        top_k: i32,
        max_len: i32,
        row_states_scratch: *mut u8,
        stream: CUstream,
    ) -> CUresult;
}
