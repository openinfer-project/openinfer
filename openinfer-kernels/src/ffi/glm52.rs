use super::Half;
use cudarc::driver::sys::{CUresult, CUstream};

mod flashmla_sparse;
pub use flashmla_sparse::*;

unsafe extern "C" {
    pub fn glm52_deepgemm_mn_major_tma_aligned_f32_cuda(
        input: *const f32,
        output: *mut f32,
        rows: i32,
        scale_cols: i32,
        aligned_rows: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_grouped_fp8_contract_cuda(
        operand_kind: i32,
        groups: i32,
        m_capacity: i32,
        n: i32,
        k: i32,
        weight_scale_rows: i32,
        weight_scale_cols: i32,
        activation_scale_cols: i32,
        activation_scale_tma_rows: i32,
        psum_entries: i32,
        expert_alignment: i32,
    ) -> CUresult;

    pub fn glm52_deepgemm_grouped_fp8_metadata_cuda(
        psum_expert: *const i32,
        expert_offsets: *mut i64,
        w13_problem_sizes: *mut i32,
        w2_problem_sizes: *mut i32,
        groups: i32,
        m_capacity: i32,
        expert_alignment: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_grouped_fp8_launch_cuda(
        operand_kind: i32,
        a: *const u8,
        a_scale: *const f32,
        b: *const u8,
        b_scale: *const f32,
        psum_expert: *const i32,
        out: *mut Half,
        groups: i32,
        m_capacity: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;
}
