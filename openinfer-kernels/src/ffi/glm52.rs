use super::Half;
use cudarc::driver::sys::{CUresult, CUstream};

mod flashmla_sparse;
mod indexer;
pub use flashmla_sparse::*;
pub use indexer::*;

unsafe extern "C" {
    // --- PP8 stage-boundary P2P spine (Slice 0) -------------------------------
    pub fn glm52_pp_source_inject(epoch: *mut u64, stream: CUstream) -> CUresult;

    pub fn glm52_pp_wait_hidden(
        my_flag: *const u64,
        epoch: *mut u64,
        up_ack: *mut u64,
        err_code: *mut u32,
        deadline_ns: u64,
        ring: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_pp_send_hidden(
        src_hidden: *const core::ffi::c_void,
        peer_hidden: *mut core::ffi::c_void,
        peer_flag: *mut u64,
        epoch: *const u64,
        down_ack: *const u64,
        deltas: *mut u64,
        err_code: *mut u32,
        words: i32,
        ring: i32,
        warmup: u64,
        n_samples: u64,
        deadline_ns: u64,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_pp_dummy_burn(burn_ns: u64, stream: CUstream) -> CUresult;

    pub fn glm52_router_noaux_tc_cuda(
        hidden: *const Half,
        gate_weight: *const Half,
        e_score_correction_bias: *const f32,
        logits: *mut f32,
        topk_weight: *mut f32,
        topk_idx: *mut i32,
        active_tokens: i32,
        padded_tokens: i32,
        hidden_dim: i32,
        n_experts: i32,
        topk: i32,
        route_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_fp8_per_token_group_quant_bf16_cuda(
        input: *const Half,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        hidden_dim: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_silu_and_mul_per_token_group_quant_bf16_cuda(
        input: *const Half,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        hidden_size: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_silu_and_mul_weighted_per_token_group_quant_bf16_cuda(
        input: *const Half,
        topk_weights: *const f32,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        hidden_size: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_mn_major_tma_aligned_f32_cuda(
        input: *const f32,
        output: *mut f32,
        rows: i32,
        scale_cols: i32,
        aligned_rows: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_grouped_offset_tma_aligned_f32_cuda(
        input: *const f32,
        expert_offsets: *const i64,
        output: *mut f32,
        m_capacity: i32,
        scale_cols: i32,
        groups: i32,
        padded_rows: i32,
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

    pub fn glm52_trtllm_grouped_fp8_contract_cuda(
        operand_kind: i32,
        groups: i32,
        m_capacity: i32,
        n: i32,
        k: i32,
        weight_scale_rows: i32,
        weight_scale_cols: i32,
        activation_scale_cols: i32,
        activation_scale_trtllm_rows: i32,
    ) -> CUresult;

    pub fn glm52_trtllm_grouped_fp8_workspace_size_cuda(
        operand_kind: i32,
        groups: i32,
        m_capacity: i32,
        n: i32,
        k: i32,
        workspace_bytes: *mut usize,
    ) -> CUresult;

    pub fn glm52_trtllm_grouped_fp8_launch_cuda(
        operand_kind: i32,
        a: *const u8,
        a_scale_trtllm: *const f32,
        b: *const u8,
        b_scale: *const f32,
        expert_offsets: *const i64,
        out: *mut Half,
        workspace: *mut std::ffi::c_void,
        workspace_bytes: usize,
        groups: i32,
        m_capacity: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_trtllm_fp8_linear_contract_cuda(
        m: i32,
        n: i32,
        k: i32,
        weight_scale_rows: i32,
        weight_scale_cols: i32,
        activation_scale_cols: i32,
    ) -> CUresult;

    pub fn glm52_trtllm_fp8_linear_workspace_size_cuda(
        m: i32,
        n: i32,
        k: i32,
        workspace_bytes: *mut usize,
    ) -> CUresult;

    pub fn glm52_trtllm_fp8_linear_launch_cuda(
        a: *const u8,
        a_scale: *const f32,
        b: *const u8,
        b_scale: *const f32,
        out: *mut Half,
        workspace: *mut std::ffi::c_void,
        workspace_bytes: usize,
        m: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    // --- MLA decode assembly (projections -> FlashMLA glue) -------------------
    pub fn glm52_mla_query_assemble_cuda(
        ql_nope: *const Half,
        q_pe: *const Half,
        cos: *const Half,
        sin: *const Half,
        query: *mut Half,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_mla_cache_pack_cuda(
        ckv_fp8: *const u8,
        ckv_scales: *const f32,
        k_pe: *const Half,
        cos: *const Half,
        sin: *const Half,
        cache_token: *mut u8,
        stream: CUstream,
    ) -> CUresult;
}
