use super::Half;
use cudarc::driver::sys::{CUresult, CUstream};

mod flashmla_sparse;
pub use flashmla_sparse::*;
mod sparse_mla;
pub use sparse_mla::*;
mod indexer;
pub use indexer::*;
mod indexer_rope;
pub use indexer_rope::*;
mod topk;
pub use topk::*;
mod deepgemm_mqa;
pub use deepgemm_mqa::*;

unsafe extern "C" {
    pub fn glm52_decode_feed_launch_cuda(
        argmax_indices: *const i32,
        token_ids: *mut u32,
        positions: *mut u32,
        slot_mapping: *mut i64,
        seq_lens: *mut i32,
        rows: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_grouped_fp8_metadata_cuda(
        psum_expert: *const i32,
        expert_offsets: *mut i64,
        masked_m: *mut i32,
        row_map: *mut i32,
        groups: i32,
        m_capacity: i32,
        expert_alignment: i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_masked_out_to_aligned_cuda(
        masked_out: *const Half,
        masked_m: *const i32,
        expert_offsets: *const i64,
        aligned_out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_masked_grouped_fp8_launch_cuda(
        operand_kind: i32,
        a: *const u8,
        a_scale: *const f32,
        b: *const u8,
        b_scale: *const f32,
        masked_m: *const i32,
        out: *mut Half,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    // --- MLA decode assembly (projections -> FlashMLA glue) -------------------
    pub fn glm52_mla_query_assemble_cuda(
        ql_nope: *const Half,
        q_pe_base: *const Half,
        q_pe_offset: i32,
        q_pe_head_stride: i32,
        num_q_heads: i32,
        cos: *const Half,
        sin: *const Half,
        query: *mut Half,
        tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_mla_cache_pack_cuda(
        ckv_fp8: *const u8,
        ckv_scales: *const f32,
        k_pe: *const Half,
        cos: *const Half,
        sin: *const Half,
        cache: *mut u8,
        slot_mapping: *const i64,
        max_slots: i64,
        tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_mla_ckv_split_cuda(
        ckv: *const Half,
        kv_c: *mut Half,
        k_pe: *mut Half,
        tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_indexer_weights_fold_cuda(
        weights: *const Half,
        q_scale: *const f32,
        softmax_scale: f32,
        n_heads_scale: f32,
        out: *mut f32,
        heads: i32,
        stream: CUstream,
    ) -> CUresult;

    // --- FP8 per-token-group quant (shared by MLA cache, MoE, dense) ----------
    pub fn glm52_fp8_per_token_group_quant_bf16_cuda(
        input: *const Half,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        hidden_dim: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_fp8_per_token_group_quant_bf16_masked_cuda(
        input: *const Half,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        hidden_dim: i32,
        group_size: i32,
        row_bound: *const i64,
        row_map: *const i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_silu_and_mul_weighted_per_token_group_quant_bf16_masked_cuda(
        input: *const Half,
        topk_weights: *const f32,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        hidden_size: i32,
        group_size: i32,
        row_bound: *const i64,
        row_map: *const i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;

    // --- MoE router (csrc/glm52/glm52_router.cu) ------------------------------
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

    pub fn glm52_fp8_weight_only_gemv_batched_cuda(
        activation: *const Half,
        weight: *const u8,
        weight_scale: *const f32,
        out: *mut Half,
        scratch: *mut f32,
        scratch_floats: usize,
        batch: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_silu_and_mul_bf16_cuda(
        input: *const Half,
        output: *mut Half,
        rows: i32,
        inter: i32,
        stream: CUstream,
    ) -> CUresult;

    // --- TP8 low-latency MoE: whole-layer cooperative kernel + LL packets ----
    pub fn glm52_moe_tp8_max_blocks_cuda(out_blocks: *mut i32) -> CUresult;

    pub fn glm52_moe_tp8_alloc_ll_cuda(
        bytes: usize,
        device_ordinals: *const i32,
        n_devices: i32,
        out_vas: *mut u64,
    ) -> CUresult;

    pub fn glm52_moe_tp8_free_ll_cuda(p: *mut std::ffi::c_void) -> CUresult;

    pub fn glm52_moe_tp8_layer_launch_cuda(
        normed2: *const Half,
        topk_idx: *const i32,
        topk_prob: *const f32,
        w13: *const u8,
        w13_scale: *const f32,
        w2: *const u8,
        w2_scale: *const f32,
        mlp_out: *mut Half,
        guidx: *mut i32,
        guprob: *mut f32,
        gucnt: *mut i32,
        gused: *mut i32,
        bpart: *mut f32,
        ug: *mut Half,
        cpart: *mut f32,
        rs_local: *mut std::ffi::c_void,
        peer_rs: *const *const std::ffi::c_void,
        epoch_dev: *mut u64,
        active_rows: *const i32,
        layer_slot: i32,
        nranks: i32,
        myrank: i32,
        grid_blocks: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_moe_tp8_epoch_advance_cuda(
        epoch_dev: *mut std::ffi::c_void,
        stream: CUstream,
    ) -> CUresult;

    // --- TP8 attention allreduce: o_proj/dense-down epilogue collective -----
    pub fn glm52_tp8_ar_launch_cuda(
        partial: *const Half,
        out: *mut Half,
        ar_local: *mut std::ffi::c_void,
        peer_ar: *const *const std::ffi::c_void,
        epoch_dev: *const u64,
        active_rows: *const i32,
        layer_slot: i32,
        rows: i32,
        nranks: i32,
        myrank: i32,
        stream: CUstream,
    ) -> CUresult;
}
