use super::Half;
use cudarc::driver::sys::{CUresult, CUstream};

// DeepSeek V4 private kernels (feature `deepseek-v4`).
// Sources: csrc/deepseek_v4/*.cu (+ tools/tilelang/deepseek_v4).
unsafe extern "C" {
    pub fn deepseek_bf16_to_f32_cuda(
        input: *const Half,
        output: *mut f32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_f32_to_bf16_cuda(
        input: *const f32,
        output: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_fp8_linear_cuda(
        x: *const Half,
        weight: *const u8,
        weight_scale: *const u8,
        out: *mut Half,
        seq_len: i32,
        in_dim: i32,
        out_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_fp8_w1_w3_with_workspace_cuda(
        x: *const Half,
        w1_weight: *const u8,
        w1_scale: *const u8,
        w3_weight: *const u8,
        w3_scale: *const u8,
        gate_out: *mut Half,
        up_out: *mut Half,
        act: *mut u8,
        act_bytes: usize,
        act_scale: *mut u8,
        act_scale_bytes: usize,
        seq_len: i32,
        in_dim: i32,
        out_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_fp8_w2_swiglu_with_workspace_cuda(
        gate: *const Half,
        up: *const Half,
        weight: *const u8,
        weight_scale: *const u8,
        out: *mut Half,
        act: *mut u8,
        act_bytes: usize,
        act_scale: *mut u8,
        act_scale_bytes: usize,
        seq_len: i32,
        in_dim: i32,
        out_dim: i32,
        limit: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_fp4_linear_cuda(
        x: *const Half,
        weight: *const u8,
        weight_scale: *const u8,
        out: *mut Half,
        seq_len: i32,
        in_dim: i32,
        out_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_moe_fp4_grouped_w1_w3_with_workspace_cuda(
        x: *const Half,
        w1_weights: *const *const u8,
        w1_scales: *const *const u8,
        w3_weights: *const *const u8,
        w3_scales: *const *const u8,
        expert_indptr: *const i32,
        gate_out: *mut Half,
        up_out: *mut Half,
        act: *mut u8,
        act_bytes: usize,
        act_scale: *mut u8,
        act_scale_bytes: usize,
        rows: i32,
        in_dim: i32,
        out_dim: i32,
        local_experts: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_moe_fp4_grouped_w2_swiglu_with_workspace_cuda(
        gate: *const Half,
        up: *const Half,
        weights: *const *const u8,
        scales: *const *const u8,
        expert_indptr: *const i32,
        out: *mut Half,
        act: *mut u8,
        act_bytes: usize,
        act_scale: *mut u8,
        act_scale_bytes: usize,
        rows: i32,
        in_dim: i32,
        out_dim: i32,
        local_experts: i32,
        limit: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hash_gate_cuda(
        x: *const Half,
        gate_weight: *const Half,
        tid2eid: *const i64,
        token_ids: *const u32,
        route_weights: *mut f32,
        route_indices: *mut i32,
        seq_len: i32,
        hidden_dim: i32,
        n_experts: i32,
        topk: i32,
        route_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_score_gate_cuda(
        x: *const Half,
        gate_weight: *const Half,
        gate_bias: *const f32,
        route_weights: *mut f32,
        route_indices: *mut i32,
        seq_len: i32,
        hidden_dim: i32,
        n_experts: i32,
        topk: i32,
        route_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_score_gate_debug_cuda(
        x: *const Half,
        gate_weight: *const Half,
        gate_bias: *const f32,
        raw_scores: *mut f32,
        original_scores: *mut f32,
        select_scores: *mut f32,
        route_weights: *mut f32,
        route_indices: *mut i32,
        seq_len: i32,
        hidden_dim: i32,
        n_experts: i32,
        topk: i32,
        route_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_moe_local_mapping_cuda(
        route_indices: *const i32,
        pos_to_token: *mut i32,
        pos_to_token_topk: *mut i32,
        token_topk_to_pos: *mut i32,
        expert_indptr: *mut i32,
        expert_cursor: *mut i32,
        local_count: *mut i32,
        seq_len: i32,
        topk: i32,
        global_start: i32,
        local_experts: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_moe_expand_to_fused_cuda(
        x: *const Half,
        pos_to_token: *const i32,
        expanded: *mut Half,
        hidden_dim: i32,
        num_expanded: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_moe_reduce_fused_f32_cuda(
        expanded: *const Half,
        route_weights: *const f32,
        token_topk_to_pos: *const i32,
        accum: *mut f32,
        seq_len: i32,
        hidden_dim: i32,
        topk: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_add_f32_bf16_to_bf16_cuda(
        a: *const f32,
        b: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_head_rms_norm_cuda(
        x: *const Half,
        out: *mut Half,
        seq_len: i32,
        num_heads: i32,
        head_dim: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_apply_rope_q_kv_cuda(
        q: *mut Half,
        kv: *mut Half,
        cos_cache: *const f32,
        sin_cache: *const f32,
        seq_len: i32,
        local_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        start_pos: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_apply_rope_q_kv_batch_cuda(
        q: *mut Half,
        kv: *mut Half,
        cos_cache: *const f32,
        sin_cache: *const f32,
        start_pos: *const i32,
        seq_len: i32,
        local_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_fill_rope_cache_cuda(
        inv_freq: *const f32,
        cos_cache: *mut f32,
        sin_cache: *mut f32,
        max_seq_len: i32,
        pairs: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_indexed_attention_prefill_cuda(
        q: *const Half,
        kv: *const Half,
        attn_sink: *const f32,
        topk_idxs: *const i32,
        out: *mut Half,
        seq_len: i32,
        kv_len: i32,
        local_heads: i32,
        head_dim: i32,
        topk: i32,
        softmax_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_indexer_scores_prefill_cuda(
        q: *const Half,
        kv: *const Half,
        weights: *const Half,
        scores: *mut f32,
        seq_len: i32,
        local_heads: i32,
        head_dim: i32,
        compressed_len: i32,
        score_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_indexer_topk_prefill_cuda(
        scores: *const f32,
        topk_idxs: *mut i32,
        seq_len: i32,
        compressed_len: i32,
        topk: i32,
        ratio: i32,
        offset: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_indexer_scores_decode_cuda(
        q: *const Half,
        kv: *const Half,
        weights: *const Half,
        scores: *mut f32,
        local_heads: i32,
        head_dim: i32,
        compressed_len: i32,
        score_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_indexer_scores_decode_batch_cuda(
        q: *const Half,
        kv: *const Half,
        weights: *const Half,
        compressed_len: *const i32,
        cache_base: *const i32,
        scores: *mut f32,
        batch: i32,
        local_heads: i32,
        head_dim: i32,
        max_compressed_len: i32,
        score_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_indexer_topk_decode_cuda(
        scores: *const f32,
        topk_idxs: *mut i32,
        compressed_len: i32,
        topk: i32,
        offset: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_concat_topk_indices_cuda(
        a: *const i32,
        b: *const i32,
        out: *mut i32,
        seq_len: i32,
        a_topk: i32,
        b_topk: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_window_topk_indices_cuda(
        out: *mut i32,
        seq_len: i32,
        window_size: i32,
        topk: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_window_topk_indices_decode_cuda(
        out: *mut i32,
        start_pos: i32,
        window_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_window_topk_indices_decode_batch_cuda(
        out: *mut i32,
        start_pos: *const i32,
        cache_base: *const i32,
        batch: i32,
        window_size: i32,
        topk: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_compress_topk_indices_cuda(
        out: *mut i32,
        seq_len: i32,
        compressed: i32,
        ratio: i32,
        offset: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_compress_topk_indices_decode_cuda(
        out: *mut i32,
        compressed: i32,
        offset: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_compress_topk_indices_decode_batch_cuda(
        out: *mut i32,
        compressed_len: *const i32,
        cache_base: *const i32,
        batch: i32,
        topk: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_window_and_compress_topk_indices_cuda(
        out: *mut i32,
        seq_len: i32,
        window_size: i32,
        window_topk: i32,
        compressed: i32,
        ratio: i32,
        compress_offset: i32,
        topk: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hadamard_fp4_quant_bf16_cuda(
        x: *mut Half,
        rows: i32,
        groups: i32,
        dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_apply_rope_hidden_cuda(
        x: *mut Half,
        cos_cache: *const f32,
        sin_cache: *const f32,
        seq_len: i32,
        local_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        start_pos: i32,
        inverse: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_apply_rope_hidden_batch_cuda(
        x: *mut Half,
        cos_cache: *const f32,
        sin_cache: *const f32,
        start_pos: *const i32,
        seq_len: i32,
        local_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        inverse: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_apply_rope_hidden_strided_cuda(
        x: *mut Half,
        cos_cache: *const f32,
        sin_cache: *const f32,
        seq_len: i32,
        local_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        start_pos: i32,
        position_stride: i32,
        inverse: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_bf16_linear_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        seq_len: i32,
        in_dim: i32,
        out_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_fp8_act_quant_nope_bf16_cuda(
        x: *mut Half,
        seq_len: i32,
        local_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        block_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_bf16_copy_rows_cuda(
        src: *const Half,
        dst: *mut Half,
        hidden_dim: i32,
        rows: i32,
        src_start_row: i32,
        dst_start_row: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_bf16_copy_rows_indexed_cuda(
        src: *const Half,
        dst: *mut Half,
        src_rows: *const i32,
        dst_rows: *const i32,
        hidden_dim: i32,
        rows: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_compressor_nonoverlap_prefill_cuda(
        x: *const Half,
        wkv: *const Half,
        wgate: *const Half,
        ape: *const f32,
        norm: *const Half,
        weighted: *mut f32,
        out: *mut Half,
        seq_len: i32,
        hidden_dim: i32,
        head_dim: i32,
        ratio: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_compressor_overlap_prefill_cuda(
        x: *const Half,
        wkv: *const Half,
        wgate: *const Half,
        ape: *const f32,
        norm: *const Half,
        weighted: *mut f32,
        out: *mut Half,
        seq_len: i32,
        hidden_dim: i32,
        head_dim: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_compressor_nonoverlap_decode_at_cuda(
        x: *const Half,
        wkv: *const Half,
        wgate: *const Half,
        ape: *const f32,
        norm: *const Half,
        kv_state: *mut f32,
        score_state: *mut f32,
        weighted: *mut f32,
        out: *mut Half,
        start_pos: i32,
        hidden_dim: i32,
        head_dim: i32,
        ratio: i32,
        state_offset: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_compressor_overlap_decode_cuda(
        x: *const Half,
        wkv: *const Half,
        wgate: *const Half,
        ape: *const f32,
        norm: *const Half,
        kv_state: *mut f32,
        score_state: *mut f32,
        weighted: *mut f32,
        out: *mut Half,
        start_pos: i32,
        hidden_dim: i32,
        head_dim: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_compressor_overlap_decode_at_cuda(
        x: *const Half,
        wkv: *const Half,
        wgate: *const Half,
        ape: *const f32,
        norm: *const Half,
        kv_state: *mut f32,
        score_state: *mut f32,
        weighted: *mut f32,
        out: *mut Half,
        start_pos: i32,
        hidden_dim: i32,
        head_dim: i32,
        state_offset: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_concat_seq_bf16_cuda(
        a: *const Half,
        b: *const Half,
        out: *mut Half,
        a_seq_len: i32,
        b_seq_len: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hc_expand_cuda(
        x: *const Half,
        out: *mut Half,
        seq_len: i32,
        hc: i32,
        dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hc_mixes_cuda(
        x: *const Half,
        hc_fn: *const f32,
        mixes: *mut f32,
        raw_mixes: *mut f32,
        rms_scales: *mut f32,
        seq_len: i32,
        hc: i32,
        dim: i32,
        mix_hc: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hc_split_sinkhorn_cuda(
        mixes: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        seq_len: i32,
        hc: i32,
        sinkhorn_iters: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hc_pre_output_cuda(
        x: *const Half,
        pre: *const f32,
        out: *mut Half,
        seq_len: i32,
        hc: i32,
        dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hc_pre_from_mixes_cuda(
        x: *const Half,
        mixes: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        post: *mut f32,
        comb: *mut f32,
        out: *mut Half,
        seq_len: i32,
        hc: i32,
        dim: i32,
        sinkhorn_iters: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hc_pre_norm_from_mixes_cuda(
        x: *const Half,
        mixes: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        norm_weight: *const Half,
        post: *mut f32,
        comb: *mut f32,
        out: *mut Half,
        seq_len: i32,
        hc: i32,
        dim: i32,
        sinkhorn_iters: i32,
        hc_eps: f32,
        norm_eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hc_head_pre_cuda(
        mixes: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        pre: *mut f32,
        seq_len: i32,
        hc: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hc_post_cuda(
        x: *const Half,
        residual: *const Half,
        post: *const f32,
        comb: *const f32,
        out: *mut Half,
        seq_len: i32,
        hc: i32,
        dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_hc_post_f32_branch_cuda(
        x: *const f32,
        residual: *const Half,
        post: *const f32,
        comb: *const f32,
        out: *mut Half,
        seq_len: i32,
        hc: i32,
        dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_last_token_bf16_logits_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut f32,
        seq_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_bf16_logits_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut f32,
        seq_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: CUstream,
    ) -> CUresult;
}

// Added during rebase onto main: cutedsl indexer, pplx EP, ratio-4 decode topk.
unsafe extern "C" {
    pub fn deepseek_cutedsl_indexer_scores_exact_bf16_cuda(
        q: *const Half,
        kv: *const Half,
        weights: *const Half,
        scores: *mut f32,
        seq_len: i32,
        compressed_len: i32,
        score_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_pplx_padded_expert_indptr_cuda(
        recv_tokens_per_expert: *const i32,
        expert_indptr: *mut i32,
        local_experts: i32,
        expert_padding: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepseek_ratio4_decode_topk_indices_batch_cuda(
        scores: *const f32,
        start_pos: *const i32,
        window_base: *const i32,
        compressed_len: *const i32,
        compressed_base: *const i32,
        topk_idxs: *mut i32,
        batch: i32,
        window_size: i32,
        max_compressed_len: i32,
        index_topk: i32,
        stream: CUstream,
    ) -> CUresult;

}
