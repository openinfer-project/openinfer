use cudarc::driver::sys::{CUresult, CUstream};

// Half type (16-bit float) - same layout as CUDA half
pub type Half = u16;

// CUDA kernels - all use half precision
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

    pub fn rms_norm_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn rms_norm_batched_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn add_cuda(
        a: *const Half,
        b: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn bf16_to_f32_cuda(
        input: *const Half,
        output: *mut f32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn f32_to_bf16_cuda(
        input: *const f32,
        output: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn scale_f32_cuda(values: *mut f32, scale: f32, n: i32, stream: CUstream) -> CUresult;

    pub fn repeat_f32_for_reduce_scatter_cuda(
        local: *const f32,
        repeated: *mut f32,
        local_elems: i32,
        world_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn fused_add_rms_norm_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn fused_add_rms_norm_batched_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        batch_size: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn silu_mul_triton_aot_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn embedding_batched_cuda(
        embed: *const Half,
        token_ids: *const u32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn embedding_batched_vocab_shard_cuda(
        embed: *const Half,
        token_ids: *const u32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        vocab_start: u32,
        part_vocab_size: u32,
        stream: CUstream,
    ) -> CUresult;

    pub fn argmax_cuda(x: *const Half, out: *mut i32, n: i32, stream: CUstream);

    pub fn argmax_batch_bf16_cuda(
        x: *const Half,
        values: *mut Half,
        indices: *mut i32,
        rows: i32,
        n: i32,
        stream: CUstream,
    );

    pub fn flashinfer_top1_cuda(
        logits: *const Half,
        top1_value_scratch: *mut Half,
        row_states_scratch: *mut u8,
        output: *mut i32,
        vocab_size: i32,
        stream: CUstream,
    );

    pub fn flashinfer_top1_batch_cuda(
        logits: *const Half,
        top1_values: *mut Half,
        row_states_scratch: *mut u8,
        output: *mut i32,
        num_rows: i32,
        vocab_size: i32,
        stream: CUstream,
    );

    pub fn gpu_sample_flashinfer_cuda(
        logits: *const Half,
        probs_scratch: *mut f32,
        valid_scratch: *mut u8,
        output: *mut i32,
        vocab_size: i32,
        inv_temperature: f32,
        top_k: i32,
        top_p: f32,
        seed: u64,
        stream: CUstream,
    );

    pub fn gemm_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> i32;

    pub fn gemm_graphsafe_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_int4_expert_metadata_probe_cuda(
        weight_shape: *const i32,
        weight_shape_entries: usize,
        local_experts: i32,
        in_dim: i32,
        out_dim: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_moe_expert_major_route_cuda(
        topk_idx: *const i32,
        pos_to_token: *mut i32,
        token_topk_to_pos: *mut i32,
        expert_indptr: *mut u32,
        expert_cursor: *mut u32,
        local_count: *mut u32,
        active_tokens: i32,
        topk: i32,
        global_start: i32,
        local_experts: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_moe_expand_to_expert_major_cuda(
        hidden: *const Half,
        pos_to_token: *const i32,
        expert_major_hidden: *mut Half,
        hidden_dim: i32,
        routed_capacity: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_moe_reduce_expert_major_f32_cuda(
        expert_major_output: *const Half,
        topk_weight: *const f32,
        token_topk_to_pos: *const i32,
        out: *mut f32,
        active_tokens: i32,
        hidden_dim: i32,
        topk: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_add_f32_bf16_to_bf16_cuda(
        a: *const f32,
        b: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_scaled_add_f32_bf16_to_bf16_cuda(
        a: *const f32,
        scale: f32,
        b: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_split_qkv_a_cuda(
        qkv_a: *const Half,
        q_a: *mut Half,
        compressed: *mut Half,
        k_rope: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_rope_assemble_prefill_cuda(
        q_proj: *const Half,
        k_rope: *const Half,
        kv_b: *const Half,
        cos: *const Half,
        sin: *const Half,
        q_attn: *mut Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        seq_len: i32,
        local_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_rope_split_decode_cuda(
        q_proj: *const Half,
        k_rope: *const Half,
        cos: *const Half,
        sin: *const Half,
        positions: *const i32,
        q_nope: *mut Half,
        q_pe: *mut Half,
        append_kpe: *mut Half,
        batch_size: i32,
        local_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_rope_apply_kpe_cuda(
        k_rope: *const Half,
        cos: *const Half,
        sin: *const Half,
        positions: *const i32,
        append_kpe: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_flashinfer_single_prefill_mla_cuda(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        local_heads: i32,
        seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_mla_absorb_q_nope_cuda(
        kv_b_proj: *const Half,
        q_nope: *const Half,
        q_abs_nope: *mut Half,
        batch_size: i32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_mla_v_up_cuda(
        kv_b_proj: *const Half,
        latent: *const Half,
        output: *mut Half,
        batch_size: i32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_mla_paged_kv_append_cuda(
        ckv_cache: *mut Half,
        kpe_cache: *mut Half,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len: *const i32,
        append_ckv: *const Half,
        append_kpe: *const Half,
        batch_indices: *const i32,
        positions: *const i32,
        nnz: i32,
        ckv_stride_page: i64,
        ckv_stride_n: i64,
        kpe_stride_page: i64,
        kpe_stride_n: i64,
        page_size: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_flashinfer_batch_decode_mla_cuda(
        q_nope: *const Half,
        q_pe: *const Half,
        output: *mut Half,
        ckv_cache: *const Half,
        kpe_cache: *const Half,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        num_qo_heads: i32,
        ckv_stride_page: i64,
        ckv_stride_n: i64,
        kpe_stride_page: i64,
        kpe_stride_n: i64,
        page_size: i32,
        batch_size: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_marlin_int4_reorder_scale_cuda(
        weight_scale_checkpoint: *const Half,
        weight_scale_marlin: *mut Half,
        in_dim: i32,
        out_dim: i32,
        local_experts: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_int4_reorder_weight_cuda(
        weight_packed_checkpoint_offset_binary: *const u8,
        weight_packed_marlin: *mut u8,
        in_dim: i32,
        out_dim: i32,
        local_experts: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_int4_fuse_w13_cuda(
        gate_weight_packed_marlin: *const u8,
        up_weight_packed_marlin: *const u8,
        w13_weight_packed_marlin: *mut u8,
        gate_scale_marlin: *const Half,
        up_scale_marlin: *const Half,
        w13_scale_marlin: *mut Half,
        in_dim: i32,
        intermediate_dim: i32,
        local_experts: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_wna16_gemm_cuda(
        input: *const Half,
        output: *mut Half,
        c_tmp: *mut f32,
        b_qweight: *const u8,
        b_scales: *const Half,
        workspace: *mut i32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        num_tokens_post_padded: *const i32,
        topk_weights: *const f32,
        workspace_len: i32,
        sorted_token_ids_len: i32,
        moe_block_size: i32,
        top_k: i32,
        mul_topk_weights: bool,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        local_experts: i32,
        group_size: i32,
        sm_count: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_w13_swiglu_cuda(
        w13: *const Half,
        out: *mut Half,
        rows: i32,
        intermediate_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_sum_topk_rows_f32_cuda(
        route_output: *const Half,
        out: *mut f32,
        active_tokens: i32,
        topk: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_moe_marlin_align_block_size_cuda(
        topk_idx: *const i32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_padded: *mut i32,
        expert_offsets: *mut u32,
        expert_cursor: *mut u32,
        active_tokens: i32,
        topk: i32,
        global_start: i32,
        local_experts: i32,
        block_size: i32,
        max_padded_tokens: i32,
        max_m_blocks: i32,
        stream: CUstream,
    ) -> CUresult;

    // Embedding lookup reading token_id from decode_meta[0] (CUDA Graph safe)
    pub fn embedding_decode_cuda(
        embed: *const Half,
        token_id: *const u32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn silu_mul_fused_cuda(
        gate_up: *const Half,
        out: *mut Half,
        intermediate_size: i32,
        bs: i32,
        stream: CUstream,
    );

    pub fn cublas_init();
    pub fn cublas_destroy();
    pub fn cuda_set_device(device_ordinal: i32) -> i32;

    // Prefill QK norm + RoPE only (no KV cache write). For paged prefill path.
    pub fn prefill_qk_norm_rope_only_cuda(
        q_batch: *mut Half,
        k_batch: *mut Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        start_pos: i32,
        rms_eps: f32,
        stream: CUstream,
    );

    // Qwen3.5 full-attention prefill prep: Q/K norm + partial RoPE + KV cache write.
    pub fn prefill_attention_hd256_prep_cuda(
        q_full_batch: *const Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        q_batch_out: *mut Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        start_pos_ptr: *const i32,
        rotary_dim: i32,
        rms_eps: f32,
        max_seq_len: i32,
        stream: CUstream,
    );

    // Apply sigmoid(gate) from interleaved q_full onto attention output in-place.
    pub fn attention_gate_batch_hd256_cuda(
        q_full_batch: *const Half,
        attn_out: *mut Half,
        num_q_heads: i32,
        seq_len: i32,
        stream: CUstream,
    );

    pub fn qk_norm_partial_rope_batched_decode_hd256_cuda(
        q_full_batch: *const Half,
        k_batch: *mut Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        q_batch_out: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        batch_size: i32,
        rotary_dim: i32,
        rms_eps: f32,
        stream: CUstream,
    );

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

    pub fn deepseek_pplx_padded_expert_indptr_cuda(
        recv_tokens_per_expert: *const i32,
        expert_indptr: *mut i32,
        local_experts: i32,
        expert_padding: i32,
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

    pub fn kimi_k2_router_noaux_tc_cuda(
        hidden: *const Half,
        gate_weight: *const Half,
        e_score_correction_bias: *const f32,
        logits: *mut f32,
        scores: *mut f32,
        choice_scores: *mut f32,
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

    #[cfg(feature = "deepseek-v4")]
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

    pub fn deepseek_compressor_nonoverlap_decode_cuda(
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

    // ========================================================================
    // Qwen3.5 kernels
    // ========================================================================

    // Batched (1+weight) RMSNorm — one block per token
    pub fn rms_norm_batched_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    );

    // (1+weight) RMSNorm — Qwen3.5 / Gemma style
    pub fn rms_norm_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    // Per-head RMSNorm with F32 weight + SiLU gate
    pub fn rms_norm_gated_cuda(
        x: *const Half,
        weight: *const f32,
        gate: *const Half,
        out: *mut Half,
        num_heads: i32,
        head_dim: i32,
        eps: f32,
        stream: CUstream,
    );

    // Gated delta rule recurrent decode (single step)
    pub fn gated_delta_rule_decode_cuda(
        qkv: *const Half,
        b_proj: *const Half,
        a_proj: *const Half,
        dt_bias: *const Half,
        A_log: *const f32,
        state: *mut f32,
        output: *mut Half,
        num_key_heads: i32,
        num_value_heads: i32,
        key_dim: i32,
        val_dim: i32,
        stream: CUstream,
    );

    // Causal depthwise conv1d prefill (parallel over sequence)
    pub fn conv1d_prefill_cuda(
        x_seq: *const Half,
        conv_weight: *const Half,
        conv_state: *mut Half,
        out_seq: *mut Half,
        num_channels: i32,
        seq_len: i32,
        kernel_size: i32,
        stream: CUstream,
    );

    pub fn gated_delta_rule_prefill_chunk_prepare_cuda(
        qkv: *const Half,
        b_proj: *const Half,
        a_proj: *const Half,
        dt_bias: *const Half,
        a_log: *const f32,
        q_out: *mut Half,
        k_out: *mut Half,
        v_out: *mut Half,
        g_out: *mut f32,
        beta_out: *mut f32,
        num_key_heads: i32,
        num_value_heads: i32,
        qkv_dim: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gated_delta_rule_prefill_chunk_cumsum_cuda(
        g_in: *const f32,
        g_out: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gated_delta_rule_prefill_chunk_a_cuda(
        k: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        a_tril: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gated_delta_rule_prefill_chunk_solve_cuda(
        a_tril: *const f32,
        a_inv: *mut Half,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gated_delta_rule_prefill_chunk_recompute_cuda(
        k: *const Half,
        v: *const Half,
        beta: *const f32,
        w: *mut Half,
        u: *mut Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    // Chunk-wise GDR prefill stage 1 (Triton AOT): recurrent chunk-state update.
    // Expected future inputs:
    //   k / w: [seq_len, num_value_heads, 128] bf16
    //   u / v_new: [seq_len, num_value_heads, 128] bf16
    //   g_cumsum: [seq_len, num_value_heads] fp32
    //   initial_state / final_state: [num_value_heads, 128, 128] fp32 in [H, K, V] (V contiguous)
    //   chunk_state: [num_chunks, num_value_heads, 128, 128] fp32
    pub fn gated_delta_rule_prefill_chunk_state_cuda(
        k: *const Half,
        w: *const Half,
        u: *const Half,
        g_cumsum: *const f32,
        initial_state: *const f32,
        chunk_state: *mut f32,
        v_new: *mut Half,
        final_state: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    // Chunk-wise GDR prefill stage 2 (Triton AOT): chunk output accumulation.
    // Expected future inputs:
    //   q / k / v_new: [seq_len, num_value_heads, 128] bf16
    //   chunk_state: [num_chunks, num_value_heads, 128, 128] fp32
    //   g_cumsum: [seq_len, num_value_heads] fp32
    //   output: [seq_len, num_value_heads * 128] bf16
    pub fn gated_delta_rule_prefill_chunk_o_cuda(
        q: *const Half,
        k: *const Half,
        v_new: *const Half,
        chunk_state: *const f32,
        g_cumsum: *const f32,
        output: *mut Half,
        seq_len: i32,
        num_value_heads: i32,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    // ========================================================================
    // Paged attention (FlashInfer)
    // ========================================================================

    // Batched QK RMSNorm + RoPE for decode with per-request positions.
    pub fn qk_norm_rope_batched_decode_cuda(
        q: *mut Half,
        k: *mut Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        batch_size: i32,
        rms_eps: f32,
        stream: CUstream,
    );

    // Scatter contiguous KV → paged layout (one layer, FlashInfer prefill append).
    pub fn paged_kv_scatter_cuda(
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        src_k: *const Half,
        src_v: *const Half,
        batch_indices: *const i32,
        positions: *const i32,
        nnz: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        stride_page: i64,
        src_stride_n: i64,
        src_stride_h: i64,
        stream: CUstream,
    ) -> i32;

    // Return the number of Q tiles for batch prefill (needed to size plan arrays).
    pub fn batch_prefill_paged_num_tiles(
        seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    pub fn batch_prefill_paged_num_tiles_with_cta_tile_q(
        seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        cta_tile_q_override: i32,
    ) -> i32;

    // Return the CTA tile size for batch prefill planning.
    pub fn batch_prefill_cta_tile_q(
        total_seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    pub fn batch_prefill_cta_tile_q_with_override(
        total_seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        cta_tile_q_override: i32,
    ) -> i32;

    // Batch prefill with paged KV cache (FlashInfer BatchPrefill, causal, kNone).
    pub fn batch_prefill_paged_cuda(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        q_indptr: *const i32,
        request_indices: *const i32,
        qo_tile_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        total_num_rows: *const u32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        seq_len: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn batch_prefill_paged_cuda_with_cta_tile_q(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        q_indptr: *const i32,
        request_indices: *const i32,
        qo_tile_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        total_num_rows: *const u32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        seq_len: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        cta_tile_q_override: i32,
        stream: CUstream,
    ) -> i32;

    // Single-request prefill with contiguous HND KV cache (FlashInfer SinglePrefill, causal).
    pub fn single_prefill_cuda(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    // Paged attention decode for HEAD_DIM=256 (Qwen3.5-4B full-attention layers).
    pub fn paged_attention_decode_cuda_hd256(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    // Batch prefill with paged KV for HEAD_DIM=256 (Qwen3.5-4B multi-token prefill).
    pub fn batch_prefill_paged_cuda_hd256(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        q_indptr: *const i32,
        request_indices: *const i32,
        qo_tile_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        total_num_rows: *const u32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        seq_len: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    // Paged attention decode (FlashInfer BatchDecode, no partition-KV).
    pub fn paged_attention_decode_cuda(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    // Paged attention decode (FlashInfer BatchDecode, partition-KV / split-K).
    pub fn paged_attention_decode_split_kv_cuda(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        o_indptr: *const i32,
        block_valid_mask: *const u8,
        tmp_v: *mut Half,
        tmp_s: *mut f32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

}
