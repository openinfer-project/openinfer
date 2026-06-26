use anyhow::{Result, ensure};

use super::{
    GLM52_INDEX_HEAD_DIM, GLM52_INDEX_HEADS, GLM52_QK_ROPE_HEAD_DIM, Glm52DecodeArena, bytes_bf16,
    bytes_f32, bytes_i32, bytes_i64, bytes_u8,
};

impl Glm52DecodeArena {
    pub(super) fn validate_allocations(&self) -> Result<()> {
        ensure!(
            self.hidden.len() == self.plan.batch_capacity * self.plan.hidden
                && self.normed.len() == self.plan.batch_capacity * self.plan.hidden
                && self.linear_input_fp8.len()
                    == self.plan.batch_capacity * self.plan.linear_quant_max_in
                && self.linear_input_scale.len()
                    == self.plan.batch_capacity * self.plan.linear_quant_scale_cols
                && self.attention_q_a.len()
                    == self.plan.batch_capacity * self.plan.attention_q_a_out
                && self.attention_q_a_normed.len()
                    == self.plan.batch_capacity * self.plan.attention_q_a_out
                && self.attention_q_b.len()
                    == self.plan.batch_capacity * self.plan.attention_q_b_out
                && self.attention_kv_a.len()
                    == self.plan.batch_capacity * self.plan.attention_kv_a_out
                && self.attention_kv_a_normed.len()
                    == self.plan.batch_capacity * self.plan.attention_kv_lora_rank
                && self.attention_k_rope.len() == self.plan.batch_capacity * GLM52_QK_ROPE_HEAD_DIM
                && self.attention_kv_b.len()
                    == self.plan.batch_capacity * self.plan.attention_kv_b_out
                && self.attention_out.len()
                    == self.plan.batch_capacity * self.plan.attention_o_proj_in
                && self.indexer_scores.len()
                    == self.plan.batch_capacity * self.plan.indexer_score_heads
                && self.indexer_wk.len() == self.plan.batch_capacity * GLM52_INDEX_HEAD_DIM
                && self.indexer_wq_b.len()
                    == self.plan.batch_capacity * (GLM52_INDEX_HEADS * GLM52_INDEX_HEAD_DIM)
                && self.indexer_topk_idx.len() == self.plan.batch_capacity * self.plan.indexer_topk
                && self.indexer_topk_weight.len()
                    == self.plan.batch_capacity * self.plan.indexer_topk
                && self.dense_gate_up.len()
                    == self.plan.batch_capacity * self.plan.dense_intermediate * 2
                && self.dense_activated.len()
                    == self.plan.batch_capacity * self.plan.dense_intermediate
                && self.shared_gate_up.len()
                    == self.plan.batch_capacity * self.plan.moe_intermediate * 2
                && self.shared_activated.len()
                    == self.plan.batch_capacity * self.plan.moe_intermediate
                && self.logits.len() == self.plan.batch_capacity * self.plan.vocab
                && self.router_logits.len() == self.plan.batch_capacity * self.plan.routed_experts
                && self.topk_idx.len() == self.plan.batch_capacity * self.plan.topk
                && self.topk_weight.len() == self.plan.batch_capacity * self.plan.topk
                && self.deepep_recv_x.len()
                    == self.plan.deepep_worst_expanded_tokens * self.plan.hidden
                && self.moe_w13_input_fp8.len()
                    == self.plan.deepep_worst_expanded_tokens * self.plan.hidden
                && self.moe_w13_input_scale.len()
                    == self.plan.deepep_worst_expanded_tokens * self.plan.moe_w13_scale_cols
                && self.moe_w13_input_scale_tma.len()
                    == self.plan.moe_w13_scale_tma_aligned_rows * self.plan.moe_w13_scale_cols
                && self.moe_w13_input_scale_trtllm_offset_tma.len()
                    == self.plan.moe_trtllm_grouped_offset_rows * self.plan.moe_w13_scale_cols
                && self.moe_w13_output_bf16.len()
                    == self.plan.deepep_worst_expanded_tokens * self.plan.moe_intermediate * 2
                && self.moe_w2_input_fp8.len()
                    == self.plan.deepep_worst_expanded_tokens * self.plan.moe_intermediate
                && self.moe_w2_input_scale.len()
                    == self.plan.deepep_worst_expanded_tokens * self.plan.moe_w2_scale_cols
                && self.moe_w2_input_scale_tma.len()
                    == self.plan.moe_w2_scale_tma_aligned_rows * self.plan.moe_w2_scale_cols
                && self.moe_w2_input_scale_trtllm_offset_tma.len()
                    == self.plan.moe_trtllm_grouped_offset_rows * self.plan.moe_w2_scale_cols
                && self.moe_w2_output_bf16.len()
                    == self.plan.deepep_worst_expanded_tokens * self.plan.hidden
                && self.moe_gemm_expert_offsets.len() == self.plan.moe_gemm_expert_offsets_len
                && self.moe_w13_problem_sizes.len() == self.plan.moe_gemm_problem_sizes_len
                && self.moe_w2_problem_sizes.len() == self.plan.moe_gemm_problem_sizes_len
                && self.deepep_recv_topk_weight.len() == self.plan.deepep_worst_expanded_tokens
                && self.deepep_recv_src_metadata.len() == self.plan.deepep_src_metadata_len
                && self.deepep_combined.len() == self.plan.batch_capacity * self.plan.hidden,
            "GLM5.2 decode arena allocation shape does not match plan: {:?}",
            self.plan
        );
        ensure!(
            self.allocated_bytes() == self.plan.total_bytes,
            "GLM5.2 decode arena byte accounting drifted: allocated={}, plan={}",
            self.allocated_bytes(),
            self.plan.total_bytes
        );
        Ok(())
    }

    fn allocated_bytes(&self) -> usize {
        bytes_bf16(self.hidden.len())
            + bytes_bf16(self.normed.len())
            + bytes_u8(self.linear_input_fp8.len())
            + bytes_f32(self.linear_input_scale.len())
            + bytes_bf16(self.attention_q_a.len())
            + bytes_bf16(self.attention_q_a_normed.len())
            + bytes_bf16(self.attention_q_b.len())
            + bytes_bf16(self.attention_kv_a.len())
            + bytes_bf16(self.attention_kv_a_normed.len())
            + bytes_bf16(self.attention_k_rope.len())
            + bytes_bf16(self.attention_kv_b.len())
            + bytes_bf16(self.attention_out.len())
            + bytes_bf16(self.indexer_scores.len())
            + bytes_bf16(self.indexer_wk.len())
            + bytes_bf16(self.indexer_wq_b.len())
            + bytes_i32(self.indexer_topk_idx.len())
            + bytes_f32(self.indexer_topk_weight.len())
            + bytes_bf16(self.dense_gate_up.len())
            + bytes_bf16(self.dense_activated.len())
            + bytes_bf16(self.shared_gate_up.len())
            + bytes_bf16(self.shared_activated.len())
            + bytes_bf16(self.logits.len())
            + bytes_f32(self.router_logits.len())
            + bytes_i32(self.topk_idx.len())
            + bytes_f32(self.topk_weight.len())
            + bytes_bf16(self.deepep_recv_x.len())
            + bytes_u8(self.moe_w13_input_fp8.len())
            + bytes_f32(self.moe_w13_input_scale.len())
            + bytes_f32(self.moe_w13_input_scale_tma.len())
            + bytes_f32(self.moe_w13_input_scale_trtllm_offset_tma.len())
            + bytes_bf16(self.moe_w13_output_bf16.len())
            + bytes_u8(self.moe_w2_input_fp8.len())
            + bytes_f32(self.moe_w2_input_scale.len())
            + bytes_f32(self.moe_w2_input_scale_tma.len())
            + bytes_f32(self.moe_w2_input_scale_trtllm_offset_tma.len())
            + bytes_bf16(self.moe_w2_output_bf16.len())
            + bytes_i64(self.moe_gemm_expert_offsets.len())
            + bytes_i32(self.moe_w13_problem_sizes.len())
            + bytes_i32(self.moe_w2_problem_sizes.len())
            + bytes_f32(self.deepep_recv_topk_weight.len())
            + bytes_i32(self.deepep_recv_src_metadata.len())
            + bytes_bf16(self.deepep_combined.len())
    }
}
