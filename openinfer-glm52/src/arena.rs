use anyhow::{Result, ensure};
use bytesize::ByteSize;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::{
    glm52_deepgemm_tma_aligned_rows, glm52_trtllm_grouped_offset_padded_rows,
};

use crate::{
    config::{
        GLM52_DENSE_INTERMEDIATE, GLM52_EXPERT_INTERMEDIATE, GLM52_INDEX_HEAD_DIM,
        GLM52_INDEX_HEADS, GLM52_INDEX_TOPK, GLM52_KV_A_OUT, GLM52_KV_B_OUT, GLM52_KV_LORA_RANK,
        GLM52_O_PROJ_IN, GLM52_Q_B_OUT, GLM52_Q_LORA_RANK, GLM52_QK_ROPE_HEAD_DIM, GLM52_VOCAB,
    },
    deepep::{GLM52_EP_WORLD, Glm52DeepEpShape},
    weights::Glm52RankGpuContext,
};

mod validation;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DecodeArenaPlan {
    pub(crate) batch_capacity: usize,
    pub(crate) hidden: usize,
    pub(crate) routed_experts: usize,
    pub(crate) topk: usize,
    pub(crate) local_experts: usize,
    pub(crate) expert_alignment: usize,
    pub(crate) dense_intermediate: usize,
    pub(crate) moe_intermediate: usize,
    pub(crate) vocab: usize,
    pub(crate) linear_quant_max_in: usize,
    pub(crate) linear_quant_scale_cols: usize,
    pub(crate) attention_q_a_out: usize,
    pub(crate) attention_q_b_out: usize,
    pub(crate) attention_kv_a_out: usize,
    pub(crate) attention_kv_lora_rank: usize,
    pub(crate) attention_kv_b_out: usize,
    pub(crate) attention_o_proj_in: usize,
    pub(crate) indexer_score_heads: usize,
    pub(crate) indexer_topk: usize,
    pub(crate) moe_quant_group_size: usize,
    pub(crate) moe_w13_scale_cols: usize,
    pub(crate) moe_w2_scale_cols: usize,
    pub(crate) moe_w13_scale_tma_aligned_rows: usize,
    pub(crate) moe_w2_scale_tma_aligned_rows: usize,
    pub(crate) moe_trtllm_grouped_offset_rows: usize,
    pub(crate) moe_gemm_expert_offsets_len: usize,
    pub(crate) moe_gemm_problem_sizes_len: usize,
    pub(crate) deepep_worst_recv_tokens: usize,
    pub(crate) deepep_worst_expanded_tokens: usize,
    pub(crate) deepep_src_metadata_len: usize,
    pub(crate) total_bytes: usize,
}

impl Glm52DecodeArenaPlan {
    pub(crate) fn tp1_dp8() -> Result<Self> {
        let deepep = Glm52DeepEpShape::tp1_dp8_h200().decode_capacity()?;
        let shape = deepep.shape;
        let moe_quant_group_size = 128;
        let linear_quant_max_in = GLM52_O_PROJ_IN;
        let linear_quant_scale_cols = linear_quant_max_in / moe_quant_group_size;
        let moe_w13_scale_cols = shape.hidden / moe_quant_group_size;
        let moe_w2_scale_cols = GLM52_EXPERT_INTERMEDIATE / moe_quant_group_size;
        let moe_w13_scale_tma_aligned_rows =
            glm52_deepgemm_tma_aligned_rows(deepep.worst_expanded_tokens);
        let moe_w2_scale_tma_aligned_rows =
            glm52_deepgemm_tma_aligned_rows(deepep.worst_expanded_tokens);
        let moe_trtllm_grouped_offset_rows = glm52_trtllm_grouped_offset_padded_rows(
            deepep.worst_expanded_tokens,
            shape.local_experts,
        );
        let moe_gemm_expert_offsets_len = shape.local_experts + 1;
        let moe_gemm_problem_sizes_len = shape.local_experts * 3;
        let batch = shape.decode_max_tokens_per_rank;
        let total_bytes = bytes_bf16(batch * shape.hidden)
            + bytes_bf16(batch * shape.hidden)
            + bytes_u8(batch * linear_quant_max_in)
            + bytes_f32(batch * linear_quant_scale_cols)
            + bytes_bf16(batch * GLM52_Q_LORA_RANK)
            + bytes_bf16(batch * GLM52_Q_LORA_RANK)
            + bytes_bf16(batch * GLM52_Q_B_OUT)
            + bytes_bf16(batch * GLM52_KV_A_OUT)
            + bytes_bf16(batch * GLM52_KV_LORA_RANK)
            + bytes_bf16(batch * GLM52_QK_ROPE_HEAD_DIM)
            + bytes_bf16(batch * GLM52_KV_B_OUT)
            + bytes_bf16(batch * GLM52_O_PROJ_IN)
            + bytes_bf16(batch * GLM52_INDEX_HEADS)
            + bytes_bf16(batch * GLM52_INDEX_HEAD_DIM)
            + bytes_bf16(batch * (GLM52_INDEX_HEADS * GLM52_INDEX_HEAD_DIM))
            + bytes_i32(batch * GLM52_INDEX_TOPK)
            + bytes_f32(batch * GLM52_INDEX_TOPK)
            + bytes_bf16(batch * GLM52_DENSE_INTERMEDIATE * 2)
            + bytes_bf16(batch * GLM52_DENSE_INTERMEDIATE)
            + bytes_bf16(batch * GLM52_EXPERT_INTERMEDIATE * 2)
            + bytes_bf16(batch * GLM52_EXPERT_INTERMEDIATE)
            + bytes_bf16(batch * GLM52_VOCAB)
            + bytes_bf16(deepep.worst_expanded_tokens * shape.hidden)
            + bytes_u8(deepep.worst_expanded_tokens * shape.hidden)
            + bytes_f32(deepep.worst_expanded_tokens * moe_w13_scale_cols)
            + bytes_f32(moe_w13_scale_tma_aligned_rows * moe_w13_scale_cols)
            + bytes_f32(moe_trtllm_grouped_offset_rows * moe_w13_scale_cols)
            + bytes_bf16(deepep.worst_expanded_tokens * GLM52_EXPERT_INTERMEDIATE * 2)
            + bytes_u8(deepep.worst_expanded_tokens * GLM52_EXPERT_INTERMEDIATE)
            + bytes_f32(deepep.worst_expanded_tokens * moe_w2_scale_cols)
            + bytes_f32(moe_w2_scale_tma_aligned_rows * moe_w2_scale_cols)
            + bytes_f32(moe_trtllm_grouped_offset_rows * moe_w2_scale_cols)
            + bytes_bf16(deepep.worst_expanded_tokens * shape.hidden)
            + bytes_f32(shape.decode_max_tokens_per_rank * shape.routed_experts)
            + bytes_i32(shape.decode_max_tokens_per_rank * shape.topk)
            + bytes_f32(shape.decode_max_tokens_per_rank * shape.topk)
            + bytes_i64(moe_gemm_expert_offsets_len)
            + bytes_i32(moe_gemm_problem_sizes_len) * 2
            + bytes_f32(deepep.worst_expanded_tokens)
            + bytes_i32(deepep.src_metadata_len)
            + bytes_bf16(batch * shape.hidden);
        Ok(Self {
            batch_capacity: shape.decode_max_tokens_per_rank,
            hidden: shape.hidden,
            routed_experts: shape.routed_experts,
            topk: shape.topk,
            local_experts: shape.local_experts,
            expert_alignment: shape.expert_alignment,
            dense_intermediate: GLM52_DENSE_INTERMEDIATE,
            moe_intermediate: GLM52_EXPERT_INTERMEDIATE,
            vocab: GLM52_VOCAB,
            linear_quant_max_in,
            linear_quant_scale_cols,
            attention_q_a_out: GLM52_Q_LORA_RANK,
            attention_q_b_out: GLM52_Q_B_OUT,
            attention_kv_a_out: GLM52_KV_A_OUT,
            attention_kv_lora_rank: GLM52_KV_LORA_RANK,
            attention_kv_b_out: GLM52_KV_B_OUT,
            attention_o_proj_in: GLM52_O_PROJ_IN,
            indexer_score_heads: GLM52_INDEX_HEADS,
            indexer_topk: GLM52_INDEX_TOPK,
            moe_quant_group_size,
            moe_w13_scale_cols,
            moe_w2_scale_cols,
            moe_w13_scale_tma_aligned_rows,
            moe_w2_scale_tma_aligned_rows,
            moe_trtllm_grouped_offset_rows,
            moe_gemm_expert_offsets_len,
            moe_gemm_problem_sizes_len,
            deepep_worst_recv_tokens: deepep.worst_recv_tokens,
            deepep_worst_expanded_tokens: deepep.worst_expanded_tokens,
            deepep_src_metadata_len: deepep.src_metadata_len,
            total_bytes,
        })
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let deepep = Glm52DeepEpShape::tp1_dp8_h200().decode_capacity()?;
        ensure!(
            self.batch_capacity == deepep.shape.decode_max_tokens_per_rank
                && self.hidden == deepep.shape.hidden
                && self.routed_experts == deepep.shape.routed_experts
                && self.topk == deepep.shape.topk
                && self.local_experts == deepep.shape.local_experts,
            "GLM5.2 decode arena plan drifted from TP1/DP8/EP8 constants: {self:?}"
        );
        ensure!(
            self.dense_intermediate == GLM52_DENSE_INTERMEDIATE
                && self.moe_intermediate == GLM52_EXPERT_INTERMEDIATE
                && self.vocab == GLM52_VOCAB
                && self.linear_quant_max_in == GLM52_O_PROJ_IN
                && self
                    .linear_quant_max_in
                    .is_multiple_of(self.moe_quant_group_size)
                && self.linear_quant_scale_cols
                    == self.linear_quant_max_in / self.moe_quant_group_size
                && self.attention_q_a_out == GLM52_Q_LORA_RANK
                && self.attention_q_b_out == GLM52_Q_B_OUT
                && self.attention_kv_a_out == GLM52_KV_A_OUT
                && self.attention_kv_lora_rank == GLM52_KV_LORA_RANK
                && self.attention_kv_b_out == GLM52_KV_B_OUT
                && self.attention_o_proj_in == GLM52_O_PROJ_IN
                && self.indexer_score_heads == GLM52_INDEX_HEADS
                && self.indexer_topk == GLM52_INDEX_TOPK,
            "GLM5.2 decode arena full-layer plan drifted: {self:?}"
        );
        ensure!(
            self.moe_intermediate == GLM52_EXPERT_INTERMEDIATE
                && self.moe_quant_group_size == 128
                && self.hidden.is_multiple_of(self.moe_quant_group_size)
                && self
                    .moe_intermediate
                    .is_multiple_of(self.moe_quant_group_size)
                && self.moe_w13_scale_cols == self.hidden / self.moe_quant_group_size
                && self.moe_w2_scale_cols == self.moe_intermediate / self.moe_quant_group_size
                && self.moe_w13_scale_tma_aligned_rows
                    == glm52_deepgemm_tma_aligned_rows(self.deepep_worst_expanded_tokens)
                && self.moe_w2_scale_tma_aligned_rows
                    == glm52_deepgemm_tma_aligned_rows(self.deepep_worst_expanded_tokens)
                && self.moe_trtllm_grouped_offset_rows
                    == glm52_trtllm_grouped_offset_padded_rows(
                        self.deepep_worst_expanded_tokens,
                        self.local_experts
                    )
                && self.moe_gemm_expert_offsets_len == self.local_experts + 1
                && self.moe_gemm_problem_sizes_len == self.local_experts * 3,
            "GLM5.2 decode arena MoE quant plan drifted: {self:?}"
        );
        ensure!(
            self.deepep_worst_recv_tokens == deepep.worst_recv_tokens,
            "GLM5.2 decode arena recv-token cap is inconsistent: {self:?}"
        );
        ensure!(
            self.deepep_worst_expanded_tokens == deepep.worst_expanded_tokens
                && self.deepep_src_metadata_len == deepep.src_metadata_len,
            "GLM5.2 decode arena DeepEP capacities drifted from the model contract: {self:?}"
        );
        ensure!(
            self.deepep_worst_expanded_tokens
                .is_multiple_of(self.expert_alignment),
            "GLM5.2 decode arena expanded rows must be expert-aligned: {self:?}"
        );
        ensure!(
            self.total_bytes > 0,
            "GLM5.2 decode arena plan has zero bytes"
        );
        Ok(())
    }
}

pub(crate) struct Glm52DecodeArena {
    pub(crate) plan: Glm52DecodeArenaPlan,
    pub(crate) hidden: CudaSlice<bf16>,
    pub(crate) normed: CudaSlice<bf16>,
    pub(crate) linear_input_fp8: CudaSlice<u8>,
    pub(crate) linear_input_scale: CudaSlice<f32>,
    pub(crate) attention_q_a: CudaSlice<bf16>,
    pub(crate) attention_q_a_normed: CudaSlice<bf16>,
    pub(crate) attention_q_b: CudaSlice<bf16>,
    pub(crate) attention_kv_a: CudaSlice<bf16>,
    pub(crate) attention_kv_a_normed: CudaSlice<bf16>,
    pub(crate) attention_k_rope: CudaSlice<bf16>,
    pub(crate) attention_kv_b: CudaSlice<bf16>,
    pub(crate) attention_out: CudaSlice<bf16>,
    pub(crate) indexer_scores: CudaSlice<bf16>,
    pub(crate) indexer_wk: CudaSlice<bf16>,
    pub(crate) indexer_wq_b: CudaSlice<bf16>,
    pub(crate) indexer_topk_idx: CudaSlice<i32>,
    pub(crate) indexer_topk_weight: CudaSlice<f32>,
    pub(crate) dense_gate_up: CudaSlice<bf16>,
    pub(crate) dense_activated: CudaSlice<bf16>,
    pub(crate) shared_gate_up: CudaSlice<bf16>,
    pub(crate) shared_activated: CudaSlice<bf16>,
    pub(crate) logits: CudaSlice<bf16>,
    pub(crate) router_logits: CudaSlice<f32>,
    pub(crate) topk_idx: CudaSlice<i32>,
    pub(crate) topk_weight: CudaSlice<f32>,
    pub(crate) deepep_recv_x: CudaSlice<bf16>,
    pub(crate) moe_w13_input_fp8: CudaSlice<u8>,
    pub(crate) moe_w13_input_scale: CudaSlice<f32>,
    pub(crate) moe_w13_input_scale_tma: CudaSlice<f32>,
    pub(crate) moe_w13_input_scale_trtllm_offset_tma: CudaSlice<f32>,
    pub(crate) moe_w13_output_bf16: CudaSlice<bf16>,
    pub(crate) moe_w2_input_fp8: CudaSlice<u8>,
    pub(crate) moe_w2_input_scale: CudaSlice<f32>,
    pub(crate) moe_w2_input_scale_tma: CudaSlice<f32>,
    pub(crate) moe_w2_input_scale_trtllm_offset_tma: CudaSlice<f32>,
    pub(crate) moe_w2_output_bf16: CudaSlice<bf16>,
    pub(crate) moe_gemm_expert_offsets: CudaSlice<i64>,
    pub(crate) moe_w13_problem_sizes: CudaSlice<i32>,
    pub(crate) moe_w2_problem_sizes: CudaSlice<i32>,
    pub(crate) deepep_recv_topk_weight: CudaSlice<f32>,
    pub(crate) deepep_recv_src_metadata: CudaSlice<i32>,
    pub(crate) deepep_combined: CudaSlice<bf16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52MoeGemmMetadataValidation {
    pub(crate) offsets_valid: bool,
    pub(crate) w13_problem_sizes_valid: bool,
    pub(crate) w2_problem_sizes_valid: bool,
    pub(crate) active_experts: usize,
    pub(crate) expanded_rows: usize,
    pub(crate) trtllm_grouped_offset_scale_rows_required: usize,
    pub(crate) trtllm_grouped_offset_scale_rows_covered: bool,
}

impl Glm52DecodeArena {
    pub(crate) fn new(ctx: &Glm52RankGpuContext) -> Result<Self> {
        ctx.set_current()?;
        let plan = Glm52DecodeArenaPlan::tp1_dp8()?;
        plan.validate()?;
        let stream = ctx.stream();
        let arena = Self {
            plan,
            hidden: stream.alloc_zeros(plan.batch_capacity * plan.hidden)?,
            normed: stream.alloc_zeros(plan.batch_capacity * plan.hidden)?,
            linear_input_fp8: stream.alloc_zeros(plan.batch_capacity * plan.linear_quant_max_in)?,
            linear_input_scale: stream
                .alloc_zeros(plan.batch_capacity * plan.linear_quant_scale_cols)?,
            attention_q_a: stream.alloc_zeros(plan.batch_capacity * plan.attention_q_a_out)?,
            attention_q_a_normed: stream
                .alloc_zeros(plan.batch_capacity * plan.attention_q_a_out)?,
            attention_q_b: stream.alloc_zeros(plan.batch_capacity * plan.attention_q_b_out)?,
            attention_kv_a: stream.alloc_zeros(plan.batch_capacity * plan.attention_kv_a_out)?,
            attention_kv_a_normed: stream
                .alloc_zeros(plan.batch_capacity * plan.attention_kv_lora_rank)?,
            attention_k_rope: stream.alloc_zeros(plan.batch_capacity * GLM52_QK_ROPE_HEAD_DIM)?,
            attention_kv_b: stream.alloc_zeros(plan.batch_capacity * plan.attention_kv_b_out)?,
            attention_out: stream.alloc_zeros(plan.batch_capacity * plan.attention_o_proj_in)?,
            indexer_scores: stream.alloc_zeros(plan.batch_capacity * plan.indexer_score_heads)?,
            indexer_wk: stream.alloc_zeros(plan.batch_capacity * GLM52_INDEX_HEAD_DIM)?,
            indexer_wq_b: stream
                .alloc_zeros(plan.batch_capacity * (GLM52_INDEX_HEADS * GLM52_INDEX_HEAD_DIM))?,
            indexer_topk_idx: stream.alloc_zeros(plan.batch_capacity * plan.indexer_topk)?,
            indexer_topk_weight: stream.alloc_zeros(plan.batch_capacity * plan.indexer_topk)?,
            dense_gate_up: stream.alloc_zeros(plan.batch_capacity * plan.dense_intermediate * 2)?,
            dense_activated: stream.alloc_zeros(plan.batch_capacity * plan.dense_intermediate)?,
            shared_gate_up: stream.alloc_zeros(plan.batch_capacity * plan.moe_intermediate * 2)?,
            shared_activated: stream.alloc_zeros(plan.batch_capacity * plan.moe_intermediate)?,
            logits: stream.alloc_zeros(plan.batch_capacity * plan.vocab)?,
            router_logits: stream.alloc_zeros(plan.batch_capacity * plan.routed_experts)?,
            topk_idx: stream.alloc_zeros(plan.batch_capacity * plan.topk)?,
            topk_weight: stream.alloc_zeros(plan.batch_capacity * plan.topk)?,
            deepep_recv_x: stream.alloc_zeros(plan.deepep_worst_expanded_tokens * plan.hidden)?,
            moe_w13_input_fp8: stream
                .alloc_zeros(plan.deepep_worst_expanded_tokens * plan.hidden)?,
            moe_w13_input_scale: stream
                .alloc_zeros(plan.deepep_worst_expanded_tokens * plan.moe_w13_scale_cols)?,
            moe_w13_input_scale_tma: stream
                .alloc_zeros(plan.moe_w13_scale_tma_aligned_rows * plan.moe_w13_scale_cols)?,
            moe_w13_input_scale_trtllm_offset_tma: stream
                .alloc_zeros(plan.moe_trtllm_grouped_offset_rows * plan.moe_w13_scale_cols)?,
            moe_w13_output_bf16: stream
                .alloc_zeros(plan.deepep_worst_expanded_tokens * plan.moe_intermediate * 2)?,
            moe_w2_input_fp8: stream
                .alloc_zeros(plan.deepep_worst_expanded_tokens * plan.moe_intermediate)?,
            moe_w2_input_scale: stream
                .alloc_zeros(plan.deepep_worst_expanded_tokens * plan.moe_w2_scale_cols)?,
            moe_w2_input_scale_tma: stream
                .alloc_zeros(plan.moe_w2_scale_tma_aligned_rows * plan.moe_w2_scale_cols)?,
            moe_w2_input_scale_trtllm_offset_tma: stream
                .alloc_zeros(plan.moe_trtllm_grouped_offset_rows * plan.moe_w2_scale_cols)?,
            moe_w2_output_bf16: stream
                .alloc_zeros(plan.deepep_worst_expanded_tokens * plan.hidden)?,
            moe_gemm_expert_offsets: stream.alloc_zeros(plan.moe_gemm_expert_offsets_len)?,
            moe_w13_problem_sizes: stream.alloc_zeros(plan.moe_gemm_problem_sizes_len)?,
            moe_w2_problem_sizes: stream.alloc_zeros(plan.moe_gemm_problem_sizes_len)?,
            deepep_recv_topk_weight: stream.alloc_zeros(plan.deepep_worst_expanded_tokens)?,
            deepep_recv_src_metadata: stream.alloc_zeros(plan.deepep_src_metadata_len)?,
            deepep_combined: stream.alloc_zeros(plan.batch_capacity * plan.hidden)?,
        };
        arena.validate_allocations()?;
        log::debug!(
            "GLM5.2 decode arena allocated: batch_cap={}, worst_recv={}, worst_expanded={}, bytes={}",
            arena.plan.batch_capacity,
            arena.plan.deepep_worst_recv_tokens,
            arena.plan.deepep_worst_expanded_tokens,
            ByteSize(arena.plan.total_bytes as u64)
        );
        Ok(arena)
    }

    pub(crate) fn total_bytes(&self) -> usize {
        self.plan.total_bytes
    }

    pub(crate) fn seed_deepep_decode_smoke_routes(
        &mut self,
        ctx: &Glm52RankGpuContext,
        num_tokens: usize,
    ) -> Result<()> {
        ensure!(
            (1..=self.plan.batch_capacity).contains(&num_tokens),
            "GLM5.2 DeepEP decode smoke tokens {num_tokens} out of 1..={}",
            self.plan.batch_capacity
        );
        let route_len = num_tokens * self.plan.topk;
        let mut topk_idx = Vec::with_capacity(route_len);
        let topk_weight = vec![1.0f32 / self.plan.topk as f32; route_len];
        for _ in 0..num_tokens {
            for ep_rank in 0..GLM52_EP_WORLD {
                topk_idx.push((ep_rank * self.plan.local_experts) as i32);
            }
        }
        let stream = ctx.stream();
        stream.memcpy_htod(&topk_idx, &mut self.topk_idx.slice_mut(0..route_len))?;
        stream.memcpy_htod(&topk_weight, &mut self.topk_weight.slice_mut(0..route_len))?;
        Ok(())
    }

    pub(crate) fn seed_router_smoke_hidden(
        &mut self,
        ctx: &Glm52RankGpuContext,
        num_tokens: usize,
    ) -> Result<()> {
        ensure!(
            (1..=self.plan.batch_capacity).contains(&num_tokens),
            "GLM5.2 router smoke tokens {num_tokens} out of 1..={}",
            self.plan.batch_capacity
        );
        let elems = num_tokens * self.plan.hidden;
        let mut hidden = Vec::with_capacity(elems);
        for token in 0..num_tokens {
            for dim in 0..self.plan.hidden {
                let value = ((token + 1) * ((dim % 17) + 1)) as f32 * 0.0001;
                hidden.push(bf16::from_f32(value));
            }
        }
        ctx.stream()
            .memcpy_htod(&hidden, &mut self.hidden.slice_mut(0..elems))?;
        Ok(())
    }

    pub(crate) fn seed_moe_quant_smoke_inputs(
        &mut self,
        ctx: &Glm52RankGpuContext,
        rows: usize,
    ) -> Result<()> {
        self.validate_moe_quant_rows(rows)?;
        let mut recv_x = Vec::with_capacity(rows * self.plan.hidden);
        for row in 0..rows {
            for dim in 0..self.plan.hidden {
                let value = ((row + 1) * ((dim % 31) + 1)) as f32 * 0.0002;
                recv_x.push(bf16::from_f32(value));
            }
        }
        let stream = ctx.stream();
        stream.memcpy_htod(&recv_x, &mut self.deepep_recv_x.slice_mut(0..recv_x.len()))?;
        self.seed_moe_w13_output_smoke(ctx, rows)?;
        Ok(())
    }

    pub(crate) fn seed_moe_w13_output_smoke(
        &mut self,
        ctx: &Glm52RankGpuContext,
        rows: usize,
    ) -> Result<()> {
        self.validate_moe_quant_rows(rows)?;
        let mut w13_output = Vec::with_capacity(rows * self.plan.moe_intermediate * 2);
        for row in 0..rows {
            for dim in 0..(self.plan.moe_intermediate * 2) {
                let value = ((row + 3) * ((dim % 23) + 1)) as f32 * 0.0003;
                w13_output.push(bf16::from_f32(value));
            }
        }
        ctx.stream().memcpy_htod(
            &w13_output,
            &mut self.moe_w13_output_bf16.slice_mut(0..w13_output.len()),
        )?;
        Ok(())
    }

    pub(crate) fn validate_moe_quant_smoke_outputs(
        &self,
        ctx: &Glm52RankGpuContext,
        rows: usize,
    ) -> Result<(bool, bool)> {
        self.validate_moe_quant_rows(rows)?;
        let w13_scale_len = rows * self.plan.moe_w13_scale_cols;
        let w13_quant_len = rows * self.plan.hidden;
        let w2_scale_len = rows * self.plan.moe_w2_scale_cols;
        let w2_quant_len = rows * self.plan.moe_intermediate;
        let stream = ctx.stream();
        let w13_scales = stream.clone_dtoh(&self.moe_w13_input_scale.slice(0..w13_scale_len))?;
        let w13_quant = stream.clone_dtoh(&self.moe_w13_input_fp8.slice(0..w13_quant_len))?;
        let w2_scales = stream.clone_dtoh(&self.moe_w2_input_scale.slice(0..w2_scale_len))?;
        let w2_quant = stream.clone_dtoh(&self.moe_w2_input_fp8.slice(0..w2_quant_len))?;
        Ok((
            quant_output_valid(&w13_scales, &w13_quant),
            quant_output_valid(&w2_scales, &w2_quant),
        ))
    }

    pub(crate) fn validate_weighted_swiglu_scale_output(
        &self,
        ctx: &Glm52RankGpuContext,
        rows: usize,
    ) -> Result<bool> {
        self.validate_moe_quant_rows(rows)?;
        let stream = ctx.stream();
        let weights = stream.clone_dtoh(&self.deepep_recv_topk_weight.slice(0..rows))?;
        let Some((row, weight)) = weights
            .iter()
            .copied()
            .enumerate()
            .find(|(_, weight)| weight.is_finite() && *weight > 0.0)
        else {
            return Ok(false);
        };

        let group_size = self.plan.moe_quant_group_size;
        let input_stride = self.plan.moe_intermediate * 2;
        let gate_offset = row * input_stride;
        let up_offset = gate_offset + self.plan.moe_intermediate;
        let scale_offset = row * self.plan.moe_w2_scale_cols;
        let gate = stream.clone_dtoh(
            &self
                .moe_w13_output_bf16
                .slice(gate_offset..gate_offset + group_size),
        )?;
        let up = stream.clone_dtoh(
            &self
                .moe_w13_output_bf16
                .slice(up_offset..up_offset + group_size),
        )?;
        let actual = stream.clone_dtoh(
            &self
                .moe_w2_input_scale
                .slice(scale_offset..scale_offset + 1),
        )?[0];

        const FP8_MAX: f32 = 448.0;
        const MIN_SWIGLU_SCALE: f32 = 1.0 / (FP8_MAX * 512.0);
        let max_abs = gate
            .iter()
            .zip(up.iter())
            .map(|(gate, up)| {
                let gate = gate.to_f32();
                let up = up.to_f32();
                let sigmoid = 1.0 / (1.0 + (-gate).exp());
                (gate * sigmoid * up * weight).abs()
            })
            .fold(0.0f32, f32::max);
        let expected = (max_abs / FP8_MAX).max(MIN_SWIGLU_SCALE);
        let tolerance = (expected.abs() * 1.0e-3).max(1.0e-7);
        Ok(actual.is_finite() && (actual - expected).abs() <= tolerance)
    }

    pub(crate) fn validate_deepgemm_scale_layout_outputs(
        &self,
        ctx: &Glm52RankGpuContext,
        rows: usize,
    ) -> Result<(bool, bool)> {
        self.validate_moe_quant_rows(rows)?;
        let aligned_rows = glm52_deepgemm_tma_aligned_rows(rows);
        let stream = ctx.stream();

        let w13_scale_len = rows * self.plan.moe_w13_scale_cols;
        let w13_tma_len = aligned_rows * self.plan.moe_w13_scale_cols;
        let w13_scales = stream.clone_dtoh(&self.moe_w13_input_scale.slice(0..w13_scale_len))?;
        let w13_tma = stream.clone_dtoh(&self.moe_w13_input_scale_tma.slice(0..w13_tma_len))?;

        let w2_scale_len = rows * self.plan.moe_w2_scale_cols;
        let w2_tma_len = aligned_rows * self.plan.moe_w2_scale_cols;
        let w2_scales = stream.clone_dtoh(&self.moe_w2_input_scale.slice(0..w2_scale_len))?;
        let w2_tma = stream.clone_dtoh(&self.moe_w2_input_scale_tma.slice(0..w2_tma_len))?;

        Ok((
            deepgemm_scale_layout_valid(
                &w13_scales,
                &w13_tma,
                rows,
                self.plan.moe_w13_scale_cols,
                aligned_rows,
            ),
            deepgemm_scale_layout_valid(
                &w2_scales,
                &w2_tma,
                rows,
                self.plan.moe_w2_scale_cols,
                aligned_rows,
            ),
        ))
    }

    pub(crate) fn validate_trtllm_grouped_offset_scale_layout_outputs(
        &self,
        ctx: &Glm52RankGpuContext,
    ) -> Result<(bool, bool)> {
        let stream = ctx.stream();
        let expert_offsets = stream.clone_dtoh(
            &self
                .moe_gemm_expert_offsets
                .slice(0..self.plan.moe_gemm_expert_offsets_len),
        )?;
        let w13_scale_len = self.plan.deepep_worst_expanded_tokens * self.plan.moe_w13_scale_cols;
        let w13_offset_len =
            self.plan.moe_trtllm_grouped_offset_rows * self.plan.moe_w13_scale_cols;
        let w13_scales = stream.clone_dtoh(&self.moe_w13_input_scale.slice(0..w13_scale_len))?;
        let w13_offset = stream.clone_dtoh(
            &self
                .moe_w13_input_scale_trtllm_offset_tma
                .slice(0..w13_offset_len),
        )?;

        let w2_scale_len = self.plan.deepep_worst_expanded_tokens * self.plan.moe_w2_scale_cols;
        let w2_offset_len = self.plan.moe_trtllm_grouped_offset_rows * self.plan.moe_w2_scale_cols;
        let w2_scales = stream.clone_dtoh(&self.moe_w2_input_scale.slice(0..w2_scale_len))?;
        let w2_offset = stream.clone_dtoh(
            &self
                .moe_w2_input_scale_trtllm_offset_tma
                .slice(0..w2_offset_len),
        )?;

        Ok((
            trtllm_grouped_offset_scale_layout_valid(
                &w13_scales,
                &w13_offset,
                &expert_offsets,
                self.plan.deepep_worst_expanded_tokens,
                self.plan.moe_w13_scale_cols,
                self.plan.moe_trtllm_grouped_offset_rows,
            ),
            trtllm_grouped_offset_scale_layout_valid(
                &w2_scales,
                &w2_offset,
                &expert_offsets,
                self.plan.deepep_worst_expanded_tokens,
                self.plan.moe_w2_scale_cols,
                self.plan.moe_trtllm_grouped_offset_rows,
            ),
        ))
    }

    pub(crate) fn validate_moe_gemm_metadata_outputs(
        &self,
        ctx: &Glm52RankGpuContext,
        psum_expert: &[i32],
    ) -> Result<Glm52MoeGemmMetadataValidation> {
        ensure!(
            psum_expert.len() == self.plan.local_experts,
            "GLM5.2 MoE GEMM metadata validation expected {} psum entries, got {}",
            self.plan.local_experts,
            psum_expert.len()
        );
        let stream = ctx.stream();
        let expert_offsets = stream.clone_dtoh(
            &self
                .moe_gemm_expert_offsets
                .slice(0..self.plan.moe_gemm_expert_offsets_len),
        )?;
        let w13_problem_sizes = stream.clone_dtoh(
            &self
                .moe_w13_problem_sizes
                .slice(0..self.plan.moe_gemm_problem_sizes_len),
        )?;
        let w2_problem_sizes = stream.clone_dtoh(
            &self
                .moe_w2_problem_sizes
                .slice(0..self.plan.moe_gemm_problem_sizes_len),
        )?;

        let mut previous_end = 0usize;
        let mut active_experts = 0usize;
        let mut offsets_valid = true;
        let mut w13_problem_sizes_valid = true;
        let mut w2_problem_sizes_valid = true;
        for (expert, raw_end) in psum_expert.iter().copied().enumerate() {
            offsets_valid &= raw_end >= 0;
            let end = raw_end.max(0) as usize;
            let start = if expert == 0 {
                0
            } else {
                align_up(previous_end, self.plan.expert_alignment)
            };
            let m = if end >= start {
                end - start
            } else {
                offsets_valid = false;
                0
            };
            offsets_valid &= start <= self.plan.deepep_worst_expanded_tokens
                && end <= self.plan.deepep_worst_expanded_tokens
                && expert_offsets[expert] == start as i64;
            if m > 0 {
                active_experts += 1;
            }

            let base = expert * 3;
            w13_problem_sizes_valid &= w13_problem_sizes[base] == m as i32
                && w13_problem_sizes[base + 1] == self.plan.moe_intermediate as i32 * 2
                && w13_problem_sizes[base + 2] == self.plan.hidden as i32;
            w2_problem_sizes_valid &= w2_problem_sizes[base] == m as i32
                && w2_problem_sizes[base + 1] == self.plan.hidden as i32
                && w2_problem_sizes[base + 2] == self.plan.moe_intermediate as i32;
            previous_end = end;
        }

        let expanded_rows = align_up(previous_end, self.plan.expert_alignment);
        offsets_valid &= expanded_rows <= self.plan.deepep_worst_expanded_tokens
            && expert_offsets[self.plan.local_experts] == expanded_rows as i64;
        let trtllm_grouped_offset_scale_rows_required = glm52_trtllm_grouped_offset_padded_rows(
            self.plan.deepep_worst_expanded_tokens,
            self.plan.local_experts,
        );
        let trtllm_grouped_offset_scale_rows_covered = self.plan.moe_trtllm_grouped_offset_rows
            >= trtllm_grouped_offset_scale_rows_required
            && self.moe_w13_input_scale_trtllm_offset_tma.len()
                >= trtllm_grouped_offset_scale_rows_required * self.plan.moe_w13_scale_cols
            && self.moe_w2_input_scale_trtllm_offset_tma.len()
                >= trtllm_grouped_offset_scale_rows_required * self.plan.moe_w2_scale_cols;

        Ok(Glm52MoeGemmMetadataValidation {
            offsets_valid,
            w13_problem_sizes_valid,
            w2_problem_sizes_valid,
            active_experts,
            expanded_rows,
            trtllm_grouped_offset_scale_rows_required,
            trtllm_grouped_offset_scale_rows_covered,
        })
    }

    pub(crate) fn validate_moe_gemm_smoke_outputs(
        &self,
        ctx: &Glm52RankGpuContext,
    ) -> Result<(bool, bool)> {
        let stream = ctx.stream();
        let expert_offsets = stream.clone_dtoh(
            &self
                .moe_gemm_expert_offsets
                .slice(0..self.plan.moe_gemm_expert_offsets_len),
        )?;
        let Some((start, end)) =
            first_active_expert_range(&expert_offsets, self.plan.deepep_worst_expanded_tokens)?
        else {
            return Ok((true, true));
        };

        Ok((
            sampled_bf16_rows_nonzero(
                stream,
                &self.moe_w13_output_bf16,
                start,
                end,
                self.plan.moe_intermediate * 2,
            )?,
            sampled_bf16_rows_nonzero(
                stream,
                &self.moe_w2_output_bf16,
                start,
                end,
                self.plan.hidden,
            )?,
        ))
    }

    fn validate_moe_quant_rows(&self, rows: usize) -> Result<()> {
        ensure!(
            (1..=self.plan.deepep_worst_expanded_tokens).contains(&rows),
            "GLM5.2 MoE quant smoke rows {rows} out of 1..={}",
            self.plan.deepep_worst_expanded_tokens
        );
        Ok(())
    }

    pub(crate) fn validate_router_smoke_routes(
        &self,
        ctx: &Glm52RankGpuContext,
        num_tokens: usize,
    ) -> Result<(bool, bool)> {
        ensure!(
            (1..=self.plan.batch_capacity).contains(&num_tokens),
            "GLM5.2 router smoke tokens {num_tokens} out of 1..={}",
            self.plan.batch_capacity
        );
        let route_len = num_tokens * self.plan.topk;
        let topk_idx = ctx
            .stream()
            .clone_dtoh(&self.topk_idx.slice(0..route_len))?;
        let topk_weight = ctx
            .stream()
            .clone_dtoh(&self.topk_weight.slice(0..route_len))?;

        let mut routes_valid = true;
        let mut weights_normalized = true;
        for token in 0..num_tokens {
            let base = token * self.plan.topk;
            let mut seen = vec![false; self.plan.routed_experts];
            let mut sum = 0.0f32;
            for route in 0..self.plan.topk {
                let expert = topk_idx[base + route];
                let weight = topk_weight[base + route];
                routes_valid &= expert >= 0;
                if expert >= 0 {
                    let expert = expert as usize;
                    routes_valid &= expert < self.plan.routed_experts && !seen[expert];
                    if expert < self.plan.routed_experts {
                        seen[expert] = true;
                    }
                }
                routes_valid &= weight.is_finite() && weight > 0.0;
                sum += weight;
            }
            weights_normalized &= (sum - 1.0).abs() <= 1.0e-3;
        }
        Ok((routes_valid, weights_normalized))
    }
}

const fn bytes_bf16(elements: usize) -> usize {
    elements * std::mem::size_of::<bf16>()
}

const fn bytes_f32(elements: usize) -> usize {
    elements * std::mem::size_of::<f32>()
}

const fn bytes_i32(elements: usize) -> usize {
    elements * std::mem::size_of::<i32>()
}

const fn bytes_i64(elements: usize) -> usize {
    elements * std::mem::size_of::<i64>()
}

const fn bytes_u8(elements: usize) -> usize {
    elements * std::mem::size_of::<u8>()
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn quant_output_valid(scales: &[f32], quantized: &[u8]) -> bool {
    scales.iter().all(|scale| scale.is_finite() && *scale > 0.0)
        && quantized.iter().any(|value| *value != 0)
}

fn first_active_expert_range(
    expert_offsets: &[i64],
    capacity: usize,
) -> Result<Option<(usize, usize)>> {
    ensure!(
        expert_offsets.len() >= 2,
        "GLM5.2 MoE GEMM smoke needs at least two expert offsets, got {}",
        expert_offsets.len()
    );
    for window in expert_offsets.windows(2) {
        let [start, end] = window else {
            unreachable!("windows(2) always yields two entries")
        };
        ensure!(
            *start >= 0 && *end >= *start && *end as usize <= capacity,
            "GLM5.2 MoE GEMM smoke expert offsets invalid: start={start}, end={end}, capacity={capacity}"
        );
        if end > start {
            return Ok(Some((*start as usize, *end as usize)));
        }
    }
    Ok(None)
}

fn sampled_bf16_rows_nonzero(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    data: &CudaSlice<bf16>,
    start_row: usize,
    end_row: usize,
    stride: usize,
) -> Result<bool> {
    ensure!(
        start_row < end_row && end_row * stride <= data.len(),
        "GLM5.2 MoE GEMM smoke sample range invalid: rows={start_row}..{end_row}, stride={stride}, len={}",
        data.len()
    );
    let sample_cols = stride.min(1024);
    for row in start_row..end_row.min(start_row + 4) {
        let start = row * stride;
        let sample = stream.clone_dtoh(&data.slice(start..start + sample_cols))?;
        if sample
            .iter()
            .any(|value| value.to_f32().is_finite() && *value != bf16::ZERO)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn deepgemm_scale_layout_valid(
    row_major: &[f32],
    tma_major: &[f32],
    rows: usize,
    scale_cols: usize,
    aligned_rows: usize,
) -> bool {
    if row_major.len() < rows * scale_cols || tma_major.len() < aligned_rows * scale_cols {
        return false;
    }
    for col in 0..scale_cols {
        for row in 0..aligned_rows {
            let value = tma_major[col * aligned_rows + row];
            if row < rows {
                if value != row_major[row * scale_cols + col] {
                    return false;
                }
            } else if value != 0.0 {
                return false;
            }
        }
    }
    true
}

pub(crate) fn trtllm_grouped_offset_scale_layout_valid(
    row_major: &[f32],
    offset_major: &[f32],
    expert_offsets: &[i64],
    m_capacity: usize,
    scale_cols: usize,
    padded_rows: usize,
) -> bool {
    if expert_offsets.len() < 2
        || row_major.len() < m_capacity * scale_cols
        || offset_major.len() < padded_rows * scale_cols
    {
        return false;
    }
    for col in 0..scale_cols {
        for dst_row in 0..padded_rows {
            let mut expected = 0.0f32;
            for expert in 0..(expert_offsets.len() - 1) {
                let src_start = expert_offsets[expert];
                let src_end = expert_offsets[expert + 1];
                if src_start < 0 || src_end < src_start || src_end as usize > m_capacity {
                    return false;
                }
                let src_start = src_start as usize;
                let src_end = src_end as usize;
                let dst_start = glm52_trtllm_grouped_offset_padded_rows(src_start, expert);
                let dst_end = dst_start + (src_end - src_start);
                if dst_row >= dst_start && dst_row < dst_end {
                    let src_row = src_start + (dst_row - dst_start);
                    expected = row_major[src_row * scale_cols + col];
                    break;
                }
            }
            if offset_major[col * padded_rows + dst_row] != expected {
                return false;
            }
        }
    }
    true
}
