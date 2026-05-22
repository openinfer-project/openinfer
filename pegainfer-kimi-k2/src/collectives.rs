//! Kimi-K2.6 TP8/EP8 collective API draft.
//!
//! This module defines the tensor-parallel and expert-parallel contracts used
//! by the eventual CUDA/PPLX implementation. The production EP target is still
//! pplx-garden dispatch/combine. The current runner is separate from
//! this header draft and uses a temporary NCCL-sum bridge over the 8 rank
//! streams for TP embedding/projection and MoE shared/routed combines; see
//! `runner::KimiK2Runtime` for the actual call sites.
//! The functions intentionally stop at validation and return `Unsupported`:
//! no collective transport body lives in this header crate.

use crate::{
    config::{
        KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HEADS, KIMI_K2_HIDDEN, KIMI_K2_Q_PROJ_OUT,
        KIMI_K2_QK_NOPE_HEAD_DIM, KIMI_K2_QK_ROPE_HEAD_DIM, KIMI_K2_ROUTED_EXPERTS, KIMI_K2_TOPK,
        KIMI_K2_VOCAB, KimiK2ParallelShape,
    },
    tensor::{
        Bf16, DType, EpRank, F32, HeaderError, HeaderResult, Layout, Shape2, Shape3, StreamHandle,
        TensorMut, TensorRef, TokenBatch, TpRank, U32, VocabShard,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectiveBackend {
    Nccl,
    PplxGarden,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorParallelRole {
    Attention,
    DenseMlp,
    SharedExpert,
    Logits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearShard {
    /// Each TP rank owns output rows. GEMM produces `[tokens, out/tp]`.
    ColumnParallel,
    /// Each TP rank owns input columns. GEMM produces a partial full output.
    RowParallel,
    /// Every TP rank owns the full matrix; no TP collective is attached.
    Replicated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TpPostOp {
    None,
    AllReduceSum,
    ReduceScatterSum,
    AllGatherConcat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TpShardPolicy {
    pub role: TensorParallelRole,
    pub op: KimiLinearOp,
    pub weight_shard: LinearShard,
    pub post_op: TpPostOp,
    pub input_dim: usize,
    pub output_dim: usize,
    pub local_input_dim: usize,
    pub local_output_dim: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KimiLinearOp {
    QaProj,
    QbProj,
    KvAWithMqaProj,
    KvBProj,
    OProj,
    DenseGate,
    DenseUp,
    DenseDown,
    SharedGate,
    SharedUp,
    SharedDown,
    LmHead,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionTpPolicy {
    pub total_heads: usize,
    pub local_heads: usize,
    pub q_nope_dim: usize,
    pub q_rope_dim: usize,
    pub q_head_dim: usize,
    pub v_head_dim: usize,
    pub q_proj_local_out: usize,
    pub kv_b_local_out: usize,
    pub o_proj_partial_out: usize,
    pub o_proj_post_op: TpPostOp,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParallelPlan {
    pub shape: KimiK2ParallelShape,
    pub tp_rank: TpRank,
    pub ep_rank: EpRank,
    pub attention: AttentionTpPolicy,
    pub dense_policies: Vec<TpShardPolicy>,
    pub shared_expert_policies: Vec<TpShardPolicy>,
    pub lm_head_policy: TpShardPolicy,
    pub vocab_shard: VocabShard,
    pub moe: EpAgRsPlan,
    pub pplx: PplxDispatchCombinePlan,
}

impl ParallelPlan {
    #[must_use]
    pub fn tp8_ep8(tp_rank: usize, ep_rank: usize, max_batch_size: usize) -> Self {
        Self::new(
            KimiK2ParallelShape::tp8_ep8(),
            tp_rank,
            ep_rank,
            max_batch_size,
        )
    }

    #[must_use]
    pub fn new(
        shape: KimiK2ParallelShape,
        tp_rank: usize,
        ep_rank: usize,
        max_batch_size: usize,
    ) -> Self {
        let tp = TpRank {
            rank: tp_rank,
            world: shape.tp_world,
        };
        let ep = EpRank {
            rank: ep_rank,
            world: shape.ep_world,
        };
        let vocab_start = tp_rank * shape.vocab_per_tp;
        let vocab_end = vocab_start + shape.vocab_per_tp;

        Self {
            shape,
            tp_rank: tp,
            ep_rank: ep,
            attention: attention_tp_policy(shape),
            dense_policies: dense_tp_policies(shape),
            shared_expert_policies: shared_expert_tp_policies(shape),
            lm_head_policy: lm_head_tp_policy(shape),
            vocab_shard: VocabShard {
                range: vocab_start..vocab_end,
            },
            moe: EpAgRsPlan::new(shape, max_batch_size),
            pplx: PplxDispatchCombinePlan::new(shape, max_batch_size),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectiveBatch {
    /// User-visible request rows in this scheduler step.
    pub batch_size: usize,
    /// Real local tokens on this rank before padding.
    pub local_tokens: usize,
    /// Real tokens across all EP ranks.
    pub global_tokens: usize,
    /// Padded local token rows used by collective buffers and CUDA graphs.
    pub padded_local_tokens: usize,
    /// Padded global token rows, normally `padded_local_tokens * ep_world`.
    pub padded_global_tokens: usize,
}

impl CollectiveBatch {
    #[must_use]
    pub fn from_local(
        batch: TokenBatch,
        local_tokens: usize,
        padded_local_tokens: usize,
        ep_world: usize,
    ) -> Self {
        Self {
            batch_size: batch.batch_size,
            local_tokens,
            global_tokens: local_tokens * ep_world,
            padded_local_tokens,
            padded_global_tokens: padded_local_tokens * ep_world,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EpAgRsPlan {
    pub backend: CollectiveBackend,
    pub ep_world: usize,
    pub global_experts: usize,
    pub local_experts: usize,
    pub topk: usize,
    pub hidden_dim: usize,
    pub expert_intermediate_dim: usize,
    pub max_batch_size: usize,
    pub max_local_tokens: usize,
    pub max_global_tokens: usize,
}

impl EpAgRsPlan {
    #[must_use]
    pub fn new(shape: KimiK2ParallelShape, max_batch_size: usize) -> Self {
        Self {
            backend: CollectiveBackend::Nccl,
            ep_world: shape.ep_world,
            global_experts: KIMI_K2_ROUTED_EXPERTS,
            local_experts: shape.local_experts,
            topk: KIMI_K2_TOPK,
            hidden_dim: KIMI_K2_HIDDEN,
            expert_intermediate_dim: KIMI_K2_EXPERT_INTERMEDIATE,
            max_batch_size,
            max_local_tokens: max_batch_size,
            max_global_tokens: max_batch_size * shape.ep_world,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PplxDispatchCombinePlan {
    pub backend: CollectiveBackend,
    pub ep_world: usize,
    pub global_experts: usize,
    pub local_experts: usize,
    pub topk: usize,
    pub hidden_dim: usize,
    pub max_batch_size: usize,
    pub max_local_tokens: usize,
    pub max_dispatch_rows: usize,
    pub expert_padding: usize,
}

impl PplxDispatchCombinePlan {
    #[must_use]
    pub fn new(shape: KimiK2ParallelShape, max_batch_size: usize) -> Self {
        Self {
            backend: CollectiveBackend::PplxGarden,
            ep_world: shape.ep_world,
            global_experts: KIMI_K2_ROUTED_EXPERTS,
            local_experts: shape.local_experts,
            topk: KIMI_K2_TOPK,
            hidden_dim: KIMI_K2_HIDDEN,
            max_batch_size,
            max_local_tokens: max_batch_size,
            max_dispatch_rows: max_batch_size * KIMI_K2_TOPK,
            expert_padding: 16,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectiveScratch {
    pub global_hidden_bf16: Shape2,
    pub global_token_ids_u32: Shape2,
    pub router_topk_ids_u32: Shape2,
    pub router_topk_weights_f32: Shape2,
    pub local_expert_indptr_u32: Shape2,
    pub expert_major_hidden_bf16: Shape2,
    pub expert_major_output_f32: Shape2,
    pub partial_routed_f32: Shape2,
    pub local_routed_f32: Shape2,
    pub pplx_send_bf16: Shape2,
    pub pplx_recv_bf16: Shape2,
    pub local_logits_f32: Shape2,
    pub full_logits_f32: Shape2,
}

impl CollectiveScratch {
    #[must_use]
    pub fn new(plan: &ParallelPlan, max_padded_local_tokens: usize) -> Self {
        let max_padded_global_tokens = max_padded_local_tokens * plan.shape.ep_world;
        let max_routes = max_padded_global_tokens * KIMI_K2_TOPK;

        Self {
            global_hidden_bf16: Shape2 {
                rows: max_padded_global_tokens,
                cols: KIMI_K2_HIDDEN,
            },
            global_token_ids_u32: Shape2 {
                rows: max_padded_global_tokens,
                cols: 1,
            },
            router_topk_ids_u32: Shape2 {
                rows: max_padded_global_tokens,
                cols: KIMI_K2_TOPK,
            },
            router_topk_weights_f32: Shape2 {
                rows: max_padded_global_tokens,
                cols: KIMI_K2_TOPK,
            },
            local_expert_indptr_u32: Shape2 {
                rows: plan.shape.local_experts + 1,
                cols: 1,
            },
            expert_major_hidden_bf16: Shape2 {
                rows: max_routes,
                cols: KIMI_K2_HIDDEN,
            },
            expert_major_output_f32: Shape2 {
                rows: max_routes,
                cols: KIMI_K2_HIDDEN,
            },
            partial_routed_f32: Shape2 {
                rows: max_padded_global_tokens,
                cols: KIMI_K2_HIDDEN,
            },
            local_routed_f32: Shape2 {
                rows: max_padded_local_tokens,
                cols: KIMI_K2_HIDDEN,
            },
            pplx_send_bf16: Shape2 {
                rows: max_padded_local_tokens * KIMI_K2_TOPK,
                cols: KIMI_K2_HIDDEN,
            },
            pplx_recv_bf16: Shape2 {
                rows: max_routes,
                cols: KIMI_K2_HIDDEN,
            },
            local_logits_f32: Shape2 {
                rows: max_padded_local_tokens,
                cols: plan.shape.vocab_per_tp,
            },
            full_logits_f32: Shape2 {
                rows: max_padded_local_tokens,
                cols: KIMI_K2_VOCAB,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionTpTensors {
    pub q_local: TensorMut<Bf16>,
    pub kv_a_replicated: TensorRef<Bf16>,
    pub kv_b_local: TensorMut<Bf16>,
    pub attn_local_out: TensorRef<Bf16>,
    pub o_proj_partial: TensorRef<Bf16>,
    pub hidden_out: TensorMut<Bf16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EpAgRsTensors {
    pub local_hidden: TensorRef<Bf16>,
    pub local_token_ids: TensorRef<U32>,
    pub global_hidden: TensorMut<Bf16>,
    pub global_token_ids: TensorMut<U32>,
    pub topk_ids: TensorMut<U32>,
    pub topk_weights: TensorMut<F32>,
    pub local_expert_indptr: TensorMut<U32>,
    pub expert_major_hidden: TensorMut<Bf16>,
    pub expert_major_output: TensorMut<F32>,
    pub partial_routed: TensorMut<F32>,
    pub local_routed: TensorMut<F32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PplxDispatchCombineTensors {
    pub local_hidden: TensorRef<Bf16>,
    pub topk_ids: TensorRef<U32>,
    pub topk_weights: TensorRef<F32>,
    pub send_hidden: TensorMut<Bf16>,
    pub recv_hidden: TensorMut<Bf16>,
    pub recv_tokens_per_expert: TensorMut<U32>,
    pub local_expert_indptr: TensorMut<U32>,
    pub expert_output: TensorRef<F32>,
    pub combined_hidden: TensorMut<F32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogitsAllGatherTensors {
    pub local_logits: TensorRef<F32>,
    pub full_logits: TensorMut<F32>,
}

/// Attention heads are TP-sharded. MLA `kv_a` stays replicated because the
/// compressed MQA projection is small and feeds local `kv_b` head shards.
pub fn validate_attention_tp_collective(
    plan: &ParallelPlan,
    batch: CollectiveBatch,
    tensors: AttentionTpTensors,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    validate_tp8_ep8(plan)?;
    validate_padded_batch(batch)?;
    expect_bf16(tensors.q_local.dtype, "q_local")?;
    expect_bf16(tensors.kv_a_replicated.dtype, "kv_a_replicated")?;
    expect_bf16(tensors.kv_b_local.dtype, "kv_b_local")?;
    expect_bf16(tensors.attn_local_out.dtype, "attn_local_out")?;
    expect_bf16(tensors.o_proj_partial.dtype, "o_proj_partial")?;
    expect_bf16(tensors.hidden_out.dtype, "hidden_out")?;

    expect_layout(tensors.q_local.layout, Layout::HeadMajor, "q_local")?;
    expect_layout(tensors.kv_b_local.layout, Layout::HeadMajor, "kv_b_local")?;
    expect_layout(tensors.hidden_out.layout, Layout::RowMajor, "hidden_out")?;

    expect_len(
        tensors.q_local.ptr.len,
        batch.padded_local_tokens * plan.attention.q_proj_local_out,
        "q_local",
    )?;
    expect_len(
        tensors.kv_a_replicated.ptr.len,
        batch.padded_local_tokens * (crate::config::KIMI_K2_KV_A_OUT),
        "kv_a_replicated",
    )?;
    expect_len(
        tensors.kv_b_local.ptr.len,
        batch.padded_local_tokens * plan.attention.kv_b_local_out,
        "kv_b_local",
    )?;
    expect_len(
        tensors.attn_local_out.ptr.len,
        batch.padded_local_tokens * plan.attention.local_heads * crate::config::KIMI_K2_V_HEAD_DIM,
        "attn_local_out",
    )?;
    expect_len(
        tensors.o_proj_partial.ptr.len,
        batch.padded_local_tokens * KIMI_K2_HIDDEN,
        "o_proj_partial",
    )?;
    expect_len(
        tensors.hidden_out.ptr.len,
        batch.padded_local_tokens * KIMI_K2_HIDDEN,
        "hidden_out",
    )?;
    unsupported("attention TP collective transport is not implemented in header crate")
}

/// Historical AG/RS sketch retained only as a shape comparison aid. Kimi EP
/// execution should use `validate_pplx_dispatch_combine_path`.
pub fn validate_ep_ag_rs_correctness_path(
    plan: &EpAgRsPlan,
    batch: CollectiveBatch,
    tensors: EpAgRsTensors,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    validate_ep_plan(plan)?;
    validate_padded_batch(batch)?;
    expect_bf16(tensors.local_hidden.dtype, "local_hidden")?;
    expect_u32(tensors.local_token_ids.dtype, "local_token_ids")?;
    expect_bf16(tensors.global_hidden.dtype, "global_hidden")?;
    expect_u32(tensors.global_token_ids.dtype, "global_token_ids")?;
    expect_u32(tensors.topk_ids.dtype, "topk_ids")?;
    expect_f32(tensors.topk_weights.dtype, "topk_weights")?;
    expect_u32(tensors.local_expert_indptr.dtype, "local_expert_indptr")?;
    expect_bf16(tensors.expert_major_hidden.dtype, "expert_major_hidden")?;
    expect_f32(tensors.expert_major_output.dtype, "expert_major_output")?;
    expect_f32(tensors.partial_routed.dtype, "partial_routed")?;
    expect_f32(tensors.local_routed.dtype, "local_routed")?;

    expect_layout(
        tensors.local_hidden.layout,
        Layout::RowMajor,
        "local_hidden",
    )?;
    expect_layout(
        tensors.global_hidden.layout,
        Layout::RowMajor,
        "global_hidden",
    )?;
    expect_layout(
        tensors.expert_major_hidden.layout,
        Layout::ExpertMajor,
        "expert_major_hidden",
    )?;
    expect_layout(
        tensors.expert_major_output.layout,
        Layout::ExpertMajor,
        "expert_major_output",
    )?;

    expect_len(
        tensors.local_hidden.ptr.len,
        batch.padded_local_tokens * plan.hidden_dim,
        "local_hidden",
    )?;
    expect_len(
        tensors.local_token_ids.ptr.len,
        batch.padded_local_tokens,
        "local_token_ids",
    )?;
    expect_len(
        tensors.global_hidden.ptr.len,
        batch.padded_global_tokens * plan.hidden_dim,
        "global_hidden",
    )?;
    expect_len(
        tensors.global_token_ids.ptr.len,
        batch.padded_global_tokens,
        "global_token_ids",
    )?;
    expect_len(
        tensors.topk_ids.ptr.len,
        batch.padded_global_tokens * plan.topk,
        "topk_ids",
    )?;
    expect_len(
        tensors.topk_weights.ptr.len,
        batch.padded_global_tokens * plan.topk,
        "topk_weights",
    )?;
    expect_len(
        tensors.local_expert_indptr.ptr.len,
        plan.local_experts + 1,
        "local_expert_indptr",
    )?;
    expect_len(
        tensors.expert_major_hidden.ptr.len,
        batch.padded_global_tokens * plan.topk * plan.hidden_dim,
        "expert_major_hidden",
    )?;
    expect_len(
        tensors.expert_major_output.ptr.len,
        batch.padded_global_tokens * plan.topk * plan.hidden_dim,
        "expert_major_output",
    )?;
    expect_len(
        tensors.partial_routed.ptr.len,
        batch.padded_global_tokens * plan.hidden_dim,
        "partial_routed",
    )?;
    expect_len(
        tensors.local_routed.ptr.len,
        batch.padded_local_tokens * plan.hidden_dim,
        "local_routed",
    )?;
    unsupported("EP AG/RS NCCL transport is not implemented in header crate")
}

/// PPLX sparse MoE path: dispatch selected token/expert rows, run local
/// expert-major grouped GEMMs, and combine weighted expert outputs back to
/// local hidden rows. This path is eager-only in the DSV4 precedent.
pub fn validate_pplx_dispatch_combine_path(
    plan: &PplxDispatchCombinePlan,
    batch: CollectiveBatch,
    tensors: PplxDispatchCombineTensors,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    validate_pplx_plan(plan)?;
    validate_padded_batch(batch)?;
    expect_bf16(tensors.local_hidden.dtype, "local_hidden")?;
    expect_u32(tensors.topk_ids.dtype, "topk_ids")?;
    expect_f32(tensors.topk_weights.dtype, "topk_weights")?;
    expect_bf16(tensors.send_hidden.dtype, "send_hidden")?;
    expect_bf16(tensors.recv_hidden.dtype, "recv_hidden")?;
    expect_u32(
        tensors.recv_tokens_per_expert.dtype,
        "recv_tokens_per_expert",
    )?;
    expect_u32(tensors.local_expert_indptr.dtype, "local_expert_indptr")?;
    expect_f32(tensors.expert_output.dtype, "expert_output")?;
    expect_f32(tensors.combined_hidden.dtype, "combined_hidden")?;

    expect_layout(
        tensors.local_hidden.layout,
        Layout::RowMajor,
        "local_hidden",
    )?;
    expect_layout(
        tensors.send_hidden.layout,
        Layout::ExpertMajor,
        "send_hidden",
    )?;
    expect_layout(
        tensors.recv_hidden.layout,
        Layout::ExpertMajor,
        "recv_hidden",
    )?;
    expect_layout(
        tensors.expert_output.layout,
        Layout::ExpertMajor,
        "expert_output",
    )?;
    expect_layout(
        tensors.combined_hidden.layout,
        Layout::RowMajor,
        "combined_hidden",
    )?;

    let local_routes = batch.padded_local_tokens * plan.topk;
    let global_routes = batch.padded_global_tokens * plan.topk;
    expect_len(
        tensors.local_hidden.ptr.len,
        batch.padded_local_tokens * plan.hidden_dim,
        "local_hidden",
    )?;
    expect_len(tensors.topk_ids.ptr.len, local_routes, "topk_ids")?;
    expect_len(tensors.topk_weights.ptr.len, local_routes, "topk_weights")?;
    expect_len(
        tensors.send_hidden.ptr.len,
        local_routes * plan.hidden_dim,
        "send_hidden",
    )?;
    expect_len(
        tensors.recv_hidden.ptr.len,
        global_routes * plan.hidden_dim,
        "recv_hidden",
    )?;
    expect_len(
        tensors.recv_tokens_per_expert.ptr.len,
        plan.local_experts,
        "recv_tokens_per_expert",
    )?;
    expect_len(
        tensors.local_expert_indptr.ptr.len,
        plan.local_experts + 1,
        "local_expert_indptr",
    )?;
    expect_len(
        tensors.expert_output.ptr.len,
        global_routes * plan.hidden_dim,
        "expert_output",
    )?;
    expect_len(
        tensors.combined_hidden.ptr.len,
        batch.padded_local_tokens * plan.hidden_dim,
        "combined_hidden",
    )?;
    unsupported("PPLX dispatch/combine transport is not implemented in header crate")
}

/// Final lm_head is vocab-parallel: each TP rank computes `[tokens, 20480]`
/// logits for TP8, then all-gathers to `[tokens, 163840]` before sampling.
pub fn validate_logits_all_gather(
    plan: &ParallelPlan,
    local_tokens: usize,
    padded_batch_size: usize,
    tensors: LogitsAllGatherTensors,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    validate_tp8_ep8(plan)?;
    if local_tokens > padded_batch_size {
        return shape_err("local_tokens must not exceed padded_batch_size");
    }
    expect_f32(tensors.local_logits.dtype, "local_logits")?;
    expect_f32(tensors.full_logits.dtype, "full_logits")?;
    expect_layout(
        tensors.local_logits.layout,
        Layout::RowMajor,
        "local_logits",
    )?;
    expect_layout(tensors.full_logits.layout, Layout::RowMajor, "full_logits")?;
    expect_len(
        tensors.local_logits.ptr.len,
        padded_batch_size * plan.shape.vocab_per_tp,
        "local_logits",
    )?;
    expect_len(
        tensors.full_logits.ptr.len,
        padded_batch_size * KIMI_K2_VOCAB,
        "full_logits",
    )?;
    unsupported("logits all-gather transport is not implemented in header crate")
}

#[must_use]
pub fn attention_tp_policy(shape: KimiK2ParallelShape) -> AttentionTpPolicy {
    let local_heads = KIMI_K2_HEADS / shape.tp_world;
    AttentionTpPolicy {
        total_heads: KIMI_K2_HEADS,
        local_heads,
        q_nope_dim: KIMI_K2_QK_NOPE_HEAD_DIM,
        q_rope_dim: KIMI_K2_QK_ROPE_HEAD_DIM,
        q_head_dim: crate::config::KIMI_K2_Q_HEAD_DIM,
        v_head_dim: crate::config::KIMI_K2_V_HEAD_DIM,
        q_proj_local_out: KIMI_K2_Q_PROJ_OUT / shape.tp_world,
        kv_b_local_out: crate::config::KIMI_K2_KV_B_OUT / shape.tp_world,
        o_proj_partial_out: KIMI_K2_HIDDEN,
        o_proj_post_op: TpPostOp::AllReduceSum,
    }
}

#[must_use]
pub fn dense_tp_policies(shape: KimiK2ParallelShape) -> Vec<TpShardPolicy> {
    vec![
        column_policy(
            TensorParallelRole::DenseMlp,
            KimiLinearOp::DenseGate,
            KIMI_K2_HIDDEN,
            crate::config::KIMI_K2_DENSE_INTERMEDIATE,
            shape.tp_world,
        ),
        column_policy(
            TensorParallelRole::DenseMlp,
            KimiLinearOp::DenseUp,
            KIMI_K2_HIDDEN,
            crate::config::KIMI_K2_DENSE_INTERMEDIATE,
            shape.tp_world,
        ),
        row_policy(
            TensorParallelRole::DenseMlp,
            KimiLinearOp::DenseDown,
            crate::config::KIMI_K2_DENSE_INTERMEDIATE,
            KIMI_K2_HIDDEN,
            shape.tp_world,
            TpPostOp::AllReduceSum,
        ),
    ]
}

#[must_use]
pub fn shared_expert_tp_policies(shape: KimiK2ParallelShape) -> Vec<TpShardPolicy> {
    vec![
        column_policy(
            TensorParallelRole::SharedExpert,
            KimiLinearOp::SharedGate,
            KIMI_K2_HIDDEN,
            KIMI_K2_EXPERT_INTERMEDIATE,
            shape.tp_world,
        ),
        column_policy(
            TensorParallelRole::SharedExpert,
            KimiLinearOp::SharedUp,
            KIMI_K2_HIDDEN,
            KIMI_K2_EXPERT_INTERMEDIATE,
            shape.tp_world,
        ),
        row_policy(
            TensorParallelRole::SharedExpert,
            KimiLinearOp::SharedDown,
            KIMI_K2_EXPERT_INTERMEDIATE,
            KIMI_K2_HIDDEN,
            shape.tp_world,
            TpPostOp::AllReduceSum,
        ),
    ]
}

#[must_use]
pub fn lm_head_tp_policy(shape: KimiK2ParallelShape) -> TpShardPolicy {
    column_policy(
        TensorParallelRole::Logits,
        KimiLinearOp::LmHead,
        KIMI_K2_HIDDEN,
        KIMI_K2_VOCAB,
        shape.tp_world,
    )
}

#[must_use]
pub fn local_attention_q_shape(plan: &ParallelPlan, batch: CollectiveBatch) -> Shape3 {
    Shape3 {
        outer: batch.padded_local_tokens,
        middle: plan.attention.local_heads,
        inner: plan.attention.q_head_dim,
    }
}

#[must_use]
pub fn local_attention_v_shape(plan: &ParallelPlan, batch: CollectiveBatch) -> Shape3 {
    Shape3 {
        outer: batch.padded_local_tokens,
        middle: plan.attention.local_heads,
        inner: plan.attention.v_head_dim,
    }
}

fn column_policy(
    role: TensorParallelRole,
    op: KimiLinearOp,
    input_dim: usize,
    output_dim: usize,
    tp_world: usize,
) -> TpShardPolicy {
    TpShardPolicy {
        role,
        op,
        weight_shard: LinearShard::ColumnParallel,
        post_op: TpPostOp::None,
        input_dim,
        output_dim,
        local_input_dim: input_dim,
        local_output_dim: output_dim / tp_world,
    }
}

fn row_policy(
    role: TensorParallelRole,
    op: KimiLinearOp,
    input_dim: usize,
    output_dim: usize,
    tp_world: usize,
    post_op: TpPostOp,
) -> TpShardPolicy {
    TpShardPolicy {
        role,
        op,
        weight_shard: LinearShard::RowParallel,
        post_op,
        input_dim,
        output_dim,
        local_input_dim: input_dim / tp_world,
        local_output_dim: output_dim,
    }
}

fn validate_tp8_ep8(plan: &ParallelPlan) -> HeaderResult<()> {
    if plan.shape.tp_world != 8 || plan.shape.ep_world != 8 {
        return shape_err("Kimi-K2.6 draft currently targets TP8/EP8 only");
    }
    if plan.tp_rank.rank >= plan.tp_rank.world {
        return shape_err("tp rank must be smaller than tp world");
    }
    if plan.ep_rank.rank >= plan.ep_rank.world {
        return shape_err("ep rank must be smaller than ep world");
    }
    if KIMI_K2_HEADS % plan.shape.tp_world != 0 {
        return shape_err("attention heads must divide tp world");
    }
    if KIMI_K2_ROUTED_EXPERTS % plan.shape.ep_world != 0 {
        return shape_err("routed experts must divide ep world");
    }
    Ok(())
}

fn validate_ep_plan(plan: &EpAgRsPlan) -> HeaderResult<()> {
    if plan.backend != CollectiveBackend::Nccl {
        return shape_err("EP AG/RS plan must use Nccl backend");
    }
    validate_ep_common(
        plan.ep_world,
        plan.global_experts,
        plan.local_experts,
        plan.topk,
        plan.hidden_dim,
    )
}

fn validate_pplx_plan(plan: &PplxDispatchCombinePlan) -> HeaderResult<()> {
    if plan.backend != CollectiveBackend::PplxGarden {
        return shape_err("PPLX dispatch/combine plan must use PplxGarden backend");
    }
    validate_ep_common(
        plan.ep_world,
        plan.global_experts,
        plan.local_experts,
        plan.topk,
        plan.hidden_dim,
    )
}

fn validate_ep_common(
    ep_world: usize,
    global_experts: usize,
    local_experts: usize,
    topk: usize,
    hidden_dim: usize,
) -> HeaderResult<()> {
    if ep_world != 8 {
        return shape_err("Kimi-K2.6 EP plan currently targets EP8 only");
    }
    if global_experts != KIMI_K2_ROUTED_EXPERTS {
        return shape_err("unexpected Kimi-K2.6 routed expert count");
    }
    if local_experts != KIMI_K2_ROUTED_EXPERTS / ep_world {
        return shape_err("local experts must equal global experts / ep world");
    }
    if topk != KIMI_K2_TOPK {
        return shape_err("Kimi-K2.6 MoE topk must be 8");
    }
    if hidden_dim != KIMI_K2_HIDDEN {
        return shape_err("unexpected Kimi-K2.6 hidden dim");
    }
    Ok(())
}

fn validate_padded_batch(batch: CollectiveBatch) -> HeaderResult<()> {
    if batch.batch_size == 0 {
        return shape_err("batch_size must be non-zero");
    }
    if batch.local_tokens == 0 {
        return shape_err("local_tokens must be non-zero");
    }
    if batch.local_tokens > batch.padded_local_tokens {
        return shape_err("local_tokens must not exceed padded_local_tokens");
    }
    if batch.global_tokens > batch.padded_global_tokens {
        return shape_err("global_tokens must not exceed padded_global_tokens");
    }
    if batch.padded_local_tokens < batch.batch_size {
        return shape_err("padded_local_tokens must cover batch_size");
    }
    Ok(())
}

fn expect_bf16(dtype: DType, name: &str) -> HeaderResult<()> {
    expect_dtype(dtype, DType::Bf16, name)
}

fn expect_f32(dtype: DType, name: &str) -> HeaderResult<()> {
    expect_dtype(dtype, DType::F32, name)
}

fn expect_u32(dtype: DType, name: &str) -> HeaderResult<()> {
    expect_dtype(dtype, DType::U32, name)
}

fn expect_dtype(actual: DType, expected: DType, name: &str) -> HeaderResult<()> {
    if actual != expected {
        return shape_err(format!(
            "{name} dtype mismatch: expected {expected:?}, got {actual:?}"
        ));
    }
    Ok(())
}

fn expect_layout(actual: Layout, expected: Layout, name: &str) -> HeaderResult<()> {
    if actual != expected {
        return shape_err(format!(
            "{name} layout mismatch: expected {expected:?}, got {actual:?}"
        ));
    }
    Ok(())
}

fn expect_len(actual: usize, expected: usize, name: &str) -> HeaderResult<()> {
    if actual < expected {
        return shape_err(format!(
            "{name} len mismatch: expected at least {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn shape_err<T>(message: impl Into<String>) -> HeaderResult<T> {
    Err(HeaderError::Shape {
        message: message.into(),
    })
}

fn unsupported<T>(message: impl Into<String>) -> HeaderResult<T> {
    Err(HeaderError::Unsupported {
        message: message.into(),
    })
}
