//! Kimi-K2.6 MoE router and expert-major layout headers.
//!
//! This module is a compile-checked API sketch.  GPU entry points validate
//! shapes and then report `Unsupported`; CUDA bodies belong in the eventual
//! kernel crate.

use crate::config::{KIMI_K2_HIDDEN, KIMI_K2_ROUTED_EXPERTS, KIMI_K2_TOPK};
use crate::tensor::{
    Bf16, DType, EpRank, F32, HeaderError, HeaderResult, Layout, Shape2, StreamHandle, TensorMut,
    TensorRef, TokenBatch,
};

pub const KIMI_K2_ROUTER_N_GROUP: usize = 1;
pub const KIMI_K2_ROUTER_TOPK_GROUP: usize = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterShape {
    pub hidden_dim: usize,
    pub routed_experts: usize,
    pub topk: usize,
    pub n_group: usize,
    pub topk_group: usize,
}

impl RouterShape {
    #[must_use]
    pub const fn kimi_k2() -> Self {
        Self {
            hidden_dim: KIMI_K2_HIDDEN,
            routed_experts: KIMI_K2_ROUTED_EXPERTS,
            topk: KIMI_K2_TOPK,
            n_group: KIMI_K2_ROUTER_N_GROUP,
            topk_group: KIMI_K2_ROUTER_TOPK_GROUP,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterWeights {
    /// `language_model.model.layers.*.mlp.gate.weight`, `[384, 7168]`.
    pub gate_weight: TensorRef<Bf16>,
    /// `e_score_correction_bias`, `[384]`, added only for expert choice.
    pub e_score_correction_bias: TensorRef<F32>,
    pub gate_shape: Shape2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterOutput {
    /// Final route weights `[tokens, 8]` from original sigmoid scores,
    /// normalized per token. Kimi's `2.827` routed scale is applied after the
    /// routed expert sum to match vLLM.
    pub topk_weight: TensorMut<F32>,
    /// Global expert ids `[tokens, 8]`.
    pub topk_idx: TensorMut<i32>,
    pub tokens: usize,
    pub topk: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterScratch {
    /// FP32 `hidden @ gate.weight.T`, `[padded_tokens, 384]`.
    pub logits: TensorMut<F32>,
    /// FP32 `sigmoid(logits)`, `[padded_tokens, 384]`.
    pub scores: TensorMut<F32>,
    /// FP32 `scores + e_score_correction_bias`, `[padded_tokens, 384]`.
    pub choice_scores: TensorMut<F32>,
    pub padded_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterPlan {
    pub batch: TokenBatch,
    pub hidden: TensorRef<Bf16>,
    pub hidden_shape: Shape2,
    pub ep_rank: EpRank,
    pub local_experts: usize,
    pub global_expert_start: usize,
    pub shape: RouterShape,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertMajorIndptr {
    /// Exclusive prefix over local expert rows, `[local_experts + 1]`.
    pub indptr: TensorMut<i32>,
    pub local_experts: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertMajorRouteMap {
    /// Expert-major position -> flattened token id, `[tokens * topk]`.
    pub pos_to_token: TensorMut<i32>,
    /// Expert-major position -> flattened `token * topk + route`, `[tokens * topk]`.
    pub pos_to_token_topk: TensorMut<i32>,
    /// Flattened `token * topk + route` -> expert-major position, or `-1` for
    /// non-local routes, `[tokens * topk]`.
    pub token_topk_to_pos: TensorMut<i32>,
    /// Per-local-expert cursor/count scratch, `[local_experts]`.
    pub expert_cursor: TensorMut<i32>,
    /// Number of local expanded rows, `[1]`.
    pub local_count: TensorMut<i32>,
    pub route_capacity: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertMajorBuffers {
    /// BF16 hidden rows packed by local expert, `[local_count, 7168]`.
    pub expanded_hidden: TensorMut<Bf16>,
    /// Routed expert output in the same expert-major row order.
    pub expanded_output: TensorMut<Bf16>,
    pub hidden_dim: usize,
    pub route_capacity: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertMajorLayout {
    pub indptr: ExpertMajorIndptr,
    pub map: ExpertMajorRouteMap,
    pub buffers: ExpertMajorBuffers,
    pub global_expert_start: usize,
    pub local_experts: usize,
    pub tokens: usize,
    pub topk: usize,
}

impl RouterPlan {
    #[must_use]
    pub fn new(batch: TokenBatch, hidden: TensorRef<Bf16>, ep_rank: EpRank) -> Self {
        let shape = RouterShape::kimi_k2();
        let local_experts = if ep_rank.world == 0 {
            0
        } else {
            shape.routed_experts / ep_rank.world
        };
        Self {
            batch,
            hidden,
            hidden_shape: Shape2 {
                rows: batch.padded_tokens,
                cols: shape.hidden_dim,
            },
            ep_rank,
            local_experts,
            global_expert_start: ep_rank.rank * local_experts,
            shape,
        }
    }

    pub fn try_new(
        batch: TokenBatch,
        hidden: TensorRef<Bf16>,
        ep_rank: EpRank,
    ) -> HeaderResult<Self> {
        let plan = Self::new(batch, hidden, ep_rank);
        validate_router_plan(&plan)?;
        Ok(plan)
    }
}

pub fn validate_router_weights(weights: &RouterWeights) -> HeaderResult<()> {
    ensure_dtype_layout(
        weights.gate_weight.dtype,
        weights.gate_weight.layout,
        DType::Bf16,
        Layout::RowMajor,
        "router gate_weight",
    )?;
    ensure_dtype_layout(
        weights.e_score_correction_bias.dtype,
        weights.e_score_correction_bias.layout,
        DType::F32,
        Layout::RowMajor,
        "router e_score_correction_bias",
    )?;
    ensure_eq_usize(
        weights.gate_shape.rows,
        KIMI_K2_ROUTED_EXPERTS,
        "router gate rows",
    )?;
    ensure_eq_usize(weights.gate_shape.cols, KIMI_K2_HIDDEN, "router gate cols")?;
    ensure_len(
        weights.gate_weight.ptr.len,
        KIMI_K2_ROUTED_EXPERTS * KIMI_K2_HIDDEN,
        "router gate_weight",
    )?;
    ensure_len(
        weights.e_score_correction_bias.ptr.len,
        KIMI_K2_ROUTED_EXPERTS,
        "router e_score_correction_bias",
    )
}

pub fn validate_router_plan(plan: &RouterPlan) -> HeaderResult<()> {
    let shape = RouterShape::kimi_k2();
    if plan.shape != shape {
        return Err(HeaderError::Shape {
            message: format!(
                "router shape mismatch: expected {shape:?}, got {:?}",
                plan.shape
            ),
        });
    }
    if plan.batch.batch_size == 0 {
        return Err(HeaderError::Shape {
            message: "router batch_size must be positive".to_string(),
        });
    }
    if plan.batch.active_tokens > plan.batch.padded_tokens {
        return Err(HeaderError::Shape {
            message: format!(
                "router active_tokens={} exceed padded_tokens={}",
                plan.batch.active_tokens, plan.batch.padded_tokens
            ),
        });
    }
    ensure_eq_usize(plan.hidden_shape.cols, KIMI_K2_HIDDEN, "router hidden cols")?;
    ensure_eq_usize(
        plan.hidden_shape.rows,
        plan.batch.padded_tokens,
        "router hidden rows",
    )?;
    ensure_dtype_layout(
        plan.hidden.dtype,
        plan.hidden.layout,
        DType::Bf16,
        Layout::RowMajor,
        "router hidden",
    )?;
    ensure_len(
        plan.hidden.ptr.len,
        plan.batch.padded_tokens * KIMI_K2_HIDDEN,
        "router hidden",
    )?;
    if plan.ep_rank.world == 0 {
        return Err(HeaderError::Shape {
            message: "router ep world must be positive".to_string(),
        });
    }
    if plan.ep_rank.rank >= plan.ep_rank.world {
        return Err(HeaderError::Shape {
            message: format!(
                "router ep rank {} out of world {}",
                plan.ep_rank.rank, plan.ep_rank.world
            ),
        });
    }
    if !KIMI_K2_ROUTED_EXPERTS.is_multiple_of(plan.ep_rank.world) {
        return Err(HeaderError::Shape {
            message: format!(
                "router experts={} must divide evenly by ep_world={}",
                KIMI_K2_ROUTED_EXPERTS, plan.ep_rank.world
            ),
        });
    }
    let local_experts = KIMI_K2_ROUTED_EXPERTS / plan.ep_rank.world;
    ensure_eq_usize(plan.local_experts, local_experts, "router local_experts")?;
    ensure_eq_usize(
        plan.global_expert_start,
        plan.ep_rank.rank * local_experts,
        "router global_expert_start",
    )
}

pub fn validate_router_output(output: &RouterOutput, tokens: usize) -> HeaderResult<()> {
    ensure_eq_usize(output.tokens, tokens, "router output tokens")?;
    ensure_eq_usize(output.topk, KIMI_K2_TOPK, "router output topk")?;
    ensure_dtype_layout(
        output.topk_weight.dtype,
        output.topk_weight.layout,
        DType::F32,
        Layout::RowMajor,
        "router topk_weight",
    )?;
    ensure_dtype_layout(
        output.topk_idx.dtype,
        output.topk_idx.layout,
        DType::I32,
        Layout::RowMajor,
        "router topk_idx",
    )?;
    ensure_len(
        output.topk_weight.ptr.len,
        tokens * KIMI_K2_TOPK,
        "router topk_weight",
    )?;
    ensure_len(
        output.topk_idx.ptr.len,
        tokens * KIMI_K2_TOPK,
        "router topk_idx",
    )
}

pub fn validate_router_scratch(scratch: &RouterScratch, padded_tokens: usize) -> HeaderResult<()> {
    ensure_eq_usize(
        scratch.padded_tokens,
        padded_tokens,
        "router scratch padded_tokens",
    )?;
    for (name, tensor) in [
        ("router logits", scratch.logits),
        ("router scores", scratch.scores),
        ("router choice_scores", scratch.choice_scores),
    ] {
        ensure_dtype_layout(
            tensor.dtype,
            tensor.layout,
            DType::F32,
            Layout::RowMajor,
            name,
        )?;
        ensure_len(tensor.ptr.len, padded_tokens * KIMI_K2_ROUTED_EXPERTS, name)?;
    }
    Ok(())
}

pub fn validate_expert_major_layout(layout: &ExpertMajorLayout) -> HeaderResult<()> {
    ensure_eq_usize(
        layout.local_experts,
        layout.indptr.local_experts,
        "expert-major local_experts",
    )?;
    ensure_eq_usize(layout.topk, KIMI_K2_TOPK, "expert-major topk")?;
    ensure_eq_usize(
        layout.buffers.hidden_dim,
        KIMI_K2_HIDDEN,
        "expert-major hidden_dim",
    )?;
    ensure_eq_usize(
        layout.map.route_capacity,
        layout.tokens * layout.topk,
        "expert-major route_capacity",
    )?;
    ensure_eq_usize(
        layout.buffers.route_capacity,
        layout.map.route_capacity,
        "expert-major buffer route_capacity",
    )?;
    ensure_len(
        layout.indptr.indptr.ptr.len,
        layout.local_experts + 1,
        "expert-major indptr",
    )?;
    ensure_dtype_layout(
        layout.indptr.indptr.dtype,
        layout.indptr.indptr.layout,
        DType::I32,
        Layout::ExpertMajor,
        "expert-major indptr",
    )?;
    for (name, tensor) in [
        ("expert-major pos_to_token", layout.map.pos_to_token),
        (
            "expert-major pos_to_token_topk",
            layout.map.pos_to_token_topk,
        ),
        (
            "expert-major token_topk_to_pos",
            layout.map.token_topk_to_pos,
        ),
    ] {
        ensure_dtype_layout(
            tensor.dtype,
            tensor.layout,
            DType::I32,
            Layout::ExpertMajor,
            name,
        )?;
        ensure_len(tensor.ptr.len, layout.map.route_capacity, name)?;
    }
    ensure_dtype_layout(
        layout.map.expert_cursor.dtype,
        layout.map.expert_cursor.layout,
        DType::I32,
        Layout::ExpertMajor,
        "expert-major expert_cursor",
    )?;
    ensure_len(
        layout.map.expert_cursor.ptr.len,
        layout.local_experts,
        "expert-major expert_cursor",
    )?;
    ensure_dtype_layout(
        layout.map.local_count.dtype,
        layout.map.local_count.layout,
        DType::I32,
        Layout::RowMajor,
        "expert-major local_count",
    )?;
    ensure_len(
        layout.map.local_count.ptr.len,
        1,
        "expert-major local_count",
    )?;
    for (name, tensor) in [
        (
            "expert-major expanded_hidden",
            layout.buffers.expanded_hidden,
        ),
        (
            "expert-major expanded_output",
            layout.buffers.expanded_output,
        ),
    ] {
        ensure_dtype_layout(
            tensor.dtype,
            tensor.layout,
            DType::Bf16,
            Layout::ExpertMajor,
            name,
        )?;
        ensure_len(
            tensor.ptr.len,
            layout.map.route_capacity * KIMI_K2_HIDDEN,
            name,
        )?;
    }
    Ok(())
}

pub fn router_forward_cuda_header(
    _stream: StreamHandle,
    plan: &RouterPlan,
    weights: &RouterWeights,
    scratch: &RouterScratch,
    output: &RouterOutput,
) -> HeaderResult<()> {
    validate_router_plan(plan)?;
    validate_router_weights(weights)?;
    validate_router_scratch(scratch, plan.batch.padded_tokens)?;
    validate_router_output(output, plan.batch.active_tokens)?;
    Err(HeaderError::Unsupported {
        message: "Kimi-K2.6 router CUDA body is intentionally not implemented in headers"
            .to_string(),
    })
}

pub fn expert_major_plan_cuda_header(
    _stream: StreamHandle,
    router_output: &RouterOutput,
    layout: &ExpertMajorLayout,
) -> HeaderResult<()> {
    validate_router_output(router_output, layout.tokens)?;
    validate_expert_major_layout(layout)?;
    Err(HeaderError::Unsupported {
        message:
            "Kimi-K2.6 expert-major layout CUDA body is intentionally not implemented in headers"
                .to_string(),
    })
}

fn ensure_dtype_layout(
    actual_dtype: DType,
    actual_layout: Layout,
    expected_dtype: DType,
    expected_layout: Layout,
    name: &str,
) -> HeaderResult<()> {
    if actual_dtype != expected_dtype || actual_layout != expected_layout {
        return Err(HeaderError::Shape {
            message: format!(
                "{name} expects {expected_dtype:?}/{expected_layout:?}, got {actual_dtype:?}/{actual_layout:?}"
            ),
        });
    }
    Ok(())
}

fn ensure_len(actual: usize, expected: usize, name: &str) -> HeaderResult<()> {
    if actual < expected {
        return Err(HeaderError::Shape {
            message: format!("{name} too small: have {actual}, need {expected}"),
        });
    }
    Ok(())
}

fn ensure_eq_usize(actual: usize, expected: usize, name: &str) -> HeaderResult<()> {
    if actual != expected {
        return Err(HeaderError::Shape {
            message: format!("{name} mismatch: expected {expected}, got {actual}"),
        });
    }
    Ok(())
}
