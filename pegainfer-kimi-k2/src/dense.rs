//! BF16 dense, shared-expert, and logits operator headers for Kimi-K2.6.
//!
//! This module intentionally contains no CUDA bodies. The signatures mirror the
//! existing batched PegaInfer primitives: row-major BF16 weights, column-major
//! `[dim, tokens]` activations, FlashInfer-backed RMSNorm/top1 where available,
//! and cuBLAS-backed GEMM.

use crate::{
    config::{
        KIMI_K2_DENSE_INTERMEDIATE, KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_VOCAB,
    },
    tensor::{
        Bf16, DType, HeaderError, HeaderResult, Layout, Shape2, StreamHandle, TensorMut, TensorRef,
        TokenBatch, U8, U32, VocabShard,
    },
};

pub const FLASHINFER_TOP1_ROW_STATES_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearRole {
    Embedding,
    DenseGate,
    DenseUp,
    DenseDown,
    SharedGate,
    SharedUp,
    SharedDown,
    Router,
    LmHead,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearBackend {
    CublasGemm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmsNormBackend {
    FlashInferBatch,
    FlashInferVec,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Top1Backend {
    FlashInfer,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bf16Linear {
    pub role: LinearRole,
    pub weight: TensorRef<Bf16>,
    pub shape: Shape2,
    pub backend: LinearBackend,
}

impl Bf16Linear {
    #[must_use]
    pub const fn new(
        role: LinearRole,
        weight: TensorRef<Bf16>,
        out_dim: usize,
        in_dim: usize,
    ) -> Self {
        Self {
            role,
            weight,
            shape: Shape2 {
                rows: out_dim,
                cols: in_dim,
            },
            backend: LinearBackend::CublasGemm,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RmsNorm {
    pub weight: TensorRef<Bf16>,
    pub hidden_dim: usize,
    pub eps: f32,
    pub backend: RmsNormBackend,
}

impl RmsNorm {
    #[must_use]
    pub const fn flashinfer_batch(weight: TensorRef<Bf16>, hidden_dim: usize, eps: f32) -> Self {
        Self {
            weight,
            hidden_dim,
            eps,
            backend: RmsNormBackend::FlashInferBatch,
        }
    }

    #[must_use]
    pub const fn flashinfer_vec(weight: TensorRef<Bf16>, hidden_dim: usize, eps: f32) -> Self {
        Self {
            weight,
            hidden_dim,
            eps,
            backend: RmsNormBackend::FlashInferVec,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseMlpWeights {
    pub gate_proj: Bf16Linear,
    pub up_proj: Bf16Linear,
    pub down_proj: Bf16Linear,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedExpertWeights {
    pub gate_proj: Bf16Linear,
    pub up_proj: Bf16Linear,
    pub down_proj: Bf16Linear,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogitsHead {
    pub final_norm: RmsNorm,
    pub lm_head: Bf16Linear,
    pub vocab_shard: Option<VocabShard>,
    pub top1_backend: Top1Backend,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseScratch {
    pub batch: TokenBatch,
    pub normed_hidden: TensorMut<Bf16>,
    pub gate: TensorMut<Bf16>,
    pub up: TensorMut<Bf16>,
    pub activation: TensorMut<Bf16>,
    pub local_logits: TensorMut<Bf16>,
    pub full_logits: Option<TensorMut<Bf16>>,
    pub top1_value_scratch: TensorMut<Bf16>,
    pub top1_row_states: TensorMut<U8>,
    pub top1_token_ids: TensorMut<U32>,
}

pub fn embedding_batch(
    stream: StreamHandle,
    embed_tokens: TensorRef<Bf16>,
    token_ids: TensorRef<U32>,
    batch: TokenBatch,
    out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    let _ = stream;
    ensure_batch(batch)?;
    ensure_ref(
        "embed_tokens",
        embed_tokens,
        DType::Bf16,
        Layout::RowMajor,
        KIMI_K2_VOCAB * KIMI_K2_HIDDEN,
    )?;
    ensure_ref(
        "token_ids",
        token_ids,
        DType::U32,
        Layout::ColumnMajor,
        batch.padded_tokens,
    )?;
    ensure_mut(
        "embedding_out",
        out,
        DType::Bf16,
        Layout::ColumnMajor,
        KIMI_K2_HIDDEN * batch.padded_tokens,
    )?;
    Ok(())
}

pub fn rms_norm_batch(
    stream: StreamHandle,
    norm: &RmsNorm,
    hidden: TensorRef<Bf16>,
    batch: TokenBatch,
    out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    let _ = stream;
    ensure_batch(batch)?;
    ensure_norm_backend(norm, RmsNormBackend::FlashInferBatch, "rms_norm_batch")?;
    ensure_ref(
        "rms_hidden",
        hidden,
        DType::Bf16,
        Layout::ColumnMajor,
        norm.hidden_dim * batch.padded_tokens,
    )?;
    ensure_mut(
        "rms_out",
        out,
        DType::Bf16,
        Layout::ColumnMajor,
        norm.hidden_dim * batch.padded_tokens,
    )?;
    Ok(())
}

pub fn rms_norm_vec(
    stream: StreamHandle,
    norm: &RmsNorm,
    hidden: TensorRef<Bf16>,
    out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    let _ = stream;
    ensure_norm_backend(norm, RmsNormBackend::FlashInferVec, "rms_norm_vec")?;
    ensure_ref(
        "rms_hidden",
        hidden,
        DType::Bf16,
        Layout::ColumnMajor,
        norm.hidden_dim,
    )?;
    ensure_mut(
        "rms_out",
        out,
        DType::Bf16,
        Layout::ColumnMajor,
        norm.hidden_dim,
    )?;
    Ok(())
}

pub fn fused_add_rms_norm_batch(
    stream: StreamHandle,
    hidden_in_out: TensorMut<Bf16>,
    residual: TensorRef<Bf16>,
    norm: &RmsNorm,
    batch: TokenBatch,
    normed_out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    let _ = stream;
    ensure_batch(batch)?;
    ensure_norm_backend(
        norm,
        RmsNormBackend::FlashInferBatch,
        "fused_add_rms_norm_batch",
    )?;
    let len = norm.hidden_dim * batch.padded_tokens;
    ensure_mut(
        "fused_hidden_in_out",
        hidden_in_out,
        DType::Bf16,
        Layout::ColumnMajor,
        len,
    )?;
    ensure_ref(
        "fused_residual",
        residual,
        DType::Bf16,
        Layout::ColumnMajor,
        len,
    )?;
    ensure_mut(
        "fused_normed_out",
        normed_out,
        DType::Bf16,
        Layout::ColumnMajor,
        len,
    )?;
    Ok(())
}

pub fn bf16_gemm_batch(
    stream: StreamHandle,
    linear: &Bf16Linear,
    x: TensorRef<Bf16>,
    batch: TokenBatch,
    out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    let _ = stream;
    ensure_batch(batch)?;
    ensure_linear(linear)?;
    ensure_ref(
        "gemm_x",
        x,
        DType::Bf16,
        Layout::ColumnMajor,
        linear.shape.cols * batch.padded_tokens,
    )?;
    ensure_mut(
        "gemm_out",
        out,
        DType::Bf16,
        Layout::ColumnMajor,
        linear.shape.rows * batch.padded_tokens,
    )?;
    Ok(())
}

pub fn silu_mul_batch(
    stream: StreamHandle,
    gate: TensorRef<Bf16>,
    up: TensorRef<Bf16>,
    intermediate_dim: usize,
    batch: TokenBatch,
    out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    let _ = stream;
    ensure_batch(batch)?;
    let len = intermediate_dim * batch.padded_tokens;
    ensure_ref("silu_gate", gate, DType::Bf16, Layout::ColumnMajor, len)?;
    ensure_ref("silu_up", up, DType::Bf16, Layout::ColumnMajor, len)?;
    ensure_mut("silu_out", out, DType::Bf16, Layout::ColumnMajor, len)?;
    Ok(())
}

pub fn silu_mul_dense(
    stream: StreamHandle,
    gate: TensorRef<Bf16>,
    up: TensorRef<Bf16>,
    batch: TokenBatch,
    out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    silu_mul_batch(stream, gate, up, KIMI_K2_DENSE_INTERMEDIATE, batch, out)
}

pub fn silu_mul_shared(
    stream: StreamHandle,
    gate: TensorRef<Bf16>,
    up: TensorRef<Bf16>,
    batch: TokenBatch,
    out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    silu_mul_batch(stream, gate, up, KIMI_K2_EXPERT_INTERMEDIATE, batch, out)
}

pub fn dense_mlp_gate_up(
    stream: StreamHandle,
    weights: &DenseMlpWeights,
    hidden: TensorRef<Bf16>,
    batch: TokenBatch,
    gate_out: TensorMut<Bf16>,
    up_out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    ensure_dense_mlp(weights)?;
    bf16_gemm_batch(stream, &weights.gate_proj, hidden, batch, gate_out)?;
    bf16_gemm_batch(stream, &weights.up_proj, hidden, batch, up_out)
}

pub fn dense_mlp_down(
    stream: StreamHandle,
    weights: &DenseMlpWeights,
    activation: TensorRef<Bf16>,
    batch: TokenBatch,
    out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    ensure_dense_mlp(weights)?;
    bf16_gemm_batch(stream, &weights.down_proj, activation, batch, out)
}

pub fn shared_expert_gate_up(
    stream: StreamHandle,
    weights: &SharedExpertWeights,
    hidden: TensorRef<Bf16>,
    batch: TokenBatch,
    gate_out: TensorMut<Bf16>,
    up_out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    ensure_shared_expert(weights)?;
    bf16_gemm_batch(stream, &weights.gate_proj, hidden, batch, gate_out)?;
    bf16_gemm_batch(stream, &weights.up_proj, hidden, batch, up_out)
}

pub fn shared_expert_down(
    stream: StreamHandle,
    weights: &SharedExpertWeights,
    activation: TensorRef<Bf16>,
    batch: TokenBatch,
    out: TensorMut<Bf16>,
) -> HeaderResult<()> {
    ensure_shared_expert(weights)?;
    bf16_gemm_batch(stream, &weights.down_proj, activation, batch, out)
}

pub fn lm_head_sharded_linear(
    stream: StreamHandle,
    head: &LogitsHead,
    normed_last_hidden: TensorRef<Bf16>,
    batch: TokenBatch,
    local_logits: TensorMut<Bf16>,
) -> HeaderResult<()> {
    ensure_logits_head(head)?;
    bf16_gemm_batch(
        stream,
        &head.lm_head,
        normed_last_hidden,
        batch,
        local_logits,
    )
}

pub fn logits_all_gather_planned(
    stream: StreamHandle,
    local_logits: TensorRef<Bf16>,
    local_vocab: usize,
    batch: TokenBatch,
    full_logits: TensorMut<Bf16>,
) -> HeaderResult<()> {
    let _ = stream;
    ensure_batch(batch)?;
    ensure_ref(
        "local_logits",
        local_logits,
        DType::Bf16,
        Layout::ColumnMajor,
        local_vocab * batch.padded_tokens,
    )?;
    ensure_mut(
        "full_logits",
        full_logits,
        DType::Bf16,
        Layout::ColumnMajor,
        KIMI_K2_VOCAB * batch.padded_tokens,
    )?;
    Err(HeaderError::Unsupported {
        message: "final logits all-gather awaits TP collective wiring".to_string(),
    })
}

pub fn greedy_top1_batch(
    stream: StreamHandle,
    logits: TensorRef<Bf16>,
    vocab_size: usize,
    batch: TokenBatch,
    top1_value_scratch: TensorMut<Bf16>,
    row_states_scratch: TensorMut<U8>,
    out_token_ids: TensorMut<U32>,
) -> HeaderResult<()> {
    let _ = stream;
    ensure_batch(batch)?;
    ensure_ref(
        "top1_logits",
        logits,
        DType::Bf16,
        Layout::ColumnMajor,
        vocab_size * batch.padded_tokens,
    )?;
    ensure_mut(
        "top1_value_scratch",
        top1_value_scratch,
        DType::Bf16,
        Layout::ColumnMajor,
        batch.padded_tokens,
    )?;
    ensure_mut(
        "top1_row_states",
        row_states_scratch,
        DType::U8,
        Layout::RowMajor,
        FLASHINFER_TOP1_ROW_STATES_BYTES,
    )?;
    ensure_mut(
        "top1_token_ids",
        out_token_ids,
        DType::U32,
        Layout::ColumnMajor,
        batch.padded_tokens,
    )?;
    Ok(())
}

fn ensure_dense_mlp(weights: &DenseMlpWeights) -> HeaderResult<()> {
    ensure_linear_dims(
        &weights.gate_proj,
        LinearRole::DenseGate,
        KIMI_K2_DENSE_INTERMEDIATE,
        KIMI_K2_HIDDEN,
    )?;
    ensure_linear_dims(
        &weights.up_proj,
        LinearRole::DenseUp,
        KIMI_K2_DENSE_INTERMEDIATE,
        KIMI_K2_HIDDEN,
    )?;
    ensure_linear_dims(
        &weights.down_proj,
        LinearRole::DenseDown,
        KIMI_K2_HIDDEN,
        KIMI_K2_DENSE_INTERMEDIATE,
    )
}

fn ensure_shared_expert(weights: &SharedExpertWeights) -> HeaderResult<()> {
    ensure_linear_dims(
        &weights.gate_proj,
        LinearRole::SharedGate,
        KIMI_K2_EXPERT_INTERMEDIATE,
        KIMI_K2_HIDDEN,
    )?;
    ensure_linear_dims(
        &weights.up_proj,
        LinearRole::SharedUp,
        KIMI_K2_EXPERT_INTERMEDIATE,
        KIMI_K2_HIDDEN,
    )?;
    ensure_linear_dims(
        &weights.down_proj,
        LinearRole::SharedDown,
        KIMI_K2_HIDDEN,
        KIMI_K2_EXPERT_INTERMEDIATE,
    )
}

fn ensure_logits_head(head: &LogitsHead) -> HeaderResult<()> {
    ensure_norm(&head.final_norm)?;
    if head.final_norm.hidden_dim != KIMI_K2_HIDDEN {
        return shape_error("final_norm dim must match Kimi hidden size");
    }
    if head.lm_head.shape.cols != KIMI_K2_HIDDEN {
        return shape_error("lm_head input dim must match Kimi hidden size");
    }
    if let Some(shard) = &head.vocab_shard {
        if shard.range.end <= shard.range.start || shard.range.end > KIMI_K2_VOCAB {
            return shape_error("lm_head vocab shard range is invalid");
        }
        if head.lm_head.shape.rows != shard.range.end - shard.range.start {
            return shape_error("lm_head rows must match vocab shard width");
        }
    } else if head.lm_head.shape.rows != KIMI_K2_VOCAB {
        return shape_error("unsharded lm_head rows must match full vocab");
    }
    ensure_linear(&head.lm_head)
}

fn ensure_linear_dims(
    linear: &Bf16Linear,
    role: LinearRole,
    out_dim: usize,
    in_dim: usize,
) -> HeaderResult<()> {
    if linear.role != role {
        return shape_error(format!(
            "linear role {:?} does not match expected {:?}",
            linear.role, role
        ));
    }
    if linear.shape.rows != out_dim || linear.shape.cols != in_dim {
        return shape_error(format!(
            "{:?} shape is [{}, {}], expected [{}, {}]",
            role, linear.shape.rows, linear.shape.cols, out_dim, in_dim
        ));
    }
    ensure_linear(linear)
}

fn ensure_linear(linear: &Bf16Linear) -> HeaderResult<()> {
    ensure_ref(
        "linear_weight",
        linear.weight,
        DType::Bf16,
        Layout::RowMajor,
        linear.shape.rows * linear.shape.cols,
    )
}

fn ensure_norm(norm: &RmsNorm) -> HeaderResult<()> {
    if norm.hidden_dim == 0 {
        return shape_error("RMSNorm hidden_dim must be non-zero");
    }
    if norm.eps <= 0.0 {
        return shape_error("RMSNorm eps must be positive");
    }
    ensure_ref(
        "rms_weight",
        norm.weight,
        DType::Bf16,
        Layout::ColumnMajor,
        norm.hidden_dim,
    )
}

fn ensure_norm_backend(
    norm: &RmsNorm,
    expected: RmsNormBackend,
    api_name: &str,
) -> HeaderResult<()> {
    if norm.backend != expected {
        return shape_error(format!(
            "{api_name} requires {:?}, got {:?}",
            expected, norm.backend
        ));
    }
    ensure_norm(norm)
}

fn ensure_batch(batch: TokenBatch) -> HeaderResult<()> {
    if batch.batch_size == 0 {
        return shape_error("batch_size must be non-zero");
    }
    if batch.active_tokens == 0 {
        return shape_error("active_tokens must be non-zero");
    }
    if batch.active_tokens > batch.padded_tokens {
        return shape_error("active_tokens cannot exceed padded_tokens");
    }
    Ok(())
}

fn ensure_ref<T>(
    name: &str,
    tensor: TensorRef<T>,
    dtype: DType,
    layout: Layout,
    min_len: usize,
) -> HeaderResult<()> {
    if tensor.dtype != dtype {
        return shape_error(format!("{name} dtype {:?} != {:?}", tensor.dtype, dtype));
    }
    if tensor.layout != layout {
        return shape_error(format!("{name} layout {:?} != {:?}", tensor.layout, layout));
    }
    if tensor.ptr.len < min_len {
        return shape_error(format!(
            "{name} len {} < required {min_len}",
            tensor.ptr.len
        ));
    }
    Ok(())
}

fn ensure_mut<T>(
    name: &str,
    tensor: TensorMut<T>,
    dtype: DType,
    layout: Layout,
    min_len: usize,
) -> HeaderResult<()> {
    if tensor.dtype != dtype {
        return shape_error(format!("{name} dtype {:?} != {:?}", tensor.dtype, dtype));
    }
    if tensor.layout != layout {
        return shape_error(format!("{name} layout {:?} != {:?}", tensor.layout, layout));
    }
    if tensor.ptr.len < min_len {
        return shape_error(format!(
            "{name} len {} < required {min_len}",
            tensor.ptr.len
        ));
    }
    Ok(())
}

fn shape_error<T>(message: impl Into<String>) -> HeaderResult<T> {
    Err(HeaderError::Shape {
        message: message.into(),
    })
}
