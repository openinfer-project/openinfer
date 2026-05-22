//! Kimi-K2.6 text-only MLA attention header/API draft.
//!
//! This module is intentionally a compile-checked contract only. It names the
//! buffers, cache layouts, batch metadata, and backend selection needed by the
//! future CUDA implementation, while the functions below only validate shapes
//! and return launch headers.

use crate::config::{
    KIMI_K2_HEADS, KIMI_K2_HIDDEN, KIMI_K2_KV_A_OUT, KIMI_K2_KV_B_OUT, KIMI_K2_KV_LORA_RANK,
    KIMI_K2_MAX_CONTEXT, KIMI_K2_O_PROJ_IN, KIMI_K2_Q_HEAD_DIM, KIMI_K2_Q_LORA_RANK,
    KIMI_K2_Q_PROJ_OUT, KIMI_K2_QK_NOPE_HEAD_DIM, KIMI_K2_QK_ROPE_HEAD_DIM, KIMI_K2_ROPE_THETA,
    KIMI_K2_V_HEAD_DIM, KIMI_K2_YARN_BETA_FAST, KIMI_K2_YARN_BETA_SLOW, KIMI_K2_YARN_FACTOR,
    KIMI_K2_YARN_ORIGINAL_MAX_POS, KimiK2ParallelShape,
};
use crate::tensor::{
    Bf16, DType, F32, HeaderError, HeaderResult, Shape2, Shape3, StreamHandle, TensorMut,
    TensorRef, TokenBatch, U32,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor1Ref<T> {
    pub tensor: TensorRef<T>,
    pub len: usize,
}

impl<T> Tensor1Ref<T> {
    #[must_use]
    pub const fn new(tensor: TensorRef<T>, len: usize) -> Self {
        Self { tensor, len }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor1Mut<T> {
    pub tensor: TensorMut<T>,
    pub len: usize,
}

impl<T> Tensor1Mut<T> {
    #[must_use]
    pub const fn new(tensor: TensorMut<T>, len: usize) -> Self {
        Self { tensor, len }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor2Ref<T> {
    pub tensor: TensorRef<T>,
    pub shape: Shape2,
}

impl<T> Tensor2Ref<T> {
    #[must_use]
    pub const fn new(tensor: TensorRef<T>, shape: Shape2) -> Self {
        Self { tensor, shape }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor2Mut<T> {
    pub tensor: TensorMut<T>,
    pub shape: Shape2,
}

impl<T> Tensor2Mut<T> {
    #[must_use]
    pub const fn new(tensor: TensorMut<T>, shape: Shape2) -> Self {
        Self { tensor, shape }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor3Ref<T> {
    pub tensor: TensorRef<T>,
    pub shape: Shape3,
}

impl<T> Tensor3Ref<T> {
    #[must_use]
    pub const fn new(tensor: TensorRef<T>, shape: Shape3) -> Self {
        Self { tensor, shape }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor3Mut<T> {
    pub tensor: TensorMut<T>,
    pub shape: Shape3,
}

impl<T> Tensor3Mut<T> {
    #[must_use]
    pub const fn new(tensor: TensorMut<T>, shape: Shape3) -> Self {
        Self { tensor, shape }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlaRuntimeShape {
    pub parallel: KimiK2ParallelShape,
    pub local_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_head_dim: usize,
    pub v_head_dim: usize,
}

impl MlaRuntimeShape {
    #[must_use]
    pub fn new(parallel: KimiK2ParallelShape) -> Self {
        Self {
            parallel,
            local_heads: parallel.heads_per_tp,
            q_lora_rank: KIMI_K2_Q_LORA_RANK,
            kv_lora_rank: KIMI_K2_KV_LORA_RANK,
            qk_nope_head_dim: KIMI_K2_QK_NOPE_HEAD_DIM,
            qk_rope_head_dim: KIMI_K2_QK_ROPE_HEAD_DIM,
            q_head_dim: KIMI_K2_Q_HEAD_DIM,
            v_head_dim: KIMI_K2_V_HEAD_DIM,
        }
    }

    #[must_use]
    pub fn single_rank() -> Self {
        Self::new(KimiK2ParallelShape::new(1, 1))
    }

    #[must_use]
    pub fn local_q_proj_out(self) -> usize {
        self.local_heads * self.q_head_dim
    }

    #[must_use]
    pub fn local_kv_b_out(self) -> usize {
        self.local_heads * (self.qk_nope_head_dim + self.v_head_dim)
    }

    #[must_use]
    pub fn local_o_proj_in(self) -> usize {
        self.local_heads * self.v_head_dim
    }

    pub fn validate(self) -> HeaderResult<()> {
        if self.parallel.tp_world == 0 || self.parallel.ep_world == 0 {
            return shape_err("parallel world sizes must be non-zero");
        }
        if KIMI_K2_HEADS % self.parallel.tp_world != 0 {
            return shape_err("attention heads must divide evenly across tensor parallel ranks");
        }
        if self.local_heads == 0 {
            return shape_err("local_heads must be non-zero");
        }
        if self.q_head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim {
            return shape_err("q_head_dim must equal qk_nope_head_dim + qk_rope_head_dim");
        }
        if self.q_lora_rank != KIMI_K2_Q_LORA_RANK
            || self.kv_lora_rank != KIMI_K2_KV_LORA_RANK
            || self.qk_nope_head_dim != KIMI_K2_QK_NOPE_HEAD_DIM
            || self.qk_rope_head_dim != KIMI_K2_QK_ROPE_HEAD_DIM
            || self.v_head_dim != KIMI_K2_V_HEAD_DIM
        {
            return shape_err("runtime MLA shape must match Kimi-K2.6 text config");
        }
        Ok(())
    }
}

impl Default for MlaRuntimeShape {
    fn default() -> Self {
        Self::single_rank()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlaProjectionWeights {
    pub q_a_proj: Tensor2Ref<Bf16>,
    pub q_a_layernorm: Tensor1Ref<Bf16>,
    pub q_b_proj: Tensor2Ref<Bf16>,
    pub kv_a_proj_with_mqa: Tensor2Ref<Bf16>,
    pub kv_a_layernorm: Tensor1Ref<Bf16>,
    pub kv_b_proj: Tensor2Ref<Bf16>,
    pub o_proj: Tensor2Ref<Bf16>,
}

impl MlaProjectionWeights {
    pub fn validate(self, shape: MlaRuntimeShape) -> HeaderResult<()> {
        shape.validate()?;
        expect_2d_ref(
            "q_a_proj",
            self.q_a_proj,
            KIMI_K2_Q_LORA_RANK,
            KIMI_K2_HIDDEN,
        )?;
        expect_1d_ref("q_a_layernorm", self.q_a_layernorm, KIMI_K2_Q_LORA_RANK)?;
        expect_2d_ref(
            "q_b_proj",
            self.q_b_proj,
            shape.local_q_proj_out(),
            KIMI_K2_Q_LORA_RANK,
        )?;
        expect_2d_ref(
            "kv_a_proj_with_mqa",
            self.kv_a_proj_with_mqa,
            KIMI_K2_KV_A_OUT,
            KIMI_K2_HIDDEN,
        )?;
        expect_1d_ref("kv_a_layernorm", self.kv_a_layernorm, KIMI_K2_KV_LORA_RANK)?;
        expect_2d_ref(
            "kv_b_proj",
            self.kv_b_proj,
            shape.local_kv_b_out(),
            KIMI_K2_KV_LORA_RANK,
        )?;
        expect_2d_ref(
            "o_proj",
            self.o_proj,
            KIMI_K2_HIDDEN,
            shape.local_o_proj_in(),
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlaProjectionBuffers {
    pub hidden: Tensor2Ref<Bf16>,
    pub q_a: Tensor2Mut<Bf16>,
    pub q: Tensor3Mut<Bf16>,
    pub q_nope: Tensor3Mut<Bf16>,
    pub q_rope: Tensor3Mut<Bf16>,
    pub kv_a_with_mqa: Tensor2Mut<Bf16>,
    pub compressed_kv: Tensor2Mut<Bf16>,
    pub k_rope: Tensor3Mut<Bf16>,
    pub kv_a_normed: Tensor2Mut<Bf16>,
    pub k_nope: Tensor3Mut<Bf16>,
    pub value: Tensor3Mut<Bf16>,
    pub attention_out: Tensor3Mut<Bf16>,
    pub o_proj_out: Tensor2Mut<Bf16>,
}

impl MlaProjectionBuffers {
    pub fn validate(self, batch: TokenBatch, shape: MlaRuntimeShape) -> HeaderResult<()> {
        validate_token_batch(batch)?;
        shape.validate()?;
        let tokens = batch.padded_tokens;
        expect_2d_ref("hidden", self.hidden, tokens, KIMI_K2_HIDDEN)?;
        expect_2d_mut("q_a", self.q_a, tokens, shape.q_lora_rank)?;
        expect_3d_mut("q", self.q, tokens, shape.local_heads, shape.q_head_dim)?;
        expect_3d_mut(
            "q_nope",
            self.q_nope,
            tokens,
            shape.local_heads,
            shape.qk_nope_head_dim,
        )?;
        expect_3d_mut(
            "q_rope",
            self.q_rope,
            tokens,
            shape.local_heads,
            shape.qk_rope_head_dim,
        )?;
        expect_2d_mut(
            "kv_a_with_mqa",
            self.kv_a_with_mqa,
            tokens,
            KIMI_K2_KV_A_OUT,
        )?;
        expect_2d_mut(
            "compressed_kv",
            self.compressed_kv,
            tokens,
            shape.kv_lora_rank,
        )?;
        expect_3d_mut("k_rope", self.k_rope, tokens, 1, shape.qk_rope_head_dim)?;
        expect_2d_mut("kv_a_normed", self.kv_a_normed, tokens, shape.kv_lora_rank)?;
        expect_3d_mut(
            "k_nope",
            self.k_nope,
            tokens,
            shape.local_heads,
            shape.qk_nope_head_dim,
        )?;
        expect_3d_mut(
            "value",
            self.value,
            tokens,
            shape.local_heads,
            shape.v_head_dim,
        )?;
        expect_3d_mut(
            "attention_out",
            self.attention_out,
            tokens,
            shape.local_heads,
            shape.v_head_dim,
        )?;
        expect_2d_mut("o_proj_out", self.o_proj_out, tokens, KIMI_K2_HIDDEN)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlaProjectionHeader {
    pub batch: TokenBatch,
    pub shape: MlaRuntimeShape,
    pub weights: MlaProjectionWeights,
    pub buffers: MlaProjectionBuffers,
}

pub fn mla_projection_header(
    batch: TokenBatch,
    shape: MlaRuntimeShape,
    weights: MlaProjectionWeights,
    buffers: MlaProjectionBuffers,
) -> HeaderResult<MlaProjectionHeader> {
    weights.validate(shape)?;
    buffers.validate(batch, shape)?;
    Ok(MlaProjectionHeader {
        batch,
        shape,
        weights,
        buffers,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YarnRopeConfig {
    pub dim: usize,
    pub theta: f32,
    pub factor: f32,
    pub original_max_position: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub max_position: usize,
}

impl YarnRopeConfig {
    #[must_use]
    pub const fn kimi_k2(max_position: usize) -> Self {
        Self {
            dim: KIMI_K2_QK_ROPE_HEAD_DIM,
            theta: KIMI_K2_ROPE_THETA,
            factor: KIMI_K2_YARN_FACTOR,
            original_max_position: KIMI_K2_YARN_ORIGINAL_MAX_POS,
            beta_fast: KIMI_K2_YARN_BETA_FAST,
            beta_slow: KIMI_K2_YARN_BETA_SLOW,
            max_position,
        }
    }

    pub fn validate(self) -> HeaderResult<()> {
        if self.dim != KIMI_K2_QK_ROPE_HEAD_DIM {
            return shape_err("YARN RoPE dim must be 64 for Kimi-K2.6 MLA");
        }
        if self.max_position == 0 || self.max_position > KIMI_K2_MAX_CONTEXT {
            return shape_err("YARN RoPE max_position must be in 1..=KIMI_K2_MAX_CONTEXT");
        }
        if self.theta <= 0.0 || self.factor <= 0.0 {
            return shape_err("YARN RoPE theta and factor must be positive");
        }
        if self.original_max_position == 0 {
            return shape_err("YARN RoPE original_max_position must be non-zero");
        }
        if self.beta_slow <= 0.0 || self.beta_fast < self.beta_slow {
            return shape_err("YARN RoPE beta range is invalid");
        }
        Ok(())
    }
}

impl Default for YarnRopeConfig {
    fn default() -> Self {
        Self::kimi_k2(KIMI_K2_MAX_CONTEXT)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct YarnRopeCache {
    pub cos: Tensor2Ref<F32>,
    pub sin: Tensor2Ref<F32>,
}

impl YarnRopeCache {
    pub fn validate(self, config: YarnRopeConfig) -> HeaderResult<()> {
        config.validate()?;
        expect_2d_ref("rope.cos", self.cos, config.max_position, config.dim)?;
        expect_2d_ref("rope.sin", self.sin, config.max_position, config.dim)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YarnRopeCacheHeader {
    pub config: YarnRopeConfig,
    pub cache: YarnRopeCache,
}

pub fn yarn_rope_cache_header(
    config: YarnRopeConfig,
    cache: YarnRopeCache,
) -> HeaderResult<YarnRopeCacheHeader> {
    cache.validate(config)?;
    Ok(YarnRopeCacheHeader { config, cache })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartialRopeHeader {
    pub batch: TokenBatch,
    pub shape: MlaRuntimeShape,
    pub positions: Tensor1Ref<U32>,
    pub cache: YarnRopeCache,
    pub q_rope: Tensor3Mut<Bf16>,
    pub k_rope: Tensor3Mut<Bf16>,
}

pub fn partial_yarn_rope_header(
    batch: TokenBatch,
    shape: MlaRuntimeShape,
    positions: Tensor1Ref<U32>,
    cache_header: YarnRopeCacheHeader,
    q_rope: Tensor3Mut<Bf16>,
    k_rope: Tensor3Mut<Bf16>,
) -> HeaderResult<PartialRopeHeader> {
    validate_token_batch(batch)?;
    shape.validate()?;
    cache_header.cache.validate(cache_header.config)?;
    expect_1d_ref("positions", positions, batch.padded_tokens)?;
    expect_3d_mut(
        "q_rope",
        q_rope,
        batch.padded_tokens,
        shape.local_heads,
        shape.qk_rope_head_dim,
    )?;
    expect_3d_mut(
        "k_rope",
        k_rope,
        batch.padded_tokens,
        1,
        shape.qk_rope_head_dim,
    )?;
    Ok(PartialRopeHeader {
        batch,
        shape,
        positions,
        cache: cache_header.cache,
        q_rope,
        k_rope,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionBackend {
    FlashInfer,
    HandWrittenFallback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchDecodeAttentionPath {
    ExpandedCorrectness,
    CompressedProduction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlashInferMlaSupport {
    pub paged_decode: bool,
    pub qk_dim_192: bool,
    pub v_dim_128: bool,
    pub bf16: bool,
}

impl FlashInferMlaSupport {
    #[must_use]
    pub const fn assume_reusable() -> Self {
        Self {
            paged_decode: true,
            qk_dim_192: true,
            v_dim_128: true,
            bf16: true,
        }
    }

    #[must_use]
    pub const fn missing_qk192_v128() -> Self {
        Self {
            paged_decode: true,
            qk_dim_192: false,
            v_dim_128: false,
            bf16: true,
        }
    }

    #[must_use]
    pub const fn selects_flashinfer(self) -> bool {
        self.paged_decode && self.qk_dim_192 && self.v_dim_128 && self.bf16
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchDecodeAttentionPlan {
    pub batch: TokenBatch,
    pub shape: MlaRuntimeShape,
    pub max_kv_len: usize,
    pub path: BatchDecodeAttentionPath,
    pub backend: AttentionBackend,
    pub flashinfer_support: FlashInferMlaSupport,
    pub cuda_graph_bucket: Option<usize>,
}

impl BatchDecodeAttentionPlan {
    pub fn new(
        batch: TokenBatch,
        shape: MlaRuntimeShape,
        max_kv_len: usize,
        path: BatchDecodeAttentionPath,
        flashinfer_support: FlashInferMlaSupport,
        cuda_graph_bucket: Option<usize>,
    ) -> HeaderResult<Self> {
        validate_decode_batch(batch)?;
        shape.validate()?;
        if max_kv_len == 0 || max_kv_len > KIMI_K2_MAX_CONTEXT {
            return shape_err("max_kv_len must be in 1..=KIMI_K2_MAX_CONTEXT");
        }
        if let Some(bucket) = cuda_graph_bucket {
            if bucket < batch.padded_tokens {
                return shape_err("cuda_graph_bucket must be >= padded decode batch size");
            }
        }
        let backend = if flashinfer_support.selects_flashinfer() {
            AttentionBackend::FlashInfer
        } else {
            AttentionBackend::HandWrittenFallback
        };
        Ok(Self {
            batch,
            shape,
            max_kv_len,
            path,
            backend,
            flashinfer_support,
            cuda_graph_bucket,
        })
    }

    #[must_use]
    pub fn batch_size(self) -> usize {
        self.batch.batch_size
    }

    #[must_use]
    pub fn padded_batch_size(self) -> usize {
        self.batch.padded_tokens
    }

    #[must_use]
    pub fn supports_bs_gt_one(self) -> bool {
        self.batch.batch_size > 1 && self.batch.active_tokens == self.batch.batch_size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeRequestMeta {
    pub positions: Tensor1Ref<U32>,
    pub kv_lens: Tensor1Ref<U32>,
    pub block_tables: Option<Tensor2Ref<U32>>,
}

impl DecodeRequestMeta {
    pub fn validate(self, batch: TokenBatch, max_kv_len: usize) -> HeaderResult<()> {
        validate_decode_batch(batch)?;
        expect_1d_ref("decode.positions", self.positions, batch.padded_tokens)?;
        expect_1d_ref("decode.kv_lens", self.kv_lens, batch.padded_tokens)?;
        if let Some(block_tables) = self.block_tables {
            if block_tables.shape.rows != batch.padded_tokens || block_tables.shape.cols == 0 {
                return shape_err("decode.block_tables must be [padded_bs, nonzero_blocks]");
            }
            expect_dtype("decode.block_tables", block_tables.tensor.dtype, DType::U32)?;
        }
        if max_kv_len == 0 || max_kv_len > KIMI_K2_MAX_CONTEXT {
            return shape_err("decode max_kv_len is outside the supported context");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpandedKvLayout {
    DenseBatchMajor,
    PagedFlashInfer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpandedKvCache {
    pub k: TensorRef<Bf16>,
    pub v: TensorRef<Bf16>,
    pub layout: ExpandedKvLayout,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub heads: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
}

impl ExpandedKvCache {
    pub fn validate(
        self,
        shape: MlaRuntimeShape,
        batch_size: usize,
        max_kv_len: usize,
    ) -> HeaderResult<()> {
        shape.validate()?;
        if self.batch_size < batch_size {
            return shape_err("expanded KV cache batch_size must cover the active batch");
        }
        if self.max_seq_len < max_kv_len {
            return shape_err("expanded KV cache max_seq_len must cover the decode plan");
        }
        if self.heads != shape.local_heads
            || self.qk_head_dim != shape.q_head_dim
            || self.v_head_dim != shape.v_head_dim
        {
            return shape_err("expanded KV cache head dimensions must match MLA expanded Q/K/V");
        }
        expect_dtype("expanded_k_cache", self.k.dtype, DType::Bf16)?;
        expect_dtype("expanded_v_cache", self.v.dtype, DType::Bf16)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpandedPrefillAttentionHeader {
    pub batch: TokenBatch,
    pub shape: MlaRuntimeShape,
    pub q: Tensor3Ref<Bf16>,
    pub k: Tensor3Ref<Bf16>,
    pub v: Tensor3Ref<Bf16>,
    pub out: Tensor3Mut<Bf16>,
    pub causal: bool,
    pub stream: StreamHandle,
}

pub fn expanded_prefill_attention_header(
    batch: TokenBatch,
    shape: MlaRuntimeShape,
    q: Tensor3Ref<Bf16>,
    k: Tensor3Ref<Bf16>,
    v: Tensor3Ref<Bf16>,
    out: Tensor3Mut<Bf16>,
    causal: bool,
    stream: StreamHandle,
) -> HeaderResult<ExpandedPrefillAttentionHeader> {
    validate_token_batch(batch)?;
    shape.validate()?;
    let tokens = batch.padded_tokens;
    expect_3d_ref(
        "expanded_prefill.q",
        q,
        tokens,
        shape.local_heads,
        shape.q_head_dim,
    )?;
    expect_3d_ref(
        "expanded_prefill.k",
        k,
        tokens,
        shape.local_heads,
        shape.q_head_dim,
    )?;
    expect_3d_ref(
        "expanded_prefill.v",
        v,
        tokens,
        shape.local_heads,
        shape.v_head_dim,
    )?;
    expect_3d_mut(
        "expanded_prefill.out",
        out,
        tokens,
        shape.local_heads,
        shape.v_head_dim,
    )?;
    Ok(ExpandedPrefillAttentionHeader {
        batch,
        shape,
        q,
        k,
        v,
        out,
        causal,
        stream,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpandedDecodeAttentionHeader {
    pub plan: BatchDecodeAttentionPlan,
    pub meta: DecodeRequestMeta,
    pub q: Tensor3Ref<Bf16>,
    pub new_k: Tensor3Ref<Bf16>,
    pub new_v: Tensor3Ref<Bf16>,
    pub cache: ExpandedKvCache,
    pub out: Tensor3Mut<Bf16>,
    pub stream: StreamHandle,
}

pub fn expanded_decode_attention_header(
    plan: BatchDecodeAttentionPlan,
    meta: DecodeRequestMeta,
    q: Tensor3Ref<Bf16>,
    new_k: Tensor3Ref<Bf16>,
    new_v: Tensor3Ref<Bf16>,
    cache: ExpandedKvCache,
    out: Tensor3Mut<Bf16>,
    stream: StreamHandle,
) -> HeaderResult<ExpandedDecodeAttentionHeader> {
    if plan.path != BatchDecodeAttentionPath::ExpandedCorrectness {
        return unsupported("expanded decode header requires ExpandedCorrectness path");
    }
    meta.validate(plan.batch, plan.max_kv_len)?;
    cache.validate(plan.shape, plan.batch.batch_size, plan.max_kv_len)?;
    let bs = plan.batch.padded_tokens;
    expect_3d_ref(
        "expanded_decode.q",
        q,
        bs,
        plan.shape.local_heads,
        plan.shape.q_head_dim,
    )?;
    expect_3d_ref(
        "expanded_decode.new_k",
        new_k,
        bs,
        plan.shape.local_heads,
        plan.shape.q_head_dim,
    )?;
    expect_3d_ref(
        "expanded_decode.new_v",
        new_v,
        bs,
        plan.shape.local_heads,
        plan.shape.v_head_dim,
    )?;
    expect_3d_mut(
        "expanded_decode.out",
        out,
        bs,
        plan.shape.local_heads,
        plan.shape.v_head_dim,
    )?;
    Ok(ExpandedDecodeAttentionHeader {
        plan,
        meta,
        q,
        new_k,
        new_v,
        cache,
        out,
        stream,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompressedKvLayout {
    DenseBatchMajor,
    Paged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedKvCache {
    pub compressed_kv: TensorRef<Bf16>,
    pub k_rope: TensorRef<Bf16>,
    pub layout: CompressedKvLayout,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub kv_lora_rank: usize,
    pub rope_dim: usize,
}

impl CompressedKvCache {
    pub fn validate(
        self,
        shape: MlaRuntimeShape,
        batch_size: usize,
        max_kv_len: usize,
    ) -> HeaderResult<()> {
        shape.validate()?;
        if self.batch_size < batch_size {
            return shape_err("compressed KV cache batch_size must cover the active batch");
        }
        if self.max_seq_len < max_kv_len {
            return shape_err("compressed KV cache max_seq_len must cover the decode plan");
        }
        if self.kv_lora_rank != shape.kv_lora_rank || self.rope_dim != shape.qk_rope_head_dim {
            return shape_err(
                "compressed KV cache dimensions must be [kv_lora_rank=512, rope_dim=64]",
            );
        }
        expect_dtype("compressed_kv_cache", self.compressed_kv.dtype, DType::Bf16)?;
        expect_dtype("compressed_k_rope_cache", self.k_rope.dtype, DType::Bf16)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedKvWriteHeader {
    pub batch: TokenBatch,
    pub shape: MlaRuntimeShape,
    pub meta: DecodeRequestMeta,
    pub compressed_kv: Tensor2Ref<Bf16>,
    pub k_rope: Tensor3Ref<Bf16>,
    pub cache: CompressedKvCache,
    pub stream: StreamHandle,
}

pub fn compressed_kv_write_header(
    batch: TokenBatch,
    shape: MlaRuntimeShape,
    meta: DecodeRequestMeta,
    compressed_kv: Tensor2Ref<Bf16>,
    k_rope: Tensor3Ref<Bf16>,
    cache: CompressedKvCache,
    stream: StreamHandle,
) -> HeaderResult<CompressedKvWriteHeader> {
    validate_decode_batch(batch)?;
    meta.validate(batch, cache.max_seq_len)?;
    cache.validate(shape, batch.batch_size, cache.max_seq_len)?;
    expect_2d_ref(
        "compressed_kv.write_input",
        compressed_kv,
        batch.padded_tokens,
        shape.kv_lora_rank,
    )?;
    expect_3d_ref(
        "compressed_k_rope.write_input",
        k_rope,
        batch.padded_tokens,
        1,
        shape.qk_rope_head_dim,
    )?;
    Ok(CompressedKvWriteHeader {
        batch,
        shape,
        meta,
        compressed_kv,
        k_rope,
        cache,
        stream,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedPrefillAttentionHeader {
    pub batch: TokenBatch,
    pub shape: MlaRuntimeShape,
    pub q: Tensor3Ref<Bf16>,
    pub compressed_kv: Tensor2Ref<Bf16>,
    pub k_rope: Tensor3Ref<Bf16>,
    pub kv_b_proj: Tensor2Ref<Bf16>,
    pub out: Tensor3Mut<Bf16>,
    pub stream: StreamHandle,
}

pub fn compressed_prefill_attention_header(
    batch: TokenBatch,
    shape: MlaRuntimeShape,
    q: Tensor3Ref<Bf16>,
    compressed_kv: Tensor2Ref<Bf16>,
    k_rope: Tensor3Ref<Bf16>,
    kv_b_proj: Tensor2Ref<Bf16>,
    out: Tensor3Mut<Bf16>,
    stream: StreamHandle,
) -> HeaderResult<CompressedPrefillAttentionHeader> {
    validate_token_batch(batch)?;
    shape.validate()?;
    let tokens = batch.padded_tokens;
    expect_3d_ref(
        "compressed_prefill.q",
        q,
        tokens,
        shape.local_heads,
        shape.q_head_dim,
    )?;
    expect_2d_ref(
        "compressed_prefill.compressed_kv",
        compressed_kv,
        tokens,
        shape.kv_lora_rank,
    )?;
    expect_3d_ref(
        "compressed_prefill.k_rope",
        k_rope,
        tokens,
        1,
        shape.qk_rope_head_dim,
    )?;
    expect_2d_ref(
        "compressed_prefill.kv_b_proj",
        kv_b_proj,
        shape.local_kv_b_out(),
        shape.kv_lora_rank,
    )?;
    expect_3d_mut(
        "compressed_prefill.out",
        out,
        tokens,
        shape.local_heads,
        shape.v_head_dim,
    )?;
    Ok(CompressedPrefillAttentionHeader {
        batch,
        shape,
        q,
        compressed_kv,
        k_rope,
        kv_b_proj,
        out,
        stream,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedDecodeAttentionHeader {
    pub plan: BatchDecodeAttentionPlan,
    pub meta: DecodeRequestMeta,
    pub q: Tensor3Ref<Bf16>,
    pub new_compressed_kv: Tensor2Ref<Bf16>,
    pub new_k_rope: Tensor3Ref<Bf16>,
    pub cache: CompressedKvCache,
    pub kv_b_proj: Tensor2Ref<Bf16>,
    pub out: Tensor3Mut<Bf16>,
    pub stream: StreamHandle,
}

pub fn compressed_decode_attention_header(
    plan: BatchDecodeAttentionPlan,
    meta: DecodeRequestMeta,
    q: Tensor3Ref<Bf16>,
    new_compressed_kv: Tensor2Ref<Bf16>,
    new_k_rope: Tensor3Ref<Bf16>,
    cache: CompressedKvCache,
    kv_b_proj: Tensor2Ref<Bf16>,
    out: Tensor3Mut<Bf16>,
    stream: StreamHandle,
) -> HeaderResult<CompressedDecodeAttentionHeader> {
    if plan.path != BatchDecodeAttentionPath::CompressedProduction {
        return unsupported("compressed decode header requires CompressedProduction path");
    }
    meta.validate(plan.batch, plan.max_kv_len)?;
    cache.validate(plan.shape, plan.batch.batch_size, plan.max_kv_len)?;
    let bs = plan.batch.padded_tokens;
    expect_3d_ref(
        "compressed_decode.q",
        q,
        bs,
        plan.shape.local_heads,
        plan.shape.q_head_dim,
    )?;
    expect_2d_ref(
        "compressed_decode.new_compressed_kv",
        new_compressed_kv,
        bs,
        plan.shape.kv_lora_rank,
    )?;
    expect_3d_ref(
        "compressed_decode.new_k_rope",
        new_k_rope,
        bs,
        1,
        plan.shape.qk_rope_head_dim,
    )?;
    expect_2d_ref(
        "compressed_decode.kv_b_proj",
        kv_b_proj,
        plan.shape.local_kv_b_out(),
        plan.shape.kv_lora_rank,
    )?;
    expect_3d_mut(
        "compressed_decode.out",
        out,
        bs,
        plan.shape.local_heads,
        plan.shape.v_head_dim,
    )?;
    Ok(CompressedDecodeAttentionHeader {
        plan,
        meta,
        q,
        new_compressed_kv,
        new_k_rope,
        cache,
        kv_b_proj,
        out,
        stream,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionApiSummary {
    pub full_q_proj_out: usize,
    pub full_kv_a_out: usize,
    pub full_kv_b_out: usize,
    pub full_o_proj_in: usize,
    pub qk_dim: usize,
    pub v_dim: usize,
}

#[must_use]
pub const fn attention_api_summary() -> AttentionApiSummary {
    AttentionApiSummary {
        full_q_proj_out: KIMI_K2_Q_PROJ_OUT,
        full_kv_a_out: KIMI_K2_KV_A_OUT,
        full_kv_b_out: KIMI_K2_KV_B_OUT,
        full_o_proj_in: KIMI_K2_O_PROJ_IN,
        qk_dim: KIMI_K2_Q_HEAD_DIM,
        v_dim: KIMI_K2_V_HEAD_DIM,
    }
}

fn validate_token_batch(batch: TokenBatch) -> HeaderResult<()> {
    if batch.batch_size == 0 {
        return shape_err("batch_size must be non-zero");
    }
    if batch.active_tokens == 0 {
        return shape_err("active_tokens must be non-zero");
    }
    if batch.padded_tokens < batch.active_tokens {
        return shape_err("padded_tokens must be >= active_tokens");
    }
    Ok(())
}

fn validate_decode_batch(batch: TokenBatch) -> HeaderResult<()> {
    validate_token_batch(batch)?;
    if batch.active_tokens != batch.batch_size {
        return shape_err("decode batch must have one active token per request");
    }
    if batch.padded_tokens < batch.batch_size {
        return shape_err("decode padded_tokens must be >= batch_size");
    }
    Ok(())
}

fn expect_1d_ref<T>(name: &str, tensor: Tensor1Ref<T>, len: usize) -> HeaderResult<()> {
    if tensor.len != len {
        return shape_err(format!("{name} expected len {len}, got {}", tensor.len));
    }
    expect_dtype(name, tensor.tensor.dtype, dtype_for::<T>())
}

fn expect_2d_ref<T>(
    name: &str,
    tensor: Tensor2Ref<T>,
    rows: usize,
    cols: usize,
) -> HeaderResult<()> {
    if tensor.shape.rows != rows || tensor.shape.cols != cols {
        return shape_err(format!(
            "{name} expected shape [{rows}, {cols}], got [{}, {}]",
            tensor.shape.rows, tensor.shape.cols
        ));
    }
    expect_dtype(name, tensor.tensor.dtype, dtype_for::<T>())
}

fn expect_2d_mut<T>(
    name: &str,
    tensor: Tensor2Mut<T>,
    rows: usize,
    cols: usize,
) -> HeaderResult<()> {
    if tensor.shape.rows != rows || tensor.shape.cols != cols {
        return shape_err(format!(
            "{name} expected shape [{rows}, {cols}], got [{}, {}]",
            tensor.shape.rows, tensor.shape.cols
        ));
    }
    expect_dtype(name, tensor.tensor.dtype, dtype_for::<T>())
}

fn expect_3d_ref<T>(
    name: &str,
    tensor: Tensor3Ref<T>,
    outer: usize,
    middle: usize,
    inner: usize,
) -> HeaderResult<()> {
    if tensor.shape.outer != outer || tensor.shape.middle != middle || tensor.shape.inner != inner {
        return shape_err(format!(
            "{name} expected shape [{outer}, {middle}, {inner}], got [{}, {}, {}]",
            tensor.shape.outer, tensor.shape.middle, tensor.shape.inner
        ));
    }
    expect_dtype(name, tensor.tensor.dtype, dtype_for::<T>())
}

fn expect_3d_mut<T>(
    name: &str,
    tensor: Tensor3Mut<T>,
    outer: usize,
    middle: usize,
    inner: usize,
) -> HeaderResult<()> {
    if tensor.shape.outer != outer || tensor.shape.middle != middle || tensor.shape.inner != inner {
        return shape_err(format!(
            "{name} expected shape [{outer}, {middle}, {inner}], got [{}, {}, {}]",
            tensor.shape.outer, tensor.shape.middle, tensor.shape.inner
        ));
    }
    expect_dtype(name, tensor.tensor.dtype, dtype_for::<T>())
}

fn dtype_for<T>() -> DType {
    let name = std::any::type_name::<T>();
    if name == std::any::type_name::<Bf16>() {
        DType::Bf16
    } else if name == std::any::type_name::<F32>() {
        DType::F32
    } else if name == std::any::type_name::<U32>() {
        DType::U32
    } else {
        DType::U8
    }
}

fn expect_dtype(name: &str, got: DType, expected: DType) -> HeaderResult<()> {
    if got != expected {
        return shape_err(format!("{name} expected dtype {expected:?}, got {got:?}"));
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
