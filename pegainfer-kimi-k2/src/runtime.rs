//! Text-only Kimi-K2.6 batch decode runtime header.
//!
//! This module describes the orchestration surface for one-token-per-request
//! decode. It validates the batch plan and names the buffers/operators the real
//! runtime will wire later; it intentionally does not launch CUDA work.

use crate::{
    config::{
        KIMI_K2_DENSE_INTERMEDIATE, KIMI_K2_DENSE_LAYERS, KIMI_K2_EXPERT_INTERMEDIATE,
        KIMI_K2_HEADS, KIMI_K2_HIDDEN, KIMI_K2_INT4_GROUP_SIZE, KIMI_K2_KV_A_OUT, KIMI_K2_KV_B_OUT,
        KIMI_K2_KV_LORA_RANK, KIMI_K2_LAYERS, KIMI_K2_MAX_CONTEXT, KIMI_K2_MOE_LAYERS,
        KIMI_K2_O_PROJ_IN, KIMI_K2_Q_LORA_RANK, KIMI_K2_Q_PROJ_OUT, KIMI_K2_QK_ROPE_HEAD_DIM,
        KIMI_K2_ROUTED_EXPERTS, KIMI_K2_TOPK, KIMI_K2_VOCAB, KimiK2ParallelShape,
    },
    tensor::{
        Bf16, DType, DevicePtr, F32, HeaderError, HeaderResult, Layout, Shape2, Shape3,
        StreamHandle, TensorMut, TensorRef, TokenBatch, U8, U32, VocabShard,
    },
};

pub const KIMI_K2_DECODE_BATCH_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiK2DecodeGraphContract {
    pub device_resident_metadata: bool,
    pub no_device_to_host: bool,
    pub no_host_sync: bool,
    pub no_step_allocation: bool,
    pub stable_pointers: bool,
    pub preallocated_scratch: bool,
    pub ep_outside_graph: bool,
}

impl KimiK2DecodeGraphContract {
    #[must_use]
    pub const fn graph_ready() -> Self {
        Self {
            device_resident_metadata: true,
            no_device_to_host: true,
            no_host_sync: true,
            no_step_allocation: true,
            stable_pointers: true,
            preallocated_scratch: true,
            ep_outside_graph: true,
        }
    }

    pub fn validate(self) -> HeaderResult<()> {
        if !self.device_resident_metadata {
            return Err(shape_error(
                "Kimi decode metadata must stay device resident for CUDA Graph capture",
            ));
        }
        if !self.no_device_to_host {
            return Err(shape_error(
                "Kimi decode hot path must not perform device-to-host transfers",
            ));
        }
        if !self.no_host_sync {
            return Err(shape_error(
                "Kimi decode hot path must not synchronize with host",
            ));
        }
        if !self.no_step_allocation {
            return Err(shape_error(
                "Kimi decode hot path must not allocate during a decode step",
            ));
        }
        if !self.stable_pointers {
            return Err(shape_error(
                "Kimi decode buffers must keep stable pointers for CUDA Graph replay",
            ));
        }
        if !self.preallocated_scratch {
            return Err(shape_error("Kimi decode scratch must be preallocated"));
        }
        if !self.ep_outside_graph {
            return Err(shape_error(
                "Kimi PPLX EP dispatch/combine must stay outside CUDA Graph capture",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    Dense,
    Moe,
}

impl LayerKind {
    #[must_use]
    pub const fn for_layer(layer_idx: usize) -> Self {
        if layer_idx < KIMI_K2_DENSE_LAYERS {
            Self::Dense
        } else {
            Self::Moe
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedTensor<T> {
    pub name: &'static str,
    pub tensor: TensorRef<T>,
    pub shape: Shape2,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormWeights {
    pub weight: NamedTensor<Bf16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlaAttentionWeights {
    pub q_a_proj: NamedTensor<Bf16>,
    pub q_a_layernorm: NormWeights,
    pub q_b_proj: NamedTensor<Bf16>,
    pub kv_a_proj_with_mqa: NamedTensor<Bf16>,
    pub kv_a_layernorm: NormWeights,
    pub kv_b_proj: NamedTensor<Bf16>,
    pub o_proj: NamedTensor<Bf16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseMlpWeights {
    pub gate_proj: NamedTensor<Bf16>,
    pub up_proj: NamedTensor<Bf16>,
    pub down_proj: NamedTensor<Bf16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedExpertWeights {
    pub gate_proj: NamedTensor<Bf16>,
    pub up_proj: NamedTensor<Bf16>,
    pub down_proj: NamedTensor<Bf16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterWeights {
    pub gate: NamedTensor<Bf16>,
    pub e_score_correction_bias: NamedTensor<F32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Int4GroupedWeight {
    pub weight_packed: TensorRef<U8>,
    pub weight_scale: TensorRef<Bf16>,
    pub logical_shape: Shape2,
    pub group_size: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutedExpertWeights {
    pub gate_proj: Int4GroupedWeight,
    pub up_proj: Int4GroupedWeight,
    pub down_proj: Int4GroupedWeight,
    pub expert_range: std::ops::Range<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeLayerWeights {
    pub router: RouterWeights,
    pub shared_expert: SharedExpertWeights,
    pub routed_experts: RoutedExpertWeights,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerWeights {
    pub layer_idx: usize,
    pub kind: LayerKind,
    pub input_layernorm: NormWeights,
    pub self_attn: MlaAttentionWeights,
    pub post_attention_layernorm: NormWeights,
    pub dense_mlp: Option<DenseMlpWeights>,
    pub moe: Option<MoeLayerWeights>,
}

impl LayerWeights {
    pub fn validate_header(&self) -> HeaderResult<()> {
        if self.layer_idx >= KIMI_K2_LAYERS {
            return Err(shape_error(format!(
                "layer_idx {} outside 0..{}",
                self.layer_idx, KIMI_K2_LAYERS
            )));
        }
        if self.kind != LayerKind::for_layer(self.layer_idx) {
            return Err(shape_error(format!(
                "layer {} kind {:?} does not match Kimi-K2.6 layout",
                self.layer_idx, self.kind
            )));
        }
        match self.kind {
            LayerKind::Dense => {
                if self.dense_mlp.is_none() || self.moe.is_some() {
                    return Err(shape_error("dense layer must carry dense_mlp only"));
                }
            }
            LayerKind::Moe => {
                if self.moe.is_none() || self.dense_mlp.is_some() {
                    return Err(shape_error("MoE layer must carry moe only"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiK2ModelWeights {
    pub token_embedding: NamedTensor<Bf16>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: NormWeights,
    pub lm_head: NamedTensor<Bf16>,
    pub vocab_shard: VocabShard,
}

impl KimiK2ModelWeights {
    pub fn validate_header(&self) -> HeaderResult<()> {
        if self.layers.len() != KIMI_K2_LAYERS {
            return Err(shape_error(format!(
                "expected {} layers, got {}",
                KIMI_K2_LAYERS,
                self.layers.len()
            )));
        }
        for (expected_idx, layer) in self.layers.iter().enumerate() {
            if layer.layer_idx != expected_idx {
                return Err(shape_error(format!(
                    "layer header at slot {expected_idx} has layer_idx {}",
                    layer.layer_idx
                )));
            }
            layer.validate_header()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiK2DecodeRow {
    pub request_id: u64,
    pub token_id: u32,
    pub seq_len_before: usize,
    pub position: u32,
    pub cache_slot: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiK2KvStateHeader {
    pub request_id: u64,
    pub seq_len: usize,
    pub max_len: usize,
    pub cache_slot: usize,
    pub page_table: TensorMut<U32>,
}

impl KimiK2KvStateHeader {
    pub fn plan_decode_row(&self, token_id: u32) -> HeaderResult<KimiK2DecodeRow> {
        if self.seq_len >= self.max_len || self.seq_len >= KIMI_K2_MAX_CONTEXT {
            return Err(shape_error(format!(
                "request {} cannot decode past seq_len={} max_len={} model_context={}",
                self.request_id, self.seq_len, self.max_len, KIMI_K2_MAX_CONTEXT
            )));
        }
        Ok(KimiK2DecodeRow {
            request_id: self.request_id,
            token_id,
            seq_len_before: self.seq_len,
            position: self.seq_len as u32,
            cache_slot: self.cache_slot,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlaDecodeCache {
    pub compressed_kv: TensorMut<Bf16>,
    pub k_rope: TensorMut<Bf16>,
    pub layout: Layout,
    pub compressed_shape: Shape3,
    pub k_rope_shape: Shape3,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiK2DecodeBuffers {
    pub batch: TokenBatch,
    pub token_ids: TensorMut<U32>,
    pub positions: TensorMut<U32>,
    pub hidden: TensorMut<Bf16>,
    pub normed: TensorMut<Bf16>,
    pub residual: TensorMut<Bf16>,
    pub q_a: TensorMut<Bf16>,
    pub q: TensorMut<Bf16>,
    pub compressed_kv_step: TensorMut<Bf16>,
    pub k_rope_step: TensorMut<Bf16>,
    pub kv_b: TensorMut<Bf16>,
    pub attn_out: TensorMut<Bf16>,
    pub attn_proj: TensorMut<Bf16>,
    pub dense_gate: TensorMut<Bf16>,
    pub dense_up: TensorMut<Bf16>,
    pub mlp_out: TensorMut<Bf16>,
    pub router_logits: TensorMut<F32>,
    pub router_scores: TensorMut<F32>,
    pub topk_ids: TensorMut<U32>,
    pub topk_weights: TensorMut<F32>,
    pub expert_major_hidden: TensorMut<Bf16>,
    pub expert_indptr: TensorMut<U32>,
    pub logits: TensorMut<Bf16>,
    pub mla_cache: MlaDecodeCache,
}

impl KimiK2DecodeBuffers {
    pub fn validate_for_plan(&self, batch: TokenBatch) -> HeaderResult<()> {
        if batch.batch_size == 0 || batch.active_tokens == 0 {
            return Err(shape_error("decode batch must contain at least one row"));
        }
        if batch.batch_size > batch.padded_tokens {
            return Err(shape_error(format!(
                "batch_size {} exceeds padded_tokens {}",
                batch.batch_size, batch.padded_tokens
            )));
        }
        if self.batch.padded_tokens < batch.padded_tokens {
            return Err(shape_error(format!(
                "decode buffers padded capacity {} is smaller than requested {}",
                self.batch.padded_tokens, batch.padded_tokens
            )));
        }
        expect_dtype("token_ids", self.token_ids.dtype, DType::U32)?;
        expect_dtype("positions", self.positions.dtype, DType::U32)?;
        expect_dtype("hidden", self.hidden.dtype, DType::Bf16)?;
        expect_dtype("logits", self.logits.dtype, DType::Bf16)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeAttentionPath {
    ExpandedCorrectness,
    CompressedMla,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiK2BatchDecodePlan {
    pub stream: StreamHandle,
    pub batch: TokenBatch,
    pub rows: Vec<KimiK2DecodeRow>,
    pub attention_path: DecodeAttentionPath,
    pub graph_contract: KimiK2DecodeGraphContract,
    pub parallel: KimiK2ParallelShape,
    pub layer_kinds: Vec<LayerKind>,
}

impl KimiK2BatchDecodePlan {
    pub fn new(
        stream: StreamHandle,
        token_ids: &[u32],
        kv_states: &[KimiK2KvStateHeader],
        bufs: &KimiK2DecodeBuffers,
        parallel: KimiK2ParallelShape,
        attention_path: DecodeAttentionPath,
        graph_contract: KimiK2DecodeGraphContract,
    ) -> HeaderResult<Self> {
        graph_contract.validate()?;
        if token_ids.len() != kv_states.len() {
            return Err(shape_error(format!(
                "token_ids length {} does not match kv_states length {}",
                token_ids.len(),
                kv_states.len()
            )));
        }
        let batch_size = token_ids.len();
        let padded_tokens = decode_bucket_for(batch_size)?;
        let batch = TokenBatch {
            batch_size,
            active_tokens: batch_size,
            padded_tokens,
        };
        bufs.validate_for_plan(batch)?;

        let rows = token_ids
            .iter()
            .zip(kv_states)
            .map(|(&token_id, kv)| kv.plan_decode_row(token_id))
            .collect::<HeaderResult<Vec<_>>>()?;
        let layer_kinds = (0..KIMI_K2_LAYERS).map(LayerKind::for_layer).collect();

        Ok(Self {
            stream,
            batch,
            rows,
            attention_path,
            graph_contract,
            parallel,
            layer_kinds,
        })
    }

    #[must_use]
    pub fn supports_multi_batch(&self) -> bool {
        self.batch.batch_size > 1
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiK2RuntimeHeader {
    pub stream: StreamHandle,
    pub parallel: KimiK2ParallelShape,
    pub enable_cuda_graph: bool,
    pub default_attention_path: DecodeAttentionPath,
    pub graph_contract: KimiK2DecodeGraphContract,
}

impl KimiK2RuntimeHeader {
    #[must_use]
    pub fn tp8_ep8(stream: StreamHandle) -> Self {
        Self {
            stream,
            parallel: KimiK2ParallelShape::tp8_ep8(),
            enable_cuda_graph: true,
            default_attention_path: DecodeAttentionPath::CompressedMla,
            graph_contract: KimiK2DecodeGraphContract::graph_ready(),
        }
    }

    pub fn plan_batch_decode(
        &self,
        token_ids: &[u32],
        kv_states: &[KimiK2KvStateHeader],
        bufs: &KimiK2DecodeBuffers,
    ) -> HeaderResult<KimiK2BatchDecodePlan> {
        KimiK2BatchDecodePlan::new(
            self.stream,
            token_ids,
            kv_states,
            bufs,
            self.parallel,
            self.default_attention_path,
            self.graph_contract,
        )
    }

    pub fn batch_decode(
        &self,
        token_ids: &[u32],
        kv_states: &[KimiK2KvStateHeader],
        bufs: &mut KimiK2DecodeBuffers,
        _weights: &KimiK2ModelWeights,
    ) -> HeaderResult<()> {
        let _plan = self.plan_batch_decode(token_ids, kv_states, bufs)?;
        Err(HeaderError::Unsupported {
            message: "Kimi-K2.6 batch decode execution body is intentionally not implemented in this header crate".to_string(),
        })
    }
}

pub fn decode_bucket_for(batch_size: usize) -> HeaderResult<usize> {
    KIMI_K2_DECODE_BATCH_BUCKETS
        .iter()
        .copied()
        .find(|&bucket| batch_size <= bucket)
        .ok_or_else(|| {
            shape_error(format!(
                "batch_size {batch_size} exceeds max decode bucket {}",
                KIMI_K2_DECODE_BATCH_BUCKETS.last().copied().unwrap_or(0)
            ))
        })
}

#[must_use]
pub const fn dense_layer_count() -> usize {
    KIMI_K2_DENSE_LAYERS
}

#[must_use]
pub const fn moe_layer_count() -> usize {
    KIMI_K2_MOE_LAYERS
}

#[must_use]
pub const fn default_dense_mlp_shapes() -> (Shape2, Shape2, Shape2) {
    (
        Shape2 {
            rows: KIMI_K2_DENSE_INTERMEDIATE,
            cols: KIMI_K2_HIDDEN,
        },
        Shape2 {
            rows: KIMI_K2_DENSE_INTERMEDIATE,
            cols: KIMI_K2_HIDDEN,
        },
        Shape2 {
            rows: KIMI_K2_HIDDEN,
            cols: KIMI_K2_DENSE_INTERMEDIATE,
        },
    )
}

#[must_use]
pub const fn default_mla_shapes() -> (Shape2, Shape2, Shape2, Shape2) {
    (
        Shape2 {
            rows: KIMI_K2_Q_LORA_RANK,
            cols: KIMI_K2_HIDDEN,
        },
        Shape2 {
            rows: KIMI_K2_Q_PROJ_OUT,
            cols: KIMI_K2_Q_LORA_RANK,
        },
        Shape2 {
            rows: KIMI_K2_KV_A_OUT,
            cols: KIMI_K2_HIDDEN,
        },
        Shape2 {
            rows: KIMI_K2_KV_B_OUT,
            cols: KIMI_K2_KV_LORA_RANK,
        },
    )
}

#[must_use]
pub const fn default_decode_tensor_shapes(batch: usize) -> (Shape2, Shape3, Shape2) {
    (
        Shape2 {
            rows: batch,
            cols: KIMI_K2_HIDDEN,
        },
        Shape3 {
            outer: batch,
            middle: KIMI_K2_HEADS,
            inner: KIMI_K2_QK_ROPE_HEAD_DIM,
        },
        Shape2 {
            rows: batch,
            cols: KIMI_K2_VOCAB,
        },
    )
}

#[must_use]
pub const fn default_routed_expert_shape() -> Shape2 {
    Shape2 {
        rows: KIMI_K2_EXPERT_INTERMEDIATE,
        cols: KIMI_K2_HIDDEN,
    }
}

#[must_use]
pub const fn default_o_proj_shape() -> Shape2 {
    Shape2 {
        rows: KIMI_K2_HIDDEN,
        cols: KIMI_K2_O_PROJ_IN,
    }
}

#[must_use]
pub const fn default_int4_group_size() -> usize {
    KIMI_K2_INT4_GROUP_SIZE
}

#[must_use]
pub const fn default_topk() -> usize {
    KIMI_K2_TOPK
}

#[must_use]
pub const fn routed_experts() -> usize {
    KIMI_K2_ROUTED_EXPERTS
}

#[must_use]
pub const fn null_tensor_ref<T>(dtype: DType, layout: Layout) -> TensorRef<T> {
    TensorRef {
        ptr: DevicePtr::new(0, 0),
        dtype,
        layout,
    }
}

#[must_use]
pub const fn null_tensor_mut<T>(dtype: DType, layout: Layout) -> TensorMut<T> {
    TensorMut {
        ptr: DevicePtr::new(0, 0),
        dtype,
        layout,
    }
}

fn expect_dtype(name: &str, actual: DType, expected: DType) -> HeaderResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(shape_error(format!(
            "{name} expected dtype {expected:?}, got {actual:?}"
        )))
    }
}

fn shape_error(message: impl Into<String>) -> HeaderError {
    HeaderError::Shape {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_contract_requires_no_d2h() {
        let mut contract = KimiK2DecodeGraphContract::graph_ready();
        contract.no_device_to_host = false;
        let err = contract.validate().unwrap_err();
        assert!(err.to_string().contains("device-to-host"));
    }

    #[test]
    fn graph_contract_requires_no_step_allocation() {
        let mut contract = KimiK2DecodeGraphContract::graph_ready();
        contract.no_step_allocation = false;
        let err = contract.validate().unwrap_err();
        assert!(err.to_string().contains("allocate during a decode step"));
    }

    #[test]
    fn graph_contract_requires_ep_outside_graph() {
        let mut contract = KimiK2DecodeGraphContract::graph_ready();
        contract.ep_outside_graph = false;
        let err = contract.validate().unwrap_err();
        assert!(err.to_string().contains("outside CUDA Graph"));
    }
}
