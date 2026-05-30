//! Text-only Kimi-K2.6 safetensors index manifest.
//!
//! The manifest is intentionally built from `model.safetensors.index.json`
//! instead of tensor headers: this stage decides ownership and required tensor
//! names without touching the 595GB weight payload.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    ops::Range,
    path::Path,
    sync::Arc,
};

use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits,
    result as cuda_result,
};
use half::bf16;
use memmap2::Mmap;
use pegainfer_kernels::ffi;
use pegainfer_kernels::ops::{
    KimiInt4ExpertRole, KimiInt4NibbleOrder, KimiInt4WeightManifest, KimiMarlinFusedW13Int4Weight,
    KimiMarlinInt4ExpertWeights, KimiMarlinInt4Weight, kimi_marlin_int4_fuse_w13,
    kimi_marlin_int4_reorder_scale, kimi_marlin_int4_reorder_weight,
};
use pegainfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec, GpuWeight};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use crate::config::{
    KIMI_K2_DENSE_INTERMEDIATE, KIMI_K2_DENSE_LAYERS, KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN,
    KIMI_K2_INT4_GROUP_SIZE, KIMI_K2_LAYERS, KIMI_K2_MOE_LAYERS, KIMI_K2_Q_HEAD_DIM,
    KIMI_K2_QK_NOPE_HEAD_DIM, KIMI_K2_ROUTED_EXPERTS, KIMI_K2_V_HEAD_DIM, KimiK2ParallelShape,
};

pub(crate) const KIMI_K2_WEIGHT_INDEX: &str = "model.safetensors.index.json";
const TEXT_PREFIX: &str = "language_model.";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiTensorEntry {
    pub name: String,
    pub shard: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiAttentionManifest {
    pub input_layernorm: KimiTensorEntry,
    pub q_a_proj: KimiTensorEntry,
    pub q_a_layernorm: KimiTensorEntry,
    pub q_b_proj: KimiTensorEntry,
    pub kv_a_proj_with_mqa: KimiTensorEntry,
    pub kv_a_layernorm: KimiTensorEntry,
    pub kv_b_proj: KimiTensorEntry,
    pub o_proj: KimiTensorEntry,
    pub post_attention_layernorm: KimiTensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiDenseMlpManifest {
    pub gate_proj: KimiTensorEntry,
    pub up_proj: KimiTensorEntry,
    pub down_proj: KimiTensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRouterManifest {
    pub gate_weight: KimiTensorEntry,
    pub e_score_correction_bias: KimiTensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiSharedExpertManifest {
    pub gate_proj: KimiTensorEntry,
    pub up_proj: KimiTensorEntry,
    pub down_proj: KimiTensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiInt4ProjectionManifest {
    pub weight_packed: KimiTensorEntry,
    pub weight_scale: KimiTensorEntry,
    pub weight_shape: KimiTensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRoutedExpertManifest {
    pub expert_idx: usize,
    pub gate_proj: KimiInt4ProjectionManifest,
    pub up_proj: KimiInt4ProjectionManifest,
    pub down_proj: KimiInt4ProjectionManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiMoeLayerManifest {
    pub router: KimiRouterManifest,
    pub shared_experts: KimiSharedExpertManifest,
    pub routed_experts: Vec<KimiRoutedExpertManifest>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum KimiLayerKindManifest {
    Dense(KimiDenseMlpManifest),
    Moe(KimiMoeLayerManifest),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiLayerManifest {
    pub layer_idx: usize,
    pub attention: KimiAttentionManifest,
    pub kind: KimiLayerKindManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRankWeightPlan {
    pub rank: usize,
    pub tp_rank: usize,
    pub ep_rank: usize,
    pub attention_head_range: Range<usize>,
    pub vocab_range: Range<usize>,
    pub local_expert_range: Range<usize>,
    pub replicated_router: bool,
    pub tensor_count: usize,
    pub shard_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum KimiTensorLoadSlice {
    Full,
    RowRange { start: usize, end: usize },
    ColRange { start: usize, end: usize },
}

impl KimiTensorLoadSlice {
    fn local_shape(&self, full_shape: &[usize]) -> Result<Vec<usize>> {
        match *self {
            Self::Full => Ok(full_shape.to_vec()),
            Self::RowRange { start, end } => {
                ensure!(
                    full_shape.len() == 2 && start <= end && end <= full_shape[0],
                    "Kimi row slice [{start}..{end}) is invalid for shape {:?}",
                    full_shape
                );
                Ok(vec![end - start, full_shape[1]])
            }
            Self::ColRange { start, end } => {
                ensure!(
                    full_shape.len() == 2 && start <= end && end <= full_shape[1],
                    "Kimi col slice [{start}..{end}) is invalid for shape {:?}",
                    full_shape
                );
                Ok(vec![full_shape[0], end - start])
            }
        }
    }

    fn local_bytes(&self, full_shape: &[usize], dtype: Dtype) -> Result<usize> {
        Ok(self.local_shape(full_shape)?.iter().product::<usize>() * dtype_element_bytes(dtype)?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiTensorLoadSpec {
    pub name: String,
    pub shard: String,
    pub slice: KimiTensorLoadSlice,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiShardTensorLoadPlan {
    pub shard: String,
    pub tensors: Vec<KimiTensorLoadSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRankSlicedLoadPlan {
    pub rank: usize,
    pub shards: Vec<KimiShardTensorLoadPlan>,
    pub tensor_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiTopWeightNames {
    pub token_embedding: String,
    pub final_norm: String,
    pub lm_head: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiAttentionWeightNames {
    pub input_layernorm: String,
    pub q_a_proj: String,
    pub q_a_layernorm: String,
    pub q_b_proj: String,
    pub kv_a_proj_with_mqa: String,
    pub kv_a_layernorm: String,
    pub kv_b_proj: String,
    pub o_proj: String,
    pub post_attention_layernorm: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiDenseMlpWeightNames {
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRouterWeightNames {
    pub gate_weight: String,
    pub e_score_correction_bias: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiSharedExpertWeightNames {
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiInt4ProjectionWeightNames {
    pub weight_packed: String,
    pub weight_scale: String,
    pub weight_shape: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRoutedExpertWeightNames {
    pub global_expert: usize,
    pub gate_proj: KimiInt4ProjectionWeightNames,
    pub up_proj: KimiInt4ProjectionWeightNames,
    pub down_proj: KimiInt4ProjectionWeightNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiMoeLayerWeightNames {
    pub router: KimiRouterWeightNames,
    pub shared_experts: KimiSharedExpertWeightNames,
    pub routed_experts: Vec<KimiRoutedExpertWeightNames>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum KimiLayerWeightKindNames {
    Dense(KimiDenseMlpWeightNames),
    Moe(KimiMoeLayerWeightNames),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiLayerWeightNames {
    pub layer_idx: usize,
    pub attention: KimiAttentionWeightNames,
    pub kind: KimiLayerWeightKindNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRankWeightNames {
    pub rank: usize,
    pub plan: KimiRankWeightPlan,
    pub top: KimiTopWeightNames,
    pub layers: Vec<KimiLayerWeightNames>,
}

pub(crate) struct KimiGpuRawTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub data: CudaSlice<u8>,
}

pub(crate) struct KimiRankGpuWeights {
    pub rank: usize,
    pub tensors: BTreeMap<String, KimiGpuRawTensor>,
    pub total_bytes: usize,
}

pub(crate) struct KimiRouterGpuWeights<'a> {
    pub gate_weight: &'a KimiGpuRawTensor,
    pub e_score_correction_bias: &'a KimiGpuRawTensor,
}

pub(crate) struct KimiRouterDeviceWeights {
    pub gate_weight: GpuWeight<KIMI_K2_ROUTED_EXPERTS, KIMI_K2_HIDDEN>,
    pub e_score_correction_bias: CudaSlice<f32>,
}

pub(crate) struct KimiInt4ProjectionGpuWeights<'a> {
    pub weight_packed: &'a KimiGpuRawTensor,
    pub weight_scale: &'a KimiGpuRawTensor,
    pub weight_shape: &'a KimiGpuRawTensor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KimiInt4ProjectionRole {
    Gate,
    Up,
    Down,
}

impl KimiInt4ProjectionRole {
    const fn dims(self) -> (usize, usize) {
        match self {
            Self::Gate | Self::Up => (KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN),
            Self::Down => (KIMI_K2_HIDDEN, KIMI_K2_EXPERT_INTERMEDIATE),
        }
    }

    const fn kernel_role(self) -> KimiInt4ExpertRole {
        match self {
            Self::Gate => KimiInt4ExpertRole::W1Gate,
            Self::Up => KimiInt4ExpertRole::W3Up,
            Self::Down => KimiInt4ExpertRole::W2Down,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiExpertMajorProjectionPlan {
    pub role: KimiInt4ProjectionRole,
    pub local_experts: usize,
    pub out_dim: usize,
    pub in_dim: usize,
    pub packed_i32_shape_per_expert: [usize; 2],
    pub scale_bf16_shape_per_expert: [usize; 2],
    pub packed_bytes: usize,
    pub scale_bytes: usize,
    pub shape_bytes: usize,
}

pub(crate) struct KimiExpertMajorProjectionMarlinBuffers {
    pub role: KimiInt4ProjectionRole,
    pub plan: KimiExpertMajorProjectionPlan,
    pub manifest: KimiInt4WeightManifest,
    pub weight_packed_marlin_uint4b8: CudaSlice<u8>,
    pub weight_scale_marlin_permuted: CudaSlice<bf16>,
}

impl KimiExpertMajorProjectionMarlinBuffers {
    pub(crate) fn as_marlin_weight(&self) -> KimiMarlinInt4Weight<'_> {
        KimiMarlinInt4Weight {
            manifest: self.manifest,
            weight_packed_uint4b8: &self.weight_packed_marlin_uint4b8,
            weight_scale_permuted: &self.weight_scale_marlin_permuted,
        }
    }

    fn package_bytes(&self) -> usize {
        self.weight_packed_marlin_uint4b8.len()
            + self.weight_scale_marlin_permuted.len() * std::mem::size_of::<bf16>()
    }
}

pub(crate) struct KimiExpertMajorW13MarlinBuffers {
    pub local_experts: usize,
    pub in_dim: usize,
    pub intermediate_dim: usize,
    pub group_size: usize,
    pub weight_packed_marlin_uint4b8: CudaSlice<u8>,
    pub weight_scale_marlin_permuted: CudaSlice<bf16>,
}

impl KimiExpertMajorW13MarlinBuffers {
    pub(crate) fn as_marlin_weight(&self) -> KimiMarlinFusedW13Int4Weight<'_> {
        KimiMarlinFusedW13Int4Weight {
            local_experts: self.local_experts,
            in_dim: self.in_dim,
            intermediate_dim: self.intermediate_dim,
            group_size: self.group_size,
            weight_packed_uint4b8: &self.weight_packed_marlin_uint4b8,
            weight_scale_permuted: &self.weight_scale_marlin_permuted,
        }
    }

    fn package_bytes(&self) -> usize {
        self.weight_packed_marlin_uint4b8.len()
            + self.weight_scale_marlin_permuted.len() * std::mem::size_of::<bf16>()
    }
}

pub(crate) struct KimiMoeLayerExpertMarlinWeights {
    pub layer_idx: usize,
    pub w13: KimiExpertMajorW13MarlinBuffers,
    pub down: KimiExpertMajorProjectionMarlinBuffers,
    pub total_bytes: usize,
}

pub(crate) struct KimiRankExpertMarlinWeights {
    pub rank: usize,
    pub local_expert_range: Range<usize>,
    pub layers: Vec<KimiMoeLayerExpertMarlinWeights>,
    pub total_bytes: usize,
}

impl KimiMoeLayerExpertMarlinWeights {
    pub(crate) fn as_marlin_weights(&self) -> KimiMarlinInt4ExpertWeights<'_> {
        KimiMarlinInt4ExpertWeights {
            w13: self.w13.as_marlin_weight(),
            w2_down: self.down.as_marlin_weight(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct KimiRankGpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub device_ordinal: usize,
}

// SAFETY: each Kimi rank owns one CUDA context/stream pair and the runner
// drives that pair from the rank worker thread.
unsafe impl Send for KimiRankGpuContext {}
unsafe impl Sync for KimiRankGpuContext {}

fn dtype_element_bytes(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::BF16 => Ok(2),
        Dtype::F32 | Dtype::I32 => Ok(4),
        Dtype::U8 => Ok(1),
        other => bail!("Kimi loader does not support dtype {:?}", other),
    }
}

mod context;
mod load;
mod manifest;
mod package;
#[cfg(test)]
use load::sliced_tensor_bytes;
use manifest::validate_rank_tensor_catalog;
#[cfg(test)]
mod tests;

pub(crate) use load::{ensure_text_only_model_index, load_rank_sliced_weights_to_gpu};
pub(crate) use manifest::KimiK2WeightManifest;
