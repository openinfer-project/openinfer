//! Text-only Kimi-K2.6 weight loading.
//!
//! This file is the `crate::weights` module root. The sibling `weights/`
//! directory contains the implementation modules, following the repository's
//! flat Rust module layout (`foo.rs` + `foo/`, no `mod.rs`).

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    ops::Range,
    path::Path,
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits,
    result as cuda_result,
};
use half::bf16;
use log::debug;
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

const KIMI_K2_WEIGHT_INDEX: &str = "model.safetensors.index.json";
const TEXT_PREFIX: &str = "language_model.";

mod context;
mod load;
mod manifest;
mod package;
#[cfg(test)]
mod tests;

pub(crate) use context::KimiRankGpuContext;
pub(crate) use load::{
    KimiRankSlicedLoadPlan, ensure_text_only_model_index, load_rank_sliced_weights_to_gpu,
};
pub(crate) use manifest::{
    KimiK2WeightManifest, KimiLayerWeightKindNames, KimiLayerWeightNames, KimiRankWeightNames,
};
pub(crate) use package::{
    KimiGpuRawTensor, KimiRankExpertMarlinWeights, KimiRankGpuWeights, KimiRouterDeviceWeights,
    KimiRouterGpuWeights,
};
