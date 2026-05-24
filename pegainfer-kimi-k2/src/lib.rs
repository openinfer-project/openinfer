//! Text-only Kimi-K2.6 model crate.
//!
//! The current crate stage owns the compile-checked operator API surface and
//! text-only config probing. CUDA/runtime bodies land behind these headers.

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::path::Path;

use anyhow::Result;
use pegainfer_core::engine::{EngineHandle, EngineLoadOptions};

#[cfg(feature = "kimi-k2")]
pub mod batch_decode_trace;
#[cfg(feature = "kimi-k2")]
pub mod collectives;
pub mod config;
#[cfg(feature = "kernel-report")]
pub mod kernel_report;
#[cfg(feature = "kimi-k2")]
pub mod layers;
#[cfg(feature = "kimi-k2")]
mod runner;
pub mod tensor;
pub mod tokenizer;
#[cfg(feature = "kimi-k2")]
mod typed_scratch;
#[cfg(feature = "kimi-k2")]
pub mod weights;

pub use config::{KimiK2TextConfig, KimiModelKind, probe_config_json, probe_model};
#[cfg(feature = "kimi-k2")]
pub use runner::{KimiK2RankPlacement, KimiK2RunnerConfig};
#[cfg(feature = "kimi-k2")]
pub use weights::{
    KIMI_K2_WEIGHT_INDEX, KimiAttentionGpuWeights, KimiDenseMlpGpuWeights,
    KimiInt4ProjectionGpuWeights, KimiK2WeightManifest, KimiLayerGpuWeights,
    KimiLayerKindGpuWeights, KimiMoeLayerGpuWeights, KimiRankGpuContext, KimiRankGpuWeights,
    KimiRankShardPlan, KimiRankSlicedLoadPlan, KimiRankTypedGpuWeights, KimiRankWeightHeaders,
    KimiRankWeightNames, KimiRankWeightPlan, KimiRoutedExpertGpuWeights, KimiRouterGpuWeights,
    KimiShardTensorLoadPlan, KimiSharedExpertGpuWeights, KimiTensorHeader, KimiTensorLoadSlice,
    KimiTensorLoadSpec, KimiTopGpuWeights, load_rank_sliced_weight_headers,
    load_rank_sliced_weights_to_gpu, load_rank_weight_headers, load_rank_weights_to_gpu,
};

#[cfg(feature = "kimi-k2")]
pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    runner::start_engine(model_path, options)
}

#[cfg(not(feature = "kimi-k2"))]
pub fn start_engine(_model_path: &Path, _options: EngineLoadOptions) -> Result<EngineHandle> {
    anyhow::bail!("Kimi-K2 runtime is feature-gated; rebuild with --features kimi-k2")
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {
        assert_eq!(crate::config::KIMI_K2_HIDDEN, 7168);
    }
}
