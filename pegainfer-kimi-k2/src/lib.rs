//! Text-only Kimi-K2.6 model crate.
//!
//! The current crate stage owns the compile-checked operator API surface and
//! text-only config probing. CUDA/runtime bodies land behind these headers.

use std::path::Path;

use anyhow::Result;
use pegainfer_core::engine::{EngineHandle, EngineLoadOptions};

pub mod attention;
pub mod batch_decode_trace;
pub mod collectives;
pub mod config;
pub mod dense;
mod direct;
pub mod experts;
#[cfg(feature = "kernel-report")]
pub mod kernel_report;
pub mod router;
pub mod tensor;
pub mod tokenizer;
pub mod weights;

pub use config::{KimiK2TextConfig, KimiModelKind, probe_config_json, probe_model};
pub use direct::{KimiK2DirectRuntimeConfig, KimiK2RankPlacement};
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

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    direct::start_engine(model_path, options)
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {
        assert_eq!(crate::config::KIMI_K2_HIDDEN, 7168);
    }
}
