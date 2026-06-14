use std::path::PathBuf;

mod backend;
mod generation;
mod helpers;
mod layers;
mod moe;
mod readiness;
mod routing;
#[cfg(test)]
mod tests;
mod types;

use crate::{
    Config,
    model::{DriverRankModel, ExpertRankModel},
};

use backend::EpBackendRuntime;

pub use types::{
    BatchedGenerationResult, DecodeGraphReadinessReport, GenerationResult, GenerationStats,
};

pub struct DeepSeekV2LiteEp2Generator {
    model_path: PathBuf,
    device_ordinals: Vec<usize>,
    config: Config,
    rank0: DriverRankModel,
    rank1: ExpertRankModel,
    backend: EpBackendRuntime,
}

// SAFETY: The generator is driven by exactly one worker thread after load. It
// switches CUDA devices explicitly before every rank-local op and recreates the
// thread-local cuBLAS handle when the active device changes.
unsafe impl Send for DeepSeekV2LiteEp2Generator {}
