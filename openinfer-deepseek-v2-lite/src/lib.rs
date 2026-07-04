mod attribution;
mod config;
mod device;
mod engine;
mod ep;
mod host_ops;
mod model;
mod nccl_backend;
mod runtime;
mod scheduler;
mod weights;

use std::path::Path;

use anyhow::Result;
use openinfer_engine::engine::EpBackend;
use openinfer_engine::engine::{EngineHandle, EngineLoadOptions};

pub use attribution::{CallSiteRollup, DecodeAttributionProfile, SectionRollup, SectionSample};
pub use config::Config;
use config::SUPPORTED_HIDDEN_SIZE;
use ep::SUPPORTED_ROUTED_EXPERTS;
pub use runtime::{
    BatchedGenerationResult, DecodeGraphReadinessReport, DeepSeekV2LiteEp2Generator,
    GenerationResult, GenerationStats,
};

pub fn probe_config_json(json: &serde_json::Value) -> Result<bool> {
    let Some(model_type) = json.get("model_type").and_then(serde_json::Value::as_str) else {
        return Ok(false);
    };
    if model_type != "deepseek_v2" {
        return Ok(false);
    }
    let n_routed_experts = json
        .get("n_routed_experts")
        .and_then(serde_json::Value::as_u64);
    let hidden_size = json.get("hidden_size").and_then(serde_json::Value::as_u64);
    let is_lite = n_routed_experts.is_some_and(|value| value == SUPPORTED_ROUTED_EXPERTS as u64)
        && hidden_size.is_some_and(|value| value == SUPPORTED_HIDDEN_SIZE as u64);
    if !is_lite {
        anyhow::bail!(
            "unsupported DeepSeek-V2 config: DeepSeek-V2-Lite first gate expects hidden_size={} and n_routed_experts={}, got hidden_size={:?}, n_routed_experts={:?}",
            SUPPORTED_HIDDEN_SIZE,
            SUPPORTED_ROUTED_EXPERTS,
            hidden_size,
            n_routed_experts
        );
    }
    Ok(true)
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    engine::start_engine(model_path, options)
}

/// Start the DeepSeek-V2-Lite engine for the server. The binary forwards the
/// user's `cuda_graph` request uniformly; whether to honor it is the model's
/// call. The server EP=2 path does not enable CUDA Graph capture, so it ignores
/// the request (warning if one came in). The diagnostic decode graph probe lives
/// in the attribution gate. The EP=2 topology (devices `0..1`) is fixed by the
/// model.
pub fn launch(model_path: &Path, cuda_graph: bool) -> Result<EngineHandle> {
    if cuda_graph {
        log::warn!("DeepSeek V2 Lite does not support CUDA Graph; ignoring --cuda-graph=true");
    }
    engine::start_engine(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            enable_prefill_profile: false,
            device_ordinals: vec![0, 1],
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        },
    )
}
