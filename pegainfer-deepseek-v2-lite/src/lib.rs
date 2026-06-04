mod attribution;
mod config;
mod device;
mod engine;
mod ep;
mod host_ops;
mod model;
mod nccl_backend;
mod runtime;
mod weights;

use std::path::Path;

use anyhow::Result;
use pegainfer_engine::engine::{EngineHandle, EngineLoadOptions};

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
