pub mod kernel_plan;

mod batch_decode;
mod batch_decode_buffers;
mod batch_decode_dag;
pub mod batch_decode_trace;
mod config;
mod executor;
pub mod kernel_bench;
mod lora;
mod prefill;
mod scheduler;
mod unified_forward;
mod weights;

use std::path::Path;

use anyhow::Result;
use pegainfer_core::engine::{EngineHandle, EngineLoadOptions, ModelInfo};

pub use kernel_plan::kernel_plan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3LoraOptions {
    pub max_loras: usize,
    pub max_lora_rank: usize,
}

impl Qwen3LoraOptions {
    pub const DEFAULT_MAX_LORAS: usize = 1;
    pub const DEFAULT_MAX_LORA_RANK: usize = 64;
    pub const SUPPORTED_MAX_LORA_RANKS: [usize; 9] = [1, 8, 16, 32, 64, 128, 256, 320, 512];

    pub fn validate(self) -> Result<Self> {
        anyhow::ensure!(self.max_loras > 0, "max_loras must be >= 1");
        anyhow::ensure!(
            Self::is_supported_max_lora_rank(self.max_lora_rank),
            "max_lora_rank must be one of: {}",
            Self::supported_max_lora_ranks_display()
        );
        Ok(self)
    }

    pub fn is_supported_max_lora_rank(rank: usize) -> bool {
        Self::SUPPORTED_MAX_LORA_RANKS.contains(&rank)
    }

    pub fn supported_max_lora_ranks_display() -> String {
        Self::SUPPORTED_MAX_LORA_RANKS
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl Default for Qwen3LoraOptions {
    fn default() -> Self {
        Self {
            max_loras: Self::DEFAULT_MAX_LORAS,
            max_lora_rank: Self::DEFAULT_MAX_LORA_RANK,
        }
    }
}

/// Low-level Qwen3 execution interface.
///
/// This is the production phase boundary used by the Qwen3 scheduler and by
/// model-local benchmarks. The root server should use `start_engine` instead.
pub mod runtime {
    pub use crate::executor::{
        DecodePlan, DecodeRequestResult, DecodeResult, DecodeStepItem, PrefillPlan,
        PrefillRequestResult, PrefillResult, PrefillStepItem, Qwen3Executor, RequestId,
        UnifiedPlan, UnifiedResult,
    };
}

pub fn probe_model(model_path: &Path) -> Result<Option<ModelInfo>> {
    let config_path = model_path.join("config.json");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let json: serde_json::Value = serde_json::from_str(&content)?;
    if json.get("text_config").is_some() {
        return Ok(None);
    }

    Ok(Some(ModelInfo {
        id: "qwen3-4b",
        display_name: "Qwen3-4B".to_string(),
        model_path: model_path.to_path_buf(),
        max_model_len: json
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok()),
    }))
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    let model_path = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path must be valid UTF-8"))?;
    scheduler::start_qwen3(model_path, enable_cuda_graph, &device_ordinals, seed)
}

pub fn start_engine_with_lora_control(
    model_path: &Path,
    options: EngineLoadOptions,
    lora_options: Qwen3LoraOptions,
) -> Result<EngineHandle> {
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    let model_path = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path must be valid UTF-8"))?;
    scheduler::start_qwen3_with_lora_control(
        model_path,
        enable_cuda_graph,
        &device_ordinals,
        seed,
        lora_options.validate()?,
    )
}
