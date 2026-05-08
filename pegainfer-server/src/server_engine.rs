use std::fmt;

use anyhow::Result;

pub use pegainfer_core::engine::{FinishReason, TokenLogprob};

// ── Model type detection ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    #[cfg(feature = "deepseek-v4")]
    DeepSeekV4,
    Qwen3,
    Qwen35,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "deepseek-v4")]
            Self::DeepSeekV4 => write!(f, "DeepSeek V4"),
            Self::Qwen3 => write!(f, "Qwen3"),
            Self::Qwen35 => write!(f, "Qwen3.5"),
        }
    }
}

/// Detect model type from config.json.
pub fn detect_model_type(model_path: &str) -> Result<ModelType> {
    let config_path = format!("{}/config.json", model_path);
    let content = std::fs::read_to_string(&config_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    if json
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|model_type| model_type == "deepseek_v4")
    {
        #[cfg(feature = "deepseek-v4")]
        return Ok(ModelType::DeepSeekV4);
        #[cfg(not(feature = "deepseek-v4"))]
        anyhow::bail!(
            "DeepSeek V4 support is feature-gated; rebuild pegainfer-server with --features deepseek-v4"
        );
    }

    if json.get("text_config").is_some() {
        return Ok(ModelType::Qwen35);
    }

    Ok(ModelType::Qwen3)
}
