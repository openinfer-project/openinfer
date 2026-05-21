use std::{fmt, path::Path};

use anyhow::{Context, Result};

pub use pegainfer_core::engine::{FinishReason, TokenLogprob};

// ── Model type detection ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    #[cfg(feature = "deepseek-v2-lite")]
    DeepSeekV2Lite,
    #[cfg(feature = "deepseek-v4")]
    DeepSeekV4,
    KimiK2,
    Qwen3,
    Qwen35,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "deepseek-v2-lite")]
            Self::DeepSeekV2Lite => write!(f, "DeepSeek-V2-Lite"),
            #[cfg(feature = "deepseek-v4")]
            Self::DeepSeekV4 => write!(f, "DeepSeek V4"),
            Self::KimiK2 => write!(f, "Kimi-K2.6"),
            Self::Qwen3 => write!(f, "Qwen3"),
            Self::Qwen35 => write!(f, "Qwen3.5"),
        }
    }
}

/// Detect model type from config.json.
pub fn detect_model_type(model_path: impl AsRef<Path>) -> Result<ModelType> {
    let config_path = model_path.as_ref().join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;

    if json
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|model_type| model_type == "deepseek_v2")
    {
        #[cfg(feature = "deepseek-v2-lite")]
        {
            pegainfer_deepseek_v2_lite::probe_config_json(&json)?;
            return Ok(ModelType::DeepSeekV2Lite);
        }
        #[cfg(not(feature = "deepseek-v2-lite"))]
        {
            anyhow::bail!(
                "DeepSeek-V2-Lite support is feature-gated; rebuild pegainfer-server with --features deepseek-v2-lite"
            );
        }
    }

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

    if json
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|model_type| model_type == "kimi_k25" || model_type == "kimi_k2")
        || json
            .get("text_config")
            .and_then(|text| text.get("model_type"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|model_type| model_type == "kimi_k2")
    {
        pegainfer_kimi_k2::probe_config_json(&json)?;
        return Ok(ModelType::KimiK2);
    }

    if json.get("text_config").is_some() {
        return Ok(ModelType::Qwen35);
    }

    Ok(ModelType::Qwen3)
}
