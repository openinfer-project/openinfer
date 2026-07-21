use std::fmt;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
pub use openinfer_core::engine::FinishReason;
pub use openinfer_core::engine::TokenLogprob;

// ── Model type detection ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    #[cfg(feature = "deepseek-v2-lite")]
    DeepSeekV2Lite,
    #[cfg(feature = "glm52")]
    Glm52,
    #[cfg(feature = "kimi-k2")]
    KimiK2,
    #[cfg(feature = "qwen3")]
    Qwen3,
    #[cfg(feature = "qwen35-4b")]
    Qwen35,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // By-value match so the no-model-features build (empty enum) still
        // type-checks: an empty match is only exhaustive for owned values.
        match *self {
            #[cfg(feature = "deepseek-v2-lite")]
            Self::DeepSeekV2Lite => write!(f, "DeepSeek-V2-Lite"),
            #[cfg(feature = "glm52")]
            Self::Glm52 => write!(f, "GLM5.2"),
            #[cfg(feature = "kimi-k2")]
            Self::KimiK2 => write!(f, "Kimi-K2.6"),
            #[cfg(feature = "qwen3")]
            Self::Qwen3 => write!(f, "Qwen3"),
            #[cfg(feature = "qwen35-4b")]
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
            openinfer_deepseek_v2_lite::probe_config_json(&json)?;
            return Ok(ModelType::DeepSeekV2Lite);
        }
        #[cfg(not(feature = "deepseek-v2-lite"))]
        {
            anyhow::bail!(
                "DeepSeek-V2-Lite support is feature-gated; rebuild openinfer-server with --features deepseek-v2-lite"
            );
        }
    }

    if json
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|model_type| model_type == "glm_moe_dsa")
    {
        #[cfg(feature = "glm52")]
        {
            openinfer_glm52::probe_config_json(&json)?;
            return Ok(ModelType::Glm52);
        }
        #[cfg(not(feature = "glm52"))]
        anyhow::bail!(
            "GLM5.2 support is feature-gated; rebuild openinfer-server with --features glm52"
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
        #[cfg(feature = "kimi-k2")]
        {
            openinfer_kimi_k2::probe_config_json(&json)?;
            return Ok(ModelType::KimiK2);
        }
        #[cfg(not(feature = "kimi-k2"))]
        anyhow::bail!(
            "Kimi-K2 support is feature-gated; rebuild openinfer-server with --features kimi-k2"
        );
    }

    if json.get("text_config").is_some() {
        #[cfg(feature = "qwen35-4b")]
        return Ok(ModelType::Qwen35);
        #[cfg(not(feature = "qwen35-4b"))]
        anyhow::bail!(
            "Qwen3.5 support is feature-gated; rebuild openinfer-server with --features qwen35-4b"
        );
    }

    #[cfg(feature = "qwen3")]
    return Ok(ModelType::Qwen3);
    #[cfg(not(feature = "qwen3"))]
    anyhow::bail!(
        "Qwen3 support is feature-gated; rebuild openinfer-server with --features qwen3-4b"
    );
}
