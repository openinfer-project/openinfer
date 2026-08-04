use std::fmt;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
pub use pegainfer_core::engine::FinishReason;
pub use pegainfer_core::engine::TokenLogprob;

// ── Model type detection ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    #[cfg(feature = "deepseek-v2-lite")]
    DeepSeekV2Lite,
    #[cfg(feature = "gemma4")]
    Gemma4,
    #[cfg(feature = "glm52")]
    Glm52,
    #[cfg(feature = "kimi-k2")]
    KimiK2,
    #[cfg(feature = "qwen3")]
    Qwen3,
    #[cfg(feature = "qwen35")]
    Qwen35,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // By-value match so the no-model-features build (empty enum) still
        // type-checks: an empty match is only exhaustive for owned values.
        match *self {
            #[cfg(feature = "deepseek-v2-lite")]
            Self::DeepSeekV2Lite => write!(f, "DeepSeek-V2-Lite"),
            #[cfg(feature = "gemma4")]
            Self::Gemma4 => write!(f, "Gemma 4"),
            #[cfg(feature = "glm52")]
            Self::Glm52 => write!(f, "GLM5.2"),
            #[cfg(feature = "kimi-k2")]
            Self::KimiK2 => write!(f, "Kimi-K2.6"),
            #[cfg(feature = "qwen3")]
            Self::Qwen3 => write!(f, "Qwen3"),
            #[cfg(feature = "qwen35")]
            Self::Qwen35 => write!(f, "Qwen3.5"),
        }
    }
}

fn config_model_type(json: &serde_json::Value) -> Option<&str> {
    json.get("model_type").and_then(serde_json::Value::as_str)
}

fn text_config_model_type(json: &serde_json::Value) -> Option<&str> {
    json.get("text_config")
        .and_then(|text| text.get("model_type"))
        .and_then(serde_json::Value::as_str)
}

fn seen_config_field(json: &serde_json::Value, key: &str) -> String {
    json.get(key)
        .map_or_else(|| "missing".to_string(), serde_json::Value::to_string)
}

fn unrecognized_config_error(json: &serde_json::Value) -> anyhow::Error {
    let families = [
        ("DeepSeek-V2-Lite", cfg!(feature = "deepseek-v2-lite")),
        ("Gemma 4", cfg!(feature = "gemma4")),
        ("GLM5.2", cfg!(feature = "glm52")),
        ("Kimi-K2.6", cfg!(feature = "kimi-k2")),
        ("Qwen3", cfg!(feature = "qwen3")),
        ("Qwen3.5", cfg!(feature = "qwen35")),
    ]
    .iter()
    .filter_map(|&(name, compiled)| compiled.then_some(name))
    .collect::<Vec<_>>()
    .join(", ");
    anyhow::anyhow!(
        "unrecognized model config: model_type={}, architectures={}; \
         model families compiled into this build: {families}",
        seen_config_field(json, "model_type"),
        seen_config_field(json, "architectures"),
    )
}

/// Detect model type from config.json.
pub fn detect_model_type(model_path: impl AsRef<Path>) -> Result<ModelType> {
    let config_path = model_path.as_ref().join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;

    if config_model_type(&json) == Some("deepseek_v2") {
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

    if config_model_type(&json) == Some("glm_moe_dsa") {
        #[cfg(feature = "glm52")]
        {
            pegainfer_glm52::probe_config_json(&json)?;
            return Ok(ModelType::Glm52);
        }
        #[cfg(not(feature = "glm52"))]
        anyhow::bail!(
            "GLM5.2 support is feature-gated; rebuild pegainfer-server with --features glm52"
        );
    }

    if matches!(config_model_type(&json), Some("kimi_k25" | "kimi_k2"))
        || text_config_model_type(&json) == Some("kimi_k2")
    {
        #[cfg(feature = "kimi-k2")]
        {
            pegainfer_kimi_k2::probe_config_json(&json)?;
            return Ok(ModelType::KimiK2);
        }
        #[cfg(not(feature = "kimi-k2"))]
        anyhow::bail!(
            "Kimi-K2 support is feature-gated; rebuild pegainfer-server with --features kimi-k2"
        );
    }

    if matches!(config_model_type(&json), Some("gemma4" | "gemma4_unified"))
        || matches!(
            text_config_model_type(&json),
            Some("gemma4_text" | "gemma4_unified_text")
        )
    {
        #[cfg(feature = "gemma4")]
        {
            pegainfer_gemma4::probe_config_json(&json)?;
            return Ok(ModelType::Gemma4);
        }
        #[cfg(not(feature = "gemma4"))]
        anyhow::bail!(
            "Gemma 4 support is feature-gated; rebuild pegainfer-server with --features gemma4"
        );
    }

    if config_model_type(&json) == Some("qwen3_5")
        || text_config_model_type(&json) == Some("qwen3_5_text")
    {
        #[cfg(feature = "qwen35")]
        {
            pegainfer_qwen35::probe_config_json(&json)?;
            return Ok(ModelType::Qwen35);
        }
        #[cfg(not(feature = "qwen35"))]
        anyhow::bail!(
            "Qwen3.5 support is feature-gated; rebuild pegainfer-server with --features qwen35"
        );
    }

    if config_model_type(&json) == Some("qwen3") {
        #[cfg(feature = "qwen3")]
        {
            pegainfer_qwen3::probe_config_json(&json)?;
            return Ok(ModelType::Qwen3);
        }
        #[cfg(not(feature = "qwen3"))]
        anyhow::bail!(
            "Qwen3 support is feature-gated; rebuild pegainfer-server with --features qwen3"
        );
    }

    Err(unrecognized_config_error(&json))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect(json: &str) -> Result<ModelType> {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), json).unwrap();
        detect_model_type(dir.path())
    }

    #[test]
    #[cfg(feature = "qwen3")]
    fn qwen3_identity_routes_to_qwen3() {
        let json = r#"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}"#;
        assert_eq!(detect(json).unwrap(), ModelType::Qwen3);
    }

    #[test]
    #[cfg(feature = "qwen35")]
    fn qwen35_identity_routes_to_qwen35() {
        let json = r#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],"text_config":{"model_type":"qwen3_5_text"}}"#;
        assert_eq!(detect(json).unwrap(), ModelType::Qwen35);
    }

    #[test]
    #[cfg(not(feature = "gemma4"))]
    fn gemma4_12b_feature_gated() {
        let json = r#"{"model_type":"gemma4_unified","architectures":["Gemma4UnifiedForConditionalGeneration"],"text_config":{"model_type":"gemma4_unified_text"}}"#;
        let err = detect(json).unwrap_err().to_string();
        assert!(err.contains("feature-gated"));
        assert!(err.contains("--features gemma4"));
    }

    #[test]
    #[cfg(feature = "gemma4")]
    fn gemma4_12b_identity_routes_to_gemma4() {
        let json = r#"{"model_type":"gemma4_unified","architectures":["Gemma4UnifiedForConditionalGeneration"],"text_config":{"model_type":"gemma4_unified_text","head_dim":256,"global_head_dim":512,"sliding_window":1024,"attention_k_eq_v":true,"num_kv_shared_layers":0,"hidden_activation":"gelu_pytorch_tanh","enable_moe_block":false,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"]}}"#;
        assert_eq!(detect(json).unwrap(), ModelType::Gemma4);
    }

    #[test]
    #[cfg(feature = "gemma4")]
    fn gemma4_26b_identity_routes_to_gemma4() {
        let json = r#"{"model_type":"gemma4","architectures":["Gemma4ForConditionalGeneration"],"text_config":{"model_type":"gemma4_text","head_dim":256,"global_head_dim":512,"sliding_window":1024,"attention_k_eq_v":true,"num_kv_shared_layers":0,"hidden_activation":"gelu_pytorch_tanh","enable_moe_block":true,"num_experts":128,"top_k_experts":8,"moe_intermediate_size":704,"intermediate_size":2112,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"]}}"#;
        assert_eq!(detect(json).unwrap(), ModelType::Gemma4);
    }

    #[test]
    fn gemma3_previous_generation_rejected() {
        let json = r#"{"model_type":"gemma3","architectures":["Gemma3ForConditionalGeneration"],"text_config":{"model_type":"gemma3_text"}}"#;
        let err = detect(json).unwrap_err().to_string();
        assert!(err.contains("gemma3"), "{err}");
    }

    #[test]
    fn unknown_family_without_text_config_rejected() {
        let json = r#"{"model_type":"frobnicate_lm","architectures":["FrobnicateForCausalLM"]}"#;
        let err = detect(json).unwrap_err().to_string();
        assert!(err.contains("frobnicate_lm"));
    }

    #[test]
    fn empty_config_rejected() {
        let err = detect("{}").unwrap_err().to_string();
        assert!(err.contains("unrecognized"));
    }

    #[test]
    #[cfg(feature = "qwen3")]
    fn qwen3_bad_architectures_rejected() {
        let json = r#"{"model_type":"qwen3","architectures":["SomethingElse"]}"#;
        detect(json).unwrap_err();
    }

    #[test]
    fn invalid_typed_identity_fields_are_shown_verbatim() {
        let err = detect(r#"{"model_type":123,"architectures":"Foo"}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("model_type=123"));
        assert!(err.contains("architectures=\"Foo\""));
    }
}
