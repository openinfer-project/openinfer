use anyhow::Result;
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Nested `rope_parameters` in text_config.
#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: String,
}

/// `text_config` sub-object inside Higgs config.json.
#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

fn default_max_position_embeddings() -> usize {
    32768
}

impl TextConfig {
    /// Resolve `rope_theta` from nested `rope_parameters` if present,
    /// otherwise fall back to a direct `rope_theta` field (not present in Higgs
    /// but supported for robustness).
    pub fn rope_theta(&self) -> f32 {
        if let Some(ref rp) = self.rope_parameters {
            rp.rope_theta
        } else {
            1_000_000.0
        }
    }
}

/// `audio_encoder_config` sub-object.
#[derive(Clone, Debug, Deserialize)]
pub struct AudioEncoderConfig {
    pub num_codebooks: usize,
    pub vocab_size: usize,
    pub out_dim: usize,
    pub use_delay_pattern: bool,
    pub tie_word_embeddings: bool,
    pub model_type: String,
    pub encoder_type: String,
}

/// Top-level Higgs config.
#[derive(Clone, Debug, Deserialize)]
pub struct HiggsConfig {
    pub text_config: TextConfig,
    pub audio_encoder_config: AudioEncoderConfig,
    pub audio_token_id: i64,
    pub model_type: String,
}

impl HiggsConfig {
    /// Load config from a directory containing `config.json`.
    pub fn from_path(model_path: &Path) -> Result<Self> {
        let config_path = model_path.join("config.json");
        let content = fs::read_to_string(&config_path)?;
        let config: HiggsConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_model_path() -> PathBuf {
        let env_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("docs/private/higgs-audio-v3-tts-4b"));
        // Resolve relative paths against the workspace root (CARGO_MANIFEST_DIR/..).
        if env_path.is_absolute() {
            env_path
        } else {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root")
                .join(&env_path)
        }
    }

    #[test]
    fn parse_higgs_config_and_assert_facts() {
        let config = HiggsConfig::from_path(&test_model_path()).expect("failed to load config");

        // text_config facts
        assert_eq!(config.text_config.hidden_size, 2560);
        assert_eq!(config.text_config.intermediate_size, 9728);
        assert_eq!(config.text_config.num_hidden_layers, 36);
        assert_eq!(config.text_config.num_attention_heads, 32);
        assert_eq!(config.text_config.num_key_value_heads, 8);
        assert_eq!(config.text_config.head_dim, 128);
        assert_eq!(config.text_config.vocab_size, 151936);
        assert!((config.text_config.rms_norm_eps - 1e-6).abs() < 1e-10);
        assert!(config.text_config.tie_word_embeddings);
        assert_eq!(config.text_config.max_position_embeddings, 32768);
        assert_eq!(config.text_config.hidden_act, "silu");

        // rope_theta from nested rope_parameters
        assert_eq!(config.text_config.rope_theta(), 1_000_000.0);

        // audio_encoder_config facts
        assert_eq!(config.audio_encoder_config.num_codebooks, 8);
        assert_eq!(config.audio_encoder_config.vocab_size, 1026);
        assert_eq!(config.audio_encoder_config.out_dim, 2560);
        assert!(config.audio_encoder_config.use_delay_pattern);
        assert!(config.audio_encoder_config.tie_word_embeddings);

        // top-level
        assert_eq!(config.audio_token_id, -100);
        assert_eq!(config.model_type, "higgs_multimodal_qwen3");
    }
}
