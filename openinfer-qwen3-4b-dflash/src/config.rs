use anyhow::{Result, bail};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct DFlashInnerConfig {
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DFlashConfig {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub block_size: usize,
    pub dflash_config: DFlashInnerConfig,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_target_layers: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
}

impl DFlashConfig {
    pub fn from_model_dir(model_path: &Path) -> Result<Self> {
        let content = fs::read_to_string(model_path.join("config.json"))?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self
            .architectures
            .iter()
            .all(|name| name != "DFlashDraftModel")
        {
            bail!("DFlash config architectures must include DFlashDraftModel");
        }
        if self.attention_bias {
            bail!("DFlash v1 expects bias-free Qwen3 projections");
        }
        if self.attention_dropout != 0.0 {
            bail!("DFlash inference expects attention_dropout=0");
        }
        if self.num_hidden_layers == 0 {
            bail!("DFlash draft must have at least one layer");
        }
        if self.num_hidden_layers != 5 {
            bail!(
                "openinfer-qwen3-4b-dflash supports only Qwen3-4B-DFlash-b16 with 5 draft layers, got {}",
                self.num_hidden_layers
            );
        }
        if self.block_size != 16 {
            bail!(
                "openinfer-qwen3-4b-dflash supports only Qwen3-4B-DFlash-b16 block_size=16, got {}",
                self.block_size
            );
        }
        if self.dflash_config.mask_token_id != 151669 {
            bail!(
                "openinfer-qwen3-4b-dflash supports only Qwen3-4B-DFlash-b16 mask_token_id=151669, got {}",
                self.dflash_config.mask_token_id
            );
        }
        if self.hidden_size == 0 || self.head_dim == 0 {
            bail!("DFlash hidden_size/head_dim must be positive");
        }
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 {
            bail!("DFlash attention/KV head counts must be positive");
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            bail!("DFlash GQA requires attention heads divisible by KV heads");
        }
        if self.dflash_config.target_layer_ids.len() != self.num_hidden_layers {
            bail!(
                "DFlash target_layer_ids len {} must match draft layers {}",
                self.dflash_config.target_layer_ids.len(),
                self.num_hidden_layers
            );
        }
        if self
            .dflash_config
            .target_layer_ids
            .iter()
            .any(|&layer| layer >= self.num_target_layers)
        {
            bail!("DFlash target_layer_ids must be within num_target_layers");
        }
        if self.dflash_config.target_layer_ids.as_slice() != [1, 9, 17, 25, 33] {
            bail!(
                "openinfer-qwen3-4b-dflash supports only Qwen3-4B-DFlash-b16 target_layer_ids=[1, 9, 17, 25, 33], got {:?}",
                self.dflash_config.target_layer_ids
            );
        }
        Ok(())
    }

    pub fn target_layer_count(&self) -> usize {
        self.dflash_config.target_layer_ids.len()
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL_DFLASH: &str = "/home/hezhaozhao/models/Qwen3-4B-DFlash-b16";

    #[test]
    fn parses_local_dflash_config() {
        let path = Path::new(LOCAL_DFLASH);
        if !path.exists() {
            eprintln!("skipping: {LOCAL_DFLASH} does not exist");
            return;
        }
        let config = DFlashConfig::from_model_dir(path).expect("config");
        assert_eq!(config.num_hidden_layers, 5);
        assert_eq!(config.block_size, 16);
        assert_eq!(config.dflash_config.mask_token_id, 151669);
        assert_eq!(config.dflash_config.target_layer_ids, [1, 9, 17, 25, 33]);
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.intermediate_size, 9728);
    }
}
