use anyhow::Result;
use serde::Deserialize;
use std::fs;

pub(crate) const PREFILL_ATTENTION_CTA_TILE_Q: i32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TensorParallelConfig {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
    pub(crate) eos_token_id: u32,
    pub(crate) tie_word_embeddings: bool,
    #[serde(skip)]
    pub(crate) stop_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct DFlashConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) num_target_layers: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) max_position_embeddings: usize,
    pub(crate) block_size: usize,
    pub(crate) dflash_config: DFlashInnerConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct DFlashInnerConfig {
    pub(crate) mask_token_id: u32,
    pub(crate) target_layer_ids: Vec<usize>,
}

fn default_max_position_embeddings() -> usize {
    40960
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    eos_token_id: EosTokenIds,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosTokenIds {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Single(token_id) => vec![token_id],
            Self::Multiple(token_ids) => token_ids,
        }
    }
}

impl Config {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let mut config: Config = serde_json::from_str(&content)?;
        config.stop_token_ids = Self::load_stop_token_ids(model_path, config.eos_token_id)?;
        Ok(config)
    }

    pub(crate) fn lm_head_tensor_name(&self) -> &'static str {
        if self.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        }
    }

    pub(crate) fn local_num_attention_heads(&self, tp: TensorParallelConfig) -> usize {
        self.num_attention_heads / tp.world_size
    }

    pub(crate) fn local_num_key_value_heads(&self, tp: TensorParallelConfig) -> usize {
        self.num_key_value_heads / tp.world_size
    }

    pub(crate) fn local_intermediate_size(&self, tp: TensorParallelConfig) -> usize {
        self.intermediate_size / tp.world_size
    }

    pub(crate) fn local_q_dim(&self, tp: TensorParallelConfig) -> usize {
        self.local_num_attention_heads(tp) * self.head_dim
    }

    pub(crate) fn local_kv_dim(&self, tp: TensorParallelConfig) -> usize {
        self.local_num_key_value_heads(tp) * self.head_dim
    }

    fn load_stop_token_ids(model_path: &str, fallback_eos_token_id: u32) -> Result<Vec<u32>> {
        let generation_config_path = format!("{}/generation_config.json", model_path);
        match fs::read_to_string(&generation_config_path) {
            Ok(content) => {
                let generation_config: GenerationConfig = serde_json::from_str(&content)?;
                let mut stop_token_ids = generation_config.eos_token_id.into_vec();
                stop_token_ids.dedup();
                Ok(stop_token_ids)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Ok(vec![fallback_eos_token_id])
            }
            Err(err) => Err(err.into()),
        }
    }
}

impl DFlashConfig {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        Ok(serde_json::from_str(&content)?)
    }

    pub(crate) fn validate_for_target(&self, target: &Config) -> Result<()> {
        anyhow::ensure!(
            self.hidden_size == target.hidden_size,
            "DFlash hidden_size {} does not match target {}",
            self.hidden_size,
            target.hidden_size
        );
        anyhow::ensure!(
            self.num_target_layers == target.num_hidden_layers,
            "DFlash num_target_layers {} does not match target layers {}",
            self.num_target_layers,
            target.num_hidden_layers
        );
        anyhow::ensure!(
            self.num_attention_heads == target.num_attention_heads
                && self.num_key_value_heads == target.num_key_value_heads
                && self.head_dim == target.head_dim,
            "DFlash attention geometry does not match target"
        );
        anyhow::ensure!(
            self.vocab_size == target.vocab_size,
            "DFlash vocab_size {} does not match target {}",
            self.vocab_size,
            target.vocab_size
        );
        anyhow::ensure!(
            self.rope_theta == target.rope_theta,
            "DFlash rope_theta {} does not match target {}",
            self.rope_theta,
            target.rope_theta
        );
        anyhow::ensure!(
            self.max_position_embeddings >= target.max_position_embeddings,
            "DFlash max_position_embeddings {} is smaller than target {}",
            self.max_position_embeddings,
            target.max_position_embeddings
        );
        anyhow::ensure!(
            self.block_size >= 2,
            "DFlash block_size must be >= 2, got {}",
            self.block_size
        );
        anyhow::ensure!(
            self.dflash_config.mask_token_id < target.vocab_size as u32,
            "DFlash mask_token_id {} is outside target vocab_size {}",
            self.dflash_config.mask_token_id,
            target.vocab_size
        );
        anyhow::ensure!(
            self.dflash_config.target_layer_ids.len() == self.num_hidden_layers,
            "DFlash target_layer_ids length {} does not match draft layers {}",
            self.dflash_config.target_layer_ids.len(),
            self.num_hidden_layers
        );
        anyhow::ensure!(
            self.dflash_config
                .target_layer_ids
                .iter()
                .all(|&layer| layer < target.num_hidden_layers),
            "DFlash target_layer_ids must be within target layer count"
        );
        anyhow::ensure!(
            self.dflash_config
                .target_layer_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "DFlash target_layer_ids must be strictly increasing"
        );
        Ok(())
    }
}

impl TensorParallelConfig {
    pub(crate) fn validate_for(self, config: &Config) -> Result<()> {
        if self.world_size == 0 {
            return Err(anyhow::anyhow!("tensor_parallel.world_size must be >= 1"));
        }
        if self.rank >= self.world_size {
            return Err(anyhow::anyhow!(
                "tensor_parallel.rank {} must be < world_size {}",
                self.rank,
                self.world_size
            ));
        }
        if !config.num_attention_heads.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "num_attention_heads={} not divisible by tp world_size={}",
                config.num_attention_heads,
                self.world_size
            ));
        }
        if !config.num_key_value_heads.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "num_key_value_heads={} not divisible by tp world_size={}",
                config.num_key_value_heads,
                self.world_size
            ));
        }
        if !config.intermediate_size.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "intermediate_size={} not divisible by tp world_size={}",
                config.intermediate_size,
                self.world_size
            ));
        }
        Ok(())
    }

    pub(crate) fn shard_range(self, total: usize) -> (usize, usize) {
        let shard_len = total / self.world_size;
        (self.rank * shard_len, shard_len)
    }

    pub(crate) fn is_sharded(self) -> bool {
        self.world_size > 1
    }
}
