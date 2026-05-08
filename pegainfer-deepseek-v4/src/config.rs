use std::{fs, path::Path};

use anyhow::{Result, ensure};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type")]
    pub kind: String,
    pub factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
    #[serde(rename = "original_max_position_embeddings")]
    pub original_seq_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorParallelConfig {
    pub rank: usize,
    pub world_size: usize,
}

impl TensorParallelConfig {
    pub fn mp8(rank: usize) -> Self {
        Self {
            rank,
            world_size: 8,
        }
    }

    pub fn validate_for(self, config: &Config) -> Result<()> {
        ensure!(
            self.world_size == 8,
            "DeepSeek V4 Flash mp checkpoint expects world_size=8"
        );
        ensure!(
            self.rank < self.world_size,
            "rank {} must be < world_size {}",
            self.rank,
            self.world_size
        );
        ensure!(
            config.vocab_size.is_multiple_of(self.world_size),
            "vocab_size={} must be divisible by world_size={}",
            config.vocab_size,
            self.world_size
        );
        ensure!(
            config.num_attention_heads.is_multiple_of(self.world_size),
            "num_attention_heads={} must be divisible by world_size={}",
            config.num_attention_heads,
            self.world_size
        );
        ensure!(
            config.n_routed_experts.is_multiple_of(self.world_size),
            "n_routed_experts={} must be divisible by world_size={}",
            config.n_routed_experts,
            self.world_size
        );
        ensure!(
            config.o_groups.is_multiple_of(self.world_size),
            "o_groups={} must be divisible by world_size={}",
            config.o_groups,
            self.world_size
        );
        Ok(())
    }

    pub fn local_vocab_size(self, config: &Config) -> usize {
        config.vocab_size / self.world_size
    }

    pub fn local_attention_heads(self, config: &Config) -> usize {
        config.num_attention_heads / self.world_size
    }

    pub fn local_experts(self, config: &Config) -> usize {
        config.n_routed_experts / self.world_size
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub vocab_size: usize,
    #[serde(rename = "hidden_size")]
    pub dim: usize,
    #[serde(rename = "moe_intermediate_size")]
    pub moe_inter_dim: usize,
    #[serde(rename = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(default, rename = "num_nextn_predict_layers")]
    pub n_mtp_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub sliding_window: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    #[serde(rename = "num_experts_per_tok")]
    pub n_activated_experts: usize,
    #[serde(rename = "num_hash_layers")]
    pub n_hash_layers: usize,
    pub scoring_func: String,
    pub routed_scaling_factor: f32,
    pub swiglu_limit: f32,
    pub rms_norm_eps: f32,
    #[serde(default = "default_hc_mult")]
    pub hc_mult: usize,
    #[serde(default = "default_hc_sinkhorn_iters")]
    pub hc_sinkhorn_iters: usize,
    #[serde(default = "default_hc_eps")]
    pub hc_eps: f32,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub max_position_embeddings: usize,
    pub rope_scaling: RopeScaling,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub compress_ratios: Vec<usize>,
}

fn default_hc_mult() -> usize {
    4
}

fn default_hc_sinkhorn_iters() -> usize {
    20
}

fn default_hc_eps() -> f32 {
    1.0e-6
}

impl Config {
    pub fn from_model_dir(model_path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(model_path.as_ref().join("config.json"))?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "deepseek_v4",
            "expected model_type=deepseek_v4, got {}",
            self.model_type
        );
        ensure!(
            self.dim == 4096,
            "expected hidden_size=4096, got {}",
            self.dim
        );
        ensure!(
            self.n_layers == 43,
            "expected num_hidden_layers=43, got {}",
            self.n_layers
        );
        ensure!(
            self.num_attention_heads == 64,
            "expected num_attention_heads=64, got {}",
            self.num_attention_heads
        );
        ensure!(
            self.num_key_value_heads == 1,
            "expected num_key_value_heads=1, got {}",
            self.num_key_value_heads
        );
        ensure!(
            self.head_dim == 512,
            "expected head_dim=512, got {}",
            self.head_dim
        );
        ensure!(
            self.q_lora_rank == 1024,
            "expected q_lora_rank=1024, got {}",
            self.q_lora_rank
        );
        ensure!(
            self.qk_rope_head_dim == 64,
            "expected qk_rope_head_dim=64, got {}",
            self.qk_rope_head_dim
        );
        ensure!(
            self.o_lora_rank == 1024,
            "expected o_lora_rank=1024, got {}",
            self.o_lora_rank
        );
        ensure!(
            self.n_routed_experts == 256,
            "expected n_routed_experts=256, got {}",
            self.n_routed_experts
        );
        ensure!(
            self.n_activated_experts == 6,
            "expected num_experts_per_tok=6, got {}",
            self.n_activated_experts
        );
        ensure!(
            self.vocab_size == 129_280,
            "expected vocab_size=129280, got {}",
            self.vocab_size
        );
        ensure!(
            self.bos_token_id == 0,
            "expected bos_token_id=0, got {}",
            self.bos_token_id
        );
        ensure!(
            self.eos_token_id == 1,
            "expected eos_token_id=1, got {}",
            self.eos_token_id
        );
        ensure!(
            self.compress_ratios.len() == self.n_layers + self.n_mtp_layers,
            "compress_ratios length {} does not match n_layers+n_mtp_layers {}",
            self.compress_ratios.len(),
            self.n_layers + self.n_mtp_layers
        );
        ensure!(
            self.max_position_embeddings == 1_048_576,
            "expected max_position_embeddings=1048576, got {}",
            self.max_position_embeddings
        );
        ensure!(
            self.rope_scaling.kind == "yarn",
            "expected rope_scaling.type=yarn, got {}",
            self.rope_scaling.kind
        );
        ensure!(
            self.rope_scaling.original_seq_len == 65_536,
            "expected original_max_position_embeddings=65536, got {}",
            self.rope_scaling.original_seq_len
        );
        Ok(())
    }
}
