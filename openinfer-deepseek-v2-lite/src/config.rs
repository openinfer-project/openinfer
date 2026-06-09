use std::{fs, path::Path};

use anyhow::{Result, ensure};
use serde::Deserialize;

use crate::ep::SUPPORTED_ROUTED_EXPERTS;

pub(crate) const SUPPORTED_HIDDEN_SIZE: usize = 2048;

#[derive(Clone, Debug, Deserialize)]
pub struct RopeScaling {
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub factor: f32,
    pub mscale: f32,
    pub mscale_all_dim: f32,
    pub original_max_position_embeddings: usize,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    #[serde(rename = "num_experts_per_tok")]
    pub num_experts_per_token: usize,
    pub routed_scaling_factor: f32,
    pub scoring_func: String,
    pub topk_method: String,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn from_model_dir(model_path: impl AsRef<Path>) -> Result<Self> {
        let config_path = model_path.as_ref().join("config.json");
        let content = fs::read_to_string(&config_path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate_lite()?;
        Ok(config)
    }

    pub fn validate_lite(&self) -> Result<()> {
        ensure!(
            self.model_type == "deepseek_v2",
            "DeepSeek-V2-Lite expects model_type=deepseek_v2, got {}",
            self.model_type
        );
        ensure!(
            self.hidden_size == SUPPORTED_HIDDEN_SIZE,
            "DeepSeek-V2-Lite expects hidden_size={}, got {}",
            SUPPORTED_HIDDEN_SIZE,
            self.hidden_size
        );
        ensure!(
            self.num_hidden_layers == 27,
            "DeepSeek-V2-Lite expects num_hidden_layers=27, got {}",
            self.num_hidden_layers
        );
        ensure!(
            self.num_attention_heads == 16,
            "DeepSeek-V2-Lite expects num_attention_heads=16, got {}",
            self.num_attention_heads
        );
        ensure!(
            self.num_key_value_heads == 16,
            "DeepSeek-V2-Lite expects num_key_value_heads=16, got {}",
            self.num_key_value_heads
        );
        ensure!(
            self.q_lora_rank.is_none(),
            "DeepSeek-V2-Lite first gate expects q_lora_rank=null, got {:?}",
            self.q_lora_rank
        );
        ensure!(
            self.kv_lora_rank == 512,
            "DeepSeek-V2-Lite expects kv_lora_rank=512, got {}",
            self.kv_lora_rank
        );
        ensure!(
            self.qk_nope_head_dim == 128 && self.qk_rope_head_dim == 64 && self.v_head_dim == 128,
            "DeepSeek-V2-Lite expects qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128; got {}/{}/{}",
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim
        );
        ensure!(
            self.n_routed_experts == SUPPORTED_ROUTED_EXPERTS,
            "DeepSeek-V2-Lite expects n_routed_experts={}, got {}",
            SUPPORTED_ROUTED_EXPERTS,
            self.n_routed_experts
        );
        ensure!(
            self.n_shared_experts == 2,
            "DeepSeek-V2-Lite expects n_shared_experts=2, got {}",
            self.n_shared_experts
        );
        ensure!(
            self.num_experts_per_token == 6,
            "DeepSeek-V2-Lite expects num_experts_per_tok=6, got {}",
            self.num_experts_per_token
        );
        ensure!(
            self.first_k_dense_replace == 1,
            "DeepSeek-V2-Lite expects first_k_dense_replace=1, got {}",
            self.first_k_dense_replace
        );
        ensure!(
            self.moe_layer_freq == 1,
            "DeepSeek-V2-Lite expects moe_layer_freq=1, got {}",
            self.moe_layer_freq
        );
        ensure!(
            self.intermediate_size > 0 && self.moe_intermediate_size > 0,
            "DeepSeek-V2-Lite intermediate sizes must be positive"
        );
        ensure!(
            self.scoring_func == "softmax",
            "DeepSeek-V2-Lite first gate expects scoring_func=softmax, got {}",
            self.scoring_func
        );
        ensure!(
            self.topk_method == "greedy",
            "DeepSeek-V2-Lite first gate expects topk_method=greedy, got {}",
            self.topk_method
        );
        ensure!(
            self.n_group == 1 && self.topk_group == 1,
            "DeepSeek-V2-Lite greedy routing expects n_group=1 and topk_group=1, got {}/{}",
            self.n_group,
            self.topk_group
        );
        ensure!(
            !self.norm_topk_prob,
            "DeepSeek-V2-Lite first gate expects norm_topk_prob=false"
        );
        ensure!(
            (self.routed_scaling_factor - 1.0).abs() < f32::EPSILON,
            "DeepSeek-V2-Lite first gate expects routed_scaling_factor=1.0, got {}",
            self.routed_scaling_factor
        );
        if let Some(rope_scaling) = &self.rope_scaling {
            ensure!(
                rope_scaling.kind == "yarn",
                "DeepSeek-V2-Lite first gate expects rope_scaling.type=yarn, got {}",
                rope_scaling.kind
            );
            ensure!(
                rope_scaling.original_max_position_embeddings > 0
                    && rope_scaling.original_max_position_embeddings
                        <= self.max_position_embeddings,
                "DeepSeek-V2-Lite rope_scaling original_max_position_embeddings={} must be in 1..={}",
                rope_scaling.original_max_position_embeddings,
                self.max_position_embeddings
            );
            ensure!(
                rope_scaling.factor >= 1.0
                    && rope_scaling.beta_fast > 0
                    && rope_scaling.beta_slow > 0
                    && rope_scaling.mscale > 0.0
                    && rope_scaling.mscale_all_dim > 0.0,
                "DeepSeek-V2-Lite rope_scaling values must be positive"
            );
        }
        Ok(())
    }

    pub fn is_moe_layer(&self, layer: usize) -> bool {
        layer >= self.first_k_dense_replace
            && (layer - self.first_k_dense_replace).is_multiple_of(self.moe_layer_freq)
    }

    pub fn query_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn q_proj_rows(&self) -> usize {
        self.num_attention_heads * self.query_head_dim()
    }

    pub fn kv_a_proj_rows(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    pub fn kv_b_proj_rows(&self) -> usize {
        self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim)
    }

    pub fn o_proj_cols(&self) -> usize {
        self.num_attention_heads * self.v_head_dim
    }

    pub fn shared_moe_intermediate(&self) -> usize {
        self.n_shared_experts * self.moe_intermediate_size
    }

    pub fn supported_plain_rope_context(&self) -> usize {
        self.rope_scaling
            .as_ref()
            .map_or(self.max_position_embeddings, |rope_scaling| {
                rope_scaling.original_max_position_embeddings
            })
    }
}

#[cfg(test)]
pub(crate) fn test_lite_config() -> Config {
    Config {
        model_type: "deepseek_v2".to_string(),
        bos_token_id: 100_000,
        eos_token_id: 100_001,
        vocab_size: 102_400,
        hidden_size: 2048,
        intermediate_size: 10944,
        moe_intermediate_size: 1408,
        num_hidden_layers: 27,
        first_k_dense_replace: 1,
        moe_layer_freq: 1,
        num_attention_heads: 16,
        num_key_value_heads: 16,
        q_lora_rank: None,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        n_routed_experts: 64,
        n_shared_experts: 2,
        num_experts_per_token: 6,
        routed_scaling_factor: 1.0,
        scoring_func: "softmax".to_string(),
        topk_method: "greedy".to_string(),
        n_group: 1,
        topk_group: 1,
        norm_topk_prob: false,
        rms_norm_eps: 0.000_001,
        rope_theta: 10000.0,
        rope_scaling: Some(RopeScaling {
            beta_fast: 32,
            beta_slow: 1,
            factor: 40.0,
            mscale: 0.707,
            mscale_all_dim: 0.707,
            original_max_position_embeddings: 4096,
            kind: "yarn".to_string(),
        }),
        max_position_embeddings: 163_840,
        tie_word_embeddings: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yarn_config_limits_first_gate_to_original_context() {
        let config = test_lite_config();

        assert_eq!(config.supported_plain_rope_context(), 4096);
    }
}
