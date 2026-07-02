use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;

use crate::config::Config35;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DFlashLayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Debug)]
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
    pub(crate) mask_token_id: u32,
    pub(crate) target_layer_ids: Vec<usize>,
    pub(crate) layer_types: Vec<DFlashLayerType>,
    pub(crate) sliding_window: usize,
    pub(crate) anchor_first: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct DFlashInnerConfig {
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f32,
}

#[derive(Deserialize)]
struct RawDFlashConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_target_layers: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default)]
    block_size: Option<usize>,
    #[serde(default)]
    dflash_config: Option<DFlashInnerConfig>,
    #[serde(default)]
    mask_token_id: Option<u32>,
    #[serde(default)]
    target_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    layer_types: Option<Vec<DFlashLayerType>>,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    num_anchors: Option<usize>,
}

fn default_max_position_embeddings() -> usize {
    40960
}

impl DFlashConfig {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let raw: RawDFlashConfig = serde_json::from_str(&content)?;
        let rope_theta = raw
            .rope_theta
            .or(raw.rope_parameters.map(|r| r.rope_theta))
            .context("Qwen3.5 DFlash config missing rope_theta / rope_parameters.rope_theta")?;
        let (block_size, mask_token_id, target_layer_ids) = match raw.dflash_config {
            Some(inner) => (
                inner.block_size,
                inner.mask_token_id,
                inner.target_layer_ids,
            ),
            None => (
                raw.block_size
                    .context("Qwen3.5 DFlash config missing block_size (no dflash_config block)")?,
                raw.mask_token_id.context(
                    "Qwen3.5 DFlash config missing mask_token_id (no dflash_config block)",
                )?,
                raw.target_layer_ids.context(
                    "Qwen3.5 DFlash config missing target_layer_ids (no dflash_config block)",
                )?,
            ),
        };
        let layer_types = raw
            .layer_types
            .unwrap_or_else(|| vec![DFlashLayerType::FullAttention; raw.num_hidden_layers]);
        let sliding_window = raw.sliding_window.unwrap_or(raw.max_position_embeddings);

        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            num_target_layers: raw.num_target_layers,
            head_dim: raw.head_dim,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta,
            max_position_embeddings: raw.max_position_embeddings,
            block_size,
            mask_token_id,
            target_layer_ids,
            layer_types,
            sliding_window,
            anchor_first: raw.num_anchors.is_some(),
        })
    }

    pub(crate) fn anchor_first(&self) -> bool {
        self.anchor_first
    }

    pub(crate) fn validate_for_target(&self, target: &Config35) -> Result<()> {
        anyhow::ensure!(
            self.hidden_size == target.hidden_size,
            "Qwen3.5 DFlash hidden_size {} does not match target {}",
            self.hidden_size,
            target.hidden_size
        );
        anyhow::ensure!(
            self.num_target_layers == target.num_hidden_layers,
            "Qwen3.5 DFlash num_target_layers {} does not match target layers {}",
            self.num_target_layers,
            target.num_hidden_layers
        );
        anyhow::ensure!(
            self.num_attention_heads > 0
                && self.num_key_value_heads > 0
                && self.head_dim > 0
                && self.num_attention_heads % self.num_key_value_heads == 0,
            "Qwen3.5 DFlash attention geometry is invalid: heads={}, kv_heads={}, head_dim={}",
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim
        );
        anyhow::ensure!(
            self.vocab_size == target.vocab_size,
            "Qwen3.5 DFlash vocab_size {} does not match target {}",
            self.vocab_size,
            target.vocab_size
        );
        anyhow::ensure!(
            (self.rope_theta - target.rope_theta).abs() < f32::EPSILON,
            "Qwen3.5 DFlash rope_theta {} does not match target {}",
            self.rope_theta,
            target.rope_theta
        );
        anyhow::ensure!(
            self.max_position_embeddings >= target.max_position_embeddings,
            "Qwen3.5 DFlash max_position_embeddings {} is smaller than target {}",
            self.max_position_embeddings,
            target.max_position_embeddings
        );
        anyhow::ensure!(
            self.block_size >= 2,
            "Qwen3.5 DFlash block_size must be >= 2, got {}",
            self.block_size
        );
        anyhow::ensure!(
            self.mask_token_id < target.vocab_size as u32,
            "Qwen3.5 DFlash mask_token_id {} is outside target vocab_size {}",
            self.mask_token_id,
            target.vocab_size
        );
        anyhow::ensure!(
            !self.target_layer_ids.is_empty(),
            "Qwen3.5 DFlash target_layer_ids must not be empty"
        );
        anyhow::ensure!(
            self.target_layer_ids
                .iter()
                .all(|&layer| layer < target.num_hidden_layers),
            "Qwen3.5 DFlash target_layer_ids must be within target layer count"
        );
        anyhow::ensure!(
            self.target_layer_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "Qwen3.5 DFlash target_layer_ids must be strictly increasing"
        );
        anyhow::ensure!(
            self.layer_types.len() == self.num_hidden_layers,
            "Qwen3.5 DFlash layer_types length {} does not match draft layers {}",
            self.layer_types.len(),
            self.num_hidden_layers
        );
        if self
            .layer_types
            .iter()
            .any(|layer| *layer == DFlashLayerType::SlidingAttention)
        {
            anyhow::ensure!(
                self.sliding_window >= self.block_size,
                "Qwen3.5 DFlash sliding_window {} must cover block_size {}",
                self.sliding_window,
                self.block_size
            );
        }
        Ok(())
    }
}
