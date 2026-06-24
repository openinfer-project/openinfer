use anyhow::Result;
use serde::Deserialize;
use std::fs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    partial_rotary_factor: f64,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    rope_parameters: RopeParameters,
    max_position_embeddings: Option<usize>,
    eos_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    text_config: TextConfig,
    max_position_embeddings: Option<usize>,
}

/// Qwen3.5 model configuration (text-only).
#[derive(Debug)]
pub(crate) struct Config35 {
    // Common
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) eos_token_id: u32,

    // Full attention params
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,

    // Linear attention params
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_conv_kernel_dim: usize,

    // RoPE
    pub(crate) rope_theta: f32,
    pub(crate) rotary_dim: usize,
    pub(crate) max_position_embeddings: usize,

    // Layer layout
    pub(crate) layer_types: Vec<LayerType>,
}

/// GDN dims the Triton-AOT kernels are built for; a mismatched model is rejected at load.
const GDN_AOT_KEY_HEAD_DIM: usize = 128;
const GDN_AOT_VALUE_HEAD_DIM: usize = 128;
const GDN_AOT_NUM_VALUE_HEADS: usize = 32;

impl Config35 {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let raw: RawConfig = serde_json::from_str(&content)?;
        let root_max_position_embeddings = raw.max_position_embeddings;
        let t = raw.text_config;

        let layer_types: Vec<LayerType> = t
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" => Ok(LayerType::FullAttention),
                "linear_attention" => Ok(LayerType::LinearAttention),
                other => Err(anyhow::anyhow!("Unknown layer type: {}", other)),
            })
            .collect::<Result<_>>()?;

        anyhow::ensure!(
            layer_types.len() == t.num_hidden_layers,
            "layer_types length {} != num_hidden_layers {}",
            layer_types.len(),
            t.num_hidden_layers
        );

        let rotary_dim = (t.head_dim as f64 * t.rope_parameters.partial_rotary_factor) as usize;
        anyhow::ensure!(rotary_dim > 0, "Qwen3.5 rotary_dim must be positive");
        let max_position_embeddings = t
            .max_position_embeddings
            .or(root_max_position_embeddings)
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing max_position_embeddings"))?;
        anyhow::ensure!(
            max_position_embeddings > 0,
            "Qwen3.5 max_position_embeddings must be positive"
        );

        anyhow::ensure!(
            t.linear_key_head_dim == GDN_AOT_KEY_HEAD_DIM
                && t.linear_value_head_dim == GDN_AOT_VALUE_HEAD_DIM
                && t.linear_num_value_heads == GDN_AOT_NUM_VALUE_HEADS,
            "Qwen3.5 GDN Triton-AOT kernels are baked for key/value head dim {}/{}, \
             {} value heads; config has {}/{}, {}. Rebuild openinfer-kernels to match.",
            GDN_AOT_KEY_HEAD_DIM,
            GDN_AOT_VALUE_HEAD_DIM,
            GDN_AOT_NUM_VALUE_HEADS,
            t.linear_key_head_dim,
            t.linear_value_head_dim,
            t.linear_num_value_heads,
        );

        Ok(Self {
            hidden_size: t.hidden_size,
            intermediate_size: t.intermediate_size,
            num_hidden_layers: t.num_hidden_layers,
            vocab_size: t.vocab_size,
            rms_norm_eps: t.rms_norm_eps as f32,
            eos_token_id: t.eos_token_id,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            linear_num_key_heads: t.linear_num_key_heads,
            linear_key_head_dim: t.linear_key_head_dim,
            linear_num_value_heads: t.linear_num_value_heads,
            linear_value_head_dim: t.linear_value_head_dim,
            linear_conv_kernel_dim: t.linear_conv_kernel_dim,
            rope_theta: t.rope_parameters.rope_theta as f32,
            rotary_dim,
            max_position_embeddings,
            layer_types,
        })
    }

    /// Number of full attention layers in the model.
    pub(crate) fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == LayerType::FullAttention)
            .count()
    }

    /// Total Q dimension for full attention (includes gate).
    pub(crate) fn full_attn_q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim * 2
    }

    /// Q dimension for full attention (without gate).
    pub(crate) fn full_attn_q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// KV dimension for full attention.
    pub(crate) fn full_attn_kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// QKV projection output dimension for linear attention.
    pub(crate) fn linear_attn_qkv_dim(&self) -> usize {
        let q_dim = self.linear_num_key_heads * self.linear_key_head_dim;
        let k_dim = q_dim;
        let v_dim = self.linear_num_value_heads * self.linear_value_head_dim;
        q_dim + k_dim + v_dim
    }

    /// Z projection output dimension for linear attention.
    pub(crate) fn linear_attn_z_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
}
