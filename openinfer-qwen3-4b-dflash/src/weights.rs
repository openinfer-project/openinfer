use anyhow::{Context, Result, bail};
use log::info;
use openinfer_core::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use openinfer_core::weight_loader::{
    deserialize_shards, load_shard_info, load_tensor_1d, load_tensor_2d, mmap_shards,
    precompute_rope,
};
use std::collections::HashMap;
use std::path::Path;

use crate::config::DFlashConfig;

pub(crate) struct DFlashAttention {
    pub(crate) q_proj: DeviceMatrix,
    pub(crate) k_proj: DeviceMatrix,
    pub(crate) v_proj: DeviceMatrix,
    pub(crate) o_proj: DeviceMatrix,
    pub(crate) q_norm: DeviceVec,
    pub(crate) k_norm: DeviceVec,
}

pub(crate) struct DFlashMlp {
    pub(crate) gate_up_proj: DeviceMatrix,
    pub(crate) down_proj: DeviceMatrix,
}

pub(crate) struct DFlashLayer {
    pub(crate) input_layernorm: DeviceVec,
    pub(crate) attention: DFlashAttention,
    pub(crate) post_attention_layernorm: DeviceVec,
    pub(crate) mlp: DFlashMlp,
}

pub struct DFlashDraftModel {
    pub(crate) ctx: DeviceContext,
    pub(crate) config: DFlashConfig,
    pub(crate) layers: Vec<DFlashLayer>,
    pub(crate) fc: DeviceMatrix,
    pub(crate) hidden_norm: DeviceVec,
    pub(crate) norm: DeviceVec,
    pub(crate) cos_cache: DeviceVec,
    pub(crate) sin_cache: DeviceVec,
}

// SAFETY: The model owns one CUDA context/stream and is intended to run on one
// worker thread at a time, matching other OpenInfer model structs.
unsafe impl Send for DFlashDraftModel {}
unsafe impl Sync for DFlashDraftModel {}

impl DFlashDraftModel {
    pub fn load(model_path: &Path, device_ordinal: usize) -> Result<Self> {
        info!(
            "Loading Qwen3-4B DFlash draft model from {}",
            model_path.display()
        );
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let config = DFlashConfig::from_model_dir(model_path)?;
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("DFlash model path must be valid UTF-8"))?;
        let (shard_paths, weight_map) = load_shard_info(model_path_str)?;
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let fc = load_tensor_2d(&ctx, &shards, &weight_map, "fc.weight")
            .context("load DFlash fc.weight")?;
        ensure_matrix_shape(
            "fc.weight",
            &fc,
            config.hidden_size,
            config.hidden_size * config.target_layer_count(),
        )?;
        let hidden_norm = load_tensor_1d(&ctx, &shards, &weight_map, "hidden_norm.weight")?;
        let norm = load_tensor_1d(&ctx, &shards, &weight_map, "norm.weight")?;
        ensure_vec_len("hidden_norm.weight", &hidden_norm, config.hidden_size)?;
        ensure_vec_len("norm.weight", &norm, config.hidden_size)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(load_layer(&ctx, &shards, &weight_map, &config, layer_idx)?);
        }
        let (cos_cache, sin_cache) = precompute_rope(
            &ctx,
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )?;

        Ok(Self {
            ctx,
            config,
            layers,
            fc,
            hidden_norm,
            norm,
            cos_cache,
            sin_cache,
        })
    }

    pub fn config(&self) -> &DFlashConfig {
        &self.config
    }

    pub fn target_layer_ids(&self) -> &[usize] {
        &self.config.dflash_config.target_layer_ids
    }

    pub fn mask_token_id(&self) -> u32 {
        self.config.dflash_config.mask_token_id
    }

    pub fn device_context(&self) -> &DeviceContext {
        &self.ctx
    }
}

fn load_layer(
    ctx: &DeviceContext,
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    config: &DFlashConfig,
    layer_idx: usize,
) -> Result<DFlashLayer> {
    let prefix = format!("layers.{layer_idx}");
    let q_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.self_attn.q_proj.weight"),
    )?;
    let k_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.self_attn.k_proj.weight"),
    )?;
    let v_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.self_attn.v_proj.weight"),
    )?;
    let o_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.self_attn.o_proj.weight"),
    )?;
    ensure_matrix_shape("q_proj", &q_proj, config.q_dim(), config.hidden_size)?;
    ensure_matrix_shape("k_proj", &k_proj, config.kv_dim(), config.hidden_size)?;
    ensure_matrix_shape("v_proj", &v_proj, config.kv_dim(), config.hidden_size)?;
    ensure_matrix_shape("o_proj", &o_proj, config.hidden_size, config.q_dim())?;

    let gate_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.mlp.gate_proj.weight"),
    )?;
    let up_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.mlp.up_proj.weight"),
    )?;
    let gate_up_proj = DeviceMatrix::vstack(ctx, &[&gate_proj, &up_proj])?;
    let down_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.mlp.down_proj.weight"),
    )?;
    ensure_matrix_shape(
        "gate_up_proj",
        &gate_up_proj,
        2 * config.intermediate_size,
        config.hidden_size,
    )?;
    ensure_matrix_shape(
        "down_proj",
        &down_proj,
        config.hidden_size,
        config.intermediate_size,
    )?;

    let input_layernorm = load_tensor_1d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.input_layernorm.weight"),
    )?;
    let post_attention_layernorm = load_tensor_1d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.post_attention_layernorm.weight"),
    )?;
    let q_norm = load_tensor_1d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.self_attn.q_norm.weight"),
    )?;
    let k_norm = load_tensor_1d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.self_attn.k_norm.weight"),
    )?;
    ensure_vec_len("input_layernorm", &input_layernorm, config.hidden_size)?;
    ensure_vec_len(
        "post_attention_layernorm",
        &post_attention_layernorm,
        config.hidden_size,
    )?;
    ensure_vec_len("q_norm", &q_norm, config.head_dim)?;
    ensure_vec_len("k_norm", &k_norm, config.head_dim)?;

    Ok(DFlashLayer {
        input_layernorm,
        attention: DFlashAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        },
        post_attention_layernorm,
        mlp: DFlashMlp {
            gate_up_proj,
            down_proj,
        },
    })
}

fn ensure_matrix_shape(name: &str, matrix: &DeviceMatrix, rows: usize, cols: usize) -> Result<()> {
    if matrix.rows != rows || matrix.cols != cols {
        bail!(
            "{name} shape mismatch: expected [{rows}, {cols}], got [{}, {}]",
            matrix.rows,
            matrix.cols
        );
    }
    Ok(())
}

fn ensure_vec_len(name: &str, vector: &DeviceVec, len: usize) -> Result<()> {
    if vector.len != len {
        bail!("{name} length mismatch: expected {len}, got {}", vector.len);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL_DFLASH: &str = "/home/hezhaozhao/models/Qwen3-4B-DFlash-b16";

    #[test]
    fn loads_local_dflash_weights() {
        let path = Path::new(LOCAL_DFLASH);
        if !path.exists() {
            eprintln!("skipping: {LOCAL_DFLASH} does not exist");
            return;
        }
        let model = DFlashDraftModel::load(path, 0).expect("load model");
        assert_eq!(model.layers.len(), 5);
        assert_eq!(model.fc.rows, 2560);
        assert_eq!(model.fc.cols, 12800);
    }
}
