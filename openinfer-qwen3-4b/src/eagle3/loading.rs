use anyhow::{Context, Result};
use log::debug;

use crate::config::Eagle3Config;
use crate::weights::Qwen3Model;
use openinfer_core::tensor::{DeviceContext, DeviceMatrix};
use openinfer_core::weight_loader::{
    deserialize_shards, load_shard_info, load_tensor_1d, load_tensor_2d, load_tensor_bool_host,
    load_tensor_i64_host, mmap_shards, precompute_rope,
};

use super::{Eagle3DraftModel, Eagle3Layer};

impl Eagle3DraftModel {
    /// Load an EAGLE-3 drafter from a HF safetensors directory, validating its
    /// geometry against the already-loaded Qwen3 `target`.
    ///
    /// Tensor names follow the `Eagle3LlamaForCausalLM` layout: a single
    /// `midlayer.*` decoder block, a `fc` fusion projection, `norm`, a draft
    /// `lm_head`, and the `d2t`/`t2d` vocab-remap tables. There is no
    /// `embed_tokens` — the draft reuses the target's embedding at runtime.
    pub(crate) fn from_safetensors_for_target(
        ctx: &DeviceContext,
        model_path: &str,
        target: &Qwen3Model,
    ) -> Result<Self> {
        let config = Eagle3Config::from_file(model_path)
            .with_context(|| format!("load EAGLE-3 config from {model_path}"))?;
        config.validate_for_target(target.config())?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        debug!(
            "Loading EAGLE-3 drafter from {model_path}: {} shard(s)",
            shard_paths.len()
        );
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        // ---- the single "midlayer" decoder block ----
        // q/k/v share `2 * hidden_size` input columns (embed ++ hidden concat).
        let q_proj = load_tensor_2d(
            ctx,
            &shards,
            &weight_map,
            "midlayer.self_attn.q_proj.weight",
        )?;
        let k_proj = load_tensor_2d(
            ctx,
            &shards,
            &weight_map,
            "midlayer.self_attn.k_proj.weight",
        )?;
        let v_proj = load_tensor_2d(
            ctx,
            &shards,
            &weight_map,
            "midlayer.self_attn.v_proj.weight",
        )?;
        let q_dim = q_proj.rows;
        let kv_dim = k_proj.rows;
        let qkv_proj = DeviceMatrix::vstack(ctx, &[&q_proj, &k_proj, &v_proj])?;
        drop(q_proj);
        drop(k_proj);
        drop(v_proj);

        let gate_proj = load_tensor_2d(ctx, &shards, &weight_map, "midlayer.mlp.gate_proj.weight")?;
        let up_proj = load_tensor_2d(ctx, &shards, &weight_map, "midlayer.mlp.up_proj.weight")?;
        let gate_up_proj = DeviceMatrix::vstack(ctx, &[&gate_proj, &up_proj])?;
        drop(gate_proj);
        drop(up_proj);

        let midlayer = Eagle3Layer {
            input_layernorm: load_tensor_1d(
                ctx,
                &shards,
                &weight_map,
                "midlayer.input_layernorm.weight",
            )?,
            hidden_norm: load_tensor_1d(ctx, &shards, &weight_map, "midlayer.hidden_norm.weight")?,
            qkv_proj,
            o_proj: load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                "midlayer.self_attn.o_proj.weight",
            )?,
            post_attention_layernorm: load_tensor_1d(
                ctx,
                &shards,
                &weight_map,
                "midlayer.post_attention_layernorm.weight",
            )?,
            gate_up_proj,
            down_proj: load_tensor_2d(ctx, &shards, &weight_map, "midlayer.mlp.down_proj.weight")?,
            q_dim,
            kv_dim,
        };

        // ---- fusion, final norm, draft head, vocab-remap tables ----
        let fc = load_tensor_2d(ctx, &shards, &weight_map, "fc.weight")?;
        // Capture-compatibility invariant: EAGLE-3 fuses exactly THREE captured
        // target layers (low/mid/high) — `fc` maps `[3 * hidden] -> [hidden]`. This
        // is what the capture path provides (`aux_hidden_state_layers` returns 3,
        // concatenated to `3 * hidden`); a mismatch here means the drafter expects a
        // different number of captured layers than we feed it.
        anyhow::ensure!(
            fc.rows == config.hidden_size && fc.cols == 3 * config.hidden_size,
            "EAGLE-3 fc must be [hidden {}, 3*hidden {}] (fuses 3 captured layers), got [{}, {}]",
            config.hidden_size,
            3 * config.hidden_size,
            fc.rows,
            fc.cols
        );
        let norm = load_tensor_1d(ctx, &shards, &weight_map, "norm.weight")?;
        let lm_head = load_tensor_2d(ctx, &shards, &weight_map, "lm_head.weight")?;

        let d2t = load_tensor_i64_host(&shards, &weight_map, "d2t")?;
        let t2d = load_tensor_bool_host(&shards, &weight_map, "t2d")?;
        anyhow::ensure!(
            d2t.len() == config.draft_vocab_size,
            "EAGLE-3 d2t length {} does not match draft_vocab_size {}",
            d2t.len(),
            config.draft_vocab_size
        );
        anyhow::ensure!(
            t2d.len() == config.vocab_size,
            "EAGLE-3 t2d length {} does not match vocab_size {}",
            t2d.len(),
            config.vocab_size
        );

        let (cos_cache, sin_cache) = precompute_rope(
            ctx,
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )?;
        ctx.sync()?;

        Ok(Self {
            config,
            fc,
            midlayer,
            norm,
            lm_head,
            d2t,
            t2d,
            cos_cache,
            sin_cache,
        })
    }
}
