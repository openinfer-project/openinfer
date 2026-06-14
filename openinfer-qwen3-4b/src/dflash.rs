use anyhow::{Context, Result};
use cudarc::driver::CudaSlice;
use log::debug;

use crate::config::DFlashConfig;
use crate::weights::{Attention, MLP, Qwen3Model, TransformerBlock};
use openinfer_core::ops;
use openinfer_core::tensor::HiddenStates;
use openinfer_core::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use openinfer_core::weight_loader::{
    deserialize_shards, load_shard_info, load_tensor_1d, load_tensor_2d, mmap_shards,
    precompute_rope,
};

pub(crate) struct DFlashDraftModel {
    config: DFlashConfig,
    layers: Vec<TransformerBlock>,
    norm: DeviceVec,
    hidden_norm: DeviceVec,
    fc: DeviceMatrix,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
}

pub(crate) struct DFlashRequestState {
    layers: Vec<DFlashLayerCache>,
    pending_context: DFlashPendingContext,
    scratch: DFlashDraftScratch,
    committed_len: usize,
    max_cache_len: usize,
}

struct DFlashLayerCache {
    k: HiddenStates,
    v: HiddenStates,
}

struct DFlashPendingContext {
    buffer: HiddenStates,
    len: usize,
    capacity: usize,
}

pub(crate) struct DFlashDraftOutput<'a> {
    pub(crate) logits: &'a HiddenStates,
    pub(crate) context_len: usize,
    pub(crate) committed_len: usize,
}

struct DFlashDraftScratch {
    max_context_len: usize,
    block_token_ids_h: Vec<u32>,
    token_ids_d: CudaSlice<u32>,
    context_projected: HiddenStates,
    context_hidden: HiddenStates,
    hidden: HiddenStates,
    hidden_out: HiddenStates,
    normed: HiddenStates,
    tail_input: HiddenStates,
    q_batch: HiddenStates,
    k_tail: HiddenStates,
    v_tail: HiddenStates,
    attn_output: HiddenStates,
    o_buf: HiddenStates,
    gate_out: HiddenStates,
    up_out: HiddenStates,
    act_out: HiddenStates,
    logits_normed: HiddenStates,
    logits: HiddenStates,
}

impl DFlashRequestState {
    pub(crate) fn pending_context_len(&self) -> Option<usize> {
        (self.pending_context.len > 0).then_some(self.pending_context.len)
    }
}

impl DFlashPendingContext {
    fn new(ctx: &DeviceContext, hidden_dim: usize, capacity: usize) -> Result<Self> {
        anyhow::ensure!(
            capacity > 0,
            "DFlash pending context capacity must be non-zero"
        );
        let mut buffer = HiddenStates::zeros(ctx, hidden_dim, capacity)?;
        buffer.seq_len = 0;
        Ok(Self {
            buffer,
            len: 0,
            capacity,
        })
    }

    fn append_from(
        &mut self,
        ctx: &DeviceContext,
        src: &HiddenStates,
        src_token_offset: usize,
        token_count: usize,
        max_capacity: usize,
    ) -> Result<()> {
        let required_len = self
            .len
            .checked_add(token_count)
            .context("DFlash pending context length overflow")?;
        anyhow::ensure!(
            required_len <= max_capacity,
            "DFlash pending context length {} exceeds request capacity {}",
            required_len,
            max_capacity
        );
        self.ensure_capacity(ctx, required_len, max_capacity)?;
        self.buffer.seq_len = self.capacity;
        ops::copy_hidden_token_range_into(
            ctx,
            src,
            src_token_offset,
            &mut self.buffer,
            self.len,
            token_count,
        )?;
        self.len = required_len;
        self.buffer.seq_len = self.len;
        Ok(())
    }

    fn ensure_capacity(
        &mut self,
        ctx: &DeviceContext,
        required_len: usize,
        max_capacity: usize,
    ) -> Result<()> {
        if required_len <= self.capacity {
            return Ok(());
        }
        let doubled = self
            .capacity
            .checked_mul(2)
            .context("DFlash pending context capacity overflow")?;
        let new_capacity = required_len.max(doubled).min(max_capacity);
        anyhow::ensure!(
            new_capacity >= required_len,
            "DFlash pending context capacity {} cannot fit {} tokens",
            new_capacity,
            required_len
        );
        let mut next = HiddenStates::zeros(ctx, self.buffer.hidden_dim, new_capacity)?;
        if self.len > 0 {
            self.buffer.seq_len = self.capacity;
            ops::copy_hidden_token_range_into(ctx, &self.buffer, 0, &mut next, 0, self.len)?;
        }
        next.seq_len = self.len;
        self.buffer = next;
        self.capacity = new_capacity;
        Ok(())
    }

    fn activate_for_read(&mut self) {
        self.buffer.seq_len = self.len;
    }

    fn clear(&mut self) {
        self.len = 0;
        self.buffer.seq_len = 0;
    }
}

impl DFlashDraftScratch {
    fn new(ctx: &DeviceContext, config: &DFlashConfig, max_context_len: usize) -> Result<Self> {
        let block_size = config.block_size;
        let hidden_size = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let inter_dim = config.intermediate_size;
        let tail_capacity = max_context_len + block_size;

        Ok(Self {
            max_context_len,
            block_token_ids_h: vec![config.dflash_config.mask_token_id; block_size],
            token_ids_d: ctx.stream.alloc_zeros(block_size)?,
            context_projected: HiddenStates::zeros(ctx, hidden_size, max_context_len)?,
            context_hidden: HiddenStates::zeros(ctx, hidden_size, max_context_len)?,
            hidden: HiddenStates::zeros(ctx, hidden_size, block_size)?,
            hidden_out: HiddenStates::zeros(ctx, hidden_size, block_size)?,
            normed: HiddenStates::zeros(ctx, hidden_size, block_size)?,
            tail_input: HiddenStates::zeros(ctx, hidden_size, tail_capacity)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, block_size)?,
            k_tail: HiddenStates::zeros(ctx, kv_dim, tail_capacity)?,
            v_tail: HiddenStates::zeros(ctx, kv_dim, tail_capacity)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, block_size)?,
            o_buf: HiddenStates::zeros(ctx, hidden_size, block_size)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, block_size)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, block_size)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, block_size)?,
            logits_normed: HiddenStates::zeros(ctx, hidden_size, block_size)?,
            logits: HiddenStates::zeros(ctx, config.vocab_size, block_size)?,
        })
    }

    fn ensure_context_capacity(
        &mut self,
        ctx: &DeviceContext,
        config: &DFlashConfig,
        context_len: usize,
    ) -> Result<()> {
        if context_len > self.max_context_len {
            *self = Self::new(ctx, config, context_len)?;
        }
        let block_size = config.block_size;
        let tail_len = context_len + block_size;

        self.context_projected.seq_len = context_len;
        self.context_hidden.seq_len = context_len;
        self.hidden.seq_len = block_size;
        self.hidden_out.seq_len = block_size;
        self.normed.seq_len = block_size;
        self.tail_input.seq_len = tail_len;
        self.q_batch.seq_len = block_size;
        self.k_tail.seq_len = tail_len;
        self.v_tail.seq_len = tail_len;
        self.attn_output.seq_len = block_size;
        self.o_buf.seq_len = block_size;
        self.gate_out.seq_len = block_size;
        self.up_out.seq_len = block_size;
        self.act_out.seq_len = block_size;
        self.logits_normed.seq_len = block_size;
        self.logits.seq_len = block_size;
        Ok(())
    }
}

impl DFlashDraftModel {
    pub(crate) fn from_safetensors_for_target(
        ctx: &DeviceContext,
        model_path: &str,
        target: &Qwen3Model,
    ) -> Result<Self> {
        let config = DFlashConfig::from_file(model_path)
            .with_context(|| format!("load DFlash config from {model_path}"))?;
        config.validate_for_target(target.config())?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        debug!(
            "Loading DFlash drafter from {model_path}: {} shard(s)",
            shard_paths.len()
        );
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("layers.{layer_idx}");

            let q_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.self_attn.q_proj.weight"),
            )?;
            let k_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.self_attn.k_proj.weight"),
            )?;
            let v_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.self_attn.v_proj.weight"),
            )?;
            let q_dim = q_proj.rows;
            let kv_dim = k_proj.rows;
            let qkv_proj = DeviceMatrix::vstack(ctx, &[&q_proj, &k_proj, &v_proj])?;
            drop(q_proj);
            drop(k_proj);
            drop(v_proj);

            let gate_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.mlp.gate_proj.weight"),
            )?;
            let up_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.mlp.up_proj.weight"),
            )?;
            let gate_up_proj = DeviceMatrix::vstack(ctx, &[&gate_proj, &up_proj])?;
            drop(gate_proj);
            drop(up_proj);

            layers.push(TransformerBlock {
                input_layernorm: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention: Attention {
                    qkv_proj,
                    o_proj: load_tensor_2d(
                        ctx,
                        &shards,
                        &weight_map,
                        &format!("{prefix}.self_attn.o_proj.weight"),
                    )?,
                    q_norm: load_tensor_1d(
                        ctx,
                        &shards,
                        &weight_map,
                        &format!("{prefix}.self_attn.q_norm.weight"),
                    )?,
                    k_norm: load_tensor_1d(
                        ctx,
                        &shards,
                        &weight_map,
                        &format!("{prefix}.self_attn.k_norm.weight"),
                    )?,
                    q_dim,
                    kv_dim,
                },
                post_attention_layernorm: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                mlp: MLP {
                    gate_up_proj,
                    down_proj: load_tensor_2d(
                        ctx,
                        &shards,
                        &weight_map,
                        &format!("{prefix}.mlp.down_proj.weight"),
                    )?,
                },
            });
        }

        let norm = load_tensor_1d(ctx, &shards, &weight_map, "norm.weight")?;
        let hidden_norm = load_tensor_1d(ctx, &shards, &weight_map, "hidden_norm.weight")?;
        let fc = load_tensor_2d(ctx, &shards, &weight_map, "fc.weight")?;
        let (cos_cache, sin_cache) = precompute_rope(
            ctx,
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )?;
        ctx.sync()?;

        Ok(Self {
            config,
            layers,
            norm,
            hidden_norm,
            fc,
            cos_cache,
            sin_cache,
        })
    }

    pub(crate) fn block_size(&self) -> usize {
        self.config.block_size
    }

    pub(crate) fn config(&self) -> &DFlashConfig {
        &self.config
    }

    pub(crate) fn mask_token_id(&self) -> u32 {
        self.config.dflash_config.mask_token_id
    }

    pub(crate) fn target_layer_ids(&self) -> &[usize] {
        &self.config.dflash_config.target_layer_ids
    }

    pub(crate) fn tune_gemm_algos(&self, target: &Qwen3Model) -> Result<()> {
        let ctx = target.device_ctx();
        let block_size = self.block_size().min(ops::GEMM_LT_MAX_N);
        let hidden = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let context_dim = self.context_feature_dim();

        let fc_samples = [(&self.fc, 0)];
        for n in 1..=block_size {
            ops::gemm_lt_tune(ctx, &fc_samples, hidden, n)?;
        }

        let kv_samples: Vec<_> = self
            .layers
            .iter()
            .flat_map(|layer| {
                [
                    (&layer.attention.qkv_proj, q_dim),
                    (&layer.attention.qkv_proj, q_dim + kv_dim),
                ]
            })
            .collect();
        let min_tail_n = self.block_size() + 1;
        let max_tail_n = (self.block_size() * 2).min(ops::GEMM_LT_MAX_N);
        for n in min_tail_n..=max_tail_n {
            ops::gemm_lt_tune(ctx, &kv_samples, kv_dim, n)?;
        }

        log::info!(
            "Qwen3 DFlash cublasLt tuned: fc M={} K={} N=1..{}, kv M={} K={} N={}..{}",
            hidden,
            context_dim,
            block_size,
            kv_dim,
            hidden,
            min_tail_n,
            max_tail_n,
        );
        Ok(())
    }

    pub(crate) fn new_request_state(
        &self,
        ctx: &DeviceContext,
        max_cache_len: usize,
    ) -> Result<DFlashRequestState> {
        anyhow::ensure!(
            max_cache_len <= self.config.max_position_embeddings,
            "DFlash request cache length {} exceeds max_position_embeddings {}",
            max_cache_len,
            self.config.max_position_embeddings
        );
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let mut layers = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            layers.push(DFlashLayerCache {
                k: HiddenStates::zeros(ctx, kv_dim, max_cache_len)?,
                v: HiddenStates::zeros(ctx, kv_dim, max_cache_len)?,
            });
        }
        Ok(DFlashRequestState {
            layers,
            pending_context: DFlashPendingContext::new(
                ctx,
                self.context_feature_dim(),
                self.config.block_size.min(max_cache_len),
            )?,
            scratch: DFlashDraftScratch::new(ctx, &self.config, self.config.block_size)?,
            committed_len: 0,
            max_cache_len,
        })
    }

    pub(crate) fn append_pending_context(
        &self,
        ctx: &DeviceContext,
        state: &mut DFlashRequestState,
        captured_hidden: &HiddenStates,
        token_offset: usize,
        token_count: usize,
    ) -> Result<()> {
        anyhow::ensure!(token_count > 0, "DFlash context append needs tokens");
        anyhow::ensure!(
            captured_hidden.hidden_dim == self.context_feature_dim(),
            "DFlash captured hidden dim {} does not match expected {}",
            captured_hidden.hidden_dim,
            self.context_feature_dim()
        );
        anyhow::ensure!(
            token_offset + token_count <= captured_hidden.seq_len,
            "DFlash captured hidden token range exceeds source"
        );
        let required_committed_len = state
            .committed_len
            .checked_add(state.pending_context.len)
            .and_then(|len| len.checked_add(token_count))
            .and_then(|len| len.checked_add(self.block_size()))
            .context("DFlash pending context cache length overflow")?;
        anyhow::ensure!(
            required_committed_len <= state.max_cache_len,
            "DFlash pending context would exceed cache: committed={}, pending={}, append={}, block={}, max={}",
            state.committed_len,
            state.pending_context.len,
            token_count,
            self.block_size(),
            state.max_cache_len
        );
        state.pending_context.append_from(
            ctx,
            captured_hidden,
            token_offset,
            token_count,
            state.max_cache_len,
        )?;
        Ok(())
    }

    pub(crate) fn draft_logits<'a>(
        &self,
        target: &Qwen3Model,
        state: &'a mut DFlashRequestState,
        current_token: u32,
    ) -> Result<DFlashDraftOutput<'a>> {
        let ctx = target.device_ctx();
        let Some(context_len) = state.pending_context_len() else {
            anyhow::bail!("DFlash draft requested before target hidden context is available");
        };
        let block_size = self.block_size();
        let tail_len = context_len + block_size;
        anyhow::ensure!(
            state.committed_len + tail_len <= state.max_cache_len,
            "DFlash draft cache overflow: committed={}, tail={}, max={}",
            state.committed_len,
            tail_len,
            state.max_cache_len
        );

        state
            .scratch
            .ensure_context_capacity(ctx, &self.config, context_len)?;
        state.scratch.block_token_ids_h.fill(self.mask_token_id());
        state.scratch.block_token_ids_h[0] = current_token;
        ctx.stream.memcpy_htod(
            &state.scratch.block_token_ids_h,
            &mut state.scratch.token_ids_d,
        )?;
        target.get_embeddings_batch_into(&state.scratch.token_ids_d, &mut state.scratch.hidden)?;

        state.pending_context.activate_for_read();
        self.project_context_into(ctx, &state.pending_context.buffer, &mut state.scratch)?;
        state.pending_context.clear();

        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        debug_assert_eq!(state.scratch.hidden.hidden_dim, hidden_size);
        debug_assert_eq!(state.scratch.q_batch.hidden_dim, q_dim);
        debug_assert_eq!(state.scratch.k_tail.hidden_dim, kv_dim);
        debug_assert_eq!(state.scratch.gate_out.hidden_dim, inter_dim);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            ops::rms_norm_batch_into(
                ctx,
                &state.scratch.hidden,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                &mut state.scratch.normed,
            );

            ops::copy_hidden_token_range_into(
                ctx,
                &state.scratch.context_hidden,
                0,
                &mut state.scratch.tail_input,
                0,
                context_len,
            )?;
            ops::copy_hidden_token_range_into(
                ctx,
                &state.scratch.normed,
                0,
                &mut state.scratch.tail_input,
                context_len,
                block_size,
            )?;

            ops::gemm_rows_into(
                ctx,
                &layer.attention.qkv_proj,
                0,
                q_dim,
                &state.scratch.normed,
                &mut state.scratch.q_batch,
            );
            ops::gemm_rows_into(
                ctx,
                &layer.attention.qkv_proj,
                q_dim,
                kv_dim,
                &state.scratch.tail_input,
                &mut state.scratch.k_tail,
            );
            ops::gemm_rows_into(
                ctx,
                &layer.attention.qkv_proj,
                q_dim + kv_dim,
                kv_dim,
                &state.scratch.tail_input,
                &mut state.scratch.v_tail,
            );

            ops::dflash_qk_norm_rope_into(
                ctx,
                &mut state.scratch.q_batch,
                &mut state.scratch.k_tail,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache,
                &self.sin_cache,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                state.committed_len + context_len,
                state.committed_len,
                self.config.rms_norm_eps,
            )?;

            let cache = &mut state.layers[layer_idx];
            ops::copy_hidden_token_range_into(
                ctx,
                &state.scratch.k_tail,
                0,
                &mut cache.k,
                state.committed_len,
                tail_len,
            )?;
            ops::copy_hidden_token_range_into(
                ctx,
                &state.scratch.v_tail,
                0,
                &mut cache.v,
                state.committed_len,
                tail_len,
            )?;
            ops::single_prefill_nhd_noncausal_into(
                ctx,
                &state.scratch.q_batch,
                &cache.k,
                &cache.v,
                &mut state.scratch.attn_output,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                state.committed_len + tail_len,
            )?;

            ops::gemm_into(
                ctx,
                &layer.attention.o_proj,
                &state.scratch.attn_output,
                &mut state.scratch.o_buf,
            );
            openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
                ctx,
                &mut state.scratch.hidden,
                &state.scratch.o_buf,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut state.scratch.normed,
            )?;

            ops::gemm_rows_into(
                ctx,
                &layer.mlp.gate_up_proj,
                0,
                inter_dim,
                &state.scratch.normed,
                &mut state.scratch.gate_out,
            );
            ops::gemm_rows_into(
                ctx,
                &layer.mlp.gate_up_proj,
                inter_dim,
                inter_dim,
                &state.scratch.normed,
                &mut state.scratch.up_out,
            );
            ops::silu_mul_batch_into(
                ctx,
                &state.scratch.gate_out,
                &state.scratch.up_out,
                &mut state.scratch.act_out,
            )?;
            ops::gemm_into(
                ctx,
                &layer.mlp.down_proj,
                &state.scratch.act_out,
                &mut state.scratch.o_buf,
            );
            ops::add_batch_into(
                ctx,
                &state.scratch.hidden,
                &state.scratch.o_buf,
                &mut state.scratch.hidden_out,
            )?;
            std::mem::swap(&mut state.scratch.hidden, &mut state.scratch.hidden_out);
        }

        let committed_len = state.committed_len;
        state.committed_len += context_len;
        self.compute_logits_with_target_head_into(target, &mut state.scratch)?;
        Ok(DFlashDraftOutput {
            logits: &state.scratch.logits,
            context_len,
            committed_len,
        })
    }

    fn context_feature_dim(&self) -> usize {
        self.config.hidden_size * self.target_layer_ids().len()
    }

    fn project_context_into(
        &self,
        ctx: &DeviceContext,
        context_features: &HiddenStates,
        scratch: &mut DFlashDraftScratch,
    ) -> Result<()> {
        ops::gemm_into(
            ctx,
            &self.fc,
            context_features,
            &mut scratch.context_projected,
        );
        ops::rms_norm_batch_into(
            ctx,
            &scratch.context_projected,
            &self.hidden_norm,
            self.config.rms_norm_eps,
            &mut scratch.context_hidden,
        );
        Ok(())
    }

    fn compute_logits_with_target_head_into(
        &self,
        target: &Qwen3Model,
        scratch: &mut DFlashDraftScratch,
    ) -> Result<()> {
        let ctx = target.device_ctx();
        ops::rms_norm_batch_into(
            ctx,
            &scratch.hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut scratch.logits_normed,
        );
        ops::gemm_into(
            ctx,
            target.output_projection(),
            &scratch.logits_normed,
            &mut scratch.logits,
        );
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn validate_dflash_config_for_target(
    dflash_path: &str,
    target_config: &crate::config::Config,
) -> Result<DFlashConfig> {
    let config = DFlashConfig::from_file(dflash_path)?;
    config.validate_for_target(target_config)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::validate_dflash_config_for_target;
    use crate::config::Config;
    use std::path::Path;

    #[test]
    fn downloaded_dflash_config_matches_qwen3_4b() {
        let target_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/data/models/Qwen3-4B".to_string());
        let dflash_path = std::env::var("OPENINFER_DFLASH_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/data/models/Qwen3-4B-DFlash-b16".to_string());
        if !Path::new(&target_path).join("config.json").exists()
            || !Path::new(&dflash_path).join("config.json").exists()
        {
            eprintln!(
                "skipping DFlash config test; set OPENINFER_TEST_MODEL_PATH and OPENINFER_DFLASH_TEST_MODEL_PATH"
            );
            return;
        }

        let target = Config::from_file(&target_path).expect("target config");
        let dflash = validate_dflash_config_for_target(&dflash_path, &target)
            .expect("DFlash config should match target");

        assert_eq!(dflash.block_size, 16);
        assert_eq!(dflash.dflash_config.mask_token_id, 151669);
        assert_eq!(
            dflash.dflash_config.target_layer_ids,
            vec![1, 9, 17, 25, 33]
        );
    }
}
