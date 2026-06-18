use anyhow::Result;
use cudarc::driver::CudaSlice;
use openinfer_core::ops;
use openinfer_core::tensor::HiddenStates;

use crate::weights::{DFlashDraftModel, DFlashLayer};

pub struct DFlashTargetHidden<'a> {
    /// HF reference layout: `[seq_len, target_layer_count * hidden_size]`.
    pub concatenated: &'a HiddenStates,
}

pub struct DFlashDraftCache {
    pub(crate) q_len: usize,
    pub(crate) state: DFlashDraftState,
    pub(crate) step: DFlashStepContext,
    pub(crate) scratch: ForwardBuffers,
}

pub(crate) struct DFlashDraftState {
    pub(crate) max_seq_len: usize,
    pub(crate) seq_len: usize,
    pub(crate) layers: Vec<DFlashLayerPastKv>,
}

pub(crate) struct DFlashStepContext {
    pub(crate) max_len: usize,
    pub(crate) len: usize,
    pub(crate) valid: bool,
    pub(crate) layers: Vec<DFlashLayerStepContext>,
}

pub(crate) struct DFlashLayerStepContext {
    pub(crate) k_ctx: HiddenStates,
    pub(crate) v_ctx: HiddenStates,
}

pub(crate) struct DFlashLayerPastKv {
    pub(crate) k_past: HiddenStates,
    pub(crate) v_past: HiddenStates,
}

pub(crate) struct ForwardBuffers {
    pub(crate) hidden_out: HiddenStates,
    pub(crate) target_projected: HiddenStates,
    pub(crate) target_normed: HiddenStates,
    pub(crate) normed: HiddenStates,
    pub(crate) q: HiddenStates,
    pub(crate) q_ctx_scratch: HiddenStates,
    pub(crate) k_ctx: HiddenStates,
    pub(crate) k_noise: HiddenStates,
    pub(crate) v_ctx: HiddenStates,
    pub(crate) v_noise: HiddenStates,
    pub(crate) k_all: HiddenStates,
    pub(crate) v_all: HiddenStates,
    pub(crate) attn_out: HiddenStates,
    pub(crate) o_buf: HiddenStates,
    pub(crate) gate_up: HiddenStates,
    pub(crate) act_out: HiddenStates,
    pub(crate) positions_q: CudaSlice<i32>,
    pub(crate) positions_ctx: CudaSlice<i32>,
}

impl DFlashDraftModel {
    pub fn create_draft_cache(
        &self,
        q_len: usize,
        max_step_context_len: usize,
        max_seq_len: usize,
    ) -> Result<DFlashDraftCache> {
        anyhow::ensure!(q_len > 0, "DFlash scratch requires q_len greater than zero");
        anyhow::ensure!(
            max_step_context_len > 0,
            "DFlash cache requires max_step_context_len greater than zero"
        );
        anyhow::ensure!(
            max_seq_len >= max_step_context_len + q_len,
            "DFlash cache max_seq_len {} must fit at least one step: context {} + q_len {}",
            max_seq_len,
            max_step_context_len,
            q_len
        );
        Ok(DFlashDraftCache {
            q_len,
            state: DFlashDraftState::new(self, max_seq_len)?,
            step: DFlashStepContext::new(self, max_step_context_len)?,
            scratch: ForwardBuffers::new(self, q_len, max_step_context_len)?,
        })
    }

    pub fn forward(
        &self,
        noise_embedding: &HiddenStates,
        target_hidden: DFlashTargetHidden<'_>,
        position_ids: &[i32],
    ) -> Result<HiddenStates> {
        let (q_len, ctx_len) =
            self.validate_forward_inputs(noise_embedding, &target_hidden, position_ids)?;
        let mut bufs = ForwardBuffers::new(self, q_len, ctx_len)?;
        self.project_target_hidden(target_hidden, &mut bufs)?;
        self.run_forward(noise_embedding, ctx_len, position_ids, &mut bufs)?;
        Ok(bufs.normed)
    }

    pub fn forward_with_cache<'a>(
        &self,
        noise_embedding: &HiddenStates,
        target_hidden: DFlashTargetHidden<'_>,
        position_ids: &[i32],
        cache: &'a mut DFlashDraftCache,
    ) -> Result<&'a HiddenStates> {
        let (q_len, ctx_len) =
            self.validate_forward_inputs(noise_embedding, &target_hidden, position_ids)?;
        anyhow::ensure!(
            cache.q_len == q_len && cache.step.max_len >= ctx_len,
            "DFlash cache shape mismatch: cache q_len={}, max_step_context_len={} but input q_len={}, ctx_len={}",
            cache.q_len,
            cache.step.max_len,
            q_len,
            ctx_len
        );
        cache.reset();
        self.prepare_step_context(target_hidden, position_ids, cache)?;
        self.run_forward(noise_embedding, ctx_len, position_ids, &mut cache.scratch)?;
        cache.step.valid = false;
        Ok(&cache.scratch.normed)
    }

    pub fn prepare_step_context(
        &self,
        target_hidden: DFlashTargetHidden<'_>,
        position_ids: &[i32],
        cache: &mut DFlashDraftCache,
    ) -> Result<()> {
        let config = &self.config;
        let ctx_len = target_hidden.concatenated.seq_len;
        anyhow::ensure!(
            ctx_len <= cache.step.max_len,
            "DFlash step context length {} exceeds cache capacity {}",
            ctx_len,
            cache.step.max_len
        );
        anyhow::ensure!(
            cache.state.seq_len + ctx_len + cache.q_len <= cache.state.max_seq_len,
            "DFlash draft cache would exceed capacity: past {} + ctx {} + q {} > {}",
            cache.state.seq_len,
            ctx_len,
            cache.q_len,
            cache.state.max_seq_len
        );
        anyhow::ensure!(
            ctx_len > 0,
            "DFlash step context must contain at least one token"
        );
        anyhow::ensure!(
            position_ids.len() >= ctx_len,
            "position_ids len {} < ctx_len {}",
            position_ids.len(),
            ctx_len
        );
        anyhow::ensure!(
            target_hidden.concatenated.hidden_dim
                == config.target_layer_count() * config.hidden_size,
            "target_hidden hidden_dim {} != {}",
            target_hidden.concatenated.hidden_dim,
            config.target_layer_count() * config.hidden_size
        );
        set_step_context_len(&mut cache.scratch, &mut cache.step.layers, ctx_len);
        let mut positions_ctx = cache.scratch.positions_ctx.slice_mut(..ctx_len);
        self.ctx
            .stream
            .memcpy_htod(&position_ids[..ctx_len], &mut positions_ctx)?;

        ops::gemm_into_checked(
            &self.ctx,
            &self.fc,
            target_hidden.concatenated,
            &mut cache.scratch.target_projected,
        )?;
        ops::rms_norm_batch_into(
            &self.ctx,
            &cache.scratch.target_projected,
            &self.hidden_norm,
            config.rms_norm_eps,
            &mut cache.scratch.target_normed,
        );
        for (layer, cached) in self.layers.iter().zip(cache.step.layers.iter_mut()) {
            ops::gemm_into_checked(
                &self.ctx,
                &layer.attention.k_proj,
                &cache.scratch.target_normed,
                &mut cached.k_ctx,
            )?;
            ops::gemm_into_checked(
                &self.ctx,
                &layer.attention.v_proj,
                &cache.scratch.target_normed,
                &mut cached.v_ctx,
            )?;
            ops::qk_norm_rope_batch_decode_into(
                &self.ctx,
                &mut cache.scratch.q_ctx_scratch,
                &mut cached.k_ctx,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache,
                &self.sin_cache,
                &cache.scratch.positions_ctx,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
            );
        }
        cache.step.len = ctx_len;
        cache.step.valid = true;
        Ok(())
    }

    pub fn forward_with_draft_cache<'a>(
        &self,
        noise_embedding: &HiddenStates,
        position_ids: &[i32],
        cache: &'a mut DFlashDraftCache,
    ) -> Result<&'a HiddenStates> {
        anyhow::ensure!(cache.step.valid, "DFlash step context is not prepared");
        anyhow::ensure!(
            noise_embedding.hidden_dim == self.config.hidden_size,
            "noise_embedding hidden_dim {} != {}",
            noise_embedding.hidden_dim,
            self.config.hidden_size
        );
        anyhow::ensure!(
            noise_embedding.seq_len == cache.q_len,
            "noise_embedding q_len {} != scratch q_len {}",
            noise_embedding.seq_len,
            cache.q_len
        );
        anyhow::ensure!(
            position_ids.len() == cache.step.len + cache.q_len,
            "position_ids len {} != step_context_len + q_len {}",
            position_ids.len(),
            cache.step.len + cache.q_len
        );
        anyhow::ensure!(
            cache.state.seq_len + cache.step.len + cache.q_len <= cache.state.max_seq_len,
            "DFlash draft cache would exceed capacity: past {} + ctx {} + q {} > {}",
            cache.state.seq_len,
            cache.step.len,
            cache.q_len,
            cache.state.max_seq_len
        );
        let past_len = cache.state.seq_len;
        self.run_forward_with_draft_cache(noise_embedding, past_len, position_ids, cache)?;
        cache.step.valid = false;
        Ok(&cache.scratch.normed)
    }

    pub(crate) fn validate_forward_inputs(
        &self,
        noise_embedding: &HiddenStates,
        target_hidden: &DFlashTargetHidden<'_>,
        position_ids: &[i32],
    ) -> Result<(usize, usize)> {
        let config = &self.config;
        anyhow::ensure!(
            noise_embedding.hidden_dim == config.hidden_size,
            "noise_embedding hidden_dim {} != {}",
            noise_embedding.hidden_dim,
            config.hidden_size
        );
        let ctx_len = target_hidden.concatenated.seq_len;
        let q_len = noise_embedding.seq_len;
        anyhow::ensure!(
            ctx_len > 0,
            "DFlash forward requires at least one target-hidden token"
        );
        anyhow::ensure!(
            q_len > 0,
            "DFlash forward requires at least one noise token"
        );
        anyhow::ensure!(
            target_hidden.concatenated.hidden_dim
                == config.target_layer_count() * config.hidden_size,
            "target_hidden hidden_dim {} != {}",
            target_hidden.concatenated.hidden_dim,
            config.target_layer_count() * config.hidden_size
        );
        anyhow::ensure!(
            position_ids.len() == ctx_len + q_len,
            "position_ids len {} != ctx_len + q_len {}",
            position_ids.len(),
            ctx_len + q_len
        );
        Ok((q_len, ctx_len))
    }

    fn project_target_hidden(
        &self,
        target_hidden: DFlashTargetHidden<'_>,
        bufs: &mut ForwardBuffers,
    ) -> Result<()> {
        let config = &self.config;
        ops::gemm_into_checked(
            &self.ctx,
            &self.fc,
            target_hidden.concatenated,
            &mut bufs.target_projected,
        )?;
        ops::rms_norm_batch_into(
            &self.ctx,
            &bufs.target_projected,
            &self.hidden_norm,
            config.rms_norm_eps,
            &mut bufs.target_normed,
        );
        Ok(())
    }

    pub(crate) fn run_forward(
        &self,
        noise_embedding: &HiddenStates,
        ctx_len: usize,
        position_ids: &[i32],
        bufs: &mut ForwardBuffers,
    ) -> Result<()> {
        let q_len = noise_embedding.seq_len;
        let mut positions_q = bufs.positions_q.slice_mut(..q_len);
        self.ctx
            .stream
            .memcpy_htod(&position_ids[ctx_len..], &mut positions_q)?;
        let mut positions_ctx = bufs.positions_ctx.slice_mut(..ctx_len);
        self.ctx
            .stream
            .memcpy_htod(&position_ids[..ctx_len], &mut positions_ctx)?;

        let mut hidden = clone_hidden(&self.ctx, noise_embedding)?;
        for layer in &self.layers {
            self.forward_layer(layer, &mut hidden, bufs)?;
        }
        ops::rms_norm_batch_into(
            &self.ctx,
            &hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );
        Ok(())
    }

    fn run_forward_with_draft_cache(
        &self,
        noise_embedding: &HiddenStates,
        past_len: usize,
        position_ids: &[i32],
        cache: &mut DFlashDraftCache,
    ) -> Result<()> {
        let ctx_len = cache.step.len;
        let q_len = noise_embedding.seq_len;
        let total_len = past_len + ctx_len + q_len;
        let mut positions_q = cache.scratch.positions_q.slice_mut(..q_len);
        self.ctx
            .stream
            .memcpy_htod(&position_ids[ctx_len..], &mut positions_q)?;

        let mut hidden = clone_hidden(&self.ctx, noise_embedding)?;
        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            self.forward_layer_with_draft_cache(
                layer,
                past_len,
                total_len,
                &cache.step.layers[layer_idx],
                &mut cache.state.layers[layer_idx],
                &mut hidden,
                &mut cache.scratch,
            )?;
        }
        ops::rms_norm_batch_into(
            &self.ctx,
            &hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut cache.scratch.normed,
        );
        cache.state.seq_len = total_len;
        set_past_seq_len(&mut cache.state.layers, total_len);
        Ok(())
    }

    pub(crate) fn forward_layer(
        &self,
        layer: &DFlashLayer,
        hidden: &mut HiddenStates,
        bufs: &mut ForwardBuffers,
    ) -> Result<()> {
        let config = &self.config;
        let q_len = hidden.seq_len;
        let ctx_len = bufs.target_normed.seq_len;

        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            config.rms_norm_eps,
            &mut bufs.normed,
        );

        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q,
        )?;
        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k_noise,
        )?;
        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v_noise,
        )?;

        ops::qk_norm_rope_batch_decode_into(
            &self.ctx,
            &mut bufs.q,
            &mut bufs.k_noise,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions_q,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
        );
        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.target_normed,
            &mut bufs.k_ctx,
        )?;
        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.target_normed,
            &mut bufs.v_ctx,
        )?;
        // Normalize and rotate context K with its own positions. Q has already
        // been prepared above; q_ctx_scratch only reuses the shared Q/K kernel.
        ops::qk_norm_rope_batch_decode_into(
            &self.ctx,
            &mut bufs.q_ctx_scratch,
            &mut bufs.k_ctx,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions_ctx,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
        );
        concat_kv(
            &self.ctx,
            &bufs.k_ctx,
            &bufs.k_noise,
            ctx_len,
            q_len,
            &mut bufs.k_all,
        )?;
        concat_kv(
            &self.ctx,
            &bufs.v_ctx,
            &bufs.v_noise,
            ctx_len,
            q_len,
            &mut bufs.v_all,
        )?;

        ops::single_prefill_nhd_noncausal_into(
            &self.ctx,
            &bufs.q,
            &bufs.k_all,
            &bufs.v_all,
            &mut bufs.attn_out,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )?;
        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.o_buf,
        )?;
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        ops::gemm_into_checked(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.gate_up,
        )?;
        ops::silu_mul_fused_batch_into(&self.ctx, &bufs.gate_up, &mut bufs.act_out);
        ops::gemm_into_checked(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        )?;
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);
        Ok(())
    }

    fn forward_layer_with_draft_cache(
        &self,
        layer: &DFlashLayer,
        past_len: usize,
        total_len: usize,
        step_context: &DFlashLayerStepContext,
        past: &mut DFlashLayerPastKv,
        hidden: &mut HiddenStates,
        bufs: &mut ForwardBuffers,
    ) -> Result<()> {
        let config = &self.config;
        let q_len = hidden.seq_len;
        let ctx_len = bufs.target_normed.seq_len;

        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            config.rms_norm_eps,
            &mut bufs.normed,
        );

        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q,
        )?;
        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k_noise,
        )?;
        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v_noise,
        )?;

        ops::qk_norm_rope_batch_decode_into(
            &self.ctx,
            &mut bufs.q,
            &mut bufs.k_noise,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions_q,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
        );

        append_kv(
            &self.ctx,
            &step_context.k_ctx,
            &bufs.k_noise,
            past_len,
            ctx_len,
            q_len,
            &mut past.k_past,
        )?;
        append_kv(
            &self.ctx,
            &step_context.v_ctx,
            &bufs.v_noise,
            past_len,
            ctx_len,
            q_len,
            &mut past.v_past,
        )?;
        past.k_past.seq_len = total_len;
        past.v_past.seq_len = total_len;

        ops::single_prefill_nhd_noncausal_into(
            &self.ctx,
            &bufs.q,
            &past.k_past,
            &past.v_past,
            &mut bufs.attn_out,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )?;
        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.o_buf,
        )?;
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        ops::gemm_into_checked(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.gate_up,
        )?;
        ops::silu_mul_fused_batch_into(&self.ctx, &bufs.gate_up, &mut bufs.act_out);
        ops::gemm_into_checked(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        )?;
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);
        Ok(())
    }
}

impl DFlashDraftCache {
    pub fn seq_len(&self) -> usize {
        self.state.seq_len
    }

    pub fn reset(&mut self) {
        self.state.seq_len = 0;
        self.step.len = 0;
        self.step.valid = false;
        set_past_seq_len(&mut self.state.layers, 0);
    }

    pub fn crop(&mut self, seq_len: usize) -> Result<()> {
        anyhow::ensure!(
            seq_len <= self.state.seq_len,
            "cannot crop DFlash draft cache from {} to larger length {}",
            self.state.seq_len,
            seq_len
        );
        self.state.seq_len = seq_len;
        self.step.valid = false;
        self.step.len = 0;
        set_past_seq_len(&mut self.state.layers, seq_len);
        Ok(())
    }
}

impl DFlashDraftState {
    fn new(model: &DFlashDraftModel, max_seq_len: usize) -> Result<Self> {
        let config = &model.config;
        let kv_dim = config.kv_dim();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(DFlashLayerPastKv {
                k_past: HiddenStates::zeros(&model.ctx, kv_dim, max_seq_len)?,
                v_past: HiddenStates::zeros(&model.ctx, kv_dim, max_seq_len)?,
            });
        }
        Ok(Self {
            max_seq_len,
            seq_len: 0,
            layers,
        })
    }
}

impl DFlashStepContext {
    fn new(model: &DFlashDraftModel, max_len: usize) -> Result<Self> {
        let config = &model.config;
        let kv_dim = config.kv_dim();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(DFlashLayerStepContext {
                k_ctx: HiddenStates::zeros(&model.ctx, kv_dim, max_len)?,
                v_ctx: HiddenStates::zeros(&model.ctx, kv_dim, max_len)?,
            });
        }
        Ok(Self {
            max_len,
            len: 0,
            valid: false,
            layers,
        })
    }
}

impl ForwardBuffers {
    pub(crate) fn new(model: &DFlashDraftModel, q_len: usize, ctx_len: usize) -> Result<Self> {
        let config = &model.config;
        let ctx = &model.ctx;
        let hidden = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden, q_len)?,
            target_projected: HiddenStates::zeros(ctx, hidden, ctx_len)?,
            target_normed: HiddenStates::zeros(ctx, hidden, ctx_len)?,
            normed: HiddenStates::zeros(ctx, hidden, q_len)?,
            q: HiddenStates::zeros(ctx, q_dim, q_len)?,
            q_ctx_scratch: HiddenStates::zeros(ctx, q_dim, ctx_len)?,
            k_ctx: HiddenStates::zeros(ctx, kv_dim, ctx_len)?,
            k_noise: HiddenStates::zeros(ctx, kv_dim, q_len)?,
            v_ctx: HiddenStates::zeros(ctx, kv_dim, ctx_len)?,
            v_noise: HiddenStates::zeros(ctx, kv_dim, q_len)?,
            k_all: HiddenStates::zeros(ctx, kv_dim, ctx_len + q_len)?,
            v_all: HiddenStates::zeros(ctx, kv_dim, ctx_len + q_len)?,
            attn_out: HiddenStates::zeros(ctx, q_dim, q_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden, q_len)?,
            gate_up: HiddenStates::zeros(ctx, 2 * config.intermediate_size, q_len)?,
            act_out: HiddenStates::zeros(ctx, config.intermediate_size, q_len)?,
            positions_q: ctx.stream.alloc_zeros(q_len)?,
            positions_ctx: ctx.stream.alloc_zeros(ctx_len)?,
        })
    }
}

pub(crate) fn clone_hidden(
    ctx: &openinfer_core::tensor::DeviceContext,
    input: &HiddenStates,
) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;
    let src = input.data.slice(..input.hidden_dim * input.seq_len);
    let mut dst = out.data.slice_mut(..input.hidden_dim * input.seq_len);
    ctx.stream.memcpy_dtod(&src, &mut dst)?;
    Ok(out)
}

pub(crate) fn concat_kv(
    ctx: &openinfer_core::tensor::DeviceContext,
    ctx_part: &HiddenStates,
    noise_part: &HiddenStates,
    ctx_len: usize,
    q_len: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    debug_assert_eq!(ctx_part.seq_len, ctx_len);
    debug_assert_eq!(noise_part.seq_len, q_len);
    debug_assert_eq!(ctx_part.hidden_dim, noise_part.hidden_dim);
    debug_assert_eq!(out.hidden_dim, ctx_part.hidden_dim);
    debug_assert_eq!(out.seq_len, ctx_len + q_len);
    let ctx_src = ctx_part.data.slice(..ctx_part.hidden_dim * ctx_len);
    let mut ctx_dst = out.data.slice_mut(..ctx_part.hidden_dim * ctx_len);
    ctx.stream.memcpy_dtod(&ctx_src, &mut ctx_dst)?;
    let noise_src = noise_part.data.slice(..noise_part.hidden_dim * q_len);
    let offset = ctx_part.hidden_dim * ctx_len;
    let mut noise_dst = out
        .data
        .slice_mut(offset..offset + noise_part.hidden_dim * q_len);
    ctx.stream.memcpy_dtod(&noise_src, &mut noise_dst)?;
    Ok(())
}

pub(crate) fn append_kv(
    ctx: &openinfer_core::tensor::DeviceContext,
    ctx_part: &HiddenStates,
    noise_part: &HiddenStates,
    past_len: usize,
    ctx_len: usize,
    q_len: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    debug_assert_eq!(ctx_part.seq_len, ctx_len);
    debug_assert_eq!(noise_part.seq_len, q_len);
    debug_assert_eq!(ctx_part.hidden_dim, noise_part.hidden_dim);
    debug_assert_eq!(out.hidden_dim, ctx_part.hidden_dim);
    debug_assert!(past_len + ctx_len + q_len <= out.data.len());
    let ctx_src = ctx_part.data.slice(..ctx_part.hidden_dim * ctx_len);
    let ctx_offset = ctx_part.hidden_dim * past_len;
    let mut ctx_dst = out
        .data
        .slice_mut(ctx_offset..ctx_offset + ctx_part.hidden_dim * ctx_len);
    ctx.stream.memcpy_dtod(&ctx_src, &mut ctx_dst)?;
    let noise_src = noise_part.data.slice(..noise_part.hidden_dim * q_len);
    let noise_offset = ctx_part.hidden_dim * (past_len + ctx_len);
    let mut noise_dst = out
        .data
        .slice_mut(noise_offset..noise_offset + noise_part.hidden_dim * q_len);
    ctx.stream.memcpy_dtod(&noise_src, &mut noise_dst)?;
    Ok(())
}

pub(crate) fn set_step_context_len(
    bufs: &mut ForwardBuffers,
    layers: &mut [DFlashLayerStepContext],
    ctx_len: usize,
) {
    bufs.target_projected.seq_len = ctx_len;
    bufs.target_normed.seq_len = ctx_len;
    bufs.q_ctx_scratch.seq_len = ctx_len;
    bufs.k_ctx.seq_len = ctx_len;
    bufs.v_ctx.seq_len = ctx_len;
    for layer in layers {
        layer.k_ctx.seq_len = ctx_len;
        layer.v_ctx.seq_len = ctx_len;
    }
}

pub(crate) fn set_past_seq_len(layers: &mut [DFlashLayerPastKv], seq_len: usize) {
    for layer in layers {
        layer.k_past.seq_len = seq_len;
        layer.v_past.seq_len = seq_len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use std::path::Path;

    const LOCAL_DFLASH: &str = "/home/hezhaozhao/models/Qwen3-4B-DFlash-b16";

    #[test]
    fn draft_forward_smoke_local_model() {
        let path = Path::new(LOCAL_DFLASH);
        if !path.exists() {
            eprintln!("skipping: {LOCAL_DFLASH} does not exist");
            return;
        }

        let model = DFlashDraftModel::load(path, 0).expect("load model");
        let config = model.config();
        let ctx_len = 1;
        let q_len = 1;
        let noise_host = vec![bf16::ZERO; config.hidden_size * q_len];
        let target_host =
            vec![bf16::ZERO; config.hidden_size * config.target_layer_count() * ctx_len];
        let noise_embedding = HiddenStates {
            data: model.ctx.stream.clone_htod(&noise_host).expect("noise h2d"),
            hidden_dim: config.hidden_size,
            seq_len: q_len,
        };
        let target_hidden = HiddenStates {
            data: model
                .ctx
                .stream
                .clone_htod(&target_host)
                .expect("target h2d"),
            hidden_dim: config.hidden_size * config.target_layer_count(),
            seq_len: ctx_len,
        };

        let out = model
            .forward(
                &noise_embedding,
                DFlashTargetHidden {
                    concatenated: &target_hidden,
                },
                &[0, 1],
            )
            .expect("forward");
        model.ctx.sync().expect("sync");
        assert_eq!(out.hidden_dim, config.hidden_size);
        assert_eq!(out.seq_len, q_len);
    }
}
