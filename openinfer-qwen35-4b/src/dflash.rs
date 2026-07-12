//! Qwen3.5 DFlash draft foundation.
//!
//! The scheduler keeps this path opt-in and currently runs speculative
//! verify/commit only when a single greedy request is active.

use anyhow::{Context, Result};

use crate::dflash::config::DFlashConfig;
use crate::dflash::state::DFlashContextScratch;
use crate::weights::Qwen35Model;
use openinfer_core::ops;
use openinfer_core::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

pub(crate) mod config;
mod loading;
mod reservation;
mod scratch;
mod state;

pub(crate) use reservation::DFlashMemoryReservation;
pub(crate) use scratch::DFlashBatchScratch;
pub(crate) use state::DFlashRequestState;

pub(crate) const DFLASH_MAX_ACTIVE_REQUESTS: usize = 1;
pub(crate) const DFLASH_MAX_VERIFIED_CONTEXT_TOKENS: usize = 2048;

pub(crate) struct DFlashDraftModel {
    config: DFlashConfig,
    layers: Vec<DFlashBlock>,
    norm: DeviceVec,
    hidden_norm: DeviceVec,
    fc: DeviceMatrix,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
}

pub(crate) struct DFlashAttention {
    pub(super) qkv_proj: DeviceMatrix,
    pub(super) o_proj: DeviceMatrix,
    pub(super) q_norm: DeviceVec,
    pub(super) k_norm: DeviceVec,
}

pub(crate) struct DFlashMlp {
    pub(super) gate_up_proj: DeviceMatrix,
    pub(super) down_proj: DeviceMatrix,
}

pub(crate) struct DFlashBlock {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attention: DFlashAttention,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: DFlashMlp,
}

impl DFlashDraftModel {
    pub(crate) fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Largest sequence position the draft can cache. `validate_for_target`
    /// guarantees this is `>=` the target's, but the draft's per-step in-fill
    /// block writes `block_size` transient positions past the committed length,
    /// so the usable context is `max_position_embeddings - block_size`.
    pub(crate) fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    pub(crate) fn mask_token_id(&self) -> u32 {
        self.config.mask_token_id
    }

    pub(crate) fn target_layer_ids(&self) -> &[usize] {
        &self.config.target_layer_ids
    }

    /// Anchor-first block layout (a checkpoint property, see
    /// [`DFlashConfig::anchor_first`]) — drives both the verify span and the
    /// draft-block slice start, independently of the markov head.
    pub(crate) fn anchor_first(&self) -> bool {
        self.config.anchor_first()
    }

    pub(crate) fn verify_span(&self) -> usize {
        if self.anchor_first() {
            self.block_size() + 1
        } else {
            self.block_size()
        }
    }

    pub(crate) fn tune_gemm_algos(&self, target: &Qwen35Model) -> Result<()> {
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
            "Qwen3.5 DFlash cublasLt tuned: fc M={} K={} N=1..{}, kv M={} K={} N={}..{}",
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

    /// Allocate the lane-level batched draft scratch once, sized for the whole
    /// decode batch. The per-request `DFlashRequestState` no longer owns scratch.
    pub(crate) fn new_batch_scratch(
        &self,
        ctx: &DeviceContext,
        max_decode_batch_size: usize,
    ) -> Result<DFlashBatchScratch> {
        DFlashBatchScratch::new(ctx, &self.config, max_decode_batch_size)
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
        DFlashRequestState::new(
            ctx,
            self.layers.len(),
            kv_dim,
            self.context_feature_dim(),
            self.config.hidden_size,
            self.config.block_size,
            max_cache_len,
        )
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

    /// Batched draft forward over all active requests at once.
    ///
    /// The *dense* ops (embedding, rms_norm, q / o / gate_up / down GEMMs, silu,
    /// add, fused_add_rms_norm, logits) run ONCE over an `active_batch *
    /// block_size` batched buffer. The *varlen* ops (context projection, tail
    /// concat, k/v GEMMs, rope, KV copy, attention) still loop per request,
    /// slicing each request's `block_size` rows at offset `i * block_size` in the
    /// batched buffers — those are Step 2/3's job to batch via CUDA-kernel changes.
    ///
    /// Returns the batched logits (`active_batch * block_size` rows): request `i`
    /// owns rows `[i * block_size, (i + 1) * block_size)`.
    pub(crate) fn draft_logits_batched<'a>(
        &self,
        target: &Qwen35Model,
        states: &mut [&mut DFlashRequestState],
        current_tokens: &[u32],
        scratch: &'a mut DFlashBatchScratch,
    ) -> Result<&'a HiddenStates> {
        let ctx = target.device_ctx();
        let active_batch = states.len();
        anyhow::ensure!(
            active_batch > 0,
            "DFlash batched draft needs active requests"
        );
        anyhow::ensure!(
            states.len() == current_tokens.len(),
            "DFlash batched draft: {} states vs {} current tokens",
            states.len(),
            current_tokens.len()
        );
        let block_size = self.block_size();
        let batch_block_rows = active_batch * block_size;

        // Each request's committed context length for this round; advancing
        // `committed_len` is deferred until after the layer loop (the rope start
        // positions and KV write offsets read the pre-advance value).
        let mut context_lens = Vec::with_capacity(active_batch);
        for (i, state) in states.iter().enumerate() {
            let Some(context_len) = state.pending_context_len() else {
                anyhow::bail!(
                    "DFlash draft requested before target hidden context is available (request slot {i})"
                );
            };
            let tail_len = context_len + block_size;
            anyhow::ensure!(
                state.committed_len + tail_len <= state.max_cache_len,
                "DFlash draft cache overflow: committed={}, tail={}, max={}",
                state.committed_len,
                tail_len,
                state.max_cache_len
            );
            context_lens.push(context_len);
        }

        scratch.activate_dense(batch_block_rows);

        // Build the batched token id buffer: each request's block is
        // [current_token, mask, mask, ...].
        scratch.block_token_ids_h[..batch_block_rows].fill(self.mask_token_id());
        for (i, &current_token) in current_tokens.iter().enumerate() {
            scratch.block_token_ids_h[i * block_size] = current_token;
        }
        // token_ids_d holds `max_batch * block_size` ids; copy only the active
        // prefix. The embedding kernel reads `out.seq_len = batch_block_rows` ids
        // from the buffer start, so the active prefix is what it consumes.
        let mut token_ids_dst = scratch.token_ids_d.slice_mut(..batch_block_rows);
        ctx.stream.memcpy_htod(
            &scratch.block_token_ids_h[..batch_block_rows],
            &mut token_ids_dst,
        )?;
        target.get_embeddings_batch_into(&scratch.token_ids_d, &mut scratch.hidden)?;

        // Per-request context projection: varlen (each request's committed
        // prefix differs), persisted in the request so every layer can read it.
        for (i, state) in states.iter_mut().enumerate() {
            let context_len = context_lens[i];
            state
                .context
                .ensure_capacity(ctx, self.config.hidden_size, context_len)?;
            state.pending_context.activate_for_read();
            self.project_context_into(ctx, &state.pending_context.buffer, &mut state.context);
            state.pending_context.clear();
        }

        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        debug_assert_eq!(scratch.hidden.hidden_dim, hidden_size);
        debug_assert_eq!(scratch.q_batch.hidden_dim, q_dim);
        debug_assert_eq!(scratch.k_tail.hidden_dim, kv_dim);
        debug_assert_eq!(scratch.gate_out.hidden_dim, inter_dim);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Dense: input layernorm over the whole batch.
            ops::rms_norm_batch_into(
                ctx,
                &scratch.hidden,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                &mut scratch.normed,
            );

            // Dense: Q projection over the whole batch (per-token, no cross-request
            // mixing). Computed before the per-request loop reads `normed`, and
            // before the post-attention norm overwrites it.
            ops::gemm_rows_into(
                ctx,
                &layer.attention.qkv_proj,
                0,
                q_dim,
                &scratch.normed,
                &mut scratch.q_batch,
            );

            // Per-request varlen attention: tail concat, k/v GEMMs, rope, KV copy,
            // single-request prefill. Each request slices its `block_size` rows at
            // offset `i * block_size` of the batched `normed`/`q_batch`/`attn_output`.
            for (i, state) in states.iter_mut().enumerate() {
                let context_len = context_lens[i];
                let tail_len = context_len + block_size;
                let row_offset = i * block_size;
                scratch.ensure_tail_capacity(ctx, &self.config, tail_len)?;

                // tail_input = [context_hidden(context_len) | normed_block(block_size)].
                ops::copy_hidden_token_range_into(
                    ctx,
                    &state.context.context_hidden,
                    0,
                    &mut scratch.tail_input,
                    0,
                    context_len,
                )?;
                ops::copy_hidden_token_range_into(
                    ctx,
                    &scratch.normed,
                    row_offset,
                    &mut scratch.tail_input,
                    context_len,
                    block_size,
                )?;

                ops::gemm_rows_into(
                    ctx,
                    &layer.attention.qkv_proj,
                    q_dim,
                    kv_dim,
                    &scratch.tail_input,
                    &mut scratch.k_tail,
                );
                ops::gemm_rows_into(
                    ctx,
                    &layer.attention.qkv_proj,
                    q_dim + kv_dim,
                    kv_dim,
                    &scratch.tail_input,
                    &mut scratch.v_tail,
                );

                ops::dflash_qk_norm_rope_into(
                    ctx,
                    &mut scratch.q_batch,
                    row_offset,
                    block_size,
                    &mut scratch.k_tail,
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
                    &scratch.k_tail,
                    0,
                    &mut cache.k,
                    state.committed_len,
                    tail_len,
                )?;
                ops::copy_hidden_token_range_into(
                    ctx,
                    &scratch.v_tail,
                    0,
                    &mut cache.v,
                    state.committed_len,
                    tail_len,
                )?;
                let cache_len = state.committed_len + tail_len;
                // The reference checkpoint forwards every draft block with
                // `is_causal=false`; `layer_types` remains validated metadata.
                ops::single_prefill_nhd_noncausal_into(
                    ctx,
                    &scratch.q_batch,
                    row_offset,
                    block_size,
                    &cache.k,
                    &cache.v,
                    &mut scratch.attn_output,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    cache_len,
                )?;
            }

            // Dense: o_proj + residual + post-attention norm + MLP over the batch.
            ops::gemm_into(
                ctx,
                &layer.attention.o_proj,
                &scratch.attn_output,
                &mut scratch.o_buf,
            );
            openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
                ctx,
                &mut scratch.hidden,
                &scratch.o_buf,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut scratch.normed,
            )?;

            ops::gemm_rows_into(
                ctx,
                &layer.mlp.gate_up_proj,
                0,
                inter_dim,
                &scratch.normed,
                &mut scratch.gate_out,
            );
            ops::gemm_rows_into(
                ctx,
                &layer.mlp.gate_up_proj,
                inter_dim,
                inter_dim,
                &scratch.normed,
                &mut scratch.up_out,
            );
            ops::silu_mul_batch_into(
                ctx,
                &scratch.gate_out,
                &scratch.up_out,
                &mut scratch.act_out,
            )?;
            ops::gemm_into(
                ctx,
                &layer.mlp.down_proj,
                &scratch.act_out,
                &mut scratch.o_buf,
            );
            ops::add_batch_into(
                ctx,
                &scratch.hidden,
                &scratch.o_buf,
                &mut scratch.hidden_out,
            )?;
            std::mem::swap(&mut scratch.hidden, &mut scratch.hidden_out);
        }

        for (i, state) in states.iter_mut().enumerate() {
            state.committed_len += context_lens[i];
        }
        self.compute_logits_with_target_head_into(target, scratch);
        Ok(&scratch.logits)
    }

    fn context_feature_dim(&self) -> usize {
        self.config.hidden_size * self.target_layer_ids().len()
    }

    fn project_context_into(
        &self,
        ctx: &DeviceContext,
        context_features: &HiddenStates,
        context: &mut DFlashContextScratch,
    ) {
        ops::gemm_into(
            ctx,
            &self.fc,
            context_features,
            &mut context.context_projected,
        );
        ops::rms_norm_batch_into(
            ctx,
            &context.context_projected,
            &self.hidden_norm,
            self.config.rms_norm_eps,
            &mut context.context_hidden,
        );
    }

    fn compute_logits_with_target_head_into(
        &self,
        target: &Qwen35Model,
        scratch: &mut DFlashBatchScratch,
    ) {
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
    }
}
