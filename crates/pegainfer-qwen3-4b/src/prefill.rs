use anyhow::Result;

use super::config::PREFILL_ATTENTION_CTA_TILE_Q;
use super::weights::{Qwen3Model, TransformerBlock};
use pegainfer_core::kv_pool::{KvLayout, KvState};
use pegainfer_core::ops;
use pegainfer_core::ops::PrefillPagedPlan;
use pegainfer_core::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated scratch buffers for one prefill forward pass.
/// Created once per prefill pass, eliminating
/// per-layer `cuMemAllocAsync` overhead (~11k calls / 88ms at seq=2048).
///
/// Buffer reuse across steps (all kernels serialized on a single stream):
///   `normed`  reused for `normed2`  (steps 1-4 done before step 8)
///   `o_buf`   reused for `mlp_out`  (step 7 done before step 12)
pub(super) struct PrefillBuffers {
    /// Output ping-pong: layer writes result here; caller swaps with the incoming hidden.
    pub(super) hidden_out: HiddenStates, // hidden_dim × seq_len
    pub(super) normed: HiddenStates, // hidden_dim × seq_len (reused for normed2)
    pub(super) q_batch: HiddenStates, // q_dim × seq_len
    pub(super) k_batch: HiddenStates, // kv_dim × seq_len
    pub(super) v_batch: HiddenStates, // kv_dim × seq_len
    pub(super) o_buf: HiddenStates,  // hidden_dim × seq_len (reused for mlp_out)
    pub(super) gate_up_out: HiddenStates, // 2*inter_dim × seq_len
    pub(super) act_out: HiddenStates, // inter_dim × seq_len
    pub(super) attn_output: HiddenStates, // q_dim × seq_len
}

impl PrefillBuffers {
    pub(super) fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            gate_up_out: HiddenStates::zeros(ctx, 2 * inter_dim, seq_len)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, seq_len)?,
        })
    }
}

impl Qwen3Model {
    #[fastrace::trace(name = "get_embeddings_batch")]
    pub(super) fn get_embeddings_batch(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        let seq_len = token_ids.len();
        let hidden_dim = self.config.hidden_size;

        // Copy token IDs to GPU
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(token_ids)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let mut out = HiddenStates::zeros(&self.ctx, hidden_dim, seq_len)?;
        ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids_gpu, &mut out)?;

        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_batch_paged(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        start_pos: usize,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &pegainfer_core::kv_pool::KvLayout,
        plan: &PrefillPagedPlan,
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        let num_heads = self.local_num_attention_heads();
        let num_kv_heads = self.local_num_key_value_heads();
        let head_dim = self.config.head_dim;

        // 1. RMSNorm → bufs.normed
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        // 2. QKV projections from fused qkv_proj
        let q_dim = layer.attention.q_dim;
        let kv_dim = layer.attention.kv_dim;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            0,
            q_dim,
            &bufs.normed,
            &mut bufs.q_batch,
        );
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.v_batch,
        );

        // 3. Paged prefill: norm+RoPE → append K/V to paged → batch attention
        ops::prefill_attention_paged_into(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            kv_buffer,
            layout,
            layer_idx,
            plan,
            &mut bufs.attn_output,
            num_heads,
            num_kv_heads,
            head_dim,
            start_pos,
            self.config.rms_norm_eps,
        )?;

        // 4. O projection → bufs.o_buf (as o_batch)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // 5+6. Residual add + MLP RMSNorm (fused): hidden += o_buf; normed = rms_norm(hidden)
        ops::fused_add_rms_norm_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        // 7. MLP: fused gate+up GEMM → silu_mul → down → bufs.o_buf
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.gate_up_out,
        );
        ops::silu_mul_fused_batch_into(&self.ctx, &bufs.gate_up_out, &mut bufs.act_out);
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // 8. Residual add: attn_residual + mlp_out → bufs.hidden_out (old hidden_in, free to overwrite)
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        // Swap: hidden = layer output, bufs.hidden_out = attn_residual (free next layer)
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }

    // ── Batch prefill ──────────────────────────────────────────────────

    /// Batch prefill: process multiple prompts in a single forward pass.
    ///
    /// Compute logits for ALL positions in the hidden states.
    ///
    /// Used when `echo=true` to return prompt token log-probabilities.
    /// Applies final RMS norm + lm_head projection in a single batched GEMM.
    /// Returns `HiddenStates` with shape `[vocab_size, total_tokens]`.
    pub(crate) fn compute_all_position_logits(
        &self,
        hidden: &HiddenStates,
    ) -> Result<HiddenStates> {
        let mut normed = HiddenStates::zeros(&self.ctx, hidden.hidden_dim, hidden.seq_len)?;
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut normed,
        );
        ops::gemm(&self.ctx, self.output_projection(), &normed)
    }

    /// Concatenates all prompts' tokens, runs one GEMM per layer for the
    /// entire batch, and uses FlashInfer's multi-request causal attention.
    /// Returns per-request logits (last token of each prompt).
    ///
    /// If `echo` is true, also returns all-position logits as a
    /// `HiddenStates [vocab_size, total_tokens]` for prompt logprobs.
    pub(crate) fn batch_prefill(
        &self,
        prompts: &[&[u32]],
        kv_states: &mut [&mut KvState],
        echo: bool,
    ) -> Result<(Vec<DeviceVec>, Option<HiddenStates>)> {
        let batch_size = prompts.len();
        assert_eq!(batch_size, kv_states.len());

        let seq_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let start_positions: Vec<usize> = kv_states.iter().map(|kv| kv.seq_len()).collect();

        // Concatenate all tokens
        let all_tokens: Vec<u32> = prompts.iter().flat_map(|p| p.iter().copied()).collect();
        let hidden = self.get_embeddings_batch(&all_tokens)?;

        // Allocate pages and advance for each request
        for (i, kv) in kv_states.iter_mut().enumerate() {
            kv.ensure_capacity(start_positions[i] + seq_lens[i])?;
            kv.advance(seq_lens[i]);
        }

        // Build batch plan (all descs must reflect post-advance state)
        let descs: Vec<_> = kv_states.iter().map(|kv| kv.desc()).collect();
        let plan = PrefillPagedPlan::new_batch_with_cta_tile_q(
            &self.ctx,
            &descs,
            &start_positions,
            &seq_lens,
            self.local_num_attention_heads(),
            self.local_num_key_value_heads(),
            self.config.head_dim,
            PREFILL_ATTENTION_CTA_TILE_Q,
        )?;

        // Forward through all layers
        let kv_buffer = kv_states[0].buffer();
        let layout = *kv_states[0].layout();
        let hidden = self.process_all_layers_batch_multi(hidden, &layout, kv_buffer, &plan)?;

        // All-position logits for echo (before we extract last-token logits)
        let all_logits = if echo {
            Some(self.compute_all_position_logits(&hidden)?)
        } else {
            None
        };

        // Extract per-request last-token logits
        let mut logits_vec = Vec::with_capacity(batch_size);
        let mut offset = 0;
        for &seq_len in &seq_lens {
            let last_idx = offset + seq_len - 1;
            let last_hidden = ops::extract_vec(&self.ctx, &hidden, last_idx)?;
            let normed = ops::rms_norm(
                &self.ctx,
                &last_hidden,
                &self.norm,
                self.config.rms_norm_eps,
            )?;
            let logits = ops::linear(&self.ctx, &normed, self.output_projection())?;
            logits_vec.push(logits);
            offset += seq_len;
        }

        Ok((logits_vec, all_logits))
    }

    fn process_all_layers_batch_multi(
        &self,
        mut hidden: HiddenStates,
        layout: &KvLayout,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        plan: &PrefillPagedPlan,
    ) -> Result<HiddenStates> {
        let total_tokens = hidden.seq_len;
        let inter_dim = self.local_intermediate_size();
        let q_dim = self.local_q_dim();
        let kv_dim = self.local_kv_dim();

        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            total_tokens,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch_paged(
                layer_idx,
                layer,
                &mut hidden,
                0, // start_pos unused for multi-request (plan has per-token positions)
                kv_buffer,
                layout,
                plan,
                &mut bufs,
            )?;
        }

        Ok(hidden)
    }
}
