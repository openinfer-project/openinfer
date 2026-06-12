//! Unified forward pass: prefill + decode tokens in a single forward pass.
//!
//! GEMM ops (QKV proj, O proj, MLP) process all tokens together. Attention is
//! one BatchPrefill varlen call covering both: each decode request enters the
//! plan as a qo_len=1 row over its full KV history — the same shape a 1-token
//! prefill chunk already exercises.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;

use super::config::PREFILL_ATTENTION_CTA_TILE_Q;
use super::prefill::PrefillBuffers;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::lora::{DeviceLoraTokenGroup, build_lora_token_ranges, prepare_lora_token_groups};
use openinfer_core::kv_pool::KvLayout;
use openinfer_core::ops;
use openinfer_core::ops::PrefillPagedPlan;
use openinfer_core::tensor::HiddenStates;
use openinfer_kv_cache::KvView;

impl Qwen3Model {
    /// Unified step: prefill + decode in one forward pass.
    ///
    /// Returns batched last-token logits `[vocab_size, n_prefill + n_decode]`:
    /// prefill request columns first (in request order), then decode columns.
    pub(crate) fn unified_step(
        &self,
        prefill_prompts: &[&[u32]],
        prefill_views: &[KvView],
        prefill_lora_adapters: &[Option<&str>],
        decode_tokens: &[u32],
        decode_views: &[KvView],
        decode_lora_adapters: &[Option<&str>],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
    ) -> Result<HiddenStates> {
        let num_prefill_reqs = prefill_prompts.len();
        let num_decode_reqs = decode_tokens.len();
        assert_eq!(num_prefill_reqs, prefill_views.len());
        assert_eq!(num_prefill_reqs, prefill_lora_adapters.len());
        assert_eq!(num_decode_reqs, decode_views.len());
        assert_eq!(num_decode_reqs, decode_lora_adapters.len());
        assert!(num_prefill_reqs > 0 && num_decode_reqs > 0);

        let prefill_seq_lens: Vec<usize> = prefill_prompts.iter().map(|p| p.len()).collect();
        let total_prefill: usize = prefill_seq_lens.iter().sum();
        let total_tokens = total_prefill + num_decode_reqs;
        let mut lora_ranges = build_lora_token_ranges(
            prefill_seq_lens.iter().copied(),
            prefill_lora_adapters.iter().copied(),
        );
        lora_ranges.extend(
            build_lora_token_ranges(
                std::iter::repeat_n(1, num_decode_reqs),
                decode_lora_adapters.iter().copied(),
            )
            .into_iter()
            .map(|mut range| {
                range.token_offset += total_prefill;
                range
            }),
        );
        let lora_groups = prepare_lora_token_groups(&self.ctx, &lora_ranges)?;

        // ── 1. Concatenate all tokens and get embeddings ──────────────
        let mut all_tokens: Vec<u32> = Vec::with_capacity(total_tokens);
        for prompt in prefill_prompts {
            all_tokens.extend_from_slice(prompt);
        }
        all_tokens.extend_from_slice(decode_tokens);
        let hidden = self.get_embeddings_batch(&all_tokens)?;

        // ── 2. Derive positions from views ────────────────────────────
        let prefill_start_positions: Vec<usize> = prefill_views
            .iter()
            .zip(prefill_seq_lens.iter())
            .map(|(v, &slen)| v.seq_len() - slen)
            .collect();

        let decode_positions: Vec<usize> = decode_views.iter().map(|v| v.seq_len() - 1).collect();

        // ── 3. Build metadata ─────────────────────────────────────────

        // One attention plan over prefill requests + decode rows (qo_len=1,
        // start at the decode position so the row attends its full history).
        let page_indices: Vec<Vec<i32>> = prefill_views
            .iter()
            .chain(decode_views.iter())
            .map(|v| v.page_indices().to_vec())
            .collect();
        let last_page_lens: Vec<usize> = prefill_views
            .iter()
            .chain(decode_views.iter())
            .map(openinfer_kv_cache::KvView::last_page_len)
            .collect();
        let mut start_positions = prefill_start_positions.clone();
        start_positions.extend_from_slice(&decode_positions);
        let mut seq_lens = prefill_seq_lens.clone();
        seq_lens.extend(std::iter::repeat_n(1, num_decode_reqs));
        let plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            &self.ctx,
            &page_indices,
            &last_page_lens,
            &start_positions,
            &seq_lens,
            self.local_num_attention_heads(),
            self.local_num_key_value_heads(),
            self.config.head_dim,
            PREFILL_ATTENTION_CTA_TILE_Q,
        )?;

        // ── 4. Process layers ─────────────────────────────────────────
        let hidden =
            self.unified_layers(hidden, total_tokens, &plan, &lora_groups, kv_buffer, layout)?;

        // ── 5. Extract logits ─────────────────────────────────────────
        // Last token of each prefill sequence, then every decode token —
        // one gather + one batched lm_head GEMM for the whole step.
        let mut last_indices = Vec::with_capacity(num_prefill_reqs + num_decode_reqs);
        let mut offset = 0usize;
        for &seq_len in &prefill_seq_lens {
            last_indices.push((offset + seq_len - 1) as i32);
            offset += seq_len;
        }
        for i in 0..num_decode_reqs {
            last_indices.push((total_prefill + i) as i32);
        }
        self.batch_token_logits(&hidden, &last_indices)
    }

    fn unified_layers(
        &self,
        mut hidden: HiddenStates,
        total_tokens: usize,
        plan: &PrefillPagedPlan,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
    ) -> Result<HiddenStates> {
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
            self.unified_forward_layer(
                layer_idx,
                layer,
                &mut hidden,
                &mut bufs,
                plan,
                lora_groups,
                kv_buffer,
                layout,
            )?;
        }

        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn unified_forward_layer(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        bufs: &mut PrefillBuffers,
        plan: &PrefillPagedPlan,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
    ) -> Result<()> {
        let num_heads = self.local_num_attention_heads();
        let num_kv_heads = self.local_num_key_value_heads();
        let head_dim = self.config.head_dim;

        // ── 1. RMSNorm → normed [all tokens] ─────────────────────────
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        // ── 2. QKV projections from fused qkv_proj [all tokens] ─────
        let q_dim_l = layer.attention.q_dim;
        let kv_dim_l = layer.attention.kv_dim;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            0,
            q_dim_l,
            &bufs.normed,
            &mut bufs.q_batch,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.q_proj.as_ref(),
            &bufs.normed,
            &mut bufs.q_batch,
            0,
        )?;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim_l,
            kv_dim_l,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.k_proj.as_ref(),
            &bufs.normed,
            &mut bufs.k_batch,
            0,
        )?;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim_l + kv_dim_l,
            kv_dim_l,
            &bufs.normed,
            &mut bufs.v_batch,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.v_proj.as_ref(),
            &bufs.normed,
            &mut bufs.v_batch,
            0,
        )?;

        // ── 3. Paged prefill: norm+RoPE → append K/V to paged → batch
        // attention. Positions and tile layout both come from the plan, so
        // the kernel runs the exact tile size the plan was built for.
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
            self.config.rms_norm_eps,
        )?;

        // ── 6. O projection [all tokens] ─────────────────────────────
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.o_proj.as_ref(),
            &bufs.attn_output,
            &mut bufs.o_buf,
            0,
        )?;
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // ── 7+8. Residual add + MLP RMSNorm (fused) ─────────────────
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        ops::gemm_rows_into(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            0,
            self.local_intermediate_size(),
            &bufs.normed,
            &mut bufs.gate_out,
        );
        ops::gemm_rows_into(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            self.local_intermediate_size(),
            self.local_intermediate_size(),
            &bufs.normed,
            &mut bufs.up_out,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.gate_proj.as_ref(),
            &bufs.normed,
            &mut bufs.gate_out,
            0,
        )?;
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.up_proj.as_ref(),
            &bufs.normed,
            &mut bufs.up_out,
            0,
        )?;
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.down_proj.as_ref(),
            &bufs.act_out,
            &mut bufs.o_buf,
            0,
        )?;
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // ── 9. Residual add → hidden_out ─────────────────────────────
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }
}
