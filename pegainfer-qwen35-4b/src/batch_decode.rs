//! Batched decode for Qwen3.5: N requests, 1 token each, shared full-attn kernels
//! and per-request recurrent-state updates for linear attention.

use anyhow::Result;
use cudarc::driver::{DevicePtr, DevicePtrMut};

use super::batch_decode_graph::{BATCH_BUCKETS, BatchDecodeGraphState, bucket_for};
use super::decode_buffers::BatchDecodeBuffers35;
use super::recurrent_state::RecurrentState;
use super::weights::{FullAttentionLayer, LayerKind, LinearAttentionLayer, Qwen35Model};
use crate::ops;
use pegainfer_core::kv_pool::{KvLayout, KvState};

impl Qwen35Model {
    pub(crate) fn select_tokens_batch_varied(
        &self,
        bufs: &mut BatchDecodeBuffers35,
        params: &[&pegainfer_core::sampler::SamplingParams],
        rng: &mut rand::rngs::StdRng,
    ) -> Result<Vec<u32>> {
        let random_vals: Vec<f32> = params.iter().map(|_| rand::RngExt::random(rng)).collect();
        ops::select_batch_tokens_into(
            &self.ctx,
            &bufs.logits,
            params,
            &random_vals,
            &mut bufs.sample_row_indices,
            &mut bufs.sample_probs,
            &mut bufs.sample_top1_value,
            &mut bufs.sample_row_states,
            &mut bufs.sample_valid,
            &mut bufs.sample_out,
        )
    }

    fn batch_decode_full_attention(
        &self,
        attn: &FullAttentionLayer,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        layer_idx: usize,
        bs: usize,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        ops::gemm_into(&self.ctx, &attn.q_proj, &bufs.normed, &mut bufs.q_full);
        ops::gemm_into(&self.ctx, &attn.k_proj, &bufs.normed, &mut bufs.k_attn);
        ops::gemm_into(&self.ctx, &attn.v_proj, &bufs.normed, &mut bufs.v_attn);

        ops::qk_norm_partial_rope_batched_decode_hd256_into(
            &self.ctx,
            &bufs.q_full,
            &mut bufs.q_attn,
            &mut bufs.k_attn,
            &attn.q_norm,
            &attn.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions_d,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.rotary_dim,
            eps,
        );

        ops::paged_attention_batch_decode_hd256_into(
            &self.ctx,
            &bufs.q_attn,
            &bufs.k_attn,
            &bufs.v_attn,
            kv_buffer,
            layout,
            layer_idx,
            &bufs.page_indices_d,
            &bufs.page_indptr_d,
            &bufs.last_page_len_d,
            &bufs.positions_d,
            &bufs.request_indices_d,
            &bufs.kv_tile_indices_d,
            &bufs.kv_chunk_size_d,
            &mut bufs.attn_out_full,
            self.config.num_attention_heads,
            bs,
        )?;

        unsafe {
            let (qf_ptr, _gqf) = bufs.q_full.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = bufs.attn_out_full.data.device_ptr_mut(&self.ctx.stream);
            crate::ffi::attention_gate_batch_hd256_cuda(
                qf_ptr as *const crate::ffi::Half,
                out_ptr as *mut crate::ffi::Half,
                self.config.num_attention_heads as i32,
                bs as i32,
                self.ctx.stream.cu_stream(),
            );
        }

        ops::gemm_into(
            &self.ctx,
            &attn.o_proj,
            &bufs.attn_out_full,
            &mut bufs.attn_results,
        );
        Ok(())
    }

    // =========================================================================
    // CUDA Graph batch decode
    // =========================================================================

    /// CUDA Graph batch decode step.
    ///
    /// Pads the batch to the nearest bucket size, fills padding positions with
    /// dummy KV metadata (pointing to the reserved padding page), then
    /// captures or replays a per-bucket CUDA Graph for the full kernel sequence.
    ///
    /// Recurrent state is owned by `graph_state.slot_states`: the caller must
    /// pack active requests into positions 0..batch_size before calling. After
    /// the call, `slot_states[i]` contains the updated state for request i.
    pub(crate) fn batch_decode_graph(
        &self,
        token_ids: &[u32],
        kv_states: &mut [&mut KvState],
        graph_state: &mut BatchDecodeGraphState,
    ) -> Result<()> {
        let bs = token_ids.len();
        anyhow::ensure!(bs > 0, "batch_decode_graph requires at least one request");
        anyhow::ensure!(bs == kv_states.len(), "token_ids / kv_states len mismatch");
        anyhow::ensure!(
            bs <= super::batch_decode_graph::MAX_BATCH,
            "batch size {bs} exceeds MAX_BATCH={}",
            super::batch_decode_graph::MAX_BATCH
        );

        let padded_bs = bucket_for(bs);

        // Advance KV states and collect positions. Slot seq_len is incremented
        // on the CPU outside the graph so it never appears inside the capture.
        let mut positions = Vec::with_capacity(bs);
        for (i, kv) in kv_states.iter_mut().enumerate() {
            let pos = kv.seq_len();
            self.ensure_rope_cache_covers(pos + 1)?;
            kv.ensure_capacity(pos + 1)?;
            kv.advance(1);
            graph_state.slot_states[i].seq_len += 1;
            positions.push(pos as i32);
        }

        graph_state.buffers.set_batch_size(padded_bs);

        // H2D: token_ids and positions — zero-padded to bucket size.
        let mut token_ids_padded = token_ids.to_vec();
        token_ids_padded.resize(padded_bs, 0);
        positions.resize(padded_bs, 0);
        self.ctx
            .stream
            .memcpy_htod(&token_ids_padded, &mut graph_state.buffers.token_ids_d)?;
        self.ctx
            .stream
            .memcpy_htod(&positions, &mut graph_state.buffers.positions_d)?;

        // H2D: paged KV metadata with padding slots pointing to padding_page_id.
        let kv_refs: Vec<&KvState> = kv_states.iter().map(|s| &**s).collect();
        graph_state
            .buffers
            .sync_paged_meta(&self.ctx, &kv_refs, padded_bs)?;

        let kv_buffer = kv_states[0].buffer();
        let layout = *kv_states[0].layout();
        let bucket_idx = BATCH_BUCKETS.iter().position(|&b| b == padded_bs).unwrap();

        // Take graphs out of graph_state to avoid split-borrow in the closure.
        let mut graphs = std::mem::take(&mut graph_state.graphs);
        let result = graphs[bucket_idx].run_or_capture(&self.ctx, || {
            self.batch_decode_kernels_graph(
                kv_buffer,
                &layout,
                padded_bs,
                &mut graph_state.slot_states,
                &mut graph_state.buffers,
            )
        });
        graph_state.graphs = graphs;
        result
    }

    fn batch_decode_kernels_graph(
        &self,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        padded_bs: usize,
        slot_states: &mut [RecurrentState],
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_d,
            &mut bufs.hidden,
        )?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        for layer in &self.layers {
            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden,
                &layer.input_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            match &layer.attn {
                LayerKind::FullAttention(attn) => {
                    self.batch_decode_full_attention(
                        attn, kv_buffer, layout, full_idx, padded_bs, bufs,
                    )?;
                    full_idx += 1;
                }
                LayerKind::LinearAttention(attn) => {
                    self.batch_decode_linear_attention_slots(
                        attn,
                        slot_states,
                        linear_idx,
                        padded_bs,
                        bufs,
                    )?;
                    linear_idx += 1;
                }
            }

            ops::add_batch_into(
                &self.ctx,
                &bufs.hidden,
                &bufs.attn_results,
                &mut bufs.hidden_mid,
            )?;

            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            ops::gemm_into(
                &self.ctx,
                &layer.mlp.gate_proj,
                &bufs.normed,
                &mut bufs.gate_out,
            );
            ops::gemm_into(
                &self.ctx,
                &layer.mlp.up_proj,
                &bufs.normed,
                &mut bufs.up_out,
            );
            ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
            ops::gemm_into(
                &self.ctx,
                &layer.mlp.down_proj,
                &bufs.act_out,
                &mut bufs.mlp_out,
            );

            ops::add_batch_into(&self.ctx, &bufs.hidden_mid, &bufs.mlp_out, &mut bufs.hidden)?;
        }

        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden,
            &self.norm,
            eps,
            &mut bufs.normed,
        )?;
        ops::gemm_into(
            &self.ctx,
            &self.embed_tokens,
            &bufs.normed,
            &mut bufs.logits,
        );
        debug_assert_eq!(bufs.logits.seq_len, padded_bs);

        Ok(())
    }

    /// Linear attention decode over slot-indexed recurrent state.
    ///
    /// Iterates 0..`padded_bs`. Real requests are in 0..real_bs; padding slots
    /// (real_bs..padded_bs) run but their output columns are ignored by the caller.
    /// All GPU addresses are stable per slot index, making this CUDA Graph safe.
    fn batch_decode_linear_attention_slots(
        &self,
        attn: &LinearAttentionLayer,
        slot_states: &mut [RecurrentState],
        layer_idx: usize,
        padded_bs: usize,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        ops::gemm_into(&self.ctx, &attn.in_proj_qkv, &bufs.normed, &mut bufs.qkv);
        ops::gemm_into(&self.ctx, &attn.in_proj_z, &bufs.normed, &mut bufs.z);
        ops::gemm_into(&self.ctx, &attn.in_proj_b, &bufs.normed, &mut bufs.b_proj);
        ops::gemm_into(&self.ctx, &attn.in_proj_a, &bufs.normed, &mut bufs.a_proj);

        for (slot_idx, slot_state) in slot_states.iter_mut().enumerate().take(padded_bs) {
            let layer_state = &mut slot_state.layers[layer_idx];

            ops::extract_vec_into(&self.ctx, &bufs.qkv, slot_idx, &mut bufs.qkv_tmp)?;
            ops::conv1d_decode_into(
                &self.ctx,
                &bufs.qkv_tmp,
                &attn.conv1d_weight,
                &mut layer_state.conv_state,
                &mut bufs.qkv_conv_tmp,
                self.config.linear_conv_kernel_dim,
            );
            ops::extract_vec_into(&self.ctx, &bufs.b_proj, slot_idx, &mut bufs.b_tmp)?;
            ops::extract_vec_into(&self.ctx, &bufs.a_proj, slot_idx, &mut bufs.a_tmp)?;

            ops::gated_delta_rule_decode_vec_into(
                &self.ctx,
                &bufs.qkv_conv_tmp,
                &bufs.b_tmp,
                &bufs.a_tmp,
                &attn.dt_bias,
                &attn.a_log,
                &mut layer_state.state,
                &mut bufs.gdr_tmp,
                self.config.linear_num_key_heads,
                self.config.linear_num_value_heads,
                self.config.linear_key_head_dim,
                self.config.linear_value_head_dim,
            );
            ops::write_vec_into(&self.ctx, &bufs.gdr_tmp, &mut bufs.gdr_out, slot_idx)?;
        }

        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &bufs.gdr_out,
            &attn.norm_weight,
            &bufs.z,
            &mut bufs.normed_gated,
            self.config.linear_num_value_heads,
            self.config.linear_value_head_dim,
            self.config.rms_norm_eps,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.out_proj,
            &bufs.normed_gated,
            &mut bufs.attn_results,
        );
        Ok(())
    }
}
