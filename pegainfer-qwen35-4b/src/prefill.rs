use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

/// Maximum prefill sequence length for Qwen3.5 full-attention paged kernels.
pub(crate) const MAX_SEQ: usize = 20_000;
const HEAD_DIM: usize = 256;

use super::prefill_buffers::GdrChunkwiseScratch35;
use super::recurrent_state::RecurrentState;
use super::weights::{
    FullAttentionLayer, LayerKind, LinearAttentionLayer, Qwen35Model, TransformerBlock35,
};
use crate::ffi;
use crate::ops;
use crate::ops::PrefillPagedPlan;
use pegainfer_core::kv_cache::KVCache;
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::tensor::{DeviceVec, HiddenStates};

fn checked_prefill_end_pos(
    base_pos: usize,
    seq_len: usize,
    max_position_embeddings: usize,
) -> Result<usize> {
    let end_pos = base_pos.checked_add(seq_len).ok_or_else(|| {
        anyhow::anyhow!("Qwen3.5 prefill position overflow: base_pos={base_pos}, seq_len={seq_len}")
    })?;
    anyhow::ensure!(
        end_pos <= MAX_SEQ,
        "Qwen3.5 prefill HND staging supports end_pos <= {MAX_SEQ}, requested end_pos={end_pos}; max_position_embeddings={}",
        max_position_embeddings
    );
    Ok(end_pos)
}

impl Qwen35Model {
    pub(super) fn prefill_forward(
        &self,
        token_ids: &[u32],
        kv_cache: &mut KVCache,
        kv_state: &mut KvState,
        recurrent: &mut RecurrentState,
    ) -> Result<DeviceVec> {
        let seq_len = token_ids.len();
        anyhow::ensure!(seq_len > 0, "prefill_forward requires at least one token");
        let c = &self.config;
        let base_pos = kv_state.seq_len();
        let end_pos = checked_prefill_end_pos(base_pos, seq_len, c.max_position_embeddings)?;
        self.ensure_rope_cache_covers(end_pos)?;

        kv_cache.init_if_needed(&self.ctx, c.head_dim)?;

        // Get embeddings for all tokens
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(token_ids)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let hidden_dim = c.hidden_size;
        let mut hidden_batch = HiddenStates::zeros(&self.ctx, hidden_dim, seq_len)?;
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &token_ids_gpu,
            &mut hidden_batch,
        )?;

        // Advance paged KV state and build prefill plan (shared across all layers).
        kv_state.ensure_capacity(end_pos)?;
        kv_state.advance(seq_len);
        let kv_desc = kv_state.desc();
        let prefill_plan = PrefillPagedPlan::new(
            &self.ctx,
            &kv_desc,
            base_pos,
            seq_len,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )?;

        // Process layers
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        let mut gdr_chunkwise_scratch = GdrChunkwiseScratch35::new(&self.ctx, c, seq_len)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_batch = self.prefill_layer(
                layer_idx,
                layer,
                &hidden_batch,
                &mut gdr_chunkwise_scratch,
                &mut linear_idx,
                &mut full_idx,
                kv_cache,
                kv_state,
                &prefill_plan,
                recurrent,
            )?;
        }

        // All layers processed. Advance write-buffer seq_len for next prefill call.
        kv_cache.advance_seq_len(seq_len);
        recurrent.seq_len += seq_len;
        debug_assert_eq!(
            kv_cache.len(),
            kv_state.seq_len(),
            "kv_cache and kv_state seq_len diverged"
        );

        // Extract last token's hidden state
        let last_hidden = ops::extract_vec(&self.ctx, &hidden_batch, seq_len - 1)?;

        // Final norm (1+weight offset)
        let normed = {
            let mut out = DeviceVec::zeros(&self.ctx, hidden_dim)?;
            ops::rms_norm_offset_into(
                &self.ctx,
                &last_hidden,
                &self.norm,
                c.rms_norm_eps,
                &mut out,
            )?;
            out
        };

        // LM head (tied embeddings)
        ops::linear(&self.ctx, &normed, &self.embed_tokens)
    }

    /// Process one layer during prefill. Returns updated hidden_batch.
    #[allow(clippy::too_many_arguments)]
    fn prefill_layer(
        &self,
        _layer_idx: usize,
        layer: &TransformerBlock35,
        hidden_batch: &HiddenStates,
        gdr_chunkwise_scratch: &mut GdrChunkwiseScratch35,
        linear_idx: &mut usize,
        full_idx: &mut usize,
        kv_cache: &mut KVCache,
        kv_state: &KvState,
        prefill_plan: &PrefillPagedPlan,
        recurrent: &mut RecurrentState,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let seq_len = hidden_batch.seq_len;

        // 1. Input layernorm — per-token (no batched offset norm kernel yet)
        // Use standard batched norm and add the offset correction manually
        // Actually we need the (1+w) variant. Process token by token for now.
        let mut normed_batch =
            self.batched_rms_norm_offset(hidden_batch, &layer.input_layernorm, eps)?;

        // 2. Attention / Linear attention — per-token for correctness
        let attn_out_dim = match &layer.attn {
            LayerKind::FullAttention(_) => c.full_attn_q_dim(),
            LayerKind::LinearAttention(_) => c.linear_attn_z_dim(),
        };

        // Batch project, then per-token attention/recurrent
        let attn_results = match &layer.attn {
            LayerKind::FullAttention(attn) => self.prefill_full_attention(
                attn,
                &normed_batch,
                full_idx,
                kv_cache,
                kv_state,
                prefill_plan,
                attn_out_dim,
                seq_len,
            )?,
            LayerKind::LinearAttention(attn) => self.prefill_linear_attention(
                attn,
                &normed_batch,
                linear_idx,
                recurrent,
                gdr_chunkwise_scratch,
                seq_len,
            )?,
        };

        // 3. Residual + post-attention layernorm
        let hidden_plus_attn = ops::add_batch(&self.ctx, hidden_batch, &attn_results)?;

        // Post-attention layernorm (1+weight offset, batched per-token)
        normed_batch =
            self.batched_rms_norm_offset(&hidden_plus_attn, &layer.post_attention_layernorm, eps)?;

        // 4. MLP (batched)
        let gate_out = ops::gemm(&self.ctx, &layer.mlp.gate_proj, &normed_batch)?;
        let up_out = ops::gemm(&self.ctx, &layer.mlp.up_proj, &normed_batch)?;
        let act_out = ops::silu_mul_batch(&self.ctx, &gate_out, &up_out)?;
        let mlp_out = ops::gemm(&self.ctx, &layer.mlp.down_proj, &act_out)?;

        // 5. Residual
        ops::add_batch(&self.ctx, &hidden_plus_attn, &mlp_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_full_attention(
        &self,
        attn: &FullAttentionLayer,
        normed_batch: &HiddenStates,
        full_idx: &mut usize,
        kv_cache: &mut KVCache,
        kv_state: &KvState,
        prefill_plan: &PrefillPagedPlan,
        _attn_out_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let attn_out_dim = c.full_attn_q_dim();
        let eps = c.rms_norm_eps;
        let q_full_batch = ops::gemm(&self.ctx, &attn.q_proj, normed_batch)?;
        let k_batch = ops::gemm(&self.ctx, &attn.k_proj, normed_batch)?;
        let v_batch = ops::gemm(&self.ctx, &attn.v_proj, normed_batch)?;
        let mut attn_out_batch = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;

        let base_pos = kv_cache.len();
        let (kc, vc) = kv_cache.get_cache_mut(&self.ctx, *full_idx)?;
        let mut q_prepped = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;
        let start_pos_cpu: CudaSlice<i32> = self
            .ctx
            .stream
            .clone_htod(&[base_pos as i32])
            .map_err(|e| anyhow::anyhow!("H2D start_pos failed: {e}"))?;

        // Step 1: QK norm + partial RoPE + write processed K/V to HND write buffers.
        unsafe {
            let (qf_ptr, _) = q_full_batch.data.device_ptr(&self.ctx.stream);
            let (k_ptr, _) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (qp_ptr, _) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (kc_ptr, _) = kc.data.device_ptr_mut(&self.ctx.stream);
            let (vc_ptr, _) = vc.data.device_ptr_mut(&self.ctx.stream);
            let (sp_ptr, _) = start_pos_cpu.device_ptr(&self.ctx.stream);
            ffi::prefill_attention_hd256_prep_cuda(
                qf_ptr as *const ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                qp_ptr as *mut ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                c.num_attention_heads as i32,
                c.num_key_value_heads as i32,
                seq_len as i32,
                sp_ptr as *const i32,
                c.rotary_dim as i32,
                eps,
                MAX_SEQ as i32,
                self.ctx.stream.cu_stream(),
            );
        }

        // Step 2: Scatter processed K/V from HND write buffers to paged pool.
        // Offset src by base_pos so nnz-index 0 maps to HND position base_pos.
        let layout = kv_state.layout();
        let layer_k_off = (*full_idx * layout.layer_stride) as i64;
        let layer_v_off = layer_k_off + layout.kv_block_len as i64;
        let stride_page = layout.page_stride as i64;
        let src_stride_n = HEAD_DIM as i64;
        let src_stride_h = (MAX_SEQ * HEAD_DIM) as i64;
        {
            let (buf_ptr, _gbuf) = kv_state.buffer().device_ptr(&self.ctx.stream);
            let (kc_ptr, _gkc) = kc.data.device_ptr(&self.ctx.stream);
            let (vc_ptr, _gvc) = vc.data.device_ptr(&self.ctx.stream);
            // Offset by base_pos so scatter nnz-index i reads from HND position (base_pos + i)
            let elem_off = base_pos as u64 * HEAD_DIM as u64 * 2; // bf16 = 2 bytes
            let kc_scatter = (kc_ptr + elem_off) as *const ffi::Half;
            let vc_scatter = (vc_ptr + elem_off) as *const ffi::Half;
            let (pi_ptr, _gpi) = prefill_plan.page_indices_d().device_ptr(&self.ctx.stream);
            let (pip_ptr, _gpip) = prefill_plan.page_indptr_d().device_ptr(&self.ctx.stream);
            let (lpl_ptr, _glpl) = prefill_plan.last_page_len_d().device_ptr(&self.ctx.stream);
            let (bi_ptr, _gbi) = prefill_plan.batch_indices_d().device_ptr(&self.ctx.stream);
            let (pos_ptr, _gpos) = prefill_plan.positions_d().device_ptr(&self.ctx.stream);
            let result = unsafe {
                ffi::paged_kv_scatter_cuda(
                    buf_ptr as *const ffi::Half,
                    layer_k_off,
                    layer_v_off,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    kc_scatter,
                    vc_scatter,
                    bi_ptr as *const i32,
                    pos_ptr as *const i32,
                    seq_len as i32,
                    c.num_key_value_heads as i32,
                    HEAD_DIM as i32,
                    layout.page_size as i32,
                    stride_page,
                    src_stride_n,
                    src_stride_h,
                    self.ctx.stream.cu_stream(),
                )
            };
            anyhow::ensure!(
                result == 0,
                "paged_kv_scatter_cuda (prefill) failed: {result}"
            );
        }

        // Step 3: Batch prefill paged attention (HD=256).
        let sm_scale = 1.0f32 / f32::sqrt(HEAD_DIM as f32);
        {
            let (buf_ptr, _gbuf) = kv_state.buffer().device_ptr(&self.ctx.stream);
            let (qp_ptr, _gqp) = q_prepped.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = attn_out_batch.data.device_ptr_mut(&self.ctx.stream);
            let (pi_ptr, _gpi) = prefill_plan.page_indices_d().device_ptr(&self.ctx.stream);
            let (pip_ptr, _gpip) = prefill_plan.page_indptr_d().device_ptr(&self.ctx.stream);
            let (lpl_ptr, _glpl) = prefill_plan.last_page_len_d().device_ptr(&self.ctx.stream);
            let (qi_ptr, _gqi) = prefill_plan.q_indptr_d().device_ptr(&self.ctx.stream);
            let (ri_ptr, _gri) = prefill_plan
                .request_indices_d()
                .device_ptr(&self.ctx.stream);
            let (qti_ptr, _gqti) = prefill_plan
                .qo_tile_indices_d()
                .device_ptr(&self.ctx.stream);
            let (kti_ptr, _gkti) = prefill_plan
                .kv_tile_indices_d()
                .device_ptr(&self.ctx.stream);
            let (kcs_ptr, _gkcs) = prefill_plan.kv_chunk_size_d().device_ptr(&self.ctx.stream);
            let (tnr_ptr, _gtnr) = prefill_plan.total_num_rows_d().device_ptr(&self.ctx.stream);
            let result = unsafe {
                ffi::batch_prefill_paged_cuda_hd256(
                    qp_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    buf_ptr as *const ffi::Half,
                    layer_k_off,
                    layer_v_off,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    qi_ptr as *const i32,
                    ri_ptr as *const i32,
                    qti_ptr as *const i32,
                    kti_ptr as *const i32,
                    kcs_ptr as *const i32,
                    tnr_ptr as *const u32,
                    c.num_attention_heads as i32,
                    c.num_key_value_heads as i32,
                    HEAD_DIM as i32,
                    layout.page_size as i32,
                    seq_len as i32,
                    prefill_plan.batch_size(),
                    prefill_plan.num_tiles(),
                    stride_page,
                    sm_scale,
                    self.ctx.stream.cu_stream(),
                )
            };
            anyhow::ensure!(
                result == 0,
                "batch_prefill_paged_cuda_hd256 failed: {result}"
            );
        }

        // Step 4: Apply gate from q_full_batch.
        {
            let (qf_ptr, _gqf) = q_full_batch.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = attn_out_batch.data.device_ptr_mut(&self.ctx.stream);
            unsafe {
                ffi::attention_gate_batch_hd256_cuda(
                    qf_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    c.num_attention_heads as i32,
                    seq_len as i32,
                    self.ctx.stream.cu_stream(),
                );
            }
        }

        *full_idx += 1;

        // O projection (batched)
        ops::gemm(&self.ctx, &attn.o_proj, &attn_out_batch)
    }

    fn prefill_linear_attention(
        &self,
        attn: &LinearAttentionLayer,
        normed_batch: &HiddenStates,
        linear_idx: &mut usize,
        recurrent: &mut RecurrentState,
        gdr_chunkwise_scratch: &mut GdrChunkwiseScratch35,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;

        // Batch projections
        let qkv_batch = ops::gemm(&self.ctx, &attn.in_proj_qkv, normed_batch)?;
        let z_batch = ops::gemm(&self.ctx, &attn.in_proj_z, normed_batch)?;
        let b_batch = ops::gemm(&self.ctx, &attn.in_proj_b, normed_batch)?;
        let a_batch = ops::gemm(&self.ctx, &attn.in_proj_a, normed_batch)?;

        let qkv_dim = c.linear_attn_qkv_dim();
        let z_dim = c.linear_attn_z_dim();
        let layer_state = &mut recurrent.layers[*linear_idx];

        let mut qkv_conv_batch = HiddenStates::zeros(&self.ctx, qkv_dim, seq_len)?;
        ops::conv1d_prefill_batch_into(
            &self.ctx,
            &qkv_batch,
            &attn.conv1d_weight,
            &mut layer_state.conv_state,
            &mut qkv_conv_batch,
            c.linear_conv_kernel_dim,
        );

        let mut gdr_out_batch = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        ops::gated_delta_rule_prefill_chunkwise_into(
            &self.ctx,
            &qkv_conv_batch,
            &b_batch,
            &a_batch,
            &attn.dt_bias,
            &attn.a_log,
            &mut layer_state.state,
            gdr_chunkwise_scratch,
            &mut gdr_out_batch,
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        )?;

        let mut normed_out_batch = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &gdr_out_batch,
            &attn.norm_weight,
            &z_batch,
            &mut normed_out_batch,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.rms_norm_eps,
        );

        *linear_idx += 1;

        // Output projection (batched)
        ops::gemm(&self.ctx, &attn.out_proj, &normed_out_batch)
    }

    fn batched_rms_norm_offset(
        &self,
        x: &HiddenStates,
        weight: &DeviceVec,
        eps: f32,
    ) -> Result<HiddenStates> {
        let mut out = HiddenStates::zeros(&self.ctx, x.hidden_dim, x.seq_len)?;
        ops::rms_norm_batch_offset_into(&self.ctx, x, weight, eps, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_SEQ, checked_prefill_end_pos};

    #[test]
    fn checked_prefill_end_pos_accepts_staging_cap() {
        assert_eq!(
            checked_prefill_end_pos(0, MAX_SEQ, 262_144).unwrap(),
            MAX_SEQ
        );
        assert_eq!(
            checked_prefill_end_pos(MAX_SEQ - 1, 1, 262_144).unwrap(),
            MAX_SEQ
        );
    }

    #[test]
    fn checked_prefill_end_pos_rejects_past_staging_cap() {
        let err = checked_prefill_end_pos(0, MAX_SEQ + 1, 262_144)
            .unwrap_err()
            .to_string();
        assert!(err.contains("end_pos <= 20000"));
        assert!(err.contains("requested end_pos=20001"));
    }

    #[test]
    fn checked_prefill_end_pos_rejects_overflow() {
        let err = checked_prefill_end_pos(usize::MAX, 1, 262_144)
            .unwrap_err()
            .to_string();
        assert!(err.contains("prefill position overflow"));
    }
}
