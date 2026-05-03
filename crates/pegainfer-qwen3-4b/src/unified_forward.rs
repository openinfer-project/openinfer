//! Unified forward pass: prefill + decode tokens in a single forward pass.
//!
//! GEMM ops (QKV proj, O proj, MLP) process all tokens together.
//! Attention splits: prefill tokens use FlashInfer BatchPrefill, decode tokens
//! use FlashInfer BatchDecode, outputs are concatenated.

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use super::prefill::PrefillBuffers;
use super::weights::{Qwen3Model, TransformerBlock};
use pegainfer_core::ffi;
use pegainfer_core::kv_pool::{KvLayout, KvState};
use pegainfer_core::ops;
use pegainfer_core::ops::PrefillPagedPlan;
use pegainfer_core::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Decode attention metadata (allocated per unified step, not CUDA-graph safe).
#[allow(clippy::struct_field_names)]
struct DecodeAttentionMeta {
    page_indices_d: CudaSlice<i32>,
    page_indptr_d: CudaSlice<i32>,
    last_page_len_d: CudaSlice<i32>,
    positions_d: CudaSlice<i32>,
    request_indices_d: CudaSlice<i32>,
    kv_tile_indices_d: CudaSlice<i32>,
    kv_chunk_size_d: CudaSlice<i32>,
}

impl DecodeAttentionMeta {
    fn build(
        ctx: &DeviceContext,
        kv_states: &[&mut KvState],
        decode_positions: &[usize],
    ) -> Result<Self> {
        let num_decode = kv_states.len();

        let mut all_page_indices = Vec::new();
        let mut indptr = vec![0i32];
        let mut last_page_lens = Vec::with_capacity(num_decode);
        let mut chunk_sizes = Vec::with_capacity(num_decode);

        for kv in kv_states {
            let pages = kv.page_indices_i32();
            all_page_indices.extend_from_slice(&pages);
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(kv.last_page_len() as i32);
            chunk_sizes.push(kv.seq_len() as i32);
        }

        let request_indices: Vec<i32> = (0..num_decode as i32).collect();
        let kv_tile_indices = vec![0i32; num_decode];
        let positions: Vec<i32> = decode_positions.iter().map(|&p| p as i32).collect();

        Ok(Self {
            page_indices_d: ctx.stream.clone_htod(&all_page_indices)?,
            page_indptr_d: ctx.stream.clone_htod(&indptr)?,
            last_page_len_d: ctx.stream.clone_htod(&last_page_lens)?,
            positions_d: ctx.stream.clone_htod(&positions)?,
            request_indices_d: ctx.stream.clone_htod(&request_indices)?,
            kv_tile_indices_d: ctx.stream.clone_htod(&kv_tile_indices)?,
            kv_chunk_size_d: ctx.stream.clone_htod(&chunk_sizes)?,
        })
    }
}

/// Byte offset for column `col` in a bf16 buffer with `hidden_dim` rows.
fn col_byte_offset(hidden_dim: usize, col: usize) -> u64 {
    (col * hidden_dim * std::mem::size_of::<bf16>()) as u64
}

impl Qwen3Model {
    /// Unified step: prefill + decode in one forward pass.
    ///
    /// Returns `(prefill_logits, decode_logits)` — one `DeviceVec` per request.
    pub(crate) fn unified_step(
        &self,
        prefill_prompts: &[&[u32]],
        prefill_kv_states: &mut [&mut KvState],
        decode_tokens: &[u32],
        decode_kv_states: &mut [&mut KvState],
    ) -> Result<(Vec<DeviceVec>, Vec<DeviceVec>)> {
        let num_prefill_reqs = prefill_prompts.len();
        let num_decode_reqs = decode_tokens.len();
        assert_eq!(num_prefill_reqs, prefill_kv_states.len());
        assert_eq!(num_decode_reqs, decode_kv_states.len());
        assert!(num_prefill_reqs > 0 && num_decode_reqs > 0);

        let prefill_seq_lens: Vec<usize> = prefill_prompts.iter().map(|p| p.len()).collect();
        let total_prefill: usize = prefill_seq_lens.iter().sum();
        let total_tokens = total_prefill + num_decode_reqs;

        // ── 1. Concatenate all tokens and get embeddings ──────────────
        let mut all_tokens: Vec<u32> = Vec::with_capacity(total_tokens);
        for prompt in prefill_prompts {
            all_tokens.extend_from_slice(prompt);
        }
        all_tokens.extend_from_slice(decode_tokens);
        let hidden = self.get_embeddings_batch(&all_tokens)?;

        // ── 2. Prepare KV states ──────────────────────────────────────
        let prefill_start_positions: Vec<usize> =
            prefill_kv_states.iter().map(|kv| kv.seq_len()).collect();
        for (i, kv) in prefill_kv_states.iter_mut().enumerate() {
            kv.ensure_capacity(prefill_start_positions[i] + prefill_seq_lens[i])?;
            kv.advance(prefill_seq_lens[i]);
        }

        let mut decode_positions = Vec::with_capacity(num_decode_reqs);
        for kv in decode_kv_states.iter_mut() {
            let pos = kv.seq_len();
            kv.ensure_capacity(pos + 1)?;
            kv.advance(1);
            decode_positions.push(pos);
        }

        // ── 3. Build metadata ─────────────────────────────────────────

        // Unified positions for RoPE (all tokens)
        let mut positions: Vec<i32> = Vec::with_capacity(total_tokens);
        for (i, &seq_len) in prefill_seq_lens.iter().enumerate() {
            let start = prefill_start_positions[i];
            positions.extend((start..start + seq_len).map(|p| p as i32));
        }
        for &pos in &decode_positions {
            positions.push(pos as i32);
        }
        let positions_d = self.ctx.stream.clone_htod(&positions)?;

        // Prefill plan (for prefill attention + KV scatter)
        let prefill_descs: Vec<_> = prefill_kv_states.iter().map(|kv| kv.desc()).collect();
        let prefill_plan = PrefillPagedPlan::new_batch(
            &self.ctx,
            &prefill_descs,
            &prefill_start_positions,
            &prefill_seq_lens,
            self.local_num_attention_heads(),
            self.local_num_key_value_heads(),
            self.config.head_dim,
        )?;

        // Decode attention metadata (built AFTER advance so seq_lens reflect new state)
        let decode_meta =
            DecodeAttentionMeta::build(&self.ctx, decode_kv_states, &decode_positions)?;

        // ── 4. Process layers ─────────────────────────────────────────
        let kv_buffer = prefill_kv_states[0].buffer();
        let layout = *prefill_kv_states[0].layout();

        let hidden = self.unified_layers(
            hidden,
            total_prefill,
            num_decode_reqs,
            &positions_d,
            &prefill_plan,
            &decode_meta,
            kv_buffer,
            &layout,
        )?;

        // ── 5. Extract logits ─────────────────────────────────────────
        // Prefill: last token of each sequence
        let mut prefill_logits = Vec::with_capacity(num_prefill_reqs);
        let mut offset = 0;
        for &seq_len in &prefill_seq_lens {
            let last_idx = offset + seq_len - 1;
            let last_hidden = ops::extract_vec(&self.ctx, &hidden, last_idx)?;
            let normed = ops::rms_norm(
                &self.ctx,
                &last_hidden,
                &self.norm,
                self.config.rms_norm_eps,
            )?;
            let logits = ops::linear(&self.ctx, &normed, self.output_projection())?;
            prefill_logits.push(logits);
            offset += seq_len;
        }

        // Decode: all decode tokens
        let mut decode_logits = Vec::with_capacity(num_decode_reqs);
        for i in 0..num_decode_reqs {
            let idx = total_prefill + i;
            let last_hidden = ops::extract_vec(&self.ctx, &hidden, idx)?;
            let normed = ops::rms_norm(
                &self.ctx,
                &last_hidden,
                &self.norm,
                self.config.rms_norm_eps,
            )?;
            let logits = ops::linear(&self.ctx, &normed, self.output_projection())?;
            decode_logits.push(logits);
        }

        Ok((prefill_logits, decode_logits))
    }

    fn unified_layers(
        &self,
        mut hidden: HiddenStates,
        total_prefill: usize,
        num_decode: usize,
        positions_d: &CudaSlice<i32>,
        prefill_plan: &PrefillPagedPlan,
        decode_meta: &DecodeAttentionMeta,
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
    ) -> Result<HiddenStates> {
        let total_tokens = total_prefill + num_decode;
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
                total_prefill,
                num_decode,
                positions_d,
                prefill_plan,
                decode_meta,
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
        total_prefill: usize,
        num_decode: usize,
        positions_d: &CudaSlice<i32>,
        prefill_plan: &PrefillPagedPlan,
        decode_meta: &DecodeAttentionMeta,
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
    ) -> Result<()> {
        let total_tokens = total_prefill + num_decode;
        let num_heads = self.local_num_attention_heads();
        let num_kv_heads = self.local_num_key_value_heads();
        let head_dim = self.config.head_dim;
        let kv_dim = self.local_kv_dim();
        let q_dim = self.local_q_dim();
        let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

        let k_offset = (layer_idx * layout.layer_stride) as i64;
        let v_offset = (layer_idx * layout.layer_stride + layout.kv_block_len) as i64;
        let stride_page = layout.page_stride as i64;

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
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim_l,
            kv_dim_l,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim_l + kv_dim_l,
            kv_dim_l,
            &bufs.normed,
            &mut bufs.v_batch,
        );

        // ── 3. QK norm + RoPE [all tokens, per-token positions] ──────
        {
            let (q_ptr, _gq) = bufs.q_batch.data.device_ptr_mut(&self.ctx.stream);
            let (k_ptr, _gk) = bufs.k_batch.data.device_ptr_mut(&self.ctx.stream);
            let (qn_ptr, _gqn) = layer.attention.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _gkn) = layer.attention.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _gc) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _gs) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (pos_ptr, _gp) = positions_d.device_ptr(&self.ctx.stream);

            unsafe {
                ffi::qk_norm_rope_batched_decode_cuda(
                    q_ptr as *mut ffi::Half,
                    k_ptr as *mut ffi::Half,
                    qn_ptr as *const ffi::Half,
                    kn_ptr as *const ffi::Half,
                    cos_ptr as *const ffi::Half,
                    sin_ptr as *const ffi::Half,
                    pos_ptr as *const i32,
                    num_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    total_tokens as i32,
                    self.config.rms_norm_eps,
                    self.ctx.stream.cu_stream(),
                );
            }
        }

        // ── 4. KV cache write ────────────────────────────────────────
        // 4a. Prefill tokens → paged KV scatter
        {
            let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&self.ctx.stream);
            let (k_ptr, _gk) = bufs.k_batch.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _gv) = bufs.v_batch.data.device_ptr(&self.ctx.stream);
            let (pi_ptr, _) = prefill_plan.page_indices_d().device_ptr(&self.ctx.stream);
            let (pip_ptr, _) = prefill_plan.page_indptr_d().device_ptr(&self.ctx.stream);
            let (lpl_ptr, _) = prefill_plan.last_page_len_d().device_ptr(&self.ctx.stream);
            let (bi_ptr, _) = prefill_plan.batch_indices_d().device_ptr(&self.ctx.stream);
            let (ppos_ptr, _) = prefill_plan.positions_d().device_ptr(&self.ctx.stream);

            let src_stride_n = kv_dim as i64;
            let src_stride_h = head_dim as i64;

            let result = unsafe {
                ffi::paged_kv_scatter_cuda(
                    buf_ptr as *const ffi::Half,
                    k_offset,
                    v_offset,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    k_ptr as *const ffi::Half,
                    v_ptr as *const ffi::Half,
                    bi_ptr as *const i32,
                    ppos_ptr as *const i32,
                    total_prefill as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    layout.page_size as i32,
                    stride_page,
                    src_stride_n,
                    src_stride_h,
                    self.ctx.stream.cu_stream(),
                )
            };
            if result != 0 {
                anyhow::bail!(
                    "unified paged_kv_scatter failed for layer {layer_idx} with error {result}"
                );
            }
        }

        // 4b. Decode tokens → paged KV append
        {
            let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&self.ctx.stream);
            let (k_base, _gk) = bufs.k_batch.data.device_ptr(&self.ctx.stream);
            let (v_base, _gv) = bufs.v_batch.data.device_ptr(&self.ctx.stream);
            let (pi_ptr, _) = decode_meta.page_indices_d.device_ptr(&self.ctx.stream);
            let (pip_ptr, _) = decode_meta.page_indptr_d.device_ptr(&self.ctx.stream);
            let (lpl_ptr, _) = decode_meta.last_page_len_d.device_ptr(&self.ctx.stream);
            let (pos_ptr, _) = decode_meta.positions_d.device_ptr(&self.ctx.stream);
            let (ri_ptr, _) = decode_meta.request_indices_d.device_ptr(&self.ctx.stream);

            let k_decode = k_base + col_byte_offset(kv_dim, total_prefill);
            let v_decode = v_base + col_byte_offset(kv_dim, total_prefill);
            let src_stride_n = kv_dim as i64;
            let src_stride_h = head_dim as i64;

            let result = unsafe {
                ffi::paged_kv_scatter_cuda(
                    buf_ptr as *const ffi::Half,
                    k_offset,
                    v_offset,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    k_decode as *const ffi::Half,
                    v_decode as *const ffi::Half,
                    ri_ptr as *const i32,
                    pos_ptr as *const i32,
                    num_decode as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    layout.page_size as i32,
                    stride_page,
                    src_stride_n,
                    src_stride_h,
                    self.ctx.stream.cu_stream(),
                )
            };
            if result != 0 {
                anyhow::bail!(
                    "unified paged_kv_scatter failed for layer {layer_idx} with error {result}"
                );
            }
        }

        // ── 5. Attention (split: prefill + decode) ────────────────────
        // 5a. Prefill attention → attn_output[0..total_prefill]
        {
            let (q_ptr, _gq) = bufs.q_batch.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _go) = bufs.attn_output.data.device_ptr_mut(&self.ctx.stream);
            let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&self.ctx.stream);
            let (pi_ptr, _) = prefill_plan.page_indices_d().device_ptr(&self.ctx.stream);
            let (pip_ptr, _) = prefill_plan.page_indptr_d().device_ptr(&self.ctx.stream);
            let (lpl_ptr, _) = prefill_plan.last_page_len_d().device_ptr(&self.ctx.stream);
            let (qi_ptr, _) = prefill_plan.q_indptr_d().device_ptr(&self.ctx.stream);
            let (ri_ptr, _) = prefill_plan
                .request_indices_d()
                .device_ptr(&self.ctx.stream);
            let (qti_ptr, _) = prefill_plan
                .qo_tile_indices_d()
                .device_ptr(&self.ctx.stream);
            let (kti_ptr, _) = prefill_plan
                .kv_tile_indices_d()
                .device_ptr(&self.ctx.stream);
            let (kcs_ptr, _) = prefill_plan.kv_chunk_size_d().device_ptr(&self.ctx.stream);
            let (tnr_ptr, _) = prefill_plan.total_num_rows_d().device_ptr(&self.ctx.stream);

            let result = unsafe {
                ffi::batch_prefill_paged_cuda(
                    q_ptr as *const ffi::Half,
                    o_ptr as *mut ffi::Half,
                    buf_ptr as *const ffi::Half,
                    k_offset,
                    v_offset,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    qi_ptr as *const i32,
                    ri_ptr as *const i32,
                    qti_ptr as *const i32,
                    kti_ptr as *const i32,
                    kcs_ptr as *const i32,
                    tnr_ptr as *const u32,
                    num_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    layout.page_size as i32,
                    total_prefill as i32,
                    prefill_plan.batch_size(),
                    prefill_plan.num_tiles(),
                    stride_page,
                    sm_scale,
                    self.ctx.stream.cu_stream(),
                )
            };
            if result != 0 {
                anyhow::bail!(
                    "unified prefill attention failed for layer {layer_idx} with error {result}"
                );
            }
        }

        // 5b. Decode attention → attn_output[total_prefill..]
        {
            let (q_base, _gq) = bufs.q_batch.data.device_ptr(&self.ctx.stream);
            let (o_base, _go) = bufs.attn_output.data.device_ptr_mut(&self.ctx.stream);
            let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&self.ctx.stream);
            let (pi_ptr, _) = decode_meta.page_indices_d.device_ptr(&self.ctx.stream);
            let (pip_ptr, _) = decode_meta.page_indptr_d.device_ptr(&self.ctx.stream);
            let (lpl_ptr, _) = decode_meta.last_page_len_d.device_ptr(&self.ctx.stream);
            let (ri_ptr, _) = decode_meta.request_indices_d.device_ptr(&self.ctx.stream);
            let (kti_ptr, _) = decode_meta.kv_tile_indices_d.device_ptr(&self.ctx.stream);
            let (kcs_ptr, _) = decode_meta.kv_chunk_size_d.device_ptr(&self.ctx.stream);

            let q_decode = q_base + col_byte_offset(q_dim, total_prefill);
            let o_decode = o_base + col_byte_offset(q_dim, total_prefill);

            let result = unsafe {
                ffi::paged_attention_decode_cuda(
                    q_decode as *const ffi::Half,
                    o_decode as *mut ffi::Half,
                    buf_ptr as *const ffi::Half,
                    k_offset,
                    v_offset,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    ri_ptr as *const i32,
                    kti_ptr as *const i32,
                    kcs_ptr as *const i32,
                    num_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    layout.page_size as i32,
                    num_decode as i32,
                    stride_page,
                    sm_scale,
                    self.ctx.stream.cu_stream(),
                )
            };
            if result != 0 {
                anyhow::bail!(
                    "unified decode attention failed for layer {layer_idx} with error {result}"
                );
            }
        }

        // ── 6. O projection [all tokens] ─────────────────────────────
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // ── 7+8. Residual add + MLP RMSNorm (fused) ─────────────────
        ops::fused_add_rms_norm_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

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

        // ── 9. Residual add → hidden_out ─────────────────────────────
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }
}
