//! Unified forward pass: prefill + decode tokens in a single forward pass.
//!
//! GEMM ops (QKV proj, O proj, MLP) process all tokens together.
//! Attention splits: prefill tokens use FlashInfer BatchPrefill, decode tokens
//! use FlashInfer BatchDecode, outputs are concatenated.

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use super::config::PREFILL_ATTENTION_CTA_TILE_Q;
use super::prefill::PrefillBuffers;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::lora::{DeviceLoraTokenGroup, build_lora_token_ranges, prepare_lora_token_groups};
use openinfer_core::ffi;
use openinfer_core::kv_pool::KvLayout;
use openinfer_core::ops;
use openinfer_core::ops::PrefillPagedPlan;
use openinfer_core::tensor::{DeviceContext, HiddenStates};
use openinfer_kv_cache::KvView;

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
        kv_views: &[&KvView],
        decode_positions: &[usize],
    ) -> Result<Self> {
        let num_decode = kv_views.len();

        let mut all_page_indices = Vec::new();
        let mut indptr = vec![0i32];
        let mut last_page_lens = Vec::with_capacity(num_decode);
        let mut chunk_sizes = Vec::with_capacity(num_decode);

        for kv in kv_views {
            all_page_indices.extend_from_slice(kv.page_indices());
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

        // Prefill plan from views
        let page_indices: Vec<Vec<i32>> = prefill_views
            .iter()
            .map(|v| v.page_indices().to_vec())
            .collect();
        let last_page_lens: Vec<usize> = prefill_views
            .iter()
            .map(openinfer_kv_cache::KvView::last_page_len)
            .collect();
        let prefill_plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            &self.ctx,
            &page_indices,
            &last_page_lens,
            &prefill_start_positions,
            &prefill_seq_lens,
            self.local_num_attention_heads(),
            self.local_num_key_value_heads(),
            self.config.head_dim,
            PREFILL_ATTENTION_CTA_TILE_Q,
        )?;

        // Decode attention metadata
        let decode_view_refs: Vec<&KvView> = decode_views.iter().collect();
        let decode_meta =
            DecodeAttentionMeta::build(&self.ctx, &decode_view_refs, &decode_positions)?;

        // ── 4. Process layers ─────────────────────────────────────────
        let hidden = self.unified_layers(
            hidden,
            total_prefill,
            num_decode_reqs,
            &positions_d,
            &prefill_plan,
            &decode_meta,
            &lora_groups,
            kv_buffer,
            layout,
        )?;

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
        total_prefill: usize,
        num_decode: usize,
        positions_d: &CudaSlice<i32>,
        prefill_plan: &PrefillPagedPlan,
        decode_meta: &DecodeAttentionMeta,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
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
        total_prefill: usize,
        num_decode: usize,
        positions_d: &CudaSlice<i32>,
        prefill_plan: &PrefillPagedPlan,
        decode_meta: &DecodeAttentionMeta,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
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
                    (self.cos_cache.data.len() / head_dim) as i32,
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
