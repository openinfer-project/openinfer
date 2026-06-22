use anyhow::Result;
use half::bf16;
use openinfer_core::ops;
use openinfer_core::tensor::{DeviceContext, HiddenStates};

use crate::batch_buffers::DFlashBatchBuffers;
use crate::forward::DFlashTargetHidden;
use crate::weights::{DFlashDraftModel, DFlashLayer};

pub struct DFlashBatchInput<'a> {
    pub noise_embedding: &'a HiddenStates,
    pub target_hidden: DFlashTargetHidden<'a>,
    pub position_ids: &'a [i32],
}

pub struct DFlashHostBatchInput<'a> {
    pub noise_embedding: &'a [bf16],
    pub target_hidden: &'a [bf16],
    pub position_ids: &'a [i32],
}

impl DFlashDraftModel {
    pub fn create_batch_buffers(
        &self,
        max_batch_size: usize,
        max_q_len: usize,
        max_ctx_len: usize,
    ) -> Result<DFlashBatchBuffers> {
        DFlashBatchBuffers::new(self, max_batch_size, max_q_len, max_ctx_len)
    }

    pub fn forward_batch<'a>(
        &self,
        requests: &[DFlashBatchInput<'_>],
        bufs: &'a mut DFlashBatchBuffers,
    ) -> Result<&'a HiddenStates> {
        anyhow::ensure!(!requests.is_empty(), "DFlash batch is empty");
        anyhow::ensure!(
            requests.len() <= bufs.max_batch_size,
            "DFlash batch size {} exceeds buffer capacity {}",
            requests.len(),
            bufs.max_batch_size
        );
        // All requests in an exact-shape batch share one (q_len, ctx_len); read
        // it from the first, then narrow the buffer's active shape to match.
        let (q_len, ctx_len) = self.validate_forward_inputs(
            requests[0].noise_embedding,
            &requests[0].target_hidden,
            requests[0].position_ids,
        )?;
        anyhow::ensure!(
            q_len <= bufs.max_q_len && ctx_len <= bufs.max_ctx_len,
            "DFlash batch shape q_len={}, ctx_len={} exceeds buffer capacity q_len={}, ctx_len={}",
            q_len,
            ctx_len,
            bufs.max_q_len,
            bufs.max_ctx_len,
        );
        // Exact-shape batch: the first request is fully validated above, so the
        // rest only need to match the three lengths that fix (q_len, ctx_len)
        // — re-running the full validator per request just repeats the same
        // hidden_dim / positivity checks against the same config.
        for req in &requests[1..] {
            anyhow::ensure!(
                req.noise_embedding.seq_len == q_len
                    && req.noise_embedding.hidden_dim == requests[0].noise_embedding.hidden_dim,
                "DFlash exact-shape batch noise_embedding shape mismatch"
            );
            anyhow::ensure!(
                req.target_hidden.concatenated.seq_len == ctx_len,
                "DFlash exact-shape batch target_hidden seq_len mismatch"
            );
            anyhow::ensure!(
                req.position_ids.len() == ctx_len + q_len,
                "DFlash exact-shape batch position_ids len mismatch"
            );
        }
        bufs.set_active_shape(requests.len(), q_len, ctx_len);
        compact_inputs(self.device_context(), requests, bufs)?;
        self.forward_compact_batch(requests.len(), bufs)?;
        Ok(&bufs.normed)
    }

    pub fn forward_host_batch<'a>(
        &self,
        requests: &[DFlashHostBatchInput<'_>],
        bufs: &'a mut DFlashBatchBuffers,
    ) -> Result<&'a HiddenStates> {
        anyhow::ensure!(!requests.is_empty(), "DFlash host batch is empty");
        anyhow::ensure!(
            requests.len() <= bufs.max_batch_size,
            "DFlash host batch size {} exceeds buffer capacity {}",
            requests.len(),
            bufs.max_batch_size
        );
        let config = self.config();
        let hidden = config.hidden_size;
        let target_hidden_dim = config.hidden_size * config.target_layer_count();
        // Derive the shared (q_len, ctx_len) from the first request, the same
        // way forward_batch derives it from device tensors.
        let first = &requests[0];
        anyhow::ensure!(
            first.noise_embedding.len() % hidden == 0,
            "noise_embedding len {} is not a multiple of hidden_size {}",
            first.noise_embedding.len(),
            hidden,
        );
        let q_len = first.noise_embedding.len() / hidden;
        anyhow::ensure!(
            first.target_hidden.len() % target_hidden_dim == 0,
            "target_hidden len {} is not a multiple of target_hidden_dim {}",
            first.target_hidden.len(),
            target_hidden_dim,
        );
        let ctx_len = first.target_hidden.len() / target_hidden_dim;
        anyhow::ensure!(q_len > 0, "DFlash host batch q_len must be positive");
        anyhow::ensure!(ctx_len > 0, "DFlash host batch ctx_len must be positive");
        anyhow::ensure!(
            q_len <= bufs.max_q_len && ctx_len <= bufs.max_ctx_len,
            "DFlash host batch shape q_len={}, ctx_len={} exceeds buffer capacity q_len={}, ctx_len={}",
            q_len,
            ctx_len,
            bufs.max_q_len,
            bufs.max_ctx_len,
        );
        let noise_len = q_len * hidden;
        let target_len = ctx_len * target_hidden_dim;
        let position_len = ctx_len + q_len;
        for req in &requests[1..] {
            anyhow::ensure!(
                req.noise_embedding.len() == noise_len,
                "noise_embedding len {} != {}",
                req.noise_embedding.len(),
                noise_len
            );
            anyhow::ensure!(
                req.target_hidden.len() == target_len,
                "target_hidden len {} != {}",
                req.target_hidden.len(),
                target_len
            );
            anyhow::ensure!(
                req.position_ids.len() == position_len,
                "position_ids len {} != {}",
                req.position_ids.len(),
                position_len
            );
        }
        bufs.set_active_shape(requests.len(), q_len, ctx_len);
        compact_host_inputs(self.device_context(), requests, bufs)?;
        self.forward_compact_batch(requests.len(), bufs)?;
        Ok(&bufs.normed)
    }

    fn forward_compact_batch(
        &self,
        batch_size: usize,
        bufs: &mut DFlashBatchBuffers,
    ) -> Result<()> {
        let config = self.config();
        ops::gemm_into_checked(
            self.device_context(),
            &self.fc,
            &bufs.target_hidden,
            &mut bufs.target_projected,
        )?;
        ops::rms_norm_batch_into(
            self.device_context(),
            &bufs.target_projected,
            &self.hidden_norm,
            config.rms_norm_eps,
            &mut bufs.target_normed,
        );
        copy_hidden(
            self.device_context(),
            &bufs.noise,
            0,
            &mut bufs.hidden,
            0,
            config.hidden_size,
            bufs.total_q_len,
        )?;
        for layer in &self.layers {
            self.forward_compact_batch_layer(layer, batch_size, bufs)?;
        }
        ops::rms_norm_batch_into(
            self.device_context(),
            &bufs.hidden,
            &self.norm,
            config.rms_norm_eps,
            &mut bufs.normed,
        );
        Ok(())
    }

    fn forward_compact_batch_layer(
        &self,
        layer: &DFlashLayer,
        batch_size: usize,
        bufs: &mut DFlashBatchBuffers,
    ) -> Result<()> {
        let config = self.config();
        let ctx = self.device_context();
        ops::rms_norm_batch_into(
            ctx,
            &bufs.hidden,
            &layer.input_layernorm,
            config.rms_norm_eps,
            &mut bufs.normed,
        );
        ops::gemm_into_checked(ctx, &layer.attention.q_proj, &bufs.normed, &mut bufs.q)?;
        ops::gemm_into_checked(
            ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k_noise,
        )?;
        ops::gemm_into_checked(
            ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v_noise,
        )?;
        ops::qk_norm_rope_batch_decode_into(
            ctx,
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
            ctx,
            &layer.attention.k_proj,
            &bufs.target_normed,
            &mut bufs.k_ctx,
        )?;
        ops::gemm_into_checked(
            ctx,
            &layer.attention.v_proj,
            &bufs.target_normed,
            &mut bufs.v_ctx,
        )?;
        // Context-K needs norm + RoPE but has no corresponding Q. The K-only
        // kernel launches num_kv_heads blocks per token instead of
        // num_q_heads + num_kv_heads, dropping 80% of the joint kernel's work
        // (the dead Q branch) for Qwen3-4B's 16:4 GQA ratio.
        ops::k_norm_rope_batch_decode_into(
            ctx,
            &mut bufs.k_ctx,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions_ctx,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
        );

        // Concatenate per-request [ctx | noise] K/V into the contiguous layout
        // the ragged attention kernel expects. Two strided segment copies per
        // tensor (ctx segment at offset 0, noise segment at offset ctx_len)
        // replace the old 2 * batch_size memcpy_dtod loop (`compact_kv`):
        // bs=32 dropped from 128 launches/layer to 4.
        let kv_seg_total = bufs.ctx_len + bufs.q_len;
        ops::strided_segment_copy_into(
            ctx,
            &bufs.k_ctx,
            &mut bufs.k_all,
            bufs.ctx_len,
            kv_seg_total,
            0,
            batch_size,
        )?;
        ops::strided_segment_copy_into(
            ctx,
            &bufs.k_noise,
            &mut bufs.k_all,
            bufs.q_len,
            kv_seg_total,
            bufs.ctx_len,
            batch_size,
        )?;
        ops::strided_segment_copy_into(
            ctx,
            &bufs.v_ctx,
            &mut bufs.v_all,
            bufs.ctx_len,
            kv_seg_total,
            0,
            batch_size,
        )?;
        ops::strided_segment_copy_into(
            ctx,
            &bufs.v_noise,
            &mut bufs.v_all,
            bufs.q_len,
            kv_seg_total,
            bufs.ctx_len,
            batch_size,
        )?;
        bufs.prepare_ragged_plan(self, batch_size)?;
        let cached_plan = bufs.ragged_plan.take().expect("ragged plan exists");
        let attention_result = ops::batch_prefill_ragged_nhd_noncausal_into(
            ctx,
            &bufs.q,
            &bufs.k_all,
            &bufs.v_all,
            &mut bufs.attn_out,
            &cached_plan.plan,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        );
        bufs.ragged_plan = Some(cached_plan);
        attention_result?;
        ops::gemm_into_checked(
            ctx,
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.o_buf,
        )?;
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            ctx,
            &mut bufs.hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            config.rms_norm_eps,
            &mut bufs.normed,
        )?;
        ops::gemm_into_checked(
            ctx,
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.gate_up,
        )?;
        ops::silu_mul_fused_batch_into(ctx, &bufs.gate_up, &mut bufs.act_out);
        ops::gemm_into_checked(ctx, &layer.mlp.down_proj, &bufs.act_out, &mut bufs.o_buf)?;
        ops::add_batch_into(ctx, &bufs.hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(&mut bufs.hidden, &mut bufs.hidden_out);
        Ok(())
    }
}

fn compact_inputs(
    ctx: &DeviceContext,
    requests: &[DFlashBatchInput<'_>],
    bufs: &mut DFlashBatchBuffers,
) -> Result<()> {
    let hidden = bufs.noise.hidden_dim;
    let target_hidden = bufs.target_hidden.hidden_dim;
    let mut pos_q = Vec::with_capacity(bufs.total_q_len);
    let mut pos_ctx = Vec::with_capacity(bufs.total_ctx_len);
    for (i, req) in requests.iter().enumerate() {
        copy_hidden(
            ctx,
            req.noise_embedding,
            0,
            &mut bufs.noise,
            i * bufs.q_len,
            hidden,
            bufs.q_len,
        )?;
        copy_hidden(
            ctx,
            req.target_hidden.concatenated,
            0,
            &mut bufs.target_hidden,
            i * bufs.ctx_len,
            target_hidden,
            bufs.ctx_len,
        )?;
        pos_ctx.extend_from_slice(&req.position_ids[..bufs.ctx_len]);
        pos_q.extend_from_slice(&req.position_ids[bufs.ctx_len..]);
    }
    let mut dst_q = bufs.positions_q.slice_mut(..pos_q.len());
    ctx.stream.memcpy_htod(&pos_q, &mut dst_q)?;
    let mut dst_ctx = bufs.positions_ctx.slice_mut(..pos_ctx.len());
    ctx.stream.memcpy_htod(&pos_ctx, &mut dst_ctx)?;
    Ok(())
}

fn compact_host_inputs(
    ctx: &DeviceContext,
    requests: &[DFlashHostBatchInput<'_>],
    bufs: &mut DFlashBatchBuffers,
) -> Result<()> {
    let hidden = bufs.noise.hidden_dim;
    let target_hidden = bufs.target_hidden.hidden_dim;
    let q_len = bufs.q_len;
    let ctx_len = bufs.ctx_len;
    let batch_size = requests.len();

    // Stitch all requests into contiguous host slices, then upload each tensor
    // in a single H2D copy — matches Qwen3's batch metadata upload pattern and
    // avoids one launch per request per tensor.
    let mut noise_flat = Vec::with_capacity(batch_size * q_len * hidden);
    let mut target_flat = Vec::with_capacity(batch_size * ctx_len * target_hidden);
    let mut pos_q = Vec::with_capacity(batch_size * q_len);
    let mut pos_ctx = Vec::with_capacity(batch_size * ctx_len);
    for req in requests {
        noise_flat.extend_from_slice(req.noise_embedding);
        target_flat.extend_from_slice(req.target_hidden);
        pos_ctx.extend_from_slice(&req.position_ids[..ctx_len]);
        pos_q.extend_from_slice(&req.position_ids[ctx_len..]);
    }

    let mut noise_dst = bufs.noise.data.slice_mut(..noise_flat.len());
    ctx.stream.memcpy_htod(&noise_flat, &mut noise_dst)?;
    let mut target_dst = bufs.target_hidden.data.slice_mut(..target_flat.len());
    ctx.stream.memcpy_htod(&target_flat, &mut target_dst)?;
    let mut dst_q = bufs.positions_q.slice_mut(..pos_q.len());
    ctx.stream.memcpy_htod(&pos_q, &mut dst_q)?;
    let mut dst_ctx = bufs.positions_ctx.slice_mut(..pos_ctx.len());
    ctx.stream.memcpy_htod(&pos_ctx, &mut dst_ctx)?;
    Ok(())
}

pub(crate) fn copy_hidden(
    ctx: &DeviceContext,
    src: &HiddenStates,
    src_token_offset: usize,
    dst: &mut HiddenStates,
    dst_token_offset: usize,
    hidden_dim: usize,
    token_count: usize,
) -> Result<()> {
    debug_assert_eq!(src.hidden_dim, hidden_dim);
    debug_assert_eq!(dst.hidden_dim, hidden_dim);
    debug_assert!(src_token_offset + token_count <= src.seq_len);
    debug_assert!(dst_token_offset + token_count <= dst.seq_len);
    let len = hidden_dim * token_count;
    let src_offset = hidden_dim * src_token_offset;
    let dst_offset = hidden_dim * dst_token_offset;
    let src_view = src.data.slice(src_offset..src_offset + len);
    let mut dst_view = dst.data.slice_mut(dst_offset..dst_offset + len);
    ctx.stream.memcpy_dtod(&src_view, &mut dst_view)?;
    Ok(())
}
