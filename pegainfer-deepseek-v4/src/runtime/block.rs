use super::*;

pub fn block_prefill_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    ptr_cache: &MoeGroupedPtrCache,
    layer: usize,
    input: &HcHiddenStates,
    token_ids: &CudaSlice<u32>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<HcHiddenStates> {
    ctx.set_current()?;
    let block = weights.block(layer)?;

    let (attn_input, attn_hc) = hc_pre_bf16_hidden(
        ctx,
        config,
        input,
        &block.hc_attn_fn,
        &block.hc_attn_scale,
        &block.hc_attn_base,
    )?;
    let attn_norm = rms_norm_bf16_hidden(ctx, &attn_input, &block.attn_norm, config.rms_norm_eps)?;
    let attn_out = attention_prefill_rank_local_bf16_hidden(
        ctx,
        config,
        layer,
        &attn_norm,
        &block.attn,
        rope,
        start_pos,
    )?;
    let after_attn = hc_post_bf16_hidden(ctx, &attn_out, input, &attn_hc)?;

    let (ffn_input, ffn_hc) = hc_pre_bf16_hidden(
        ctx,
        config,
        &after_attn,
        &block.hc_ffn_fn,
        &block.hc_ffn_scale,
        &block.hc_ffn_base,
    )?;
    let ffn_norm = rms_norm_bf16_hidden(ctx, &ffn_input, &block.ffn_norm, config.rms_norm_eps)?;
    let ffn_out =
        moe_rank_local_bf16_hidden(ctx, config, weights, ptr_cache, layer, &ffn_norm, token_ids)?;
    hc_post_bf16_hidden(ctx, &ffn_out, &after_attn, &ffn_hc)
}

pub(crate) fn block_prefill_rank_lane_bf16_hidden_with_decode_cache(
    ctx: &RankGpuContext,
    weights: &RankWeightView<'_>,
    ptr_cache: &MoeGroupedPtrCache,
    comm: &Comm,
    config: &Config,
    layer: usize,
    input: &HcHiddenStates,
    token_ids: &CudaSlice<u32>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
    prefill_window_topk: &PrefillWindowTopk,
) -> Result<HcHiddenStates> {
    ensure!(
        layer < config.n_layers,
        "rank-lane block prefill cache layer {layer} out of range"
    );
    let block = weights.block(layer)?;

    let (attn_input, attn_hc) = hc_pre_bf16_hidden(
        ctx,
        config,
        input,
        &block.hc_attn_fn,
        &block.hc_attn_scale,
        &block.hc_attn_base,
    )
    .with_context(|| format!("hc_pre attention layer {layer}"))?;
    let attn_norm = rms_norm_bf16_hidden(ctx, &attn_input, &block.attn_norm, config.rms_norm_eps)
        .with_context(|| format!("attention rms_norm layer {layer}"))?;
    let mut attn_out = match config.compress_ratios[layer] {
        0 => attention_prefill_rank_local_bf16_hidden_with_cache(
            ctx,
            config,
            layer,
            &attn_norm,
            &block.attn,
            rope,
            start_pos,
            &mut cache.kv,
            prefill_window_topk,
        )?,
        4 => attention_prefill_compressed_overlap_rank_local_collective_bf16_hidden_with_cache(
            ctx,
            config,
            &attn_norm,
            &block.attn,
            rope,
            layer,
            start_pos,
            cache,
            comm,
            prefill_window_topk,
        )?,
        _ => attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden_with_cache(
            ctx,
            config,
            &attn_norm,
            &block.attn,
            rope,
            layer,
            start_pos,
            cache,
        )?,
    };
    all_reduce_hidden_fp32_in_place(ctx, &mut attn_out, comm)
        .with_context(|| format!("attention all_reduce layer {layer}"))?;
    let after_attn = hc_post_bf16_hidden(ctx, &attn_out, input, &attn_hc)
        .with_context(|| format!("hc_post attention layer {layer}"))?;

    let (ffn_input, ffn_hc) = hc_pre_bf16_hidden(
        ctx,
        config,
        &after_attn,
        &block.hc_ffn_fn,
        &block.hc_ffn_scale,
        &block.hc_ffn_base,
    )
    .with_context(|| format!("hc_pre ffn layer {layer}"))?;
    let ffn_norm = rms_norm_bf16_hidden(ctx, &ffn_input, &block.ffn_norm, config.rms_norm_eps)
        .with_context(|| format!("ffn rms_norm layer {layer}"))?;
    let ffn_out = moe_rank_lane_bf16_hidden(
        ctx, config, weights, ptr_cache, comm, layer, &ffn_norm, token_ids,
    )
    .with_context(|| format!("prefill MoE rank-lane layer {layer}"))?;
    hc_post_bf16_hidden(ctx, &ffn_out, &after_attn, &ffn_hc)
        .with_context(|| format!("hc_post ffn layer {layer}"))
}

pub(crate) fn block_decode_rank_lane_bf16_hidden_with_scratch(
    ctx: &RankGpuContext,
    weights: &RankWeightView<'_>,
    ptr_cache: &MoeGroupedPtrCache,
    comm: &Comm,
    moe_comm: &Comm,
    config: &Config,
    layer: usize,
    input: &HcHiddenStates,
    token_ids: &CudaSlice<u32>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
    hc_pre_norm_scratch: &mut HcPreNormScratch,
    shared_expert_scratch: &mut SharedExpertScratch,
    moe_ag_rs_scratch: &mut MoeAgRsScratch,
    attention_projection_scratch: &mut AttentionProjectionScratch,
    attention_output_scratch: &mut AttentionOutputScratch,
    attention_index_scratch: &mut AttentionIndexScratch,
    attention_aux_scratch: &mut AttentionAuxScratch,
    attention_hc_post_scratch: &mut CudaSlice<f32>,
    attention_hc_out: &mut HcHiddenStates,
    layer_out: &mut HcHiddenStates,
) -> Result<()> {
    ensure!(
        input.seq_len == 1,
        "rank lane block decode expects HC seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        layer < config.n_layers,
        "rank lane block decode layer {layer} out of range"
    );
    let block = weights.block(layer)?;

    let (attn_norm, attn_hc) = hc_pre_norm_bf16_hidden_scratch(
        ctx,
        config,
        input,
        &block.hc_attn_fn,
        &block.hc_attn_scale,
        &block.hc_attn_base,
        &block.attn_norm,
        hc_pre_norm_scratch,
    )
    .with_context(|| format!("hc_pre_norm attention layer {layer}"))?;
    match config.compress_ratios[layer] {
        0 => {
            let attn_out = attention_decode_rank_local_bf16_hidden_with_scratch(
                ctx,
                config,
                layer,
                attn_norm,
                &block.attn,
                rope,
                start_pos,
                &mut cache.kv,
                attention_projection_scratch,
                attention_output_scratch,
                attention_index_scratch,
            )
            .with_context(|| format!("attention decode layer {layer}"))?;
            all_reduce_hidden_fp32_hc_post_view_into(
                ctx,
                attn_out,
                input,
                &attn_hc,
                comm,
                attention_hc_post_scratch,
                attention_hc_out,
            )
        }
        4 => {
            let attn_out =
                attention_decode_compressed_overlap_rank_local_collective_bf16_hidden_with_scratch(
                    ctx,
                    config,
                    layer,
                    attn_norm,
                    &block.attn,
                    rope,
                    start_pos,
                    cache,
                    comm,
                    attention_projection_scratch,
                    attention_output_scratch,
                    attention_index_scratch,
                    attention_aux_scratch,
                )
                .with_context(|| format!("attention decode layer {layer}"))?;
            all_reduce_hidden_fp32_hc_post_view_into(
                ctx,
                attn_out,
                input,
                &attn_hc,
                comm,
                attention_hc_post_scratch,
                attention_hc_out,
            )
        }
        _ => {
            let attn_out = attention_decode_compressed_nonoverlap_rank_local_bf16_hidden(
                ctx,
                config,
                layer,
                attn_norm,
                &block.attn,
                rope,
                start_pos,
                cache,
            )
            .with_context(|| format!("attention decode layer {layer}"))?;
            all_reduce_hidden_fp32_hc_post_view_into(
                ctx,
                &attn_out,
                input,
                &attn_hc,
                comm,
                attention_hc_post_scratch,
                attention_hc_out,
            )
        }
    }
    .with_context(|| format!("attention all_reduce hc_post layer {layer}"))?;

    let (ffn_norm, ffn_hc) = hc_pre_norm_bf16_hidden_scratch(
        ctx,
        config,
        attention_hc_out,
        &block.hc_ffn_fn,
        &block.hc_ffn_scale,
        &block.hc_ffn_base,
        &block.ffn_norm,
        hc_pre_norm_scratch,
    )
    .with_context(|| format!("hc_pre_norm ffn layer {layer}"))?;

    let ffn_out = decode_moe_ag_rs_bf16_hidden_with_scratch(
        ctx,
        config,
        weights,
        ptr_cache,
        moe_comm,
        layer,
        ffn_norm,
        token_ids,
        shared_expert_scratch,
        moe_ag_rs_scratch,
    )
    .with_context(|| format!("decode MoE AG/RS layer {layer}"))?;
    hc_post_bf16_hidden_view_into(ctx, ffn_out, attention_hc_out, &ffn_hc, layer_out)
        .with_context(|| format!("hc_post ffn layer {layer}"))?;
    Ok(())
}

pub(crate) fn block_decode_rank_lane_bf16_hidden_batch_with_scratch(
    ctx: &RankGpuContext,
    weights: &RankWeightView<'_>,
    ptr_cache: &MoeGroupedPtrCache,
    comm: &Comm,
    moe_comm: &Comm,
    config: &Config,
    layer: usize,
    input: &HcHiddenStates,
    token_ids: &CudaSlice<u32>,
    rope: &DeepSeekRopeCache,
    batch_meta: &DecodeBatchMeta<'_>,
    cache: &mut LayerDecodeCache,
    hc_pre_norm_scratch: &mut HcPreNormScratch,
    shared_expert_scratch: &mut SharedExpertScratch,
    moe_ag_rs_scratch: &mut MoeAgRsScratch,
    attention_projection_scratch: &mut AttentionProjectionScratch,
    attention_output_scratch: &mut AttentionOutputScratch,
    attention_index_scratch: &mut AttentionIndexScratch,
    _attention_aux_scratch: &mut AttentionAuxScratch,
    attention_hc_post_scratch: &mut CudaSlice<f32>,
    attention_hc_out: &mut HcHiddenStates,
    layer_out: &mut HcHiddenStates,
) -> Result<()> {
    ensure!(
        input.seq_len == batch_meta.batch,
        "rank lane block batch decode seq_len mismatch: input={}, batch={}",
        input.seq_len,
        batch_meta.batch
    );
    ensure!(
        token_ids.len() >= batch_meta.batch,
        "rank lane block batch token capacity too small: need {}, have {}",
        batch_meta.batch,
        token_ids.len()
    );
    ensure!(
        layer < config.n_layers,
        "rank lane block batch decode layer {layer} out of range"
    );
    let block = weights.block(layer)?;

    let (attn_norm, attn_hc) = hc_pre_norm_bf16_hidden_scratch(
        ctx,
        config,
        input,
        &block.hc_attn_fn,
        &block.hc_attn_scale,
        &block.hc_attn_base,
        &block.attn_norm,
        hc_pre_norm_scratch,
    )
    .with_context(|| format!("hc_pre_norm batch attention layer {layer}"))?;
    match config.compress_ratios[layer] {
        0 => {
            let attn_out = attention_decode_rank_local_bf16_hidden_batch_with_scratch(
                ctx,
                config,
                layer,
                attn_norm,
                &block.attn,
                rope,
                batch_meta,
                &mut cache.kv,
                attention_projection_scratch,
                attention_output_scratch,
                attention_index_scratch,
            )
            .with_context(|| format!("attention batch decode layer {layer}"))?;
            all_reduce_hidden_fp32_hc_post_view_into(
                ctx,
                attn_out,
                input,
                &attn_hc,
                comm,
                attention_hc_post_scratch,
                attention_hc_out,
            )
        }
        4 => {
            let attn_out =
                attention_decode_compressed_overlap_rank_local_collective_bf16_hidden_batch_with_scratch(
                    ctx,
                    config,
                    layer,
                    attn_norm,
                    &block.attn,
                    rope,
                    batch_meta,
                    cache,
                    comm,
                    attention_projection_scratch,
                    attention_output_scratch,
                    attention_index_scratch,
                    _attention_aux_scratch,
                )
                .with_context(|| format!("attention overlap batch decode layer {layer}"))?;
            all_reduce_hidden_fp32_hc_post_view_into(
                ctx,
                attn_out,
                input,
                &attn_hc,
                comm,
                attention_hc_post_scratch,
                attention_hc_out,
            )
        }
        _ => {
            let attn_out =
                attention_decode_compressed_nonoverlap_rank_local_bf16_hidden_batch_with_scratch(
                    ctx,
                    config,
                    layer,
                    attn_norm,
                    &block.attn,
                    rope,
                    batch_meta,
                    cache,
                    attention_projection_scratch,
                    attention_output_scratch,
                    attention_index_scratch,
                )
                .with_context(|| format!("attention compressed batch decode layer {layer}"))?;
            all_reduce_hidden_fp32_hc_post_view_into(
                ctx,
                attn_out,
                input,
                &attn_hc,
                comm,
                attention_hc_post_scratch,
                attention_hc_out,
            )
        }
    }
    .with_context(|| format!("attention all_reduce batch hc_post layer {layer}"))?;

    let (ffn_norm, ffn_hc) = hc_pre_norm_bf16_hidden_scratch(
        ctx,
        config,
        attention_hc_out,
        &block.hc_ffn_fn,
        &block.hc_ffn_scale,
        &block.hc_ffn_base,
        &block.ffn_norm,
        hc_pre_norm_scratch,
    )
    .with_context(|| format!("hc_pre_norm batch ffn layer {layer}"))?;

    let ffn_out = decode_moe_ag_rs_bf16_hidden_with_scratch(
        ctx,
        config,
        weights,
        ptr_cache,
        moe_comm,
        layer,
        ffn_norm,
        token_ids,
        shared_expert_scratch,
        moe_ag_rs_scratch,
    )
    .with_context(|| format!("decode batch MoE AG/RS layer {layer}"))?;
    hc_post_bf16_hidden_view_into(ctx, ffn_out, attention_hc_out, &ffn_hc, layer_out)
        .with_context(|| format!("hc_post batch ffn layer {layer}"))?;
    Ok(())
}

fn attention_decode_compressed_overlap_rank_local_collective_bf16_hidden_with_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
    comm: &Comm,
    attention_projection_scratch: &mut AttentionProjectionScratch,
    attention_output_scratch: &'a mut AttentionOutputScratch,
    attention_index_scratch: &mut AttentionIndexScratch,
    attention_aux_scratch: &mut AttentionAuxScratch,
) -> Result<&'a Bf16HiddenStates> {
    ensure!(input.seq_len == 1, "ratio-4 decode expects seq_len=1");
    ensure!(
        config.compress_ratios[layer] == 4,
        "ratio-4 decode called for layer {layer} with ratio {}",
        config.compress_ratios[layer]
    );

    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor weights"))?;
    let indexer = attn
        .indexer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing ratio-4 indexer weights"))?;
    let compressor_state = cache
        .compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor state"))?;

    let mut projections = attention_project_bf16_hidden_scratch(
        ctx,
        config,
        input,
        attn,
        attention_projection_scratch,
    )?;
    apply_rope_attention_projections_view(ctx, &mut projections, rope, start_pos)?;
    copy_bf16_rows_to_cache(
        ctx,
        projections.kv,
        &mut cache.kv,
        0,
        start_pos % config.sliding_window,
        1,
    )?;
    if let Some(compressed_kv) = compressor_overlap_decode_bf16_hidden_with_dim_scratch(
        ctx,
        config,
        input,
        compressor,
        rope,
        start_pos,
        config.head_dim,
        compressor_state,
        false,
        attention_aux_scratch,
    )? {
        copy_bf16_rows_to_cache(
            ctx,
            compressed_kv,
            &mut cache.kv,
            0,
            config.sliding_window + start_pos / 4,
            1,
        )?;
    }

    let indexer_kv = cache
        .indexer_kv
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing indexer kv cache"))?;
    let indexer_state = cache
        .indexer_compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing indexer compressor state"))?;
    let score_len = indexer_scores_decode_bf16_hidden_scratch(
        ctx,
        config,
        input,
        projections.qr,
        indexer,
        rope,
        start_pos,
        indexer_kv,
        indexer_state,
        attention_aux_scratch,
    )?;

    if let Some(score_len) = score_len {
        let mut scores = attention_aux_scratch.indexer_scores.slice_mut(0..score_len);
        comm.all_reduce_in_place(&mut scores, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("NCCL decode indexer score all-reduce failed: {err:?}")
            })?;
    }

    let window_topk = window_topk_indices_decode_into(
        ctx,
        start_pos,
        config.sliding_window,
        &mut attention_index_scratch.window_idxs,
    )?;
    let compressed_len = (start_pos + 1) / 4;
    let (topk_idxs, topk) = if compressed_len > 0 {
        let scores = attention_aux_scratch
            .indexer_scores
            .slice(0..compressed_len);
        let compress_topk = indexer_topk_indices_decode_into(
            ctx,
            config,
            &scores,
            compressed_len,
            config.sliding_window,
            &mut attention_index_scratch.compress_idxs,
        )?;
        concat_topk_indices_into(
            ctx,
            &attention_index_scratch.window_idxs,
            window_topk,
            &attention_index_scratch.compress_idxs,
            compress_topk,
            1,
            &mut attention_index_scratch.topk_idxs,
        )?;
        (
            &attention_index_scratch.topk_idxs,
            window_topk + compress_topk,
        )
    } else {
        (&attention_index_scratch.window_idxs, window_topk)
    };
    indexed_attention_cache_bf16_hidden_view_into(
        ctx,
        config,
        &projections,
        &cache.kv,
        attn,
        &topk_idxs,
        topk,
        &mut attention_output_scratch.attn_out,
    )?;
    attention_output_project_bf16_hidden_scratch(
        ctx,
        attn,
        rope,
        start_pos,
        attention_output_scratch,
    )
}

fn attention_decode_compressed_overlap_rank_local_collective_bf16_hidden_batch_with_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    batch_meta: &DecodeBatchMeta<'_>,
    cache: &mut LayerDecodeCache,
    comm: &Comm,
    attention_projection_scratch: &mut AttentionProjectionScratch,
    attention_output_scratch: &'a mut AttentionOutputScratch,
    attention_index_scratch: &mut AttentionIndexScratch,
    attention_aux_scratch: &mut AttentionAuxScratch,
) -> Result<&'a Bf16HiddenStates> {
    ensure!(
        input.seq_len == batch_meta.batch,
        "ratio-4 batch decode seq_len mismatch: input={}, batch={}",
        input.seq_len,
        batch_meta.batch
    );
    ensure!(
        config.compress_ratios[layer] == 4,
        "ratio-4 batch decode called for layer {layer} with ratio {}",
        config.compress_ratios[layer]
    );
    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor weights"))?;
    let indexer = attn
        .indexer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing ratio-4 indexer weights"))?;
    let compressor_state = cache
        .compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor state"))?;
    let indexer_kv = cache
        .indexer_kv
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing indexer kv cache"))?;
    let indexer_state = cache
        .indexer_compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing indexer compressor state"))?;

    let projections = attention_project_bf16_hidden_scratch(
        ctx,
        config,
        input,
        attn,
        attention_projection_scratch,
    )?;
    apply_rope_q_kv_batch_in_place(
        ctx,
        projections.q,
        projections.kv,
        rope,
        projections.local_heads,
        projections.head_dim,
        batch_meta.start_pos,
    )?;
    copy_bf16_rows_to_cache_indexed(
        ctx,
        projections.kv,
        &mut cache.kv,
        batch_meta.src_rows,
        batch_meta.window_dst_rows,
        batch_meta.batch,
    )?;
    let compressed_slots = batch_meta.compressed_slots;
    for row in 0..batch_meta.batch {
        let row_input = copy_bf16_row_to_hidden(ctx, input, row)?;
        let state_offset = batch_meta.slot_ids_host[row] * 8;
        if let Some(compressed_kv) = compressor_overlap_decode_bf16_hidden_with_dim_at(
            ctx,
            config,
            &row_input,
            compressor,
            rope,
            batch_meta.start_pos_host[row],
            config.head_dim,
            compressor_state,
            state_offset,
            false,
        )? {
            let dst = decode_cache_compressed_row(
                config.sliding_window,
                compressed_slots,
                batch_meta.slot_ids_host[row],
                batch_meta.start_pos_host[row] / 4,
            );
            copy_bf16_rows_to_cache(ctx, &compressed_kv, &mut cache.kv, 0, dst, 1)?;
        }
        if let Some(indexer_compressed_kv) = compressor_overlap_decode_bf16_hidden_with_dim_at(
            ctx,
            config,
            &row_input,
            &indexer.compressor,
            rope,
            batch_meta.start_pos_host[row],
            config.index_head_dim,
            indexer_state,
            state_offset,
            true,
        )? {
            let dst = batch_meta.slot_ids_host[row] * compressed_slots
                + batch_meta.start_pos_host[row] / 4;
            copy_bf16_rows_to_cache(ctx, &indexer_compressed_kv, indexer_kv, 0, dst, 1)?;
        }
    }

    let max_compressed_len = batch_meta
        .start_pos_host
        .iter()
        .map(|pos| (pos + 1) / 4)
        .max()
        .unwrap_or(0);
    let window_topk = window_topk_indices_decode_batch_into(
        ctx,
        batch_meta.start_pos,
        batch_meta.window_base,
        batch_meta.batch,
        config.sliding_window,
        &mut attention_index_scratch.window_idxs,
    )?;
    let (topk_idxs, topk) = if max_compressed_len > 0 {
        let local_heads = attention_aux_scratch.local_index_heads;
        fp8_linear_bf16_hidden_into(
            ctx,
            projections.qr,
            &indexer.wq_b,
            &mut attention_aux_scratch.indexer_q,
        )?;
        apply_rope_hidden_batch_in_place(
            ctx,
            &mut attention_aux_scratch.indexer_q,
            rope,
            local_heads,
            config.index_head_dim,
            batch_meta.start_pos,
            false,
        )?;
        hadamard_fp4_quant_bf16_hidden_in_place(
            ctx,
            &mut attention_aux_scratch.indexer_q,
            local_heads,
            config.index_head_dim,
        )?;
        bf16_linear_bf16_hidden_into(
            ctx,
            input,
            &indexer.weights_proj,
            &mut attention_aux_scratch.indexer_weights,
        )?;
        indexer_scores_decode_batch_into(
            ctx,
            config,
            IndexerDecodeBatchInputs {
                q: &attention_aux_scratch.indexer_q,
                kv_cache: indexer_kv,
                weights: &attention_aux_scratch.indexer_weights,
                compressed_len: batch_meta.compressed_len,
                cache_base: batch_meta.compressed_base,
                batch: batch_meta.batch,
                max_compressed_len,
            },
            &mut attention_aux_scratch.indexer_scores,
        )?;
        let score_len = batch_meta.batch * max_compressed_len;
        let mut scores = attention_aux_scratch.indexer_scores.slice_mut(0..score_len);
        comm.all_reduce_in_place(&mut scores, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("NCCL batch decode indexer score all-reduce failed: {err:?}")
            })?;
        let compress_topk = indexer_topk_indices_decode_batch_into(
            ctx,
            config,
            &attention_aux_scratch.indexer_scores,
            batch_meta.compressed_len,
            batch_meta.compressed_base,
            batch_meta.batch,
            max_compressed_len,
            &mut attention_index_scratch.compress_idxs,
        )?;
        concat_topk_indices_into(
            ctx,
            &attention_index_scratch.window_idxs,
            window_topk,
            &attention_index_scratch.compress_idxs,
            compress_topk,
            batch_meta.batch,
            &mut attention_index_scratch.topk_idxs,
        )?;
        (
            &attention_index_scratch.topk_idxs,
            window_topk + compress_topk,
        )
    } else {
        (&attention_index_scratch.window_idxs, window_topk)
    };
    indexed_attention_cache_bf16_hidden_view_into(
        ctx,
        config,
        &projections,
        &cache.kv,
        attn,
        topk_idxs,
        topk,
        &mut attention_output_scratch.attn_out,
    )?;
    attention_output_project_bf16_hidden_batch_scratch(
        ctx,
        attn,
        rope,
        batch_meta.start_pos,
        batch_meta.batch,
        attention_output_scratch,
    )
}
