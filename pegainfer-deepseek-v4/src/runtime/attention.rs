use super::*;

pub fn attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    layer: usize,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        start_pos == 0,
        "compressed attention prefill currently supports start_pos=0 only"
    );
    ensure!(
        layer < config.compress_ratios.len(),
        "layer {layer} out of range"
    );
    let ratio = config.compress_ratios[layer];
    ensure!(ratio > 0, "layer {layer} is not compressed");
    ensure!(ratio != 4, "ratio=4 uses the indexer/overlap path");
    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor weights"))?;

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    if input.seq_len < ratio {
        let mut attn_out = sparse_attention_prefill_bf16_hidden(ctx, config, &projections, attn)?;
        return attention_output_project_bf16_hidden(
            ctx,
            &mut attn_out,
            attn,
            rope,
            projections.local_heads,
            projections.head_dim,
            start_pos,
        );
    }
    let compressed_kv = compressor_nonoverlap_prefill_bf16_hidden(
        ctx, config, input, compressor, ratio, rope, start_pos,
    )?;
    let kv = concat_seq_bf16_hidden(ctx, &projections.kv, &compressed_kv)?;
    let (topk_idxs, topk) = window_and_compress_topk_indices(
        ctx,
        projections.q.seq_len,
        config.sliding_window,
        ratio,
        projections.kv.seq_len,
    )?;
    let indexed_projections = AttentionProjections {
        qr: projections.qr,
        q: projections.q,
        kv,
        local_heads: projections.local_heads,
        head_dim: projections.head_dim,
    };
    let mut attn_out = indexed_attention_prefill_bf16_hidden(
        ctx,
        config,
        &indexed_projections,
        attn,
        &topk_idxs,
        topk,
    )?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        indexed_projections.local_heads,
        indexed_projections.head_dim,
        start_pos,
    )
}

pub(crate) fn attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden_with_cache(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    layer: usize,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        start_pos == 0,
        "compressed attention prefill cache path currently supports start_pos=0 only"
    );
    ensure!(
        layer < config.compress_ratios.len(),
        "layer {layer} out of range"
    );
    let ratio = config.compress_ratios[layer];
    ensure!(ratio > 0, "layer {layer} is not compressed");
    ensure!(ratio != 4, "ratio=4 uses the indexer/overlap path");
    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor weights"))?;
    let compressor_state = cache
        .compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor decode state"))?;

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    copy_window_prefill_to_ring_cache(ctx, &projections.kv, &mut cache.kv, config.sliding_window)?;
    init_nonoverlap_compressor_state_from_prefill(
        ctx,
        config,
        input,
        compressor,
        ratio,
        rope,
        compressor_state,
    )?;
    if input.seq_len < ratio {
        let mut attn_out = sparse_attention_prefill_bf16_hidden(ctx, config, &projections, attn)?;
        return attention_output_project_bf16_hidden(
            ctx,
            &mut attn_out,
            attn,
            rope,
            projections.local_heads,
            projections.head_dim,
            start_pos,
        );
    }
    let compressed_kv = compressor_nonoverlap_prefill_bf16_hidden(
        ctx, config, input, compressor, ratio, rope, start_pos,
    )?;
    copy_bf16_rows_to_cache(
        ctx,
        &compressed_kv,
        &mut cache.kv,
        0,
        config.sliding_window,
        compressed_kv.seq_len,
    )?;
    let kv = concat_seq_bf16_hidden(ctx, &projections.kv, &compressed_kv)?;
    let (topk_idxs, topk) = window_and_compress_topk_indices(
        ctx,
        projections.q.seq_len,
        config.sliding_window,
        ratio,
        projections.kv.seq_len,
    )?;
    let indexed_projections = AttentionProjections {
        qr: projections.qr,
        q: projections.q,
        kv,
        local_heads: projections.local_heads,
        head_dim: projections.head_dim,
    };
    let mut attn_out = indexed_attention_prefill_bf16_hidden(
        ctx,
        config,
        &indexed_projections,
        attn,
        &topk_idxs,
        topk,
    )?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        indexed_projections.local_heads,
        indexed_projections.head_dim,
        start_pos,
    )
}

pub fn attention_prefill_compressed_nonoverlap_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "compressed attention group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input), rope) in ranks.iter().zip(ropes.iter()) {
        out.push(
            attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden(
                ctx, config, input, attn, rope, layer, start_pos,
            )?,
        );
    }
    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub(crate) fn attention_prefill_compressed_nonoverlap_group_bf16_hidden_with_cache(
    ranks: &mut [(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
        &mut LayerDecodeCache,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "compressed attention cache group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input, cache), rope) in ranks.iter_mut().zip(ropes.iter()) {
        out.push(
            attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden_with_cache(
                ctx, config, input, attn, rope, layer, start_pos, cache,
            )?,
        );
    }
    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

fn finish_compressed_overlap_attention_rank_local(
    ctx: &RankGpuContext,
    config: &Config,
    projections: AttentionProjections,
    compressed_kv: Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    topk_idxs: &CudaSlice<i32>,
    topk: usize,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    let kv = concat_seq_bf16_hidden(ctx, &projections.kv, &compressed_kv)?;
    let indexed_projections = AttentionProjections {
        qr: projections.qr,
        q: projections.q,
        kv,
        local_heads: projections.local_heads,
        head_dim: projections.head_dim,
    };
    let mut attn_out = indexed_attention_prefill_bf16_hidden(
        ctx,
        config,
        &indexed_projections,
        attn,
        topk_idxs,
        topk,
    )?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        indexed_projections.local_heads,
        indexed_projections.head_dim,
        start_pos,
    )
}

pub fn attention_prefill_compressed_overlap_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    layer: usize,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        start_pos == 0,
        "ratio-4 overlap attention prefill currently supports start_pos=0 only"
    );
    ensure!(
        layer < config.compress_ratios.len(),
        "layer {layer} out of range"
    );
    ensure!(
        config.compress_ratios[layer] == 4,
        "ratio-4 overlap attention called for layer {layer} with compress_ratio={}",
        config.compress_ratios[layer]
    );
    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor weights"))?;

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    if input.seq_len < 4 {
        let mut attn_out = sparse_attention_prefill_bf16_hidden(ctx, config, &projections, attn)?;
        return attention_output_project_bf16_hidden(
            ctx,
            &mut attn_out,
            attn,
            rope,
            projections.local_heads,
            projections.head_dim,
            start_pos,
        );
    }
    let compressed_kv =
        compressor_overlap_prefill_bf16_hidden(ctx, config, input, compressor, rope, start_pos)?;
    let (topk_idxs, topk) = window_and_compress_topk_indices(
        ctx,
        projections.q.seq_len,
        config.sliding_window,
        4,
        projections.kv.seq_len,
    )?;
    finish_compressed_overlap_attention_rank_local(
        ctx,
        config,
        projections,
        compressed_kv,
        attn,
        rope,
        &topk_idxs,
        topk,
        start_pos,
    )
}

pub fn attention_prefill_compressed_overlap_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "ratio-4 attention group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    if ranks[0].3.seq_len < 4 {
        let mut out = Vec::with_capacity(ranks.len());
        for ((ctx, attn, _comm, input), rope) in ranks.iter().zip(ropes.iter()) {
            out.push(attention_prefill_compressed_overlap_rank_local_bf16_hidden(
                ctx, config, input, attn, rope, layer, start_pos,
            )?);
        }
        let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
            .iter()
            .zip(out.iter_mut())
            .map(|((ctx, _, comm, _), hidden)| (*ctx, *comm, hidden))
            .collect();
        all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
        return Ok(out);
    }

    let mut projections = Vec::with_capacity(ranks.len());
    let mut compressed_kvs = Vec::with_capacity(ranks.len());
    let mut index_scores = Vec::with_capacity(ranks.len());
    let mut compressed_len = None;
    for ((ctx, attn, _comm, input), rope) in ranks.iter().zip(ropes.iter()) {
        ensure!(
            config.compress_ratios[layer] == 4,
            "ratio-4 group attention called for layer {layer} with compress_ratio={}",
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

        let mut rank_projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
        apply_rope_attention_projections(ctx, &mut rank_projections, rope, start_pos)?;
        let rank_compressed_kv = compressor_overlap_prefill_bf16_hidden(
            ctx, config, input, compressor, rope, start_pos,
        )?;
        let (rank_scores, rank_compressed_len) = indexer_scores_prefill_bf16_hidden(
            ctx,
            config,
            input,
            &rank_projections.qr,
            indexer,
            rope,
            start_pos,
        )?;
        if let Some(expected) = compressed_len {
            ensure!(
                rank_compressed_len == expected,
                "indexer compressed len mismatch: expected {}, got {}",
                expected,
                rank_compressed_len
            );
        } else {
            compressed_len = Some(rank_compressed_len);
        }

        projections.push(rank_projections);
        compressed_kvs.push(rank_compressed_kv);
        index_scores.push(rank_scores);
    }

    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for ((_, _, comm, _), scores) in ranks.iter().zip(index_scores.iter_mut()) {
        if let Err(err) = comm.all_reduce_in_place(scores, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!(
                "NCCL indexer score all-reduce failed: {err:?}"
            ));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

    let compressed_len =
        compressed_len.ok_or_else(|| anyhow::anyhow!("ratio-4 group has no compressed len"))?;
    let mut out = Vec::with_capacity(ranks.len());
    for (((((ctx, attn, _comm, input), rope), rank_projections), rank_compressed_kv), scores) in
        ranks
            .iter()
            .zip(ropes.iter())
            .zip(projections.into_iter())
            .zip(compressed_kvs.into_iter())
            .zip(index_scores.iter())
    {
        let (window_idxs, window_topk) =
            window_topk_indices(ctx, input.seq_len, config.sliding_window)?;
        let (compress_idxs, compress_topk) = indexer_topk_indices_prefill(
            ctx,
            config,
            scores,
            input.seq_len,
            compressed_len,
            rank_projections.kv.seq_len,
        )?;
        let topk_idxs = concat_topk_indices(
            ctx,
            &window_idxs,
            window_topk,
            &compress_idxs,
            compress_topk,
            input.seq_len,
        )?;
        out.push(finish_compressed_overlap_attention_rank_local(
            ctx,
            config,
            rank_projections,
            rank_compressed_kv,
            attn,
            rope,
            &topk_idxs,
            window_topk + compress_topk,
            start_pos,
        )?);
    }

    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub(crate) fn attention_prefill_compressed_overlap_group_bf16_hidden_with_cache(
    ranks: &mut [(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
        &mut LayerDecodeCache,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "ratio-4 attention cache group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );

    let mut projections = Vec::with_capacity(ranks.len());
    let mut compressed_kvs = Vec::with_capacity(ranks.len());
    let mut index_scores = Vec::with_capacity(ranks.len());
    let mut compressed_len = None;
    for (((ctx, attn, _comm, input, cache), rope), rank) in
        ranks.iter_mut().zip(ropes.iter()).zip(0..)
    {
        ensure!(
            config.compress_ratios[layer] == 4,
            "ratio-4 group attention cache path called for layer {layer} with compress_ratio={}",
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

        let mut rank_projections = attention_project_bf16_hidden(ctx, config, input, attn)
            .with_context(|| {
                format!("ratio-4 cache attention_project layer {layer} rank {rank}")
            })?;
        apply_rope_attention_projections(ctx, &mut rank_projections, rope, start_pos)
            .with_context(|| format!("ratio-4 cache apply_rope layer {layer} rank {rank}"))?;
        copy_window_prefill_to_ring_cache(
            ctx,
            &rank_projections.kv,
            &mut cache.kv,
            config.sliding_window,
        )
        .with_context(|| format!("ratio-4 cache raw kv layer {layer} rank {rank}"))?;
        init_overlap_compressor_state_from_prefill(
            ctx,
            config,
            input,
            compressor,
            rope,
            config.head_dim,
            compressor_state,
            false,
        )
        .with_context(|| format!("ratio-4 cache compressor tail layer {layer} rank {rank}"))?;
        init_overlap_compressor_state_from_prefill(
            ctx,
            config,
            input,
            &indexer.compressor,
            rope,
            config.index_head_dim,
            indexer_state,
            true,
        )
        .with_context(|| format!("ratio-4 cache indexer tail layer {layer} rank {rank}"))?;

        if input.seq_len < 4 {
            projections.push(rank_projections);
            compressed_kvs.push(None);
            index_scores.push(None);
            continue;
        }

        let rank_compressed_kv = compressor_overlap_prefill_bf16_hidden(
            ctx, config, input, compressor, rope, start_pos,
        )?;
        copy_bf16_rows_to_cache(
            ctx,
            &rank_compressed_kv,
            &mut cache.kv,
            0,
            config.sliding_window,
            rank_compressed_kv.seq_len,
        )
        .with_context(|| format!("ratio-4 cache compressed kv layer {layer} rank {rank}"))?;
        let indexer_compressed_kv = compressor_overlap_prefill_bf16_hidden_with_dim(
            ctx,
            config,
            input,
            &indexer.compressor,
            rope,
            start_pos,
            config.index_head_dim,
        )?;
        copy_bf16_rows_to_cache(
            ctx,
            &indexer_compressed_kv,
            indexer_kv,
            0,
            0,
            indexer_compressed_kv.seq_len,
        )
        .with_context(|| format!("ratio-4 cache indexer kv layer {layer} rank {rank}"))?;
        let (rank_scores, rank_compressed_len) = indexer_scores_prefill_bf16_hidden(
            ctx,
            config,
            input,
            &rank_projections.qr,
            indexer,
            rope,
            start_pos,
        )?;
        if let Some(expected) = compressed_len {
            ensure!(
                rank_compressed_len == expected,
                "indexer compressed len mismatch: expected {}, got {}",
                expected,
                rank_compressed_len
            );
        } else {
            compressed_len = Some(rank_compressed_len);
        }

        projections.push(rank_projections);
        compressed_kvs.push(Some(rank_compressed_kv));
        index_scores.push(Some(rank_scores));
    }

    if let Some(_compressed_len) = compressed_len {
        group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
        for ((_, _, comm, _, _), scores) in ranks.iter().zip(index_scores.iter_mut()) {
            let Some(scores) = scores.as_mut() else {
                let _ = group_end();
                return Err(anyhow::anyhow!("missing rank indexer scores"));
            };
            if let Err(err) = comm.all_reduce_in_place(scores, &ReduceOp::Sum) {
                let _ = group_end();
                return Err(anyhow::anyhow!(
                    "NCCL indexer score all-reduce failed: {err:?}"
                ));
            }
        }
        group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;
    }

    let mut out = Vec::with_capacity(ranks.len());
    for (
        ((((ctx, attn, _comm, input, _cache), rope), rank_projections), rank_compressed_kv),
        scores,
    ) in ranks
        .iter()
        .zip(ropes.iter())
        .zip(projections.into_iter())
        .zip(compressed_kvs.into_iter())
        .zip(index_scores.iter())
    {
        if input.seq_len < 4 {
            let mut attn_out =
                sparse_attention_prefill_bf16_hidden(ctx, config, &rank_projections, attn)?;
            out.push(attention_output_project_bf16_hidden(
                ctx,
                &mut attn_out,
                attn,
                rope,
                rank_projections.local_heads,
                rank_projections.head_dim,
                start_pos,
            )?);
            continue;
        }
        let compressed_len =
            compressed_len.ok_or_else(|| anyhow::anyhow!("ratio-4 group has no compressed len"))?;
        let rank_compressed_kv =
            rank_compressed_kv.ok_or_else(|| anyhow::anyhow!("missing compressed kv"))?;
        let scores = scores
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing indexer scores"))?;
        let (window_idxs, window_topk) =
            window_topk_indices(ctx, input.seq_len, config.sliding_window)?;
        let (compress_idxs, compress_topk) = indexer_topk_indices_prefill(
            ctx,
            config,
            scores,
            input.seq_len,
            compressed_len,
            rank_projections.kv.seq_len,
        )?;
        let topk_idxs = concat_topk_indices(
            ctx,
            &window_idxs,
            window_topk,
            &compress_idxs,
            compress_topk,
            input.seq_len,
        )?;
        out.push(finish_compressed_overlap_attention_rank_local(
            ctx,
            config,
            rank_projections,
            rank_compressed_kv,
            attn,
            rope,
            &topk_idxs,
            window_topk + compress_topk,
            start_pos,
        )?);
    }

    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub fn indexed_attention_prefill_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    projections: &AttentionProjections,
    attn: &AttentionWeights<'_>,
    topk_idxs: &CudaSlice<i32>,
    topk: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(topk > 0, "indexed attention topk must be positive");
    ensure!(
        topk_idxs.len() == projections.q.seq_len * topk,
        "indexed attention topk shape mismatch: expected {}, got {}",
        projections.q.seq_len * topk,
        topk_idxs.len()
    );
    ensure!(
        attn.attn_sink.tensor.dtype == safetensors::Dtype::F32,
        "attn_sink {} must be F32, got {:?}",
        attn.attn_sink.name,
        attn.attn_sink.tensor.dtype
    );
    ensure!(
        attn.attn_sink.tensor.shape == [projections.local_heads],
        "attn_sink {} shape mismatch: expected {:?}, got {:?}",
        attn.attn_sink.name,
        [projections.local_heads],
        attn.attn_sink.tensor.shape
    );

    let mut out = Bf16HiddenStates::zeros(ctx, projections.q.hidden_dim, projections.q.seq_len)?;
    {
        let (q_ptr, _q_guard) = projections.q.data.device_ptr(&ctx.stream);
        let (kv_ptr, _kv_guard) = projections.kv.data.device_ptr(&ctx.stream);
        let (sink_ptr, _sink_guard) = attn.attn_sink.tensor.data.device_ptr(&ctx.stream);
        let (topk_ptr, _topk_guard) = topk_idxs.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_indexed_attention_prefill_cuda(
                q_ptr as *const ffi::Half,
                kv_ptr as *const ffi::Half,
                sink_ptr as *const f32,
                topk_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                projections.q.seq_len as i32,
                projections.kv.seq_len as i32,
                projections.local_heads as i32,
                projections.head_dim as i32,
                topk as i32,
                1.0f32 / (config.head_dim as f32).sqrt(),
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn indexed_attention_cache_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    projections: &AttentionProjections,
    kv_cache: &Bf16Cache,
    attn: &AttentionWeights<'_>,
    topk_idxs: &CudaSlice<i32>,
    topk: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(topk > 0, "indexed cache attention topk must be positive");
    ensure!(
        projections.q.seq_len == 1,
        "indexed cache attention currently expects decode seq_len=1, got {}",
        projections.q.seq_len
    );
    ensure!(
        kv_cache.hidden_dim == projections.head_dim,
        "kv cache hidden dim mismatch: expected {}, got {}",
        projections.head_dim,
        kv_cache.hidden_dim
    );
    ensure!(
        topk_idxs.len() == topk,
        "indexed cache attention topk shape mismatch: expected {}, got {}",
        topk,
        topk_idxs.len()
    );
    ensure!(
        attn.attn_sink.tensor.dtype == safetensors::Dtype::F32,
        "attn_sink {} must be F32, got {:?}",
        attn.attn_sink.name,
        attn.attn_sink.tensor.dtype
    );
    ensure!(
        attn.attn_sink.tensor.shape == [projections.local_heads],
        "attn_sink {} shape mismatch: expected {:?}, got {:?}",
        attn.attn_sink.name,
        [projections.local_heads],
        attn.attn_sink.tensor.shape
    );

    let mut out = Bf16HiddenStates::zeros(ctx, projections.q.hidden_dim, projections.q.seq_len)?;
    {
        let (q_ptr, _q_guard) = projections.q.data.device_ptr(&ctx.stream);
        let (kv_ptr, _kv_guard) = kv_cache.data.device_ptr(&ctx.stream);
        let (sink_ptr, _sink_guard) = attn.attn_sink.tensor.data.device_ptr(&ctx.stream);
        let (topk_ptr, _topk_guard) = topk_idxs.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_indexed_attention_prefill_cuda(
                q_ptr as *const ffi::Half,
                kv_ptr as *const ffi::Half,
                sink_ptr as *const f32,
                topk_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                projections.q.seq_len as i32,
                kv_cache.slots as i32,
                projections.local_heads as i32,
                projections.head_dim as i32,
                topk as i32,
                1.0f32 / (config.head_dim as f32).sqrt(),
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn attention_output_project_bf16_hidden(
    ctx: &RankGpuContext,
    attn_out: &mut Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    local_heads: usize,
    head_dim: usize,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    apply_rope_hidden_in_place(ctx, attn_out, rope, local_heads, head_dim, start_pos, true)?;
    let low_rank = bf16_linear_bf16_hidden(ctx, attn_out, &attn.wo_a)?;
    fp8_linear_bf16_hidden(ctx, &low_rank, &attn.wo_b)
}

pub fn attention_prefill_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        config.compress_ratios[layer] == 0,
        "rank-local prefill attention currently only supports non-compressed layers, layer {layer} has compress_ratio={}",
        config.compress_ratios[layer]
    );
    ensure!(
        start_pos == 0,
        "rank-local prefill attention currently supports start_pos=0 only, got {start_pos}"
    );
    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    let mut attn_out = sparse_attention_prefill_bf16_hidden(ctx, config, &projections, attn)?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )
}

pub(crate) fn attention_prefill_rank_local_bf16_hidden_with_cache(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    kv_cache: &mut Bf16Cache,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        config.compress_ratios[layer] == 0,
        "rank-local prefill attention cache path only supports non-compressed layers, layer {layer} has compress_ratio={}",
        config.compress_ratios[layer]
    );
    ensure!(
        start_pos == 0,
        "rank-local prefill attention cache path currently supports start_pos=0 only, got {start_pos}"
    );
    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    copy_window_prefill_to_ring_cache(ctx, &projections.kv, kv_cache, config.sliding_window)?;
    let mut attn_out = sparse_attention_prefill_bf16_hidden(ctx, config, &projections, attn)?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )
}

pub fn attention_prefill_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "attention group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input), rope) in ranks.iter().zip(ropes.iter()) {
        out.push(attention_prefill_rank_local_bf16_hidden(
            ctx, config, layer, input, attn, rope, start_pos,
        )?);
    }
    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub fn attention_decode_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    kv_cache: &mut Bf16Cache,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        config.compress_ratios[layer] == 0,
        "rank-local decode attention currently only supports non-compressed layers, layer {layer} has compress_ratio={}",
        config.compress_ratios[layer]
    );
    ensure!(
        input.seq_len == 1,
        "rank-local decode attention expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        kv_cache.hidden_dim == config.head_dim,
        "decode kv cache hidden dim mismatch: expected {}, got {}",
        config.head_dim,
        kv_cache.hidden_dim
    );
    ensure!(
        kv_cache.slots >= config.sliding_window,
        "decode kv cache slots {} smaller than sliding_window {}",
        kv_cache.slots,
        config.sliding_window
    );

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)
        .with_context(|| format!("attention_project layer {layer}"))?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)
        .with_context(|| format!("apply_rope_attention_projections layer {layer}"))?;
    copy_bf16_rows_to_cache(
        ctx,
        &projections.kv,
        kv_cache,
        0,
        start_pos % config.sliding_window,
        1,
    )
    .with_context(|| format!("copy kv to cache layer {layer} pos {start_pos}"))?;
    let (topk_idxs, topk) = window_topk_indices_decode(ctx, start_pos, config.sliding_window)
        .with_context(|| format!("window_topk_indices_decode layer {layer} pos {start_pos}"))?;
    let mut attn_out = indexed_attention_cache_bf16_hidden(
        ctx,
        config,
        &projections,
        kv_cache,
        attn,
        &topk_idxs,
        topk,
    )
    .with_context(|| format!("indexed_attention_cache layer {layer} topk {topk}"))?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )
    .with_context(|| format!("attention_output_project layer {layer}"))
}

pub fn attention_decode_group_bf16_hidden(
    ranks: &mut [(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
        &mut Bf16Cache,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "attention decode group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input, kv_cache), rope) in ranks.iter_mut().zip(ropes.iter()) {
        out.push(attention_decode_rank_local_bf16_hidden(
            ctx, config, layer, input, attn, rope, start_pos, kv_cache,
        )?);
    }
    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub(crate) fn attention_decode_compressed_nonoverlap_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
) -> Result<Bf16HiddenStates> {
    ensure!(input.seq_len == 1, "compressed decode expects seq_len=1");
    ensure!(
        layer < config.compress_ratios.len(),
        "compressed decode layer {layer} out of range"
    );
    let ratio = config.compress_ratios[layer];
    ensure!(
        ratio > 0 && ratio != 4,
        "non-overlap decode called for ratio {ratio}"
    );
    ensure!(
        cache.kv.hidden_dim == config.head_dim && cache.kv.slots >= config.sliding_window,
        "compressed decode kv cache shape mismatch: hidden_dim={}, slots={}",
        cache.kv.hidden_dim,
        cache.kv.slots
    );
    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor weights"))?;
    let compressor_state = cache
        .compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor decode state"))?;

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    copy_bf16_rows_to_cache(
        ctx,
        &projections.kv,
        &mut cache.kv,
        0,
        start_pos % config.sliding_window,
        1,
    )?;
    if let Some(compressed_kv) = compressor_nonoverlap_decode_bf16_hidden(
        ctx,
        config,
        input,
        compressor,
        ratio,
        rope,
        start_pos,
        compressor_state,
    )? {
        copy_bf16_rows_to_cache(
            ctx,
            &compressed_kv,
            &mut cache.kv,
            0,
            config.sliding_window + start_pos / ratio,
            1,
        )?;
    }

    let (window_idxs, window_topk) =
        window_topk_indices_decode(ctx, start_pos, config.sliding_window)?;
    let compressed_len = (start_pos + 1) / ratio;
    let (topk_idxs, topk) = if compressed_len > 0 {
        let (compress_idxs, compress_topk) =
            compress_topk_indices_decode(ctx, start_pos, ratio, config.sliding_window)?;
        (
            concat_topk_indices(
                ctx,
                &window_idxs,
                window_topk,
                &compress_idxs,
                compress_topk,
                1,
            )?,
            window_topk + compress_topk,
        )
    } else {
        (window_idxs, window_topk)
    };
    let mut attn_out = indexed_attention_cache_bf16_hidden(
        ctx,
        config,
        &projections,
        &cache.kv,
        attn,
        &topk_idxs,
        topk,
    )?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )
}

pub(crate) fn attention_decode_compressed_nonoverlap_group_bf16_hidden(
    ranks: &mut [(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
        &mut LayerDecodeCache,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "compressed decode group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input, cache), rope) in ranks.iter_mut().zip(ropes.iter()) {
        out.push(
            attention_decode_compressed_nonoverlap_rank_local_bf16_hidden(
                ctx, config, layer, input, attn, rope, start_pos, cache,
            )?,
        );
    }
    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub(crate) fn attention_decode_compressed_overlap_group_bf16_hidden(
    ranks: &mut [(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
        &mut LayerDecodeCache,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "ratio-4 decode group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    ensure!(
        config.compress_ratios[layer] == 4,
        "ratio-4 decode called for layer {layer} with ratio {}",
        config.compress_ratios[layer]
    );

    let mut projections = Vec::with_capacity(ranks.len());
    let mut index_scores: Vec<Option<CudaSlice<f32>>> = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input, cache), rope) in ranks.iter_mut().zip(ropes.iter()) {
        ensure!(input.seq_len == 1, "ratio-4 decode expects seq_len=1");
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

        let mut rank_projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
        apply_rope_attention_projections(ctx, &mut rank_projections, rope, start_pos)?;
        copy_bf16_rows_to_cache(
            ctx,
            &rank_projections.kv,
            &mut cache.kv,
            0,
            start_pos % config.sliding_window,
            1,
        )?;
        if let Some(compressed_kv) = compressor_overlap_decode_bf16_hidden(
            ctx,
            config,
            input,
            compressor,
            rope,
            start_pos,
            compressor_state,
        )? {
            copy_bf16_rows_to_cache(
                ctx,
                &compressed_kv,
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
        let scores = indexer_scores_decode_bf16_hidden(
            ctx,
            config,
            input,
            &rank_projections.qr,
            indexer,
            rope,
            start_pos,
            indexer_kv,
            indexer_state,
        )?;
        projections.push(rank_projections);
        index_scores.push(scores);
    }

    if let Some(compressed_len) = index_scores
        .iter()
        .find_map(|scores| scores.as_ref().map(|scores| scores.len()))
    {
        group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
        for ((_, _, comm, _, _), scores) in ranks.iter().zip(index_scores.iter_mut()) {
            let Some(scores) = scores.as_mut() else {
                let _ = group_end();
                return Err(anyhow::anyhow!(
                    "missing rank indexer scores for compressed_len={compressed_len}"
                ));
            };
            if scores.len() != compressed_len {
                let _ = group_end();
                return Err(anyhow::anyhow!(
                    "indexer decode score len mismatch: expected {}, got {}",
                    compressed_len,
                    scores.len()
                ));
            }
            if let Err(err) = comm.all_reduce_in_place(scores, &ReduceOp::Sum) {
                let _ = group_end();
                return Err(anyhow::anyhow!(
                    "NCCL decode indexer score all-reduce failed: {err:?}"
                ));
            }
        }
        group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;
    }

    let mut out = Vec::with_capacity(ranks.len());
    for ((((ctx, attn, _comm, _input, cache), rope), rank_projections), scores) in ranks
        .iter_mut()
        .zip(ropes.iter())
        .zip(projections.into_iter())
        .zip(index_scores.iter())
    {
        let (window_idxs, window_topk) =
            window_topk_indices_decode(ctx, start_pos, config.sliding_window)?;
        let compressed_len = (start_pos + 1) / 4;
        let (topk_idxs, topk) = if compressed_len > 0 {
            let scores = scores
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing indexer decode scores"))?;
            let (compress_idxs, compress_topk) = indexer_topk_indices_decode(
                ctx,
                config,
                scores,
                compressed_len,
                config.sliding_window,
            )?;
            (
                concat_topk_indices(
                    ctx,
                    &window_idxs,
                    window_topk,
                    &compress_idxs,
                    compress_topk,
                    1,
                )?,
                window_topk + compress_topk,
            )
        } else {
            (window_idxs, window_topk)
        };
        let mut attn_out = indexed_attention_cache_bf16_hidden(
            ctx,
            config,
            &rank_projections,
            &cache.kv,
            attn,
            &topk_idxs,
            topk,
        )?;
        out.push(attention_output_project_bf16_hidden(
            ctx,
            &mut attn_out,
            attn,
            rope,
            rank_projections.local_heads,
            rank_projections.head_dim,
            start_pos,
        )?);
    }

    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}
