use super::*;

pub fn block_prefill_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
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
    let ffn_out = moe_rank_local_bf16_hidden(ctx, config, weights, layer, &ffn_norm, token_ids)?;
    hc_post_bf16_hidden(ctx, &ffn_out, &after_attn, &ffn_hc)
}

pub fn block_prefill_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &HcHiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<HcHiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "block prefill group must contain at least one rank"
    );
    ensure!(
        ranks.len() == ropes.len(),
        "block prefill ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    ensure!(
        layer < config.n_layers,
        "group block prefill layer {layer} out of range"
    );

    let blocks = ranks
        .iter()
        .enumerate()
        .map(|(rank, (_, weights, _, _, _))| {
            weights
                .block(layer)
                .with_context(|| format!("load block view layer {layer} rank {rank}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut attn_inputs = Vec::with_capacity(ranks.len());
    let mut attn_hc = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, input, _), block) in ranks.iter().zip(blocks.iter()) {
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_attn_fn,
            &block.hc_attn_scale,
            &block.hc_attn_base,
        )?;
        attn_inputs.push(pre);
        attn_hc.push(state);
    }

    let mut attn_norms = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (block, attn_input)) in
        ranks.iter().zip(blocks.iter().zip(attn_inputs.iter()))
    {
        attn_norms.push(rms_norm_bf16_hidden(
            ctx,
            attn_input,
            &block.attn_norm,
            config.rms_norm_eps,
        )?);
    }

    let attention_group = ranks
        .iter()
        .zip(blocks.iter())
        .zip(attn_norms.iter())
        .map(|(((ctx, _, comm, _, _), block), attn_norm)| (*ctx, &block.attn, *comm, attn_norm))
        .collect::<Vec<_>>();
    let attn_out = match config.compress_ratios[layer] {
        0 => {
            attention_prefill_group_bf16_hidden(&attention_group, config, layer, ropes, start_pos)?
        }
        4 => attention_prefill_compressed_overlap_group_bf16_hidden(
            &attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )?,
        _ => attention_prefill_compressed_nonoverlap_group_bf16_hidden(
            &attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )?,
    };

    let mut after_attn = Vec::with_capacity(ranks.len());
    for (rank, (((ctx, _, _, input, _), attn_out), state)) in ranks
        .iter()
        .zip(attn_out.iter())
        .zip(attn_hc.iter())
        .enumerate()
    {
        after_attn.push(
            hc_post_bf16_hidden(ctx, attn_out, input, state)
                .with_context(|| format!("hc_post attention layer {layer} rank {rank}"))?,
        );
    }

    let mut ffn_inputs = Vec::with_capacity(ranks.len());
    let mut ffn_hc = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, _, _), (block, input))) in ranks
        .iter()
        .zip(blocks.iter().zip(after_attn.iter()))
        .enumerate()
    {
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_ffn_fn,
            &block.hc_ffn_scale,
            &block.hc_ffn_base,
        )
        .with_context(|| format!("hc_pre ffn layer {layer} rank {rank}"))?;
        ffn_inputs.push(pre);
        ffn_hc.push(state);
    }

    let mut ffn_norms = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, _, _), (block, ffn_input))) in ranks
        .iter()
        .zip(blocks.iter().zip(ffn_inputs.iter()))
        .enumerate()
    {
        ffn_norms.push(
            rms_norm_bf16_hidden(ctx, ffn_input, &block.ffn_norm, config.rms_norm_eps)
                .with_context(|| format!("ffn rms_norm layer {layer} rank {rank}"))?,
        );
    }

    let moe_group = ranks
        .iter()
        .zip(ffn_norms.iter())
        .map(|((ctx, weights, comm, _, token_ids), ffn_norm)| {
            (*ctx, *weights, *comm, ffn_norm, *token_ids)
        })
        .collect::<Vec<_>>();
    let ffn_out = moe_group_bf16_hidden(&moe_group, config, layer)
        .with_context(|| format!("moe_group_bf16_hidden layer {layer}"))?;

    let mut out = Vec::with_capacity(ranks.len());
    for (rank, (((ctx, _, _, _, _), ffn_out), (input, state))) in ranks
        .iter()
        .zip(ffn_out.iter())
        .zip(after_attn.iter().zip(ffn_hc.iter()))
        .enumerate()
    {
        out.push(
            hc_post_bf16_hidden(ctx, ffn_out, input, state)
                .with_context(|| format!("hc_post ffn layer {layer} rank {rank}"))?,
        );
    }
    Ok(out)
}

pub(crate) fn block_prefill_group_bf16_hidden_with_decode_cache(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &HcHiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
    caches: &mut [LayerDecodeCache],
) -> Result<Vec<HcHiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "block prefill cache group must contain at least one rank"
    );
    ensure!(
        ranks.len() == ropes.len(),
        "block prefill cache ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    ensure!(
        ranks.len() == caches.len(),
        "block prefill cache ranks/cache length mismatch: ranks={}, caches={}",
        ranks.len(),
        caches.len()
    );
    ensure!(
        layer < config.n_layers,
        "group block prefill cache layer {layer} out of range"
    );

    let blocks = ranks
        .iter()
        .enumerate()
        .map(|(rank, (_, weights, _, _, _))| {
            weights
                .block(layer)
                .with_context(|| format!("load block view layer {layer} rank {rank}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut attn_inputs = Vec::with_capacity(ranks.len());
    let mut attn_hc = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, input, _), block) in ranks.iter().zip(blocks.iter()) {
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_attn_fn,
            &block.hc_attn_scale,
            &block.hc_attn_base,
        )?;
        attn_inputs.push(pre);
        attn_hc.push(state);
    }

    let mut attn_norms = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (block, attn_input)) in
        ranks.iter().zip(blocks.iter().zip(attn_inputs.iter()))
    {
        attn_norms.push(rms_norm_bf16_hidden(
            ctx,
            attn_input,
            &block.attn_norm,
            config.rms_norm_eps,
        )?);
    }

    let mut attention_group = ranks
        .iter()
        .zip(blocks.iter())
        .zip(attn_norms.iter())
        .zip(caches.iter_mut())
        .map(|((((ctx, _, comm, _, _), block), attn_norm), cache)| {
            (*ctx, &block.attn, *comm, attn_norm, cache)
        })
        .collect::<Vec<_>>();
    let attn_out = match config.compress_ratios[layer] {
        0 => {
            let mut out = Vec::with_capacity(attention_group.len());
            for ((ctx, attn, _comm, input, cache), rope) in
                attention_group.iter_mut().zip(ropes.iter())
            {
                out.push(attention_prefill_rank_local_bf16_hidden_with_cache(
                    ctx,
                    config,
                    layer,
                    input,
                    attn,
                    rope,
                    start_pos,
                    &mut cache.kv,
                )?);
            }
            let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> =
                attention_group
                    .iter()
                    .zip(out.iter_mut())
                    .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
                    .collect();
            all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
            out
        }
        4 => attention_prefill_compressed_overlap_group_bf16_hidden_with_cache(
            &mut attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )?,
        _ => attention_prefill_compressed_nonoverlap_group_bf16_hidden_with_cache(
            &mut attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )?,
    };

    let mut after_attn = Vec::with_capacity(ranks.len());
    for (rank, (((ctx, _, _, input, _), attn_out), state)) in ranks
        .iter()
        .zip(attn_out.iter())
        .zip(attn_hc.iter())
        .enumerate()
    {
        after_attn.push(
            hc_post_bf16_hidden(ctx, attn_out, input, state)
                .with_context(|| format!("hc_post attention layer {layer} rank {rank}"))?,
        );
    }

    let mut ffn_inputs = Vec::with_capacity(ranks.len());
    let mut ffn_hc = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, _, _), (block, input))) in ranks
        .iter()
        .zip(blocks.iter().zip(after_attn.iter()))
        .enumerate()
    {
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_ffn_fn,
            &block.hc_ffn_scale,
            &block.hc_ffn_base,
        )
        .with_context(|| format!("hc_pre ffn layer {layer} rank {rank}"))?;
        ffn_inputs.push(pre);
        ffn_hc.push(state);
    }

    let mut ffn_norms = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, _, _), (block, ffn_input))) in ranks
        .iter()
        .zip(blocks.iter().zip(ffn_inputs.iter()))
        .enumerate()
    {
        ffn_norms.push(
            rms_norm_bf16_hidden(ctx, ffn_input, &block.ffn_norm, config.rms_norm_eps)
                .with_context(|| format!("ffn rms_norm layer {layer} rank {rank}"))?,
        );
    }

    let moe_group = ranks
        .iter()
        .zip(ffn_norms.iter())
        .map(|((ctx, weights, comm, _, token_ids), ffn_norm)| {
            (*ctx, *weights, *comm, ffn_norm, *token_ids)
        })
        .collect::<Vec<_>>();
    let ffn_out = moe_group_bf16_hidden(&moe_group, config, layer)
        .with_context(|| format!("moe_group_bf16_hidden layer {layer}"))?;

    let mut out = Vec::with_capacity(ranks.len());
    for (rank, (((ctx, _, _, _, _), ffn_out), (input, state))) in ranks
        .iter()
        .zip(ffn_out.iter())
        .zip(after_attn.iter().zip(ffn_hc.iter()))
        .enumerate()
    {
        out.push(
            hc_post_bf16_hidden(ctx, ffn_out, input, state)
                .with_context(|| format!("hc_post ffn layer {layer} rank {rank}"))?,
        );
    }
    Ok(out)
}

pub fn block_decode_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &HcHiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
    caches: &mut [LayerDecodeCache],
) -> Result<Vec<HcHiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "block decode group must contain at least one rank"
    );
    ensure!(
        ranks.len() == ropes.len(),
        "block decode ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    ensure!(
        ranks.len() == caches.len(),
        "block decode ranks/cache length mismatch: ranks={}, caches={}",
        ranks.len(),
        caches.len()
    );
    ensure!(
        layer < config.n_layers,
        "group block decode layer {layer} out of range"
    );
    let blocks = ranks
        .iter()
        .map(|(_, weights, _, _, _)| weights.block(layer))
        .collect::<Result<Vec<_>>>()?;

    let mut attn_inputs = Vec::with_capacity(ranks.len());
    let mut attn_hc = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, input, _), block)) in ranks.iter().zip(blocks.iter()).enumerate() {
        ensure!(
            input.seq_len == 1,
            "block decode expects HC seq_len=1, got {}",
            input.seq_len
        );
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_attn_fn,
            &block.hc_attn_scale,
            &block.hc_attn_base,
        )
        .with_context(|| format!("hc_pre attention layer {layer} rank {rank}"))?;
        attn_inputs.push(pre);
        attn_hc.push(state);
    }

    let mut attn_norms = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, _, _), (block, attn_input))) in ranks
        .iter()
        .zip(blocks.iter().zip(attn_inputs.iter()))
        .enumerate()
    {
        attn_norms.push(
            rms_norm_bf16_hidden(ctx, attn_input, &block.attn_norm, config.rms_norm_eps)
                .with_context(|| format!("attention rms_norm layer {layer} rank {rank}"))?,
        );
    }

    let mut attention_group = ranks
        .iter()
        .zip(blocks.iter())
        .zip(attn_norms.iter())
        .zip(caches.iter_mut())
        .map(|((((ctx, _, comm, _, _), block), attn_norm), cache)| {
            (*ctx, &block.attn, *comm, attn_norm, cache)
        })
        .collect::<Vec<_>>();
    let attn_out = match config.compress_ratios[layer] {
        0 => {
            let mut out = Vec::with_capacity(attention_group.len());
            for (rank, ((ctx, attn, _comm, input, cache), rope)) in
                attention_group.iter_mut().zip(ropes.iter()).enumerate()
            {
                out.push(
                    attention_decode_rank_local_bf16_hidden(
                        ctx,
                        config,
                        layer,
                        input,
                        attn,
                        rope,
                        start_pos,
                        &mut cache.kv,
                    )
                    .with_context(|| {
                        format!("attention_decode_rank_local layer {layer} rank {rank}")
                    })?,
                );
            }
            let mut attn_reduce: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> =
                attention_group
                    .iter()
                    .zip(out.iter_mut())
                    .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
                    .collect();
            all_reduce_hidden_group_fp32(&mut attn_reduce)
                .with_context(|| format!("attention all_reduce layer {layer}"))?;
            out
        }
        4 => attention_decode_compressed_overlap_group_bf16_hidden(
            &mut attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )
        .with_context(|| format!("attention_decode_compressed_overlap layer {layer}"))?,
        _ => attention_decode_compressed_nonoverlap_group_bf16_hidden(
            &mut attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )
        .with_context(|| format!("attention_decode_compressed_nonoverlap layer {layer}"))?,
    };

    let mut after_attn = Vec::with_capacity(ranks.len());
    for (((ctx, _, _, input, _), attn_out), state) in
        ranks.iter().zip(attn_out.iter()).zip(attn_hc.iter())
    {
        after_attn.push(hc_post_bf16_hidden(ctx, attn_out, input, state)?);
    }

    let mut ffn_inputs = Vec::with_capacity(ranks.len());
    let mut ffn_hc = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (block, input)) in
        ranks.iter().zip(blocks.iter().zip(after_attn.iter()))
    {
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_ffn_fn,
            &block.hc_ffn_scale,
            &block.hc_ffn_base,
        )?;
        ffn_inputs.push(pre);
        ffn_hc.push(state);
    }

    let mut ffn_norms = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (block, ffn_input)) in
        ranks.iter().zip(blocks.iter().zip(ffn_inputs.iter()))
    {
        ffn_norms.push(rms_norm_bf16_hidden(
            ctx,
            ffn_input,
            &block.ffn_norm,
            config.rms_norm_eps,
        )?);
    }

    let moe_group = ranks
        .iter()
        .zip(ffn_norms.iter())
        .map(|((ctx, weights, comm, _, token_ids), ffn_norm)| {
            (*ctx, *weights, *comm, ffn_norm, *token_ids)
        })
        .collect::<Vec<_>>();
    let ffn_out = moe_group_bf16_hidden(&moe_group, config, layer)?;

    let mut out = Vec::with_capacity(ranks.len());
    for (((ctx, _, _, _, _), ffn_out), (input, state)) in ranks
        .iter()
        .zip(ffn_out.iter())
        .zip(after_attn.iter().zip(ffn_hc.iter()))
    {
        out.push(hc_post_bf16_hidden(ctx, ffn_out, input, state)?);
    }
    Ok(out)
}

pub fn prefill_logits_group_bf16_hidden(
    ranks: &[(&RankGpuContext, &RankWeightView<'_>, &Comm, &CudaSlice<u32>)],
    config: &Config,
    seq_len: usize,
) -> Result<Vec<F32Logits>> {
    ensure!(
        !ranks.is_empty(),
        "full prefill group must contain at least one rank"
    );
    ensure!(seq_len > 0, "full prefill seq_len must be positive");

    let hidden = embedding_vocab_parallel_group(ranks, config, seq_len)?;
    let mut hcs = ranks
        .iter()
        .zip(hidden.iter())
        .map(|((ctx, _, _, _), hidden)| hc_expand_bf16_hidden(ctx, hidden, config.hc_mult))
        .collect::<Result<Vec<_>>>()?;

    for layer in 0..config.n_layers {
        let ropes = ranks
            .iter()
            .map(|(ctx, _, _, _)| precompute_rope_cache(ctx, config, layer, seq_len))
            .collect::<Result<Vec<_>>>()?;
        let rope_refs = ropes.iter().collect::<Vec<_>>();
        let block_inputs = ranks
            .iter()
            .zip(hcs.iter())
            .map(|((ctx, weights, comm, token_ids), hc)| (*ctx, *weights, *comm, hc, *token_ids))
            .collect::<Vec<_>>();
        hcs = block_prefill_group_bf16_hidden(&block_inputs, config, layer, &rope_refs, 0)?;
    }

    let logits_inputs = ranks
        .iter()
        .zip(hcs.iter())
        .map(|((ctx, weights, comm, _), hc)| (*ctx, *weights, *comm, hc))
        .collect::<Vec<_>>();
    final_logits_group_bf16_hidden(&logits_inputs, config)
}

pub fn prefill_logits_and_decode_cache_group_bf16_hidden(
    ranks: &[(&RankGpuContext, &RankWeightView<'_>, &Comm, &CudaSlice<u32>)],
    config: &Config,
    seq_len: usize,
    caches: &mut [Vec<LayerDecodeCache>],
) -> Result<Vec<F32Logits>> {
    ensure!(
        !ranks.is_empty(),
        "full prefill cache group must contain at least one rank"
    );
    ensure!(seq_len > 0, "full prefill cache seq_len must be positive");
    ensure!(
        caches.len() == config.n_layers,
        "prefill cache layer count mismatch: have {}, need {}",
        caches.len(),
        config.n_layers
    );
    for (layer, rank_caches) in caches.iter().enumerate() {
        ensure!(
            rank_caches.len() == ranks.len(),
            "prefill cache rank count mismatch at layer {layer}: have {}, need {}",
            rank_caches.len(),
            ranks.len()
        );
    }

    let hidden = embedding_vocab_parallel_group(ranks, config, seq_len)?;
    let mut hcs = ranks
        .iter()
        .zip(hidden.iter())
        .map(|((ctx, _, _, _), hidden)| hc_expand_bf16_hidden(ctx, hidden, config.hc_mult))
        .collect::<Result<Vec<_>>>()?;

    for (layer, layer_caches) in caches.iter_mut().enumerate().take(config.n_layers) {
        let ropes = ranks
            .iter()
            .map(|(ctx, _, _, _)| precompute_rope_cache(ctx, config, layer, seq_len))
            .collect::<Result<Vec<_>>>()?;
        let rope_refs = ropes.iter().collect::<Vec<_>>();
        let block_inputs = ranks
            .iter()
            .zip(hcs.iter())
            .map(|((ctx, weights, comm, token_ids), hc)| (*ctx, *weights, *comm, hc, *token_ids))
            .collect::<Vec<_>>();
        hcs = block_prefill_group_bf16_hidden_with_decode_cache(
            &block_inputs,
            config,
            layer,
            &rope_refs,
            0,
            layer_caches,
        )?;
    }

    let logits_inputs = ranks
        .iter()
        .zip(hcs.iter())
        .map(|((ctx, weights, comm, _), hc)| (*ctx, *weights, *comm, hc))
        .collect::<Vec<_>>();
    final_logits_group_bf16_hidden(&logits_inputs, config)
}
