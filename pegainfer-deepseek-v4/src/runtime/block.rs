use super::*;

use std::thread;

struct RankComm(usize);

// SAFETY: Each NCCL communicator is used only by its owning rank lane. The
// parallel block path sends one distinct communicator reference to one scoped
// thread, matching cudarc's TP worker ownership pattern.
unsafe impl Send for RankComm {}

impl RankComm {
    fn new(comm: &Comm) -> Self {
        Self(std::ptr::from_ref(comm) as usize)
    }

    fn get(self) -> &'static Comm {
        // SAFETY: the pointer comes from a `Comm` borrowed by a scoped thread.
        // The caller only uses it before the scope exits.
        unsafe { &*(self.0 as *const Comm) }
    }
}

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

pub fn block_decode_group_rank_threads_bf16_hidden(
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
        "rank-thread block decode group must contain at least one rank"
    );
    ensure!(
        ranks.len() == ropes.len(),
        "rank-thread block decode ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    ensure!(
        ranks.len() == caches.len(),
        "rank-thread block decode ranks/cache length mismatch: ranks={}, caches={}",
        ranks.len(),
        caches.len()
    );
    ensure!(
        layer < config.n_layers,
        "rank-thread group block decode layer {layer} out of range"
    );

    let mut out = Vec::with_capacity(ranks.len());
    thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(ranks.len());
        for (rank, (((ctx, weights, comm, input, token_ids), rope), cache)) in ranks
            .iter()
            .zip(ropes.iter())
            .zip(caches.iter_mut())
            .enumerate()
        {
            let comm = RankComm::new(comm);
            handles.push(scope.spawn(move || -> Result<(usize, HcHiddenStates)> {
                let comm = comm.get();
                let hidden = block_decode_rank_lane_bf16_hidden(
                    ctx, weights, comm, config, layer, input, token_ids, rope, start_pos, cache,
                )
                .with_context(|| format!("rank-thread block decode layer {layer} rank {rank}"))?;
                Ok((rank, hidden))
            }));
        }

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            results.push(
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("rank-thread block decode worker panicked"))??,
            );
        }
        results.sort_by_key(|(rank, _)| *rank);
        for (_, hidden) in results {
            out.push(hidden);
        }
        Ok(())
    })?;
    Ok(out)
}

pub fn block_decode_rank_lane_bf16_hidden(
    ctx: &RankGpuContext,
    weights: &RankWeightView<'_>,
    comm: &Comm,
    config: &Config,
    layer: usize,
    input: &HcHiddenStates,
    token_ids: &CudaSlice<u32>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
) -> Result<HcHiddenStates> {
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
    let mut attn_out = attention_decode_rank_local_collective_bf16_hidden(
        ctx,
        config,
        layer,
        &attn_norm,
        &block.attn,
        rope,
        start_pos,
        cache,
        comm,
    )
    .with_context(|| format!("attention decode layer {layer}"))?;
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

    let ffn = weights.ffn(layer)?;
    let routed = if layer < config.n_hash_layers {
        hash_route_bf16_hidden(ctx, config, &ffn_norm, &ffn, token_ids)?
    } else {
        score_route_bf16_hidden(ctx, config, &ffn_norm, &ffn)?
    };
    let mut routed_out =
        decode_routed_moe_rank_local_f32_hidden(ctx, config, weights, layer, &ffn_norm, &routed)?;
    let shared_out = shared_expert_forward_bf16_hidden(ctx, &ffn_norm, &ffn, config.swiglu_limit)?;
    all_reduce_f32_hidden_in_place(&mut routed_out, comm)
        .with_context(|| format!("moe routed all_reduce layer {layer}"))?;
    let ffn_out = add_f32_bf16_to_bf16_hidden(ctx, &routed_out, &shared_out)?;
    hc_post_bf16_hidden(ctx, &ffn_out, &after_attn, &ffn_hc)
        .with_context(|| format!("hc_post ffn layer {layer}"))
}

fn attention_decode_rank_local_collective_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
    comm: &Comm,
) -> Result<Bf16HiddenStates> {
    match config.compress_ratios[layer] {
        0 => attention_decode_rank_local_bf16_hidden(
            ctx,
            config,
            layer,
            input,
            attn,
            rope,
            start_pos,
            &mut cache.kv,
        ),
        4 => attention_decode_compressed_overlap_rank_local_collective_bf16_hidden(
            ctx, config, layer, input, attn, rope, start_pos, cache, comm,
        ),
        _ => attention_decode_compressed_nonoverlap_rank_local_bf16_hidden(
            ctx, config, layer, input, attn, rope, start_pos, cache,
        ),
    }
}

fn attention_decode_compressed_overlap_rank_local_collective_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
    comm: &Comm,
) -> Result<Bf16HiddenStates> {
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
    let mut scores = indexer_scores_decode_bf16_hidden(
        ctx,
        config,
        input,
        &projections.qr,
        indexer,
        rope,
        start_pos,
        indexer_kv,
        indexer_state,
    )?;

    if let Some(scores) = scores.as_mut() {
        comm.all_reduce_in_place(scores, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("NCCL decode indexer score all-reduce failed: {err:?}")
            })?;
    }

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
    ropes: &[Vec<DeepSeekRopeCache>],
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
    ensure!(
        ropes.len() == config.n_layers,
        "prefill rope layer count mismatch: have {}, need {}",
        ropes.len(),
        config.n_layers
    );
    for (layer, rank_ropes) in ropes.iter().enumerate() {
        ensure!(
            rank_ropes.len() == ranks.len(),
            "prefill rope rank count mismatch at layer {layer}: have {}, need {}",
            rank_ropes.len(),
            ranks.len()
        );
        for (rank, rope) in rank_ropes.iter().enumerate() {
            ensure!(
                rope.max_seq_len >= seq_len,
                "prefill rope cache too short at layer {layer} rank {rank}: have {}, need {}",
                rope.max_seq_len,
                seq_len
            );
        }
    }

    let hidden = embedding_vocab_parallel_group(ranks, config, seq_len)?;
    let mut hcs = ranks
        .iter()
        .zip(hidden.iter())
        .map(|((ctx, _, _, _), hidden)| hc_expand_bf16_hidden(ctx, hidden, config.hc_mult))
        .collect::<Result<Vec<_>>>()?;

    for (layer, layer_caches) in caches.iter_mut().enumerate().take(config.n_layers) {
        let rope_refs = ropes[layer].iter().collect::<Vec<_>>();
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
