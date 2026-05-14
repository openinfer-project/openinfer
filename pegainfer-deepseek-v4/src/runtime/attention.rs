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

pub(crate) fn attention_prefill_compressed_overlap_rank_local_collective_bf16_hidden_with_cache(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    layer: usize,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
    comm: &Comm,
    prefill_window_topk: &PrefillWindowTopk,
) -> Result<Bf16HiddenStates> {
    ensure!(
        config.compress_ratios[layer] == 4,
        "ratio-4 rank-lane attention cache path called for layer {layer} with compress_ratio={}",
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

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)
        .with_context(|| format!("ratio-4 rank-lane attention_project layer {layer}"))?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)
        .with_context(|| format!("ratio-4 rank-lane apply_rope layer {layer}"))?;
    copy_window_prefill_to_ring_cache(ctx, &projections.kv, &mut cache.kv, config.sliding_window)
        .with_context(|| format!("ratio-4 rank-lane raw kv layer {layer}"))?;
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
    .with_context(|| format!("ratio-4 rank-lane compressor tail layer {layer}"))?;
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
    .with_context(|| format!("ratio-4 rank-lane indexer tail layer {layer}"))?;

    if input.seq_len < 4 {
        let (topk_idxs, topk) =
            prefill_window_topk.parts(projections.q.seq_len, config.sliding_window)?;
        let mut attn_out = sparse_attention_prefill_bf16_hidden_with_topk(
            ctx,
            config,
            &projections,
            attn,
            topk_idxs,
            topk,
        )?;
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
    copy_bf16_rows_to_cache(
        ctx,
        &compressed_kv,
        &mut cache.kv,
        0,
        config.sliding_window,
        compressed_kv.seq_len,
    )
    .with_context(|| format!("ratio-4 rank-lane compressed kv layer {layer}"))?;
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
    .with_context(|| format!("ratio-4 rank-lane indexer kv layer {layer}"))?;
    let (mut scores, compressed_len) = indexer_scores_prefill_bf16_hidden(
        ctx,
        config,
        input,
        &projections.qr,
        indexer,
        rope,
        start_pos,
    )?;
    comm.all_reduce_in_place(&mut scores, &ReduceOp::Sum)
        .map_err(|err| anyhow::anyhow!("NCCL indexer score all-reduce failed: {err:?}"))?;

    let (window_idxs, window_topk) =
        prefill_window_topk.parts(input.seq_len, config.sliding_window)?;
    let (compress_idxs, compress_topk) = indexer_topk_indices_prefill(
        ctx,
        config,
        &scores,
        input.seq_len,
        compressed_len,
        projections.kv.seq_len,
    )?;
    let topk_idxs = concat_topk_indices(
        ctx,
        &window_idxs,
        window_topk,
        &compress_idxs,
        compress_topk,
        input.seq_len,
    )?;
    finish_compressed_overlap_attention_rank_local(
        ctx,
        config,
        projections,
        compressed_kv,
        attn,
        rope,
        &topk_idxs,
        window_topk + compress_topk,
        start_pos,
    )
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
        kv_cache.hidden_dim == projections.head_dim,
        "kv cache hidden dim mismatch: expected {}, got {}",
        projections.head_dim,
        kv_cache.hidden_dim
    );
    ensure!(
        topk_idxs.len() >= projections.q.seq_len * topk,
        "indexed cache attention topk capacity too small: need {}, have {}",
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
    indexed_attention_cache_bf16_hidden_into(
        ctx,
        config,
        projections,
        kv_cache,
        attn,
        topk_idxs,
        topk,
        &mut out,
    )?;
    Ok(out)
}

pub(crate) fn indexed_attention_cache_bf16_hidden_into(
    ctx: &RankGpuContext,
    config: &Config,
    projections: &AttentionProjections,
    kv_cache: &Bf16Cache,
    attn: &AttentionWeights<'_>,
    topk_idxs: &CudaSlice<i32>,
    topk: usize,
    out: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(topk > 0, "indexed cache attention topk must be positive");
    ensure!(
        kv_cache.hidden_dim == projections.head_dim,
        "kv cache hidden dim mismatch: expected {}, got {}",
        projections.head_dim,
        kv_cache.hidden_dim
    );
    ensure!(
        topk_idxs.len() >= projections.q.seq_len * topk,
        "indexed cache attention topk capacity too small: need {}, have {}",
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
    ensure!(
        out.hidden_dim == projections.q.hidden_dim,
        "indexed cache attention output dim mismatch: expected {}, got {}",
        projections.q.hidden_dim,
        out.hidden_dim
    );
    ensure!(
        out.data.len() >= projections.q.hidden_dim * projections.q.seq_len,
        "indexed cache attention output capacity too small: need {}, have {}",
        projections.q.hidden_dim * projections.q.seq_len,
        out.data.len()
    );
    out.seq_len = projections.q.seq_len;
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
    Ok(())
}

pub(crate) fn indexed_attention_cache_bf16_hidden_view_into(
    ctx: &RankGpuContext,
    config: &Config,
    projections: &AttentionProjectionsView<'_>,
    kv_cache: &Bf16Cache,
    attn: &AttentionWeights<'_>,
    topk_idxs: &CudaSlice<i32>,
    topk: usize,
    out: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(topk > 0, "indexed cache attention topk must be positive");
    ensure!(
        kv_cache.hidden_dim == projections.head_dim,
        "kv cache hidden dim mismatch: expected {}, got {}",
        projections.head_dim,
        kv_cache.hidden_dim
    );
    ensure!(
        topk_idxs.len() >= topk * projections.q.seq_len,
        "indexed cache attention topk capacity too small: need {}, have {}",
        topk * projections.q.seq_len,
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
    ensure!(
        out.hidden_dim == projections.q.hidden_dim,
        "indexed cache attention output dim mismatch: expected {}, got {}",
        projections.q.hidden_dim,
        out.hidden_dim
    );
    ensure!(
        out.data.len() >= projections.q.hidden_dim * projections.q.seq_len,
        "indexed cache attention output capacity too small: need {}, have {}",
        projections.q.hidden_dim * projections.q.seq_len,
        out.data.len()
    );
    out.seq_len = projections.q.seq_len;
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
    Ok(())
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

pub(crate) fn attention_output_project_bf16_hidden_scratch<'a>(
    ctx: &RankGpuContext,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    scratch: &'a mut AttentionOutputScratch,
) -> Result<&'a Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        scratch.attn_out.seq_len <= scratch.seq_capacity,
        "attention output scratch logical seq_len {} exceeds capacity {}",
        scratch.attn_out.seq_len,
        scratch.seq_capacity
    );
    apply_rope_hidden_in_place(
        ctx,
        &mut scratch.attn_out,
        rope,
        scratch.local_heads,
        scratch.head_dim,
        start_pos,
        true,
    )?;
    bf16_linear_bf16_hidden_into(ctx, &scratch.attn_out, &attn.wo_a, &mut scratch.low_rank)?;
    fp8_linear_bf16_hidden_into(ctx, &scratch.low_rank, &attn.wo_b, &mut scratch.out)?;
    Ok(&scratch.out)
}

pub(crate) fn attention_output_project_bf16_hidden_batch_scratch<'a>(
    ctx: &RankGpuContext,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: &CudaSlice<i32>,
    batch: usize,
    scratch: &'a mut AttentionOutputScratch,
) -> Result<&'a Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(batch > 0, "attention output batch must be positive");
    ensure!(
        scratch.attn_out.seq_len == batch,
        "attention output batch mismatch: attn_out seq_len={}, batch={batch}",
        scratch.attn_out.seq_len
    );
    ensure!(
        start_pos.len() >= batch,
        "attention output start_pos capacity too small: need {}, have {}",
        batch,
        start_pos.len()
    );
    ensure!(
        batch <= scratch.seq_capacity,
        "attention output scratch capacity too small: need {}, have {}",
        batch,
        scratch.seq_capacity
    );
    apply_rope_hidden_batch_in_place(
        ctx,
        &mut scratch.attn_out,
        rope,
        scratch.local_heads,
        scratch.head_dim,
        start_pos,
        true,
    )?;
    bf16_linear_bf16_hidden_into(ctx, &scratch.attn_out, &attn.wo_a, &mut scratch.low_rank)?;
    fp8_linear_bf16_hidden_into(ctx, &scratch.low_rank, &attn.wo_b, &mut scratch.out)?;
    Ok(&scratch.out)
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
    prefill_window_topk: &PrefillWindowTopk,
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
    let (topk_idxs, topk) =
        prefill_window_topk.parts(projections.q.seq_len, config.sliding_window)?;
    let mut attn_out = sparse_attention_prefill_bf16_hidden_with_topk(
        ctx,
        config,
        &projections,
        attn,
        topk_idxs,
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

pub(crate) fn attention_decode_rank_local_bf16_hidden_with_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    kv_cache: &mut Bf16Cache,
    attention_projection_scratch: &mut AttentionProjectionScratch,
    attention_output_scratch: &'a mut AttentionOutputScratch,
    attention_index_scratch: &mut AttentionIndexScratch,
) -> Result<&'a Bf16HiddenStates> {
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

    let mut projections = attention_project_bf16_hidden_scratch(
        ctx,
        config,
        input,
        attn,
        attention_projection_scratch,
    )
    .with_context(|| format!("attention_project layer {layer}"))?;
    apply_rope_attention_projections_view(ctx, &mut projections, rope, start_pos)
        .with_context(|| format!("apply_rope_attention_projections layer {layer}"))?;
    copy_bf16_rows_to_cache(
        ctx,
        projections.kv,
        kv_cache,
        0,
        start_pos % config.sliding_window,
        1,
    )
    .with_context(|| format!("copy kv to cache layer {layer} pos {start_pos}"))?;
    let topk = window_topk_indices_decode_into(
        ctx,
        start_pos,
        config.sliding_window,
        &mut attention_index_scratch.window_idxs,
    )
    .with_context(|| format!("window_topk_indices_decode layer {layer} pos {start_pos}"))?;
    indexed_attention_cache_bf16_hidden_view_into(
        ctx,
        config,
        &projections,
        kv_cache,
        attn,
        &attention_index_scratch.window_idxs,
        topk,
        &mut attention_output_scratch.attn_out,
    )
    .with_context(|| format!("indexed_attention_cache layer {layer} topk {topk}"))?;
    attention_output_project_bf16_hidden_scratch(
        ctx,
        attn,
        rope,
        start_pos,
        attention_output_scratch,
    )
    .with_context(|| format!("attention_output_project layer {layer}"))
}

pub(crate) fn attention_decode_rank_local_bf16_hidden_batch_with_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    batch_meta: &DecodeBatchMeta<'_>,
    kv_cache: &mut Bf16Cache,
    attention_projection_scratch: &mut AttentionProjectionScratch,
    attention_output_scratch: &'a mut AttentionOutputScratch,
    attention_index_scratch: &mut AttentionIndexScratch,
) -> Result<&'a Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        config.compress_ratios[layer] == 0,
        "rank-local batch decode only supports non-compressed layers, layer {layer} has compress_ratio={}",
        config.compress_ratios[layer]
    );
    ensure!(
        input.seq_len == batch_meta.batch,
        "rank-local batch decode input seq_len mismatch: input={}, batch={}",
        input.seq_len,
        batch_meta.batch
    );
    ensure!(
        kv_cache.hidden_dim == config.head_dim,
        "batch decode kv cache hidden dim mismatch: expected {}, got {}",
        config.head_dim,
        kv_cache.hidden_dim
    );

    let projections = attention_project_bf16_hidden_scratch(
        ctx,
        config,
        input,
        attn,
        attention_projection_scratch,
    )
    .with_context(|| format!("attention_project batch layer {layer}"))?;
    apply_rope_q_kv_batch_in_place(
        ctx,
        projections.q,
        projections.kv,
        rope,
        projections.local_heads,
        projections.head_dim,
        batch_meta.start_pos,
    )
    .with_context(|| format!("apply_rope_attention_projections batch layer {layer}"))?;
    copy_bf16_rows_to_cache_indexed(
        ctx,
        projections.kv,
        kv_cache,
        batch_meta.src_rows,
        batch_meta.window_dst_rows,
        batch_meta.batch,
    )
    .with_context(|| format!("copy batch kv to cache layer {layer}"))?;
    let topk = window_topk_indices_decode_batch_into(
        ctx,
        batch_meta.start_pos,
        batch_meta.window_base,
        batch_meta.batch,
        config.sliding_window,
        &mut attention_index_scratch.window_idxs,
    )
    .with_context(|| format!("window_topk_indices_decode batch layer {layer}"))?;
    indexed_attention_cache_bf16_hidden_view_into(
        ctx,
        config,
        &projections,
        kv_cache,
        attn,
        &attention_index_scratch.window_idxs,
        topk,
        &mut attention_output_scratch.attn_out,
    )
    .with_context(|| format!("indexed_attention_cache batch layer {layer} topk {topk}"))?;
    attention_output_project_bf16_hidden_batch_scratch(
        ctx,
        attn,
        rope,
        batch_meta.start_pos,
        batch_meta.batch,
        attention_output_scratch,
    )
    .with_context(|| format!("attention_output_project batch layer {layer}"))
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

pub(crate) fn attention_decode_compressed_nonoverlap_rank_local_bf16_hidden_batch_with_scratch<
    'a,
>(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    batch_meta: &DecodeBatchMeta<'_>,
    cache: &mut LayerDecodeCache,
    attention_projection_scratch: &mut AttentionProjectionScratch,
    attention_output_scratch: &'a mut AttentionOutputScratch,
    attention_index_scratch: &mut AttentionIndexScratch,
) -> Result<&'a Bf16HiddenStates> {
    ensure!(
        input.seq_len == batch_meta.batch,
        "compressed non-overlap batch decode seq_len mismatch: input={}, batch={}",
        input.seq_len,
        batch_meta.batch
    );
    ensure!(
        layer < config.compress_ratios.len(),
        "compressed batch decode layer {layer} out of range"
    );
    let ratio = config.compress_ratios[layer];
    ensure!(
        ratio > 0 && ratio != 4,
        "non-overlap batch decode called for ratio {ratio}"
    );
    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor weights"))?;
    let compressor_state = cache
        .compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor decode state"))?;

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
        let state_offset = batch_meta.slot_ids_host[row] * ratio;
        if let Some(compressed_kv) = compressor_nonoverlap_decode_bf16_hidden_at(
            ctx,
            config,
            &row_input,
            compressor,
            ratio,
            rope,
            batch_meta.start_pos_host[row],
            compressor_state,
            state_offset,
        )? {
            let dst = decode_cache_compressed_row(
                config.sliding_window,
                compressed_slots,
                batch_meta.slot_ids_host[row],
                batch_meta.start_pos_host[row] / ratio,
            );
            copy_bf16_rows_to_cache(ctx, &compressed_kv, &mut cache.kv, 0, dst, 1)?;
        }
    }

    let window_topk = window_topk_indices_decode_batch_into(
        ctx,
        batch_meta.start_pos,
        batch_meta.window_base,
        batch_meta.batch,
        config.sliding_window,
        &mut attention_index_scratch.window_idxs,
    )?;
    let max_compressed_len = batch_meta
        .start_pos_host
        .iter()
        .map(|pos| (pos + 1) / ratio)
        .max()
        .unwrap_or(0);
    let (topk_idxs, topk) = if max_compressed_len > 0 {
        let compress_topk = compress_topk_indices_decode_batch_into(
            ctx,
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
