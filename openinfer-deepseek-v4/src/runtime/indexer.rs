use super::*;

pub fn hadamard_fp4_quant_bf16_hidden_in_place(
    ctx: &RankGpuContext,
    hidden: &mut Bf16HiddenStates,
    groups: usize,
    dim: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(groups > 0, "Hadamard groups must be positive");
    ensure!(dim > 0, "Hadamard dim must be positive");
    ensure!(
        hidden.hidden_dim == groups * dim,
        "Hadamard hidden dim mismatch: expected {}, got {}",
        groups * dim,
        hidden.hidden_dim
    );
    ensure!(
        dim.is_power_of_two(),
        "Hadamard dim must be a power of two, got {}",
        dim
    );
    ensure!(
        dim.is_multiple_of(32),
        "FP4 quant dim must be divisible by 32, got {}",
        dim
    );

    {
        let (x_ptr, _x_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hadamard_fp4_quant_bf16_cuda(
                x_ptr as *mut ffi::Half,
                hidden.seq_len as i32,
                groups as i32,
                dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn indexer_scores_prefill_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    qr: &Bf16HiddenStates,
    indexer: &IndexerWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<(CudaSlice<f32>, usize)> {
    ctx.set_current()?;
    ensure!(
        start_pos == 0,
        "indexer prefill scores currently supports start_pos=0 only"
    );
    ensure!(
        input.hidden_dim == config.dim,
        "indexer input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        qr.hidden_dim == config.q_lora_rank,
        "indexer qr dim mismatch: expected {}, got {}",
        config.q_lora_rank,
        qr.hidden_dim
    );
    ensure!(
        qr.seq_len == input.seq_len,
        "indexer qr/input seq mismatch: qr={}, input={}",
        qr.seq_len,
        input.seq_len
    );
    ensure!(
        input.seq_len >= 4,
        "indexer prefill needs at least one ratio-4 block, got seq_len={}",
        input.seq_len
    );

    let local_heads = config.index_n_heads / 8;
    let mut q = fp8_linear_bf16_hidden(ctx, qr, &indexer.wq_b)?;
    ensure!(
        q.hidden_dim == local_heads * config.index_head_dim,
        "indexer q dim mismatch: expected {}, got {}",
        local_heads * config.index_head_dim,
        q.hidden_dim
    );
    apply_rope_hidden_in_place(
        ctx,
        &mut q,
        rope,
        local_heads,
        config.index_head_dim,
        start_pos,
        false,
    )?;
    hadamard_fp4_quant_bf16_hidden_in_place(ctx, &mut q, local_heads, config.index_head_dim)?;

    let mut compressed_kv = compressor_overlap_prefill_bf16_hidden_with_dim(
        ctx,
        config,
        input,
        &indexer.compressor,
        rope,
        start_pos,
        config.index_head_dim,
    )?;
    hadamard_fp4_quant_bf16_hidden_in_place(ctx, &mut compressed_kv, 1, config.index_head_dim)?;
    let weights = bf16_linear_bf16_hidden(ctx, input, &indexer.weights_proj)?;
    ensure!(
        weights.hidden_dim == local_heads,
        "indexer weights dim mismatch: expected {}, got {}",
        local_heads,
        weights.hidden_dim
    );

    let compressed_len = compressed_kv.seq_len;
    let mut scores = ctx.stream.alloc_zeros(input.seq_len * compressed_len)?;
    {
        let (q_ptr, _q_guard) = q.data.device_ptr(&ctx.stream);
        let (kv_ptr, _kv_guard) = compressed_kv.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = weights.data.device_ptr(&ctx.stream);
        let (scores_ptr, _scores_guard) = scores.device_ptr_mut(&ctx.stream);
        let score_scale =
            1.0f32 / (config.index_head_dim as f32).sqrt() / (config.index_n_heads as f32).sqrt();
        ensure!(
            local_heads == 8 && config.index_head_dim == 128,
            "CuTeDSL indexer score runtime expects local_heads=8 and head_dim=128, got local_heads={} head_dim={}",
            local_heads,
            config.index_head_dim
        );
        let result = unsafe {
            ffi::deepseek_cutedsl_indexer_scores_exact_bf16_cuda(
                q_ptr as *const ffi::Half,
                kv_ptr as *const ffi::Half,
                weights_ptr as *const ffi::Half,
                scores_ptr as *mut f32,
                input.seq_len as i32,
                compressed_len as i32,
                score_scale,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok((scores, compressed_len))
}

pub fn indexer_topk_indices_prefill(
    ctx: &RankGpuContext,
    config: &Config,
    scores: &CudaSlice<f32>,
    seq_len: usize,
    compressed_len: usize,
    offset: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(
        compressed_len > 0,
        "indexer compressed_len must be positive"
    );
    ensure!(
        scores.len() == seq_len * compressed_len,
        "indexer scores shape mismatch: expected {}, got {}",
        seq_len * compressed_len,
        scores.len()
    );
    let topk = config.index_topk.min(compressed_len);
    let mut topk_idxs = ctx.stream.alloc_zeros(seq_len * topk)?;
    {
        let (scores_ptr, _scores_guard) = scores.device_ptr(&ctx.stream);
        let (topk_ptr, _topk_guard) = topk_idxs.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_indexer_topk_prefill_cuda(
                scores_ptr as *const f32,
                topk_ptr as *mut i32,
                seq_len as i32,
                compressed_len as i32,
                topk as i32,
                4,
                offset as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok((topk_idxs, topk))
}

pub fn indexer_scores_decode_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    qr: &Bf16HiddenStates,
    indexer: &IndexerWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    kv_cache: &mut Bf16Cache,
    compressor_state: &mut CompressorDecodeState,
) -> Result<Option<CudaSlice<f32>>> {
    ctx.set_current()?;
    ensure!(
        input.hidden_dim == config.dim,
        "indexer decode input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == 1,
        "indexer decode expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        qr.hidden_dim == config.q_lora_rank && qr.seq_len == 1,
        "indexer decode qr shape mismatch: hidden_dim={}, seq_len={}",
        qr.hidden_dim,
        qr.seq_len
    );
    ensure!(
        kv_cache.hidden_dim == config.index_head_dim,
        "indexer decode kv cache dim mismatch: expected {}, got {}",
        config.index_head_dim,
        kv_cache.hidden_dim
    );

    let local_heads = config.index_n_heads / 8;
    let mut q = fp8_linear_bf16_hidden(ctx, qr, &indexer.wq_b)?;
    ensure!(
        q.hidden_dim == local_heads * config.index_head_dim && q.seq_len == 1,
        "indexer decode q shape mismatch: hidden_dim={}, seq_len={}",
        q.hidden_dim,
        q.seq_len
    );
    apply_rope_hidden_in_place(
        ctx,
        &mut q,
        rope,
        local_heads,
        config.index_head_dim,
        start_pos,
        false,
    )?;
    hadamard_fp4_quant_bf16_hidden_in_place(ctx, &mut q, local_heads, config.index_head_dim)?;

    if let Some(compressed_kv) = compressor_overlap_decode_bf16_hidden_with_dim(
        ctx,
        config,
        input,
        &indexer.compressor,
        rope,
        start_pos,
        config.index_head_dim,
        compressor_state,
        true,
    )? {
        copy_bf16_rows_to_cache(ctx, &compressed_kv, kv_cache, 0, start_pos / 4, 1)?;
    }

    let compressed_len = (start_pos + 1) / 4;
    if compressed_len == 0 {
        return Ok(None);
    }
    ensure!(
        compressed_len <= kv_cache.slots,
        "indexer decode compressed_len {} exceeds kv cache slots {}",
        compressed_len,
        kv_cache.slots
    );

    let weights = bf16_linear_bf16_hidden(ctx, input, &indexer.weights_proj)?;
    ensure!(
        weights.hidden_dim == local_heads && weights.seq_len == 1,
        "indexer decode weights shape mismatch: hidden_dim={}, seq_len={}",
        weights.hidden_dim,
        weights.seq_len
    );
    let mut scores = ctx.stream.alloc_zeros(compressed_len)?;
    {
        let (q_ptr, _q_guard) = q.data.device_ptr(&ctx.stream);
        let (kv_ptr, _kv_guard) = kv_cache.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = weights.data.device_ptr(&ctx.stream);
        let (scores_ptr, _scores_guard) = scores.device_ptr_mut(&ctx.stream);
        let score_scale =
            1.0f32 / (config.index_head_dim as f32).sqrt() / (config.index_n_heads as f32).sqrt();
        let result = unsafe {
            ffi::deepseek_indexer_scores_decode_cuda(
                q_ptr as *const ffi::Half,
                kv_ptr as *const ffi::Half,
                weights_ptr as *const ffi::Half,
                scores_ptr as *mut f32,
                local_heads as i32,
                config.index_head_dim as i32,
                compressed_len as i32,
                score_scale,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(Some(scores))
}

pub(crate) fn indexer_scores_decode_bf16_hidden_scratch(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    qr: &Bf16HiddenStates,
    indexer: &IndexerWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    kv_cache: &mut Bf16Cache,
    compressor_state: &mut CompressorDecodeState,
    scratch: &mut AttentionAuxScratch,
) -> Result<Option<usize>> {
    ctx.set_current()?;
    ensure!(
        input.hidden_dim == config.dim,
        "indexer decode input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == 1,
        "indexer decode expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        qr.hidden_dim == config.q_lora_rank && qr.seq_len == 1,
        "indexer decode qr shape mismatch: hidden_dim={}, seq_len={}",
        qr.hidden_dim,
        qr.seq_len
    );
    ensure!(
        kv_cache.hidden_dim == config.index_head_dim,
        "indexer decode kv cache dim mismatch: expected {}, got {}",
        config.index_head_dim,
        kv_cache.hidden_dim
    );

    let local_heads = scratch.local_index_heads;
    fp8_linear_bf16_hidden_into(ctx, qr, &indexer.wq_b, &mut scratch.indexer_q)?;
    ensure!(
        scratch.indexer_q.hidden_dim == local_heads * config.index_head_dim
            && scratch.indexer_q.seq_len == 1,
        "indexer decode q shape mismatch: hidden_dim={}, seq_len={}",
        scratch.indexer_q.hidden_dim,
        scratch.indexer_q.seq_len
    );
    apply_rope_hidden_in_place(
        ctx,
        &mut scratch.indexer_q,
        rope,
        local_heads,
        config.index_head_dim,
        start_pos,
        false,
    )?;
    hadamard_fp4_quant_bf16_hidden_in_place(
        ctx,
        &mut scratch.indexer_q,
        local_heads,
        config.index_head_dim,
    )?;

    if let Some(compressed_kv) = compressor_overlap_decode_bf16_hidden_with_dim_scratch(
        ctx,
        config,
        input,
        &indexer.compressor,
        rope,
        start_pos,
        config.index_head_dim,
        compressor_state,
        true,
        scratch,
    )? {
        copy_bf16_rows_to_cache(ctx, compressed_kv, kv_cache, 0, start_pos / 4, 1)?;
    }

    let compressed_len = (start_pos + 1) / 4;
    if compressed_len == 0 {
        return Ok(None);
    }
    ensure!(
        compressed_len <= kv_cache.slots,
        "indexer decode compressed_len {} exceeds kv cache slots {}",
        compressed_len,
        kv_cache.slots
    );
    ensure!(
        compressed_len <= scratch.max_compressed_len,
        "indexer score scratch capacity too small: need {}, have {}",
        compressed_len,
        scratch.max_compressed_len
    );

    bf16_linear_bf16_hidden_into(
        ctx,
        input,
        &indexer.weights_proj,
        &mut scratch.indexer_weights,
    )?;
    ensure!(
        scratch.indexer_weights.hidden_dim == local_heads && scratch.indexer_weights.seq_len == 1,
        "indexer decode weights shape mismatch: hidden_dim={}, seq_len={}",
        scratch.indexer_weights.hidden_dim,
        scratch.indexer_weights.seq_len
    );
    {
        let (q_ptr, _q_guard) = scratch.indexer_q.data.device_ptr(&ctx.stream);
        let (kv_ptr, _kv_guard) = kv_cache.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = scratch.indexer_weights.data.device_ptr(&ctx.stream);
        let (scores_ptr, _scores_guard) = scratch.indexer_scores.device_ptr_mut(&ctx.stream);
        let score_scale =
            1.0f32 / (config.index_head_dim as f32).sqrt() / (config.index_n_heads as f32).sqrt();
        let result = unsafe {
            ffi::deepseek_indexer_scores_decode_cuda(
                q_ptr as *const ffi::Half,
                kv_ptr as *const ffi::Half,
                weights_ptr as *const ffi::Half,
                scores_ptr as *mut f32,
                local_heads as i32,
                config.index_head_dim as i32,
                compressed_len as i32,
                score_scale,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(Some(compressed_len))
}

pub(crate) struct IndexerDecodeBatchInputs<'a> {
    pub(crate) q: &'a Bf16HiddenStates,
    pub(crate) kv_cache: &'a Bf16Cache,
    pub(crate) weights: &'a Bf16HiddenStates,
    pub(crate) compressed_len: &'a CudaSlice<i32>,
    pub(crate) cache_base: &'a CudaSlice<i32>,
    pub(crate) batch: usize,
    pub(crate) max_compressed_len: usize,
}

pub(crate) fn indexer_scores_decode_batch_into(
    ctx: &RankGpuContext,
    config: &Config,
    input: IndexerDecodeBatchInputs<'_>,
    scores: &mut CudaSlice<f32>,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(input.batch > 0, "indexer decode batch must be positive");
    ensure!(
        input.q.seq_len == input.batch,
        "indexer batch q seq_len mismatch: expected {}, got {}",
        input.batch,
        input.q.seq_len
    );
    ensure!(
        input.weights.seq_len == input.batch,
        "indexer batch weights seq_len mismatch: expected {}, got {}",
        input.batch,
        input.weights.seq_len
    );
    ensure!(
        input.kv_cache.hidden_dim == config.index_head_dim,
        "indexer batch kv cache dim mismatch: expected {}, got {}",
        config.index_head_dim,
        input.kv_cache.hidden_dim
    );
    ensure!(
        input.compressed_len.len() >= input.batch,
        "indexer batch compressed_len capacity too small: need {}, have {}",
        input.batch,
        input.compressed_len.len()
    );
    ensure!(
        input.cache_base.len() >= input.batch,
        "indexer batch cache_base capacity too small: need {}, have {}",
        input.batch,
        input.cache_base.len()
    );
    ensure!(
        scores.len() >= input.batch * input.max_compressed_len,
        "indexer batch scores capacity too small: need {}, have {}",
        input.batch * input.max_compressed_len,
        scores.len()
    );
    let local_heads = config.index_n_heads / 8;
    ensure!(
        input.q.hidden_dim == local_heads * config.index_head_dim,
        "indexer batch q hidden mismatch: expected {}, got {}",
        local_heads * config.index_head_dim,
        input.q.hidden_dim
    );
    ensure!(
        input.weights.hidden_dim == local_heads,
        "indexer batch weights hidden mismatch: expected {}, got {}",
        local_heads,
        input.weights.hidden_dim
    );
    {
        let (q_ptr, _q_guard) = input.q.data.device_ptr(&ctx.stream);
        let (kv_ptr, _kv_guard) = input.kv_cache.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = input.weights.data.device_ptr(&ctx.stream);
        let (compressed_ptr, _compressed_guard) = input.compressed_len.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = input.cache_base.device_ptr(&ctx.stream);
        let (scores_ptr, _scores_guard) = scores.device_ptr_mut(&ctx.stream);
        let score_scale =
            1.0f32 / (config.index_head_dim as f32).sqrt() / (config.index_n_heads as f32).sqrt();
        let result = unsafe {
            ffi::deepseek_indexer_scores_decode_batch_cuda(
                q_ptr as *const ffi::Half,
                kv_ptr as *const ffi::Half,
                weights_ptr as *const ffi::Half,
                compressed_ptr as *const i32,
                base_ptr as *const i32,
                scores_ptr as *mut f32,
                input.batch as i32,
                local_heads as i32,
                config.index_head_dim as i32,
                input.max_compressed_len as i32,
                score_scale,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn indexer_topk_indices_decode(
    ctx: &RankGpuContext,
    config: &Config,
    scores: &CudaSlice<f32>,
    compressed_len: usize,
    offset: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(
        compressed_len > 0,
        "indexer decode compressed_len must be positive"
    );
    let topk = config.index_topk.min(compressed_len);
    let mut topk_idxs = unsafe { ctx.stream.alloc(topk)? };
    indexer_topk_indices_decode_into(ctx, config, scores, compressed_len, offset, &mut topk_idxs)?;
    Ok((topk_idxs, topk))
}

pub(crate) fn indexer_topk_indices_decode_into<S>(
    ctx: &RankGpuContext,
    config: &Config,
    scores: &S,
    compressed_len: usize,
    offset: usize,
    topk_idxs: &mut CudaSlice<i32>,
) -> Result<usize>
where
    S: DevicePtr<f32>,
{
    ctx.set_current()?;
    ensure!(
        compressed_len > 0,
        "indexer decode compressed_len must be positive"
    );
    ensure!(
        scores.len() == compressed_len,
        "indexer decode scores shape mismatch: expected {}, got {}",
        compressed_len,
        scores.len()
    );
    let topk = config.index_topk.min(compressed_len);
    ensure!(
        topk_idxs.len() >= topk,
        "indexer decode top-k output capacity too small: need {}, have {}",
        topk,
        topk_idxs.len()
    );
    {
        let (scores_ptr, _scores_guard) = scores.device_ptr(&ctx.stream);
        let (topk_ptr, _topk_guard) = topk_idxs.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_indexer_topk_decode_cuda(
                scores_ptr as *const f32,
                topk_ptr as *mut i32,
                compressed_len as i32,
                topk as i32,
                offset as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(topk)
}

pub(crate) struct Ratio4DecodeTopkBatchInputs<'a> {
    pub(crate) scores: &'a CudaSlice<f32>,
    pub(crate) start_pos: &'a CudaSlice<i32>,
    pub(crate) window_base: &'a CudaSlice<i32>,
    pub(crate) compressed_len: &'a CudaSlice<i32>,
    pub(crate) compressed_base: &'a CudaSlice<i32>,
    pub(crate) batch: usize,
    pub(crate) max_compressed_len: usize,
}

pub(crate) fn ratio4_decode_topk_indices_batch_into(
    ctx: &RankGpuContext,
    config: &Config,
    input: Ratio4DecodeTopkBatchInputs<'_>,
    topk_idxs: &mut CudaSlice<i32>,
) -> Result<usize> {
    ctx.set_current()?;
    ensure!(input.batch > 0, "ratio4 top-k batch must be positive");
    ensure!(
        input.max_compressed_len > 0,
        "ratio4 top-k max_compressed_len must be positive"
    );
    ensure!(
        input.scores.len() >= input.batch * input.max_compressed_len,
        "ratio4 top-k scores capacity too small: need {}, have {}",
        input.batch * input.max_compressed_len,
        input.scores.len()
    );
    ensure!(
        input.start_pos.len() >= input.batch
            && input.window_base.len() >= input.batch
            && input.compressed_len.len() >= input.batch
            && input.compressed_base.len() >= input.batch,
        "ratio4 top-k metadata capacity too small for batch {}",
        input.batch
    );
    let index_topk = config.index_topk.min(input.max_compressed_len);
    let total_topk = config.sliding_window + index_topk;
    ensure!(
        topk_idxs.len() >= input.batch * total_topk,
        "ratio4 top-k output capacity too small: need {}, have {}",
        input.batch * total_topk,
        topk_idxs.len()
    );
    {
        let (scores_ptr, _scores_guard) = input.scores.device_ptr(&ctx.stream);
        let (start_ptr, _start_guard) = input.start_pos.device_ptr(&ctx.stream);
        let (window_base_ptr, _window_base_guard) = input.window_base.device_ptr(&ctx.stream);
        let (compressed_len_ptr, _compressed_len_guard) =
            input.compressed_len.device_ptr(&ctx.stream);
        let (compressed_base_ptr, _compressed_base_guard) =
            input.compressed_base.device_ptr(&ctx.stream);
        let (topk_ptr, _topk_guard) = topk_idxs.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_ratio4_decode_topk_indices_batch_cuda(
                scores_ptr as *const f32,
                start_ptr as *const i32,
                window_base_ptr as *const i32,
                compressed_len_ptr as *const i32,
                compressed_base_ptr as *const i32,
                topk_ptr as *mut i32,
                input.batch as i32,
                config.sliding_window as i32,
                input.max_compressed_len as i32,
                index_topk as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(total_topk)
}
