use super::*;

pub fn attention_project_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
) -> Result<AttentionProjections> {
    ctx.set_current()?;
    let qr = fp8_linear_bf16_hidden(ctx, input, &attn.wq_a)?;
    let qr_norm = rms_norm_bf16_hidden(ctx, &qr, &attn.q_norm, config.rms_norm_eps)?;
    let q_raw = fp8_linear_bf16_hidden(ctx, &qr_norm, &attn.wq_b)?;
    let local_heads = q_raw.hidden_dim / config.head_dim;
    ensure!(
        local_heads * config.head_dim == q_raw.hidden_dim,
        "wq_b output dim {} is not divisible by head_dim {}",
        q_raw.hidden_dim,
        config.head_dim
    );
    let q = head_rms_norm_bf16_hidden(
        ctx,
        &q_raw,
        local_heads,
        config.head_dim,
        config.rms_norm_eps,
    )?;
    let kv_raw = fp8_linear_bf16_hidden(ctx, input, &attn.wkv)?;
    let kv = rms_norm_bf16_hidden(ctx, &kv_raw, &attn.kv_norm, config.rms_norm_eps)?;
    Ok(AttentionProjections {
        qr: qr_norm,
        q,
        kv,
        local_heads,
        head_dim: config.head_dim,
    })
}

pub(crate) fn attention_project_bf16_hidden_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    scratch: &'a mut AttentionProjectionScratch,
) -> Result<AttentionProjectionsView<'a>> {
    ctx.set_current()?;
    ensure!(
        input.seq_len <= scratch.seq_capacity,
        "attention projection scratch seq capacity too small: need {}, have {}",
        input.seq_len,
        scratch.seq_capacity
    );
    fp8_linear_bf16_hidden_into(ctx, input, &attn.wq_a, &mut scratch.qr_raw)?;
    rms_norm_bf16_hidden_into(
        ctx,
        &scratch.qr_raw,
        &attn.q_norm,
        config.rms_norm_eps,
        &mut scratch.qr,
    )?;
    fp8_linear_bf16_hidden_into(ctx, &scratch.qr, &attn.wq_b, &mut scratch.q_raw)?;
    ensure!(
        scratch.local_heads * config.head_dim == scratch.q_raw.hidden_dim,
        "wq_b output dim {} is not divisible by head_dim {}",
        scratch.q_raw.hidden_dim,
        config.head_dim
    );
    head_rms_norm_bf16_hidden_into(
        ctx,
        &scratch.q_raw,
        scratch.local_heads,
        config.head_dim,
        config.rms_norm_eps,
        &mut scratch.q,
    )?;
    fp8_linear_bf16_hidden_into(ctx, input, &attn.wkv, &mut scratch.kv_raw)?;
    rms_norm_bf16_hidden_into(
        ctx,
        &scratch.kv_raw,
        &attn.kv_norm,
        config.rms_norm_eps,
        &mut scratch.kv,
    )?;
    Ok(AttentionProjectionsView {
        qr: &scratch.qr,
        q: &mut scratch.q,
        kv: &mut scratch.kv,
        local_heads: scratch.local_heads,
        head_dim: scratch.head_dim,
    })
}

pub fn precompute_rope_cache(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    max_seq_len: usize,
) -> Result<DeepSeekRopeCache> {
    ctx.set_current()?;
    ensure!(
        layer < config.compress_ratios.len(),
        "layer {layer} out of range"
    );
    ensure!(max_seq_len > 0, "max_seq_len must be positive");
    let rotary_dim = config.qk_rope_head_dim;
    ensure!(
        rotary_dim.is_multiple_of(2),
        "rotary_dim must be even, got {rotary_dim}"
    );

    let compress = config.compress_ratios[layer] > 0;
    let base = if compress {
        config.compress_rope_theta
    } else {
        config.rope_theta
    };
    let original_seq_len = if compress {
        config.rope_scaling.original_seq_len
    } else {
        0
    };
    let factor = config.rope_scaling.factor;
    let beta_fast = config.rope_scaling.beta_fast as f32;
    let beta_slow = config.rope_scaling.beta_slow as f32;

    let mut inv_freq = Vec::with_capacity(rotary_dim / 2);
    for i in 0..rotary_dim / 2 {
        let exponent = (2 * i) as f32 / rotary_dim as f32;
        inv_freq.push(1.0 / base.powf(exponent));
    }
    if original_seq_len > 0 {
        let find_correction_dim = |num_rotations: f32| -> f32 {
            rotary_dim as f32
                * ((original_seq_len as f32) / (num_rotations * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = find_correction_dim(beta_fast).floor().max(0.0);
        let high = find_correction_dim(beta_slow)
            .ceil()
            .min((rotary_dim - 1) as f32);
        let high = if (high - low).abs() < f32::EPSILON {
            high + 0.001
        } else {
            high
        };
        for (i, freq) in inv_freq.iter_mut().enumerate() {
            let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
        }
    }

    let pairs = rotary_dim / 2;
    let inv_freq_gpu = ctx.stream.clone_htod(&inv_freq)?;
    let mut cos = ctx.stream.alloc_zeros::<f32>(max_seq_len * pairs)?;
    let mut sin = ctx.stream.alloc_zeros::<f32>(max_seq_len * pairs)?;
    {
        let (inv_ptr, _inv_guard) = inv_freq_gpu.device_ptr(&ctx.stream);
        let (cos_ptr, _cos_guard) = cos.device_ptr_mut(&ctx.stream);
        let (sin_ptr, _sin_guard) = sin.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_fill_rope_cache_cuda(
                inv_ptr as *const f32,
                cos_ptr as *mut f32,
                sin_ptr as *mut f32,
                max_seq_len as i32,
                pairs as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    ctx.sync()?;
    Ok(DeepSeekRopeCache {
        cos,
        sin,
        max_seq_len,
        rotary_dim,
    })
}

pub fn apply_rope_attention_projections(
    ctx: &RankGpuContext,
    projections: &mut AttentionProjections,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        start_pos + projections.q.seq_len <= rope.max_seq_len,
        "RoPE range [{}..{}) exceeds cache len {}",
        start_pos,
        start_pos + projections.q.seq_len,
        rope.max_seq_len
    );
    ensure!(
        rope.rotary_dim <= projections.head_dim,
        "rotary_dim {} exceeds head_dim {}",
        rope.rotary_dim,
        projections.head_dim
    );
    ensure!(
        projections.kv.hidden_dim == projections.head_dim,
        "kv dim {} must equal head_dim {}",
        projections.kv.hidden_dim,
        projections.head_dim
    );

    apply_rope_q_kv_in_place(
        ctx,
        &mut projections.q,
        &mut projections.kv,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )?;
    fp8_act_quant_nope_bf16_hidden_in_place(
        ctx,
        &mut projections.kv,
        1,
        projections.head_dim,
        rope.rotary_dim,
        64,
    )?;
    Ok(())
}

pub(crate) fn apply_rope_attention_projections_view(
    ctx: &RankGpuContext,
    projections: &mut AttentionProjectionsView<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        start_pos + projections.q.seq_len <= rope.max_seq_len,
        "RoPE range [{}..{}) exceeds cache len {}",
        start_pos,
        start_pos + projections.q.seq_len,
        rope.max_seq_len
    );
    ensure!(
        rope.rotary_dim <= projections.head_dim,
        "rotary_dim {} exceeds head_dim {}",
        rope.rotary_dim,
        projections.head_dim
    );
    ensure!(
        projections.kv.hidden_dim == projections.head_dim,
        "kv dim {} must equal head_dim {}",
        projections.kv.hidden_dim,
        projections.head_dim
    );

    apply_rope_q_kv_in_place(
        ctx,
        projections.q,
        projections.kv,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )?;
    fp8_act_quant_nope_bf16_hidden_in_place(
        ctx,
        projections.kv,
        1,
        projections.head_dim,
        rope.rotary_dim,
        64,
    )?;
    Ok(())
}

pub(crate) fn apply_rope_q_kv_in_place(
    ctx: &RankGpuContext,
    q: &mut Bf16HiddenStates,
    kv: &mut Bf16HiddenStates,
    rope: &DeepSeekRopeCache,
    local_heads: usize,
    head_dim: usize,
    start_pos: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        q.hidden_dim == local_heads * head_dim,
        "RoPE q hidden dim mismatch: expected {}, got {}",
        local_heads * head_dim,
        q.hidden_dim
    );
    ensure!(
        kv.hidden_dim == head_dim,
        "RoPE kv hidden dim mismatch: expected {}, got {}",
        head_dim,
        kv.hidden_dim
    );
    ensure!(
        q.seq_len == kv.seq_len,
        "RoPE q/kv seq_len mismatch: q={}, kv={}",
        q.seq_len,
        kv.seq_len
    );
    ensure!(
        start_pos + q.seq_len <= rope.max_seq_len,
        "RoPE q/kv range [{}..{}) exceeds cache len {}",
        start_pos,
        start_pos + q.seq_len,
        rope.max_seq_len
    );
    ensure!(
        rope.rotary_dim <= head_dim,
        "rotary_dim {} exceeds head_dim {}",
        rope.rotary_dim,
        head_dim
    );

    {
        let (q_ptr, _q_guard) = q.data.device_ptr_mut(&ctx.stream);
        let (kv_ptr, _kv_guard) = kv.data.device_ptr_mut(&ctx.stream);
        let (cos_ptr, _cos_guard) = rope.cos.device_ptr(&ctx.stream);
        let (sin_ptr, _sin_guard) = rope.sin.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_apply_rope_q_kv_cuda(
                q_ptr as *mut ffi::Half,
                kv_ptr as *mut ffi::Half,
                cos_ptr as *const f32,
                sin_ptr as *const f32,
                q.seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rope.rotary_dim as i32,
                start_pos as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub(crate) fn apply_rope_q_kv_batch_in_place(
    ctx: &RankGpuContext,
    q: &mut Bf16HiddenStates,
    kv: &mut Bf16HiddenStates,
    rope: &DeepSeekRopeCache,
    local_heads: usize,
    head_dim: usize,
    start_pos: &CudaSlice<i32>,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        q.hidden_dim == local_heads * head_dim,
        "batch RoPE q hidden dim mismatch: expected {}, got {}",
        local_heads * head_dim,
        q.hidden_dim
    );
    ensure!(
        kv.hidden_dim == head_dim,
        "batch RoPE kv hidden dim mismatch: expected {}, got {}",
        head_dim,
        kv.hidden_dim
    );
    ensure!(
        q.seq_len == kv.seq_len,
        "batch RoPE q/kv seq_len mismatch: q={}, kv={}",
        q.seq_len,
        kv.seq_len
    );
    ensure!(
        start_pos.len() >= q.seq_len,
        "batch RoPE start_pos capacity too small: need {}, have {}",
        q.seq_len,
        start_pos.len()
    );
    ensure!(
        rope.rotary_dim <= head_dim,
        "rotary_dim {} exceeds head_dim {}",
        rope.rotary_dim,
        head_dim
    );

    {
        let (q_ptr, _q_guard) = q.data.device_ptr_mut(&ctx.stream);
        let (kv_ptr, _kv_guard) = kv.data.device_ptr_mut(&ctx.stream);
        let (cos_ptr, _cos_guard) = rope.cos.device_ptr(&ctx.stream);
        let (sin_ptr, _sin_guard) = rope.sin.device_ptr(&ctx.stream);
        let (start_ptr, _start_guard) = start_pos.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_apply_rope_q_kv_batch_cuda(
                q_ptr as *mut ffi::Half,
                kv_ptr as *mut ffi::Half,
                cos_ptr as *const f32,
                sin_ptr as *const f32,
                start_ptr as *const i32,
                q.seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rope.rotary_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn fp8_act_quant_nope_bf16_hidden_in_place(
    ctx: &RankGpuContext,
    hidden: &mut Bf16HiddenStates,
    local_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    block_size: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        hidden.hidden_dim == local_heads * head_dim,
        "FP8 in-place quant hidden dim mismatch: expected {}, got {}",
        local_heads * head_dim,
        hidden.hidden_dim
    );
    ensure!(
        rotary_dim < head_dim,
        "FP8 in-place quant rotary_dim {} must be smaller than head_dim {}",
        rotary_dim,
        head_dim
    );
    let nope_dim = head_dim - rotary_dim;
    ensure!(
        nope_dim.is_multiple_of(block_size),
        "FP8 in-place quant nope_dim {} must be divisible by block_size {}",
        nope_dim,
        block_size
    );

    {
        let (x_ptr, _x_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_fp8_act_quant_nope_bf16_cuda(
                x_ptr as *mut ffi::Half,
                hidden.seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rotary_dim as i32,
                block_size as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn apply_rope_hidden_in_place(
    ctx: &RankGpuContext,
    hidden: &mut Bf16HiddenStates,
    rope: &DeepSeekRopeCache,
    local_heads: usize,
    head_dim: usize,
    start_pos: usize,
    inverse: bool,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        hidden.hidden_dim == local_heads * head_dim,
        "RoPE hidden dim mismatch: expected {}, got {}",
        local_heads * head_dim,
        hidden.hidden_dim
    );
    ensure!(
        start_pos + hidden.seq_len <= rope.max_seq_len,
        "RoPE range [{}..{}) exceeds cache len {}",
        start_pos,
        start_pos + hidden.seq_len,
        rope.max_seq_len
    );
    ensure!(
        rope.rotary_dim <= head_dim,
        "rotary_dim {} exceeds head_dim {}",
        rope.rotary_dim,
        head_dim
    );

    {
        let (x_ptr, _x_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let (cos_ptr, _cos_guard) = rope.cos.device_ptr(&ctx.stream);
        let (sin_ptr, _sin_guard) = rope.sin.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_apply_rope_hidden_cuda(
                x_ptr as *mut ffi::Half,
                cos_ptr as *const f32,
                sin_ptr as *const f32,
                hidden.seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rope.rotary_dim as i32,
                start_pos as i32,
                if inverse { 1 } else { 0 },
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub(crate) fn apply_rope_hidden_batch_in_place(
    ctx: &RankGpuContext,
    hidden: &mut Bf16HiddenStates,
    rope: &DeepSeekRopeCache,
    local_heads: usize,
    head_dim: usize,
    start_pos: &CudaSlice<i32>,
    inverse: bool,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        hidden.hidden_dim == local_heads * head_dim,
        "batch RoPE hidden dim mismatch: expected {}, got {}",
        local_heads * head_dim,
        hidden.hidden_dim
    );
    ensure!(
        start_pos.len() >= hidden.seq_len,
        "batch RoPE hidden start_pos capacity too small: need {}, have {}",
        hidden.seq_len,
        start_pos.len()
    );
    ensure!(
        rope.rotary_dim <= head_dim,
        "rotary_dim {} exceeds head_dim {}",
        rope.rotary_dim,
        head_dim
    );

    {
        let (x_ptr, _x_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let (cos_ptr, _cos_guard) = rope.cos.device_ptr(&ctx.stream);
        let (sin_ptr, _sin_guard) = rope.sin.device_ptr(&ctx.stream);
        let (start_ptr, _start_guard) = start_pos.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_apply_rope_hidden_batch_cuda(
                x_ptr as *mut ffi::Half,
                cos_ptr as *const f32,
                sin_ptr as *const f32,
                start_ptr as *const i32,
                hidden.seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rope.rotary_dim as i32,
                if inverse { 1 } else { 0 },
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn apply_rope_hidden_strided_in_place(
    ctx: &RankGpuContext,
    hidden: &mut Bf16HiddenStates,
    rope: &DeepSeekRopeCache,
    local_heads: usize,
    head_dim: usize,
    start_pos: usize,
    position_stride: usize,
    inverse: bool,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        hidden.hidden_dim == local_heads * head_dim,
        "strided RoPE hidden dim mismatch: expected {}, got {}",
        local_heads * head_dim,
        hidden.hidden_dim
    );
    ensure!(position_stride > 0, "position_stride must be positive");
    ensure!(
        start_pos + (hidden.seq_len - 1) * position_stride < rope.max_seq_len,
        "strided RoPE range start={} len={} stride={} exceeds cache len {}",
        start_pos,
        hidden.seq_len,
        position_stride,
        rope.max_seq_len
    );
    ensure!(
        rope.rotary_dim <= head_dim,
        "rotary_dim {} exceeds head_dim {}",
        rope.rotary_dim,
        head_dim
    );

    {
        let (x_ptr, _x_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let (cos_ptr, _cos_guard) = rope.cos.device_ptr(&ctx.stream);
        let (sin_ptr, _sin_guard) = rope.sin.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_apply_rope_hidden_strided_cuda(
                x_ptr as *mut ffi::Half,
                cos_ptr as *const f32,
                sin_ptr as *const f32,
                hidden.seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rope.rotary_dim as i32,
                start_pos as i32,
                position_stride as i32,
                if inverse { 1 } else { 0 },
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn sparse_attention_prefill_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    projections: &AttentionProjections,
    attn: &AttentionWeights<'_>,
) -> Result<Bf16HiddenStates> {
    let (topk_idxs, topk) = window_topk_indices(ctx, projections.q.seq_len, config.sliding_window)?;
    sparse_attention_prefill_bf16_hidden_with_topk(ctx, config, projections, attn, &topk_idxs, topk)
}

pub(crate) fn sparse_attention_prefill_bf16_hidden_with_topk(
    ctx: &RankGpuContext,
    config: &Config,
    projections: &AttentionProjections,
    attn: &AttentionWeights<'_>,
    topk_idxs: &CudaSlice<i32>,
    topk: usize,
) -> Result<Bf16HiddenStates> {
    indexed_attention_prefill_bf16_hidden(ctx, config, projections, attn, topk_idxs, topk)
}

pub(crate) struct PrefillWindowTopk {
    indices: CudaSlice<i32>,
    topk: usize,
    seq_len: usize,
    window_size: usize,
}

impl PrefillWindowTopk {
    pub(crate) fn new(ctx: &RankGpuContext, seq_len: usize, window_size: usize) -> Result<Self> {
        let (indices, topk) = window_topk_indices(ctx, seq_len, window_size)?;
        Ok(Self {
            indices,
            topk,
            seq_len,
            window_size,
        })
    }

    pub(crate) fn parts(
        &self,
        seq_len: usize,
        window_size: usize,
    ) -> Result<(&CudaSlice<i32>, usize)> {
        ensure!(
            self.seq_len == seq_len && self.window_size == window_size,
            "prefill window topk mismatch: cached seq_len={}, window_size={}, requested seq_len={}, window_size={}",
            self.seq_len,
            self.window_size,
            seq_len,
            window_size
        );
        Ok((&self.indices, self.topk))
    }
}

pub fn window_topk_indices(
    ctx: &RankGpuContext,
    seq_len: usize,
    window_size: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(seq_len > 0, "seq_len must be positive");
    ensure!(window_size > 0, "window_size must be positive");
    let topk = seq_len.min(window_size);
    let mut data = unsafe { ctx.stream.alloc(seq_len * topk)? };
    {
        let (out_ptr, _out_guard) = data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_window_topk_indices_cuda(
                out_ptr as *mut i32,
                seq_len as i32,
                window_size as i32,
                topk as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok((data, topk))
}

pub fn window_topk_indices_decode(
    ctx: &RankGpuContext,
    start_pos: usize,
    window_size: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(window_size > 0, "window_size must be positive");
    let mut data = unsafe { ctx.stream.alloc(window_size)? };
    let topk = window_topk_indices_decode_into(ctx, start_pos, window_size, &mut data)?;
    Ok((data, topk))
}

pub(crate) fn window_topk_indices_decode_into(
    ctx: &RankGpuContext,
    start_pos: usize,
    window_size: usize,
    out: &mut CudaSlice<i32>,
) -> Result<usize> {
    ctx.set_current()?;
    ensure!(window_size > 0, "window_size must be positive");
    ensure!(
        out.len() >= window_size,
        "window top-k decode output capacity too small: need {}, have {}",
        window_size,
        out.len()
    );
    {
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_window_topk_indices_decode_cuda(
                out_ptr as *mut i32,
                start_pos as i32,
                window_size as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(window_size)
}

pub(crate) fn window_topk_indices_decode_batch_into(
    ctx: &RankGpuContext,
    start_pos: &CudaSlice<i32>,
    cache_base: &CudaSlice<i32>,
    batch: usize,
    window_size: usize,
    out: &mut CudaSlice<i32>,
) -> Result<usize> {
    ctx.set_current()?;
    ensure!(batch > 0, "window top-k decode batch must be positive");
    ensure!(window_size > 0, "window_size must be positive");
    ensure!(
        start_pos.len() >= batch,
        "window top-k batch start_pos capacity too small: need {}, have {}",
        batch,
        start_pos.len()
    );
    ensure!(
        cache_base.len() >= batch,
        "window top-k batch cache_base capacity too small: need {}, have {}",
        batch,
        cache_base.len()
    );
    ensure!(
        out.len() >= batch * window_size,
        "window top-k batch output capacity too small: need {}, have {}",
        batch * window_size,
        out.len()
    );
    {
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
        let (start_ptr, _start_guard) = start_pos.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = cache_base.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_window_topk_indices_decode_batch_cuda(
                out_ptr as *mut i32,
                start_ptr as *const i32,
                base_ptr as *const i32,
                batch as i32,
                window_size as i32,
                window_size as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(window_size)
}

pub fn compress_topk_indices(
    ctx: &RankGpuContext,
    seq_len: usize,
    ratio: usize,
    offset: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(seq_len > 0, "seq_len must be positive");
    ensure!(ratio > 0, "compress ratio must be positive");
    let compressed = seq_len / ratio;
    ensure!(
        compressed > 0,
        "seq_len {seq_len} is smaller than ratio {ratio}"
    );
    let mut data = unsafe { ctx.stream.alloc(seq_len * compressed)? };
    {
        let (out_ptr, _out_guard) = data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compress_topk_indices_cuda(
                out_ptr as *mut i32,
                seq_len as i32,
                compressed as i32,
                ratio as i32,
                offset as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok((data, compressed))
}

pub fn compress_topk_indices_decode(
    ctx: &RankGpuContext,
    start_pos: usize,
    ratio: usize,
    offset: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(ratio > 0, "compress ratio must be positive");
    let compressed = (start_pos + 1) / ratio;
    let mut data = unsafe { ctx.stream.alloc(compressed)? };
    {
        let (out_ptr, _out_guard) = data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compress_topk_indices_decode_cuda(
                out_ptr as *mut i32,
                compressed as i32,
                offset as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok((data, compressed))
}

pub(crate) fn compress_topk_indices_decode_batch_into(
    ctx: &RankGpuContext,
    compressed_len: &CudaSlice<i32>,
    cache_base: &CudaSlice<i32>,
    batch: usize,
    topk: usize,
    out: &mut CudaSlice<i32>,
) -> Result<usize> {
    ctx.set_current()?;
    ensure!(batch > 0, "compress top-k decode batch must be positive");
    ensure!(
        compressed_len.len() >= batch,
        "compress top-k batch compressed_len capacity too small: need {}, have {}",
        batch,
        compressed_len.len()
    );
    ensure!(
        cache_base.len() >= batch,
        "compress top-k batch cache_base capacity too small: need {}, have {}",
        batch,
        cache_base.len()
    );
    ensure!(
        out.len() >= batch * topk,
        "compress top-k batch output capacity too small: need {}, have {}",
        batch * topk,
        out.len()
    );
    {
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
        let (compressed_ptr, _compressed_guard) = compressed_len.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = cache_base.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compress_topk_indices_decode_batch_cuda(
                out_ptr as *mut i32,
                compressed_ptr as *const i32,
                base_ptr as *const i32,
                batch as i32,
                topk as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(topk)
}

pub fn window_and_compress_topk_indices(
    ctx: &RankGpuContext,
    seq_len: usize,
    window_size: usize,
    ratio: usize,
    compress_offset: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(seq_len > 0, "seq_len must be positive");
    ensure!(window_size > 0, "window_size must be positive");
    ensure!(ratio > 0, "compress ratio must be positive");
    let window_topk = seq_len.min(window_size);
    let compressed = seq_len / ratio;
    ensure!(
        compressed > 0,
        "seq_len {seq_len} is smaller than ratio {ratio}"
    );
    let topk = window_topk + compressed;
    let mut data = unsafe { ctx.stream.alloc(seq_len * topk)? };
    {
        let (out_ptr, _out_guard) = data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_window_and_compress_topk_indices_cuda(
                out_ptr as *mut i32,
                seq_len as i32,
                window_size as i32,
                window_topk as i32,
                compressed as i32,
                ratio as i32,
                compress_offset as i32,
                topk as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok((data, topk))
}

pub fn concat_seq_bf16_hidden(
    ctx: &RankGpuContext,
    a: &Bf16HiddenStates,
    b: &Bf16HiddenStates,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        a.hidden_dim == b.hidden_dim,
        "concat hidden dim mismatch: a={}, b={}",
        a.hidden_dim,
        b.hidden_dim
    );
    let mut out = Bf16HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len + b.seq_len)?;
    {
        let (a_ptr, _a_guard) = a.data.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = b.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_concat_seq_bf16_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                a.seq_len as i32,
                b.seq_len as i32,
                a.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}
