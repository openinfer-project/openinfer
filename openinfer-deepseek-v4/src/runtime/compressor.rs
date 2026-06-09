use super::*;

pub fn compressor_nonoverlap_prefill_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    ratio: usize,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(ratio > 1, "compress ratio must be > 1");
    ensure!(ratio != 4, "ratio=4 uses the overlap compressor path");
    ensure!(
        start_pos == 0,
        "non-overlap compressor prefill currently supports start_pos=0 only"
    );
    ensure!(
        input.hidden_dim == config.dim,
        "compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    let compressed_len = input.seq_len / ratio;
    ensure!(
        compressed_len > 0,
        "input seq_len {} is smaller than ratio {}",
        input.seq_len,
        ratio
    );
    ensure!(
        compressor.ape.tensor.dtype == safetensors::Dtype::F32,
        "compressor ape {} must be F32, got {:?}",
        compressor.ape.name,
        compressor.ape.tensor.dtype
    );
    ensure!(
        compressor.wkv.tensor.dtype == safetensors::Dtype::BF16,
        "compressor wkv {} must be BF16, got {:?}",
        compressor.wkv.name,
        compressor.wkv.tensor.dtype
    );
    ensure!(
        compressor.wgate.tensor.dtype == safetensors::Dtype::BF16,
        "compressor wgate {} must be BF16, got {:?}",
        compressor.wgate.name,
        compressor.wgate.tensor.dtype
    );
    ensure!(
        compressor.norm.tensor.dtype == safetensors::Dtype::BF16,
        "compressor norm {} must be BF16, got {:?}",
        compressor.norm.name,
        compressor.norm.tensor.dtype
    );
    ensure!(
        compressor.ape.tensor.shape == [ratio, config.head_dim],
        "compressor ape {} shape mismatch: expected {:?}, got {:?}",
        compressor.ape.name,
        [ratio, config.head_dim],
        compressor.ape.tensor.shape
    );
    ensure!(
        compressor.wkv.tensor.shape == [config.head_dim, config.dim],
        "compressor wkv {} shape mismatch: expected {:?}, got {:?}",
        compressor.wkv.name,
        [config.head_dim, config.dim],
        compressor.wkv.tensor.shape
    );
    ensure!(
        compressor.wgate.tensor.shape == [config.head_dim, config.dim],
        "compressor wgate {} shape mismatch: expected {:?}, got {:?}",
        compressor.wgate.name,
        [config.head_dim, config.dim],
        compressor.wgate.tensor.shape
    );
    ensure!(
        compressor.norm.tensor.shape == [config.head_dim],
        "compressor norm {} shape mismatch: expected {:?}, got {:?}",
        compressor.norm.name,
        [config.head_dim],
        compressor.norm.tensor.shape
    );

    let mut weighted: CudaSlice<f32> = ctx.stream.alloc_zeros(compressed_len * config.head_dim)?;
    let mut out = Bf16HiddenStates::zeros(ctx, config.head_dim, compressed_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = weighted.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compressor_nonoverlap_prefill_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                weighted_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hidden_dim as i32,
                config.head_dim as i32,
                ratio as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    apply_rope_hidden_strided_in_place(
        ctx,
        &mut out,
        rope,
        1,
        config.head_dim,
        start_pos,
        ratio,
        false,
    )?;
    fp8_act_quant_nope_bf16_hidden_in_place(
        ctx,
        &mut out,
        1,
        config.head_dim,
        rope.rotary_dim,
        64,
    )?;
    Ok(out)
}

pub fn compressor_overlap_prefill_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    compressor_overlap_prefill_bf16_hidden_with_dim(
        ctx,
        config,
        input,
        compressor,
        rope,
        start_pos,
        config.head_dim,
    )
}

pub fn compressor_overlap_prefill_bf16_hidden_with_dim(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    head_dim: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        start_pos == 0,
        "overlap compressor prefill currently supports start_pos=0 only"
    );
    ensure!(
        input.hidden_dim == config.dim,
        "overlap compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    let ratio = 4;
    let compressed_len = input.seq_len / ratio;
    ensure!(
        compressed_len > 0,
        "input seq_len {} is smaller than ratio {}",
        input.seq_len,
        ratio
    );
    ensure!(
        compressor.ape.tensor.dtype == safetensors::Dtype::F32,
        "overlap compressor ape {} must be F32, got {:?}",
        compressor.ape.name,
        compressor.ape.tensor.dtype
    );
    ensure!(
        compressor.wkv.tensor.dtype == safetensors::Dtype::BF16,
        "overlap compressor wkv {} must be BF16, got {:?}",
        compressor.wkv.name,
        compressor.wkv.tensor.dtype
    );
    ensure!(
        compressor.wgate.tensor.dtype == safetensors::Dtype::BF16,
        "overlap compressor wgate {} must be BF16, got {:?}",
        compressor.wgate.name,
        compressor.wgate.tensor.dtype
    );
    ensure!(
        compressor.norm.tensor.dtype == safetensors::Dtype::BF16,
        "overlap compressor norm {} must be BF16, got {:?}",
        compressor.norm.name,
        compressor.norm.tensor.dtype
    );
    ensure!(
        compressor.ape.tensor.shape == [ratio, 2 * head_dim],
        "overlap compressor ape {} shape mismatch: expected {:?}, got {:?}",
        compressor.ape.name,
        [ratio, 2 * head_dim],
        compressor.ape.tensor.shape
    );
    ensure!(
        compressor.wkv.tensor.shape == [2 * head_dim, config.dim],
        "overlap compressor wkv {} shape mismatch: expected {:?}, got {:?}",
        compressor.wkv.name,
        [2 * head_dim, config.dim],
        compressor.wkv.tensor.shape
    );
    ensure!(
        compressor.wgate.tensor.shape == [2 * head_dim, config.dim],
        "overlap compressor wgate {} shape mismatch: expected {:?}, got {:?}",
        compressor.wgate.name,
        [2 * head_dim, config.dim],
        compressor.wgate.tensor.shape
    );
    ensure!(
        compressor.norm.tensor.shape == [head_dim],
        "overlap compressor norm {} shape mismatch: expected {:?}, got {:?}",
        compressor.norm.name,
        [head_dim],
        compressor.norm.tensor.shape
    );

    let mut weighted: CudaSlice<f32> = ctx.stream.alloc_zeros(compressed_len * head_dim)?;
    let mut out = Bf16HiddenStates::zeros(ctx, head_dim, compressed_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = weighted.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compressor_overlap_prefill_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                weighted_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hidden_dim as i32,
                head_dim as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    apply_rope_hidden_strided_in_place(ctx, &mut out, rope, 1, head_dim, start_pos, ratio, false)?;
    if head_dim == config.head_dim {
        fp8_act_quant_nope_bf16_hidden_in_place(ctx, &mut out, 1, head_dim, rope.rotary_dim, 64)?;
    }
    Ok(out)
}

pub fn compressor_nonoverlap_decode_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    ratio: usize,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    state: &mut CompressorDecodeState,
) -> Result<Option<Bf16HiddenStates>> {
    compressor_nonoverlap_decode_bf16_hidden_at(
        ctx, config, input, compressor, ratio, rope, start_pos, state, 0,
    )
}

pub(crate) fn compressor_nonoverlap_decode_bf16_hidden_at(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    ratio: usize,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    state: &mut CompressorDecodeState,
    state_offset: usize,
) -> Result<Option<Bf16HiddenStates>> {
    ctx.set_current()?;
    ensure!(ratio > 1, "compress ratio must be > 1");
    ensure!(ratio != 4, "ratio=4 uses the overlap compressor path");
    ensure!(
        input.hidden_dim == config.dim,
        "decode compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == 1,
        "decode compressor expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        state.hidden_dim == config.head_dim && state_offset + ratio <= state.slots,
        "decode compressor state mismatch: hidden_dim={}, slots={}, offset={}, need {} rows",
        state.hidden_dim,
        state.slots,
        state_offset,
        ratio
    );

    let should_compress = (start_pos + 1).is_multiple_of(ratio);
    let mut weighted = if should_compress {
        Some(ctx.stream.alloc_zeros::<f32>(config.head_dim)?)
    } else {
        None
    };
    let mut out = if should_compress {
        Some(Bf16HiddenStates::zeros(ctx, config.head_dim, 1)?)
    } else {
        None
    };
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (kv_state_ptr, _kv_state_guard) = state.kv.device_ptr_mut(&ctx.stream);
        let (score_state_ptr, _score_state_guard) = state.score.device_ptr_mut(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = if let Some(weighted) = weighted.as_mut() {
            let (ptr, guard) = weighted.device_ptr_mut(&ctx.stream);
            (ptr as *mut f32, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let (out_ptr, _out_guard) = if let Some(out) = out.as_mut() {
            let (ptr, guard) = out.data.device_ptr_mut(&ctx.stream);
            (ptr as *mut ffi::Half, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let result = unsafe {
            ffi::deepseek_compressor_nonoverlap_decode_at_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                kv_state_ptr as *mut f32,
                score_state_ptr as *mut f32,
                weighted_ptr,
                out_ptr,
                start_pos as i32,
                input.hidden_dim as i32,
                config.head_dim as i32,
                ratio as i32,
                state_offset as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    if let Some(mut out) = out {
        let rope_start = start_pos + 1 - ratio;
        apply_rope_hidden_strided_in_place(
            ctx,
            &mut out,
            rope,
            1,
            config.head_dim,
            rope_start,
            ratio,
            false,
        )?;
        fp8_act_quant_nope_bf16_hidden_in_place(
            ctx,
            &mut out,
            1,
            config.head_dim,
            rope.rotary_dim,
            64,
        )?;
        Ok(Some(out))
    } else {
        Ok(None)
    }
}

pub(crate) fn compressor_nonoverlap_decode_bf16_hidden_at_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    ratio: usize,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    state: &mut CompressorDecodeState,
    state_offset: usize,
    scratch: &'a mut AttentionAuxScratch,
) -> Result<Option<&'a Bf16HiddenStates>> {
    ctx.set_current()?;
    ensure!(ratio > 1, "compress ratio must be > 1");
    ensure!(ratio != 4, "ratio=4 uses the overlap compressor path");
    ensure!(
        input.hidden_dim == config.dim,
        "decode compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == 1,
        "decode compressor expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        state.hidden_dim == config.head_dim && state_offset + ratio <= state.slots,
        "decode compressor state mismatch: hidden_dim={}, slots={}, offset={}, need {} rows",
        state.hidden_dim,
        state.slots,
        state_offset,
        ratio
    );
    ensure!(
        config.head_dim <= scratch.max_head_dim,
        "decode compressor scratch head_dim capacity too small: need {}, have {}",
        config.head_dim,
        scratch.max_head_dim
    );
    ensure!(
        scratch.compressor_weighted.len() >= config.head_dim,
        "decode compressor weighted scratch too small: need {}, have {}",
        config.head_dim,
        scratch.compressor_weighted.len()
    );
    ensure!(
        scratch.compressor_out.data.len() >= config.head_dim,
        "decode compressor output scratch too small: need {}, have {}",
        config.head_dim,
        scratch.compressor_out.data.len()
    );

    let should_compress = (start_pos + 1).is_multiple_of(ratio);
    scratch.compressor_out.hidden_dim = config.head_dim;
    scratch.compressor_out.seq_len = 1;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (kv_state_ptr, _kv_state_guard) = state.kv.device_ptr_mut(&ctx.stream);
        let (score_state_ptr, _score_state_guard) = state.score.device_ptr_mut(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = if should_compress {
            let (ptr, guard) = scratch.compressor_weighted.device_ptr_mut(&ctx.stream);
            (ptr as *mut f32, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let (out_ptr, _out_guard) = if should_compress {
            let (ptr, guard) = scratch.compressor_out.data.device_ptr_mut(&ctx.stream);
            (ptr as *mut ffi::Half, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let result = unsafe {
            ffi::deepseek_compressor_nonoverlap_decode_at_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                kv_state_ptr as *mut f32,
                score_state_ptr as *mut f32,
                weighted_ptr,
                out_ptr,
                start_pos as i32,
                input.hidden_dim as i32,
                config.head_dim as i32,
                ratio as i32,
                state_offset as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    if should_compress {
        let rope_start = start_pos + 1 - ratio;
        apply_rope_hidden_strided_in_place(
            ctx,
            &mut scratch.compressor_out,
            rope,
            1,
            config.head_dim,
            rope_start,
            ratio,
            false,
        )?;
        fp8_act_quant_nope_bf16_hidden_in_place(
            ctx,
            &mut scratch.compressor_out,
            1,
            config.head_dim,
            rope.rotary_dim,
            64,
        )?;
        Ok(Some(&scratch.compressor_out))
    } else {
        Ok(None)
    }
}

pub fn compressor_overlap_decode_bf16_hidden_with_dim(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    head_dim: usize,
    state: &mut CompressorDecodeState,
    rotate_fp4: bool,
) -> Result<Option<Bf16HiddenStates>> {
    compressor_overlap_decode_bf16_hidden_with_dim_at(
        ctx, config, input, compressor, rope, start_pos, head_dim, state, 0, rotate_fp4,
    )
}

pub(crate) fn compressor_overlap_decode_bf16_hidden_with_dim_at(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    head_dim: usize,
    state: &mut CompressorDecodeState,
    state_offset: usize,
    rotate_fp4: bool,
) -> Result<Option<Bf16HiddenStates>> {
    ctx.set_current()?;
    ensure!(
        input.hidden_dim == config.dim,
        "overlap decode compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == 1,
        "overlap decode compressor expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        state.hidden_dim == 2 * head_dim && state_offset + 8 <= state.slots,
        "overlap decode compressor state mismatch: hidden_dim={}, slots={}, offset={}, need 8 rows",
        state.hidden_dim,
        state.slots,
        state_offset
    );

    let should_compress = (start_pos + 1).is_multiple_of(4);
    let mut weighted = if should_compress {
        Some(ctx.stream.alloc_zeros::<f32>(head_dim)?)
    } else {
        None
    };
    let mut out = if should_compress {
        Some(Bf16HiddenStates::zeros(ctx, head_dim, 1)?)
    } else {
        None
    };
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (kv_state_ptr, _kv_state_guard) = state.kv.device_ptr_mut(&ctx.stream);
        let (score_state_ptr, _score_state_guard) = state.score.device_ptr_mut(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = if let Some(weighted) = weighted.as_mut() {
            let (ptr, guard) = weighted.device_ptr_mut(&ctx.stream);
            (ptr as *mut f32, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let (out_ptr, _out_guard) = if let Some(out) = out.as_mut() {
            let (ptr, guard) = out.data.device_ptr_mut(&ctx.stream);
            (ptr as *mut ffi::Half, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let result = unsafe {
            ffi::deepseek_compressor_overlap_decode_at_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                kv_state_ptr as *mut f32,
                score_state_ptr as *mut f32,
                weighted_ptr,
                out_ptr,
                start_pos as i32,
                input.hidden_dim as i32,
                head_dim as i32,
                state_offset as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    if let Some(mut out) = out {
        let rope_start = start_pos + 1 - 4;
        apply_rope_hidden_strided_in_place(ctx, &mut out, rope, 1, head_dim, rope_start, 4, false)?;
        if rotate_fp4 {
            hadamard_fp4_quant_bf16_hidden_in_place(ctx, &mut out, 1, head_dim)?;
        } else {
            fp8_act_quant_nope_bf16_hidden_in_place(
                ctx,
                &mut out,
                1,
                head_dim,
                rope.rotary_dim,
                64,
            )?;
        }
        Ok(Some(out))
    } else {
        Ok(None)
    }
}

pub(crate) fn compressor_overlap_decode_bf16_hidden_with_dim_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    head_dim: usize,
    state: &mut CompressorDecodeState,
    rotate_fp4: bool,
    scratch: &'a mut AttentionAuxScratch,
) -> Result<Option<&'a Bf16HiddenStates>> {
    ctx.set_current()?;
    ensure!(
        input.hidden_dim == config.dim,
        "overlap decode compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == 1,
        "overlap decode compressor expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        state.hidden_dim == 2 * head_dim && state.slots >= 8,
        "overlap decode compressor state mismatch: hidden_dim={}, slots={}, expected at least {}x8",
        state.hidden_dim,
        state.slots,
        2 * head_dim
    );
    ensure!(
        head_dim <= scratch.max_head_dim,
        "overlap decode compressor scratch head_dim capacity too small: need {}, have {}",
        head_dim,
        scratch.max_head_dim
    );

    let should_compress = (start_pos + 1).is_multiple_of(4);
    scratch.compressor_out.hidden_dim = head_dim;
    scratch.compressor_out.seq_len = 1;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (kv_state_ptr, _kv_state_guard) = state.kv.device_ptr_mut(&ctx.stream);
        let (score_state_ptr, _score_state_guard) = state.score.device_ptr_mut(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = if should_compress {
            let (ptr, guard) = scratch.compressor_weighted.device_ptr_mut(&ctx.stream);
            (ptr as *mut f32, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let (out_ptr, _out_guard) = if should_compress {
            let (ptr, guard) = scratch.compressor_out.data.device_ptr_mut(&ctx.stream);
            (ptr as *mut ffi::Half, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let result = unsafe {
            ffi::deepseek_compressor_overlap_decode_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                kv_state_ptr as *mut f32,
                score_state_ptr as *mut f32,
                weighted_ptr,
                out_ptr,
                start_pos as i32,
                input.hidden_dim as i32,
                head_dim as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    if should_compress {
        let rope_start = start_pos + 1 - 4;
        apply_rope_hidden_strided_in_place(
            ctx,
            &mut scratch.compressor_out,
            rope,
            1,
            head_dim,
            rope_start,
            4,
            false,
        )?;
        if rotate_fp4 {
            hadamard_fp4_quant_bf16_hidden_in_place(ctx, &mut scratch.compressor_out, 1, head_dim)?;
        } else {
            fp8_act_quant_nope_bf16_hidden_in_place(
                ctx,
                &mut scratch.compressor_out,
                1,
                head_dim,
                rope.rotary_dim,
                64,
            )?;
        }
        Ok(Some(&scratch.compressor_out))
    } else {
        Ok(None)
    }
}

pub fn compressor_overlap_decode_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    state: &mut CompressorDecodeState,
) -> Result<Option<Bf16HiddenStates>> {
    ctx.set_current()?;
    compressor_overlap_decode_bf16_hidden_with_dim(
        ctx,
        config,
        input,
        compressor,
        rope,
        start_pos,
        config.head_dim,
        state,
        false,
    )
}

pub(crate) fn init_nonoverlap_compressor_state_from_prefill(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    ratio: usize,
    rope: &DeepSeekRopeCache,
    state: &mut CompressorDecodeState,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(ratio > 1, "compress ratio must be > 1");
    ensure!(ratio != 4, "ratio=4 uses overlap compressor state");
    let tail = input.seq_len % ratio;
    if tail == 0 {
        return Ok(());
    }
    let start = input.seq_len - tail;
    for pos in start..input.seq_len {
        let row = copy_bf16_row_to_hidden(ctx, input, pos)?;
        let _ = compressor_nonoverlap_decode_bf16_hidden(
            ctx, config, &row, compressor, ratio, rope, pos, state,
        )?;
    }
    Ok(())
}

pub(crate) fn init_overlap_compressor_state_from_prefill(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    head_dim: usize,
    state: &mut CompressorDecodeState,
    rotate_fp4: bool,
) -> Result<()> {
    ctx.set_current()?;
    let start = input.seq_len.saturating_sub(8);
    for pos in start..input.seq_len {
        let row = copy_bf16_row_to_hidden(ctx, input, pos)?;
        let _ = compressor_overlap_decode_bf16_hidden_with_dim(
            ctx, config, &row, compressor, rope, pos, head_dim, state, rotate_fp4,
        )?;
    }
    Ok(())
}

pub fn concat_topk_indices(
    ctx: &RankGpuContext,
    a: &CudaSlice<i32>,
    a_topk: usize,
    b: &CudaSlice<i32>,
    b_topk: usize,
    seq_len: usize,
) -> Result<CudaSlice<i32>> {
    ctx.set_current()?;
    ensure!(seq_len > 0, "top-k concat seq_len must be positive");
    ensure!(
        a.len() == seq_len * a_topk,
        "top-k concat left shape mismatch: expected {}, got {}",
        seq_len * a_topk,
        a.len()
    );
    ensure!(
        b.len() == seq_len * b_topk,
        "top-k concat right shape mismatch: expected {}, got {}",
        seq_len * b_topk,
        b.len()
    );
    let mut out = unsafe { ctx.stream.alloc(seq_len * (a_topk + b_topk))? };
    concat_topk_indices_into(ctx, a, a_topk, b, b_topk, seq_len, &mut out)?;
    Ok(out)
}

pub(crate) fn concat_topk_indices_into(
    ctx: &RankGpuContext,
    a: &CudaSlice<i32>,
    a_topk: usize,
    b: &CudaSlice<i32>,
    b_topk: usize,
    seq_len: usize,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(seq_len > 0, "top-k concat seq_len must be positive");
    ensure!(
        a.len() >= seq_len * a_topk,
        "top-k concat left capacity too small: need {}, have {}",
        seq_len * a_topk,
        a.len()
    );
    ensure!(
        b.len() >= seq_len * b_topk,
        "top-k concat right capacity too small: need {}, have {}",
        seq_len * b_topk,
        b.len()
    );
    ensure!(
        out.len() >= seq_len * (a_topk + b_topk),
        "top-k concat output capacity too small: need {}, have {}",
        seq_len * (a_topk + b_topk),
        out.len()
    );
    {
        let (a_ptr, _a_guard) = a.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = b.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_concat_topk_indices_cuda(
                a_ptr as *const i32,
                b_ptr as *const i32,
                out_ptr as *mut i32,
                seq_len as i32,
                a_topk as i32,
                b_topk as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}
