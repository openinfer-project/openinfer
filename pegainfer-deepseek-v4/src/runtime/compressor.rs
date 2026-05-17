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
    ensure!(
        state_offset % ratio == 0,
        "decode compressor state_offset {state_offset} must be multiple of ratio {ratio}"
    );

    // Per Review msg 027a1df4 constraint #3: route through batched FFI.
    let slot_id = state_offset / ratio;
    let should_compress = (start_pos + 1).is_multiple_of(ratio);
    let mut weighted = ctx.stream.alloc_zeros::<f32>(config.head_dim)?;
    let mut out = Bf16HiddenStates::zeros(ctx, config.head_dim, 1)?;
    let mut start_pos_d_one: CudaSlice<i32> = unsafe { ctx.stream.alloc(1)? };
    ctx.stream
        .memcpy_htod(&[start_pos as i32], &mut start_pos_d_one)?;
    let mut slot_ids_d_one: CudaSlice<i32> = unsafe { ctx.stream.alloc(1)? };
    ctx.stream
        .memcpy_htod(&[slot_id as i32], &mut slot_ids_d_one)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (kv_state_ptr, _kv_state_guard) = state.kv.device_ptr_mut(&ctx.stream);
        let (score_state_ptr, _score_state_guard) = state.score.device_ptr_mut(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = weighted.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let (start_pos_ptr, _start_pos_guard) = start_pos_d_one.device_ptr(&ctx.stream);
        let (slot_ids_ptr, _slot_ids_guard) = slot_ids_d_one.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compressor_nonoverlap_decode_batch_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                kv_state_ptr as *mut f32,
                score_state_ptr as *mut f32,
                weighted_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                start_pos_ptr as *const i32,
                slot_ids_ptr as *const i32,
                1,
                input.hidden_dim as i32,
                config.head_dim as i32,
                ratio as i32,
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
        let _ = (weighted, out);
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
    // Per Review msg 027a1df4 constraint #3: single-row path goes through the
    // same batched FFI to avoid two semantics. state_offset is the legacy
    // per-row addressing — derive slot_id from it (must be multiple of ratio
    // by construction; all callers pass state_offset = slot_id * ratio).
    ensure!(
        state_offset % ratio == 0,
        "decode compressor state_offset {state_offset} must be multiple of ratio {ratio}"
    );
    let slot_id = state_offset / ratio;
    let mut start_pos_d_one: CudaSlice<i32> = unsafe { ctx.stream.alloc(1)? };
    ctx.stream
        .memcpy_htod(&[start_pos as i32], &mut start_pos_d_one)?;
    let mut slot_ids_d_one: CudaSlice<i32> = unsafe { ctx.stream.alloc(1)? };
    ctx.stream
        .memcpy_htod(&[slot_id as i32], &mut slot_ids_d_one)?;
    let compressed_out = compressor_nonoverlap_decode_bf16_hidden_batch_scratch(
        ctx,
        config,
        input,
        compressor,
        ratio,
        &start_pos_d_one,
        &slot_ids_d_one,
        1,
        slot_id,
        state,
        scratch,
    )?;
    let should_compress = (start_pos + 1).is_multiple_of(ratio);
    if should_compress {
        let rope_start = start_pos + 1 - ratio;
        apply_rope_hidden_strided_in_place(
            ctx,
            &mut scratch.compressor_out_batch,
            rope,
            1,
            config.head_dim,
            rope_start,
            ratio,
            false,
        )?;
        fp8_act_quant_nope_bf16_hidden_in_place(
            ctx,
            &mut scratch.compressor_out_batch,
            1,
            config.head_dim,
            rope.rotary_dim,
            64,
        )?;
        Ok(Some(&scratch.compressor_out_batch))
    } else {
        let _ = compressed_out;
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
    ensure!(
        state_offset % 8 == 0,
        "overlap decode compressor state_offset {state_offset} must be multiple of 8 (state_offset_mul)"
    );

    // Per Review msg 027a1df4 constraint #3: route through batched FFI.
    let slot_id = state_offset / 8;
    let should_compress = (start_pos + 1).is_multiple_of(4);
    let mut weighted = ctx.stream.alloc_zeros::<f32>(head_dim)?;
    let mut out = Bf16HiddenStates::zeros(ctx, head_dim, 1)?;
    let mut start_pos_d_one: CudaSlice<i32> = unsafe { ctx.stream.alloc(1)? };
    ctx.stream
        .memcpy_htod(&[start_pos as i32], &mut start_pos_d_one)?;
    let mut slot_ids_d_one: CudaSlice<i32> = unsafe { ctx.stream.alloc(1)? };
    ctx.stream
        .memcpy_htod(&[slot_id as i32], &mut slot_ids_d_one)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (kv_state_ptr, _kv_state_guard) = state.kv.device_ptr_mut(&ctx.stream);
        let (score_state_ptr, _score_state_guard) = state.score.device_ptr_mut(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = weighted.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let (start_pos_ptr, _start_pos_guard) = start_pos_d_one.device_ptr(&ctx.stream);
        let (slot_ids_ptr, _slot_ids_guard) = slot_ids_d_one.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compressor_overlap_decode_batch_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                kv_state_ptr as *mut f32,
                score_state_ptr as *mut f32,
                weighted_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                start_pos_ptr as *const i32,
                slot_ids_ptr as *const i32,
                1,
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
        let _ = (weighted, out);
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
    // Per Review msg 027a1df4 constraint #3: single-row path goes through the
    // same batched FFI to avoid two semantics. slot_id is implicitly 0 here
    // (single-row overlap was state-local).
    let slot_id: usize = 0;
    let mut start_pos_d_one: CudaSlice<i32> = unsafe { ctx.stream.alloc(1)? };
    ctx.stream
        .memcpy_htod(&[start_pos as i32], &mut start_pos_d_one)?;
    let mut slot_ids_d_one: CudaSlice<i32> = unsafe { ctx.stream.alloc(1)? };
    ctx.stream
        .memcpy_htod(&[slot_id as i32], &mut slot_ids_d_one)?;
    let compressed_out = compressor_overlap_decode_bf16_hidden_batch_scratch(
        ctx,
        config,
        input,
        compressor,
        head_dim,
        &start_pos_d_one,
        &slot_ids_d_one,
        1,
        slot_id,
        state,
        scratch,
    )?;
    let should_compress = (start_pos + 1).is_multiple_of(4);
    if should_compress {
        let rope_start = start_pos + 1 - 4;
        apply_rope_hidden_strided_in_place(
            ctx,
            &mut scratch.compressor_out_batch,
            rope,
            1,
            head_dim,
            rope_start,
            4,
            false,
        )?;
        if rotate_fp4 {
            hadamard_fp4_quant_bf16_hidden_in_place(
                ctx,
                &mut scratch.compressor_out_batch,
                1,
                head_dim,
            )?;
        } else {
            fp8_act_quant_nope_bf16_hidden_in_place(
                ctx,
                &mut scratch.compressor_out_batch,
                1,
                head_dim,
                rope.rotary_dim,
                64,
            )?;
        }
        Ok(Some(&scratch.compressor_out_batch))
    } else {
        let _ = compressed_out;
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

// =============================================================================
// task #46 — batched decode compressor wrappers.
//
// These wrappers drive the new batched FFI from `pegainfer-kernels`:
//   * `deepseek_compressor_nonoverlap_decode_batch_cuda`
//   * `deepseek_compressor_overlap_decode_batch_cuda`
//
// They run the 3 (non-overlap) / 4 (overlap) batched kernels (projection,
// weighted softmax, RMS norm[, shift]) for an entire active set in one shot.
// Per-row state offsets are computed in-kernel from `batch_meta.slot_ids`
// and `batch_meta.start_pos` device arrays (no host->device sync per step).
//
// Mask handling (Review msg 027a1df4 option B): false-mask rows early-exit
// inside the kernel; the corresponding rows of `compressor_*_batch` scratch
// buffers are undefined output. Callers MUST mask via host
// `batch_meta.start_pos_host[row]` before reading or scattering any row.
//
// Post-kernel RoPE and FP8 quantization remain per-row in the caller for
// this iteration. The existing strided RoPE / FP8 APIs assume a uniform
// per-row start_pos which is not true here; a per-row-variable RoPE kernel
// is deferred to a follow-up PR. The dominant cost (per-row projection
// GEMM) is what this batching collapses.
// =============================================================================

pub(crate) fn compressor_nonoverlap_decode_bf16_hidden_batch_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    ratio: usize,
    start_pos_d: &CudaSlice<i32>,
    slot_ids_d: &CudaSlice<i32>,
    batch: usize,
    max_slot_id: usize,
    state: &mut CompressorDecodeState,
    scratch: &'a mut AttentionAuxScratch,
) -> Result<&'a Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(ratio > 1, "compress ratio must be > 1");
    ensure!(ratio != 4, "ratio=4 uses the overlap compressor path");
    ensure!(
        input.hidden_dim == config.dim,
        "batched decode compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == batch,
        "batched decode compressor seq_len mismatch: input={}, batch={}",
        input.seq_len,
        batch
    );
    ensure!(batch > 0, "batched decode compressor batch must be > 0");
    ensure!(
        start_pos_d.len() >= batch,
        "batched decode compressor start_pos device array too small: need {}, have {}",
        batch,
        start_pos_d.len()
    );
    ensure!(
        slot_ids_d.len() >= batch,
        "batched decode compressor slot_ids device array too small: need {}, have {}",
        batch,
        slot_ids_d.len()
    );
    ensure!(
        state.hidden_dim == config.head_dim,
        "batched decode compressor state hidden_dim mismatch: state={}, head_dim={}",
        state.hidden_dim,
        config.head_dim
    );
    ensure!(
        config.head_dim <= scratch.max_head_dim,
        "batched decode compressor scratch head_dim capacity too small: need {}, have {}",
        config.head_dim,
        scratch.max_head_dim
    );
    ensure!(
        scratch.compressor_weighted_batch.len() >= batch * config.head_dim,
        "batched decode compressor weighted scratch too small: need {}, have {}",
        batch * config.head_dim,
        scratch.compressor_weighted_batch.len()
    );
    ensure!(
        scratch.compressor_out_batch.seq_capacity() >= batch,
        "batched decode compressor out scratch seq capacity too small: need {}, have {}",
        batch,
        scratch.compressor_out_batch.seq_capacity()
    );
    ensure!(
        scratch.compressor_out_batch.hidden_dim >= config.head_dim,
        "batched decode compressor out scratch hidden_dim too small: need {}, have {}",
        config.head_dim,
        scratch.compressor_out_batch.hidden_dim
    );
    // Per-row slot capacity check: each row stores `ratio` rows in the per-slot
    // state window; max slot index must fit within state.slots.
    ensure!(
        (max_slot_id + 1) * ratio <= state.slots,
        "batched decode compressor state slots too small: max_slot_id={}, ratio={}, slots={}",
        max_slot_id,
        ratio,
        state.slots
    );

    scratch.compressor_out_batch.hidden_dim = config.head_dim;
    scratch.compressor_out_batch.seq_len = batch;

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (kv_state_ptr, _kv_state_guard) = state.kv.device_ptr_mut(&ctx.stream);
        let (score_state_ptr, _score_state_guard) = state.score.device_ptr_mut(&ctx.stream);
        let (weighted_ptr, _weighted_guard) =
            scratch.compressor_weighted_batch.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) =
            scratch.compressor_out_batch.data.device_ptr_mut(&ctx.stream);
        let (start_pos_ptr, _start_pos_guard) = start_pos_d.device_ptr(&ctx.stream);
        let (slot_ids_ptr, _slot_ids_guard) = slot_ids_d.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compressor_nonoverlap_decode_batch_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                kv_state_ptr as *mut f32,
                score_state_ptr as *mut f32,
                weighted_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                start_pos_ptr as *const i32,
                slot_ids_ptr as *const i32,
                batch as i32,
                input.hidden_dim as i32,
                config.head_dim as i32,
                ratio as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(&scratch.compressor_out_batch)
}

pub(crate) fn compressor_overlap_decode_bf16_hidden_batch_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    head_dim: usize,
    start_pos_d: &CudaSlice<i32>,
    slot_ids_d: &CudaSlice<i32>,
    batch: usize,
    max_slot_id: usize,
    state: &mut CompressorDecodeState,
    scratch: &'a mut AttentionAuxScratch,
) -> Result<&'a Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        input.hidden_dim == config.dim,
        "batched overlap compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == batch,
        "batched overlap compressor seq_len mismatch: input={}, batch={}",
        input.seq_len,
        batch
    );
    ensure!(batch > 0, "batched overlap compressor batch must be > 0");
    ensure!(
        start_pos_d.len() >= batch,
        "batched overlap compressor start_pos device array too small: need {}, have {}",
        batch,
        start_pos_d.len()
    );
    ensure!(
        slot_ids_d.len() >= batch,
        "batched overlap compressor slot_ids device array too small: need {}, have {}",
        batch,
        slot_ids_d.len()
    );
    ensure!(
        state.hidden_dim == head_dim,
        "batched overlap compressor state hidden_dim mismatch: state={}, head_dim={}",
        state.hidden_dim,
        head_dim
    );
    ensure!(
        head_dim <= scratch.max_head_dim,
        "batched overlap compressor scratch head_dim capacity too small: need {}, have {}",
        head_dim,
        scratch.max_head_dim
    );
    ensure!(
        scratch.compressor_weighted_batch.len() >= batch * head_dim,
        "batched overlap compressor weighted scratch too small: need {}, have {}",
        batch * head_dim,
        scratch.compressor_weighted_batch.len()
    );
    ensure!(
        scratch.compressor_out_batch.seq_capacity() >= batch,
        "batched overlap compressor out scratch seq capacity too small: need {}, have {}",
        batch,
        scratch.compressor_out_batch.seq_capacity()
    );
    ensure!(
        scratch.compressor_out_batch.hidden_dim >= head_dim,
        "batched overlap compressor out scratch hidden_dim too small: need {}, have {}",
        head_dim,
        scratch.compressor_out_batch.hidden_dim
    );
    // Overlap stores 8 rows per slot (state_offset_mul=8).
    ensure!(
        (max_slot_id + 1) * 8 <= state.slots,
        "batched overlap compressor state slots too small: max_slot_id={}, slots={}",
        max_slot_id,
        state.slots
    );

    scratch.compressor_out_batch.hidden_dim = head_dim;
    scratch.compressor_out_batch.seq_len = batch;

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (kv_state_ptr, _kv_state_guard) = state.kv.device_ptr_mut(&ctx.stream);
        let (score_state_ptr, _score_state_guard) = state.score.device_ptr_mut(&ctx.stream);
        let (weighted_ptr, _weighted_guard) =
            scratch.compressor_weighted_batch.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) =
            scratch.compressor_out_batch.data.device_ptr_mut(&ctx.stream);
        let (start_pos_ptr, _start_pos_guard) = start_pos_d.device_ptr(&ctx.stream);
        let (slot_ids_ptr, _slot_ids_guard) = slot_ids_d.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compressor_overlap_decode_batch_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                kv_state_ptr as *mut f32,
                score_state_ptr as *mut f32,
                weighted_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                start_pos_ptr as *const i32,
                slot_ids_ptr as *const i32,
                batch as i32,
                input.hidden_dim as i32,
                head_dim as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(&scratch.compressor_out_batch)
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
