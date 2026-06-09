use super::*;

pub fn hc_expand_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    hc: usize,
) -> Result<HcHiddenStates> {
    ctx.set_current()?;
    ensure!(hc > 0, "HC multiplier must be positive");
    let mut out = HcHiddenStates::uninit(ctx, input.hidden_dim, input.seq_len, hc)?;
    hc_expand_bf16_hidden_into(ctx, input, hc, &mut out)?;
    Ok(out)
}

pub(crate) fn hc_expand_bf16_hidden_into(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    hc: usize,
    out: &mut HcHiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(hc > 0, "HC multiplier must be positive");
    ensure!(
        out.hidden_dim == input.hidden_dim,
        "HC expand scratch hidden dim mismatch: out={}, input={}",
        out.hidden_dim,
        input.hidden_dim
    );
    ensure!(
        out.seq_capacity() >= input.seq_len,
        "HC expand scratch capacity too small: out={}, input={}",
        out.seq_capacity(),
        input.seq_len
    );
    ensure!(
        out.hc == hc,
        "HC expand scratch multiplier mismatch: out={}, expected={hc}",
        out.hc
    );
    out.seq_len = input.seq_len;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_expand_cuda(
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                hc as i32,
                input.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn hc_head_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &HcHiddenStates,
    hc_fn: &TensorRef<'_>,
    hc_scale: &TensorRef<'_>,
    hc_base: &TensorRef<'_>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        input.hc == config.hc_mult,
        "HC head multiplier mismatch: expected {}, got {}",
        config.hc_mult,
        input.hc
    );
    ensure!(
        input.hidden_dim == config.dim,
        "HC head hidden dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        hc_fn.tensor.dtype == safetensors::Dtype::F32,
        "HC head fn {} must be F32, got {:?}",
        hc_fn.name,
        hc_fn.tensor.dtype
    );
    ensure!(
        hc_scale.tensor.dtype == safetensors::Dtype::F32,
        "HC head scale {} must be F32, got {:?}",
        hc_scale.name,
        hc_scale.tensor.dtype
    );
    ensure!(
        hc_base.tensor.dtype == safetensors::Dtype::F32,
        "HC head base {} must be F32, got {:?}",
        hc_base.name,
        hc_base.tensor.dtype
    );

    let hc_dim = input.hc * input.hidden_dim;
    ensure!(
        hc_fn.tensor.shape == [input.hc, hc_dim],
        "HC head fn {} shape mismatch: expected {:?}, got {:?}",
        hc_fn.name,
        [input.hc, hc_dim],
        hc_fn.tensor.shape
    );
    ensure!(
        hc_scale.tensor.shape == [1],
        "HC head scale {} shape mismatch: expected {:?}, got {:?}",
        hc_scale.name,
        [1],
        hc_scale.tensor.shape
    );
    ensure!(
        hc_base.tensor.shape == [input.hc],
        "HC head base {} shape mismatch: expected {:?}, got {:?}",
        hc_base.name,
        [input.hc],
        hc_base.tensor.shape
    );

    let mut mixes: CudaSlice<f32> = unsafe { ctx.stream.alloc(input.seq_len * input.hc)? };
    let mut pre: CudaSlice<f32> = unsafe { ctx.stream.alloc(input.seq_len * input.hc)? };
    let mut out = Bf16HiddenStates::uninit(ctx, input.hidden_dim, input.seq_len)?;

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (fn_ptr, _fn_guard) = hc_fn.tensor.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_mixes_cuda(
                x_ptr as *const ffi::Half,
                fn_ptr as *const f32,
                mixes_ptr as *mut f32,
                ptr::null_mut(),
                ptr::null_mut(),
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                input.hc as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    {
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = hc_scale.tensor.data.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = hc_base.tensor.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = pre.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_head_pre_cuda(
                mixes_ptr as *const f32,
                scale_ptr as *const f32,
                base_ptr as *const f32,
                pre_ptr as *mut f32,
                input.seq_len as i32,
                input.hc as i32,
                config.hc_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = pre.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_pre_output_cuda(
                x_ptr as *const ffi::Half,
                pre_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_head_bf16_hidden_into(
    ctx: &RankGpuContext,
    config: &Config,
    input: &HcHiddenStates,
    hc_fn: &TensorRef<'_>,
    hc_scale: &TensorRef<'_>,
    hc_base: &TensorRef<'_>,
    mixes: &mut CudaSlice<f32>,
    pre: &mut CudaSlice<f32>,
    out: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        input.hc == config.hc_mult,
        "HC head multiplier mismatch: expected {}, got {}",
        config.hc_mult,
        input.hc
    );
    ensure!(
        input.hidden_dim == config.dim,
        "HC head hidden dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        hc_fn.tensor.dtype == safetensors::Dtype::F32,
        "HC head fn {} must be F32, got {:?}",
        hc_fn.name,
        hc_fn.tensor.dtype
    );
    ensure!(
        hc_scale.tensor.dtype == safetensors::Dtype::F32,
        "HC head scale {} must be F32, got {:?}",
        hc_scale.name,
        hc_scale.tensor.dtype
    );
    ensure!(
        hc_base.tensor.dtype == safetensors::Dtype::F32,
        "HC head base {} must be F32, got {:?}",
        hc_base.name,
        hc_base.tensor.dtype
    );

    let hc_dim = input.hc * input.hidden_dim;
    ensure!(
        hc_fn.tensor.shape == [input.hc, hc_dim],
        "HC head fn {} shape mismatch: expected {:?}, got {:?}",
        hc_fn.name,
        [input.hc, hc_dim],
        hc_fn.tensor.shape
    );
    ensure!(
        hc_scale.tensor.shape == [1],
        "HC head scale {} shape mismatch: expected {:?}, got {:?}",
        hc_scale.name,
        [1],
        hc_scale.tensor.shape
    );
    ensure!(
        hc_base.tensor.shape == [input.hc],
        "HC head base {} shape mismatch: expected {:?}, got {:?}",
        hc_base.name,
        [input.hc],
        hc_base.tensor.shape
    );
    ensure!(
        mixes.len() >= input.seq_len * input.hc,
        "HC head mixes scratch too small: have {}, need {}",
        mixes.len(),
        input.seq_len * input.hc
    );
    ensure!(
        pre.len() >= input.seq_len * input.hc,
        "HC head pre scratch too small: have {}, need {}",
        pre.len(),
        input.seq_len * input.hc
    );
    ensure!(
        out.hidden_dim == input.hidden_dim,
        "HC head output dim mismatch: out={}, input={}",
        out.hidden_dim,
        input.hidden_dim
    );
    ensure!(
        out.seq_capacity() >= input.seq_len,
        "HC head output capacity too small: out={}, input={}",
        out.seq_capacity(),
        input.seq_len
    );
    out.seq_len = input.seq_len;

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (fn_ptr, _fn_guard) = hc_fn.tensor.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_mixes_cuda(
                x_ptr as *const ffi::Half,
                fn_ptr as *const f32,
                mixes_ptr as *mut f32,
                ptr::null_mut(),
                ptr::null_mut(),
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                input.hc as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    {
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = hc_scale.tensor.data.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = hc_base.tensor.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = pre.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_head_pre_cuda(
                mixes_ptr as *const f32,
                scale_ptr as *const f32,
                base_ptr as *const f32,
                pre_ptr as *mut f32,
                input.seq_len as i32,
                input.hc as i32,
                config.hc_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = pre.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_pre_output_cuda(
                x_ptr as *const ffi::Half,
                pre_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(())
}

pub fn rank_local_logits_from_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    head: &TensorRef<'_>,
) -> Result<F32Logits> {
    ctx.set_current()?;
    ensure!(
        head.tensor.dtype == safetensors::Dtype::BF16,
        "head weight {} must be BF16, got {:?}",
        head.name,
        head.tensor.dtype
    );
    ensure!(
        head.tensor.shape.len() == 2,
        "head weight {} must be rank-2, got {:?}",
        head.name,
        head.tensor.shape
    );
    let vocab_size = head.tensor.shape[0];
    let hidden_dim = head.tensor.shape[1];
    ensure!(
        hidden_dim == input.hidden_dim,
        "head input dim mismatch: head expects {}, got {}",
        hidden_dim,
        input.hidden_dim
    );
    ensure!(input.seq_len > 0, "logits input seq_len must be positive");

    let mut out = unsafe { ctx.stream.alloc(vocab_size)? };
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (head_ptr, _head_guard) = head.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_last_token_bf16_logits_cuda(
                x_ptr as *const ffi::Half,
                head_ptr as *const ffi::Half,
                out_ptr as *mut f32,
                input.seq_len as i32,
                input.hidden_dim as i32,
                vocab_size as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(F32Logits {
        data: out,
        vocab_size,
    })
}

pub(crate) fn rank_local_logits_from_hidden_into(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    head: &TensorRef<'_>,
    out: &mut F32Logits,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        head.tensor.dtype == safetensors::Dtype::BF16,
        "head weight {} must be BF16, got {:?}",
        head.name,
        head.tensor.dtype
    );
    ensure!(
        head.tensor.shape.len() == 2,
        "head weight {} must be rank-2, got {:?}",
        head.name,
        head.tensor.shape
    );
    let vocab_size = head.tensor.shape[0];
    let hidden_dim = head.tensor.shape[1];
    ensure!(
        hidden_dim == input.hidden_dim,
        "head input dim mismatch: head expects {}, got {}",
        hidden_dim,
        input.hidden_dim
    );
    ensure!(input.seq_len > 0, "logits input seq_len must be positive");
    ensure!(
        out.vocab_size == vocab_size,
        "local logits scratch vocab mismatch: out={}, expected={vocab_size}",
        out.vocab_size
    );
    ensure!(
        out.data.len() >= vocab_size,
        "local logits scratch capacity too small: out={}, expected={vocab_size}",
        out.data.len()
    );
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (head_ptr, _head_guard) = head.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_last_token_bf16_logits_cuda(
                x_ptr as *const ffi::Half,
                head_ptr as *const ffi::Half,
                out_ptr as *mut f32,
                input.seq_len as i32,
                input.hidden_dim as i32,
                vocab_size as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub(crate) fn rank_local_logits_from_hidden_all(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    head: &TensorRef<'_>,
) -> Result<F32BatchLogits> {
    ctx.set_current()?;
    ensure!(
        head.tensor.dtype == safetensors::Dtype::BF16,
        "head weight {} must be BF16, got {:?}",
        head.name,
        head.tensor.dtype
    );
    ensure!(
        head.tensor.shape.len() == 2,
        "head weight {} must be rank-2, got {:?}",
        head.name,
        head.tensor.shape
    );
    let vocab_size = head.tensor.shape[0];
    let hidden_dim = head.tensor.shape[1];
    ensure!(
        hidden_dim == input.hidden_dim,
        "head input dim mismatch: head expects {}, got {}",
        hidden_dim,
        input.hidden_dim
    );
    ensure!(input.seq_len > 0, "logits input seq_len must be positive");
    let mut out = unsafe { ctx.stream.alloc(input.seq_len * vocab_size)? };
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (head_ptr, _head_guard) = head.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_bf16_logits_cuda(
                x_ptr as *const ffi::Half,
                head_ptr as *const ffi::Half,
                out_ptr as *mut f32,
                input.seq_len as i32,
                input.hidden_dim as i32,
                vocab_size as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(F32BatchLogits {
        data: out,
        vocab_size,
        seq_len: input.seq_len,
    })
}

pub fn final_logits_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    input: &HcHiddenStates,
) -> Result<F32Logits> {
    ctx.set_current()?;
    let hidden = hc_head_bf16_hidden(
        ctx,
        config,
        input,
        &weights.hc_head_fn()?,
        &weights.hc_head_scale()?,
        &weights.hc_head_base()?,
    )?;
    let normed = rms_norm_bf16_hidden(ctx, &hidden, &weights.norm()?, config.rms_norm_eps)?;
    rank_local_logits_from_hidden(ctx, &normed, &weights.head()?)
}

pub(crate) fn final_logits_rank_local_bf16_hidden_into(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    input: &HcHiddenStates,
    scratch: &mut FinalLogitsScratch,
) -> Result<()> {
    ctx.set_current()?;
    hc_head_bf16_hidden_into(
        ctx,
        config,
        input,
        &weights.hc_head_fn()?,
        &weights.hc_head_scale()?,
        &weights.hc_head_base()?,
        &mut scratch.hc_mixes,
        &mut scratch.hc_pre,
        &mut scratch.hc_out,
    )?;
    rms_norm_bf16_hidden_into(
        ctx,
        &scratch.hc_out,
        &weights.norm()?,
        config.rms_norm_eps,
        &mut scratch.normed,
    )?;
    rank_local_logits_from_hidden_into(
        ctx,
        &scratch.normed,
        &weights.head()?,
        &mut scratch.local_logits,
    )
}

pub(crate) fn all_gather_logits(
    ctx: &RankGpuContext,
    comm: &Comm,
    local: &F32Logits,
    world_size: usize,
) -> Result<F32Logits> {
    ctx.set_current()?;
    ensure!(
        world_size > 0,
        "logits all-gather world size must be positive"
    );
    ensure!(local.vocab_size > 0, "local vocab size must be positive");
    let mut gathered = F32Logits {
        data: ctx.stream.alloc_zeros(local.vocab_size * world_size)?,
        vocab_size: local.vocab_size * world_size,
    };
    comm.all_gather(&local.data, &mut gathered.data)
        .map_err(|err| anyhow::anyhow!("NCCL logits all-gather failed: {err:?}"))?;
    Ok(gathered)
}

pub(crate) fn all_gather_logits_into(
    ctx: &RankGpuContext,
    comm: &Comm,
    local: &F32Logits,
    world_size: usize,
    gathered: &mut F32Logits,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        world_size > 0,
        "logits all-gather world size must be positive"
    );
    ensure!(local.vocab_size > 0, "local vocab size must be positive");
    ensure!(
        gathered.vocab_size == local.vocab_size * world_size,
        "gathered logits vocab mismatch: gathered={}, expected={}",
        gathered.vocab_size,
        local.vocab_size * world_size
    );
    ensure!(
        gathered.data.len() >= local.vocab_size * world_size,
        "gathered logits capacity too small: gathered={}, expected={}",
        gathered.data.len(),
        local.vocab_size * world_size
    );
    comm.all_gather(&local.data, &mut gathered.data)
        .map_err(|err| anyhow::anyhow!("NCCL logits all-gather failed: {err:?}"))?;
    Ok(())
}

pub fn hc_pre_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &HcHiddenStates,
    hc_fn: &TensorRef<'_>,
    hc_scale: &TensorRef<'_>,
    hc_base: &TensorRef<'_>,
) -> Result<(Bf16HiddenStates, HcPreState)> {
    ctx.set_current()?;
    ensure!(
        input.hc == config.hc_mult,
        "HC input multiplier mismatch: expected {}, got {}",
        config.hc_mult,
        input.hc
    );
    ensure!(
        input.hidden_dim == config.dim,
        "HC input hidden dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        hc_fn.tensor.dtype == safetensors::Dtype::F32,
        "HC fn {} must be F32, got {:?}",
        hc_fn.name,
        hc_fn.tensor.dtype
    );
    ensure!(
        hc_scale.tensor.dtype == safetensors::Dtype::F32,
        "HC scale {} must be F32, got {:?}",
        hc_scale.name,
        hc_scale.tensor.dtype
    );
    ensure!(
        hc_base.tensor.dtype == safetensors::Dtype::F32,
        "HC base {} must be F32, got {:?}",
        hc_base.name,
        hc_base.tensor.dtype
    );

    let mix_hc = (2 + input.hc) * input.hc;
    let hc_dim = input.hc * input.hidden_dim;
    ensure!(
        hc_fn.tensor.shape == [mix_hc, hc_dim],
        "HC fn {} shape mismatch: expected {:?}, got {:?}",
        hc_fn.name,
        [mix_hc, hc_dim],
        hc_fn.tensor.shape
    );
    ensure!(
        hc_scale.tensor.shape == [3],
        "HC scale {} shape mismatch: expected {:?}, got {:?}",
        hc_scale.name,
        [3],
        hc_scale.tensor.shape
    );
    ensure!(
        hc_base.tensor.shape == [mix_hc],
        "HC base {} shape mismatch: expected {:?}, got {:?}",
        hc_base.name,
        [mix_hc],
        hc_base.tensor.shape
    );

    let mut mixes: CudaSlice<f32> = unsafe { ctx.stream.alloc(input.seq_len * mix_hc)? };
    let mut post: CudaSlice<f32> = unsafe { ctx.stream.alloc(input.seq_len * input.hc)? };
    let mut comb: CudaSlice<f32> =
        unsafe { ctx.stream.alloc(input.seq_len * input.hc * input.hc)? };
    let mut out = Bf16HiddenStates::uninit(ctx, input.hidden_dim, input.seq_len)?;

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (fn_ptr, _fn_guard) = hc_fn.tensor.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_mixes_cuda(
                x_ptr as *const ffi::Half,
                fn_ptr as *const f32,
                mixes_ptr as *mut f32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                mix_hc as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    let use_fused_pre = input.seq_len == 1
        && input.hc == 4
        && config.hc_sinkhorn_iters == 20
        && (config.hc_eps - 1.0e-6).abs() <= 1.0e-12;

    if use_fused_pre {
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = hc_scale.tensor.data.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = hc_base.tensor.data.device_ptr(&ctx.stream);
        let (post_ptr, _post_guard) = post.device_ptr_mut(&ctx.stream);
        let (comb_ptr, _comb_guard) = comb.device_ptr_mut(&ctx.stream);
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_pre_from_mixes_cuda(
                x_ptr as *const ffi::Half,
                mixes_ptr as *const f32,
                scale_ptr as *const f32,
                base_ptr as *const f32,
                post_ptr as *mut f32,
                comb_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                config.hc_sinkhorn_iters as i32,
                config.hc_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    } else {
        let mut pre: CudaSlice<f32> = unsafe { ctx.stream.alloc(input.seq_len * input.hc)? };
        {
            let (mixes_ptr, _mixes_guard) = mixes.device_ptr(&ctx.stream);
            let (scale_ptr, _scale_guard) = hc_scale.tensor.data.device_ptr(&ctx.stream);
            let (base_ptr, _base_guard) = hc_base.tensor.data.device_ptr(&ctx.stream);
            let (pre_ptr, _pre_guard) = pre.device_ptr_mut(&ctx.stream);
            let (post_ptr, _post_guard) = post.device_ptr_mut(&ctx.stream);
            let (comb_ptr, _comb_guard) = comb.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                ffi::deepseek_hc_split_sinkhorn_cuda(
                    mixes_ptr as *const f32,
                    scale_ptr as *const f32,
                    base_ptr as *const f32,
                    pre_ptr as *mut f32,
                    post_ptr as *mut f32,
                    comb_ptr as *mut f32,
                    input.seq_len as i32,
                    input.hc as i32,
                    config.hc_sinkhorn_iters as i32,
                    config.hc_eps,
                    ctx.stream.cu_stream(),
                )
            };
            result.result()?;
        }

        {
            let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
            let (pre_ptr, _pre_guard) = pre.device_ptr(&ctx.stream);
            let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                ffi::deepseek_hc_pre_output_cuda(
                    x_ptr as *const ffi::Half,
                    pre_ptr as *const f32,
                    out_ptr as *mut ffi::Half,
                    input.seq_len as i32,
                    input.hc as i32,
                    input.hidden_dim as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result()?;
        }
    }

    Ok((
        out,
        HcPreState {
            post,
            comb,
            seq_len: input.seq_len,
            hc: input.hc,
        },
    ))
}

pub fn hc_pre_norm_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &HcHiddenStates,
    hc_fn: &TensorRef<'_>,
    hc_scale: &TensorRef<'_>,
    hc_base: &TensorRef<'_>,
    norm_weight: &TensorRef<'_>,
) -> Result<(Bf16HiddenStates, HcPreState)> {
    let use_fused_pre_norm = input.seq_len == 1
        && input.hc == 4
        && config.hc_sinkhorn_iters == 20
        && (config.hc_eps - 1.0e-6).abs() <= 1.0e-12;
    if !use_fused_pre_norm {
        let (pre, state) = hc_pre_bf16_hidden(ctx, config, input, hc_fn, hc_scale, hc_base)?;
        let normed = rms_norm_bf16_hidden(ctx, &pre, norm_weight, config.rms_norm_eps)?;
        return Ok((normed, state));
    }

    ctx.set_current()?;
    ensure!(
        input.hidden_dim == config.dim,
        "HC norm input hidden dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        hc_fn.tensor.dtype == safetensors::Dtype::F32,
        "HC fn {} must be F32, got {:?}",
        hc_fn.name,
        hc_fn.tensor.dtype
    );
    ensure!(
        hc_scale.tensor.dtype == safetensors::Dtype::F32,
        "HC scale {} must be F32, got {:?}",
        hc_scale.name,
        hc_scale.tensor.dtype
    );
    ensure!(
        hc_base.tensor.dtype == safetensors::Dtype::F32,
        "HC base {} must be F32, got {:?}",
        hc_base.name,
        hc_base.tensor.dtype
    );
    ensure!(
        norm_weight.tensor.dtype == safetensors::Dtype::BF16,
        "HC fused norm weight {} must be BF16, got {:?}",
        norm_weight.name,
        norm_weight.tensor.dtype
    );

    let mix_hc = (2 + input.hc) * input.hc;
    let hc_dim = input.hc * input.hidden_dim;
    ensure!(
        hc_fn.tensor.shape == [mix_hc, hc_dim],
        "HC fn {} shape mismatch: expected {:?}, got {:?}",
        hc_fn.name,
        [mix_hc, hc_dim],
        hc_fn.tensor.shape
    );
    ensure!(
        hc_scale.tensor.shape == [3],
        "HC scale {} shape mismatch: expected {:?}, got {:?}",
        hc_scale.name,
        [3],
        hc_scale.tensor.shape
    );
    ensure!(
        hc_base.tensor.shape == [mix_hc],
        "HC base {} shape mismatch: expected {:?}, got {:?}",
        hc_base.name,
        [mix_hc],
        hc_base.tensor.shape
    );
    ensure!(
        norm_weight.tensor.shape == [input.hidden_dim],
        "HC fused norm weight {} shape mismatch: expected {:?}, got {:?}",
        norm_weight.name,
        [input.hidden_dim],
        norm_weight.tensor.shape
    );

    let mut mixes: CudaSlice<f32> = unsafe { ctx.stream.alloc(input.seq_len * mix_hc)? };
    let mut post: CudaSlice<f32> = unsafe { ctx.stream.alloc(input.seq_len * input.hc)? };
    let mut comb: CudaSlice<f32> =
        unsafe { ctx.stream.alloc(input.seq_len * input.hc * input.hc)? };
    let mut out = Bf16HiddenStates::uninit(ctx, input.hidden_dim, input.seq_len)?;

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (fn_ptr, _fn_guard) = hc_fn.tensor.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_mixes_cuda(
                x_ptr as *const ffi::Half,
                fn_ptr as *const f32,
                mixes_ptr as *mut f32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                mix_hc as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = hc_scale.tensor.data.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = hc_base.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = norm_weight.tensor.data.device_ptr(&ctx.stream);
        let (post_ptr, _post_guard) = post.device_ptr_mut(&ctx.stream);
        let (comb_ptr, _comb_guard) = comb.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_pre_norm_from_mixes_cuda(
                x_ptr as *const ffi::Half,
                mixes_ptr as *const f32,
                scale_ptr as *const f32,
                base_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                post_ptr as *mut f32,
                comb_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                config.hc_sinkhorn_iters as i32,
                config.hc_eps,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok((
        out,
        HcPreState {
            post,
            comb,
            seq_len: input.seq_len,
            hc: input.hc,
        },
    ))
}

pub(crate) fn hc_pre_norm_bf16_hidden_scratch<'a>(
    ctx: &RankGpuContext,
    config: &Config,
    input: &HcHiddenStates,
    hc_fn: &TensorRef<'_>,
    hc_scale: &TensorRef<'_>,
    hc_base: &TensorRef<'_>,
    norm_weight: &TensorRef<'_>,
    scratch: &'a mut HcPreNormScratch,
) -> Result<(&'a Bf16HiddenStates, HcPreStateView<'a>)> {
    ctx.set_current()?;
    ensure!(
        input.hc == config.hc_mult,
        "HC fused norm input multiplier mismatch: expected {}, got {}",
        config.hc_mult,
        input.hc
    );
    ensure!(
        input.hidden_dim == config.dim,
        "HC fused norm input hidden dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len <= scratch.seq_capacity,
        "HC pre-norm scratch seq capacity too small: need {}, have {}",
        input.seq_len,
        scratch.seq_capacity
    );
    ensure!(
        scratch.hidden_dim == input.hidden_dim && scratch.hc == input.hc,
        "HC pre-norm scratch shape mismatch: scratch dim/hc={}/{}, input dim/hc={}/{}",
        scratch.hidden_dim,
        scratch.hc,
        input.hidden_dim,
        input.hc
    );
    ensure!(
        hc_fn.tensor.dtype == safetensors::Dtype::F32,
        "HC fn {} must be F32, got {:?}",
        hc_fn.name,
        hc_fn.tensor.dtype
    );
    ensure!(
        hc_scale.tensor.dtype == safetensors::Dtype::F32,
        "HC scale {} must be F32, got {:?}",
        hc_scale.name,
        hc_scale.tensor.dtype
    );
    ensure!(
        hc_base.tensor.dtype == safetensors::Dtype::F32,
        "HC base {} must be F32, got {:?}",
        hc_base.name,
        hc_base.tensor.dtype
    );
    ensure!(
        norm_weight.tensor.dtype == safetensors::Dtype::BF16,
        "HC fused norm weight {} must be BF16, got {:?}",
        norm_weight.name,
        norm_weight.tensor.dtype
    );

    let mix_hc = (2 + input.hc) * input.hc;
    let hc_dim = input.hc * input.hidden_dim;
    ensure!(
        hc_fn.tensor.shape == [mix_hc, hc_dim],
        "HC fn {} shape mismatch: expected {:?}, got {:?}",
        hc_fn.name,
        [mix_hc, hc_dim],
        hc_fn.tensor.shape
    );
    ensure!(
        hc_scale.tensor.shape == [3],
        "HC scale {} shape mismatch: expected {:?}, got {:?}",
        hc_scale.name,
        [3],
        hc_scale.tensor.shape
    );
    ensure!(
        hc_base.tensor.shape == [mix_hc],
        "HC base {} shape mismatch: expected {:?}, got {:?}",
        hc_base.name,
        [mix_hc],
        hc_base.tensor.shape
    );
    ensure!(
        norm_weight.tensor.shape == [input.hidden_dim],
        "HC fused norm weight {} shape mismatch: expected {:?}, got {:?}",
        norm_weight.name,
        [input.hidden_dim],
        norm_weight.tensor.shape
    );

    scratch.out.seq_len = input.seq_len;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (fn_ptr, _fn_guard) = hc_fn.tensor.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = scratch.mixes.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_mixes_cuda(
                x_ptr as *const ffi::Half,
                fn_ptr as *const f32,
                mixes_ptr as *mut f32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                mix_hc as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = scratch.mixes.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = hc_scale.tensor.data.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = hc_base.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = norm_weight.tensor.data.device_ptr(&ctx.stream);
        let (post_ptr, _post_guard) = scratch.post.device_ptr_mut(&ctx.stream);
        let (comb_ptr, _comb_guard) = scratch.comb.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = scratch.out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_pre_norm_from_mixes_cuda(
                x_ptr as *const ffi::Half,
                mixes_ptr as *const f32,
                scale_ptr as *const f32,
                base_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                post_ptr as *mut f32,
                comb_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                config.hc_sinkhorn_iters as i32,
                config.hc_eps,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok((
        &scratch.out,
        HcPreStateView {
            post: &scratch.post,
            comb: &scratch.comb,
            seq_len: input.seq_len,
            hc: input.hc,
        },
    ))
}

pub fn hc_post_bf16_hidden(
    ctx: &RankGpuContext,
    branch_out: &Bf16HiddenStates,
    residual: &HcHiddenStates,
    pre_state: &HcPreState,
) -> Result<HcHiddenStates> {
    ctx.set_current()?;
    ensure!(
        branch_out.hidden_dim == residual.hidden_dim,
        "HC post hidden dim mismatch: branch={}, residual={}",
        branch_out.hidden_dim,
        residual.hidden_dim
    );
    ensure!(
        branch_out.seq_len == residual.seq_len,
        "HC post seq len mismatch: branch={}, residual={}",
        branch_out.seq_len,
        residual.seq_len
    );
    ensure!(
        pre_state.seq_len == branch_out.seq_len,
        "HC post pre-state seq len mismatch: state={}, branch={}",
        pre_state.seq_len,
        branch_out.seq_len
    );
    ensure!(
        pre_state.hc == residual.hc,
        "HC post pre-state multiplier mismatch: state={}, residual={}",
        pre_state.hc,
        residual.hc
    );

    let mut out =
        HcHiddenStates::uninit(ctx, branch_out.hidden_dim, branch_out.seq_len, residual.hc)?;
    {
        let (x_ptr, _x_guard) = branch_out.data.device_ptr(&ctx.stream);
        let (residual_ptr, _residual_guard) = residual.data.device_ptr(&ctx.stream);
        let (post_ptr, _post_guard) = pre_state.post.device_ptr(&ctx.stream);
        let (comb_ptr, _comb_guard) = pre_state.comb.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_post_cuda(
                x_ptr as *const ffi::Half,
                residual_ptr as *const ffi::Half,
                post_ptr as *const f32,
                comb_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                branch_out.seq_len as i32,
                residual.hc as i32,
                branch_out.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub(crate) fn hc_post_bf16_hidden_view_into(
    ctx: &RankGpuContext,
    branch_out: &Bf16HiddenStates,
    residual: &HcHiddenStates,
    pre_state: &HcPreStateView<'_>,
    out: &mut HcHiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        branch_out.hidden_dim == residual.hidden_dim,
        "HC post hidden dim mismatch: branch={}, residual={}",
        branch_out.hidden_dim,
        residual.hidden_dim
    );
    ensure!(
        branch_out.seq_len == residual.seq_len,
        "HC post seq len mismatch: branch={}, residual={}",
        branch_out.seq_len,
        residual.seq_len
    );
    ensure!(
        pre_state.seq_len == branch_out.seq_len,
        "HC post pre-state seq len mismatch: state={}, branch={}",
        pre_state.seq_len,
        branch_out.seq_len
    );
    ensure!(
        pre_state.hc == residual.hc,
        "HC post pre-state multiplier mismatch: state={}, residual={}",
        pre_state.hc,
        residual.hc
    );
    ensure!(
        out.hidden_dim == branch_out.hidden_dim,
        "HC post output hidden dim mismatch: out={}, branch={}",
        out.hidden_dim,
        branch_out.hidden_dim
    );
    ensure!(
        out.seq_capacity() >= branch_out.seq_len,
        "HC post output capacity too small: out={}, branch={}",
        out.seq_capacity(),
        branch_out.seq_len
    );
    ensure!(
        out.hc == residual.hc,
        "HC post output multiplier mismatch: out={}, residual={}",
        out.hc,
        residual.hc
    );
    out.seq_len = branch_out.seq_len;

    {
        let (x_ptr, _x_guard) = branch_out.data.device_ptr(&ctx.stream);
        let (residual_ptr, _residual_guard) = residual.data.device_ptr(&ctx.stream);
        let (post_ptr, _post_guard) = pre_state.post.device_ptr(&ctx.stream);
        let (comb_ptr, _comb_guard) = pre_state.comb.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_post_cuda(
                x_ptr as *const ffi::Half,
                residual_ptr as *const ffi::Half,
                post_ptr as *const f32,
                comb_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                branch_out.seq_len as i32,
                residual.hc as i32,
                branch_out.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn embedding_rank_local(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    token_ids: &CudaSlice<u32>,
    seq_len: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let mut out = Bf16HiddenStates::uninit(ctx, config.dim, seq_len)?;
    embedding_rank_local_into(ctx, config, weights, token_ids, seq_len, &mut out)?;
    Ok(out)
}

pub(crate) fn embedding_rank_local_into(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    token_ids: &CudaSlice<u32>,
    seq_len: usize,
    out: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    let embed = weights.embed()?;
    ensure!(
        embed.tensor.shape == [config.vocab_size / 8, config.dim],
        "unexpected embed shape {:?}",
        embed.tensor.shape
    );
    ensure!(
        out.hidden_dim == config.dim,
        "embedding scratch hidden dim mismatch: out={}, expected={}",
        out.hidden_dim,
        config.dim
    );
    ensure!(
        out.seq_capacity() >= seq_len,
        "embedding scratch capacity too small: out={}, requested={seq_len}",
        out.seq_capacity()
    );
    out.seq_len = seq_len;

    {
        let (embed_ptr, _embed_guard) = embed.tensor.data.device_ptr(&ctx.stream);
        let (token_ptr, _token_guard) = token_ids.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let vocab_start = (weights.rank() * (config.vocab_size / 8)) as u32;
        let part_vocab_size = (config.vocab_size / 8) as u32;

        let result = unsafe {
            ffi::embedding_batched_vocab_shard_cuda(
                embed_ptr as *const ffi::Half,
                token_ptr as *const u32,
                out_ptr as *mut ffi::Half,
                config.dim as i32,
                seq_len as i32,
                vocab_start,
                part_vocab_size,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(())
}

pub fn rms_norm_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    weight: &crate::model::TensorRef<'_>,
    eps: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        weight.tensor.dtype == safetensors::Dtype::BF16,
        "RMSNorm weight {} must be BF16, got {:?}",
        weight.name,
        weight.tensor.dtype
    );
    ensure!(
        weight.tensor.shape == [input.hidden_dim],
        "RMSNorm weight {} shape mismatch: expected {:?}, got {:?}",
        weight.name,
        [input.hidden_dim],
        weight.tensor.shape
    );

    let mut out = Bf16HiddenStates::uninit(ctx, input.hidden_dim, input.seq_len)?;
    rms_norm_bf16_hidden_into(ctx, input, weight, eps, &mut out)?;
    Ok(out)
}

pub(crate) fn rms_norm_bf16_hidden_into(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    weight: &crate::model::TensorRef<'_>,
    eps: f32,
    out: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        weight.tensor.dtype == safetensors::Dtype::BF16,
        "RMSNorm weight {} must be BF16, got {:?}",
        weight.name,
        weight.tensor.dtype
    );
    ensure!(
        weight.tensor.shape == [input.hidden_dim],
        "RMSNorm weight {} shape mismatch: expected {:?}, got {:?}",
        weight.name,
        [input.hidden_dim],
        weight.tensor.shape
    );
    ensure!(
        out.hidden_dim == input.hidden_dim,
        "RMSNorm output dim mismatch: expected {}, got {}",
        input.hidden_dim,
        out.hidden_dim
    );
    ensure!(
        out.data.len() >= input.hidden_dim * input.seq_len,
        "RMSNorm output capacity too small: need {}, have {}",
        input.hidden_dim * input.seq_len,
        out.data.len()
    );
    out.seq_len = input.seq_len;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = weight.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::rms_norm_batched_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                input.hidden_dim as i32,
                input.seq_len as i32,
                eps,
                ctx.stream.cu_stream(),
            );
        }
    }
    Ok(())
}

pub fn fp8_linear_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    linear: &QuantLinearRef<'_>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        linear.weight.tensor.dtype == safetensors::Dtype::F8_E4M3,
        "FP8 linear weight {} must be F8_E4M3, got {:?}",
        linear.weight.name,
        linear.weight.tensor.dtype
    );
    ensure!(
        linear.scale.tensor.dtype == safetensors::Dtype::F8_E8M0,
        "FP8 linear scale {} must be F8_E8M0, got {:?}",
        linear.scale.name,
        linear.scale.tensor.dtype
    );
    ensure!(
        linear.weight.tensor.shape.len() == 2,
        "FP8 linear weight {} must be rank-2, got {:?}",
        linear.weight.name,
        linear.weight.tensor.shape
    );
    let out_dim = linear.weight.tensor.shape[0];
    let in_dim = linear.weight.tensor.shape[1];
    ensure!(
        in_dim == input.hidden_dim,
        "FP8 linear input dim mismatch: weight {} expects {}, got {}",
        linear.weight.name,
        in_dim,
        input.hidden_dim
    );
    ensure!(
        linear.scale.tensor.shape == [out_dim.div_ceil(128), in_dim.div_ceil(128)],
        "FP8 linear scale {} shape mismatch: expected {:?}, got {:?}",
        linear.scale.name,
        [out_dim.div_ceil(128), in_dim.div_ceil(128)],
        linear.scale.tensor.shape
    );

    let mut out = Bf16HiddenStates::uninit(ctx, out_dim, input.seq_len)?;
    fp8_linear_bf16_hidden_into(ctx, input, linear, &mut out)?;
    Ok(out)
}

pub(crate) fn fp8_linear_bf16_hidden_into(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    linear: &QuantLinearRef<'_>,
    out: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        linear.weight.tensor.dtype == safetensors::Dtype::F8_E4M3,
        "FP8 linear weight {} must be F8_E4M3, got {:?}",
        linear.weight.name,
        linear.weight.tensor.dtype
    );
    ensure!(
        linear.scale.tensor.dtype == safetensors::Dtype::F8_E8M0,
        "FP8 linear scale {} must be F8_E8M0, got {:?}",
        linear.scale.name,
        linear.scale.tensor.dtype
    );
    ensure!(
        linear.weight.tensor.shape.len() == 2,
        "FP8 linear weight {} must be rank-2, got {:?}",
        linear.weight.name,
        linear.weight.tensor.shape
    );
    let out_dim = linear.weight.tensor.shape[0];
    let in_dim = linear.weight.tensor.shape[1];
    ensure!(
        in_dim == input.hidden_dim,
        "FP8 linear input dim mismatch: weight {} expects {}, got {}",
        linear.weight.name,
        in_dim,
        input.hidden_dim
    );
    ensure!(
        linear.scale.tensor.shape == [out_dim.div_ceil(128), in_dim.div_ceil(128)],
        "FP8 linear scale {} shape mismatch: expected {:?}, got {:?}",
        linear.scale.name,
        [out_dim.div_ceil(128), in_dim.div_ceil(128)],
        linear.scale.tensor.shape
    );
    ensure!(
        out.hidden_dim == out_dim,
        "FP8 linear output dim mismatch: expected {}, got {}",
        out_dim,
        out.hidden_dim
    );
    ensure!(
        out.data.len() >= out_dim * input.seq_len,
        "FP8 linear output seq capacity too small: need {}, have {}",
        out_dim * input.seq_len,
        out.data.len()
    );
    out.seq_len = input.seq_len;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = linear.weight.tensor.data.device_ptr(&ctx.stream);
        let (s_ptr, _s_guard) = linear.scale.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_fp8_linear_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const u8,
                s_ptr as *const u8,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                in_dim as i32,
                out_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub(crate) fn fp8_w1_w3_bf16_hidden_into(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    w1: &QuantLinearRef<'_>,
    w3: &QuantLinearRef<'_>,
    gate_out: &mut Bf16HiddenStates,
    up_out: &mut Bf16HiddenStates,
    act_workspace: &mut CudaSlice<u8>,
    act_scale_workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        w1.weight.tensor.dtype == safetensors::Dtype::F8_E4M3
            && w3.weight.tensor.dtype == safetensors::Dtype::F8_E4M3,
        "shared W1/W3 weights must be F8_E4M3, got {:?}/{:?}",
        w1.weight.tensor.dtype,
        w3.weight.tensor.dtype
    );
    ensure!(
        w1.scale.tensor.dtype == safetensors::Dtype::F8_E8M0
            && w3.scale.tensor.dtype == safetensors::Dtype::F8_E8M0,
        "shared W1/W3 scales must be F8_E8M0, got {:?}/{:?}",
        w1.scale.tensor.dtype,
        w3.scale.tensor.dtype
    );
    ensure!(
        w1.weight.tensor.shape.len() == 2 && w3.weight.tensor.shape.len() == 2,
        "shared W1/W3 weights must be rank-2, got {:?}/{:?}",
        w1.weight.tensor.shape,
        w3.weight.tensor.shape
    );
    let out_dim = w1.weight.tensor.shape[0];
    let in_dim = w1.weight.tensor.shape[1];
    ensure!(
        w3.weight.tensor.shape == [out_dim, in_dim],
        "shared W1/W3 shape mismatch: w1={:?}, w3={:?}",
        w1.weight.tensor.shape,
        w3.weight.tensor.shape
    );
    ensure!(
        in_dim == input.hidden_dim,
        "shared W1/W3 input dim mismatch: weight expects {}, got {}",
        in_dim,
        input.hidden_dim
    );
    let scale_shape = [out_dim.div_ceil(128), in_dim.div_ceil(128)];
    ensure!(
        w1.scale.tensor.shape == scale_shape && w3.scale.tensor.shape == scale_shape,
        "shared W1/W3 scale shape mismatch: expected {:?}, got {:?}/{:?}",
        scale_shape,
        w1.scale.tensor.shape,
        w3.scale.tensor.shape
    );
    ensure!(
        gate_out.hidden_dim == out_dim && up_out.hidden_dim == out_dim,
        "shared W1/W3 output dim mismatch: gate={}, up={}, expected={}",
        gate_out.hidden_dim,
        up_out.hidden_dim,
        out_dim
    );
    ensure!(
        gate_out.seq_capacity() >= input.seq_len && up_out.seq_capacity() >= input.seq_len,
        "shared W1/W3 output capacity too small: gate={}, up={}, required={}",
        gate_out.seq_capacity(),
        up_out.seq_capacity(),
        input.seq_len
    );
    let act_bytes = input.seq_len * in_dim;
    let act_scale_bytes = input.seq_len * in_dim.div_ceil(128);
    ensure!(
        act_workspace.len() >= act_bytes,
        "shared W1/W3 act workspace too small: have {}, need {act_bytes}",
        act_workspace.len()
    );
    ensure!(
        act_scale_workspace.len() >= act_scale_bytes,
        "shared W1/W3 act scale workspace too small: have {}, need {act_scale_bytes}",
        act_scale_workspace.len()
    );
    gate_out.seq_len = input.seq_len;
    up_out.seq_len = input.seq_len;
    {
        let act_workspace_len = act_workspace.len();
        let act_scale_workspace_len = act_scale_workspace.len();
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (w1_ptr, _w1_guard) = w1.weight.tensor.data.device_ptr(&ctx.stream);
        let (w1_scale_ptr, _w1_scale_guard) = w1.scale.tensor.data.device_ptr(&ctx.stream);
        let (w3_ptr, _w3_guard) = w3.weight.tensor.data.device_ptr(&ctx.stream);
        let (w3_scale_ptr, _w3_scale_guard) = w3.scale.tensor.data.device_ptr(&ctx.stream);
        let (gate_ptr, _gate_guard) = gate_out.data.device_ptr_mut(&ctx.stream);
        let (up_ptr, _up_guard) = up_out.data.device_ptr_mut(&ctx.stream);
        let (act_ptr, _act_guard) = act_workspace.device_ptr_mut(&ctx.stream);
        let (act_scale_ptr, _act_scale_guard) = act_scale_workspace.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_fp8_w1_w3_with_workspace_cuda(
                x_ptr as *const ffi::Half,
                w1_ptr as *const u8,
                w1_scale_ptr as *const u8,
                w3_ptr as *const u8,
                w3_scale_ptr as *const u8,
                gate_ptr as *mut ffi::Half,
                up_ptr as *mut ffi::Half,
                act_ptr as *mut u8,
                act_workspace_len,
                act_scale_ptr as *mut u8,
                act_scale_workspace_len,
                input.seq_len as i32,
                in_dim as i32,
                out_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fp8_w2_swiglu_bf16_hidden_into(
    ctx: &RankGpuContext,
    gate: &Bf16HiddenStates,
    up: &Bf16HiddenStates,
    limit: f32,
    w2: &QuantLinearRef<'_>,
    out: &mut Bf16HiddenStates,
    act_workspace: &mut CudaSlice<u8>,
    act_scale_workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        w2.weight.tensor.dtype == safetensors::Dtype::F8_E4M3,
        "shared W2 weight {} must be F8_E4M3, got {:?}",
        w2.weight.name,
        w2.weight.tensor.dtype
    );
    ensure!(
        w2.scale.tensor.dtype == safetensors::Dtype::F8_E8M0,
        "shared W2 scale {} must be F8_E8M0, got {:?}",
        w2.scale.name,
        w2.scale.tensor.dtype
    );
    ensure!(
        w2.weight.tensor.shape.len() == 2,
        "shared W2 weight {} must be rank-2, got {:?}",
        w2.weight.name,
        w2.weight.tensor.shape
    );
    let out_dim = w2.weight.tensor.shape[0];
    let in_dim = w2.weight.tensor.shape[1];
    ensure!(
        gate.hidden_dim == in_dim && up.hidden_dim == in_dim,
        "shared W2 SwiGLU input dim mismatch: gate={}, up={}, expected={}",
        gate.hidden_dim,
        up.hidden_dim,
        in_dim
    );
    ensure!(
        gate.seq_len == up.seq_len,
        "shared W2 SwiGLU seq len mismatch: gate={}, up={}",
        gate.seq_len,
        up.seq_len
    );
    ensure!(
        w2.scale.tensor.shape == [out_dim.div_ceil(128), in_dim.div_ceil(128)],
        "shared W2 scale {} shape mismatch: expected {:?}, got {:?}",
        w2.scale.name,
        [out_dim.div_ceil(128), in_dim.div_ceil(128)],
        w2.scale.tensor.shape
    );
    ensure!(
        out.hidden_dim == out_dim,
        "shared W2 output dim mismatch: expected {}, got {}",
        out_dim,
        out.hidden_dim
    );
    ensure!(
        out.seq_capacity() >= gate.seq_len,
        "shared W2 output capacity too small: out={}, required={}",
        out.seq_capacity(),
        gate.seq_len
    );
    let act_bytes = gate.seq_len * in_dim;
    let act_scale_bytes = gate.seq_len * in_dim.div_ceil(128);
    ensure!(
        act_workspace.len() >= act_bytes,
        "shared W2 act workspace too small: have {}, need {act_bytes}",
        act_workspace.len()
    );
    ensure!(
        act_scale_workspace.len() >= act_scale_bytes,
        "shared W2 act scale workspace too small: have {}, need {act_scale_bytes}",
        act_scale_workspace.len()
    );
    out.seq_len = gate.seq_len;
    {
        let act_workspace_len = act_workspace.len();
        let act_scale_workspace_len = act_scale_workspace.len();
        let (gate_ptr, _gate_guard) = gate.data.device_ptr(&ctx.stream);
        let (up_ptr, _up_guard) = up.data.device_ptr(&ctx.stream);
        let (weight_ptr, _weight_guard) = w2.weight.tensor.data.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = w2.scale.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let (act_ptr, _act_guard) = act_workspace.device_ptr_mut(&ctx.stream);
        let (act_scale_ptr, _act_scale_guard) = act_scale_workspace.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_fp8_w2_swiglu_with_workspace_cuda(
                gate_ptr as *const ffi::Half,
                up_ptr as *const ffi::Half,
                weight_ptr as *const u8,
                scale_ptr as *const u8,
                out_ptr as *mut ffi::Half,
                act_ptr as *mut u8,
                act_workspace_len,
                act_scale_ptr as *mut u8,
                act_scale_workspace_len,
                gate.seq_len as i32,
                in_dim as i32,
                out_dim as i32,
                limit,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn fp4_linear_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    linear: &QuantLinearRef<'_>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        linear.weight.tensor.dtype == safetensors::Dtype::F4,
        "FP4 linear weight {} must be F4, got {:?}",
        linear.weight.name,
        linear.weight.tensor.dtype
    );
    ensure!(
        linear.scale.tensor.dtype == safetensors::Dtype::F8_E8M0,
        "FP4 linear scale {} must be F8_E8M0, got {:?}",
        linear.scale.name,
        linear.scale.tensor.dtype
    );
    ensure!(
        linear.weight.tensor.shape.len() == 2,
        "FP4 linear weight {} must be rank-2, got {:?}",
        linear.weight.name,
        linear.weight.tensor.shape
    );
    let out_dim = linear.weight.tensor.shape[0];
    let in_dim = linear.weight.tensor.shape[1];
    ensure!(
        in_dim == input.hidden_dim,
        "FP4 linear input dim mismatch: weight {} expects {}, got {}",
        linear.weight.name,
        in_dim,
        input.hidden_dim
    );
    ensure!(
        in_dim.is_multiple_of(32),
        "FP4 linear input dim must be divisible by 32, got {in_dim}"
    );
    ensure!(
        linear.scale.tensor.shape == [out_dim, in_dim / 32],
        "FP4 linear scale {} shape mismatch: expected {:?}, got {:?}",
        linear.scale.name,
        [out_dim, in_dim / 32],
        linear.scale.tensor.shape
    );

    let mut out = Bf16HiddenStates::uninit(ctx, out_dim, input.seq_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = linear.weight.tensor.data.device_ptr(&ctx.stream);
        let (s_ptr, _s_guard) = linear.scale.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_fp4_linear_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const u8,
                s_ptr as *const u8,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                in_dim as i32,
                out_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn bf16_linear_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    weight: &TensorRef<'_>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        weight.tensor.dtype == safetensors::Dtype::BF16,
        "BF16 linear weight {} must be BF16, got {:?}",
        weight.name,
        weight.tensor.dtype
    );
    ensure!(
        weight.tensor.shape.len() == 2,
        "BF16 linear weight {} must be rank-2, got {:?}",
        weight.name,
        weight.tensor.shape
    );
    let out_dim = weight.tensor.shape[0];
    let in_dim = weight.tensor.shape[1];
    ensure!(
        in_dim == input.hidden_dim,
        "BF16 linear input dim mismatch: weight {} expects {}, got {}",
        weight.name,
        in_dim,
        input.hidden_dim
    );

    let mut out = Bf16HiddenStates::uninit(ctx, out_dim, input.seq_len)?;
    bf16_linear_bf16_hidden_into(ctx, input, weight, &mut out)?;
    Ok(out)
}

pub(crate) fn bf16_linear_bf16_hidden_into(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    weight: &TensorRef<'_>,
    out: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        weight.tensor.dtype == safetensors::Dtype::BF16,
        "BF16 linear weight {} must be BF16, got {:?}",
        weight.name,
        weight.tensor.dtype
    );
    ensure!(
        weight.tensor.shape.len() == 2,
        "BF16 linear weight {} must be rank-2, got {:?}",
        weight.name,
        weight.tensor.shape
    );
    let out_dim = weight.tensor.shape[0];
    let in_dim = weight.tensor.shape[1];
    ensure!(
        in_dim == input.hidden_dim,
        "BF16 linear input dim mismatch: weight {} expects {}, got {}",
        weight.name,
        in_dim,
        input.hidden_dim
    );
    ensure!(
        out.hidden_dim == out_dim,
        "BF16 linear output dim mismatch: expected {}, got {}",
        out_dim,
        out.hidden_dim
    );
    ensure!(
        out.data.len() >= out_dim * input.seq_len,
        "BF16 linear output seq capacity too small: need {}, have {}",
        out_dim * input.seq_len,
        out.data.len()
    );
    out.seq_len = input.seq_len;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = weight.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_bf16_linear_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                in_dim as i32,
                out_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn shared_expert_forward_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    ffn: &FfnWeights<'_>,
    swiglu_limit: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let inter_dim = ffn.shared_w1.weight.tensor.shape[0];
    let out_dim = ffn.shared_w2.weight.tensor.shape[0];
    let mut gate = Bf16HiddenStates::uninit(ctx, inter_dim, input.seq_len)?;
    let mut up = Bf16HiddenStates::uninit(ctx, inter_dim, input.seq_len)?;
    let mut out = Bf16HiddenStates::uninit(ctx, out_dim, input.seq_len)?;
    let max_fp8_input_dim = input.hidden_dim.max(inter_dim);
    let mut fp8_act_workspace =
        unsafe { ctx.stream.alloc::<u8>(input.seq_len * max_fp8_input_dim)? };
    let mut fp8_act_scale_workspace = unsafe {
        ctx.stream
            .alloc::<u8>(input.seq_len * max_fp8_input_dim.div_ceil(128))?
    };
    fp8_w1_w3_bf16_hidden_into(
        ctx,
        input,
        &ffn.shared_w1,
        &ffn.shared_w3,
        &mut gate,
        &mut up,
        &mut fp8_act_workspace,
        &mut fp8_act_scale_workspace,
    )?;
    fp8_w2_swiglu_bf16_hidden_into(
        ctx,
        &gate,
        &up,
        swiglu_limit,
        &ffn.shared_w2,
        &mut out,
        &mut fp8_act_workspace,
        &mut fp8_act_scale_workspace,
    )?;
    Ok(out)
}

pub(crate) fn shared_expert_forward_bf16_hidden_scratch<'a>(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    ffn: &FfnWeights<'_>,
    swiglu_limit: f32,
    scratch: &'a mut SharedExpertScratch,
) -> Result<&'a Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        input.seq_len <= scratch.seq_capacity,
        "shared expert scratch seq capacity too small: need {}, have {}",
        input.seq_len,
        scratch.seq_capacity
    );
    fp8_w1_w3_bf16_hidden_into(
        ctx,
        input,
        &ffn.shared_w1,
        &ffn.shared_w3,
        &mut scratch.gate,
        &mut scratch.up,
        &mut scratch.fp8_act_workspace,
        &mut scratch.fp8_act_scale_workspace,
    )?;
    fp8_w2_swiglu_bf16_hidden_into(
        ctx,
        &scratch.gate,
        &scratch.up,
        swiglu_limit,
        &ffn.shared_w2,
        &mut scratch.out,
        &mut scratch.fp8_act_workspace,
        &mut scratch.fp8_act_scale_workspace,
    )?;
    Ok(&scratch.out)
}

pub fn add_bf16_hidden(
    ctx: &RankGpuContext,
    a: &Bf16HiddenStates,
    b: &Bf16HiddenStates,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        a.hidden_dim == b.hidden_dim,
        "add hidden dim mismatch: a={}, b={}",
        a.hidden_dim,
        b.hidden_dim
    );
    ensure!(
        a.seq_len == b.seq_len,
        "add seq len mismatch: a={}, b={}",
        a.seq_len,
        b.seq_len
    );

    let mut out = Bf16HiddenStates::uninit(ctx, a.hidden_dim, a.seq_len)?;
    {
        let (a_ptr, _a_guard) = a.data.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = b.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::add_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                (out.hidden_dim * out.seq_len) as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn head_rms_norm_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        input.hidden_dim == num_heads * head_dim,
        "head RMSNorm input dim mismatch: expected {}, got {}",
        num_heads * head_dim,
        input.hidden_dim
    );
    let mut out = Bf16HiddenStates::uninit(ctx, input.hidden_dim, input.seq_len)?;
    head_rms_norm_bf16_hidden_into(ctx, input, num_heads, head_dim, eps, &mut out)?;
    Ok(out)
}

pub(crate) fn head_rms_norm_bf16_hidden_into(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
    out: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        input.hidden_dim == num_heads * head_dim,
        "head RMSNorm input dim mismatch: expected {}, got {}",
        num_heads * head_dim,
        input.hidden_dim
    );
    ensure!(
        out.hidden_dim == input.hidden_dim,
        "head RMSNorm output dim mismatch: expected {}, got {}",
        input.hidden_dim,
        out.hidden_dim
    );
    ensure!(
        out.data.len() >= input.hidden_dim * input.seq_len,
        "head RMSNorm output capacity too small: need {}, have {}",
        input.hidden_dim * input.seq_len,
        out.data.len()
    );
    out.seq_len = input.seq_len;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_head_rms_norm_cuda(
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                num_heads as i32,
                head_dim as i32,
                eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}
