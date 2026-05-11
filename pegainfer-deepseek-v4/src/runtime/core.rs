use super::*;

pub fn hc_expand_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    hc: usize,
) -> Result<HcHiddenStates> {
    ctx.set_current()?;
    ensure!(hc > 0, "HC multiplier must be positive");
    let mut out = HcHiddenStates::zeros(ctx, input.hidden_dim, input.seq_len, hc)?;
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
    Ok(out)
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

    let mut mixes: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * input.hc)?;
    let mut pre: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * input.hc)?;
    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;

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

    let mut out = ctx.stream.alloc_zeros(vocab_size)?;
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

    let mut mixes: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * mix_hc)?;
    let mut post: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * input.hc)?;
    let mut comb: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(input.seq_len * input.hc * input.hc)?;
    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;

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
        let mut pre: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * input.hc)?;
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
        HcHiddenStates::zeros(ctx, branch_out.hidden_dim, branch_out.seq_len, residual.hc)?;
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

pub fn embedding_rank_local(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    token_ids: &CudaSlice<u32>,
    seq_len: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let embed = weights.embed()?;
    ensure!(
        embed.tensor.shape == [config.vocab_size / 8, config.dim],
        "unexpected embed shape {:?}",
        embed.tensor.shape
    );
    let mut out = Bf16HiddenStates::zeros(ctx, config.dim, seq_len)?;

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

    Ok(out)
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

    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;
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
    Ok(out)
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

    let mut out = Bf16HiddenStates::zeros(ctx, out_dim, input.seq_len)?;
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
    Ok(out)
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

    let mut out = Bf16HiddenStates::zeros(ctx, out_dim, input.seq_len)?;
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

    let mut out = Bf16HiddenStates::zeros(ctx, out_dim, input.seq_len)?;
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
    Ok(out)
}

pub fn swiglu_clamp_bf16_hidden(
    ctx: &RankGpuContext,
    gate: &Bf16HiddenStates,
    up: &Bf16HiddenStates,
    limit: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        gate.hidden_dim == up.hidden_dim,
        "SwiGLU hidden dim mismatch: gate={}, up={}",
        gate.hidden_dim,
        up.hidden_dim
    );
    ensure!(
        gate.seq_len == up.seq_len,
        "SwiGLU seq len mismatch: gate={}, up={}",
        gate.seq_len,
        up.seq_len
    );

    let mut out = Bf16HiddenStates::zeros(ctx, gate.hidden_dim, gate.seq_len)?;
    {
        let (gate_ptr, _gate_guard) = gate.data.device_ptr(&ctx.stream);
        let (up_ptr, _up_guard) = up.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_swiglu_clamp_cuda(
                gate_ptr as *const ffi::Half,
                up_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                (gate.hidden_dim * gate.seq_len) as i32,
                limit,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn local_expert_forward_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    expert: &ExpertWeights<'_>,
    swiglu_limit: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let gate = fp4_linear_bf16_hidden(ctx, input, &expert.w1)?;
    let up = fp4_linear_bf16_hidden(ctx, input, &expert.w3)?;
    let activated = swiglu_clamp_bf16_hidden(ctx, &gate, &up, swiglu_limit)?;
    fp4_linear_bf16_hidden(ctx, &activated, &expert.w2)
}

pub fn shared_expert_forward_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    ffn: &FfnWeights<'_>,
    swiglu_limit: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let gate = fp8_linear_bf16_hidden(ctx, input, &ffn.shared_w1)?;
    let up = fp8_linear_bf16_hidden(ctx, input, &ffn.shared_w3)?;
    let activated = swiglu_clamp_bf16_hidden(ctx, &gate, &up, swiglu_limit)?;
    fp8_linear_bf16_hidden(ctx, &activated, &ffn.shared_w2)
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

    let mut out = Bf16HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
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
    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;
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
    Ok(out)
}
