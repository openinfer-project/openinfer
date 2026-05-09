use super::*;

pub fn hash_route_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    ffn: &FfnWeights<'_>,
    token_ids: &CudaSlice<u32>,
) -> Result<RoutedExperts> {
    ctx.set_current()?;
    ensure!(
        ffn.gate_weight.tensor.dtype == safetensors::Dtype::BF16,
        "gate weight {} must be BF16, got {:?}",
        ffn.gate_weight.name,
        ffn.gate_weight.tensor.dtype
    );
    ensure!(
        ffn.gate_weight.tensor.shape == [config.n_routed_experts, input.hidden_dim],
        "gate weight {} shape mismatch: expected {:?}, got {:?}",
        ffn.gate_weight.name,
        [config.n_routed_experts, input.hidden_dim],
        ffn.gate_weight.tensor.shape
    );
    let tid2eid = ffn
        .gate_tid2eid
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("hash routing requires gate.tid2eid"))?;
    ensure!(
        tid2eid.tensor.dtype == safetensors::Dtype::I64,
        "gate tid2eid {} must be I64, got {:?}",
        tid2eid.name,
        tid2eid.tensor.dtype
    );
    ensure!(
        tid2eid.tensor.shape == [config.vocab_size, config.n_activated_experts],
        "gate tid2eid {} shape mismatch: expected {:?}, got {:?}",
        tid2eid.name,
        [config.vocab_size, config.n_activated_experts],
        tid2eid.tensor.shape
    );

    let mut weights = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    let mut indices = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (gate_ptr, _gate_guard) = ffn.gate_weight.tensor.data.device_ptr(&ctx.stream);
        let (tid2eid_ptr, _tid2eid_guard) = tid2eid.tensor.data.device_ptr(&ctx.stream);
        let (token_ptr, _token_guard) = token_ids.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = weights.device_ptr_mut(&ctx.stream);
        let (indices_ptr, _indices_guard) = indices.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hash_gate_cuda(
                x_ptr as *const ffi::Half,
                gate_ptr as *const ffi::Half,
                tid2eid_ptr as *const i64,
                token_ptr as *const u32,
                weights_ptr as *mut f32,
                indices_ptr as *mut i32,
                input.seq_len as i32,
                input.hidden_dim as i32,
                config.n_routed_experts as i32,
                config.n_activated_experts as i32,
                config.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(RoutedExperts {
        weights,
        indices,
        topk: config.n_activated_experts,
        seq_len: input.seq_len,
    })
}

pub fn score_route_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    ffn: &FfnWeights<'_>,
) -> Result<RoutedExperts> {
    ctx.set_current()?;
    ensure!(
        ffn.gate_weight.tensor.dtype == safetensors::Dtype::BF16,
        "gate weight {} must be BF16, got {:?}",
        ffn.gate_weight.name,
        ffn.gate_weight.tensor.dtype
    );
    ensure!(
        ffn.gate_weight.tensor.shape == [config.n_routed_experts, input.hidden_dim],
        "gate weight {} shape mismatch: expected {:?}, got {:?}",
        ffn.gate_weight.name,
        [config.n_routed_experts, input.hidden_dim],
        ffn.gate_weight.tensor.shape
    );
    let bias = ffn
        .gate_bias
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("score routing requires gate.bias"))?;
    ensure!(
        bias.tensor.dtype == safetensors::Dtype::F32,
        "gate bias {} must be F32, got {:?}",
        bias.name,
        bias.tensor.dtype
    );
    ensure!(
        bias.tensor.shape == [config.n_routed_experts],
        "gate bias {} shape mismatch: expected {:?}, got {:?}",
        bias.name,
        [config.n_routed_experts],
        bias.tensor.shape
    );

    let mut weights = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    let mut indices = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (gate_ptr, _gate_guard) = ffn.gate_weight.tensor.data.device_ptr(&ctx.stream);
        let (bias_ptr, _bias_guard) = bias.tensor.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = weights.device_ptr_mut(&ctx.stream);
        let (indices_ptr, _indices_guard) = indices.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_score_gate_cuda(
                x_ptr as *const ffi::Half,
                gate_ptr as *const ffi::Half,
                bias_ptr as *const f32,
                weights_ptr as *mut f32,
                indices_ptr as *mut i32,
                input.seq_len as i32,
                input.hidden_dim as i32,
                config.n_routed_experts as i32,
                config.n_activated_experts as i32,
                config.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(RoutedExperts {
        weights,
        indices,
        topk: config.n_activated_experts,
        seq_len: input.seq_len,
    })
}

pub fn accumulate_weighted_expert_output(
    ctx: &RankGpuContext,
    expert_out: &Bf16HiddenStates,
    routed: &RoutedExperts,
    global_expert: usize,
    accum: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        expert_out.hidden_dim == accum.hidden_dim,
        "expert output dim mismatch: expert={}, accum={}",
        expert_out.hidden_dim,
        accum.hidden_dim
    );
    ensure!(
        expert_out.seq_len == accum.seq_len,
        "expert output seq len mismatch: expert={}, accum={}",
        expert_out.seq_len,
        accum.seq_len
    );
    ensure!(
        routed.seq_len == expert_out.seq_len,
        "route seq len mismatch: route={}, expert={}",
        routed.seq_len,
        expert_out.seq_len
    );

    {
        let (expert_ptr, _expert_guard) = expert_out.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = routed.weights.device_ptr(&ctx.stream);
        let (indices_ptr, _indices_guard) = routed.indices.device_ptr(&ctx.stream);
        let (accum_ptr, _accum_guard) = accum.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_weighted_expert_accum_cuda(
                expert_ptr as *const ffi::Half,
                weights_ptr as *const f32,
                indices_ptr as *const i32,
                accum_ptr as *mut ffi::Half,
                expert_out.seq_len as i32,
                expert_out.hidden_dim as i32,
                routed.topk as i32,
                global_expert as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn add_f32_bf16_to_bf16_hidden(
    ctx: &RankGpuContext,
    a: &F32HiddenStates,
    b: &Bf16HiddenStates,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        a.hidden_dim == b.hidden_dim,
        "add f32/bf16 hidden dim mismatch: a={}, b={}",
        a.hidden_dim,
        b.hidden_dim
    );
    ensure!(
        a.seq_len == b.seq_len,
        "add f32/bf16 seq len mismatch: a={}, b={}",
        a.seq_len,
        b.seq_len
    );
    let mut out = Bf16HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    {
        let (a_ptr, _a_guard) = a.data.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = b.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_add_f32_bf16_to_bf16_cuda(
                a_ptr as *const f32,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                (a.hidden_dim * a.seq_len) as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn accumulate_expert_output_f32(
    ctx: &RankGpuContext,
    expert_out: &Bf16HiddenStates,
    accum: &mut F32HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        expert_out.hidden_dim == accum.hidden_dim,
        "expert output dim mismatch: expert={}, accum={}",
        expert_out.hidden_dim,
        accum.hidden_dim
    );
    ensure!(
        expert_out.seq_len == accum.seq_len,
        "expert output seq len mismatch: expert={}, accum={}",
        expert_out.seq_len,
        accum.seq_len
    );
    {
        let (expert_ptr, _expert_guard) = expert_out.data.device_ptr(&ctx.stream);
        let (accum_ptr, _accum_guard) = accum.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_expert_accum_f32_cuda(
                expert_ptr as *const ffi::Half,
                accum_ptr as *mut f32,
                (expert_out.hidden_dim * expert_out.seq_len) as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn routed_local_experts_forward_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &Bf16HiddenStates,
    routed: &RoutedExperts,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;
    let local_experts = config.n_routed_experts / weights.world_size();
    let global_start = weights.rank() * local_experts;
    let route_indices = ctx.stream.clone_dtoh(&routed.indices)?;
    ctx.sync()?;
    let mut active_local = vec![false; local_experts];
    for expert in route_indices {
        if expert < 0 {
            continue;
        }
        let expert = expert as usize;
        if (global_start..global_start + local_experts).contains(&expert) {
            active_local[expert - global_start] = true;
        }
    }
    for local_expert in 0..local_experts {
        if !active_local[local_expert] {
            continue;
        }
        let expert = weights.local_expert(layer, local_expert)?;
        let expert_out =
            local_expert_forward_bf16_hidden(ctx, input, &expert, config.swiglu_limit)?;
        accumulate_weighted_expert_output(
            ctx,
            &expert_out,
            routed,
            global_start + local_expert,
            &mut out,
        )?;
    }
    Ok(out)
}

pub fn routed_local_experts_forward_f32_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &Bf16HiddenStates,
    routed: &RoutedExperts,
) -> Result<F32HiddenStates> {
    ctx.set_current()?;
    let mut out = F32HiddenStates {
        data: ctx.stream.alloc_zeros(input.hidden_dim * input.seq_len)?,
        hidden_dim: input.hidden_dim,
        seq_len: input.seq_len,
    };
    let local_experts = config.n_routed_experts / weights.world_size();
    let global_start = weights.rank() * local_experts;
    let route_indices = ctx.stream.clone_dtoh(&routed.indices)?;
    ctx.sync()?;
    let mut active_local = vec![false; local_experts];
    for expert in route_indices {
        if expert < 0 {
            continue;
        }
        let expert = expert as usize;
        if (global_start..global_start + local_experts).contains(&expert) {
            active_local[expert - global_start] = true;
        }
    }
    for local_expert in 0..local_experts {
        if !active_local[local_expert] {
            continue;
        }
        let global_expert = global_start + local_expert;
        let expert = weights.local_expert(layer, local_expert)?;
        let expert_out = local_expert_forward_weighted_bf16_hidden(
            ctx,
            input,
            &expert,
            routed,
            global_expert,
            config.swiglu_limit,
        )?;
        accumulate_expert_output_f32(ctx, &expert_out, &mut out)?;
    }
    Ok(out)
}

pub fn hash_routed_moe_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &Bf16HiddenStates,
    token_ids: &CudaSlice<u32>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let ffn = weights.ffn(layer)?;
    let routed = hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?;
    let routed_out =
        routed_local_experts_forward_bf16_hidden(ctx, config, weights, layer, input, &routed)?;
    let shared = shared_expert_forward_bf16_hidden(ctx, input, &ffn, config.swiglu_limit)?;
    add_bf16_hidden(ctx, &routed_out, &shared)
}

pub fn moe_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &Bf16HiddenStates,
    token_ids: &CudaSlice<u32>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let ffn = weights.ffn(layer)?;
    let routed = if layer < config.n_hash_layers {
        hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?
    } else {
        score_route_bf16_hidden(ctx, config, input, &ffn)?
    };
    let routed_out =
        routed_local_experts_forward_bf16_hidden(ctx, config, weights, layer, input, &routed)?;
    let shared = shared_expert_forward_bf16_hidden(ctx, input, &ffn, config.swiglu_limit)?;
    add_bf16_hidden(ctx, &routed_out, &shared)
}

pub fn hash_routed_moe_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &Bf16HiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "MoE group must contain at least one rank"
    );

    let mut routed_out = Vec::with_capacity(ranks.len());
    let mut shared_out = Vec::with_capacity(ranks.len());
    for (ctx, weights, _comm, input, token_ids) in ranks {
        let ffn = weights.ffn(layer)?;
        let routed = hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?;
        routed_out.push(routed_local_experts_forward_f32_hidden(
            ctx, config, weights, layer, input, &routed,
        )?);
        shared_out.push(shared_expert_forward_bf16_hidden(
            ctx,
            input,
            &ffn,
            config.swiglu_limit,
        )?);
    }

    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for ((_, _, comm, _, _), hidden) in ranks.iter().zip(routed_out.iter_mut()) {
        if let Err(err) = comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!(
                "NCCL MoE routed all-reduce failed: {err:?}"
            ));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (routed, shared)) in
        ranks.iter().zip(routed_out.iter().zip(shared_out.iter()))
    {
        out.push(add_f32_bf16_to_bf16_hidden(ctx, routed, shared)?);
    }
    Ok(out)
}

pub fn moe_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &Bf16HiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "MoE group must contain at least one rank"
    );

    let mut routed_out = Vec::with_capacity(ranks.len());
    let mut shared_out = Vec::with_capacity(ranks.len());
    for (ctx, weights, _comm, input, token_ids) in ranks {
        let ffn = weights.ffn(layer)?;
        let routed = if layer < config.n_hash_layers {
            hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?
        } else {
            score_route_bf16_hidden(ctx, config, input, &ffn)?
        };
        routed_out.push(routed_local_experts_forward_f32_hidden(
            ctx, config, weights, layer, input, &routed,
        )?);
        shared_out.push(shared_expert_forward_bf16_hidden(
            ctx,
            input,
            &ffn,
            config.swiglu_limit,
        )?);
    }

    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for ((_, _, comm, _, _), hidden) in ranks.iter().zip(routed_out.iter_mut()) {
        if let Err(err) = comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!(
                "NCCL MoE routed all-reduce failed: {err:?}"
            ));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (routed, shared)) in
        ranks.iter().zip(routed_out.iter().zip(shared_out.iter()))
    {
        out.push(add_f32_bf16_to_bf16_hidden(ctx, routed, shared)?);
    }
    Ok(out)
}
