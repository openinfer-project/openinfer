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

pub fn build_moe_fused_route_plan(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    routed: RoutedExperts,
    input_hidden_dim: usize,
) -> Result<MoeFusedRoutePlan> {
    ctx.set_current()?;
    ensure!(
        input_hidden_dim == config.dim,
        "MoE fused route input dim mismatch: expected {}, got {}",
        config.dim,
        input_hidden_dim
    );
    ensure!(
        routed.topk == config.n_activated_experts,
        "MoE fused route topk mismatch: expected {}, got {}",
        config.n_activated_experts,
        routed.topk
    );
    ensure!(
        config.n_routed_experts.is_multiple_of(weights.world_size()),
        "n_routed_experts={} must be divisible by world_size={}",
        config.n_routed_experts,
        weights.world_size()
    );

    let local_experts = config.n_routed_experts / weights.world_size();
    let global_start = weights.rank() * local_experts;
    let num_expanded = routed.seq_len * routed.topk * local_experts;
    let mut pos_to_expert = ctx.stream.alloc_zeros(num_expanded)?;
    let mut pos_to_token = ctx.stream.alloc_zeros(num_expanded)?;
    let mut pos_to_token_topk = ctx.stream.alloc_zeros(num_expanded)?;
    let mut token_topk_to_pos = ctx.stream.alloc_zeros(routed.seq_len * routed.topk)?;
    let mut expert_start = ctx.stream.alloc_zeros(local_experts)?;
    let mut expert_end = ctx.stream.alloc_zeros(local_experts)?;
    let mut num_tokens_per_expert = ctx.stream.alloc_zeros(local_experts)?;
    {
        let (indices_ptr, _indices_guard) = routed.indices.device_ptr(&ctx.stream);
        let (pos_to_expert_ptr, _pos_to_expert_guard) = pos_to_expert.device_ptr_mut(&ctx.stream);
        let (pos_to_token_ptr, _pos_to_token_guard) = pos_to_token.device_ptr_mut(&ctx.stream);
        let (pos_to_token_topk_ptr, _pos_to_token_topk_guard) =
            pos_to_token_topk.device_ptr_mut(&ctx.stream);
        let (token_topk_to_pos_ptr, _token_topk_to_pos_guard) =
            token_topk_to_pos.device_ptr_mut(&ctx.stream);
        let (expert_start_ptr, _expert_start_guard) = expert_start.device_ptr_mut(&ctx.stream);
        let (expert_end_ptr, _expert_end_guard) = expert_end.device_ptr_mut(&ctx.stream);
        let (num_tokens_ptr, _num_tokens_guard) = num_tokens_per_expert.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_moe_local_mapping_cuda(
                indices_ptr as *const i32,
                pos_to_expert_ptr as *mut i32,
                pos_to_token_ptr as *mut i32,
                pos_to_token_topk_ptr as *mut i32,
                token_topk_to_pos_ptr as *mut i32,
                expert_start_ptr as *mut i32,
                expert_end_ptr as *mut i32,
                num_tokens_ptr as *mut i32,
                routed.seq_len as i32,
                routed.topk as i32,
                global_start as i32,
                local_experts as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(MoeFusedRoutePlan {
        routed,
        pos_to_expert,
        pos_to_token,
        pos_to_token_topk,
        token_topk_to_pos,
        expert_start,
        expert_end,
        num_tokens_per_expert,
        local_experts,
        global_start,
        num_expanded,
    })
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

pub fn expand_moe_fused_input(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    plan: &MoeFusedRoutePlan,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        plan.routed.seq_len == input.seq_len,
        "MoE expand route seq len mismatch: route={}, input={}",
        plan.routed.seq_len,
        input.seq_len
    );
    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, plan.num_expanded)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (pos_ptr, _pos_guard) = plan.pos_to_token.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_moe_expand_to_fused_cuda(
                x_ptr as *const ffi::Half,
                pos_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hidden_dim as i32,
                plan.routed.topk as i32,
                plan.local_experts as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn reduce_moe_fused_output_f32(
    ctx: &RankGpuContext,
    expanded: &Bf16HiddenStates,
    plan: &MoeFusedRoutePlan,
    output_hidden_dim: usize,
) -> Result<F32HiddenStates> {
    ctx.set_current()?;
    ensure!(
        expanded.hidden_dim == output_hidden_dim,
        "MoE reduce hidden dim mismatch: expanded={}, output={}",
        expanded.hidden_dim,
        output_hidden_dim
    );
    let mut out = F32HiddenStates {
        data: ctx
            .stream
            .alloc_zeros(output_hidden_dim * plan.routed.seq_len)?,
        hidden_dim: output_hidden_dim,
        seq_len: plan.routed.seq_len,
    };
    {
        let (expanded_ptr, _expanded_guard) = expanded.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = plan.routed.weights.device_ptr(&ctx.stream);
        let (map_ptr, _map_guard) = plan.token_topk_to_pos.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_moe_reduce_fused_f32_cuda(
                expanded_ptr as *const ffi::Half,
                weights_ptr as *const f32,
                map_ptr as *const i32,
                out_ptr as *mut f32,
                plan.routed.seq_len as i32,
                output_hidden_dim as i32,
                plan.routed.topk as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

fn accumulate_weighted_bf16_to_f32_hidden(
    ctx: &RankGpuContext,
    expert_out: &Bf16HiddenStates,
    route_weights: &CudaSlice<f32>,
    route: usize,
    out: &mut F32HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        expert_out.seq_len == 1,
        "weighted MoE accumulate expects seq_len=1, got {}",
        expert_out.seq_len
    );
    ensure!(
        out.seq_len == 1,
        "weighted MoE accumulate output expects seq_len=1, got {}",
        out.seq_len
    );
    ensure!(
        expert_out.hidden_dim == out.hidden_dim,
        "weighted MoE accumulate hidden dim mismatch: expert={}, out={}",
        expert_out.hidden_dim,
        out.hidden_dim
    );
    {
        let (expert_ptr, _expert_guard) = expert_out.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = route_weights.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_moe_accumulate_weighted_bf16_to_f32_cuda(
                expert_ptr as *const ffi::Half,
                weights_ptr as *const f32,
                route as i32,
                out_ptr as *mut f32,
                out.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub(crate) fn decode_routed_moe_rank_local_f32_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &Bf16HiddenStates,
    routed: &RoutedExperts,
) -> Result<F32HiddenStates> {
    ctx.set_current()?;
    ensure!(
        input.seq_len == 1,
        "decode routed MoE expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        routed.seq_len == 1,
        "decode routed MoE route seq_len mismatch: {}",
        routed.seq_len
    );
    ensure!(
        routed.topk == config.n_activated_experts,
        "decode routed MoE topk mismatch: route={}, config={}",
        routed.topk,
        config.n_activated_experts
    );
    ensure!(
        config.n_routed_experts.is_multiple_of(weights.world_size()),
        "n_routed_experts={} must be divisible by world_size={}",
        config.n_routed_experts,
        weights.world_size()
    );
    let local_experts = config.n_routed_experts / weights.world_size();
    let global_start = weights.rank() * local_experts;
    let route_indices = ctx.stream.clone_dtoh(&routed.indices)?;
    ctx.sync()?;

    let mut out = F32HiddenStates {
        data: ctx.stream.alloc_zeros(input.hidden_dim)?,
        hidden_dim: input.hidden_dim,
        seq_len: 1,
    };
    for (route, expert_id) in route_indices.iter().copied().enumerate().take(routed.topk) {
        if expert_id < global_start as i32
            || expert_id >= (global_start + local_experts) as i32
        {
            continue;
        }
        let local_expert = expert_id as usize - global_start;
        let expert = weights.local_expert(layer, local_expert)?;
        let expert_out =
            local_expert_forward_bf16_hidden(ctx, input, &expert, config.swiglu_limit)?;
        accumulate_weighted_bf16_to_f32_hidden(ctx, &expert_out, &routed.weights, route, &mut out)?;
    }
    Ok(out)
}

pub fn local_experts_forward_packed_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    expanded_input: &Bf16HiddenStates,
    plan: &MoeFusedRoutePlan,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let mut expanded_out = Bf16HiddenStates::zeros(ctx, config.dim, plan.num_expanded)?;
    let expert_start = ctx.stream.clone_dtoh(&plan.expert_start)?;
    let expert_end = ctx.stream.clone_dtoh(&plan.expert_end)?;
    ctx.sync()?;
    for local_expert in 0..plan.local_experts {
        let start = expert_start[local_expert].max(0) as usize;
        let end = expert_end[local_expert].max(0) as usize;
        if end <= start {
            continue;
        }
        let token_count = end - start;
        let expert = weights.local_expert(layer, local_expert)?;
        let input_offset = start * expanded_input.hidden_dim;
        let input_len = token_count * expanded_input.hidden_dim;
        let input_view = Bf16HiddenView {
            data: expanded_input
                .data
                .slice(input_offset..input_offset + input_len),
            hidden_dim: expanded_input.hidden_dim,
            seq_len: token_count,
        };
        let gate = fp4_linear_bf16_view(ctx, &input_view, &expert.w1)?;
        let up = fp4_linear_bf16_view(ctx, &input_view, &expert.w3)?;
        let activated = swiglu_clamp_bf16_hidden(ctx, &gate, &up, config.swiglu_limit)?;
        let expert_out = fp4_linear_bf16_hidden(ctx, &activated, &expert.w2)?;
        let output_offset = start * expanded_out.hidden_dim;
        let output_len = token_count * expanded_out.hidden_dim;
        {
            let (src_ptr, _src_guard) = expert_out.data.device_ptr(&ctx.stream);
            let mut output_view = expanded_out
                .data
                .slice_mut(output_offset..output_offset + output_len);
            let (dst_ptr, _dst_guard) = output_view.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                ffi::deepseek_bf16_copy_rows_cuda(
                    src_ptr as *const ffi::Half,
                    dst_ptr as *mut ffi::Half,
                    expanded_out.hidden_dim as i32,
                    token_count as i32,
                    0,
                    0,
                    ctx.stream.cu_stream(),
                )
            };
            result.result()?;
        }
    }
    Ok(expanded_out)
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
    let plan = build_moe_fused_route_plan(ctx, config, weights, routed, input.hidden_dim)?;
    let expanded_input = expand_moe_fused_input(ctx, input, &plan)?;
    let expanded_out = local_experts_forward_packed_bf16_hidden(
        ctx,
        config,
        weights,
        layer,
        &expanded_input,
        &plan,
    )?;
    let routed_out = reduce_moe_fused_output_f32(ctx, &expanded_out, &plan, input.hidden_dim)?;
    let shared = shared_expert_forward_bf16_hidden(ctx, input, &ffn, config.swiglu_limit)?;
    add_f32_bf16_to_bf16_hidden(ctx, &routed_out, &shared)
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
    let plan = build_moe_fused_route_plan(ctx, config, weights, routed, input.hidden_dim)?;
    let expanded_input = expand_moe_fused_input(ctx, input, &plan)?;
    let expanded_out = local_experts_forward_packed_bf16_hidden(
        ctx,
        config,
        weights,
        layer,
        &expanded_input,
        &plan,
    )?;
    let routed_out = reduce_moe_fused_output_f32(ctx, &expanded_out, &plan, input.hidden_dim)?;
    let shared = shared_expert_forward_bf16_hidden(ctx, input, &ffn, config.swiglu_limit)?;
    add_f32_bf16_to_bf16_hidden(ctx, &routed_out, &shared)
}

fn routed_moe_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &Bf16HiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
    hash_only: bool,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "MoE group must contain at least one rank"
    );

    let mut routed_out = Vec::with_capacity(ranks.len());
    let mut shared_out = Vec::with_capacity(ranks.len());
    for (ctx, weights, _comm, input, token_ids) in ranks {
        let ffn = weights.ffn(layer)?;
        let routed = if hash_only || layer < config.n_hash_layers {
            hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?
        } else {
            score_route_bf16_hidden(ctx, config, input, &ffn)?
        };
        let plan = build_moe_fused_route_plan(ctx, config, weights, routed, input.hidden_dim)?;
        let expanded_input = expand_moe_fused_input(ctx, input, &plan)?;
        let expanded_out = local_experts_forward_packed_bf16_hidden(
            ctx,
            config,
            weights,
            layer,
            &expanded_input,
            &plan,
        )?;
        routed_out.push(reduce_moe_fused_output_f32(
            ctx,
            &expanded_out,
            &plan,
            input.hidden_dim,
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
    routed_moe_group_bf16_hidden(ranks, config, layer, true)
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
    routed_moe_group_bf16_hidden(ranks, config, layer, false)
}
