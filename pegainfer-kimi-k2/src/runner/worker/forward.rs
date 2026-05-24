use super::{runtime::*, *};

pub(super) fn forward_decode_batch_next_token_kernels(
    device_ctx: &DeviceContext,
    decode_aux_ctx: &DeviceContext,
    comm: &Comm,
    cache: &KimiOneTokenForwardCache,
    expert_kernels: &KimiRankExpertMarlinWeights,
    decode_arena: &mut KimiWorkerDecodeArena,
    #[cfg(feature = "pplx-ep")] mut pplx: Option<&mut PplxDecodeContext<'_>>,
) -> Result<()> {
    typed_ops::embedding_vocab_shard_into(
        device_ctx,
        &cache.token_embedding,
        &decode_arena.token_ids_d,
        &mut decode_arena.scratch.mla.hidden,
        cache.vocab_start as u32,
    )?;
    all_reduce_hidden_via_f32_in_place(
        device_ctx,
        &mut decode_arena.scratch.mla.hidden,
        &mut decode_arena.scratch.comm.hidden_allreduce_f32,
        comm,
    )?;

    for layer in &cache.layers {
        forward_mla_decode_layer_into(device_ctx, &layer.attention, decode_arena, layer.layer_idx)
            .with_context(|| format!("Kimi MLA batch decode layer {}", layer.layer_idx))?;
        all_reduce_hidden_via_f32_in_place(
            device_ctx,
            &mut decode_arena.scratch.mla.projected,
            &mut decode_arena.scratch.comm.hidden_allreduce_f32,
            comm,
        )?;
        typed_ops::add_into(
            device_ctx,
            &decode_arena.scratch.mla.hidden,
            &decode_arena.scratch.mla.projected,
            &mut decode_arena.scratch.mla.normed,
        )?;
        std::mem::swap(
            &mut decode_arena.scratch.mla.hidden,
            &mut decode_arena.scratch.mla.normed,
        );
        match &layer.kind {
            KimiLayerForwardKindCache::Dense(dense) => {
                forward_dense_mlp_decode_into(
                    device_ctx,
                    comm,
                    dense,
                    &layer.attention.post_attention_norm,
                    &mut decode_arena.scratch,
                )
                .with_context(|| {
                    format!("Kimi dense batch decode MLP layer {}", layer.layer_idx)
                })?;
            }
            KimiLayerForwardKindCache::Moe(moe) => {
                #[cfg(feature = "pplx-ep")]
                if let Some(pplx_ctx) = pplx.as_mut() {
                    super::moe_pplx::forward_moe_layer_decode_pplx(
                        device_ctx,
                        decode_aux_ctx,
                        comm,
                        pplx_ctx.ep,
                        layer.layer_idx,
                        moe,
                        &layer.attention.post_attention_norm,
                        expert_kernels,
                        &mut decode_arena.scratch,
                        pplx_ctx.scratch,
                    )
                    .with_context(|| {
                        format!("Kimi MoE PPLX batch decode layer {}", layer.layer_idx)
                    })?;
                } else {
                    forward_moe_layer_decode_into(
                        device_ctx,
                        decode_aux_ctx,
                        comm,
                        layer.layer_idx,
                        moe,
                        &layer.attention.post_attention_norm,
                        expert_kernels,
                        &mut decode_arena.scratch,
                    )
                    .with_context(|| format!("Kimi MoE batch decode layer {}", layer.layer_idx))?;
                }
                #[cfg(not(feature = "pplx-ep"))]
                {
                    forward_moe_layer_decode_into(
                        device_ctx,
                        decode_aux_ctx,
                        comm,
                        layer.layer_idx,
                        moe,
                        &layer.attention.post_attention_norm,
                        expert_kernels,
                        &mut decode_arena.scratch,
                    )
                    .with_context(|| format!("Kimi MoE batch decode layer {}", layer.layer_idx))?;
                }
            }
        }
    }

    let active_len = decode_arena.scratch.mla.hidden.seq_len;
    typed_ops::rms_norm_into(
        device_ctx,
        &decode_arena.scratch.mla.hidden,
        &cache.final_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut decode_arena.scratch.mla.normed,
    )?;
    typed_ops::gemm_runtime_out_graphsafe_into(
        device_ctx,
        &cache.lm_head,
        &decode_arena.scratch.mla.normed,
        &mut decode_arena.logits,
    )?;
    launch_local_top1_batch(
        device_ctx,
        &decode_arena.logits,
        active_len,
        &mut decode_arena.scratch.sampling.top1_value_scratch,
        &mut decode_arena.scratch.sampling.top1_out,
    )
}

pub(super) fn forward_mla_decode_layer_into(
    ctx: &DeviceContext,
    attention: &KimiAttentionForwardCache,
    arena: &mut KimiWorkerDecodeArena,
    layer_idx: usize,
) -> Result<()> {
    let KimiWorkerDecodeArena {
        layout,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        batch_indices_d,
        positions_d,
        request_indices_d,
        kv_tile_indices_d,
        kv_chunk_size_d,
        cos_d,
        sin_d,
        layer_caches,
        scratch,
        ..
    } = arena;
    let layer_cache = layer_caches
        .get_mut(layer_idx)
        .ok_or_else(|| anyhow::anyhow!("Kimi decode layer cache {layer_idx} out of range"))?;

    typed_ops::rms_norm_into(
        ctx,
        &scratch.mla.hidden,
        &attention.input_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.mla.normed,
    )?;
    typed_ops::gemm_graphsafe_into(
        ctx,
        &attention.fused_qkv_a_proj,
        &scratch.mla.normed,
        &mut scratch.mla.qkv_a,
    )?;
    kimi_mla_split_qkv_a(
        ctx,
        &scratch.mla.qkv_a,
        &mut scratch.mla.q_a,
        &mut scratch.mla.compressed_kv,
        &mut scratch.mla.k_rope,
    )?;
    typed_ops::rms_norm_into(
        ctx,
        &scratch.mla.q_a,
        &attention.q_a_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.mla.q_a_normed,
    )?;
    typed_ops::gemm_graphsafe_into(
        ctx,
        &attention.q_b_proj,
        &scratch.mla.q_a_normed,
        &mut scratch.mla.q_proj,
    )?;
    typed_ops::rms_norm_into(
        ctx,
        &scratch.mla.compressed_kv,
        &attention.kv_a_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.mla.compressed_normed,
    )?;
    kimi_mla_rope_split_decode(
        ctx,
        &scratch.mla.q_proj,
        &scratch.mla.k_rope,
        cos_d,
        sin_d,
        positions_d,
        &mut scratch.mla.q_nope,
        &mut scratch.mla.q_pe,
        &mut scratch.mla.append_kpe,
    )?;
    kimi_mla_absorb_q_nope(
        ctx,
        &attention.kv_b_proj,
        &scratch.mla.q_nope,
        &mut scratch.mla.q_abs_nope,
    )?;
    kimi_mla_paged_kv_append(
        ctx,
        &mut layer_cache.ckv_cache,
        &mut layer_cache.kpe_cache,
        *layout,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        &scratch.mla.compressed_normed,
        &scratch.mla.append_kpe,
        batch_indices_d,
        positions_d,
    )?;
    kimi_flashinfer_batch_decode_mla(
        ctx,
        &scratch.mla.q_abs_nope,
        &scratch.mla.q_pe,
        &mut scratch.mla.latent,
        &layer_cache.ckv_cache,
        &layer_cache.kpe_cache,
        *layout,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        request_indices_d,
        kv_tile_indices_d,
        kv_chunk_size_d,
        kimi_mla_softmax_scale(),
    )?;
    kimi_mla_v_up(
        ctx,
        &attention.kv_b_proj,
        &scratch.mla.latent,
        &mut scratch.mla.attn_out,
    )?;
    typed_ops::gemm_graphsafe_into(
        ctx,
        &attention.o_proj,
        &scratch.mla.attn_out,
        &mut scratch.mla.projected,
    )?;
    Ok(())
}

pub(super) fn forward_dense_mlp_batch_into(
    ctx: &DeviceContext,
    comm: &Comm,
    dense: &KimiDenseForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
    next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
) -> Result<()> {
    let seq_len = hidden.seq_len;
    pegainfer_kernels::typed_pipeline! {
        ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS, seq_len = seq_len, gemm = prefill;
        tensor gate_up: DENSE_GATE_UP_DIM;
        tensor activated: DENSE_ACTIVATED_DIM;
        tensor mlp_out: KIMI_K2_HIDDEN;

        rms_norm(hidden => normed, post_attention_norm);
        gemm(normed => &mut gate_up, dense.gate_up_proj);
        silu_mul<DENSE_ACTIVATED_DIM>(&gate_up => &mut activated);
        gemm(&activated => &mut mlp_out, dense.down_proj);
    }
    comm.all_reduce_in_place(&mut mlp_out.data, &ReduceOp::Sum)
        .map_err(|err| {
            anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
        })?;
    pegainfer_kernels::typed_pipeline! {
        ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS;
        add(hidden, &mlp_out => next_hidden);
        swap(hidden, next_hidden);
    }
    Ok(())
}

pub(super) fn forward_dense_mlp_decode_into(
    ctx: &DeviceContext,
    comm: &Comm,
    dense: &KimiDenseForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    pegainfer_kernels::typed_pipeline! {
        ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS;
        rms_norm (&scratch.mla.hidden           => &mut scratch.mla.normed,         post_attention_norm);
        gemm     (&scratch.mla.normed           => &mut scratch.dense_mlp.gate_up,  dense.gate_up_proj);
        silu_mul<DENSE_ACTIVATED_DIM> (&scratch.dense_mlp.gate_up => &mut scratch.dense_mlp.activated);
        gemm     (&scratch.dense_mlp.activated  => &mut scratch.mla.projected,      dense.down_proj);
    }
    all_reduce_hidden_via_f32_in_place(
        ctx,
        &mut scratch.mla.projected,
        &mut scratch.comm.hidden_allreduce_f32,
        comm,
    )?;
    typed_ops::add_into(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &mut scratch.mla.normed,
    )?;
    std::mem::swap(&mut scratch.mla.hidden, &mut scratch.mla.normed);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_moe_layer_batch_into(
    ctx: &DeviceContext,
    comm: &Comm,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    expert_kernels: &KimiRankExpertMarlinWeights,
    hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
    next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
) -> Result<()> {
    let seq_len = hidden.seq_len;
    pegainfer_kernels::typed_pipeline! {
        ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS, seq_len = seq_len, gemm = prefill;
        tensor shared_gate_up: SHARED_GATE_UP_DIM;
        tensor shared_activated: SHARED_ACTIVATED_DIM;
        tensor shared_out: KIMI_K2_HIDDEN;

        rms_norm(hidden => normed, post_attention_norm);
        gemm(normed => &mut shared_gate_up, moe.shared_gate_up_proj);
        silu_mul<SHARED_ACTIVATED_DIM>(&shared_gate_up => &mut shared_activated);
        gemm(&shared_activated => &mut shared_out, moe.shared_down_proj);
    }
    comm.all_reduce_in_place(&mut shared_out.data, &ReduceOp::Sum)
        .map_err(|err| {
            anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
        })?;

    let mut router_logits = ctx.stream.alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_scores = ctx.stream.alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_choice_scores = ctx.stream.alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_topk_weight = ctx.stream.alloc_zeros(seq_len * KIMI_K2_TOPK)?;
    let mut router_topk_idx = ctx.stream.alloc_zeros(seq_len * KIMI_K2_TOPK)?;
    {
        let mut scratch = KimiRouterScratch {
            logits: &mut router_logits,
            scores: &mut router_scores,
            choice_scores: &mut router_choice_scores,
        };
        let mut output = KimiRouterOutput {
            topk_weight: &mut router_topk_weight,
            topk_idx: &mut router_topk_idx,
        };
        kimi_router_noaux_tc_launch(
            ctx,
            KimiRouterConfig::kimi_k2(),
            KimiRouterBatch {
                batch_size: seq_len,
                active_tokens: seq_len,
                padded_tokens: seq_len,
            },
            normed,
            &moe.router.gate_weight,
            &moe.router.e_score_correction_bias,
            &mut scratch,
            &mut output,
        )?;
    }

    let marlin_block_size = kimi_marlin_block_size(seq_len);
    let mut route_workspace = KimiMarlinRouteWorkspace::new(ctx, seq_len, marlin_block_size)?;
    let routing = kimi_moe_marlin_align_block_size(
        ctx,
        &mut route_workspace,
        &router_topk_idx,
        seq_len,
        seq_len,
        expert_kernels.local_expert_range.start,
    )?;
    let layer_weights = expert_kernels
        .layers
        .iter()
        .find(|layer| layer.layer_idx == layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!("Kimi rank expert Marlin package missing layer {layer_idx}")
        })?
        .as_marlin_weights();

    let mut marlin_workspace = KimiMarlinWna16Workspace::new(
        ctx,
        routing.max_m_blocks,
        KIMI_K2_HIDDEN,
        marlin_block_size,
    )?;
    let mut w13_out = GpuTensor::<MARLIN_W13_OUT_DIM>::zeros(ctx, routing.route_elems)?;
    kimi_marlin_wna16_w13_gemm(
        ctx,
        &mut marlin_workspace,
        &routing,
        normed,
        &layer_weights.w13,
        &router_topk_weight,
        &mut w13_out,
    )?;
    let mut activated = GpuTensor::<KIMI_K2_EXPERT_INTERMEDIATE>::zeros(ctx, routing.route_elems)?;
    kimi_marlin_w13_swiglu(ctx, &w13_out, &mut activated)?;
    let mut expert_output = GpuTensor::<KIMI_K2_HIDDEN>::zeros(ctx, routing.route_elems)?;
    kimi_marlin_wna16_w2_gemm(
        ctx,
        &mut marlin_workspace,
        &routing,
        &activated,
        &layer_weights.w2_down,
        &router_topk_weight,
        &mut expert_output,
    )?;

    let mut routed_out_f32 = ctx.stream.alloc_zeros(seq_len * KIMI_K2_HIDDEN)?;
    kimi_marlin_sum_topk_rows_f32(ctx, &expert_output, seq_len, &mut routed_out_f32)?;
    all_reduce_f32_in_place(&mut routed_out_f32, comm)?;
    scale_f32_in_place(
        ctx,
        &mut routed_out_f32,
        seq_len * KIMI_K2_HIDDEN,
        KIMI_K2_ROUTER_SCALE,
    )?;
    pegainfer_kernels::typed_pipeline! {
        ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS;
        add(hidden, &shared_out => next_hidden);
    }
    kimi_add_f32_bf16_to_bf16(ctx, &routed_out_f32, next_hidden, hidden)?;
    Ok(())
}

pub(super) fn forward_moe_layer_decode_into(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    comm: &Comm,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    expert_kernels: &KimiRankExpertMarlinWeights,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    let seq_len = scratch.mla.hidden.seq_len;

    // Shared expert path (main stream)
    typed_ops::rms_norm_into(
        ctx,
        &scratch.mla.hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.mla.normed,
    )?;
    let norm_ready = ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE layer {layer_idx} record norm_ready"))?;
    aux_ctx
        .stream
        .wait(&norm_ready)
        .with_context(|| format!("Kimi MoE layer {layer_idx} aux wait norm_ready"))?;

    pegainfer_kernels::typed_pipeline! {
        ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS;
        gemm     (&scratch.mla.normed           => &mut scratch.shared_expert.gate_up,  moe.shared_gate_up_proj);
        silu_mul<SHARED_ACTIVATED_DIM> (&scratch.shared_expert.gate_up => &mut scratch.shared_expert.activated);
        gemm     (&scratch.shared_expert.activated => &mut scratch.mla.projected,       moe.shared_down_proj);
    }
    all_reduce_hidden_via_f32_in_place(
        ctx,
        &mut scratch.mla.projected,
        &mut scratch.comm.hidden_allreduce_f32,
        comm,
    )?;

    // Router + routed experts (aux stream)
    {
        let mut router_scratch = KimiRouterScratch {
            logits: &mut scratch.router.router_logits.data,
            scores: &mut scratch.router.router_scores.data,
            choice_scores: &mut scratch.router.router_choice_scores.data,
        };
        let mut router_output = KimiRouterOutput {
            topk_weight: &mut scratch.router.router_topk_weight.data,
            topk_idx: &mut scratch.router.router_topk_idx.data,
        };
        kimi_router_noaux_tc_launch(
            aux_ctx,
            KimiRouterConfig::kimi_k2(),
            KimiRouterBatch {
                batch_size: seq_len,
                active_tokens: seq_len,
                padded_tokens: seq_len,
            },
            &scratch.mla.normed,
            &moe.router.gate_weight,
            &moe.router.e_score_correction_bias,
            &mut router_scratch,
            &mut router_output,
        )?;
    }

    let routing = kimi_moe_marlin_align_block_size(
        aux_ctx,
        &mut scratch.marlin_route_workspace,
        &scratch.router.router_topk_idx.data,
        seq_len,
        seq_len,
        expert_kernels.local_expert_range.start,
    )?;
    let layer_weights = expert_kernels
        .layers
        .iter()
        .find(|layer| layer.layer_idx == layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!("Kimi rank expert Marlin package missing layer {layer_idx}")
        })?
        .as_marlin_weights();

    aux_ctx
        .stream
        .memset_zeros(&mut scratch.marlin.w13_out.data)?;
    kimi_marlin_wna16_w13_gemm(
        aux_ctx,
        &mut scratch.marlin_workspace,
        &routing,
        &scratch.mla.normed,
        &layer_weights.w13,
        &scratch.router.router_topk_weight.data,
        &mut scratch.marlin.w13_out,
    )?;
    kimi_marlin_w13_swiglu(
        aux_ctx,
        &scratch.marlin.w13_out,
        &mut scratch.marlin.activated,
    )?;
    aux_ctx
        .stream
        .memset_zeros(&mut scratch.marlin.expert_output.data)?;
    kimi_marlin_wna16_w2_gemm(
        aux_ctx,
        &mut scratch.marlin_workspace,
        &routing,
        &scratch.marlin.activated,
        &layer_weights.w2_down,
        &scratch.router.router_topk_weight.data,
        &mut scratch.marlin.expert_output,
    )?;
    kimi_marlin_sum_topk_rows_f32(
        aux_ctx,
        &scratch.marlin.expert_output,
        seq_len,
        &mut scratch.comm.routed_out_f32,
    )?;
    repeat_f32_for_reduce_scatter_into(
        aux_ctx,
        &scratch.comm.routed_out_f32,
        &mut scratch.comm.routed_reduce_scatter_send_f32,
        seq_len * KIMI_K2_HIDDEN,
        KIMI_K2_EP_WORLD,
    )?;

    let routed_local_done = aux_ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE layer {layer_idx} record routed_local_done"))?;
    ctx.stream
        .wait(&routed_local_done)
        .with_context(|| format!("Kimi MoE layer {layer_idx} main wait routed_local_done"))?;
    reduce_scatter_f32_hidden_into(
        &scratch.comm.routed_reduce_scatter_send_f32,
        seq_len * KIMI_K2_EP_WORLD,
        KIMI_K2_HIDDEN,
        &mut scratch.comm.routed_out_f32,
        seq_len,
        KIMI_K2_EP_WORLD,
        comm,
    )?;

    typed_ops::add_into(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &mut scratch.mla.normed,
    )?;
    kimi_scaled_add_f32_bf16_to_bf16(
        ctx,
        &scratch.comm.routed_out_f32,
        KIMI_K2_ROUTER_SCALE,
        &scratch.mla.normed,
        &mut scratch.mla.hidden,
    )?;
    Ok(())
}
