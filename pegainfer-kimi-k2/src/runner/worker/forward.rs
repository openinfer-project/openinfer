use super::{runtime::*, *};

pub(super) fn all_reduce_bf16_rows_in_place(
    values: &mut CudaSlice<half::bf16>,
    rows: usize,
    row_len: usize,
    comm: &Comm,
) -> Result<()> {
    ensure!(
        values.len() >= rows * row_len,
        "Kimi row-wise bf16 all-reduce len {} < rows {} * row_len {}",
        values.len(),
        rows,
        row_len
    );
    for row in 0..rows {
        let start = row * row_len;
        let end = start + row_len;
        let mut view = values.slice_mut(start..end);
        comm.all_reduce_in_place(&mut view, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("Kimi row-wise bf16 all-reduce failed: status={:?}", err.0)
            })?;
    }
    Ok(())
}

pub(super) fn forward_decode_batch_next_token_kernels(
    device_ctx: &DeviceContext,
    decode_aux_ctx: &DeviceContext,
    comm: Option<&Comm>,
    cache: &KimiOneTokenForwardCache,
    expert_kernels: &KimiRankExpertMarlinWeights,
    decode_arena: &mut KimiWorkerDecodeArena,
    active_len: usize,
    local_heads: usize,
    #[cfg(feature = "pplx-ep")] mut pplx: Option<&mut PplxDecodeContext<'_>>,
) -> Result<()> {
    #[cfg(not(feature = "pplx-ep"))]
    let _ = active_len;

    typed_ops::embedding_vocab_shard_into(
        device_ctx,
        &cache.token_embedding,
        &decode_arena.token_ids_d,
        &mut decode_arena.scratch.mla.hidden,
        cache.vocab_start as u32,
    )?;
    maybe_all_reduce_hidden_via_f32_in_place(
        device_ctx,
        &mut decode_arena.scratch.mla.hidden,
        &mut decode_arena.scratch.comm.hidden_allreduce_f32,
        comm,
    )?;

    for layer in &cache.layers {
        forward_mla_decode_layer_into(
            device_ctx,
            &layer.attention,
            decode_arena,
            layer.layer_idx,
            local_heads,
        )
        .with_context(|| format!("Kimi MLA batch decode layer {}", layer.layer_idx))?;
        maybe_all_reduce_hidden_via_f32_in_place(
            device_ctx,
            &mut decode_arena.scratch.mla.projected,
            &mut decode_arena.scratch.comm.hidden_allreduce_f32,
            comm,
        )?;
        typed_ops::fused_add_rms_norm_round_into(
            device_ctx,
            &mut decode_arena.scratch.mla.hidden,
            &decode_arena.scratch.mla.projected,
            &layer.attention.post_attention_norm,
            KIMI_K2_RMS_NORM_EPS,
            &mut decode_arena.scratch.mla.normed,
        )?;
        match &layer.kind {
            KimiLayerForwardKindCache::Dense(dense) => {
                forward_dense_mlp_decode_normed_into(
                    device_ctx,
                    comm,
                    dense,
                    &mut decode_arena.scratch,
                )
                .with_context(|| {
                    format!("Kimi dense batch decode MLP layer {}", layer.layer_idx)
                })?;
            }
            KimiLayerForwardKindCache::Moe(moe) => {
                #[cfg(feature = "pplx-ep")]
                if let Some(pplx_ctx) = pplx.as_mut() {
                    let arena_seq_len = decode_arena.scratch.mla.hidden.seq_len;
                    decode_arena.scratch.set_moe_seq_len(active_len)?;
                    let pplx_result = crate::runner::moe_pplx::forward_moe_layer_decode_pplx_normed(
                        device_ctx,
                        decode_aux_ctx,
                        comm,
                        pplx_ctx.ep,
                        layer.layer_idx,
                        moe,
                        expert_kernels,
                        &mut decode_arena.scratch,
                        pplx_ctx.scratch,
                    );
                    let restore_result = decode_arena.scratch.set_moe_seq_len(arena_seq_len);
                    restore_result?;
                    pplx_result.with_context(|| {
                        format!("Kimi MoE PPLX batch decode layer {}", layer.layer_idx)
                    })?;
                } else {
                    forward_moe_layer_decode_normed_into(
                        device_ctx,
                        decode_aux_ctx,
                        comm,
                        layer.layer_idx,
                        moe,
                        expert_kernels,
                        &mut decode_arena.scratch,
                    )
                    .with_context(|| format!("Kimi MoE batch decode layer {}", layer.layer_idx))?;
                }
                #[cfg(not(feature = "pplx-ep"))]
                {
                    forward_moe_layer_decode_normed_into(
                        device_ctx,
                        decode_aux_ctx,
                        comm,
                        layer.layer_idx,
                        moe,
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

fn forward_mla_decode_layer_into(
    ctx: &DeviceContext,
    attention: &KimiAttentionForwardCache,
    arena: &mut KimiWorkerDecodeArena,
    layer_idx: usize,
    local_heads: usize,
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
    kimi_mla_split_qkv_a_norm(
        ctx,
        &scratch.mla.qkv_a,
        &attention.q_a_norm,
        &attention.kv_a_norm,
        &mut scratch.mla.q_a_normed,
        &mut scratch.mla.compressed_normed,
        &mut scratch.mla.k_rope,
        KIMI_K2_RMS_NORM_EPS,
    )?;
    typed_ops::gemm_dm_typed_to_hs_graphsafe(
        ctx,
        &attention.q_b_proj,
        &scratch.mla.q_a_normed,
        &mut scratch.mla.q_proj,
    )?;
    kimi_mla_rope_split_decode_rt(
        ctx,
        &scratch.mla.q_proj,
        &scratch.mla.k_rope,
        cos_d,
        sin_d,
        positions_d,
        &mut scratch.mla.q_nope,
        &mut scratch.mla.q_pe,
        &mut scratch.mla.append_kpe,
        local_heads,
    )?;
    kimi_mla_absorb_q_nope_rt(
        ctx,
        &attention.kv_b_proj,
        &scratch.mla.q_nope,
        &mut scratch.mla.q_abs_nope,
        local_heads,
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
    kimi_flashinfer_batch_decode_mla_rt(
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
        local_heads,
    )?;
    kimi_mla_v_up_rt(
        ctx,
        &attention.kv_b_proj,
        &scratch.mla.latent,
        &mut scratch.mla.attn_out,
        local_heads,
    )?;
    typed_ops::gemm_dm_hs_to_typed_graphsafe(
        ctx,
        &attention.o_proj,
        &scratch.mla.attn_out,
        &mut scratch.mla.projected,
    )?;
    Ok(())
}

pub(super) fn forward_mla_prompt_len1_batch_layer_into(
    ctx: &DeviceContext,
    comm: Option<&Comm>,
    attention: &KimiAttentionForwardCache,
    arena: &mut KimiWorkerDecodeArena,
    layer_idx: usize,
    local_heads: usize,
) -> Result<()> {
    let KimiWorkerDecodeArena {
        layout,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        batch_indices_d,
        positions_d,
        cos_d,
        sin_d,
        layer_caches,
        scratch,
        ..
    } = arena;
    let layer_cache = layer_caches.get_mut(layer_idx).ok_or_else(|| {
        anyhow::anyhow!("Kimi prompt_len1 prefill layer cache {layer_idx} out of range")
    })?;

    typed_ops::rms_norm_into(
        ctx,
        &scratch.mla.hidden,
        &attention.input_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.mla.normed,
    )?;
    typed_ops::gemm_per_token_into(
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
        &scratch.mla.compressed_kv,
        &attention.kv_a_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.mla.compressed_normed,
    )?;
    kimi_mla_rope_apply_kpe(
        ctx,
        &scratch.mla.k_rope,
        cos_d,
        sin_d,
        positions_d,
        &mut scratch.mla.append_kpe,
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
    typed_ops::gemm_dm_typed_to_hs_per_token(
        ctx,
        &attention.kv_b_proj,
        &scratch.mla.compressed_normed,
        &mut scratch.mla.kv_b,
    )?;
    kimi_mla_extract_prefill_v_rt(
        ctx,
        &scratch.mla.kv_b,
        &mut scratch.mla.attn_out,
        local_heads,
    )?;
    typed_ops::gemm_dm_hs_to_typed_per_token(
        ctx,
        &attention.o_proj,
        &scratch.mla.attn_out,
        &mut scratch.mla.projected,
    )?;
    if let Some(comm) = comm {
        all_reduce_bf16_rows_in_place(
            &mut scratch.mla.projected.data,
            scratch.mla.projected.seq_len,
            KIMI_K2_HIDDEN,
            comm,
        )?;
    }
    typed_ops::add_into(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &mut scratch.mla.normed,
    )?;
    std::mem::swap(&mut scratch.mla.hidden, &mut scratch.mla.normed);
    Ok(())
}

pub(super) fn forward_dense_mlp_batch_into(
    ctx: &DeviceContext,
    comm: Option<&Comm>,
    dense: &KimiDenseForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
    next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
) -> Result<()> {
    let seq_len = hidden.seq_len;
    typed_ops::rms_norm_into(
        ctx,
        hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        normed,
    )?;
    let mut gate_up = HiddenStates::zeros(ctx, dense.gate_up_proj.rows, seq_len)?;
    typed_ops::gemm_dm_typed_to_hs(ctx, &dense.gate_up_proj, normed, &mut gate_up)?;
    let mut activated = HiddenStates::zeros(ctx, dense.down_proj.cols, seq_len)?;
    typed_ops::silu_mul_hs_fused_into(ctx, &gate_up, &mut activated)?;
    let mut mlp_out = GpuTensor::<KIMI_K2_HIDDEN>::zeros(ctx, seq_len)?;
    typed_ops::gemm_dm_hs_to_typed(ctx, &dense.down_proj, &activated, &mut mlp_out)?;
    if let Some(comm) = comm {
        comm.all_reduce_in_place(&mut mlp_out.data, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
            })?;
    }
    typed_ops::add_into(ctx, hidden, &mlp_out, next_hidden)?;
    std::mem::swap(hidden, next_hidden);
    Ok(())
}

pub(super) fn forward_dense_mlp_prefill_scratch_into(
    ctx: &DeviceContext,
    comm: Option<&Comm>,
    dense: &KimiDenseForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    let seq_len = scratch.mla.hidden.seq_len;
    typed_ops::rms_norm_into(
        ctx,
        &scratch.mla.hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.mla.normed,
    )?;
    typed_ops::gemm_dm_typed_to_hs_per_token(
        ctx,
        &dense.gate_up_proj,
        &scratch.mla.normed,
        &mut scratch.dense_mlp.gate_up,
    )?;
    typed_ops::silu_mul_hs_fused_into(
        ctx,
        &scratch.dense_mlp.gate_up,
        &mut scratch.dense_mlp.activated,
    )?;
    typed_ops::gemm_dm_hs_to_typed_per_token(
        ctx,
        &dense.down_proj,
        &scratch.dense_mlp.activated,
        &mut scratch.mla.projected,
    )?;
    if let Some(comm) = comm {
        all_reduce_bf16_rows_in_place(
            &mut scratch.mla.projected.data,
            seq_len,
            KIMI_K2_HIDDEN,
            comm,
        )?;
    }
    typed_ops::add_into(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &mut scratch.mla.normed,
    )?;
    std::mem::swap(&mut scratch.mla.hidden, &mut scratch.mla.normed);
    Ok(())
}

fn forward_dense_mlp_decode_normed_into(
    ctx: &DeviceContext,
    comm: Option<&Comm>,
    dense: &KimiDenseForwardCache,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    typed_ops::gemm_dm_typed_to_hs_graphsafe(
        ctx,
        &dense.gate_up_proj,
        &scratch.mla.normed,
        &mut scratch.dense_mlp.gate_up,
    )?;
    typed_ops::silu_mul_hs_fused_into(
        ctx,
        &scratch.dense_mlp.gate_up,
        &mut scratch.dense_mlp.activated,
    )?;
    typed_ops::gemm_dm_hs_to_typed_graphsafe(
        ctx,
        &dense.down_proj,
        &scratch.dense_mlp.activated,
        &mut scratch.mla.projected,
    )?;
    maybe_all_reduce_hidden_via_f32_in_place(
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
    comm: Option<&Comm>,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    expert_kernels: &KimiRankExpertMarlinWeights,
    hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
    next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
) -> Result<()> {
    let seq_len = hidden.seq_len;
    typed_ops::rms_norm_into(
        ctx,
        hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        normed,
    )?;
    let mut shared_gate_up = HiddenStates::zeros(ctx, moe.shared_gate_up_proj.rows, seq_len)?;
    typed_ops::gemm_dm_typed_to_hs(ctx, &moe.shared_gate_up_proj, normed, &mut shared_gate_up)?;
    let mut shared_activated = HiddenStates::zeros(ctx, moe.shared_down_proj.cols, seq_len)?;
    typed_ops::silu_mul_hs_fused_into(ctx, &shared_gate_up, &mut shared_activated)?;
    let mut shared_out = GpuTensor::<KIMI_K2_HIDDEN>::zeros(ctx, seq_len)?;
    typed_ops::gemm_dm_hs_to_typed(
        ctx,
        &moe.shared_down_proj,
        &shared_activated,
        &mut shared_out,
    )?;
    if let Some(comm) = comm {
        comm.all_reduce_in_place(&mut shared_out.data, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
            })?;
    }

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
    let nccl_comm = comm.ok_or_else(|| {
        anyhow::anyhow!("NCCL MoE batch routed path requires TP comm (use PPLX for TP1)")
    })?;
    all_reduce_f32_in_place(&mut routed_out_f32, nccl_comm)?;
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

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_moe_layer_prefill_scratch_into(
    ctx: &DeviceContext,
    comm: Option<&Comm>,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    expert_kernels: &KimiRankExpertMarlinWeights,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    let seq_len = scratch.mla.hidden.seq_len;
    typed_ops::rms_norm_into(
        ctx,
        &scratch.mla.hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.mla.normed,
    )?;
    typed_ops::gemm_dm_typed_to_hs_per_token(
        ctx,
        &moe.shared_gate_up_proj,
        &scratch.mla.normed,
        &mut scratch.shared_expert.gate_up,
    )?;
    typed_ops::silu_mul_hs_fused_into(
        ctx,
        &scratch.shared_expert.gate_up,
        &mut scratch.shared_expert.activated,
    )?;
    typed_ops::gemm_dm_hs_to_typed_per_token(
        ctx,
        &moe.shared_down_proj,
        &scratch.shared_expert.activated,
        &mut scratch.mla.projected,
    )?;
    if let Some(comm) = comm {
        all_reduce_bf16_rows_in_place(
            &mut scratch.mla.projected.data,
            seq_len,
            KIMI_K2_HIDDEN,
            comm,
        )?;
    }

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
        let router_batch = KimiRouterBatch {
            batch_size: seq_len,
            active_tokens: seq_len,
            padded_tokens: seq_len,
        };
        kimi_router_noaux_tc_per_token_launch(
            ctx,
            KimiRouterConfig::kimi_k2(),
            router_batch,
            &scratch.mla.normed,
            &moe.router.gate_weight,
            &moe.router.e_score_correction_bias,
            &mut router_scratch,
            &mut router_output,
        )?;
    }

    let routing = kimi_moe_marlin_align_block_size(
        ctx,
        &mut scratch.prompt_len1_moe.route_workspace,
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

    scratch.marlin.w13_out.seq_len = routing.route_elems;
    scratch.marlin.activated.seq_len = routing.route_elems;
    scratch.marlin.expert_output.seq_len = routing.route_elems;
    {
        let mut active_w13 = scratch
            .marlin
            .w13_out
            .data
            .slice_mut(0..routing.route_elems * MARLIN_W13_OUT_DIM);
        ctx.stream
            .memset_zeros(&mut active_w13)
            .with_context(|| format!("Kimi MoE layer {layer_idx} zero prompt w13 scratch"))?;
    }
    kimi_marlin_wna16_w13_gemm(
        ctx,
        &mut scratch.prompt_len1_moe.marlin_workspace,
        &routing,
        &scratch.mla.normed,
        &layer_weights.w13,
        &scratch.router.router_topk_weight.data,
        &mut scratch.marlin.w13_out,
    )?;
    kimi_marlin_w13_swiglu(ctx, &scratch.marlin.w13_out, &mut scratch.marlin.activated)?;
    {
        let mut active_expert = scratch
            .marlin
            .expert_output
            .data
            .slice_mut(0..routing.route_elems * KIMI_K2_HIDDEN);
        ctx.stream
            .memset_zeros(&mut active_expert)
            .with_context(|| format!("Kimi MoE layer {layer_idx} zero prompt expert scratch"))?;
    }
    kimi_marlin_wna16_w2_gemm(
        ctx,
        &mut scratch.prompt_len1_moe.marlin_workspace,
        &routing,
        &scratch.marlin.activated,
        &layer_weights.w2_down,
        &scratch.router.router_topk_weight.data,
        &mut scratch.marlin.expert_output,
    )?;

    kimi_marlin_sum_topk_rows_f32(
        ctx,
        &scratch.marlin.expert_output,
        seq_len,
        &mut scratch.comm.routed_out_f32,
    )?;
    let nccl_comm = comm.ok_or_else(|| {
        anyhow::anyhow!("NCCL MoE batch routed path requires TP comm (use PPLX for TP1)")
    })?;
    all_reduce_f32_rows_in_place(
        &mut scratch.comm.routed_out_f32,
        seq_len,
        KIMI_K2_HIDDEN,
        nccl_comm,
    )?;
    scale_f32_in_place(
        ctx,
        &mut scratch.comm.routed_out_f32,
        seq_len * KIMI_K2_HIDDEN,
        KIMI_K2_ROUTER_SCALE,
    )?;
    typed_ops::add_into(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &mut scratch.mla.normed,
    )?;
    kimi_add_f32_bf16_to_bf16(
        ctx,
        &scratch.comm.routed_out_f32,
        &scratch.mla.normed,
        &mut scratch.mla.hidden,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_moe_layer_decode_normed_into(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    comm: Option<&Comm>,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    expert_kernels: &KimiRankExpertMarlinWeights,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    let norm_ready = ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE layer {layer_idx} record fused norm_ready"))?;
    aux_ctx
        .stream
        .wait(&norm_ready)
        .with_context(|| format!("Kimi MoE layer {layer_idx} aux wait fused norm_ready"))?;
    forward_moe_layer_decode_normed_after_event_into(
        ctx,
        aux_ctx,
        comm,
        layer_idx,
        moe,
        expert_kernels,
        scratch,
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_moe_layer_decode_normed_after_event_into(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    comm: Option<&Comm>,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    expert_kernels: &KimiRankExpertMarlinWeights,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    let seq_len = scratch.mla.hidden.seq_len;
    typed_ops::gemm_dm_typed_to_hs_graphsafe(
        ctx,
        &moe.shared_gate_up_proj,
        &scratch.mla.normed,
        &mut scratch.shared_expert.gate_up,
    )?;
    typed_ops::silu_mul_hs_fused_into(
        ctx,
        &scratch.shared_expert.gate_up,
        &mut scratch.shared_expert.activated,
    )?;
    typed_ops::gemm_dm_hs_to_typed_graphsafe(
        ctx,
        &moe.shared_down_proj,
        &scratch.shared_expert.activated,
        &mut scratch.mla.projected,
    )?;
    maybe_all_reduce_hidden_via_f32_in_place(
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
    let nccl_comm = comm.ok_or_else(|| {
        anyhow::anyhow!("NCCL MoE routed path requires TP comm (use PPLX for TP1)")
    })?;
    reduce_scatter_f32_hidden_into(
        &scratch.comm.routed_reduce_scatter_send_f32,
        seq_len * KIMI_K2_EP_WORLD,
        KIMI_K2_HIDDEN,
        &mut scratch.comm.routed_out_f32,
        seq_len,
        KIMI_K2_EP_WORLD,
        nccl_comm,
    )?;

    kimi_residual_add_scaled_f32(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &scratch.comm.routed_out_f32,
        KIMI_K2_ROUTER_SCALE,
        &mut scratch.mla.normed,
    )?;
    std::mem::swap(&mut scratch.mla.hidden, &mut scratch.mla.normed);
    Ok(())
}
