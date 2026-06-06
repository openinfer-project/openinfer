use super::{forward::*, runtime::*, *};
use crate::config::KIMI_K2_VOCAB;

impl KimiRankThreadState {
    pub(super) fn enable_pplx(&mut self, ep_backend: pegainfer_comm::EpBackend) -> Result<()> {
        self.ctx.set_current()?;
        self.ep_backend = Some(ep_backend);
        self.enable_cuda_graph = false;
        Ok(())
    }

    pub(super) fn init_tp_comm(&mut self, id: Id, world_size: usize) -> Result<()> {
        ensure!(
            self.tp_comm.is_none(),
            "Kimi rank {} TP comm already attached",
            self.sliced_load_plan.rank
        );
        self.ctx.set_current()?;
        let rank = self.sliced_load_plan.rank;
        let device_ctx = self.ctx.as_device_context();
        let comm = Comm::from_rank(device_ctx.stream, rank, world_size, id)
            .map_err(|err| anyhow::anyhow!("Kimi rank {rank} NCCL init failed: {err:?}"))?;
        self.tp_comm = Some(OwnedRankComm(comm));
        Ok(())
    }

    pub(super) fn load_sliced_weights(
        &mut self,
        model_path: &Path,
    ) -> Result<KimiRankWeightLoadReport> {
        let started = Instant::now();
        let rank = self.sliced_load_plan.rank;
        debug!("kimi-k2: rank {rank} start rank weight init");
        let load_output = load_rank_sliced_weights_to_gpu(
            &self.ctx,
            model_path,
            &self.sliced_load_plan,
            &self.weight_names,
        )
        .with_context(|| {
            format!(
                "failed to load Kimi rank {} sliced weights to GPU",
                self.sliced_load_plan.rank
            )
        })?;
        let weights = load_output.weights;
        let expert_kernel_weights = load_output.expert_kernel_weights;
        let tensor_count = load_output.loaded_tensor_count;
        let total_bytes = load_output.loaded_total_bytes;
        debug!("kimi-k2: rank {rank} start one-token forward cache build");
        let cache_started = Instant::now();
        let one_token_cache =
            KimiOneTokenForwardCache::from_gpu_weights(&self.ctx, &weights, &self.weight_names)
                .with_context(|| {
                    format!(
                        "failed to build Kimi rank {} one-token forward cache",
                        self.sliced_load_plan.rank
                    )
                })?;
        debug!(
            "kimi-k2: rank {rank} one-token forward cache build cost {:.2}s",
            cache_started.elapsed().as_secs_f64()
        );
        let decode_arenas =
            KimiWorkerDecodeArenas::new(one_token_cache.vocab_rows, &self.local_dims);
        let report = KimiRankWeightLoadReport::from_loaded_weights(
            tensor_count,
            total_bytes,
            &expert_kernel_weights,
        );
        let loaded = KimiRankLoadedWeights {
            gpu: weights,
            expert_kernels: expert_kernel_weights,
            one_token_cache,
            decode_arenas,
        };
        ensure!(
            loaded.gpu.rank == report.rank,
            "Kimi loaded rank {} does not match report rank {}",
            loaded.gpu.rank,
            report.rank
        );
        ensure!(
            loaded.expert_kernels.layers.len() == report.expert_kernel_layers,
            "Kimi expert kernel layer count {} does not match report count {}",
            loaded.expert_kernels.layers.len(),
            report.expert_kernel_layers
        );
        self.loaded = Some(loaded);
        debug!(
            "kimi-k2: rank {rank} rank weight init cost {:.2}s: tensors={}, bytes={}, expert_layers={}",
            started.elapsed().as_secs_f64(),
            tensor_count,
            ByteSize(total_bytes as u64),
            report.expert_kernel_layers
        );
        Ok(report)
    }

    pub(super) fn forward_prompt_next_token(
        &mut self,
        slot: usize,
        decode_batch_size: usize,
        input_ids: &[u32],
        ep_max_seq_len: usize,
        logprobs: usize,
    ) -> Result<KimiOneTokenForwardReport> {
        self.forward_prompt_next_token_inner(
            slot,
            decode_batch_size,
            input_ids,
            ep_max_seq_len,
            logprobs,
        )
    }

    pub(super) fn ensure_decode_arena(&mut self, decode_batch_size: usize) -> Result<()> {
        self.ctx.set_current()?;
        let device_ctx = self.ctx.as_device_context();
        let (rank, arena_batch_size) = {
            let loaded = self.loaded.as_mut().ok_or_else(|| {
                anyhow::anyhow!("Kimi rank weights must be loaded before decode arena allocation")
            })?;
            let arena = loaded
                .decode_arenas
                .get_mut(&device_ctx, decode_batch_size)?;
            (loaded.gpu.rank, arena.batch_size)
        };
        if self.ep_backend.is_some() {
            ensure_pplx_decode_scratch(
                &device_ctx,
                rank,
                &mut self.moe_pplx_scratch,
                arena_batch_size,
            )?;
        }
        Ok(())
    }

    pub(super) fn forward_decode_batch_next_tokens(
        &mut self,
        token_ids: &[u32],
        append_positions: &[usize],
        slots: &[usize],
        decode_batch_size: usize,
        logprobs: &[usize],
    ) -> Result<Vec<KimiOneTokenForwardReport>> {
        ensure!(!token_ids.is_empty(), "Kimi batch decode requires tokens");
        ensure!(
            token_ids.len() == append_positions.len() && token_ids.len() == slots.len(),
            "Kimi batch decode input length mismatch: tokens={}, positions={}, slots={}",
            token_ids.len(),
            append_positions.len(),
            slots.len()
        );
        ensure!(
            logprobs.len() == token_ids.len(),
            "Kimi batch decode logprobs length mismatch: tokens={}, logprobs={}",
            token_ids.len(),
            logprobs.len()
        );
        self.ctx.set_current()?;
        let loaded = self.loaded.as_mut().ok_or_else(|| {
            anyhow::anyhow!("Kimi rank weights must be loaded before batch decode")
        })?;
        let tp_comm_ref = self.tp_comm.as_ref().map(super::OwnedRankComm::get);
        let device_ctx = self.ctx.as_device_context();
        let decode_aux_ctx = DeviceContext {
            ctx: Arc::clone(&self.decode_aux_ctx.ctx),
            stream: Arc::clone(&self.decode_aux_ctx.stream),
            device_ordinal: self.decode_aux_ctx.device_ordinal,
        };
        let KimiRankLoadedWeights {
            gpu,
            expert_kernels,
            one_token_cache: cache,
            decode_arenas,
        } = loaded;
        let rank = gpu.rank;
        let active_len = token_ids.len();
        ensure!(
            (1..=KIMI_DECODE_MAX_BATCH).contains(&decode_batch_size),
            "Kimi decode batch size {decode_batch_size} must be in 1..={KIMI_DECODE_MAX_BATCH}"
        );
        ensure!(
            active_len <= decode_batch_size,
            "Kimi active decode rows {active_len} exceed decode batch size {decode_batch_size}"
        );
        let decode_arena = decode_arenas.get_mut(&device_ctx, decode_batch_size)?;
        #[cfg(feature = "kernel-call-trace")]
        if rank == 0 && call_trace::is_enabled() {
            let kv_len = append_positions
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            for call in crate::batch_decode_trace::trace_decode_kernel_calls(
                "",
                decode_arena.batch_size,
                kv_len,
            )? {
                call_trace::record_call(call);
            }
        }
        decode_arena
            .configure_batch_decode(&device_ctx, slots, append_positions)
            .with_context(|| format!("Kimi rank {rank} configure batch decode KV page table"))?;
        decode_arena
            .upload_batch_tokens(&device_ctx, token_ids)
            .with_context(|| format!("Kimi rank {rank} upload batch decode tokens"))?;

        let local_heads = self.local_dims.local_heads;
        let forward_result = if self.enable_cuda_graph {
            let mut graph = std::mem::take(&mut decode_arena.graph);
            let graph_barrier = Arc::clone(&self.collective_barrier);
            let result = graph.run_or_capture_synchronized(
                &device_ctx,
                |_| {
                    graph_barrier.wait();
                },
                || {
                    forward_decode_batch_next_token_kernels(
                        &device_ctx,
                        &decode_aux_ctx,
                        tp_comm_ref,
                        cache,
                        expert_kernels,
                        decode_arena,
                        active_len,
                        local_heads,
                        None,
                    )
                },
            );
            decode_arena.graph = graph;
            result
        } else {
            if self.ep_backend.is_some() {
                ensure_pplx_decode_scratch(
                    &device_ctx,
                    rank,
                    &mut self.moe_pplx_scratch,
                    decode_arena.batch_size,
                )?;
            }
            let mut pplx_ctx = match self.ep_backend.as_mut() {
                Some(ep) => {
                    let scratch = self.moe_pplx_scratch.as_mut().ok_or_else(|| {
                        anyhow::anyhow!("Kimi rank {rank} PPLX decode scratch is missing")
                    })?;
                    Some(PplxDecodeContext { ep, scratch })
                }
                None => None,
            };
            forward_decode_batch_next_token_kernels(
                &device_ctx,
                &decode_aux_ctx,
                tp_comm_ref,
                cache,
                expert_kernels,
                decode_arena,
                active_len,
                local_heads,
                pplx_ctx.as_mut(),
            )
        };
        forward_result?;

        let local_top1 = read_local_top1_batch_values(
            &device_ctx,
            &decode_arena.logits,
            active_len,
            &mut decode_arena.scratch.sampling.top1_value_scratch,
            &mut decode_arena.scratch.sampling.top1_out,
        )?;
        let host_logits = if logprobs.iter().any(|&k| k > 0) {
            ensure!(
                cache.vocab_start == 0 && cache.vocab_rows == KIMI_K2_VOCAB,
                "Kimi logprobs require an unsharded vocab (TP1); a vocab shard's \
                 logsumexp is not the global one (#236)"
            );
            Some(
                device_ctx
                    .stream
                    .clone_dtoh(&decode_arena.logits.data)
                    .with_context(|| format!("Kimi rank {rank} D2H decode logits for logprobs"))?,
            )
        } else {
            None
        };
        let mut reports = Vec::with_capacity(active_len);
        for (row, (local_next, local_top_logit_f32)) in local_top1.into_iter().enumerate() {
            let logprob = match &host_logits {
                Some(host) if logprobs[row] > 0 => Some(host_token_logprob(
                    &host[row * cache.vocab_rows..(row + 1) * cache.vocab_rows],
                    local_next as usize,
                    logprobs[row],
                )),
                _ => None,
            };
            reports.push(KimiOneTokenForwardReport {
                rank,
                batch_slot: slots[row],
                input_token_id: token_ids[row],
                local_next_token_id: local_next,
                local_next_token_global_id: cache.vocab_start as u32 + local_next,
                local_top_logit_f32,
                vocab_start: cache.vocab_start,
                vocab_rows: cache.vocab_rows,
                dense_layers_executed: KIMI_K2_DENSE_LAYERS,
                moe_layers_executed: KIMI_K2_MOE_LAYERS,
                logprob,
            });
        }
        Ok(reports)
    }

    fn forward_prompt_next_token_inner(
        &mut self,
        slot: usize,
        decode_batch_size: usize,
        input_ids: &[u32],
        ep_max_seq_len: usize,
        logprobs: usize,
    ) -> Result<KimiOneTokenForwardReport> {
        ensure!(!input_ids.is_empty(), "Kimi prompt forward requires tokens");
        self.ctx.set_current()?;
        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Kimi rank weights must be loaded before forward"))?;
        let tp_comm_ref = self.tp_comm.as_ref().map(super::OwnedRankComm::get);
        let device_ctx = self.ctx.as_device_context();
        let KimiRankLoadedWeights {
            gpu,
            expert_kernels,
            one_token_cache: cache,
            decode_arenas,
        } = loaded;
        let rank = gpu.rank;
        let seq_len = input_ids.len();
        let input_token_id = *input_ids
            .last()
            .ok_or_else(|| anyhow::anyhow!("Kimi prompt ids unexpectedly empty"))?;
        let decode_arena = decode_arenas.get_mut(&device_ctx, decode_batch_size)?;
        decode_arena
            .configure_slot_prefill(&device_ctx, slot, seq_len)
            .with_context(|| {
                format!("Kimi rank {rank} configure slot {slot} prefill KV page table")
            })?;

        let mut hidden = GpuTensor::<KIMI_K2_HIDDEN>::zeros(&device_ctx, seq_len)?;
        let token_ids = device_ctx.stream.clone_htod(input_ids)?;
        typed_ops::embedding_vocab_shard_into(
            &device_ctx,
            &cache.token_embedding,
            &token_ids,
            &mut hidden,
            cache.vocab_start as u32,
        )?;
        self.collective_barrier.wait();
        if let Some(comm) = tp_comm_ref {
            device_ctx
                .sync()
                .with_context(|| format!("Kimi rank {} sync before first TP all-reduce", rank))?;
            comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum)
                .map_err(|err| {
                    anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
                })?;
        }

        let (cos_host, sin_host) = build_yarn_rope_cache(seq_len);
        let cos = device_ctx.stream.clone_htod(&cos_host)?;
        let sin = device_ctx.stream.clone_htod(&sin_host)?;
        let mut normed = GpuTensor::<KIMI_K2_HIDDEN>::zeros(&device_ctx, seq_len)?;
        let mut next_hidden = GpuTensor::<KIMI_K2_HIDDEN>::zeros(&device_ctx, seq_len)?;
        let decode_aux_ctx = DeviceContext {
            ctx: Arc::clone(&self.decode_aux_ctx.ctx),
            stream: Arc::clone(&self.decode_aux_ctx.stream),
            device_ordinal: self.decode_aux_ctx.device_ordinal,
        };
        let mut pplx_prefill_scratch = if tp_comm_ref.is_none() && ep_max_seq_len > 0 {
            Some(
                crate::runner::moe_pplx::KimiMoePplxScratch::new_prefill(
                    &device_ctx,
                    ep_max_seq_len,
                )
                .with_context(|| {
                    format!(
                        "Kimi rank {rank} PPLX prefill scratch (ep_max_seq_len={ep_max_seq_len})"
                    )
                })?,
            )
        } else {
            None
        };

        let mut dense_layers_executed = 0usize;
        let mut moe_layers_executed = 0usize;
        for layer in &cache.layers {
            Self::forward_mla_prefill(
                &device_ctx,
                tp_comm_ref,
                layer.layer_idx,
                &layer.attention,
                &cos,
                &sin,
                decode_arena,
                &mut hidden,
                &mut normed,
                &mut next_hidden,
                self.local_dims.local_heads,
            )
            .with_context(|| format!("Kimi MLA prefill layer {}", layer.layer_idx))?;
            match &layer.kind {
                KimiLayerForwardKindCache::Dense(dense) => {
                    Self::forward_dense_mlp(
                        &device_ctx,
                        tp_comm_ref,
                        dense,
                        &layer.attention.post_attention_norm,
                        &mut hidden,
                        &mut normed,
                        &mut next_hidden,
                    )
                    .with_context(|| format!("Kimi dense MLP layer {}", layer.layer_idx))?;
                    dense_layers_executed += 1;
                }
                KimiLayerForwardKindCache::Moe(moe) => {
                    if let Some(pplx_scratch) = pplx_prefill_scratch.as_mut() {
                        crate::runner::moe_pplx::forward_moe_layer_prefill_pplx(
                            &device_ctx,
                            &decode_aux_ctx,
                            self.ep_backend.as_mut().expect("TP1 requires PPLX"),
                            layer.layer_idx,
                            moe,
                            &layer.attention.post_attention_norm,
                            expert_kernels,
                            &mut hidden,
                            &mut normed,
                            &mut next_hidden,
                            pplx_scratch,
                        )
                        .with_context(|| {
                            format!("Kimi MoE PPLX prefill layer {}", layer.layer_idx)
                        })?;
                    } else {
                        Self::forward_moe_layer(
                            &device_ctx,
                            tp_comm_ref,
                            layer.layer_idx,
                            moe,
                            &layer.attention.post_attention_norm,
                            expert_kernels,
                            &mut hidden,
                            &mut normed,
                            &mut next_hidden,
                        )
                        .with_context(|| format!("Kimi MoE layer {}", layer.layer_idx))?;
                    }
                    moe_layers_executed += 1;
                }
            }
        }

        typed_ops::rms_norm_into(
            &device_ctx,
            &hidden,
            &cache.final_norm,
            KIMI_K2_RMS_NORM_EPS,
            &mut normed,
        )?;
        let mut logits_hidden = HiddenStates::zeros(&device_ctx, cache.vocab_rows, seq_len)?;
        typed_ops::gemm_runtime_out_into(&device_ctx, &cache.lm_head, &normed, &mut logits_hidden)?;
        let logits_offset = (seq_len - 1) * cache.vocab_rows;
        let logits_last = logits_hidden
            .data
            .slice(logits_offset..logits_offset + cache.vocab_rows);
        let mut logits_data = device_ctx.stream.alloc_zeros(cache.vocab_rows)?;
        device_ctx
            .stream
            .memcpy_dtod(&logits_last, &mut logits_data)?;
        let logits = DeviceVec {
            data: logits_data,
            len: cache.vocab_rows,
        };
        let (local_next, local_top_logit_f32) = sample_local_top1_with_value(&device_ctx, &logits)?;
        let logprob = if logprobs > 0 {
            ensure!(
                cache.vocab_start == 0 && cache.vocab_rows == KIMI_K2_VOCAB,
                "Kimi logprobs require an unsharded vocab (TP1); a vocab \
                 shard's logsumexp is not the global one (#236)"
            );
            let host = device_ctx
                .stream
                .clone_dtoh(&logits.data)
                .with_context(|| format!("Kimi rank {rank} D2H prefill logits"))?;
            Some(host_token_logprob(&host, local_next as usize, logprobs))
        } else {
            None
        };

        let report = KimiOneTokenForwardReport {
            rank,
            batch_slot: slot,
            input_token_id,
            local_next_token_id: local_next,
            local_next_token_global_id: cache.vocab_start as u32 + local_next,
            local_top_logit_f32,
            vocab_start: cache.vocab_start,
            vocab_rows: cache.vocab_rows,
            dense_layers_executed,
            moe_layers_executed,
            logprob,
        };
        Ok(report)
    }

    fn forward_mla_prefill(
        ctx: &DeviceContext,
        comm: Option<&Comm>,
        layer_idx: usize,
        attention: &KimiAttentionForwardCache,
        cos: &CudaSlice<half::bf16>,
        sin: &CudaSlice<half::bf16>,
        decode_arena: &mut KimiWorkerDecodeArena,
        hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
        normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
        next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
        local_heads: usize,
    ) -> Result<()> {
        let seq_len = hidden.seq_len;
        let q_proj_out = local_heads * KIMI_K2_MLA_Q_HEAD_DIM;
        let kv_b_out = attention.kv_b_proj.rows;
        pegainfer_kernels::typed_pipeline! {
            ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS, seq_len = seq_len, gemm = prefill;
            tensor qkv_a: KIMI_K2_MLA_QKV_A_OUT;
            tensor q_a: KIMI_K2_Q_LORA_RANK;
            tensor q_a_normed: KIMI_K2_Q_LORA_RANK;
            tensor compressed_kv: KIMI_K2_MLA_KV_LORA_RANK;
            tensor k_rope: KIMI_K2_MLA_ROPE_DIM;
            tensor compressed_normed: KIMI_K2_MLA_KV_LORA_RANK;
            tensor append_kpe: KIMI_K2_MLA_ROPE_DIM;

            rms_norm(hidden => normed, attention.input_norm);
            gemm(normed => &mut qkv_a, attention.fused_qkv_a_proj);
            try kimi_mla_split_qkv_a(ctx, &qkv_a, &mut q_a, &mut compressed_kv, &mut k_rope);
            rms_norm(&q_a => &mut q_a_normed, attention.q_a_norm);
        }
        let mut q_proj = HiddenStates::zeros(ctx, q_proj_out, seq_len)?;
        typed_ops::gemm_dm_typed_to_hs(ctx, &attention.q_b_proj, &q_a_normed, &mut q_proj)?;
        typed_ops::rms_norm_into(
            ctx,
            &compressed_kv,
            &attention.kv_a_norm,
            KIMI_K2_RMS_NORM_EPS,
            &mut compressed_normed,
        )?;
        kimi_mla_rope_apply_kpe(
            ctx,
            &k_rope,
            cos,
            sin,
            &decode_arena.positions_d,
            &mut append_kpe,
        )?;
        decode_arena.append_prefill_layer_kv(ctx, layer_idx, &compressed_normed, &append_kpe)?;
        let mut kv_b = HiddenStates::zeros(ctx, kv_b_out, seq_len)?;
        typed_ops::gemm_dm_typed_to_hs(ctx, &attention.kv_b_proj, &compressed_normed, &mut kv_b)?;

        let mut k_cache = ctx
            .stream
            .alloc_zeros(seq_len * local_heads * KIMI_K2_MLA_Q_HEAD_DIM)?;
        let mut v_cache = ctx
            .stream
            .alloc_zeros(seq_len * local_heads * KIMI_K2_MLA_V_HEAD_DIM)?;
        let mut q_attn = HiddenStates::zeros(ctx, q_proj_out, seq_len)?;
        kimi_mla_rope_assemble_prefill_rt(
            ctx,
            &q_proj,
            &k_rope,
            &kv_b,
            cos,
            sin,
            &mut q_attn,
            &mut k_cache,
            &mut v_cache,
            local_heads,
        )?;

        let o_proj_in = local_heads * KIMI_K2_MLA_V_HEAD_DIM;
        let mut attn_out = HiddenStates::zeros(ctx, o_proj_in, seq_len)?;
        kimi_flashinfer_single_prefill_mla_rt(
            ctx,
            &q_attn,
            &k_cache,
            &v_cache,
            &mut attn_out,
            kimi_mla_softmax_scale(),
            local_heads,
        )?;
        let mut projected = GpuTensor::<KIMI_K2_HIDDEN>::zeros(ctx, seq_len)?;
        typed_ops::gemm_dm_hs_to_typed(ctx, &attention.o_proj, &attn_out, &mut projected)?;
        if let Some(comm) = comm {
            comm.all_reduce_in_place(&mut projected.data, &ReduceOp::Sum)
                .map_err(|err| {
                    anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
                })?;
        }
        typed_ops::add_into(ctx, hidden, &projected, next_hidden)?;
        std::mem::swap(hidden, next_hidden);
        Ok(())
    }

    fn forward_dense_mlp(
        ctx: &DeviceContext,
        comm: Option<&Comm>,
        dense: &KimiDenseForwardCache,
        post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
        hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
        normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
        next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    ) -> Result<()> {
        forward_dense_mlp_batch_into(
            ctx,
            comm,
            dense,
            post_attention_norm,
            hidden,
            normed,
            next_hidden,
        )
    }

    fn forward_moe_layer(
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
        crate::runner::moe_nccl::forward_moe_layer_batch_into(
            ctx,
            comm,
            layer_idx,
            moe,
            post_attention_norm,
            expert_kernels,
            hidden,
            normed,
            next_hidden,
        )
    }
}
fn ensure_pplx_decode_scratch(
    ctx: &DeviceContext,
    rank: usize,
    scratch: &mut Option<crate::runner::moe_pplx::KimiMoePplxScratch>,
    batch_size: usize,
) -> Result<()> {
    let needs_alloc = match scratch.as_ref() {
        Some(scratch) => scratch.max_local_output_tokens < batch_size,
        None => true,
    };
    if needs_alloc {
        *scratch = Some(
            crate::runner::moe_pplx::KimiMoePplxScratch::new_decode(ctx, batch_size).with_context(
                || format!("Kimi rank {rank} PPLX decode scratch allocation bs{batch_size}"),
            )?,
        );
    }
    Ok(())
}
