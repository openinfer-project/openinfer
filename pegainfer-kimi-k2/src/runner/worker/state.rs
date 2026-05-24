use super::{forward::*, runtime::*, *};

impl KimiRankThreadState {
    #[cfg(feature = "pplx-ep")]
    pub(super) fn enable_pplx(&mut self, ep_backend: pegainfer_comm::EpBackend) -> Result<()> {
        self.ctx.set_current()?;
        if self.moe_pplx_scratch.is_none() {
            self.moe_pplx_scratch = Some(
                super::moe_pplx::KimiMoePplxScratch::new(
                    &self.ctx.as_device_context(),
                    KIMI_DECODE_MAX_BATCH,
                )
                .with_context(|| {
                    format!(
                        "Kimi rank {} PPLX scratch allocation",
                        self.sliced_load_plan.rank
                    )
                })?,
            );
        }
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
        let comm = Comm::from_rank(self.ctx.stream.clone(), rank, world_size, id)
            .map_err(|err| anyhow::anyhow!("Kimi rank {rank} NCCL init failed: {err:?}"))?;
        self.tp_comm = Some(OwnedRankComm(comm));
        Ok(())
    }

    pub(super) fn load_sliced_weights(
        &mut self,
        model_path: &Path,
    ) -> Result<KimiRankWeightLoadReport> {
        let mut weights =
            load_rank_sliced_weights_to_gpu(&self.ctx, model_path, &self.sliced_load_plan)
                .with_context(|| {
                    format!(
                        "failed to load Kimi rank {} sliced weights to GPU",
                        self.sliced_load_plan.rank
                    )
                })?;
        weights.typed_view(&self.weight_names).with_context(|| {
            format!(
                "failed to validate Kimi rank {} typed GPU weight view",
                self.sliced_load_plan.rank
            )
        })?;
        let tensor_count = weights.tensors.len();
        let total_bytes = weights.total_bytes;
        let expert_kernel_weights = weights
            .pack_rank_expert_marlin_weights(&self.ctx, &self.weight_names)
            .with_context(|| {
                format!(
                    "failed to package Kimi rank {} expert Marlin weights",
                    self.sliced_load_plan.rank
                )
            })?;
        let one_token_cache =
            KimiOneTokenForwardCache::from_gpu_weights(&self.ctx, &weights, &self.weight_names)
                .with_context(|| {
                    format!(
                        "failed to build Kimi rank {} one-token forward cache",
                        self.sliced_load_plan.rank
                    )
                })?;
        let decode_arenas =
            KimiWorkerDecodeArenas::new(&self.ctx.as_device_context(), one_token_cache.vocab_rows)
                .with_context(|| {
                    format!(
                        "failed to allocate Kimi rank {} decode arenas",
                        self.sliced_load_plan.rank
                    )
                })?;
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
        self.weight_report = Some(report.clone());
        Ok(report)
    }

    pub(super) fn forward_prompt_next_token(
        &mut self,
        slot: usize,
        decode_batch_size: usize,
        input_ids: &[u32],
    ) -> Result<KimiOneTokenForwardReport> {
        self.forward_prompt_next_token_inner(slot, decode_batch_size, input_ids)
    }

    pub(super) fn forward_decode_batch_next_tokens(
        &mut self,
        token_ids: &[u32],
        append_positions: &[usize],
        slots: &[usize],
        decode_batch_size: usize,
    ) -> Result<Vec<KimiOneTokenForwardReport>> {
        ensure!(!token_ids.is_empty(), "Kimi batch decode requires tokens");
        ensure!(
            token_ids.len() == append_positions.len() && token_ids.len() == slots.len(),
            "Kimi batch decode input length mismatch: tokens={}, positions={}, slots={}",
            token_ids.len(),
            append_positions.len(),
            slots.len()
        );
        self.ctx.set_current()?;
        let loaded = self.loaded.as_mut().ok_or_else(|| {
            anyhow::anyhow!("Kimi rank weights must be loaded before batch decode")
        })?;
        let tp_comm = self.tp_comm.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Kimi rank {} TP comm must be attached before batch decode",
                loaded.gpu.rank
            )
        })?;
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
        #[cfg(feature = "kernel-call-trace")]
        if rank == 0 && call_trace::is_enabled() {
            let kv_len = append_positions
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            for call in
                crate::batch_decode_trace::trace_decode_kernel_calls("", decode_batch_size, kv_len)?
            {
                call_trace::record_call(call);
            }
        }
        ensure!(
            (1..=KIMI_DECODE_MAX_BATCH).contains(&decode_batch_size),
            "Kimi decode batch size {decode_batch_size} must be in 1..={KIMI_DECODE_MAX_BATCH}"
        );
        ensure!(
            active_len <= decode_batch_size,
            "Kimi active decode rows {active_len} exceed decode batch size {decode_batch_size}"
        );
        let decode_arena = decode_arenas.get_mut(decode_batch_size)?;
        decode_arena
            .configure_batch_decode(&device_ctx, slots, append_positions)
            .with_context(|| format!("Kimi rank {rank} configure batch decode KV page table"))?;
        decode_arena
            .upload_batch_tokens(&device_ctx, token_ids)
            .with_context(|| format!("Kimi rank {rank} upload batch decode tokens"))?;

        if self.enable_cuda_graph {
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
                        tp_comm.get(),
                        cache,
                        expert_kernels,
                        decode_arena,
                        #[cfg(feature = "pplx-ep")]
                        None,
                    )
                },
            );
            decode_arena.graph = graph;
            result?;
        } else {
            #[cfg(feature = "pplx-ep")]
            let mut pplx_ctx = self
                .ep_backend
                .as_mut()
                .zip(self.moe_pplx_scratch.as_mut())
                .map(|(ep, scratch)| PplxDecodeContext { ep, scratch });
            forward_decode_batch_next_token_kernels(
                &device_ctx,
                &decode_aux_ctx,
                tp_comm.get(),
                cache,
                expert_kernels,
                decode_arena,
                #[cfg(feature = "pplx-ep")]
                pplx_ctx.as_mut(),
            )?;
        }

        let local_top1 = read_local_top1_batch_values(
            &device_ctx,
            &decode_arena.logits,
            active_len,
            &mut decode_arena.scratch.sampling.top1_value_scratch,
            &mut decode_arena.scratch.sampling.top1_out,
        )?;
        let mut reports = Vec::with_capacity(active_len);
        for (row, (local_next, local_top_logit_f32)) in local_top1.into_iter().enumerate() {
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
            });
        }
        Ok(reports)
    }

    pub(super) fn forward_prompt_next_token_inner(
        &mut self,
        slot: usize,
        decode_batch_size: usize,
        input_ids: &[u32],
    ) -> Result<KimiOneTokenForwardReport> {
        ensure!(!input_ids.is_empty(), "Kimi prompt forward requires tokens");
        self.ctx.set_current()?;
        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Kimi rank weights must be loaded before forward"))?;
        let tp_comm = self.tp_comm.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Kimi rank {} TP comm must be attached before forward",
                loaded.gpu.rank
            )
        })?;
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
        let decode_arena = decode_arenas.get_mut(decode_batch_size)?;
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
        device_ctx
            .sync()
            .with_context(|| format!("Kimi rank {} sync before first TP all-reduce", rank))?;
        tp_comm
            .get()
            .all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
            })?;

        let (cos_host, sin_host) = build_yarn_rope_cache(seq_len);
        let cos = device_ctx.stream.clone_htod(&cos_host)?;
        let sin = device_ctx.stream.clone_htod(&sin_host)?;
        let mut normed = GpuTensor::<KIMI_K2_HIDDEN>::zeros(&device_ctx, seq_len)?;
        let mut next_hidden = GpuTensor::<KIMI_K2_HIDDEN>::zeros(&device_ctx, seq_len)?;

        let mut dense_layers_executed = 0usize;
        let mut moe_layers_executed = 0usize;
        for layer in &cache.layers {
            Self::forward_mla_prefill(
                &device_ctx,
                tp_comm.get(),
                layer.layer_idx,
                &layer.attention,
                &cos,
                &sin,
                decode_arena,
                &mut hidden,
                &mut normed,
                &mut next_hidden,
            )
            .with_context(|| format!("Kimi MLA prefill layer {}", layer.layer_idx))?;
            match &layer.kind {
                KimiLayerForwardKindCache::Dense(dense) => {
                    Self::forward_dense_mlp(
                        &device_ctx,
                        tp_comm.get(),
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
                    Self::forward_moe_layer(
                        &device_ctx,
                        tp_comm.get(),
                        layer.layer_idx,
                        moe,
                        &layer.attention.post_attention_norm,
                        expert_kernels,
                        &mut hidden,
                        &mut normed,
                        &mut next_hidden,
                    )
                    .with_context(|| format!("Kimi MoE layer {}", layer.layer_idx))?;
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

        Ok(KimiOneTokenForwardReport {
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
        })
    }

    fn forward_mla_prefill(
        ctx: &DeviceContext,
        comm: &Comm,
        layer_idx: usize,
        attention: &KimiAttentionForwardCache,
        cos: &CudaSlice<half::bf16>,
        sin: &CudaSlice<half::bf16>,
        decode_arena: &mut KimiWorkerDecodeArena,
        hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
        normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
        next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    ) -> Result<()> {
        let seq_len = hidden.seq_len;
        pegainfer_kernels::typed_pipeline! {
            ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS, seq_len = seq_len, gemm = prefill;
            tensor qkv_a: KIMI_K2_MLA_QKV_A_OUT;
            tensor q_a: KIMI_K2_Q_LORA_RANK;
            tensor q_a_normed: KIMI_K2_Q_LORA_RANK;
            tensor q_proj: KIMI_K2_MLA_Q_LOCAL_OUT_TP8;
            tensor compressed_kv: KIMI_K2_MLA_KV_LORA_RANK;
            tensor k_rope: KIMI_K2_MLA_ROPE_DIM;
            tensor compressed_normed: KIMI_K2_MLA_KV_LORA_RANK;
            tensor kv_b: KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8;
            tensor append_kpe: KIMI_K2_MLA_ROPE_DIM;
            tensor q_attn: KIMI_K2_MLA_Q_LOCAL_OUT_TP8;

            rms_norm(hidden => normed, attention.input_norm);
            gemm(normed => &mut qkv_a, attention.fused_qkv_a_proj);
            try kimi_mla_split_qkv_a(ctx, &qkv_a, &mut q_a, &mut compressed_kv, &mut k_rope);
            rms_norm(&q_a => &mut q_a_normed, attention.q_a_norm);
            gemm(&q_a_normed => &mut q_proj, attention.q_b_proj);
            rms_norm(&compressed_kv => &mut compressed_normed, attention.kv_a_norm);
            try kimi_mla_rope_apply_kpe(ctx, &k_rope, cos, sin, &decode_arena.positions_d, &mut append_kpe);
            try decode_arena.append_prefill_layer_kv(ctx, layer_idx, &compressed_normed, &append_kpe);
            gemm(&compressed_normed => &mut kv_b, attention.kv_b_proj);
        }
        let mut k_cache = ctx
            .stream
            .alloc_zeros(seq_len * KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_Q_HEAD_DIM)?;
        let mut v_cache = ctx
            .stream
            .alloc_zeros(seq_len * KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_V_HEAD_DIM)?;
        kimi_mla_rope_assemble_prefill(
            ctx,
            &q_proj,
            &k_rope,
            &kv_b,
            cos,
            sin,
            &mut q_attn,
            &mut k_cache,
            &mut v_cache,
        )?;

        pegainfer_kernels::typed_pipeline! {
            ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS, seq_len = seq_len, gemm = prefill;
            tensor attn_out: KIMI_K2_MLA_O_LOCAL_IN_TP8;
            tensor projected: KIMI_K2_HIDDEN;

            try kimi_flashinfer_single_prefill_mla(ctx, &q_attn, &k_cache, &v_cache, &mut attn_out, kimi_mla_softmax_scale());
            gemm(&attn_out => &mut projected, attention.o_proj);
        }
        comm.all_reduce_in_place(&mut projected.data, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
            })?;
        pegainfer_kernels::typed_pipeline! {
            ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS;
            add(hidden, &projected => next_hidden);
            swap(hidden, next_hidden);
        }
        Ok(())
    }

    fn forward_dense_mlp(
        ctx: &DeviceContext,
        comm: &Comm,
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
        comm: &Comm,
        layer_idx: usize,
        moe: &KimiMoeForwardCache,
        post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
        expert_kernels: &KimiRankExpertMarlinWeights,
        hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
        normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
        next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    ) -> Result<()> {
        forward_moe_layer_batch_into(
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
