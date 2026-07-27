//! GLM5.2 native TP4 prefill executor, layer-outer over the coordinator
//! chunk: every layer stage (norm, MLA front, KV pack, o_proj, MoE) runs at
//! chunk M so the fp8 GEMMs stay large and each MoE layer reads its expert
//! bank once per chunk (not once per tile). Only two stages sub-tile:
//! the DSA indexer (32 rows — the DeepGEMM paged-MQA AOT batch) and the
//! FlashMLA sparse attention (`PREFILL_ATTN_TILE_ROWS`). TP reductions ride
//! NCCL bf16 all-reduces (`Glm52MoeTpState::prefill_allreduce`).
//!
//! Causality note: packing the whole chunk's KV before attention is safe —
//! the indexer masks each query to `positions[row] + 1` keys, so keys packed
//! for later in-chunk positions are never selected for earlier queries.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::Glm52IndexerCacheLayout;
use openinfer_kernels::ops::Glm52MoeQuantShape;
use openinfer_kernels::ops::add_into;
use openinfer_kernels::ops::argmax_batch_bf16_split_partials_len;
use openinfer_kernels::ops::argmax_bf16_split_into;
use openinfer_kernels::ops::embedding_rows_into;
use openinfer_kernels::ops::fused_add_rms_norm_round_into;
use openinfer_kernels::ops::gemm_strided_batched_bf16;
use openinfer_kernels::ops::glm52_flashmla_sparse_prefill_launch;
use openinfer_kernels::ops::glm52_fp8_per_token_group_quant_bf16_ue8m0_launch;
use openinfer_kernels::ops::glm52_mla_cache_pack_launch;
use openinfer_kernels::ops::glm52_mla_query_assemble_launch;
use openinfer_kernels::ops::glm52_prefill_moe_gather_rows_launch;
use openinfer_kernels::ops::glm52_prefill_unpack_pages_launch;
use openinfer_kernels::ops::glm52_vocab_parallel_pack_launch;
use openinfer_kernels::ops::glm52_vocab_parallel_unpack_launch;
use openinfer_kernels::ops::rms_norm_rows_into;
use openinfer_kernels::tensor::DeviceContext;
use openinfer_kernels::tensor::DeviceMatrix;
use openinfer_kernels::tensor::DeviceVec;
use openinfer_kernels::tensor::HiddenStatesRef;
use openinfer_sample::BatchSamplingRow;
use openinfer_sample::BatchSamplingScratch;
use openinfer_sample::gpu_sample_batch_into;
use openinfer_sample::mix_seed;

use crate::bookend::glm52_final_norm_into;
use crate::bookend::glm52_lm_head_into;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_KV_A_OUT;
use crate::config::GLM52_KV_LORA_RANK;
use crate::config::GLM52_QK_HEAD_DIM;
use crate::config::GLM52_QK_NOPE_HEAD_DIM;
use crate::config::GLM52_RMS_EPS;
use crate::config::GLM52_ROPE_HALF;
use crate::config::GLM52_VOCAB;
use crate::dense::Glm52DenseMlpWeights;
use crate::dense::glm52_dense_mlp_prefill_into;
use crate::fp8::Glm52Fp8GemmScratch;
use crate::fp8::fp8_linear_large_m_into;
use crate::indexer::Glm52IndexerPrefillScratch;
use crate::layer::Glm52DecoderLayerWeights;
use crate::layer::Glm52LayerCaches;
use crate::layer::Glm52LayerIndexer;
use crate::layer::Glm52LayerMlp;
use crate::mla_front::Glm52MlaFront;
use crate::mla_front::Glm52MlaLayerWeights;
use crate::mla_front::glm52_mla_prefill_front_into;
use crate::moe_tp::Glm52MoeTpPrefillScratch;
use crate::moe_tp::Glm52MoeTpRank;
use crate::moe_tp::Glm52MoeTpState;
use crate::rows::Rows;
use crate::runner::Glm52PrefillBatch;

/// FlashMLA sparse attention sub-tile (query rows per launch).
const PREFILL_ATTN_TILE_ROWS: usize = 512;
/// Dense-MLP sub-tile: bounds the 12288-wide gate|up scratch.
const PREFILL_DENSE_TILE_ROWS: usize = 2048;

const GLM52_INDEXER_TOPK: usize = 2048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Glm52TpPrefillLayout {
    kv_slots: usize,
    table_width: usize,
    chunk_rows: usize,
}

impl Glm52TpPrefillLayout {
    fn new(kv_slots: usize, table_width: usize, chunk_rows: usize) -> Result<Self> {
        ensure!(
            kv_slots > 0 && table_width > 0 && chunk_rows > 0,
            "prefill capacities must be positive"
        );
        Ok(Self {
            kv_slots,
            table_width,
            chunk_rows: chunk_rows.next_multiple_of(4),
        })
    }
}

/// Env-gated (`OPENINFER_GLM52_PREFILL_PROFILE=1`) CUDA-event section
/// profile: per-section call counts and summed GPU ms per chunk forward.
struct Glm52PrefillProfiler {
    enabled: bool,
    sections: Vec<(&'static str, usize, Vec<(CudaEvent, CudaEvent)>)>,
}

impl Glm52PrefillProfiler {
    fn new() -> Self {
        Self {
            enabled: std::env::var("OPENINFER_GLM52_PREFILL_PROFILE").as_deref() == Ok("1"),
            sections: Vec::new(),
        }
    }

    fn start(&self, ctx: &DeviceContext) -> Result<Option<CudaEvent>> {
        if !self.enabled {
            return Ok(None);
        }
        // Explicit default flags: cudarc's `None` means DISABLE_TIMING, which
        // would make `elapsed_ms` fail with CUDA_ERROR_INVALID_HANDLE.
        Ok(Some(ctx.stream.record_event(Some(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
        ))?))
    }

    fn stop(
        &mut self,
        ctx: &DeviceContext,
        name: &'static str,
        begin: Option<CudaEvent>,
    ) -> Result<()> {
        let Some(begin) = begin else {
            return Ok(());
        };
        let end = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        match self.sections.iter_mut().find(|(n, _, _)| *n == name) {
            Some((_, count, pairs)) => {
                *count += 1;
                pairs.push((begin, end));
            }
            None => self.sections.push((name, 1, vec![(begin, end)])),
        }
        Ok(())
    }

    fn report(&mut self, ctx: &DeviceContext, rows: usize) -> Result<()> {
        if !self.enabled || self.sections.is_empty() {
            self.sections.clear();
            return Ok(());
        }
        ctx.stream.synchronize()?;
        let mut lines = Vec::with_capacity(self.sections.len());
        let mut total = 0.0f64;
        for (name, count, pairs) in &self.sections {
            let mut ms = 0.0f64;
            for (begin, end) in pairs {
                ms += f64::from(begin.elapsed_ms(end)?);
            }
            total += ms;
            lines.push(format!("\"{name}\": ({count}, {ms:.3})"));
        }
        log::info!(
            "GLM5.2 TP4 prefill CUDA-event profile: device={}, rows={rows}, \
             section_total_ms={total:.3}, sections={{{}}}",
            ctx.device_ordinal,
            lines.join(", ")
        );
        self.sections.clear();
        Ok(())
    }
}

pub(crate) struct Glm52TpPrefillExecutor {
    layout: Glm52TpPrefillLayout,
    profiler: Glm52PrefillProfiler,
    // ---- chunk-scale buffers ----
    token_ids: CudaSlice<u32>,
    positions: CudaSlice<u32>,
    hidden: CudaSlice<bf16>,
    normed: CudaSlice<bf16>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    mla_front: Glm52MlaFront,
    ql_nope: CudaSlice<bf16>,
    ckv_fp8: CudaSlice<u8>,
    ckv_scales: CudaSlice<f32>,
    slot_mapping: CudaSlice<i64>,
    block_ids: CudaSlice<i32>,
    unpacked_kv: CudaSlice<bf16>,
    fp8_gemm: Glm52Fp8GemmScratch,
    attention_v: CudaSlice<bf16>,
    attention_partial: CudaSlice<bf16>,
    attention_reduced: CudaSlice<bf16>,
    mlp_out: CudaSlice<bf16>,
    // Cross-layer indexer carry at chunk scale: a full-indexer layer fills
    // it in attention-tile slices; shared layers reuse it.
    carry_slots: CudaSlice<i32>,
    carry_lens: CudaSlice<i32>,
    // ---- chunk-scale DSA indexer (unpaged MQA) ----
    indexer: Glm52IndexerPrefillScratch,
    // ---- attention sub-tile buffers ----
    query_bf16: CudaSlice<bf16>,
    attention_out: CudaSlice<bf16>,
    attention_max: CudaSlice<f32>,
    attention_lse: CudaSlice<f32>,
    attention_v_sub: CudaSlice<bf16>,
    // ---- dense-MLP sub-tile buffers ----
    dense_gate_up: CudaSlice<bf16>,
    dense_silu: CudaSlice<bf16>,
    dense_out_sub: CudaSlice<bf16>,
    dense_gemm: Glm52Fp8GemmScratch,
    // ---- MoE (chunk-scale) ----
    moe: Glm52MoeTpPrefillScratch,
    // ---- output tail (fixed 32-row blocks) ----
    output_rows: CudaSlice<i32>,
    final_hidden: Rows<GLM52_HIDDEN>,
    final_normed: Rows<GLM52_HIDDEN>,
    logits: Rows<GLM52_VOCAB>,
    argmax_partial_values: CudaSlice<f32>,
    argmax_partial_indices: CudaSlice<i32>,
    argmax_values: CudaSlice<bf16>,
    argmax_indices: CudaSlice<i32>,
}

pub(crate) struct Glm52TpPrefillModelView<'a> {
    pub(crate) layers: &'a [Glm52DecoderLayerWeights],
    pub(crate) caches: &'a mut [Glm52LayerCaches],
    pub(crate) embed: &'a DeviceMatrix,
    pub(crate) cos_table: &'a DeviceMatrix,
    pub(crate) sin_table: &'a DeviceMatrix,
    pub(crate) final_norm: &'a DeviceVec,
    pub(crate) shard_lm_head: &'a DeviceMatrix,
    pub(crate) full_lm_head: &'a DeviceMatrix,
    pub(crate) vocab_start: usize,
    pub(crate) sampling_scratch: &'a mut BatchSamplingScratch,
}

impl Glm52TpPrefillExecutor {
    pub(crate) fn new(
        ctx: &DeviceContext,
        kv_slots: usize,
        table_width: usize,
        index_cache_layout: Glm52IndexerCacheLayout,
        chunk_rows: usize,
        topology: openinfer_kernels::ops::Glm52TpTopology,
    ) -> Result<Self> {
        let layout = Glm52TpPrefillLayout::new(kv_slots, table_width, chunk_rows)?;
        let chunk = layout.chunk_rows;
        let attn = PREFILL_ATTN_TILE_ROWS;
        let dense = PREFILL_DENSE_TILE_ROWS.min(chunk);
        Ok(Self {
            profiler: Glm52PrefillProfiler::new(),
            token_ids: ctx.stream.alloc_zeros::<u32>(chunk)?,
            positions: ctx.stream.alloc_zeros::<u32>(chunk)?,
            hidden: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_HIDDEN)?,
            normed: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_HIDDEN)?,
            cos: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_ROPE_HALF)?,
            sin: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_ROPE_HALF)?,
            mla_front: Glm52MlaFront::new_prefill(ctx, chunk, 16)?,
            ql_nope: ctx
                .stream
                .alloc_zeros::<bf16>(chunk * 16 * GLM52_KV_LORA_RANK)?,
            ckv_fp8: ctx.stream.alloc_zeros::<u8>(chunk * GLM52_KV_LORA_RANK)?,
            ckv_scales: ctx.stream.alloc_zeros::<f32>(chunk * 4)?,
            slot_mapping: ctx.stream.alloc_zeros::<i64>(chunk)?,
            block_ids: ctx
                .stream
                .alloc_zeros::<i32>(layout.kv_slots.div_ceil(64))?,
            unpacked_kv: ctx
                .stream
                .alloc_zeros::<bf16>(layout.kv_slots * GLM52_KV_A_OUT)?,
            fp8_gemm: Glm52Fp8GemmScratch::new(ctx, chunk, GLM52_HIDDEN)?,
            attention_v: ctx.stream.alloc_zeros::<bf16>(chunk * 16 * 256)?,
            attention_partial: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_HIDDEN)?,
            attention_reduced: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_HIDDEN)?,
            mlp_out: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_HIDDEN)?,
            carry_slots: ctx.stream.alloc_zeros::<i32>(chunk * GLM52_INDEXER_TOPK)?,
            carry_lens: ctx.stream.alloc_zeros::<i32>(chunk)?,
            indexer: Glm52IndexerPrefillScratch::new(
                ctx,
                chunk,
                PREFILL_ATTN_TILE_ROWS,
                kv_slots,
                table_width,
                index_cache_layout,
            )?,
            query_bf16: ctx.stream.alloc_zeros::<bf16>(attn * 64 * GLM52_KV_A_OUT)?,
            attention_out: ctx
                .stream
                .alloc_zeros::<bf16>(attn * 64 * GLM52_KV_LORA_RANK)?,
            attention_max: ctx.stream.alloc_zeros::<f32>(attn * 64)?,
            attention_lse: ctx.stream.alloc_zeros::<f32>(attn * 64)?,
            attention_v_sub: ctx.stream.alloc_zeros::<bf16>(attn * 16 * 256)?,
            dense_gate_up: ctx
                .stream
                .alloc_zeros::<bf16>(dense * 2 * crate::config::GLM52_DENSE_INTERMEDIATE)?,
            dense_silu: ctx
                .stream
                .alloc_zeros::<bf16>(dense * crate::config::GLM52_DENSE_INTERMEDIATE)?,
            dense_out_sub: ctx.stream.alloc_zeros::<bf16>(dense * GLM52_HIDDEN)?,
            dense_gemm: Glm52Fp8GemmScratch::new(
                ctx,
                dense,
                crate::config::GLM52_DENSE_INTERMEDIATE,
            )?,
            moe: Glm52MoeTpPrefillScratch::new(ctx, topology, chunk)?,
            output_rows: ctx.stream.alloc_zeros(32)?,
            final_hidden: Rows::zeros(ctx, 32)?,
            final_normed: Rows::zeros(ctx, 32)?,
            logits: Rows::zeros(ctx, 32)?,
            argmax_partial_values: ctx
                .stream
                .alloc_zeros(argmax_batch_bf16_split_partials_len(32, GLM52_VOCAB))?,
            argmax_partial_indices: ctx
                .stream
                .alloc_zeros(argmax_batch_bf16_split_partials_len(32, GLM52_VOCAB))?,
            argmax_values: ctx.stream.alloc_zeros(32)?,
            argmax_indices: ctx.stream.alloc_zeros(32)?,
            layout,
        })
    }

    /// Run the complete TP4 prefill forward for one coordinator batch,
    /// layer-outer: each of the 78 layers processes the whole chunk before
    /// the next layer starts. Returns tokens only for request boundary rows.
    pub(crate) fn forward(
        &mut self,
        ctx: &DeviceContext,
        batch: &Glm52PrefillBatch,
        tp: &mut Glm52MoeTpRank,
        model: Glm52TpPrefillModelView<'_>,
    ) -> Result<Vec<u32>> {
        ensure!(
            model.layers.len() == model.caches.len() && !model.layers.is_empty(),
            "GLM5.2 TP prefill layer/cache layout is invalid"
        );
        let rows = batch.token_ids.len();
        ensure!(
            rows > 0 && rows <= self.layout.chunk_rows,
            "GLM5.2 TP prefill batch of {rows} rows exceeds the chunk capacity {}",
            self.layout.chunk_rows
        );
        let rows4 = rows.next_multiple_of(4);

        let mark = self.profiler.start(ctx)?;
        self.stage_chunk(ctx, batch, model.embed, model.cos_table, model.sin_table)?;
        self.indexer.stage_chunk(ctx, batch)?;
        self.profiler.stop(ctx, "embedding_rope_stage", mark)?;

        let mark = self.profiler.start(ctx)?;
        rms_norm_rows_into(
            ctx,
            &self.hidden,
            &model.layers[0].input_ln,
            GLM52_RMS_EPS,
            GLM52_HIDDEN,
            rows4,
            &mut self.normed,
        )?;
        self.profiler.stop(ctx, "input_norm", mark)?;

        let mut carry_ready = false;
        for layer in 0..model.layers.len() {
            let weights = &model.layers[layer];
            let cache = &mut model.caches[layer];

            let mark = self.profiler.start(ctx)?;
            glm52_mla_prefill_front_into(
                ctx,
                &weights.mla,
                rows4,
                &self.normed,
                &mut self.fp8_gemm,
                &mut self.mla_front,
            )?;
            self.profiler.stop(ctx, "mla_front", mark)?;

            let mark = self.profiler.start(ctx)?;
            self.pack_mla_cache(ctx, &weights.mla, &mut cache.mla_cache, rows)?;
            self.profiler.stop(ctx, "mla_pack_cache", mark)?;

            if !batch.block_ids.is_empty() {
                let mark = self.profiler.start(ctx)?;
                glm52_prefill_unpack_pages_launch(
                    ctx,
                    &cache.mla_cache,
                    &self.block_ids,
                    batch.block_ids.len(),
                    &mut self.unpacked_kv,
                )?;
                self.profiler.stop(ctx, "kv_page_unpack", mark)?;
            }

            match &weights.indexer {
                Glm52LayerIndexer::Full(indexer) => {
                    let mark = self.profiler.start(ctx)?;
                    let index_k_cache = cache
                        .index_k_cache
                        .as_mut()
                        .context("GLM5.2 full prefill indexer is missing its cache")?;
                    self.indexer.run_layer(
                        ctx,
                        indexer,
                        &self.normed,
                        self.mla_front.q_resid.data(),
                        &self.cos,
                        &self.sin,
                        index_k_cache,
                        &self.slot_mapping,
                        rows,
                        &mut self.fp8_gemm,
                        &mut self.carry_slots,
                        &mut self.carry_lens,
                    )?;
                    carry_ready = true;
                    self.profiler.stop(ctx, "indexer_full", mark)?;
                }
                Glm52LayerIndexer::Shared => {
                    ensure!(
                        carry_ready,
                        "GLM5.2 shared prefill indexer has no top-k carry"
                    );
                }
            }

            let mark = self.profiler.start(ctx)?;
            self.attend_chunk(ctx, &weights.mla, rows)?;
            self.profiler.stop(ctx, "sparse_attention", mark)?;

            let mark = self.profiler.start(ctx)?;
            fp8_linear_large_m_into(
                ctx,
                &weights.mla.o_proj,
                rows4,
                &self.attention_v,
                &mut self.fp8_gemm,
                &mut self.attention_partial,
            )?;
            self.profiler.stop(ctx, "o_proj", mark)?;

            let mark = self.profiler.start(ctx)?;
            self.reduce_and_norm_attention(ctx, &mut tp.state, &weights.post_attn_ln, rows)?;
            self.profiler.stop(ctx, "attention_out_reduce_norm", mark)?;

            match &weights.mlp {
                Glm52LayerMlp::Dense(dense) => {
                    let mark = self.profiler.start(ctx)?;
                    self.dense_mlp(ctx, dense, rows4)?;
                    self.profiler.stop(ctx, "dense_mlp", mark)?;
                }
                Glm52LayerMlp::MoeTp(router) => {
                    let (state, _, bank) = tp.layer_bank(layer).with_context(|| {
                        format!("GLM5.2 TP prefill layer {layer} has no expert slice bank")
                    })?;
                    let mark = self.profiler.start(ctx)?;
                    self.moe.forward(
                        ctx,
                        state,
                        router,
                        bank,
                        &self.normed,
                        rows,
                        &mut self.mlp_out,
                    )?;
                    self.profiler.stop(ctx, "moe_mlp", mark)?;
                    let mark = self.profiler.start(ctx)?;
                    state.prefill_allreduce_in_place(ctx, rows, &mut self.mlp_out)?;
                    self.profiler.stop(ctx, "moe_reduce", mark)?;
                }
                Glm52LayerMlp::MoeEp8(_) => {
                    anyhow::bail!("GLM5.2 TP prefill layer {layer} has EP weights");
                }
            }

            let mark = self.profiler.start(ctx)?;
            self.finish_layer(
                ctx,
                model.layers.get(layer + 1).map(|next| &next.input_ln),
                rows,
            )?;
            self.profiler.stop(ctx, "residual_next_norm", mark)?;
        }

        let mut outputs = Vec::with_capacity(batch.output_rows.len());
        let local_outputs: Vec<i32> = batch.output_rows.iter().map(|&row| row as i32).collect();
        let mark = self.profiler.start(ctx)?;
        for rows_block in local_outputs.chunks(32) {
            let output_base = outputs.len();
            let sampling: Vec<_> = batch
                .sampling
                .iter()
                .filter(|sample| {
                    (output_base..output_base + rows_block.len()).contains(&sample.row)
                })
                .map(|sample| {
                    let mut sample = *sample;
                    sample.row -= output_base;
                    sample
                })
                .collect();
            outputs.extend(self.output_tokens(
                ctx,
                &mut tp.state,
                model.final_norm,
                model.shard_lm_head,
                model.full_lm_head,
                model.vocab_start,
                rows_block,
                &sampling,
                batch.seed,
                model.sampling_scratch,
            )?);
        }
        self.profiler.stop(ctx, "lm_head_sampling", mark)?;
        self.profiler.report(ctx, rows)?;
        Ok(outputs)
    }

    /// Upload token ids/positions/slot mapping/block list and stage
    /// embeddings + rope rows for the whole chunk.
    fn stage_chunk(
        &mut self,
        ctx: &DeviceContext,
        batch: &Glm52PrefillBatch,
        embed: &DeviceMatrix,
        cos_table: &DeviceMatrix,
        sin_table: &DeviceMatrix,
    ) -> Result<()> {
        let rows = batch.token_ids.len();
        ensure!(
            batch.positions.len() == rows && batch.slot_mapping.len() == rows,
            "prefill chunk rows/positions mismatch"
        );
        ctx.stream
            .memcpy_htod(&batch.token_ids, &mut self.token_ids.slice_mut(..rows))?;
        ctx.stream
            .memcpy_htod(&batch.positions, &mut self.positions.slice_mut(..rows))?;
        ctx.stream.memcpy_htod(
            &batch.slot_mapping,
            &mut self.slot_mapping.slice_mut(..rows),
        )?;
        if !batch.block_ids.is_empty() {
            ensure!(
                batch.block_ids.len() <= self.block_ids.len(),
                "prefill block list exceeds scratch capacity"
            );
            ctx.stream.memcpy_htod(
                &batch.block_ids,
                &mut self.block_ids.slice_mut(..batch.block_ids.len()),
            )?;
        }
        embedding_rows_into(ctx, embed, &self.token_ids, rows, &mut self.hidden)?;
        let rows4 = rows.next_multiple_of(4);
        if rows4 > rows {
            ctx.stream.memset_zeros(
                &mut self
                    .hidden
                    .slice_mut(rows * GLM52_HIDDEN..rows4 * GLM52_HIDDEN),
            )?;
        }
        embedding_rows_into(ctx, cos_table, &self.positions, rows, &mut self.cos)?;
        embedding_rows_into(ctx, sin_table, &self.positions, rows, &mut self.sin)?;
        Ok(())
    }

    /// Per-layer chunk-scale MLA pack: the w_uk absorb bmm plus the fused
    /// canonical fp8_ds_mla pack that writes this layer's 656-byte KV rows at
    /// `slot_mapping`. The bf16 attention query is assembled later, per
    /// attention sub-tile.
    fn pack_mla_cache(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52MlaLayerWeights,
        packed_cache: &mut CudaSlice<u8>,
        rows: usize,
    ) -> Result<()> {
        gemm_strided_batched_bf16(
            ctx,
            false,
            false,
            GLM52_KV_LORA_RANK,
            rows,
            GLM52_QK_NOPE_HEAD_DIM,
            &weights.w_uk,
            GLM52_KV_LORA_RANK,
            GLM52_QK_NOPE_HEAD_DIM * GLM52_KV_LORA_RANK,
            &self.mla_front.q_full,
            16 * GLM52_QK_HEAD_DIM,
            GLM52_QK_HEAD_DIM,
            &mut self.ql_nope,
            16 * GLM52_KV_LORA_RANK,
            GLM52_KV_LORA_RANK,
            16,
        )?;
        glm52_fp8_per_token_group_quant_bf16_ue8m0_launch(
            ctx,
            Glm52MoeQuantShape {
                rows,
                width: GLM52_KV_LORA_RANK,
                group_size: 128,
            },
            &self.mla_front.kv_c,
            &mut self.ckv_fp8,
            &mut self.ckv_scales,
        )?;
        glm52_mla_cache_pack_launch(
            ctx,
            rows,
            &self.ckv_fp8,
            &self.ckv_scales,
            &self.mla_front.k_pe,
            &self.cos,
            &self.sin,
            packed_cache,
            &self.slot_mapping,
        )
    }

    /// Sparse attention over the chunk in `PREFILL_ATTN_TILE_ROWS` sub-tiles:
    /// query assembly (bf16), FlashMLA sparse prefill against the unpacked
    /// KV pool with the carried top-k slots, and the w_uv value bmm into the
    /// chunk-scale `attention_v`.
    fn attend_chunk(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52MlaLayerWeights,
        rows: usize,
    ) -> Result<()> {
        let mut sub = 0usize;
        while sub < rows {
            let t = (rows - sub).min(PREFILL_ATTN_TILE_ROWS);
            glm52_mla_query_assemble_launch(
                ctx,
                t,
                16,
                &self.ql_nope.slice(sub * 16 * GLM52_KV_LORA_RANK..),
                &self.mla_front.q_full.slice(sub * 16 * GLM52_QK_HEAD_DIM..),
                GLM52_QK_NOPE_HEAD_DIM,
                GLM52_QK_HEAD_DIM,
                &self.cos.slice(sub * GLM52_ROPE_HALF..),
                &self.sin.slice(sub * GLM52_ROPE_HALF..),
                &mut self.query_bf16,
            )?;
            let carry = self.carry_slots.slice(sub * GLM52_INDEXER_TOPK..);
            let lens = self.carry_lens.slice(sub..);
            glm52_flashmla_sparse_prefill_launch(
                ctx,
                t,
                self.layout.kv_slots,
                GLM52_INDEXER_TOPK,
                0.0625,
                &self.query_bf16,
                &self.unpacked_kv,
                &carry,
                Some(&lens),
                &mut self.attention_out,
                &mut self.attention_max,
                &mut self.attention_lse,
            )?;
            // cuBLAS is column-major: token columns advance by `16 * 256`,
            // while each head batch starts 256 elements later. The resulting
            // address is `[token][head][value]`, matching `o_proj`'s
            // row-major input.
            gemm_strided_batched_bf16(
                ctx,
                true,
                false,
                256,
                t,
                GLM52_KV_LORA_RANK,
                &weights.w_uv,
                GLM52_KV_LORA_RANK,
                256 * GLM52_KV_LORA_RANK,
                &self.attention_out,
                64 * GLM52_KV_LORA_RANK,
                GLM52_KV_LORA_RANK,
                &mut self.attention_v_sub,
                16 * 256,
                256,
                16,
            )?;
            ctx.stream.memcpy_dtod(
                &self.attention_v_sub.slice(..t * 16 * 256),
                &mut self
                    .attention_v
                    .slice_mut(sub * 16 * 256..(sub + t) * 16 * 256),
            )?;
            sub += t;
        }
        let rows4 = rows.next_multiple_of(4);
        if rows4 > rows {
            ctx.stream.memset_zeros(
                &mut self
                    .attention_v
                    .slice_mut(rows * 16 * 256..rows4 * 16 * 256),
            )?;
        }
        Ok(())
    }

    fn dense_mlp(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52DenseMlpWeights,
        rows4: usize,
    ) -> Result<()> {
        let mut sub = 0usize;
        while sub < rows4 {
            let t = (rows4 - sub).min(PREFILL_DENSE_TILE_ROWS);
            let t4 = t.next_multiple_of(4);
            glm52_dense_mlp_prefill_into(
                ctx,
                weights,
                t4,
                &self.normed.slice(sub * GLM52_HIDDEN..),
                &mut self.dense_gemm,
                &mut self.dense_gate_up,
                &mut self.dense_silu,
                &mut self.dense_out_sub,
            )?;
            ctx.stream.memcpy_dtod(
                &self.dense_out_sub.slice(..t * GLM52_HIDDEN),
                &mut self
                    .mlp_out
                    .slice_mut(sub * GLM52_HIDDEN..(sub + t) * GLM52_HIDDEN),
            )?;
            sub += t;
        }
        Ok(())
    }

    fn reduce_and_norm_attention(
        &mut self,
        ctx: &DeviceContext,
        tp: &mut Glm52MoeTpState,
        post_attn_ln: &DeviceVec,
        rows: usize,
    ) -> Result<()> {
        tp.prefill_allreduce(
            ctx,
            rows,
            &self.attention_partial,
            &mut self.attention_reduced,
        )?;
        fused_add_rms_norm_round_into(
            ctx,
            &mut self.attention_reduced,
            &self.hidden,
            post_attn_ln,
            GLM52_RMS_EPS,
            GLM52_HIDDEN,
            rows,
            &mut self.normed,
        )?;
        let rows4 = rows.next_multiple_of(4);
        if rows4 > rows {
            ctx.stream.memset_zeros(
                &mut self
                    .normed
                    .slice_mut(rows * GLM52_HIDDEN..rows4 * GLM52_HIDDEN),
            )?;
        }
        Ok(())
    }

    fn finish_layer(
        &mut self,
        ctx: &DeviceContext,
        next_input_ln: Option<&DeviceVec>,
        rows: usize,
    ) -> Result<()> {
        match next_input_ln {
            Some(weight) => {
                fused_add_rms_norm_round_into(
                    ctx,
                    &mut self.attention_reduced,
                    &self.mlp_out,
                    weight,
                    GLM52_RMS_EPS,
                    GLM52_HIDDEN,
                    rows,
                    &mut self.normed,
                )?;
                ctx.stream.memcpy_dtod(
                    &self.attention_reduced.slice(..rows * GLM52_HIDDEN),
                    &mut self.hidden.slice_mut(..rows * GLM52_HIDDEN),
                )?;
                let rows4 = rows.next_multiple_of(4);
                if rows4 > rows {
                    ctx.stream.memset_zeros(
                        &mut self
                            .normed
                            .slice_mut(rows * GLM52_HIDDEN..rows4 * GLM52_HIDDEN),
                    )?;
                }
            }
            None => {
                add_into(
                    ctx,
                    &self.attention_reduced,
                    &self.mlp_out,
                    rows * GLM52_HIDDEN,
                    &mut self.hidden,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn output_tokens(
        &mut self,
        ctx: &DeviceContext,
        tp: &mut Glm52MoeTpState,
        final_norm: &DeviceVec,
        shard_lm_head: &DeviceMatrix,
        full_lm_head: &DeviceMatrix,
        vocab_start: usize,
        rows: &[i32],
        sampling: &[crate::runner::Glm52RowSample],
        seed: u64,
        sampling_scratch: &mut BatchSamplingScratch,
    ) -> Result<Vec<u32>> {
        ensure!(
            !rows.is_empty() && rows.len() <= 32 && rows.iter().all(|&row| row >= 0),
            "GLM5.2 prefill output row set is invalid"
        );
        ctx.stream
            .memcpy_htod(rows, &mut self.output_rows.slice_mut(..rows.len()))?;
        glm52_prefill_moe_gather_rows_launch(
            ctx,
            rows.len(),
            GLM52_HIDDEN,
            &self.hidden,
            &self.output_rows,
            self.final_hidden.data_mut(),
        )?;
        if rows.len() < 32 {
            ctx.stream.memset_zeros(
                &mut self
                    .final_hidden
                    .data_mut()
                    .slice_mut(rows.len() * GLM52_HIDDEN..),
            )?;
        }
        glm52_final_norm_into(ctx, &self.final_hidden, final_norm, &mut self.final_normed)?;
        glm52_lm_head_into(ctx, &self.final_normed, shard_lm_head, &mut self.logits)?;
        argmax_bf16_split_into(
            ctx,
            self.logits.data(),
            32,
            shard_lm_head.rows,
            &mut self.argmax_partial_values,
            &mut self.argmax_partial_indices,
            &mut self.argmax_values,
            &mut self.argmax_indices,
        )?;
        glm52_vocab_parallel_pack_launch(
            ctx,
            &self.argmax_values,
            &self.argmax_indices,
            &mut self.attention_partial,
            32,
            tp.rank(),
            vocab_start,
        )?;
        tp.prefill_allreduce(
            ctx,
            32,
            &self.attention_partial,
            &mut self.attention_reduced,
        )?;
        glm52_vocab_parallel_unpack_launch(
            ctx,
            &self.attention_reduced,
            &mut self.argmax_values,
            &mut self.argmax_indices,
            32,
            tp.ranks(),
        )?;
        let mut host = vec![0i32; 32];
        ctx.stream.memcpy_dtoh(&self.argmax_indices, &mut host)?;
        ctx.stream.synchronize()?;
        let mut outputs = host
            .into_iter()
            .take(rows.len())
            .map(|token| {
                ensure!(
                    (0..GLM52_VOCAB as i32).contains(&token),
                    "GLM5.2 prefill argmax token {token} is invalid"
                );
                Ok(token as u32)
            })
            .collect::<Result<Vec<_>>>()?;
        if sampling.is_empty() {
            return Ok(outputs);
        }

        glm52_lm_head_into(ctx, &self.final_normed, full_lm_head, &mut self.logits)?;
        let logits = HiddenStatesRef {
            data: self.logits.data(),
            hidden_dim: GLM52_VOCAB,
            seq_len: 32,
        };
        let as_row = |sample: &crate::runner::Glm52RowSample| BatchSamplingRow {
            row: sample.row,
            temperature: sample.params.temperature,
            top_k: sample.params.top_k,
            top_p: sample.params.top_p,
            min_p: sample.params.min_p,
        };
        let unseeded: Vec<_> = sampling
            .iter()
            .filter(|sample| sample.params.seed.is_none())
            .map(as_row)
            .collect();
        if !unseeded.is_empty() {
            let tokens = gpu_sample_batch_into(ctx, logits, &unseeded, seed, sampling_scratch)?;
            for (row, token) in unseeded.iter().zip(tokens) {
                outputs[row.row] = token;
            }
        }
        for sample in sampling {
            let Some(request_seed) = sample.params.seed else {
                continue;
            };
            let tokens = gpu_sample_batch_into(
                ctx,
                logits,
                &[as_row(sample)],
                mix_seed(request_seed, sample.step),
                sampling_scratch,
            )?;
            outputs[sample.row] = tokens[0];
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a GPU"]
    fn w_uv_multirow_output_is_token_major() -> Result<()> {
        const ROWS: usize = 3;
        const HEADS: usize = 16;
        const SOURCE_HEADS: usize = 64;
        const K: usize = 512;
        const V: usize = 256;

        let ctx = DeviceContext::new_with_device(0)?;
        let mut weights = vec![bf16::ZERO; HEADS * V * K];
        for head in 0..HEADS {
            for value in 0..V {
                weights[head * V * K + value * K + value] = bf16::ONE;
            }
        }
        let mut latent = vec![bf16::ZERO; ROWS * SOURCE_HEADS * K];
        for token in 0..ROWS {
            for head in 0..HEADS {
                for value in 0..V {
                    latent[token * SOURCE_HEADS * K + head * K + value] =
                        bf16::from_f32((token * 64 + head * 2 + value % 2) as f32);
                }
            }
        }
        let weights = ctx.stream.clone_htod(&weights)?;
        let latent = ctx.stream.clone_htod(&latent)?;
        let mut output = ctx.stream.alloc_zeros::<bf16>(ROWS * HEADS * V)?;
        gemm_strided_batched_bf16(
            &ctx,
            true,
            false,
            V,
            ROWS,
            K,
            &weights,
            K,
            V * K,
            &latent,
            SOURCE_HEADS * K,
            K,
            &mut output,
            HEADS * V,
            V,
            HEADS,
        )?;
        let output = ctx.stream.clone_dtoh(&output)?;
        for token in 0..ROWS {
            for head in 0..HEADS {
                for value in 0..V {
                    let offset = token * HEADS * V + head * V + value;
                    let expected = (token * 64 + head * 2 + value % 2) as f32;
                    assert_eq!(output[offset].to_f32(), expected);
                }
            }
        }
        Ok(())
    }
}
