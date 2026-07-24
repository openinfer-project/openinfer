use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::Glm52DeepGemmMqaLogitsShape;
use openinfer_kernels::ops::add_into;
use openinfer_kernels::ops::argmax_batch_bf16_split_partials_len;
use openinfer_kernels::ops::argmax_bf16_split_into;
use openinfer_kernels::ops::embedding_rows_into;
use openinfer_kernels::ops::fused_add_rms_norm_round_into;
use openinfer_kernels::ops::gemm_strided_batched_bf16;
use openinfer_kernels::ops::glm52_flashmla_sparse_prefill_launch;
use openinfer_kernels::ops::glm52_mla_front_pack_fp8_launch;
use openinfer_kernels::ops::glm52_mla_query_assemble_launch;
use openinfer_kernels::ops::glm52_prefill_moe_gather_launch;
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
use crate::config::GLM52_Q_LORA_RANK;
use crate::config::GLM52_QK_HEAD_DIM;
use crate::config::GLM52_QK_NOPE_HEAD_DIM;
use crate::config::GLM52_RMS_EPS;
use crate::config::GLM52_ROPE_HALF;
use crate::config::GLM52_VOCAB;
use crate::dense::Glm52DenseMlpWeights;
use crate::dense::glm52_dense_mlp_prefill_into;
use crate::fp8::Glm52Fp8GemmScratch;
use crate::fp8::fp8_linear_large_m_into;
use crate::indexer::Glm52IndexerLayerWeights;
use crate::indexer::Glm52IndexerScratch;
use crate::indexer::glm52_indexer_forward_into;
use crate::layer::Glm52DecoderLayerWeights;
use crate::layer::Glm52LayerCaches;
use crate::layer::Glm52LayerIndexer;
use crate::layer::Glm52LayerMlp;
use crate::mla_front::Glm52MlaFront;
use crate::mla_front::Glm52MlaLayerWeights;
use crate::mla_front::glm52_mla_prefill_front_into;
use crate::moe_decode::Glm52MoeRouterWeights;
use crate::moe_tp::Glm52MoeTpPrefillScratch;
use crate::moe_tp::Glm52MoeTpRank;
use crate::moe_tp::Glm52MoeTpSliceBank;
use crate::moe_tp::Glm52MoeTpState;
use crate::rows::Rows;
use crate::runner::Glm52PrefillBatch;

pub(crate) const PREFILL_TILE_ROWS: usize = 32;
const INDEXER_TILE: usize = PREFILL_TILE_ROWS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Glm52TpPrefillLayout {
    kv_slots: usize,
    table_width: usize,
}

impl Glm52TpPrefillLayout {
    fn new(kv_slots: usize, table_width: usize) -> Result<Self> {
        ensure!(
            kv_slots > 0 && table_width > 0,
            "prefill capacities must be positive"
        );
        Ok(Self {
            kv_slots,
            table_width,
        })
    }
}

pub(crate) struct Glm52TpPrefillExecutor {
    layout: Glm52TpPrefillLayout,
    token_ids: CudaSlice<u32>,
    positions: CudaSlice<u32>,
    hidden: CudaSlice<bf16>,
    normed: CudaSlice<bf16>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    mla_front: Glm52MlaFront,
    ql_nope: CudaSlice<bf16>,
    query_bf16: CudaSlice<bf16>,
    query_fp8: CudaSlice<u8>,
    slot_mapping: CudaSlice<i64>,
    block_ids: CudaSlice<i32>,
    unpacked_kv: CudaSlice<bf16>,
    fp8_gemm: Glm52Fp8GemmScratch,
    indexer: Glm52IndexerScratch,
    tile_hidden: Rows<GLM52_HIDDEN>,
    tile_q_resid: Rows<GLM52_Q_LORA_RANK>,
    tile_cos: CudaSlice<bf16>,
    tile_sin: CudaSlice<bf16>,
    tile_slots: CudaSlice<i64>,
    tile_table: CudaSlice<i32>,
    tile_lens: CudaSlice<i32>,
    attention_out: CudaSlice<bf16>,
    attention_max: CudaSlice<f32>,
    attention_lse: CudaSlice<f32>,
    attention_v: CudaSlice<bf16>,
    attention_partial: CudaSlice<bf16>,
    attention_reduced: CudaSlice<bf16>,
    dense_gate_up: CudaSlice<bf16>,
    dense_silu: CudaSlice<bf16>,
    dense_out: CudaSlice<bf16>,
    moe: Glm52MoeTpPrefillScratch,
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
        indexer_shape: Glm52DeepGemmMqaLogitsShape,
    ) -> Result<Self> {
        let layout = Glm52TpPrefillLayout::new(kv_slots, table_width)?;
        ensure!(
            indexer_shape.batch_size == INDEXER_TILE
                && indexer_shape.block_table_stride == layout.table_width,
            "prefill indexer scratch/layout mismatch"
        );
        let work_rows = INDEXER_TILE;
        Ok(Self {
            layout,
            token_ids: ctx.stream.alloc_zeros::<u32>(work_rows)?,
            positions: ctx.stream.alloc_zeros::<u32>(work_rows)?,
            hidden: ctx.stream.alloc_zeros::<bf16>(work_rows * GLM52_HIDDEN)?,
            normed: ctx.stream.alloc_zeros::<bf16>(work_rows * GLM52_HIDDEN)?,
            cos: ctx
                .stream
                .alloc_zeros::<bf16>(work_rows * GLM52_ROPE_HALF)?,
            sin: ctx
                .stream
                .alloc_zeros::<bf16>(work_rows * GLM52_ROPE_HALF)?,
            mla_front: Glm52MlaFront::new_prefill(ctx, work_rows, 16)?,
            ql_nope: ctx
                .stream
                .alloc_zeros::<bf16>(work_rows * 16 * GLM52_KV_LORA_RANK)?,
            query_bf16: ctx
                .stream
                .alloc_zeros::<bf16>(work_rows * 64 * GLM52_KV_A_OUT)?,
            query_fp8: ctx
                .stream
                .alloc_zeros::<u8>(work_rows * 16 * GLM52_KV_A_OUT)?,
            slot_mapping: ctx.stream.alloc_zeros::<i64>(work_rows)?,
            block_ids: ctx
                .stream
                .alloc_zeros::<i32>(layout.kv_slots.div_ceil(64))?,
            unpacked_kv: ctx
                .stream
                .alloc_zeros::<bf16>(layout.kv_slots * GLM52_KV_A_OUT)?,
            fp8_gemm: Glm52Fp8GemmScratch::new(
                ctx,
                work_rows,
                crate::config::GLM52_DENSE_INTERMEDIATE,
            )?,
            indexer: Glm52IndexerScratch::new_prefill(ctx, indexer_shape)?,
            tile_hidden: Rows::zeros(ctx, INDEXER_TILE)?,
            tile_q_resid: Rows::zeros(ctx, INDEXER_TILE)?,
            tile_cos: ctx
                .stream
                .alloc_zeros::<bf16>(INDEXER_TILE * GLM52_ROPE_HALF)?,
            tile_sin: ctx
                .stream
                .alloc_zeros::<bf16>(INDEXER_TILE * GLM52_ROPE_HALF)?,
            tile_slots: ctx.stream.alloc_zeros::<i64>(INDEXER_TILE)?,
            tile_table: ctx
                .stream
                .alloc_zeros::<i32>(INDEXER_TILE * layout.table_width)?,
            tile_lens: ctx.stream.alloc_zeros::<i32>(INDEXER_TILE)?,
            attention_out: ctx
                .stream
                .alloc_zeros::<bf16>(INDEXER_TILE * 64 * GLM52_KV_LORA_RANK)?,
            attention_max: ctx.stream.alloc_zeros::<f32>(INDEXER_TILE * 64)?,
            attention_lse: ctx.stream.alloc_zeros::<f32>(INDEXER_TILE * 64)?,
            attention_v: ctx.stream.alloc_zeros::<bf16>(INDEXER_TILE * 16 * 256)?,
            attention_partial: ctx
                .stream
                .alloc_zeros::<bf16>(INDEXER_TILE * GLM52_HIDDEN)?,
            attention_reduced: ctx
                .stream
                .alloc_zeros::<bf16>(INDEXER_TILE * GLM52_HIDDEN)?,
            dense_gate_up: ctx
                .stream
                .alloc_zeros::<bf16>(INDEXER_TILE * 2 * crate::config::GLM52_DENSE_INTERMEDIATE)?,
            dense_silu: ctx
                .stream
                .alloc_zeros::<bf16>(INDEXER_TILE * crate::config::GLM52_DENSE_INTERMEDIATE)?,
            dense_out: ctx
                .stream
                .alloc_zeros::<bf16>(INDEXER_TILE * GLM52_HIDDEN)?,
            moe: Glm52MoeTpPrefillScratch::new(ctx)?,
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
        })
    }

    /// Run the complete TP4 prefill forward for one coordinator batch.
    ///
    /// The batch is tiled into fixed-size kernel calls, writes MLA/indexer KV
    /// pages in place, runs all layers, and returns tokens only for request
    /// boundary rows. Returned tokens are never fed into decode.
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
        let mut outputs = Vec::with_capacity(batch.output_rows.len());
        for start in (0..batch.token_ids.len()).step_by(PREFILL_TILE_ROWS) {
            let active = (batch.token_ids.len() - start).min(PREFILL_TILE_ROWS);
            self.stage_chunk(
                ctx,
                &batch.token_ids[start..start + active],
                &batch.positions[start..start + active],
                model.embed,
                model.cos_table,
                model.sin_table,
            )?;
            self.norm_chunk(ctx, &model.layers[0].input_ln, active)?;
            let mut carry_ready = false;
            for layer in 0..model.layers.len() {
                let weights = &model.layers[layer];
                let cache = &mut model.caches[layer];
                self.mla_front_chunk(ctx, &weights.mla, active)?;
                self.assemble_and_pack_mla(
                    ctx,
                    &weights.mla,
                    &mut cache.mla_cache,
                    &batch.slot_mapping[start..start + active],
                )?;
                if !batch.block_ids.is_empty() {
                    self.unpack_kv_pages(ctx, &cache.mla_cache, &batch.block_ids)?;
                }
                match &weights.indexer {
                    Glm52LayerIndexer::Full(indexer) => {
                        let index_k_cache = cache
                            .index_k_cache
                            .as_mut()
                            .context("GLM5.2 full prefill indexer is missing its cache")?;
                        self.run_indexer(ctx, indexer, index_k_cache, batch, start, active)?;
                        carry_ready = true;
                    }
                    Glm52LayerIndexer::Shared => {
                        ensure!(
                            carry_ready,
                            "GLM5.2 shared prefill indexer has no top-k carry"
                        );
                    }
                }
                self.attend_partial(ctx, &weights.mla, active)?;
                self.reduce_and_norm_attention(ctx, &mut tp.state, &weights.post_attn_ln, active)?;
                match &weights.mlp {
                    Glm52LayerMlp::Dense(dense) => {
                        self.dense_mlp(ctx, dense, active.next_multiple_of(4))?;
                    }
                    Glm52LayerMlp::MoeTp(router) => {
                        let (state, _, bank) = tp.layer_bank(layer).with_context(|| {
                            format!("GLM5.2 TP prefill layer {layer} has no expert slice bank")
                        })?;
                        self.moe_mlp(ctx, state, router, bank, active)?;
                    }
                    Glm52LayerMlp::MoeEp8(_) => {
                        anyhow::bail!("GLM5.2 TP prefill layer {layer} has EP weights");
                    }
                }
                self.finish_dense_layer(
                    ctx,
                    model.layers.get(layer + 1).map(|next| &next.input_ln),
                    active,
                )?;
            }
            let local_outputs: Vec<i32> = batch
                .output_rows
                .iter()
                .copied()
                .filter(|&row| (start..start + active).contains(&(row as usize)))
                .map(|row| row as i32 - start as i32)
                .collect();
            for rows in local_outputs.chunks(32) {
                let output_base = outputs.len();
                let sampling: Vec<_> = batch
                    .sampling
                    .iter()
                    .filter(|sample| (output_base..output_base + rows.len()).contains(&sample.row))
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
                    rows,
                    &sampling,
                    batch.seed,
                    model.sampling_scratch,
                )?);
            }
        }
        Ok(outputs)
    }

    fn unpack_kv_pages(
        &mut self,
        ctx: &DeviceContext,
        packed: &CudaSlice<u8>,
        block_ids: &[i32],
    ) -> Result<()> {
        ensure!(
            !block_ids.is_empty() && block_ids.len() <= self.block_ids.len(),
            "prefill block list exceeds scratch capacity"
        );
        ctx.stream
            .memcpy_htod(block_ids, &mut self.block_ids.slice_mut(..block_ids.len()))?;
        glm52_prefill_unpack_pages_launch(
            ctx,
            packed,
            &self.block_ids,
            block_ids.len(),
            &mut self.unpacked_kv,
        )
    }

    fn stage_chunk(
        &mut self,
        ctx: &DeviceContext,
        token_ids: &[u32],
        positions: &[u32],
        embed: &DeviceMatrix,
        cos_table: &DeviceMatrix,
        sin_table: &DeviceMatrix,
    ) -> Result<()> {
        let rows = token_ids.len();
        ensure!(
            rows > 0 && rows <= INDEXER_TILE && positions.len() == rows,
            "prefill chunk rows/positions mismatch"
        );
        ctx.stream
            .memcpy_htod(token_ids, &mut self.token_ids.slice_mut(..rows))?;
        ctx.stream
            .memcpy_htod(positions, &mut self.positions.slice_mut(..rows))?;
        embedding_rows_into(ctx, embed, &self.token_ids, rows, &mut self.hidden)?;
        let gemm_rows = rows.next_multiple_of(4);
        if gemm_rows > rows {
            ctx.stream.memset_zeros(
                &mut self
                    .hidden
                    .slice_mut(rows * GLM52_HIDDEN..gemm_rows * GLM52_HIDDEN),
            )?;
        }
        embedding_rows_into(ctx, cos_table, &self.positions, rows, &mut self.cos)?;
        embedding_rows_into(ctx, sin_table, &self.positions, rows, &mut self.sin)?;
        Ok(())
    }

    fn norm_chunk(&mut self, ctx: &DeviceContext, weight: &DeviceVec, rows: usize) -> Result<()> {
        let rows = rows.next_multiple_of(4);
        ensure!(
            rows > 0 && rows <= INDEXER_TILE,
            "prefill norm rows {rows} exceed tile capacity {INDEXER_TILE}"
        );
        rms_norm_rows_into(
            ctx,
            &self.hidden,
            weight,
            GLM52_RMS_EPS,
            GLM52_HIDDEN,
            rows,
            &mut self.normed,
        )
    }

    fn mla_front_chunk(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52MlaLayerWeights,
        rows: usize,
    ) -> Result<()> {
        let rows = rows.next_multiple_of(4);
        glm52_mla_prefill_front_into(
            ctx,
            weights,
            rows,
            &self.normed,
            &mut self.fp8_gemm,
            &mut self.mla_front,
        )
    }

    fn assemble_and_pack_mla(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52MlaLayerWeights,
        packed_cache: &mut CudaSlice<u8>,
        slot_mapping: &[i64],
    ) -> Result<()> {
        let rows = slot_mapping.len();
        ensure!(
            rows > 0 && rows <= INDEXER_TILE,
            "prefill MLA rows exceed scratch capacity"
        );
        ctx.stream.memcpy_htod(
            slot_mapping,
            &mut self.slot_mapping.slice_mut(..slot_mapping.len()),
        )?;
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
        glm52_mla_query_assemble_launch(
            ctx,
            rows,
            16,
            &self.ql_nope,
            &self.mla_front.q_full,
            GLM52_QK_NOPE_HEAD_DIM,
            GLM52_QK_HEAD_DIM,
            &self.cos,
            &self.sin,
            &mut self.query_bf16,
        )?;
        glm52_mla_front_pack_fp8_launch(
            ctx,
            rows,
            16,
            &self.ql_nope,
            &self.mla_front.q_full,
            GLM52_QK_NOPE_HEAD_DIM,
            GLM52_QK_HEAD_DIM,
            &self.mla_front.ckv,
            &weights.kv_a_ln.data,
            GLM52_RMS_EPS,
            &self.cos,
            &self.sin,
            &mut self.query_fp8,
            packed_cache,
            &self.slot_mapping,
        )
    }

    fn run_indexer(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52IndexerLayerWeights,
        index_k_cache: &mut CudaSlice<u8>,
        batch: &Glm52PrefillBatch,
        start: usize,
        active: usize,
    ) -> Result<()> {
        ensure!(
            active > 0 && active <= INDEXER_TILE && start + active <= batch.token_ids.len(),
            "prefill indexer tile range is invalid"
        );
        ctx.stream
            .memcpy_dtod(&self.normed, self.tile_hidden.data_mut())?;
        ctx.stream
            .memcpy_dtod(self.mla_front.q_resid.data(), self.tile_q_resid.data_mut())?;
        ctx.stream.memcpy_dtod(&self.cos, &mut self.tile_cos)?;
        ctx.stream.memcpy_dtod(&self.sin, &mut self.tile_sin)?;

        let padding_page = batch.padding_block;
        let mut slots = vec![padding_page as i64 * 64; INDEXER_TILE];
        let mut lens = vec![1i32; INDEXER_TILE];
        let mut table = vec![padding_page; INDEXER_TILE * self.layout.table_width];
        let mut request = batch.request_indptr[1..].partition_point(|&end| end as usize <= start);
        for local in 0..active {
            let row = start + local;
            while row >= batch.request_indptr[request + 1] as usize {
                request += 1;
            }
            let block_start = batch.block_indptr[request] as usize;
            let block_end = batch.block_indptr[request + 1] as usize;
            let blocks = &batch.block_ids[block_start..block_end];
            table[local * self.layout.table_width..local * self.layout.table_width + blocks.len()]
                .copy_from_slice(blocks);
            slots[local] = batch.slot_mapping[row];
            lens[local] = batch.positions[row] as i32 + 1;
        }
        ctx.stream.memcpy_htod(&slots, &mut self.tile_slots)?;
        ctx.stream.memcpy_htod(&lens, &mut self.tile_lens)?;
        ctx.stream.memcpy_htod(&table, &mut self.tile_table)?;
        glm52_indexer_forward_into(
            ctx,
            weights,
            &self.tile_hidden,
            &self.tile_q_resid,
            &self.tile_cos,
            &self.tile_sin,
            index_k_cache,
            &self.tile_slots,
            &self.tile_table,
            &self.tile_lens,
            2048,
            &mut self.indexer,
        )
    }

    fn attend_partial(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52MlaLayerWeights,
        active: usize,
    ) -> Result<()> {
        ensure!(
            active > 0 && active <= INDEXER_TILE,
            "prefill attention tile is invalid"
        );
        glm52_flashmla_sparse_prefill_launch(
            ctx,
            active,
            self.layout.kv_slots,
            2048,
            0.0625,
            &self.query_bf16,
            &self.unpacked_kv,
            &self.indexer.global_slots,
            Some(&self.indexer.topk_lens),
            &mut self.attention_out,
            &mut self.attention_max,
            &mut self.attention_lse,
        )?;
        // cuBLAS is column-major: token columns advance by `16 * 256`,
        // while each head batch starts 256 elements later. The resulting
        // address is `[token][head][value]`, matching `o_proj`'s row-major
        // input.
        gemm_strided_batched_bf16(
            ctx,
            true,
            false,
            256,
            active,
            GLM52_KV_LORA_RANK,
            &weights.w_uv,
            GLM52_KV_LORA_RANK,
            256 * GLM52_KV_LORA_RANK,
            &self.attention_out,
            64 * GLM52_KV_LORA_RANK,
            GLM52_KV_LORA_RANK,
            &mut self.attention_v,
            16 * 256,
            256,
            16,
        )?;
        let rows = active.next_multiple_of(4);
        if rows > active {
            ctx.stream.memset_zeros(
                &mut self
                    .attention_v
                    .slice_mut(active * 16 * 256..rows * 16 * 256),
            )?;
        }
        fp8_linear_large_m_into(
            ctx,
            &weights.o_proj,
            rows,
            &self.attention_v,
            &mut self.fp8_gemm,
            &mut self.attention_partial,
        )
    }

    fn dense_mlp(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52DenseMlpWeights,
        rows: usize,
    ) -> Result<()> {
        glm52_dense_mlp_prefill_into(
            ctx,
            weights,
            rows,
            &self.normed,
            &mut self.fp8_gemm,
            &mut self.dense_gate_up,
            &mut self.dense_silu,
            &mut self.dense_out,
        )
    }

    fn moe_mlp(
        &mut self,
        ctx: &DeviceContext,
        state: &mut Glm52MoeTpState,
        router: &Glm52MoeRouterWeights,
        bank: &Glm52MoeTpSliceBank,
        active: usize,
    ) -> Result<()> {
        self.moe.forward(
            ctx,
            state,
            router,
            bank,
            &self.normed,
            active,
            &mut self.dense_out,
        )
    }

    fn reduce_and_norm_attention(
        &mut self,
        ctx: &DeviceContext,
        tp: &mut Glm52MoeTpState,
        post_attn_ln: &DeviceVec,
        active: usize,
    ) -> Result<()> {
        tp.prefill_ar_launch(
            ctx,
            active,
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
            active,
            &mut self.normed,
        )?;
        let rows = active.next_multiple_of(4);
        if rows > active {
            ctx.stream.memset_zeros(
                &mut self
                    .normed
                    .slice_mut(active * GLM52_HIDDEN..rows * GLM52_HIDDEN),
            )?;
        }
        Ok(())
    }

    fn finish_dense_layer(
        &mut self,
        ctx: &DeviceContext,
        next_input_ln: Option<&DeviceVec>,
        active: usize,
    ) -> Result<()> {
        match next_input_ln {
            Some(weight) => {
                fused_add_rms_norm_round_into(
                    ctx,
                    &mut self.attention_reduced,
                    &self.dense_out,
                    weight,
                    GLM52_RMS_EPS,
                    GLM52_HIDDEN,
                    active,
                    &mut self.normed,
                )?;
                ctx.stream.memcpy_dtod(
                    &self.attention_reduced.slice(..active * GLM52_HIDDEN),
                    &mut self.hidden.slice_mut(..active * GLM52_HIDDEN),
                )?;
                let rows = active.next_multiple_of(4);
                if rows > active {
                    ctx.stream.memset_zeros(
                        &mut self
                            .normed
                            .slice_mut(active * GLM52_HIDDEN..rows * GLM52_HIDDEN),
                    )?;
                }
            }
            None => {
                add_into(
                    ctx,
                    &self.attention_reduced,
                    &self.dense_out,
                    active * GLM52_HIDDEN,
                    &mut self.hidden,
                )?;
            }
        }
        Ok(())
    }

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
        glm52_prefill_moe_gather_launch(
            ctx,
            rows.len(),
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
        tp.prefill_ar_launch(
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
