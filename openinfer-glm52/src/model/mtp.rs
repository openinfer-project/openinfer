//! Checkpoint-native GLM5.2 MTP serving lane.
//!
//! The target step keeps its final-normalized hidden rows resident. A draft round
//! packs only committed rows, shifts each sequence token one place left, and
//! runs checkpoint layer 78 once to synchronize MTP KV and produce draft 1.
//! Four single-token iterations then recycle the layer's shared-head-normalized
//! hidden. Rejected speculative KV is not copied back: the next committed
//! first pass overwrites it at the same positions.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN;
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_TOPK;
use openinfer_kernels::ops::Glm52FlashMlaSparseDecode;
use openinfer_kernels::ops::Glm52IndexerCacheLayout;
use openinfer_kernels::ops::argmax_bf16_split_into;
use openinfer_kernels::ops::embedding_rows_into;
use openinfer_kernels::ops::glm52_flashmla_sparse_decode_num_sm_parts;
use openinfer_kernels::ops::rms_norm_rows_into;
use openinfer_kernels::tensor::DeviceContext;
use openinfer_kernels::tensor::DeviceMatrix;

use super::GLM52_DECODE_BUCKETS;
use super::GLM52_MAX_BATCH_PER_RANK;
use super::INDEX_CACHE_BLOCK;
use super::NUM_SMS;
use super::build;
use super::glm52_pool_blocks;
use super::glm52_table_width;
use super::step_body::glm52_moe_ep_layer;
use crate::bookend::glm52_embed_into;
use crate::bookend::glm52_lm_head_into;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_INDEX_HEAD_DIM;
use crate::config::GLM52_MTP_LAYER;
use crate::config::GLM52_RMS_EPS;
use crate::config::GLM52_SM_SCALE;
use crate::config::GLM52_VOCAB;
use crate::indexer::Glm52IndexerScratch;
use crate::layer::Glm52DecodeStep;
use crate::layer::Glm52DecoderLayerWeights;
use crate::layer::Glm52LayerCaches;
use crate::layer::Glm52LayerIndexMode;
use crate::layer::Glm52LayerMlp;
use crate::layer::glm52_layer_attention_half;
use crate::layer::glm52_layer_finish;
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::mla_decode::glm52_select_mla_backend;
use crate::moe_ep_wo::Glm52MoeEpState;
use crate::mtp::GLM52_MTP_DRAFTS;
use crate::mtp::Glm52MtpBookendWeights;
use crate::mtp::Glm52MtpScratch;
use crate::mtp::glm52_mtp_prepare_into;
use crate::mtp::glm52_mtp_recycle_into;
use crate::rows::Rows;
use crate::runner::Glm52MtpRound;
use crate::scratch::Glm52DecodeScratch;
use crate::weights::Glm52RankGpuWeights;
use crate::weights::retype_owned;

struct Glm52MtpBucket {
    rows: usize,
    sched: Glm52MlaSchedMetadata,
    scratch: Glm52DecodeScratch,
    bookend_scratch: Glm52MtpScratch,
    embeds: Rows<GLM52_HIDDEN>,
    previous: Rows<GLM52_HIDDEN>,
    decoder_input: Rows<GLM52_HIDDEN>,
    block_table: CudaSlice<i32>,
    compute_graph: CudaGraphState,
    reuse_graph: CudaGraphState,
}

pub(super) struct Glm52NativeMtp {
    bookend: Glm52MtpBookendWeights,
    layer: Glm52DecoderLayerWeights,
    cache: Glm52LayerCaches,
    buckets: [Glm52MtpBucket; GLM52_DECODE_BUCKETS.len()],
    max_model_len: usize,
    table_width: usize,
    pages_per_slot: usize,
    positions: CudaSlice<u32>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    token_ids: CudaSlice<u32>,
    slot_mapping: CudaSlice<i64>,
    seq_lens: CudaSlice<i32>,
    shared_topk: CudaSlice<i32>,
    committed_lens: [usize; GLM52_MAX_BATCH_PER_RANK],
}

impl Glm52NativeMtp {
    pub(super) fn build(
        ctx: &DeviceContext,
        weights: &mut Glm52RankGpuWeights,
        max_model_len: usize,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{GLM52_MTP_LAYER}");
        let enorm = build::take_bf16_vec(
            ctx,
            weights,
            &format!("{prefix}.enorm.weight"),
            GLM52_HIDDEN,
        )?;
        let hnorm = build::take_bf16_vec(
            ctx,
            weights,
            &format!("{prefix}.hnorm.weight"),
            GLM52_HIDDEN,
        )?;
        let eh_proj_raw = weights.take_tensor(&format!("{prefix}.eh_proj.weight"))?;
        ensure!(
            eh_proj_raw.len() == 2 * GLM52_HIDDEN * GLM52_HIDDEN * size_of::<bf16>(),
            "GLM5.2 MTP eh_proj byte length drifted"
        );
        let eh_proj = DeviceMatrix {
            data: retype_owned::<bf16>(&ctx.stream, eh_proj_raw)?,
            rows: GLM52_HIDDEN,
            cols: 2 * GLM52_HIDDEN,
        };
        let shared_norm = build::take_bf16_vec(
            ctx,
            weights,
            &format!("{prefix}.shared_head.norm.weight"),
            GLM52_HIDDEN,
        )?;
        let bookend = Glm52MtpBookendWeights::new(enorm, hnorm, eh_proj, shared_norm)?;
        let layer = build::build_decoder_layer(
            ctx,
            weights,
            GLM52_MTP_LAYER,
            crate::Glm52MoeTopo::Ep8,
            None,
        )?;

        let num_blocks = glm52_pool_blocks(max_model_len, GLM52_MAX_BATCH_PER_RANK);
        let table_width = glm52_table_width(max_model_len);
        let index_layout = Glm52IndexerCacheLayout {
            cache_blocks: num_blocks,
            cache_block_size: INDEX_CACHE_BLOCK,
            cache_block_stride_bytes: INDEX_CACHE_BLOCK * (GLM52_INDEX_HEAD_DIM + 4),
        };
        let backend = glm52_select_mla_backend(crate::config::GLM52_HEADS)?;
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: GLM52_MAX_BATCH_PER_RANK,
            num_blocks,
            topk: GLM52_FLASHMLA_SPARSE_TOPK,
            num_sm_parts: glm52_flashmla_sparse_decode_num_sm_parts()?,
            sm_scale: GLM52_SM_SCALE,
        };
        let cache = Glm52LayerCaches {
            mla_cache: ctx.stream.alloc_zeros::<u8>(
                num_blocks
                    * GLM52_FLASHMLA_SPARSE_PAGE_SIZE
                    * GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN,
            )?,
            index_k_cache: Some(
                ctx.stream
                    .alloc_zeros::<u8>(index_layout.min_cache_bytes()?)?,
            ),
        };

        let mut buckets = Vec::with_capacity(GLM52_DECODE_BUCKETS.len());
        for rows in GLM52_DECODE_BUCKETS {
            let row_contract = Glm52FlashMlaSparseDecode {
                batch_size: rows,
                ..contract
            };
            let mqa_shape = Glm52IndexerScratch::paged_mqa_shape(
                rows,
                index_layout,
                table_width,
                NUM_SMS,
                max_model_len,
            );
            buckets.push(Glm52MtpBucket {
                rows,
                sched: Glm52MlaSchedMetadata::new_for_backend(
                    ctx,
                    row_contract,
                    crate::config::GLM52_HEADS,
                    backend,
                )?,
                scratch: Glm52DecodeScratch::new_for_backend(
                    ctx,
                    &row_contract,
                    mqa_shape,
                    crate::config::GLM52_HEADS,
                    backend,
                    false,
                )?,
                bookend_scratch: Glm52MtpScratch::new(ctx, rows)?,
                embeds: Rows::zeros(ctx, rows)?,
                previous: Rows::zeros(ctx, rows)?,
                decoder_input: Rows::zeros(ctx, rows)?,
                block_table: ctx.stream.alloc_zeros::<i32>(rows * table_width)?,
                compute_graph: CudaGraphState::new(),
                reuse_graph: CudaGraphState::new(),
            });
        }
        Ok(Self {
            bookend,
            layer,
            cache,
            buckets: buckets
                .try_into()
                .map_err(|_| anyhow::anyhow!("GLM5.2 MTP bucket count drifted"))?,
            max_model_len,
            table_width,
            pages_per_slot: (max_model_len + 1).div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE),
            positions: ctx.stream.alloc_zeros(GLM52_MAX_BATCH_PER_RANK)?,
            cos: ctx
                .stream
                .alloc_zeros(GLM52_MAX_BATCH_PER_RANK * crate::config::GLM52_ROPE_HALF)?,
            sin: ctx
                .stream
                .alloc_zeros(GLM52_MAX_BATCH_PER_RANK * crate::config::GLM52_ROPE_HALF)?,
            token_ids: ctx.stream.alloc_zeros(GLM52_MAX_BATCH_PER_RANK)?,
            slot_mapping: ctx.stream.alloc_zeros(GLM52_MAX_BATCH_PER_RANK)?,
            seq_lens: ctx.stream.alloc_zeros(GLM52_MAX_BATCH_PER_RANK)?,
            shared_topk: ctx
                .stream
                .alloc_zeros(GLM52_MAX_BATCH_PER_RANK * GLM52_FLASHMLA_SPARSE_TOPK)?,
            committed_lens: [0; GLM52_MAX_BATCH_PER_RANK],
        })
    }

    pub(super) fn reset_slots(&mut self, resets: &[usize]) -> Result<()> {
        for &slot in resets {
            ensure!(
                slot < GLM52_MAX_BATCH_PER_RANK,
                "GLM5.2 MTP reset slot {slot} is outside \
                 0..{GLM52_MAX_BATCH_PER_RANK}"
            );
            self.committed_lens[slot] = 0;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        ep: &mut Glm52MoeEpState,
        embed: &DeviceMatrix,
        lm_head: &DeviceMatrix,
        cos_table: &DeviceMatrix,
        sin_table: &DeviceMatrix,
        target_final_normed: &Rows<GLM52_HIDDEN>,
        round: &Glm52MtpRound,
    ) -> Result<Vec<[u32; GLM52_MTP_DRAFTS]>> {
        let (context_bucket, appends, proposal) = match round {
            Glm52MtpRound::Context {
                context_bucket,
                appends,
                ..
            } => (*context_bucket, appends.as_slice(), None),
            Glm52MtpRound::Propose {
                context_bucket,
                draft_bucket,
                appends,
                proposal_slots,
                ..
            } => (
                *context_bucket,
                appends.as_slice(),
                Some((*draft_bucket, proposal_slots.as_slice())),
            ),
            Glm52MtpRound::Reset { .. } => {
                unreachable!("reset-only MTP rounds return before target hidden is selected")
            }
        };
        let proposal_slots = proposal.map_or(&[][..], |(_, slots)| slots);
        ensure!(
            appends.len() <= context_bucket,
            "GLM5.2 MTP context rows {} exceed collective bucket {context_bucket}",
            appends.len(),
        );
        ensure!(
            proposal_slots.windows(2).all(|pair| pair[0] < pair[1]),
            "GLM5.2 MTP proposal slots must be strictly ascending"
        );
        let context_index = self.bucket_index(context_bucket)?;
        for (packed, append) in appends.iter().enumerate() {
            ensure!(
                append.slot < GLM52_MAX_BATCH_PER_RANK
                    && append.target_row < target_final_normed.tokens(),
                "GLM5.2 MTP append target row {} or slot {} is out of bounds \
                 (target rows {}, slots {})",
                append.target_row,
                append.slot,
                target_final_normed.tokens(),
                GLM52_MAX_BATCH_PER_RANK,
            );
            ensure!(
                append.position == self.committed_lens[append.slot],
                "GLM5.2 MTP slot {} first-pass position {} != committed {}",
                append.slot,
                append.position,
                self.committed_lens[append.slot]
            );
            let src = target_final_normed
                .data()
                .slice(append.target_row * GLM52_HIDDEN..(append.target_row + 1) * GLM52_HIDDEN);
            let mut dst = self.buckets[context_index]
                .previous
                .data_mut()
                .slice_mut(packed * GLM52_HIDDEN..(packed + 1) * GLM52_HIDDEN);
            ctx.stream.memcpy_dtod(&src, &mut dst)?;
            self.committed_lens[append.slot] += 1;
        }
        let context_inputs: Vec<(usize, u32, usize)> = appends
            .iter()
            .map(|append| (append.slot, append.input_token, append.position))
            .collect();
        self.forward(
            ctx,
            aux,
            ep,
            embed,
            lm_head,
            cos_table,
            sin_table,
            context_index,
            &context_inputs,
            Glm52LayerIndexMode::Normal,
        )?;
        let Some((draft_bucket, proposal_slots)) = proposal else {
            return Ok(Vec::new());
        };
        ensure!(
            proposal_slots.len() <= draft_bucket,
            "GLM5.2 MTP proposal rows {} exceed collective bucket {draft_bucket}",
            proposal_slots.len(),
        );
        let mut last_rows = Vec::with_capacity(proposal_slots.len());
        for &slot in proposal_slots {
            let row = appends
                .iter()
                .rposition(|append| append.slot == slot)
                .with_context(|| format!("GLM5.2 MTP proposal slot {slot} has no append"))?;
            last_rows.push(row);
        }
        let context_tokens = self.argmax_host(ctx, context_index)?;
        let draft_index = self.bucket_index(draft_bucket)?;
        for (packed, (&slot, &context_row)) in proposal_slots.iter().zip(&last_rows).enumerate() {
            let src_topk = self.buckets[context_index].scratch.idx.global_slots.slice(
                context_row * GLM52_FLASHMLA_SPARSE_TOPK
                    ..(context_row + 1) * GLM52_FLASHMLA_SPARSE_TOPK,
            );
            let mut dst_topk = self.shared_topk.slice_mut(
                packed * GLM52_FLASHMLA_SPARSE_TOPK..(packed + 1) * GLM52_FLASHMLA_SPARSE_TOPK,
            );
            ctx.stream.memcpy_dtod(&src_topk, &mut dst_topk)?;
            let src_hidden = self.buckets[context_index]
                .scratch
                .final_normed
                .data()
                .slice(context_row * GLM52_HIDDEN..(context_row + 1) * GLM52_HIDDEN);
            let mut dst_hidden = self.buckets[draft_index]
                .previous
                .data_mut()
                .slice_mut(packed * GLM52_HIDDEN..(packed + 1) * GLM52_HIDDEN);
            ctx.stream.memcpy_dtod(&src_hidden, &mut dst_hidden)?;
            ensure!(
                self.committed_lens[slot] < self.max_model_len,
                "GLM5.2 MTP slot {slot} exhausted its context cap"
            );
        }

        let mut spans = vec![[0u32; GLM52_MTP_DRAFTS]; proposal_slots.len()];
        for (span, &row) in spans.iter_mut().zip(&last_rows) {
            span[0] = context_tokens[row];
        }
        for iteration in 1..GLM52_MTP_DRAFTS {
            let inputs: Vec<(usize, u32, usize)> = proposal_slots
                .iter()
                .enumerate()
                .map(|(row, &slot)| {
                    (
                        slot,
                        spans[row][iteration - 1],
                        self.committed_lens[slot] + iteration - 1,
                    )
                })
                .collect();
            {
                let topk_len = proposal_slots.len() * GLM52_FLASHMLA_SPARSE_TOPK;
                if topk_len > 0 {
                    let src = self.shared_topk.slice(..topk_len);
                    let mut dst = self.buckets[draft_index]
                        .scratch
                        .idx
                        .global_slots
                        .slice_mut(..topk_len);
                    ctx.stream.memcpy_dtod(&src, &mut dst)?;
                }
            }
            self.forward(
                ctx,
                aux,
                ep,
                embed,
                lm_head,
                cos_table,
                sin_table,
                draft_index,
                &inputs,
                Glm52LayerIndexMode::Reuse,
            )?;
            let tokens = self.argmax_host(ctx, draft_index)?;
            for (row, span) in spans.iter_mut().enumerate() {
                span[iteration] = tokens[row];
                let src = self.buckets[draft_index]
                    .scratch
                    .final_normed
                    .data()
                    .slice(row * GLM52_HIDDEN..(row + 1) * GLM52_HIDDEN);
                let mut dst = self.buckets[draft_index]
                    .previous
                    .data_mut()
                    .slice_mut(row * GLM52_HIDDEN..(row + 1) * GLM52_HIDDEN);
                ctx.stream.memcpy_dtod(&src, &mut dst)?;
            }
        }
        Ok(spans)
    }

    fn bucket_index(&self, rows: usize) -> Result<usize> {
        self.buckets
            .iter()
            .position(|bucket| bucket.rows == rows)
            .with_context(|| format!("GLM5.2 MTP bucket {rows} is not in {GLM52_DECODE_BUCKETS:?}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        ep: &mut Glm52MoeEpState,
        embed: &DeviceMatrix,
        lm_head: &DeviceMatrix,
        cos_table: &DeviceMatrix,
        sin_table: &DeviceMatrix,
        bucket_index: usize,
        inputs: &[(usize, u32, usize)],
        index_mode: Glm52LayerIndexMode,
    ) -> Result<()> {
        let rows = self.buckets[bucket_index].rows;
        let mut tokens = [0u32; GLM52_MAX_BATCH_PER_RANK];
        let mut positions = [0u32; GLM52_MAX_BATCH_PER_RANK];
        let mut seq_lens = [1i32; GLM52_MAX_BATCH_PER_RANK];
        let mut slot_mapping = [0i64; GLM52_MAX_BATCH_PER_RANK];
        let mut pages = vec![0i32; rows * self.table_width];
        for (row, &(slot, token, position)) in inputs.iter().enumerate() {
            ensure!(
                row < rows && slot < GLM52_MAX_BATCH_PER_RANK && position < self.max_model_len,
                "GLM5.2 MTP input row {row}/{rows}, slot \
                 {slot}/{GLM52_MAX_BATCH_PER_RANK}, or position \
                 {position}/{} is out of bounds",
                self.max_model_len,
            );
            tokens[row] = token;
            positions[row] = position as u32;
            seq_lens[row] = (position + 1) as i32;
            let page_offset = position / GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
            let page = 1 + slot * self.pages_per_slot + page_offset;
            slot_mapping[row] = (page * GLM52_FLASHMLA_SPARSE_PAGE_SIZE
                + position % GLM52_FLASHMLA_SPARSE_PAGE_SIZE)
                as i64;
            for logical_page in 0..=page_offset {
                pages[row * self.table_width + logical_page] =
                    (1 + slot * self.pages_per_slot + logical_page) as i32;
            }
        }
        ctx.stream.memcpy_htod(&tokens, &mut self.token_ids)?;
        ctx.stream.memcpy_htod(&positions, &mut self.positions)?;
        ctx.stream.memcpy_htod(&seq_lens, &mut self.seq_lens)?;
        ctx.stream
            .memcpy_htod(&slot_mapping, &mut self.slot_mapping)?;
        embedding_rows_into(ctx, cos_table, &self.positions, rows, &mut self.cos)?;
        embedding_rows_into(ctx, sin_table, &self.positions, rows, &mut self.sin)?;
        ctx.stream
            .memcpy_htod(&pages, &mut self.buckets[bucket_index].block_table)?;

        let bucket = &mut self.buckets[bucket_index];
        let Glm52MtpBucket {
            sched,
            scratch,
            bookend_scratch,
            embeds,
            previous,
            decoder_input,
            block_table,
            compute_graph,
            reuse_graph,
            ..
        } = bucket;
        let step = Glm52DecodeStep {
            mla_cos: &self.cos,
            mla_sin: &self.sin,
            idx_cos: &self.cos,
            idx_sin: &self.sin,
            mla_sched: sched,
            slot_mapping: &self.slot_mapping,
            block_table,
            seq_lens: &self.seq_lens,
        };
        let graph = match index_mode {
            Glm52LayerIndexMode::Normal => compute_graph,
            Glm52LayerIndexMode::Reuse => reuse_graph,
        };
        graph.run_or_capture(ctx, || {
            glm52_embed_into(ctx, embed, &self.token_ids, embeds)?;
            glm52_mtp_prepare_into(
                ctx,
                &self.bookend,
                &self.positions,
                embeds,
                previous,
                bookend_scratch,
                decoder_input,
            )?;
            ctx.stream
                .memcpy_dtod(decoder_input.data(), scratch.hidden.data_mut())?;
            rms_norm_rows_into(
                ctx,
                scratch.hidden.data(),
                &self.layer.input_ln,
                GLM52_RMS_EPS,
                GLM52_HIDDEN,
                rows,
                scratch.layer.normed.data_mut(),
            )?;
            let mut carry_ready = index_mode == Glm52LayerIndexMode::Reuse;
            glm52_layer_attention_half(
                ctx,
                Some(aux),
                &self.layer,
                &mut self.cache,
                &step,
                scratch,
                &mut carry_ready,
                0,
                true,
                None,
                index_mode,
            )?;
            let Glm52LayerMlp::MoeEp8(moe) = &self.layer.mlp else {
                anyhow::bail!("GLM5.2 MTP layer 78 is not EP MoE")
            };
            glm52_moe_ep_layer(
                ctx,
                aux,
                ep,
                moe,
                scratch,
                rows,
                crate::weights::GLM52_EP_RANKS * rows,
            )?;
            glm52_layer_finish(ctx, scratch, 0, false)?;
            glm52_mtp_recycle_into(
                ctx,
                &self.bookend,
                &scratch.hidden,
                &mut scratch.final_normed,
            )?;
            glm52_lm_head_into(ctx, &scratch.final_normed, lm_head, &mut scratch.logits)?;
            argmax_bf16_split_into(
                ctx,
                scratch.logits.data(),
                rows,
                GLM52_VOCAB,
                &mut scratch.argmax_partial_values,
                &mut scratch.argmax_partial_indices,
                &mut scratch.argmax_values,
                &mut scratch.argmax_indices,
            )
        })
    }

    fn argmax_host(&self, ctx: &DeviceContext, bucket_index: usize) -> Result<Vec<u32>> {
        let bucket = &self.buckets[bucket_index];
        let values = ctx.stream.clone_dtoh(&bucket.scratch.argmax_values)?;
        let indices = ctx.stream.clone_dtoh(&bucket.scratch.argmax_indices)?;
        values
            .iter()
            .zip(indices)
            .enumerate()
            .map(|(row, (value, index))| {
                ensure!(
                    value.to_f32().is_finite() && index >= 0,
                    "GLM5.2 MTP row {row} produced invalid argmax value {} at index {index}",
                    value.to_f32(),
                );
                u32::try_from(index).context("GLM5.2 MTP argmax does not fit u32")
            })
            .collect()
    }
}
