//! GLM5.2 DP8/EP8 full-model decode: every rank owns the whole non-expert
//! path (embed → 78 decoder layers → final norm → lm_head → greedy argmax)
//! plus its 32 local experts, and forwards one of the
//! [`GLM52_DECODE_BUCKETS`] batch buckets per step (real request tokens in
//! occupied slots, padding rows elsewhere; prefill rides decode as *spans* —
//! several consecutive positions of one slot in a single step). Each slot
//! owns a disjoint `GLM52_MAX_MODEL_LEN`-token region of the paged KV/index
//! caches, so pad rows write only their own dead slots.
//!
//! Every step, all 8 ranks run the forward in lock-step with the SAME bucket
//! (the coordinator's global decision), each dispatching exactly that many
//! rows into every MoE layer's DeepEP collective — the collectives require
//! all ranks to enter in the same layer order 3..=77 with the agreed global
//! row count, and the per-bucket fixed row count keeps every step's kernel
//! shapes identical within a bucket (the whole-step CUDA graphs' contract).

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaSlice, PinnedHostSlice};
use half::bf16;
use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE, GLM52_FLASHMLA_SPARSE_TOPK,
    GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW, Glm52FlashMlaSparseDecode, Glm52IndexerCacheLayout,
    add_into, argmax_bf16_split_into, copy_hidden_rows_raw_into, embedding_rows_into,
    glm52_flashmla_sparse_decode_num_sm_parts, glm52_fp8_weight_only_gemv_launch,
    rms_norm_rows_into,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

use crate::bookend::{glm52_embed_into, glm52_final_norm_into, glm52_lm_head_into};
use crate::config::{
    GLM52_HIDDEN, GLM52_INDEX_HEAD_DIM, GLM52_INDEX_TOPK, GLM52_LAYERS, GLM52_RMS_EPS,
    GLM52_ROPE_HALF, GLM52_SM_SCALE, GLM52_VOCAB, glm52_layer_has_full_indexer,
};
use crate::dense::glm52_dense_mlp_forward_into;
use crate::indexer::Glm52IndexerScratch;
use crate::layer::{
    Glm52DecodeStep, Glm52DecoderLayerWeights, Glm52LayerCaches, Glm52LayerMlp,
    glm52_layer_attention_half, glm52_layer_finish, glm52_layer_finish_fused,
};
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::moe_decode::run_router_into;
use crate::moe_ep8::{Glm52MoeEp8State, glm52_moe_ep8_routed_forward};
use crate::scratch::Glm52DecodeScratch;
use crate::weights::{Glm52RankGpuWeights, retype_owned};

mod build;
mod launch_ahead;
use launch_ahead::Glm52SpeculatedStep;

/// Per-request context cap: `prompt + max_tokens - 1 <= GLM52_MAX_MODEL_LEN`.
/// Sizes each slot's region of the per-layer MLA and index-K caches at build
/// time.dan
pub(crate) const GLM52_MAX_MODEL_LEN: usize = 4096;

/// The per-rank slot count and the largest decode bucket. Each slot owns a
/// `GLM52_MAX_MODEL_LEN`-token region of the paged caches.
pub(crate) const GLM52_MAX_BATCH_PER_RANK: usize = 8;

/// The decode batch buckets, ascending. Each bucket has its own captured
/// CUDA graphs, scratch arena, and FlashMLA plans, and the batched GEMV
/// kernel is instantiated for exactly these row counts (`kBatchedGemvBatch*`
/// in `glm52_moe_gemv.cu` — a drift crashes at the launch boundary). The
/// coordinator picks the smallest bucket covering the fullest rank, so a
/// lightly-loaded fleet keeps the small-step cost; discrete buckets (not a
/// continuum) keep the whole-step graphs' fixed-shape contract.
pub(crate) const GLM52_DECODE_BUCKETS: [usize; 4] = [1, 2, 4, GLM52_MAX_BATCH_PER_RANK];

// The DeepGEMM masked grouped expert GEMM gives every local expert a fixed
// per-expert row slab; each source token contributes at most one row per
// expert, so the protocol's worst-case global token count must fit it.
// Compile-time so a future GLM52_MAX_BATCH_PER_RANK bump fails here, not at
// graph capture.
const _: () = assert!(
    crate::weights::GLM52_EP_RANKS * GLM52_MAX_BATCH_PER_RANK
        <= openinfer_kernels::ops::GLM52_DEEPGEMM_MASKED_CAP
);

// The decode feed kernel runs one 32-thread block (`glm52_decode_feed.cu`);
// a batch-cap bump past it must widen the kernel, not silently truncate.
const _: () = assert!(GLM52_MAX_BATCH_PER_RANK <= 32);

/// The step's forward shape, agreed globally by the coordinator: `bucket`
/// rows per rank (a member of [`GLM52_DECODE_BUCKETS`] — the MoE collectives
/// require every rank to enter with the same global row count), with
/// `slots[row]` naming the cache slot each forwarded row addresses for
/// `row < bucket` (active slots first, padding rows parked on free slots
/// whose cache regions are dead).
///
/// A slot may own SEVERAL rows (a *span*): one contiguous run of rows walking
/// consecutive positions of that slot's sequence — how prompt tokens batch
/// through the decode path, and the shape a DSpark verify step reuses. Within
/// a step, a later row of a span attends to the earlier rows' KV through the
/// cache: per layer every row's cache write lands before any row's attention
/// launches, and row `k`'s `seq_len` admits exactly the positions before it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StepShape {
    pub(crate) bucket: usize,
    pub(crate) slots: [u8; GLM52_MAX_BATCH_PER_RANK],
    /// Rows `0..active_rows` carry real requests; `active_rows..bucket` are
    /// padding. Carried explicitly because a padding input is NOT
    /// value-distinguishable from an active one (a single-token prompt `[0]`
    /// legally feeds `(token 0, position 0)`).
    pub(crate) active_rows: usize,
}

/// Short-context attention tier: while `seq_len <= 256` the DSA top-256 IS
/// the full token set (exactly like top-2048 is below 2048 tokens), so the
/// short-tier graph attends the same tokens at 1/8 the FlashMLA padding work
/// — the V3.2 sparse kernel always walks all `topk` index slots (`-1` pads
/// run the full load+GEMM and are only masked in softmax). Must be a multiple
/// of the 64-entry topk block.
pub(crate) const GLM52_MLA_TOPK_SHORT: usize = 256;

// Both attention tiers' `topk` feed the DSA indexer's top-k selection, whose
// buffers are sized for GLM52_INDEX_TOPK rows — pin the range here so the
// indexer forward never needs to re-check it per layer per step.
const _: () = assert!(GLM52_MLA_TOPK_SHORT > 0 && GLM52_MLA_TOPK_SHORT.is_multiple_of(64));
const _: () = assert!(GLM52_MLA_TOPK_SHORT <= GLM52_INDEX_TOPK);
const _: () = assert!(GLM52_FLASHMLA_SPARSE_TOPK > 0);
const _: () = assert!(GLM52_FLASHMLA_SPARSE_TOPK <= GLM52_INDEX_TOPK);

/// DeepGEMM paged MQA requires BLOCK_KV=64 — a kernel constraint, not a
/// model property (kept here, not in config.rs).
pub(crate) const INDEX_CACHE_BLOCK: usize = 64;
/// H200 SM count — hardware property, not a model property.
pub(crate) const NUM_SMS: usize = 132;

pub(crate) fn rope_tables(position: usize) -> (Vec<bf16>, Vec<bf16>) {
    let theta = crate::config::GLM52_ROPE_THETA as f32;
    (0..GLM52_ROPE_HALF)
        .map(|j| {
            let inv_freq = 1.0 / theta.powf(j as f32 / GLM52_ROPE_HALF as f32);
            let angle = position as f32 * inv_freq;
            (bf16::from_f32(angle.cos()), bf16::from_f32(angle.sin()))
        })
        .unzip()
}

/// One DP rank: the full non-expert model plus this rank's expert banks.
pub(crate) struct Glm52RankModel {
    layers: Vec<Glm52DecoderLayerWeights>,
    caches: Vec<Glm52LayerCaches>,
    embed: DeviceMatrix,
    final_norm: DeviceVec,
    lm_head: DeviceMatrix,
    /// Per-bucket execution state, index-aligned with
    /// [`GLM52_DECODE_BUCKETS`]. Selecting one `Glm52BucketState` selects the
    /// plans, scratch, graphs, and block table together — a graph can never
    /// be taken from one shape and restored into another.
    buckets: [Glm52BucketState; GLM52_DECODE_BUCKETS.len()],
    /// `[GLM52_MAX_BATCH_PER_RANK, blocks_per_slot]` — row b maps slot b's
    /// context into its disjoint region of the index-K cache. Static: written
    /// once at build; every bucket's table is gathered from it per step.
    block_table: CudaSlice<i32>,
    blocks_per_slot: usize,
    slot_mapping: CudaSlice<i64>,
    seq_lens: CudaSlice<i32>,
    /// Device-resident rope tables for every position (`[GLM52_MAX_MODEL_LEN,
    /// GLM52_ROPE_HALF]` as a gatherable matrix); the prologue gathers each row's
    /// position row instead of recomputing on the host and copying up.
    cos_table: DeviceMatrix,
    sin_table: DeviceMatrix,
    positions: CudaSlice<u32>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    token_ids: CudaSlice<u32>,
    /// In-flight speculative next-step replay, if any (see `decode_step`).
    speculated: Option<Glm52SpeculatedStep>,
    /// What the per-row `positions` device buffer currently holds (padding
    /// rows included): the feed kernel advances it without host readback,
    /// and a speculation must keep every row under the model-length cap.
    device_positions: [usize; GLM52_MAX_BATCH_PER_RANK],
}

/// Everything one decode bucket owns: batch-`rows` FlashMLA plans (per
/// attention tier), the scratch arena, the whole-step CUDA graphs (each
/// captured on this rank's first step in that (bucket × tier) shape and
/// replayed every step after — valid forever, since within a shape every
/// step has the same kernel sequence and the same arena pointers by
/// construction; the per-step inputs are the device buffers the prologue
/// rewrites), and the `[rows, blocks_per_slot]` block table, rewritten by
/// the prologue every step (dtod gather of each forwarded row's slot
/// region), so the captured graphs address whichever slots hold the
/// requests — and span rows their repeated slot — through device data,
/// never baked slot ids.
struct Glm52BucketState {
    rows: usize,
    scheds: [Glm52MlaSchedMetadata; 2],
    scratch: Glm52DecodeScratch,
    graphs: [CudaGraphState; 2],
    block_table: CudaSlice<i32>,
    /// Pinned landing buffers for this bucket's argmax D2H, sized exactly
    /// `rows` (`memcpy_dtoh` copies the DESTINATION's byte count). Pinned
    /// memory keeps the copy asynchronous so the next step's replay can be
    /// enqueued launch-ahead before the host blocks on the result.
    argmax_values_host: PinnedHostSlice<bf16>,
    argmax_indices_host: PinnedHostSlice<i32>,
}

/// Tier index into the per-tier arrays: every consumer selects with the same
/// index computed once per step.
const TIER_FULL: usize = 0;
const TIER_SHORT: usize = 1;

impl Glm52RankModel {
    /// The token embedding table — the DSpark draft's block embedding reuses
    /// it (the draft checkpoint's copy is byte-identical and not loaded).
    pub(crate) fn embed(&self) -> &DeviceMatrix {
        &self.embed
    }

    /// The lm_head — the DSpark draft's logits reuse it (same reuse contract
    /// as [`Self::embed`]).
    pub(crate) fn lm_head(&self) -> &DeviceMatrix {
        &self.lm_head
    }

    /// The last step's aux-hidden capture buffer for `bucket` (`[bucket,
    /// 5 * GLM52_HIDDEN]`, row = step row). Valid until the next step in the
    /// same bucket overwrites it — the draft lane consumes it between steps.
    pub(crate) fn captured(&self, bucket: usize) -> Result<&CudaSlice<bf16>> {
        let state = self
            .buckets
            .iter()
            .find(|state| state.rows == bucket)
            .with_context(|| {
                format!(
                    "GLM5.2 capture bucket {bucket} is not a member of {GLM52_DECODE_BUCKETS:?}"
                )
            })?;
        Ok(state.scratch.captured.data())
    }

    pub(crate) fn build(ctx: &DeviceContext, w: &mut Glm52RankGpuWeights) -> Result<Self> {
        let batch = GLM52_MAX_BATCH_PER_RANK;
        let num_sm_parts = glm52_flashmla_sparse_decode_num_sm_parts()?;
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: batch,
            num_blocks: batch * GLM52_MAX_MODEL_LEN.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE),
            topk: GLM52_FLASHMLA_SPARSE_TOPK,
            num_sm_parts,
            sm_scale: GLM52_SM_SCALE,
        };
        // The short tier walks only 256/64 = 4 index blocks per row — more
        // SM parts than blocks would just be empty splits for the combine.
        let contract_short = Glm52FlashMlaSparseDecode {
            topk: GLM52_MLA_TOPK_SHORT,
            num_sm_parts: num_sm_parts.min(GLM52_MLA_TOPK_SHORT / 64),
            ..contract
        };
        // Each slot owns `blocks_per_slot` consecutive index-K cache blocks;
        // the whole cache holds every slot's region back-to-back.
        let blocks_per_slot = GLM52_MAX_MODEL_LEN.div_ceil(INDEX_CACHE_BLOCK);
        let index_cache_layout = Glm52IndexerCacheLayout {
            cache_blocks: batch * blocks_per_slot,
            cache_block_size: INDEX_CACHE_BLOCK,
            cache_block_stride_bytes: INDEX_CACHE_BLOCK * (GLM52_INDEX_HEAD_DIM + 4),
        };

        let mut layers = Vec::with_capacity(GLM52_LAYERS);
        let mut caches = Vec::with_capacity(GLM52_LAYERS);
        for layer in 0..GLM52_LAYERS {
            layers.push(
                build::build_decoder_layer(ctx, w, layer)
                    .with_context(|| format!("build GLM5.2 decoder layer {layer}"))?,
            );
            caches.push(Glm52LayerCaches {
                mla_cache: ctx
                    .stream
                    .alloc_zeros::<u8>(contract.packed_kv_cache_len())?,
                index_k_cache: glm52_layer_has_full_indexer(layer)
                    .then(|| {
                        ctx.stream
                            .alloc_zeros::<u8>(index_cache_layout.min_cache_bytes()?)
                            .map_err(anyhow::Error::from)
                    })
                    .transpose()?,
            });
        }

        let embed_raw = w.take_tensor("model.embed_tokens.weight")?;
        let lm_head_raw = w.take_tensor("lm_head.weight")?;
        ensure!(
            embed_raw.len() == GLM52_VOCAB * GLM52_HIDDEN * 2
                && lm_head_raw.len() == GLM52_VOCAB * GLM52_HIDDEN * 2,
            "GLM5.2 embed/lm_head byte lengths unexpected"
        );
        let embed = DeviceMatrix {
            data: retype_owned::<bf16>(&ctx.stream, embed_raw)?,
            rows: GLM52_VOCAB,
            cols: GLM52_HIDDEN,
        };
        let lm_head = DeviceMatrix {
            data: retype_owned::<bf16>(&ctx.stream, lm_head_raw)?,
            rows: GLM52_VOCAB,
            cols: GLM52_HIDDEN,
        };
        let final_norm = build::take_bf16_vec(ctx, w, "model.norm.weight", GLM52_HIDDEN)?;

        // The MTP layer's experts are loaded (checkpoint-coverage validation)
        // but out of campaign scope — drop this rank's copy.
        let _ = w.take_expert_layer(crate::weights::GLM52_MTP_LAYER)?;

        // Row b of the block table maps slot b's context onto its own cache
        // region: block j of slot b is global block b*blocks_per_slot + j.
        let block_table_host: Vec<i32> = (0..batch * blocks_per_slot).map(|i| i as i32).collect();
        let mut block_table = ctx.stream.alloc_zeros::<i32>(batch * blocks_per_slot)?;
        ctx.stream
            .memcpy_htod(&block_table_host, &mut block_table)?;

        let mut cos_host = Vec::with_capacity(GLM52_MAX_MODEL_LEN * GLM52_ROPE_HALF);
        let mut sin_host = Vec::with_capacity(GLM52_MAX_MODEL_LEN * GLM52_ROPE_HALF);
        for position in 0..GLM52_MAX_MODEL_LEN {
            let (cos_row, sin_row) = rope_tables(position);
            cos_host.extend_from_slice(&cos_row);
            sin_host.extend_from_slice(&sin_row);
        }
        let mut cos_table_data = ctx
            .stream
            .alloc_zeros::<bf16>(GLM52_MAX_MODEL_LEN * GLM52_ROPE_HALF)?;
        let mut sin_table_data = ctx
            .stream
            .alloc_zeros::<bf16>(GLM52_MAX_MODEL_LEN * GLM52_ROPE_HALF)?;
        ctx.stream.memcpy_htod(&cos_host, &mut cos_table_data)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin_table_data)?;

        // One Glm52BucketState per decode bucket: batch-`rows` contracts
        // (num_blocks is cache geometry, not batch, so it carries over),
        // plans, scratch, and a block table pre-filled with the identity
        // prefix (the step prologue rewrites it per step).
        let mut buckets = Vec::with_capacity(GLM52_DECODE_BUCKETS.len());
        for rows in GLM52_DECODE_BUCKETS {
            let contract_rows = Glm52FlashMlaSparseDecode {
                batch_size: rows,
                ..contract
            };
            let contract_rows_short = Glm52FlashMlaSparseDecode {
                batch_size: rows,
                ..contract_short
            };
            let mqa_shape = Glm52IndexerScratch::decode_shape(
                rows,
                index_cache_layout,
                blocks_per_slot,
                NUM_SMS,
                GLM52_MAX_MODEL_LEN,
            );
            let mut bucket_table = ctx.stream.alloc_zeros::<i32>(rows * blocks_per_slot)?;
            ctx.stream.memcpy_htod(
                &block_table_host[..rows * blocks_per_slot],
                &mut bucket_table,
            )?;
            buckets.push(Glm52BucketState {
                rows,
                scheds: [
                    Glm52MlaSchedMetadata::new(ctx, contract_rows)?,
                    Glm52MlaSchedMetadata::new(ctx, contract_rows_short)?,
                ],
                scratch: Glm52DecodeScratch::new(ctx, &contract_rows, mqa_shape)?,
                graphs: [CudaGraphState::new(), CudaGraphState::new()],
                block_table: bucket_table,
                // Read only after a D2H lands in them (the write-combined
                // pages start uninitialized).
                argmax_values_host: unsafe { ctx.ctx.alloc_pinned::<bf16>(rows)? },
                argmax_indices_host: unsafe { ctx.ctx.alloc_pinned::<i32>(rows)? },
            });
        }
        let buckets: [Glm52BucketState; GLM52_DECODE_BUCKETS.len()] = buckets
            .try_into()
            .map_err(|_| anyhow::anyhow!("GLM5.2 bucket state count drifted from the const"))?;

        // Crash-early pre-flight: launch the batched weight-only GEMV once
        // per bucket, so a GLM52_DECODE_BUCKETS entry without a matching CUDA
        // template instantiation (`kBatchedGemvBatch*` in glm52_moe_gemv.cu)
        // fails at startup — not on the first mid-serving step that reaches
        // that bucket (graphs are lazily captured; nothing else exercises a
        // bucket before real traffic does). Zeroed dummy operands in the
        // smallest whitelisted linear shape (indexer wk, n=128 k=6144).
        {
            let (n, k) = (128usize, 6144usize);
            let weight = ctx.stream.alloc_zeros::<u8>(n * k)?;
            let scale = ctx
                .stream
                .alloc_zeros::<u8>(n.div_ceil(128) * k.div_ceil(128) * 4)?;
            let activation = ctx
                .stream
                .alloc_zeros::<bf16>(GLM52_MAX_BATCH_PER_RANK * k)?;
            let mut out = ctx
                .stream
                .alloc_zeros::<bf16>(GLM52_MAX_BATCH_PER_RANK * n)?;
            let mut gemv_partial = ctx.stream.alloc_zeros::<f32>(
                GLM52_MAX_BATCH_PER_RANK * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW,
            )?;
            for rows in GLM52_DECODE_BUCKETS {
                glm52_fp8_weight_only_gemv_launch(
                    ctx,
                    rows,
                    n,
                    k,
                    &activation,
                    &weight,
                    &scale,
                    Some(&mut gemv_partial),
                    &mut out,
                )
                .with_context(|| {
                    format!(
                        "GLM5.2 decode bucket {rows} has no batched GEMV instantiation \
                         (GLM52_DECODE_BUCKETS drifted from kBatchedGemvBatch* in glm52_moe_gemv.cu)"
                    )
                })?;
            }
        }

        Ok(Self {
            layers,
            caches,
            embed,
            final_norm,
            lm_head,
            buckets,
            block_table,
            blocks_per_slot,
            slot_mapping: ctx.stream.alloc_zeros::<i64>(batch)?,
            seq_lens: ctx.stream.alloc_zeros::<i32>(batch)?,
            cos_table: DeviceMatrix {
                data: cos_table_data,
                rows: GLM52_MAX_MODEL_LEN,
                cols: GLM52_ROPE_HALF,
            },
            sin_table: DeviceMatrix {
                data: sin_table_data,
                rows: GLM52_MAX_MODEL_LEN,
                cols: GLM52_ROPE_HALF,
            },
            positions: ctx.stream.alloc_zeros::<u32>(batch)?,
            cos: ctx.stream.alloc_zeros::<bf16>(batch * GLM52_ROPE_HALF)?,
            sin: ctx.stream.alloc_zeros::<bf16>(batch * GLM52_ROPE_HALF)?,
            token_ids: ctx.stream.alloc_zeros::<u32>(batch)?,
            speculated: None,
            device_positions: [0; GLM52_MAX_BATCH_PER_RANK],
        })
    }

    /// One lock-step step: feed `inputs[row]` = the `(token, position)` each
    /// forwarded row carries, return the greedy next-token id per ROW. Enters
    /// 75 MoE collectives — every other rank must be stepping concurrently
    /// WITH THE SAME BUCKET (`shape.bucket`): the coordinator agrees the
    /// bucket globally per step. Row `r` addresses the cache slot
    /// `shape.slots[r]`; a slot's span rows walk consecutive positions (see
    /// [`Glm52StepShape`]); padding rows' cache writes land in their free
    /// slot's own dead region.
    ///
    /// The step body (embed → 78 layers → lm_head → argmax) is captured into
    /// a CUDA graph on the first call in each (attention tier × bucket) shape
    /// and replayed afterwards: one graph launch instead of ~4155 kernel
    /// launches per rank per step. The prologue rewrites the device input
    /// buffers the captured kernels read (per-row rope rows, slots, seq_lens,
    /// tokens, and — for partial buckets — each forwarded row's block-table
    /// row), and the epilogue reads back the per-row argmax results — both
    /// outside the graph. Capture-time safety: stream capture records without
    /// executing, and in lock-step all ranks enter a new shape on the same
    /// global step, so the DeepEP collectives first execute together on every
    /// rank's first launch of that shape (the same argument as the kimi
    /// decode graph; the ceiling is the ~100 s DeepEP device timeout against
    /// a capture window of tens of ms — already proven by the mid-serving
    /// tier-crossing capture).
    pub(crate) fn decode_step(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        ep8: &mut Glm52MoeEp8State,
        inputs: &[(u32, usize); GLM52_MAX_BATCH_PER_RANK],
        shape: Glm52StepShape,
        flags: crate::runner::Glm52StepFlags,
    ) -> Result<[u32; GLM52_MAX_BATCH_PER_RANK]> {
        let batch = shape.bucket;
        if flags.consume {
            // Launch-ahead fast path: the coordinator says this step IS the
            // replay every rank speculatively enqueued last step. That claim
            // is global — a speculative replay is a full set of collectives,
            // so ranks must consume together or not at all. Any mismatch is
            // a protocol bug; failing the step beats a silent fallback that
            // would desync the collective pairing (measured as the ~100 s
            // DeepEP device-timeout trap).
            let speculated = self.speculated.take().context(
                "GLM5.2 launch-ahead desync: the coordinator consumed a speculation this rank \
                 never enqueued",
            )?;
            ensure!(
                speculated.bucket == batch
                    && speculated.active_rows == shape.active_rows
                    && speculated.slots[..batch] == shape.slots[..batch]
                    && speculated.expect[..batch] == inputs[..batch],
                "GLM5.2 launch-ahead desync: consumed speculation (bucket {}, slots {:?}, expect \
                 {:?}) does not match the step (bucket {batch}, slots {:?}, inputs {:?})",
                speculated.bucket,
                &speculated.slots[..speculated.bucket],
                &speculated.expect[..speculated.bucket],
                &shape.slots[..batch],
                &inputs[..batch],
            );
        } else {
            // Any stale speculation was enqueued by EVERY rank (the lease is
            // a global grant), so the stale replay's collectives pair up and
            // it degrades to a harmless recompute the prologue overwrites.
            self.speculated = None;
            self.decode_step_prologue_and_replay(ctx, aux, ep8, inputs, shape)?;
        }
        self.decode_step_harvest(ctx, inputs, shape, flags.lease)
    }

    /// The non-leased step path: validate the shape, rewrite every per-step
    /// device input buffer from the coordinator's `inputs`, and run (or lazily
    /// capture) the whole-step graph for the step's bucket × tier.
    fn decode_step_prologue_and_replay(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        ep8: &mut Glm52MoeEp8State,
        inputs: &[(u32, usize); GLM52_MAX_BATCH_PER_RANK],
        shape: Glm52StepShape,
    ) -> Result<()> {
        // The bucket state's `rows` is the lookup key — an unknown bucket is
        // a coordinator bug and fails the step before touching the GPU.
        let bucket = self
            .buckets
            .iter_mut()
            .find(|bucket| bucket.rows == shape.bucket)
            .with_context(|| {
                format!(
                    "GLM5.2 step bucket {} is not a member of {GLM52_DECODE_BUCKETS:?}",
                    shape.bucket
                )
            })?;
        let batch = shape.bucket;
        // A slot's rows must form ONE contiguous run of consecutive
        // positions: a gap would leave positions the later rows attend to
        // unwritten this step (stale data from whatever request last held the
        // slot), and a second run would re-enter a region the first already
        // wrote. Single-row slots are the trivial run.
        let mut slot_last_row = [None::<usize>; GLM52_MAX_BATCH_PER_RANK];
        for row in 0..batch {
            let slot = shape.slots[row] as usize;
            ensure!(
                slot < GLM52_MAX_BATCH_PER_RANK,
                "GLM5.2 step row {row} slot {slot} out of range in {:?}",
                &shape.slots[..batch]
            );
            match slot_last_row[slot] {
                None => {}
                Some(last) => {
                    ensure!(
                        last + 1 == row && inputs[last].1 + 1 == inputs[row].1,
                        "GLM5.2 step slot {slot} span is not one contiguous run of \
                         consecutive positions: rows {:?}, positions {:?}",
                        &shape.slots[..batch],
                        inputs[..batch].iter().map(|i| i.1).collect::<Vec<_>>()
                    );
                }
            }
            slot_last_row[slot] = Some(row);
        }
        let mut tokens_host = [0u32; GLM52_MAX_BATCH_PER_RANK];
        let mut positions_host = [0u32; GLM52_MAX_BATCH_PER_RANK];
        let mut slots_host = [0i64; GLM52_MAX_BATCH_PER_RANK];
        let mut seq_lens_host = [0i32; GLM52_MAX_BATCH_PER_RANK];
        for row in 0..batch {
            let slot = shape.slots[row] as usize;
            let (token, position) = inputs[row];
            ensure!(
                position < GLM52_MAX_MODEL_LEN,
                "GLM5.2 slot {slot} position {position} exceeds the model-length cap {GLM52_MAX_MODEL_LEN}"
            );
            tokens_host[row] = token;
            positions_host[row] = position as u32;
            // Slot b owns cache tokens [b*MAX_LEN, (b+1)*MAX_LEN) — the same
            // global slot id addresses both the MLA and index-K caches.
            slots_host[row] = (slot * GLM52_MAX_MODEL_LEN + position) as i64;
            seq_lens_host[row] = (position + 1) as i32;
        }
        ctx.stream.memcpy_htod(&tokens_host, &mut self.token_ids)?;
        ctx.stream
            .memcpy_htod(&positions_host, &mut self.positions)?;
        ctx.stream
            .memcpy_htod(&slots_host, &mut self.slot_mapping)?;
        ctx.stream.memcpy_htod(&seq_lens_host, &mut self.seq_lens)?;
        for (dst, &(_, position)) in self.device_positions.iter_mut().zip(&inputs[..batch]) {
            *dst = position;
        }
        // Gather each row's rotary table row (a bit-exact row copy).
        embedding_rows_into(ctx, &self.cos_table, &self.positions, batch, &mut self.cos)?;
        embedding_rows_into(ctx, &self.sin_table, &self.positions, batch, &mut self.sin)?;
        // Point each forwarded row's block-table row at its slot's cache
        // region — device data, so the captured graphs replay against
        // whichever slots hold the requests (span rows repeat their slot's
        // region). Every bucket's table is rewritten every step: the full
        // bucket stopped being an identity mapping once spans landed.
        for row in 0..batch {
            let slot = shape.slots[row] as usize;
            let src = self
                .block_table
                .slice(slot * self.blocks_per_slot..(slot + 1) * self.blocks_per_slot);
            let mut dst = bucket
                .block_table
                .slice_mut(row * self.blocks_per_slot..(row + 1) * self.blocks_per_slot);
            ctx.stream.memcpy_dtod(&src, &mut dst)?;
        }

        // Attention tier: while EVERY forwarded row's context fits in the
        // short top-k, top-256 selects exactly the same tokens as top-2048
        // (all of them) — the short graph only walks 1/8 of the padded
        // FlashMLA index slots. Pad rows sit at position 0 and never lift the
        // tier; a mixed-tier batch runs at the full tier (correct for every
        // row, only the padded walk length differs). Each shape has its own
        // captured graph; the lazily captured graph records without
        // executing, so a mid-serving capture holds the other ranks'
        // collectives for tens of ms — far under the ~100 s DeepEP device
        // timeout (same argument as the first-step capture).
        let short_tier = (0..batch)
            .map(|row| inputs[row].1 + 1)
            .max()
            .is_some_and(|longest| longest <= GLM52_MLA_TOPK_SHORT);
        let tier = if short_tier { TIER_SHORT } else { TIER_FULL };
        // The bucket state selected above carries the plan, scratch, graph,
        // and block table together — one coherent shape.
        let step = Glm52DecodeStep {
            mla_cos: &self.cos,
            mla_sin: &self.sin,
            idx_cos: &self.cos,
            idx_sin: &self.sin,
            mla_sched: &bucket.scheds[tier],
            slot_mapping: &self.slot_mapping,
            block_table: &bucket.block_table,
            seq_lens: &self.seq_lens,
        };
        // Every rank must pass the same global token count into the MoE
        // collectives — guaranteed by the coordinator agreeing the bucket.
        let global_tokens = crate::weights::GLM52_EP_RANKS * batch;

        let mut graph = std::mem::take(&mut bucket.graphs[tier]);
        let s = &mut bucket.scratch;
        let result = graph.run_or_capture(ctx, || {
            run_step_body(
                ctx,
                aux,
                ep8,
                &self.layers,
                &mut self.caches,
                &self.embed,
                &self.final_norm,
                &self.lm_head,
                &self.token_ids,
                &step,
                s,
                global_tokens,
            )
        });
        bucket.graphs[tier] = graph;
        result
    }
}

/// The captured region of one decode step: embed → 78 layers → lm_head →
/// device argmax over the step's `batch` rows (read from the attend plan —
/// the single source of truth for the step's row count). Shared verbatim by
/// both batch buckets; only the plan, scratch, block table, and
/// `global_tokens` differ per shape.
#[allow(clippy::too_many_arguments)]
fn run_step_body(
    ctx: &DeviceContext,
    aux: &DeviceContext,
    ep8: &mut Glm52MoeEp8State,
    layers: &[Glm52DecoderLayerWeights],
    caches: &mut [Glm52LayerCaches],
    embed: &DeviceMatrix,
    final_norm: &DeviceVec,
    lm_head: &DeviceMatrix,
    token_ids: &CudaSlice<u32>,
    step: &Glm52DecodeStep<'_>,
    s: &mut Glm52DecodeScratch,
    global_tokens: usize,
) -> Result<()> {
    let batch = step.mla_sched.batch();
    glm52_embed_into(ctx, embed, token_ids, &mut s.hidden)?;
    // Layer 0's input norm is standalone (the embedding is the residual);
    // every later layer's input norm is fused into the previous layer's
    // closing add (`glm52_layer_finish_fused`).
    rms_norm_rows_into(
        ctx,
        s.hidden.data(),
        &layers[0].input_ln,
        GLM52_RMS_EPS,
        GLM52_HIDDEN,
        batch,
        s.layer.normed.data_mut(),
    )?;
    let mut carry_ready = false;
    for (layer, (weights, cache)) in layers.iter().zip(caches.iter_mut()).enumerate() {
        let parity = layer % 2;
        glm52_layer_attention_half(
            ctx,
            Some(aux),
            weights,
            cache,
            step,
            s,
            &mut carry_ready,
            parity,
            layer == 0,
        )
        .with_context(|| format!("GLM5.2 layer {layer} attention half"))?;
        match &weights.mlp {
            Glm52LayerMlp::Dense(dense) => glm52_dense_mlp_forward_into(
                ctx,
                dense,
                s.layer.normed2.data(),
                &mut s.dense_mlp,
                s.layer.mlp_out.data_mut(),
            )?,
            Glm52LayerMlp::MoeEp8(moe) => {
                // Fork: the shared expert only needs `normed2`, so it runs on
                // the aux stream concurrently with the routed path's
                // dispatch/grouped-GEMM/combine — the cooperative collectives
                // occupy a fixed SM slice and mostly wait on peers, leaving
                // the rest of the GPU free. The events recorded here during
                // capture become graph edges; replay keeps the parallel
                // branches.
                let normed_ready = ctx.stream.record_event(None)?;
                aux.stream.wait(&normed_ready)?;
                moe.shared.forward_into(
                    aux,
                    s.layer.normed2.data(),
                    &mut s.shared_mlp,
                    s.layer.shared_out.data_mut(),
                )?;
                let shared_done = aux.stream.record_event(None)?;

                run_router_into(ctx, &moe.router, s.layer.normed2.data(), &mut s.router)?;
                let dispatched = glm52_moe_ep8_routed_forward(
                    ctx,
                    ep8,
                    &moe.bank,
                    Some((s.layer.normed2.data(), &s.router.route, batch)),
                    global_tokens,
                )
                .with_context(|| format!("GLM5.2 layer {layer} EP8 MoE"))?;
                ensure!(
                    dispatched,
                    "EP8 MoE returned no combined output for the dispatched rows"
                );
                // Join: the closing add consumes both branches.
                ctx.stream.wait(&shared_done)?;
                add_into(
                    ctx,
                    ep8.combined(),
                    s.layer.shared_out.data(),
                    batch * GLM52_HIDDEN,
                    s.layer.mlp_out.data_mut(),
                )?;
            }
            #[cfg(test)]
            Glm52LayerMlp::Moe(_) => {
                anyhow::bail!("GLM5.2 EP8 spine built an EP1 MoE layer — loader bug")
            }
        }
        if layer + 1 < layers.len() {
            glm52_layer_finish_fused(ctx, s, parity, &layers[layer + 1].input_ln)?;
        } else {
            glm52_layer_finish(ctx, s, parity)?;
        }
        // DSpark aux-hidden capture: after layer L's closing add the residual
        // stream lives in `attn[parity]` (updated in place by the fused
        // add+norm; none of the capture layers is the last layer, which lands
        // in `s.hidden` instead). Recorded into the step graph — pointer-
        // stable, ~60 KB/row per step.
        if let Some(feature) = crate::dspark::GLM52_DSPARK_AUX_LAYERS
            .iter()
            .position(|&aux| aux == layer)
        {
            copy_hidden_rows_raw_into(
                ctx,
                s.layer.attn[parity].data(),
                GLM52_HIDDEN,
                s.captured.data_mut(),
                crate::dspark::GLM52_DSPARK_CONTEXT_DIM,
                feature * GLM52_HIDDEN,
                batch,
            )?;
        }
    }

    glm52_final_norm_into(ctx, &s.hidden, final_norm, &mut s.final_normed)?;
    glm52_lm_head_into(ctx, &s.final_normed, lm_head, &mut s.logits)?;
    // Device greedy argmax per row (same semantics as a host scan: lowest
    // index wins ties, NaN never wins) — the step's egress shrinks from the
    // full vocab rows to 6 bytes per row, and the kernel chain ends on-device
    // (the graph boundary). Two-stage: per-4096-tile partials in parallel,
    // then one finalize block per row — bit-identical to the single-block
    // scan (the partials carry global indices, same total order), and each
    // row's result is independent of its slot-mates.
    argmax_bf16_split_into(
        ctx,
        s.logits.data(),
        batch,
        GLM52_VOCAB,
        &mut s.argmax_partial_values,
        &mut s.argmax_partial_indices,
        &mut s.argmax_values,
        &mut s.argmax_indices,
    )
}
