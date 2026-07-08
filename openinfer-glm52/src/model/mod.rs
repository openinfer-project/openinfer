//! GLM5.2 DP8/EP8 full-model decode: every rank owns the whole non-expert
//! path (embed → 78 decoder layers → final norm → lm_head → greedy argmax)
//! plus its 32 local experts, and forwards one of the
//! [`GLM52_DECODE_BUCKETS`] batch buckets per step (real request tokens in
//! occupied slots, padding rows elsewhere; prefill rides decode as *spans* —
//! several consecutive positions of one slot in a single step). KV lives in
//! a rank-wide pool of 64-token pages: the coordinator's `BlockPool` assigns
//! pages per request, and every step's [`Glm52StepKv`] carries each row's
//! page table row plus its flat cache write slot — padding rows ride the
//! pool's reserved padding page, whose garbage writes nobody reads.
//!
//! Every step, all 8 ranks run the forward in lock-step with the SAME bucket
//! (the coordinator's global decision), each dispatching exactly that many
//! rows into every MoE layer's DeepEP collective — the collectives require
//! all ranks to enter in the same layer order 3..=77 with the agreed global
//! row count, and the per-bucket fixed row count keeps every step's kernel
//! shapes identical within a bucket (the whole-step CUDA graphs' contract).

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr as _, PinnedHostSlice};
use half::bf16;
use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN, GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
    GLM52_FLASHMLA_SPARSE_TOPK, GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW, Glm52FlashMlaSparseDecode,
    Glm52IndexerCacheLayout, add_into, argmax_bf16_split_into, copy_hidden_rows_raw_into,
    embedding_rows_into, glm52_flashmla_sparse_decode_num_sm_parts,
    glm52_fp8_weight_only_gemv_launch, rms_norm_rows_into,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStatesRef};
use openinfer_kv_offload::KvArena;
use openinfer_sample::{
    BatchSamplingRow, BatchSamplingScratch, effectively_greedy, gpu_sample_batch_into, mix_seed,
};

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
use crate::moe_ep8::{Glm52MoeEp8LayerWeights, Glm52MoeEp8State, glm52_moe_ep8_routed_forward};
use crate::moe_tp8::Glm52MoeTp8Rank;
use crate::scratch::Glm52DecodeScratch;
use crate::weights::{Glm52RankGpuWeights, retype_owned};

mod build;
mod launch_ahead;
use launch_ahead::Glm52SpeculatedStep;

/// The per-rank slot count and the largest decode bucket. A slot is a batch
/// lane (and the draft lane's state key), not a cache region — KV pages come
/// from the rank's shared pool.
pub(crate) const GLM52_MAX_BATCH_PER_RANK: usize = 8;

/// Cache geometry is carved in units of the 64-token FlashMLA page (== the
/// index-K cache block), so the per-request context cap must sit on a page
/// boundary — the page-table width is `max_model_len / page`, and a
/// non-multiple cap would strand a partial page the table cannot address.
pub(crate) const GLM52_MODEL_LEN_ALIGN: usize = GLM52_FLASHMLA_SPARSE_PAGE_SIZE;

/// The rank-wide KV pool size for a given per-request cap: capacity for
/// every slot's full-lifetime draw plus the reserved padding page
/// (`BlockPool` block 0 — padding rows and CUDA-graph pre-capture write
/// there). A request's lifetime draw is `ceil((prompt + max_tokens)/page)`
/// with `prompt + max_tokens <= cap + 1` (`validate_request`) — one page
/// more than its KV ever writes, because kvbm appends the final generated
/// token and eagerly provisions its page (the dangling-token contract). The
/// `cap + 1` here keeps 8 concurrent max-shape requests admissible, exactly
/// like the pre-pool per-slot layout. The coordinator's `BlockPool` and the
/// rank arenas ([`Glm52RankModel::build`]) MUST agree on this count: pool
/// block ids index the arenas directly.
pub(crate) fn glm52_pool_blocks(max_model_len: usize) -> usize {
    GLM52_MAX_BATCH_PER_RANK * (max_model_len + 1).div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE) + 1
}

/// Page-table width: the pages a single request at the full cap addresses.
/// Every per-row page table (bucket block tables) is this wide; rows with
/// fewer pages are padded with the padding page id.
pub(crate) fn glm52_table_width(max_model_len: usize) -> usize {
    max_model_len.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE)
}

/// Exact GPU bytes [`Glm52RankModel::build`] allocates on a rank for a given
/// context cap — every `max_model_len`-scaled term of the build, computed
/// from the same layout formulas the allocations use (the FlashMLA packed
/// cache and index-K layouts, the device rope tables, the per-bucket indexer
/// logits scratch with its 256-rounded stride, and the block tables). The
/// launch-time VRAM probe sizes `max_model_len` against this, so a new
/// len-scaled allocation in `build` MUST be added here or the probe
/// under-charges.
pub(crate) fn glm52_arena_bytes(max_model_len: usize) -> Result<usize> {
    let num_blocks = glm52_pool_blocks(max_model_len);
    let mla = GLM52_LAYERS
        * num_blocks
        * GLM52_FLASHMLA_SPARSE_PAGE_SIZE
        * GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN;
    let (table_width, index_layout) = glm52_index_cache_layout(max_model_len);
    let index_k = (0..GLM52_LAYERS)
        .filter(|&layer| glm52_layer_has_full_indexer(layer))
        .count()
        * index_layout.min_cache_bytes()?;
    let rope_tables = 2 * max_model_len * GLM52_ROPE_HALF * size_of::<bf16>();
    let bucket_rows: usize = GLM52_DECODE_BUCKETS.iter().sum();
    let indexer_logits =
        bucket_rows * max_model_len.next_multiple_of(256) * (size_of::<bf16>() + size_of::<f32>());
    let block_tables = bucket_rows * table_width * size_of::<i32>();
    Ok(mla + index_k + rope_tables + indexer_logits + block_tables)
}

/// The page-table width and index-K cache layout for a given cap — the ONE
/// construction shared by [`Glm52RankModel::build`] and the arena ledger
/// ([`glm52_arena_bytes`]), so a layout change cannot drift between them.
fn glm52_index_cache_layout(max_model_len: usize) -> (usize, Glm52IndexerCacheLayout) {
    // The index-K cache is indexed by the same pool block ids as the MLA
    // cache, so it holds the same block count.
    let layout = Glm52IndexerCacheLayout {
        cache_blocks: glm52_pool_blocks(max_model_len),
        cache_block_size: INDEX_CACHE_BLOCK,
        cache_block_stride_bytes: INDEX_CACHE_BLOCK * (GLM52_INDEX_HEAD_DIM + 4),
    };
    (glm52_table_width(max_model_len), layout)
}

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

/// The step's KV paging, decided by the coordinator's per-rank `BlockPool`:
/// where each forwarded row's cache writes land and which pages its
/// attention/indexer walk. Uploaded by the step prologue into the bucket's
/// device block table / slot mapping (the captured graphs read only those
/// device buffers — a page's physical id is data, never baked).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StepKv {
    /// `[bucket, table_width]` row-major page ids. Row `r` holds the pages
    /// covering its request's KV through this step (span rows repeat their
    /// slot's row); entries past the covered pages — and every padding row —
    /// are the pool's padding page id.
    pub(crate) pages: Box<[i32]>,
    /// Per-row flat cache write slot: `pages[position/64]*64 + position%64`
    /// (the fp8_ds_mla packed cache and the index-K cache share this token
    /// index space). Padding rows point into the padding page.
    pub(crate) slot_mapping: [i64; GLM52_MAX_BATCH_PER_RANK],
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
    /// Width of every per-row page-table row ([`glm52_table_width`]): the
    /// pages one request at the full cap addresses.
    table_width: usize,
    /// Per-request context cap: `prompt + max_tokens - 1 <= max_model_len`.
    /// Decided at launch (VRAM probe or `--max-model-len`); sized the
    /// rank-wide per-layer MLA and index-K page pools at build time
    /// ([`glm52_pool_blocks`]).
    max_model_len: usize,
    /// Built with `--moe-topo tp8`: every MoE arm is `MoeTp8`, bucket-8
    /// steps are span steps (all 8 rows one owner rank), and the
    /// coordinator must stage the span owner on every such step.
    slot_mapping: CudaSlice<i64>,
    seq_lens: CudaSlice<i32>,
    /// Device-resident rope tables for every position (`[max_model_len,
    /// GLM52_ROPE_HALF]` as a gatherable matrix); the prologue gathers each row's
    /// position row instead of recomputing on the host and copying up.
    cos_table: DeviceMatrix,
    sin_table: DeviceMatrix,
    positions: CudaSlice<u32>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    token_ids: CudaSlice<u32>,
    /// FlashInfer batch-sampling buffers for the non-greedy rows, sized for
    /// the max bucket × vocab and shared by every bucket (the sampling pass
    /// runs outside the captured graphs, so pointer stability per bucket is
    /// not required). Allocated at build — a mid-serving step must never hit
    /// the allocator.
    sampling_scratch: BatchSamplingScratch,
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
/// rewrites), and the `[rows, table_width]` block table, uploaded by the
/// prologue every step from the coordinator's [`Glm52StepKv`] page rows, so
/// the captured graphs address whichever pool pages hold the requests —
/// and span rows their repeated slot's row — through device data, never
/// baked page ids.
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

    /// The per-request context cap this rank's cache arenas were built for.
    pub(crate) fn max_model_len(&self) -> usize {
        self.max_model_len
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

    /// The per-layer cache arenas this rank registers with the KV offload
    /// tier: one `glm52.L{n}.mla` arena per layer plus a `glm52.L{n}.idxk`
    /// sidecar on the full-indexer layers, all indexed by the same pool block
    /// ids. Registering both under one instance makes every save/load move a
    /// block's MLA page and its index-K slice together — an MLA page restored
    /// without its index-K would be silent corruption. The arenas are
    /// contiguous, so a block's stride equals its copy size.
    pub(crate) fn kv_arenas(&self, stream: &CudaStream) -> Result<Vec<KvArena>> {
        let num_blocks = glm52_pool_blocks(self.max_model_len);
        let mla_block_bytes =
            GLM52_FLASHMLA_SPARSE_PAGE_SIZE * GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN;
        let idxk_block_bytes = INDEX_CACHE_BLOCK * (GLM52_INDEX_HEAD_DIM + 4);
        let mut arenas = Vec::with_capacity(self.caches.len() * 2);
        for (layer, caches) in self.caches.iter().enumerate() {
            ensure!(
                caches.mla_cache.len() == num_blocks * mla_block_bytes,
                "GLM5.2 layer {layer} MLA arena is {} bytes, expected \
                 {num_blocks} blocks x {mla_block_bytes}",
                caches.mla_cache.len(),
            );
            let (base_ptr, _sync) = caches.mla_cache.device_ptr(stream);
            arenas.push(KvArena {
                name: format!("glm52.L{layer}.mla"),
                base_ptr,
                num_blocks,
                bytes_per_block: mla_block_bytes,
                block_stride_bytes: mla_block_bytes,
            });
            if let Some(index_k) = &caches.index_k_cache {
                ensure!(
                    index_k.len() == num_blocks * idxk_block_bytes,
                    "GLM5.2 layer {layer} index-K arena is {} bytes, expected \
                     {num_blocks} blocks x {idxk_block_bytes}",
                    index_k.len(),
                );
                let (base_ptr, _sync) = index_k.device_ptr(stream);
                arenas.push(KvArena {
                    name: format!("glm52.L{layer}.idxk"),
                    base_ptr,
                    num_blocks,
                    bytes_per_block: idxk_block_bytes,
                    block_stride_bytes: idxk_block_bytes,
                });
            }
        }
        Ok(arenas)
    }

    pub(crate) fn build(
        ctx: &DeviceContext,
        w: &mut Glm52RankGpuWeights,
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        attn_shard: Option<usize>,
    ) -> Result<Self> {
        ensure!(
            (moe_topo == crate::Glm52MoeTopo::Tp8) == attn_shard.is_some(),
            "GLM5.2 attention-TP shard must ride the tp8 topology (topo {moe_topo:?}, \
             shard {attn_shard:?})"
        );
        ensure!(
            max_model_len > 0 && max_model_len.is_multiple_of(GLM52_MODEL_LEN_ALIGN),
            "GLM5.2 max_model_len {max_model_len} must be a positive multiple of \
             {GLM52_MODEL_LEN_ALIGN} (the FlashMLA page / index-K block size)"
        );
        let batch = GLM52_MAX_BATCH_PER_RANK;
        let num_sm_parts = glm52_flashmla_sparse_decode_num_sm_parts()?;
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: batch,
            num_blocks: glm52_pool_blocks(max_model_len),
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
        let (table_width, index_cache_layout) = glm52_index_cache_layout(max_model_len);

        let mut layers = Vec::with_capacity(GLM52_LAYERS);
        let mut caches = Vec::with_capacity(GLM52_LAYERS);
        for layer in 0..GLM52_LAYERS {
            layers.push(
                build::build_decoder_layer(ctx, w, layer, moe_topo, attn_shard)
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
        // but out of campaign scope — drop this rank's copy. The TP8 bundle
        // never loads routed experts, so there is nothing to drop there.
        if moe_topo == crate::Glm52MoeTopo::Ep8 {
            let _ = w.take_expert_layer(crate::weights::GLM52_MTP_LAYER)?;
        }

        let mut cos_host = Vec::with_capacity(max_model_len * GLM52_ROPE_HALF);
        let mut sin_host = Vec::with_capacity(max_model_len * GLM52_ROPE_HALF);
        for position in 0..max_model_len {
            let (cos_row, sin_row) = rope_tables(position);
            cos_host.extend_from_slice(&cos_row);
            sin_host.extend_from_slice(&sin_row);
        }
        let mut cos_table_data = ctx
            .stream
            .alloc_zeros::<bf16>(max_model_len * GLM52_ROPE_HALF)?;
        let mut sin_table_data = ctx
            .stream
            .alloc_zeros::<bf16>(max_model_len * GLM52_ROPE_HALF)?;
        ctx.stream.memcpy_htod(&cos_host, &mut cos_table_data)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin_table_data)?;

        // One Glm52BucketState per decode bucket: batch-`rows` contracts
        // (num_blocks is cache geometry, not batch, so it carries over),
        // plans, scratch, and a zeroed block table (never read before the
        // first step prologue uploads the coordinator's page rows).
        // Attention-TP: the shard keeps 8 of 64 heads per rank; every scratch
        // buffer with a head dimension shrinks with it.
        let mla_heads = if attn_shard.is_some() {
            crate::config::GLM52_HEADS / 8
        } else {
            crate::config::GLM52_HEADS
        };
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
                table_width,
                NUM_SMS,
                max_model_len,
            );
            let bucket_table = ctx.stream.alloc_zeros::<i32>(rows * table_width)?;
            buckets.push(Glm52BucketState {
                rows,
                scheds: [
                    Glm52MlaSchedMetadata::new(ctx, contract_rows)?,
                    Glm52MlaSchedMetadata::new(ctx, contract_rows_short)?,
                ],
                scratch: Glm52DecodeScratch::new(ctx, &contract_rows, mqa_shape, mla_heads)?,
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
            table_width,
            max_model_len,
            slot_mapping: ctx.stream.alloc_zeros::<i64>(batch)?,
            seq_lens: ctx.stream.alloc_zeros::<i32>(batch)?,
            cos_table: DeviceMatrix {
                data: cos_table_data,
                rows: max_model_len,
                cols: GLM52_ROPE_HALF,
            },
            sin_table: DeviceMatrix {
                data: sin_table_data,
                rows: max_model_len,
                cols: GLM52_ROPE_HALF,
            },
            positions: ctx.stream.alloc_zeros::<u32>(batch)?,
            cos: ctx.stream.alloc_zeros::<bf16>(batch * GLM52_ROPE_HALF)?,
            sin: ctx.stream.alloc_zeros::<bf16>(batch * GLM52_ROPE_HALF)?,
            token_ids: ctx.stream.alloc_zeros::<u32>(batch)?,
            sampling_scratch: BatchSamplingScratch::new(ctx, batch, GLM52_VOCAB)?,
            speculated: None,
            device_positions: [0; GLM52_MAX_BATCH_PER_RANK],
        })
    }

    /// One lock-step step: feed `inputs[row]` = the `(token, position)` each
    /// forwarded row carries, return the next-token id per ROW (the fused
    /// greedy argmax, overwritten for the coordinator's `sampling` rows by a
    /// post-graph FlashInfer sampling pass — see [`Self::sample_rows_into`]).
    /// Enters
    /// 75 MoE collectives — every other rank must be stepping concurrently
    /// WITH THE SAME BUCKET (`shape.bucket`): the coordinator agrees the
    /// bucket globally per step. Row `r` writes and reads KV through `kv`'s
    /// page row / slot mapping; a slot's span rows walk consecutive positions
    /// (see [`Glm52StepShape`]); padding rows' cache writes land in the
    /// pool's padding page, which nobody reads meaningfully.
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_step(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        ep8: &mut Glm52MoeEp8State,
        tp8: Option<&mut Glm52MoeTp8Rank>,
        inputs: &[(u32, usize); GLM52_MAX_BATCH_PER_RANK],
        shape: Glm52StepShape,
        kv: &Glm52StepKv,
        flags: crate::runner::Glm52StepFlags,
        sampling: &[crate::runner::Glm52RowSample],
        seed: u64,
    ) -> Result<[u32; GLM52_MAX_BATCH_PER_RANK]> {
        // A launch-ahead speculation feeds this step's ARGMAX token to the
        // next step, so it can never coexist with a sampled row — the
        // coordinator withholds the lease while any non-greedy request is
        // active, and a violation here is a protocol bug.
        ensure!(
            sampling.is_empty() || (!flags.consume && !flags.lease),
            "GLM5.2 sampling rows cannot ride a launch-ahead step (the speculation feeds the \
             argmax token, not the sampled one)"
        );
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
            self.decode_step_prologue_and_replay(ctx, aux, ep8, tp8, inputs, shape, kv)?;
        }
        let mut outputs = self.decode_step_harvest(ctx, inputs, shape, flags.lease)?;
        self.sample_rows_into(ctx, shape, sampling, seed, &mut outputs)?;
        Ok(outputs)
    }

    /// Overwrite the sampled rows' tokens: a non-greedy request's committed
    /// row takes a FlashInfer temperature/top-k/top-p/min_p pass over the
    /// step's logits instead of the fused argmax. Unseeded rows ride one
    /// batched call under the step seed; a seeded row is its own single-row
    /// call with `mix_seed(request_seed, step)` — the same replayable-stream
    /// contract as `openinfer_sample::select_batch`.
    fn sample_rows_into(
        &mut self,
        ctx: &DeviceContext,
        shape: Glm52StepShape,
        sampling: &[crate::runner::Glm52RowSample],
        seed: u64,
        outputs: &mut [u32; GLM52_MAX_BATCH_PER_RANK],
    ) -> Result<()> {
        if sampling.is_empty() {
            return Ok(());
        }
        for pair in sampling.windows(2) {
            ensure!(
                pair[0].row < pair[1].row,
                "GLM5.2 sampling rows must be strictly ascending: {sampling:?}"
            );
        }
        for s in sampling {
            ensure!(
                s.row < shape.active_rows,
                "GLM5.2 sampling row {} outside the step's {} active rows",
                s.row,
                shape.active_rows
            );
            ensure!(
                !effectively_greedy(&s.params, GLM52_VOCAB),
                "GLM5.2 effectively-greedy row {} routed to the sampler (coordinator bug)",
                s.row
            );
        }
        let bucket = self
            .buckets
            .iter()
            .find(|bucket| bucket.rows == shape.bucket)
            .expect("decode_step validated the bucket");
        let logits = HiddenStatesRef {
            data: bucket.scratch.logits.data(),
            hidden_dim: GLM52_VOCAB,
            seq_len: shape.bucket,
        };
        let as_row = |s: &crate::runner::Glm52RowSample| BatchSamplingRow {
            row: s.row,
            temperature: s.params.temperature,
            top_k: s.params.top_k,
            top_p: s.params.top_p,
            min_p: s.params.min_p,
        };
        let unseeded: Vec<BatchSamplingRow> = sampling
            .iter()
            .filter(|s| s.params.seed.is_none())
            .map(as_row)
            .collect();
        if !unseeded.is_empty() {
            let tokens =
                gpu_sample_batch_into(ctx, logits, &unseeded, seed, &mut self.sampling_scratch)?;
            for (row, token) in unseeded.iter().zip(tokens) {
                outputs[row.row] = token;
            }
        }
        for s in sampling {
            let Some(request_seed) = s.params.seed else {
                continue;
            };
            let tokens = gpu_sample_batch_into(
                ctx,
                logits,
                &[as_row(s)],
                mix_seed(request_seed, s.step),
                &mut self.sampling_scratch,
            )?;
            outputs[s.row] = tokens[0];
        }
        Ok(())
    }

    /// The non-leased step path: validate the shape, rewrite every per-step
    /// device input buffer from the coordinator's `inputs`, and run (or lazily
    /// capture) the whole-step graph for the step's bucket × tier.
    #[allow(clippy::too_many_arguments)]
    fn decode_step_prologue_and_replay(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        ep8: &mut Glm52MoeEp8State,
        mut tp8: Option<&mut Glm52MoeTp8Rank>,
        inputs: &[(u32, usize); GLM52_MAX_BATCH_PER_RANK],
        shape: Glm52StepShape,
        kv: &Glm52StepKv,
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
        ensure!(
            kv.pages.len() == batch * self.table_width,
            "GLM5.2 step KV pages {} != bucket {batch} x table width {}",
            kv.pages.len(),
            self.table_width
        );
        let mut tokens_host = [0u32; GLM52_MAX_BATCH_PER_RANK];
        let mut positions_host = [0u32; GLM52_MAX_BATCH_PER_RANK];
        let mut seq_lens_host = [0i32; GLM52_MAX_BATCH_PER_RANK];
        for row in 0..batch {
            let slot = shape.slots[row] as usize;
            let (token, position) = inputs[row];
            ensure!(
                position < self.max_model_len,
                "GLM5.2 slot {slot} position {position} exceeds the model-length cap {}",
                self.max_model_len
            );
            // The coordinator's page row must place this row's write slot
            // inside the page covering its position — a drifted slot mapping
            // would write one row's KV into another request's page.
            let page =
                kv.pages[row * self.table_width + position / GLM52_FLASHMLA_SPARSE_PAGE_SIZE];
            let expect = page as i64 * GLM52_FLASHMLA_SPARSE_PAGE_SIZE as i64
                + (position % GLM52_FLASHMLA_SPARSE_PAGE_SIZE) as i64;
            ensure!(
                kv.slot_mapping[row] == expect,
                "GLM5.2 row {row} slot mapping {} does not match page {page} at position \
                 {position} (expect {expect})",
                kv.slot_mapping[row]
            );
            tokens_host[row] = token;
            positions_host[row] = position as u32;
            seq_lens_host[row] = (position + 1) as i32;
        }
        ctx.stream.memcpy_htod(&tokens_host, &mut self.token_ids)?;
        ctx.stream
            .memcpy_htod(&positions_host, &mut self.positions)?;
        ctx.stream
            .memcpy_htod(&kv.slot_mapping, &mut self.slot_mapping)?;
        ctx.stream.memcpy_htod(&seq_lens_host, &mut self.seq_lens)?;
        for (dst, &(_, position)) in self.device_positions.iter_mut().zip(&inputs[..batch]) {
            *dst = position;
        }
        // Gather each row's rotary table row (a bit-exact row copy).
        embedding_rows_into(ctx, &self.cos_table, &self.positions, batch, &mut self.cos)?;
        embedding_rows_into(ctx, &self.sin_table, &self.positions, batch, &mut self.sin)?;
        // Upload the step's page rows into the bucket's device block table —
        // device data, so the captured graphs replay against whichever pool
        // pages hold the requests (span rows repeat their slot's row, padding
        // rows ride the padding page).
        ctx.stream
            .memcpy_htod(&kv.pages[..], &mut bucket.block_table)?;
        // Want-mask for the TP8 kernels: pad rows (>= active_rows, a prefix
        // by plan construction) skip the LL wire and shrink the expert union.
        // Every rank stages the same value — the coordinator mirrors the
        // step, and LL push/wait symmetry depends on it. A leased replay
        // (consume path) skips this prologue, which is safe: the lease
        // guarantees the identical shape, so the staged value still holds.
        if let Some(rank) = tp8.as_deref_mut() {
            if !rank.slices.is_empty() {
                rank.state.stage_active_rows(ctx, shape.active_rows)?;
            }
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
                tp8,
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
    mut tp8: Option<&mut Glm52MoeTp8Rank>,
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
    // TP8 step head: advance the shared LL epoch exactly once per replayed
    // step that runs TP8 kernels (all TP8 layers of the step share the tag;
    // per-layer slot regions alternate parity across steps).
    if let Some(rank) = tp8.as_deref_mut() {
        if !rank.slices.is_empty() {
            rank.state.advance_epoch(ctx)?;
        }
    }
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
        // Attention-TP: a head-sharded layer's o_proj partial crosses the AR
        // brick inside the attention half; the layer index is its AR slot.
        let tp8_ar = if weights.mla.heads != crate::config::GLM52_HEADS {
            let rank = tp8
                .as_deref_mut()
                .context("GLM5.2 sharded attention without TP8 state")?;
            Some((&mut rank.state, layer))
        } else {
            None
        };
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
            tp8_ar,
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
                glm52_moe_ep8_layer(ctx, aux, ep8, moe, s, batch, global_tokens)
                    .with_context(|| format!("GLM5.2 layer {layer} EP8 MoE"))?;
            }
            Glm52LayerMlp::MoeTp8(router) => {
                // TP8 topology: every MoE layer runs the replicated
                // phase-kernel chain over ALL 8 global rows — the topology
                // serves exactly one shape (bucket-8; pad rows ride free
                // slots). Any other bucket is a scheduler bug.
                ensure!(
                    batch == GLM52_MAX_BATCH_PER_RANK,
                    "GLM5.2 TP8 topology stepped at bucket {batch} — replicated activations \
                     serve the single bucket-{GLM52_MAX_BATCH_PER_RANK} shape"
                );
                let (state, slot, bank) = tp8
                    .as_deref_mut()
                    .and_then(|rank| rank.layer_bank(layer))
                    .with_context(|| {
                        format!("GLM5.2 TP8 layer {layer} has no slice bank — loader drifted")
                    })?;
                // Every rank routes all rows locally — bit-identical across
                // ranks (same kernel, same replicated normed2), so the
                // kernel's union and prob table need no routing exchange.
                run_router_into(ctx, router, s.layer.normed2.data(), &mut s.router)?;
                state
                    .forward(
                        ctx,
                        slot,
                        bank,
                        s.layer.normed2.data(),
                        &s.router.route.topk_idx,
                        &s.router.route.topk_weight,
                        s.layer.mlp_out.data_mut(),
                    )
                    .with_context(|| format!("GLM5.2 layer {layer} TP8 MoE"))?;
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

/// One layer's EP8 MoE half: shared expert forked to the aux stream, routed
/// path through router + DeepEP dispatch/grouped-GEMM/combine, joined by the
/// closing add into `mlp_out`. The events recorded here during capture
/// become graph edges; replay keeps the parallel branches.
fn glm52_moe_ep8_layer(
    ctx: &DeviceContext,
    aux: &DeviceContext,
    ep8: &mut Glm52MoeEp8State,
    moe: &Glm52MoeEp8LayerWeights,
    s: &mut Glm52DecodeScratch,
    batch: usize,
    global_tokens: usize,
) -> Result<()> {
    // Fork: the shared expert only needs `normed2`, so it runs on the aux
    // stream concurrently with the routed path's dispatch/grouped-GEMM/
    // combine — the cooperative collectives occupy a fixed SM slice and
    // mostly wait on peers, leaving the rest of the GPU free.
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
    )?;
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
    )
}
