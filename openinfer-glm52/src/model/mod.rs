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

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr as _;
use cudarc::driver::PinnedHostSlice;
use half::bf16;
use openinfer_core::cuda_graph::CudaGraphDumpSummary;
use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN;
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_TOPK;
use openinfer_kernels::ops::GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW;
use openinfer_kernels::ops::GLM52_MLA_CACHE_BYTES;
use openinfer_kernels::ops::Glm52FlashMlaSparseDecode;
use openinfer_kernels::ops::Glm52IndexerCacheLayout;
use openinfer_kernels::ops::Glm52VllmFixupKind;
use openinfer_kernels::ops::embedding_rows_into;
use openinfer_kernels::ops::glm52_flashmla_sparse_decode_num_sm_parts;
use openinfer_kernels::ops::glm52_fp8_weight_only_gemv_launch;
use openinfer_kernels::ops::glm52_vllm_rope_fixup_launch;
use openinfer_kernels::tensor::DeviceContext;
use openinfer_kernels::tensor::DeviceMatrix;
use openinfer_kernels::tensor::DeviceVec;
use openinfer_kernels::tensor::HiddenStatesRef;
use openinfer_kv_offload::KvArena;
use openinfer_sample::BatchSamplingRow;
use openinfer_sample::BatchSamplingScratch;
use openinfer_sample::effectively_greedy;
use openinfer_sample::gpu_sample_batch_into;
use openinfer_sample::mix_seed;

use crate::bookend::glm52_lm_head_into;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_INDEX_HEAD_DIM;
use crate::config::GLM52_INDEX_TOPK;
use crate::config::GLM52_LAYERS;
use crate::config::GLM52_ROPE_HALF;
use crate::config::GLM52_SM_SCALE;
use crate::config::GLM52_VOCAB;
use crate::config::glm52_layer_has_full_indexer;
use crate::indexer::Glm52IndexerScratch;
use crate::layer::Glm52DecodeStep;
use crate::layer::Glm52DecoderLayerWeights;
use crate::layer::Glm52LayerCaches;
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::mla_decode::glm52_mla_backend_preflight;
use crate::mla_decode::glm52_select_mla_backend;
use crate::moe_ep_wo::Glm52MoeEpState;
use crate::moe_tp::Glm52MoeTpRank;
use crate::scratch::Glm52DecodeScratch;
use crate::weights::Glm52RankGpuWeights;
use crate::weights::retype_owned;

mod build;
mod launch_ahead;
mod step_body;
use launch_ahead::Glm52SpeculatedStep;
use step_body::run_step_body;

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

// The min-latency GEMV (router logits, indexer weights_proj) dispatches
// tokens 1..=8; a bucket bump must extend glm52_min_gemv.cuh first.
const _: () =
    assert!(GLM52_MAX_BATCH_PER_RANK <= openinfer_kernels::ops::GLM52_MIN_GEMV_MAX_TOKENS);

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
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
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

// There used to be a short-context attention tier here (topk 256 while every
// row's context fit in it — lossless, 1/8 the index walk). Dropped: the
// serving traffic is agent workloads whose contexts start well past 2048, so
// the tier was dead weight — 2x the pre-captured graphs and a second MLA
// schedule per bucket. To bring it back, restore the (bucket x tier) arrays
// from git history; both decode backends already accept any topk multiple
// of 64, and the checked-in FlashInfer cubin closure covers topk 256.

// The attention `topk` feeds the DSA indexer's top-k selection, whose
// buffers are sized for GLM52_INDEX_TOPK rows — pin the range here so the
// indexer forward never needs to re-check it per layer per step.
const _: () = assert!(GLM52_FLASHMLA_SPARSE_TOPK > 0);
const _: () = assert!(GLM52_FLASHMLA_SPARSE_TOPK <= GLM52_INDEX_TOPK);

/// DeepGEMM paged MQA requires BLOCK_KV=64 — a kernel constraint, not a
/// model property (kept here, not in config.rs).
pub(crate) const INDEX_CACHE_BLOCK: usize = 64;
/// The DeepGEMM MQA indexer's persistent-grid size. 132 is the H200 SM count
/// and is baked into the AOT instantiation (`kAotNumSms` in
/// glm52_deepgemm_mqa.cu, enforced by its `num_sms == kAotNumSms` gate), so
/// it deliberately stays 132 on 152-SM GB300 — the schedule metadata drives
/// correctness; "fixing" this to the live SM count breaks the AOT gate.
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
    /// Full vocabulary head retained for DSpark and non-greedy sampling.
    lm_head: DeviceMatrix,
    /// Contiguous vocabulary shard used only by attention-TP decode. EP8
    /// computes the full head directly and leaves this absent.
    decode_lm_head: Option<DeviceMatrix>,
    decode_vocab_start: usize,
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
    /// Token stride of this rank's MLA arena. FlashMLA fp8_ds_mla uses 656
    /// bytes; TP4 FlashInfer uses the standard 576-byte E4M3 layout.
    mla_cache_bytes_per_token: usize,
    /// EP rank count of the launch topology (8 for EP8, 4 for EP4, 1 for the
    /// tensor-replicated topologies): the factor between a step's per-rank
    /// bucket and the MoE collectives' agreed `global_tokens`.
    ep_ranks: usize,
    /// Built with `--moe-topo tp`: every MoE arm is `MoeTp`, bucket-8
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

/// Everything one decode bucket owns: the MLA schedule and whole-step
/// graph, shared scratch, and the device block table.
struct Glm52BucketState {
    rows: usize,
    sched: Glm52MlaSchedMetadata,
    scratch: Glm52DecodeScratch,
    graph: CudaGraphState,
    block_table: CudaSlice<i32>,
    /// Pinned landing buffers for this bucket's argmax D2H, sized exactly
    /// `rows` (`memcpy_dtoh` copies the DESTINATION's byte count). Pinned
    /// memory keeps the copy asynchronous so the next step's replay can be
    /// enqueued launch-ahead before the host blocks on the result.
    argmax_values_host: PinnedHostSlice<bf16>,
    argmax_indices_host: PinnedHostSlice<i32>,
}

/// Attention layers occupy AR slots `0..GLM52_LAYERS`; the tail reuses the
/// same fixed-order transport to gather vocabulary-shard top-1 candidates.
const VOCAB_AR_SLOT: usize = GLM52_LAYERS;
impl Glm52RankModel {
    /// Export one already pre-captured whole-step bucket graph. The scheduler
    /// selects the topology's serving shape; this method only enforces that
    /// the requested bucket belongs to this model and is ready for replay.
    pub(crate) fn dump_decode_graph_png(
        &self,
        bucket: usize,
        png_path: &std::path::Path,
        title: &str,
    ) -> Result<CudaGraphDumpSummary> {
        let state = self
            .buckets
            .iter()
            .find(|state| state.rows == bucket)
            .with_context(|| {
                format!(
                    "GLM5.2 graph dump bucket {bucket} is not a member of {GLM52_DECODE_BUCKETS:?}"
                )
            })?;
        ensure!(
            state.graph.is_captured(),
            "GLM5.2 bucket-{bucket} graph dump requested before pre-capture"
        );
        state
            .graph
            .dump_png(png_path, title)
            .with_context(|| format!("dump GLM5.2 rank-0 bucket-{bucket} decode CUDA Graph"))
    }

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
        Ok(state
            .scratch
            .captured
            .as_ref()
            .context("GLM5.2 DSpark context capture is disabled")?
            .data())
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
        let mla_block_bytes = GLM52_FLASHMLA_SPARSE_PAGE_SIZE * self.mla_cache_bytes_per_token;
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

    /// Deinterleave the RoPE dims of vLLM-restored pages, across every MLA and
    /// index-K arena (vLLM P/D: the peer stores rotated RoPE pairs interleaved,
    /// openinfer's kernels read the block-out placement — same values, permuted
    /// dims). Runs on the model's compute stream, so it orders after the
    /// caller-synchronized pegaflow H2D and before any subsequent step kernels.
    /// NOT idempotent — the scheduler calls it exactly once per restored page.
    pub(crate) fn vllm_rope_fixup(&mut self, ctx: &DeviceContext, pages: &[i32]) -> Result<()> {
        ensure!(!pages.is_empty(), "GLM5.2 vLLM rope fixup needs pages");
        ensure!(
            self.mla_cache_bytes_per_token == GLM52_MLA_CACHE_BYTES,
            "GLM5.2 vLLM P/D requires the fp8_ds_mla cache row ({GLM52_MLA_CACHE_BYTES} B/token), \
             this rank packs {} B/token",
            self.mla_cache_bytes_per_token
        );
        let pages_dev = ctx
            .stream
            .clone_htod(pages)
            .map_err(|err| anyhow::anyhow!("GLM5.2 vLLM rope fixup pages H2D: {err}"))?;
        let mla_stride = GLM52_FLASHMLA_SPARSE_PAGE_SIZE * self.mla_cache_bytes_per_token;
        let idxk_stride = INDEX_CACHE_BLOCK * (GLM52_INDEX_HEAD_DIM + 4);
        for caches in &mut self.caches {
            glm52_vllm_rope_fixup_launch(
                ctx,
                &mut caches.mla_cache,
                mla_stride,
                Glm52VllmFixupKind::Mla,
                &pages_dev,
                pages.len(),
            )?;
            if let Some(index_k) = &mut caches.index_k_cache {
                glm52_vllm_rope_fixup_launch(
                    ctx,
                    index_k,
                    idxk_stride,
                    Glm52VllmFixupKind::IndexK,
                    &pages_dev,
                    pages.len(),
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn build(
        ctx: &DeviceContext,
        w: &mut Glm52RankGpuWeights,
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        attn_shard: Option<usize>,
        dspark_enabled: bool,
    ) -> Result<Self> {
        ensure!(
            moe_topo.uses_tensor_replicated_moe() == attn_shard.is_some(),
            "GLM5.2 attention-TP shard must ride a tensor-replicated topology (topo {moe_topo:?}, \
             shard {attn_shard:?})"
        );
        ensure!(
            max_model_len > 0 && max_model_len.is_multiple_of(GLM52_MODEL_LEN_ALIGN),
            "GLM5.2 max_model_len {max_model_len} must be a positive multiple of \
             {GLM52_MODEL_LEN_ALIGN} (the FlashMLA page / index-K block size)"
        );
        let batch = GLM52_MAX_BATCH_PER_RANK;
        let mla_heads = if attn_shard.is_some() {
            crate::config::GLM52_HEADS / moe_topo.device_count()
        } else {
            crate::config::GLM52_HEADS
        };
        let mla_backend = glm52_select_mla_backend(mla_heads)?;
        let mla_cache_bytes_per_token = mla_backend.cache_bytes_per_token();
        log::info!(
            "GLM5.2 MLA backend: {:?} ({} heads/rank, {} bytes/cache token)",
            mla_backend,
            mla_heads,
            mla_cache_bytes_per_token
        );
        let num_sm_parts = if attn_shard.is_some() {
            1
        } else {
            glm52_flashmla_sparse_decode_num_sm_parts()?
        };
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: batch,
            num_blocks: glm52_pool_blocks(max_model_len),
            topk: GLM52_FLASHMLA_SPARSE_TOPK,
            num_sm_parts,
            sm_scale: GLM52_SM_SCALE,
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
                mla_cache: ctx.stream.alloc_zeros::<u8>(
                    contract.num_blocks
                        * GLM52_FLASHMLA_SPARSE_PAGE_SIZE
                        * mla_cache_bytes_per_token,
                )?,
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
        let (decode_lm_head, decode_vocab_start) = if let Some(rank) = attn_shard {
            let ranks = moe_topo.device_count();
            ensure!(
                rank < ranks && GLM52_VOCAB.is_multiple_of(ranks),
                "GLM5.2 vocab TP shard {rank}/{ranks} cannot partition {} rows",
                GLM52_VOCAB
            );
            let rows = GLM52_VOCAB / ranks;
            let start = rank * rows;
            let mut data = ctx.stream.alloc_zeros::<bf16>(rows * GLM52_HIDDEN)?;
            ctx.stream.memcpy_dtod(
                &lm_head
                    .data
                    .slice(start * GLM52_HIDDEN..(start + rows) * GLM52_HIDDEN),
                &mut data,
            )?;
            (
                Some(DeviceMatrix {
                    data,
                    rows,
                    cols: GLM52_HIDDEN,
                }),
                start,
            )
        } else {
            (None, 0)
        };
        let final_norm = build::take_bf16_vec(ctx, w, "model.norm.weight", GLM52_HIDDEN)?;
        w.ensure_consumed()?;

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
        // Attention-TP scratch follows the head shard and selected MLA cache
        // layout; both were fixed before the per-layer arenas were allocated.
        let mut buckets = Vec::with_capacity(GLM52_DECODE_BUCKETS.len());
        for rows in GLM52_DECODE_BUCKETS {
            let contract_rows = Glm52FlashMlaSparseDecode {
                batch_size: rows,
                ..contract
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
                sched: Glm52MlaSchedMetadata::new_for_backend(
                    ctx,
                    contract_rows,
                    mla_heads,
                    mla_backend,
                )?,
                scratch: Glm52DecodeScratch::new_for_backend(
                    ctx,
                    &contract_rows,
                    mqa_shape,
                    mla_heads,
                    mla_backend,
                    dspark_enabled,
                )?,
                graph: CudaGraphState::new(),
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
        let mut buckets = buckets;

        if mla_backend == crate::mla_decode::Glm52MlaBackend::FlashInferFp8 {
            for bucket in &mut buckets {
                glm52_mla_backend_preflight(
                    ctx,
                    &bucket.sched,
                    &mut bucket.scratch.mla_attend,
                    &caches[0].mla_cache,
                )?;
            }
            ctx.sync()?;
        }

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
            decode_lm_head,
            decode_vocab_start,
            buckets,
            table_width,
            max_model_len,
            mla_cache_bytes_per_token,
            ep_ranks: moe_topo.expected_ep_size(),
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
        ep8: Option<&mut Glm52MoeEpState>,
        tp: Option<&mut Glm52MoeTpRank>,
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
            self.decode_step_prologue_and_replay(ctx, aux, ep8, tp, inputs, shape, kv)?;
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
            .iter_mut()
            .find(|bucket| bucket.rows == shape.bucket)
            .expect("decode_step validated the bucket");
        // TP greedy decode writes compact shard logits into the shared
        // buffer. Sampling needs the full distribution, so only sampled
        // steps pay one eager full-head recompute after the graph.
        if self.decode_lm_head.is_some() {
            glm52_lm_head_into(
                ctx,
                &bucket.scratch.final_normed,
                &self.lm_head,
                &mut bucket.scratch.logits,
            )?;
        }
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
        ep8: Option<&mut Glm52MoeEpState>,
        mut tp: Option<&mut Glm52MoeTpRank>,
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
        if let Some(rank) = tp.as_deref_mut() {
            if !rank.slices.is_empty() {
                rank.state.stage_active_rows(ctx, shape.active_rows)?;
            }
        }

        // The bucket state selected above carries the plan, scratch, graph,
        // and block table together — one coherent shape.
        let step = Glm52DecodeStep {
            mla_cos: &self.cos,
            mla_sin: &self.sin,
            idx_cos: &self.cos,
            idx_sin: &self.sin,
            mla_sched: &bucket.sched,
            slot_mapping: &self.slot_mapping,
            block_table: &bucket.block_table,
            seq_lens: &self.seq_lens,
        };
        // Every rank must pass the same global token count into the MoE
        // collectives — guaranteed by the coordinator agreeing the bucket.
        // (`ep_ranks` is 1 on tensor-replicated topologies, where the value
        // is never consumed.)
        let global_tokens = self.ep_ranks * batch;

        let s = &mut bucket.scratch;
        let decode_lm_head = self.decode_lm_head.as_ref().unwrap_or(&self.lm_head);
        let mut graph = std::mem::take(&mut bucket.graph);
        let result = graph.run_or_capture(ctx, || {
            run_step_body(
                ctx,
                aux,
                ep8,
                tp,
                &self.layers,
                &mut self.caches,
                &self.embed,
                &self.final_norm,
                decode_lm_head,
                self.decode_vocab_start,
                &self.token_ids,
                &step,
                s,
                global_tokens,
            )
        });
        bucket.graph = graph;
        result
    }
}
