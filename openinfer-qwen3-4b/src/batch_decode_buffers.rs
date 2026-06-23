//! Pre-allocated GPU buffers for batched decode (multiple requests, 1 token each).

use anyhow::Result;

use cudarc::driver::CudaSlice;

use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_core::tensor::{DeviceContext, HiddenStates};
use openinfer_kv_cache::KvView;

/// Bucket sizes for CUDA Graph capture. Actual batch is padded to the nearest bucket.
/// Based on vLLM's cudagraph capture list up to 256; graphs are captured lazily per
/// bucket, and activation buffers are shared (sized once at the largest bucket), so
/// extra buckets cost capture time on first hit, not memory.
///
/// Buckets 8/16 are viable only because decode GEMMs at N <= GEMM_LT_MAX_N run
/// tuned cublasLt algos: cuBLAS's GemmEx heuristic skips split-K for batch in
/// [8, 16] (RTX 5090 ctx1024: 9.2/9.3ms steps vs 7.9ms at bs20), while the Lt
/// heuristic list has full-speed candidates at every small N.
pub(crate) const BATCH_BUCKETS: &[usize] = &[
    1, 2, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
    160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256,
];
const DECODE_ATTENTION_PATH_COUNT: usize = 2;
// Split-KV decode attention: the non-partitioned kernel issues one CTA per
// (request x kv-head), starving SMs at small batch. The path is therefore
// chosen by batch (CTA count vs SM count), NOT context length — at bs=1 the
// 8 CTAs underfill the GPU at any seq_len, so SplitKv wins across the whole
// context range. SPLIT_KV_MAX_BATCH_SIZE caps it where NonPartition's CTAs
// already saturate the SMs (bs<=8 wins big, ~bs16 even, bs32 within ~1%).
// 64-token chunks measured fastest on RTX 5090 (128/256 are 1-7% slower, 32
// past the merge-overhead knee). Measurements: docs/models/qwen3/decode-attention.md.
const SPLIT_KV_CHUNK_TOKENS: usize = 64;
const SPLIT_KV_MAX_CHUNKS_PER_REQUEST: usize = 64;
const SPLIT_KV_MAX_BATCH_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeAttentionPath {
    NonPartition,
    SplitKv,
}

impl DecodeAttentionPath {
    fn graph_slot(self) -> usize {
        match self {
            Self::NonPartition => 0,
            Self::SplitKv => 1,
        }
    }
}

/// Find the smallest bucket >= `bs`. Panics if bs > largest bucket.
pub(crate) fn bucket_for(bs: usize) -> usize {
    for &b in BATCH_BUCKETS {
        if b >= bs {
            return b;
        }
    }
    panic!(
        "batch size {bs} exceeds largest bucket {}",
        BATCH_BUCKETS.last().unwrap()
    );
}

/// Pre-allocated buffers for batch decode. All tensors are sized for `max_batch_size`.
///
/// Uses `HiddenStates` (2D) instead of `DeviceVec` (1D) — the "seq_len" dimension
/// is actually the batch dimension (one token per request).
pub(crate) struct BatchDecodeBuffers {
    pub(crate) max_batch_size: usize,

    // Per-layer intermediates [dim, max_batch_size]
    pub(crate) normed: HiddenStates,
    pub(crate) q: HiddenStates,
    pub(crate) k: HiddenStates,
    pub(crate) v: HiddenStates,
    pub(crate) attn_out: HiddenStates,
    pub(crate) attn_proj: HiddenStates,
    /// Fused QKV projection output [q_dim + 2*kv_dim, bs]
    pub(crate) qkv_out: HiddenStates,
    /// Split MLP gate projection output [intermediate_size, bs].
    pub(crate) gate_out: HiddenStates,
    /// Split MLP up projection output [intermediate_size, bs].
    pub(crate) up_out: HiddenStates,
    pub(crate) mlp_act: HiddenStates,
    pub(crate) mlp_out: HiddenStates,
    pub(crate) hidden: HiddenStates,
    pub(crate) logits: HiddenStates,

    // GPU metadata
    pub(crate) token_ids_d: CudaSlice<u32>,
    pub(crate) positions_d: CudaSlice<i32>,
    pub(crate) lora_token_slots_d: CudaSlice<i32>,

    // Paged attention metadata (concatenated across requests, CSR format)
    pub(crate) page_indices_d: CudaSlice<i32>,
    pub(crate) page_indptr_d: CudaSlice<i32>,
    pub(crate) last_page_len_d: CudaSlice<i32>,
    pub(crate) request_indices_d: CudaSlice<i32>,
    pub(crate) kv_tile_indices_d: CudaSlice<i32>,
    pub(crate) kv_chunk_size_d: CudaSlice<i32>,

    // Split-K paged attention metadata/workspace.
    pub(crate) split_request_indices_d: CudaSlice<i32>,
    pub(crate) split_kv_tile_indices_d: CudaSlice<i32>,
    pub(crate) split_kv_chunk_size_d: CudaSlice<i32>,
    pub(crate) split_o_indptr_d: CudaSlice<i32>,
    pub(crate) split_block_valid_mask_d: CudaSlice<u8>,
    pub(crate) split_tmp_v: CudaSlice<half::bf16>,
    pub(crate) split_tmp_s: CudaSlice<f32>,
    pub(crate) split_padded_slots: usize,
    max_seq_len: usize,

    /// Padding page index for bucket CUDA Graph. Padding slots point here.
    padding_page_id: i32,

    /// One CudaGraphState per `(bucket, attention_path)`, captured on the
    /// full-SM `ctx.stream` — used by the normal decode-only path.
    pub(crate) graphs: Vec<CudaGraphState>,

    /// Parallel cache captured on the Green Context decode-partition stream,
    /// used when decode runs concurrently with prefill (SplitConcurrent). A
    /// graph captured on `ctx.stream` would replay on all SMs, so the split
    /// path needs its own graphs whose nodes are pinned to the decode partition.
    pub(crate) graphs_split: Vec<CudaGraphState>,
}

impl BatchDecodeBuffers {
    pub(crate) fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_batch_size: usize,
        max_total_pages: usize,
        padding_page_id: i32,
        num_qo_heads: usize,
    ) -> Result<Self> {
        let bs = max_batch_size;
        // The split-KV path is gated on padded_bs <= SPLIT_KV_MAX_BATCH_SIZE,
        // so its workspace only needs slots for that many requests (~16 MiB
        // instead of ~128 MiB at bucket 256).
        let max_split_slots = bs.min(SPLIT_KV_MAX_BATCH_SIZE) * SPLIT_KV_MAX_CHUNKS_PER_REQUEST;

        Ok(Self {
            max_batch_size: bs,
            normed: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            q: HiddenStates::zeros(ctx, q_dim, bs)?,
            k: HiddenStates::zeros(ctx, kv_dim, bs)?,
            v: HiddenStates::zeros(ctx, kv_dim, bs)?,
            attn_out: HiddenStates::zeros(ctx, q_dim, bs)?,
            attn_proj: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            qkv_out: HiddenStates::zeros(ctx, q_dim + 2 * kv_dim, bs)?,
            gate_out: HiddenStates::zeros(ctx, intermediate_size, bs)?,
            up_out: HiddenStates::zeros(ctx, intermediate_size, bs)?,
            mlp_act: HiddenStates::zeros(ctx, intermediate_size, bs)?,
            mlp_out: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            hidden: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            logits: HiddenStates::zeros(ctx, vocab_size, bs)?,
            token_ids_d: ctx.stream.alloc_zeros(bs)?,
            positions_d: ctx.stream.alloc_zeros(bs)?,
            lora_token_slots_d: ctx.stream.alloc_zeros(bs)?,
            // Paged attention: worst case all requests use max_total_pages + padding slots
            page_indices_d: ctx.stream.alloc_zeros(max_total_pages + bs)?,
            page_indptr_d: ctx.stream.alloc_zeros(bs + 1)?,
            last_page_len_d: ctx.stream.alloc_zeros(bs)?,
            request_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_tile_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_chunk_size_d: ctx.stream.alloc_zeros(bs)?,
            split_request_indices_d: ctx.stream.alloc_zeros(max_split_slots)?,
            split_kv_tile_indices_d: ctx.stream.alloc_zeros(max_split_slots)?,
            split_kv_chunk_size_d: ctx.stream.alloc_zeros(1)?,
            split_o_indptr_d: ctx.stream.alloc_zeros(bs + 1)?,
            split_block_valid_mask_d: ctx.stream.alloc_zeros(max_split_slots)?,
            split_tmp_v: ctx.stream.alloc_zeros(max_split_slots * q_dim)?,
            split_tmp_s: ctx.stream.alloc_zeros(max_split_slots * num_qo_heads)?,
            split_padded_slots: 0,
            max_seq_len: 0,
            padding_page_id,
            graphs: BATCH_BUCKETS
                .iter()
                .flat_map(|_| (0..DECODE_ATTENTION_PATH_COUNT).map(|_| CudaGraphState::new()))
                .collect(),
            graphs_split: BATCH_BUCKETS
                .iter()
                .flat_map(|_| (0..DECODE_ATTENTION_PATH_COUNT).map(|_| CudaGraphState::new()))
                .collect(),
        })
    }

    /// Set actual batch size for this step. Adjusts the seq_len field on all HiddenStates.
    pub(crate) fn set_batch_size(&mut self, bs: usize) {
        assert!(bs <= self.max_batch_size);
        self.normed.seq_len = bs;
        self.q.seq_len = bs;
        self.k.seq_len = bs;
        self.v.seq_len = bs;
        self.attn_out.seq_len = bs;
        self.attn_proj.seq_len = bs;
        self.qkv_out.seq_len = bs;
        self.gate_out.seq_len = bs;
        self.up_out.seq_len = bs;
        self.mlp_act.seq_len = bs;
        self.mlp_out.seq_len = bs;
        self.hidden.seq_len = bs;
        self.logits.seq_len = bs;
    }

    /// Sync paged attention metadata from multiple KvViews to GPU buffers.
    ///
    /// `padded_bs` >= `kv_views.len()`: padding slots (if any) point to the
    /// reserved padding page with seq_len=1 so FlashInfer accesses valid memory.
    pub(crate) fn sync_paged_meta(
        &mut self,
        ctx: &DeviceContext,
        kv_views: &[&KvView],
        padded_bs: usize,
    ) -> Result<()> {
        let real_bs = kv_views.len();
        debug_assert!(padded_bs >= real_bs);

        // Build concatenated page_indices and CSR indptr
        let mut all_page_indices = Vec::new();
        let mut indptr = vec![0i32];
        let mut last_page_lens = Vec::with_capacity(padded_bs);
        let mut chunk_sizes = Vec::with_capacity(padded_bs);
        self.max_seq_len = 0;

        for kv in kv_views {
            all_page_indices.extend_from_slice(kv.page_indices());
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(kv.last_page_len() as i32);
            chunk_sizes.push(kv.seq_len() as i32);
            self.max_seq_len = self.max_seq_len.max(kv.seq_len());
        }

        // Padding slots: 1 page (the padding page), seq_len=1, last_page_len=1
        for _ in real_bs..padded_bs {
            all_page_indices.push(self.padding_page_id);
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(1);
            chunk_sizes.push(1);
        }

        let request_indices: Vec<i32> = (0..padded_bs as i32).collect();
        let kv_tile_indices = vec![0i32; padded_bs];

        ctx.stream
            .memcpy_htod(&all_page_indices, &mut self.page_indices_d)?;
        ctx.stream.memcpy_htod(&indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(&last_page_lens, &mut self.last_page_len_d)?;
        ctx.stream
            .memcpy_htod(&chunk_sizes, &mut self.kv_chunk_size_d)?;
        ctx.stream
            .memcpy_htod(&request_indices, &mut self.request_indices_d)?;
        ctx.stream
            .memcpy_htod(&kv_tile_indices, &mut self.kv_tile_indices_d)?;
        self.sync_split_kv_meta(ctx, kv_views, padded_bs)?;

        Ok(())
    }

    fn sync_split_kv_meta(
        &mut self,
        ctx: &DeviceContext,
        kv_views: &[&KvView],
        padded_bs: usize,
    ) -> Result<()> {
        // Past the batch cap the step always takes the non-partitioned path
        // (see attention_path), and the workspace has no slots for it anyway.
        if padded_bs > SPLIT_KV_MAX_BATCH_SIZE {
            return Ok(());
        }
        let split_chunk_size =
            SPLIT_KV_CHUNK_TOKENS.max(self.max_seq_len.div_ceil(SPLIT_KV_MAX_CHUNKS_PER_REQUEST));
        let split_padded_slots = padded_bs * SPLIT_KV_MAX_CHUNKS_PER_REQUEST;
        let mut split_request_indices = Vec::with_capacity(split_padded_slots);
        let mut split_kv_tile_indices = Vec::with_capacity(split_padded_slots);
        let mut split_o_indptr = Vec::with_capacity(padded_bs + 1);
        let mut split_block_valid_mask = Vec::with_capacity(split_padded_slots);
        split_o_indptr.push(0);

        for (request_idx, kv) in kv_views.iter().enumerate() {
            let chunks = kv.seq_len().div_ceil(split_chunk_size).max(1);
            debug_assert!(chunks <= SPLIT_KV_MAX_CHUNKS_PER_REQUEST);
            for chunk_idx in 0..chunks {
                split_request_indices.push(request_idx as i32);
                split_kv_tile_indices.push(chunk_idx as i32);
                split_block_valid_mask.push(1);
            }
            split_o_indptr.push(split_request_indices.len() as i32);
        }

        for _ in kv_views.len()..padded_bs {
            split_o_indptr.push(split_request_indices.len() as i32);
        }

        while split_request_indices.len() < split_padded_slots {
            split_request_indices.push(0);
            split_kv_tile_indices.push(0);
            split_block_valid_mask.push(0);
        }

        let split_kv_chunk_size = [split_chunk_size as i32];
        ctx.stream
            .memcpy_htod(&split_request_indices, &mut self.split_request_indices_d)?;
        ctx.stream
            .memcpy_htod(&split_kv_tile_indices, &mut self.split_kv_tile_indices_d)?;
        ctx.stream
            .memcpy_htod(&split_kv_chunk_size, &mut self.split_kv_chunk_size_d)?;
        ctx.stream
            .memcpy_htod(&split_o_indptr, &mut self.split_o_indptr_d)?;
        ctx.stream
            .memcpy_htod(&split_block_valid_mask, &mut self.split_block_valid_mask_d)?;
        self.split_padded_slots = split_padded_slots;

        Ok(())
    }

    pub(crate) fn attention_path(padded_bs: usize) -> DecodeAttentionPath {
        if padded_bs <= SPLIT_KV_MAX_BATCH_SIZE {
            DecodeAttentionPath::SplitKv
        } else {
            DecodeAttentionPath::NonPartition
        }
    }

    pub(crate) fn graph_index(bucket_idx: usize, path: DecodeAttentionPath) -> usize {
        bucket_idx * DECODE_ATTENTION_PATH_COUNT + path.graph_slot()
    }
}
