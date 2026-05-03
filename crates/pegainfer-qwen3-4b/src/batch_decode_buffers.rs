//! Pre-allocated GPU buffers for batched decode (multiple requests, 1 token each).

use anyhow::Result;

use cudarc::driver::CudaSlice;

use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::tensor::{DeviceContext, HiddenStates};

/// Bucket sizes for CUDA Graph capture. Actual batch is padded to the nearest bucket.
pub(crate) const BATCH_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

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
    /// Fused gate+up projection output [2*intermediate_size, bs]
    pub(crate) gate_up_out: HiddenStates,
    pub(crate) mlp_act: HiddenStates,
    pub(crate) mlp_out: HiddenStates,
    pub(crate) hidden: HiddenStates,
    pub(crate) logits: HiddenStates,

    // GPU metadata
    pub(crate) token_ids_d: CudaSlice<u32>,
    pub(crate) positions_d: CudaSlice<i32>,

    // Paged attention metadata (concatenated across requests, CSR format)
    pub(crate) page_indices_d: CudaSlice<i32>,
    pub(crate) page_indptr_d: CudaSlice<i32>,
    pub(crate) last_page_len_d: CudaSlice<i32>,
    pub(crate) request_indices_d: CudaSlice<i32>,
    pub(crate) kv_tile_indices_d: CudaSlice<i32>,
    pub(crate) kv_chunk_size_d: CudaSlice<i32>,

    /// Padding page index for bucket CUDA Graph. Padding slots point here.
    padding_page_id: i32,

    /// One CudaGraphState per bucket (indexed by BATCH_BUCKETS position).
    pub(crate) graphs: Vec<CudaGraphState>,
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
    ) -> Result<Self> {
        let bs = max_batch_size;

        Ok(Self {
            max_batch_size: bs,
            normed: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            q: HiddenStates::zeros(ctx, q_dim, bs)?,
            k: HiddenStates::zeros(ctx, kv_dim, bs)?,
            v: HiddenStates::zeros(ctx, kv_dim, bs)?,
            attn_out: HiddenStates::zeros(ctx, q_dim, bs)?,
            attn_proj: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            qkv_out: HiddenStates::zeros(ctx, q_dim + 2 * kv_dim, bs)?,
            gate_up_out: HiddenStates::zeros(ctx, 2 * intermediate_size, bs)?,
            mlp_act: HiddenStates::zeros(ctx, intermediate_size, bs)?,
            mlp_out: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            hidden: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            logits: HiddenStates::zeros(ctx, vocab_size, bs)?,
            token_ids_d: ctx.stream.alloc_zeros(bs)?,
            positions_d: ctx.stream.alloc_zeros(bs)?,
            // Paged attention: worst case all requests use max_total_pages + padding slots
            page_indices_d: ctx.stream.alloc_zeros(max_total_pages + bs)?,
            page_indptr_d: ctx.stream.alloc_zeros(bs + 1)?,
            last_page_len_d: ctx.stream.alloc_zeros(bs)?,
            request_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_tile_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_chunk_size_d: ctx.stream.alloc_zeros(bs)?,
            padding_page_id,
            graphs: BATCH_BUCKETS
                .iter()
                .map(|_| CudaGraphState::new())
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
        self.gate_up_out.seq_len = bs;
        self.mlp_act.seq_len = bs;
        self.mlp_out.seq_len = bs;
        self.hidden.seq_len = bs;
        self.logits.seq_len = bs;
    }

    /// Sync paged attention metadata from multiple KvStates to GPU buffers.
    ///
    /// `padded_bs` >= `kv_states.len()`: padding slots (if any) point to the
    /// reserved padding page with seq_len=1 so FlashInfer accesses valid memory.
    pub(crate) fn sync_paged_meta(
        &mut self,
        ctx: &DeviceContext,
        kv_states: &[&KvState],
        padded_bs: usize,
    ) -> Result<()> {
        let real_bs = kv_states.len();
        debug_assert!(padded_bs >= real_bs);

        // Build concatenated page_indices and CSR indptr
        let mut all_page_indices = Vec::new();
        let mut indptr = vec![0i32];
        let mut last_page_lens = Vec::with_capacity(padded_bs);
        let mut chunk_sizes = Vec::with_capacity(padded_bs);

        for kv in kv_states {
            let pages = kv.page_indices_i32();
            all_page_indices.extend_from_slice(&pages);
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(kv.last_page_len() as i32);
            chunk_sizes.push(kv.seq_len() as i32);
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

        Ok(())
    }
}
