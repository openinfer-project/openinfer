use std::sync::Arc;

use anyhow::{Result, bail};
use cudarc::driver::CudaSlice;
use half::bf16;

use crate::page_pool::{OwnedPagePermit, PageId, PagePool};
use crate::tensor::DeviceContext;

/// Page-first geometry: dimensions and derived strides for one page.
///
/// Pure value type — no GPU, no allocation. Shared by `KvPool`, `KvState`, `KvDesc`.
#[derive(Clone, Copy, Debug)]
pub struct KvLayout {
    pub page_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Elements in one K (or V) block: page_size × num_kv_heads × head_dim.
    pub kv_block_len: usize,
    /// Elements between layers within a page: 2 × kv_block_len (K then V).
    pub layer_stride: usize,
    /// Elements per page (all layers): num_layers × layer_stride.
    pub page_stride: usize,
}

impl KvLayout {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, page_size: usize) -> Self {
        let kv_block_len = page_size * num_kv_heads * head_dim;
        let layer_stride = 2 * kv_block_len;
        let page_stride = num_layers * layer_stride;
        Self {
            page_size,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_block_len,
            layer_stride,
            page_stride,
        }
    }

    pub fn kernel_layout(&self) -> pegainfer_kernels::paged_kv::PagedKvLayout {
        pegainfer_kernels::paged_kv::PagedKvLayout {
            page_size: self.page_size,
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            kv_block_len: self.kv_block_len,
            layer_stride: self.layer_stride,
            page_stride: self.page_stride,
        }
    }
}

struct KvPoolInner {
    pool: PagePool,
    buffer: CudaSlice<bf16>,
    layout: KvLayout,
    /// Reserved page for bucket CUDA Graph padding. FlashInfer metadata for
    /// padding slots points here with seq_len=1. KV append writes harmless
    /// garbage; attention output is discarded.
    padding_permit: OwnedPagePermit,
}

/// Shared KV backing storage with page-based allocation.
///
/// Owns a single GPU buffer in page-first layout: each page is a contiguous
/// chunk holding all layers' K and V for `page_size` tokens.
///
/// ```text
/// page_i: [L0_K | L0_V | L1_K | L1_V | ... | Ln_K | Ln_V]
///          └─ page_size × num_kv_heads × head_dim each ─┘
/// ```
///
/// Cheaply clonable (Arc). Multiple `KvState`s share the same pool.
#[derive(Clone)]
pub struct KvPool {
    inner: Arc<KvPoolInner>,
}

impl KvPool {
    pub fn new(
        ctx: &DeviceContext,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        num_pages: usize,
    ) -> Result<Self> {
        let layout = KvLayout::new(num_layers, num_kv_heads, head_dim, page_size);
        let total_elements = num_pages * layout.page_stride;

        let buffer: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(total_elements)
            .map_err(|e| anyhow::anyhow!("KvPool alloc failed: {e}"))?;

        let pool = PagePool::new(num_pages);
        let padding_permit = pool
            .try_acquire_many(1)
            .expect("pool must have at least 1 page for padding");

        Ok(Self {
            inner: Arc::new(KvPoolInner {
                pool,
                buffer,
                layout,
                padding_permit,
            }),
        })
    }

    /// Create an empty per-request KV state.
    pub fn alloc(&self) -> KvState {
        KvState {
            permit: self.inner.pool.try_acquire_many(0).expect("zero acquire"),
            seq_len: 0,
            pool: self.clone(),
        }
    }

    pub fn layout(&self) -> &KvLayout {
        &self.inner.layout
    }

    pub fn capacity_pages(&self) -> usize {
        self.inner.pool.capacity_pages()
    }

    pub fn available_pages(&self) -> usize {
        self.inner.pool.available_pages()
    }

    /// Page index reserved for bucket CUDA Graph padding.
    pub fn padding_page_id(&self) -> i32 {
        self.inner.padding_permit.pages()[0].index() as i32
    }
}

/// Per-request KV state. Parallels `RecurrentState` for linear attention.
///
/// Holds an RAII page permit that auto-returns pages on drop.
pub struct KvState {
    permit: OwnedPagePermit,
    seq_len: usize,
    pool: KvPool,
}

/// Pages needed to hold `token_count` tokens with the given `page_size`.
fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

impl KvState {
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn num_pages(&self) -> usize {
        self.permit.pages().len()
    }

    pub fn last_page_len(&self) -> usize {
        if self.seq_len == 0 {
            0
        } else {
            let rem = self.seq_len % self.pool.inner.layout.page_size;
            if rem == 0 {
                self.pool.inner.layout.page_size
            } else {
                rem
            }
        }
    }

    /// Page indices as i32 for GPU upload.
    pub fn page_indices_i32(&self) -> Vec<i32> {
        self.permit
            .pages()
            .iter()
            .map(|p| p.index() as i32)
            .collect()
    }

    pub fn buffer(&self) -> &CudaSlice<bf16> {
        &self.pool.inner.buffer
    }

    pub fn layout(&self) -> &KvLayout {
        &self.pool.inner.layout
    }

    /// Ensure capacity for at least `token_count` tokens total.
    pub fn ensure_capacity(&mut self, token_count: usize) -> Result<()> {
        let needed = pages_needed(token_count, self.pool.inner.layout.page_size);
        let held = self.permit.len();
        if needed <= held {
            return Ok(());
        }
        let grow = needed - held;
        if !self.permit.try_grow(grow) {
            bail!(
                "KvState: out of pages (need {grow} more, {} available)",
                self.pool.available_pages()
            );
        }
        Ok(())
    }

    /// Advance sequence length after writing tokens.
    pub fn advance(&mut self, count: usize) {
        self.seq_len += count;
    }

    /// Build kernel-facing metadata for this request's KV.
    pub fn desc(&self) -> KvDesc<'_> {
        KvDesc {
            layout: self.pool.inner.layout,
            buffer: &self.pool.inner.buffer,
            pages: self.permit.pages(),
            seq_len: self.seq_len,
            last_page_len: self.last_page_len(),
        }
    }

    /// Reset for a new request: return all pages, zero seq_len.
    pub fn reset(&mut self) {
        self.permit = self
            .pool
            .inner
            .pool
            .try_acquire_many(0)
            .expect("zero acquire");
        self.seq_len = 0;
    }
}

/// Kernel-facing metadata describing this request's paged KV layout.
///
/// Bundles pool geometry (strides) with request state (page list, seq_len).
/// Forward code passes this opaquely to FFI calls.
pub struct KvDesc<'a> {
    layout: KvLayout,
    buffer: &'a CudaSlice<bf16>,
    pages: &'a [PageId],
    seq_len: usize,
    last_page_len: usize,
}

impl KvDesc<'_> {
    pub fn layout(&self) -> &KvLayout {
        &self.layout
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn last_page_len(&self) -> usize {
        self.last_page_len
    }

    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Page indices for this request (CPU-side).
    pub fn page_indices(&self) -> &[PageId] {
        self.pages
    }

    /// The backing buffer. Kernel integration will extract device pointers
    /// via `buffer.device_ptr(&stream)`.
    pub fn buffer(&self) -> &CudaSlice<bf16> {
        self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pool(page_size: usize, num_pages: usize) -> KvPool {
        let ctx = DeviceContext::new().expect("GPU required for kv_pool tests");
        // Tiny allocation: 1 layer, 1 head, dim=1 — just enough to exercise logic.
        KvPool::new(&ctx, 1, 1, 1, page_size, num_pages).expect("KvPool::new failed")
    }

    #[test]
    fn stride_geometry_qwen35() {
        // Qwen3.5-4B: 8 full attn layers, 4 KV heads, head_dim=256, page_size=16
        let l = KvLayout::new(8, 4, 256, 16);

        assert_eq!(l.kv_block_len, 16 * 4 * 256); // 16384 elements
        assert_eq!(l.layer_stride, 2 * 16384); // 32768 (K + V)
        assert_eq!(l.page_stride, 8 * 32768); // 262144 per page
        // bytes: 262144 * 2 = 524288 = 512 KB per page
        assert_eq!(l.page_stride * 2, 512 * 1024);
    }

    #[test]
    fn stride_geometry_qwen3() {
        // Qwen3-4B: 36 full attn layers, 4 KV heads, head_dim=128, page_size=16
        let l = KvLayout::new(36, 4, 128, 16);

        // 36 layers × 16384 = 589824 elements per page
        assert_eq!(l.page_stride, 589_824);
        // bytes: ~1.125 MB per page
    }

    #[test]
    fn kv_state_lifecycle() {
        // 5 pages total: 1 reserved for padding, 4 available for requests
        let pool = test_pool(16, 5);
        let mut kv = pool.alloc();

        // Starts empty
        assert_eq!(kv.seq_len(), 0);
        let desc = kv.desc();
        assert_eq!(desc.num_pages(), 0);
        assert_eq!(desc.last_page_len(), 0);

        // Prefill 10 tokens: needs 1 page
        kv.ensure_capacity(10).unwrap();
        kv.advance(10);
        assert_eq!(kv.seq_len(), 10);
        let desc = kv.desc();
        assert_eq!(desc.num_pages(), 1);
        assert_eq!(desc.last_page_len(), 10);
        assert_eq!(pool.available_pages(), 3);

        // Decode to 16: still 1 page, last_page_len=16 (full)
        for _ in 10..16 {
            kv.ensure_capacity(kv.seq_len() + 1).unwrap();
            kv.advance(1);
        }
        let desc = kv.desc();
        assert_eq!(desc.num_pages(), 1);
        assert_eq!(desc.last_page_len(), 16);

        // One more token crosses page boundary: 2 pages
        kv.ensure_capacity(kv.seq_len() + 1).unwrap();
        kv.advance(1);
        assert_eq!(kv.seq_len(), 17);
        let desc = kv.desc();
        assert_eq!(desc.num_pages(), 2);
        assert_eq!(desc.last_page_len(), 1);
        assert_eq!(pool.available_pages(), 2);

        // Reset returns all pages
        kv.reset();
        assert_eq!(kv.seq_len(), 0);
        assert_eq!(pool.available_pages(), 4);
    }

    #[test]
    fn kv_state_out_of_pages() {
        // 3 pages total: 1 padding, 2 available → 32 tokens max
        let pool = test_pool(16, 3);
        let mut kv = pool.alloc();

        kv.ensure_capacity(32).unwrap(); // exactly fits 2 pages
        assert!(kv.ensure_capacity(33).is_err()); // needs 3rd page
    }

    #[test]
    fn kv_state_drop_returns_pages() {
        // 5 pages total: 1 padding, 4 available
        let pool = test_pool(16, 5);
        {
            let mut kv = pool.alloc();
            kv.ensure_capacity(20).unwrap(); // 2 pages
            kv.advance(20);
            assert_eq!(pool.available_pages(), 2);
        }
        // Drop returns pages
        assert_eq!(pool.available_pages(), 4);
    }
}
