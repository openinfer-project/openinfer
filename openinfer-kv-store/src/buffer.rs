use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr;
use half::bf16;

use crate::KvLayout;

struct Inner {
    buffer: CudaSlice<bf16>,
    layout: KvLayout,
    num_blocks: usize,
}

/// GPU KV cache buffer without an allocator.
///
/// Owns the device memory and layout geometry but delegates block
/// allocation to an external `BlockManager` (kvbm-logical).
#[derive(Clone)]
pub struct KvBuffer {
    inner: Arc<Inner>,
}

impl KvBuffer {
    pub fn new(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        num_blocks: usize,
    ) -> anyhow::Result<Self> {
        let layout = KvLayout::new(num_layers, num_kv_heads, head_dim, page_size);
        let total_elements = num_blocks * layout.page_stride;
        let buffer: CudaSlice<bf16> = stream
            .alloc_zeros(total_elements)
            .map_err(|e| anyhow::anyhow!("KvBuffer alloc failed: {e}"))?;
        Ok(Self {
            inner: Arc::new(Inner {
                buffer,
                layout,
                num_blocks,
            }),
        })
    }

    pub fn layout(&self) -> &KvLayout {
        &self.inner.layout
    }

    pub fn buffer(&self) -> &CudaSlice<bf16> {
        &self.inner.buffer
    }

    /// Base device address of the fused KV buffer.
    ///
    /// Stable for the buffer's lifetime — cudarc allocations don't move — so
    /// the KV-offload connector registers this once with pegaflow and the
    /// page-first [`KvLayout`] strides reach every (layer, block, K/V) segment
    /// from it. The returned address outlives the transient stream-ordering
    /// guard precisely because the `Arc<Inner>` keeps the slice alive.
    pub fn device_ptr(&self, stream: &CudaStream) -> u64 {
        let (ptr, _guard) = self.inner.buffer.device_ptr(stream);
        ptr
    }

    pub fn num_blocks(&self) -> usize {
        self.inner.num_blocks
    }
}
