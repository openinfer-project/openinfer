use std::mem::ManuallyDrop;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr;
use half::bf16;
use openinfer_kernels::imported::ImportedKvArena;

use crate::KvLayout;

/// Backing memory for the fused KV arena.
///
/// The default path owns a normal cudarc `CudaSlice`. The offload path is a
/// non-owning view over a PegaFlow-allocated arena the executor imported over
/// CUDA IPC; the `CudaSlice` must not free that pointer, so it is wrapped in
/// `ManuallyDrop`. Whoever owns the `ImportedKvArena` controls the mapping
/// lifetime: it must outlive every kernel that touches this buffer and be
/// closed before the server frees the allocation.
enum Backing {
    Owned(CudaSlice<bf16>),
    ImportedView(ManuallyDrop<CudaSlice<bf16>>),
}

impl Backing {
    fn view(&self) -> &CudaSlice<bf16> {
        match self {
            Backing::Owned(slice) => slice,
            Backing::ImportedView(view) => view,
        }
    }
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::ImportedView(view) = self {
            // Drop the view WITHOUT freeing the imported pointer: leak()
            // reclaims the raw ptr and runs the CudaSlice field teardown
            // (events/stream) without any cuMemFree. The mapping itself is
            // closed by the `ImportedKvArena` owner.
            // SAFETY: `view` is not used again; ManuallyDrop::take moves it out.
            let slice = unsafe { ManuallyDrop::take(view) };
            let _raw_ptr = slice.leak();
        }
    }
}

struct Inner {
    backing: Backing,
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
        let slice: CudaSlice<bf16> = stream
            .alloc_zeros(total_elements)
            .map_err(|e| anyhow::anyhow!("KvBuffer alloc failed: {e}"))?;
        Ok(Self::from_backing(
            Backing::Owned(slice),
            layout,
            num_blocks,
        ))
    }

    /// Build the KV buffer as a view over a PegaFlow-allocated arena the
    /// caller imported via CUDA IPC. The caller keeps the `ImportedKvArena`
    /// alive for as long as any kernel touches this buffer and closes it
    /// before the server frees the allocation. The server zeroed the arena at
    /// allocation, so a KV hit never reads stale device memory.
    pub fn new_imported(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        num_blocks: usize,
        arena: &ImportedKvArena,
    ) -> anyhow::Result<Self> {
        let layout = KvLayout::new(num_layers, num_kv_heads, head_dim, page_size);
        let total_elements = num_blocks * layout.page_stride;
        let bytes = total_elements
            .checked_mul(std::mem::size_of::<bf16>())
            .ok_or_else(|| anyhow::anyhow!("KvBuffer size overflows usize"))?;
        anyhow::ensure!(
            arena.size_bytes() >= bytes,
            "imported arena is {} bytes but the KV layout needs {bytes}",
            arena.size_bytes()
        );
        // Wrap the imported pointer as a CudaSlice view for the attention
        // kernels.
        // SAFETY: the caller keeps the mapping valid for `total_elements` bf16;
        // the view is ManuallyDrop so it never frees the pointer.
        let view = unsafe { stream.upgrade_device_ptr::<bf16>(arena.device_ptr(), total_elements) };
        Ok(Self::from_backing(
            Backing::ImportedView(ManuallyDrop::new(view)),
            layout,
            num_blocks,
        ))
    }

    fn from_backing(backing: Backing, layout: KvLayout, num_blocks: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                backing,
                layout,
                num_blocks,
            }),
        }
    }

    pub fn layout(&self) -> &KvLayout {
        &self.inner.layout
    }

    pub fn buffer(&self) -> &CudaSlice<bf16> {
        self.inner.backing.view()
    }

    /// Base device address of the fused KV buffer.
    ///
    /// Stable for the buffer's lifetime, so the page-first [`KvLayout`] strides
    /// reach every (layer, block, K/V) segment from it.
    pub fn device_ptr(&self, stream: &CudaStream) -> u64 {
        let (ptr, _guard) = self.inner.backing.view().device_ptr(stream);
        ptr
    }

    pub fn num_blocks(&self) -> usize {
        self.inner.num_blocks
    }
}
