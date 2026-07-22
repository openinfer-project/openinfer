use std::mem::ManuallyDrop;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::bf16;

use openinfer_kernels::exportable::VmmExportableBuffer;

use crate::KvLayout;

/// Backing memory for the fused KV arena.
///
/// The default path owns a normal cudarc `CudaSlice`. The exportable path
/// allocates through the VMM API so the arena can be shared with an
/// out-of-process PegaFlow server (and, unlike CUDA IPC, registered into the
/// NIC for GPUDirect RDMA). In the VMM case the `CudaSlice` is only a *view*
/// over the VMM pointer — it must not free it — so it is wrapped in
/// `ManuallyDrop`; the `VmmExportableBuffer` owns teardown (unmap/free/release).
enum Backing {
    Owned(CudaSlice<bf16>),
    Vmm {
        /// Non-owning view for the attention kernels. Never dropped as a
        /// `CudaSlice` — that would `cuMemFreeAsync` a VMM pointer. Torn down
        /// via `ManuallyDrop::drop` (which does nothing to the memory) while the
        /// `vmm` field performs the real VMM teardown.
        view: ManuallyDrop<CudaSlice<bf16>>,
        vmm: VmmExportableBuffer,
    },
}

impl Backing {
    fn view(&self) -> &CudaSlice<bf16> {
        match self {
            Backing::Owned(slice) => slice,
            Backing::Vmm { view, .. } => view,
        }
    }
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Vmm { view, .. } = self {
            // Drop the view WITHOUT freeing the VMM pointer: leak() reclaims the
            // raw ptr and runs the CudaSlice field teardown (events/stream)
            // without any cuMemFree. The `vmm` field then frees the memory
            // correctly via its own Drop.
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

    /// Allocate the fused arena through the CUDA VMM API so it can be exported
    /// to an out-of-process PegaFlow server as a POSIX file descriptor.
    pub fn new_exportable(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        num_blocks: usize,
    ) -> anyhow::Result<Self> {
        let layout = KvLayout::new(num_layers, num_kv_heads, head_dim, page_size);
        let total_elements = num_blocks * layout.page_stride;
        let bytes = total_elements
            .checked_mul(std::mem::size_of::<bf16>())
            .ok_or_else(|| anyhow::anyhow!("KvBuffer size overflows usize"))?;
        let vmm = VmmExportableBuffer::alloc_zeroed(stream, bytes)
            .map_err(|e| anyhow::anyhow!("KvBuffer VMM alloc failed: {e}"))?;
        // Wrap the VMM pointer as a CudaSlice view for the attention kernels.
        // SAFETY: `vmm` keeps the pointer valid for `total_elements` bf16 and
        // owns teardown; the view is ManuallyDrop so it never frees the pointer.
        let view = unsafe { stream.upgrade_device_ptr::<bf16>(vmm.device_ptr(), total_elements) };
        Ok(Self::from_backing(
            Backing::Vmm {
                view: ManuallyDrop::new(view),
                vmm,
            },
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

    /// Exported POSIX fd for the fused allocation, or `None` for a
    /// non-exportable (default) buffer. The connector sends this to PegaFlow
    /// over the fd side-channel; the buffer retains ownership.
    pub fn export_fd(&self) -> Option<std::os::fd::BorrowedFd<'_>> {
        match &self.inner.backing {
            Backing::Owned(_) => None,
            Backing::Vmm { vmm, .. } => Some(vmm.export_fd()),
        }
    }

    /// Byte size of the exported VMM allocation (granularity-aligned), or `None`
    /// for a non-exportable buffer. PegaFlow reserves/maps exactly this size.
    pub fn export_alloc_size(&self) -> Option<usize> {
        match &self.inner.backing {
            Backing::Owned(_) => None,
            Backing::Vmm { vmm, .. } => Some(vmm.alloc_size()),
        }
    }

    /// Base device address of the fused KV buffer.
    ///
    /// Stable for the buffer's lifetime, so the KV-offload connector registers
    /// this once with pegaflow and the page-first [`KvLayout`] strides reach
    /// every (layer, block, K/V) segment from it.
    pub fn device_ptr(&self, stream: &CudaStream) -> u64 {
        let (ptr, _guard) = self.inner.backing.view().device_ptr(stream);
        ptr
    }

    pub fn num_blocks(&self) -> usize {
        self.inner.num_blocks
    }
}
