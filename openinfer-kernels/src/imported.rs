//! GPU memory imported from an out-of-process PegaFlow server.
//!
//! PegaFlow allocates the fused KV arena in its own process (which is what
//! lets it register the memory into the NIC for GPUDirect RDMA — owner-side
//! `ibv_reg_mr`/dma-buf registration works, an IPC-imported pointer never
//! does) and returns a CUDA IPC handle in the registration response. This side
//! only runs compute kernels on the imported mapping, which CUDA IPC fully
//! supports.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use cudarc::driver::result::DriverError;
use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::{self};

/// Byte length of a serialized `CUipcMemHandle`.
pub const IPC_HANDLE_BYTES: usize = 64;

/// A device mapping of a PegaFlow-owned allocation, imported via CUDA IPC.
///
/// The server owns the memory and frees it on unregister; this mapping must be
/// closed (drop) before the engine unregisters. The pointer is stable for the
/// mapping's lifetime.
pub struct ImportedKvArena {
    stream: Arc<CudaStream>,
    ptr: sys::CUdeviceptr,
    size_bytes: usize,
}

impl ImportedKvArena {
    /// Import `ipc_handle` (the 64-byte `CUipcMemHandle` from PegaFlow's
    /// registration response) on `stream`'s device. `size_bytes` is the arena
    /// size this side needs; the actual allocation is queried from the driver
    /// and must cover it — trusting the client's own number would let a
    /// server-side under-allocation turn into silent out-of-bounds writes.
    pub fn open(
        stream: &Arc<CudaStream>,
        ipc_handle: &[u8],
        size_bytes: usize,
    ) -> Result<Self, DriverError> {
        if ipc_handle.len() != IPC_HANDLE_BYTES || size_bytes == 0 {
            return Err(DriverError(CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        let ctx = stream.context();
        ctx.bind_to_thread()?;

        let mut handle = sys::CUipcMemHandle { reserved: [0; 64] };
        for (dst, src) in handle.reserved.iter_mut().zip(ipc_handle) {
            *dst = *src as i8;
        }
        let mut ptr: sys::CUdeviceptr = 0;
        // SAFETY: the handle references a live server-owned allocation; the
        // server keeps it alive until this instance unregisters.
        unsafe {
            sys::cuIpcOpenMemHandle_v2(
                &raw mut ptr,
                handle,
                sys::CUipcMem_flags_enum::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as u32,
            )
            .result()?;
        }

        let mut range_base: sys::CUdeviceptr = 0;
        let mut range_size: usize = 0;
        // SAFETY: ptr is a live imported mapping; the driver reports the
        // allocation it belongs to.
        let range = unsafe {
            sys::cuMemGetAddressRange_v2(&raw mut range_base, &raw mut range_size, ptr).result()
        };
        let actual = match range {
            Ok(()) => range_size - (ptr - range_base) as usize,
            Err(err) => {
                // SAFETY: close the mapping we just opened before bailing.
                unsafe { sys::cuIpcCloseMemHandle(ptr).result().ok() };
                return Err(err);
            }
        };
        if actual < size_bytes {
            // SAFETY: as above.
            unsafe { sys::cuIpcCloseMemHandle(ptr).result().ok() };
            return Err(DriverError(CUresult::CUDA_ERROR_INVALID_VALUE));
        }

        Ok(Self {
            stream: Arc::clone(stream),
            ptr,
            size_bytes,
        })
    }

    /// Base device address of the imported arena. Stable for the mapping's
    /// lifetime.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Arena size in bytes as registered with the server.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}

impl Drop for ImportedKvArena {
    fn drop(&mut self) {
        self.stream
            .context()
            .bind_to_thread()
            .expect("bind CUDA context before closing imported KV arena");
        // SAFETY: `ptr` came from cuIpcOpenMemHandle in `open`.
        unsafe { sys::cuIpcCloseMemHandle(self.ptr).result() }
            .expect("close imported KV arena mapping");
    }
}
