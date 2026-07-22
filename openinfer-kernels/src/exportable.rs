//! GPU allocations that can be shared with an out-of-process PegaFlow server.
//!
//! The KV arena must be visible to a separate PegaFlow process so it can copy
//! to the host/SSD tiers and — unlike CUDA IPC — register it into the NIC for
//! GPUDirect RDMA. That rules out `cuMemAlloc`+`cuIpcGetMemHandle`: an imported
//! IPC pointer cannot be `ibv_reg_mr`'d. Instead we allocate through the VMM API
//! (`cuMemCreate`) with a POSIX-fd shareable handle; PegaFlow imports the fd,
//! maps it, and can hand the mapping to `ibv_reg_dmabuf_mr`.

use std::sync::Arc;

use cudarc::driver::{
    CudaStream,
    result::DriverError,
    sys::{self, CUresult},
};

/// A zeroed, pointer-stable device allocation made through the CUDA VMM API and
/// exportable to another process as a POSIX file descriptor.
///
/// Owns the reserved VA range, the physical handle, and the export fd. Drops
/// them in reverse order (unmap → free VA → release handle → close fd). The
/// device pointer is stable for the buffer's lifetime, so the KV-offload
/// connector registers it once with PegaFlow.
pub struct VmmExportableBuffer {
    stream: Arc<CudaStream>,
    handle: sys::CUmemGenericAllocationHandle,
    ptr: sys::CUdeviceptr,
    /// Allocation size rounded up to the VMM granularity (what was reserved and
    /// mapped; also what the importer must reserve).
    alloc_size: usize,
    /// Exported POSIX fd for the physical handle. Sent to PegaFlow over the fd
    /// side-channel; owned here and closed on drop.
    export_fd: std::os::fd::OwnedFd,
}

impl VmmExportableBuffer {
    /// Allocate `bytes` (rounded up to the VMM granularity) of zeroed device
    /// memory on `stream`'s device, mapped read/write and exported as a POSIX fd.
    pub fn alloc_zeroed(stream: &Arc<CudaStream>, bytes: usize) -> Result<Self, DriverError> {
        if bytes == 0 {
            return Err(DriverError(CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        let ctx = stream.context();
        ctx.bind_to_thread()?;
        // VMM location.id wants the CUdevice, not the cudarc ordinal (they
        // coincide for device 0 but not in general).
        let device_id = ctx.cu_device();

        let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        prop.requestedHandleTypes =
            sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        // Advertise the allocation as GPUDirect-RDMA capable only where the
        // device actually supports it. Consumer GeForce parts do not, and
        // `cuMemCreate` rejects the flag there (surfaces as INVALID_DEVICE).
        // Data-center parts (H100/H200/B200) set it, which is what lets the
        // importer register the allocation into the NIC.
        prop.allocFlags.gpuDirectRDMACapable = u8::from(device_supports_gdr(ctx.cu_device()));

        // Round the request up to the allocation granularity.
        let mut granularity: usize = 0;
        // SAFETY: prop is fully initialized; granularity is valid out storage.
        unsafe {
            sys::cuMemGetAllocationGranularity(
                &raw mut granularity,
                &raw const prop,
                sys::CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
            .result()?;
        }
        let alloc_size = bytes.div_ceil(granularity) * granularity;

        // Create the physical allocation.
        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        // SAFETY: prop is initialized; handle is valid out storage.
        unsafe { sys::cuMemCreate(&raw mut handle, alloc_size, &raw const prop, 0).result()? };

        // Reserve VA and map, unwinding the physical handle on any failure.
        let mut ptr: sys::CUdeviceptr = 0;
        // SAFETY: standard VMM reserve; default alignment/addr.
        if let Err(e) =
            unsafe { sys::cuMemAddressReserve(&raw mut ptr, alloc_size, 0, 0, 0).result() }
        {
            unsafe { sys::cuMemRelease(handle).result().ok() };
            return Err(e);
        }
        // SAFETY: ptr..+alloc_size just reserved; handle freshly created.
        if let Err(e) = unsafe { sys::cuMemMap(ptr, alloc_size, 0, handle, 0).result() } {
            unsafe { sys::cuMemAddressFree(ptr, alloc_size).result().ok() };
            unsafe { sys::cuMemRelease(handle).result().ok() };
            return Err(e);
        }

        // Grant this device read/write access.
        let access = sys::CUmemAccessDesc {
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            flags: sys::CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: ptr..+alloc_size is mapped; one access descriptor.
        if let Err(e) =
            unsafe { sys::cuMemSetAccess(ptr, alloc_size, &raw const access, 1).result() }
        {
            Self::teardown(ptr, alloc_size, handle);
            return Err(e);
        }

        // Export the physical handle as a POSIX fd for the PegaFlow importer.
        let mut raw_fd: std::os::raw::c_int = -1;
        // SAFETY: handle is a live POSIX-fd-capable allocation; out storage valid.
        if let Err(e) = unsafe {
            sys::cuMemExportToShareableHandle(
                (&raw mut raw_fd).cast(),
                handle,
                sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            )
            .result()
        } {
            Self::teardown(ptr, alloc_size, handle);
            return Err(e);
        }
        // SAFETY: CUDA returned a fresh owned fd we are responsible for closing.
        let export_fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(raw_fd) };

        // Zero the mapped range so a KV hit never reads stale device memory.
        // SAFETY: ptr..+alloc_size is mapped RW on the current context.
        if let Err(e) = unsafe { sys::cuMemsetD8_v2(ptr, 0, alloc_size).result() } {
            // export_fd drops (closes) here; then unwind the mapping.
            drop(export_fd);
            Self::teardown(ptr, alloc_size, handle);
            return Err(e);
        }

        Ok(Self {
            stream: Arc::clone(stream),
            handle,
            ptr,
            alloc_size,
            export_fd,
        })
    }

    /// Base device address of the mapped allocation. Stable for the buffer's
    /// lifetime.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Size actually reserved and mapped (request rounded up to granularity).
    pub fn alloc_size(&self) -> usize {
        self.alloc_size
    }

    /// Borrow the export fd to send over the fd side-channel. The buffer retains
    /// ownership; the receiver (CUDA) dups it, so the borrow is sufficient.
    pub fn export_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        self.export_fd.as_fd()
    }

    fn teardown(
        ptr: sys::CUdeviceptr,
        alloc_size: usize,
        handle: sys::CUmemGenericAllocationHandle,
    ) {
        // SAFETY: called only with a mapped ptr/reserved VA/live handle to unwind.
        unsafe {
            sys::cuMemUnmap(ptr, alloc_size).result().ok();
            sys::cuMemAddressFree(ptr, alloc_size).result().ok();
            sys::cuMemRelease(handle).result().ok();
        }
    }
}

use std::os::fd::{AsFd, FromRawFd};

/// Whether `device` supports registering a VMM allocation for GPUDirect RDMA.
/// Data-center parts (H100/H200/B200) return true; consumer GeForce parts
/// return false, and setting `gpuDirectRDMACapable` on them makes `cuMemCreate`
/// fail with INVALID_DEVICE.
fn device_supports_gdr(device: sys::CUdevice) -> bool {
    let mut supported: std::os::raw::c_int = 0;
    // SAFETY: `supported` is valid out storage; the attribute + device are valid.
    let ok = unsafe {
        sys::cuDeviceGetAttribute(
            &raw mut supported,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
            device,
        )
    };
    ok == CUresult::CUDA_SUCCESS && supported != 0
}

impl Drop for VmmExportableBuffer {
    fn drop(&mut self) {
        // Bind the owning context before touching the VMM mapping; export_fd is
        // closed by its own Drop after this returns.
        self.stream
            .context()
            .bind_to_thread()
            .expect("bind CUDA context before releasing VMM buffer");
        Self::teardown(self.ptr, self.alloc_size, self.handle);
    }
}
