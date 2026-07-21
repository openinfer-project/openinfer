//! CUDA allocations that can be shared through legacy CUDA IPC handles.

use std::sync::Arc;

use cudarc::driver::{
    CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits,
    result::DriverError,
    sys::{self, CUresult},
};

/// Allocate zeroed, pointer-stable device memory through `cuMemAlloc`.
///
/// cudarc normally uses `cuMemAllocAsync` on modern GPUs. Those allocations
/// cannot be exported by `cuIpcGetMemHandle`, while persistent KV arenas must
/// be visible to an out-of-process PegaFlow server. `CudaSlice` may release a
/// `cuMemAlloc` allocation through `cuMemFreeAsync`, so the returned value
/// retains the normal stream-ordered drop behavior.
pub fn alloc_ipc_zeros<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>, DriverError> {
    if len == 0 {
        return stream.alloc_zeros(0);
    }
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(DriverError(CUresult::CUDA_ERROR_INVALID_VALUE))?;
    stream.context().bind_to_thread()?;

    let mut ptr = 0;
    // SAFETY: `ptr` is valid output storage, the CUDA context is current, and
    // the allocation is immediately transferred into a CudaSlice owner.
    unsafe { sys::cuMemAlloc_v2(&raw mut ptr, bytes).result()? };
    // SAFETY: `ptr` owns `bytes == len * size_of::<T>()` device bytes on this
    // stream's context. `memset_zeros` establishes valid zero bits for T.
    let mut slice = unsafe { stream.upgrade_device_ptr(ptr, len) };
    stream.memset_zeros(&mut slice)?;
    Ok(slice)
}
