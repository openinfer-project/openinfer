use std::{
    ffi::c_void,
    mem::{size_of, size_of_val},
    ptr::null_mut,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

use crate::CudaError;

type GdrResult<T> = Result<T, CudaError>;

/// Reference-counted GDR context handle.
struct GdrContextHandle {
    handle: gdrapi_sys::gdr_t,
}
unsafe impl Send for GdrContextHandle {}
unsafe impl Sync for GdrContextHandle {}

impl Drop for GdrContextHandle {
    fn drop(&mut self) {
        unsafe { gdrapi_sys::gdr_close(self.handle) };
    }
}

/// Backing for the small CPU-mapped control buffers the a2a proxy writes and the
/// a2a CUDA kernel spins on. Two modes with identical semantics:
///
/// - `Gdr`: the production path. GDRCopy pins GPU memory and maps its BAR into
///   CPU address space, so the proxy pokes GPU-resident flags directly.
/// - `HostPinned`: a fallback for hosts **without the `gdrdrv` kernel module**
///   (`gdr_open` returns null when `/dev/gdrdrv` is absent and the caller has no
///   `CAP_SYS_MODULE` to `insmod` it). It mirrors GDRCopy the other way round —
///   `cuMemHostAlloc(DEVICEMAP)` + `cuMemHostGetDevicePointer` map host-pinned
///   memory into the GPU's address space, so the kernel spins on it over PCIe
///   (higher small-message latency) while the proxy reads/writes it locally. The
///   RDMA data plane is untouched (it never used GDRCopy — it registers device
///   buffers via `ibv_reg_mr`/dma-buf), so this only degrades control-flag
///   signalling latency and lets P/D benchmarks run without root kernel access.
enum GdrMode {
    Gdr(Arc<GdrContextHandle>),
    HostPinned,
}

/// Public wrapper around the GDRCopy context (or its host-pinned fallback).
pub struct GdrCopyContext {
    mode: GdrMode,
}

fn align_to(ptr: u64, alignment: usize) -> u64 {
    (ptr + alignment as u64 - 1).div_ceil(alignment as u64) * alignment as u64
}

impl GdrCopyContext {
    pub fn new() -> GdrResult<Self> {
        let handle = unsafe { gdrapi_sys::gdr_open() };
        if handle.is_null() {
            // gdrdrv kernel module not loaded — fall back to host-pinned mapped
            // memory rather than failing. Only control-flag latency degrades.
            log::warn!(
                "gdrdrv unavailable (gdr_open returned null); using host-pinned mapped \
                 memory for a2a control flags — higher small-message latency, identical \
                 semantics, RDMA data path unaffected"
            );
            return Ok(GdrCopyContext { mode: GdrMode::HostPinned });
        }
        Ok(GdrCopyContext { mode: GdrMode::Gdr(Arc::new(GdrContextHandle { handle })) })
    }

    /// True when running on the host-pinned fallback (no gdrdrv).
    pub fn is_host_pinned_fallback(&self) -> bool {
        matches!(self.mode, GdrMode::HostPinned)
    }

    fn alloc_buffer(&self, nbytes: usize) -> GdrResult<GdrBuffer> {
        match &self.mode {
            GdrMode::Gdr(context) => Self::alloc_gdr(context, nbytes),
            GdrMode::HostPinned => Self::alloc_host_pinned(nbytes),
        }
    }

    fn alloc_gdr(
        context: &Arc<GdrContextHandle>,
        nbytes: usize,
    ) -> GdrResult<GdrBuffer> {
        let mut device_ptr: u64 = 0;
        let page_size: usize = 1 << 16; // 64KB page size
        let bytesize = nbytes.div_ceil(page_size) * page_size;

        if unsafe { cuda_sys::cuMemAlloc(&mut device_ptr, bytesize + page_size) }
            != cuda_sys::CUDA_SUCCESS
        {
            return Err(CudaError::GdrCopyError("Failed to allocate GDR buffer"));
        }

        let aligned_device_ptr = align_to(device_ptr, page_size);
        let context = context.clone();
        let g = context.handle;
        let mut mh = gdrapi_sys::gdr_mh_t { h: 0 };

        let ret = unsafe {
            gdrapi_sys::gdr_pin_buffer(g, aligned_device_ptr, bytesize, 0, 0, &mut mh)
        };
        if ret != 0 {
            unsafe { cuda_sys::cuMemFree(device_ptr) };
            return Err(CudaError::GdrCopyError("Failed to pin GDR buffer"));
        }

        let mut mapped_ptr: *mut c_void = null_mut();
        let ret = unsafe { gdrapi_sys::gdr_map(g, mh, &mut mapped_ptr, bytesize) };
        if ret != 0 {
            unsafe {
                gdrapi_sys::gdr_unpin_buffer(g, mh);
                cuda_sys::cuMemFree(device_ptr);
            };
            return Err(CudaError::GdrCopyError("Failed to map GDR buffer"));
        }

        Ok(GdrBuffer {
            inner: GdrBufferInner::Gdr {
                device_ptr,
                aligned_device_ptr,
                mapped_ptr,
                bytesize,
                mh,
                context,
            },
        })
    }

    fn alloc_host_pinned(nbytes: usize) -> GdrResult<GdrBuffer> {
        let page_size: usize = 1 << 16;
        let bytesize = nbytes.div_ceil(page_size) * page_size;

        // DEVICEMAP: the allocation also gets a device-accessible address so the
        // a2a kernel can poll it over PCIe. WRITECOMBINED is deliberately NOT set
        // — the CPU both reads (flag waits) and writes these bytes.
        let mut host_ptr: *mut c_void = null_mut();
        if unsafe {
            cuda_sys::cuMemHostAlloc(
                &mut host_ptr,
                bytesize,
                cuda_sys::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        } != cuda_sys::CUDA_SUCCESS
        {
            return Err(CudaError::GdrCopyError(
                "Failed to allocate host-pinned fallback buffer",
            ));
        }

        let mut device_ptr: u64 = 0;
        if unsafe {
            cuda_sys::cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr, 0)
        } != cuda_sys::CUDA_SUCCESS
        {
            unsafe { cuda_sys::cuMemFreeHost(host_ptr) };
            return Err(CudaError::GdrCopyError(
                "Failed to map host-pinned fallback buffer to the device",
            ));
        }

        // Flags start unset; the CPU allocation is not zeroed by the driver.
        unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, bytesize) };

        Ok(GdrBuffer {
            inner: GdrBufferInner::HostPinned { host_ptr, device_ptr, bytesize },
        })
    }
}

/// Raw control buffer visible to both the GPU (`device_ptr`) and the CPU proxy.
enum GdrBufferInner {
    Gdr {
        device_ptr: u64,
        aligned_device_ptr: u64,
        mapped_ptr: *mut c_void,
        mh: gdrapi_sys::gdr_mh_t,
        bytesize: usize,
        context: Arc<GdrContextHandle>,
    },
    HostPinned {
        host_ptr: *mut c_void,
        device_ptr: u64,
        #[allow(dead_code)]
        bytesize: usize,
    },
}

struct GdrBuffer {
    inner: GdrBufferInner,
}

unsafe impl Send for GdrBuffer {}
unsafe impl Sync for GdrBuffer {}

impl Drop for GdrBuffer {
    fn drop(&mut self) {
        match &self.inner {
            GdrBufferInner::Gdr {
                device_ptr,
                mapped_ptr,
                mh,
                bytesize,
                context,
                ..
            } => {
                let g = context.handle;
                unsafe {
                    gdrapi_sys::gdr_unmap(g, *mh, *mapped_ptr, *bytesize);
                    gdrapi_sys::gdr_unpin_buffer(g, *mh);
                    cuda_sys::cuMemFree(*device_ptr);
                };
            }
            GdrBufferInner::HostPinned { host_ptr, .. } => {
                unsafe { cuda_sys::cuMemFreeHost(*host_ptr) };
            }
        }
    }
}

trait GdrRead {
    fn read(cpu_ptr: *mut c_void) -> Self;
}

impl GdrRead for u8 {
    #[inline(always)]
    fn read(cpu_ptr: *mut c_void) -> Self {
        let flag = unsafe { AtomicU8::from_ptr(cpu_ptr as *mut u8) };
        flag.load(Ordering::Acquire)
    }
}

trait GdrWrite {
    fn write(cpu_ptr: *mut c_void, value: Self);
}

impl GdrWrite for u8 {
    #[inline(always)]
    fn write(cpu_ptr: *mut c_void, value: Self) {
        let flag = unsafe { AtomicU8::from_ptr(cpu_ptr as *mut u8) };
        flag.store(value, Ordering::Release);
    }
}

impl GdrBuffer {
    /// The GPU-side address the a2a kernel reads/writes.
    fn get_device_ptr(&self) -> *mut c_void {
        match &self.inner {
            GdrBufferInner::Gdr { aligned_device_ptr, .. } => {
                *aligned_device_ptr as *mut c_void
            }
            GdrBufferInner::HostPinned { device_ptr, .. } => *device_ptr as *mut c_void,
        }
    }

    /// The CPU-side view of the same bytes the proxy thread pokes.
    fn cpu_ptr(&self) -> *mut c_void {
        match &self.inner {
            GdrBufferInner::Gdr { mapped_ptr, .. } => *mapped_ptr,
            GdrBufferInner::HostPinned { host_ptr, .. } => *host_ptr,
        }
    }

    #[inline(always)]
    fn read<T: GdrRead>(&self) -> T {
        T::read(self.cpu_ptr())
    }

    #[inline(always)]
    fn write<T: GdrWrite>(&self, value: T) {
        T::write(self.cpu_ptr(), value);
    }

    fn copy_to(&self, src: *const c_void, nbytes: usize) {
        match &self.inner {
            GdrBufferInner::Gdr { mh, mapped_ptr, .. } => unsafe {
                gdrapi_sys::gdr_copy_to_mapping(*mh, *mapped_ptr, src, nbytes);
            },
            GdrBufferInner::HostPinned { host_ptr, .. } => unsafe {
                // Host-pinned memory is plain CPU memory — a direct copy suffices.
                std::ptr::copy_nonoverlapping(
                    src as *const u8,
                    *host_ptr as *mut u8,
                    nbytes,
                );
            },
        }
    }
}

/// Byte-flag implemented over a GDR (or host-pinned) control buffer.
pub struct GdrFlag {
    buffer: GdrBuffer,
}

impl GdrFlag {
    pub fn new(context: &GdrCopyContext) -> GdrResult<Self> {
        let buffer = context.alloc_buffer(size_of::<u8>())?;
        Ok(GdrFlag { buffer })
    }

    pub fn wait(&self) {
        while !self.is_set() {
            std::hint::spin_loop();
        }
        self.set(false);
    }

    pub fn get_device_ptr(&self) -> *mut u8 {
        self.buffer.get_device_ptr() as *mut u8
    }

    pub fn set(&self, value: bool) {
        self.buffer.write(value as u8);
    }

    fn is_set(&self) -> bool {
        self.buffer.read::<u8>() != 0
    }
}

pub struct GdrVec<T: Sized> {
    buffer: GdrBuffer,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Sized> GdrVec<T> {
    pub fn new(context: &GdrCopyContext, len: usize) -> GdrResult<Self> {
        let buffer = context.alloc_buffer(len * size_of::<T>())?;
        Ok(GdrVec { buffer, len, _marker: std::marker::PhantomData })
    }

    pub fn get_device_ptr(&self) -> *mut T {
        self.buffer.get_device_ptr().cast::<T>()
    }

    pub fn copy(&self, value: &[T]) {
        debug_assert!(value.len() <= self.len);
        self.buffer.copy_to(value.as_ptr() as *const c_void, size_of_val(value));
    }
}
