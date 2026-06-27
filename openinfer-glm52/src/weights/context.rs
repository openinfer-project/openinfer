use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaStream};
use openinfer_kernels::ffi;
use openinfer_kernels::tensor::DeviceContext;

#[derive(Clone)]
pub(crate) struct Glm52RankGpuContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    device_ordinal: usize,
}

// SAFETY: a GLM5.2 rank owns one CUDA context/stream pair. The rank worker
// binds it to its thread before touching device state.
unsafe impl Send for Glm52RankGpuContext {}
unsafe impl Sync for Glm52RankGpuContext {}

impl Glm52RankGpuContext {
    pub(crate) fn new(device_ordinal: usize) -> Result<Self> {
        Self::set_current_device(device_ordinal)?;
        let ctx = CudaContext::new(device_ordinal).with_context(|| {
            format!("failed to create GLM5.2 CUDA context for device {device_ordinal}")
        })?;
        retain_async_alloc_pool(device_ordinal)?;
        unsafe {
            ctx.disable_event_tracking();
            ffi::cublas_init();
        }
        let stream = ctx.new_stream().with_context(|| {
            format!("failed to create GLM5.2 CUDA stream for device {device_ordinal}")
        })?;
        Ok(Self {
            ctx,
            stream,
            device_ordinal,
        })
    }

    pub(crate) fn set_current(&self) -> Result<()> {
        Self::set_current_device(self.device_ordinal)?;
        self.ctx.bind_to_thread().with_context(|| {
            format!(
                "failed to bind GLM5.2 CUDA context for device {} to current thread",
                self.device_ordinal
            )
        })
    }

    pub(crate) fn sync(&self) -> Result<()> {
        self.stream.synchronize().with_context(|| {
            format!(
                "failed to synchronize GLM5.2 device {}",
                self.device_ordinal
            )
        })
    }

    /// Bridge to the kernel-launch `DeviceContext` used by every glm52 op.
    pub(crate) fn as_device_context(&self) -> DeviceContext {
        DeviceContext {
            ctx: Arc::clone(&self.ctx),
            stream: Arc::clone(&self.stream),
            device_ordinal: self.device_ordinal,
        }
    }

    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    fn set_current_device(device_ordinal: usize) -> Result<()> {
        let err = unsafe { ffi::cuda_set_device(device_ordinal as i32) };
        ensure!(
            err == 0,
            "failed to set GLM5.2 CUDA device {device_ordinal}: cudaError={err}"
        );
        Ok(())
    }
}

/// Make the device's default async-allocation pool RETAIN freed blocks rather
/// than return them to the driver on every stream sync (the CUDA default release
/// threshold is 0). cudarc allocates via `cuMemAllocAsync` on this pool, and the
/// bs=1 PP decode does a host sync per stage per token — without retention every
/// per-call `alloc_zeros` round-trips the driver, which dominates the per-token
/// cost (~10x the memory-bound floor). Retention turns them into pool hits.
fn retain_async_alloc_pool(device_ordinal: usize) -> Result<()> {
    use cudarc::driver::sys;
    unsafe {
        let mut dev: sys::CUdevice = 0;
        check_cu(
            sys::cuDeviceGet(&mut dev, device_ordinal as i32),
            "cuDeviceGet",
        )?;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        check_cu(
            sys::cuDeviceGetDefaultMemPool(&mut pool, dev),
            "cuDeviceGetDefaultMemPool",
        )?;
        let mut threshold: u64 = u64::MAX;
        check_cu(
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast::<std::ffi::c_void>(),
            ),
            "cuMemPoolSetAttribute(RELEASE_THRESHOLD)",
        )?;
    }
    Ok(())
}

fn check_cu(result: cudarc::driver::sys::CUresult, what: &str) -> Result<()> {
    ensure!(
        result == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "GLM5.2 {what} failed: {result:?}"
    );
    Ok(())
}
