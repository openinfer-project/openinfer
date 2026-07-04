use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaStream};

#[derive(Clone)]
pub(crate) struct Glm52RankGpuContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    device_ordinal: usize,
}

// SAFETY: a GLM5.2 rank owns one CUDA context/stream pair. The worker binds it
// to its thread before touching device state.
unsafe impl Send for Glm52RankGpuContext {}
unsafe impl Sync for Glm52RankGpuContext {}

impl Glm52RankGpuContext {
    pub(crate) fn new(device_ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_ordinal)
            .with_context(|| format!("create GLM5.2 CUDA context for device {device_ordinal}"))?;
        ctx.bind_to_thread()
            .with_context(|| format!("bind GLM5.2 CUDA context for device {device_ordinal}"))?;
        retain_async_alloc_pool(device_ordinal)?;
        unsafe {
            ctx.disable_event_tracking();
        }
        let stream = ctx
            .new_stream()
            .with_context(|| format!("create GLM5.2 CUDA stream for device {device_ordinal}"))?;
        Ok(Self {
            ctx,
            stream,
            device_ordinal,
        })
    }

    pub(crate) fn set_current(&self) -> Result<()> {
        self.ctx.bind_to_thread().with_context(|| {
            format!(
                "bind GLM5.2 CUDA context for device {} to current thread",
                self.device_ordinal
            )
        })
    }

    pub(crate) fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .with_context(|| format!("synchronize GLM5.2 device {}", self.device_ordinal))
    }

    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The kernels-crate view of this rank's context/stream pair (shared
    /// Arcs, not a new context) — what the forward bricks take. Also performs
    /// the kernels-crate per-thread setup that `DeviceContext::new` would do:
    /// the cuBLAS handle is thread-local per device, so the calling worker
    /// thread must initialize its own (idempotent).
    pub(crate) fn device_context(&self) -> Result<openinfer_kernels::tensor::DeviceContext> {
        // SAFETY: plain device selection + idempotent handle creation.
        unsafe {
            let err = openinfer_kernels::ffi::cuda_set_device(self.device_ordinal as i32);
            ensure!(
                err == 0,
                "GLM5.2 cudaSetDevice({}) failed: cudaError={err}",
                self.device_ordinal
            );
            openinfer_kernels::ffi::cublas_init();
        }
        Ok(openinfer_kernels::tensor::DeviceContext {
            ctx: self.ctx.clone(),
            stream: self.stream.clone(),
            device_ordinal: self.device_ordinal,
        })
    }

    /// A second stream on the same context, for work that overlaps the main
    /// stream inside the decode step (fork/join via events — the shared
    /// expert runs concurrently with the MoE collectives).
    pub(crate) fn auxiliary_device_context(
        &self,
        role: &str,
    ) -> Result<openinfer_kernels::tensor::DeviceContext> {
        use anyhow::Context as _;
        let stream = self.ctx.new_stream().with_context(|| {
            format!(
                "failed to create GLM5.2 {role} stream for device {}",
                self.device_ordinal
            )
        })?;
        Ok(openinfer_kernels::tensor::DeviceContext {
            ctx: self.ctx.clone(),
            stream,
            device_ordinal: self.device_ordinal,
        })
    }
}

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
