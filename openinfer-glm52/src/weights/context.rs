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

    /// Bridge to the kernel-launch `DeviceContext`. Unreferenced until the PP8
    /// forward (Slice 3+) starts issuing kernels; kept as the canonical accessor.
    #[allow(dead_code)]
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
