use super::*;

impl KimiRankGpuContext {
    pub fn new(device_ordinal: usize) -> Result<Self> {
        Self::set_current_device(device_ordinal)?;
        let ctx = CudaContext::new(device_ordinal).with_context(|| {
            format!("failed to create CUDA context for device {device_ordinal}")
        })?;
        unsafe {
            ctx.disable_event_tracking();
        }
        let stream = ctx
            .new_stream()
            .with_context(|| format!("failed to create CUDA stream for device {device_ordinal}"))?;
        unsafe {
            ffi::cublas_init();
        }
        Ok(Self {
            ctx,
            stream,
            device_ordinal,
        })
    }

    pub fn set_current(&self) -> Result<()> {
        Self::set_current_device(self.device_ordinal)?;
        self.ctx.bind_to_thread().with_context(|| {
            format!(
                "failed to bind Kimi CUDA context for device {} to current thread",
                self.device_ordinal
            )
        })
    }

    pub(crate) fn as_device_context(&self) -> DeviceContext {
        DeviceContext {
            ctx: Arc::clone(&self.ctx),
            stream: Arc::clone(&self.stream),
            device_ordinal: self.device_ordinal,
        }
    }

    fn set_current_device(device_ordinal: usize) -> Result<()> {
        let err = unsafe { ffi::cuda_set_device(device_ordinal as i32) };
        ensure!(
            err == 0,
            "failed to set Kimi CUDA device {device_ordinal}: cudaError={err}"
        );
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .with_context(|| format!("failed to synchronize Kimi device {}", self.device_ordinal))
    }
}
