use std::cell::Cell;

use anyhow::{Result, ensure};
use openinfer_core::{ffi, tensor::DeviceContext};

thread_local! {
    static ACTIVE_DEVICE: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(crate) fn activate(ctx: &DeviceContext) -> Result<()> {
    ACTIVE_DEVICE.with(|active| {
        if active.get() == Some(ctx.device_ordinal) {
            return Ok(());
        }
        unsafe {
            ffi::cublas_destroy();
            let err = ffi::cuda_set_device(ctx.device_ordinal as i32);
            ensure!(
                err == 0,
                "failed to activate CUDA device {}: cudaError={err}",
                ctx.device_ordinal
            );
            ffi::cublas_init();
        }
        active.set(Some(ctx.device_ordinal));
        Ok(())
    })
}
