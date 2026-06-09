use anyhow::{Result, bail, ensure};
use cudarc::driver::{DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceMatrix, GpuTensor, HiddenStates};

use super::experts::{KIMI_K2_HIDDEN, KIMI_K2_SHARED_GATE_UP};

const KIMI_SHARED_GATE_UP_CUBLASLT_MAX_BATCH: usize = 64;

pub fn kimi_shared_gate_up_cublaslt_supports_batch_size(batch_size: usize) -> bool {
    (1..=KIMI_SHARED_GATE_UP_CUBLASLT_MAX_BATCH).contains(&batch_size)
}

pub fn kimi_shared_gate_up_cublaslt_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &GpuTensor<KIMI_K2_HIDDEN>,
    out: &mut HiddenStates,
) -> Result<()> {
    let batch_size = x.seq_len;
    ensure!(
        weight.rows == KIMI_K2_SHARED_GATE_UP && weight.cols == KIMI_K2_HIDDEN,
        "Kimi shared_gate_up cuBLASLt weight shape mismatch: got [{},{}], expected [{},{}]",
        weight.rows,
        weight.cols,
        KIMI_K2_SHARED_GATE_UP,
        KIMI_K2_HIDDEN
    );
    ensure!(
        out.hidden_dim == KIMI_K2_SHARED_GATE_UP && out.seq_len == batch_size,
        "Kimi shared_gate_up cuBLASLt output shape mismatch: out=[{},{}], batch_size={}",
        out.hidden_dim,
        out.seq_len,
        batch_size
    );
    ensure!(
        kimi_shared_gate_up_cublaslt_supports_batch_size(batch_size),
        "Kimi shared_gate_up cuBLASLt supports batch_size 1..={}; got {}",
        KIMI_SHARED_GATE_UP_CUBLASLT_MAX_BATCH,
        batch_size
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        let status = ffi::kimi_shared_gate_up_cublaslt_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            KIMI_K2_SHARED_GATE_UP as i32,
            batch_size as i32,
            KIMI_K2_HIDDEN as i32,
            ctx.stream.cu_stream(),
        );
        if status != 0 {
            if status >= 100_000 {
                bail!(
                    "Kimi shared_gate_up cuBLASLt failed: cublas_status={}, batch_size={}",
                    status - 100_000,
                    batch_size
                );
            }
            bail!(
                "Kimi shared_gate_up cuBLASLt launch failed: cuda_status={}, batch_size={}",
                status,
                batch_size
            );
        }
    }
    Ok(())
}
