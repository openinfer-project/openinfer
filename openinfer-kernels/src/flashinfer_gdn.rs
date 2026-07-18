//! Optional FlashInfer SM120 GDN prefill launcher.
//!
//! The generated CuTe artifact is deliberately hidden behind this small
//! pointer-level API.  Qwen3.5 owns tensor validation and state conversion;
//! this module owns whether an artifact was produced by `build.rs`.

use anyhow::{Result, anyhow};
use cudarc::driver::sys::{CUresult, CUstream};

use crate::ffi;

/// Returns whether build.rs emitted the optional SM120 CuTe artifact.
pub const fn aot_available() -> bool {
    cfg!(flashinfer_gdn_aot)
}

fn check_cuda(result: CUresult, operation: &str) -> Result<()> {
    if result as u32 == 0 {
        Ok(())
    } else {
        Err(anyhow!("{operation} returned CUDA result {result:?}"))
    }
}

/// Launch the generated SM120 FlashInfer GDN artifact when available.
///
/// Returns `Ok(true)` when the artifact launched and `Ok(false)` when this
/// build did not generate it.  The latter is the normal path for CI,
/// unsupported architectures, and builds that intentionally retain Triton.
#[allow(clippy::too_many_arguments)]
pub fn launch_prefill(
    q: *const ffi::Half,
    k: *const ffi::Half,
    v: *const ffi::Half,
    output: *mut ffi::Half,
    alpha: *const f32,
    beta: *const f32,
    state: *mut f32,
    init_state: *const f32,
    tensormaps: *mut u8,
    cu_seqlens: *const i64,
    seq_len: i32,
    stream: CUstream,
) -> Result<bool> {
    #[cfg(flashinfer_gdn_aot)]
    {
        // SAFETY: callers obtain every pointer from live cudarc allocations
        // on the same CUDA stream and keep those allocations alive until the
        // stream-ordered launch completes.
        let result = unsafe {
            ffi::openinfer_qwen35_gdn_sm120_cuda(
                q, k, v, output, alpha, beta, state, init_state, tensormaps, cu_seqlens, seq_len,
                stream,
            )
        };
        check_cuda(result, "FlashInfer SM120 GDN prefill")?;
        return Ok(true);
    }

    #[cfg(not(flashinfer_gdn_aot))]
    {
        let _ = (
            q, k, v, output, alpha, beta, state, init_state, tensormaps, cu_seqlens, seq_len,
            stream,
        );
        Ok(false)
    }
}

/// Transpose one recurrent state matrix between the OpenInfer and FlashInfer
/// layouts.  This kernel is available whenever the Qwen3.5 CUDA kernels are
/// built, independent of whether the optional CuTe artifact exists.
pub fn transpose_state(
    src: *const f32,
    dst: *mut f32,
    num_heads: i32,
    key_dim: i32,
    value_dim: i32,
    to_flashinfer: bool,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: callers pass valid device pointers and a stream from the owning
    // DeviceContext; the CUDA kernel performs a bounds-checked out-of-place
    // transpose.
    let result = unsafe {
        ffi::gated_delta_rule_state_transpose_cuda(
            src,
            dst,
            num_heads,
            key_dim,
            value_dim,
            to_flashinfer,
            stream,
        )
    };
    check_cuda(result, "GDN state transpose")
}

/// Convert per-token log-gates to FlashInfer's positive alpha factors.
pub fn exp_gate(input: *const f32, output: *mut f32, length: i32, stream: CUstream) -> Result<()> {
    let result =
        unsafe { ffi::gated_delta_rule_prefill_gate_exp_cuda(input, output, length, stream) };
    check_cuda(result, "GDN gate exponentiation")
}
