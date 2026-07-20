//! Batched GPU logprobs reduction (#719): log-sum-exp, picked-token logprob,
//! and deterministic top-k over bf16 logits rows, replacing the per-row
//! full-vocab D2H and host O(V) passes.

use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, HiddenStates};

/// `CUDA_ERROR_NOT_SUPPORTED` — FilteredTopK needs Hopper-class smem.
const CUDA_ERROR_NOT_SUPPORTED: i32 = 801;

/// log-sum-exp + picked-token logprob for `num_rows` scored rows.
///
/// `row_indices == None` scores rows `0..num_rows` contiguously; otherwise
/// row `i` of the output scores logits row `row_indices[i]`. `picked[r]` is
/// the sampled token of scored row `r`.
pub fn logprobs_lse_bf16_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    row_indices: Option<&CudaSlice<u32>>,
    picked: &CudaSlice<u32>,
    num_rows: usize,
    out_lse: &mut CudaSlice<f32>,
    out_picked_lp: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(num_rows > 0, "logprobs lse requires num_rows > 0");
    ensure!(
        num_rows <= logits.seq_len || row_indices.is_some(),
        "logprobs lse: num_rows {num_rows} exceeds logits seq_len {} without row indices",
        logits.seq_len
    );
    if let Some(indices) = row_indices {
        ensure!(
            indices.len() >= num_rows,
            "logprobs lse row_indices too small: have {}, need {num_rows}",
            indices.len()
        );
    }
    ensure!(
        picked.len() >= num_rows,
        "logprobs lse picked too small: have {}, need {num_rows}",
        picked.len()
    );
    ensure!(
        out_lse.len() >= num_rows && out_picked_lp.len() >= num_rows,
        "logprobs lse outputs too small: have {}/{}, need {num_rows}",
        out_lse.len(),
        out_picked_lp.len()
    );

    let (logits_ptr, _lg) = logits.data.device_ptr(&ctx.stream);
    let (indices_ptr, _ig) = match row_indices {
        Some(indices) => {
            let (ptr, guard) = indices.device_ptr(&ctx.stream);
            (ptr as *const u32, Some(guard))
        }
        None => (std::ptr::null(), None),
    };
    let (picked_ptr, _pg) = picked.device_ptr(&ctx.stream);
    let (lse_ptr, _og) = out_lse.device_ptr_mut(&ctx.stream);
    let (picked_lp_ptr, _pg2) = out_picked_lp.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::logprobs_lse_bf16_cuda(
            logits_ptr as *const ffi::Half,
            indices_ptr,
            picked_ptr as *const u32,
            num_rows as i32,
            logits.hidden_dim as i32,
            lse_ptr as *mut f32,
            picked_lp_ptr as *mut f32,
            ctx.stream.cu_stream(),
        )
    };
    ensure!(
        result == 0,
        "logprobs lse launch failed with error {result}{}",
        crate::ops::ffi_exception_message(result)
    );
    Ok(())
}

/// Gather `num_rows` logits rows by index into a contiguous
/// [num_rows, hidden_dim] bf16 buffer for the FilteredTopK layout.
pub fn logprobs_gather_rows_bf16_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    row_indices: &CudaSlice<u32>,
    num_rows: usize,
    out: &mut CudaSlice<half::bf16>,
) -> Result<()> {
    ensure!(num_rows > 0, "logprobs gather requires num_rows > 0");
    ensure!(
        row_indices.len() >= num_rows,
        "logprobs gather row_indices too small: have {}, need {num_rows}",
        row_indices.len()
    );
    ensure!(
        out.len() >= num_rows * logits.hidden_dim,
        "logprobs gather output too small: have {}, need {}",
        out.len(),
        num_rows * logits.hidden_dim
    );

    let (logits_ptr, _lg) = logits.data.device_ptr(&ctx.stream);
    let (indices_ptr, _ig) = row_indices.device_ptr(&ctx.stream);
    let (out_ptr, _og) = out.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::logprobs_gather_rows_bf16_cuda(
            logits_ptr as *const ffi::Half,
            indices_ptr as *const u32,
            out_ptr as *mut ffi::Half,
            num_rows as i32,
            logits.hidden_dim as i32,
            ctx.stream.cu_stream(),
        )
    };
    ensure!(
        result == 0,
        "logprobs gather launch failed with error {result}{}",
        crate::ops::ffi_exception_message(result)
    );
    Ok(())
}

/// Deterministic top-k over a contiguous [num_rows, hidden_dim] bf16 logits
/// block (FilteredTopK, smallest-index tie-break).
///
/// Returns `Ok(false)` when the GPU cannot run FilteredTopK (pre-Hopper
/// smem), in which case the caller should fall back to the host path.
pub fn logprobs_topk_bf16_into(
    ctx: &DeviceContext,
    logits: &CudaSlice<half::bf16>,
    num_rows: usize,
    hidden_dim: usize,
    top_k: usize,
    out_values: &mut CudaSlice<half::bf16>,
    out_indices: &mut CudaSlice<i32>,
) -> Result<bool> {
    ensure!(num_rows > 0, "logprobs top-k requires num_rows > 0");
    ensure!(top_k > 0, "logprobs top-k requires top_k > 0");
    ensure!(
        logits.len() >= num_rows * hidden_dim,
        "logprobs top-k input too small: have {}, need {}",
        logits.len(),
        num_rows * hidden_dim
    );
    ensure!(
        out_values.len() >= num_rows * top_k && out_indices.len() >= num_rows * top_k,
        "logprobs top-k outputs too small: have {}/{}, need {}",
        out_values.len(),
        out_indices.len(),
        num_rows * top_k
    );

    let (logits_ptr, _lg) = logits.device_ptr(&ctx.stream);
    let (values_ptr, _vg) = out_values.device_ptr_mut(&ctx.stream);
    let (indices_ptr, _ig) = out_indices.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::logprobs_topk_bf16_cuda(
            logits_ptr as *const ffi::Half,
            num_rows as i32,
            hidden_dim as i32,
            top_k as i32,
            values_ptr as *mut ffi::Half,
            indices_ptr as *mut i32,
            ctx.stream.cu_stream(),
        )
    };
    if result == CUDA_ERROR_NOT_SUPPORTED {
        return Ok(false);
    }
    ensure!(
        result == 0,
        "logprobs top-k launch failed with error {result}{}",
        crate::ops::ffi_exception_message(result)
    );
    Ok(true)
}
