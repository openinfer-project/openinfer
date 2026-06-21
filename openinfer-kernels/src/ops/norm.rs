use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// RMSNorm into pre-allocated output buffer
pub fn rms_norm_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(x.len, out.len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// RMSNorm (allocating)
pub fn rms_norm(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
) -> Result<DeviceVec> {
    let mut out = DeviceVec::zeros(ctx, x.len)?;
    rms_norm_into(ctx, x, weight, eps, &mut out)?;
    Ok(out)
}

/// Fused add + RMSNorm: hidden += residual; out = rms_norm(hidden, weight)
/// Saves one global read of hidden compared to separate add + rms_norm.
pub fn fused_add_rms_norm_into(
    ctx: &DeviceContext,
    hidden: &mut DeviceVec,
    residual: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(hidden.len, residual.len);
    assert_eq!(hidden.len, out.len);
    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::fused_add_rms_norm_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// Batched fused add + RMSNorm for HiddenStates.
/// hidden[i] += residual[i]; out[i] = rms_norm(hidden[i], weight) for each batch element.
pub fn fused_add_rms_norm_batch_into(
    ctx: &DeviceContext,
    hidden: &mut HiddenStates,
    residual: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) {
    assert_eq!(hidden.hidden_dim, residual.hidden_dim);
    assert_eq!(hidden.hidden_dim, out.hidden_dim);
    assert_eq!(hidden.seq_len, residual.seq_len);
    assert_eq!(hidden.seq_len, out.seq_len);
    assert_eq!(weight.len, hidden.hidden_dim);
    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::fused_add_rms_norm_batched_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.hidden_dim as i32,
            hidden.seq_len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
}

/// Batched exact-preserving fused add + RMSNorm.
/// hidden[i] = bf16(hidden[i] + residual[i]); out[i] = rms_norm(hidden[i], weight).
pub fn fused_add_rms_norm_round_batch_into(
    ctx: &DeviceContext,
    hidden: &mut HiddenStates,
    residual: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(hidden.hidden_dim, residual.hidden_dim);
    assert_eq!(hidden.hidden_dim, out.hidden_dim);
    assert_eq!(hidden.seq_len, residual.seq_len);
    assert_eq!(hidden.seq_len, out.seq_len);
    assert_eq!(weight.len, hidden.hidden_dim);
    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::fused_add_rms_norm_round_batched_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.hidden_dim as i32,
            hidden.seq_len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Batched RMSNorm into pre-allocated output buffer (zero allocation).
pub fn rms_norm_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) {
    assert_eq!(weight.len, x.hidden_dim);
    assert_eq!(out.hidden_dim, x.hidden_dim);
    assert_eq!(out.seq_len, x.seq_len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.hidden_dim as i32,
            x.seq_len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
}

/// Batched (1+weight) RMSNorm over HiddenStates — one kernel launch for all tokens.
pub fn rms_norm_batch_offset_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(weight.len, x.hidden_dim);
    assert_eq!(out.hidden_dim, x.hidden_dim);
    assert_eq!(out.seq_len, x.seq_len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.hidden_dim as i32,
            x.seq_len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// (1+weight) RMSNorm into pre-allocated output buffer (Gemma/Qwen3.5 style)
pub fn rms_norm_offset_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(x.len, out.len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// Batched per-head RMSNorm with F32 weight + SiLU gate multiplication.
/// HiddenStates are flattened as (seq_len * num_heads) contiguous head slices.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gated_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &CudaSlice<f32>,
    gate: &HiddenStates,
    out: &mut HiddenStates,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let total_heads = x.seq_len * num_heads;
    assert_eq!(x.hidden_dim, num_heads * head_dim);
    assert_eq!(gate.hidden_dim, x.hidden_dim);
    assert_eq!(gate.seq_len, x.seq_len);
    assert_eq!(out.hidden_dim, x.hidden_dim);
    assert_eq!(out.seq_len, x.seq_len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_gated_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const f32,
            g_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            total_heads as i32,
            head_dim as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
}
