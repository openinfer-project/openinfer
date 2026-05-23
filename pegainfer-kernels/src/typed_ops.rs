//! Compile-time dimension-safe GPU operations on `GpuTensor<DIM>` / `GpuWeight<OUT, IN>`.
//!
//! Each function's signature encodes the shape contract — passing a tensor with the
//! wrong dimension is a compile error, not a runtime panic.

use anyhow::Result;
use cudarc::driver::{DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, GpuTensor, GpuWeight, NormWeight};

// ── GEMM ─────────────────────────────────────────────────────────────

/// `Y = W @ X` — compile-time shape: `W:[OUT,IN]`, `X:[IN,bs]`, `Y:[OUT,bs]`.
pub fn gemm_into<const OUT: usize, const IN: usize>(
    ctx: &DeviceContext,
    w: &GpuWeight<OUT, IN>,
    x: &GpuTensor<IN>,
    y: &mut GpuTensor<OUT>,
) -> Result<()> {
    debug_assert_eq!(y.seq_len, x.seq_len);
    let (w_ptr, _gw) = w.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);
    launch_gemm(
        w_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        OUT,
        x.seq_len,
        IN,
        x.seq_len == 1,
        ctx,
    )
}

/// Graph-safe variant: always uses workspace-free cuBLAS handle.
pub fn gemm_graphsafe_into<const OUT: usize, const IN: usize>(
    ctx: &DeviceContext,
    w: &GpuWeight<OUT, IN>,
    x: &GpuTensor<IN>,
    y: &mut GpuTensor<OUT>,
) -> Result<()> {
    debug_assert_eq!(y.seq_len, x.seq_len);
    let (w_ptr, _gw) = w.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);
    launch_gemm(
        w_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        OUT,
        x.seq_len,
        IN,
        true,
        ctx,
    )
}

// ── RMSNorm ──────────────────────────────────────────────────────────

/// Batched RMSNorm: `out[i] = rms_norm(x[i], w)`. Same DIM enforced at compile time.
pub fn rms_norm_into<const DIM: usize>(
    ctx: &DeviceContext,
    x: &GpuTensor<DIM>,
    w: &NormWeight<DIM>,
    eps: f32,
    out: &mut GpuTensor<DIM>,
) {
    debug_assert_eq!(out.seq_len, x.seq_len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = w.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            DIM as i32,
            x.seq_len as i32,
            eps,
            ctx.stream.cu_stream(),
        );
    }
}

/// Fused `hidden += residual; out = rms_norm(hidden, w)`. All three must be same DIM.
pub fn fused_add_rms_norm_into<const DIM: usize>(
    ctx: &DeviceContext,
    hidden: &mut GpuTensor<DIM>,
    residual: &GpuTensor<DIM>,
    w: &NormWeight<DIM>,
    eps: f32,
    out: &mut GpuTensor<DIM>,
) {
    debug_assert_eq!(hidden.seq_len, residual.seq_len);
    debug_assert_eq!(hidden.seq_len, out.seq_len);
    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = w.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::fused_add_rms_norm_batched_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            DIM as i32,
            hidden.seq_len as i32,
            eps,
            ctx.stream.cu_stream(),
        );
    }
}

// ── Elementwise ──────────────────────────────────────────────────────

/// `out = a + b` — same DIM enforced at compile time.
pub fn add_into<const DIM: usize>(
    ctx: &DeviceContext,
    a: &GpuTensor<DIM>,
    b: &GpuTensor<DIM>,
    out: &mut GpuTensor<DIM>,
) -> Result<()> {
    debug_assert_eq!(a.seq_len, b.seq_len);
    debug_assert_eq!(a.seq_len, out.seq_len);
    let n = DIM * a.seq_len;
    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::add_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

/// Fused SiLU-mul: `gate_up:[2*INTER, bs]` → `out:[INTER, bs]`.
pub fn silu_mul_fused_into<const INTER: usize>(
    ctx: &DeviceContext,
    gate_up: &GpuTensor<{ 2 * INTER }>,
    out: &mut GpuTensor<INTER>,
) where
    [(); 2 * INTER]:,
{
    debug_assert_eq!(gate_up.seq_len, out.seq_len);
    let (gu_ptr, _g0) = gate_up.data.device_ptr(&ctx.stream);
    let (o_ptr, _g1) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::silu_mul_fused_cuda(
            gu_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            INTER as i32,
            gate_up.seq_len as i32,
            ctx.stream.cu_stream(),
        );
    }
}

// ── Internal ─────────────────────────────────────────────────────────

fn launch_gemm(
    w_ptr: *const ffi::Half,
    x_ptr: *const ffi::Half,
    y_ptr: *mut ffi::Half,
    m: usize,
    n: usize,
    k: usize,
    graphsafe: bool,
    ctx: &DeviceContext,
) -> Result<()> {
    unsafe {
        let status = if graphsafe {
            ffi::gemm_graphsafe_cuda(
                w_ptr,
                x_ptr,
                y_ptr,
                m as i32,
                n as i32,
                k as i32,
                ctx.stream.cu_stream(),
            )
        } else {
            ffi::gemm_cuda(
                w_ptr,
                x_ptr,
                y_ptr,
                m as i32,
                n as i32,
                k as i32,
                ctx.stream.cu_stream(),
            )
        };
        if status != 0 {
            if status >= 100_000 {
                anyhow::bail!(
                    "cuBLAS GEMM failed: cublas_status={}, m={m}, n={n}, k={k}",
                    status - 100_000
                );
            }
            anyhow::bail!("CUDA GEMM launch failed: cuda_status={status}, m={m}, n={n}, k={k}");
        }
    }
    Ok(())
}
