use anyhow::{Result, bail};
use cudarc::driver::{DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

/// GEMM on a row sub-range of a weight matrix: Y = W[row_offset..row_offset+M, :] @ X
pub fn gemm_rows_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    gemm_rows_into_checked(ctx, weight, row_offset, num_rows, x, out)
        .expect("GEMM row-range launch failed");
}

/// Checked row-range GEMM. New hot paths should use this form so cuBLAS launch
/// failures surface at the operator boundary instead of at a later collective.
pub fn gemm_rows_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert!(row_offset + num_rows <= weight.rows);
    assert_eq!(weight.cols, x.hidden_dim);
    assert_eq!(out.hidden_dim, num_rows);
    assert_eq!(out.seq_len, x.seq_len);

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let w_sub = w_ptr + (row_offset * weight.cols * std::mem::size_of::<half::bf16>()) as u64;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        w_sub as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        num_rows,
        x.seq_len,
        weight.cols,
        x.seq_len == 1,
        ctx,
    )
}

/// Matrix-vector multiplication: y = A @ x (via cuBLAS GEMM with N=1)
/// A: (M, K) row-major, x: (K,), y: (M,)
pub fn gemv(ctx: &DeviceContext, a: &DeviceMatrix, x: &DeviceVec, y: &mut DeviceVec) -> Result<()> {
    assert_eq!(a.cols, x.len, "A cols {} != x len {}", a.cols, x.len);
    assert_eq!(a.rows, y.len, "A rows {} != y len {}", a.rows, y.len);

    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        a_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        a.rows,
        1,
        a.cols,
        true,
        ctx,
    )
}
/// Linear layer: y = weight @ x
pub fn linear(ctx: &DeviceContext, x: &DeviceVec, weight: &DeviceMatrix) -> Result<DeviceVec> {
    let mut y = DeviceVec::zeros(ctx, weight.rows)?;
    gemv(ctx, weight, x, &mut y)?;
    Ok(y)
}

/// GEMM: Y = weight @ X (batched linear projection)
/// weight: [out_dim, in_dim] row-major, X: HiddenStates [in_dim, seq_len], Y: HiddenStates [out_dim, seq_len]
pub fn gemm(ctx: &DeviceContext, weight: &DeviceMatrix, x: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, weight.rows, x.seq_len)?;
    gemm_into_checked(ctx, weight, x, &mut out)?;
    Ok(out)
}

/// GEMM into pre-allocated output buffer (zero allocation).
/// For seq_len=1, uses the graph-safe cuBLAS handle (no workspace) for lower
/// latency while preserving numerical parity with the prefill path.
pub fn gemm_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    gemm_into_checked(ctx, weight, x, out).expect("GEMM launch failed");
}

/// Checked GEMM using the default policy: graph-safe handle for single-token
/// decode, workspace-backed handle for prefill/batched projections.
pub fn gemm_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    gemm_into_with_policy(ctx, weight, x, out, x.seq_len == 1)
}

/// Checked GEMM that always uses the workspace-free cuBLAS handle. Kimi decode
/// uses this for active-batch sizes 1..=4 so graph-readiness is not tied to a
/// bs1-only condition.
pub fn gemm_graphsafe_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    gemm_into_with_policy(ctx, weight, x, out, true)
}

fn gemm_into_with_policy(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
    graphsafe: bool,
) -> Result<()> {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert_eq!(
        out.seq_len, x.seq_len,
        "out seq_len {} != x seq_len {}",
        out.seq_len, x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        w_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        weight.rows,
        x.seq_len,
        weight.cols,
        graphsafe,
        ctx,
    )
}

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
            if status >= 100000 {
                bail!(
                    "cuBLAS GEMM failed: cublas_status={}, m={}, n={}, k={}, graphsafe={}",
                    status - 100000,
                    m,
                    n,
                    k,
                    graphsafe
                );
            }
            bail!(
                "CUDA GEMM launch failed: cuda_status={}, m={}, n={}, k={}, graphsafe={}",
                status,
                m,
                n,
                k,
                graphsafe
            );
        }
    }
    Ok(())
}
