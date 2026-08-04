use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

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

/// Slice-level batched RMSNorm over `rows` rows of `dim`: same
/// `flashinfer::norm::RMSNorm` template as [`rms_norm_into`] with
/// batch_size=rows (one CTA per row), so each row is bit-identical to the
/// single-row launch. For callers whose buffers live in a persistent decode
/// arena rather than owned `HiddenStates`.
pub fn rms_norm_rows_into(
    ctx: &DeviceContext,
    x: &CudaSlice<bf16>,
    weight: &DeviceVec,
    eps: f32,
    dim: usize,
    rows: usize,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(rows > 0 && dim > 0, "rms_norm_rows needs positive rows/dim");
    ensure!(
        x.len() >= rows * dim && out.len() >= rows * dim,
        "rms_norm_rows buffers too small for {rows}x{dim}: x {}, out {}",
        x.len(),
        out.len()
    );
    ensure!(
        weight.len == dim,
        "rms_norm_rows weight len {} != dim {dim}",
        weight.len
    );
    let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            dim as i32,
            rows as i32,
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

/// LayerNorm (with bias) of a single bf16 vector `[dim]` — GLM5.2 DSA indexer
/// k_norm (eps=1e-6, has bias). Wraps `flashinfer::norm::LayerNorm` (same
/// vendored template that `rms_norm_cuda` wraps). Unlike RMSNorm, LayerNorm
/// subtracts the mean and applies a per-element bias. gamma/beta are f32
/// (FlashInfer's LayerNorm template requires f32 weight types).
pub fn layer_norm_into(
    ctx: &DeviceContext,
    x: &CudaSlice<bf16>,
    gamma: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    eps: f32,
    dim: usize,
    rows: usize,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(rows > 0 && dim > 0, "layer_norm needs positive rows/dim");
    ensure!(
        x.len() >= rows * dim && out.len() >= rows * dim,
        "layer_norm x/out too small for {rows}x{dim}: x {}, out {}",
        x.len(),
        out.len()
    );
    ensure!(gamma.len() >= dim, "layer_norm gamma too small");
    ensure!(beta.len() >= dim, "layer_norm beta too small");
    let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = gamma.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = beta.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::layer_norm_cuda(
            x_ptr as *const ffi::Half,
            g_ptr as *const f32,
            b_ptr as *const f32,
            o_ptr as *mut ffi::Half,
            dim as i32,
            rows as i32,
            eps,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow::anyhow!("GLM5.2 LayerNorm launch failed: {err}"))
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

/// Slice-level twin of [`fused_add_rms_norm_round_batch_into`] — same kernel
/// — for callers whose buffers live in a persistent decode arena rather than
/// owned `HiddenStates`: `hidden += residual` (sum rounded to bf16), then
/// `out = rms_norm(hidden, weight)` over `seq_len` rows of `hidden_dim`.
pub fn fused_add_rms_norm_round_into(
    ctx: &DeviceContext,
    hidden: &mut CudaSlice<bf16>,
    residual: &CudaSlice<bf16>,
    weight: &DeviceVec,
    eps: f32,
    hidden_dim: usize,
    seq_len: usize,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let n = hidden_dim * seq_len;
    ensure!(
        hidden.len() >= n && residual.len() >= n && out.len() >= n,
        "fused_add_rms_norm_round_into buffers too small for {seq_len}x{hidden_dim}: hidden {}, residual {}, out {}",
        hidden.len(),
        residual.len(),
        out.len()
    );
    ensure!(
        weight.len == hidden_dim,
        "fused_add_rms_norm_round_into weight len {} != hidden_dim {hidden_dim}",
        weight.len
    );
    let (h_ptr, _gh) = hidden.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::fused_add_rms_norm_round_batched_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden_dim as i32,
            seq_len as i32,
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
