//! Compile-time dimension-safe GPU operations on `GpuTensor<DIM>`.
//!
//! Weight parameters are typed (`GpuWeight`, `NormWeight`) so tensor and weight
//! dimensions are checked through const generics instead of runtime matrix metadata.

use anyhow::Result;
use cudarc::driver::{DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, GpuTensor, GpuWeight, HiddenStates, NormWeight};

// ── GEMM ─────────────────────────────────────────────────────────────

/// `Y = W @ X` — compile-time shape: `X:[IN,bs]`, `Y:[OUT,bs]`.
pub fn gemm_into<const OUT: usize, const IN: usize>(
    ctx: &DeviceContext,
    w: &GpuWeight<OUT, IN>,
    x: &GpuTensor<IN>,
    y: &mut GpuTensor<OUT>,
) -> Result<()> {
    anyhow::ensure!(
        y.seq_len == x.seq_len,
        "typed GEMM seq_len mismatch: input={}, output={}",
        x.seq_len,
        y.seq_len
    );
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
    anyhow::ensure!(
        y.seq_len == x.seq_len,
        "typed graphsafe GEMM seq_len mismatch: input={}, output={}",
        x.seq_len,
        y.seq_len
    );
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
) -> Result<()> {
    anyhow::ensure!(
        out.seq_len == x.seq_len,
        "typed RMSNorm seq_len mismatch: input={}, output={}",
        x.seq_len,
        out.seq_len
    );
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
    Ok(())
}

/// Fused `hidden += residual; out = rms_norm(hidden, w)`. All three must be same DIM.
pub fn fused_add_rms_norm_into<const DIM: usize>(
    ctx: &DeviceContext,
    hidden: &mut GpuTensor<DIM>,
    residual: &GpuTensor<DIM>,
    w: &NormWeight<DIM>,
    eps: f32,
    out: &mut GpuTensor<DIM>,
) -> Result<()> {
    anyhow::ensure!(
        hidden.seq_len == residual.seq_len && hidden.seq_len == out.seq_len,
        "typed fused_add_rms_norm seq_len mismatch: hidden={}, residual={}, output={}",
        hidden.seq_len,
        residual.seq_len,
        out.seq_len
    );
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
    Ok(())
}

// ── Elementwise ──────────────────────────────────────────────────────

/// `out = a + b` — same DIM enforced at compile time.
pub fn add_into<const DIM: usize>(
    ctx: &DeviceContext,
    a: &GpuTensor<DIM>,
    b: &GpuTensor<DIM>,
    out: &mut GpuTensor<DIM>,
) -> Result<()> {
    anyhow::ensure!(
        a.seq_len == b.seq_len && a.seq_len == out.seq_len,
        "typed add seq_len mismatch: a={}, b={}, output={}",
        a.seq_len,
        b.seq_len,
        out.seq_len
    );
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
) -> Result<()>
where
    [(); 2 * INTER]:,
{
    anyhow::ensure!(
        gate_up.seq_len == out.seq_len,
        "typed silu_mul seq_len mismatch: gate_up={}, output={}",
        gate_up.seq_len,
        out.seq_len
    );
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
    Ok(())
}

// ── Embedding ────────────────────────────────────────────────────────

/// Batched vocab-shard embedding lookup: `out[t] = embed[token_ids[t] - vocab_start]`.
pub fn embedding_vocab_shard_into<const DIM: usize>(
    ctx: &DeviceContext,
    embed: &GpuTensor<DIM>,
    token_ids: &cudarc::driver::CudaSlice<u32>,
    out: &mut GpuTensor<DIM>,
    vocab_start: u32,
) -> Result<()> {
    anyhow::ensure!(
        token_ids.len() >= out.seq_len,
        "embedding token_ids too small: got {}, need {}",
        token_ids.len(),
        out.seq_len
    );
    let vocab_rows = u32::try_from(embed.seq_len)
        .map_err(|_| anyhow::anyhow!("embedding vocab rows exceed u32: {}", embed.seq_len))?;
    let (embed_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (token_ptr, _gt) = token_ids.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::embedding_batched_vocab_shard_cuda(
            embed_ptr as *const ffi::Half,
            token_ptr as *const u32,
            out_ptr as *mut ffi::Half,
            DIM as i32,
            out.seq_len as i32,
            vocab_start,
            vocab_rows,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

/// Runtime-output GEMM for row-major weights with static input width:
/// `Y = W @ X`, `W:[runtime_out, IN]`, `X:[IN,bs]`, `Y:[runtime_out,bs]`.
pub fn gemm_runtime_out_into<const IN: usize>(
    ctx: &DeviceContext,
    w: &GpuTensor<IN>,
    x: &GpuTensor<IN>,
    y: &mut HiddenStates,
) -> Result<()> {
    gemm_runtime_out_impl(ctx, w, x, y, false)
}

/// Graph-safe runtime-output GEMM variant for decode capture.
pub fn gemm_runtime_out_graphsafe_into<const IN: usize>(
    ctx: &DeviceContext,
    w: &GpuTensor<IN>,
    x: &GpuTensor<IN>,
    y: &mut HiddenStates,
) -> Result<()> {
    gemm_runtime_out_impl(ctx, w, x, y, true)
}

// ── bf16 ↔ f32 conversion ───────────────────────────────────────────

/// Convert bf16 tensor to f32 buffer (for deterministic all-reduce).
pub fn bf16_to_f32_into<const DIM: usize>(
    ctx: &DeviceContext,
    x: &GpuTensor<DIM>,
    out: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<()> {
    let n = DIM * x.seq_len;
    anyhow::ensure!(
        out.len() >= n,
        "bf16_to_f32 scratch too small: have {}, need {n}",
        out.len()
    );
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::bf16_to_f32_cuda(
            x_ptr as *const ffi::Half,
            o_ptr as *mut f32,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

/// Convert f32 buffer back to bf16 tensor.
pub fn f32_to_bf16_into<const DIM: usize>(
    ctx: &DeviceContext,
    x: &cudarc::driver::CudaSlice<f32>,
    out: &mut GpuTensor<DIM>,
) -> Result<()> {
    let n = DIM * out.seq_len;
    anyhow::ensure!(
        x.len() >= n,
        "f32_to_bf16 input too small: have {}, need {n}",
        x.len()
    );
    let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::f32_to_bf16_cuda(
            x_ptr as *const f32,
            o_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
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

fn gemm_runtime_out_impl<const IN: usize>(
    ctx: &DeviceContext,
    w: &GpuTensor<IN>,
    x: &GpuTensor<IN>,
    y: &mut HiddenStates,
    graphsafe: bool,
) -> Result<()> {
    anyhow::ensure!(
        y.hidden_dim == w.seq_len,
        "runtime-out GEMM output hidden mismatch: weight rows={}, output hidden={}",
        w.seq_len,
        y.hidden_dim
    );
    anyhow::ensure!(
        y.seq_len == x.seq_len,
        "runtime-out GEMM seq_len mismatch: input={}, output={}",
        x.seq_len,
        y.seq_len
    );
    let (w_ptr, _gw) = w.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);
    launch_gemm(
        w_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        w.seq_len,
        x.seq_len,
        IN,
        graphsafe || x.seq_len == 1,
        ctx,
    )
}
