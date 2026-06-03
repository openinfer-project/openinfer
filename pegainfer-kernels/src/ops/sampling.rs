use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

const FLASHINFER_TOPK_ROW_STATES_BYTES: usize = 1024 * 1024;

/// Argmax — returns the index of the maximum element.
///
/// Allocates a temporary output buffer. Used by benchmarks; model code uses
/// `gpu_sample_into` for both greedy and non-greedy paths.
pub fn argmax(ctx: &DeviceContext, x: &DeviceVec) -> Result<u32> {
    let mut out_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out_gpu.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_cuda(
                x_ptr as *const ffi::Half,
                out_ptr as *mut i32,
                x.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    let result = ctx
        .stream
        .clone_dtoh(&out_gpu)
        .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
    ctx.sync()?;

    Ok(result[0] as u32)
}

pub fn argmax_batch_bf16_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    values: &mut CudaSlice<half::bf16>,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    let rows = logits.seq_len;
    if rows == 0 {
        return Err(anyhow!("argmax batch requires at least one row"));
    }
    if values.len() < rows {
        return Err(anyhow!(
            "argmax batch values scratch too small: have {}, need {}",
            values.len(),
            rows
        ));
    }
    if out.len() < rows {
        return Err(anyhow!(
            "argmax batch output too small: have {}, need {}",
            out.len(),
            rows
        ));
    }

    let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (values_ptr, _gv) = values.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::argmax_batch_bf16_cuda(
            logits_ptr as *const ffi::Half,
            values_ptr as *mut ffi::Half,
            out_ptr as *mut i32,
            rows as i32,
            logits.hidden_dim as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

pub fn argmax_batch_bf16_split_partials_len(rows: usize, vocab: usize) -> usize {
    const TILE_ELEMS: usize = 4096;
    rows * vocab.div_ceil(TILE_ELEMS)
}

/// GPU sampling: temperature → softmax → top-k → top-p → multinomial.
/// Allocates a temporary output buffer — use `gpu_sample_into` for the decode loop.
pub fn gpu_sample(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    random_val: f32,
) -> Result<u32> {
    let mut valid_scratch: CudaSlice<u8> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;
    let mut out_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    gpu_sample_core(
        ctx,
        logits,
        probs_scratch,
        top1_value_scratch,
        row_states_scratch,
        &mut valid_scratch,
        &mut out_gpu,
        temperature,
        top_k,
        top_p,
        random_val,
    )
}

/// GPU sampling into pre-allocated buffers — zero allocation, suitable for decode loop.
pub fn gpu_sample_into(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    valid_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    random_val: f32,
) -> Result<u32> {
    gpu_sample_core(
        ctx,
        logits,
        probs_scratch,
        top1_value_scratch,
        row_states_scratch,
        valid_scratch,
        out,
        temperature,
        top_k,
        top_p,
        random_val,
    )
}

fn gpu_sample_core(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    _top1_value_scratch: &mut CudaSlice<half::bf16>,
    _row_states_scratch: &mut CudaSlice<u8>,
    valid_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    random_val: f32,
) -> Result<u32> {
    if (temperature <= 0.0 || top_k == 1) && top_p >= 1.0 {
        let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_cuda(
                l_ptr as *const ffi::Half,
                o_ptr as *mut i32,
                logits.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    } else {
        let inv_temperature = 1.0 / temperature;

        let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (p_ptr, _gp) = probs_scratch.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gv) = valid_scratch.device_ptr_mut(&ctx.stream);
        let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::gpu_sample_flashinfer_cuda(
                l_ptr as *const ffi::Half,
                p_ptr as *mut f32,
                v_ptr as *mut u8,
                o_ptr as *mut i32,
                logits.len as i32,
                inv_temperature,
                top_k,
                top_p,
                u64::from(random_val.to_bits()),
                ctx.stream.cu_stream(),
            );
        }
    }
    let result = ctx
        .stream
        .clone_dtoh(out)
        .map_err(|e| anyhow!("D2H sample read failed: {}", e))?;
    ctx.sync()?;

    Ok(result[0] as u32)
}

pub fn flashinfer_topk_row_states_bytes() -> usize {
    FLASHINFER_TOPK_ROW_STATES_BYTES
}

pub fn flashinfer_top1_batch_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    top1_values: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    let rows = logits.seq_len;
    if top1_values.len() < rows {
        return Err(anyhow!(
            "top1 values scratch too small: have {}, need {}",
            top1_values.len(),
            rows
        ));
    }
    if out.len() < rows {
        return Err(anyhow!(
            "top1 output too small: have {}, need {}",
            out.len(),
            rows
        ));
    }
    if row_states_scratch.len() < FLASHINFER_TOPK_ROW_STATES_BYTES {
        return Err(anyhow!(
            "top1 row states scratch too small: have {}, need {}",
            row_states_scratch.len(),
            FLASHINFER_TOPK_ROW_STATES_BYTES
        ));
    }

    let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = top1_values.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = row_states_scratch.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::flashinfer_top1_batch_cuda(
            l_ptr as *const ffi::Half,
            v_ptr as *mut ffi::Half,
            r_ptr as *mut u8,
            o_ptr as *mut i32,
            rows as i32,
            logits.hidden_dim as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}
