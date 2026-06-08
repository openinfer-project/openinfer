use anyhow::{Result, anyhow};
use cudarc::driver::CudaSlice;

use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

pub use pegainfer_kernels::ops::{
    argmax, argmax_batch_bf16_indexed_into, argmax_batch_bf16_into,
    flashinfer_topk_row_states_bytes,
};

/// GPU sampling: temperature -> softmax -> top-k -> top-p -> multinomial.
///
/// Root owns request sampling policy; the kernels crate only sees primitive
/// launch parameters.
pub fn gpu_sample(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    params: &SamplingParams,
    random_val: f32,
) -> Result<u32> {
    pegainfer_kernels::ops::gpu_sample(
        ctx,
        logits,
        probs_scratch,
        top1_value_scratch,
        row_states_scratch,
        params.temperature,
        params.top_k,
        params.top_p,
        random_val,
    )
}

/// GPU sampling into pre-allocated buffers.
pub fn gpu_sample_into(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    valid_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
    params: &SamplingParams,
    random_val: f32,
) -> Result<u32> {
    pegainfer_kernels::ops::gpu_sample_into(
        ctx,
        logits,
        probs_scratch,
        top1_value_scratch,
        row_states_scratch,
        valid_scratch,
        out,
        params.temperature,
        params.top_k,
        params.top_p,
        random_val,
    )
}

/// Pick the next token for each row in a decode batch.
///
/// Greedy rows are selected together with indexed batched argmax. Non-greedy
/// rows still use the existing per-row sampler because each row may have its
/// own random value and sampling parameters.
#[allow(clippy::too_many_arguments)]
pub fn select_batch_tokens_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    params: &[&SamplingParams],
    random_vals: &[f32],
    row_indices_scratch: &mut CudaSlice<i32>,
    probs_scratch: &mut CudaSlice<f32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    valid_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
) -> Result<Vec<u32>> {
    let batch_size = params.len();
    let mut tokens = vec![0; batch_size];
    let greedy_rows = params
        .iter()
        .enumerate()
        .filter_map(|(i, params_i)| params_i.is_greedy().then_some(i as i32))
        .collect::<Vec<_>>();

    if !greedy_rows.is_empty() {
        // Batch sampling for greedy rows.
        if row_indices_scratch.len() < greedy_rows.len() {
            return Err(anyhow!(
                "row_indices_scratch too small: have {}, need {}",
                row_indices_scratch.len(),
                greedy_rows.len()
            ));
        }

        ctx.stream
            .memcpy_htod(&greedy_rows, row_indices_scratch)
            .map_err(|e| anyhow!("H2D indexed argmax rows failed: {}", e))?;

        argmax_batch_bf16_indexed_into(
            ctx,
            logits,
            row_indices_scratch,
            greedy_rows.len(),
            top1_value_scratch,
            out,
        )?;

        let out_host = ctx
            .stream
            .clone_dtoh(out)
            .map_err(|e| anyhow!("D2H indexed batch argmax read failed: {}", e))?;
        ctx.sync()?;

        for (i, row) in greedy_rows.iter().enumerate() {
            tokens[*row as usize] = out_host[i] as u32;
        }
    }

    // Per-row sampling for non-greedy rows.
    for (i, params_i) in params.iter().enumerate() {
        if params_i.is_greedy() {
            continue;
        }
        let logits_i = pegainfer_kernels::ops::extract_vec(ctx, logits, i)?;
        tokens[i] = gpu_sample_into(
            ctx,
            &logits_i,
            probs_scratch,
            top1_value_scratch,
            row_states_scratch,
            valid_scratch,
            out,
            params_i,
            random_vals[i],
        )?;
    }

    Ok(tokens)
}
