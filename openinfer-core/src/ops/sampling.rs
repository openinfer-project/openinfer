use anyhow::{Result, anyhow};
use cudarc::driver::CudaSlice;

use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, HiddenStates};

pub use openinfer_kernels::ops::{
    BatchSamplingRow, BatchSamplingScratch, argmax, argmax_batch_bf16_into,
    argmax_batch_bf16_split_indexed_into, argmax_batch_bf16_split_partials_len,
    flashinfer_top1_row_states_bytes,
};

/// Pick the next token for each row in a decode batch.
///
/// Greedy rows are selected together with indexed batched argmax. Non-greedy
/// rows are compacted and sent through one batched FlashInfer sampling call.
#[allow(clippy::too_many_arguments)]
pub fn select_batch_tokens_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    params: &[&SamplingParams],
    sample_seed: u64,
    row_indices_scratch: &mut CudaSlice<i32>,
    argmax_partial_values_scratch: &mut CudaSlice<f32>,
    argmax_partial_indices_scratch: &mut CudaSlice<i32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    out: &mut CudaSlice<i32>,
    batch_sampling_scratch: &mut BatchSamplingScratch,
) -> Result<Vec<u32>> {
    let batch_size = params.len();
    let vocab_size = logits.hidden_dim;
    let mut tokens = vec![0; batch_size];
    let is_greedy =
        |params_i: &&SamplingParams| sampling_params_effectively_greedy(params_i, vocab_size);
    let greedy_rows = params
        .iter()
        .enumerate()
        .filter_map(|(i, params_i)| is_greedy(params_i).then_some(i as i32))
        .collect::<Vec<_>>();

    if !greedy_rows.is_empty() {
        // Batched argmax for greedy rows.
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

        argmax_batch_bf16_split_indexed_into(
            ctx,
            logits,
            row_indices_scratch,
            greedy_rows.len(),
            argmax_partial_values_scratch,
            argmax_partial_indices_scratch,
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

    let sampling_rows = params
        .iter()
        .enumerate()
        .filter(|(_, params_i)| !is_greedy(params_i))
        .map(|(i, params_i)| BatchSamplingRow {
            row: i,
            temperature: params_i.temperature,
            top_k: params_i.top_k,
            top_p: params_i.top_p,
        })
        .collect::<Vec<_>>();
    if !sampling_rows.is_empty() {
        let sampled = openinfer_kernels::ops::gpu_sample_batch_into(
            ctx,
            logits.as_ref(),
            &sampling_rows,
            sample_seed,
            batch_sampling_scratch,
        )?;
        for (row, token) in sampling_rows.iter().zip(sampled) {
            tokens[row.row] = token;
        }
    }

    Ok(tokens)
}

/// Whether a request can use argmax without changing sampling semantics.
///
/// Besides explicit greedy params, a top-p threshold at or below `1 / vocab`
/// leaves only the argmax token in the nucleus for any normalized distribution.
pub fn sampling_params_effectively_greedy(params: &SamplingParams, vocab_size: usize) -> bool {
    params.is_greedy()
        || (vocab_size > 0
            && params.top_p.is_finite()
            && params.top_p > 0.0
            && params.top_p <= 1.0 / vocab_size as f32)
}
