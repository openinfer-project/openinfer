use anyhow::{Result, anyhow};
use cudarc::driver::CudaSlice;

use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

pub use openinfer_kernels::ops::{
    argmax, argmax_batch_bf16_into, argmax_batch_bf16_split_indexed_into,
    argmax_batch_bf16_split_into, argmax_batch_bf16_split_partials_len,
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
    openinfer_kernels::ops::gpu_sample(
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
    openinfer_kernels::ops::gpu_sample_into(
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

/// Pick the next token for each real row in a decode batch.
///
/// `logits` may contain CUDA Graph padding rows after the real batch. Greedy
/// rows are selected together with batched argmax. Non-greedy rows still use
/// the existing per-row sampler because each row may have its own random value
/// and sampling parameters.
#[allow(clippy::too_many_arguments)]
pub fn select_batch_tokens_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    params: &[&SamplingParams],
    random_vals: &[f32],
    row_indices_scratch: &mut CudaSlice<i32>,
    argmax_partial_values_scratch: &mut CudaSlice<f32>,
    argmax_partial_indices_scratch: &mut CudaSlice<i32>,
    probs_scratch: &mut CudaSlice<f32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    valid_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
) -> Result<Vec<u32>> {
    let batch_size = params.len();
    if batch_size > logits.seq_len {
        return Err(anyhow!(
            "sampling params len {} exceeds logits seq_len {}",
            batch_size,
            logits.seq_len
        ));
    }
    if random_vals.len() < batch_size {
        return Err(anyhow!(
            "sampling random_vals len {} is smaller than params len {}",
            random_vals.len(),
            batch_size
        ));
    }
    let mut tokens = vec![0; batch_size];
    if batch_size == 0 {
        return Ok(tokens);
    }
    if params.iter().all(|params_i| params_i.is_greedy()) {
        argmax_batch_bf16_split_into(
            ctx,
            logits,
            argmax_partial_values_scratch,
            argmax_partial_indices_scratch,
            top1_value_scratch,
            out,
        )?;
        let out_host = ctx
            .stream
            .clone_dtoh(out)
            .map_err(|e| anyhow!("D2H batch argmax read failed: {}", e))?;
        ctx.sync()?;
        for (token, sampled) in tokens.iter_mut().zip(out_host) {
            *token = sampled as u32;
        }
        return Ok(tokens);
    }

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

    // Per-row sampling for non-greedy rows.
    for (i, params_i) in params.iter().enumerate() {
        if params_i.is_greedy() {
            continue;
        }
        let logits_i = openinfer_kernels::ops::extract_vec(ctx, logits, i)?;
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

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    fn logits_with_argmax_rows(
        ctx: &DeviceContext,
        hidden_dim: usize,
        max_tokens: &[usize],
        tied_tokens: &[(usize, usize)],
    ) -> HiddenStates {
        let seq_len = max_tokens.len();
        let mut host = vec![bf16::from_f32(-1.0); seq_len * hidden_dim];
        for (row, &token) in max_tokens.iter().enumerate() {
            host[row * hidden_dim + token] = bf16::from_f32(10.0 + row as f32);
        }
        for &(row, token) in tied_tokens {
            host[row * hidden_dim + token] = host[row * hidden_dim + max_tokens[row]];
        }
        HiddenStates {
            data: ctx.stream.clone_htod(&host).expect("copy logits"),
            hidden_dim,
            seq_len,
        }
    }

    struct SelectScratch {
        row_indices: CudaSlice<i32>,
        argmax_values: CudaSlice<f32>,
        argmax_indices: CudaSlice<i32>,
        probs: CudaSlice<f32>,
        top1_values: CudaSlice<bf16>,
        row_states: CudaSlice<u8>,
        valid: CudaSlice<u8>,
        out: CudaSlice<i32>,
    }

    impl SelectScratch {
        fn new(ctx: &DeviceContext, rows: usize, vocab: usize) -> Self {
            let partials = argmax_batch_bf16_split_partials_len(rows, vocab);
            Self {
                row_indices: ctx.stream.alloc_zeros(rows).expect("row indices"),
                argmax_values: ctx.stream.alloc_zeros(partials).expect("argmax values"),
                argmax_indices: ctx.stream.alloc_zeros(partials).expect("argmax indices"),
                probs: ctx.stream.alloc_zeros(vocab).expect("probs"),
                top1_values: ctx.stream.alloc_zeros(rows).expect("top1 values"),
                row_states: ctx
                    .stream
                    .alloc_zeros(flashinfer_topk_row_states_bytes())
                    .expect("row states"),
                valid: ctx.stream.alloc_zeros(1).expect("valid"),
                out: ctx.stream.alloc_zeros(rows).expect("out"),
            }
        }
    }

    #[test]
    fn all_greedy_contiguous_argmax_matches_indexed_greedy_rows() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let logits = logits_with_argmax_rows(&ctx, 8193, &[17, 5001, 8192, 4095], &[(3, 4096)]);
        let greedy = SamplingParams::default();
        let mut sampled = SamplingParams::default();
        sampled.temperature = 1.0;
        let random_vals = [0.0; 4];

        let mut contiguous = SelectScratch::new(&ctx, logits.seq_len, logits.hidden_dim);
        let all_greedy = [&greedy, &greedy, &greedy, &greedy];
        let contiguous_tokens = select_batch_tokens_into(
            &ctx,
            &logits,
            &all_greedy,
            &random_vals,
            &mut contiguous.row_indices,
            &mut contiguous.argmax_values,
            &mut contiguous.argmax_indices,
            &mut contiguous.probs,
            &mut contiguous.top1_values,
            &mut contiguous.row_states,
            &mut contiguous.valid,
            &mut contiguous.out,
        )
        .expect("contiguous greedy select");

        let mut indexed = SelectScratch::new(&ctx, logits.seq_len, logits.hidden_dim);
        let mixed = [&greedy, &sampled, &greedy, &greedy];
        let indexed_tokens = select_batch_tokens_into(
            &ctx,
            &logits,
            &mixed,
            &random_vals,
            &mut indexed.row_indices,
            &mut indexed.argmax_values,
            &mut indexed.argmax_indices,
            &mut indexed.probs,
            &mut indexed.top1_values,
            &mut indexed.row_states,
            &mut indexed.valid,
            &mut indexed.out,
        )
        .expect("mixed select");

        for row in [0usize, 2, 3] {
            assert_eq!(
                contiguous_tokens[row], indexed_tokens[row],
                "greedy row {row} differs between contiguous and indexed argmax"
            );
        }
        assert_eq!(contiguous_tokens, vec![17, 5001, 8192, 4095]);
    }

    #[test]
    fn all_greedy_contiguous_argmax_ignores_cuda_graph_padding_rows() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let logits = logits_with_argmax_rows(
            &ctx,
            8193,
            &[17, 5001, 8192, 4095, 7, 123, 456, 789],
            &[(3, 4096)],
        );
        let greedy = SamplingParams::default();
        let real_rows = [&greedy, &greedy, &greedy, &greedy, &greedy];
        let random_vals = [0.0; 5];
        let mut scratch = SelectScratch::new(&ctx, logits.seq_len, logits.hidden_dim);

        let tokens = select_batch_tokens_into(
            &ctx,
            &logits,
            &real_rows,
            &random_vals,
            &mut scratch.row_indices,
            &mut scratch.argmax_values,
            &mut scratch.argmax_indices,
            &mut scratch.probs,
            &mut scratch.top1_values,
            &mut scratch.row_states,
            &mut scratch.valid,
            &mut scratch.out,
        )
        .expect("padded greedy select");

        assert_eq!(tokens, vec![17, 5001, 8192, 4095, 7]);
    }
}
