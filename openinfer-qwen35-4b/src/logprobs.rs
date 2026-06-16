use anyhow::Result;
use openinfer_core::engine::TokenLogprob;
use openinfer_core::tensor::{DeviceContext, HiddenStates};

pub(crate) fn snapshot_requested_logprobs(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    requested_top_k: &[usize],
) -> Result<Vec<Option<Vec<f32>>>> {
    anyhow::ensure!(
        requested_top_k.len() <= logits.seq_len,
        "Qwen3.5 logprobs request/logits row mismatch: requested={}, logits_rows={}",
        requested_top_k.len(),
        logits.seq_len
    );
    if !requested_top_k.iter().any(|&top_k| top_k > 0) {
        return Ok(vec![None; requested_top_k.len()]);
    }

    requested_top_k
        .iter()
        .enumerate()
        .map(|(i, &top_k)| {
            if top_k == 0 {
                Ok(None)
            } else {
                let row = crate::ops::extract_vec(ctx, logits, i)?;
                Ok(Some(row.to_host(ctx)?))
            }
        })
        .collect()
}

pub(crate) fn compute_logprobs_from_cpu(
    logits_f32: &[f32],
    sampled_token: u32,
    top_k: usize,
) -> Option<TokenLogprob> {
    if logits_f32.is_empty() {
        return None;
    }

    let max_val = logits_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits_f32.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();

    let sampled_logprob = logits_f32.get(sampled_token as usize)? - log_sum_exp;

    let k = top_k.min(logits_f32.len());
    let mut top = Vec::with_capacity(k);
    if k > 0 {
        let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (idx, &val) in logits_f32.iter().enumerate() {
            if best.len() < k || val > best.last().unwrap().1 {
                let pos = best.partition_point(|&(_, v)| v > val);
                best.insert(pos, (idx as u32, val));
                if best.len() > k {
                    best.pop();
                }
            }
        }
        for (idx, val) in best {
            top.push((idx, val - log_sum_exp));
        }
    }

    Some(TokenLogprob {
        logprob: sampled_logprob,
        top_logprobs: top,
    })
}
