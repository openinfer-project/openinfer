use anyhow::Result;
use openinfer_core::tensor::DeviceContext;
use openinfer_core::tensor::HiddenStates;

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
