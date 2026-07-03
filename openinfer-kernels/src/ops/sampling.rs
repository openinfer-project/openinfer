use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates, HiddenStatesRef};

/// One non-greedy row of a batched sampling call.
///
/// `temperature` must be > 0 and `top_p` in (0, 1] — greedy rows
/// (`temperature <= 0` or `top_k == 1`) belong on the argmax path.
/// `top_k <= 0` means disabled. `min_p` in [0, 1); `0.0` means disabled —
/// `gpu_sample_batch_into` partitions min_p rows into their own pass, so
/// callers may mix freely.
#[derive(Clone, Copy, Debug)]
pub struct BatchSamplingRow {
    /// Row index into the logits arena.
    pub row: usize,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
}

/// Device buffers for `gpu_sample_batch_into`, sized for `max_rows` x `vocab`.
pub struct BatchSamplingScratch {
    probs: CudaSlice<f32>,
    row_indices: CudaSlice<i32>,
    temperature: CudaSlice<f32>,
    top_k: CudaSlice<i32>,
    top_p: CudaSlice<f32>,
    min_p: CudaSlice<f32>,
    topk_row_states: CudaSlice<u8>,
    valid: CudaSlice<u8>,
    out: CudaSlice<i32>,
    softmax_workspace: CudaSlice<u8>,
    max_rows: usize,
    vocab: usize,
}

impl BatchSamplingScratch {
    pub fn new(ctx: &DeviceContext, max_rows: usize, vocab: usize) -> Result<Self> {
        ensure!(
            max_rows > 0 && vocab > 0,
            "batch sampling scratch requires max_rows > 0 and vocab > 0"
        );
        // OnlineSoftmax vocab-splitting path: batch x ceil(vocab / 8192)
        // partials of {f32 max, f32 denominator}, plus alignment slack.
        let softmax_workspace_bytes = max_rows * vocab.div_ceil(8192) * 8 + 256;
        let topk_row_states_bytes = unsafe { ffi::gpu_sample_topk_renorm_row_states_bytes_cuda() };
        let alloc = |n: usize| -> Result<CudaSlice<f32>> {
            ctx.stream
                .alloc_zeros(n)
                .map_err(|e| anyhow!("batch sampling scratch alloc failed: {e}"))
        };
        Ok(Self {
            probs: alloc(max_rows * vocab)?,
            row_indices: ctx.stream.alloc_zeros(max_rows)?,
            temperature: alloc(max_rows)?,
            top_k: ctx.stream.alloc_zeros(max_rows)?,
            top_p: alloc(max_rows)?,
            min_p: alloc(max_rows)?,
            topk_row_states: ctx.stream.alloc_zeros(topk_row_states_bytes)?,
            valid: ctx.stream.alloc_zeros(max_rows)?,
            out: ctx.stream.alloc_zeros(max_rows)?,
            softmax_workspace: ctx.stream.alloc_zeros(softmax_workspace_bytes)?,
            max_rows,
            vocab,
        })
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }
}

/// Batched temperature/top-k/top-p sampling: gathers the requested bf16 arena
/// rows, then runs FlashInfer's batched softmax + sampling — three kernel
/// launches, one sync, and one D2H for the whole batch.
///
/// `seed` must be fresh per decode step (one philox seed per call; rows
/// decorrelate through the philox subsequence). Returns one token per row, in
/// `rows` order.
///
/// min_p rows run as their own pass: if they shared a call, every row would
/// ride the min_p kernel, whose u-scaling (`u * q`) and survivor predicate
/// differ from the fused fast path — a min_p == 0 row could then sample a
/// different token than it would alone. Partitioning here (not in callers)
/// keeps "min_p == 0 rows take the original path" true for every caller, at
/// the cost of a second full pass (gather + softmax + sample, own sync) only
/// when a batch actually mixes.
pub fn gpu_sample_batch_into(
    ctx: &DeviceContext,
    logits: HiddenStatesRef<'_>,
    rows: &[BatchSamplingRow],
    seed: u64,
    scratch: &mut BatchSamplingScratch,
) -> Result<Vec<u32>> {
    if rows.iter().all(|r| r.min_p > 0.0) || rows.iter().all(|r| r.min_p <= 0.0) {
        return sample_uniform_batch_into(ctx, logits, rows, seed, scratch);
    }
    let (minp, plain): (Vec<BatchSamplingRow>, Vec<BatchSamplingRow>) =
        rows.iter().copied().partition(|r| r.min_p > 0.0);
    let plain_tokens = sample_uniform_batch_into(ctx, logits, &plain, seed, scratch)?;
    // Distinct philox key for the second pass: both passes restart their
    // subsequences at 0, so reusing `seed` would hand minp row i the same
    // uniform stream as plain row i and correlate their tokens.
    let minp_seed = seed ^ 0x9E37_79B9_7F4A_7C15;
    let minp_tokens = sample_uniform_batch_into(ctx, logits, &minp, minp_seed, scratch)?;
    let mut plain_it = plain_tokens.into_iter();
    let mut minp_it = minp_tokens.into_iter();
    Ok(rows
        .iter()
        .map(|r| {
            if r.min_p > 0.0 {
                minp_it.next().expect("minp token per minp row")
            } else {
                plain_it.next().expect("plain token per plain row")
            }
        })
        .collect())
}

/// The target distribution at each requested arena row: gather + softmax +
/// top-k/top-p renorm, written to `probs_out` as `n_rows x vocab` f32 —
/// filtered tokens are exact zeros. This is the verify-side input to
/// speculative rejection sampling; it is distribution-equivalent to the
/// sampling fast path (the fused sampler filters at draw time, this filters
/// then draws).
///
/// `min_p` rows are rejected: the min_p mask is applied inside the sampling
/// kernel, not as a renorm, so a min_p target distribution is not
/// representable here yet — callers must keep such requests off the
/// speculative path.
pub fn gpu_verify_probs_into(
    ctx: &DeviceContext,
    logits: HiddenStatesRef<'_>,
    rows: &[BatchSamplingRow],
    probs_out: &mut CudaSlice<f32>,
    scratch: &mut BatchSamplingScratch,
) -> Result<()> {
    let n = rows.len();
    ensure!(n > 0, "verify probs requires at least one row");
    ensure!(
        n <= scratch.max_rows,
        "verify probs scratch too small: {n} rows > capacity {}",
        scratch.max_rows
    );
    ensure!(
        logits.hidden_dim == scratch.vocab,
        "verify probs vocab mismatch: logits {} vs scratch {}",
        logits.hidden_dim,
        scratch.vocab
    );
    ensure!(
        probs_out.len() >= n * scratch.vocab,
        "verify probs output {} < {n} x {}",
        probs_out.len(),
        scratch.vocab
    );

    let mut row_indices = Vec::with_capacity(n);
    let mut temperature = Vec::with_capacity(n);
    let mut top_k = Vec::with_capacity(n);
    let mut top_p = Vec::with_capacity(n);
    let mut has_top_k_filter = false;
    let mut has_top_p_filter = false;
    for r in rows {
        ensure!(
            r.row < logits.seq_len,
            "verify probs row {} out of arena range {}",
            r.row,
            logits.seq_len
        );
        ensure!(
            r.temperature > 0.0 && r.temperature.is_finite(),
            "verify probs temperature {} must be finite and > 0",
            r.temperature
        );
        ensure!(
            r.top_p > 0.0 && r.top_p <= 1.0,
            "verify probs top_p {} must be in (0, 1]",
            r.top_p
        );
        ensure!(
            r.min_p == 0.0,
            "verify probs cannot represent a min_p target distribution (min_p {})",
            r.min_p
        );
        row_indices.push(i32::try_from(r.row)?);
        temperature.push(r.temperature);
        let vocab = i32::try_from(scratch.vocab)?;
        let clamped_top_k = if r.top_k > 0 && r.top_k < vocab {
            has_top_k_filter = true;
            r.top_k
        } else {
            vocab
        };
        top_k.push(clamped_top_k);
        if r.top_p < 1.0 {
            has_top_p_filter = true;
        }
        top_p.push(r.top_p);
    }
    ctx.stream
        .memcpy_htod(&row_indices, &mut scratch.row_indices)?;
    ctx.stream
        .memcpy_htod(&temperature, &mut scratch.temperature)?;
    ctx.stream.memcpy_htod(&top_k, &mut scratch.top_k)?;
    ctx.stream.memcpy_htod(&top_p, &mut scratch.top_p)?;

    let softmax_workspace_bytes = scratch.softmax_workspace.len();
    let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (indices_ptr, _gi) = scratch.row_indices.device_ptr(&ctx.stream);
    let (probs_ptr, _gp) = probs_out.device_ptr_mut(&ctx.stream);
    let (temp_ptr, _gt) = scratch.temperature.device_ptr(&ctx.stream);
    let (top_k_ptr, _gk) = scratch.top_k.device_ptr(&ctx.stream);
    let (top_p_ptr, _gtp) = scratch.top_p.device_ptr(&ctx.stream);
    let row_states = if has_top_k_filter {
        Some(scratch.topk_row_states.device_ptr_mut(&ctx.stream))
    } else {
        None
    };
    let row_states_ptr = row_states.as_ref().map_or(0, |(ptr, _guard)| *ptr);
    let (ws_ptr, _gw) = scratch.softmax_workspace.device_ptr_mut(&ctx.stream);
    let err = unsafe {
        ffi::gpu_verify_probs_flashinfer_cuda(
            logits_ptr as *const ffi::Half,
            indices_ptr as *const i32,
            probs_ptr as *mut f32,
            temp_ptr as *const f32,
            top_k_ptr as *const i32,
            top_p_ptr as *const f32,
            row_states_ptr as *mut u8,
            ws_ptr as *mut u8,
            softmax_workspace_bytes,
            n as i32,
            scratch.vocab as i32,
            i32::from(has_top_k_filter),
            i32::from(has_top_p_filter),
            crate::tensor::active_cu_stream(ctx),
        )
    };
    ensure!(err == 0, "verify probs kernel failed: cudaError {err}");
    Ok(())
}

/// Chain speculative (rejection) sampling over per-row verify spans.
///
/// For each of the `batch` rows: accept draft token `i` with probability
/// `min(1, p_target(x_i) / q_draft(x_i))`; the first rejection resamples from
/// `relu(target − draft)` renormalized and stops; full acceptance emits the
/// bonus token sampled from the target at position K. Output row layout is
/// `K+1` token ids, `-1`-filled after the stop — exactly the "longest
/// accepted prefix + one model token" contract `accept_greedy` has.
///
/// With `onehot_draft` (a greedy/argmax proposer), `draft_probs` is derived
/// on-device from `draft_token_ids` — the degenerate proposal
/// `q(x) = δ(x − draft)`, under which acceptance is `min(1, p_target(draft))`
/// and the residual is the target with the draft token's mass removed:
/// rejection sampling stays distribution-exact for a deterministic proposer.
///
/// `target_probs` must be post-filter probabilities (`gpu_verify_probs_into`),
/// `batch x (K+1) x vocab`; `draft_probs` is `batch x K x vocab` scratch.
/// One philox draw sequence per row; pass a fresh `seed`/`offset` per step
/// (same discipline as the batched sampler).
///
/// Returns `(accepted, emitted)` per row. **`emitted` is the accepted-prefix
/// length** (what the commit logic consumes); `accepted` keeps counting
/// hypothetical acceptances *past* the first rejection — it is FlashInfer's
/// acceptance-rate telemetry, not a commit signal.
#[allow(clippy::too_many_arguments)]
pub fn gpu_spec_accept_into(
    ctx: &DeviceContext,
    draft_probs: &mut CudaSlice<f32>,
    draft_token_ids: &CudaSlice<i32>,
    target_probs: &mut CudaSlice<f32>,
    output_token_ids: &mut CudaSlice<i32>,
    batch: usize,
    num_spec_tokens: usize,
    vocab: usize,
    seed: u64,
    offset: u64,
) -> Result<(Vec<i32>, Vec<i32>)> {
    ensure!(
        batch > 0 && num_spec_tokens > 0,
        "spec accept requires batch > 0 and K > 0"
    );
    ensure!(
        draft_probs.len() >= batch * num_spec_tokens * vocab
            && target_probs.len() >= batch * (num_spec_tokens + 1) * vocab
            && draft_token_ids.len() >= batch * num_spec_tokens
            && output_token_ids.len() >= batch * (num_spec_tokens + 1),
        "spec accept buffer too small for batch {batch} x K {num_spec_tokens} x vocab {vocab}"
    );
    // The kernel accumulates into the counters, so they start at zero.
    let mut accepted: CudaSlice<i32> = ctx.stream.alloc_zeros(batch)?;
    let mut emitted: CudaSlice<i32> = ctx.stream.alloc_zeros(batch)?;
    {
        let (dp, _g1) = draft_probs.device_ptr_mut(&ctx.stream);
        let (ids, _g2) = draft_token_ids.device_ptr(&ctx.stream);
        let (tp, _g3) = target_probs.device_ptr_mut(&ctx.stream);
        let (out, _g4) = output_token_ids.device_ptr_mut(&ctx.stream);
        let (acc, _g5) = accepted.device_ptr_mut(&ctx.stream);
        let (emi, _g6) = emitted.device_ptr_mut(&ctx.stream);
        let err = unsafe {
            ffi::gpu_chain_speculative_sampling_cuda(
                dp as *mut f32,
                ids as *const i32,
                tp as *mut f32,
                out as *mut i32,
                acc as *mut i32,
                emi as *mut i32,
                batch as i32,
                num_spec_tokens as i32,
                vocab as i32,
                1,
                seed,
                offset,
                crate::tensor::active_cu_stream(ctx),
            )
        };
        ensure!(
            err == 0,
            "chain speculative sampling failed: cudaError {err}"
        );
    }
    let accepted_h = ctx.stream.clone_dtoh(&accepted)?;
    let emitted_h = ctx.stream.clone_dtoh(&emitted)?;
    ctx.sync()?;
    Ok((accepted_h, emitted_h))
}

/// One homogeneous FlashInfer pass (rows are all-min_p or all-plain).
fn sample_uniform_batch_into(
    ctx: &DeviceContext,
    logits: HiddenStatesRef<'_>,
    rows: &[BatchSamplingRow],
    seed: u64,
    scratch: &mut BatchSamplingScratch,
) -> Result<Vec<u32>> {
    let n = rows.len();
    ensure!(n > 0, "batch sampling requires at least one row");
    ensure!(
        n <= scratch.max_rows,
        "batch sampling scratch too small: {n} rows > capacity {}",
        scratch.max_rows
    );
    ensure!(
        logits.hidden_dim == scratch.vocab,
        "batch sampling vocab mismatch: logits {} vs scratch {}",
        logits.hidden_dim,
        scratch.vocab
    );

    let mut row_indices = Vec::with_capacity(n);
    let mut temperature = Vec::with_capacity(n);
    let mut top_k = Vec::with_capacity(n);
    let mut top_p = Vec::with_capacity(n);
    let mut min_p = Vec::with_capacity(n);
    let mut has_top_k_filter = false;
    let mut has_top_p_filter = false;
    let mut has_min_p_filter = false;
    for r in rows {
        ensure!(
            r.row < logits.seq_len,
            "batch sampling row {} out of arena range {}",
            r.row,
            logits.seq_len
        );
        ensure!(
            r.temperature > 0.0 && r.temperature.is_finite(),
            "batch sampling temperature {} must be finite and > 0 (greedy rows take the argmax path)",
            r.temperature
        );
        ensure!(
            r.top_p > 0.0 && r.top_p <= 1.0,
            "batch sampling top_p {} must be in (0, 1]",
            r.top_p
        );
        ensure!(
            (0.0..1.0).contains(&r.min_p) && r.min_p.is_finite(),
            "batch sampling min_p {} must be in [0, 1)",
            r.min_p
        );
        row_indices.push(i32::try_from(r.row)?);
        temperature.push(r.temperature);
        // FlashInfer reads top_k as u32; "disabled" is any k >= vocab.
        let vocab = i32::try_from(scratch.vocab)?;
        let clamped_top_k = if r.top_k > 0 && r.top_k < vocab {
            has_top_k_filter = true;
            r.top_k
        } else {
            vocab
        };
        top_k.push(clamped_top_k);
        if r.top_p < 1.0 {
            has_top_p_filter = true;
        }
        top_p.push(r.top_p);
        if r.min_p > 0.0 {
            has_min_p_filter = true;
        }
        min_p.push(r.min_p);
    }
    ctx.stream
        .memcpy_htod(&row_indices, &mut scratch.row_indices)?;
    ctx.stream
        .memcpy_htod(&temperature, &mut scratch.temperature)?;
    ctx.stream.memcpy_htod(&top_k, &mut scratch.top_k)?;
    ctx.stream.memcpy_htod(&top_p, &mut scratch.top_p)?;
    if has_min_p_filter {
        ctx.stream.memcpy_htod(&min_p, &mut scratch.min_p)?;
    }

    {
        let softmax_workspace_bytes = scratch.softmax_workspace.len();
        let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = scratch.row_indices.device_ptr(&ctx.stream);
        let (probs_ptr, _gp) = scratch.probs.device_ptr_mut(&ctx.stream);
        let (temp_ptr, _gt) = scratch.temperature.device_ptr(&ctx.stream);
        let (top_k_ptr, _gk) = scratch.top_k.device_ptr(&ctx.stream);
        let (top_p_ptr, _gtp) = scratch.top_p.device_ptr(&ctx.stream);
        let (min_p_ptr, _gmp) = scratch.min_p.device_ptr(&ctx.stream);
        // topk_row_states is only read on the min_p pipeline; the fast path
        // hands the kernel a null instead of borrowing the buffer.
        let row_states = if has_min_p_filter {
            Some(scratch.topk_row_states.device_ptr_mut(&ctx.stream))
        } else {
            None
        };
        let row_states_ptr = row_states.as_ref().map_or(0, |(ptr, _guard)| *ptr);
        let (valid_ptr, _gv) = scratch.valid.device_ptr_mut(&ctx.stream);
        let (out_ptr, _go) = scratch.out.device_ptr_mut(&ctx.stream);
        let (ws_ptr, _gw) = scratch.softmax_workspace.device_ptr_mut(&ctx.stream);

        let err = unsafe {
            ffi::gpu_sample_batch_flashinfer_cuda(
                logits_ptr as *const ffi::Half,
                indices_ptr as *const i32,
                probs_ptr as *mut f32,
                temp_ptr as *const f32,
                top_k_ptr as *const i32,
                top_p_ptr as *const f32,
                if has_min_p_filter {
                    min_p_ptr as *const f32
                } else {
                    std::ptr::null()
                },
                row_states_ptr as *mut u8,
                valid_ptr as *mut u8,
                out_ptr as *mut i32,
                ws_ptr as *mut u8,
                softmax_workspace_bytes,
                n as i32,
                scratch.vocab as i32,
                i32::from(has_top_k_filter),
                i32::from(has_top_p_filter),
                seed,
                0,
                crate::tensor::active_cu_stream(ctx),
            )
        };
        ensure!(
            err == 0,
            "batch sampling kernel failed with error {err}{}",
            crate::ops::ffi_exception_message(err)
        );
    }

    let out = ctx
        .stream
        .clone_dtoh(&scratch.out)
        .map_err(|e| anyhow!("D2H batch sample read failed: {e}"))?;
    let valid = ctx
        .stream
        .clone_dtoh(&scratch.valid)
        .map_err(|e| anyhow!("D2H batch sample valid read failed: {e}"))?;
    ctx.sync()?;

    let mut tokens = Vec::with_capacity(n);
    for (i, r) in rows.iter().enumerate() {
        ensure!(
            valid[i] != 0,
            "batch sampling produced no valid token for arena row {} (probs failed to cover u)",
            r.row
        );
        ensure!(
            out[i] >= 0 && (out[i] as usize) < scratch.vocab,
            "batch sampling token {} for arena row {} out of vocab range {}",
            out[i],
            r.row,
            scratch.vocab
        );
        tokens.push(out[i] as u32);
    }
    Ok(tokens)
}

/// Argmax — returns the index of the maximum element.
///
/// Allocates a temporary output buffer. Model decode paths use batched argmax
/// through `openinfer-sample`'s `select_batch`.
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
                crate::tensor::active_cu_stream(ctx),
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

/// Single-row bf16 argmax into pre-allocated device outputs (`value[0]`,
/// `index[0]`). Slice-level twin of [`argmax_batch_bf16_into`] (same kernel,
/// rows=1, lowest index wins ties, NaN never wins) for callers whose logits
/// live in a persistent decode arena. The bf16 top value is emitted so the
/// caller can keep the crash-early non-finite guard after the 2-byte D2H.
pub fn argmax_bf16_into(
    ctx: &DeviceContext,
    logits: &CudaSlice<half::bf16>,
    n: usize,
    value: &mut CudaSlice<half::bf16>,
    index: &mut CudaSlice<i32>,
) -> Result<()> {
    if n == 0 || logits.len() < n {
        return Err(anyhow!(
            "argmax_bf16_into logits too small: have {}, need {n}",
            logits.len()
        ));
    }
    if value.is_empty() || index.is_empty() {
        return Err(anyhow!("argmax_bf16_into outputs must hold one element"));
    }
    let (x_ptr, _gx) = logits.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = value.device_ptr_mut(&ctx.stream);
    let (i_ptr, _gi) = index.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::argmax_batch_bf16_cuda(
            x_ptr as *const ffi::Half,
            v_ptr as *mut ffi::Half,
            i_ptr as *mut i32,
            1,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
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
            crate::tensor::active_cu_stream(ctx),
        );
    }

    Ok(())
}

pub fn argmax_batch_bf16_split_partials_len(rows: usize, vocab: usize) -> usize {
    const TILE_ELEMS: usize = 4096;
    rows * vocab.div_ceil(TILE_ELEMS)
}

/// Partial-buffer length for the Markov-step argmax, whose tiles are 1024
/// elements: its read is one logits row + one bias row, so it needs 4x the
/// blocks of the full-row batched argmax to not be latency-bound (see
/// `MARKOV_STEP_TILE_ELEMS` in `argmax.cu`).
pub fn markov_step_argmax_partials_len(rows: usize, vocab: usize) -> usize {
    const TILE_ELEMS: usize = 1024;
    rows * vocab.div_ceil(TILE_ELEMS)
}

/// Row-wise two-stage bf16 argmax over `rows` rows of `n`: tile-parallel
/// partials then one finalize block per row. Same per-row total order as
/// [`argmax_bf16_into`] (lowest GLOBAL index wins ties, NaN never wins) — the
/// partials carry global indices, so each row's result is bit-identical to
/// the single-block scan (and independent of the other rows) while each vocab
/// row spreads over ~n/4096 CTAs instead of one. `partial_*` must hold
/// `argmax_batch_bf16_split_partials_len(rows, n)` elements; `values`/`indices`
/// hold one element per row.
#[allow(clippy::too_many_arguments)]
pub fn argmax_bf16_split_into(
    ctx: &DeviceContext,
    logits: &CudaSlice<half::bf16>,
    rows: usize,
    n: usize,
    partial_values: &mut CudaSlice<f32>,
    partial_indices: &mut CudaSlice<i32>,
    values: &mut CudaSlice<half::bf16>,
    indices: &mut CudaSlice<i32>,
) -> Result<()> {
    if rows == 0 || n == 0 || logits.len() < rows * n {
        return Err(anyhow!(
            "argmax_bf16_split_into logits too small: have {}, need {}",
            logits.len(),
            rows * n
        ));
    }
    if values.len() < rows || indices.len() < rows {
        return Err(anyhow!(
            "argmax_bf16_split_into outputs must hold {rows} elements: have {}/{}",
            values.len(),
            indices.len()
        ));
    }
    let needed = argmax_batch_bf16_split_partials_len(rows, n);
    if partial_values.len() < needed || partial_indices.len() < needed {
        return Err(anyhow!(
            "argmax_bf16_split_into partials too small: {}/{} need {needed}",
            partial_values.len(),
            partial_indices.len()
        ));
    }
    let (x_ptr, _gx) = logits.device_ptr(&ctx.stream);
    let (pv_ptr, _gpv) = partial_values.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = partial_indices.device_ptr_mut(&ctx.stream);
    let (v_ptr, _gv) = values.device_ptr_mut(&ctx.stream);
    let (i_ptr, _gi) = indices.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::argmax_batch_bf16_split_cuda(
            x_ptr as *const ffi::Half,
            v_ptr as *mut ffi::Half,
            i_ptr as *mut i32,
            pv_ptr as *mut f32,
            pi_ptr as *mut i32,
            rows as i32,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// Two-stage indexed batched argmax: tile-parallel partials then a per-row
/// finalize. Lowest index wins ties; each vocab row spreads over many CTAs
/// instead of one.
#[allow(clippy::too_many_arguments)]
pub fn argmax_batch_bf16_split_indexed_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    row_indices: &CudaSlice<i32>,
    rows: usize,
    partial_values: &mut CudaSlice<f32>,
    partial_indices: &mut CudaSlice<i32>,
    values: &mut CudaSlice<half::bf16>,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    if rows == 0 {
        return Err(anyhow!(
            "argmax split indexed batch requires at least one row"
        ));
    }
    if row_indices.len() < rows {
        return Err(anyhow!(
            "argmax split indexed row scratch too small: have {}, need {}",
            row_indices.len(),
            rows
        ));
    }
    let needed_partials = argmax_batch_bf16_split_partials_len(rows, logits.hidden_dim);
    if partial_values.len() < needed_partials || partial_indices.len() < needed_partials {
        return Err(anyhow!(
            "argmax split indexed partials scratch too small: have {}/{}, need {}",
            partial_values.len(),
            partial_indices.len(),
            needed_partials
        ));
    }
    if values.len() < rows || out.len() < rows {
        return Err(anyhow!(
            "argmax split indexed outputs too small: have {}/{}, need {}",
            values.len(),
            out.len(),
            rows
        ));
    }

    let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (row_indices_ptr, _gr) = row_indices.device_ptr(&ctx.stream);
    let (pv_ptr, _gpv) = partial_values.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = partial_indices.device_ptr_mut(&ctx.stream);
    let (values_ptr, _gv) = values.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::argmax_batch_bf16_split_indexed_cuda(
            logits_ptr as *const ffi::Half,
            row_indices_ptr as *const i32,
            values_ptr as *mut ffi::Half,
            out_ptr as *mut i32,
            pv_ptr as *mut f32,
            pi_ptr as *mut i32,
            rows as i32,
            logits.hidden_dim as i32,
            crate::tensor::active_cu_stream(ctx),
        );
    }

    Ok(())
}

/// DSpark Markov-head step argmax. For each request `row`, argmax over
/// `base[row*block_size + step] + bias[row]` and write the chosen token id as
/// u32 (so it feeds straight back as the next step's prev-token lookup).
/// `base` is the request-major block logits `[rows*block_size, vocab]`; `bias`
/// is the per-request Markov logit bias `[rows, vocab]` for this step. `partial_*`
/// must hold `argmax_batch_bf16_split_partials_len(rows, vocab)` elements.
/// `sampled_tokens` receives the request-major block token at
/// `row * block_size + step`, allowing callers to D2H the finished block once.
#[allow(clippy::too_many_arguments)]
pub fn markov_step_argmax_into(
    ctx: &DeviceContext,
    base: &HiddenStates,
    bias: &HiddenStates,
    block_size: usize,
    step: usize,
    rows: usize,
    partial_values: &mut CudaSlice<f32>,
    partial_indices: &mut CudaSlice<i32>,
    out_tokens: &mut CudaSlice<u32>,
    sampled_tokens: &mut CudaSlice<u32>,
) -> Result<()> {
    if rows == 0 {
        return Err(anyhow!("markov step argmax requires at least one row"));
    }
    let vocab = base.hidden_dim;
    if bias.hidden_dim != vocab {
        return Err(anyhow!(
            "markov step bias vocab {} != base vocab {}",
            bias.hidden_dim,
            vocab
        ));
    }
    if base.seq_len < rows * block_size {
        return Err(anyhow!(
            "markov step base rows {} < rows*block_size {}",
            base.seq_len,
            rows * block_size
        ));
    }
    if bias.seq_len < rows {
        return Err(anyhow!("markov step bias rows {} < {}", bias.seq_len, rows));
    }
    if out_tokens.len() < rows {
        return Err(anyhow!(
            "markov step out too small: {} < {}",
            out_tokens.len(),
            rows
        ));
    }
    if sampled_tokens.len() < rows * block_size {
        return Err(anyhow!(
            "markov sampled-token scratch too small: {} < {}",
            sampled_tokens.len(),
            rows * block_size
        ));
    }
    let needed = markov_step_argmax_partials_len(rows, vocab);
    if partial_values.len() < needed || partial_indices.len() < needed {
        return Err(anyhow!(
            "markov step partials too small: {}/{} need {}",
            partial_values.len(),
            partial_indices.len(),
            needed
        ));
    }

    let (base_ptr, _gb) = base.data.device_ptr(&ctx.stream);
    let (bias_ptr, _gbi) = bias.data.device_ptr(&ctx.stream);
    let (pv_ptr, _gpv) = partial_values.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = partial_indices.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = out_tokens.device_ptr_mut(&ctx.stream);
    let (sampled_ptr, _gs) = sampled_tokens.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::markov_step_argmax_cuda(
            base_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            block_size as i32,
            step as i32,
            rows as i32,
            vocab as i32,
            pv_ptr as *mut f32,
            pi_ptr as *mut i32,
            out_ptr as *mut u32,
            sampled_ptr as *mut u32,
            crate::tensor::active_cu_stream(ctx),
        );
    }

    Ok(())
}

pub fn flashinfer_top1_row_states_bytes() -> usize {
    unsafe { ffi::flashinfer_top1_row_states_bytes_cuda() }
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
    let row_states_bytes = flashinfer_top1_row_states_bytes();
    if row_states_scratch.len() < row_states_bytes {
        return Err(anyhow!(
            "top1 row states scratch too small: have {}, need {}",
            row_states_scratch.len(),
            row_states_bytes
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
            crate::tensor::active_cu_stream(ctx),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    const VOCAB: usize = 32768; // >= 24576 so OnlineSoftmax takes the vocab-splitting path
    const ARENA_ROWS: usize = 8;

    /// Arena where every row not under test is poisoned with a dominant logit
    /// at `POISON_TOKEN` — a broken row gather makes every assertion fail.
    const POISON_TOKEN: usize = 7777;

    fn arena_with_rows(ctx: &DeviceContext, rows: &[(usize, Vec<f32>)]) -> HiddenStates {
        let mut host = vec![bf16::from_f32(0.0); ARENA_ROWS * VOCAB];
        for r in 0..ARENA_ROWS {
            host[r * VOCAB + POISON_TOKEN] = bf16::from_f32(20.0);
        }
        for (row, values) in rows {
            assert_eq!(values.len(), VOCAB);
            for (i, v) in values.iter().enumerate() {
                host[row * VOCAB + i] = bf16::from_f32(*v);
            }
        }
        let data = ctx.stream.clone_htod(&host).expect("htod logits");
        HiddenStates {
            data,
            hidden_dim: VOCAB,
            seq_len: ARENA_ROWS,
        }
    }

    fn flat_row(fill: f32) -> Vec<f32> {
        vec![fill; VOCAB]
    }

    #[test]
    fn batch_sampling_honors_top_k_top_p_and_gathers_rows() {
        let ctx = DeviceContext::new().expect("create CUDA context");

        // Row 1: top_k=5 — five high tokens; the unmasked tail would win ~83%
        // of draws (32k tokens at e^2 vs five at e^8..e^10), so a missing
        // top-k mask fails immediately.
        let top5: Vec<usize> = vec![11, 503, 1024, 9000, 32000];
        let mut row_k = flat_row(2.0);
        for (i, &t) in top5.iter().enumerate() {
            row_k[t] = 10.0 - 0.5 * i as f32;
        }

        // Row 4: top_p=0.5 with one token holding ~83% of the mass — the
        // nucleus is exactly that token, so every draw must return it.
        let mut row_p = flat_row(0.0);
        row_p[222] = 12.0;

        // Row 6: near-zero temperature sharpens to argmax.
        let mut row_t = flat_row(0.0);
        row_t[31999] = 4.0;

        let logits = arena_with_rows(&ctx, &[(1, row_k), (4, row_p), (6, row_t)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [
            BatchSamplingRow {
                row: 1,
                temperature: 1.0,
                top_k: 5,
                top_p: 1.0,
                min_p: 0.0,
            },
            BatchSamplingRow {
                row: 4,
                temperature: 1.0,
                top_k: -1,
                top_p: 0.5,
                min_p: 0.0,
            },
            BatchSamplingRow {
                row: 6,
                temperature: 0.05,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
        ];

        for seed in 0..64u64 {
            let tokens = gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, seed, &mut scratch)
                .expect("sample");
            assert!(
                top5.contains(&(tokens[0] as usize)),
                "seed {seed}: top_k=5 row sampled {} outside the top-5 set",
                tokens[0]
            );
            assert_eq!(
                tokens[1], 222,
                "seed {seed}: top_p=0.5 row escaped the single-token nucleus"
            );
            assert_eq!(
                tokens[2], 31999,
                "seed {seed}: near-zero temperature row missed the argmax"
            );
        }
    }

    #[test]
    fn batch_sampling_top_p_only_small_nucleus_collapses_to_argmax() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let mut row = flat_row(0.0);
        row[123] = 2.0;
        row[456] = 1.5;
        row[789] = 1.0;
        let logits = arena_with_rows(&ctx, &[(2, row)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [BatchSamplingRow {
            row: 2,
            temperature: 1.0,
            top_k: -1,
            top_p: 1e-6,
            min_p: 0.0,
        }];
        let tokens =
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 17, &mut scratch).expect("sample");
        assert_eq!(tokens, vec![123], "tiny top_p should collapse to argmax");
    }

    #[test]
    fn batch_sampling_same_seed_is_deterministic() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        // Two flat rows: uniform over 32768 tokens, so different seeds
        // colliding on both rows is ~1e-9.
        let logits = arena_with_rows(&ctx, &[(2, flat_row(0.0)), (5, flat_row(0.0))]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [
            BatchSamplingRow {
                row: 2,
                temperature: 1.0,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
            BatchSamplingRow {
                row: 5,
                temperature: 1.0,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
        ];

        let a =
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 42, &mut scratch).expect("sample");
        let b =
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 42, &mut scratch).expect("sample");
        let c =
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 43, &mut scratch).expect("sample");
        assert_eq!(a, b, "same seed must reproduce the same tokens");
        assert_ne!(a, c, "different seeds must diverge on flat rows");
    }

    #[test]
    fn verify_probs_renorm_matches_the_sampling_law() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        // Row with a known 3-token support: P = {0.6652, 0.2447, 0.0900}
        // (softmax of {2, 1, 0} with the rest at -120), plus top_k=2 on a
        // second view of the same row so the renorm must zero token 300 and
        // renormalize the survivors to {0.7311, 0.2689}.
        let mut row = flat_row(-120.0);
        row[100] = 2.0;
        row[200] = 1.0;
        row[300] = 0.0;
        let logits = arena_with_rows(&ctx, &[(1, row.clone()), (5, row)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [
            BatchSamplingRow {
                row: 1,
                temperature: 1.0,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
            BatchSamplingRow {
                row: 5,
                temperature: 1.0,
                top_k: 2,
                top_p: 1.0,
                min_p: 0.0,
            },
        ];
        let mut probs: CudaSlice<f32> = ctx.stream.alloc_zeros(2 * VOCAB).expect("probs");
        gpu_verify_probs_into(&ctx, logits.as_ref(), &rows, &mut probs, &mut scratch)
            .expect("verify probs");
        let host = ctx.stream.clone_dtoh(&probs).expect("D2H");
        ctx.sync().expect("sync");

        // Unfiltered row: sums to 1, matches the closed-form softmax.
        let r0 = &host[..VOCAB];
        let sum0: f32 = r0.iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-3, "row0 sum {sum0}");
        assert!((r0[100] - 0.6652).abs() < 5e-3, "P(100) {}", r0[100]);
        assert!((r0[200] - 0.2447).abs() < 5e-3, "P(200) {}", r0[200]);
        assert!((r0[300] - 0.0900).abs() < 5e-3, "P(300) {}", r0[300]);

        // top_k=2 row: token 300 is an exact zero (renorm, not epsilon), the
        // survivors renormalize, and the sum is 1 again.
        let r1 = &host[VOCAB..2 * VOCAB];
        assert_eq!(r1[300], 0.0, "top-k filtered token must be exactly zero");
        assert!((r1[100] - 0.7311).abs() < 5e-3, "renorm P(100) {}", r1[100]);
        assert!((r1[200] - 0.2689).abs() < 5e-3, "renorm P(200) {}", r1[200]);
        let sum1: f32 = r1.iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-3, "row1 sum {sum1}");
    }

    #[test]
    fn verify_probs_rejects_min_p_rows() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let logits = arena_with_rows(&ctx, &[(0, flat_row(0.0))]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [BatchSamplingRow {
            row: 0,
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.1,
        }];
        let mut probs: CudaSlice<f32> = ctx.stream.alloc_zeros(VOCAB).expect("probs");
        let err = gpu_verify_probs_into(&ctx, logits.as_ref(), &rows, &mut probs, &mut scratch)
            .expect_err("min_p rows must be rejected");
        assert!(err.to_string().contains("min_p"), "unexpected error: {err}");
    }

    #[test]
    fn spec_accept_full_acceptance_and_certain_rejection() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let k = 2usize;
        // Hand-built target probs, batch=2, K+1=3 positions each.
        // Row 0: target puts prob 1.0 on the draft token at every position →
        // certain acceptance; bonus position puts 1.0 on token 777.
        // Row 1: target puts 0.0 on the draft at position 0 (prob 1.0 on 555)
        // → certain first-step rejection; the residual is exactly {555: 1.0}.
        let mut target = vec![0.0f32; 2 * (k + 1) * VOCAB];
        target[100] = 1.0; // row0 pos0 -> 100
        target[VOCAB + 200] = 1.0; // row0 pos1 -> 200
        target[2 * VOCAB + 777] = 1.0; // row0 bonus -> 777
        let r1 = 3 * VOCAB;
        target[r1 + 555] = 1.0; // row1 pos0: all mass on 555, draft is 100
        target[r1 + VOCAB + 200] = 1.0; // never reached
        target[r1 + 2 * VOCAB + 300] = 1.0; // never reached
        let mut target_d: CudaSlice<f32> =
            ctx.stream.clone_htod(&target).expect("target probs H2D");
        let drafts: Vec<i32> = vec![100, 200, /* row1 */ 100, 200];
        let drafts_d = ctx.stream.clone_htod(&drafts).expect("draft ids H2D");
        let mut draft_probs: CudaSlice<f32> =
            ctx.stream.alloc_zeros(2 * k * VOCAB).expect("draft probs");
        let mut out: CudaSlice<i32> = ctx.stream.alloc_zeros(2 * (k + 1)).expect("out");

        let (accepted, emitted) = gpu_spec_accept_into(
            &ctx,
            &mut draft_probs,
            &drafts_d,
            &mut target_d,
            &mut out,
            2,
            k,
            VOCAB,
            0x5eed,
            0,
        )
        .expect("spec accept");
        let out_h = ctx.stream.clone_dtoh(&out).expect("D2H");
        ctx.sync().expect("sync");

        // Row 0: both drafts accepted (p_target = 1), bonus token 777 emitted.
        assert_eq!(&out_h[..3], &[100, 200, 777], "row0 {:?}", &out_h[..3]);
        assert_eq!(emitted[0], 2, "row0 emitted prefix");
        assert_eq!(accepted[0], 2, "row0 acceptance telemetry");
        // Row 1: draft rejected at pos 0 (p_target = 0); the residual
        // relu(target - onehot(100)) is exactly {555: 1.0}, so the resample is
        // deterministic; the tail is -1-filled. `emitted` is the commit
        // signal; `accepted` may exceed it (it keeps counting hypothetical
        // acceptances past the rejection - telemetry, asserted only >= 0).
        assert_eq!(&out_h[3..], &[555, -1, -1], "row1 {:?}", &out_h[3..]);
        assert_eq!(emitted[1], 0, "row1 emitted prefix");
        assert!(accepted[1] >= 0, "row1 telemetry sane");
    }

    #[test]
    fn batch_sampling_applies_per_row_temperature() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        // Effective 2-token distribution: logit ln(3) vs 0, everything else
        // at -120 so the 32766-token tail stays negligible even after the
        // temperature=4 flattening (e^-30 x 32766 ≈ 3e-9). P(token 100) =
        // 0.75 at temperature 1, 3^(1/4)/(3^(1/4)+1) ≈ 0.568 at temperature
        // 4. Fixed seed sequence + deterministic kernel make the observed
        // counts reproducible.
        let mut row = flat_row(-120.0);
        row[100] = 3.0f32.ln();
        row[200] = 0.0;
        let logits = arena_with_rows(&ctx, &[(3, row.clone()), (7, row)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [
            BatchSamplingRow {
                row: 3,
                temperature: 1.0,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
            BatchSamplingRow {
                row: 7,
                temperature: 4.0,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
        ];

        let draws = 300;
        let mut hits = [0u32; 2];
        for seed in 0..draws {
            let tokens =
                gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, seed as u64, &mut scratch)
                    .expect("sample");
            for (i, &t) in tokens.iter().enumerate() {
                assert!(
                    t == 100 || t == 200,
                    "row {i} sampled {t}, outside the 2-token support"
                );
                if t == 100 {
                    hits[i] += 1;
                }
            }
        }
        let freq_t1 = f64::from(hits[0]) / f64::from(draws);
        let freq_t4 = f64::from(hits[1]) / f64::from(draws);
        assert!(
            (0.65..=0.85).contains(&freq_t1),
            "temperature=1 row frequency {freq_t1} outside [0.65, 0.85] (expected 0.75)"
        );
        assert!(
            (0.47..=0.67).contains(&freq_t4),
            "temperature=4 row frequency {freq_t4} outside [0.47, 0.67] (expected 0.568)"
        );
    }

    #[test]
    fn batch_sampling_min_p_masks_below_threshold() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        // Two-token support: P(100) ≈ 0.7, P(200) ≈ 0.28, tail ≈ 0 (logits
        // ln(0.7/0.28) apart, rest at -120). min_p thresholds against the max
        // prob: 0.5 * 0.7 = 0.35 keeps only token 100; 0.2 * 0.7 = 0.14 keeps
        // both.
        let mut row = flat_row(-120.0);
        row[100] = (0.7f32 / 0.28).ln();
        row[200] = 0.0;
        let logits = arena_with_rows(&ctx, &[(1, row.clone()), (3, row)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");

        let strict = [BatchSamplingRow {
            row: 1,
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.5,
        }];
        for seed in 0..64u64 {
            let tokens = gpu_sample_batch_into(&ctx, logits.as_ref(), &strict, seed, &mut scratch)
                .expect("sample");
            assert_eq!(
                tokens[0], 100,
                "seed {seed}: min_p=0.5 must mask the 0.28-prob token"
            );
        }

        let loose = [BatchSamplingRow {
            row: 3,
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.2,
        }];
        let mut saw_minor = false;
        for seed in 0..128u64 {
            let tokens = gpu_sample_batch_into(&ctx, logits.as_ref(), &loose, seed, &mut scratch)
                .expect("sample");
            assert!(
                tokens[0] == 100 || tokens[0] == 200,
                "seed {seed}: min_p=0.2 sampled {} outside the surviving pair",
                tokens[0]
            );
            saw_minor |= tokens[0] == 200;
        }
        assert!(
            saw_minor,
            "min_p=0.2 never sampled the 0.28-prob token in 128 draws (~1e-19 if unmasked)"
        );
    }

    #[test]
    fn batch_sampling_min_p_composes_with_top_k_and_top_p() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        // Five spaced tokens; top_k=3 keeps {11, 503, 1024}, then the top-p /
        // min_p stages cut deeper. With min_p=0.6 after top-k renorm the
        // survivor set is exactly the argmax.
        let picks: Vec<usize> = vec![11, 503, 1024, 9000, 32000];
        let mut row = flat_row(-120.0);
        for (i, &t) in picks.iter().enumerate() {
            row[t] = 8.0 - 1.0 * i as f32;
        }
        let logits = arena_with_rows(&ctx, &[(2, row)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [BatchSamplingRow {
            row: 2,
            temperature: 1.0,
            top_k: 3,
            top_p: 0.99,
            min_p: 0.6,
        }];
        for seed in 0..64u64 {
            let tokens = gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, seed, &mut scratch)
                .expect("sample");
            assert_eq!(
                tokens[0], 11,
                "seed {seed}: top_k=3 + min_p=0.6 must collapse to the argmax"
            );
        }
    }

    #[test]
    fn batch_sampling_rejects_greedy_rows() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let logits = arena_with_rows(&ctx, &[]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [BatchSamplingRow {
            row: 0,
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.0,
        }];
        assert!(
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 1, &mut scratch).is_err(),
            "temperature=0 must be rejected — greedy rows take the argmax path"
        );
    }
}
