use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates, HiddenStatesRef};

const FLASHINFER_TOPK_ROW_STATES_BYTES: usize = 1024 * 1024;

/// One non-greedy row of a batched sampling call.
///
/// `temperature` must be > 0 and `top_p` in (0, 1] — greedy rows
/// (`temperature <= 0` or `top_k == 1`) belong on the argmax path.
/// `top_k <= 0` means disabled.
#[derive(Clone, Copy, Debug)]
pub struct BatchSamplingRow {
    /// Row index into the logits arena.
    pub row: usize,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
}

/// Device buffers for `gpu_sample_batch_into`, sized for `max_rows` x `vocab`.
pub struct BatchSamplingScratch {
    probs: CudaSlice<f32>,
    row_indices: CudaSlice<i32>,
    temperature: CudaSlice<f32>,
    top_k: CudaSlice<i32>,
    top_p: CudaSlice<f32>,
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
pub fn gpu_sample_batch_into(
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
        row_indices.push(i32::try_from(r.row)?);
        temperature.push(r.temperature);
        // FlashInfer reads top_k as u32; "disabled" is any k >= vocab.
        let vocab = i32::try_from(scratch.vocab)?;
        top_k.push(if r.top_k <= 0 || r.top_k > vocab {
            vocab
        } else {
            r.top_k
        });
        top_p.push(r.top_p);
    }
    ctx.stream
        .memcpy_htod(&row_indices, &mut scratch.row_indices)?;
    ctx.stream
        .memcpy_htod(&temperature, &mut scratch.temperature)?;
    ctx.stream.memcpy_htod(&top_k, &mut scratch.top_k)?;
    ctx.stream.memcpy_htod(&top_p, &mut scratch.top_p)?;

    {
        let softmax_workspace_bytes = scratch.softmax_workspace.len();
        let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = scratch.row_indices.device_ptr(&ctx.stream);
        let (probs_ptr, _gp) = scratch.probs.device_ptr_mut(&ctx.stream);
        let (temp_ptr, _gt) = scratch.temperature.device_ptr(&ctx.stream);
        let (top_k_ptr, _gk) = scratch.top_k.device_ptr(&ctx.stream);
        let (top_p_ptr, _gtp) = scratch.top_p.device_ptr(&ctx.stream);
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
                valid_ptr as *mut u8,
                out_ptr as *mut i32,
                ws_ptr as *mut u8,
                softmax_workspace_bytes,
                n as i32,
                scratch.vocab as i32,
                seed,
                0,
                ctx.stream.cu_stream(),
            )
        };
        ensure!(err == 0, "batch sampling kernel failed: cudaError {err}");
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

pub fn argmax_batch_bf16_indexed_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    row_indices: &CudaSlice<i32>,
    rows: usize,
    values: &mut CudaSlice<half::bf16>,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    if rows == 0 {
        return Err(anyhow!("argmax indexed batch requires at least one row"));
    }
    if row_indices.len() < rows {
        return Err(anyhow!(
            "argmax indexed row scratch too small: have {}, need {}",
            row_indices.len(),
            rows
        ));
    }
    if values.len() < rows {
        return Err(anyhow!(
            "argmax indexed values scratch too small: have {}, need {}",
            values.len(),
            rows
        ));
    }
    if out.len() < rows {
        return Err(anyhow!(
            "argmax indexed output too small: have {}, need {}",
            out.len(),
            rows
        ));
    }

    let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (row_indices_ptr, _gr) = row_indices.device_ptr(&ctx.stream);
    let (values_ptr, _gv) = values.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::argmax_batch_bf16_indexed_cuda(
            logits_ptr as *const ffi::Half,
            row_indices_ptr as *const i32,
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
    // temperature <= 0 is argmax regardless of top_p (the temperature -> 0
    // limit), and top_k == 1 leaves a single token for top_p to renormalize.
    // Routing these to the sampler would also divide by temperature = 0.
    if temperature <= 0.0 || top_k == 1 {
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
            },
            BatchSamplingRow {
                row: 4,
                temperature: 1.0,
                top_k: -1,
                top_p: 0.5,
            },
            BatchSamplingRow {
                row: 6,
                temperature: 0.05,
                top_k: -1,
                top_p: 1.0,
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
            },
            BatchSamplingRow {
                row: 5,
                temperature: 1.0,
                top_k: -1,
                top_p: 1.0,
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
            },
            BatchSamplingRow {
                row: 7,
                temperature: 4.0,
                top_k: -1,
                top_p: 1.0,
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
    fn batch_sampling_rejects_greedy_rows() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let logits = arena_with_rows(&ctx, &[]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [BatchSamplingRow {
            row: 0,
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
        }];
        assert!(
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 1, &mut scratch).is_err(),
            "temperature=0 must be rejected — greedy rows take the argmax path"
        );
    }
}
