//! Speculative-decoding sampling primitives: the verify-side target
//! distribution (`gpu_verify_probs_into`) and chain rejection sampling
//! (`gpu_spec_accept_into`). Split out of `sampling.rs` to keep that file
//! under the module size budget; the shared parameter packing and scratch
//! buffers live in the sibling `sampling` module.

use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::sampling::{BatchSamplingRow, BatchSamplingScratch, prepare_sampling_params};
use crate::ffi;
use crate::tensor::{DeviceContext, HiddenStatesRef};

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
        n <= scratch.max_rows(),
        "verify probs scratch too small: {n} rows > capacity {}",
        scratch.max_rows()
    );
    ensure!(
        logits.hidden_dim == scratch.vocab(),
        "verify probs vocab mismatch: logits {} vs scratch {}",
        logits.hidden_dim,
        scratch.vocab()
    );
    ensure!(
        probs_out.len() >= n * scratch.vocab(),
        "verify probs output {} < {n} x {}",
        probs_out.len(),
        scratch.vocab()
    );
    // A min_p target distribution is not representable as a renorm, so reject
    // such rows before packing the shared parameters.
    for r in rows {
        ensure!(
            r.min_p == 0.0,
            "verify probs cannot represent a min_p target distribution (min_p {})",
            r.min_p
        );
    }

    let params = prepare_sampling_params(rows, logits.seq_len, scratch.vocab())?;
    let has_top_k_filter = params.has_top_k_filter;
    let has_top_p_filter = params.has_top_p_filter;
    ctx.stream
        .memcpy_htod(&params.row_indices, &mut scratch.row_indices)?;
    ctx.stream
        .memcpy_htod(&params.temperature, &mut scratch.temperature)?;
    ctx.stream.memcpy_htod(&params.top_k, &mut scratch.top_k)?;
    ctx.stream.memcpy_htod(&params.top_p, &mut scratch.top_p)?;

    let vocab = scratch.vocab();
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
            vocab as i32,
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
/// `onehot_draft` selects the proposal semantics. With `onehot_draft == true`
/// (a greedy/argmax proposer) `draft_probs` is derived on-device from
/// `draft_token_ids` — the degenerate proposal `q(x) = δ(x − draft)`, under
/// which acceptance is `min(1, p_target(draft))` and the residual is the
/// target with the draft token's mass removed. With `onehot_draft == false`
/// the caller must supply the true `draft_probs` the proposer sampled from;
/// rejection sampling stays distribution-exact either way.
///
/// `target_probs` must be post-filter probabilities (`gpu_verify_probs_into`),
/// `batch x (K+1) x vocab`; `draft_probs` is `batch x K x vocab` scratch.
///
/// **Seeded requests are not representable here.** The chain kernel folds the
/// row index into the philox subsequence, so a fixed-seed request's draw
/// stream depends on batch composition — the exact replay trap the batched
/// sampler documents and dodges with per-row calls, and a per-row path does
/// not exist for the chain. Callers must keep seeded requests off the
/// speculative path entirely (a gate the verify scheduler enforces the same
/// way it already keeps min_p rows off it). Pass a fresh `seed`/`offset` per
/// verify step for the unseeded rows.
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
    onehot_draft: bool,
    seed: u64,
    offset: u64,
    scratch: &mut SpecAcceptScratch,
) -> Result<SpecAcceptCounts> {
    ensure!(
        batch > 0 && num_spec_tokens > 0 && vocab > 0,
        "spec accept requires batch > 0, K > 0, and vocab > 0"
    );
    ensure!(
        batch <= scratch.max_batch,
        "spec accept scratch too small: batch {batch} > capacity {}",
        scratch.max_batch
    );
    ensure!(
        draft_probs.len() >= batch * num_spec_tokens * vocab
            && target_probs.len() >= batch * (num_spec_tokens + 1) * vocab
            && draft_token_ids.len() >= batch * num_spec_tokens
            && output_token_ids.len() >= batch * (num_spec_tokens + 1),
        "spec accept buffer too small for batch {batch} x K {num_spec_tokens} x vocab {vocab}"
    );
    // The kernel accumulates into the counters, so they must start at zero on
    // every call — memset instead of a fresh allocation (this runs every
    // verify step; scratch buffers are allocate-once by convention).
    ctx.stream.memset_zeros(&mut scratch.accepted)?;
    ctx.stream.memset_zeros(&mut scratch.emitted)?;
    {
        let (dp, _g1) = draft_probs.device_ptr_mut(&ctx.stream);
        let (ids, _g2) = draft_token_ids.device_ptr(&ctx.stream);
        let (tp, _g3) = target_probs.device_ptr_mut(&ctx.stream);
        let (out, _g4) = output_token_ids.device_ptr_mut(&ctx.stream);
        let (acc, _g5) = scratch.accepted.device_ptr_mut(&ctx.stream);
        let (emi, _g6) = scratch.emitted.device_ptr_mut(&ctx.stream);
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
                i32::from(onehot_draft),
                seed,
                offset,
                crate::tensor::active_cu_stream(ctx),
            )
        };
        ensure!(
            err == 0,
            "chain speculative sampling failed with error {err}{}",
            crate::ops::ffi_exception_message(err)
        );
    }
    let accepted_view = scratch.accepted.slice(..batch);
    let emitted_view = scratch.emitted.slice(..batch);
    let acceptance_telemetry = ctx.stream.clone_dtoh(&accepted_view)?;
    let emitted = ctx.stream.clone_dtoh(&emitted_view)?;
    ctx.sync()?;
    Ok(SpecAcceptCounts {
        emitted,
        acceptance_telemetry,
    })
}

/// Per-row results of one chain-rejection call, named so the commit signal
/// and the telemetry cannot be swapped silently (they are both `Vec<i32>`).
pub struct SpecAcceptCounts {
    /// The accepted-prefix length per row — **the commit signal**: the row's
    /// output holds `emitted` accepted drafts plus one resampled/bonus token,
    /// then `-1` filler.
    pub emitted: Vec<i32>,
    /// FlashInfer's acceptance-rate telemetry: keeps counting hypothetical
    /// acceptances *past* the first rejection. Never commit on this.
    pub acceptance_telemetry: Vec<i32>,
}

/// Allocate-once device counters for [`gpu_spec_accept_into`] (it runs every
/// verify step; per-call `alloc_zeros` is against the scratch convention).
pub struct SpecAcceptScratch {
    accepted: CudaSlice<i32>,
    emitted: CudaSlice<i32>,
    max_batch: usize,
}

impl SpecAcceptScratch {
    pub fn new(ctx: &DeviceContext, max_batch: usize) -> Result<Self> {
        ensure!(max_batch > 0, "spec accept scratch requires max_batch > 0");
        Ok(Self {
            accepted: ctx.stream.alloc_zeros(max_batch)?,
            emitted: ctx.stream.alloc_zeros(max_batch)?,
            max_batch,
        })
    }
}

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaSlice;

    use super::super::sampling::test_support::{ARENA_ROWS, VOCAB, arena_with_rows, flat_row};
    use super::super::sampling::{BatchSamplingRow, BatchSamplingScratch};
    use super::{SpecAcceptScratch, gpu_spec_accept_into, gpu_verify_probs_into};
    use crate::tensor::DeviceContext;

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

        let mut scratch = SpecAcceptScratch::new(&ctx, 2).expect("scratch");
        let counts = gpu_spec_accept_into(
            &ctx,
            &mut draft_probs,
            &drafts_d,
            &mut target_d,
            &mut out,
            2,
            k,
            VOCAB,
            /*onehot_draft=*/ true,
            0x5eed,
            0,
            &mut scratch,
        )
        .expect("spec accept");
        let (accepted, emitted) = (counts.acceptance_telemetry, counts.emitted);
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
        // Telemetry keeps counting past the rejection: pos 1's draft (200) is
        // hypothetically accepted (target pos1 mass is 1.0 on 200), so the
        // acceptance counter reads exactly 1 while the commit signal reads 0.
        assert_eq!(accepted[1], 1, "row1 acceptance telemetry");
    }

    #[test]
    fn spec_accept_partial_acceptance_matches_the_rate_law() {
        // A onehot (greedy) proposer whose draft token holds 0.7 of the target
        // mass must be accepted with probability exactly min(1, 0.7) = 0.7. Run
        // one big batch of i.i.d. rows (per-row philox decorrelates them) and
        // check the empirical acceptance rate converges — the acceptance
        // probability is the whole correctness claim of rejection sampling, and
        // the corner-point tests above only exercise p ∈ {0, 1}.
        let ctx = DeviceContext::new().expect("create CUDA context");
        let vocab = 128usize;
        let batch = 512usize;
        let k = 1usize;
        let (tok_a, tok_b, tok_c) = (10usize, 20usize, 30usize);

        // Every row identical: pos0 target = {A: 0.7, B: 0.3}; the draft is A,
        // so on rejection the residual relu(target - onehot(A)) is {B: 1.0}.
        // pos1 (the bonus) is deterministic on C, so an accepted row emits
        // [A, C] and a rejected row emits [B, -1].
        let mut target = vec![0.0f32; batch * (k + 1) * vocab];
        for row in 0..batch {
            let base = row * (k + 1) * vocab;
            target[base + tok_a] = 0.7;
            target[base + tok_b] = 0.3;
            target[base + vocab + tok_c] = 1.0;
        }
        let mut target_d: CudaSlice<f32> =
            ctx.stream.clone_htod(&target).expect("target probs H2D");
        let drafts: Vec<i32> = vec![tok_a as i32; batch * k];
        let drafts_d = ctx.stream.clone_htod(&drafts).expect("draft ids H2D");
        let mut draft_probs: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(batch * k * vocab)
            .expect("draft probs");
        let mut out: CudaSlice<i32> = ctx.stream.alloc_zeros(batch * (k + 1)).expect("out");

        let mut scratch = SpecAcceptScratch::new(&ctx, batch).expect("scratch");
        let counts = gpu_spec_accept_into(
            &ctx,
            &mut draft_probs,
            &drafts_d,
            &mut target_d,
            &mut out,
            batch,
            k,
            vocab,
            /*onehot_draft=*/ true,
            0xC0FFEE,
            0,
            &mut scratch,
        )
        .expect("spec accept");
        let emitted = counts.emitted;
        let out_h = ctx.stream.clone_dtoh(&out).expect("D2H");
        ctx.sync().expect("sync");

        let mut accept_count = 0usize;
        for row in 0..batch {
            let o = &out_h[row * (k + 1)..row * (k + 1) + 2];
            if emitted[row] == 1 {
                assert_eq!(o, &[tok_a as i32, tok_c as i32], "accepted row {row}");
                accept_count += 1;
            } else {
                assert_eq!(emitted[row], 0, "row {row} emitted");
                assert_eq!(o, &[tok_b as i32, -1], "rejected row {row}");
            }
        }
        let rate = accept_count as f64 / batch as f64;
        // Bernoulli(0.7) over 512 draws: std ≈ 0.020, so 0.7 ± 0.07 is > 3σ.
        assert!(
            (0.63..=0.77).contains(&rate),
            "acceptance rate {rate} outside [0.63, 0.77] (expected 0.70)"
        );
    }

    #[test]
    fn spec_accept_explicit_draft_probs_corner_points() {
        // onehot_draft = false: the caller supplies the true proposal
        // distribution. Row 0: q puts 0.5 on the draft while the target puts
        // 1.0 there -> accept probability min(1, 1.0/0.5) = 1, certain accept;
        // the bonus position is deterministic on 777. Row 1: q puts 1.0 on the
        // draft while the target puts 0 there -> certain rejection; the
        // residual relu(target - draft) is exactly {555: 1.0}.
        let ctx = DeviceContext::new().expect("create CUDA context");
        let k = 1usize;
        let mut target = vec![0.0f32; 2 * (k + 1) * VOCAB];
        target[100] = 1.0; // row0 pos0 -> all mass on the draft
        target[VOCAB + 777] = 1.0; // row0 bonus -> 777
        let r1 = 2 * VOCAB;
        target[r1 + 555] = 1.0; // row1 pos0: no mass on the draft
        target[r1 + VOCAB + 300] = 1.0; // never reached
        let mut target_d: CudaSlice<f32> =
            ctx.stream.clone_htod(&target).expect("target probs H2D");
        let mut draft = vec![0.0f32; 2 * k * VOCAB];
        draft[100] = 0.5; // row0: q(draft) = 0.5
        draft[200] = 0.5;
        draft[VOCAB + 100] = 1.0; // row1: q(draft) = 1.0
        let mut draft_d: CudaSlice<f32> = ctx.stream.clone_htod(&draft).expect("draft probs H2D");
        let drafts_d = ctx.stream.clone_htod(&[100i32, 100]).expect("draft ids");
        let mut out: CudaSlice<i32> = ctx.stream.alloc_zeros(2 * (k + 1)).expect("out");
        let mut scratch = SpecAcceptScratch::new(&ctx, 2).expect("scratch");

        let counts = gpu_spec_accept_into(
            &ctx,
            &mut draft_d,
            &drafts_d,
            &mut target_d,
            &mut out,
            2,
            k,
            VOCAB,
            /*onehot_draft=*/ false,
            0xFACE,
            0,
            &mut scratch,
        )
        .expect("spec accept");
        let out_h = ctx.stream.clone_dtoh(&out).expect("D2H");
        ctx.sync().expect("sync");

        assert_eq!(&out_h[..2], &[100, 777], "row0 {:?}", &out_h[..2]);
        assert_eq!(counts.emitted[0], 1, "row0 emitted prefix");
        assert_eq!(&out_h[2..], &[555, -1], "row1 {:?}", &out_h[2..]);
        assert_eq!(counts.emitted[1], 0, "row1 emitted prefix");
    }
}
