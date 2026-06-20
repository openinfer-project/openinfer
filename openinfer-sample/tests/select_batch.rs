//! Model-free integration tests for the unified batched sampler.
//!
//! Synthetic logits arenas (no model weights): each test pins down one behaviour
//! the crate adds on top of the raw kernels — greedy/non-greedy routing, per-row
//! token placement, scratch reuse, and the capacity invariant. Raw kernel
//! correctness (top-k/top-p/temperature, philox determinism) is owned by
//! `openinfer-kernels`; here we check that the door wires rows to the right path
//! and lands each token at the right index. Requires a GPU.

use std::collections::HashSet;

use half::bf16;
use openinfer_kernels::tensor::{DeviceContext, HiddenStates};
use openinfer_sample::{SampleScratch, SamplingParams, select_batch};

/// Build a logits arena from per-row logits (row `i` is token `i`'s logits).
fn make_arena(ctx: &DeviceContext, rows: &[Vec<f32>]) -> HiddenStates {
    let vocab = rows[0].len();
    assert!(rows.iter().all(|r| r.len() == vocab), "ragged rows");
    let mut hs = HiddenStates::zeros(ctx, vocab, rows.len()).unwrap();
    let flat: Vec<bf16> = rows
        .iter()
        .flat_map(|r| r.iter().map(|&x| bf16::from_f32(x)))
        .collect();
    ctx.stream.memcpy_htod(&flat, &mut hs.data).unwrap();
    ctx.sync().unwrap();
    hs
}

fn refs(params: &[SamplingParams]) -> Vec<&SamplingParams> {
    params.iter().collect()
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: -1,
        top_p: 1.0,
        ignore_eos: false,
    }
}

fn sampling(temperature: f32, top_k: i32, top_p: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k,
        top_p,
        ignore_eos: false,
    }
}

/// A one-hot-ish logits row: a single peak above a flat floor.
fn peak_row(vocab: usize, peak: usize, height: f32) -> Vec<f32> {
    let mut r = vec![0.0; vocab];
    r[peak] = height;
    r
}

#[test]
fn greedy_batch_picks_each_rows_argmax() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 16;
    let peaks = [3usize, 7, 0, 15];
    let rows: Vec<Vec<f32>> = peaks.iter().map(|&p| peak_row(vocab, p, 10.0)).collect();
    let arena = make_arena(&ctx, &rows);
    let params = vec![greedy(); peaks.len()];
    let mut scratch = SampleScratch::new(&ctx, vocab, peaks.len()).unwrap();

    let tokens = select_batch(&ctx, &arena, &refs(&params), 0, &mut scratch).unwrap();

    assert_eq!(tokens, vec![3, 7, 0, 15]);
}

#[test]
fn top_k_one_routes_through_greedy_path() {
    // High temperature but top_k == 1 => is_greedy() => argmax, not sampling.
    let ctx = DeviceContext::new().unwrap();
    let vocab = 16;
    let arena = make_arena(&ctx, &[peak_row(vocab, 5, 10.0)]);
    let params = vec![sampling(1.0, 1, 1.0)];
    let mut scratch = SampleScratch::new(&ctx, vocab, 1).unwrap();

    let tokens = select_batch(&ctx, &arena, &refs(&params), 123, &mut scratch).unwrap();

    assert_eq!(tokens, vec![5]);
}

#[test]
fn mixed_batch_routes_and_places_each_row() {
    // Row 0 greedy, row 1 sampled, row 2 greedy. Row 1's top_p (0.5) sits above
    // the 1/vocab argmax floor so it genuinely takes the sampler path; its dist
    // is near one-hot (peak prob ~1) so the nucleus is the peak and the sample
    // collapses there. Verifies both paths run in one call and each token lands
    // at its own row index.
    let ctx = DeviceContext::new().unwrap();
    let vocab = 16;
    let rows = vec![
        peak_row(vocab, 2, 12.0),
        peak_row(vocab, 9, 12.0),
        peak_row(vocab, 4, 12.0),
    ];
    let arena = make_arena(&ctx, &rows);
    let params = vec![greedy(), sampling(1.0, -1, 0.5), greedy()];
    let mut scratch = SampleScratch::new(&ctx, vocab, 3).unwrap();

    let tokens = select_batch(&ctx, &arena, &refs(&params), 7, &mut scratch).unwrap();

    assert_eq!(tokens, vec![2, 9, 4]);
}

#[test]
fn sampling_is_seed_deterministic_and_actually_samples() {
    // Uniform logits: argmax would always pick index 0, so seeing >1 token over
    // many seeds proves the non-greedy path samples; equal seeds proves it is
    // reproducible. Also exercises scratch reuse across many calls.
    let ctx = DeviceContext::new().unwrap();
    let vocab = 8;
    let arena = make_arena(&ctx, &[vec![0.0; vocab]]);
    let params = vec![sampling(1.0, -1, 1.0)];
    let mut scratch = SampleScratch::new(&ctx, vocab, 1).unwrap();

    let a = select_batch(&ctx, &arena, &refs(&params), 42, &mut scratch).unwrap();
    let b = select_batch(&ctx, &arena, &refs(&params), 42, &mut scratch).unwrap();
    assert_eq!(a, b, "same seed must be deterministic");

    let mut seen = HashSet::new();
    for s in 0..64u64 {
        seen.insert(select_batch(&ctx, &arena, &refs(&params), s, &mut scratch).unwrap()[0]);
    }
    assert!(
        seen.len() > 1,
        "uniform sampling should explore multiple tokens, saw {seen:?}"
    );
}

#[test]
fn tiny_top_p_routes_to_argmax_even_under_bf16_ties() {
    // Regression for the per-step-philox migration (no per-request RNG). A tiny
    // top_p must collapse to the same token greedy picks. The hard case is a
    // bf16 tie at the top: argmax breaks it by lowest index, but the rejection
    // sampler would pick either tied token depending on the seed (the old
    // per-request seed masked this; a per-step seed exposes it). Because the
    // softmax max is always >= 1/vocab, top_p <= 1/vocab makes the nucleus a
    // single token, so the row routes to the deterministic argmax path and
    // matches greedy for every seed. Large vocab matches the model's
    // OnlineSoftmax regime.
    let ctx = DeviceContext::new().unwrap();
    let vocab = 151_936;
    let lo = 879usize;
    let hi = 6941usize;
    let mut row = vec![0.0f32; vocab];
    row[lo] = 8.0; // bf16-exact, identical to `hi` -> a true top tie
    row[hi] = 8.0;
    let arena = make_arena(&ctx, &[row]);
    let mut scratch = SampleScratch::new(&ctx, vocab, 1).unwrap();

    let greedy_tok = select_batch(&ctx, &arena, &[&greedy()], 0, &mut scratch).unwrap();
    assert_eq!(
        greedy_tok,
        vec![lo as u32],
        "argmax breaks the tie by lowest index"
    );

    let tiny = sampling(1.0, -1, 1e-6); // 1e-6 < 1/vocab (~6.6e-6)
    for s in 0..64u64 {
        assert_eq!(
            select_batch(&ctx, &arena, &[&tiny], s, &mut scratch).unwrap(),
            greedy_tok,
            "seed {s}: tiny top_p must match greedy, not sample the tied peer"
        );
    }
}

#[test]
fn batch_larger_than_scratch_is_rejected() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 8;
    let arena = make_arena(&ctx, &[peak_row(vocab, 1, 5.0), peak_row(vocab, 2, 5.0)]);
    let params = vec![greedy(); 2];
    let mut scratch = SampleScratch::new(&ctx, vocab, 1).unwrap();

    assert!(select_batch(&ctx, &arena, &refs(&params), 0, &mut scratch).is_err());
}
