//! Golden tests for the batched device logprobs reduction (#719).
//!
//! The device path (`logprobs_lse_bf16_into`, `logprobs_gather_rows_bf16_into`,
//! and `logprobs_topk_bf16_into`) must reproduce
//! `openinfer_sample::token_logprob_from_row` semantics over the same bf16
//! logits: identical top-k token ids in (value desc, id asc) order, and
//! logprobs within a small tolerance — the host accumulates `exp` in f64
//! sequentially while the device block-reduces f64 partials, so the LSE can
//! differ by a few fp32 ULPs. Requires a GPU.

use half::bf16;
use openinfer_kernels::ops::{
    logprobs_gather_rows_bf16_into, logprobs_lse_bf16_into, logprobs_topk_bf16_into,
};
use openinfer_kernels::tensor::{DeviceContext, HiddenStates};
use openinfer_sample::token_logprob_from_row;

/// Logprob tolerance: covers fp32-ULP LSE divergence between the sequential
/// host sum and the device tree reduction (values are O(10); fp32 eps ~ 1e-7).
const LP_TOLERANCE: f32 = 1e-4;

/// Deterministic xorshift64 PRNG so the test needs no extra deps.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        lo + unit * (hi - lo)
    }
}

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

/// One scored job: (arena row, picked token, top_k).
struct Job {
    row: u32,
    picked: u32,
    top_k: usize,
}

/// Run the device batch exactly the way the executor does and assemble
/// host-side `(picked logprob, top logprobs)` results.
fn run_device_batch(
    ctx: &DeviceContext,
    hs: &HiddenStates,
    jobs: &[Job],
) -> Vec<(f32, Vec<(u32, f32)>)> {
    let m = jobs.len();
    let vocab = hs.hidden_dim;
    let k_dev = jobs.iter().map(|j| j.top_k).max().unwrap_or(0).min(vocab);

    let rows: Vec<u32> = jobs.iter().map(|j| j.row).collect();
    let picked: Vec<u32> = jobs.iter().map(|j| j.picked).collect();
    let mut row_indices = ctx.stream.alloc_zeros::<u32>(m).unwrap();
    let mut picked_dev = ctx.stream.alloc_zeros::<u32>(m).unwrap();
    ctx.stream.memcpy_htod(&rows, &mut row_indices).unwrap();
    ctx.stream.memcpy_htod(&picked, &mut picked_dev).unwrap();

    let mut lse = ctx.stream.alloc_zeros::<f32>(m).unwrap();
    let mut picked_lp = ctx.stream.alloc_zeros::<f32>(m).unwrap();
    logprobs_lse_bf16_into(
        ctx,
        hs,
        Some(&row_indices),
        &picked_dev,
        m,
        &mut lse,
        &mut picked_lp,
    )
    .unwrap();

    let (values, indices) = if k_dev > 0 {
        let mut gathered = ctx.stream.alloc_zeros::<bf16>(m * vocab).unwrap();
        logprobs_gather_rows_bf16_into(ctx, hs, &row_indices, m, &mut gathered).unwrap();
        let mut values = ctx.stream.alloc_zeros::<bf16>(m * k_dev).unwrap();
        let mut indices = ctx.stream.alloc_zeros::<i32>(m * k_dev).unwrap();
        let ok = logprobs_topk_bf16_into(
            ctx,
            &gathered,
            m,
            vocab,
            k_dev,
            &mut values,
            &mut indices,
        )
        .unwrap();
        assert!(ok, "FilteredTopK unsupported on this GPU");
        (values, indices)
    } else {
        (
            ctx.stream.alloc_zeros::<bf16>(1).unwrap(),
            ctx.stream.alloc_zeros::<i32>(1).unwrap(),
        )
    };

    let lse_h = ctx.stream.clone_dtoh(&lse).unwrap();
    let picked_lp_h = ctx.stream.clone_dtoh(&picked_lp).unwrap();
    let values_h = ctx.stream.clone_dtoh(&values).unwrap();
    let indices_h = ctx.stream.clone_dtoh(&indices).unwrap();
    ctx.sync().unwrap();

    jobs.iter()
        .enumerate()
        .map(|(out_idx, job)| {
            let lse = lse_h[out_idx];
            let top = if job.top_k > 0 {
                let base = out_idx * k_dev;
                (0..job.top_k)
                    .map(|t| (indices_h[base + t] as u32, values_h[base + t].to_f32() - lse))
                    .collect()
            } else {
                Vec::new()
            };
            (picked_lp_h[out_idx], top)
        })
        .collect()
}

/// Host reference for one arena row, reading the bf16 values back so both
/// sides see bit-identical inputs.
fn host_reference(
    ctx: &DeviceContext,
    hs: &HiddenStates,
    jobs: &[Job],
) -> Vec<(f32, Vec<(u32, f32)>)> {
    let all = ctx.stream.clone_dtoh(&hs.data).unwrap();
    ctx.sync().unwrap();
    let vocab = hs.hidden_dim;
    jobs.iter()
        .map(|job| {
            let row: Vec<f32> = all[job.row as usize * vocab..(job.row as usize + 1) * vocab]
                .iter()
                .map(|x| x.to_f32())
                .collect();
            let r = token_logprob_from_row(&row, job.picked, job.top_k).unwrap();
            (r.logprob, r.top_logprobs)
        })
        .collect()
}

fn assert_matches(device: &[(f32, Vec<(u32, f32)>)], reference: &[(f32, Vec<(u32, f32)>)]) {
    assert_eq!(device.len(), reference.len());
    for (i, (d, r)) in device.iter().zip(reference).enumerate() {
        assert!(
            (d.0 - r.0).abs() <= LP_TOLERANCE,
            "job {i}: picked logprob {} vs reference {}",
            d.0,
            r.0
        );
        let d_ids: Vec<u32> = d.1.iter().map(|t| t.0).collect();
        let r_ids: Vec<u32> = r.1.iter().map(|t| t.0).collect();
        assert_eq!(d_ids, r_ids, "job {i}: top-k token ids differ");
        for (t, (dt, rt)) in d.1.iter().zip(&r.1).enumerate() {
            assert!(
                (dt.1 - rt.1).abs() <= LP_TOLERANCE,
                "job {i} top {t}: logprob {} vs reference {}",
                dt.1,
                rt.1
            );
        }
    }
}

#[test]
fn golden_random_rows_match_host_reference() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 151_936;
    let mut rng = Rng(0x5EED_1234_5678_9ABC);
    let rows: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..vocab).map(|_| rng.next_f32(-20.0, 20.0)).collect())
        .collect();
    let arena = make_arena(&ctx, &rows);

    let jobs = vec![
        Job { row: 0, picked: 1234, top_k: 5 },
        Job { row: 3, picked: 0, top_k: 1 },
        Job { row: 7, picked: vocab as u32 - 1, top_k: 20 },
        Job { row: 5, picked: 999, top_k: 0 },
    ];
    let device = run_device_batch(&ctx, &arena, &jobs);
    let reference = host_reference(&ctx, &arena, &jobs);
    assert_matches(&device, &reference);
}

#[test]
fn golden_tie_heavy_rows_keep_index_order() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 4096;
    // Only a handful of distinct values: massive ties, including at the top-k
    // boundary — the device must keep (value desc, token id asc) order and the
    // smallest-id tie-break selection, exactly like the host insertion pass.
    let mut rng = Rng(0xDEAD_BEEF_0BAD_F00D);
    let rows: Vec<Vec<f32>> = (0..4)
        .map(|_| {
            (0..vocab)
                .map(|_| [3.0f32, 1.5, 0.0, -2.5][(rng.next_f32(0.0, 4.0) as usize) % 4])
                .collect()
        })
        .collect();
    let arena = make_arena(&ctx, &rows);

    let jobs = vec![
        Job { row: 0, picked: 4000, top_k: 3 },
        Job { row: 1, picked: 7, top_k: 100 },
        Job { row: 2, picked: 1, top_k: 1 },
        Job { row: 3, picked: 2048, top_k: 2048 },
    ];
    let device = run_device_batch(&ctx, &arena, &jobs);
    let reference = host_reference(&ctx, &arena, &jobs);
    assert_matches(&device, &reference);
}

#[test]
fn golden_chosen_only_rows_match() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 32_000;
    let mut rng = Rng(0x0BAD_5EED_CAFE_F00D);
    let rows: Vec<Vec<f32>> = (0..3)
        .map(|_| (0..vocab).map(|_| rng.next_f32(-10.0, 10.0)).collect())
        .collect();
    let arena = make_arena(&ctx, &rows);

    // top_k = 0: chosen-token logprob only, empty alternatives.
    let jobs: Vec<Job> = (0..3)
        .map(|i| Job {
            row: i,
            picked: (i * 977 + 13) % vocab as u32,
            top_k: 0,
        })
        .collect();
    let device = run_device_batch(&ctx, &arena, &jobs);
    let reference = host_reference(&ctx, &arena, &jobs);
    assert_matches(&device, &reference);
}
