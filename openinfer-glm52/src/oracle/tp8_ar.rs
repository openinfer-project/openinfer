//! TP8 attention-allreduce brick gate (M4 K1): 8 ranks push synthetic
//! partials through the LL two-shot allreduce (reduce-scatter + broadcast)
//! and every rank must get the bit-identical, exactly-predicted sum back.
//! No model weights involved.
//!
//! Covers: slot/parity addressing across steps (3 steps flip parity and
//! revisit a region holding stale packets — the tag discipline must reject
//! them), the want-mask (full-capacity launch with a device active count of
//! 3: pad rows must skip the wire and come back zero-filled over a NaN
//! canary), cross-rank bit-identity (the replicated-activation topology
//! relies on it), and a standalone timing loop (kill line 10 us/layer at 8 rows; the one-shot form measured 13.2
//! here — byte-bound at 7x payload egress — which is why the brick is
//! two-shot; the stream-enqueued standalone chain is an upper bound, since
//! production rides the whole-step graph).

use std::sync::{Arc, Barrier, Mutex};
use std::time::Instant;

use anyhow::{Result, ensure};
use half::bf16;
use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_kernels::ops::{
    GLM52_TP8_AR_CHUNK_PACKETS, GLM52_TP8_HIDDEN, GLM52_TP8_RANKS, GLM52_TP8_TOKENS,
    Glm52Tp8LlBuffer, glm52_moe_tp8_epoch_advance, glm52_tp8_ar_buffer_bytes, glm52_tp8_ar_launch,
};
use openinfer_kernels::tensor::DeviceContext;

const RANKS: usize = GLM52_TP8_RANKS;
const H: usize = GLM52_TP8_HIDDEN;
/// Enough slots for the full 78-layer chain the perf phase drives.
const SLOTS: usize = 78;
const PERF_WARM_STEPS: usize = 50;
const PERF_TIMED_STEPS: usize = 200;

/// Synthetic partial: exact in bf16 (4*(9k+2m)/16 with 9k+2m < 128), so the
/// fixed-order f32 accumulation reproduces on the host bit-for-bit.
fn partial_value(step: usize, rank: usize, row: usize, h: usize) -> f32 {
    let base = ((rank + 1 + step) % 9) * (row + 1) * 4 + (h % 13) * 8;
    base as f32 / 16.0
}

fn expected_row(step: usize, row: usize) -> Vec<bf16> {
    (0..H)
        .map(|h| {
            let mut acc = 0.0f32;
            for rank in 0..RANKS {
                acc += bf16::from_f32(partial_value(step, rank, row, h)).to_f32();
            }
            bf16::from_f32(acc)
        })
        .collect()
}

#[test]
#[ignore = "requires 8 NVLink GPUs (jz-38 8xH200)"]
fn tp8_ar_brick_gate() -> Result<()> {
    let vas: Arc<Mutex<Vec<Vec<u64>>>> = Arc::new(Mutex::new(vec![Vec::new(); RANKS]));
    let barrier = Arc::new(Barrier::new(RANKS));

    let handles: Vec<_> = (0..RANKS)
        .map(|rank| {
            let vas = Arc::clone(&vas);
            let barrier = Arc::clone(&barrier);
            std::thread::Builder::new()
                .name(format!("tp8-ar-gate-rank-{rank}"))
                .spawn(move || -> Result<(Vec<Vec<bf16>>, f64)> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let ordinals: Vec<usize> = (0..RANKS).collect();
                    let buf = Glm52Tp8LlBuffer::alloc(glm52_tp8_ar_buffer_bytes(SLOTS), &ordinals)?;
                    vas.lock().unwrap()[rank] = (0..RANKS).map(|i| buf.addr_for(i)).collect();
                    barrier.wait();
                    // peer_ar[dst] = the VA THIS rank uses to reach dst's
                    // buffer, pre-offset to this rank's src slot.
                    let peer_ar: [u64; RANKS] = {
                        let published = vas.lock().unwrap();
                        std::array::from_fn(|dst| {
                            published[dst][rank] + (rank * GLM52_TP8_AR_CHUNK_PACKETS * 16) as u64
                        })
                    };
                    let ar_local = buf.addr_for(rank);

                    let mut epoch_dev = ctx.stream.alloc_zeros::<u64>(1)?;
                    let mut active_dev = ctx.stream.alloc_zeros::<i32>(1)?;
                    let mut partial = ctx.stream.alloc_zeros::<bf16>(GLM52_TP8_TOKENS * H)?;
                    let mut out = ctx.stream.alloc_zeros::<bf16>(GLM52_TP8_TOKENS * H)?;

                    // --- correctness: 3 steps x 2 slots; step 3 re-enters
                    // step 1's parity region, which still holds tag-1
                    // packets — the recv must reject them and wait for
                    // tag 3. Step 2 runs the production partial shape:
                    // full-capacity launch, device want-mask active=3 —
                    // pad rows must skip the wire and zero-fill the NaN
                    // canary.
                    let mut results: Vec<Vec<bf16>> = Vec::new();
                    for step in 0..3 {
                        let active = if step == 1 { 3 } else { GLM52_TP8_TOKENS };
                        ctx.stream.memcpy_htod(&[active as i32], &mut active_dev)?;
                        let host: Vec<bf16> = (0..GLM52_TP8_TOKENS * H)
                            .map(|i| bf16::from_f32(partial_value(step, rank, i / H, i % H)))
                            .collect();
                        ctx.stream.memcpy_htod(&host, &mut partial)?;
                        ctx.stream.memcpy_htod(
                            &vec![bf16::from_f32(f32::NAN); GLM52_TP8_TOKENS * H],
                            &mut out,
                        )?;
                        glm52_moe_tp8_epoch_advance(&ctx, &mut epoch_dev)?;
                        for slot in [0usize, 1] {
                            glm52_tp8_ar_launch(
                                &ctx,
                                slot,
                                GLM52_TP8_TOKENS,
                                &partial,
                                &mut out,
                                ar_local,
                                peer_ar,
                                &epoch_dev,
                                Some(&active_dev),
                                rank,
                            )?;
                        }
                        let got = ctx.stream.clone_dtoh(&out)?;
                        ctx.stream.synchronize()?;
                        results.push(got);
                        // Keep epoch sequences aligned before the region is
                        // revisited with a new tag.
                        barrier.wait();
                    }

                    // --- perf: the 78-slot step chain captured into a CUDA
                    // graph and replayed — the production shape. A
                    // stream-launched loop measures HOST launch throughput
                    // instead (8 rank threads x 235 launches/step hammering
                    // the driver ≈ 10 us/layer of pure enqueue — the D7
                    // lesson).
                    ctx.stream.synchronize()?;
                    barrier.wait();
                    let mut graph = CudaGraphState::new();
                    for _ in 0..PERF_WARM_STEPS {
                        graph.run_or_capture(&ctx, || {
                            glm52_moe_tp8_epoch_advance(&ctx, &mut epoch_dev)?;
                            for slot in 0..SLOTS {
                                glm52_tp8_ar_launch(
                                    &ctx,
                                    slot,
                                    GLM52_TP8_TOKENS,
                                    &partial,
                                    &mut out,
                                    ar_local,
                                    peer_ar,
                                    &epoch_dev,
                                    Some(&active_dev),
                                    rank,
                                )?;
                            }
                            Ok(())
                        })?;
                    }
                    ctx.stream.synchronize()?;
                    barrier.wait();
                    let t0 = Instant::now();
                    for _ in 0..PERF_TIMED_STEPS {
                        graph.launch_captured(&ctx)?;
                    }
                    ctx.stream.synchronize()?;
                    let us_per_layer =
                        t0.elapsed().as_secs_f64() * 1e6 / (PERF_TIMED_STEPS * SLOTS) as f64;

                    // Streams idle before anyone's buffer unmaps on drop.
                    barrier.wait();
                    Ok((results, us_per_layer))
                })
                .expect("spawn tp8 ar gate rank thread")
        })
        .collect();

    let mut all: Vec<(Vec<Vec<bf16>>, f64)> = Vec::new();
    for h in handles {
        all.push(h.join().expect("tp8 ar gate rank thread panicked")?);
    }

    for (step, &active) in [GLM52_TP8_TOKENS, 3, GLM52_TP8_TOKENS].iter().enumerate() {
        let mut expected = Vec::with_capacity(GLM52_TP8_TOKENS * H);
        for row in 0..active {
            expected.extend(expected_row(step, row));
        }
        // Want-mask contract: pad rows come back zero-filled (NaN canary
        // overwritten), not stale and not reduced.
        expected.resize(GLM52_TP8_TOKENS * H, bf16::from_f32(0.0));
        for (rank, (results, _)) in all.iter().enumerate() {
            ensure!(
                results[step] == expected,
                "rank {rank} step {step} AR result differs from the exact host sum"
            );
        }
    }

    let timings: Vec<f64> = all.iter().map(|(_, us)| *us).collect();
    let worst = timings.iter().cloned().fold(0.0f64, f64::max);
    println!("tp8_ar standalone: per-rank us/layer = {timings:.3?}, worst {worst:.2}");
    ensure!(
        worst <= 10.0,
        "AR standalone {worst:.2} us/layer blows the kill line (R4 anchor 5.8)"
    );
    Ok(())
}
