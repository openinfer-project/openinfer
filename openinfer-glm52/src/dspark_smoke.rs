//! DSpark draft perf smoke: measures `propose()` wall time (one draft round =
//! 7 draft tokens) against the bandwidth-bound floor, on a single GPU.
//!
//! Uses the zero-weight synthetic model at the checkpoint's exact geometry —
//! GPU kernel time is value-independent, so no checkpoint or 8-GPU box is
//! needed. Requires ~11 GB free VRAM (weights 3.9 GB + embed/lm_head 3.8 GB +
//! bs=8 slot states 2.8 GB).
//!
//! Run (single GPU):
//! ```text
//! cargo test --release -p openinfer-glm52 --features glm52 --lib \
//!   dspark_propose_smoke -- --nocapture --ignored
//! ```

use std::time::Instant;

use anyhow::Result;
use half::bf16;

use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix};

use crate::config::{GLM52_HIDDEN, GLM52_VOCAB};
use crate::dspark::{
    GLM52_DSPARK_CONTEXT_DIM, GLM52_DSPARK_DRAFTS, Glm52DsparkModel, Glm52DsparkScratch,
    Glm52DsparkSlotState,
};

const WARMUP: usize = 5;
const ITERS: usize = 50;

/// One propose round per iteration, one captured context row per round —
/// the bs=1 steady-state shape (accept length only changes the cheap fc
/// input rows). Returns per-round times.
fn bench_propose(
    ctx: &DeviceContext,
    model: &Glm52DsparkModel,
    embed: &DeviceMatrix,
    lm_head: &DeviceMatrix,
    scratch: &mut Glm52DsparkScratch,
    batch: usize,
) -> Result<Vec<f64>> {
    let mut states = (0..batch)
        .map(|_| Glm52DsparkSlotState::new(ctx))
        .collect::<Result<Vec<_>>>()?;
    let captured: cudarc::driver::CudaSlice<bf16> =
        ctx.stream.alloc_zeros(GLM52_DSPARK_CONTEXT_DIM)?;

    let mut times_ms = Vec::with_capacity(ITERS);
    for round in 0..WARMUP + ITERS {
        for state in states.iter_mut() {
            state.append_captured_row(ctx, &captured, 0)?;
        }
        // Each round drains 1 pending row into committed, so the anchor sits
        // at `round + 1`.
        let anchors = vec![(7u32, round + 1); batch];
        ctx.sync()?;
        let started = Instant::now();
        let mut state_refs: Vec<&mut Glm52DsparkSlotState> = states.iter_mut().collect();
        let drafts = model.propose(ctx, embed, lm_head, &mut state_refs, &anchors, scratch)?;
        ctx.sync()?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        assert_eq!(drafts.len(), batch);
        assert!(drafts.iter().all(|d| d.len() == GLM52_DSPARK_DRAFTS));
        if round >= WARMUP {
            times_ms.push(elapsed);
        }
    }
    Ok(times_ms)
}

#[test]
#[ignore = "GPU perf smoke — needs a CUDA device with ~11 GB free VRAM"]
fn dspark_propose_smoke() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let model = Glm52DsparkModel::synthetic(&ctx)?;
    let zero_head = || -> Result<DeviceMatrix> {
        Ok(DeviceMatrix {
            data: ctx.stream.alloc_zeros(GLM52_VOCAB * GLM52_HIDDEN)?,
            rows: GLM52_VOCAB,
            cols: GLM52_HIDDEN,
        })
    };
    let embed = zero_head()?;
    let lm_head = zero_head()?;
    let mut scratch = Glm52DsparkScratch::new(&ctx)?;

    for batch in [1usize, 8] {
        let mut times = bench_propose(&ctx, &model, &embed, &lm_head, &mut scratch, batch)?;
        times.sort_by(f64::total_cmp);
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let (min, p50, max) = (times[0], times[times.len() / 2], times[times.len() - 1]);
        println!(
            "dspark propose bs={batch}: avg {avg:.3} ms, min {min:.3} ms, p50 {p50:.3} ms, \
             max {max:.3} ms ({ITERS} rounds, 7 drafts/round)"
        );
    }
    Ok(())
}
