//! M5 split-KV-read feasibility probe: the whole campaign account rests on
//! one unverified assumption — that `-1`-diluted top-k index lists make the
//! sparse FlashMLA kernel proportionally cheaper (invalid entries load page 0
//! repeatedly, absorbed by L2; softmax masks them to -inf). If the walk over
//! the padded index blocks is the bottleneck instead of the KV bytes, the
//! account collapses. Synthetic buffers only — no checkpoint, single GPU.
//!
//! Variants (batch 8, topk 2048, 172 MiB KV pool per "layer" — L2-busting):
//!   full             2048 valid slots/row across the whole pool
//!   eighth-segment    256 valid slots/row confined to the row's 1/8 slot
//!                     range + 1792 × -1 (the M5 shape: dilution + locality)
//!   eighth-scattered  256 valid slots/row across the whole pool + -1 pad
//!                     (isolates dilution from locality)
//!   dregs               64 valid slots/row + 1984 × -1 (extreme dilution —
//!                     the index-walk floor)
//!
//! Timing = CUDA-graph capture of 78 launches (one per production layer)
//! cycling 8 distinct KV pools + index sets (reuse distance 1.4 GiB, so L2
//! cannot carry a pool between same-pool launches), replayed 50×. The D7
//! lesson applies: stream-driven loops measure host launch walls, graphs
//! measure the device.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_HEADS, GLM52_FLASHMLA_SPARSE_PAGE_SIZE, GLM52_FLASHMLA_SPARSE_TOPK,
    Glm52FlashMlaSparseDecode, glm52_flashmla_sparse_decode_launch,
    glm52_flashmla_sparse_decode_metadata_launch, glm52_flashmla_sparse_decode_num_sm_parts,
};
use openinfer_kernels::tensor::DeviceContext;

const BATCH: usize = 8;
const TOPK: usize = GLM52_FLASHMLA_SPARSE_TOPK;
/// 4096 pages × 64 tok × 656 B = 172 MiB per pool — several× the H200 L2.
const NUM_BLOCKS: usize = 4096;
const POOLS: usize = 8;
const LAYERS: usize = 78;
const REPLAYS: usize = 50;

/// Deterministic xorshift so runs are comparable without pulling in rand.
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct ProbeBuffers {
    kv: Vec<CudaSlice<u8>>,
    indices: Vec<CudaSlice<i32>>,
    q: CudaSlice<bf16>,
    sched: CudaSlice<i32>,
    num_splits: CudaSlice<i32>,
    latent: CudaSlice<bf16>,
    lse: CudaSlice<f32>,
    lse_accum: CudaSlice<f32>,
    o_accum: CudaSlice<f32>,
    contract: Glm52FlashMlaSparseDecode,
}

/// `valid_per_row` slots per row (rest -1); `segment` confines row r's slots
/// to the pool's r-th eighth (the M5 locality shape).
fn build_indices(rng: &mut XorShift, valid_per_row: usize, segment: bool) -> Vec<i32> {
    build_indices_with_topk(rng, TOPK, valid_per_row, segment)
}

fn build_indices_with_topk(
    rng: &mut XorShift,
    topk: usize,
    valid_per_row: usize,
    segment: bool,
) -> Vec<i32> {
    let total_slots = NUM_BLOCKS * GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
    let mut host = vec![-1i32; BATCH * topk];
    for row in 0..BATCH {
        let (base, range) = if segment {
            (row * total_slots / 8, total_slots / 8)
        } else {
            (0, total_slots)
        };
        for k in 0..valid_per_row {
            host[row * topk + k] = (base + rng.below(range)) as i32;
        }
    }
    host
}

fn alloc_probe(ctx: &DeviceContext) -> Result<ProbeBuffers> {
    alloc_probe_with_topk(ctx, TOPK)
}

fn alloc_probe_with_topk(ctx: &DeviceContext, topk: usize) -> Result<ProbeBuffers> {
    let num_sm_parts = glm52_flashmla_sparse_decode_num_sm_parts()?;
    let contract = Glm52FlashMlaSparseDecode {
        batch_size: BATCH,
        num_blocks: NUM_BLOCKS,
        topk,
        num_sm_parts,
        sm_scale: 1.0 / (576f32).sqrt(),
    };
    contract.validate()?;

    // Benign fp8 bytes: 0x30 is a small positive e4m3 everywhere it lands
    // (values and scales stay finite — NaN-free softmax).
    let kv_host = vec![0x30u8; contract.packed_kv_cache_len()];
    let mut kv = Vec::with_capacity(POOLS);
    for _ in 0..POOLS {
        // SAFETY: fully written by the memcpy below before any kernel reads.
        let mut pool = unsafe { ctx.stream.alloc::<u8>(contract.packed_kv_cache_len()) }?;
        ctx.stream.memcpy_htod(&kv_host, &mut pool)?;
        kv.push(pool);
    }

    let mut rng = XorShift(0x5eed_5eed_5eed_5eed);
    let q_host: Vec<bf16> = (0..contract.q_len())
        .map(|_| bf16::from_f32((rng.below(2000) as f32 - 1000.0) / 1000.0))
        .collect();
    let mut q = unsafe { ctx.stream.alloc::<bf16>(contract.q_len()) }?;
    ctx.stream.memcpy_htod(&q_host, &mut q)?;

    let mut sched = ctx
        .stream
        .alloc_zeros::<i32>(contract.tile_scheduler_metadata_len())?;
    let mut num_splits = ctx.stream.alloc_zeros::<i32>(contract.num_splits_len())?;
    glm52_flashmla_sparse_decode_metadata_launch(
        ctx,
        BATCH,
        topk,
        num_sm_parts,
        &mut sched,
        &mut num_splits,
    )?;

    Ok(ProbeBuffers {
        kv,
        indices: Vec::new(),
        q,
        sched,
        num_splits,
        latent: ctx.stream.alloc_zeros::<bf16>(contract.latent_len())?,
        lse: ctx.stream.alloc_zeros::<f32>(contract.lse_len())?,
        lse_accum: ctx.stream.alloc_zeros::<f32>(contract.lse_accum_len())?,
        o_accum: ctx.stream.alloc_zeros::<f32>(contract.o_accum_len())?,
        contract,
    })
}

fn stage_indices(
    ctx: &DeviceContext,
    bufs: &mut ProbeBuffers,
    valid_per_row: usize,
    segment: bool,
) -> Result<()> {
    let mut rng = XorShift(0x1dee_c0de_0000_0001 ^ valid_per_row as u64);
    bufs.indices.clear();
    for _ in 0..POOLS {
        let host = build_indices(&mut rng, valid_per_row, segment);
        let mut dev = unsafe { ctx.stream.alloc::<i32>(host.len()) }?;
        ctx.stream.memcpy_htod(&host, &mut dev)?;
        bufs.indices.push(dev);
    }
    Ok(())
}

fn stage_indices_with_topk(
    ctx: &DeviceContext,
    bufs: &mut ProbeBuffers,
    topk: usize,
    valid_per_row: usize,
) -> Result<()> {
    let mut rng = XorShift(0x1dee_c0de_0000_0001 ^ (topk as u64) << 32 ^ valid_per_row as u64);
    bufs.indices.clear();
    for _ in 0..POOLS {
        let host = build_indices_with_topk(&mut rng, topk, valid_per_row, false);
        let mut dev = unsafe { ctx.stream.alloc::<i32>(host.len()) }?;
        ctx.stream.memcpy_htod(&host, &mut dev)?;
        bufs.indices.push(dev);
    }
    Ok(())
}

fn time_variant(ctx: &DeviceContext, bufs: &mut ProbeBuffers, label: &str) -> Result<f64> {
    let mut graph = CudaGraphState::new();
    // Warm (captures on the first call), then timed replays.
    for _ in 0..3 {
        let contract = bufs.contract;
        let kv = &bufs.kv;
        let indices = &bufs.indices;
        let q = &bufs.q;
        let sched = &bufs.sched;
        let num_splits = &bufs.num_splits;
        let latent = &mut bufs.latent;
        let lse = &mut bufs.lse;
        let lse_accum = &mut bufs.lse_accum;
        let o_accum = &mut bufs.o_accum;
        graph.run_or_capture(ctx, || {
            for layer in 0..LAYERS {
                glm52_flashmla_sparse_decode_launch(
                    ctx,
                    contract,
                    q,
                    &kv[layer % POOLS],
                    &indices[layer % POOLS],
                    sched,
                    num_splits,
                    latent,
                    lse,
                    lse_accum,
                    o_accum,
                )?;
            }
            Ok(())
        })?;
    }
    ctx.stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..REPLAYS {
        graph.launch_captured(ctx)?;
    }
    ctx.stream.synchronize()?;
    let us_per_launch = t0.elapsed().as_secs_f64() * 1e6 / (REPLAYS * LAYERS) as f64;
    println!("{label:>18}: {us_per_launch:7.2} us/launch");
    Ok(us_per_launch)
}

/// Second question, after dilution was falsified (2026-07-08, idle H200:
/// dregs-64 still 0.92x of full — the kernel does fixed work per padded index
/// block): does shrinking the ACTUAL topk parameter scale? If time tracks
/// topk with a small floor, M5 pivots to a strided 1/8 partition of the
/// top-2048 list compacted into a topk=256 launch (KV is replicated, so any
/// disjoint partition is valid for the lse-merge; stride-8 by rank id is
/// perfectly balanced and needs no device staging at all).
#[test]
#[ignore = "requires one sm90a GPU (H200); synthetic buffers, no checkpoint"]
fn splitkv_topk_scaling_probe() -> Result<()> {
    let ctx = DeviceContext::new_with_device(0)?;
    let mut results = Vec::new();
    for topk in [2048usize, 1024, 512, 256, 128] {
        let mut bufs = alloc_probe_with_topk(&ctx, topk)?;
        stage_indices_with_topk(&ctx, &mut bufs, topk, topk)?;
        let us = time_variant(&ctx, &mut bufs, &format!("topk-{topk}"))?;
        results.push((topk, us));
    }
    let full = results[0].1;
    for &(topk, us) in &results[1..] {
        println!("topk {topk}: {:.2}x of topk-2048", us / full);
    }
    let quarter = results
        .iter()
        .find(|(t, _)| *t == 256)
        .expect("256 in the sweep")
        .1;
    anyhow::ensure!(
        quarter < 0.35 * full,
        "M5 compaction route broken too: topk-256 launch {quarter:.2} us is not well below          topk-2048 {full:.2} us — the kernel does not scale with topk"
    );
    Ok(())
}

#[test]
#[ignore = "requires one sm90a GPU (H200); synthetic buffers, no checkpoint"]
fn splitkv_dilution_probe() -> Result<()> {
    let ctx = DeviceContext::new_with_device(0)?;
    let mut bufs = alloc_probe(&ctx)?;

    stage_indices(&ctx, &mut bufs, TOPK, false)?;
    let full = time_variant(&ctx, &mut bufs, "full-2048")?;

    stage_indices(&ctx, &mut bufs, TOPK / 8, true)?;
    let eighth_seg = time_variant(&ctx, &mut bufs, "eighth-segment")?;

    stage_indices(&ctx, &mut bufs, TOPK / 8, false)?;
    let eighth_scat = time_variant(&ctx, &mut bufs, "eighth-scattered")?;

    stage_indices(&ctx, &mut bufs, 64, false)?;
    let dregs = time_variant(&ctx, &mut bufs, "dregs-64")?;

    println!(
        "ratios: eighth-segment {:.2}x, eighth-scattered {:.2}x, dregs {:.2}x (of full)",
        eighth_seg / full,
        eighth_scat / full,
        dregs / full
    );
    // Feasibility bar, not a regression gate: the M5 account needs the
    // diluted list to cost well under half of the full walk. If this fires,
    // the index walk (not KV bytes) owns the kernel and M5 is dead as
    // designed.
    anyhow::ensure!(
        eighth_seg < 0.5 * full,
        "M5 account broken: 1/8-diluted launch {eighth_seg:.2} us is not well below full \
         {full:.2} us — the padded index walk dominates, not KV bytes"
    );
    let _ = GLM52_FLASHMLA_SPARSE_HEADS;
    Ok(())
}
