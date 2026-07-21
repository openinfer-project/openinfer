//! M5b perf probe: the right-sized sparse MLA decode vs the FlashMLA sparse
//! decode it replaces, at the production shape (bucket-8, topk 2048, fp8
//! cache). The campaign account: FlashMLA measured 22.6 µs where the DRAM
//! floor is ~3.2 µs (M5 falsification numbers); the tilelang prototype hit
//! 10.5 µs at a 2× byte handicap (bf16 KV). Kill line: the new kernel over
//! 12 µs including combine → reopen the account before wiring anything.
//!
//! Timing = CUDA-graph capture of 78 launches (one per production layer)
//! cycling 8 distinct 172 MiB KV pools + index sets (reuse distance 1.4 GiB,
//! so L2 cannot carry a pool between same-pool launches), replayed 50×.
//! Stream-driven loops measure host launch walls; graphs measure the device
//! (the D7 lesson).

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_TOPK;
use openinfer_kernels::ops::Glm52FlashMlaSparseDecode;
use openinfer_kernels::ops::Glm52SparseMlaDecode;
use openinfer_kernels::ops::glm52_flashmla_sparse_decode_launch;
use openinfer_kernels::ops::glm52_flashmla_sparse_decode_metadata_launch;
use openinfer_kernels::ops::glm52_flashmla_sparse_decode_num_sm_parts;
use openinfer_kernels::ops::glm52_sparse_mla_decode_launch;
use openinfer_kernels::tensor::DeviceContext;

const BATCH: usize = 8;
const HEADS: usize = 8; // attention-TP shard
const TOPK: usize = GLM52_FLASHMLA_SPARSE_TOPK;
/// 4096 pages × 64 tok × 656 B = 172 MiB per pool — several× the H200 L2.
const NUM_BLOCKS: usize = 4096;
const POOLS: usize = 8;
const LAYERS: usize = 78;
const REPLAYS: usize = 50;

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

struct SharedBuffers {
    kv: Vec<CudaSlice<u8>>,
    indices: Vec<CudaSlice<i32>>,
    q: CudaSlice<bf16>,
    latent: CudaSlice<bf16>,
}

fn alloc_shared(ctx: &DeviceContext) -> Result<SharedBuffers> {
    let cache_len = NUM_BLOCKS * GLM52_FLASHMLA_SPARSE_PAGE_SIZE * 656;
    // Benign fp8 bytes: 0x30 is a small positive e4m3 everywhere it lands
    // (values and scales stay finite — NaN-free softmax).
    let kv_host = vec![0x30u8; cache_len];
    let mut kv = Vec::with_capacity(POOLS);
    for _ in 0..POOLS {
        // SAFETY: fully written by the memcpy below before any kernel reads.
        let mut pool = unsafe { ctx.stream.alloc::<u8>(cache_len) }?;
        ctx.stream.memcpy_htod(&kv_host, &mut pool)?;
        kv.push(pool);
    }

    let total_slots = NUM_BLOCKS * GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
    let mut rng = XorShift(0x5eed_5eed_5eed_5eed);
    let mut indices = Vec::with_capacity(POOLS);
    for _ in 0..POOLS {
        let host: Vec<i32> = (0..BATCH * TOPK)
            .map(|_| rng.below(total_slots) as i32)
            .collect();
        let mut dev = unsafe { ctx.stream.alloc::<i32>(host.len()) }?;
        ctx.stream.memcpy_htod(&host, &mut dev)?;
        indices.push(dev);
    }

    let q_len = BATCH * 64 * 576;
    let q_host: Vec<bf16> = (0..q_len)
        .map(|_| bf16::from_f32((rng.below(2000) as f32 - 1000.0) / 1000.0))
        .collect();
    let mut q = unsafe { ctx.stream.alloc::<bf16>(q_len) }?;
    ctx.stream.memcpy_htod(&q_host, &mut q)?;

    Ok(SharedBuffers {
        kv,
        indices,
        q,
        latent: ctx.stream.alloc_zeros::<bf16>(BATCH * 64 * 512)?,
    })
}

fn time_graph(
    ctx: &DeviceContext,
    label: &str,
    mut launch_layer: impl FnMut(usize) -> Result<()>,
) -> Result<f64> {
    let mut graph = CudaGraphState::new();
    for _ in 0..3 {
        graph.run_or_capture(ctx, || {
            for layer in 0..LAYERS {
                launch_layer(layer)?;
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
    println!("{label:>24}: {us_per_launch:7.2} us/launch");
    Ok(us_per_launch)
}

#[test]
#[ignore = "requires one sm90a GPU (H200); synthetic buffers, no checkpoint"]
fn sparse_mla_perf_probe() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let mut bufs = alloc_shared(&ctx)?;

    let contract = Glm52SparseMlaDecode {
        batch_size: BATCH,
        num_blocks: NUM_BLOCKS,
        topk: TOPK,
        heads: HEADS,
        sm_scale: 0.0625,
    };
    let mut o_part = ctx.stream.alloc_zeros::<f32>(contract.o_part_len())?;
    let mut ml_part = ctx.stream.alloc_zeros::<f32>(contract.ml_part_len())?;
    {
        let kv = &bufs.kv;
        let indices = &bufs.indices;
        let q = &bufs.q;
        let latent = &mut bufs.latent;
        time_graph(&ctx, "rightsize incl combine", |layer| {
            glm52_sparse_mla_decode_launch(
                &ctx,
                contract,
                q,
                &kv[layer % POOLS],
                &indices[layer % POOLS],
                &mut o_part,
                &mut ml_part,
                latent,
            )
        })?;
    }

    let num_sm_parts = glm52_flashmla_sparse_decode_num_sm_parts()?;
    let flash = Glm52FlashMlaSparseDecode {
        batch_size: BATCH,
        num_blocks: NUM_BLOCKS,
        topk: TOPK,
        num_sm_parts,
        sm_scale: 0.0625,
    };
    let mut sched = ctx
        .stream
        .alloc_zeros::<i32>(flash.tile_scheduler_metadata_len())?;
    let mut num_splits = ctx.stream.alloc_zeros::<i32>(flash.num_splits_len())?;
    glm52_flashmla_sparse_decode_metadata_launch(
        &ctx,
        BATCH,
        TOPK,
        num_sm_parts,
        &mut sched,
        &mut num_splits,
    )?;
    let mut lse = ctx.stream.alloc_zeros::<f32>(flash.lse_len())?;
    let mut lse_accum = ctx.stream.alloc_zeros::<f32>(flash.lse_accum_len())?;
    let mut o_accum = ctx.stream.alloc_zeros::<f32>(flash.o_accum_len())?;
    {
        let kv = &bufs.kv;
        let indices = &bufs.indices;
        let q = &bufs.q;
        let latent = &mut bufs.latent;
        time_graph(&ctx, "flashmla incl combine", |layer| {
            glm52_flashmla_sparse_decode_launch(
                &ctx,
                flash,
                q,
                &kv[layer % POOLS],
                &indices[layer % POOLS],
                &sched,
                &num_splits,
                latent,
                &mut lse,
                &mut lse_accum,
                &mut o_accum,
            )
        })?;
    }
    Ok(())
}
