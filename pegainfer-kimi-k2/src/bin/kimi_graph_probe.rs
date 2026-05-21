use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use cudarc::nccl::{
    ReduceOp,
    safe::{Comm, Id},
};
use half::bf16;
use pegainfer_kernels::{
    ops::{gemm_graphsafe_into_checked, scale_f32_in_place},
    tensor::{DeviceContext, DeviceMatrix, HiddenStates},
};
use serde::Serialize;

#[derive(Parser)]
#[command(about = "Kimi-K2 CUDA Graph capture probe for decode building blocks")]
struct Cli {
    #[arg(long, value_enum, default_value_t = Probe::LocalKernel)]
    probe: Probe,
    #[arg(long, default_value_t = 8)]
    world_size: usize,
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    #[arg(long, default_value_t = 7168)]
    hidden: usize,
    #[arg(long, default_value_t = 896)]
    gemm_out: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
enum Probe {
    LocalKernel,
    Gemm,
    NcclAllReduce,
    NcclReduceScatter,
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    probe: Probe,
    world_size: usize,
    batch_size: usize,
    hidden: usize,
    gemm_out: usize,
    capture_ok: bool,
    replay_ok: bool,
    elapsed_ms: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let started = Instant::now();
    match cli.probe {
        Probe::LocalKernel => run_local_kernel(cli.batch_size, cli.hidden)?,
        Probe::Gemm => run_gemm(cli.batch_size, cli.hidden, cli.gemm_out)?,
        Probe::NcclAllReduce => run_nccl_probe(cli.world_size, cli.batch_size, cli.hidden, false)?,
        Probe::NcclReduceScatter => {
            run_nccl_probe(cli.world_size, cli.batch_size, cli.hidden, true)?;
        }
    }
    let report = ProbeReport {
        probe: cli.probe,
        world_size: cli.world_size,
        batch_size: cli.batch_size,
        hidden: cli.hidden,
        gemm_out: cli.gemm_out,
        capture_ok: true,
        replay_ok: true,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_local_kernel(batch_size: usize, hidden: usize) -> Result<()> {
    let ctx = DeviceContext::new_with_device(0)?;
    let mut values: CudaSlice<f32> = ctx.stream.alloc_zeros(batch_size * hidden)?;
    capture_and_replay(&ctx, || {
        scale_f32_in_place(&ctx, &mut values, batch_size * hidden, 1.0)
    })
}

fn run_gemm(batch_size: usize, hidden: usize, out: usize) -> Result<()> {
    let ctx = DeviceContext::new_with_device(0)?;
    let weight = vec![bf16::ZERO; out * hidden];
    let weight = DeviceMatrix::from_host(&ctx, &weight, out, hidden)?;
    let x = HiddenStates::zeros(&ctx, hidden, batch_size)?;
    let mut y = HiddenStates::zeros(&ctx, out, batch_size)?;
    capture_and_replay(&ctx, || {
        gemm_graphsafe_into_checked(&ctx, &weight, &x, &mut y)
    })
}

fn run_nccl_probe(
    world_size: usize,
    batch_size: usize,
    hidden: usize,
    reduce_scatter: bool,
) -> Result<()> {
    if world_size == 0 {
        bail!("world_size must be positive");
    }
    let id = Id::new().map_err(|err| anyhow!("NCCL unique id creation failed: {err:?}"))?;
    let begin = Arc::new(Barrier::new(world_size));
    let enqueued = Arc::new(Barrier::new(world_size));
    let captured = Arc::new(Barrier::new(world_size));
    let launched = Arc::new(Barrier::new(world_size));
    let (tx, rx) = mpsc::channel();

    for rank in 0..world_size {
        let tx = tx.clone();
        let begin = Arc::clone(&begin);
        let enqueued = Arc::clone(&enqueued);
        let captured = Arc::clone(&captured);
        let launched = Arc::clone(&launched);
        thread::Builder::new()
            .name(format!("kimi-graph-probe-rank-{rank}"))
            .spawn(move || {
                let result = run_nccl_rank(
                    rank,
                    world_size,
                    batch_size,
                    hidden,
                    id,
                    reduce_scatter,
                    begin,
                    enqueued,
                    captured,
                    launched,
                );
                let _ = tx.send((rank, result));
            })
            .with_context(|| format!("spawn graph probe rank {rank}"))?;
    }
    drop(tx);

    let mut failures = Vec::new();
    for (rank, result) in rx {
        if let Err(err) = result {
            failures.push(format!("rank {rank}: {err:#}"));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        bail!("graph probe failed:\n{}", failures.join("\n"))
    }
}

#[allow(clippy::too_many_arguments)]
fn run_nccl_rank(
    rank: usize,
    world_size: usize,
    batch_size: usize,
    hidden: usize,
    id: Id,
    reduce_scatter: bool,
    begin: Arc<Barrier>,
    enqueued: Arc<Barrier>,
    captured: Arc<Barrier>,
    launched: Arc<Barrier>,
) -> Result<()> {
    let ctx = DeviceContext::new_with_device(rank)?;
    let comm = Comm::from_rank(ctx.stream.clone(), rank, world_size, id)
        .map_err(|err| anyhow!("NCCL comm init failed: {err:?}"))?;
    let elems = batch_size * hidden;
    let mut all_reduce_values: CudaSlice<f32> = ctx.stream.alloc_zeros(elems)?;
    let reduce_scatter_send: CudaSlice<f32> = ctx.stream.alloc_zeros(elems * world_size)?;
    let mut reduce_scatter_recv: CudaSlice<f32> = ctx.stream.alloc_zeros(elems)?;

    if reduce_scatter {
        comm.reduce_scatter(
            &reduce_scatter_send,
            &mut reduce_scatter_recv,
            &ReduceOp::Sum,
        )
        .map_err(|err| anyhow!("warmup reduce_scatter failed: {:?}", err.0))?;
    } else {
        comm.all_reduce_in_place(&mut all_reduce_values, &ReduceOp::Sum)
            .map_err(|err| anyhow!("warmup all_reduce failed: {:?}", err.0))?;
    }
    ctx.sync()?;

    begin.wait();
    ctx.stream
        .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .map_err(|err| anyhow!("rank {rank} begin_capture failed: {err}"))?;
    if reduce_scatter {
        comm.reduce_scatter(
            &reduce_scatter_send,
            &mut reduce_scatter_recv,
            &ReduceOp::Sum,
        )
        .map_err(|err| anyhow!("capture reduce_scatter enqueue failed: {:?}", err.0))?;
    } else {
        comm.all_reduce_in_place(&mut all_reduce_values, &ReduceOp::Sum)
            .map_err(|err| anyhow!("capture all_reduce enqueue failed: {:?}", err.0))?;
    }
    enqueued.wait();
    let graph = ctx
        .stream
        .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|err| anyhow!("rank {rank} end_capture failed: {err}"))?
        .ok_or_else(|| anyhow!("rank {rank} end_capture returned empty graph"))?;
    captured.wait();
    graph
        .launch()
        .map_err(|err| anyhow!("rank {rank} graph launch failed: {err}"))?;
    launched.wait();
    ctx.sync()
}

fn capture_and_replay<F>(ctx: &DeviceContext, f: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    ctx.stream
        .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .map_err(|err| anyhow!("begin_capture failed: {err}"))?;
    f()?;
    let graph = ctx
        .stream
        .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|err| anyhow!("end_capture failed: {err}"))?
        .ok_or_else(|| anyhow!("end_capture returned empty graph"))?;
    graph
        .launch()
        .map_err(|err| anyhow!("graph launch failed: {err}"))?;
    ctx.sync()
}
