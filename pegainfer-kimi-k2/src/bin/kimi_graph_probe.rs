use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use cudarc::driver::sys::CUevent_flags_enum::CU_EVENT_DEFAULT;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use cudarc::driver::{CudaGraph, CudaSlice};
use cudarc::nccl::{
    ReduceOp,
    safe::{Comm, Id},
};
use half::bf16;
use pegainfer_kernels::{
    ops::{gemm_graphsafe_into_checked, repeat_f32_for_reduce_scatter_into, scale_f32_in_place},
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
    #[arg(long, default_value_t = 100)]
    replay_iters: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
enum Probe {
    LocalKernel,
    Gemm,
    NcclAllReduce,
    NcclReduceScatter,
    NcclTwoStreamOverlap,
    RoutedBridgeCompare,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    overlap: Option<OverlapReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bridge: Option<BridgeReport>,
}

#[derive(Debug, Serialize)]
struct OverlapReport {
    replay_iters: usize,
    sequential_avg_us_max_rank: f32,
    overlap_avg_us_max_rank: f32,
    speedup_max_rank: f32,
    sequential_avg_us_by_rank: Vec<f32>,
    overlap_avg_us_by_rank: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct BridgeReport {
    replay_iters: usize,
    all_reduce_avg_us_max_rank: f32,
    repeat_reduce_scatter_avg_us_max_rank: f32,
    repeat_reduce_scatter_over_all_reduce: f32,
    all_reduce_avg_us_by_rank: Vec<f32>,
    repeat_reduce_scatter_avg_us_by_rank: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct RankOverlapTiming {
    sequential_avg_us: f32,
    overlap_avg_us: f32,
}

#[derive(Clone, Copy, Debug)]
struct RankBridgeTiming {
    all_reduce_avg_us: f32,
    repeat_reduce_scatter_avg_us: f32,
}

fn run_routed_bridge_compare_probe(
    world_size: usize,
    batch_size: usize,
    hidden: usize,
    replay_iters: usize,
) -> Result<BridgeReport> {
    if world_size == 0 {
        bail!("world_size must be positive");
    }
    if replay_iters == 0 {
        bail!("replay_iters must be positive");
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
            .name(format!("kimi-graph-probe-bridge-rank-{rank}"))
            .spawn(move || {
                let result = run_routed_bridge_compare_rank(
                    rank,
                    world_size,
                    batch_size,
                    hidden,
                    id,
                    begin,
                    enqueued,
                    captured,
                    launched,
                    replay_iters,
                );
                let _ = tx.send((rank, result));
            })
            .with_context(|| format!("spawn routed bridge graph probe rank {rank}"))?;
    }
    drop(tx);

    let mut failures = Vec::new();
    let mut rank_timings = Vec::with_capacity(world_size);
    for (rank, result) in rx {
        match result {
            Ok(timing) => rank_timings.push((rank, timing)),
            Err(err) => failures.push(format!("rank {rank}: {err:#}")),
        }
    }
    if !failures.is_empty() {
        bail!("routed bridge graph probe failed:\n{}", failures.join("\n"));
    }
    rank_timings.sort_by_key(|(rank, _)| *rank);
    let all_reduce_avg_us_by_rank: Vec<f32> = rank_timings
        .iter()
        .map(|(_, timing)| timing.all_reduce_avg_us)
        .collect();
    let repeat_reduce_scatter_avg_us_by_rank: Vec<f32> = rank_timings
        .iter()
        .map(|(_, timing)| timing.repeat_reduce_scatter_avg_us)
        .collect();
    let all_reduce_avg_us_max_rank = all_reduce_avg_us_by_rank
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let repeat_reduce_scatter_avg_us_max_rank = repeat_reduce_scatter_avg_us_by_rank
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    Ok(BridgeReport {
        replay_iters,
        all_reduce_avg_us_max_rank,
        repeat_reduce_scatter_avg_us_max_rank,
        repeat_reduce_scatter_over_all_reduce: repeat_reduce_scatter_avg_us_max_rank
            / all_reduce_avg_us_max_rank.max(f32::EPSILON),
        all_reduce_avg_us_by_rank,
        repeat_reduce_scatter_avg_us_by_rank,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_routed_bridge_compare_rank(
    rank: usize,
    world_size: usize,
    batch_size: usize,
    hidden: usize,
    id: Id,
    begin: Arc<Barrier>,
    enqueued: Arc<Barrier>,
    captured: Arc<Barrier>,
    launched: Arc<Barrier>,
    replay_iters: usize,
) -> Result<RankBridgeTiming> {
    let ctx = DeviceContext::new_with_device(rank)?;
    let comm = Comm::from_rank(ctx.stream.clone(), rank, world_size, id)
        .map_err(|err| anyhow!("NCCL comm init failed: {err:?}"))?;
    let elems = batch_size * hidden;
    let mut all_reduce_values: CudaSlice<f32> = ctx.stream.alloc_zeros(elems)?;
    let local: CudaSlice<f32> = ctx.stream.alloc_zeros(elems)?;
    let mut repeated: CudaSlice<f32> = ctx.stream.alloc_zeros(elems * world_size)?;
    let mut reduce_scatter_recv: CudaSlice<f32> = ctx.stream.alloc_zeros(elems)?;

    comm.all_reduce_in_place(&mut all_reduce_values, &ReduceOp::Sum)
        .map_err(|err| anyhow!("warmup all_reduce failed: {:?}", err.0))?;
    repeat_f32_for_reduce_scatter_into(&ctx, &local, &mut repeated, elems, world_size)?;
    comm.reduce_scatter(&repeated, &mut reduce_scatter_recv, &ReduceOp::Sum)
        .map_err(|err| anyhow!("warmup repeat+reduce_scatter failed: {:?}", err.0))?;
    ctx.sync()?;

    let all_reduce_graph = capture_routed_bridge_graph(
        rank,
        &ctx,
        &comm,
        &mut all_reduce_values,
        &local,
        &mut repeated,
        &mut reduce_scatter_recv,
        elems,
        world_size,
        false,
        &begin,
        &enqueued,
        &captured,
    )?;
    let all_reduce_avg_us = replay_graph_timed(
        rank,
        &ctx,
        &all_reduce_graph,
        replay_iters,
        &launched,
        "routed-all-reduce",
    )?;

    let repeat_reduce_scatter_graph = capture_routed_bridge_graph(
        rank,
        &ctx,
        &comm,
        &mut all_reduce_values,
        &local,
        &mut repeated,
        &mut reduce_scatter_recv,
        elems,
        world_size,
        true,
        &begin,
        &enqueued,
        &captured,
    )?;
    let repeat_reduce_scatter_avg_us = replay_graph_timed(
        rank,
        &ctx,
        &repeat_reduce_scatter_graph,
        replay_iters,
        &launched,
        "routed-repeat-reduce-scatter",
    )?;

    Ok(RankBridgeTiming {
        all_reduce_avg_us,
        repeat_reduce_scatter_avg_us,
    })
}

#[allow(clippy::too_many_arguments)]
fn capture_routed_bridge_graph(
    rank: usize,
    ctx: &DeviceContext,
    comm: &Comm,
    all_reduce_values: &mut CudaSlice<f32>,
    local: &CudaSlice<f32>,
    repeated: &mut CudaSlice<f32>,
    reduce_scatter_recv: &mut CudaSlice<f32>,
    elems: usize,
    world_size: usize,
    repeat_reduce_scatter: bool,
    begin: &Barrier,
    enqueued: &Barrier,
    captured: &Barrier,
) -> Result<CudaGraph> {
    begin.wait();
    ctx.stream
        .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .map_err(|err| anyhow!("rank {rank} begin_capture failed: {err}"))?;
    if repeat_reduce_scatter {
        repeat_f32_for_reduce_scatter_into(ctx, local, repeated, elems, world_size)?;
        comm.reduce_scatter(repeated, reduce_scatter_recv, &ReduceOp::Sum)
            .map_err(|err| anyhow!("capture repeat+reduce_scatter failed: {:?}", err.0))?;
    } else {
        comm.all_reduce_in_place(all_reduce_values, &ReduceOp::Sum)
            .map_err(|err| anyhow!("capture all_reduce failed: {:?}", err.0))?;
    }
    enqueued.wait();
    let graph = ctx
        .stream
        .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|err| anyhow!("rank {rank} end_capture failed: {err}"))?
        .ok_or_else(|| anyhow!("rank {rank} end_capture returned empty graph"))?;
    captured.wait();
    Ok(graph)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let started = Instant::now();
    let mut bridge = None;
    let overlap = match cli.probe {
        Probe::LocalKernel => {
            run_local_kernel(cli.batch_size, cli.hidden)?;
            None
        }
        Probe::Gemm => {
            run_gemm(cli.batch_size, cli.hidden, cli.gemm_out)?;
            None
        }
        Probe::NcclAllReduce => {
            run_nccl_probe(cli.world_size, cli.batch_size, cli.hidden, false)?;
            None
        }
        Probe::NcclReduceScatter => {
            run_nccl_probe(cli.world_size, cli.batch_size, cli.hidden, true)?;
            None
        }
        Probe::NcclTwoStreamOverlap => Some(run_nccl_two_stream_overlap_probe(
            cli.world_size,
            cli.batch_size,
            cli.hidden,
            cli.replay_iters,
        )?),
        Probe::RoutedBridgeCompare => {
            bridge = Some(run_routed_bridge_compare_probe(
                cli.world_size,
                cli.batch_size,
                cli.hidden,
                cli.replay_iters,
            )?);
            None
        }
    };
    let report = ProbeReport {
        probe: cli.probe,
        world_size: cli.world_size,
        batch_size: cli.batch_size,
        hidden: cli.hidden,
        gemm_out: cli.gemm_out,
        capture_ok: true,
        replay_ok: true,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        overlap,
        bridge,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_nccl_two_stream_overlap_probe(
    world_size: usize,
    batch_size: usize,
    hidden: usize,
    replay_iters: usize,
) -> Result<OverlapReport> {
    if world_size == 0 {
        bail!("world_size must be positive");
    }
    if replay_iters == 0 {
        bail!("replay_iters must be positive");
    }
    let main_id =
        Id::new().map_err(|err| anyhow!("main NCCL unique id creation failed: {err:?}"))?;
    let aux_id = Id::new().map_err(|err| anyhow!("aux NCCL unique id creation failed: {err:?}"))?;
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
            .name(format!("kimi-graph-probe-overlap-rank-{rank}"))
            .spawn(move || {
                let result = run_nccl_two_stream_overlap_rank(
                    rank,
                    world_size,
                    batch_size,
                    hidden,
                    main_id,
                    aux_id,
                    begin,
                    enqueued,
                    captured,
                    launched,
                    replay_iters,
                );
                let _ = tx.send((rank, result));
            })
            .with_context(|| format!("spawn two-stream graph probe rank {rank}"))?;
    }
    drop(tx);

    let mut failures = Vec::new();
    let mut rank_timings = Vec::with_capacity(world_size);
    for (rank, result) in rx {
        match result {
            Ok(timing) => rank_timings.push((rank, timing)),
            Err(err) => failures.push(format!("rank {rank}: {err:#}")),
        }
    }
    if failures.is_empty() {
        let mut timings = vec![
            RankOverlapTiming {
                sequential_avg_us: 0.0,
                overlap_avg_us: 0.0,
            };
            world_size
        ];
        for (rank, timing) in rank_timings {
            timings[rank] = timing;
        }
        let sequential_avg_us_by_rank = timings
            .iter()
            .map(|timing| timing.sequential_avg_us)
            .collect::<Vec<_>>();
        let overlap_avg_us_by_rank = timings
            .iter()
            .map(|timing| timing.overlap_avg_us)
            .collect::<Vec<_>>();
        let sequential_avg_us_max_rank = sequential_avg_us_by_rank
            .iter()
            .copied()
            .fold(0.0, f32::max);
        let overlap_avg_us_max_rank = overlap_avg_us_by_rank.iter().copied().fold(0.0, f32::max);
        Ok(OverlapReport {
            replay_iters,
            sequential_avg_us_max_rank,
            overlap_avg_us_max_rank,
            speedup_max_rank: sequential_avg_us_max_rank / overlap_avg_us_max_rank.max(0.001),
            sequential_avg_us_by_rank,
            overlap_avg_us_by_rank,
        })
    } else {
        bail!("two-stream graph probe failed:\n{}", failures.join("\n"))
    }
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

#[allow(clippy::too_many_arguments)]
fn run_nccl_two_stream_overlap_rank(
    rank: usize,
    world_size: usize,
    batch_size: usize,
    hidden: usize,
    main_id: Id,
    aux_id: Id,
    begin: Arc<Barrier>,
    enqueued: Arc<Barrier>,
    captured: Arc<Barrier>,
    launched: Arc<Barrier>,
    replay_iters: usize,
) -> Result<RankOverlapTiming> {
    let main_ctx = DeviceContext::new_with_device(rank)?;
    let aux_stream = main_ctx
        .ctx
        .new_stream()
        .map_err(|err| anyhow!("rank {rank} aux stream creation failed: {err}"))?;
    let aux_ctx = DeviceContext {
        ctx: Arc::clone(&main_ctx.ctx),
        stream: Arc::clone(&aux_stream),
        device_ordinal: main_ctx.device_ordinal,
    };
    let main_comm = Comm::from_rank(main_ctx.stream.clone(), rank, world_size, main_id)
        .map_err(|err| anyhow!("main NCCL comm init failed: {err:?}"))?;
    let aux_comm = Comm::from_rank(aux_ctx.stream.clone(), rank, world_size, aux_id)
        .map_err(|err| anyhow!("aux NCCL comm init failed: {err:?}"))?;

    let elems = batch_size * hidden;
    let mut all_reduce_values: CudaSlice<f32> = main_ctx.stream.alloc_zeros(elems)?;
    let reduce_scatter_send: CudaSlice<f32> = aux_ctx.stream.alloc_zeros(elems * world_size)?;
    let mut reduce_scatter_recv: CudaSlice<f32> = aux_ctx.stream.alloc_zeros(elems)?;

    main_comm
        .all_reduce_in_place(&mut all_reduce_values, &ReduceOp::Sum)
        .map_err(|err| anyhow!("warmup main all_reduce failed: {:?}", err.0))?;
    aux_comm
        .reduce_scatter(
            &reduce_scatter_send,
            &mut reduce_scatter_recv,
            &ReduceOp::Sum,
        )
        .map_err(|err| anyhow!("warmup aux reduce_scatter failed: {:?}", err.0))?;
    main_ctx.sync()?;

    let sequential = capture_two_stream_nccl_graph(
        rank,
        &main_ctx,
        &aux_ctx,
        &main_comm,
        &aux_comm,
        &mut all_reduce_values,
        &reduce_scatter_send,
        &mut reduce_scatter_recv,
        false,
        &begin,
        &enqueued,
        &captured,
    )?;
    let sequential_avg_us = replay_graph_timed(
        rank,
        &main_ctx,
        &sequential,
        replay_iters,
        &launched,
        "sequential",
    )?;

    let overlap = capture_two_stream_nccl_graph(
        rank,
        &main_ctx,
        &aux_ctx,
        &main_comm,
        &aux_comm,
        &mut all_reduce_values,
        &reduce_scatter_send,
        &mut reduce_scatter_recv,
        true,
        &begin,
        &enqueued,
        &captured,
    )?;
    let overlap_avg_us = replay_graph_timed(
        rank,
        &main_ctx,
        &overlap,
        replay_iters,
        &launched,
        "overlap",
    )?;

    Ok(RankOverlapTiming {
        sequential_avg_us,
        overlap_avg_us,
    })
}

#[allow(clippy::too_many_arguments)]
fn capture_two_stream_nccl_graph(
    rank: usize,
    main_ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    main_comm: &Comm,
    aux_comm: &Comm,
    all_reduce_values: &mut CudaSlice<f32>,
    reduce_scatter_send: &CudaSlice<f32>,
    reduce_scatter_recv: &mut CudaSlice<f32>,
    overlap: bool,
    begin: &Barrier,
    enqueued: &Barrier,
    captured: &Barrier,
) -> Result<CudaGraph> {
    begin.wait();
    main_ctx
        .stream
        .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .map_err(|err| anyhow!("rank {rank} begin_capture failed: {err}"))?;

    let aux_start = main_ctx
        .stream
        .record_event(None)
        .map_err(|err| anyhow!("rank {rank} record aux_start failed: {err}"))?;
    aux_ctx
        .stream
        .wait(&aux_start)
        .map_err(|err| anyhow!("rank {rank} aux wait aux_start failed: {err}"))?;

    if overlap {
        aux_comm
            .reduce_scatter(reduce_scatter_send, reduce_scatter_recv, &ReduceOp::Sum)
            .map_err(|err| anyhow!("capture aux reduce_scatter enqueue failed: {:?}", err.0))?;
        main_comm
            .all_reduce_in_place(all_reduce_values, &ReduceOp::Sum)
            .map_err(|err| anyhow!("capture main all_reduce enqueue failed: {:?}", err.0))?;
    } else {
        main_comm
            .all_reduce_in_place(all_reduce_values, &ReduceOp::Sum)
            .map_err(|err| anyhow!("capture main all_reduce enqueue failed: {:?}", err.0))?;
        let main_done = main_ctx
            .stream
            .record_event(None)
            .map_err(|err| anyhow!("rank {rank} record main_done failed: {err}"))?;
        aux_ctx
            .stream
            .wait(&main_done)
            .map_err(|err| anyhow!("rank {rank} aux wait main_done failed: {err}"))?;
        aux_comm
            .reduce_scatter(reduce_scatter_send, reduce_scatter_recv, &ReduceOp::Sum)
            .map_err(|err| anyhow!("capture aux reduce_scatter enqueue failed: {:?}", err.0))?;
    }

    let aux_done = aux_ctx
        .stream
        .record_event(None)
        .map_err(|err| anyhow!("rank {rank} record aux_done failed: {err}"))?;
    main_ctx
        .stream
        .wait(&aux_done)
        .map_err(|err| anyhow!("rank {rank} main wait aux_done failed: {err}"))?;
    enqueued.wait();

    let graph = main_ctx
        .stream
        .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|err| anyhow!("rank {rank} end_capture failed: {err}"))?
        .ok_or_else(|| anyhow!("rank {rank} end_capture returned empty graph"))?;
    captured.wait();
    Ok(graph)
}

fn replay_graph_timed(
    rank: usize,
    ctx: &DeviceContext,
    graph: &CudaGraph,
    replay_iters: usize,
    launched: &Barrier,
    label: &str,
) -> Result<f32> {
    graph
        .launch()
        .map_err(|err| anyhow!("rank {rank} warmup {label} graph launch failed: {err}"))?;
    launched.wait();
    ctx.sync()?;

    launched.wait();
    let start = ctx
        .stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .map_err(|err| anyhow!("rank {rank} record {label} start failed: {err}"))?;
    for _ in 0..replay_iters {
        graph
            .launch()
            .map_err(|err| anyhow!("rank {rank} {label} graph launch failed: {err}"))?;
    }
    let end = ctx
        .stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .map_err(|err| anyhow!("rank {rank} record {label} end failed: {err}"))?;
    end.synchronize()
        .map_err(|err| anyhow!("rank {rank} sync {label} end failed: {err}"))?;
    launched.wait();
    let elapsed_ms = start
        .elapsed_ms(&end)
        .map_err(|err| anyhow!("rank {rank} elapsed {label} failed: {err}"))?;
    Ok(elapsed_ms * 1000.0 / replay_iters as f32)
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
