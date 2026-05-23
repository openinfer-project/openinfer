use std::ffi::c_void;
use std::io::{BufRead, BufReader};
use std::process::Command;
use std::ptr;
use std::sync::{Arc, Barrier};
use std::thread;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use pegainfer_comm::ScalarType;
use pegainfer_comm::bootstrap::{
    EpModelShape, PplxBootstrapParams, build_intra_node_backends_for_devices,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 256)]
    n_experts: usize,
    #[arg(long, default_value_t = 6)]
    topk: usize,
    #[arg(long, default_value_t = 4096)]
    hidden_dim: usize,
    #[arg(long, default_value_t = 8)]
    world_size: usize,
    #[arg(long, default_value_t = 1)]
    max_num_tokens: usize,
    #[arg(long, default_value_t = 64)]
    max_private_tokens: usize,
    #[arg(long, default_value_t = 16)]
    expert_padding: usize,
    #[arg(long, default_value_t = 1)]
    nets_per_gpu: u8,
    #[arg(long, default_value_t = 20)]
    warmup: usize,
    #[arg(long, default_value_t = 100)]
    repeats: usize,

    /// Sweep preset model shapes (dsv4, kimi-k2) x token counts.
    /// Overrides --n-experts/--topk/--hidden-dim/--max-num-tokens.
    #[arg(long)]
    sweep: bool,
}

#[derive(Clone, Debug)]
struct BenchConfig {
    label: String,
    shape: EpModelShape,
    world_size: usize,
    max_num_tokens: usize,
    max_private_tokens: Option<usize>,
    expert_padding: usize,
    nets_per_gpu: u8,
    warmup: usize,
    repeats: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct IterTimes {
    dispatch_send_us: f64,
    dispatch_recv_us: f64,
    combine_send_us: f64,
    combine_recv_us: f64,
}

impl IterTimes {
    fn split_sum_us(self) -> f64 {
        self.dispatch_send_us
            + self.dispatch_recv_us
            + self.combine_send_us
            + self.combine_recv_us
    }
}

#[derive(Debug)]
struct Stats {
    mean: f64,
    min: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
}

struct GpuContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl GpuContext {
    fn new(device: usize) -> Result<Self> {
        let ctx = CudaContext::new(device).with_context(|| {
            format!("failed to create CUDA context for device {device}")
        })?;
        let stream = ctx.new_stream().with_context(|| {
            format!("failed to create CUDA stream for device {device}")
        })?;
        Ok(Self { ctx, stream })
    }

    fn sync(&self) -> Result<()> {
        self.stream.synchronize().context("failed to synchronize stream")
    }
}

struct SweepRow {
    label: String,
    summary_line: String,
}

const DSV4_SHAPE: EpModelShape =
    EpModelShape { n_routed_experts: 256, n_activated_experts: 6, hidden_dim: 4096 };

const KIMI_K2_SHAPE: EpModelShape =
    EpModelShape { n_routed_experts: 384, n_activated_experts: 8, hidden_dim: 7168 };

fn sweep_configs(args: &Args) -> Vec<BenchConfig> {
    let shapes = [("dsv4", DSV4_SHAPE), ("kimi-k2", KIMI_K2_SHAPE)];
    let token_counts = [1, 4, 8, 32, 128, 256];
    let mut configs = Vec::new();
    for &(name, shape) in &shapes {
        for &tokens in &token_counts {
            configs.push(BenchConfig {
                label: format!("{name}/tok={tokens}"),
                shape,
                world_size: args.world_size,
                max_num_tokens: tokens,
                max_private_tokens: None,
                expert_padding: args.expert_padding,
                nets_per_gpu: args.nets_per_gpu,
                warmup: args.warmup,
                repeats: args.repeats,
            });
        }
    }
    configs
}

fn run_sweep(args: &Args) -> Result<()> {
    let configs = sweep_configs(args);
    let exe = std::env::current_exe().context("resolve current executable")?;
    let mut rows = Vec::with_capacity(configs.len());

    for config in &configs {
        eprintln!("[sweep] running {}", config.label);
        let mut child = Command::new(&exe)
            .args([
                "--n-experts",
                &config.shape.n_routed_experts.to_string(),
                "--topk",
                &config.shape.n_activated_experts.to_string(),
                "--hidden-dim",
                &config.shape.hidden_dim.to_string(),
                "--world-size",
                &config.world_size.to_string(),
                "--max-num-tokens",
                &config.max_num_tokens.to_string(),
                "--expert-padding",
                &config.expert_padding.to_string(),
                "--nets-per-gpu",
                &config.nets_per_gpu.to_string(),
                "--warmup",
                &config.warmup.to_string(),
                "--repeats",
                &config.repeats.to_string(),
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn subprocess for {}", config.label))?;

        let stdout = child.stdout.take().unwrap();
        let mut max_rank_line = None;
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let line = line?;
            println!("{}", line);
            if line.starts_with("max_rank_split_us:") {
                max_rank_line = Some(line);
            }
        }

        let status = child.wait()?;
        ensure!(status.success(), "{} exited with {status}", config.label);

        if let Some(line) = max_rank_line {
            rows.push(SweepRow { label: config.label.clone(), summary_line: line });
        }
        println!();
    }

    println!("=== sweep summary (max_rank_split_sum_us) ===");
    for row in &rows {
        println!("{:<20} {}", row.label, row.summary_line);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.world_size > 0, "world_size must be positive");
    ensure!(args.repeats > 0, "repeats must be positive");
    ensure!(args.nets_per_gpu > 0, "nets_per_gpu must be positive for pplx bootstrap");

    if args.sweep {
        run_sweep(&args)?;
    } else {
        let config = BenchConfig {
            label: format!(
                "e={}/k={}/h={}",
                args.n_experts, args.topk, args.hidden_dim
            ),
            shape: EpModelShape {
                n_routed_experts: args.n_experts,
                n_activated_experts: args.topk,
                hidden_dim: args.hidden_dim,
            },
            world_size: args.world_size,
            max_num_tokens: args.max_num_tokens,
            max_private_tokens: Some(args.max_private_tokens),
            expert_padding: args.expert_padding,
            nets_per_gpu: args.nets_per_gpu,
            warmup: args.warmup,
            repeats: args.repeats,
        };
        let rank_results = run_config(&config)?;
        print_report(&rank_results);
    }
    Ok(())
}

fn run_config(config: &BenchConfig) -> Result<Vec<Vec<IterTimes>>> {
    let devices: Vec<usize> = (0..config.world_size).collect();
    let params = PplxBootstrapParams {
        max_num_tokens: config.max_num_tokens,
        expert_padding: config.expert_padding,
        max_private_tokens: config.max_private_tokens,
        nets_per_gpu: config.nets_per_gpu,
        imm_base: 0x8a2a_0000,
    };

    eprintln!(
        "[{}] bootstrap: world={} max_tokens={} experts={} topk={} hidden={}",
        config.label,
        config.world_size,
        config.max_num_tokens,
        config.shape.n_routed_experts,
        config.shape.n_activated_experts,
        config.shape.hidden_dim,
    );
    let (backends, _resources) =
        build_intra_node_backends_for_devices(config.shape, &devices, params)?;

    let barrier = Arc::new(Barrier::new(config.world_size));
    let mut rank_results: Vec<Vec<IterTimes>> = Vec::with_capacity(config.world_size);
    thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(config.world_size);
        for (rank, backend) in backends.into_iter().enumerate() {
            let barrier = Arc::clone(&barrier);
            let config = config;
            handles.push(scope.spawn(move || {
                run_rank(rank, backend, config, barrier)
                    .with_context(|| format!("rank {rank}"))
            }));
        }
        for handle in handles {
            rank_results.push(handle.join().expect("rank bench thread panicked")?);
        }
        Ok(())
    })?;

    Ok(rank_results)
}

fn run_rank(
    rank: usize,
    mut backend: pegainfer_comm::EpBackend,
    config: &BenchConfig,
    barrier: Arc<Barrier>,
) -> Result<Vec<IterTimes>> {
    let gpu = GpuContext::new(rank)?;

    let hidden = config.shape.hidden_dim;
    let topk = config.shape.n_activated_experts;
    let local_experts = config.shape.n_routed_experts / config.world_size;
    let max_private_tokens = config.max_private_tokens.unwrap_or_else(|| {
        let avg =
            (config.max_num_tokens * topk).div_ceil(config.shape.n_routed_experts);
        (avg + avg / 5 + 1) * local_experts
    });
    let max_recv_tokens = compute_max_recv_tokens(
        config.max_num_tokens,
        topk,
        local_experts,
        config.world_size,
        max_private_tokens,
        config.expert_padding,
    );

    let x_host =
        vec![bf16::from_f32((rank + 1) as f32); config.max_num_tokens * hidden];
    let indices_host = route_indices(
        rank,
        config.world_size,
        config.max_num_tokens,
        topk,
        local_experts,
    );
    let weights_host = vec![1.0f32 / topk as f32; config.max_num_tokens * topk];

    let x = gpu.stream.clone_htod(&x_host)?;
    let indices = gpu.stream.clone_htod(&indices_host)?;
    let weights = gpu.stream.clone_htod(&weights_host)?;
    let mut recv_tokens_per_expert = gpu.stream.alloc_zeros::<i32>(local_experts)?;
    let mut out_x = gpu.stream.alloc_zeros::<bf16>(max_recv_tokens * hidden)?;
    let expert_y = gpu.stream.alloc_zeros::<bf16>(max_recv_tokens * hidden)?;
    let mut out_tokens =
        gpu.stream.alloc_zeros::<bf16>(config.max_num_tokens * hidden)?;
    gpu.sync()?;

    let total_iters = config.warmup + config.repeats;
    let mut measured = Vec::with_capacity(config.repeats);
    barrier.wait();
    for iter in 0..total_iters {
        let record = iter >= config.warmup;
        let mut times = IterTimes::default();

        times.dispatch_send_us = time_stage(&gpu, record, || {
            dispatch_send(
                &mut backend,
                config.max_num_tokens,
                hidden,
                topk,
                &x,
                &indices,
                &weights,
                &gpu,
            )
        })?;
        times.dispatch_recv_us = time_stage(&gpu, record, || {
            dispatch_recv(
                &mut backend,
                hidden,
                &mut recv_tokens_per_expert,
                &mut out_x,
                &gpu,
            )
        })?;
        times.combine_send_us = time_stage(&gpu, record, || {
            combine_send(&mut backend, hidden, &expert_y, &gpu)
        })?;
        times.combine_recv_us = time_stage(&gpu, record, || {
            combine_recv(
                &mut backend,
                config.max_num_tokens,
                hidden,
                topk,
                &mut out_tokens,
                &indices,
                &weights,
                &gpu,
            )
        })?;

        if record {
            measured.push(times);
        }
    }
    gpu.sync()?;
    barrier.wait();
    Ok(measured)
}

fn dispatch_send(
    backend: &mut pegainfer_comm::EpBackend,
    num_tokens: usize,
    hidden: usize,
    topk: usize,
    x: &CudaSlice<bf16>,
    indices: &CudaSlice<i32>,
    weights: &CudaSlice<f32>,
    gpu: &GpuContext,
) -> Result<()> {
    let stream = gpu.stream.cu_stream() as u64;
    let (x_ptr, _x_guard) = x.device_ptr(&gpu.stream);
    let (idx_ptr, _idx_guard) = indices.device_ptr(&gpu.stream);
    let (w_ptr, _w_guard) = weights.device_ptr(&gpu.stream);
    backend
        .dispatch_send(
            num_tokens,
            x_ptr as *const c_void,
            hidden * std::mem::size_of::<u16>(),
            ptr::null(),
            0,
            0,
            idx_ptr as *const i32,
            topk,
            w_ptr as *const f32,
            topk,
            ptr::null(),
            stream,
        )
        .map_err(anyhow::Error::from)
}

fn dispatch_recv(
    backend: &mut pegainfer_comm::EpBackend,
    hidden: usize,
    recv_tokens_per_expert: &mut CudaSlice<i32>,
    out_x: &mut CudaSlice<bf16>,
    gpu: &GpuContext,
) -> Result<()> {
    let stream = gpu.stream.cu_stream() as u64;
    let (out_num_ptr, _g0) = recv_tokens_per_expert.device_ptr_mut(&gpu.stream);
    let (out_x_ptr, _g1) = out_x.device_ptr_mut(&gpu.stream);
    backend
        .dispatch_recv(
            out_num_ptr as *mut i32,
            out_x_ptr as *mut c_void,
            hidden * std::mem::size_of::<u16>(),
            ptr::null_mut(),
            0,
            0,
            stream,
        )
        .map_err(anyhow::Error::from)
}

fn combine_send(
    backend: &mut pegainfer_comm::EpBackend,
    hidden: usize,
    expert_y: &CudaSlice<bf16>,
    gpu: &GpuContext,
) -> Result<()> {
    let stream = gpu.stream.cu_stream() as u64;
    let (expert_ptr, _g) = expert_y.device_ptr(&gpu.stream);
    backend
        .combine_send(
            expert_ptr as *const c_void,
            hidden * std::mem::size_of::<u16>(),
            stream,
        )
        .map_err(anyhow::Error::from)
}

fn combine_recv(
    backend: &mut pegainfer_comm::EpBackend,
    num_tokens: usize,
    hidden: usize,
    topk: usize,
    out_tokens: &mut CudaSlice<bf16>,
    indices: &CudaSlice<i32>,
    weights: &CudaSlice<f32>,
    gpu: &GpuContext,
) -> Result<()> {
    let stream = gpu.stream.cu_stream() as u64;
    let (out_ptr, _g0) = out_tokens.device_ptr_mut(&gpu.stream);
    let (idx_ptr, _g1) = indices.device_ptr(&gpu.stream);
    let (w_ptr, _g2) = weights.device_ptr(&gpu.stream);
    backend
        .combine_recv(
            num_tokens,
            0,
            ScalarType::BF16,
            out_ptr as *mut c_void,
            hidden,
            idx_ptr as *const i32,
            topk,
            w_ptr as *const f32,
            topk,
            ptr::null(),
            true,
            stream,
        )
        .map_err(anyhow::Error::from)
}

fn time_stage<F>(gpu: &GpuContext, record: bool, f: F) -> Result<f64>
where
    F: FnOnce() -> Result<()>,
{
    if !record {
        f()?;
        return Ok(0.0);
    }
    let start = gpu
        .ctx
        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = gpu
        .ctx
        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(&gpu.stream)?;
    f()?;
    end.record(&gpu.stream)?;
    Ok(start.elapsed_ms(&end)? as f64 * 1000.0)
}

fn route_indices(
    rank: usize,
    world_size: usize,
    max_num_tokens: usize,
    topk: usize,
    local_experts: usize,
) -> Vec<i32> {
    let mut out = Vec::with_capacity(max_num_tokens * topk);
    for token in 0..max_num_tokens {
        for k in 0..topk {
            let dst_rank = (rank + k + 1) % world_size;
            let local_expert = (token * topk + k) % local_experts;
            out.push((dst_rank * local_experts + local_expert) as i32);
        }
    }
    out
}

fn compute_max_recv_tokens(
    max_num_tokens: usize,
    topk: usize,
    local_experts: usize,
    world_size: usize,
    max_private_tokens: usize,
    expert_padding: usize,
) -> usize {
    let num_tokens_total = max_num_tokens * world_size;
    let padded_recv = round_up(
        std::cmp::max(
            std::cmp::min(
                num_tokens_total * topk + local_experts * (expert_padding - 1),
                num_tokens_total * local_experts,
            ),
            local_experts * expert_padding,
        ),
        expert_padding,
    );
    max_private_tokens * world_size + padded_recv
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn max_rank_split_stats(rank_results: &[Vec<IterTimes>]) -> Stats {
    let repeats = rank_results.first().map_or(0, Vec::len);
    let mut max_split_by_iter = Vec::with_capacity(repeats);
    for iter in 0..repeats {
        let max_us = rank_results
            .iter()
            .map(|rank| rank[iter].split_sum_us())
            .fold(0.0, f64::max);
        max_split_by_iter.push(max_us);
    }
    stats(&max_split_by_iter)
}

fn print_report_header(label: &str) {
    println!("--- {label} ---");
}

fn print_report(rank_results: &[Vec<IterTimes>]) {
    let mut dispatch_send = Vec::new();
    let mut dispatch_recv = Vec::new();
    let mut combine_send = Vec::new();
    let mut combine_recv = Vec::new();
    let mut split_sum = Vec::new();
    for rank in rank_results {
        for &t in rank {
            dispatch_send.push(t.dispatch_send_us);
            dispatch_recv.push(t.dispatch_recv_us);
            combine_send.push(t.combine_send_us);
            combine_recv.push(t.combine_recv_us);
            split_sum.push(t.split_sum_us());
        }
    }

    print_stats("dispatch_send_us", &dispatch_send);
    print_stats("dispatch_recv_us", &dispatch_recv);
    print_stats("combine_send_us", &combine_send);
    print_stats("combine_recv_us", &combine_recv);
    print_stats("split_sum_us", &split_sum);

    let s = max_rank_split_stats(rank_results);
    println!(
        "max_rank_split_us: mean={:.1} p50={:.1} p95={:.1} p99={:.1} max={:.1}",
        s.mean, s.p50, s.p95, s.p99, s.max
    );
}

fn print_stats(name: &str, values: &[f64]) {
    let s = stats(values);
    println!(
        "{name}: mean={:.2} min={:.2} p50={:.2} p95={:.2} p99={:.2} max={:.2}",
        s.mean, s.min, s.p50, s.p95, s.p99, s.max
    );
}

fn stats(values: &[f64]) -> Stats {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    Stats {
        mean,
        min: sorted[0],
        p50: percentile(&sorted, 0.50),
        p95: percentile(&sorted, 0.95),
        p99: percentile(&sorted, 0.99),
        max: sorted[sorted.len() - 1],
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}
