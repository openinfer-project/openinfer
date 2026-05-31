use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use log::info;
use pegainfer_core::{engine::EpBackend, parallel::ParallelConfig};
use pegainfer_kernels::tensor::KernelCall;
use pegainfer_kimi_k2::batch_decode_trace::{
    MODEL, RuntimeTraceOptions, trace_decode_kernel_calls,
    trace_runtime_decode_kernel_calls_with_options,
};
use pegainfer_kimi_k2::kernel_report::{MeasuredCall, bench_key, measure_call};
use serde::Serialize;

const DEFAULT_ITERS: u64 = 32;

#[derive(Parser)]
#[command(about = "Kimi-K2 per-op kernel report runner")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run one op provider selected from the decode DAG.
    Run(RunArgs),
    /// Print the rank0 decode DAG without measuring.
    Trace(TraceArgs),
}

#[derive(Parser)]
struct RunArgs {
    #[arg(long)]
    op: String,
    #[arg(long = "batch-size")]
    batch_size: usize,
    #[arg(long = "kv-len")]
    kv_len: usize,
    #[arg(long, default_value_t = DEFAULT_ITERS)]
    iters: u64,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value = "models/Kimi-K2.5")]
    model_path: String,
    #[arg(long, value_enum, default_value_t = TraceSource::Runtime)]
    source: TraceSource,
    #[arg(long = "tp-world", default_value_t = 8)]
    tp_world: usize,
    #[arg(long = "dp-world", default_value_t = 1)]
    dp_world: usize,
    #[arg(long = "ep-backend", value_enum, default_value_t = EpBackendArg::Nccl)]
    ep_backend: EpBackendArg,
}

#[derive(Parser)]
struct TraceArgs {
    #[arg(long = "batch-size")]
    batch_size: usize,
    #[arg(long = "kv-len")]
    kv_len: usize,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value = "models/Kimi-K2.5")]
    model_path: String,
    #[arg(long, value_enum, default_value_t = TraceSource::Runtime)]
    source: TraceSource,
    #[arg(long = "tp-world", default_value_t = 8)]
    tp_world: usize,
    #[arg(long = "dp-world", default_value_t = 1)]
    dp_world: usize,
    #[arg(long = "ep-backend", value_enum, default_value_t = EpBackendArg::Nccl)]
    ep_backend: EpBackendArg,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TraceSource {
    Runtime,
    Static,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum EpBackendArg {
    Nccl,
    Pplx,
}

impl From<EpBackendArg> for EpBackend {
    fn from(value: EpBackendArg) -> Self {
        match value {
            EpBackendArg::Nccl => EpBackend::Nccl,
            EpBackendArg::Pplx => EpBackend::Pplx,
        }
    }
}

#[derive(Serialize)]
struct KernelReport {
    schema: u32,
    report_type: String,
    model: String,
    rank_scope: String,
    op: String,
    batch_size: usize,
    kv_len: usize,
    iters: u64,
    selected_call: KernelCall,
    key: String,
    measured: MeasuredCall,
}

fn main() -> Result<()> {
    pegainfer_core::logging::init_default();
    match Cli::parse().command {
        Command::Run(args) => run(args),
        Command::Trace(args) => trace(args),
    }
}

fn run(args: RunArgs) -> Result<()> {
    validate_common(args.batch_size, args.kv_len, args.iters)?;
    let schedule = load_schedule(
        args.source,
        &args.model_path,
        args.batch_size,
        args.kv_len,
        args.tp_world,
        args.dp_world,
        args.ep_backend,
    )?;
    let call = schedule
        .iter()
        .find(|call| call.op == args.op)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("decode DAG does not contain op `{}`", args.op))?;
    let measured = measure_call(&call, args.iters)?;
    let report = KernelReport {
        schema: 1,
        report_type: "kimi_kernel_report".to_string(),
        model: MODEL.to_string(),
        rank_scope:
            "rank0 local provider; all-rank NCCL and EP imbalance need dedicated H20 harness"
                .to_string(),
        op: args.op,
        batch_size: args.batch_size,
        kv_len: args.kv_len,
        iters: args.iters,
        key: bench_key(&call)?,
        selected_call: call,
        measured,
    };
    let out = args.out.unwrap_or_else(|| {
        PathBuf::from(format!(
            "target/kernel_reports/{MODEL}/{}-bs{}-kv{}.json",
            report.op, report.batch_size, report.kv_len
        ))
    });
    write_json(&out, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    info!("wrote {}", out.display());
    Ok(())
}

fn trace(args: TraceArgs) -> Result<()> {
    if args.batch_size == 0 || args.kv_len == 0 {
        bail!("--batch-size and --kv-len must be greater than zero");
    }
    let schedule = load_schedule(
        args.source,
        &args.model_path,
        args.batch_size,
        args.kv_len,
        args.tp_world,
        args.dp_world,
        args.ep_backend,
    )?;
    let out = args.out.unwrap_or_else(|| {
        PathBuf::from(format!(
            "target/kernel_reports/{MODEL}/decode-trace-rank0-bs{}-kv{}.json",
            args.batch_size, args.kv_len
        ))
    });
    write_json(&out, &schedule)?;
    println!("{}", serde_json::to_string_pretty(&schedule)?);
    info!("wrote {}", out.display());
    Ok(())
}

fn load_schedule(
    source: TraceSource,
    model_path: &str,
    batch_size: usize,
    kv_len: usize,
    tp_world: usize,
    dp_world: usize,
    ep_backend: EpBackendArg,
) -> Result<Vec<KernelCall>> {
    match source {
        TraceSource::Runtime => {
            let parallel = ParallelConfig::new(tp_world, dp_world);
            trace_runtime_decode_kernel_calls_with_options(
                model_path,
                batch_size,
                kv_len,
                RuntimeTraceOptions {
                    parallel,
                    ep_backend: ep_backend.into(),
                    device_ordinals: (0..parallel.ep_world()).collect(),
                },
            )
        }
        TraceSource::Static => trace_decode_kernel_calls(model_path, batch_size, kv_len),
    }
}

fn validate_common(batch_size: usize, kv_len: usize, iters: u64) -> Result<()> {
    if batch_size == 0 || kv_len == 0 || iters == 0 {
        bail!("--batch-size, --kv-len, and --iters must be greater than zero");
    }
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}
