use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use pegainfer_kernels::{
    ops::{
        KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_INT4_GROUP_SIZE, KIMI_K2_LOCAL_EXPERTS,
    },
    tensor::KernelCall,
};
use pegainfer_kimi_k2::kernel_report::{MeasuredCall, measure_call};
use serde::Serialize;

const DEFAULT_ITERS: u64 = 16;
const DEFAULT_MIN_ACTIVE_ROWS: usize = 7;
const DEFAULT_H20_BF16_TFLOPS: f64 = 148.0;
const DEFAULT_H20_HBM_GBPS: f64 = 4_800.0;
const QUANTILES: [usize; 6] = [0, 50, 90, 95, 99, 100];
const BF16_BYTES: usize = 2;
const I32_BYTES: usize = 4;

#[derive(Parser)]
#[command(about = "Replay Kimi-K2 PPLX Marlin compute kernels from runtime route histograms")]
struct Cli {
    #[arg(long)]
    trace: PathBuf,
    #[arg(long, default_value_t = DEFAULT_ITERS)]
    iters: u64,
    #[arg(long, default_value_t = DEFAULT_MIN_ACTIVE_ROWS)]
    min_active_rows: usize,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = DEFAULT_H20_BF16_TFLOPS)]
    peak_tflops: f64,
    #[arg(long, default_value_t = DEFAULT_H20_HBM_GBPS)]
    peak_gbps: f64,
    #[arg(long)]
    ridge_flop_per_byte: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Serialize)]
struct ReplayReport {
    schema: u32,
    report_type: &'static str,
    config: ReplayConfig,
    samples: Vec<SampleReport>,
}

#[derive(Serialize)]
struct ReplayConfig {
    trace: String,
    iters: u64,
    min_active_rows: usize,
    quantiles: Vec<String>,
    peak_tflops: f64,
    peak_gbps: f64,
    ridge_flop_per_byte: f64,
}

#[derive(Serialize)]
struct SampleReport {
    sample: RouteSample,
    measured: Vec<ReplayMeasurement>,
}

#[derive(Clone, Debug, Serialize)]
struct RouteSample {
    source_index: usize,
    source_label: String,
    quantiles: Vec<String>,
    layer_idx: usize,
    rank: usize,
    ep_rank: usize,
    active_rows: usize,
    arena_rows: usize,
    local_expert_start: usize,
    recv_counts: Vec<i32>,
    recv_total_routes: usize,
    padded_rows: usize,
    active_local_experts: usize,
    max_count_per_expert: usize,
    recv_capacity: usize,
    expert_padding: usize,
    block_size: usize,
}

#[derive(Serialize)]
struct ReplayMeasurement {
    op: &'static str,
    label: String,
    measured: MeasuredCall,
    mean_us: Option<f64>,
    flops: u128,
    bytes: u128,
    arithmetic_intensity_flop_per_byte: Option<f64>,
    roofline_bound: ReplayBound,
    achieved_tflops: Option<f64>,
    achieved_gbps: Option<f64>,
    roofline_peak_pct: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReplayBound {
    Compute,
    Memory,
    Control,
    NoWork,
    Unsupported,
}

#[derive(Clone, Copy)]
enum ReplayOp {
    Routing,
    W13,
    SwiGlu,
    W2,
}

impl ReplayOp {
    const ALL: [Self; 4] = [Self::Routing, Self::W13, Self::SwiGlu, Self::W2];

    const fn op(self) -> &'static str {
        match self {
            Self::Routing => "kimi_pplx_build_marlin_routing_on_stream",
            Self::W13 => "kimi_marlin_wna16_pplx_w13_gemm",
            Self::SwiGlu => "kimi_marlin_w13_swiglu_pplx",
            Self::W2 => "kimi_marlin_wna16_pplx_w2_gemm",
        }
    }

    const fn short_name(self) -> &'static str {
        match self {
            Self::Routing => "routing",
            Self::W13 => "w13",
            Self::SwiGlu => "swiglu",
            Self::W2 => "w2",
        }
    }
}

#[derive(Clone, Copy)]
struct WorkEstimate {
    flops: u128,
    bytes: u128,
    bound: ReplayBound,
}

fn main() -> Result<()> {
    pegainfer_core::logging::init_default();
    let cli = Cli::parse();
    ensure!(cli.iters > 0, "--iters must be greater than zero");
    ensure!(
        cli.peak_tflops.is_finite() && cli.peak_tflops > 0.0,
        "--peak-tflops must be a positive finite number"
    );
    ensure!(
        cli.peak_gbps.is_finite() && cli.peak_gbps > 0.0,
        "--peak-gbps must be a positive finite number"
    );
    let ridge_flop_per_byte = cli
        .ridge_flop_per_byte
        .unwrap_or(cli.peak_tflops * 1_000.0 / cli.peak_gbps);
    ensure!(
        ridge_flop_per_byte.is_finite() && ridge_flop_per_byte > 0.0,
        "derived --ridge-flop-per-byte must be a positive finite number"
    );

    let trace_bytes = fs::read(&cli.trace)
        .with_context(|| format!("failed to read trace {}", cli.trace.display()))?;
    let calls: Vec<KernelCall> = serde_json::from_slice(&trace_bytes)
        .with_context(|| format!("failed to parse trace {}", cli.trace.display()))?;
    let samples = select_samples(parse_route_samples(&calls)?, cli.min_active_rows)?;
    let mut reports = Vec::with_capacity(samples.len());
    for sample in samples {
        let mut measured = Vec::new();
        for replay_op in ReplayOp::ALL {
            measured.push(measure_replay_op(
                &sample,
                replay_op,
                cli.iters,
                ridge_flop_per_byte,
                cli.peak_tflops,
                cli.peak_gbps,
            )?);
        }
        reports.push(SampleReport { sample, measured });
    }

    let report = ReplayReport {
        schema: 1,
        report_type: "kimi_pplx_marlin_replay",
        config: ReplayConfig {
            trace: cli.trace.display().to_string(),
            iters: cli.iters,
            min_active_rows: cli.min_active_rows,
            quantiles: QUANTILES.iter().map(|q| format!("p{q}")).collect(),
            peak_tflops: cli.peak_tflops,
            peak_gbps: cli.peak_gbps,
            ridge_flop_per_byte,
        },
        samples: reports,
    };

    let out = cli
        .out
        .unwrap_or_else(|| PathBuf::from("target/kernel_reports/kimi-k2/pplx-marlin-replay.json"));
    write_json(&out, &report)?;
    match cli.format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        OutputFormat::Text => print_text(&report, &out),
    }
    Ok(())
}

fn parse_route_samples(calls: &[KernelCall]) -> Result<Vec<RouteSample>> {
    calls
        .iter()
        .enumerate()
        .filter(|(_, call)| call.op == "kimi_pplx_route_histogram")
        .map(|(source_index, call)| route_sample_from_call(source_index, call))
        .collect()
}

fn route_sample_from_call(source_index: usize, call: &KernelCall) -> Result<RouteSample> {
    let recv_counts = parse_recv_counts(attr(call, "recv_counts")?)?;
    ensure!(
        recv_counts.len() == KIMI_K2_LOCAL_EXPERTS,
        "{} recv_counts len {} must equal local experts {KIMI_K2_LOCAL_EXPERTS}",
        call.label,
        recv_counts.len()
    );
    let recv_total_routes = attr_usize(call, "recv_total_routes")?;
    let route_sum = route_elems_from_counts(&recv_counts)?;
    ensure!(
        route_sum == recv_total_routes,
        "{} recv_counts sum {route_sum} != recv_total_routes {recv_total_routes}",
        call.label
    );
    let expert_padding = attr_usize(call, "expert_padding")?;
    let padded_rows = attr_usize(call, "padded_rows")?;
    let padded_calc = padded_rows_from_counts(&recv_counts, expert_padding)?;
    ensure!(
        padded_calc == padded_rows,
        "{} recv_counts padded rows {padded_calc} != padded_rows {padded_rows}",
        call.label
    );
    let num_tokens_post_padded = attr_usize(call, "num_tokens_post_padded")?;
    ensure!(
        num_tokens_post_padded == padded_rows,
        "{} device num_tokens_post_padded {num_tokens_post_padded} != padded_rows {padded_rows}",
        call.label
    );
    let local_experts = attr_usize(call, "local_experts")?;
    ensure!(
        local_experts == KIMI_K2_LOCAL_EXPERTS,
        "{} local_experts {local_experts} != {KIMI_K2_LOCAL_EXPERTS}",
        call.label
    );
    Ok(RouteSample {
        source_index,
        source_label: call.label.clone(),
        quantiles: Vec::new(),
        layer_idx: attr_usize(call, "layer_idx")?,
        rank: attr_usize(call, "rank")?,
        ep_rank: attr_usize(call, "ep_rank")?,
        active_rows: attr_usize(call, "active_rows")?,
        arena_rows: attr_usize(call, "arena_rows")?,
        local_expert_start: attr_usize(call, "local_expert_start")?,
        recv_counts,
        recv_total_routes,
        padded_rows,
        active_local_experts: attr_usize(call, "active_local_experts")?,
        max_count_per_expert: attr_usize(call, "max_count_per_expert")?,
        recv_capacity: attr_usize(call, "recv_capacity")?,
        expert_padding,
        block_size: attr_usize(call, "block_size")?,
    })
}

fn select_samples(samples: Vec<RouteSample>, min_active_rows: usize) -> Result<Vec<RouteSample>> {
    let mut samples = samples
        .into_iter()
        .filter(|sample| {
            sample.active_rows >= min_active_rows
                && sample.recv_total_routes > 0
                && sample.padded_rows > 0
        })
        .collect::<Vec<_>>();
    if samples.is_empty() {
        bail!(
            "trace contains no non-empty PPLX route histograms with active_rows >= {min_active_rows}"
        );
    }
    samples.sort_by_key(|sample| {
        (
            sample.padded_rows,
            sample.recv_total_routes,
            sample.active_local_experts,
            sample.max_count_per_expert,
            sample.layer_idx,
            sample.rank,
            sample.source_index,
        )
    });
    let mut selected = BTreeMap::<usize, Vec<String>>::new();
    for quantile in QUANTILES {
        let index = percentile_index(samples.len(), quantile);
        selected
            .entry(index)
            .or_default()
            .push(format!("p{quantile}"));
    }
    Ok(selected
        .into_iter()
        .map(|(index, quantiles)| {
            let mut sample = samples[index].clone();
            sample.quantiles = quantiles;
            sample
        })
        .collect())
}

fn percentile_index(len: usize, quantile: usize) -> usize {
    debug_assert!(len > 0);
    debug_assert!(quantile <= 100);
    (quantile * (len - 1) + 50) / 100
}

fn measure_replay_op(
    sample: &RouteSample,
    replay_op: ReplayOp,
    iters: u64,
    ridge_flop_per_byte: f64,
    peak_tflops: f64,
    peak_gbps: f64,
) -> Result<ReplayMeasurement> {
    let label = format!(
        "decode.moe.pplx_{}.replay.{}.L{}.r{}",
        replay_op.short_name(),
        sample.quantiles.join("+"),
        sample.layer_idx,
        sample.rank
    );
    let call = KernelCall::new(replay_op.op(), label.clone())
        .attr("pplx_route_elems", sample.recv_total_routes.to_string())
        .attr("pplx_recv_capacity", sample.recv_capacity.to_string())
        .attr("expert_padding", sample.expert_padding.to_string())
        .attr("pplx_recv_counts", counts_csv(&sample.recv_counts));
    let measured = measure_call(&call, iters).with_context(|| {
        format!(
            "failed to replay {} from {}",
            replay_op.op(),
            sample.source_label
        )
    })?;
    let work = work_estimate(sample, replay_op);
    let mean_us = measured.stats.as_ref().map(|stats| stats.mean_us);
    let arithmetic_intensity_flop_per_byte = arithmetic_intensity(work.flops, work.bytes);
    let roofline_bound = roofline_bound(work, &measured, ridge_flop_per_byte);
    let achieved_tflops = achieved(work.flops, mean_us, 1.0e12);
    let achieved_gbps = achieved(work.bytes, mean_us, 1.0e9);
    let roofline_peak_pct = peak_pct(
        roofline_bound,
        achieved_tflops,
        achieved_gbps,
        peak_tflops,
        peak_gbps,
    );
    Ok(ReplayMeasurement {
        op: replay_op.op(),
        label,
        measured,
        mean_us,
        flops: work.flops,
        bytes: work.bytes,
        arithmetic_intensity_flop_per_byte,
        roofline_bound,
        achieved_tflops,
        achieved_gbps,
        roofline_peak_pct,
    })
}

fn work_estimate(sample: &RouteSample, replay_op: ReplayOp) -> WorkEstimate {
    match replay_op {
        ReplayOp::Routing => WorkEstimate {
            flops: 0,
            bytes: routing_bytes(sample.recv_capacity, sample.expert_padding),
            bound: ReplayBound::Control,
        },
        ReplayOp::W13 => WorkEstimate {
            flops: gemm_flops(
                sample.padded_rows,
                2 * KIMI_K2_EXPERT_INTERMEDIATE,
                KIMI_K2_HIDDEN,
            ),
            bytes: marlin_gemm_bytes(
                sample.padded_rows,
                2 * KIMI_K2_EXPERT_INTERMEDIATE,
                KIMI_K2_HIDDEN,
                sample.active_local_experts,
            ),
            bound: ReplayBound::Compute,
        },
        ReplayOp::SwiGlu => WorkEstimate {
            flops: 0,
            bytes: swiglu_bytes(sample.padded_rows),
            bound: ReplayBound::Memory,
        },
        ReplayOp::W2 => WorkEstimate {
            flops: gemm_flops(
                sample.padded_rows,
                KIMI_K2_HIDDEN,
                KIMI_K2_EXPERT_INTERMEDIATE,
            ),
            bytes: marlin_gemm_bytes(
                sample.padded_rows,
                KIMI_K2_HIDDEN,
                KIMI_K2_EXPERT_INTERMEDIATE,
                sample.active_local_experts,
            ),
            bound: ReplayBound::Compute,
        },
    }
}

fn attr<'a>(call: &'a KernelCall, name: &str) -> Result<&'a str> {
    call.attrs
        .iter()
        .find(|attr| attr.name == name)
        .map(|attr| attr.value.as_str())
        .with_context(|| format!("{} missing attr `{name}`", call.label))
}

fn attr_usize(call: &KernelCall, name: &str) -> Result<usize> {
    attr(call, name)?
        .parse::<usize>()
        .with_context(|| format!("{} attr `{name}` is not usize", call.label))
}

fn parse_recv_counts(value: &str) -> Result<Vec<i32>> {
    value
        .split(',')
        .enumerate()
        .map(|(idx, part)| {
            let part = part.trim();
            ensure!(!part.is_empty(), "recv_counts entry {idx} is empty");
            let count = part
                .parse::<i32>()
                .with_context(|| format!("recv_counts entry `{part}` is not i32"))?;
            ensure!(count >= 0, "recv_counts entry {idx} is negative: {count}");
            Ok(count)
        })
        .collect()
}

fn counts_csv(counts: &[i32]) -> String {
    counts
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn route_elems_from_counts(counts: &[i32]) -> Result<usize> {
    counts.iter().try_fold(0usize, |acc, &count| {
        let count =
            usize::try_from(count).map_err(|_| anyhow::anyhow!("negative recv count {count}"))?;
        acc.checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("recv count sum overflows usize"))
    })
}

fn padded_rows_from_counts(counts: &[i32], expert_padding: usize) -> Result<usize> {
    ensure!(expert_padding > 0, "expert_padding must be positive");
    counts.iter().try_fold(0usize, |acc, &count| {
        let count =
            usize::try_from(count).map_err(|_| anyhow::anyhow!("negative recv count {count}"))?;
        let padded = if count == 0 {
            0
        } else {
            count.div_ceil(expert_padding) * expert_padding
        };
        acc.checked_add(padded)
            .ok_or_else(|| anyhow::anyhow!("padded row sum overflows usize"))
    })
}

fn gemm_flops(m: usize, n: usize, k: usize) -> u128 {
    2_u128 * m as u128 * n as u128 * k as u128
}

fn marlin_gemm_bytes(m: usize, n: usize, k: usize, active_experts: usize) -> u128 {
    m as u128 * k as u128 * BF16_BYTES as u128
        + int4_weight_bytes(n, k, active_experts)
        + m as u128 * n as u128 * BF16_BYTES as u128
}

fn int4_weight_bytes(out_dim: usize, in_dim: usize, active_experts: usize) -> u128 {
    let packed = active_experts as u128 * out_dim as u128 * in_dim.div_ceil(2) as u128;
    let scales = active_experts as u128
        * out_dim as u128
        * (in_dim / KIMI_K2_INT4_GROUP_SIZE) as u128
        * BF16_BYTES as u128;
    packed + scales
}

fn swiglu_bytes(rows: usize) -> u128 {
    rows as u128 * (3 * KIMI_K2_EXPERT_INTERMEDIATE) as u128 * BF16_BYTES as u128
}

fn routing_bytes(recv_capacity: usize, expert_padding: usize) -> u128 {
    let counts = KIMI_K2_LOCAL_EXPERTS * I32_BYTES;
    let sorted_token_ids = recv_capacity * I32_BYTES;
    let expert_ids = recv_capacity.div_ceil(expert_padding) * I32_BYTES;
    let num_tokens_post_padded = I32_BYTES;
    (counts + sorted_token_ids + expert_ids + num_tokens_post_padded) as u128
}

fn arithmetic_intensity(flops: u128, bytes: u128) -> Option<f64> {
    if flops == 0 || bytes == 0 {
        return None;
    }
    Some(flops as f64 / bytes as f64)
}

fn roofline_bound(
    work: WorkEstimate,
    measured: &MeasuredCall,
    ridge_flop_per_byte: f64,
) -> ReplayBound {
    if !measured.supported {
        return ReplayBound::Unsupported;
    }
    match work.bound {
        ReplayBound::Control => ReplayBound::Control,
        ReplayBound::Memory => ReplayBound::Memory,
        ReplayBound::Compute => match arithmetic_intensity(work.flops, work.bytes) {
            Some(ai) if ai >= ridge_flop_per_byte => ReplayBound::Compute,
            Some(_) => ReplayBound::Memory,
            None => ReplayBound::NoWork,
        },
        ReplayBound::NoWork | ReplayBound::Unsupported => work.bound,
    }
}

fn achieved(work: u128, mean_us: Option<f64>, scale: f64) -> Option<f64> {
    if work == 0 {
        return None;
    }
    let seconds = mean_us? * 1.0e-6;
    (seconds > 0.0).then_some(work as f64 / seconds / scale)
}

fn peak_pct(
    bound: ReplayBound,
    achieved_tflops: Option<f64>,
    achieved_gbps: Option<f64>,
    peak_tflops: f64,
    peak_gbps: f64,
) -> Option<f64> {
    match bound {
        ReplayBound::Compute => achieved_tflops.map(|achieved| achieved / peak_tflops * 100.0),
        ReplayBound::Memory => achieved_gbps.map(|achieved| achieved / peak_gbps * 100.0),
        ReplayBound::Control | ReplayBound::NoWork | ReplayBound::Unsupported => None,
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn print_text(report: &ReplayReport, out: &Path) {
    println!(
        "Kimi PPLX Marlin replay samples={} iters={} min_active_rows={} peak={:.2} TFLOP/s,{:.2} GB/s ridge={:.2} flop/byte",
        report.samples.len(),
        report.config.iters,
        report.config.min_active_rows,
        report.config.peak_tflops,
        report.config.peak_gbps,
        report.config.ridge_flop_per_byte
    );
    println!("wrote {}", out.display());
    println!(
        "{:<8} {:>5} {:>4} {:>6} {:>6} {:>6} {:>7} {:<45} {:>9} {:>8} {:>10} {:>8} {:>10}",
        "q",
        "layer",
        "rank",
        "active",
        "recv",
        "padded",
        "experts",
        "op",
        "mean_us",
        "AI",
        "GB/s",
        "%peak",
        "bound"
    );
    for sample_report in &report.samples {
        let sample = &sample_report.sample;
        let quantiles = sample.quantiles.join("+");
        for measurement in &sample_report.measured {
            println!(
                "{:<8} {:>5} {:>4} {:>6} {:>6} {:>6} {:>7} {:<45} {:>9} {:>8} {:>10} {:>8} {:>10}",
                quantiles,
                sample.layer_idx,
                sample.rank,
                sample.active_rows,
                sample.recv_total_routes,
                sample.padded_rows,
                sample.active_local_experts,
                measurement.op,
                display_opt(measurement.mean_us),
                display_opt(measurement.arithmetic_intensity_flop_per_byte),
                display_opt(measurement.achieved_gbps),
                display_opt(measurement.roofline_peak_pct),
                format!("{:?}", measurement.roofline_bound),
            );
        }
    }
}

fn display_opt(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.2}"))
}
