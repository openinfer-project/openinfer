use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use pegainfer_bench::MeasuredCall;
use pegainfer_kernels::ops::{
    KIMI_K2_MLA_KV_LORA_RANK, KIMI_K2_MLA_NOPE_DIM, KIMI_K2_MLA_Q_HEAD_DIM, KIMI_K2_MLA_ROPE_DIM,
    KIMI_K2_MLA_V_HEAD_DIM,
};
use pegainfer_kernels::tensor::{AxisSpec, Bf16, Contiguous1D, F32, HiddenStatesLayout, I32, U32};
use pegainfer_kernels::tensor::{KernelCall, TensorSpec};
use pegainfer_kimi_k2::{
    kernel_report::measure_call,
    tp1_pplx_decode_bench::{
        BenchSpec, BoundKind, MeasureKind, TP1_PPLX_ARENA_ROWS, default_active_rows,
        default_ctx_lens, specs,
    },
};
use serde::Serialize;

const DEFAULT_ITERS: u64 = 32;
const DEFAULT_H20_BF16_TFLOPS: f64 = 148.0;
const DEFAULT_H20_HBM_GBPS: f64 = 4_800.0;

#[derive(Parser)]
#[command(about = "Kimi-K2 TP1 DP8 PPLX decode operator bench")]
struct Cli {
    #[arg(long = "active-rows")]
    active_rows: Option<String>,
    #[arg(long = "ctx-lens")]
    ctx_lens: Option<String>,
    #[arg(long, default_value_t = DEFAULT_ITERS)]
    iters: u64,
    #[arg(long, default_value = "text")]
    format: String,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    measure: bool,
    #[arg(long)]
    ridge_flop_per_byte: Option<f64>,
    #[arg(long, default_value_t = DEFAULT_H20_BF16_TFLOPS)]
    peak_tflops: f64,
    #[arg(long, default_value_t = DEFAULT_H20_HBM_GBPS)]
    peak_gbps: f64,
}

#[derive(Serialize)]
struct BenchReport {
    schema: u32,
    report_type: &'static str,
    rank_scope: &'static str,
    config: BenchConfig,
    rows: Vec<BenchRow>,
}

#[derive(Serialize)]
struct BenchConfig {
    active_rows: Vec<usize>,
    ctx_lens: Vec<usize>,
    arena_rows: usize,
    iters: u64,
    measure: bool,
    ridge_flop_per_byte: f64,
    peak_tflops: f64,
    peak_gbps: f64,
}

#[derive(Serialize)]
struct BenchRow {
    spec: BenchSpec,
    measured: MeasuredCall,
    total_mean_us: Option<f64>,
    arithmetic_intensity_flop_per_byte: Option<f64>,
    roofline_bound: RooflineBound,
    achieved_tflops: Option<f64>,
    achieved_gbps: Option<f64>,
    roofline_peak_pct: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RooflineBound {
    Compute,
    Memory,
    Control,
    Comm,
    NoWork,
    EstimateOnly,
}

fn main() -> Result<()> {
    pegainfer_core::logging::init_default();
    let cli = Cli::parse();
    if cli.iters == 0 {
        bail!("--iters must be greater than zero");
    }
    if cli.format != "text" && cli.format != "json" {
        bail!("--format must be `text` or `json`");
    }
    if !(cli.peak_tflops.is_finite() && cli.peak_tflops > 0.0) {
        bail!("--peak-tflops must be a positive finite number");
    }
    if !(cli.peak_gbps.is_finite() && cli.peak_gbps > 0.0) {
        bail!("--peak-gbps must be a positive finite number");
    }
    let ridge_flop_per_byte = cli
        .ridge_flop_per_byte
        .unwrap_or(cli.peak_tflops * 1_000.0 / cli.peak_gbps);
    if !(ridge_flop_per_byte.is_finite() && ridge_flop_per_byte > 0.0) {
        bail!("derived --ridge-flop-per-byte must be a positive finite number");
    }
    let active_rows = cli
        .active_rows
        .as_deref()
        .map(parse_usize_csv)
        .transpose()?
        .unwrap_or_else(|| default_active_rows().to_vec());
    let ctx_lens = cli
        .ctx_lens
        .as_deref()
        .map(parse_usize_csv)
        .transpose()?
        .unwrap_or_else(|| default_ctx_lens().to_vec());
    validate_rows(&active_rows)?;
    validate_ctx_lens(&ctx_lens)?;

    let mut rows = Vec::new();
    for &ctx_len in &ctx_lens {
        for &active in &active_rows {
            for spec in specs(active, TP1_PPLX_ARENA_ROWS, ctx_len) {
                let measured = if cli.measure {
                    measure_spec(&spec, cli.iters)?
                } else {
                    unsupported("measurement disabled")
                };
                rows.push(row_from_measurement(
                    spec,
                    measured,
                    ridge_flop_per_byte,
                    cli.peak_tflops,
                    cli.peak_gbps,
                ));
            }
        }
    }

    let report = BenchReport {
        schema: 1,
        report_type: "kimi_tp1_pplx_decode_bench",
        rank_scope: "single TP1 DP rank local compute plus PPLX comm accounting; comm timing needs all-rank harness",
        config: BenchConfig {
            active_rows,
            ctx_lens,
            arena_rows: TP1_PPLX_ARENA_ROWS,
            iters: cli.iters,
            measure: cli.measure,
            ridge_flop_per_byte,
            peak_tflops: cli.peak_tflops,
            peak_gbps: cli.peak_gbps,
        },
        rows,
    };

    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from("target/kernel_reports/kimi-k2/tp1-pplx-decode-bench.json")
    });
    write_json(&out, &report)?;
    match cli.format.as_str() {
        "json" => println!("{}", serde_json::to_string_pretty(&report)?),
        "text" => print_text(&report, &out),
        _ => unreachable!("format validated"),
    }
    Ok(())
}

fn parse_usize_csv(value: &str) -> Result<Vec<usize>> {
    value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .with_context(|| format!("invalid usize `{part}`"))
        })
        .collect()
}

fn validate_rows(rows: &[usize]) -> Result<()> {
    if rows.is_empty() {
        bail!("active row list must not be empty");
    }
    for &row in rows {
        if !(1..=TP1_PPLX_ARENA_ROWS).contains(&row) {
            bail!("active row {row} must be in 1..={TP1_PPLX_ARENA_ROWS}");
        }
    }
    Ok(())
}

fn validate_ctx_lens(ctx_lens: &[usize]) -> Result<()> {
    if ctx_lens.is_empty() {
        bail!("ctx-len list must not be empty");
    }
    for &ctx_len in ctx_lens {
        if ctx_len == 0 {
            bail!("ctx-len must be greater than zero");
        }
    }
    Ok(())
}

fn measure_spec(spec: &BenchSpec, iters: u64) -> Result<MeasuredCall> {
    if spec.measure != MeasureKind::ExistingProvider {
        return Ok(unsupported("estimate-only spec has no local provider"));
    }
    let Some(call) = kernel_call(spec) else {
        return Ok(unsupported("no KernelCall adapter for TP1 PPLX spec"));
    };
    measure_call(&call, iters)
        .with_context(|| format!("failed to measure {} ({})", spec.label, spec.op))
}

fn row_from_measurement(
    spec: BenchSpec,
    measured: MeasuredCall,
    ridge_flop_per_byte: f64,
    peak_tflops: f64,
    peak_gbps: f64,
) -> BenchRow {
    let total_mean_us = measured
        .stats
        .as_ref()
        .map(|stats| stats.mean_us * spec.calls_per_decode_step as f64);
    let arithmetic_intensity_flop_per_byte = arithmetic_intensity(&spec);
    let roofline_bound = roofline_bound(&spec, &measured, ridge_flop_per_byte);
    let achieved_tflops = achieved(spec.flops_per_decode_step, total_mean_us, 1.0e12);
    let achieved_gbps = achieved(spec.bytes_per_decode_step, total_mean_us, 1.0e9);
    let roofline_peak_pct = peak_pct(
        roofline_bound,
        achieved_tflops,
        achieved_gbps,
        peak_tflops,
        peak_gbps,
    );
    BenchRow {
        spec,
        measured,
        total_mean_us,
        arithmetic_intensity_flop_per_byte,
        roofline_bound,
        achieved_tflops,
        achieved_gbps,
        roofline_peak_pct,
    }
}

fn arithmetic_intensity(spec: &BenchSpec) -> Option<f64> {
    if spec.flops_per_decode_step == 0 || spec.bytes_per_decode_step == 0 {
        return None;
    }
    Some(spec.flops_per_decode_step as f64 / spec.bytes_per_decode_step as f64)
}

fn roofline_bound(
    spec: &BenchSpec,
    measured: &MeasuredCall,
    ridge_flop_per_byte: f64,
) -> RooflineBound {
    match spec.bound {
        BoundKind::Comm => return RooflineBound::Comm,
        BoundKind::Control => return RooflineBound::Control,
        BoundKind::Compute | BoundKind::Memory | BoundKind::Mixed => {}
    }
    if !measured.supported {
        return RooflineBound::EstimateOnly;
    }
    match (
        spec.flops_per_decode_step,
        spec.bytes_per_decode_step,
        arithmetic_intensity(spec),
    ) {
        (0, 0, _) => RooflineBound::NoWork,
        (0, _, _) => RooflineBound::Memory,
        (_, 0, _) => RooflineBound::Compute,
        (_, _, Some(ai)) if ai >= ridge_flop_per_byte => RooflineBound::Compute,
        (_, _, Some(_)) => RooflineBound::Memory,
        _ => RooflineBound::NoWork,
    }
}

fn achieved(work: u128, total_mean_us: Option<f64>, scale: f64) -> Option<f64> {
    if work == 0 {
        return None;
    }
    let seconds = total_mean_us? * 1.0e-6;
    (seconds > 0.0).then_some(work as f64 / seconds / scale)
}

fn peak_pct(
    bound: RooflineBound,
    achieved_tflops: Option<f64>,
    achieved_gbps: Option<f64>,
    peak_tflops: f64,
    peak_gbps: f64,
) -> Option<f64> {
    match bound {
        RooflineBound::Compute => achieved_tflops.map(|achieved| achieved / peak_tflops * 100.0),
        RooflineBound::Memory => achieved_gbps.map(|achieved| achieved / peak_gbps * 100.0),
        RooflineBound::Control
        | RooflineBound::Comm
        | RooflineBound::NoWork
        | RooflineBound::EstimateOnly => None,
    }
}

fn unsupported(reason: impl Into<String>) -> MeasuredCall {
    MeasuredCall {
        supported: false,
        reason: Some(reason.into()),
        stats: None,
    }
}

fn kernel_call(spec: &BenchSpec) -> Option<KernelCall> {
    match spec.op {
        "gemm_graphsafe" => {
            let (rows, out_dim, in_dim) = (spec.m?, spec.n?, spec.k?);
            Some(
                KernelCall::new("gemm_graphsafe", spec.label)
                    .input("weight", weight(out_dim, in_dim))
                    .input("x", hidden(in_dim, rows))
                    .output("out", hidden(out_dim, rows)),
            )
        }
        "gemm_dm_typed_to_hs_graphsafe" => {
            let (rows, out_dim, in_dim) = (spec.m?, spec.n?, spec.k?);
            Some(
                KernelCall::new("gemm_dm_typed_to_hs_graphsafe", spec.label)
                    .input("weight", weight(out_dim, in_dim))
                    .input("x", hidden(in_dim, rows))
                    .output("out", hidden(out_dim, rows)),
            )
        }
        "gemm_dm_hs_to_typed_graphsafe" => {
            let (rows, out_dim, in_dim) = (spec.m?, spec.n?, spec.k?);
            Some(
                KernelCall::new("gemm_dm_hs_to_typed_graphsafe", spec.label)
                    .input("weight", weight(out_dim, in_dim))
                    .input("x", hidden(in_dim, rows))
                    .output("out", hidden(out_dim, rows)),
            )
        }
        "rms_norm_batch" => {
            let (hidden_dim, batch) = hidden_batch_or_arena(spec);
            Some(
                KernelCall::new("rms_norm_batch", spec.label)
                    .input("x", hidden(hidden_dim, batch))
                    .input("weight", vector_bf16("hidden", hidden_dim))
                    .output("out", hidden(hidden_dim, batch)),
            )
        }
        "fused_add_rms_norm_round_batch" => {
            let (hidden_dim, batch) = hidden_batch_or_arena(spec);
            Some(
                KernelCall::new("fused_add_rms_norm_round_batch", spec.label)
                    .input("hidden", hidden(hidden_dim, batch))
                    .input("residual", hidden(hidden_dim, batch))
                    .input("weight", vector_bf16("hidden", hidden_dim))
                    .output("hidden", hidden(hidden_dim, batch))
                    .output("normed", hidden(hidden_dim, batch)),
            )
        }
        "silu_mul_batch" => {
            let (hidden_dim, batch) = (spec.m?, spec.n?);
            Some(
                KernelCall::new("silu_mul_batch", spec.label)
                    .input("gate", hidden(hidden_dim, batch))
                    .input("up", hidden(hidden_dim, batch))
                    .output("out", hidden(hidden_dim, batch)),
            )
        }
        "silu_mul_hs_fused_into" => {
            let batch = spec.n.unwrap_or(spec.active_rows);
            let inter = spec.m.unwrap_or(crate_expert_intermediate());
            Some(
                KernelCall::new("silu_mul_hs_fused_into", spec.label)
                    .input("gate_up", hidden(2 * inter, batch))
                    .output("out", hidden(inter, batch)),
            )
        }
        "add_batch" => {
            let (hidden_dim, batch) = hidden_batch_or_arena(spec);
            Some(
                KernelCall::new("add_batch", spec.label)
                    .input("a", hidden(hidden_dim, batch))
                    .input("b", hidden(hidden_dim, batch))
                    .output("out", hidden(hidden_dim, batch)),
            )
        }
        "embedding_batch_vocab_shard" => {
            let (vocab, hidden_dim, batch) = (crate_vocab(), crate_hidden(), spec.arena_rows);
            Some(
                KernelCall::new("embedding_batch_vocab_shard", spec.label)
                    .input("weight", weight(vocab, hidden_dim))
                    .input(
                        "token_ids",
                        TensorSpec::new::<U32, Contiguous1D>([AxisSpec::named("batch", batch)]),
                    )
                    .output("out", hidden(hidden_dim, batch)),
            )
        }
        "kimi_mla_split_qkv_a_norm" => {
            let batch = spec.arena_rows;
            Some(
                KernelCall::new("kimi_mla_split_qkv_a_norm", spec.label)
                    .input("qkv_a", hidden(crate_qkv_a_out(), batch))
                    .output("q_a_normed", hidden(crate_q_lora_rank(), batch))
                    .output(
                        "compressed_kv_normed",
                        hidden(KIMI_K2_MLA_KV_LORA_RANK, batch),
                    )
                    .output("k_rope", hidden(KIMI_K2_MLA_ROPE_DIM, batch)),
            )
        }
        "kimi_mla_rope_split_decode_rt" => {
            let batch = spec.arena_rows;
            let local_heads = crate_local_heads();
            Some(
                KernelCall::new("kimi_mla_rope_split_decode_rt", spec.label)
                    .input(
                        "q_proj",
                        hidden(local_heads * KIMI_K2_MLA_Q_HEAD_DIM, batch),
                    )
                    .input("k_rope", hidden(KIMI_K2_MLA_ROPE_DIM, batch))
                    .output("q_nope", hidden(local_heads * KIMI_K2_MLA_NOPE_DIM, batch))
                    .output("q_pe", hidden(local_heads * KIMI_K2_MLA_ROPE_DIM, batch))
                    .output("append_kpe", hidden(KIMI_K2_MLA_ROPE_DIM, batch)),
            )
        }
        "kimi_mla_absorb_q_nope_rt" => {
            let batch = spec.arena_rows;
            let local_heads = crate_local_heads();
            Some(
                KernelCall::new("kimi_mla_absorb_q_nope_rt", spec.label)
                    .input("q_nope", hidden(local_heads * KIMI_K2_MLA_NOPE_DIM, batch))
                    .input(
                        "kv_b_proj",
                        weight(
                            local_heads * (KIMI_K2_MLA_NOPE_DIM + KIMI_K2_MLA_V_HEAD_DIM),
                            KIMI_K2_MLA_KV_LORA_RANK,
                        ),
                    )
                    .output(
                        "q_abs_nope",
                        hidden(local_heads * KIMI_K2_MLA_KV_LORA_RANK, batch),
                    ),
            )
        }
        "kimi_flashinfer_batch_decode_mla_rt" => {
            let batch = spec.arena_rows;
            let local_heads = crate_local_heads();
            Some(
                KernelCall::new("kimi_flashinfer_batch_decode_mla_rt", spec.label)
                    .input(
                        "q_abs_nope",
                        hidden(local_heads * KIMI_K2_MLA_KV_LORA_RANK, batch),
                    )
                    .input("q_pe", hidden(local_heads * KIMI_K2_MLA_ROPE_DIM, batch))
                    .output(
                        "latent",
                        hidden(local_heads * KIMI_K2_MLA_KV_LORA_RANK, batch),
                    )
                    .attr("kv_len", spec.ctx_len.to_string()),
            )
        }
        "kimi_mla_v_up_rt" => {
            let batch = spec.arena_rows;
            let local_heads = crate_local_heads();
            Some(
                KernelCall::new("kimi_mla_v_up_rt", spec.label)
                    .input(
                        "latent",
                        hidden(local_heads * KIMI_K2_MLA_KV_LORA_RANK, batch),
                    )
                    .input(
                        "kv_b_proj",
                        weight(
                            local_heads * (KIMI_K2_MLA_NOPE_DIM + KIMI_K2_MLA_V_HEAD_DIM),
                            KIMI_K2_MLA_KV_LORA_RANK,
                        ),
                    )
                    .output("out", hidden(local_heads * KIMI_K2_MLA_V_HEAD_DIM, batch)),
            )
        }
        "top1_batch" => {
            let (vocab, batch) = (spec.m?, spec.n?);
            Some(
                KernelCall::new("top1_batch", spec.label)
                    .input("logits", hidden(vocab, batch))
                    .output(
                        "token_ids",
                        TensorSpec::new::<I32, Contiguous1D>([AxisSpec::named("batch", batch)]),
                    ),
            )
        }
        "argmax_batch_bf16" => {
            let batch = spec.active_rows;
            Some(
                KernelCall::new("argmax_batch_bf16", spec.label)
                    .input("logits", hidden(crate_vocab(), batch))
                    .output(
                        "values",
                        TensorSpec::new::<Bf16, Contiguous1D>([AxisSpec::named("batch", batch)]),
                    )
                    .output(
                        "token_ids",
                        TensorSpec::new::<I32, Contiguous1D>([AxisSpec::named("batch", batch)]),
                    ),
            )
        }
        "kimi_router_noaux_tc" => {
            let batch = spec.active_rows;
            Some(
                KernelCall::new("kimi_router_noaux_tc", spec.label)
                    .input("hidden", hidden(crate_hidden(), batch))
                    .input(
                        "gate_weight",
                        weight(crate_routed_experts(), crate_hidden()),
                    )
                    .output(
                        "topk_weight",
                        TensorSpec::new::<F32, Contiguous1D>([AxisSpec::named(
                            "route",
                            batch * crate_topk(),
                        )]),
                    )
                    .output(
                        "topk_idx",
                        TensorSpec::new::<I32, Contiguous1D>([AxisSpec::named(
                            "route",
                            batch * crate_topk(),
                        )]),
                    ),
            )
        }
        "kimi_residual_add_scaled_f32" => {
            let batch = spec.active_rows;
            Some(
                KernelCall::new("kimi_residual_add_scaled_f32", spec.label)
                    .input("hidden", hidden(crate_hidden(), batch))
                    .input("projected", hidden(crate_hidden(), batch))
                    .input(
                        "routed_f32",
                        TensorSpec::new::<F32, Contiguous1D>([AxisSpec::named(
                            "elem",
                            batch * crate_hidden(),
                        )]),
                    )
                    .output("out", hidden(crate_hidden(), batch))
                    .attr("scale", "2.827".to_string()),
            )
        }
        _ => None,
    }
}

fn hidden_batch_or_arena(spec: &BenchSpec) -> (usize, usize) {
    (
        spec.m.unwrap_or(crate_hidden()),
        spec.n.unwrap_or(spec.arena_rows),
    )
}

fn hidden(hidden_dim: usize, batch: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, HiddenStatesLayout>([
        AxisSpec::named("hidden", hidden_dim),
        AxisSpec::named("batch", batch),
    ])
}

fn weight(out_dim: usize, in_dim: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, pegainfer_kernels::tensor::RowMajor2D>([
        AxisSpec::named("out", out_dim),
        AxisSpec::named("in", in_dim),
    ])
}

fn vector_bf16(axis: &str, len: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, Contiguous1D>([AxisSpec::named(axis, len)])
}

fn crate_hidden() -> usize {
    7168
}

fn crate_vocab() -> usize {
    163_840
}

fn crate_local_heads() -> usize {
    64
}

fn crate_q_lora_rank() -> usize {
    1536
}

fn crate_qkv_a_out() -> usize {
    crate_q_lora_rank() + KIMI_K2_MLA_KV_LORA_RANK + KIMI_K2_MLA_ROPE_DIM
}

fn crate_expert_intermediate() -> usize {
    2048
}

fn crate_routed_experts() -> usize {
    384
}

fn crate_topk() -> usize {
    8
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn print_text(report: &BenchReport, out: &Path) {
    println!(
        "Kimi TP1 PPLX decode bench rows={} arena_rows={} iters={} measure={} peak={:.2} TFLOP/s,{:.2} GB/s ridge={:.2} flop/byte",
        report.rows.len(),
        report.config.arena_rows,
        report.config.iters,
        report.config.measure,
        report.config.peak_tflops,
        report.config.peak_gbps,
        report.config.ridge_flop_per_byte
    );
    println!("wrote {}", out.display());
    println!(
        "{:>3} {:>5} {:<18} {:<34} {:<28} {:>7} {:>7} {:>10} {:>10} {:>8} {:>10} {:>10} note",
        "bs",
        "ctx",
        "stage",
        "op",
        "shape",
        "us",
        "AI",
        "TFLOP/s",
        "GB/s",
        "%peak",
        "kind",
        "roofline"
    );
    for row in &report.rows {
        let mean_us = display_opt(row.total_mean_us);
        let ai = display_opt(row.arithmetic_intensity_flop_per_byte);
        let tflops = display_opt(row.achieved_tflops);
        let gbps = display_opt(row.achieved_gbps);
        let peak_pct = display_opt(row.roofline_peak_pct);
        let stage = format!("{:?}", row.spec.stage);
        let bound = format!("{:?}", row.spec.bound);
        let roofline = format!("{:?}", row.roofline_bound);
        let shape = row.spec.shape.as_deref().unwrap_or("-");
        let note = match &row.measured.reason {
            Some(reason) if row.spec.notes.is_empty() => reason.clone(),
            Some(reason) => format!("{}; measurement: {reason}", row.spec.notes),
            None => row.spec.notes.clone(),
        };
        println!(
            "{:>3} {:>5} {:<18} {:<34} {:<28} {:>7} {:>7} {:>10} {:>10} {:>8} {:>10} {:>10} {}",
            row.spec.active_rows,
            row.spec.ctx_len,
            stage,
            row.spec.label,
            shape,
            mean_us,
            ai,
            tflops,
            gbps,
            peak_pct,
            bound,
            roofline,
            note
        );
    }
}

fn display_opt(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.2}"))
}
