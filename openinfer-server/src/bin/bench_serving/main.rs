//! In-process inference benchmark CLI.
//!
//! Usage:
//!   cargo run -r --bin bench_serving -- [GLOBAL_OPTIONS] <SUBCOMMAND> [OPTIONS]
//!
//! Examples:
//!   cargo run -r --bin bench_serving -- request --prompt "Tell me a story" --output-len 128
//!   cargo run -r --bin bench_serving -- request --prompt-len 512 --output-len 64
//!   cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512 --output-lens 32,128
//!   cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use comfy_table::{Cell, CellAlignment};
use log::{debug, info};
use openinfer::logging;
use openinfer::scheduler::SchedulerHandle;
use openinfer::server_engine::{ModelType, detect_model_type};
use openinfer_core::engine::{EngineLoadOptions, EpBackend};
#[cfg(feature = "kimi-k2")]
use openinfer_core::parallel::ParallelConfig;
use openinfer_vllm_support::load_tokenizer as load_vllm_tokenizer;
use vllm_text::tokenizer::DynTokenizer;

mod cli;
mod exec;
mod metrics;
mod prompt;
mod render;
mod report;
mod runners;
use cli::*;
use exec::*;
use metrics::*;
use prompt::*;
use render::*;
use report::*;
use runners::*;

const SNAPSHOT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench_snapshots");
const SNAPSHOT_PREFILL_OUTPUT_LEN: usize = 1;
const SNAPSHOT_DECODE_PROMPT_LEN: usize = 1024;
const SNAPSHOT_DECODE_OUTPUT_LEN: usize = 256;

fn snapshot_prefill_prompt_len(model_type: ModelType) -> usize {
    match model_type {
        // Kimi serves TP1/DP8, where the PPLX fabric buffers cap prompts at
        // 2048 tokens (full-lifetime KV cap is 8192) — probe the largest
        // prompt the serving shape admits.
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => 2_048,
        _ => 10_000,
    }
}
const REGRESSION_TPOT_PCT: f64 = 2.0;
const REGRESSION_TTFT_PCT: f64 = 3.0;

fn command_seed(cli: &Cli) -> u64 {
    match &cli.command {
        Command::Request(args) => args.run.seed,
        Command::Matrix(args) => args.run.seed,
        Command::Curve(args) => args.run.seed,
        Command::Snapshot(args) => args.run.seed,
        Command::Compare(_) => 42,
    }
}

#[cfg(feature = "kimi-k2")]
fn kimi_parallel_config(tp_size: usize, dp_size: usize) -> Result<ParallelConfig> {
    ensure!(tp_size > 0, "--tp-size must be positive");
    ensure!(dp_size > 0, "--dp-size must be positive");
    Ok(ParallelConfig::new(tp_size, dp_size))
}

fn shell_output(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

fn git_short_commit() -> String {
    shell_output("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into())
}

fn gpu_name() -> String {
    shell_output(
        "nvidia-smi",
        &["--query-gpu=name", "--format=csv,noheader", "--id=0"],
    )
    .unwrap_or_else(|| "unknown".into())
}

/// Produce a filesystem-safe slug from a GPU name string.
///
/// `"NVIDIA GeForce RTX 5070 Ti"` → `"rtx-5070-ti"`
fn gpu_slug_from(name: &str) -> String {
    let stripped = name
        .strip_prefix("NVIDIA GeForce ")
        .or_else(|| name.strip_prefix("NVIDIA "))
        .unwrap_or(name);
    stripped
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

fn today_date() -> String {
    shell_output("date", &["+%Y-%m-%d"]).unwrap_or_else(|| "unknown".into())
}

fn model_display_name(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn delta_pct(current: f64, baseline: f64) -> f64 {
    if baseline == 0.0 {
        return 0.0;
    }
    (current - baseline) / baseline * 100.0
}

fn format_delta(pct: f64) -> String {
    if pct >= 0.0 {
        format!("+{pct:.1}%")
    } else {
        format!("{pct:.1}%")
    }
}

fn run_snapshot(
    model: &mut dyn BenchModel,
    cli: &Cli,
    model_type: ModelType,
    args: &SnapshotArgs,
) -> Result<()> {
    let prefill_prompt_len = snapshot_prefill_prompt_len(model_type);

    info!("Running prefill-heavy ({prefill_prompt_len},{SNAPSHOT_PREFILL_OUTPUT_LEN})");
    let prefill_tokens = synthetic_prompt_tokens(prefill_prompt_len);
    let prefill_timings = measure_timings(
        model,
        std::slice::from_ref(&prefill_tokens),
        SNAPSHOT_PREFILL_OUTPUT_LEN,
        &args.run,
        cli.cuda_profiler_capture,
    )?;
    let prefill_metrics = build_request_metrics(&prefill_timings);

    info!("Running decode-heavy ({SNAPSHOT_DECODE_PROMPT_LEN},{SNAPSHOT_DECODE_OUTPUT_LEN})");
    let decode_tokens = synthetic_prompt_tokens(SNAPSHOT_DECODE_PROMPT_LEN);
    let decode_timings = measure_timings(
        model,
        std::slice::from_ref(&decode_tokens),
        SNAPSHOT_DECODE_OUTPUT_LEN,
        &args.run,
        cli.cuda_profiler_capture,
    )?;
    let decode_metrics = build_request_metrics(&decode_timings);

    let model_name = model_display_name(&cli.model_path);
    let gpu = gpu_name();
    let parallel = match model_type {
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => Some(format!(
            "tp{}-dp{}-{}",
            cli.tp_size,
            cli.dp_size,
            format!("{:?}", cli.ep_backend).to_lowercase()
        )),
        _ => None,
    };
    let report = SnapshotReport {
        commit: git_short_commit(),
        date: today_date(),
        model: model_name.clone(),
        gpu: gpu.clone(),
        parallel,
        prefill_heavy: SnapshotProfile {
            prompt_len: prefill_prompt_len,
            output_len: SNAPSHOT_PREFILL_OUTPUT_LEN,
            metrics: prefill_metrics,
        },
        decode_heavy: SnapshotProfile {
            prompt_len: SNAPSHOT_DECODE_PROMPT_LEN,
            output_len: SNAPSHOT_DECODE_OUTPUT_LEN,
            metrics: decode_metrics,
        },
    };

    let dir = Path::new(SNAPSHOT_DIR).join(gpu_slug_from(&gpu));
    fs::create_dir_all(&dir)?;
    let filename = model_name.to_lowercase();
    let path = dir.join(format!("{filename}.json"));
    let snapshot_json = serde_json::to_string_pretty(&report)?;
    fs::write(&path, format!("{snapshot_json}\n"))?;

    println!("{}", render_snapshot_text(&report, &path));
    Ok(())
}

fn render_snapshot_text(report: &SnapshotReport, path: &Path) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "bench_serving snapshot\n");
    let _ = writeln!(out, "model:  {}", report.model);
    let _ = writeln!(out, "gpu:    {}", report.gpu);
    if let Some(parallel) = &report.parallel {
        let _ = writeln!(out, "shape:  {parallel}");
    }
    let _ = writeln!(out, "commit: {}\n", report.commit);
    let _ = writeln!(
        out,
        "prefill_heavy ({},{}):",
        report.prefill_heavy.prompt_len, report.prefill_heavy.output_len
    );
    let _ = writeln!(
        out,
        "  TTFT  p50={:.2}ms  p99={:.2}ms",
        report.prefill_heavy.metrics.ttft_ms.p50_ms, report.prefill_heavy.metrics.ttft_ms.p99_ms
    );
    let _ = writeln!(
        out,
        "\ndecode_heavy ({},{}):",
        report.decode_heavy.prompt_len, report.decode_heavy.output_len
    );
    if let Some(tpot) = &report.decode_heavy.metrics.steady_tpot_ms {
        let _ = writeln!(
            out,
            "  TPOT  p50={:.2}ms  p99={:.2}ms",
            tpot.p50_ms, tpot.p99_ms
        );
    }
    let _ = writeln!(out, "\nwritten to {}", path.display());
    out
}

fn run_compare(args: &CompareArgs) -> Result<()> {
    let current_content = fs::read_to_string(&args.path).with_context(|| {
        format!(
            "snapshot not found: {}\nrun `bench_serving snapshot` first",
            args.path
        )
    })?;
    let current: SnapshotReport =
        serde_json::from_str(&current_content).context("failed to parse current snapshot")?;

    // Resolve repo-root-relative path for git show
    let abs_path = fs::canonicalize(&args.path)?;
    let toplevel =
        shell_output("git", &["rev-parse", "--show-toplevel"]).context("not a git repository")?;
    let root = PathBuf::from(&toplevel);
    let rel_path = abs_path
        .strip_prefix(&root)
        .context("snapshot file is outside the git repository")?;

    let git_output = std::process::Command::new("git")
        .args(["show", &format!("{}:{}", args.baseline, rel_path.display())])
        .output()
        .context("failed to run git show")?;

    if !git_output.status.success() {
        anyhow::bail!(
            "no baseline at {}:{}\ncommit the current snapshot to establish a baseline",
            args.baseline,
            rel_path.display()
        );
    }

    let baseline: SnapshotReport =
        serde_json::from_slice(&git_output.stdout).context("failed to parse baseline snapshot")?;

    // Guard against comparing snapshots with different profile shapes
    ensure!(
        current.prefill_heavy.prompt_len == baseline.prefill_heavy.prompt_len
            && current.prefill_heavy.output_len == baseline.prefill_heavy.output_len
            && current.decode_heavy.prompt_len == baseline.decode_heavy.prompt_len
            && current.decode_heavy.output_len == baseline.decode_heavy.output_len,
        "profile shape mismatch: current ({},{}) + ({},{}) vs baseline ({},{}) + ({},{})\n\
         the snapshot profiles were changed — re-baseline by committing a fresh snapshot",
        current.prefill_heavy.prompt_len,
        current.prefill_heavy.output_len,
        current.decode_heavy.prompt_len,
        current.decode_heavy.output_len,
        baseline.prefill_heavy.prompt_len,
        baseline.prefill_heavy.output_len,
        baseline.decode_heavy.prompt_len,
        baseline.decode_heavy.output_len,
    );
    println!("{}", render_comparison(&current, &baseline, &args.baseline));
    Ok(())
}

fn render_comparison(
    current: &SnapshotReport,
    baseline: &SnapshotReport,
    ref_name: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "bench_serving compare\n");
    let _ = writeln!(
        out,
        "comparing {} (working tree) vs {} ({ref_name})\n",
        current.commit, baseline.commit
    );

    let mut table = new_table();
    table.set_header(vec![
        Cell::new("metric"),
        Cell::new("current").set_alignment(CellAlignment::Right),
        Cell::new("baseline").set_alignment(CellAlignment::Right),
        Cell::new("delta").set_alignment(CellAlignment::Right),
    ]);

    let pf = &current.prefill_heavy;
    let pf_b = &baseline.prefill_heavy;
    let pf_label = format!("({},{})", pf.prompt_len, pf.output_len);

    for (stat, cur, base) in [
        (
            "p50",
            pf.metrics.ttft_ms.p50_ms,
            pf_b.metrics.ttft_ms.p50_ms,
        ),
        (
            "p99",
            pf.metrics.ttft_ms.p99_ms,
            pf_b.metrics.ttft_ms.p99_ms,
        ),
    ] {
        table.add_row(vec![
            key_cell(format!("TTFT {stat} {pf_label}")),
            numeric_cell(format!("{cur:.2}ms")),
            numeric_cell(format!("{base:.2}ms")),
            numeric_cell(format_delta(delta_pct(cur, base))),
        ]);
    }

    let dc_label = format!(
        "({},{})",
        current.decode_heavy.prompt_len, current.decode_heavy.output_len
    );
    if let (Some(cur_tpot), Some(base_tpot)) = (
        &current.decode_heavy.metrics.steady_tpot_ms,
        &baseline.decode_heavy.metrics.steady_tpot_ms,
    ) {
        for (stat, cur, base) in [
            ("p50", cur_tpot.p50_ms, base_tpot.p50_ms),
            ("p99", cur_tpot.p99_ms, base_tpot.p99_ms),
        ] {
            table.add_row(vec![
                key_cell(format!("TPOT {stat} {dc_label}")),
                numeric_cell(format!("{cur:.2}ms")),
                numeric_cell(format!("{base:.2}ms")),
                numeric_cell(format_delta(delta_pct(cur, base))),
            ]);
        }
    }

    push_table(&mut out, &table);

    // Regression check
    let mut regressions = Vec::new();
    let ttft_d = delta_pct(
        current.prefill_heavy.metrics.ttft_ms.p50_ms,
        baseline.prefill_heavy.metrics.ttft_ms.p50_ms,
    );
    if ttft_d > REGRESSION_TTFT_PCT {
        regressions.push(format!(
            "TTFT p50 {ttft_d:+.1}% > {REGRESSION_TTFT_PCT}% threshold"
        ));
    }
    if let (Some(cur), Some(base)) = (
        &current.decode_heavy.metrics.steady_tpot_ms,
        &baseline.decode_heavy.metrics.steady_tpot_ms,
    ) {
        let tpot_d = delta_pct(cur.p50_ms, base.p50_ms);
        if tpot_d > REGRESSION_TPOT_PCT {
            regressions.push(format!(
                "TPOT p50 {tpot_d:+.1}% > {REGRESSION_TPOT_PCT}% threshold"
            ));
        }
    }

    out.push('\n');
    if regressions.is_empty() {
        let _ = writeln!(
            out,
            "no regression detected (threshold: TPOT >{REGRESSION_TPOT_PCT}%, TTFT >{REGRESSION_TTFT_PCT}%)"
        );
    } else {
        let _ = writeln!(out, "REGRESSION DETECTED:");
        for r in &regressions {
            let _ = writeln!(out, "  {r}");
        }
    }

    out
}

fn dispatch(
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    model: &mut dyn BenchModel,
    tokenizer: &DynTokenizer,
) -> Result<()> {
    if let Command::Snapshot(args) = &cli.command {
        run_snapshot(model, cli, model_type, args)
    } else {
        let report = run_command(cli, model_type, load_ms, cuda_graph, model, tokenizer)?;
        emit_report(cli, &report)
    }
}

fn main() -> Result<()> {
    logging::init_default();

    let cli = Cli::parse();

    // Compare needs no model loading
    if let Command::Compare(ref args) = cli.command {
        return run_compare(args);
    }

    debug!(
        "bench_serving starting: command={} model_path={} cuda_graph={} format={:?}",
        match &cli.command {
            Command::Request(_) => "request",
            Command::Matrix(_) => "matrix",
            Command::Curve(_) => "curve",
            Command::Snapshot(_) => "snapshot",
            Command::Compare(_) => "compare",
        },
        cli.model_path,
        cli.cuda_graph,
        cli.format
    );
    let model_type = detect_model_type(&cli.model_path)
        .with_context(|| format!("failed to detect model type from {}", cli.model_path))?;
    debug!("Detected model type: {:?}", model_type);
    let load_start = Instant::now();

    // Shared tail for every scheduler-backed model: load the tokenizer, stamp
    // the elapsed load time, wrap the handle, and dispatch. The per-model arms
    // below differ only in how they construct the engine handle.
    let finish = |handle: SchedulerHandle, cuda_graph: bool| -> Result<()> {
        let tokenizer = load_vllm_tokenizer(&cli.model_path)?;
        let load_ms = dur_ms(load_start.elapsed());
        let mut bench = SchedulerBenchModel { handle };
        dispatch(
            &cli, model_type, load_ms, cuda_graph, &mut bench, &tokenizer,
        )
    };

    match model_type {
        #[cfg(feature = "deepseek-v2-lite")]
        ModelType::DeepSeekV2Lite => {
            // Distinct bench type (not scheduler-backed), so it keeps its own tail.
            let generator = openinfer_deepseek_v2_lite::DeepSeekV2LiteEp2Generator::load(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: false,
                    enable_prefill_profile: false,
                    device_ordinals: vec![0, 1],
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
            )?;
            let tokenizer = load_vllm_tokenizer(&cli.model_path)?;
            let load_ms = dur_ms(load_start.elapsed());
            let mut bench = DeepSeekV2LiteBenchModel { generator };
            dispatch(&cli, model_type, load_ms, false, &mut bench, &tokenizer)
        }
        #[cfg(feature = "deepseek-v4")]
        ModelType::DeepSeekV4 => {
            let handle = openinfer_deepseek_v4::start_engine(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: false,
                    enable_prefill_profile: false,
                    device_ordinals: (0..8).collect(),
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
            )?;
            finish(handle, false)
        }
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => {
            let parallel = kimi_parallel_config(cli.tp_size, cli.dp_size)?;
            let handle = openinfer_kimi_k2::start_engine(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: cli.cuda_graph,
                    enable_prefill_profile: false,
                    device_ordinals: (0..parallel.ep_world()).collect(),
                    parallel_config: Some(parallel),
                    ep_backend: cli.ep_backend.into(),
                    seed: command_seed(&cli),
                },
            )?;
            finish(handle, cli.cuda_graph)
        }
        #[cfg(feature = "qwen3-4b")]
        ModelType::Qwen3 => {
            let handle = openinfer_qwen3_4b::start_engine(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: cli.cuda_graph,
                    enable_prefill_profile: false,
                    device_ordinals: vec![0],
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
            )?;
            finish(handle, cli.cuda_graph)
        }
        #[cfg(feature = "qwen35-4b")]
        ModelType::Qwen35 => {
            let handle = openinfer_qwen35_4b::start_engine_with_capacity(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: cli.cuda_graph,
                    enable_prefill_profile: false,
                    device_ordinals: vec![0],
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
                4,
            )?;
            finish(handle, cli.cuda_graph)
        }
    }
}

#[cfg(all(test, feature = "deepseek-v2-lite"))]
mod tests {
    use std::time::Duration;

    use openinfer::sampler::SamplingParams;

    use super::*;

    #[test]
    fn dsv2_lite_sampling_contract_accepts_bench_params() {
        let sampling = SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        };

        assert_dsv2_lite_sampling_contract(&sampling);
    }

    #[test]
    #[should_panic(expected = "supports greedy decoding only")]
    fn dsv2_lite_sampling_contract_rejects_non_greedy_params() {
        let sampling = SamplingParams {
            temperature: 0.8,
            top_k: -1,
            top_p: 0.95,
            ignore_eos: true,
        };

        assert_dsv2_lite_sampling_contract(&sampling);
    }

    #[test]
    #[should_panic(expected = "requires ignore_eos=true")]
    fn dsv2_lite_sampling_contract_rejects_eos_enabled_params() {
        let sampling = SamplingParams {
            ignore_eos: false,
            ..SamplingParams::default()
        };

        assert_dsv2_lite_sampling_contract(&sampling);
    }

    #[test]
    fn dsv2_lite_attribution_timings_preserve_decode_steps() {
        let timings = timings_from_dsv2_lite_attribution(
            vec![11, 304, 608],
            3,
            60_000,
            Some(20_000),
            &[19_000, 18_000],
        );

        assert_eq!(timings.ttft, Duration::from_micros(20_000));
        assert_eq!(
            timings.tbt,
            vec![Duration::from_micros(19_000), Duration::from_micros(18_000)]
        );
        assert_eq!(timings.total, Duration::from_micros(60_000));
        assert_eq!(timings.emitted_tokens, 3);
        assert_eq!(timings.generated_tokens, vec![11, 304, 608]);
        assert_eq!(timings.decode_tokens_for_rate, 2);
        assert_eq!(timings.decode_time_for_rate, Duration::from_micros(37_000));
    }

    #[test]
    fn dsv2_lite_batched_timings_use_shared_decode_time_for_rate() {
        let timings = timings_from_dsv2_lite_batched_generation(
            openinfer_deepseek_v2_lite::BatchedGenerationResult {
                tokens: vec![vec![11, 304, 608], vec![11, 304, 608]],
                prefill_next_token_us: vec![20_000, 21_000],
                per_token_decode_us: vec![19_000, 18_000],
                total_generation_us: 80_000,
                stats: openinfer_deepseek_v2_lite::GenerationStats::default(),
            },
            3,
        );

        assert_eq!(timings.len(), 2);
        assert_eq!(timings[0].decode_tokens_for_rate, 4);
        assert_eq!(
            timings[0].decode_time_for_rate,
            Duration::from_micros(37_000)
        );
        assert_eq!(timings[1].decode_tokens_for_rate, 0);
        assert_eq!(timings[1].decode_time_for_rate, Duration::ZERO);

        let metrics = build_request_metrics(&timings);
        assert_eq!(metrics.steady_tpot_ms.unwrap().p50_ms, 18.0);
        assert!(
            metrics.decode_tok_s.unwrap() > 100.0,
            "batched decode tok/s should use one shared step duration instead of duplicating it per row"
        );
    }

    #[test]
    #[should_panic(expected = "timing count mismatch")]
    fn dsv2_lite_attribution_timings_fail_on_missing_decode_samples() {
        let _ = timings_from_dsv2_lite_attribution(
            vec![11, 304, 608],
            3,
            60_000,
            Some(20_000),
            &[19_000],
        );
    }

    #[test]
    #[should_panic(expected = "generated token count mismatch")]
    fn dsv2_lite_attribution_timings_fail_on_short_generation() {
        let _ =
            timings_from_dsv2_lite_attribution(vec![11, 304], 3, 60_000, Some(20_000), &[19_000]);
    }

    #[test]
    #[should_panic(expected = "zero-duration")]
    fn dsv2_lite_attribution_timings_fail_on_zero_decode_samples() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 60_000, Some(20_000), &[0]);
    }

    #[test]
    #[should_panic(expected = "total generation timing is zero")]
    fn dsv2_lite_attribution_timings_fail_on_zero_total_generation() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 0, Some(20_000), &[19_000]);
    }

    #[test]
    #[should_panic(expected = "TTFT timing is missing or zero")]
    fn dsv2_lite_attribution_timings_fail_on_missing_ttft() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 60_000, None, &[19_000]);
    }

    #[test]
    #[should_panic(expected = "TTFT timing is missing or zero")]
    fn dsv2_lite_attribution_timings_fail_on_zero_ttft() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 60_000, Some(0), &[19_000]);
    }
}
