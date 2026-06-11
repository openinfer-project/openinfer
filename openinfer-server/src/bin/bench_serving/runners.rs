//! Benchmark drivers for the request / matrix / curve commands:
//! timing collection, metric assembly, and report emission.

use std::fmt::Write as _;
use std::fs;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use cudarc::driver::Profiler;
use cudarc::runtime::result::device as cuda_device;
use log::{debug, info};
use openinfer::sampler::SamplingParams;
use openinfer::server_engine::ModelType;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vllm_text::tokenizer::DynTokenizer;

use crate::cli::*;
use crate::exec::*;
use crate::metrics::*;
use crate::prompt::*;
use crate::render::*;
use crate::report::*;

pub(crate) const DEFAULT_REQUEST_PROMPT: &str = "Tell me a story";
pub(crate) const DEFAULT_CURVE_PROMPT_LEN: usize = 512;

pub(crate) fn normalize_sizes(values: &[usize], flag: &str) -> Result<Vec<usize>> {
    ensure!(!values.is_empty(), "{flag} must not be empty");
    ensure!(values.iter().all(|v| *v > 0), "{flag} values must be > 0");
    let mut normalized = values.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    Ok(normalized)
}

pub(crate) fn validate_run_args(args: &RunArgs) -> Result<()> {
    ensure!(args.iters > 0, "--iters must be > 0");
    Ok(())
}

pub(crate) fn measure_timings(
    model: &mut dyn BenchModel,
    prompts: &[Vec<u32>],
    output_len: usize,
    run: &RunArgs,
    cuda_profiler_capture: bool,
) -> Result<Vec<GenTimings>> {
    ensure!(output_len > 0, "--output-len must be > 0");
    ensure!(!prompts.is_empty(), "concurrency must be > 0");
    model.validate_concurrency(prompts.len())?;
    validate_run_args(run)?;

    let sampling = SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    };
    let mut rng = StdRng::seed_from_u64(run.seed);

    for _ in 0..run.warmup {
        let _ = model.timed_generation_batch(prompts, output_len, &sampling, &mut rng);
    }

    let profiler = if cuda_profiler_capture {
        info!(
            "Starting CUDA profiler capture around {} measured iterations",
            run.iters
        );
        cuda_device::set(0).context("failed to set CUDA device before profiler capture")?;
        Some(Profiler::new().context("failed to start CUDA profiler capture")?)
    } else {
        None
    };

    let mut timings = Vec::with_capacity(run.iters * prompts.len());
    for _ in 0..run.iters {
        timings.extend(model.timed_generation_batch(prompts, output_len, &sampling, &mut rng));
    }
    drop(profiler);
    Ok(timings)
}

pub(crate) fn build_request_metrics(timings: &[GenTimings]) -> RequestMetrics {
    let ttfts: Vec<Duration> = timings.iter().map(|t| t.ttft).collect();
    let e2e: Vec<Duration> = timings.iter().map(|t| t.total).collect();
    let first_steps: Vec<Duration> = timings
        .iter()
        .filter_map(|t| t.tbt.first().copied())
        .collect();
    let steady: Vec<Duration> = timings
        .iter()
        .flat_map(|t| t.tbt.iter().skip(1).copied())
        .collect();
    let generated: Vec<usize> = timings.iter().map(|t| t.emitted_tokens).collect();
    let generated_token_traces: Vec<GeneratedTokenTrace> = timings
        .iter()
        .map(|timing| generated_token_trace(&timing.generated_tokens))
        .collect();

    let total_emitted: usize = timings.iter().map(|t| t.emitted_tokens).sum();
    let total_request_time: Duration = timings.iter().map(|t| t.total).sum();
    let total_decode_steps: usize = timings.iter().map(|t| t.decode_tokens_for_rate).sum();
    let total_decode_time: Duration = timings.iter().map(|t| t.decode_time_for_rate).sum();

    RequestMetrics {
        ttft_ms: summarize_durations(&ttfts),
        first_decode_step_ms: (!first_steps.is_empty()).then(|| summarize_durations(&first_steps)),
        steady_tpot_ms: (!steady.is_empty()).then(|| summarize_durations(&steady)),
        e2e_ms: summarize_durations(&e2e),
        generated_tokens: summarize_counts(&generated),
        generated_token_traces,
        request_tok_s: aggregate_tok_s(total_emitted, total_request_time),
        decode_tok_s: aggregate_tok_s(total_decode_steps, total_decode_time),
    }
}

pub(crate) fn build_request_iterations(timings: &[GenTimings]) -> Vec<RequestIterationTiming> {
    timings
        .iter()
        .enumerate()
        .map(|(index, timing)| {
            let steady: Vec<Duration> = timing.tbt.iter().skip(1).copied().collect();
            RequestIterationTiming {
                index,
                ttft_ms: dur_ms(timing.ttft),
                first_decode_step_ms: timing.tbt.first().copied().map(dur_ms),
                steady_tpot_ms: (!steady.is_empty()).then(|| summarize_durations(&steady)),
                e2e_ms: dur_ms(timing.total),
                generated_tokens: timing.emitted_tokens,
                generated_token_trace: generated_token_trace(&timing.generated_tokens),
            }
        })
        .collect()
}

pub(crate) fn run_info(
    cli: &Cli,
    command: &'static str,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
) -> RunInfo {
    RunInfo {
        command,
        model_path: cli.model_path.clone(),
        model_type: format!("{model_type:?}"),
        cuda_graph,
        load_ms,
        label: cli.label.clone(),
    }
}

pub(crate) fn bench_request(
    model: &mut dyn BenchModel,
    tokenizer: &DynTokenizer,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    args: &RequestArgs,
) -> Result<BenchReport> {
    let mut prompt = resolve_prompt_input(
        &args.prompt_input,
        tokenizer,
        Some(DEFAULT_REQUEST_PROMPT),
        None,
    )?;
    // A `--prompt-len` workload is synthetic: give every concurrent request a
    // distinct seeded-random prompt so the decode streams diverge and MoE
    // routing is realistic. An explicit `--prompt`/`--prompt-file` (or the
    // default text) is the caller's chosen prompt and is replicated as-is.
    let synthetic = args.prompt_input.prompt_len.is_some();
    let prompts: Vec<Vec<u32>> = if synthetic {
        // 0 = one distinct prompt per request (fully diverse). Otherwise tile
        // `distinct` unique prompts across the batch: idx → idx % distinct.
        let distinct = if args.distinct_prompts == 0 {
            args.concurrency
        } else {
            args.distinct_prompts.min(args.concurrency)
        };
        prompt.descriptor.source = format!(
            "synthetic-random[{SYNTHETIC_TOKEN_LO}..{SYNTHETIC_TOKEN_HI}) seed={} distinct={distinct}/{}",
            args.run.seed, args.concurrency
        );
        (0..args.concurrency)
            .map(|idx| synthetic_random_prompt(prompt.tokens.len(), args.run.seed, idx % distinct))
            .collect()
    } else {
        vec![prompt.tokens.clone(); args.concurrency]
    };
    info!(
        "Starting request benchmark: prompt_tokens={} output_len={} concurrency={} warmup={} iters={} seed={} source={}",
        prompt.descriptor.prompt_tokens,
        args.output_len,
        args.concurrency,
        args.run.warmup,
        args.run.iters,
        args.run.seed,
        prompt.descriptor.source,
    );
    let timings = measure_timings(
        model,
        &prompts,
        args.output_len,
        &args.run,
        cli.cuda_profiler_capture,
    )?;
    Ok(BenchReport::Request(Box::new(RequestReport {
        run: run_info(cli, "request", model_type, load_ms, cuda_graph),
        workload: RequestWorkload {
            prompt: prompt.descriptor,
            output_len: args.output_len,
            concurrency: args.concurrency,
            warmup: args.run.warmup,
            iters: args.run.iters,
            seed: args.run.seed,
        },
        metrics: build_request_metrics(&timings),
        iterations: build_request_iterations(&timings),
    })))
}

pub(crate) fn bench_matrix(
    model: &mut dyn BenchModel,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    args: &MatrixArgs,
) -> Result<BenchReport> {
    validate_run_args(&args.run)?;
    let prompt_lens = normalize_sizes(&args.prompt_lens, "--prompt-lens")?;
    let output_lens = normalize_sizes(&args.output_lens, "--output-lens")?;
    info!(
        "Starting matrix benchmark: prompt_lens={:?} output_lens={:?} warmup={} iters={} seed={}",
        prompt_lens, output_lens, args.run.warmup, args.run.iters, args.run.seed
    );

    let mut cells = Vec::with_capacity(prompt_lens.len() * output_lens.len());
    for &prompt_len in &prompt_lens {
        let prompt_tokens = synthetic_prompt_tokens(prompt_len);
        for &output_len in &output_lens {
            debug!(
                "Running matrix cell: prompt_len={} output_len={}",
                prompt_len, output_len
            );
            let timings = measure_timings(
                model,
                std::slice::from_ref(&prompt_tokens),
                output_len,
                &args.run,
                cli.cuda_profiler_capture,
            )?;
            let metrics = build_request_metrics(&timings);
            cells.push(MatrixCell {
                prompt_len,
                output_len,
                ttft_ms: metrics.ttft_ms,
                e2e_ms: metrics.e2e_ms,
                first_decode_step_ms: metrics.first_decode_step_ms,
                steady_tpot_ms: metrics.steady_tpot_ms,
                generated_tokens: metrics.generated_tokens,
                request_tok_s: metrics.request_tok_s,
                decode_tok_s: metrics.decode_tok_s,
            });
        }
    }

    Ok(BenchReport::Matrix(MatrixReport {
        run: run_info(cli, "matrix", model_type, load_ms, cuda_graph),
        workload: MatrixWorkload {
            prompt_lens,
            output_lens,
            warmup: args.run.warmup,
            iters: args.run.iters,
            seed: args.run.seed,
            synthetic_pattern: SYNTHETIC_PATTERN,
        },
        cells,
    }))
}

pub(crate) fn bench_curve(
    model: &mut dyn BenchModel,
    tokenizer: &DynTokenizer,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    args: &CurveArgs,
) -> Result<BenchReport> {
    ensure!(args.window > 0, "--window must be > 0");
    ensure!(args.output_len >= 2, "--output-len must be >= 2 for curve");

    let prompt = resolve_prompt_input(
        &args.prompt_input,
        tokenizer,
        None,
        Some(DEFAULT_CURVE_PROMPT_LEN),
    )?;
    info!(
        "Starting curve benchmark: prompt_tokens={} output_len={} window={} warmup={} iters={} seed={}",
        prompt.descriptor.prompt_tokens,
        args.output_len,
        args.window,
        args.run.warmup,
        args.run.iters,
        args.run.seed
    );
    let timings = measure_timings(
        model,
        std::slice::from_ref(&prompt.tokens),
        args.output_len,
        &args.run,
        cli.cuda_profiler_capture,
    )?;

    let mut tbt_by_pos: Vec<Vec<Duration>> = Vec::new();
    for timing in &timings {
        for (idx, &duration) in timing.tbt.iter().enumerate() {
            if idx >= tbt_by_pos.len() {
                tbt_by_pos.push(Vec::with_capacity(args.run.iters));
            }
            tbt_by_pos[idx].push(duration);
        }
    }

    let mut windows = Vec::new();
    let mut pos = 0usize;
    while pos < tbt_by_pos.len() {
        let end = (pos + args.window).min(tbt_by_pos.len());
        let mut samples = Vec::new();
        for bucket in &tbt_by_pos[pos..end] {
            samples.extend_from_slice(bucket);
        }
        if !samples.is_empty() {
            let stats = summarize_durations(&samples);
            windows.push(CurveWindow {
                ctx_start: prompt.descriptor.prompt_tokens + pos + 1,
                ctx_end: prompt.descriptor.prompt_tokens + end,
                decode_tok_s: (stats.avg_ms > 0.0).then(|| 1000.0 / stats.avg_ms),
                tpot_ms: stats,
            });
        }
        pos = end;
    }

    Ok(BenchReport::Curve(CurveReport {
        run: run_info(cli, "curve", model_type, load_ms, cuda_graph),
        workload: CurveWorkload {
            prompt: prompt.descriptor,
            output_len: args.output_len,
            window: args.window,
            warmup: args.run.warmup,
            iters: args.run.iters,
            seed: args.run.seed,
        },
        windows,
    }))
}

pub(crate) fn render_text(report: &BenchReport) -> String {
    let mut out = String::new();
    match report {
        BenchReport::Request(report) => {
            let _ = writeln!(out, "bench_serving request\n");
            push_table(&mut out, &render_request_meta(report));
            out.push('\n');
            push_table(
                &mut out,
                &render_duration_table(
                    std::iter::once(("ttft_ms".to_string(), report.metrics.ttft_ms.clone()))
                        .chain(
                            report
                                .metrics
                                .first_decode_step_ms
                                .clone()
                                .into_iter()
                                .map(|stats| ("first_decode_step_ms".to_string(), stats)),
                        )
                        .chain(
                            report
                                .metrics
                                .steady_tpot_ms
                                .clone()
                                .into_iter()
                                .map(|stats| ("steady_tpot_ms".to_string(), stats)),
                        )
                        .chain(std::iter::once((
                            "e2e_ms".to_string(),
                            report.metrics.e2e_ms.clone(),
                        )))
                        .collect(),
                ),
            );
            out.push('\n');
            push_table(&mut out, &render_request_summary(report));
        }
        BenchReport::Matrix(report) => {
            let _ = writeln!(out, "bench_serving matrix\n");
            push_table(&mut out, &render_matrix_meta(report));
            out.push('\n');
            push_table(&mut out, &render_matrix_table(report));
        }
        BenchReport::Curve(report) => {
            let _ = writeln!(out, "bench_serving curve\n");
            push_table(&mut out, &render_curve_meta(report));
            out.push('\n');
            push_table(&mut out, &render_curve_table(report));
        }
        BenchReport::Mixed(report) => out.push_str(&render_mixed_text(report)),
    }
    out
}

pub(crate) fn emit_report(cli: &Cli, report: &BenchReport) -> Result<()> {
    let rendered = match cli.format {
        OutputFormat::Text => render_text(report),
        OutputFormat::Json => serde_json::to_string_pretty(report)?,
    };

    if let Some(path) = &cli.out {
        fs::write(path, &rendered).with_context(|| format!("failed to write report to {path}"))?;
        info!("Wrote benchmark report to {}", path);
    }

    println!("{rendered}");
    Ok(())
}

pub(crate) fn run_command(
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    model: &mut dyn BenchModel,
    tokenizer: &DynTokenizer,
) -> Result<BenchReport> {
    match &cli.command {
        Command::Request(args) => {
            bench_request(model, tokenizer, cli, model_type, load_ms, cuda_graph, args)
        }
        Command::Matrix(args) => bench_matrix(model, cli, model_type, load_ms, cuda_graph, args),
        Command::Curve(args) => {
            bench_curve(model, tokenizer, cli, model_type, load_ms, cuda_graph, args)
        }
        Command::Mixed(args) => {
            crate::mixed::run_mixed_load(model, cli, model_type, load_ms, cuda_graph, args)
        }
        Command::Snapshot(_) | Command::Compare(_) => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Snapshot / Compare
// ---------------------------------------------------------------------------
