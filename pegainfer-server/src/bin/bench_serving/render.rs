//! comfy_table renderers for the text report format.

use std::fmt::Write as _;
use std::io::IsTerminal;
use std::io::stdout;
use std::time::Duration;

use comfy_table::Cell;
use comfy_table::CellAlignment;
use comfy_table::Table;
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::ASCII_FULL_CONDENSED;
use comfy_table::presets::UTF8_FULL_CONDENSED;

use crate::metrics::summarize_durations;
use crate::report::CurveReport;
use crate::report::DecodeReport;
use crate::report::DurationStats;
use crate::report::MatrixReport;
use crate::report::MixedLoadReport;
use crate::report::PrefillReport;
use crate::report::RequestReport;
use crate::report::RunInfo;
use crate::snapshot::format_delta;

pub(crate) fn new_table() -> Table {
    let mut table = Table::new();
    if stdout().is_terminal() {
        table.load_preset(UTF8_FULL_CONDENSED);
        table.apply_modifier(UTF8_ROUND_CORNERS);
    } else {
        table.load_preset(ASCII_FULL_CONDENSED);
    }
    table
}

pub(crate) fn key_cell(label: impl Into<String>) -> Cell {
    Cell::new(label.into())
}

pub(crate) fn value_cell(value: impl Into<String>) -> Cell {
    Cell::new(value.into())
}

pub(crate) fn numeric_cell(value: impl Into<String>) -> Cell {
    Cell::new(value.into()).set_alignment(CellAlignment::Right)
}

pub(crate) fn format_rate(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"))
}

pub(crate) fn format_duration_ms(value: f64) -> String {
    format!("{value:.2}")
}

pub(crate) fn format_count_avg(value: f64) -> String {
    format!("{value:.2}")
}

pub(crate) fn push_table(out: &mut String, table: &Table) {
    out.push_str(&table.to_string());
    out.push('\n');
}

pub(crate) fn render_run_summary(report: &RunInfo) -> Table {
    let mut table = new_table();
    table.add_row(vec![
        key_cell("model"),
        value_cell(format!("{} ({})", report.model_path, report.model_type)),
    ]);
    table.add_row(vec![
        key_cell("cuda_graph"),
        value_cell(report.cuda_graph.to_string()),
    ]);
    table.add_row(vec![
        key_cell("load_ms"),
        numeric_cell(format_duration_ms(report.load_ms)),
    ]);
    if let Some(label) = &report.label {
        table.add_row(vec![key_cell("label"), value_cell(label.clone())]);
    }
    table
}

pub(crate) fn render_request_meta(report: &RequestReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("prompt_source"),
        value_cell(report.workload.prompt.source.clone()),
    ]);
    table.add_row(vec![
        key_cell("prompt_tokens"),
        numeric_cell(report.workload.prompt.prompt_tokens.to_string()),
    ]);
    if let Some(preview) = &report.workload.prompt.prompt_preview {
        table.add_row(vec![
            key_cell("prompt"),
            value_cell(format!("\"{preview}\"")),
        ]);
    }
    table.add_row(vec![
        key_cell("output_len"),
        numeric_cell(report.workload.output_len.to_string()),
    ]);
    table.add_row(vec![
        key_cell("warmup / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.warmup, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

pub(crate) fn render_duration_table(rows: Vec<(String, DurationStats)>) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("metric"),
        Cell::new("avg_ms").set_alignment(CellAlignment::Right),
        Cell::new("p50_ms").set_alignment(CellAlignment::Right),
        Cell::new("p95_ms").set_alignment(CellAlignment::Right),
        Cell::new("p99_ms").set_alignment(CellAlignment::Right),
        Cell::new("max_ms").set_alignment(CellAlignment::Right),
        Cell::new("samples").set_alignment(CellAlignment::Right),
    ]);
    for (label, stats) in rows {
        table.add_row(vec![
            key_cell(label),
            numeric_cell(format_duration_ms(stats.avg_ms)),
            numeric_cell(format_duration_ms(stats.p50_ms)),
            numeric_cell(format_duration_ms(stats.p95_ms)),
            numeric_cell(format_duration_ms(stats.p99_ms)),
            numeric_cell(format_duration_ms(stats.max_ms)),
            numeric_cell(stats.samples.to_string()),
        ]);
    }
    table
}

pub(crate) fn render_request_summary(report: &RequestReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("metric"),
        Cell::new("value").set_alignment(CellAlignment::Right),
    ]);
    table.add_row(vec![
        key_cell("generated_tokens_avg"),
        numeric_cell(format_count_avg(report.metrics.generated_tokens.avg)),
    ]);
    table.add_row(vec![
        key_cell("generated_tokens_min"),
        numeric_cell(report.metrics.generated_tokens.min.to_string()),
    ]);
    table.add_row(vec![
        key_cell("generated_tokens_max"),
        numeric_cell(report.metrics.generated_tokens.max.to_string()),
    ]);
    table.add_row(vec![
        key_cell("generated_token_runs"),
        numeric_cell(report.metrics.generated_tokens.samples.to_string()),
    ]);
    table.add_row(vec![
        key_cell("request_tok_s"),
        numeric_cell(format_rate(report.metrics.request_tok_s)),
    ]);
    table.add_row(vec![
        key_cell("decode_tok_s"),
        numeric_cell(format_rate(report.metrics.decode_tok_s)),
    ]);
    table
}

fn join_csv(values: &[usize]) -> String {
    values
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn render_prefill_meta(report: &PrefillReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("prompt_lens"),
        value_cell(join_csv(&report.workload.prompt_lens)),
    ]);
    table.add_row(vec![
        key_cell("batches"),
        value_cell(join_csv(&report.workload.batches)),
    ]);
    table.add_row(vec![
        key_cell("kv_capacity_tokens"),
        value_cell(
            report
                .workload
                .kv_capacity_tokens
                .map_or_else(|| "unknown".to_string(), |c| c.to_string()),
        ),
    ]);
    table.add_row(vec![
        key_cell("distinct_prompts"),
        numeric_cell(report.workload.distinct_prompts.to_string()),
    ]);
    table.add_row(vec![
        key_cell("warmup / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.warmup, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

pub(crate) fn render_prefill_table(report: &PrefillReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("prompt_tok").set_alignment(CellAlignment::Right),
        Cell::new("batch").set_alignment(CellAlignment::Right),
        Cell::new("total_tok").set_alignment(CellAlignment::Right),
        Cell::new("ttft_p50").set_alignment(CellAlignment::Right),
        Cell::new("ttft_p99").set_alignment(CellAlignment::Right),
        Cell::new("ttft_max").set_alignment(CellAlignment::Right),
        Cell::new("prefill_tok/s").set_alignment(CellAlignment::Right),
        Cell::new("samples").set_alignment(CellAlignment::Right),
    ]);
    for cell in &report.cells {
        table.add_row(vec![
            numeric_cell(cell.prompt_len.to_string()),
            numeric_cell(cell.batch.to_string()),
            numeric_cell(cell.total_tokens.to_string()),
            numeric_cell(format_duration_ms(cell.ttft_ms.p50_ms)),
            numeric_cell(format_duration_ms(cell.ttft_ms.p99_ms)),
            numeric_cell(format_duration_ms(cell.ttft_ms.max_ms)),
            numeric_cell(format_rate(cell.prefill_tok_s)),
            numeric_cell(cell.ttft_ms.samples.to_string()),
        ]);
    }
    table
}

pub(crate) fn render_decode_meta(report: &DecodeReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("ctxs"),
        value_cell(join_csv(&report.workload.ctxs)),
    ]);
    table.add_row(vec![
        key_cell("batches"),
        value_cell(join_csv(&report.workload.batches)),
    ]);
    table.add_row(vec![
        key_cell("kv_capacity_tokens"),
        value_cell(
            report
                .workload
                .kv_capacity_tokens
                .map_or_else(|| "unknown".to_string(), |c| c.to_string()),
        ),
    ]);
    table.add_row(vec![
        key_cell("decode_steps / warmup_steps"),
        value_cell(format!(
            "{} / {}",
            report.workload.decode_steps, report.workload.warmup_steps
        )),
    ]);
    table.add_row(vec![
        key_cell("distinct_prompts / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.distinct_prompts, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

pub(crate) fn render_decode_table(report: &DecodeReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("ctx").set_alignment(CellAlignment::Right),
        Cell::new("batch").set_alignment(CellAlignment::Right),
        Cell::new("peak_tok").set_alignment(CellAlignment::Right),
        Cell::new("tpot_p50").set_alignment(CellAlignment::Right),
        Cell::new("tpot_p99").set_alignment(CellAlignment::Right),
        Cell::new("tpot_max").set_alignment(CellAlignment::Right),
        Cell::new("decode_tok/s").set_alignment(CellAlignment::Right),
        Cell::new("samples").set_alignment(CellAlignment::Right),
    ]);
    for cell in &report.cells {
        table.add_row(vec![
            numeric_cell(cell.ctx.to_string()),
            numeric_cell(cell.batch.to_string()),
            numeric_cell(cell.peak_tokens.to_string()),
            numeric_cell(format_duration_ms(cell.tpot_ms.p50_ms)),
            numeric_cell(format_duration_ms(cell.tpot_ms.p99_ms)),
            numeric_cell(format_duration_ms(cell.tpot_ms.max_ms)),
            numeric_cell(format_rate(cell.decode_tok_s)),
            numeric_cell(cell.tpot_ms.samples.to_string()),
        ]);
    }
    table
}

pub(crate) fn render_matrix_meta(report: &MatrixReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("prompt_lens"),
        value_cell(
            report
                .workload
                .prompt_lens
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ),
    ]);
    table.add_row(vec![
        key_cell("output_lens"),
        value_cell(
            report
                .workload
                .output_lens
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ),
    ]);
    table.add_row(vec![
        key_cell("synthetic_pattern"),
        value_cell(report.workload.synthetic_pattern),
    ]);
    table.add_row(vec![
        key_cell("warmup / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.warmup, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

pub(crate) fn render_matrix_table(report: &MatrixReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("prompt_tok").set_alignment(CellAlignment::Right),
        Cell::new("output_tok").set_alignment(CellAlignment::Right),
        Cell::new("ttft_avg").set_alignment(CellAlignment::Right),
        Cell::new("ttft_p95").set_alignment(CellAlignment::Right),
        Cell::new("e2e_avg").set_alignment(CellAlignment::Right),
        Cell::new("req_tok/s").set_alignment(CellAlignment::Right),
        Cell::new("decode_tok/s").set_alignment(CellAlignment::Right),
        Cell::new("gen_avg").set_alignment(CellAlignment::Right),
    ]);
    for cell in &report.cells {
        table.add_row(vec![
            numeric_cell(cell.prompt_len.to_string()),
            numeric_cell(cell.output_len.to_string()),
            numeric_cell(format_duration_ms(cell.ttft_ms.avg_ms)),
            numeric_cell(format_duration_ms(cell.ttft_ms.p95_ms)),
            numeric_cell(format_duration_ms(cell.e2e_ms.avg_ms)),
            numeric_cell(format_rate(cell.request_tok_s)),
            numeric_cell(format_rate(cell.decode_tok_s)),
            numeric_cell(format_count_avg(cell.generated_tokens.avg)),
        ]);
    }
    table
}

pub(crate) fn render_curve_meta(report: &CurveReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("prompt_source"),
        value_cell(report.workload.prompt.source.clone()),
    ]);
    table.add_row(vec![
        key_cell("prompt_tokens"),
        numeric_cell(report.workload.prompt.prompt_tokens.to_string()),
    ]);
    if let Some(preview) = &report.workload.prompt.prompt_preview {
        table.add_row(vec![
            key_cell("prompt"),
            value_cell(format!("\"{preview}\"")),
        ]);
    }
    table.add_row(vec![
        key_cell("output_len"),
        numeric_cell(report.workload.output_len.to_string()),
    ]);
    table.add_row(vec![
        key_cell("window"),
        numeric_cell(report.workload.window.to_string()),
    ]);
    table.add_row(vec![
        key_cell("warmup / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.warmup, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

pub(crate) fn render_curve_table(report: &CurveReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("ctx_range"),
        Cell::new("avg_ms").set_alignment(CellAlignment::Right),
        Cell::new("p50_ms").set_alignment(CellAlignment::Right),
        Cell::new("p95_ms").set_alignment(CellAlignment::Right),
        Cell::new("p99_ms").set_alignment(CellAlignment::Right),
        Cell::new("tok/s").set_alignment(CellAlignment::Right),
        Cell::new("samples").set_alignment(CellAlignment::Right),
    ]);
    for window in &report.windows {
        table.add_row(vec![
            value_cell(format!("{}-{}", window.ctx_start, window.ctx_end)),
            numeric_cell(format_duration_ms(window.tpot_ms.avg_ms)),
            numeric_cell(format_duration_ms(window.tpot_ms.p50_ms)),
            numeric_cell(format_duration_ms(window.tpot_ms.p95_ms)),
            numeric_cell(format_duration_ms(window.tpot_ms.p99_ms)),
            numeric_cell(format_rate(window.decode_tok_s)),
            numeric_cell(window.tpot_ms.samples.to_string()),
        ]);
    }
    table
}

pub(crate) fn render_mixed_text(report: &MixedLoadReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "bench_serving mixed-load ITL\n");

    let cfg = &report.config;
    let mut meta = render_run_summary(&report.run);
    meta.add_row(vec![
        key_cell("commit / gpu"),
        value_cell(format!("{} / {}", report.commit, report.gpu)),
    ]);
    meta.add_row(vec![
        key_cell("bg (prompt,conc,out)"),
        value_cell(format!(
            "({},{},{})",
            cfg.bg_prompt_len, cfg.bg_concurrency, cfg.bg_output_len
        )),
    ]);
    meta.add_row(vec![
        key_cell("injection (prompt,out)"),
        value_cell(format!(
            "({},{})  warm_frac={}",
            cfg.inj_prompt_len, cfg.inj_output_len, cfg.inj_warm_frac
        )),
    ]);
    meta.add_row(vec![
        key_cell("qps / num_injections"),
        value_cell(format!("{} / {}", cfg.qps, cfg.num_injections)),
    ]);
    meta.add_row(vec![
        key_cell("warmup / seed"),
        value_cell(format!("{} / {}", cfg.warmup, cfg.seed)),
    ]);
    meta.add_row(vec![
        key_cell("max_batch / max_prefill_tokens"),
        value_cell(format!(
            "{} / {}",
            cfg.max_batch,
            cfg.max_prefill_tokens
                .map_or_else(|| "default".to_string(), |v| v.to_string())
        )),
    ]);
    push_table(&mut out, &meta);
    out.push('\n');

    let mut rows = Vec::new();
    if let Some(baseline) = &report.baseline_itl {
        rows.push(("baseline_itl".to_string(), baseline.clone()));
    }
    rows.push(("mixed_itl_all".to_string(), report.mixed_itl.all.clone()));
    if let Some(steady) = &report.mixed_itl.steady {
        rows.push(("mixed_itl_steady".to_string(), steady.clone()));
    }
    if let Some(stall) = &report.mixed_itl.stall {
        rows.push(("mixed_itl_stall".to_string(), stall.clone()));
    }
    push_table(&mut out, &render_duration_table(rows));
    out.push('\n');

    let total = report.mixed_itl.total_gap_count;
    let stalled = report.mixed_itl.stall_gap_count;
    let stall_pct = if total > 0 {
        100.0 * stalled as f64 / total as f64
    } else {
        0.0
    };
    let _ = writeln!(out, "stall gaps: {stalled}/{total} ({stall_pct:.1}%)");
    let _ = writeln!(
        out,
        "background generated tokens: min={} max={} avg={:.2} runs={}",
        report.background_generated_tokens.min,
        report.background_generated_tokens.max,
        report.background_generated_tokens.avg,
        report.background_generated_tokens.samples
    );
    if let Some(trace) = report.background_generated_token_traces.first() {
        let _ = writeln!(out, "background hash0: {} len={}", trace.hash, trace.len);
    }

    let dur = |ms: f64| Duration::from_secs_f64(ms / 1000.0);
    let prefill_line = |label: &str, ms: &[Duration]| {
        if ms.is_empty() {
            return String::new();
        }
        let s = summarize_durations(ms);
        format!(
            "{label}: p50={:.2}ms  p99={:.2}ms  max={:.2}ms (n={})\n",
            s.p50_ms,
            s.p99_ms,
            s.max_ms,
            ms.len()
        )
    };
    if !report.injections.is_empty() {
        let cold: Vec<Duration> = report
            .injections
            .iter()
            .filter(|r| !r.warm)
            .map(|r| dur(r.prefill_ms))
            .collect();
        let warm: Vec<Duration> = report
            .injections
            .iter()
            .filter(|r| r.warm)
            .map(|r| dur(r.prefill_ms))
            .collect();
        out.push_str(&prefill_line("injected prefill (cold)", &cold));
        out.push_str(&prefill_line("injected prefill (warm)", &warm));
        if let Some(first) = report.injections.first() {
            let min_generated = report
                .injections
                .iter()
                .map(|r| r.generated_tokens)
                .min()
                .unwrap_or(0);
            let max_generated = report
                .injections
                .iter()
                .map(|r| r.generated_tokens)
                .max()
                .unwrap_or(0);
            let _ = writeln!(
                out,
                "injection generated tokens: min={min_generated} max={max_generated} hash0={} len={}",
                first.generated_token_trace.hash, first.generated_token_trace.len
            );
        }
    }

    let d = &report.decision_inputs;
    match (
        d.baseline_p50_ms,
        d.baseline_p99_ms,
        d.p99_delta_pct,
        d.p99_delta_ms,
    ) {
        (Some(bp50), Some(bp99), Some(dpct), Some(dms)) => {
            let _ = writeln!(
                out,
                "\nITL p50: baseline {:.2}ms → mixed {:.2}ms",
                bp50, d.mixed_p50_ms
            );
            let _ = writeln!(
                out,
                "ITL p99: baseline {:.2}ms → mixed {:.2}ms ({}, {:+.2}ms)",
                bp99,
                d.mixed_p99_ms,
                format_delta(dpct),
                dms
            );
        }
        _ => {
            let _ = writeln!(
                out,
                "\nITL (mixed, no baseline): p50={:.2}ms  p99={:.2}ms",
                d.mixed_p50_ms, d.mixed_p99_ms
            );
        }
    }

    for warning in &report.warnings {
        let _ = writeln!(out, "warning: {warning}");
    }
    out
}
