//! Snapshot generation and git-baseline comparison: the regression-
//! trackable profiles plus their git/gpu/date provenance helpers.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use comfy_table::Cell;
use comfy_table::CellAlignment;
use log::info;
use openinfer::server_engine::ModelType;

use crate::cli::Cli;
use crate::cli::CompareArgs;
use crate::cli::MixedArgs;
use crate::cli::RunArgs;
use crate::cli::SnapshotArgs;
use crate::exec::BenchModel;
use crate::prompt::synthetic_prompt_tokens;
use crate::render::key_cell;
use crate::render::new_table;
use crate::render::numeric_cell;
use crate::render::push_table;
use crate::report::BenchReport;
use crate::report::SnapshotMixedItl;
use crate::report::SnapshotProfile;
use crate::report::SnapshotReport;
use crate::runners::build_request_metrics;
use crate::runners::measure_timings;

pub(crate) const SNAPSHOT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench_snapshots");
pub(crate) const SNAPSHOT_PREFILL_OUTPUT_LEN: usize = 1;
pub(crate) const SNAPSHOT_DECODE_PROMPT_LEN: usize = 1024;
pub(crate) const SNAPSHOT_DECODE_OUTPUT_LEN: usize = 256;

pub(crate) fn snapshot_prefill_prompt_len(model_type: ModelType) -> usize {
    match model_type {
        // Kimi serves TP1/DP8, where the PPLX fabric buffers cap prompts at
        // 2048 tokens (full-lifetime KV cap is 8192) — probe the largest
        // prompt the serving shape admits.
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => 2_048,
        _ => 10_000,
    }
}
pub(crate) const REGRESSION_TPOT_PCT: f64 = 2.0;
pub(crate) const REGRESSION_TTFT_PCT: f64 = 3.0;

/// Canonical mixed-load cell folded into the snapshot: 4 background decode
/// streams (512-prompt / 1024-out) with a 4k **cold** prompt arriving at
/// 0.5 req/s. This is the committed reference point from
/// docs/benchmarks/mixed-load-itl.md; it fits the 16 GB KV budget so it runs on
/// every supported card. Pinned (not driven by `--warmup`/`--seed`) so the
/// tracked profile stays shape-stable across refreshes.
fn snapshot_mixed_args() -> MixedArgs {
    MixedArgs {
        bg_prompt_len: 512,
        bg_concurrency: 4,
        bg_output_len: 1024,
        inj_prompt_len: 4096,
        inj_output_len: 1,
        qps: 0.5,
        num_injections: 5,
        skip_baseline: false,
        inj_warm_frac: 0.0,
        head_start_tokens: 8,
        run: RunArgs {
            warmup: 5,
            iters: 20,
            seed: 42,
        },
    }
}

pub(crate) fn shell_output(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

pub(crate) fn git_short_commit() -> String {
    shell_output("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into())
}

pub(crate) fn gpu_name() -> String {
    shell_output(
        "nvidia-smi",
        &["--query-gpu=name", "--format=csv,noheader", "--id=0"],
    )
    .unwrap_or_else(|| "unknown".into())
}

/// Produce a filesystem-safe slug from a GPU name string.
///
/// `"NVIDIA GeForce RTX 5070 Ti"` → `"rtx-5070-ti"`
pub(crate) fn gpu_slug_from(name: &str) -> String {
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

pub(crate) fn today_date() -> String {
    shell_output("date", &["+%Y-%m-%d"]).unwrap_or_else(|| "unknown".into())
}

pub(crate) fn model_display_name(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

pub(crate) fn delta_pct(current: f64, baseline: f64) -> f64 {
    if baseline == 0.0 {
        return 0.0;
    }
    (current - baseline) / baseline * 100.0
}

pub(crate) fn format_delta(pct: f64) -> String {
    if pct >= 0.0 {
        format!("+{pct:.1}%")
    } else {
        format!("{pct:.1}%")
    }
}

pub(crate) fn run_snapshot(
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

    let mixed_itl = if model.scheduler_handle().is_some() {
        let margs = snapshot_mixed_args();
        info!(
            "Running mixed-load ITL (inj {} cold @ {} req/s into {}-way decode)",
            margs.inj_prompt_len, margs.qps, margs.bg_concurrency
        );
        match crate::mixed::run_mixed_load(model, cli, model_type, 0.0, cli.cuda_graph, &margs)? {
            BenchReport::Mixed(report) => Some(SnapshotMixedItl {
                config: report.config,
                baseline_itl: report.baseline_itl,
                itl: report.mixed_itl,
                warnings: report.warnings,
            }),
            _ => None,
        }
    } else {
        info!("snapshot: model exposes no scheduler handle; skipping mixed-load ITL profile");
        None
    };

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
        mixed_itl,
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

pub(crate) fn render_snapshot_text(report: &SnapshotReport, path: &Path) -> String {
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
    if let Some(m) = &report.mixed_itl {
        let _ = writeln!(
            out,
            "\nmixed_itl (inj {} cold @ {} req/s, {}-way bg):",
            m.config.inj_prompt_len, m.config.qps, m.config.bg_concurrency
        );
        if let Some(b) = &m.baseline_itl {
            let _ = writeln!(
                out,
                "  baseline  p50={:.2}ms  p99={:.2}ms",
                b.p50_ms, b.p99_ms
            );
        }
        let _ = writeln!(
            out,
            "  mixed     p50={:.2}ms  p99={:.2}ms",
            m.itl.all.p50_ms, m.itl.all.p99_ms
        );
    }
    let _ = writeln!(out, "\nwritten to {}", path.display());
    out
}

pub(crate) fn run_compare(args: &CompareArgs) -> Result<()> {
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

pub(crate) fn render_comparison(
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

    if let (Some(cm), Some(bm)) = (&current.mixed_itl, &baseline.mixed_itl) {
        if cm.config.inj_prompt_len == bm.config.inj_prompt_len
            && cm.config.qps.to_bits() == bm.config.qps.to_bits()
        {
            let ml = format!("(inj {}@{})", cm.config.inj_prompt_len, cm.config.qps);
            for (stat, cur, base) in [
                ("p50", cm.itl.all.p50_ms, bm.itl.all.p50_ms),
                ("p99", cm.itl.all.p99_ms, bm.itl.all.p99_ms),
            ] {
                table.add_row(vec![
                    key_cell(format!("ITL {stat} {ml}")),
                    numeric_cell(format!("{cur:.2}ms")),
                    numeric_cell(format!("{base:.2}ms")),
                    numeric_cell(format_delta(delta_pct(cur, base))),
                ]);
            }
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
    if current.mixed_itl.is_some() && baseline.mixed_itl.is_some() {
        let _ = writeln!(
            out,
            "(mixed-load ITL shown for context only — not gated; the stall tail is thermally/run-to-run noisy)"
        );
    }

    out
}
