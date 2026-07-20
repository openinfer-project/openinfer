//! Mixed-load ITL engine: long prompts injected at low QPS into a steady-state
//! decode batch, measuring how often (and how far) decode inter-token latency
//! stalls behind in-flight prefill. The scheduler-driving loops reuse
//! [`run_scheduler_stream`] instead of re-implementing submit/drain.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use log::info;
use openinfer::sampler::SamplingParams;
use openinfer::scheduler::SchedulerHandle;
use openinfer::server_engine::ModelType;

use crate::cli::Cli;
use crate::cli::MixedArgs;
use crate::exec::BenchModel;
use crate::exec::run_scheduler_stream;
use crate::metrics::dur_ms;
use crate::metrics::generated_token_trace;
use crate::metrics::summarize_counts;
use crate::metrics::summarize_durations;
use crate::prompt::synthetic_prompt_tokens;
use crate::prompt::synthetic_prompt_tokens_salted;
use crate::report::BenchReport;
use crate::report::DurationStats;
use crate::report::InjectionRecord;
use crate::report::MixedDecisionInputs;
use crate::report::MixedLoadConfig;
use crate::report::MixedLoadItl;
use crate::report::MixedLoadReport;
use crate::runners::run_info;
use crate::snapshot::delta_pct;
use crate::snapshot::git_short_commit;
use crate::snapshot::gpu_name;
use crate::snapshot::today_date;

/// One background decode stream's record over a mixed-load (or baseline) phase.
struct BgStream {
    /// Wall-clock instant of each emitted decode token.
    token_times: Vec<Instant>,
    /// Generated token ids for output sanity/hash checks.
    generated_tokens: Vec<u32>,
    /// True if the stream hit its `output_len` (Finished) before being stopped —
    /// signals that steady-state concurrency dropped mid-run.
    finished_early: bool,
}

struct InjectorOutcome {
    /// `[submit, last-token]` window of each injected prefill.
    windows: Vec<(Instant, Instant)>,
    records: Vec<InjectionRecord>,
    /// Injections whose prefill outlasted the `1/qps` slot (QPS not sustained).
    overruns: usize,
}

fn greedy_sampling() -> SamplingParams {
    SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    }
}

fn opt_summarize(samples: &[Duration]) -> Option<DurationStats> {
    (!samples.is_empty()).then(|| summarize_durations(samples))
}

/// Spawn `bg_concurrency` long-lived decode streams. Each records the instant of
/// every emitted token and stops when `stop` is set (or its `output_len` runs
/// out). `counters[idx]` tracks tokens emitted, for head-start coordination.
fn spawn_background_streams(
    handle: &SchedulerHandle,
    bg_prompt_len: usize,
    bg_output_len: usize,
    bg_concurrency: usize,
    stop: &Arc<AtomicBool>,
    counters: &Arc<[AtomicUsize]>,
) -> Vec<thread::JoinHandle<Result<BgStream>>> {
    (0..bg_concurrency)
        .map(|idx| {
            let handle = handle.clone();
            let stop = Arc::clone(stop);
            let counters = Arc::clone(counters);
            thread::spawn(move || -> Result<BgStream> {
                let prompt = synthetic_prompt_tokens(bg_prompt_len);
                let mut token_times = Vec::with_capacity(bg_output_len);
                let mut generated_tokens = Vec::with_capacity(bg_output_len);
                let outcome = run_scheduler_stream(
                    &handle,
                    Some(format!("mixed-bg-{idx}")),
                    prompt,
                    greedy_sampling(),
                    bg_output_len,
                    |id| {
                        token_times.push(Instant::now());
                        generated_tokens.push(id);
                        counters[idx].fetch_add(1, Ordering::Relaxed);
                        !stop.load(Ordering::Acquire)
                    },
                )?;
                // Dropping the stream cancels the request if it is still active.
                Ok(BgStream {
                    token_times,
                    generated_tokens,
                    finished_early: outcome.finished,
                })
            })
        })
        .collect()
}

/// Run a few closed-loop decode batches at the target concurrency to JIT the
/// decode CUDA graph and warm the allocator before measurement begins.
fn mixed_warmup(
    handle: &SchedulerHandle,
    bg_prompt_len: usize,
    bg_concurrency: usize,
    rounds: usize,
) -> Result<()> {
    for _ in 0..rounds {
        let workers: Vec<_> = (0..bg_concurrency)
            .map(|idx| {
                let handle = handle.clone();
                thread::spawn(move || -> Result<()> {
                    let prompt = synthetic_prompt_tokens(bg_prompt_len);
                    run_scheduler_stream(
                        &handle,
                        Some(format!("mixed-warmup-{idx}")),
                        prompt,
                        greedy_sampling(),
                        16,
                        |_| true,
                    )?;
                    Ok(())
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("warmup worker panicked")?;
        }
    }
    Ok(())
}

/// Block until every background stream has emitted `target` tokens, so injection
/// starts only after the background is in steady-state decode (past its own
/// prefill / first-decode-step). Returns false on timeout.
fn wait_for_head_start(counters: &Arc<[AtomicUsize]>, target: usize, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if counters.iter().all(|c| c.load(Ordering::Relaxed) >= target) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

/// Decide whether injection `index` is warm, spreading `warm_frac` of injections evenly across the run
fn injection_is_warm(index: usize, warm_frac: f64) -> bool {
    let count = |k: usize| (k as f64 * warm_frac).floor() as usize;
    count(index + 1) > count(index)
}

/// Submit `num_injections` long prompts paced by arrival at `qps`, draining each
/// to completion. Each `[submit, last-token]` window marks an in-flight prefill.
fn run_injector(
    handle: &SchedulerHandle,
    inj_prompt_len: usize,
    inj_output_len: usize,
    qps: f64,
    num_injections: usize,
    warm_frac: f64,
) -> Result<InjectorOutcome> {
    let period = Duration::from_secs_f64(1.0 / qps);
    let mut windows = Vec::with_capacity(num_injections);
    let mut records = Vec::with_capacity(num_injections);
    let mut overruns = 0usize;
    let warm_salt = num_injections + 100;
    let t0 = Instant::now();
    for index in 0..num_injections {
        // Evenly interleave round(warm_frac * num_injections) warm injections.
        let warm = injection_is_warm(index, warm_frac);
        // Warm → shared prompt (injection after the first hits the prefix cache).
        // Cold → distinct prompt per injection → real prefill every time.
        let salt = if warm { warm_salt } else { index + 1 };
        let prompt = synthetic_prompt_tokens_salted(inj_prompt_len, salt);
        let slot_start = Instant::now();
        let mut last = slot_start;
        let mut generated_tokens = Vec::with_capacity(inj_output_len);
        run_scheduler_stream(
            handle,
            Some(format!("mixed-inj-{index}")),
            prompt,
            greedy_sampling(),
            inj_output_len,
            |id| {
                last = Instant::now();
                generated_tokens.push(id);
                true
            },
        )?;
        windows.push((slot_start, last));
        records.push(InjectionRecord {
            index,
            warm,
            prefill_ms: dur_ms(last - slot_start),
            arrival_offset_ms: dur_ms(slot_start - t0),
            generated_tokens: generated_tokens.len(),
            generated_token_trace: generated_token_trace(&generated_tokens),
        });
        let elapsed = slot_start.elapsed();
        if elapsed < period {
            thread::sleep(period.saturating_sub(elapsed));
        } else if index + 1 < num_injections {
            overruns += 1;
        }
    }
    Ok(InjectorOutcome {
        windows,
        records,
        overruns,
    })
}

/// A background decode gap `[a, b)` is a stall if it overlaps any in-flight
/// prefill window `[s, e)`.
fn gap_overlaps_any(a: Instant, b: Instant, windows: &[(Instant, Instant)]) -> bool {
    windows.iter().any(|&(s, e)| a < e && s < b)
}

fn collect_gaps(streams: &[BgStream]) -> Vec<Duration> {
    let mut gaps = Vec::new();
    for stream in streams {
        for pair in stream.token_times.windows(2) {
            gaps.push(pair[1] - pair[0]);
        }
    }
    gaps
}

fn build_mixed_itl(streams: &[BgStream], windows: &[(Instant, Instant)]) -> Option<MixedLoadItl> {
    let mut all = Vec::new();
    let mut steady = Vec::new();
    let mut stall = Vec::new();
    for stream in streams {
        for pair in stream.token_times.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            all.push(b - a);
            if gap_overlaps_any(a, b, windows) {
                stall.push(b - a);
            } else {
                steady.push(b - a);
            }
        }
    }
    let total_gap_count = all.len();
    let stall_gap_count = stall.len();
    Some(MixedLoadItl {
        all: opt_summarize(&all)?,
        steady: opt_summarize(&steady),
        stall: opt_summarize(&stall),
        stall_gap_count,
        total_gap_count,
    })
}

/// Decode-only control: same background streams, no injector, run for `duration`.
fn run_baseline(
    handle: &SchedulerHandle,
    args: &MixedArgs,
    duration: Duration,
    warnings: &mut Vec<String>,
) -> Result<Option<DurationStats>> {
    let stop = Arc::new(AtomicBool::new(false));
    let counters: Arc<[AtomicUsize]> = (0..args.bg_concurrency)
        .map(|_| AtomicUsize::new(0))
        .collect();
    let bg_handles = spawn_background_streams(
        handle,
        args.bg_prompt_len,
        args.bg_output_len,
        args.bg_concurrency,
        &stop,
        &counters,
    );
    if !wait_for_head_start(&counters, args.head_start_tokens, Duration::from_secs(120)) {
        warnings.push("baseline: head-start not reached within 120s".to_string());
    }
    thread::sleep(duration);
    stop.store(true, Ordering::Release);

    let mut streams = Vec::with_capacity(args.bg_concurrency);
    for worker in bg_handles {
        streams.push(worker.join().expect("baseline worker panicked")?);
    }
    if streams.iter().any(|s| s.finished_early) {
        warnings.push(
            "baseline: a background stream hit --bg-output-len before the window closed"
                .to_string(),
        );
    }
    Ok(opt_summarize(&collect_gaps(&streams)))
}

pub(crate) fn run_mixed_load(
    model: &mut dyn BenchModel,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    args: &MixedArgs,
) -> Result<BenchReport> {
    ensure!(args.bg_concurrency > 0, "--bg-concurrency must be > 0");
    ensure!(args.bg_prompt_len > 0, "--bg-prompt-len must be > 0");
    ensure!(args.bg_output_len > 0, "--bg-output-len must be > 0");
    ensure!(args.inj_prompt_len > 0, "--inj-prompt-len must be > 0");
    ensure!(args.inj_output_len > 0, "--inj-output-len must be > 0");
    ensure!(args.num_injections > 0, "--num-injections must be > 0");
    ensure!(args.qps > 0.0, "--qps must be > 0");
    ensure!(
        (0.0..=1.0).contains(&args.inj_warm_frac),
        "--inj-warm-frac must be in [0.0, 1.0]"
    );

    let handle = model.scheduler_handle().context(
        "mixed-load requires a scheduler-backed continuous-batching model; \
         this model exposes no scheduler handle",
    )?;

    let mut warnings = Vec::new();

    // Slot starvation: long-lived bg streams hold every scheduler slot when
    // bg_concurrency >= max_batch, so the injector cannot admit until a bg
    // stream dies. That poisons mixed-load ITL (issue #470). Warn loudly but
    // do not fail — `max_batch == bg_concurrency` is an intentional
    // starvation / negative-control cell.
    //
    // Only Qwen3.5 wires `--max-batch` into its scheduler admission cap; the
    // other model lines ignore it and keep their own capacity, so comparing
    // `bg_concurrency` against `cli.max_batch` for them would be a false alarm.
    // `ModelType::Qwen35` only exists under the feature, so gate the whole check.
    #[cfg(feature = "qwen35-4b")]
    if matches!(model_type, ModelType::Qwen35) && args.bg_concurrency >= cli.max_batch {
        let msg = format!(
            "slot starvation: --bg-concurrency ({}) >= --max-batch ({}), \
             injector has no free slot while all background streams are live; \
             treat as invalid / negative-control unless intentional \
             (prefer --max-batch > --bg-concurrency, e.g. 5 > 4)",
            args.bg_concurrency, cli.max_batch
        );
        log::warn!("{msg}");
        warnings.push(msg);
    }

    info!(
        "mixed-load warmup: {} round(s) at bg_concurrency={}",
        args.run.warmup, args.bg_concurrency
    );
    mixed_warmup(
        &handle,
        args.bg_prompt_len,
        args.bg_concurrency,
        args.run.warmup,
    )?;

    // ---- Mixed phase ----
    info!(
        "mixed-load: {} background decode streams (prompt={}, output={}); injecting {} prompt(s) of {} tokens at {} QPS",
        args.bg_concurrency,
        args.bg_prompt_len,
        args.bg_output_len,
        args.num_injections,
        args.inj_prompt_len,
        args.qps
    );
    let stop = Arc::new(AtomicBool::new(false));
    let counters: Arc<[AtomicUsize]> = (0..args.bg_concurrency)
        .map(|_| AtomicUsize::new(0))
        .collect();
    let bg_handles = spawn_background_streams(
        &handle,
        args.bg_prompt_len,
        args.bg_output_len,
        args.bg_concurrency,
        &stop,
        &counters,
    );
    if !wait_for_head_start(&counters, args.head_start_tokens, Duration::from_secs(120)) {
        warnings.push(format!(
            "head-start of {} tokens not reached within 120s; injection started anyway",
            args.head_start_tokens
        ));
    }

    let mixed_window_start = Instant::now();
    let inj = run_injector(
        &handle,
        args.inj_prompt_len,
        args.inj_output_len,
        args.qps,
        args.num_injections,
        args.inj_warm_frac,
    )?;
    stop.store(true, Ordering::Release);
    let mixed_window = mixed_window_start.elapsed();

    let mut streams = Vec::with_capacity(args.bg_concurrency);
    for worker in bg_handles {
        streams.push(worker.join().expect("background worker panicked")?);
    }

    if inj.overruns > 0 {
        warnings.push(format!(
            "{} injection(s) overran the {:.0}ms QPS slot (prefill longer than 1/qps); arrivals were not evenly paced",
            inj.overruns,
            1000.0 / args.qps
        ));
    }
    let early = streams.iter().filter(|s| s.finished_early).count();
    if early > 0 {
        warnings.push(format!(
            "{early} background stream(s) hit --bg-output-len before the run ended; raise --bg-output-len to keep steady-state concurrency constant"
        ));
    }

    let mixed_itl = build_mixed_itl(&streams, &inj.windows).context(
        "no background decode gaps recorded; increase --bg-output-len or --num-injections",
    )?;

    // ---- Baseline phase (decode-only control over the same wall-clock) ----
    let baseline_itl = if args.skip_baseline {
        None
    } else {
        info!(
            "mixed-load baseline: decode-only for {:.1}s",
            mixed_window.as_secs_f64()
        );
        run_baseline(&handle, args, mixed_window, &mut warnings)?
    };

    let mixed_p50_ms = mixed_itl.all.p50_ms;
    let mixed_p99_ms = mixed_itl.all.p99_ms;
    let decision_inputs = MixedDecisionInputs {
        baseline_p50_ms: baseline_itl.as_ref().map(|b| b.p50_ms),
        baseline_p99_ms: baseline_itl.as_ref().map(|b| b.p99_ms),
        mixed_p50_ms,
        mixed_p99_ms,
        p99_delta_ms: baseline_itl.as_ref().map(|b| mixed_p99_ms - b.p99_ms),
        p99_delta_pct: baseline_itl
            .as_ref()
            .map(|b| delta_pct(mixed_p99_ms, b.p99_ms)),
    };

    Ok(BenchReport::Mixed(Box::new(MixedLoadReport {
        commit: git_short_commit(),
        date: today_date(),
        gpu: gpu_name(),
        run: run_info(cli, "mixed", model_type, load_ms, cuda_graph),
        config: MixedLoadConfig {
            bg_prompt_len: args.bg_prompt_len,
            bg_concurrency: args.bg_concurrency,
            bg_output_len: args.bg_output_len,
            inj_prompt_len: args.inj_prompt_len,
            inj_output_len: args.inj_output_len,
            qps: args.qps,
            num_injections: args.num_injections,
            inj_warm_frac: args.inj_warm_frac,
            warmup: args.run.warmup,
            seed: args.run.seed,
            max_batch: cli.max_batch,
            max_prefill_tokens: cli.max_prefill_tokens,
        },
        background_generated_tokens: summarize_counts(
            &streams
                .iter()
                .map(|stream| stream.generated_tokens.len())
                .collect::<Vec<_>>(),
        ),
        background_generated_token_traces: streams
            .iter()
            .map(|stream| generated_token_trace(&stream.generated_tokens))
            .collect(),
        baseline_itl,
        mixed_itl,
        injections: inj.records,
        decision_inputs,
        warnings,
    })))
}
