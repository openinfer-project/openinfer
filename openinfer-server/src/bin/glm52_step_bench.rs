//! GLM5.2 whole-step decode bench: load the engine once, then measure steady
//! bucket-{1,2,4,8} decode ms/step in-process — no HTTP, no tokenizer, no
//! 20-minute serve/probe cycle per kernel iteration.
//!
//! Slot placement is least-loaded-rank-first, so submitting `8 × bucket`
//! concurrent requests pins every rank at `bucket` resident rows and the
//! coordinator replays that bucket's whole-step graph with every row real
//! (no padding). Each request gets distinct random prompt ids because
//! identical prompts under-measure MoE decode via degenerate expert routing.
//!
//!   cargo run -r --bin glm52_step_bench --features glm52 -- \
//!       --model-path models/GLM5.2 --buckets 1,2,4,8 --steps 256
//!
//! For kernel attribution, wrap a single bucket under nsys — the trace is
//! load + short admission ramp + pure steady steps, far cleaner than
//! windowing a live server:
//!
//!   nsys profile --cuda-graph-trace=node -s none -o b8 \
//!       target/release/glm52_step_bench --model-path M --buckets 8

use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;

use openinfer::logging;
use openinfer::sampler::SamplingParams;
use openinfer::scheduler::{SchedulerHandle, SchedulerRequest, TokenEvent, TokenSink};

#[derive(Parser, Debug)]
#[command(
    name = "glm52_step_bench",
    about = "GLM5.2 steady-state decode step bench (per-rank bucket sweep, one weight load)"
)]
struct Cli {
    #[arg(long)]
    model_path: PathBuf,
    /// Per-rank decode buckets to measure; each runs 8 × bucket concurrent requests.
    #[arg(long, default_value = "1,2,4,8", value_delimiter = ',')]
    buckets: Vec<usize>,
    /// Prompt length per request (distinct random ids, span-ingested, untimed).
    #[arg(long, default_value_t = 64)]
    ctx: usize,
    /// Decode tokens per request; steady window = gaps after --warmup-steps.
    #[arg(long, default_value_t = 256)]
    steps: usize,
    /// Leading inter-token gaps dropped per stream (admission ramp + first replays).
    #[arg(long, default_value_t = 32)]
    warmup_steps: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Also measure solo span-8 ingestion: one request with this many prompt
    /// tokens, ms per span step = the DSpark span-verify / TTFT shape (one
    /// real slot, other ranks padded) — a different point than the diverse
    /// full-load bucket sweep above. 0 = off.
    #[arg(long, default_value_t = 0)]
    ingest_tokens: usize,
}

const GLM52_RANKS: usize = 8;

fn main() -> Result<()> {
    logging::init_default();
    let cli = Cli::parse();
    ensure!(!cli.buckets.is_empty(), "--buckets must be non-empty");
    ensure!(
        cli.steps > cli.warmup_steps + 8,
        "--steps ({}) must exceed --warmup-steps ({}) with room for a steady window",
        cli.steps,
        cli.warmup_steps
    );

    let load_start = Instant::now();
    let handle = openinfer_glm52::launch(
        &cli.model_path,
        openinfer_glm52::Glm52LaunchOptions {
            tp_size: 1,
            dp_size: GLM52_RANKS,
            dspark_draft_model_path: None,
            max_model_len: None,
        },
    )
    .context("failed to start GLM5.2 engine")?;
    println!("load: {:.1}s", load_start.elapsed().as_secs_f64());

    println!(
        "{:>6} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "bucket", "conc", "p50 ms", "p90 ms", "p99 ms", "mean ms", "tok/s"
    );
    for &bucket in &cli.buckets {
        let gaps = measure_bucket(&handle, bucket, &cli)
            .with_context(|| format!("bucket {bucket} failed"))?;
        let conc = GLM52_RANKS * bucket;
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let p50 = ms(percentile(&gaps, 0.50));
        let mean = ms(gaps.iter().sum::<Duration>()) / gaps.len() as f64;
        println!(
            "{:>6} {:>5} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.0}",
            bucket,
            conc,
            p50,
            ms(percentile(&gaps, 0.90)),
            ms(percentile(&gaps, 0.99)),
            mean,
            conc as f64 * 1000.0 / p50,
        );
    }
    if cli.ingest_tokens > 0 {
        // Twice: the first run doubles as warm-up, the second is the number.
        for round in 0..2 {
            let prompt = random_prompt(cli.ingest_tokens, cli.seed, 99, round);
            let started = Instant::now();
            let stamps = run_stream(
                &handle,
                format!("step-bench-ingest-{round}"),
                prompt,
                SamplingParams {
                    ignore_eos: true,
                    ..SamplingParams::default()
                },
                1,
            )?;
            ensure!(stamps.len() == 1, "ingest emitted {} tokens", stamps.len());
            let wall = started.elapsed().as_secs_f64();
            let span_steps = cli.ingest_tokens.div_ceil(8);
            println!(
                "ingest[{round}]: {} tokens, wall {:.2}s, {:.2} ms/span8-step (solo, one real slot)",
                cli.ingest_tokens,
                wall,
                wall * 1e3 / span_steps as f64,
            );
        }
    }
    Ok(())
}

/// Run `8 × bucket` concurrent streams to completion and pool their steady
/// inter-token gaps. Every stream has the same ctx and step count, so all
/// slots stay resident together: after the admission ramp (dropped via
/// --warmup-steps) each gap is one whole-step graph replay of `bucket`.
fn measure_bucket(handle: &SchedulerHandle, bucket: usize, cli: &Cli) -> Result<Vec<Duration>> {
    let conc = GLM52_RANKS * bucket;
    let params = SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    };
    let workers: Vec<_> = (0..conc)
        .map(|idx| {
            let handle = handle.clone();
            let prompt = random_prompt(cli.ctx, cli.seed, bucket as u64, idx as u64);
            let steps = cli.steps;
            let request_id = format!("step-bench-{bucket}-{idx}");
            thread::spawn(move || run_stream(&handle, request_id, prompt, params, steps))
        })
        .collect();

    let mut gaps = Vec::with_capacity(conc * cli.steps.saturating_sub(cli.warmup_steps));
    for worker in workers {
        let stamps = worker.join().expect("stream worker panicked")?;
        ensure!(
            stamps.len() == cli.steps,
            "a stream emitted {} tokens, expected {} — batch did not stay homogeneous",
            stamps.len(),
            cli.steps
        );
        // stamps[i] - stamps[i-1] spans exactly one step; skip the ramp.
        gaps.extend(
            stamps
                .windows(2)
                .skip(cli.warmup_steps)
                .map(|w| w[1].duration_since(w[0])),
        );
    }
    ensure!(!gaps.is_empty(), "no steady gaps collected");
    gaps.sort_unstable();
    Ok(gaps)
}

fn run_stream(
    handle: &SchedulerHandle,
    request_id: String,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    max_tokens: usize,
) -> Result<Vec<Instant>> {
    let (token_tx, mut token_rx) = TokenSink::standalone();
    handle
        .submit(SchedulerRequest {
            request_id: Some(request_id),
            queued_at_unix_s: None,
            prompt_tokens,
            params,
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .map_err(|e| anyhow::anyhow!("scheduler submit failed: {e}"))?;

    let mut stamps = Vec::with_capacity(max_tokens);
    loop {
        match token_rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { .. }) => stamps.push(Instant::now()),
            Some(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => return Ok(stamps),
            Some(TokenEvent::Error { message, .. }) => bail!("request failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => bail!("request rejected: {message}"),
            None => bail!("scheduler channel closed"),
        }
    }
}

/// Deterministic distinct prompt ids (LCG, safe low-id range). Distinctness
/// across requests is what exercises diverse expert routing.
fn random_prompt(ctx: usize, seed: u64, bucket: u64, idx: u64) -> Vec<u32> {
    let mut state = seed ^ (bucket << 32) ^ (idx << 16) ^ 0x9E37_79B9_7F4A_7C15;
    (0..ctx)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1000 + ((state >> 33) % 19000) as u32
        })
        .collect()
}

fn percentile(sorted: &[Duration], q: f64) -> Duration {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}
