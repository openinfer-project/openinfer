//! GLM5.2 whole-step decode bench: load the engine once, then measure steady
//! bucket-{1,2,4,8} decode ms/step in-process — no HTTP, no tokenizer, no
//! 20-minute serve/probe cycle per kernel iteration.
//!
//! EP8 slot placement is least-loaded-rank-first, so submitting `8 × bucket`
//! concurrent requests pins every rank at `bucket` resident rows. Tensor-
//! replicated topologies (`tp8`/`tp4`) have one logical rank with eight slots,
//! so `--buckets 1` submits eight streams and fills the single mirrored
//! bucket-8 shape. Each request gets distinct random prompt ids because
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
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use clap::Parser;
use openinfer::logging;
use openinfer::sampler::SamplingParams;
use openinfer::scheduler::SchedulerHandle;
use openinfer::scheduler::SchedulerRequest;
use openinfer::scheduler::TokenEvent;
use openinfer::scheduler::TokenSink;

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
    /// MoE topology: ep8 (default, all buckets), tp8, or tp4. Tensor-
    /// replicated topologies expose ONE logical rank with 8 slots, so a
    /// bucket value there IS the concurrency (1..=8). TP4 maps it onto its
    /// compact 1/2/4/8 decode graphs; TP8 always replays its single bucket-8
    /// graph (pad rows ride free slots), so smaller values measure
    /// bucket-8-with-pads.
    #[arg(long, default_value = "ep8")]
    moe_topo: String,
    /// Remote rank-host nodes, comma-separated `host:port=ranks` — same
    /// contract as the server flag (EP topologies only; remote ranks come
    /// after the local ones).
    #[arg(long, value_delimiter = ',')]
    rank_hosts: Vec<String>,
    /// Override the VRAM-derived per-request context cap — same contract as
    /// the server flag. Wide-EP topologies need this until the context-cap
    /// ledger accounts for width-scaled DeepEP buffers.
    #[arg(long)]
    max_model_len: Option<usize>,
}

const GLM52_RANKS: usize = 8;

fn main() -> Result<()> {
    logging::init_default();
    let cli = Cli::parse();
    ensure!(!cli.buckets.is_empty(), "--buckets must be non-empty");
    let moe_topo: openinfer_glm52::Glm52MoeTopo = cli.moe_topo.parse().context("--moe-topo")?;
    if is_tensor_replicated_moe(moe_topo) {
        ensure!(
            cli.buckets.iter().all(|&b| (1..=GLM52_RANKS).contains(&b)),
            "--moe-topo {moe_topo:?} holds at most {GLM52_RANKS} concurrent requests (one \
             logical rank); pass --buckets values in 1..={GLM52_RANKS}"
        );
    }
    let (tp_size, dp_size) = (moe_topo.expected_tp_size(), moe_topo.default_dp_size());
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
            tp_size,
            dp_size,
            dspark_draft_model_path: None,
            max_model_len: cli.max_model_len,
            no_prefix_cache: false,
            kv_offload: None,
            moe_topo,
            dump_graph_png: None,
            rank_hosts: cli
                .rank_hosts
                .iter()
                .map(|spec| spec.parse())
                .collect::<Result<Vec<_>>>()
                .context("--rank-hosts")?,
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
        let conc = bench_concurrency(bucket, moe_topo);
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

/// Run [`bench_concurrency`] concurrent streams (`8 × bucket` on EP8, the
/// bucket value itself on tensor-replicated TP) to completion and pool their
/// steady inter-token gaps. Every stream has the same ctx and step count, so
/// all slots stay resident together: after the admission ramp (dropped via
/// --warmup-steps) each gap is one whole-step graph replay of `bucket`.
fn measure_bucket(handle: &SchedulerHandle, bucket: usize, cli: &Cli) -> Result<Vec<Duration>> {
    let moe_topo: openinfer_glm52::Glm52MoeTopo = cli
        .moe_topo
        .parse()
        .expect("moe_topo was parsed successfully in main");
    let conc = bench_concurrency(bucket, moe_topo);
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

fn bench_concurrency(bucket: usize, moe_topo: openinfer_glm52::Glm52MoeTopo) -> usize {
    if is_tensor_replicated_moe(moe_topo) {
        // One logical rank: the bucket value is the concurrency, so TP4's
        // compact bs=1 shape (the PR's headline latency) is directly iterable.
        bucket
    } else {
        // One stream per (logical DP rank, slot): more would make the
        // scheduler run a LARGER per-rank bucket than the label says (EP4
        // has 4 logical ranks, not the EP8 constant).
        moe_topo.default_dp_size() * bucket
    }
}

fn is_tensor_replicated_moe(moe_topo: openinfer_glm52::Glm52MoeTopo) -> bool {
    matches!(
        moe_topo,
        openinfer_glm52::Glm52MoeTopo::Tp8 | openinfer_glm52::Glm52MoeTopo::Tp4
    )
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
            trace_parent: None,
            request_id: Some(request_id),
            queued_at_unix_s: None,
            data_parallel_rank: None,
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
