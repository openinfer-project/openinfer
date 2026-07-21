//! Prefill sweep: cold-prefill TTFT across prompt_len × batch.
//!
//! A prefill batch holds `batch` requests resident, each occupying
//! `⌈prompt_len / block_size⌉` KV blocks, so a big enough cell will not fit the
//! pool. The engine already knows its pool size (`EngineHandle::kv_capacity`),
//! so the sweep checks every requested cell up front and refuses the whole run
//! if any exceeds capacity — the user never computes per-token KV by hand, and
//! the run never silently under-batches or OOMs mid-sweep.

use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use log::info;
use log::warn;
use openinfer::sampler::SamplingParams;
use openinfer::scheduler::KvCapacity;
use openinfer::server_engine::ModelType;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::cli::Cli;
use crate::cli::PrefillArgs;
use crate::exec::BenchModel;
use crate::metrics::summarize_durations;
use crate::prompt::draw_distinct_prompts;
use crate::report::BenchReport;
use crate::report::PrefillCell;
use crate::report::PrefillReport;
use crate::report::PrefillWorkload;
use crate::runners::normalize_sizes;
use crate::runners::run_info;
use crate::runners::validate_run_args;

pub(crate) fn run_prefill(
    model: &mut dyn BenchModel,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    args: &PrefillArgs,
) -> Result<BenchReport> {
    validate_run_args(&args.run)?;
    let prompt_lens = normalize_sizes(&args.prompt_lens, "--prompt-lens")?;
    let batches = normalize_sizes(&args.batches, "--batches")?;

    let handle = model.scheduler_handle().context(
        "prefill requires a scheduler-backed model; deepseek-v2-lite is generator-based — use `request`",
    )?;
    let capacity = handle.kv_capacity();
    guard_capacity(capacity, &prompt_lens, &batches)?;

    let sampling = SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    };
    // Monotonic across the whole sweep so no two prompts (any cell, any
    // iteration, any request) ever repeat — every prefill stays cold.
    let mut prompt_salt = 0usize;
    let mut cells = Vec::with_capacity(prompt_lens.len() * batches.len());
    for &prompt_len in &prompt_lens {
        for &batch in &batches {
            info!("prefill cell: prompt_len={prompt_len} batch={batch}");
            cells.push(measure_prefill_cell(
                model,
                prompt_len,
                batch,
                args,
                &sampling,
                &mut prompt_salt,
            )?);
        }
    }

    Ok(BenchReport::Prefill(PrefillReport {
        run: run_info(cli, "prefill", model_type, load_ms, cuda_graph),
        workload: PrefillWorkload {
            prompt_lens,
            batches,
            distinct_prompts: args.distinct_prompts,
            warmup: args.run.warmup,
            iters: args.run.iters,
            seed: args.run.seed,
            kv_capacity_tokens: capacity.map(KvCapacity::total_tokens),
        },
        cells,
    }))
}

/// Refuse the whole sweep if any requested cell's resident KV exceeds the pool.
/// The scheduler allocates whole blocks per request, so a cell needs
/// `batch × ⌈prompt_len / block_size⌉` blocks — summing raw tokens would
/// under-count and admit a cell the scheduler then defers, contaminating TTFT.
/// A `None` capacity (model did not report it) downgrades to a warning: we
/// cannot pre-check, so a too-large cell would surface as a mid-run rejection.
fn guard_capacity(
    capacity: Option<KvCapacity>,
    prompt_lens: &[usize],
    batches: &[usize],
) -> Result<()> {
    let Some(cap) = capacity else {
        warn!(
            "model did not report KV capacity; skipping the prefill capacity guard \
             (an oversized cell will fail mid-run instead)"
        );
        return Ok(());
    };

    let blocks = |l: usize, b: usize| b.saturating_mul(cap.blocks_for(l));
    let over: Vec<String> = prompt_lens
        .iter()
        .flat_map(|&l| batches.iter().map(move |&b| (l, b)))
        .filter(|&(l, b)| blocks(l, b) > cap.total_blocks)
        .map(|(l, b)| format!("{l}×{b} ({} blocks)", blocks(l, b)))
        .collect();

    if !over.is_empty() {
        bail!(
            "KV pool holds {} blocks ({} tokens); these cells exceed it \
             (batch × ⌈prompt_len/{}⌉ blocks): {}. Lower --prompt-lens/--batches.",
            cap.total_blocks,
            cap.total_tokens(),
            cap.block_size,
            over.join(", ")
        );
    }
    info!(
        "KV capacity {} blocks ({} tokens); all {} cells fit",
        cap.total_blocks,
        cap.total_tokens(),
        prompt_lens.len() * batches.len()
    );
    Ok(())
}

fn measure_prefill_cell(
    model: &mut dyn BenchModel,
    prompt_len: usize,
    batch: usize,
    args: &PrefillArgs,
    sampling: &SamplingParams,
    prompt_salt: &mut usize,
) -> Result<PrefillCell> {
    // SchedulerBenchModel ignores the rng (greedy); kept to satisfy the trait.
    let mut rng = StdRng::seed_from_u64(args.run.seed);
    let mut ttfts: Vec<Duration> = Vec::with_capacity(args.run.iters * batch);

    for iter in 0..(args.run.warmup + args.run.iters) {
        // Fresh prompts every iteration — drawn from the sweep-global salt so no
        // prompt repeats anywhere, keeping every prefill cold (a repeat would hit
        // the prefix cache and inflate throughput).
        let prompts = draw_distinct_prompts(
            args.distinct_prompts,
            batch,
            prompt_len,
            args.run.seed,
            prompt_salt,
        );
        // output_len = 1: the request prefills and emits exactly one token, so
        // its TTFT is the prefill latency with no decode steps mixed in.
        let timings = model.timed_generation_batch(&prompts, 1, sampling, &mut rng);
        if iter < args.run.warmup {
            continue;
        }
        for timing in &timings {
            anyhow::ensure!(
                timing.emitted_tokens == 1,
                "prefill cell {prompt_len}×{batch}: a request emitted {} tokens, expected 1 \
                 (was it rejected or did the batch not stay homogeneous?)",
                timing.emitted_tokens
            );
            ttfts.push(timing.ttft);
        }
    }

    let ttft_ms = summarize_durations(&ttfts);
    let total_tokens = prompt_len.saturating_mul(batch);
    let prefill_tok_s =
        (ttft_ms.p50_ms > 0.0).then(|| total_tokens as f64 / (ttft_ms.p50_ms / 1000.0));

    Ok(PrefillCell {
        prompt_len,
        batch,
        total_tokens,
        ttft_ms,
        prefill_tok_s,
    })
}
