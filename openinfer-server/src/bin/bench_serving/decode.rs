//! Decode sweep: steady-state decode TPOT across ctx × batch.
//!
//! Decode TPOT must not include prefill. To strip it, each cell runs two rounds
//! against the scheduler:
//!   1. warm — submit the batch with `max_tokens = 1` so the ctx-length KV lands
//!      in the prefix cache (not timed);
//!   2. measure — resubmit the same prompts with `max_tokens = decode_steps`.
//!      Prefill now hits the cache (near-instant), so all requests enter decode
//!      almost together and the timed inter-token gaps are pure decode.
//!
//! The hit is verified per request via `Scheduled.cached_tokens ≈ ctx`; if the
//! prefill was not served from cache the measurement would be polluted, so the
//! cell errors out rather than report a prefill-contaminated TPOT.
//!
//! A decode batch keeps every request resident for the whole run, so the KV
//! peak is `batch × (ctx + decode_steps)` — heavier than prefill. The capacity
//! guard refuses the sweep if any cell exceeds the pool.

use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use log::{info, warn};
use openinfer::sampler::SamplingParams;
use openinfer::scheduler::{KvCapacity, SchedulerHandle, SchedulerRequest, TokenEvent, TokenSink};
use openinfer::server_engine::ModelType;

use crate::cli::{Cli, DecodeArgs};
use crate::exec::{BenchModel, run_scheduler_stream};
use crate::metrics::summarize_durations;
use crate::prompt::draw_distinct_prompts;
use crate::report::{BenchReport, DecodeCell, DecodeReport, DecodeWorkload};
use crate::runners::{normalize_sizes, run_info};

/// Tokens of the prompt allowed to be uncached and still count as a hit. Prefix
/// caching commits whole blocks (block_size 16), so the tail partial block is
/// never cached; 64 is a generous few-block slack above that.
const CACHE_HIT_SLACK: usize = 64;

pub(crate) fn run_decode(
    model: &mut dyn BenchModel,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    args: &DecodeArgs,
) -> Result<BenchReport> {
    ensure!(args.iters > 0, "--iters must be > 0");
    ensure!(
        args.decode_steps > args.warmup_steps,
        "--decode-steps ({}) must exceed --warmup-steps ({})",
        args.decode_steps,
        args.warmup_steps
    );
    let ctxs = normalize_sizes(&args.ctxs, "--ctxs")?;
    let batches = normalize_sizes(&args.batches, "--batches")?;

    let handle = model.scheduler_handle().context(
        "decode requires a scheduler-backed model; deepseek-v2-lite is generator-based — use `request`",
    )?;
    let capacity = handle.kv_capacity();
    guard_capacity(capacity, &ctxs, &batches, args.decode_steps)?;

    let sampling = SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    };
    let mut prompt_salt = 0usize;
    let mut cells = Vec::with_capacity(ctxs.len() * batches.len());
    for &ctx in &ctxs {
        for &batch in &batches {
            info!("decode cell: ctx={ctx} batch={batch}");
            cells.push(measure_decode_cell(
                &handle,
                ctx,
                batch,
                args,
                sampling,
                &mut prompt_salt,
            )?);
        }
    }

    Ok(BenchReport::Decode(DecodeReport {
        run: run_info(cli, "decode", model_type, load_ms, cuda_graph),
        workload: DecodeWorkload {
            ctxs,
            batches,
            decode_steps: args.decode_steps,
            warmup_steps: args.warmup_steps,
            distinct_prompts: args.distinct_prompts,
            iters: args.iters,
            seed: args.seed,
            kv_capacity_tokens: capacity.map(KvCapacity::total_tokens),
        },
        cells,
    }))
}

/// Refuse the sweep if any cell's peak resident KV exceeds the pool. Each of the
/// `batch` requests grows to `ctx + decode_steps` tokens and the scheduler
/// allocates whole blocks, so a cell needs `batch × ⌈(ctx + decode_steps) /
/// block_size⌉` blocks — summing raw tokens would under-count. `None` capacity
/// downgrades to a warning.
fn guard_capacity(
    capacity: Option<KvCapacity>,
    ctxs: &[usize],
    batches: &[usize],
    decode_steps: usize,
) -> Result<()> {
    let Some(cap) = capacity else {
        warn!(
            "model did not report KV capacity; skipping the decode capacity guard \
             (an oversized cell will fail mid-run instead)"
        );
        return Ok(());
    };

    let blocks = |c: usize, b: usize| b.saturating_mul(cap.blocks_for(c + decode_steps));
    let over: Vec<String> = ctxs
        .iter()
        .flat_map(|&c| batches.iter().map(move |&b| (c, b)))
        .filter(|&(c, b)| blocks(c, b) > cap.total_blocks)
        .map(|(c, b)| format!("ctx{c}×bs{b} ({} blocks)", blocks(c, b)))
        .collect();

    if !over.is_empty() {
        bail!(
            "KV pool holds {} blocks ({} tokens); these decode cells exceed it \
             (batch × ⌈(ctx + decode_steps)/{}⌉ blocks): {}. \
             Lower --ctxs/--batches/--decode-steps.",
            cap.total_blocks,
            cap.total_tokens(),
            cap.block_size,
            over.join(", ")
        );
    }
    info!(
        "KV capacity {} blocks ({} tokens); all {} decode cells fit",
        cap.total_blocks,
        cap.total_tokens(),
        ctxs.len() * batches.len()
    );
    Ok(())
}

fn measure_decode_cell(
    handle: &SchedulerHandle,
    ctx: usize,
    batch: usize,
    args: &DecodeArgs,
    sampling: SamplingParams,
    prompt_salt: &mut usize,
) -> Result<DecodeCell> {
    // Distinct cold prompts so every request owns a full ctx-length KV (true
    // N-way decode). Drawn once and reused across iterations and both rounds, so
    // round 2 (the timed one) hits the cache round 1 populated.
    let prompts = draw_distinct_prompts(args.distinct_prompts, batch, ctx, args.seed, prompt_salt);

    let mut tbts: Vec<Duration> = Vec::new();
    for iter in 0..args.iters {
        let tag = format!("{ctx}-{batch}-{iter}");
        // Round 1: warm the prefix cache (not timed).
        warm_round(handle, &prompts, sampling, &tag)?;
        // Round 2: decode with the prefill served from cache.
        let streams = measure_round(handle, &prompts, sampling, args.decode_steps, &tag)?;
        for stream in &streams {
            ensure!(
                stream.emitted == args.decode_steps,
                "decode cell ctx{ctx}×bs{batch}: a request emitted {} tokens, expected {} \
                 (rejected, or the batch did not stay homogeneous?)",
                stream.emitted,
                args.decode_steps
            );
            // A real hit covers all of the prompt bar the final partial block.
            // cached_tokens == 0 means nothing matched — and some lines never
            // emit a real count (Kimi reports 0 until prefix-usage lands;
            // Qwen3.5 emits no Scheduled at all), in which case the slack would
            // let small ctx masquerade as a hit and report a prefill-polluted
            // TPOT. Require a positive, near-full hit so those lines fail loudly
            // instead.
            ensure!(
                stream.cached_tokens > 0 && stream.cached_tokens + CACHE_HIT_SLACK >= ctx,
                "decode cell ctx{ctx}×bs{batch}: prefill was not served from cache \
                 (cached {} of {ctx} tokens) — decode TPOT would include prefill. \
                 This needs a scheduler with prefix caching that reports \
                 Scheduled.cached_tokens (qwen3 does; kimi/qwen3.5 do not yet).",
                stream.cached_tokens
            );
            tbts.extend(stream.tbt.iter().skip(args.warmup_steps).copied());
        }
    }
    ensure!(
        !tbts.is_empty(),
        "no steady decode samples collected for ctx{ctx}×bs{batch}"
    );

    let tpot_ms = summarize_durations(&tbts);
    let decode_tok_s = (tpot_ms.p50_ms > 0.0).then(|| batch as f64 * 1000.0 / tpot_ms.p50_ms);

    Ok(DecodeCell {
        ctx,
        batch,
        decode_steps: args.decode_steps,
        peak_tokens: batch.saturating_mul(ctx + args.decode_steps),
        tpot_ms,
        decode_tok_s,
    })
}

/// Submit the whole batch with `max_tokens = 1` and drain to completion so the
/// ctx-length KV is resident in the prefix cache. Not timed.
fn warm_round(
    handle: &SchedulerHandle,
    prompts: &[Vec<u32>],
    params: SamplingParams,
    tag: &str,
) -> Result<()> {
    let mut workers = Vec::with_capacity(prompts.len());
    for (idx, prompt) in prompts.iter().enumerate() {
        let handle = handle.clone();
        let prompt = prompt.clone();
        let request_id = format!("decode-warm-{tag}-{idx}");
        workers.push(thread::spawn(move || -> Result<()> {
            run_scheduler_stream(&handle, Some(request_id), prompt, params, 1, |_| true)?;
            Ok(())
        }));
    }
    for worker in workers {
        worker
            .join()
            .map_err(|_| anyhow!("decode warm worker panicked"))??;
    }
    Ok(())
}

struct StreamResult {
    cached_tokens: usize,
    emitted: usize,
    tbt: Vec<Duration>,
}

/// Submit the whole batch concurrently with `max_tokens` and record each
/// request's cached-prefix size, emitted count, and inter-token gaps.
fn measure_round(
    handle: &SchedulerHandle,
    prompts: &[Vec<u32>],
    params: SamplingParams,
    max_tokens: usize,
    tag: &str,
) -> Result<Vec<StreamResult>> {
    let mut workers = Vec::with_capacity(prompts.len());
    for (idx, prompt) in prompts.iter().enumerate() {
        let handle = handle.clone();
        let prompt = prompt.clone();
        let request_id = format!("decode-run-{tag}-{idx}");
        workers.push(thread::spawn(move || {
            measure_decode_stream(&handle, request_id, prompt, params, max_tokens)
        }));
    }
    workers
        .into_iter()
        .map(|worker| {
            worker
                .join()
                .map_err(|_| anyhow!("decode worker panicked"))?
        })
        .collect()
}

fn measure_decode_stream(
    handle: &SchedulerHandle,
    request_id: String,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    max_tokens: usize,
) -> Result<StreamResult> {
    let (token_tx, mut token_rx) = TokenSink::standalone();
    handle
        .submit(SchedulerRequest {
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
        .map_err(|e| anyhow!("scheduler submit failed: {e}"))?;

    let mut cached_tokens = 0;
    let mut emitted = 0usize;
    let mut tbt = Vec::with_capacity(max_tokens.saturating_sub(1));
    let mut first_at: Option<Instant> = None;
    let mut prev_at: Option<Instant> = None;
    loop {
        match token_rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Scheduled {
                cached_tokens: cached,
                ..
            }) => cached_tokens = cached,
            Some(TokenEvent::Token { .. }) => {
                let now = Instant::now();
                emitted += 1;
                if first_at.is_none() {
                    first_at = Some(now);
                } else if let Some(prev) = prev_at {
                    tbt.push(now - prev);
                }
                prev_at = Some(now);
            }
            Some(TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            Some(TokenEvent::Error { message, .. }) => bail!("decode request failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => {
                bail!("decode request rejected: {message}")
            }
            None => bail!("scheduler channel closed"),
        }
    }
    Ok(StreamResult {
        cached_tokens,
        emitted,
        tbt,
    })
}
