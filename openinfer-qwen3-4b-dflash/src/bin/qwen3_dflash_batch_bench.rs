use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use half::bf16;
use openinfer_core::tensor::HiddenStates;
use openinfer_qwen3_4b_dflash::{DFlashBatchInput, DFlashDraftModel, DFlashTargetHidden};
use serde::Serialize;

fn main() -> Result<()> {
    let args = Args::parse()?;
    let model = DFlashDraftModel::load(&args.model_path, args.device)?;
    let config = model.config();
    let ctx = model.device_context();
    let mut reports = Vec::new();

    for &batch_size in &args.batch_sizes {
        let mut noises = Vec::with_capacity(batch_size);
        let mut targets = Vec::with_capacity(batch_size);
        let mut positions = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let noise = deterministic_bf16(args.q_len * config.hidden_size, 0xD4A5_0000 + i as u64);
            let target = deterministic_bf16(
                args.ctx_len * config.hidden_size * config.target_layer_count(),
                0xC0DE_0000 + i as u64,
            );
            noises.push(HiddenStates {
                data: ctx.stream.clone_htod(&noise).context("noise h2d")?,
                hidden_dim: config.hidden_size,
                seq_len: args.q_len,
            });
            targets.push(HiddenStates {
                data: ctx.stream.clone_htod(&target).context("target h2d")?,
                hidden_dim: config.hidden_size * config.target_layer_count(),
                seq_len: args.ctx_len,
            });
            positions.push(
                (0..(args.ctx_len + args.q_len))
                    .map(|pos| pos as i32)
                    .collect::<Vec<_>>(),
            );
        }
        let mut bufs = model.create_batch_buffers(batch_size, args.q_len, args.ctx_len)?;
        let inputs = build_inputs(&noises, &targets, &positions);
        for _ in 0..args.warmup {
            let _ = model.forward_batch(&inputs, &mut bufs)?;
            ctx.sync()?;
        }
        let mut latencies_ms = Vec::with_capacity(args.iters);
        for _ in 0..args.iters {
            ctx.sync()?;
            let started = Instant::now();
            let _ = model.forward_batch(&inputs, &mut bufs)?;
            ctx.sync()?;
            latencies_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        let stats = Stats::from(&latencies_ms);
        let mean_s = stats.mean / 1000.0;
        reports.push(BatchReport {
            batch_size,
            ctx_len: args.ctx_len,
            q_len: args.q_len,
            warmup: args.warmup,
            iters: args.iters,
            draft_tokens_per_s: (batch_size * args.q_len) as f64 / mean_s,
            requests_per_s: batch_size as f64 / mean_s,
            latency_ms: stats,
        });
    }

    let report = Report {
        schema: 1,
        engine: "openinfer-qwen3-4b-dflash-batch",
        model_path: args.model_path.to_string_lossy().to_string(),
        device: args.device,
        hidden_size: config.hidden_size,
        target_layer_count: config.target_layer_count(),
        reports,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn build_inputs<'a>(
    noises: &'a [HiddenStates],
    targets: &'a [HiddenStates],
    positions: &'a [Vec<i32>],
) -> Vec<DFlashBatchInput<'a>> {
    noises
        .iter()
        .zip(targets.iter())
        .zip(positions.iter())
        .map(|((noise, target), position_ids)| DFlashBatchInput {
            noise_embedding: noise,
            target_hidden: DFlashTargetHidden {
                concatenated: target,
            },
            position_ids,
        })
        .collect()
}

#[derive(Clone)]
struct Args {
    model_path: PathBuf,
    device: usize,
    ctx_len: usize,
    q_len: usize,
    warmup: usize,
    iters: usize,
    batch_sizes: Vec<usize>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut model_path = PathBuf::from("/home/hezhaozhao/models/Qwen3-4B-DFlash-b16");
        let mut device = 0usize;
        let mut ctx_len = 2usize;
        let mut q_len = 16usize;
        let mut warmup = 5usize;
        let mut iters = 30usize;
        let mut batch_sizes = vec![1, 2, 4, 8, 16, 32];
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-path" => model_path = PathBuf::from(next_value(&mut args, &arg)?),
                "--device" => device = next_value(&mut args, &arg)?.parse()?,
                "--ctx-len" => ctx_len = next_value(&mut args, &arg)?.parse()?,
                "--q-len" => q_len = next_value(&mut args, &arg)?.parse()?,
                "--warmup" => warmup = next_value(&mut args, &arg)?.parse()?,
                "--iters" => iters = next_value(&mut args, &arg)?.parse()?,
                "--batch-sizes" => {
                    batch_sizes = next_value(&mut args, &arg)?
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                }
                _ => bail!("unknown argument {arg}"),
            }
        }
        if ctx_len == 0 || q_len == 0 || iters == 0 {
            bail!("--ctx-len, --q-len, and --iters must be greater than zero");
        }
        if batch_sizes.is_empty() || batch_sizes.contains(&0) {
            bail!("--batch-sizes must contain positive batch sizes");
        }
        Ok(Self {
            model_path,
            device,
            ctx_len,
            q_len,
            warmup,
            iters,
            batch_sizes,
        })
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{flag} requires a value"))
}

fn deterministic_bf16(len: usize, seed: u64) -> Vec<bf16> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((state >> 32) as u32) as f32 / (u32::MAX as f32);
        out.push(bf16::from_f32((bits * 2.0 - 1.0) * 0.125));
    }
    out
}

#[derive(Serialize)]
struct Report {
    schema: u32,
    engine: &'static str,
    model_path: String,
    device: usize,
    hidden_size: usize,
    target_layer_count: usize,
    reports: Vec<BatchReport>,
}

#[derive(Serialize)]
struct BatchReport {
    batch_size: usize,
    ctx_len: usize,
    q_len: usize,
    warmup: usize,
    iters: usize,
    draft_tokens_per_s: f64,
    requests_per_s: f64,
    latency_ms: Stats,
}

#[derive(Serialize)]
struct Stats {
    mean: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn from(values: &[f64]) -> Self {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        Self {
            mean,
            p50: percentile(&sorted, 0.50),
            p90: percentile(&sorted, 0.90),
            p99: percentile(&sorted, 0.99),
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        }
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
