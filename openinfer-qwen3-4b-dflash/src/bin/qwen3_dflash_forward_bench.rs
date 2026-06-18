use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use half::bf16;
use openinfer_core::tensor::HiddenStates;
use openinfer_qwen3_4b_dflash::{DFlashDraftModel, DFlashTargetHidden};
use safetensors::{Dtype, SafeTensors};
use serde::Serialize;

fn main() -> Result<()> {
    let args = Args::parse()?;
    let model = DFlashDraftModel::load(&args.model_path, args.device)?;
    let config = model.config();
    let ctx = model.device_context();

    let (noise, target_hidden, positions, ctx_len, q_len) = if let Some(fixture) = &args.fixture {
        let bytes = std::fs::read(fixture)
            .with_context(|| format!("failed to read fixture {}", fixture.display()))?;
        let st = SafeTensors::deserialize(&bytes).context("parse fixture")?;
        let noise = read_bf16(&st, "noise_embedding", &[1, args.q_len, config.hidden_size])?;
        let target_hidden = read_bf16(
            &st,
            "target_hidden",
            &[
                1,
                args.ctx_len,
                config.hidden_size * config.target_layer_count(),
            ],
        )?;
        let positions = read_i32(&st, "position_ids", &[1, args.ctx_len + args.q_len])?;
        (noise, target_hidden, positions, args.ctx_len, args.q_len)
    } else {
        let noise = deterministic_bf16(args.q_len * config.hidden_size, 0xD4A5_4B16);
        let target_hidden = deterministic_bf16(
            args.ctx_len * config.hidden_size * config.target_layer_count(),
            0xD4A5_C0DE,
        );
        let positions = (0..(args.ctx_len + args.q_len))
            .map(|pos| pos as i32)
            .collect::<Vec<_>>();
        (noise, target_hidden, positions, args.ctx_len, args.q_len)
    };

    let noise = HiddenStates {
        data: ctx.stream.clone_htod(&noise).context("noise h2d")?,
        hidden_dim: config.hidden_size,
        seq_len: q_len,
    };
    let target_hidden = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&target_hidden)
            .context("target hidden h2d")?,
        hidden_dim: config.hidden_size * config.target_layer_count(),
        seq_len: ctx_len,
    };
    ctx.sync()?;

    let mut cache = model.create_draft_cache(q_len, ctx_len, ctx_len + q_len)?;
    if args.draft_cache {
        model.prepare_step_context(
            DFlashTargetHidden {
                concatenated: &target_hidden,
            },
            &positions,
            &mut cache,
        )?;
        ctx.sync()?;
    }
    for _ in 0..args.warmup {
        if args.draft_cache {
            cache.reset();
            model.prepare_step_context(
                DFlashTargetHidden {
                    concatenated: &target_hidden,
                },
                &positions,
                &mut cache,
            )?;
            let _out = model.forward_with_draft_cache(&noise, &positions, &mut cache)?;
        } else {
            let _out = model.forward_with_cache(
                &noise,
                DFlashTargetHidden {
                    concatenated: &target_hidden,
                },
                &positions,
                &mut cache,
            )?;
        }
        ctx.sync()?;
    }

    let mut latencies_ms = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        ctx.sync()?;
        let started = Instant::now();
        if args.draft_cache {
            cache.reset();
            model.prepare_step_context(
                DFlashTargetHidden {
                    concatenated: &target_hidden,
                },
                &positions,
                &mut cache,
            )?;
            let _out = model.forward_with_draft_cache(&noise, &positions, &mut cache)?;
        } else {
            let _out = model.forward_with_cache(
                &noise,
                DFlashTargetHidden {
                    concatenated: &target_hidden,
                },
                &positions,
                &mut cache,
            )?;
        }
        ctx.sync()?;
        latencies_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    let report = Report {
        schema: 1,
        engine: "openinfer-qwen3-4b-dflash",
        model_path: args.model_path.to_string_lossy().to_string(),
        device: args.device,
        ctx_len: args.ctx_len,
        q_len: args.q_len,
        hidden_size: config.hidden_size,
        target_layer_count: config.target_layer_count(),
        draft_cache: args.draft_cache,
        warmup: args.warmup,
        iters: args.iters,
        latency_ms: Stats::from(&latencies_ms),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[derive(Clone)]
struct Args {
    model_path: PathBuf,
    fixture: Option<PathBuf>,
    device: usize,
    ctx_len: usize,
    q_len: usize,
    warmup: usize,
    iters: usize,
    draft_cache: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut model_path = PathBuf::from("/home/hezhaozhao/models/Qwen3-4B-DFlash-b16");
        let mut fixture = None;
        let mut device = 0usize;
        let mut ctx_len = 2usize;
        let mut q_len = 16usize;
        let mut warmup = 5usize;
        let mut iters = 30usize;
        let mut draft_cache = false;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-path" => model_path = PathBuf::from(next_value(&mut args, &arg)?),
                "--fixture" => fixture = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                "--device" => device = next_value(&mut args, &arg)?.parse()?,
                "--ctx-len" => ctx_len = next_value(&mut args, &arg)?.parse()?,
                "--q-len" => q_len = next_value(&mut args, &arg)?.parse()?,
                "--warmup" => warmup = next_value(&mut args, &arg)?.parse()?,
                "--iters" => iters = next_value(&mut args, &arg)?.parse()?,
                "--draft-cache" | "--context-cache" => draft_cache = true,
                _ => bail!("unknown argument {arg}"),
            }
        }
        if ctx_len == 0 {
            bail!("--ctx-len must be greater than zero");
        }
        if q_len == 0 {
            bail!("--q-len must be greater than zero");
        }
        if iters == 0 {
            bail!("--iters must be greater than zero");
        }
        Ok(Self {
            model_path,
            fixture,
            device,
            ctx_len,
            q_len,
            warmup,
            iters,
            draft_cache,
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
        let value = (bits * 2.0 - 1.0) * 0.125;
        out.push(bf16::from_f32(value));
    }
    out
}

#[derive(Serialize)]
struct Report {
    schema: u32,
    engine: &'static str,
    model_path: String,
    device: usize,
    ctx_len: usize,
    q_len: usize,
    hidden_size: usize,
    target_layer_count: usize,
    draft_cache: bool,
    warmup: usize,
    iters: usize,
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

fn read_bf16(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<bf16>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    if view.dtype() != Dtype::BF16 {
        bail!("{name} must be BF16, got {:?}", view.dtype());
    }
    if view.shape() != shape {
        bail!(
            "{name} shape mismatch: expected {shape:?}, got {:?}",
            view.shape()
        );
    }
    Ok(view
        .data()
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn read_i32(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<i32>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    if view.dtype() != Dtype::I32 {
        bail!("{name} must be I32, got {:?}", view.dtype());
    }
    if view.shape() != shape {
        bail!(
            "{name} shape mismatch: expected {shape:?}, got {:?}",
            view.shape()
        );
    }
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
