use std::hint::black_box;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use pegainfer_core::sampler::SamplingParams;
use pegainfer_qwen3_4b::runtime::{
    DecodePlan, DecodeStepItem, PrefillPlan, PrefillStepItem, Qwen3Executor, RequestId,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DEFAULT_CONTEXTS: &[usize] = &[128, 512, 1024, 2048, 4096, 8192, 10_000];
const DEFAULT_MEASURE_ITERS: usize = 5;
const DEFAULT_PROFILE_STEPS: usize = 32;

#[link(name = "cudart")]
unsafe extern "C" {
    #[link_name = "cudaProfilerStart"]
    fn cuda_profiler_start() -> i32;
    #[link_name = "cudaProfilerStop"]
    fn cuda_profiler_stop() -> i32;
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Mode {
    Measure,
    Profile,
}

enum ParsedArgs {
    Run(Args),
    Help,
}

struct Args {
    mode: Mode,
    model_path: String,
    contexts: Vec<usize>,
    iters: usize,
    profile_steps: usize,
    capture_range: bool,
    enable_cuda_graph: bool,
}

fn usage() -> &'static str {
    "usage: qwen3_decode_context [--mode measure|profile] [--model-path PATH] \
     [--contexts 128,512,...] [--iters N] [--profile-steps N] \
     [--capture-range] [--disable-cuda-graph]"
}

fn parse_contexts(raw: &str) -> Result<Vec<usize>> {
    let contexts: Vec<_> = raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| {
            let value = token
                .parse::<usize>()
                .with_context(|| format!("invalid context length `{token}`"))?;
            anyhow::ensure!(value > 0, "context length must be greater than zero");
            Ok(value)
        })
        .collect::<Result<_>>()?;
    anyhow::ensure!(!contexts.is_empty(), "--contexts must not be empty");
    Ok(contexts)
}

fn parse_usize(name: &str, raw: &str) -> Result<usize> {
    let value = raw
        .parse::<usize>()
        .with_context(|| format!("invalid {name} `{raw}`"))?;
    anyhow::ensure!(value > 0, "{name} must be greater than zero");
    Ok(value)
}

fn parse_args() -> Result<ParsedArgs> {
    let mut mode = Mode::Measure;
    let mut model_path =
        std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string());
    let mut contexts = DEFAULT_CONTEXTS.to_vec();
    let mut iters = DEFAULT_MEASURE_ITERS;
    let mut profile_steps = DEFAULT_PROFILE_STEPS;
    let mut capture_range = false;
    let mut enable_cuda_graph = true;

    let mut raw = std::env::args().skip(1);
    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                return Ok(ParsedArgs::Help);
            }
            "--mode" => {
                let value = raw.next().ok_or_else(|| anyhow!("--mode needs a value"))?;
                mode = match value.as_str() {
                    "measure" => Mode::Measure,
                    "profile" => Mode::Profile,
                    _ => bail!("--mode must be `measure` or `profile`"),
                };
            }
            "--model-path" => {
                model_path = raw
                    .next()
                    .ok_or_else(|| anyhow!("--model-path needs a value"))?;
            }
            "--contexts" => {
                let value = raw
                    .next()
                    .ok_or_else(|| anyhow!("--contexts needs a value"))?;
                contexts = parse_contexts(&value)?;
            }
            "--iters" => {
                let value = raw.next().ok_or_else(|| anyhow!("--iters needs a value"))?;
                iters = parse_usize("--iters", &value)?;
            }
            "--profile-steps" => {
                let value = raw
                    .next()
                    .ok_or_else(|| anyhow!("--profile-steps needs a value"))?;
                profile_steps = parse_usize("--profile-steps", &value)?;
            }
            "--capture-range" => {
                capture_range = true;
            }
            "--disable-cuda-graph" => {
                enable_cuda_graph = false;
            }
            _ => bail!("unknown argument `{arg}`\n{}", usage()),
        }
    }

    if mode == Mode::Profile {
        anyhow::ensure!(
            contexts.len() == 1,
            "--mode profile expects exactly one context"
        );
    }

    Ok(ParsedArgs::Run(Args {
        mode,
        model_path,
        contexts,
        iters,
        profile_steps,
        capture_range,
        enable_cuda_graph,
    }))
}

fn synthetic_prompt(seq_len: usize) -> Vec<u32> {
    (0..seq_len).map(|i| ((i % 1000) + 100) as u32).collect()
}

fn greedy_ignore_eos() -> SamplingParams {
    SamplingParams {
        ignore_eos: true,
        ..Default::default()
    }
}

fn next_request_id(next_id: &mut u64) -> RequestId {
    let request_id = RequestId::new(*next_id);
    *next_id += 1;
    request_id
}

fn prefill_one(
    executor: &mut Qwen3Executor,
    request_id: RequestId,
    prompt: &[u32],
    params: SamplingParams,
    rng: &mut StdRng,
) -> Result<u32> {
    let requests = [PrefillStepItem::new(
        request_id,
        prompt.to_vec(),
        params,
        0,
        false,
        rng.random(),
    )];
    let result = executor.execute_prefill(PrefillPlan {
        requests: &requests,
        echo: false,
    })?;
    Ok(result.requests[0].first_token)
}

fn decode_one_step(
    executor: &mut Qwen3Executor,
    request_id: RequestId,
    token: &mut u32,
    params: SamplingParams,
    rng: &mut StdRng,
) -> Result<Duration> {
    let requests = [DecodeStepItem::new(
        request_id,
        *token,
        params,
        0,
        rng.random(),
    )];
    let start = Instant::now();
    let result = executor.execute_decode(DecodePlan {
        requests: &requests,
    })?;
    let elapsed = start.elapsed();
    *token = result.requests[0].token;
    Ok(elapsed)
}

fn warm_decode_graph(
    executor: &mut Qwen3Executor,
    context_len: usize,
    params: SamplingParams,
    next_id: &mut u64,
    rng: &mut StdRng,
) -> Result<()> {
    let request_id = next_request_id(next_id);
    let prompt = synthetic_prompt(context_len);
    let mut token = prefill_one(executor, request_id, &prompt, params, rng)?;
    let _ = decode_one_step(executor, request_id, &mut token, params, rng)?;
    executor.drop_request(request_id)?;
    Ok(())
}

fn sorted_durations(samples: &[Duration]) -> Vec<Duration> {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted
}

fn percentile(sorted: &[Duration], percentile: f64) -> Duration {
    debug_assert!(!sorted.is_empty());
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn print_measurement_header(args: &Args) {
    println!(
        "mode=measure model_path={} cuda_graph={} iters={} contexts={:?}",
        args.model_path, args.enable_cuda_graph, args.iters, args.contexts
    );
    println!("prompt_context,kv_len_during_decode,iters,avg_ms,p50_ms,p90_ms,min_ms,max_ms");
}

fn measure_contexts(args: &Args) -> Result<()> {
    let mut executor = Qwen3Executor::from_runtime(&args.model_path, args.enable_cuda_graph, &[0])?;
    let params = greedy_ignore_eos();
    let mut rng = StdRng::seed_from_u64(42);
    let mut next_id = 0u64;

    if args.enable_cuda_graph {
        warm_decode_graph(&mut executor, 1, params, &mut next_id, &mut rng)?;
    }

    print_measurement_header(args);
    for &context_len in &args.contexts {
        let prompt = synthetic_prompt(context_len);
        let mut samples = Vec::with_capacity(args.iters);
        for _ in 0..args.iters {
            let request_id = next_request_id(&mut next_id);
            let mut token = prefill_one(&mut executor, request_id, &prompt, params, &mut rng)?;
            samples.push(decode_one_step(
                &mut executor,
                request_id,
                &mut token,
                params,
                &mut rng,
            )?);
            black_box(token);
            executor.drop_request(request_id)?;
        }

        let sorted = sorted_durations(&samples);
        let total: Duration = samples.iter().sum();
        let avg = total / samples.len() as u32;
        println!(
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
            context_len,
            context_len + 1,
            samples.len(),
            duration_ms(avg),
            duration_ms(percentile(&sorted, 0.50)),
            duration_ms(percentile(&sorted, 0.90)),
            duration_ms(sorted[0]),
            duration_ms(*sorted.last().unwrap())
        );
    }

    Ok(())
}

fn profiler_start() -> Result<()> {
    let err = unsafe { cuda_profiler_start() };
    anyhow::ensure!(err == 0, "cudaProfilerStart failed with cudaError={err}");
    Ok(())
}

fn profiler_stop() -> Result<()> {
    let err = unsafe { cuda_profiler_stop() };
    anyhow::ensure!(err == 0, "cudaProfilerStop failed with cudaError={err}");
    Ok(())
}

fn profile_context(args: &Args) -> Result<()> {
    let context_len = args.contexts[0];
    let mut executor = Qwen3Executor::from_runtime(&args.model_path, args.enable_cuda_graph, &[0])?;
    let params = greedy_ignore_eos();
    let mut rng = StdRng::seed_from_u64(42);
    let mut next_id = 0u64;

    if args.enable_cuda_graph {
        warm_decode_graph(&mut executor, context_len, params, &mut next_id, &mut rng)?;
    }

    let prompt = synthetic_prompt(context_len);
    let request_id = next_request_id(&mut next_id);
    let mut token = prefill_one(&mut executor, request_id, &prompt, params, &mut rng)?;

    println!(
        "mode=profile model_path={} cuda_graph={} prompt_context={} initial_kv_len_during_decode={} profile_steps={} capture_range={}",
        args.model_path,
        args.enable_cuda_graph,
        context_len,
        context_len + 1,
        args.profile_steps,
        args.capture_range
    );

    if args.capture_range {
        profiler_start()?;
    }

    let start = Instant::now();
    for _ in 0..args.profile_steps {
        let elapsed = decode_one_step(&mut executor, request_id, &mut token, params, &mut rng)?;
        black_box(elapsed);
        black_box(token);
    }
    let elapsed = start.elapsed();

    if args.capture_range {
        profiler_stop()?;
    }

    executor.drop_request(request_id)?;
    println!(
        "profile_total_ms={:.4} profile_avg_tpot_ms={:.4}",
        duration_ms(elapsed),
        duration_ms(elapsed / args.profile_steps as u32)
    );
    Ok(())
}

fn main() -> Result<()> {
    match parse_args()? {
        ParsedArgs::Help => {
            println!("{}", usage());
            Ok(())
        }
        ParsedArgs::Run(args) => match args.mode {
            Mode::Measure => measure_contexts(&args),
            Mode::Profile => profile_context(&args),
        },
    }
}
