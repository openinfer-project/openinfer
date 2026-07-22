//! In-process inference benchmark CLI.
//!
//! Usage:
//!   cargo run -r --bin bench_serving -- [GLOBAL_OPTIONS] <SUBCOMMAND> [OPTIONS]
//!
//! Examples:
//!   cargo run -r --bin bench_serving -- request --prompt "Tell me a story" --output-len 128
//!   cargo run -r --bin bench_serving -- request --prompt-len 512 --output-len 64
//!   cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512 --output-lens 32,128
//!   cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32

#![cfg_attr(
    not(any(
        feature = "deepseek-v2-lite",
        feature = "kimi-k2",
        feature = "qwen3",
        feature = "qwen35-4b"
    )),
    allow(unused_imports, unused_variables, dead_code)
)]

use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use log::debug;
use openinfer::logging;
use openinfer::scheduler::SchedulerHandle;
use openinfer::server_engine::ModelType;
use openinfer::server_engine::detect_model_type;
use openinfer_core::engine::EngineLoadOptions;
use openinfer_core::engine::EpBackend;
#[cfg(feature = "kimi-k2")]
use openinfer_core::parallel::ParallelConfig;
use openinfer_vllm_support::load_tokenizer as load_vllm_tokenizer;
use vllm_text::tokenizer::DynTokenizer;

mod cli;
mod decode;
mod exec;
mod metrics;
mod mixed;
mod prefill;
mod prompt;
mod render;
mod report;
mod runners;
mod snapshot;
use cli::Cli;
use cli::CliQwen35SchedulerPolicy;
use cli::Command;
use exec::BenchModel;
#[cfg(feature = "deepseek-v2-lite")]
use exec::DeepSeekV2LiteBenchModel;
use exec::SchedulerBenchModel;
use metrics::dur_ms;
use runners::emit_report;
use runners::run_command;
use snapshot::run_compare;
use snapshot::run_snapshot;

fn command_seed(cli: &Cli) -> u64 {
    match &cli.command {
        Command::Request(args) => args.run.seed,
        Command::Prefill(args) => args.run.seed,
        Command::Decode(args) => args.seed,
        Command::Matrix(args) => args.run.seed,
        Command::Curve(args) => args.run.seed,
        Command::Snapshot(args) => args.run.seed,
        Command::Compare(_) => 42,
        Command::Mixed(args) => args.run.seed,
    }
}

#[cfg(feature = "kimi-k2")]
fn kimi_parallel_config(tp_size: usize, dp_size: usize) -> Result<ParallelConfig> {
    anyhow::ensure!(tp_size > 0, "--tp-size must be positive");
    anyhow::ensure!(dp_size > 0, "--dp-size must be positive");
    Ok(ParallelConfig::new(tp_size, dp_size))
}

fn validate_qwen35_scheduler_policy(
    model_type: ModelType,
    policy: CliQwen35SchedulerPolicy,
) -> Result<()> {
    #[cfg(feature = "qwen35-4b")]
    if matches!(model_type, ModelType::Qwen35) {
        return Ok(());
    }
    #[cfg(not(feature = "qwen35-4b"))]
    let _ = model_type;
    if policy == CliQwen35SchedulerPolicy::Off {
        return Ok(());
    }
    anyhow::bail!("--qwen35-scheduler-policy=auto is supported only for Qwen3.5")
}

fn validate_qwen35_max_batch(model_type: ModelType, max_batch: usize) -> Result<()> {
    anyhow::ensure!(max_batch > 0, "--max-batch must be > 0");
    #[cfg(feature = "qwen35-4b")]
    if matches!(model_type, ModelType::Qwen35) {
        anyhow::ensure!(
            max_batch <= openinfer_qwen35_4b::MAX_DECODE_BATCH,
            "--max-batch must be <= {} for Qwen3.5",
            openinfer_qwen35_4b::MAX_DECODE_BATCH
        );
        return Ok(());
    }
    #[cfg(not(feature = "qwen35-4b"))]
    let _ = model_type;
    Ok(())
}

fn dispatch(
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    model: &mut dyn BenchModel,
    tokenizer: &DynTokenizer,
) -> Result<()> {
    if let Command::Snapshot(args) = &cli.command {
        run_snapshot(model, cli, model_type, args)
    } else {
        let report = run_command(cli, model_type, load_ms, cuda_graph, model, tokenizer)?;
        emit_report(cli, &report)
    }
}

fn main() -> Result<()> {
    logging::init_default();

    let cli = Cli::parse();

    // Compare needs no model loading
    if let Command::Compare(ref args) = cli.command {
        return run_compare(args);
    }

    debug!(
        "bench_serving starting: command={} model_path={} cuda_graph={} format={:?}",
        match &cli.command {
            Command::Request(_) => "request",
            Command::Prefill(_) => "prefill",
            Command::Decode(_) => "decode",
            Command::Matrix(_) => "matrix",
            Command::Curve(_) => "curve",
            Command::Snapshot(_) => "snapshot",
            Command::Compare(_) => "compare",
            Command::Mixed(_) => "mixed",
        },
        cli.model_path,
        cli.cuda_graph,
        cli.format
    );
    let model_type = detect_model_type(&cli.model_path)
        .with_context(|| format!("failed to detect model type from {}", cli.model_path))?;
    debug!("Detected model type: {:?}", model_type);
    validate_qwen35_scheduler_policy(model_type, cli.qwen35_scheduler_policy)?;
    validate_qwen35_max_batch(model_type, cli.max_batch)?;
    let load_start = Instant::now();

    // Shared tail for every scheduler-backed model: load the tokenizer, stamp
    // the elapsed load time, wrap the handle, and dispatch. The per-model arms
    // below differ only in how they construct the engine handle.
    let finish = |handle: SchedulerHandle, cuda_graph: bool| -> Result<()> {
        let tokenizer = load_vllm_tokenizer(&cli.model_path)?;
        let load_ms = dur_ms(load_start.elapsed());
        let mut bench = SchedulerBenchModel { handle };
        dispatch(
            &cli, model_type, load_ms, cuda_graph, &mut bench, &tokenizer,
        )
    };

    match model_type {
        #[cfg(feature = "deepseek-v2-lite")]
        ModelType::DeepSeekV2Lite => {
            // Distinct bench type (not scheduler-backed), so it keeps its own tail.
            let generator = openinfer_deepseek_v2_lite::DeepSeekV2LiteEp2Generator::load(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: false,
                    device_ordinals: vec![0, 1],
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
            )?;
            let tokenizer = load_vllm_tokenizer(&cli.model_path)?;
            let load_ms = dur_ms(load_start.elapsed());
            let mut bench = DeepSeekV2LiteBenchModel { generator };
            dispatch(&cli, model_type, load_ms, false, &mut bench, &tokenizer)
        }
        #[cfg(feature = "glm52")]
        ModelType::Glm52 => {
            anyhow::bail!("bench_serving is not supported for the GLM5.2 load-weight-only branch")
        }
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => {
            let parallel = kimi_parallel_config(cli.tp_size, cli.dp_size)?;
            let handle = openinfer_kimi_k2::start_engine(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: cli.cuda_graph,
                    device_ordinals: (0..parallel.ep_world()).collect(),
                    parallel_config: Some(parallel),
                    ep_backend: cli.ep_backend.into(),
                    seed: command_seed(&cli),
                },
            )?;
            finish(handle, cli.cuda_graph)
        }
        #[cfg(feature = "qwen3")]
        ModelType::Qwen3 => {
            // Chunked-prefill budget from --max-prefill-tokens (a huge value
            // forwards the whole prompt in one step, i.e. chunking off, for the
            // chunked-vs-not sweep); omit for the model default.
            let max_prefill_tokens = cli
                .max_prefill_tokens
                .filter(|&v| v > 0)
                .unwrap_or(openinfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS);
            let handle = openinfer_qwen3::start_engine_with_offload(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: cli.cuda_graph,
                    device_ordinals: vec![0],
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
                openinfer_qwen3::Qwen3OffloadOptions::disabled(),
                false,
                max_prefill_tokens,
                openinfer_qwen3::Qwen3MemoryOptions::default(),
                openinfer_qwen3::DecodeOverlap::Off,
                false,
                None,
                false,
            )?;
            finish(handle, cli.cuda_graph)
        }
        #[cfg(feature = "qwen35-4b")]
        ModelType::Qwen35 => {
            // Chunked-prefill budget from --max-prefill-tokens (mirrors the Qwen3
            // path); omit for the model default. Concurrent capacity from
            // --max-batch (was hardcoded to 4; see issue #470).
            let max_prefill_tokens = cli
                .max_prefill_tokens
                .filter(|&v| v > 0)
                .unwrap_or(openinfer_qwen35_4b::DEFAULT_MAX_PREFILL_TOKENS);
            let handle = openinfer_qwen35_4b::start_engine_with_capacity_and_policy(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: cli.cuda_graph,
                    device_ordinals: vec![0],
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
                cli.max_batch,
                max_prefill_tokens,
                cli.qwen35_scheduler_policy.resolve(),
            )?;
            finish(handle, cli.cuda_graph)
        }
    }
}

#[cfg(all(test, feature = "deepseek-v2-lite"))]
mod tests {
    use std::time::Duration;

    use openinfer::sampler::SamplingParams;

    use super::*;
    use crate::exec::assert_dsv2_lite_sampling_contract;
    use crate::exec::timings_from_dsv2_lite_attribution;
    use crate::exec::timings_from_dsv2_lite_batched_generation;
    use crate::runners::build_request_metrics;

    #[test]
    fn dsv2_lite_sampling_contract_accepts_bench_params() {
        let sampling = SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        };

        assert_dsv2_lite_sampling_contract(&sampling);
    }

    #[test]
    #[should_panic(expected = "supports greedy decoding only")]
    fn dsv2_lite_sampling_contract_rejects_non_greedy_params() {
        let sampling = SamplingParams {
            temperature: 0.8,
            top_k: -1,
            top_p: 0.95,
            ignore_eos: true,
            ..SamplingParams::default()
        };

        assert_dsv2_lite_sampling_contract(&sampling);
    }

    #[test]
    #[should_panic(expected = "requires ignore_eos=true")]
    fn dsv2_lite_sampling_contract_rejects_eos_enabled_params() {
        let sampling = SamplingParams {
            ignore_eos: false,
            ..SamplingParams::default()
        };

        assert_dsv2_lite_sampling_contract(&sampling);
    }

    #[test]
    fn dsv2_lite_attribution_timings_preserve_decode_steps() {
        let timings = timings_from_dsv2_lite_attribution(
            vec![11, 304, 608],
            3,
            60_000,
            Some(20_000),
            &[19_000, 18_000],
        );

        assert_eq!(timings.ttft, Duration::from_micros(20_000));
        assert_eq!(
            timings.tbt,
            vec![Duration::from_micros(19_000), Duration::from_micros(18_000)]
        );
        assert_eq!(timings.total, Duration::from_micros(60_000));
        assert_eq!(timings.emitted_tokens, 3);
        assert_eq!(timings.generated_tokens, vec![11, 304, 608]);
        assert_eq!(timings.decode_tokens_for_rate, 2);
        assert_eq!(timings.decode_time_for_rate, Duration::from_micros(37_000));
    }

    #[test]
    fn dsv2_lite_batched_timings_use_shared_decode_time_for_rate() {
        let timings = timings_from_dsv2_lite_batched_generation(
            openinfer_deepseek_v2_lite::BatchedGenerationResult {
                tokens: vec![vec![11, 304, 608], vec![11, 304, 608]],
                prefill_next_token_us: vec![20_000, 21_000],
                per_token_decode_us: vec![19_000, 18_000],
                total_generation_us: 80_000,
                stats: openinfer_deepseek_v2_lite::GenerationStats::default(),
            },
            3,
        );

        assert_eq!(timings.len(), 2);
        assert_eq!(timings[0].decode_tokens_for_rate, 4);
        assert_eq!(
            timings[0].decode_time_for_rate,
            Duration::from_micros(37_000)
        );
        assert_eq!(timings[1].decode_tokens_for_rate, 0);
        assert_eq!(timings[1].decode_time_for_rate, Duration::ZERO);

        let metrics = build_request_metrics(&timings);
        assert_eq!(metrics.steady_tpot_ms.unwrap().p50_ms, 18.0);
        assert!(
            metrics.decode_tok_s.unwrap() > 100.0,
            "batched decode tok/s should use one shared step duration instead of duplicating it per row"
        );
    }

    #[test]
    #[should_panic(expected = "timing count mismatch")]
    fn dsv2_lite_attribution_timings_fail_on_missing_decode_samples() {
        let _ = timings_from_dsv2_lite_attribution(
            vec![11, 304, 608],
            3,
            60_000,
            Some(20_000),
            &[19_000],
        );
    }

    #[test]
    #[should_panic(expected = "generated token count mismatch")]
    fn dsv2_lite_attribution_timings_fail_on_short_generation() {
        let _ =
            timings_from_dsv2_lite_attribution(vec![11, 304], 3, 60_000, Some(20_000), &[19_000]);
    }

    #[test]
    #[should_panic(expected = "zero-duration")]
    fn dsv2_lite_attribution_timings_fail_on_zero_decode_samples() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 60_000, Some(20_000), &[0]);
    }

    #[test]
    #[should_panic(expected = "total generation timing is zero")]
    fn dsv2_lite_attribution_timings_fail_on_zero_total_generation() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 0, Some(20_000), &[19_000]);
    }

    #[test]
    #[should_panic(expected = "TTFT timing is missing or zero")]
    fn dsv2_lite_attribution_timings_fail_on_missing_ttft() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 60_000, None, &[19_000]);
    }

    #[test]
    #[should_panic(expected = "TTFT timing is missing or zero")]
    fn dsv2_lite_attribution_timings_fail_on_zero_ttft() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 60_000, Some(0), &[19_000]);
    }
}
