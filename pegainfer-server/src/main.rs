use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use log::info;
use pegainfer::logging;
use pegainfer::server_engine::{ModelType, detect_model_type};
use pegainfer_core::engine::EngineLoadOptions;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

#[derive(Parser)]
#[command(name = "pegainfer", about = "Qwen3/3.5 GPU inference server")]
struct Args {
    /// Model directory containing config, tokenizer, and safetensor shards
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,

    /// Public model ID returned by the OpenAI API (/v1/models, completion `model`).
    /// Defaults to the model path when omitted.
    #[arg(long)]
    served_model_name: Option<String>,

    /// Port to listen on
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Enable CUDA Graph capture/replay on decode path (`--cuda-graph=false` to disable)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graph: bool,

    /// CUDA device ordinal for single-GPU Qwen3 loads
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,

    /// Tensor-parallel world size for Qwen3
    #[arg(long, default_value_t = 1)]
    tp_size: usize,

    /// Emit synchronized DeepSeek V4 prefill phase timing records.
    #[arg(long, default_value_t = false)]
    deepseek_prefill_profile: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    logging::init_default();

    let args = Args::parse();

    let model_type = detect_model_type(&args.model_path).with_context(|| {
        format!(
            "failed to detect model type from {}",
            args.model_path.display()
        )
    })?;
    let effective_cuda_graph = match model_type {
        #[cfg(feature = "deepseek-v2-lite")]
        ModelType::DeepSeekV2Lite => false,
        #[cfg(feature = "deepseek-v4")]
        ModelType::DeepSeekV4 => false,
        ModelType::KimiK2 => args.cuda_graph,
        ModelType::Qwen3 | ModelType::Qwen35 => args.cuda_graph,
    };

    info!("=== Rust LLM Server - {} (GPU) ===", model_type);
    info!("Loading engine...");
    let start = Instant::now();
    info!(
        "Runtime options: model_path={}, requested_cuda_graph={}, effective_cuda_graph={}, device_ordinal={}, tp_size={}",
        args.model_path.display(),
        args.cuda_graph,
        effective_cuda_graph,
        args.device_ordinal,
        args.tp_size
    );

    let handle = match model_type {
        #[cfg(feature = "deepseek-v4")]
        ModelType::DeepSeekV4 => {
            let handle = pegainfer_deepseek_v4::start_engine(
                &args.model_path,
                EngineLoadOptions {
                    enable_cuda_graph: false,
                    enable_prefill_profile: args.deepseek_prefill_profile,
                    device_ordinals: (0..8).collect(),
                    seed: 42,
                },
            )
            .context("failed to start DeepSeek V4 engine")?;

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());

            handle
        }
        #[cfg(feature = "deepseek-v2-lite")]
        ModelType::DeepSeekV2Lite => {
            let handle = pegainfer_deepseek_v2_lite::start_engine(
                &args.model_path,
                EngineLoadOptions {
                    enable_cuda_graph: false,
                    enable_prefill_profile: false,
                    device_ordinals: vec![0, 1],
                    seed: 42,
                },
            )?;

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());

            handle
        }
        ModelType::KimiK2 => {
            let handle = pegainfer_kimi_k2::start_engine(
                &args.model_path,
                EngineLoadOptions {
                    enable_cuda_graph: args.cuda_graph,
                    enable_prefill_profile: false,
                    device_ordinals: (0..8).collect(),
                    seed: 42,
                },
            )
            .context("failed to start Kimi-K2.6 text engine")?;

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());

            handle
        }
        ModelType::Qwen3 => {
            let device_ordinals: Vec<usize> = if args.tp_size == 1 {
                vec![args.device_ordinal]
            } else {
                (0..args.tp_size).collect()
            };
            let handle = pegainfer_qwen3_4b::start_engine(
                &args.model_path,
                EngineLoadOptions {
                    enable_cuda_graph: args.cuda_graph,
                    enable_prefill_profile: false,
                    device_ordinals,
                    seed: 42,
                },
            )
            .context("failed to start Qwen3 engine")?;

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());

            handle
        }
        ModelType::Qwen35 => {
            let handle = pegainfer_qwen35_4b::start_engine(
                &args.model_path,
                EngineLoadOptions {
                    enable_cuda_graph: args.cuda_graph,
                    enable_prefill_profile: false,
                    device_ordinals: vec![args.device_ordinal],
                    seed: 42,
                },
            )
            .context("failed to start Qwen3.5 engine")?;

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());

            handle
        }
    };

    pegainfer::vllm_frontend::serve(
        handle,
        &args.model_path,
        args.served_model_name.as_deref(),
        args.port,
        pegainfer::vllm_frontend::shutdown_token_from_ctrl_c(),
    )
    .await
    .context("vLLM frontend server failed")?;

    Ok(())
}
