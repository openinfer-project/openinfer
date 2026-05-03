use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use log::info;
use pegainfer::logging;
use pegainfer::model::Qwen35Model;
use pegainfer::server_engine::{ModelType, detect_model_type};
use pegainfer_core::engine::EngineLoadOptions;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

#[derive(Parser)]
#[command(name = "pegainfer", about = "Qwen3/3.5 GPU inference server")]
struct Args {
    /// Model directory containing config, tokenizer, and safetensor shards
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,

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
}

#[tokio::main]
async fn main() {
    logging::init_default();

    let args = Args::parse();

    let model_path = args
        .model_path
        .to_str()
        .expect("Model path must be valid UTF-8");
    let model_type = detect_model_type(model_path).expect("Failed to detect model type");

    info!("=== Rust LLM Server - {} (GPU) ===", model_type);
    info!("Loading engine...");
    let start = Instant::now();
    info!(
        "Runtime options: model_path={}, cuda_graph={}, device_ordinal={}, tp_size={}",
        args.model_path.display(),
        args.cuda_graph,
        args.device_ordinal,
        args.tp_size
    );

    let handle = match model_type {
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
                    device_ordinals,
                    seed: 42,
                },
            )
            .expect("Failed to start Qwen3 engine");

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());

            handle
        }
        ModelType::Qwen35 => {
            let model = Qwen35Model::from_safetensors_with_options(model_path, args.cuda_graph)
                .expect("Failed to load Qwen3.5 model");

            let handle = pegainfer::scheduler_qwen35::start(model, 42)
                .expect("Failed to start Qwen3.5 scheduler");

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());

            handle
        }
    };

    pegainfer::vllm_frontend::serve(
        handle,
        &args.model_path,
        args.port,
        pegainfer::vllm_frontend::shutdown_token_from_ctrl_c(),
    )
    .await
    .expect("vLLM frontend server failed");
}
