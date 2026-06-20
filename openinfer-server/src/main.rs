mod config;

use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use log::info;
use openinfer::logging;
use openinfer::server_engine::{ModelType, detect_model_type};
use openinfer_core::engine::EngineHandle;
#[cfg(feature = "qwen3-4b")]
use openinfer_qwen3_4b::{Qwen3LaunchOptions, Qwen3LoraOptions, Qwen3OffloadOptions};

use config::Args;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

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
    args.validate(model_type)?;

    info!("=== openinfer - {} (GPU) ===", model_type);
    info!("Loading engine...");
    let start = Instant::now();
    info!(
        "Runtime options: model_path={}, cuda_graph={}, enable_lora={}, device_ordinal={}, tp_size={}",
        args.model_path.display(),
        args.cuda_graph,
        args.enable_lora,
        args.device_ordinal,
        args.tp_size,
    );

    // Engine load (weights → GPU) runs on a blocking thread so the HTTP
    // frontend (tokenizer, chat templates) loads concurrently. The frontend
    // binds only after the engine registers, so readiness is unchanged.
    let model_path = args.model_path.clone();
    let served_model_name = args.served_model_name.clone();
    let lora_modules = args.lora_modules.clone();
    let enable_lora = args.enable_lora;
    let port = args.port;
    let engine_load = tokio::task::spawn_blocking(move || -> anyhow::Result<EngineHandle> {
        load_engine(&args, model_type)
    });

    if enable_lora {
        // LoRA routes need the engine handle when the router is built, so this
        // path stays sequential.
        let handle = engine_load
            .await
            .context("engine loader thread panicked")??;
        info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());
        let max_model_len =
            openinfer::vllm_frontend::load_max_model_len(&model_path).unwrap_or(4096);
        openinfer::vllm_frontend::serve_model_with_lora_routes(
            handle,
            model_path.to_string_lossy().into_owned(),
            served_model_name.into_iter().collect(),
            lora_modules,
            port,
            max_model_len,
            openinfer::vllm_frontend::shutdown_token_from_ctrl_c(),
        )
        .await
    } else {
        let shutdown = tokio_util::sync::CancellationToken::new();
        let engine = {
            let shutdown = shutdown.clone();
            async move {
                let handle = engine_load
                    .await
                    .context("engine loader thread panicked")??;
                info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());
                // The blocking load can't be cancelled, so SIGINT keeps its
                // default kill behavior until the engine is up; only then
                // switch to graceful shutdown.
                openinfer::vllm_frontend::cancel_token_on_ctrl_c(&shutdown);
                anyhow::Ok(handle)
            }
        };
        openinfer::vllm_frontend::serve(
            engine,
            &model_path,
            served_model_name.into_iter().collect(),
            port,
            None,
            shutdown,
        )
        .await
    }
    .context("vLLM frontend server failed")?;

    Ok(())
}

// Pure dispatch: each model crate owns its own launch policy (topology
// defaults, capability constraints, cross-arg validation). The server only
// picks the crate by detected model type and forwards the relevant CLI knobs.
fn load_engine(args: &Args, model_type: ModelType) -> anyhow::Result<EngineHandle> {
    let handle = match model_type {
        #[cfg(feature = "deepseek-v4")]
        ModelType::DeepSeekV4 => openinfer_deepseek_v4::launch(
            &args.model_path,
            args.cuda_graph,
            args.deepseek_prefill_profile,
        )
        .context("failed to start DeepSeek V4 engine")?,
        #[cfg(feature = "deepseek-v2-lite")]
        ModelType::DeepSeekV2Lite => {
            openinfer_deepseek_v2_lite::launch(&args.model_path, args.cuda_graph)
                .context("failed to start DeepSeek V2 Lite engine")?
        }
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => openinfer_kimi_k2::launch(
            &args.model_path,
            openinfer_kimi_k2::KimiLaunchOptions {
                tp_size: args.tp_size,
                dp_size: args.dp_size,
                ep_backend: args.ep_backend.into(),
                cuda_graph: args.cuda_graph,
            },
        )
        .context("failed to start Kimi-K2.6 text engine")?,
        #[cfg(feature = "qwen3-4b")]
        ModelType::Qwen3 => {
            let offload = if args.kv_offload {
                let bytes = (args.kv_offload_host_gib * f64::from(1u32 << 30)) as usize;
                Qwen3OffloadOptions::enabled(bytes)
            } else {
                Qwen3OffloadOptions::disabled()
            };
            let lora = args.enable_lora.then_some(Qwen3LoraOptions {
                max_loras: args.max_loras,
                max_lora_rank: args.max_lora_rank,
            });
            openinfer_qwen3_4b::launch(
                &args.model_path,
                Qwen3LaunchOptions {
                    device_ordinal: args.device_ordinal,
                    tp_size: args.tp_size,
                    cuda_graph: args.cuda_graph,
                    offload,
                    no_prefix_cache: args.no_prefix_cache,
                    max_prefill_tokens: args.max_prefill_tokens,
                    lora,
                    decode_overlap: args.decode_overlap.resolve(args.decode_sm_pct),
                },
            )
            .context("failed to start Qwen3 engine")?
        }
        #[cfg(feature = "qwen35-4b")]
        ModelType::Qwen35 => {
            openinfer_qwen35_4b::launch(&args.model_path, args.device_ordinal, args.cuda_graph)
                .context("failed to start Qwen3.5 engine")?
        }
    };

    Ok(handle)
}
