use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, bail};
use clap::{Parser, ValueEnum};
use log::info;
use pegainfer::logging;
use pegainfer::server_engine::{ModelType, detect_model_type};
use pegainfer::vllm_frontend::LoraModule;
use pegainfer_core::engine::{EngineLoadOptions, EpBackend};
#[cfg(feature = "kimi-k2")]
use pegainfer_core::parallel::ParallelConfig;
use pegainfer_qwen3_4b::{Qwen3LoraOptions, Qwen3OffloadOptions};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

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

    /// Enable Qwen3 LoRA serving mode.
    #[arg(long, default_value_t = false)]
    enable_lora: bool,

    /// LoRA modules to load at startup. Accepts vLLM-style `name=path`, JSON
    /// object, or JSON list object entries with `name` and `path`.
    #[arg(long = "lora-modules", value_parser = parse_lora_modules_arg)]
    lora_modules: Vec<LoraModule>,

    /// Maximum number of resident LoRA adapters in Qwen3 LoRA mode.
    #[arg(long = "max-loras", default_value_t = Qwen3LoraOptions::DEFAULT_MAX_LORAS)]
    max_loras: usize,

    /// Maximum supported LoRA rank in Qwen3 LoRA mode.
    #[arg(long = "max-lora-rank", default_value_t = Qwen3LoraOptions::DEFAULT_MAX_LORA_RANK, value_parser = parse_max_lora_rank_arg)]
    max_lora_rank: usize,

    /// CUDA device ordinal for single-GPU Qwen3 loads
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,

    /// Tensor-parallel world size for Qwen3
    #[arg(long, default_value_t = 1)]
    tp_size: usize,

    /// Data-parallel world size for Kimi-K2
    #[arg(long, default_value_t = 8)]
    dp_size: usize,

    /// Expert-parallel backend for Kimi-K2 (TP1/DP8 requires deepep; TP8/DP1 requires nccl)
    #[arg(long, default_value = "deepep")]
    ep_backend: CliEpBackend,

    /// Emit synchronized DeepSeek V4 prefill phase timing records.
    #[arg(long, default_value_t = false)]
    deepseek_prefill_profile: bool,

    /// Enable pegaflow KV offload (host-tier "L2" cache) on the single-GPU
    /// Qwen3 path. Sealed KV blocks are saved to host pinned memory and
    /// restored into HBM before prefill when a prompt's prefix has fallen out
    /// of the GPU cache.
    #[arg(long, default_value_t = false)]
    kv_offload: bool,

    /// Host pinned-memory pool size for the KV offload tier, in GiB. pegaflow
    /// allocates the whole pool up front, so RSS reflects this at startup.
    #[arg(long, default_value_t = 8.0)]
    kv_offload_host_gib: f64,

    /// vLLM-style no-prefix-cache. Without --kv-offload it disables prefix
    /// matching outright (every prefill recomputes the full prompt). With
    /// --kv-offload it is the pure-L2 mode: no cross-request HBM reuse, so every
    /// prefix is restored from the host tier — for measuring the L2 TTFT win.
    #[arg(long, default_value_t = false)]
    no_prefix_cache: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliEpBackend {
    Nccl,
    #[value(name = "deepep")]
    DeepEp,
}

impl From<CliEpBackend> for EpBackend {
    fn from(value: CliEpBackend) -> Self {
        match value {
            CliEpBackend::Nccl => Self::Nccl,
            CliEpBackend::DeepEp => Self::DeepEp,
        }
    }
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
    if !args.enable_lora && !args.lora_modules.is_empty() {
        bail!("--lora-modules requires --enable-lora");
    }
    if args.enable_lora && !matches!(model_type, ModelType::Qwen3) {
        bail!("--enable-lora is currently supported only for Qwen3");
    }
    let effective_cuda_graph = match model_type {
        #[cfg(feature = "deepseek-v2-lite")]
        ModelType::DeepSeekV2Lite => false,
        #[cfg(feature = "deepseek-v4")]
        ModelType::DeepSeekV4 => false,
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => args.cuda_graph,
        ModelType::Qwen3 if args.enable_lora => false,
        ModelType::Qwen3 | ModelType::Qwen35 => args.cuda_graph,
    };

    info!("=== Rust LLM Server - {} (GPU) ===", model_type);
    info!("Loading engine...");
    let start = Instant::now();
    info!(
        "Runtime options: model_path={}, requested_cuda_graph={}, effective_cuda_graph={}, enable_lora={}, max_loras={}, max_lora_rank={}, device_ordinal={}, tp_size={}, dp_size={}, ep_backend={:?}",
        args.model_path.display(),
        args.cuda_graph,
        effective_cuda_graph,
        args.enable_lora,
        args.max_loras,
        args.max_lora_rank,
        args.device_ordinal,
        args.tp_size,
        args.dp_size,
        args.ep_backend
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
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
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
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: 42,
                },
            )?;

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());

            handle
        }
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => {
            let parallel = kimi_parallel_config(args.tp_size, args.dp_size)?;
            let handle = pegainfer_kimi_k2::start_engine(
                &args.model_path,
                EngineLoadOptions {
                    enable_cuda_graph: args.cuda_graph,
                    enable_prefill_profile: false,
                    device_ordinals: (0..parallel.ep_world()).collect(),
                    parallel_config: Some(parallel),
                    ep_backend: args.ep_backend.into(),
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
            let options = EngineLoadOptions {
                enable_cuda_graph: effective_cuda_graph,
                enable_prefill_profile: false,
                device_ordinals,
                parallel_config: None,
                ep_backend: EpBackend::Nccl,
                seed: 42,
            };
            let offload = if args.kv_offload {
                let bytes = (args.kv_offload_host_gib * f64::from(1u32 << 30)) as usize;
                info!(
                    "Qwen3 KV offload enabled: host tier {:.1} GiB, no_prefix_cache={}",
                    args.kv_offload_host_gib, args.no_prefix_cache
                );
                Qwen3OffloadOptions::enabled(bytes)
            } else {
                Qwen3OffloadOptions::disabled()
            };
            let handle = if args.enable_lora {
                let lora_options = Qwen3LoraOptions {
                    max_loras: args.max_loras,
                    max_lora_rank: args.max_lora_rank,
                };
                info!(
                    "Starting Qwen3 engine with LoRA control; CUDA Graph is disabled; max_loras={}, max_lora_rank={}",
                    lora_options.max_loras, lora_options.max_lora_rank
                );
                pegainfer_qwen3_4b::start_engine_with_lora_control(
                    &args.model_path,
                    options,
                    lora_options,
                    offload,
                    args.no_prefix_cache,
                )
            } else {
                pegainfer_qwen3_4b::start_engine_with_offload(
                    &args.model_path,
                    options,
                    offload,
                    args.no_prefix_cache,
                )
            }
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
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: 42,
                },
            )
            .context("failed to start Qwen3.5 engine")?;

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());

            handle
        }
    };

    if args.enable_lora {
        let max_model_len =
            pegainfer::vllm_frontend::load_max_model_len(&args.model_path).unwrap_or(4096);
        pegainfer::vllm_frontend::serve_model_with_lora_routes(
            handle,
            args.model_path.to_string_lossy().into_owned(),
            args.served_model_name.into_iter().collect(),
            args.lora_modules,
            args.port,
            max_model_len,
            pegainfer::vllm_frontend::shutdown_token_from_ctrl_c(),
        )
        .await
    } else {
        pegainfer::vllm_frontend::serve(
            handle,
            &args.model_path,
            args.served_model_name.as_deref(),
            args.port,
            pegainfer::vllm_frontend::shutdown_token_from_ctrl_c(),
        )
        .await
    }
    .context("vLLM frontend server failed")?;

    Ok(())
}

#[cfg(feature = "kimi-k2")]
fn kimi_parallel_config(tp_size: usize, dp_size: usize) -> anyhow::Result<ParallelConfig> {
    if tp_size == 0 || dp_size == 0 {
        bail!("--tp-size and --dp-size must be positive");
    }
    Ok(ParallelConfig::new(tp_size, dp_size))
}

fn parse_lora_modules_arg(value: &str) -> Result<LoraModule, String> {
    if let Some((name, path)) = value.split_once('=') {
        return parse_lora_module_fields(name, path);
    }
    let json: serde_json::Value =
        serde_json::from_str(value).map_err(|error| format!("invalid --lora-modules: {error}"))?;
    match json {
        serde_json::Value::Object(map) => {
            let name = map
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "--lora-modules JSON object requires string field `name`".to_string()
                })?;
            let path = map
                .get("path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "--lora-modules JSON object requires string field `path`".to_string()
                })?;
            parse_lora_module_fields(name, path)
        }
        serde_json::Value::Array(entries) if entries.len() == 1 => {
            let Some(entry) = entries.into_iter().next() else {
                unreachable!("array length checked")
            };
            let serde_json::Value::Object(map) = entry else {
                return Err("--lora-modules JSON list entries must be objects".to_string());
            };
            let name = map
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "--lora-modules JSON object requires string field `name`".to_string()
                })?;
            let path = map
                .get("path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "--lora-modules JSON object requires string field `path`".to_string()
                })?;
            parse_lora_module_fields(name, path)
        }
        serde_json::Value::Array(_) => Err(
            "pass multiple --lora-modules values instead of one JSON list with multiple entries"
                .to_string(),
        ),
        _ => Err(
            "--lora-modules must be `name=path`, a JSON object, or a single-entry JSON list"
                .to_string(),
        ),
    }
}

fn parse_max_lora_rank_arg(value: &str) -> Result<usize, String> {
    let rank = value
        .parse::<usize>()
        .map_err(|error| format!("invalid --max-lora-rank: {error}"))?;
    if Qwen3LoraOptions::is_supported_max_lora_rank(rank) {
        Ok(rank)
    } else {
        Err(format!(
            "--max-lora-rank must be one of: {}",
            Qwen3LoraOptions::supported_max_lora_ranks_display()
        ))
    }
}

fn parse_lora_module_fields(name: &str, path: &str) -> Result<LoraModule, String> {
    if name.is_empty() {
        return Err("--lora-modules name must not be empty".to_string());
    }
    if path.is_empty() {
        return Err("--lora-modules path must not be empty".to_string());
    }
    Ok(LoraModule {
        name: name.to_string(),
        path: PathBuf::from(path),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lora_modules_name_equals_path() {
        assert_eq!(
            parse_lora_modules_arg("adapter-a=/tmp/adapter-a").expect("parse module"),
            LoraModule {
                name: "adapter-a".to_string(),
                path: PathBuf::from("/tmp/adapter-a"),
            }
        );
    }

    #[test]
    fn parses_lora_modules_json_object() {
        assert_eq!(
            parse_lora_modules_arg(r#"{"name":"adapter-a","path":"/tmp/adapter-a"}"#)
                .expect("parse module"),
            LoraModule {
                name: "adapter-a".to_string(),
                path: PathBuf::from("/tmp/adapter-a"),
            }
        );
    }

    #[test]
    fn parses_lora_modules_single_entry_json_list() {
        assert_eq!(
            parse_lora_modules_arg(r#"[{"name":"adapter-a","path":"/tmp/adapter-a"}]"#)
                .expect("parse module"),
            LoraModule {
                name: "adapter-a".to_string(),
                path: PathBuf::from("/tmp/adapter-a"),
            }
        );
    }

    #[test]
    fn parses_supported_max_lora_rank() {
        assert_eq!(parse_max_lora_rank_arg("16").expect("parse rank"), 16);
        assert_eq!(parse_max_lora_rank_arg("320").expect("parse rank"), 320);
    }

    #[test]
    fn qwen3_lora_default_rank_is_64() {
        assert_eq!(Qwen3LoraOptions::default().max_lora_rank, 64);
    }

    #[test]
    fn rejects_unsupported_max_lora_rank() {
        let error = parse_max_lora_rank_arg("7").expect_err("rank should be unsupported");

        assert!(error.contains("--max-lora-rank must be one of"));
        assert!(error.contains("16"));
    }
}
