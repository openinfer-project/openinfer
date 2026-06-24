use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use openinfer::server_engine::ModelType;
use openinfer::vllm_frontend::LoraModule;
use openinfer_core::engine::EpBackend;
#[cfg(feature = "qwen3-4b")]
use openinfer_qwen3_4b::Qwen3LoraOptions;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

#[derive(Parser)]
#[command(name = "openinfer", about = "Qwen3/3.5 GPU inference server")]
#[allow(clippy::struct_excessive_bools)] // independent CLI flags, not a state machine
pub(crate) struct Args {
    /// Model directory containing config, tokenizer, and safetensor shards
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    pub model_path: PathBuf,

    /// Public model ID returned by the OpenAI API (/v1/models, completion `model`).
    /// Defaults to the model path when omitted.
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// Port to listen on
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Enable CUDA Graph capture/replay on decode path (`--cuda-graph=false` to disable)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub cuda_graph: bool,

    /// Enable Qwen3 LoRA serving mode.
    #[arg(long, default_value_t = false)]
    pub enable_lora: bool,

    /// LoRA modules to load at startup. Accepts vLLM-style `name=path`, JSON
    /// object, or JSON list object entries with `name` and `path`.
    #[arg(long = "lora-modules", value_parser = parse_lora_modules_arg)]
    pub lora_modules: Vec<LoraModule>,

    /// Maximum number of resident LoRA adapters in Qwen3 LoRA mode.
    #[cfg(feature = "qwen3-4b")]
    #[arg(long = "max-loras", default_value_t = Qwen3LoraOptions::DEFAULT_MAX_LORAS)]
    pub max_loras: usize,

    /// Maximum supported LoRA rank in Qwen3 LoRA mode.
    #[cfg(feature = "qwen3-4b")]
    #[arg(long = "max-lora-rank", default_value_t = Qwen3LoraOptions::DEFAULT_MAX_LORA_RANK, value_parser = parse_max_lora_rank_arg)]
    pub max_lora_rank: usize,

    /// CUDA device ordinal for single-GPU Qwen3 loads
    #[arg(long, default_value_t = 0)]
    pub device_ordinal: usize,

    /// Tensor-parallel world size for Qwen3
    #[arg(long, default_value_t = 1)]
    pub tp_size: usize,

    /// Data-parallel world size for Kimi-K2
    #[arg(long, default_value_t = 8)]
    pub dp_size: usize,

    /// Expert-parallel backend for Kimi-K2 (TP1/DP8 requires deepep; TP8/DP1 requires nccl)
    #[arg(long, default_value = "deepep")]
    pub ep_backend: CliEpBackend,

    /// Emit synchronized DeepSeek V4 prefill phase timing records.
    #[arg(long, default_value_t = false)]
    pub deepseek_prefill_profile: bool,

    /// Enable pegaflow KV offload (host-tier "L2" cache) on the single-GPU
    /// Qwen3 path. Sealed KV blocks are saved to host pinned memory and
    /// restored into HBM before prefill when a prompt's prefix has fallen out
    /// of the GPU cache.
    #[arg(long, default_value_t = false)]
    pub kv_offload: bool,

    /// Host pinned-memory pool size for the KV offload tier, in GiB. pegaflow
    /// allocates the whole pool up front, so RSS reflects this at startup.
    #[arg(long, default_value_t = 8.0)]
    pub kv_offload_host_gib: f64,

    /// vLLM-style no-prefix-cache. Without --kv-offload it disables prefix
    /// matching outright (every prefill recomputes the full prompt). With
    /// --kv-offload it is the pure-L2 mode: no cross-request HBM reuse, so every
    /// prefix is restored from the host tier — for measuring the L2 TTFT win.
    #[arg(long, default_value_t = false)]
    pub no_prefix_cache: bool,

    /// Enable Qwen3 DFlash speculative decoding with this drafter model path.
    /// Single-GPU greedy only; incompatible with --enable-lora and --kv-offload,
    /// and forces the prefix cache off (it needs clean target hidden states).
    #[arg(long = "dflash-draft-model-path")]
    pub dflash_draft_model_path: Option<PathBuf>,

    /// Cap on total prompt tokens forwarded in one scheduler step (chunked
    /// prefill), for both the Qwen3 and Qwen3.5 schedulers. Prefill activation
    /// scratch scales with the step's prompt tokens, so this bounds peak VRAM
    /// under request bursts; prompts longer than the cap are split across steps
    /// so running decodes keep ticking. On Qwen3, echo requests are never
    /// split. Must be positive.
    #[arg(long, default_value_t = openinfer_qwen3_4b::DEFAULT_MAX_PREFILL_TOKENS)]
    pub max_prefill_tokens: usize,

    /// Fraction of total GPU memory the Qwen3 instance may use. The KV cache is
    /// sized from this budget after startup profiling accounts for weights,
    /// runtime buffers, activation peak, CUDA Graph capture, and margin.
    #[arg(long, default_value_t = openinfer_qwen3_4b::DEFAULT_GPU_MEMORY_UTILIZATION)]
    pub gpu_memory_utilization: f64,

    /// Additional Qwen3 GPU memory to hold back after profile-based KV sizing,
    /// in MiB. Covers allocator fragmentation and small unprofiled drift.
    #[arg(long, default_value_t = (openinfer_qwen3_4b::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES >> 20) as usize)]
    pub kv_cache_memory_margin_mib: usize,
    /// How prefill and decode share the GPU (single-GPU Qwen3 only).
    /// `off` serializes them on one stream (lowest TTFT); `stream` overlaps on
    /// two streams sharing all SMs; `green-ctx` pins each to a disjoint Green
    /// Context SM partition (lower decode ITL p99, higher TTFT).
    #[arg(long, value_enum, default_value_t = CliDecodeOverlap::Off)]
    pub decode_overlap: CliDecodeOverlap,

    /// Percent of SMs pinned to decode in `--decode-overlap green-ctx` (the rest
    /// go to prefill). Ignored in other modes.
    #[arg(long, default_value_t = 20)]
    pub decode_sm_pct: u32,

    /// Enable Qwen3 projection-GEMM and split-KV chunk-count batch-invariant
    /// pinning. Off by default; does not cover path-selection residuals. Qwen3-only.
    #[arg(long, default_value_t = false)]
    pub batch_invariant: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum CliEpBackend {
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

/// CLI selector for prefill/decode overlap. Mapped to
/// [`openinfer_qwen3_4b::DecodeOverlap`] together with `--decode-sm-pct`.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum CliDecodeOverlap {
    /// One stream; prefill and decode serialize.
    Off,
    /// Two CUDA streams sharing all SMs.
    Stream,
    /// Green Context SM partition (SM-pinned streams).
    #[value(name = "green-ctx")]
    GreenCtx,
}

impl CliDecodeOverlap {
    pub(crate) fn resolve(self, decode_sm_pct: u32) -> openinfer_qwen3_4b::DecodeOverlap {
        use openinfer_qwen3_4b::DecodeOverlap;
        match self {
            Self::Off => DecodeOverlap::Off,
            Self::Stream => DecodeOverlap::SharedSm,
            Self::GreenCtx => DecodeOverlap::GreenCtx {
                decode_pct: decode_sm_pct,
            },
        }
    }
}

impl Args {
    pub(crate) fn validate(&self, model_type: ModelType) -> Result<()> {
        if !self.enable_lora && !self.lora_modules.is_empty() {
            bail!("--lora-modules requires --enable-lora");
        }
        #[cfg(feature = "qwen3-4b")]
        let is_qwen3 = matches!(model_type, ModelType::Qwen3);
        #[cfg(not(feature = "qwen3-4b"))]
        let is_qwen3 = false;
        if self.enable_lora && !is_qwen3 {
            bail!("--enable-lora is currently supported only for Qwen3");
        }
        if self.batch_invariant && !is_qwen3 {
            bail!("--batch-invariant is currently supported only for Qwen3");
        }
        if self.batch_invariant && self.enable_lora {
            bail!(
                "--batch-invariant is not yet validated with --enable-lora; enable one at a time"
            );
        }
        if self.batch_invariant && !matches!(self.decode_overlap, CliDecodeOverlap::Off) {
            bail!(
                "--batch-invariant is not compatible with --decode-overlap; Pin falls back to per-token under a stream override"
            );
        }
        Ok(())
    }
}

pub(crate) fn parse_lora_modules_arg(value: &str) -> Result<LoraModule, String> {
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
            let Some(entry) = entries.first() else {
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

#[cfg(feature = "qwen3-4b")]
pub(crate) fn parse_max_lora_rank_arg(value: &str) -> Result<usize, String> {
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

    #[cfg(feature = "qwen3-4b")]
    #[test]
    fn parses_supported_max_lora_rank() {
        assert_eq!(parse_max_lora_rank_arg("16").expect("parse rank"), 16);
        assert_eq!(parse_max_lora_rank_arg("320").expect("parse rank"), 320);
    }

    #[cfg(feature = "qwen3-4b")]
    #[test]
    fn qwen3_lora_default_rank_is_64() {
        assert_eq!(Qwen3LoraOptions::default().max_lora_rank, 64);
    }

    #[cfg(feature = "qwen3-4b")]
    #[test]
    fn rejects_unsupported_max_lora_rank() {
        let error = parse_max_lora_rank_arg("7").expect_err("rank should be unsupported");

        assert!(error.contains("--max-lora-rank must be one of"));
        assert!(error.contains("16"));
    }
}
