use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{CommandFactory, Parser, ValueEnum};
use openinfer::server_engine::ModelType;
use openinfer::vllm_frontend::LoraModule;
use openinfer_core::engine::EpBackend;
#[cfg(feature = "qwen3")]
use openinfer_qwen3::Qwen3LoraOptions;

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

    /// Enable CUDA Graph capture/replay on decode path (`--cuda-graph=false` to
    /// disable). Rejected for GLM5.2; forced off in Qwen3 LoRA mode.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub cuda_graph: bool,

    /// Dump the live Qwen3 rank-0, batch-1 SplitKv decode CUDA Graph during
    /// startup. Writes a detailed sibling `.dot` for LLM inspection and a
    /// compact Graphviz-rendered PNG at this path. Requires CUDA driver API
    /// 12.3 or newer for kernel-name inspection.
    #[arg(long)]
    pub dump_graph_png: Option<PathBuf>,

    /// Enable Qwen3 LoRA serving mode.
    #[arg(long, default_value_t = false)]
    pub enable_lora: bool,

    /// LoRA modules to load at startup. Accepts vLLM-style `name=path`, JSON
    /// object, or JSON list object entries with `name` and `path`.
    #[arg(long = "lora-modules", value_parser = parse_lora_modules_arg)]
    pub lora_modules: Vec<LoraModule>,

    /// Maximum number of resident LoRA adapters in Qwen3 LoRA mode.
    #[cfg(feature = "qwen3")]
    #[arg(long = "max-loras", default_value_t = Qwen3LoraOptions::DEFAULT_MAX_LORAS)]
    pub max_loras: usize,

    /// Maximum supported LoRA rank in Qwen3 LoRA mode.
    #[cfg(feature = "qwen3")]
    #[arg(long = "max-lora-rank", default_value_t = Qwen3LoraOptions::DEFAULT_MAX_LORA_RANK, value_parser = parse_max_lora_rank_arg)]
    pub max_lora_rank: usize,

    /// CUDA device ordinal for single-GPU Qwen3/Qwen3.5 loads
    #[arg(long, default_value_t = 0)]
    pub device_ordinal: usize,

    /// Tensor-parallel world size (Qwen3 and Kimi-K2; GLM5.2 requires 1)
    #[arg(long, default_value_t = 1)]
    pub tp_size: usize,

    /// Data-parallel world size. Kimi-K2 and GLM5.2 both default to 8; GLM5.2
    /// requires exactly 8.
    #[arg(long)]
    pub dp_size: Option<usize>,

    /// Expert-parallel backend for Kimi-K2 (TP1/DP8 requires deepep; TP8/DP1 requires nccl)
    #[arg(long, default_value = "deepep")]
    pub ep_backend: CliEpBackend,

    /// Emit synchronized DeepSeek V4 prefill phase timing records.
    #[arg(long, default_value_t = false)]
    pub deepseek_prefill_profile: bool,

    /// Enable pegaflow KV offload (host-tier "L2" cache): single-GPU Qwen3,
    /// or GLM5.2 DP8 (one pool shared by all 8 ranks under one namespace).
    /// Sealed KV blocks are saved to host pinned memory and restored into
    /// HBM before prefill when a prompt's prefix has fallen out of the GPU
    /// cache. GLM5.2 requires the prefix cache: incompatible with
    /// --no-prefix-cache and the DSpark drafter.
    #[arg(long, default_value_t = false)]
    pub kv_offload: bool,

    /// Host pinned-memory pool size for the KV offload tier, in GiB. pegaflow
    /// allocates the whole pool up front, so RSS reflects this at startup.
    #[arg(long, default_value_t = 8.0, value_parser = parse_offload_gib, requires = "kv_offload")]
    pub kv_offload_host_gib: f64,

    /// Back the KV offload pool with 2 MiB hugepages. The box must hold a
    /// reservation covering the pool (`HugePages_Total` in /proc/meminfo;
    /// `echo N > /proc/sys/vm/nr_hugepages` as root) — allocation fails at
    /// startup otherwise.
    #[arg(long, default_value_t = false, requires = "kv_offload")]
    pub kv_offload_hugepages: bool,

    /// Join the cross-instance KV P2P mesh: pegaflow MetaServer gRPC address
    /// (e.g. `http://127.0.0.1:50056`). Saved block hashes register there and
    /// missing prefixes are pulled from peer instances over RDMA — the P/D
    /// disaggregation data plane. Requires --kv-offload, --kv-p2p-advertise-addr
    /// and --kv-p2p-nics.
    #[arg(long, requires_all = ["kv_offload", "kv_p2p_advertise_addr", "kv_p2p_nics"])]
    pub kv_p2p_metaserver_addr: Option<String>,

    /// This instance's routable IP:port for KV P2P — a literal socket address
    /// (it is also the embedded transfer-service bind address, so hostnames
    /// are rejected at startup). Peers dial it for RDMA handshakes and block
    /// queries. Must be reachable by every peer; not 0.0.0.0.
    #[arg(long, requires = "kv_p2p_metaserver_addr")]
    pub kv_p2p_advertise_addr: Option<String>,

    /// RDMA NIC device names for KV P2P (e.g. `mlx5_0`), comma-separated.
    #[arg(long, value_delimiter = ',', requires = "kv_p2p_metaserver_addr")]
    pub kv_p2p_nics: Vec<String>,

    /// P/D prefill role: barrier each request's KV saves (host tier +
    /// MetaServer registration) before its final token event, so this
    /// instance's HTTP response doubles as the KV-ready signal a router can
    /// act on. Leave off on decode instances.
    #[arg(long, default_value_t = false, requires = "kv_p2p_metaserver_addr")]
    pub kv_p2p_flush_on_finish: bool,

    /// vLLM-style no-prefix-cache. Without --kv-offload it disables prefix
    /// matching outright (every prefill recomputes the full prompt). With
    /// --kv-offload it is the pure-L2 mode: no cross-request HBM reuse, so every
    /// prefix is restored from the host tier — for measuring the L2 TTFT win.
    #[arg(long, default_value_t = false)]
    pub no_prefix_cache: bool,

    /// Speculative drafter model path: Qwen3 DFlash/DSpark decoding, or the
    /// GLM5.2 DSpark drafter (greedy AND sampled requests speculate;
    /// per-request accept stats logged). For Qwen3: single-GPU greedy only;
    /// incompatible with --enable-lora and --kv-offload, and forces the
    /// prefix cache off (it needs clean target hidden states).
    #[arg(long = "dflash-draft-model-path")]
    pub dflash_draft_model_path: Option<PathBuf>,

    /// Cap on total prompt tokens forwarded in one scheduler step. Qwen3 and
    /// Qwen3.5 only (rejected for other model lines); when omitted, they use
    /// their own crate defaults.
    #[arg(long)]
    pub max_prefill_tokens: Option<usize>,

    /// Per-request context cap: prompt + max_tokens - 1 must fit. GLM5.2 only;
    /// when omitted, GLM5.2 sizes it from post-weight-load free VRAM.
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// GLM5.2 launch-time MoE sharding topology: `ep8` (default) is the
    /// high-throughput configuration (32 whole experts per rank, DeepEP
    /// dispatch/combine, buckets 1-8); `tp8` is the low-latency
    /// configuration (1/8-intermediate slice of ALL experts per rank on
    /// every MoE layer, bucket-1 only — at most one request per rank).
    #[arg(long, default_value = "ep8")]
    pub moe_topo: String,

    /// Fraction of total GPU memory the Qwen3 instance may use. The KV cache is
    /// sized from this budget after startup profiling accounts for weights,
    /// runtime buffers, activation peak, margin, and (single-GPU only) CUDA-graph
    /// capture; tensor parallelism runs decode eagerly (no graph).
    #[cfg(feature = "qwen3")]
    #[arg(long, default_value_t = openinfer_qwen3::DEFAULT_GPU_MEMORY_UTILIZATION)]
    pub gpu_memory_utilization: f64,

    /// Additional Qwen3 GPU memory to hold back after profile-based KV sizing,
    /// in MiB. Covers allocator fragmentation and small unprofiled drift.
    #[cfg(feature = "qwen3")]
    #[arg(long, default_value_t = (openinfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES >> 20) as usize)]
    pub kv_cache_memory_margin_mib: usize,
    /// KV cache page (block) size in tokens. FlashInfer's paged attention only
    /// accepts a restricted set; 16 (default) or 64. Larger pages cut block
    /// bookkeeping overhead at the cost of coarser-grained allocation.
    #[cfg(feature = "qwen3")]
    #[arg(long, default_value_t = openinfer_qwen3::DEFAULT_KV_PAGE_SIZE)]
    pub kv_page_size: usize,
    /// How prefill and decode share the GPU (single-GPU Qwen3 only).
    /// `off` serializes them on one stream (lowest TTFT); `stream` overlaps on
    /// two streams sharing all SMs; `green-ctx` pins each to a disjoint Green
    /// Context SM partition (lower decode ITL p99, higher TTFT).
    #[arg(long, value_enum, default_value_t = CliDecodeOverlap::Off)]
    pub decode_overlap: CliDecodeOverlap,

    /// Percent of SMs pinned to decode in `--decode-overlap green-ctx` (the rest
    /// go to prefill); rejected if set in any other mode.
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=99))]
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
/// [`openinfer_qwen3::DecodeOverlap`] together with `--decode-sm-pct`.
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
    #[cfg(feature = "qwen3")]
    pub(crate) fn resolve(self, decode_sm_pct: u32) -> openinfer_qwen3::DecodeOverlap {
        use openinfer_qwen3::DecodeOverlap;
        match self {
            Self::Off => DecodeOverlap::Off,
            Self::Stream => DecodeOverlap::SharedSm,
            Self::GreenCtx => DecodeOverlap::GreenCtx {
                decode_pct: decode_sm_pct,
            },
        }
    }
}

/// Flags accepted for every model line regardless of detected type.
const CORE_ARGS: &[&str] = &["model_path", "served_model_name", "port"];

/// The CLI arg ids each model line uses — the applicability source of truth for
/// `validate()`.
fn consumed_args(model_type: ModelType) -> &'static [&'static str] {
    match model_type {
        #[cfg(feature = "deepseek-v4")]
        ModelType::DeepSeekV4 => &["cuda_graph", "deepseek_prefill_profile"],
        #[cfg(feature = "deepseek-v2-lite")]
        ModelType::DeepSeekV2Lite => &["cuda_graph"],
        #[cfg(feature = "glm52")]
        ModelType::Glm52 => &[
            "tp_size",
            "dp_size",
            "dflash_draft_model_path",
            "max_model_len",
            "no_prefix_cache",
            "kv_offload",
            "kv_offload_host_gib",
            "kv_offload_hugepages",
            "moe_topo",
        ],
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => &["tp_size", "dp_size", "ep_backend", "cuda_graph"],
        #[cfg(feature = "qwen3")]
        ModelType::Qwen3 => &[
            "cuda_graph",
            "dump_graph_png",
            "enable_lora",
            "lora_modules",
            "max_loras",
            "max_lora_rank",
            "device_ordinal",
            "tp_size",
            "kv_offload",
            "kv_offload_host_gib",
            "kv_offload_hugepages",
            "kv_p2p_metaserver_addr",
            "kv_p2p_advertise_addr",
            "kv_p2p_nics",
            "kv_p2p_flush_on_finish",
            "no_prefix_cache",
            "max_prefill_tokens",
            "gpu_memory_utilization",
            "kv_cache_memory_margin_mib",
            "kv_page_size",
            "decode_overlap",
            "decode_sm_pct",
            "batch_invariant",
            "dflash_draft_model_path",
        ],
        #[cfg(feature = "qwen35-4b")]
        ModelType::Qwen35 => &["device_ordinal", "cuda_graph", "max_prefill_tokens"],
    }
}

fn long_flag(cmd: &clap::Command, id: &str) -> String {
    cmd.get_arguments()
        .find(|arg| arg.get_id() == id)
        .and_then(clap::Arg::get_long)
        .map_or_else(|| id.to_owned(), str::to_owned)
}

/// Arg ids the user set explicitly (command line or env), for consume-or-reject.
pub(crate) fn provided_args(matches: &clap::ArgMatches) -> BTreeSet<String> {
    // matches.ids() also yields clap's synthetic struct-name group id; keep only real args.
    let cmd = Args::command();
    let real: BTreeSet<&str> = cmd
        .get_arguments()
        .map(|arg| arg.get_id().as_str())
        .collect();
    matches
        .ids()
        .map(clap::Id::as_str)
        .filter(|id| real.contains(id))
        .filter(|id| {
            matches!(
                matches.value_source(id),
                Some(
                    clap::parser::ValueSource::CommandLine | clap::parser::ValueSource::EnvVariable
                )
            )
        })
        .map(str::to_owned)
        .collect()
}

impl Args {
    pub(crate) fn validate(
        &self,
        model_type: ModelType,
        provided: &BTreeSet<String>,
    ) -> Result<()> {
        let cmd = Self::command();
        for id in provided {
            let id = id.as_str();
            if CORE_ARGS.contains(&id) || consumed_args(model_type).contains(&id) {
                continue;
            }
            bail!("--{} is not used by {model_type:?}", long_flag(&cmd, id));
        }
        if !self.enable_lora && !self.lora_modules.is_empty() {
            bail!("--lora-modules requires --enable-lora");
        }
        if !self.enable_lora
            && (provided.contains("max_loras") || provided.contains("max_lora_rank"))
        {
            bail!("--max-loras and --max-lora-rank require --enable-lora");
        }
        if self.batch_invariant && self.enable_lora {
            bail!("--batch-invariant is not supported with --enable-lora; enable one at a time");
        }
        if self.dump_graph_png.is_some() && !self.cuda_graph {
            bail!("--dump-graph-png requires --cuda-graph=true");
        }
        if self.dump_graph_png.is_some() && self.enable_lora {
            bail!(
                "--dump-graph-png is not supported with --enable-lora (LoRA disables CUDA Graph)"
            );
        }
        if self.batch_invariant && !matches!(self.decode_overlap, CliDecodeOverlap::Off) {
            bail!(
                "--batch-invariant is not compatible with --decode-overlap; the stream override would force the pinned GEMM to bail at runtime"
            );
        }
        if self.batch_invariant && self.dflash_draft_model_path.is_some() {
            bail!(
                "--batch-invariant is not supported with DFlash speculative decoding; enable one at a time"
            );
        }
        if provided.contains("decode_sm_pct")
            && !matches!(self.decode_overlap, CliDecodeOverlap::GreenCtx)
        {
            bail!("--decode-sm-pct only applies with --decode-overlap=green-ctx");
        }
        if provided.contains("device_ordinal") && self.tp_size > 1 {
            bail!(
                "--device-ordinal is ignored under tensor parallelism; tp_size>1 uses devices 0..tp_size"
            );
        }
        if !matches!(self.decode_overlap, CliDecodeOverlap::Off) && self.tp_size > 1 {
            bail!("--decode-overlap is single-GPU only; tp_size>1 has no prefill/decode overlap");
        }
        #[cfg(feature = "glm52")]
        if matches!(model_type, ModelType::Glm52) {
            if let Some(dp_size) = self.dp_size {
                if dp_size != 8 {
                    bail!(
                        "GLM5.2 requires --dp-size=8 when provided; omit --dp-size to use DP8/EP8"
                    );
                }
            }
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

fn parse_offload_gib(value: &str) -> Result<f64, String> {
    let gib = value
        .parse::<f64>()
        .map_err(|error| format!("invalid --kv-offload-host-gib: {error}"))?;
    if gib.is_finite() && gib > 0.0 {
        Ok(gib)
    } else {
        Err("--kv-offload-host-gib must be a positive, finite number of GiB".to_owned())
    }
}

#[cfg(feature = "qwen3")]
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

    #[cfg(any(feature = "glm52", feature = "qwen3"))]
    fn parse_with_provided(argv: &[&str]) -> (Args, BTreeSet<String>) {
        use clap::FromArgMatches;
        let matches = Args::command()
            .try_get_matches_from(argv)
            .expect("args parse");
        let args = Args::from_arg_matches(&matches).expect("args from matches");
        (args, provided_args(&matches))
    }

    fn all_model_types() -> Vec<ModelType> {
        [
            #[cfg(feature = "deepseek-v4")]
            ModelType::DeepSeekV4,
            #[cfg(feature = "deepseek-v2-lite")]
            ModelType::DeepSeekV2Lite,
            #[cfg(feature = "glm52")]
            ModelType::Glm52,
            #[cfg(feature = "kimi-k2")]
            ModelType::KimiK2,
            #[cfg(feature = "qwen3")]
            ModelType::Qwen3,
            #[cfg(feature = "qwen35-4b")]
            ModelType::Qwen35,
        ]
        .to_vec()
    }

    #[test]
    fn consumed_and_core_are_real_arg_ids() {
        let ids: BTreeSet<String> = Args::command()
            .get_arguments()
            .map(|arg| arg.get_id().to_string())
            .collect();
        for id in CORE_ARGS {
            assert!(ids.contains(*id), "core arg {id} is not a real CLI arg id");
        }
        for model_type in all_model_types() {
            for id in consumed_args(model_type) {
                assert!(
                    ids.contains(*id),
                    "{model_type:?} lists {id}, which is not a real CLI arg id"
                );
            }
        }
    }

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

    #[cfg(feature = "qwen3")]
    #[test]
    fn parses_supported_max_lora_rank() {
        assert_eq!(parse_max_lora_rank_arg("16").expect("parse rank"), 16);
        assert_eq!(parse_max_lora_rank_arg("320").expect("parse rank"), 320);
    }

    #[cfg(feature = "qwen3")]
    #[test]
    fn qwen3_lora_default_rank_is_64() {
        assert_eq!(Qwen3LoraOptions::default().max_lora_rank, 64);
    }

    #[cfg(feature = "qwen3")]
    #[test]
    fn qwen3_accepts_graph_png_dump() {
        let (args, provided) =
            parse_with_provided(&["openinfer", "--dump-graph-png", "decode.png"]);
        args.validate(ModelType::Qwen3, &provided)
            .expect("Qwen3 should accept a graph PNG dump with CUDA Graph enabled");
    }

    #[cfg(feature = "qwen3")]
    #[test]
    fn qwen3_graph_png_dump_requires_cuda_graph() {
        let (args, provided) = parse_with_provided(&[
            "openinfer",
            "--dump-graph-png",
            "decode.png",
            "--cuda-graph=false",
        ]);
        let error = args
            .validate(ModelType::Qwen3, &provided)
            .expect_err("graph dump without CUDA Graph should be rejected");
        assert!(error.to_string().contains("requires --cuda-graph=true"));
    }

    #[cfg(feature = "qwen3")]
    #[test]
    fn qwen3_graph_png_dump_rejects_lora() {
        let (args, provided) = parse_with_provided(&[
            "openinfer",
            "--dump-graph-png",
            "decode.png",
            "--enable-lora",
        ]);
        let error = args
            .validate(ModelType::Qwen3, &provided)
            .expect_err("graph dump with LoRA should be rejected");
        assert!(
            error
                .to_string()
                .contains("not supported with --enable-lora")
        );
    }

    #[cfg(feature = "qwen3")]
    #[test]
    fn rejects_unsupported_max_lora_rank() {
        let error = parse_max_lora_rank_arg("7").expect_err("rank should be unsupported");

        assert!(error.contains("--max-lora-rank must be one of"));
        assert!(error.contains("16"));
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_accepts_omitted_dp_size() {
        let (args, provided) = parse_with_provided(&["openinfer"]);
        args.validate(ModelType::Glm52, &provided)
            .expect("GLM5.2 should default to DP8/EP8 when --dp-size is omitted");
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_rejects_non_dp8() {
        let (args, provided) = parse_with_provided(&["openinfer", "--dp-size", "1"]);
        let error = args
            .validate(ModelType::Glm52, &provided)
            .expect_err("GLM5.2 should reject explicit non-DP8");
        assert!(error.to_string().contains("DP8/EP8"));
    }
}
