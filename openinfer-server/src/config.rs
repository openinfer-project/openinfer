use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Result;
use anyhow::bail;
use clap::CommandFactory;
use clap::Parser;
use clap::ValueEnum;
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
    /// disable). Rejected for GLM5.2; forced off in Qwen3 LoRA mode; Qwen3.5
    /// always captures and rejects `false`.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub cuda_graph: bool,

    /// Dump a live rank-0 decode CUDA Graph during startup. Qwen3 exports its
    /// batch-1 SplitKv graph; GLM5.2 exports EP8 bucket 1 or the fixed TP8
    /// bucket 8 selected by `--moe-topo`. Writes a complete sibling `.dot` for
    /// machine inspection and a folded Graphviz PNG at this path. Requires
    /// CUDA driver API 12.3 or newer for kernel-name inspection.
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

    /// Tensor-parallel world size. GLM5.2 supports TP1/EP8 today; TP4/GB300
    /// bring-up uses --tp-size=4 --moe-topo=tp4.
    #[arg(long, default_value_t = 1)]
    pub tp_size: usize,

    /// Data-parallel world size. Kimi-K2 and GLM5.2 EP8/TP8 default to 8;
    /// GLM5.2 TP4 defaults to 1.
    #[arg(long)]
    pub dp_size: Option<usize>,

    /// Expert-parallel backend for Kimi-K2 (TP1/DP8 requires deepep; TP8/DP1 requires nccl)
    #[arg(long, default_value = "deepep")]
    pub ep_backend: CliEpBackend,

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

    /// P/D decode role with a vLLM prefill peer: the shared PYTHONHASHSEED
    /// value set on every vLLM prefill process. Switches offload query keys to
    /// vLLM's prefix-cache hash scheme (requires the P side to run
    /// --prefix-caching-hash-algo xxhash_cbor) and makes a cold request wait
    /// out the producer's registration tail instead of prefilling locally.
    /// Requires --kv-pd-vllm-namespace and the P2P mesh flags.
    #[arg(long, value_parser = parse_pythonhashseed, requires_all = ["kv_p2p_metaserver_addr", "kv_pd_vllm_namespace"])]
    pub kv_pd_vllm_seed: Option<String>,

    /// The vLLM prefill peer's pegaflow-connector namespace (an 8-hex digest
    /// the connector logs at startup as `namespace=...`). Both sides must
    /// address the same content domain. The digest carries no model identity:
    /// pointing a decode node at a different model's namespace (same
    /// tokenizer, same geometry class) silently cross-loads foreign KV.
    #[arg(long, value_parser = parse_pegaflow_namespace, requires = "kv_pd_vllm_seed")]
    pub kv_pd_vllm_namespace: Option<String>,

    /// Zero-hit wait window for --kv-pd-vllm-seed mode, in milliseconds: how
    /// long a cold request keeps re-querying before giving up on the expected
    /// remote KV. On give-up, Qwen3 prefills locally; GLM5.2 rejects for the
    /// router to retry (see --kv-pd-allow-local-prefill). Must stay below the
    /// executor's 15s remote-fetch deadline (both engines enforce this at
    /// startup).
    #[arg(long, default_value_t = 5000, requires = "kv_pd_vllm_seed")]
    pub kv_pd_miss_wait_ms: u64,

    /// Debug escape hatch for --kv-pd-vllm-seed mode on models whose decode
    /// node has no prefill path (GLM5.2): admit with local prompt compute
    /// when the remote KV never materializes, instead of rejecting for the
    /// router to retry. Local prompt compute rides the decode kernels
    /// token-by-token — leave off in production. Qwen3 ignores this flag
    /// (its miss path always falls back to a real local prefill).
    #[arg(long, default_value_t = false, requires = "kv_pd_vllm_seed")]
    pub kv_pd_allow_local_prefill: bool,

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

    /// Decode-batch capacity, 1..=64. Qwen3.5 internally rounds allocation to
    /// the next graph bucket but admits only this many scheduler slots; defaults
    /// to 64.
    #[arg(long)]
    pub max_batch: Option<usize>,

    /// Qwen3.5 prefill/decode scheduler policy. Defaults to `off`; `auto` is
    /// opt-in and currently single-GPU only.
    #[arg(long, value_enum, default_value_t = CliQwen35SchedulerPolicy::Off)]
    pub qwen35_scheduler_policy: CliQwen35SchedulerPolicy,

    /// Per-request context cap: prompt + max_tokens - 1 must fit. GLM5.2 only;
    /// when omitted, GLM5.2 sizes it from post-weight-load free VRAM.
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Run GLM5.2 TP4 with prefix caching and no decode.
    #[arg(long, default_value_t = false)]
    pub glm52_prefill_only: bool,

    /// Token rows per prefill chunk. Must be a multiple of 64.
    #[arg(long, default_value_t = 16_384)]
    pub glm52_prefill_chunk_size: usize,

    /// GLM5.2 launch-time MoE sharding topology: `ep8` (default) is the
    /// high-throughput configuration (32 whole experts per rank, DeepEP
    /// dispatch/combine, buckets 1-8); `ep4` is its four-GPU counterpart
    /// (64 whole experts per rank, weight-only expert GEMMs — the GB300
    /// high-throughput topology); `tp8` is the low-latency
    /// configuration (1/8-intermediate slice of ALL experts per rank on
    /// every MoE layer, bucket-1 only — at most one request per rank);
    /// `tp4` is the GB300 four-GPU low-latency topology.
    #[arg(long, default_value = "ep8")]
    pub moe_topo: String,

    /// Stage GLM5.2 checkpoint bytes through pinned double buffers. This can
    /// substantially accelerate warm-page-cache loads; leave off for cold
    /// network-filesystem starts.
    #[arg(long)]
    pub glm52_weight_staging: bool,

    /// GLM5.2 remote rank-host nodes for cross-node EP, comma-separated
    /// `host:port=ranks` (e.g. `10.13.84.7:19000=4`). Each node contributes
    /// its ranks AFTER this process's local ranks, in list order; the total
    /// must equal the topology's rank count. Start the remote side with
    /// `--glm52-rank-host`.
    #[arg(long, value_delimiter = ',')]
    pub rank_hosts: Vec<String>,

    /// Serve as a GLM5.2 rank-host on this listen address (e.g.
    /// `0.0.0.0:19000`) instead of running an engine: a coordinator started
    /// with `--rank-hosts` connects and drives this node's GPUs. No HTTP
    /// frontend, no scheduler — a dumb worker shell.
    #[arg(long)]
    pub glm52_rank_host: Option<String>,

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

    /// Enable single-GPU Qwen3 batch-invariant serving by pinning the numeric paths and cutting
    /// each prompt's prefill chunks on its own grid. Off by default. Requires `--no-prefix-cache`;
    /// incompatible with `--kv-offload`, which keeps prefix matching on regardless.
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

/// CLI selector for the Qwen3.5 adaptive scheduler policy.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum CliQwen35SchedulerPolicy {
    /// Fixed chunked-prefill behavior.
    #[default]
    Off,
    /// Runtime-state adaptive policy.
    Auto,
}

impl CliQwen35SchedulerPolicy {
    #[cfg(feature = "qwen35")]
    pub(crate) fn resolve(self) -> openinfer_qwen35::Qwen35SchedulerPolicy {
        match self {
            Self::Off => openinfer_qwen35::Qwen35SchedulerPolicy::Off,
            Self::Auto => openinfer_qwen35::Qwen35SchedulerPolicy::Auto,
        }
    }
}

/// Flags accepted for every model line regardless of detected type.
const CORE_ARGS: &[&str] = &["model_path", "served_model_name", "port"];

/// The CLI arg ids each model line uses — the applicability source of truth for
/// `validate()`.
fn consumed_args(model_type: ModelType) -> &'static [&'static str] {
    match model_type {
        #[cfg(feature = "deepseek-v2-lite")]
        ModelType::DeepSeekV2Lite => &["cuda_graph"],
        #[cfg(feature = "glm52")]
        ModelType::Glm52 => &[
            "tp_size",
            "dp_size",
            "dflash_draft_model_path",
            "max_model_len",
            "glm52_prefill_only",
            "glm52_prefill_chunk_size",
            "no_prefix_cache",
            "kv_offload",
            "kv_offload_host_gib",
            "kv_offload_hugepages",
            "kv_p2p_metaserver_addr",
            "kv_p2p_advertise_addr",
            "kv_p2p_nics",
            "kv_pd_vllm_seed",
            "kv_pd_vllm_namespace",
            "kv_pd_miss_wait_ms",
            "kv_pd_allow_local_prefill",
            "moe_topo",
            "glm52_weight_staging",
            "dump_graph_png",
            "rank_hosts",
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
            "kv_pd_vllm_seed",
            "kv_pd_vllm_namespace",
            "kv_pd_miss_wait_ms",
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
        #[cfg(feature = "qwen35")]
        ModelType::Qwen35 => &[
            "device_ordinal",
            "tp_size",
            "cuda_graph",
            "max_prefill_tokens",
            "max_batch",
            "qwen35_scheduler_policy",
        ],
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
        if self.batch_invariant && self.kv_offload {
            bail!(
                "--batch-invariant is not supported with --kv-offload: offload keeps prefix matching \
                 on (--no-prefix-cache only disables HBM retention there), and a host-tier prefix hit \
                 shifts a prompt's chunk boundaries off the request-local grid"
            );
        }
        if self.batch_invariant && self.dflash_draft_model_path.is_some() {
            bail!(
                "--batch-invariant is not supported with DFlash speculative decoding; enable one at a time"
            );
        }
        if self.batch_invariant && self.tp_size > 1 {
            bail!("--batch-invariant is not supported with --tp-size > 1; enable one at a time");
        }
        if self.batch_invariant && !self.no_prefix_cache {
            bail!(
                "--batch-invariant requires --no-prefix-cache; prefix-cache hits move a prompt's chunk \
                 boundaries off the request-local grid, so batch-invariant prefill cannot be provided"
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
        #[cfg(feature = "qwen35")]
        if matches!(model_type, ModelType::Qwen35) {
            if let Some(max_batch) = self.max_batch {
                if !(1..=openinfer_qwen35::MAX_DECODE_BATCH).contains(&max_batch) {
                    bail!(
                        "--max-batch must be in 1..={} for Qwen3.5, got {max_batch}",
                        openinfer_qwen35::MAX_DECODE_BATCH
                    );
                }
            }
            if self.tp_size > 1
                && matches!(self.qwen35_scheduler_policy, CliQwen35SchedulerPolicy::Auto)
            {
                bail!(
                    "--qwen35-scheduler-policy=auto is single-GPU only; Qwen3.5 TP uses the fixed off policy"
                );
            }
        }
        #[cfg(feature = "glm52")]
        if matches!(model_type, ModelType::Glm52) {
            // Parse the topology here so an invalid --moe-topo string fails
            // with the real problem instead of a misleading dp/tp complaint;
            // the accepted strings live in one place (the model crate).
            let moe_topo: openinfer_glm52::Glm52MoeTopo = self
                .moe_topo
                .parse()
                .map_err(|err| anyhow::anyhow!("--moe-topo: {err}"))?;
            if let Some(dp_size) = self.dp_size {
                let expected_dp_size = moe_topo.default_dp_size();
                if dp_size != expected_dp_size {
                    bail!(
                        "GLM5.2 --moe-topo={} requires --dp-size={} when provided; omit --dp-size to use the topology default",
                        self.moe_topo,
                        expected_dp_size
                    );
                }
            }
            let expected_tp_size = moe_topo.expected_tp_size();
            if self.tp_size != expected_tp_size {
                bail!(
                    "GLM5.2 --moe-topo={} requires --tp-size={expected_tp_size}, got {}",
                    self.moe_topo,
                    self.tp_size
                );
            }
            if self.glm52_prefill_only {
                if !matches!(moe_topo, openinfer_glm52::Glm52MoeTopo::Tp4) {
                    bail!("--glm52-prefill-only requires --moe-topo=tp4");
                }
                if self.no_prefix_cache {
                    bail!("--glm52-prefill-only requires prefix caching; drop --no-prefix-cache");
                }
                if self.dflash_draft_model_path.is_some() {
                    bail!("--glm52-prefill-only is incompatible with the DSpark drafter");
                }
                if self.kv_offload || self.kv_pd_vllm_seed.is_some() {
                    bail!(
                        "--glm52-prefill-only does not support KV offload or an external P/D peer"
                    );
                }
                if self.dump_graph_png.is_some() {
                    bail!("--glm52-prefill-only does not expose a decode CUDA graph");
                }
            } else if provided.contains("glm52_prefill_chunk_size") {
                bail!("--glm52-prefill-chunk-size requires --glm52-prefill-only");
            }
            if self.glm52_prefill_chunk_size == 0
                || !self
                    .glm52_prefill_chunk_size
                    .is_multiple_of(openinfer_glm52::GLM52_PREFILL_CHUNK_ALIGN)
            {
                bail!(
                    "--glm52-prefill-chunk-size must be a positive multiple of {}, got {}",
                    openinfer_glm52::GLM52_PREFILL_CHUNK_ALIGN,
                    self.glm52_prefill_chunk_size
                );
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

/// PYTHONHASHSEED as vLLM accepts it: a decimal integer in [0, 4294967295].
/// An empty or malformed seed would derive a well-formed key space that can
/// never match the peer — a config error must fail here, not as slow requests.
fn parse_pythonhashseed(s: &str) -> Result<String, String> {
    if s.parse::<u32>().is_err() || s.starts_with('+') {
        return Err(format!(
            "PYTHONHASHSEED must be a decimal integer in [0, 4294967295], got {s:?}"
        ));
    }
    Ok(s.to_string())
}

/// A pegaflow namespace digest: exactly 8 lowercase hex chars.
fn parse_pegaflow_namespace(s: &str) -> Result<String, String> {
    if s.len() != 8
        || !s
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(format!(
            "namespace must be an 8-char lowercase hex digest, got {s:?}"
        ));
    }
    Ok(s.to_string())
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

    #[cfg(any(feature = "glm52", feature = "qwen3", feature = "qwen35"))]
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
            #[cfg(feature = "deepseek-v2-lite")]
            ModelType::DeepSeekV2Lite,
            #[cfg(feature = "glm52")]
            ModelType::Glm52,
            #[cfg(feature = "kimi-k2")]
            ModelType::KimiK2,
            #[cfg(feature = "qwen3")]
            ModelType::Qwen3,
            #[cfg(feature = "qwen35")]
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

    #[cfg(feature = "qwen35")]
    #[test]
    fn qwen35_accepts_tp_size() {
        let (args, provided) =
            parse_with_provided(&["openinfer", "--tp-size", "2", "--cuda-graph=false"]);
        args.validate(ModelType::Qwen35, &provided)
            .expect("Qwen3.5 should accept --tp-size for eager TP startup");
    }

    #[cfg(feature = "qwen35")]
    #[test]
    fn qwen35_defaults_scheduler_policy_off() {
        let (args, provided) = parse_with_provided(&["openinfer"]);
        args.validate(ModelType::Qwen35, &provided)
            .expect("Qwen3.5 should accept default scheduler-policy off");
        assert!(matches!(
            args.qwen35_scheduler_policy,
            CliQwen35SchedulerPolicy::Off
        ));
    }

    #[cfg(feature = "qwen35")]
    #[test]
    fn qwen35_accepts_scheduler_policy_off() {
        let (args, provided) =
            parse_with_provided(&["openinfer", "--qwen35-scheduler-policy", "off"]);
        args.validate(ModelType::Qwen35, &provided)
            .expect("Qwen3.5 should accept explicit scheduler-policy off");
        assert!(matches!(
            args.qwen35_scheduler_policy,
            CliQwen35SchedulerPolicy::Off
        ));
    }

    #[cfg(feature = "qwen35")]
    #[test]
    fn qwen35_rejects_tp_auto_scheduler_policy() {
        let (args, provided) = parse_with_provided(&[
            "openinfer",
            "--tp-size",
            "2",
            "--cuda-graph=false",
            "--qwen35-scheduler-policy",
            "auto",
        ]);
        let err = args
            .validate(ModelType::Qwen35, &provided)
            .expect_err("Qwen3.5 TP should reject auto scheduler-policy")
            .to_string();
        assert!(err.contains("single-GPU only"));
    }

    #[cfg(feature = "qwen35")]
    #[test]
    fn qwen35_accepts_non_bucket_scheduler_max_batch() {
        let (args, provided) = parse_with_provided(&["openinfer", "--max-batch", "5"]);
        args.validate(ModelType::Qwen35, &provided)
            .expect("Qwen3.5 should accept scheduler max_batch between decode buckets");
        assert_eq!(args.max_batch, Some(5));
    }

    #[cfg(feature = "qwen35")]
    #[test]
    fn qwen35_rejects_zero_scheduler_max_batch() {
        let (args, provided) = parse_with_provided(&["openinfer", "--max-batch", "0"]);
        let err = args
            .validate(ModelType::Qwen35, &provided)
            .expect_err("Qwen3.5 should reject zero scheduler max_batch")
            .to_string();
        assert!(
            err.contains("--max-batch must be in 1..="),
            "unexpected error: {err}"
        );
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
    fn glm52_accepts_graph_png_dump() {
        let (args, provided) =
            parse_with_provided(&["openinfer", "--dump-graph-png", "decode.png"]);
        args.validate(ModelType::Glm52, &provided)
            .expect("GLM5.2 should accept a graph PNG dump");
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
    fn glm52_rejects_non_dp8_for_ep8() {
        let (args, provided) = parse_with_provided(&["openinfer", "--dp-size", "1"]);
        let error = args
            .validate(ModelType::Glm52, &provided)
            .expect_err("GLM5.2 should reject explicit non-DP8");
        assert!(error.to_string().contains("--dp-size=8"));
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_accepts_tp4_dp1() {
        let (args, provided) =
            parse_with_provided(&["openinfer", "--moe-topo", "tp4", "--tp-size", "4"]);
        args.validate(ModelType::Glm52, &provided)
            .expect("GLM5.2 TP4 should default to DP1");
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_tp4_rejects_non_dp1() {
        let (args, provided) = parse_with_provided(&[
            "openinfer",
            "--moe-topo",
            "tp4",
            "--tp-size",
            "4",
            "--dp-size",
            "8",
        ]);
        let error = args
            .validate(ModelType::Glm52, &provided)
            .expect_err("GLM5.2 TP4 should reject explicit non-DP1");
        assert!(error.to_string().contains("--dp-size=1"));
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_tp4_rejects_omitted_tp_size() {
        let (args, provided) = parse_with_provided(&["openinfer", "--moe-topo", "tp4"]);
        let error = args
            .validate(ModelType::Glm52, &provided)
            .expect_err("GLM5.2 TP4 should reject the default --tp-size=1");
        assert!(error.to_string().contains("--tp-size=4"));
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_ep8_rejects_tp4_tp_size() {
        let (args, provided) = parse_with_provided(&["openinfer", "--tp-size", "4"]);
        let error = args
            .validate(ModelType::Glm52, &provided)
            .expect_err("GLM5.2 EP8 should reject --tp-size=4");
        assert!(error.to_string().contains("--tp-size=1"));
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_rejects_unknown_moe_topo() {
        let (args, provided) = parse_with_provided(&["openinfer", "--moe-topo", "tp2"]);
        let error = args
            .validate(ModelType::Glm52, &provided)
            .expect_err("GLM5.2 should reject an unknown topology string");
        assert!(error.to_string().contains("ep8, ep4, tp8, or tp4"));
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_accepts_ep4_default_dp4() {
        let (args, provided) = parse_with_provided(&["openinfer", "--moe-topo", "ep4"]);
        args.validate(ModelType::Glm52, &provided)
            .expect("GLM5.2 EP4 should default to DP4 with --tp-size=1");
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_ep4_rejects_non_dp4() {
        let (args, provided) =
            parse_with_provided(&["openinfer", "--moe-topo", "ep4", "--dp-size", "8"]);
        let error = args
            .validate(ModelType::Glm52, &provided)
            .expect_err("GLM5.2 EP4 should reject explicit non-DP4");
        assert!(error.to_string().contains("--dp-size=4"));
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_prefill_only_accepts_tp4_defaults() {
        let (args, provided) = parse_with_provided(&[
            "openinfer",
            "--moe-topo",
            "tp4",
            "--tp-size",
            "4",
            "--glm52-prefill-only",
        ]);
        args.validate(ModelType::Glm52, &provided)
            .expect("TP4 prefill-only defaults should validate");
        assert_eq!(
            args.glm52_prefill_chunk_size,
            openinfer_glm52::GLM52_DEFAULT_PREFILL_CHUNK_SIZE
        );
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_prefill_only_rejects_decode_features() {
        for extra in [
            vec!["--no-prefix-cache"],
            vec!["--dflash-draft-model-path", "/tmp/dspark"],
            vec!["--dump-graph-png", "/tmp/decode.png"],
        ] {
            let mut argv = vec![
                "openinfer",
                "--moe-topo",
                "tp4",
                "--tp-size",
                "4",
                "--glm52-prefill-only",
            ];
            argv.extend(extra);
            let (args, provided) = parse_with_provided(&argv);
            args.validate(ModelType::Glm52, &provided)
                .expect_err("prefill-only must reject decode-only features");
        }
    }

    #[cfg(feature = "glm52")]
    #[test]
    fn glm52_prefill_chunk_requires_mode_and_page_alignment() {
        let (args, provided) =
            parse_with_provided(&["openinfer", "--glm52-prefill-chunk-size", "16384"]);
        let error = args
            .validate(ModelType::Glm52, &provided)
            .expect_err("an inert chunk size must be rejected");
        assert!(error.to_string().contains("requires --glm52-prefill-only"));

        let (args, provided) = parse_with_provided(&[
            "openinfer",
            "--moe-topo",
            "tp4",
            "--tp-size",
            "4",
            "--glm52-prefill-only",
            "--glm52-prefill-chunk-size",
            "16001",
        ]);
        let error = args
            .validate(ModelType::Glm52, &provided)
            .expect_err("unaligned chunk must be rejected");
        assert!(error.to_string().contains("positive multiple of 64"));
    }
}
