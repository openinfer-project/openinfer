pub mod kernel_plan;

mod batch_decode;
mod batch_decode_buffers;
mod batch_decode_dag;
pub mod batch_decode_trace;
mod config;
mod dflash;
mod executor;
pub(crate) mod green_ctx;
pub mod kernel_bench;
mod lora;
mod prefill;
mod scheduler;
mod speculative;
mod unified_forward;
mod verify_graph;
mod weights;

use std::path::{Path, PathBuf};

use anyhow::Result;
use log::{info, warn};
use openinfer_core::engine::{EngineHandle, EngineLoadOptions, EpBackend, ModelInfo};

pub use kernel_plan::kernel_plan;
pub use scheduler::DEFAULT_MAX_PREFILL_TOKENS;
pub use weights::{
    DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES, Qwen3MemoryOptions,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3LoraOptions {
    pub max_loras: usize,
    pub max_lora_rank: usize,
}

impl Qwen3LoraOptions {
    pub const DEFAULT_MAX_LORAS: usize = 1;
    pub const DEFAULT_MAX_LORA_RANK: usize = 64;
    pub const SUPPORTED_MAX_LORA_RANKS: [usize; 9] = [1, 8, 16, 32, 64, 128, 256, 320, 512];

    pub fn validate(self) -> Result<Self> {
        anyhow::ensure!(self.max_loras > 0, "max_loras must be >= 1");
        anyhow::ensure!(
            Self::is_supported_max_lora_rank(self.max_lora_rank),
            "max_lora_rank must be one of: {}",
            Self::supported_max_lora_ranks_display()
        );
        Ok(self)
    }

    pub fn is_supported_max_lora_rank(rank: usize) -> bool {
        Self::SUPPORTED_MAX_LORA_RANKS.contains(&rank)
    }

    pub fn supported_max_lora_ranks_display() -> String {
        Self::SUPPORTED_MAX_LORA_RANKS
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl Default for Qwen3LoraOptions {
    fn default() -> Self {
        Self {
            max_loras: Self::DEFAULT_MAX_LORAS,
            max_lora_rank: Self::DEFAULT_MAX_LORA_RANK,
        }
    }
}

/// Prefill/decode GPU-sharing mode (`--decode-overlap`). Defined alongside the
/// stream plumbing in [`green_ctx`].
pub use green_ctx::DecodeOverlap;

/// KV-offload (pegaflow) opt-in for the single-GPU Qwen3 path.
///
/// Disabled by default — the existing GPU-only prefix cache is unchanged.
/// When enabled, the executor saves sealed KV blocks to pegaflow's host tier
/// and prefetches CPU-resident prefixes back into HBM before prefill, so a
/// prompt that has fallen out of the GPU cache still skips recompute. Only the
/// single-GPU topology is supported (tensor parallel shards KV per rank).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3OffloadOptions {
    pub enabled: bool,
    /// Host pinned-memory pool size (the CPU KV-tier capacity), in bytes.
    pub pinned_pool_bytes: usize,
}

impl Qwen3OffloadOptions {
    /// 8 GiB host tier — a few thousand dense Qwen3-4B blocks.
    pub const DEFAULT_PINNED_POOL_BYTES: usize = 8 << 30;

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            pinned_pool_bytes: 0,
        }
    }

    pub fn enabled(pinned_pool_bytes: usize) -> Self {
        Self {
            enabled: true,
            pinned_pool_bytes,
        }
    }
}

impl Default for Qwen3OffloadOptions {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Low-level Qwen3 execution interface.
///
/// This is the production phase boundary used by the Qwen3 scheduler and by
/// model-local benchmarks. The root server should use `start_engine` instead.
pub mod runtime {
    pub use crate::batch_decode_buffers::split_chunk_size_for;
    pub use crate::executor::{
        DecodePlan, DecodeRequestResult, DecodeResult, DecodeStepItem, PrefillPlan,
        PrefillRequestResult, PrefillResult, PrefillStepItem, Qwen3Executor, RequestId,
        UnifiedPlan, UnifiedResult,
    };
}

pub fn probe_model(model_path: &Path) -> Result<Option<ModelInfo>> {
    let config_path = model_path.join("config.json");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let json: serde_json::Value = serde_json::from_str(&content)?;
    if json.get("text_config").is_some() {
        return Ok(None);
    }

    Ok(Some(ModelInfo {
        id: "qwen3-4b",
        display_name: "Qwen3-4B".to_string(),
        model_path: model_path.to_path_buf(),
        max_model_len: json
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok()),
    }))
}

/// Server-facing launch knobs for the Qwen3 engine.
///
/// The binary maps raw CLI flags into this struct; [`launch`] then owns the
/// Qwen3 startup policy — the TP→device mapping and the LoRA↔CUDA-Graph
/// exclusion — and dispatches to the right low-level entry. That policy lives
/// with the model instead of leaking into the server.
#[derive(Clone, Debug)]
pub struct Qwen3LaunchOptions {
    /// CUDA device for single-GPU loads (ignored when `tp_size > 1`).
    pub device_ordinal: usize,
    /// Tensor-parallel world size; `> 1` uses devices `0..tp_size`.
    pub tp_size: usize,
    /// Whether the user requested CUDA Graph. LoRA serving forces it off.
    pub cuda_graph: bool,
    pub offload: Qwen3OffloadOptions,
    pub no_prefix_cache: bool,
    pub max_prefill_tokens: usize,
    pub memory: Qwen3MemoryOptions,
    /// `Some` switches on LoRA serving (and disables CUDA Graph).
    pub lora: Option<Qwen3LoraOptions>,
    /// How prefill and decode share the GPU (`--decode-overlap`).
    pub decode_overlap: DecodeOverlap,
    pub batch_invariant: bool,
    /// `Some` enables DFlash speculative decoding with this drafter model.
    /// Single-GPU only and mutually exclusive with LoRA and KV offload.
    pub dflash_draft_model_path: Option<PathBuf>,
    /// Publish KV block store/remove events for an out-of-band cache-aware
    /// router (e.g. a Dynamo KV router). Off for plain single-machine serving;
    /// single-GPU + base-model only (rejected with LoRA or tensor parallel).
    pub enable_kv_events: bool,
}

/// Start the Qwen3 engine from server-facing [`Qwen3LaunchOptions`].
pub fn launch(model_path: &Path, options: Qwen3LaunchOptions) -> Result<EngineHandle> {
    let device_ordinals = if options.tp_size == 1 {
        vec![options.device_ordinal]
    } else {
        (0..options.tp_size).collect()
    };
    // LoRA serving repoints adapter weights between steps, which a captured
    // decode graph bakes in — the two cannot coexist, so LoRA wins.
    let enable_cuda_graph = if options.lora.is_some() {
        if options.cuda_graph {
            warn!("Qwen3: CUDA Graph is disabled while LoRA serving is enabled");
        }
        false
    } else {
        options.cuda_graph
    };
    let engine = EngineLoadOptions {
        enable_cuda_graph,
        enable_prefill_profile: false,
        device_ordinals,
        parallel_config: None,
        ep_backend: EpBackend::Nccl,
        seed: 42,
    };
    if options.offload.enabled {
        info!(
            "Qwen3 KV offload enabled: host tier {:.1} GiB, no_prefix_cache={}",
            options.offload.pinned_pool_bytes as f64 / f64::from(1u32 << 30),
            options.no_prefix_cache
        );
    }
    anyhow::ensure!(
        !(options.dflash_draft_model_path.is_some() && options.lora.is_some()),
        "DFlash speculative decoding cannot be combined with LoRA serving"
    );
    anyhow::ensure!(
        !(options.dflash_draft_model_path.is_some()
            && !matches!(options.decode_overlap, DecodeOverlap::Off)),
        "DFlash speculative decoding cannot be combined with decode overlap \
         (--decode-overlap): the speculative path never takes the unified overlap \
         route, so the overlap streams would only waste VRAM the drafter needs"
    );
    if options.enable_kv_events && options.lora.is_some() {
        anyhow::bail!("KV block events are not supported with LoRA serving");
    }
    match options.lora {
        Some(lora) => {
            info!(
                "Starting Qwen3 engine with LoRA control; max_loras={}, max_lora_rank={}",
                lora.max_loras, lora.max_lora_rank
            );
            start_engine_with_lora_control(
                model_path,
                engine,
                lora,
                options.offload,
                options.no_prefix_cache,
                options.max_prefill_tokens,
                options.memory,
                options.decode_overlap,
                options.batch_invariant,
            )
        }
        None => start_engine_with_offload(
            model_path,
            engine,
            options.offload,
            options.no_prefix_cache,
            options.max_prefill_tokens,
            options.memory,
            options.decode_overlap,
            options.batch_invariant,
            options.dflash_draft_model_path.as_deref(),
            options.enable_kv_events,
        ),
    }
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    start_engine_with_offload(
        model_path,
        options,
        Qwen3OffloadOptions::disabled(),
        false,
        DEFAULT_MAX_PREFILL_TOKENS,
        Qwen3MemoryOptions::default(),
        DecodeOverlap::Off,
        false,
        None,
        false,
    )
}

/// Like [`start_engine`] but with pegaflow KV offload (single-GPU only). The
/// host tier persists sealed KV blocks and serves CPU-resident prefixes back
/// into HBM before prefill.
///
/// `no_prefix_cache` is the vLLM-style switch (see
/// [`Qwen3Executor::set_no_prefix_cache`](runtime::Qwen3Executor::set_no_prefix_cache)):
/// without offload it disables prefix matching outright; with offload it keeps
/// the host tier but stops cross-request HBM reuse, so every prefix is served
/// from L2 — the pure-L2 benchmark mode.
///
/// `max_prefill_tokens` caps the total prompt tokens batch-prefilled in one
/// scheduler step (see [`DEFAULT_MAX_PREFILL_TOKENS`]).
#[allow(clippy::too_many_arguments)]
pub fn start_engine_with_offload(
    model_path: &Path,
    options: EngineLoadOptions,
    offload_options: Qwen3OffloadOptions,
    no_prefix_cache: bool,
    max_prefill_tokens: usize,
    memory_options: Qwen3MemoryOptions,
    decode_overlap: DecodeOverlap,
    batch_invariant: bool,
    dflash_draft_model_path: Option<&Path>,
    enable_kv_events: bool,
) -> Result<EngineHandle> {
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    let model_path = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path must be valid UTF-8"))?;
    ensure_batch_invariant_supported(
        decode_overlap,
        batch_invariant,
        dflash_draft_model_path.is_some(),
    )?;
    apply_batch_invariant_policy(batch_invariant);
    let dflash_draft_model_path = dflash_draft_model_path
        .map(|path| {
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("DFlash draft model path must be valid UTF-8"))
        })
        .transpose()?;
    scheduler::start_qwen3(
        model_path,
        enable_cuda_graph,
        &device_ordinals,
        seed,
        offload_options,
        no_prefix_cache,
        max_prefill_tokens,
        memory_options,
        decode_overlap,
        dflash_draft_model_path,
        enable_kv_events,
    )
}

pub fn start_engine_with_lora_control(
    model_path: &Path,
    options: EngineLoadOptions,
    lora_options: Qwen3LoraOptions,
    offload_options: Qwen3OffloadOptions,
    no_prefix_cache: bool,
    max_prefill_tokens: usize,
    memory_options: Qwen3MemoryOptions,
    decode_overlap: DecodeOverlap,
    batch_invariant: bool,
) -> Result<EngineHandle> {
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    let model_path = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path must be valid UTF-8"))?;
    ensure_batch_invariant_supported(decode_overlap, batch_invariant, false)?;
    if batch_invariant {
        anyhow::bail!(
            "--batch-invariant is not supported with LoRA: the decode-GEMM pin warms and self-checks \
             only the base projections, not the per-adapter rank-K shapes, so the combination is \
             unverified"
        );
    }
    apply_batch_invariant_policy(batch_invariant);
    scheduler::start_qwen3_with_lora_control(
        model_path,
        enable_cuda_graph,
        &device_ordinals,
        seed,
        lora_options.validate()?,
        offload_options,
        no_prefix_cache,
        max_prefill_tokens,
        memory_options,
        decode_overlap,
    )
}

fn apply_batch_invariant_policy(batch_invariant: bool) {
    use openinfer_kernels::ops::{NumericPolicy, set_numeric_policy};
    let policy = if batch_invariant {
        NumericPolicy::Pin
    } else {
        NumericPolicy::Tuned
    };
    info!("Qwen3 numeric policy: {policy:?} (--batch-invariant={batch_invariant})");
    set_numeric_policy(policy);
}

fn ensure_batch_invariant_supported(
    decode_overlap: DecodeOverlap,
    batch_invariant: bool,
    dflash: bool,
) -> Result<()> {
    if batch_invariant && !matches!(decode_overlap, DecodeOverlap::Off) {
        anyhow::bail!(
            "--batch-invariant is not compatible with --decode-overlap; the stream override would force the pinned GEMM to bail at runtime"
        );
    }
    if batch_invariant && dflash {
        anyhow::bail!(
            "--batch-invariant is not supported with DFlash speculative decoding: the decode-GEMM \
             pin warms and self-checks only the base projections, so the drafter's fc/MLP GEMMs \
             are not pinned and would bail at the GEMM boundary under Pin"
        );
    }
    Ok(())
}
