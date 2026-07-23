//! GLM5.2 DP8/EP8 engine surface.
//!
//! Startup validates the official GLM5.2 FP8 checkpoint layout, loads rank
//! slices to GPU memory (the non-expert stack replicated to every rank,
//! experts placed into their packed layout at H2D time), builds the resident
//! models, and serves greedy generation with one request per rank: every
//! step all 8 ranks run the full model in lock-step and enter the
//! per-MoE-layer DeepEP collectives.

mod bookend;
mod config;
mod dense;
mod dspark;
#[cfg(test)]
mod dspark_smoke;
mod fp8;
mod indexer;
#[cfg(test)]
mod indexer_smoke;
mod layer;
mod mla_decode;
mod mla_front;
mod model;
mod moe_decode;
mod moe_ep8;
mod moe_ep_wo;
mod moe_tp;
#[cfg(test)]
mod oracle;
mod remote;
mod rows;
mod runner;
mod scheduler;
mod scratch;
mod weights;

use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use bytesize::ByteSize;
pub(crate) use config::GLM52_LAYERS;
pub(crate) use config::GLM52_ROUTED_EXPERTS;
pub use config::probe_config_json;
use openinfer_core::engine::EngineHandle;
use openinfer_core::engine::KvCapacity;
use openinfer_core::engine::LoadSnapshot;
use openinfer_kv_offload::HostConfig;
use openinfer_kv_offload::KvArena;
use openinfer_kv_offload::OffloadEngine;
use openinfer_kv_offload::OffloadHost;
use remote::Glm52RemoteNode;
pub use remote::serve_rank_host;
use runner::Glm52RankPlacement;
use runner::Glm52RankWorker;
use runner::Glm52Worker;
use tokio::sync::mpsc;
use tokio::sync::watch;
use weights::GLM52_EP_RANKS;
use weights::Glm52RankLoadBundle;
use weights::Glm52WeightManifest;

use crate::config::GLM52_MAX_CONTEXT;
use crate::model::GLM52_MODEL_LEN_ALIGN;
use crate::model::glm52_arena_bytes;
use crate::model::glm52_pool_blocks;

/// GLM5.2 parallel shape. EP8 is the production layout today; TP4 is the
/// GB300 bring-up target.
#[derive(Clone, Debug)]
pub struct Glm52LaunchOptions {
    pub tp_size: usize,
    pub dp_size: usize,
    /// DSpark drafter checkpoint dir (`RedHatAI/GLM-5.2-speculator.dspark`).
    /// Enables speculative decoding for greedy AND sampled requests (the
    /// verify span prefix-matches per-row sampled tokens — lossless): verify
    /// spans ride the decode buckets, accepted tokens commit in batches,
    /// per-request accept stats are logged on release.
    pub dspark_draft_model_path: Option<std::path::PathBuf>,
    /// Per-request context cap (`prompt + max_tokens - 1 <= max_model_len`).
    /// `None` sizes it from the post-weight-load free VRAM (fleet minimum);
    /// an explicit value is still validated against that budget so an
    /// impossible cap fails at launch, not at the first long request.
    pub max_model_len: Option<usize>,
    /// vLLM-style kill switch: disable prefix matching outright (every
    /// prefill recomputes the full prompt). Prefix caching is also forced
    /// off while the DSpark drafter is on — the draft lane needs the
    /// aux-hidden captures a skipped prefix never produces.
    pub no_prefix_cache: bool,
    /// `Some` adds the pegaflow host tier under the prefix cache: sealed KV
    /// blocks flow to one shared pinned pool on request release, and a
    /// prompt whose prefix fell out of HBM restores from it at admission.
    /// Requires the prefix cache (rejected at launch alongside the DSpark
    /// drafter or `no_prefix_cache`).
    pub kv_offload: Option<Glm52KvOffloadOptions>,
    /// Launch-time MoE sharding topology. `Ep8` (default) is the
    /// high-throughput configuration: 32 whole experts per rank, DeepEP
    /// dispatch/combine, buckets 1-8. `Tp8` is the low-latency
    /// configuration: replicated activations over head-sharded weights —
    /// every rank holds a 1/8-intermediate slice of ALL experts plus 8 of
    /// the 64 attention heads, all 8 workers mirror ONE logical rank (up to
    /// 8 concurrent requests, single bucket-8 shape), and the MoE path is
    /// the TP8 phase-kernel chain on all 75 layers. `Tp4` is the GB300
    /// four-GPU bring-up target using 16 attention heads per rank and 1/4
    /// intermediate MoE slices.
    pub moe_topo: Glm52MoeTopo,
    /// Export rank 0's already pre-captured whole-step decode graph during
    /// startup. EP8 and TP4 export bucket 1; TP8 exports its fixed bucket 8.
    /// The requested PNG gets a complete sibling `.dot` for machine
    /// inspection.
    pub dump_graph_png: Option<PathBuf>,
    /// Remote rank-host nodes (cross-node EP): each entry contributes its
    /// `ranks` workers AFTER the local ranks, in list order. The local
    /// process keeps ranks `0..device_count - Σ remote` on its own GPUs.
    /// Empty (the default) is the single-node engine, byte-for-byte.
    pub rank_hosts: Vec<Glm52RankHostSpec>,
}

/// One `--rank-hosts` entry: `host:port=ranks` (e.g. `10.13.84.7:19000=4`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Glm52RankHostSpec {
    addr: String,
    ranks: usize,
}

impl std::str::FromStr for Glm52RankHostSpec {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let (addr, ranks) = s
            .rsplit_once('=')
            .with_context(|| format!("rank-host spec `{s}` must be host:port=ranks"))?;
        let ranks: usize = ranks
            .parse()
            .with_context(|| format!("rank-host spec `{s}` has a non-numeric rank count"))?;
        ensure!(
            !addr.is_empty() && ranks > 0,
            "rank-host spec `{s}` must be host:port=ranks with ranks > 0"
        );
        Ok(Self {
            addr: addr.to_string(),
            ranks,
        })
    }
}

/// Launch-time MoE sharding topology (the expert slab is repacked during
/// H2D load, so this is a boot choice — the two layouts never co-reside).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Glm52MoeTopo {
    #[default]
    Ep8,
    /// Four-GPU expert-parallel layout (DP4/EP4, 64 whole routed experts per
    /// rank — the GB300 high-throughput target). Same DeepEP protocol as EP8
    /// with its own shim instantiation; the routed-expert GEMM runs the
    /// arch-portable weight-only mma chain instead of the sm_90a DeepGEMM
    /// masked chain.
    Ep4,
    /// Cross-tray expert-parallel widths on GB300 NVL72 (4 GPUs per tray,
    /// remote ranks behind `--rank-hosts`). Same DeepEP protocol with one
    /// shim instantiation per width; all run the weight-only chain.
    Ep16,
    Ep32,
    Ep64,
    Tp8,
    Tp4,
}

impl Glm52MoeTopo {
    #[must_use]
    pub fn default_dp_size(self) -> usize {
        match self {
            Self::Tp4 => 1,
            _ => self.device_count(),
        }
    }

    #[must_use]
    fn device_count(self) -> usize {
        match self {
            Self::Ep8 | Self::Tp8 => GLM52_EP_RANKS,
            Self::Ep4 | Self::Tp4 => 4,
            Self::Ep16 => 16,
            Self::Ep32 => 32,
            Self::Ep64 => 64,
        }
    }

    /// Number of independently scheduled request partitions. Tensor-
    /// replicated workers execute one mirrored partition in lock-step.
    #[must_use]
    pub fn logical_rank_count(self) -> usize {
        if self.uses_tensor_replicated_moe() {
            1
        } else {
            self.device_count()
        }
    }

    /// The `--tp-size` this topology requires (server validation mirrors the
    /// launch-time ensure).
    #[must_use]
    pub fn expected_tp_size(self) -> usize {
        match self {
            Self::Tp4 => 4,
            _ => 1,
        }
    }

    #[must_use]
    fn expected_ep_size(self) -> usize {
        match self {
            Self::Tp8 => GLM52_EP_RANKS,
            Self::Tp4 => 1,
            _ => self.device_count(),
        }
    }

    #[must_use]
    fn uses_ep_expert_bundles(self) -> bool {
        !self.uses_tensor_replicated_moe()
    }

    /// Whole routed experts per rank of an expert-bundle topology (EP8 → 32,
    /// EP4 → 64). Meaningless for the tensor-replicated topologies.
    #[must_use]
    fn ep_local_experts(self) -> usize {
        debug_assert!(self.uses_ep_expert_bundles());
        GLM52_ROUTED_EXPERTS / self.expected_ep_size()
    }

    #[must_use]
    fn uses_tensor_replicated_moe(self) -> bool {
        matches!(self, Self::Tp8 | Self::Tp4)
    }
}

impl std::str::FromStr for Glm52MoeTopo {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "ep4" => Ok(Self::Ep4),
            "ep8" => Ok(Self::Ep8),
            "ep16" => Ok(Self::Ep16),
            "ep32" => Ok(Self::Ep32),
            "ep64" => Ok(Self::Ep64),
            "tp8" => Ok(Self::Tp8),
            "tp4" => Ok(Self::Tp4),
            other => {
                anyhow::bail!(
                    "GLM5.2 MoE topology must be ep4, ep8, ep16, ep32, ep64, tp8, or tp4, \
                     got {other}"
                )
            }
        }
    }
}

#[cfg(test)]
mod topology_tests {
    use super::*;

    #[test]
    fn tp4_topology_shape_is_four_rank_replicated_tp() {
        assert_eq!(Glm52MoeTopo::Tp4.default_dp_size(), 1);
        assert_eq!(Glm52MoeTopo::Tp4.device_count(), 4);
        assert_eq!(Glm52MoeTopo::Tp4.logical_rank_count(), 1);
        assert_eq!(Glm52MoeTopo::Tp4.expected_tp_size(), 4);
        assert_eq!(Glm52MoeTopo::Tp4.expected_ep_size(), 1);
        assert!(!Glm52MoeTopo::Tp4.uses_ep_expert_bundles());
        assert!(Glm52MoeTopo::Tp4.uses_tensor_replicated_moe());
    }

    #[test]
    fn tp8_and_ep8_shapes_remain_unchanged() {
        for topo in [Glm52MoeTopo::Ep8, Glm52MoeTopo::Tp8] {
            assert_eq!(topo.default_dp_size(), GLM52_EP_RANKS);
            assert_eq!(topo.device_count(), GLM52_EP_RANKS);
            assert_eq!(topo.expected_tp_size(), 1);
            assert_eq!(topo.expected_ep_size(), GLM52_EP_RANKS);
        }
        assert_eq!(Glm52MoeTopo::Ep8.logical_rank_count(), GLM52_EP_RANKS);
        assert_eq!(Glm52MoeTopo::Tp8.logical_rank_count(), 1);
        assert!(Glm52MoeTopo::Ep8.uses_ep_expert_bundles());
        assert!(!Glm52MoeTopo::Ep8.uses_tensor_replicated_moe());
        assert!(!Glm52MoeTopo::Tp8.uses_ep_expert_bundles());
        assert!(Glm52MoeTopo::Tp8.uses_tensor_replicated_moe());
        assert_eq!(Glm52MoeTopo::Ep8.ep_local_experts(), 32);
    }

    #[test]
    fn ep4_topology_shape_is_four_rank_expert_parallel() {
        assert_eq!(Glm52MoeTopo::Ep4.default_dp_size(), 4);
        assert_eq!(Glm52MoeTopo::Ep4.device_count(), 4);
        assert_eq!(Glm52MoeTopo::Ep4.logical_rank_count(), 4);
        assert_eq!(Glm52MoeTopo::Ep4.expected_tp_size(), 1);
        assert_eq!(Glm52MoeTopo::Ep4.expected_ep_size(), 4);
        assert!(Glm52MoeTopo::Ep4.uses_ep_expert_bundles());
        assert!(!Glm52MoeTopo::Ep4.uses_tensor_replicated_moe());
        assert_eq!(Glm52MoeTopo::Ep4.ep_local_experts(), 64);
        assert_eq!("ep4".parse::<Glm52MoeTopo>().unwrap(), Glm52MoeTopo::Ep4);
    }

    #[test]
    fn cross_tray_ep_widths_shard_all_routed_experts() {
        for (topo, ranks, local) in [
            (Glm52MoeTopo::Ep16, 16, 16),
            (Glm52MoeTopo::Ep32, 32, 8),
            (Glm52MoeTopo::Ep64, 64, 4),
        ] {
            assert_eq!(topo.default_dp_size(), ranks);
            assert_eq!(topo.device_count(), ranks);
            assert_eq!(topo.logical_rank_count(), ranks);
            assert_eq!(topo.expected_tp_size(), 1);
            assert_eq!(topo.expected_ep_size(), ranks);
            assert!(topo.uses_ep_expert_bundles());
            assert!(!topo.uses_tensor_replicated_moe());
            assert_eq!(topo.ep_local_experts(), local);
            assert_eq!(format!("ep{ranks}").parse::<Glm52MoeTopo>().unwrap(), topo);
        }
    }
}

/// Host-tier KV offload knobs. One `PegaEngine` (one pinned pool) backs all
/// 8 DP ranks under a single namespace: the MLA latent has no TP sharding
/// and the non-expert weights are replicated, so any rank's KV for a token
/// prefix is as good as any other's — the same tolerance as reusing a
/// rank's own prefix cache (FP reduction order may differ across the batch
/// shapes that computed it, never the semantics). Any rank restores what
/// any rank saved.
#[derive(Clone, Debug)]
pub struct Glm52KvOffloadOptions {
    /// Host pinned-memory pool size in bytes, shared by all ranks.
    pub pinned_pool_bytes: usize,
    /// Back the pool with hugepages (the box must hold a reservation —
    /// check `HugePages_Total`).
    pub use_hugepages: bool,
    /// `Some` joins the cross-instance P2P mesh: saved block hashes register
    /// with the MetaServer and missing prefixes are pulled from peer
    /// instances over RDMA — the P/D disaggregation data plane.
    pub p2p: Option<Glm52P2pOptions>,
    /// `Some` when the P/D prefill peer is vLLM (pegaflow connector): offload
    /// query keys switch from kvbm lineage hashes to vLLM's prefix-cache hash
    /// scheme so this decode node can find the blocks vLLM registered.
    /// Requires `p2p` (the peer's KV lives in its pegaflow-server's pool,
    /// even on the same host).
    pub vllm_compat: Option<Glm52VllmCompatOptions>,
}

/// Cross-instance P2P KV sharing (see `openinfer_kv_offload::P2pConfig`).
#[derive(Clone, Debug)]
pub struct Glm52P2pOptions {
    /// MetaServer gRPC address, e.g. `http://10.0.0.100:50056`.
    pub metaserver_addr: String,
    /// This engine's routable `IP:port` (doubles as the embedded transfer
    /// service's bind address). Must be reachable by every peer.
    pub advertise_addr: String,
    /// RDMA NIC device names to register the pinned pool on.
    pub rdma_nics: Vec<String>,
}

/// Decode-node settings for a P/D deployment whose prefill node is vLLM with
/// the pegaflow connector (see `openinfer_kv_offload::VllmBlockHasher` and
/// `docs/models/glm52/pd-vllm-prefill.md`).
///
/// GLM5.2's decode node has no prefill path — prompt positions ride the
/// decode kernels token-by-token (~worst case seconds per request) — so the
/// contract here is stricter than qwen3's: the router appends the prefill
/// peer's first generated token to the prompt, the full original prompt's KV
/// (all full pages plus the partial tail page) must arrive from the peer, and
/// a request whose remote KV never materializes is REJECTED for the router to
/// retry, never silently prefilled locally.
#[derive(Clone, Debug)]
pub struct Glm52VllmCompatOptions {
    /// The `PYTHONHASHSEED` value shared with every vLLM prefill process.
    pub python_hash_seed: String,
    /// The P side's pegaflow-connector namespace (8-hex digest logged by the
    /// connector at startup as `namespace=...`).
    pub namespace: String,
    /// How long a cold request keeps re-querying a zero/partial hit before
    /// giving up on the expected remote KV. Covers the P side's post-response
    /// save + MetaServer-registration tail (tens of ms).
    pub miss_wait: std::time::Duration,
    /// Debug escape hatch: admit with local prompt compute when remote KV
    /// never materializes, instead of rejecting. Leave OFF in production —
    /// on GLM5.2 the fallback rides decode kernels token-by-token.
    pub allow_local_prefill: bool,
}

pub fn launch(model_path: &Path, options: Glm52LaunchOptions) -> Result<EngineHandle> {
    let Glm52LaunchOptions {
        tp_size,
        dp_size,
        dspark_draft_model_path,
        max_model_len,
        no_prefix_cache,
        kv_offload,
        moe_topo,
        dump_graph_png,
        rank_hosts,
    } = options;
    if let Some(path) = &dump_graph_png {
        openinfer_core::cuda_graph::validate_graph_dump_request(path)?;
    }
    match moe_topo {
        Glm52MoeTopo::Tp4 => {
            ensure!(
                tp_size == 4,
                "GLM5.2 TP4 requires --tp-size=4, got {tp_size}"
            );
            ensure!(
                dp_size == 1,
                "GLM5.2 TP4 requires --dp-size=1 (or omitted), got {dp_size}"
            );
        }
        _ => {
            ensure!(
                tp_size == 1,
                "GLM5.2 {moe_topo:?} requires --tp-size=1, got {tp_size}"
            );
            let expected_dp = moe_topo.default_dp_size();
            ensure!(
                dp_size == expected_dp,
                "GLM5.2 {moe_topo:?} requires --dp-size={expected_dp} (or omitted), got {dp_size}"
            );
        }
    }
    // The offload tier extends the prefix cache (restored blocks surface as
    // matched prefix), so a config that disables prefix matching while asking
    // for offload is contradictory — fail loud instead of silently idling an
    // allocated multi-GiB pinned pool.
    ensure!(
        kv_offload.is_none() || (dspark_draft_model_path.is_none() && !no_prefix_cache),
        "GLM5.2 --kv-offload requires the prefix cache: drop --no-prefix-cache and the \
         DSpark drafter (speculative decoding and prefix caching are mutually exclusive)"
    );
    // The tp8 topology mirrors KV on every rank; the host tier's restore leg
    // H2Ds into ONE rank's arena, which would silently desync the other 7.
    ensure!(
        kv_offload.is_none() || moe_topo == Glm52MoeTopo::Ep8,
        "GLM5.2 --kv-offload requires the EP8 topology (tp8 replicates KV on all ranks; \
         a host-tier restore would land on one)"
    );
    // The vLLM prefill peer's KV lives in its pegaflow-server's pool (a
    // separate process even on the same host); without the P2P mesh the
    // compat keys would query an empty local tier and every request would
    // wait out the full miss window.
    ensure!(
        kv_offload
            .as_ref()
            .is_none_or(|kv| kv.vllm_compat.is_none() || kv.p2p.is_some()),
        "GLM5.2 --kv-pd-vllm-seed requires the KV P2P mesh (--kv-p2p-metaserver-addr, \
         --kv-p2p-advertise-addr, --kv-p2p-nics)"
    );
    // The miss window must sit inside the in-flight-fetch ceiling, or the
    // registration phase could never hand over to the fetch phase.
    ensure!(
        kv_offload
            .as_ref()
            .and_then(|kv| kv.vllm_compat.as_ref())
            .is_none_or(|c| c.miss_wait < scheduler::REMOTE_FETCH_DEADLINE),
        "GLM5.2 --kv-pd-miss-wait-ms must stay below the {}s remote-fetch deadline",
        scheduler::REMOTE_FETCH_DEADLINE.as_secs(),
    );
    let remote_ranks: usize = rank_hosts.iter().map(|host| host.ranks).sum();
    if remote_ranks > 0 {
        ensure!(
            moe_topo.uses_ep_expert_bundles(),
            "GLM5.2 --rank-hosts requires an EP topology (tensor-replicated MoE \
             rendezvouses device pointers in-process)"
        );
        ensure!(
            remote_ranks < moe_topo.device_count(),
            "GLM5.2 --rank-hosts claims {remote_ranks} ranks but {moe_topo:?} has only {} \
             (the coordinator keeps at least rank 0 local)",
            moe_topo.device_count()
        );
        // Remote arenas hold device pointers that cannot cross the wire; the
        // host tier would need a per-node offload host + the Event facts
        // plane (cross-node-scaling.md) — not built yet.
        ensure!(
            kv_offload.is_none(),
            "GLM5.2 --kv-offload is not supported with --rank-hosts yet"
        );
    }
    start_engine(
        model_path,
        &Glm52LoadOptions {
            device_ordinals: (0..moe_topo.device_count() - remote_ranks).collect(),
            tp_size,
            dp_size,
            ep_size: moe_topo.expected_ep_size(),
            rank_hosts,
        },
        dspark_draft_model_path.as_deref(),
        max_model_len,
        no_prefix_cache,
        kv_offload,
        moe_topo,
        dump_graph_png,
    )
}

/// Free VRAM held back from the context-cap budget on every rank, covering
/// the post-probe allocations the exact arena ledger does not model: the
/// MLA W_UK/W_UV bf16 dequant during build (~1.1 GiB net over the freed fp8
/// kv_b), DeepEP collective buffers, the 8 whole-step graph instantiations,
/// cuBLAS workspaces, and allocator fragmentation. Measured on 8×H200
/// (jz-38, 2026-07-06): the worst rank's non-arena post-probe allocations
/// came to ~3.05 GiB, so 5 GiB leaves ~2 GiB of post-build headroom over
/// the [`GLM52_POST_BUILD_MIN_FREE_BYTES`] floor; the post-build re-probe
/// below turns any drift into a launch failure instead of a mid-serving
/// OOM.
const GLM52_VRAM_RESERVE_BYTES: usize = 5 << 30;

/// Extra reserve when the DSpark drafter is enabled: the replicated draft
/// weights (~3.8 GiB bf16) plus its dense forward scratch, which load after
/// the probe. The drafter's cap-scaled buffers are in the exact ledger
/// (`glm52_dspark_arena_bytes`), not here.
const GLM52_DSPARK_VRAM_RESERVE_BYTES: usize = 5 << 30;

/// The smallest cap worth serving with (the pre-refactor bring-up value);
/// a budget below this is a misconfiguration, not a working engine.
const GLM52_MIN_MODEL_LEN: usize = 4096;

/// Free VRAM every rank must still have AFTER the model, DeepEP contexts,
/// and the optional drafter are fully resident — headroom for the whole-step
/// graph instantiations (captured lazily by the coordinator) and allocator
/// fragmentation. The post-build re-probe fails launch below this, so a
/// ledger/reserve drift crashes at startup, not mid-serving.
const GLM52_POST_BUILD_MIN_FREE_BYTES: usize = 1 << 30;

/// The launch-time context-cap decision and the numbers behind it — the log
/// line and the tests consume the same values the decision used, so they
/// cannot drift apart.
#[derive(Clone, Copy, Debug)]
struct Glm52ContextBudget {
    max_model_len: usize,
    /// Exact bytes the cap costs a rank (build arenas + drafter lane).
    arena_bytes: usize,
    reserve_bytes: usize,
    budget_bytes: usize,
}

/// Exact cap-scaled bytes a rank allocates for a candidate cap: the build
/// arenas plus, when the drafter is enabled, the DSpark lane.
fn glm52_cap_bytes(max_model_len: usize, dspark_enabled: bool) -> Result<usize> {
    Ok(glm52_arena_bytes(max_model_len)?
        + if dspark_enabled {
            crate::dspark::glm52_dspark_arena_bytes(max_model_len)
        } else {
            0
        })
}

/// Decide the per-request context cap from the post-weight-load VRAM budget.
/// Every slot's cache region is sized `max_model_len` tokens at build, so a
/// candidate cap's cost is exact arithmetic ([`glm52_cap_bytes`]) over the
/// fleet-minimum free bytes — kept free of CUDA so the policy is
/// unit-testable. Auto mode binary-searches the largest aligned cap that
/// fits; an explicit cap must be aligned and fit, or launch fails.
fn derive_max_model_len(
    requested: Option<usize>,
    min_free_vram_bytes: usize,
    dspark_enabled: bool,
) -> Result<Glm52ContextBudget> {
    let reserve_bytes = GLM52_VRAM_RESERVE_BYTES
        + if dspark_enabled {
            GLM52_DSPARK_VRAM_RESERVE_BYTES
        } else {
            0
        };
    let budget_bytes = min_free_vram_bytes.saturating_sub(reserve_bytes);
    let max_model_len = if let Some(requested) = requested {
        ensure!(
            requested >= GLM52_MIN_MODEL_LEN,
            "GLM5.2 --max-model-len {requested} is below the minimum {GLM52_MIN_MODEL_LEN}"
        );
        ensure!(
            requested <= GLM52_MAX_CONTEXT,
            "GLM5.2 --max-model-len {requested} exceeds the checkpoint's \
             max_position_embeddings {GLM52_MAX_CONTEXT}"
        );
        ensure!(
            requested.is_multiple_of(GLM52_MODEL_LEN_ALIGN),
            "GLM5.2 --max-model-len {requested} must be a multiple of {GLM52_MODEL_LEN_ALIGN} \
             (the FlashMLA page size); nearest valid values are {} and {}",
            requested / GLM52_MODEL_LEN_ALIGN * GLM52_MODEL_LEN_ALIGN,
            requested.next_multiple_of(GLM52_MODEL_LEN_ALIGN),
        );
        let required = glm52_cap_bytes(requested, dspark_enabled)?;
        ensure!(
            required <= budget_bytes,
            "GLM5.2 --max-model-len {requested} needs {} of cache per rank but only {} \
             fits (min rank free VRAM {} - reserve {}); lower it or free VRAM",
            ByteSize(required as u64),
            ByteSize(budget_bytes as u64),
            ByteSize(min_free_vram_bytes as u64),
            ByteSize(reserve_bytes as u64),
        );
        requested
    } else {
        // Largest aligned cap whose exact cost fits the budget: the cost is
        // monotone in the cap, so binary search over the aligned candidates.
        let (mut lo, mut hi) = (0, GLM52_MAX_CONTEXT / GLM52_MODEL_LEN_ALIGN);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if glm52_cap_bytes(mid * GLM52_MODEL_LEN_ALIGN, dspark_enabled)? <= budget_bytes {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let derived = lo * GLM52_MODEL_LEN_ALIGN;
        ensure!(
            derived >= GLM52_MIN_MODEL_LEN,
            "GLM5.2 free VRAM leaves a context cap of {derived} (< {GLM52_MIN_MODEL_LEN}): \
             budget {} (min rank free VRAM {} - reserve {})",
            ByteSize(budget_bytes as u64),
            ByteSize(min_free_vram_bytes as u64),
            ByteSize(reserve_bytes as u64),
        );
        derived
    };
    Ok(Glm52ContextBudget {
        max_model_len,
        arena_bytes: glm52_cap_bytes(max_model_len, dspark_enabled)?,
        reserve_bytes,
        budget_bytes,
    })
}

#[derive(Clone, Debug)]
struct Glm52LoadOptions {
    /// Ordinals for the LOCAL ranks (`0..local_count`); remote ranks live on
    /// their rank-hosts' own devices.
    device_ordinals: Vec<usize>,
    tp_size: usize,
    dp_size: usize,
    ep_size: usize,
    rank_hosts: Vec<Glm52RankHostSpec>,
}

#[derive(Debug)]
struct StartupValidation {
    device_ordinals: Vec<usize>,
    rank_hosts: Vec<Glm52RankHostSpec>,
    rank_bundles: Vec<Glm52RankLoadBundle>,
    rank_tensor_counts: Vec<usize>,
    rank_expert_ranges: Vec<std::ops::Range<usize>>,
}

#[derive(Debug)]
/// Per-rank facts gathered while the weights landed (index = rank).
struct GpuWeightLoadReport {
    tensor_counts: Vec<usize>,
    bytes: Vec<usize>,
    free_vram_bytes: Vec<usize>,
}

struct LoadedGlm52Runtime {
    workers: Vec<Glm52Worker>,
    report: GpuWeightLoadReport,
}

fn start_engine(
    model_path: &Path,
    options: &Glm52LoadOptions,
    dspark_path: Option<&Path>,
    requested_max_model_len: Option<usize>,
    no_prefix_cache: bool,
    kv_offload: Option<Glm52KvOffloadOptions>,
    moe_topo: Glm52MoeTopo,
    dump_graph_png: Option<PathBuf>,
) -> Result<EngineHandle> {
    let dspark_enabled = dspark_path.is_some();
    let startup = validate_startup(model_path, options, moe_topo)?;
    let loaded = load_rank_weights_to_gpu(model_path, &startup, moe_topo)?;
    log::info!(
        "GLM5.2 load-weight startup complete: ranks={}, rank_plan_tensors={:?}, rank_gpu_tensors={:?}, rank_gpu_bytes={:?}",
        startup.device_ordinals.len(),
        startup.rank_tensor_counts,
        loaded.report.tensor_counts,
        format_bytes(&loaded.report.bytes),
    );

    let min_free_vram_bytes = loaded
        .report
        .free_vram_bytes
        .iter()
        .copied()
        .min()
        .expect("at least one rank loaded");
    // The q_a|kv_a packed twins allocate during rank-model build, after this
    // probe — charge them to the budget here so the derived cap still leaves
    // the post-build headroom floor.
    let qa_kva_twin_bytes = mla_front::glm52_qa_kva_twin_bytes()?;
    let budget = derive_max_model_len(
        requested_max_model_len,
        min_free_vram_bytes.saturating_sub(qa_kva_twin_bytes),
        dspark_enabled,
    )?;
    let max_model_len = budget.max_model_len;
    log::info!(
        "GLM5.2 max_model_len={max_model_len} ({}): min rank free VRAM {} after weights \
         (qa|kv_a twins {} charged), cap-scaled arenas {} across {} slots{}, reserve {}, \
         budget {}",
        if requested_max_model_len.is_some() {
            "--max-model-len"
        } else {
            "VRAM-derived"
        },
        ByteSize(min_free_vram_bytes as u64),
        ByteSize(qa_kva_twin_bytes as u64),
        ByteSize(budget.arena_bytes as u64),
        model::GLM52_MAX_BATCH_PER_RANK,
        if dspark_enabled {
            " (dspark lane included)"
        } else {
            ""
        },
        ByteSize(budget.reserve_bytes as u64),
        ByteSize(budget.budget_bytes as u64),
    );

    let eos_token_ids = read_eos_token_ids(model_path)?;
    // build_rank_models sends SetupComm, so from inside it the DeepEP
    // contexts exist and their destruction is COLLECTIVE: any startup failure
    // from here on must broadcast Shutdown to every rank BEFORE the workers'
    // sequential Drop joins them one by one (the same teardown contract as
    // the coordinator exit) — otherwise the first dropped worker blocks in
    // the destroy barrier waiting for ranks that were never told to shut
    // down, and the launch error surfaces only after the ~100 s DeepEP
    // device timeout. The TP8 LL rendezvous rejecting a topology (poison
    // pill, NVLink probe) is a real failure landing exactly in this window.
    let rank_arenas =
        match build_rank_models(&loaded.workers, max_model_len, moe_topo, dspark_enabled) {
            Ok(rank_arenas) => rank_arenas,
            Err(err) => {
                for worker in &loaded.workers {
                    let _ = worker.request_shutdown();
                }
                return Err(err);
            }
        };
    let vllm_compat = kv_offload
        .as_ref()
        .and_then(|opts| opts.vllm_compat.clone());
    let post_comm_startup = || -> Result<Option<Vec<OffloadEngine>>> {
        if let Some(dspark_path) = dspark_path {
            load_dspark_drafters(&loaded.workers, dspark_path)?;
        }
        ensure_post_build_headroom(&loaded.workers)?;
        let offload = kv_offload
            .map(|opts| build_offload_engines(&opts, rank_arenas, &startup.device_ordinals))
            .transpose()?;
        Ok(offload)
    };
    let offload = match post_comm_startup() {
        Ok(started) => started,
        Err(err) => {
            for worker in &loaded.workers {
                let _ = worker.request_shutdown();
            }
            return Err(err);
        }
    };
    let logical_ranks = moe_topo.logical_rank_count();
    let kv_total_blocks = glm52_pool_blocks(max_model_len) - 1;
    let (load_txs, load_rxs): (Vec<_>, Vec<_>) = (0..logical_ranks)
        .map(|_| {
            watch::channel(LoadSnapshot {
                kv_total_blocks: kv_total_blocks as u64,
                ..LoadSnapshot::default()
            })
        })
        .unzip();
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let (graph_dump_request, graph_dump_response) = match dump_graph_png {
        Some(path) => {
            let (response_tx, response_rx) = crossbeam_channel::bounded(1);
            (Some((path, response_tx)), Some(response_rx))
        }
        None => (None, None),
    };
    let coord_handle = std::thread::Builder::new()
        .name("glm52-coord".into())
        .spawn(move || {
            scheduler::run_dp8_coordinator(
                submit_rx,
                loaded.workers,
                &eos_token_ids,
                dspark_enabled,
                max_model_len,
                no_prefix_cache,
                offload,
                vllm_compat,
                moe_topo,
                load_txs,
                graph_dump_request,
            );
        })
        .map_err(|err| anyhow::anyhow!("failed to spawn GLM5.2 coordinator: {err}"))?;
    if let Some(response) = graph_dump_response {
        let Ok(dump_result) = response.recv() else {
            drop(submit_tx);
            coord_handle.join().map_err(|_| {
                anyhow::anyhow!("GLM5.2 coordinator panicked before reporting graph export")
            })?;
            return Err(anyhow::anyhow!(
                "GLM5.2 coordinator exited before reporting CUDA Graph export"
            ));
        };
        let summary = match dump_result {
            Ok(summary) => summary,
            Err(err) => {
                drop(submit_tx);
                coord_handle.join().map_err(|_| {
                    anyhow::anyhow!("GLM5.2 coordinator panicked after graph export failure")
                })?;
                return Err(err.context("GLM5.2 CUDA Graph export failed"));
            }
        };
        log::info!(
            "GLM5.2 decode CUDA Graph exported: nodes={}, kernels={}, edges={}, dot={}, png={}",
            summary.nodes,
            summary.kernels,
            summary.edges,
            summary.dot_path.display(),
            summary.png_path.display()
        );
    }
    // Publish the launch-time cap so the frontend clamps its config.json
    // max_position_embeddings (1M) at the API boundary instead of admitting
    // requests the scheduler would reject (same contract as qwen3/dsv2-lite).
    let servable_len = u32::try_from(max_model_len)
        .expect("max_model_len is bounded by GLM52_MAX_CONTEXT and fits u32");
    Ok(EngineHandle::new_with_join_handle(submit_tx, coord_handle)
        .with_servable_len(servable_len)
        .with_kv_capacity(KvCapacity {
            total_blocks: kv_total_blocks,
            block_size: GLM52_MODEL_LEN_ALIGN,
        })
        .with_load_watches(load_rxs))
}

/// Load the DSpark drafter on every rank (rank-local, ~3.8 GB bf16 each —
/// the draft's embed/lm_head reuse the target's, so they are never loaded).
fn load_dspark_drafters(workers: &[Glm52Worker], dspark_path: &Path) -> Result<()> {
    let started = Instant::now();
    let responses = workers
        .iter()
        .map(|worker| worker.load_dspark_async(dspark_path))
        .collect::<Result<Vec<_>>>()?;
    for (rank, response) in responses.into_iter().enumerate() {
        response.recv().map_err(|_| {
            anyhow::anyhow!("GLM5.2 rank {rank} dropped its dspark-load response")
        })??;
    }
    log::info!(
        "GLM5.2 DSpark drafter loaded on all ranks in {:.2}s (speculative decoding: verify \
         spans ride the decode buckets, accept stats logged per request)",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Re-probe every rank once everything the reserve constants stand in for is
/// resident (model arenas, dequanted MLA weights, DeepEP contexts, optional
/// drafter): if any rank is left with less headroom than the whole-step
/// graph instantiations and allocator slack need, fail the launch with the
/// numbers — a reserve/ledger drift must crash here, not as a mid-serving
/// OOM that tears the collective group down.
fn ensure_post_build_headroom(workers: &[Glm52Worker]) -> Result<()> {
    let responses = workers
        .iter()
        .map(Glm52Worker::free_vram_async)
        .collect::<Result<Vec<_>>>()?;
    let mut per_rank = Vec::with_capacity(responses.len());
    for (rank, response) in responses.into_iter().enumerate() {
        let free = response
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its VRAM-probe response"))??;
        ensure!(
            free >= GLM52_POST_BUILD_MIN_FREE_BYTES,
            "GLM5.2 rank {rank} has only {} free VRAM after build (< {} headroom for graph \
             capture); lower --max-model-len or free device memory",
            ByteSize(free as u64),
            ByteSize(GLM52_POST_BUILD_MIN_FREE_BYTES as u64),
        );
        per_rank.push(free);
    }
    log::info!(
        "GLM5.2 post-build free VRAM per rank: {:?}",
        format_bytes(&per_rank)
    );
    Ok(())
}

/// Build every rank's resident model, then create the collective contexts.
/// Two phases on purpose: the build is per-rank and can fail (OOM, packaging
/// drift) — every rank must report success BEFORE anyone enters context
/// creation, or a single failure strands peer ranks in a collective init with
/// no useful error. TP4 currently stops after the per-rank build, before
/// entering any EP8/TP8 collective setup.
fn build_rank_models(
    workers: &[Glm52Worker],
    max_model_len: usize,
    moe_topo: Glm52MoeTopo,
    dspark_enabled: bool,
) -> Result<Vec<Vec<KvArena>>> {
    let build_started = Instant::now();
    let responses = workers
        .iter()
        .map(|worker| worker.build_model_async(max_model_len, moe_topo, dspark_enabled))
        .collect::<Result<Vec<_>>>()?;
    let mut rank_arenas = Vec::with_capacity(responses.len());
    for (rank, response) in responses.into_iter().enumerate() {
        rank_arenas.push(
            response
                .recv()
                .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its build response"))??,
        );
    }
    let unique_id = if moe_topo.uses_ep_expert_bundles() {
        openinfer_kernels::ops::glm52_ep_deepep_unique_id(moe_topo.expected_ep_size())?
    } else {
        // TP allreduce bootstrap just needs one NCCL unique id; ride the EP8
        // shim's generator.
        openinfer_kernels::ops::glm52_ep_deepep_unique_id(8)?
    };
    let tp_exchange = moe_topo
        .uses_tensor_replicated_moe()
        .then(|| std::sync::Arc::new(crate::moe_tp::Glm52TpExchange::new(moe_topo.device_count())));
    let responses = workers
        .iter()
        .map(|worker| worker.setup_comm_async(unique_id, moe_topo, tp_exchange.clone()))
        .collect::<Result<Vec<_>>>()?;
    for (rank, response) in responses.into_iter().enumerate() {
        response
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its comm-setup response"))??;
    }
    log::info!(
        "GLM5.2 rank models built in {:.2}s (weights adopted in place + {:?} contexts up)",
        build_started.elapsed().as_secs_f64(),
        moe_topo
    );
    Ok(rank_arenas)
}

/// One shared pegaflow host (one pinned pool) with each rank's arenas
/// registered as its own instance under a single namespace — replicated
/// non-expert weights make DP ranks' KV interchangeable, so any rank
/// restores what any rank saved.
/// The namespace folds the layout facts that make blocks interchange-safe
/// (per-token packing, page size, layer count); pool capacity deliberately
/// stays out (a block's bytes don't depend on it).
fn build_offload_engines(
    opts: &Glm52KvOffloadOptions,
    rank_arenas: Vec<Vec<KvArena>>,
    device_ordinals: &[usize],
) -> Result<Vec<OffloadEngine>> {
    let mla_page_size = openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
    let mla_bytes_per_token = rank_arenas
        .first()
        .and_then(|arenas| arenas.iter().find(|arena| is_mla_arena_name(&arena.name)))
        .context("GLM5.2 KV offload has no MLA arena")?
        .bytes_per_block
        / mla_page_size;
    ensure!(
        rank_arenas.iter().all(|arenas| arenas
            .iter()
            .filter(|arena| is_mla_arena_name(&arena.name))
            .all(|arena| arena.bytes_per_block == mla_page_size * mla_bytes_per_token)),
        "GLM5.2 KV offload ranks disagree on MLA cache layout"
    );
    let host = OffloadHost::new(HostConfig {
        pinned_pool_bytes: opts.pinned_pool_bytes,
        use_hugepages: opts.use_hugepages,
        runtime_threads: 2,
        p2p: opts
            .p2p
            .as_ref()
            .map(|p2p| openinfer_kv_offload::P2pConfig {
                metaserver_addr: p2p.metaserver_addr.clone(),
                advertise_addr: p2p.advertise_addr.clone(),
                rdma_nics: p2p.rdma_nics.clone(),
            }),
    })
    .map_err(|err| anyhow::anyhow!("GLM5.2 KV offload host: {err}"))?;
    // vLLM-compat mode joins the *P side's* content domain: the pegaflow
    // connector derives an 8-hex namespace from vLLM config (and logs it at
    // startup); reproducing that derivation would mean chasing Python repr
    // of vLLM internals, so the operator passes it through explicitly.
    let namespace = match &opts.vllm_compat {
        Some(compat) => compat.namespace.clone(),
        None => format!(
            "openinfer-glm52-l{GLM52_LAYERS}-p{}-mla{}-idxk{}",
            mla_page_size,
            mla_bytes_per_token,
            config::GLM52_INDEX_HEAD_DIM + 4,
        ),
    };
    // vLLM-compat: the P side's connector stores MLA-model blocks page-first —
    // one host page per block, layers at offsets ordered by lexicographic
    // layer name. Byte-identical interop therefore requires registering under
    // vLLM's own layer names (same sort order ⇒ same page offsets; the
    // per-layer byte widths already match by construction) and page-first.
    let vllm_compat_active = opts.vllm_compat.is_some();
    let engines = rank_arenas
        .into_iter()
        .zip(device_ordinals)
        .enumerate()
        .map(|(rank, (mut arenas, &device_ordinal))| {
            if vllm_compat_active {
                for arena in &mut arenas {
                    arena.name = vllm_arena_name(&arena.name)?;
                }
            }
            OffloadEngine::with_arenas_on(
                std::sync::Arc::clone(&host),
                format!("glm52-rank{rank}"),
                &namespace,
                device_ordinal as i32,
                &arenas,
                vllm_compat_active,
            )
            .map_err(|err| anyhow::anyhow!("GLM5.2 KV offload rank {rank} registration: {err}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let arenas_per_rank = GLM52_LAYERS
        + (0..GLM52_LAYERS)
            .filter(|&layer| config::glm52_layer_has_full_indexer(layer))
            .count();
    log::info!(
        "GLM5.2 KV offload up: {} pinned host pool (hugepages: {}), namespace {namespace}, \
         {} rank instances x {arenas_per_rank} arenas",
        ByteSize(opts.pinned_pool_bytes as u64),
        opts.use_hugepages,
        engines.len(),
    );
    Ok(engines)
}

fn is_mla_arena_name(name: &str) -> bool {
    name.rsplit_once('.')
        .is_some_and(|(_, arena_kind)| arena_kind == "mla")
}

/// Map a native arena name (`glm52.L{n}.mla` / `glm52.L{n}.idxk`) to the name
/// vLLM registers the same cache under (`GlmMoeDsaForCausalLM`, vLLM ≥ 0.24:
/// MLA latent on every layer, indexer K only on full-indexer layers).
fn vllm_arena_name(name: &str) -> Result<String> {
    let parse = || -> Option<(usize, &str)> {
        let rest = name.strip_prefix("glm52.L")?;
        let (layer, kind) = rest.split_once('.')?;
        Some((layer.parse().ok()?, kind))
    };
    match parse() {
        Some((layer, "mla")) => Ok(format!("model.layers.{layer}.self_attn.attn")),
        Some((layer, "idxk")) => Ok(format!("model.layers.{layer}.self_attn.indexer.k_cache")),
        _ => bail!("GLM5.2 arena {name} has no vLLM-compat mapping"),
    }
}

/// EOS ids from the checkpoint's generation_config.json (`eos_token_id` is a
/// number or an array of numbers).
fn read_eos_token_ids(model_path: &Path) -> Result<Vec<u32>> {
    let path = model_path.join("generation_config.json");
    let content = std::fs::read_to_string(&path)
        .map_err(|err| anyhow::anyhow!("read {}: {err}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|err| anyhow::anyhow!("parse {}: {err}", path.display()))?;
    let field = json
        .get("eos_token_id")
        .ok_or_else(|| anyhow::anyhow!("{} missing eos_token_id", path.display()))?;
    let as_u32 = |value: &serde_json::Value| -> Result<u32> {
        value
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| anyhow::anyhow!("eos_token_id entry {value} is not a u32"))
    };
    let ids = match field {
        serde_json::Value::Array(entries) => {
            entries.iter().map(as_u32).collect::<Result<Vec<_>>>()?
        }
        other => vec![as_u32(other)?],
    };
    ensure!(!ids.is_empty(), "eos_token_id list is empty");
    Ok(ids)
}

fn validate_startup(
    model_path: &Path,
    options: &Glm52LoadOptions,
    moe_topo: Glm52MoeTopo,
) -> Result<StartupValidation> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .map_err(|err| anyhow::anyhow!("read {}: {err}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|err| anyhow::anyhow!("parse {}: {err}", config_path.display()))?;
    probe_config_json(&json)?;

    let expected_devices = moe_topo.device_count();
    let remote_ranks: usize = options.rank_hosts.iter().map(|host| host.ranks).sum();
    ensure!(
        options.device_ordinals.len() + remote_ranks == expected_devices,
        "GLM5.2 {moe_topo:?} load requires {expected_devices} ranks, got {} local ({:?}) + \
         {remote_ranks} remote",
        options.device_ordinals.len(),
        options.device_ordinals
    );
    ensure!(
        options.tp_size == moe_topo.expected_tp_size()
            && options.dp_size == moe_topo.default_dp_size()
            && options.ep_size == moe_topo.expected_ep_size(),
        "GLM5.2 {moe_topo:?} requires TP{}/DP{}/EP{}, got TP{} DP{} EP{}",
        moe_topo.expected_tp_size(),
        moe_topo.default_dp_size(),
        moe_topo.expected_ep_size(),
        options.tp_size,
        options.dp_size,
        options.ep_size
    );
    let unique_devices = options
        .device_ordinals
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    ensure!(
        unique_devices.len() == options.device_ordinals.len(),
        "GLM5.2 device ordinals must be unique, got {:?}",
        options.device_ordinals
    );

    let manifest = Glm52WeightManifest::from_model_dir(model_path)?;
    let rank_bundles = manifest.all_rank_load_bundles(moe_topo)?;
    let mut rank_tensor_counts = Vec::with_capacity(rank_bundles.len());
    let mut rank_expert_ranges = Vec::with_capacity(rank_bundles.len());
    for bundle in &rank_bundles {
        rank_tensor_counts.push(bundle.plan.tensor_count);
        rank_expert_ranges.push(bundle.plan.expert_range.clone());
    }

    log::info!(
        "GLM5.2 load-weight startup validated: model_path={}, ranks={}, device_ordinals={:?}, logical_parallel=TP{} DP{} EP{}, rank_expert_ranges={:?}, rank_plan_tensors={:?}",
        model_path.display(),
        rank_bundles.len(),
        options.device_ordinals,
        options.tp_size,
        options.dp_size,
        options.ep_size,
        rank_expert_ranges,
        rank_tensor_counts,
    );

    Ok(StartupValidation {
        device_ordinals: options.device_ordinals.clone(),
        rank_hosts: options.rank_hosts.clone(),
        rank_bundles,
        rank_tensor_counts,
        rank_expert_ranges,
    })
}

fn load_rank_weights_to_gpu(
    model_path: &Path,
    startup: &StartupValidation,
    moe_topo: Glm52MoeTopo,
) -> Result<LoadedGlm52Runtime> {
    let spawn_started = Instant::now();
    log::info!(
        "start spawn GLM5.2 rank workers: ranks={} ({} local + {} remote nodes)",
        startup.rank_bundles.len(),
        startup.device_ordinals.len(),
        startup.rank_hosts.len(),
    );
    let mut workers = Vec::with_capacity(startup.rank_bundles.len());
    for (rank, &device_ordinal) in startup.device_ordinals.iter().enumerate() {
        let placement = Glm52RankPlacement {
            rank,
            device_ordinal,
        };
        workers.push(Glm52Worker::Local(Glm52RankWorker::spawn(
            placement,
            startup.rank_bundles[rank].clone(),
        )?));
    }
    let mut next_rank = startup.device_ordinals.len();
    for host in &startup.rank_hosts {
        let remote =
            Glm52RemoteNode::connect(&host.addr, model_path, moe_topo, next_rank, host.ranks)?;
        next_rank += host.ranks;
        workers.extend(remote.into_iter().map(Glm52Worker::Remote));
    }
    log::info!(
        "spawn GLM5.2 rank workers cost {:.2}s: ranks={}",
        spawn_started.elapsed().as_secs_f64(),
        workers.len()
    );

    let load_started = Instant::now();
    log::info!(
        "start load GLM5.2 rank weights: ranks={}, rank_expert_ranges={:?}",
        workers.len(),
        startup.rank_expert_ranges,
    );
    let load_results = workers
        .iter()
        .map(|worker| worker.load_weights_async(model_path, moe_topo))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(load_results.len());
    for (rank, rx) in load_results.into_iter().enumerate() {
        let report = rx
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} worker dropped load response"))??;
        ensure!(
            report.rank == rank && report.loaded_to_gpu,
            "GLM5.2 rank {rank} invalid weight-load report: {:?}",
            report
        );
        reports.push(report);
    }
    let rank_tensor_counts = reports
        .iter()
        .map(|report| report.loaded_tensor_count)
        .collect::<Vec<_>>();
    let rank_bytes = reports
        .iter()
        .map(|report| report.loaded_total_bytes)
        .collect::<Vec<_>>();
    let rank_free_vram_bytes = reports
        .iter()
        .map(|report| report.free_vram_bytes)
        .collect::<Vec<_>>();
    log::info!(
        "GLM5.2 rank weight load cost {:.2}s: ranks={}, tensors={:?}, resident_bytes={:?}",
        load_started.elapsed().as_secs_f64(),
        reports.len(),
        rank_tensor_counts,
        format_bytes(&rank_bytes),
    );

    Ok(LoadedGlm52Runtime {
        workers,
        report: GpuWeightLoadReport {
            tensor_counts: rank_tensor_counts,
            bytes: rank_bytes,
            free_vram_bytes: rank_free_vram_bytes,
        },
    })
}

fn format_bytes(values: &[usize]) -> Vec<String> {
    values
        .iter()
        .map(|&value| ByteSize(value as u64).to_string())
        .collect()
}

#[cfg(test)]
mod max_model_len_tests {
    use super::*;

    /// Free VRAM that budgets exactly a `cap`-token context (exact ledger +
    /// reserve) — inverted through the same `glm52_cap_bytes` the derivation
    /// uses, so the tests exercise the policy, not a parallel formula.
    fn free_for(cap: usize, dspark: bool) -> usize {
        let reserve = GLM52_VRAM_RESERVE_BYTES
            + if dspark {
                GLM52_DSPARK_VRAM_RESERVE_BYTES
            } else {
                0
            };
        reserve + glm52_cap_bytes(cap, dspark).expect("cap bytes")
    }

    #[test]
    fn derived_cap_is_aligned_and_scales_with_free_vram() {
        let cap = derive_max_model_len(None, free_for(10_048, false), false)
            .expect("derive")
            .max_model_len;
        assert_eq!(cap, 10_048, "exact budget for an aligned cap derives it");
        assert!(cap.is_multiple_of(GLM52_MODEL_LEN_ALIGN));
        let larger = derive_max_model_len(None, free_for(50_048, false), false)
            .expect("derive")
            .max_model_len;
        assert!(larger > cap);
    }

    #[test]
    fn dspark_lane_shrinks_the_derived_cap() {
        let free = free_for(50_048, false);
        let plain = derive_max_model_len(None, free, false).expect("derive");
        let dspark = derive_max_model_len(None, free, true).expect("derive");
        assert!(
            dspark.max_model_len < plain.max_model_len,
            "dspark cap-scaled cost must shrink the cap"
        );
    }

    #[test]
    fn derived_cap_never_exceeds_the_checkpoint_ceiling() {
        let budget = derive_max_model_len(None, usize::MAX / 2, false).expect("derive");
        assert_eq!(budget.max_model_len, GLM52_MAX_CONTEXT);
    }

    #[test]
    fn too_little_vram_fails_instead_of_serving_a_toy_cap() {
        let err = derive_max_model_len(None, free_for(1024, false), false)
            .expect_err("sub-minimum cap must fail");
        assert!(err.to_string().contains("context cap"), "{err}");
    }

    #[test]
    fn unaligned_requested_cap_is_rejected_with_the_nearest_valid_values() {
        let err = derive_max_model_len(Some(5000), free_for(100_032, false), false)
            .expect_err("unaligned cap must fail, not silently round");
        let message = err.to_string();
        assert!(
            message.contains("4992") && message.contains("5056"),
            "{message}"
        );
    }

    #[test]
    fn requested_cap_beyond_the_budget_fails_at_launch() {
        let err = derive_max_model_len(Some(99_968), free_for(10_048, false), false)
            .expect_err("over-budget cap must fail");
        assert!(err.to_string().contains("--max-model-len"), "{err}");
    }

    #[test]
    fn requested_cap_below_the_minimum_fails() {
        derive_max_model_len(Some(1024), free_for(100_032, false), false)
            .expect_err("sub-minimum cap must fail");
    }
}
