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
mod fp8;
mod indexer;
#[cfg(test)]
mod indexer_smoke;
mod layer;
mod mla_decode;
mod model;
mod moe_decode;
mod moe_ep8;
#[cfg(test)]
mod oracle;
mod rows;
mod runner;
mod scheduler;
mod scratch;
mod weights;

use std::{collections::BTreeSet, path::Path, time::Instant};

use anyhow::{Result, ensure};
use bytesize::ByteSize;
use openinfer_core::engine::EngineHandle;
use runner::{Glm52RankPlacement, Glm52RankWorker};
use tokio::sync::mpsc;
use weights::{GLM52_EP_RANKS, Glm52RankLoadBundle, Glm52WeightManifest};

use crate::config::GLM52_MAX_CONTEXT;
use crate::model::{GLM52_MODEL_LEN_ALIGN, glm52_arena_bytes};

pub use config::{
    GLM52_DENSE_LAYERS, GLM52_HIDDEN, GLM52_INDEX_TOPK, GLM52_LAYERS, GLM52_MOE_LAYERS,
    GLM52_ROUTED_EXPERTS, GLM52_TOPK, GLM52_VOCAB, probe_config_json,
};

/// GLM5.2 parallel shape: TP1/DP8/EP8 is the only supported layout — every
/// rank holds the full non-expert stack plus its 32 routed experts, and
/// serves one request at a time.
#[derive(Clone, Debug)]
pub struct Glm52LaunchOptions {
    pub tp_size: usize,
    pub dp_size: usize,
    /// DSpark drafter checkpoint dir (`RedHatAI/GLM-5.2-speculator.dspark`).
    /// Enables greedy speculative decoding: verify spans ride the decode
    /// buckets, accepted tokens commit in batches, per-request accept stats
    /// are logged on release.
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
}

pub fn launch(model_path: &Path, options: Glm52LaunchOptions) -> Result<EngineHandle> {
    let Glm52LaunchOptions {
        tp_size,
        dp_size,
        dspark_draft_model_path,
        max_model_len,
        no_prefix_cache,
    } = options;
    ensure!(tp_size == 1, "GLM5.2 requires --tp-size=1, got {tp_size}");
    ensure!(
        dp_size == GLM52_EP_RANKS,
        "GLM5.2 requires --dp-size={GLM52_EP_RANKS} (or omitted), got {dp_size}"
    );
    start_engine(
        model_path,
        &Glm52LoadOptions {
            device_ordinals: (0..GLM52_EP_RANKS).collect(),
            tp_size,
            dp_size,
            ep_size: GLM52_EP_RANKS,
        },
        dspark_draft_model_path.as_deref(),
        max_model_len,
        no_prefix_cache,
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
    device_ordinals: Vec<usize>,
    tp_size: usize,
    dp_size: usize,
    ep_size: usize,
}

#[derive(Debug)]
struct StartupValidation {
    device_ordinals: Vec<usize>,
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
    workers: Vec<Glm52RankWorker>,
    report: GpuWeightLoadReport,
}

fn start_engine(
    model_path: &Path,
    options: &Glm52LoadOptions,
    dspark_path: Option<&Path>,
    requested_max_model_len: Option<usize>,
    no_prefix_cache: bool,
) -> Result<EngineHandle> {
    let startup = validate_startup(model_path, options)?;
    let loaded = load_rank_weights_to_gpu(model_path, &startup)?;
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
    let budget = derive_max_model_len(
        requested_max_model_len,
        min_free_vram_bytes,
        dspark_path.is_some(),
    )?;
    let max_model_len = budget.max_model_len;
    log::info!(
        "GLM5.2 max_model_len={max_model_len} ({}): min rank free VRAM {} after weights, \
         cap-scaled arenas {} across {} slots{}, reserve {}, budget {}",
        if requested_max_model_len.is_some() {
            "--max-model-len"
        } else {
            "VRAM-derived"
        },
        ByteSize(min_free_vram_bytes as u64),
        ByteSize(budget.arena_bytes as u64),
        model::GLM52_MAX_BATCH_PER_RANK,
        if dspark_path.is_some() {
            " (dspark lane included)"
        } else {
            ""
        },
        ByteSize(budget.reserve_bytes as u64),
        ByteSize(budget.budget_bytes as u64),
    );

    let eos_token_ids = read_eos_token_ids(model_path)?;
    build_rank_models(&loaded.workers, max_model_len)?;
    // From here the DeepEP contexts exist and their destruction is COLLECTIVE:
    // a startup failure must broadcast Shutdown to every rank BEFORE the
    // workers' sequential Drop joins them one by one (the same teardown
    // contract as the coordinator exit) — otherwise the first dropped worker
    // blocks in the destroy barrier waiting for ranks that were never told to
    // shut down, and the launch error surfaces only after the ~100 s DeepEP
    // device timeout.
    let post_comm_startup = || -> Result<bool> {
        let dspark_enabled = if let Some(dspark_path) = dspark_path {
            load_dspark_drafters(&loaded.workers, dspark_path)?;
            true
        } else {
            false
        };
        ensure_post_build_headroom(&loaded.workers)?;
        Ok(dspark_enabled)
    };
    let dspark_enabled = match post_comm_startup() {
        Ok(dspark_enabled) => dspark_enabled,
        Err(err) => {
            for worker in &loaded.workers {
                let _ = worker.request_shutdown();
            }
            return Err(err);
        }
    };
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
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
            );
        })
        .map_err(|err| anyhow::anyhow!("failed to spawn GLM5.2 coordinator: {err}"))?;
    // Publish the launch-time cap so the frontend clamps its config.json
    // max_position_embeddings (1M) at the API boundary instead of admitting
    // requests the scheduler would reject (same contract as qwen3/dsv2-lite).
    let servable_len = u32::try_from(max_model_len)
        .expect("max_model_len is bounded by GLM52_MAX_CONTEXT and fits u32");
    Ok(EngineHandle::new_with_join_handle(submit_tx, coord_handle).with_servable_len(servable_len))
}

/// Load the DSpark drafter on every rank (rank-local, ~3.8 GB bf16 each —
/// the draft's embed/lm_head reuse the target's, so they are never loaded).
fn load_dspark_drafters(workers: &[Glm52RankWorker], dspark_path: &Path) -> Result<()> {
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
fn ensure_post_build_headroom(workers: &[Glm52RankWorker]) -> Result<()> {
    let responses = workers
        .iter()
        .map(Glm52RankWorker::free_vram_async)
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

/// Build every rank's resident model, then create the DeepEP contexts. Two
/// phases on purpose: the build is per-rank and can fail (OOM, packaging
/// drift) — every rank must report success BEFORE anyone enters the
/// collective context creation, or a single failure strands the other seven
/// ranks in NCCL init with no timeout.
fn build_rank_models(workers: &[Glm52RankWorker], max_model_len: usize) -> Result<()> {
    let build_started = Instant::now();
    let responses = workers
        .iter()
        .map(|worker| worker.build_model_async(max_model_len))
        .collect::<Result<Vec<_>>>()?;
    for (rank, response) in responses.into_iter().enumerate() {
        response
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its build response"))??;
    }

    let unique_id = openinfer_kernels::ops::glm52_deepep_unique_id()?;
    let responses = workers
        .iter()
        .map(|worker| worker.setup_comm_async(unique_id))
        .collect::<Result<Vec<_>>>()?;
    for (rank, response) in responses.into_iter().enumerate() {
        response
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its comm-setup response"))??;
    }
    log::info!(
        "GLM5.2 rank models built in {:.2}s (weights adopted in place + DeepEP contexts up)",
        build_started.elapsed().as_secs_f64()
    );
    Ok(())
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

fn validate_startup(model_path: &Path, options: &Glm52LoadOptions) -> Result<StartupValidation> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .map_err(|err| anyhow::anyhow!("read {}: {err}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|err| anyhow::anyhow!("parse {}: {err}", config_path.display()))?;
    probe_config_json(&json)?;

    ensure!(
        options.device_ordinals.len() == GLM52_EP_RANKS,
        "GLM5.2 EP8 load requires {GLM52_EP_RANKS} devices, got {:?}",
        options.device_ordinals
    );
    ensure!(
        options.tp_size == 1
            && options.dp_size == GLM52_EP_RANKS
            && options.ep_size == GLM52_EP_RANKS,
        "GLM5.2 requires TP1/DP8/EP8, got TP{} DP{} EP{}",
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
    let rank_bundles = manifest.all_rank_load_bundles()?;
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
        rank_bundles,
        rank_tensor_counts,
        rank_expert_ranges,
    })
}

fn load_rank_weights_to_gpu(
    model_path: &Path,
    startup: &StartupValidation,
) -> Result<LoadedGlm52Runtime> {
    let spawn_started = Instant::now();
    log::info!(
        "start spawn GLM5.2 rank workers: ranks={}",
        startup.rank_bundles.len()
    );
    let mut workers = Vec::with_capacity(startup.rank_bundles.len());
    for (rank, bundle) in startup.rank_bundles.iter().enumerate() {
        let placement = Glm52RankPlacement::new(rank, startup.device_ordinals[rank])?;
        workers.push(Glm52RankWorker::spawn(placement, bundle.clone())?);
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
        .map(|worker| worker.load_weights_async(model_path))
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
