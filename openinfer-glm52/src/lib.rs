//! GLM5.2 DP8/EP8 engine surface.
//!
//! Startup validates the official GLM5.2 FP8 checkpoint layout, loads rank
//! slices to GPU memory (the non-expert stack replicated to every rank,
//! experts placed into their packed layout at H2D time), builds the resident
//! models, and serves greedy generation with one request per rank: every
//! step all 8 ranks run the full model in lock-step and enter the
//! per-MoE-layer DeepEP collectives.

mod bookend;
#[cfg(test)]
mod bookend_oracle_gate;
mod config;
mod dense;
mod fp8;
mod indexer;
#[cfg(test)]
mod indexer_oracle_gate;
#[cfg(test)]
mod indexer_smoke;
mod layer;
#[cfg(test)]
mod layer_ep8_oracle_gate;
#[cfg(test)]
mod layer_oracle_gate;
mod mla_decode;
#[cfg(test)]
mod mla_oracle_gate;
mod model;
mod moe_decode;
mod moe_ep8;
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
}

pub fn launch(model_path: &Path, options: Glm52LaunchOptions) -> Result<EngineHandle> {
    ensure!(
        options.tp_size == 1,
        "GLM5.2 requires --tp-size=1, got {}",
        options.tp_size
    );
    ensure!(
        options.dp_size == GLM52_EP_RANKS,
        "GLM5.2 requires --dp-size={GLM52_EP_RANKS} (or omitted), got {}",
        options.dp_size
    );
    start_engine(
        model_path,
        Glm52LoadOptions {
            device_ordinals: (0..GLM52_EP_RANKS).collect(),
            tp_size: options.tp_size,
            dp_size: options.dp_size,
            ep_size: GLM52_EP_RANKS,
        },
    )
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
struct GpuWeightLoadReport {
    rank_tensor_counts: Vec<usize>,
    rank_bytes: Vec<usize>,
}

struct LoadedGlm52Runtime {
    workers: Vec<Glm52RankWorker>,
    report: GpuWeightLoadReport,
}

fn start_engine(model_path: &Path, options: Glm52LoadOptions) -> Result<EngineHandle> {
    let startup = validate_startup(model_path, &options)?;
    let loaded = load_rank_weights_to_gpu(model_path, &startup)?;
    log::info!(
        "GLM5.2 load-weight startup complete: ranks={}, rank_plan_tensors={:?}, rank_gpu_tensors={:?}, rank_gpu_bytes={:?}",
        startup.device_ordinals.len(),
        startup.rank_tensor_counts,
        loaded.report.rank_tensor_counts,
        format_bytes(&loaded.report.rank_bytes),
    );

    let eos_token_ids = read_eos_token_ids(model_path)?;
    build_rank_models(&loaded.workers)?;
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let coord_handle = std::thread::Builder::new()
        .name("glm52-coord".into())
        .spawn(move || {
            scheduler::run_dp8_coordinator(submit_rx, loaded.workers, &eos_token_ids);
        })
        .map_err(|err| anyhow::anyhow!("failed to spawn GLM5.2 coordinator: {err}"))?;
    Ok(EngineHandle::new_with_join_handle(submit_tx, coord_handle))
}

/// Build every rank's resident model, then create the DeepEP contexts. Two
/// phases on purpose: the build is per-rank and can fail (OOM, packaging
/// drift) — every rank must report success BEFORE anyone enters the
/// collective context creation, or a single failure strands the other seven
/// ranks in NCCL init with no timeout.
fn build_rank_models(workers: &[Glm52RankWorker]) -> Result<()> {
    let build_started = Instant::now();
    let responses = workers
        .iter()
        .map(Glm52RankWorker::build_model_async)
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
            rank_tensor_counts,
            rank_bytes,
        },
    })
}

fn format_bytes(values: &[usize]) -> Vec<String> {
    values
        .iter()
        .map(|&value| ByteSize(value as u64).to_string())
        .collect()
}
