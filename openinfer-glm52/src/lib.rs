//! GLM5.2 model crate bring-up surface.
//!
//! This crate currently owns the fail-fast model probing and first-cut launch
//! contract. Runtime execution will land behind the same API so the server
//! already routes GLM5.2 through the normal Qwen-style `EngineHandle` path.

mod config;
// Paged-KV decode geometry (vLLM-parity page table / slot mapping). It is
// parallelism-agnostic and survives the DP8->PP8 pivot unchanged; the PP8
// MLA/indexer/KV decode slice (Slice 3, docs/models/glm52/pp-decode.md) is its
// first consumer, so it is unreferenced until then.
#[allow(dead_code)]
mod decode_meta;
mod pp;
mod runner;
mod weights;

use std::{collections::BTreeSet, path::Path, time::Instant};

use anyhow::{Result, ensure};
use openinfer_core::engine::{EngineHandle, EngineLoadOptions, EpBackend, ModelInfo};
use openinfer_core::parallel::ParallelConfig;
use runner::{Glm52RankPlacement, Glm52RankWorker, run_rejecting_dp_coordinator};
use tokio::sync::mpsc;

pub use config::{
    GLM52_DENSE_LAYERS, GLM52_HIDDEN, GLM52_INDEX_TOPK, GLM52_LAYERS, GLM52_MOE_LAYERS,
    GLM52_ROUTED_EXPERTS, GLM52_TOPK, GLM52_VOCAB, Glm52ParallelShape, load_stop_token_ids,
    probe_config_json,
};
pub use pp::{Glm52PpHopStats, Glm52PpSpineConfig, Glm52PpSpineReport, run_pp_p2p_spine};
use weights::Glm52WeightManifest;

pub fn probe_model(model_path: &Path) -> Result<Option<ModelInfo>> {
    let config_path = model_path.join("config.json");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let json: serde_json::Value = serde_json::from_str(&content)?;
    if json.get("model_type").and_then(serde_json::Value::as_str) != Some("glm_moe_dsa") {
        return Ok(None);
    }
    probe_config_json(&json)?;
    Ok(Some(ModelInfo {
        id: "glm52",
        display_name: "GLM5.2".to_string(),
        model_path: model_path.to_path_buf(),
        max_model_len: json
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok()),
    }))
}

#[derive(Clone, Debug)]
pub struct Glm52LaunchOptions {
    pub tp_size: usize,
    pub dp_size: usize,
    pub ep_backend: EpBackend,
    pub cuda_graph: bool,
}

pub fn launch(model_path: &Path, options: Glm52LaunchOptions) -> Result<EngineHandle> {
    ensure!(
        options.tp_size > 0 && options.dp_size > 0,
        "GLM5.2 --tp-size and --dp-size must be positive"
    );
    let parallel = ParallelConfig::new(options.tp_size, options.dp_size);
    start_engine(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: options.cuda_graph,
            enable_prefill_profile: false,
            device_ordinals: (0..parallel.ep_world()).collect(),
            parallel_config: Some(parallel),
            ep_backend: options.ep_backend,
            seed: 42,
        },
    )
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    let startup = validate_startup(model_path, &options)?;
    let loaded = load_rank_weights_to_gpu(model_path, &startup)?;
    log::info!(
        "GLM5.2 startup validated: shape={:?}, stop_tokens={:?}, rank_plan_tensors={:?}, rank_gpu_tensors={:?}, rank_gpu_bytes={:?}, rank0_header_bytes={}, nextn_tensors={}, cuda_graph={}",
        startup.shape,
        startup.stop_token_ids,
        startup.rank_tensor_counts,
        loaded.report.rank_tensor_counts,
        loaded.report.rank_bytes,
        startup.rank0_header_bytes,
        startup.nextn_tensor_count,
        options.enable_cuda_graph
    );
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let coord_handle = std::thread::Builder::new()
        .name("glm52-dp-coord".into())
        .spawn(move || run_rejecting_dp_coordinator(submit_rx, loaded.workers))
        .map_err(|err| anyhow::anyhow!("failed to spawn GLM5.2 DP coordinator: {err}"))?;
    Ok(EngineHandle::new_with_join_handle(submit_tx, coord_handle))
}

#[derive(Debug)]
struct StartupValidation {
    shape: Glm52ParallelShape,
    stop_token_ids: Vec<u32>,
    device_ordinals: Vec<usize>,
    rank_bundles: Vec<weights::Glm52RankLoadBundle>,
    rank_tensor_counts: Vec<usize>,
    rank0_header_bytes: usize,
    nextn_tensor_count: usize,
}

#[derive(Debug)]
struct GpuWeightLoadReport {
    rank_bytes: Vec<usize>,
    rank_tensor_counts: Vec<usize>,
}

struct LoadedGlm52Runtime {
    workers: Vec<Glm52RankWorker>,
    report: GpuWeightLoadReport,
}

fn validate_startup(model_path: &Path, options: &EngineLoadOptions) -> Result<StartupValidation> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    probe_config_json(&json)?;

    let parallel = options
        .parallel_config
        .unwrap_or_else(|| ParallelConfig::new(1, 8));
    ensure!(
        parallel.tp_world() == 1 && parallel.dp_world() == 8,
        "GLM5.2 first cut supports only TP1/DP8/EP8, got TP{}/DP{}/EP{}",
        parallel.tp_world(),
        parallel.dp_world(),
        parallel.ep_world()
    );
    ensure!(
        options.device_ordinals.len() == parallel.ep_world(),
        "GLM5.2 TP1/DP8/EP8 requires {} devices, got {:?}",
        parallel.ep_world(),
        options.device_ordinals
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
    let manifest = Glm52WeightManifest::from_model_dir(model_path)?
        .with_parallel_shape(shape_from_parallel(parallel)?)?;
    let rank_bundles = manifest.all_rank_load_bundles()?;
    let mut rank_tensor_counts = Vec::with_capacity(parallel.ep_world());
    let mut rank0_header_bytes = 0usize;
    for (rank, bundle) in rank_bundles.iter().enumerate() {
        if rank == 0 {
            let header_stats =
                weights::validate_rank_safetensor_headers(model_path, &bundle.load_plan)?;
            ensure!(
                header_stats.tensor_count == bundle.load_plan.tensor_count,
                "GLM5.2 rank0 header tensor count {} disagrees with load plan {}",
                header_stats.tensor_count,
                bundle.load_plan.tensor_count
            );
            rank0_header_bytes = header_stats.total_bytes;
        }
        rank_tensor_counts.push(bundle.plan.tensor_count);
    }

    Ok(StartupValidation {
        shape: manifest.parallel,
        stop_token_ids: load_stop_token_ids(model_path)?,
        device_ordinals: options.device_ordinals.clone(),
        rank_bundles,
        rank_tensor_counts,
        rank0_header_bytes,
        nextn_tensor_count: manifest.nextn_tensor_count,
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
    log::info!("start load GLM5.2 rank weights: ranks={}", workers.len());
    let load_results = workers
        .iter()
        .map(|worker| worker.load_sliced_weights_async(model_path))
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
    log::info!(
        "GLM5.2 rank weight load cost {:.2}s: ranks={}, tensors={:?}, resident_bytes={:?}, non_expert_fp8_projections={:?}, attention/dense/shared={:?}",
        load_started.elapsed().as_secs_f64(),
        reports.len(),
        reports
            .iter()
            .map(|report| report.tensor_count)
            .collect::<Vec<_>>(),
        reports
            .iter()
            .map(|report| report.total_bytes)
            .collect::<Vec<_>>(),
        reports
            .iter()
            .map(|report| report.non_expert_weight_contract.total_fp8_projections)
            .collect::<Vec<_>>(),
        reports
            .iter()
            .map(|report| {
                let contract = report.non_expert_weight_contract;
                (
                    contract.attention_fp8_projections,
                    contract.dense_fp8_projections,
                    contract.shared_fp8_projections,
                )
            })
            .collect::<Vec<_>>()
    );
    let rank_bytes = reports
        .iter()
        .map(|report| report.total_bytes)
        .collect::<Vec<_>>();
    let rank_tensor_counts = reports
        .iter()
        .map(|report| report.tensor_count)
        .collect::<Vec<_>>();
    Ok(LoadedGlm52Runtime {
        workers,
        report: GpuWeightLoadReport {
            rank_bytes,
            rank_tensor_counts,
        },
    })
}

fn shape_from_parallel(parallel: ParallelConfig) -> Result<Glm52ParallelShape> {
    ensure!(
        parallel.tp_world() == 1 && parallel.dp_world() == 8,
        "GLM5.2 first cut supports only TP1/DP8/EP8, got TP{}/DP{}/EP{}",
        parallel.tp_world(),
        parallel.dp_world(),
        parallel.ep_world()
    );
    Ok(Glm52ParallelShape::tp1_dp8())
}
