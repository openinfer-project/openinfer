//! GLM5.2 model crate bring-up surface.
//!
//! This crate currently owns the fail-fast model probing and first-cut launch
//! contract. Runtime execution will land behind the same API so the server
//! already routes GLM5.2 through the normal Qwen-style `EngineHandle` path.

mod config;
// Decode bookends (Slice 6): token embedding + final RMSNorm + lm_head. The PP
// stage executor (Slice 7) is its first caller, so it is unreferenced until then.
#[allow(dead_code)]
mod bookend;
// Dense-MLP decode forward (Slice 6, layers 0..first_k_dense_replace). The PP
// stage executor (Slice 7) is its first caller, so it is unreferenced until then.
#[allow(dead_code)]
mod dense;
// Paged-KV decode geometry (vLLM-parity page table / slot mapping). It is
// parallelism-agnostic and survives the DP8->PP8 pivot unchanged; the PP8
// MLA/indexer/KV decode slice (Slice 3, docs/models/glm52/pp-decode.md) is its
// first consumer, so it is unreferenced until then.
#[allow(dead_code)]
mod decode_meta;
// bs=1 PP-stage decode forward (Slice 7): composes the oracle-gated bricks into
// the per-stage decode step; the coordinator drives the eight stages serially.
mod decode;
// Single-layer MLA decode forward (Slice 3). Composes the oracle-validated GPU
// ops into one `hidden -> o`; the PP stage executor (Slice 7) is its first
// caller, so it is unreferenced until then.
// Shared fp8 block-scaled projection primitives (MLA, dense MLP, MoE shared expert).
#[allow(dead_code)]
mod fp8;
#[allow(dead_code)]
mod mla_decode;
// Typed per-stage decode model: drains the raw loader output into the brick
// weight structs the forward consumes (Slice 7). First caller is the stage
// executor, so it is unreferenced until then.
#[allow(dead_code)]
mod model;
// Single-layer routed-MoE decode forward (Slice 5). Composes the grouped FP8
// expert GEMM with the route/scatter/combine glue; the PP stage executor (Slice
// 7) is its first caller, so it is unreferenced until then.
#[allow(dead_code)]
mod moe_decode;
mod pp;
mod runner;
mod weights;

use std::{collections::BTreeSet, ops::Range, path::Path, time::Instant};

use anyhow::{Result, ensure};
use openinfer_core::engine::{EngineHandle, EngineLoadOptions, EpBackend, ModelInfo};
use runner::{Glm52StagePlacement, Glm52StageWorker, run_pp_coordinator};
use tokio::sync::mpsc;

pub use config::{
    GLM52_DENSE_LAYERS, GLM52_HIDDEN, GLM52_INDEX_TOPK, GLM52_LAYERS, GLM52_MOE_LAYERS,
    GLM52_ROUTED_EXPERTS, GLM52_TOPK, GLM52_VOCAB, load_stop_token_ids, probe_config_json,
};
pub use pp::{Glm52PpHopStats, Glm52PpSpineConfig, Glm52PpSpineReport, run_pp_p2p_spine};
use weights::Glm52WeightManifest;

/// GLM5.2 runs as 8 pipeline stages, one GPU each (PP8 TP1 EP1).
const GLM52_PP_WORLD: usize = 8;

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

/// Launch surface kept stable for the server CLI. GLM5.2 is hardwired to PP8
/// (8 pipeline stages, one GPU each), so `tp_size` / `dp_size` no longer steer
/// the parallel layout — the stages always map to device ordinals `0..8`.
#[derive(Clone, Debug)]
pub struct Glm52LaunchOptions {
    pub tp_size: usize,
    pub dp_size: usize,
    pub ep_backend: EpBackend,
    pub cuda_graph: bool,
}

pub fn launch(model_path: &Path, options: Glm52LaunchOptions) -> Result<EngineHandle> {
    start_engine(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: options.cuda_graph,
            enable_prefill_profile: false,
            device_ordinals: (0..GLM52_PP_WORLD).collect(),
            parallel_config: None,
            ep_backend: options.ep_backend,
            seed: 42,
        },
    )
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    let startup = validate_startup(model_path, &options)?;
    let loaded = load_stage_weights_to_gpu(model_path, &startup)?;
    log::info!(
        "GLM5.2 startup validated: stages={}, stop_tokens={:?}, stage_layer_ranges={:?}, stage_plan_tensors={:?}, stage_gpu_tensors={:?}, stage_gpu_bytes={:?}, stage0_header_bytes={}, nextn_tensors={}, cuda_graph={}",
        startup.device_ordinals.len(),
        startup.stop_token_ids,
        startup.stage_layer_ranges,
        startup.stage_tensor_counts,
        loaded.report.stage_tensor_counts,
        loaded.report.stage_bytes,
        startup.stage0_header_bytes,
        startup.nextn_tensor_count,
        options.enable_cuda_graph
    );
    let stop_token_ids = startup.stop_token_ids.clone();
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let coord_handle = std::thread::Builder::new()
        .name("glm52-pp-coord".into())
        .spawn(move || run_pp_coordinator(submit_rx, loaded.workers, stop_token_ids))
        .map_err(|err| anyhow::anyhow!("failed to spawn GLM5.2 PP coordinator: {err}"))?;
    Ok(EngineHandle::new_with_join_handle(submit_tx, coord_handle))
}

#[derive(Debug)]
struct StartupValidation {
    stop_token_ids: Vec<u32>,
    device_ordinals: Vec<usize>,
    stage_bundles: Vec<weights::Glm52StageLoadBundle>,
    stage_layer_ranges: Vec<Range<usize>>,
    stage_tensor_counts: Vec<usize>,
    stage0_header_bytes: usize,
    nextn_tensor_count: usize,
}

#[derive(Debug)]
struct GpuWeightLoadReport {
    stage_bytes: Vec<usize>,
    stage_tensor_counts: Vec<usize>,
}

struct LoadedGlm52Runtime {
    workers: Vec<Glm52StageWorker>,
    report: GpuWeightLoadReport,
}

fn validate_startup(model_path: &Path, options: &EngineLoadOptions) -> Result<StartupValidation> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    probe_config_json(&json)?;

    ensure!(
        options.device_ordinals.len() == GLM52_PP_WORLD,
        "GLM5.2 PP{GLM52_PP_WORLD} requires {GLM52_PP_WORLD} devices (one GPU per stage), got {:?}",
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

    let manifest = Glm52WeightManifest::from_model_dir(model_path)?;
    let stage_bundles = manifest.all_stage_load_bundles(GLM52_PP_WORLD)?;
    let stage_layer_ranges = stage_bundles
        .iter()
        .map(|bundle| bundle.plan.layers.clone())
        .collect::<Vec<_>>();
    let mut stage_tensor_counts = Vec::with_capacity(stage_bundles.len());
    let mut stage0_header_bytes = 0usize;
    for (stage, bundle) in stage_bundles.iter().enumerate() {
        if stage == 0 {
            let header_stats =
                weights::validate_stage_safetensor_headers(model_path, &bundle.load_plan)?;
            ensure!(
                header_stats.tensor_count == bundle.load_plan.tensor_count,
                "GLM5.2 stage0 header tensor count {} disagrees with load plan {}",
                header_stats.tensor_count,
                bundle.load_plan.tensor_count
            );
            stage0_header_bytes = header_stats.total_bytes;
        }
        stage_tensor_counts.push(bundle.plan.tensor_count);
    }

    Ok(StartupValidation {
        stop_token_ids: load_stop_token_ids(model_path)?,
        device_ordinals: options.device_ordinals.clone(),
        stage_bundles,
        stage_layer_ranges,
        stage_tensor_counts,
        stage0_header_bytes,
        nextn_tensor_count: manifest.nextn_tensor_count,
    })
}

fn load_stage_weights_to_gpu(
    model_path: &Path,
    startup: &StartupValidation,
) -> Result<LoadedGlm52Runtime> {
    let spawn_started = Instant::now();
    log::info!(
        "start spawn GLM5.2 stage workers: stages={}",
        startup.stage_bundles.len()
    );
    let mut workers = Vec::with_capacity(startup.stage_bundles.len());
    for (stage, bundle) in startup.stage_bundles.iter().enumerate() {
        let placement = Glm52StagePlacement::new(stage, startup.device_ordinals[stage])?;
        workers.push(Glm52StageWorker::spawn(placement, bundle.clone())?);
    }
    log::info!(
        "spawn GLM5.2 stage workers cost {:.2}s: stages={}",
        spawn_started.elapsed().as_secs_f64(),
        workers.len()
    );

    let load_started = Instant::now();
    log::info!("start load GLM5.2 stage weights: stages={}", workers.len());
    let load_results = workers
        .iter()
        .map(|worker| worker.load_sliced_weights_async(model_path))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(load_results.len());
    for (stage, rx) in load_results.into_iter().enumerate() {
        let report = rx
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 stage {stage} worker dropped load response"))??;
        ensure!(
            report.stage == stage && report.loaded_to_gpu,
            "GLM5.2 stage {stage} invalid weight-load report: {:?}",
            report
        );
        reports.push(report);
    }
    log::info!(
        "GLM5.2 stage weight load cost {:.2}s: stages={}, tensors={:?}, resident_bytes={:?}, non_expert_fp8_projections={:?}, attention/dense/shared={:?}",
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
    let stage_bytes = reports
        .iter()
        .map(|report| report.total_bytes)
        .collect::<Vec<_>>();
    let stage_tensor_counts = reports
        .iter()
        .map(|report| report.tensor_count)
        .collect::<Vec<_>>();
    Ok(LoadedGlm52Runtime {
        workers,
        report: GpuWeightLoadReport {
            stage_bytes,
            stage_tensor_counts,
        },
    })
}
