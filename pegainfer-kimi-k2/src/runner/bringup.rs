use std::{
    collections::BTreeSet,
    path::Path,
    sync::{Arc, Barrier},
    thread,
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use bytesize::ByteSize;
use crossbeam_channel::bounded;
use log::{debug, info};
use pegainfer_core::{
    engine::{EngineHandle, EngineLoadOptions, EpBackend, GenerateRequest},
    parallel::ParallelConfig,
};
use tokio::sync::mpsc;

use crate::{
    config::{KimiK2ParallelShape, load_stop_token_ids},
    runner::{
        affinity::pin_scheduler_thread,
        config::KimiK2RunnerConfig,
        executor::{ForwardExecutor, Tp1Dp8ForwardExecutor, Tp8Dp1ForwardExecutor},
        load_balancer::DpLoadBalancer,
        scheduler::{KimiK2Scheduler, dp::DpCoordinator},
        worker::{KimiRankWeightLoadReport, KimiRankWorker, build_placements},
    },
    weights::{KimiRankGpuContext, KimiRankSlicedLoadPlan, ensure_text_only_model_index},
};

pub(crate) fn start_engine(model_path: &Path, options: &EngineLoadOptions) -> Result<EngineHandle> {
    let parallel = resolve_parallel_config(options);
    info!(
        "kimi-k2: resolving engine startup: model_path={}, tp_size={}, dp_size={}, ep_size={}, ep_backend={:?}, devices={:?}",
        model_path.display(),
        parallel.tp_world(),
        parallel.dp_world(),
        parallel.ep_world(),
        options.ep_backend,
        options.device_ordinals
    );
    ensure!(
        options.device_ordinals.len() == parallel.ep_world(),
        "Kimi-K2 {:?} requires {} devices, got {:?}",
        parallel,
        parallel.ep_world(),
        options.device_ordinals
    );

    match (parallel.tp_world(), parallel.dp_world()) {
        (8, 1) => start_engine_tp8_dp1(model_path, options, parallel),
        (1, 8) => start_engine_tp1_dp8(model_path, options, parallel),
        _ => bail!(
            "Kimi-K2 TP{}/DP{} not yet supported (v1: TP8DP1 or TP1DP8)",
            parallel.tp_world(),
            parallel.dp_world()
        ),
    }
}

fn resolve_parallel_config(options: &EngineLoadOptions) -> ParallelConfig {
    options
        .parallel_config
        .unwrap_or_else(|| ParallelConfig::new(8, 1))
}

fn build_runner_config(
    model_path: &Path,
    options: &EngineLoadOptions,
    parallel: ParallelConfig,
    shape: KimiK2ParallelShape,
) -> Result<KimiK2RunnerConfig> {
    let started = Instant::now();
    info!("kimi-k2: start build runner config");
    let mut weight_manifest = ensure_text_only_model_index(model_path)?;
    weight_manifest = weight_manifest.with_parallel_shape(shape)?;
    let placements = build_placements(&options.device_ordinals)?;
    let thread_placement = crate::runner::affinity::KimiRankThreadPlacementPlan::for_devices(
        &options.device_ordinals,
    )?;
    let rank_weight_names = (0..placements.len())
        .map(|rank| weight_manifest.rank_weight_names(rank))
        .collect::<Result<Vec<_>>>()?;
    let rank_sliced_load_plans = (0..placements.len())
        .map(|rank| weight_manifest.rank_sliced_load_plan(rank))
        .collect::<Result<Vec<_>>>()?;
    let pplx_thread_placement = pegainfer_core::cpu_topology::RankThreadPlacementPlan::for_devices(
        &options.device_ordinals,
    )?;
    let config = KimiK2RunnerConfig {
        model_path: model_path.to_path_buf(),
        parallel,
        local_dims: shape.local_dims(),
        rank_weight_names,
        rank_sliced_load_plans,
        placements,
        thread_placement,
        pplx_thread_placement,
        enable_cuda_graph: options.enable_cuda_graph,
        ep_backend: options.ep_backend,
    };
    info!(
        "kimi-k2: build runner config cost {:.2}s: ranks={}",
        started.elapsed().as_secs_f64(),
        config.placements.len()
    );
    debug!(
        "kimi-k2: runner config detail: tensors_per_rank={:?}",
        config
            .rank_sliced_load_plans
            .iter()
            .map(|plan| plan.tensor_count)
            .collect::<Vec<_>>()
    );
    Ok(config)
}

fn start_engine_tp8_dp1(
    model_path: &Path,
    options: &EngineLoadOptions,
    parallel: ParallelConfig,
) -> Result<EngineHandle> {
    info!("kimi-k2: starting TP8/DP1 engine");
    let config = build_runner_config(
        model_path,
        options,
        parallel,
        KimiK2ParallelShape::tp8_ep8(),
    )?;
    let stop_token_ids = load_stop_token_ids(model_path)?;
    let executor = build_tp8_dp1_executor(&config)?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
    let (init_tx, init_rx) = bounded::<Result<()>>(1);
    let scheduler_handle = thread::Builder::new()
        .name("kimi-k2-scheduler".into())
        .spawn(move || {
            pin_scheduler_thread(&config.thread_placement);
            let mut scheduler = match KimiK2Scheduler::new(executor, stop_token_ids) {
                Ok(scheduler) => scheduler,
                Err(err) => {
                    let _ = init_tx.send(Err(err));
                    return;
                }
            };
            let _ = init_tx.send(Ok(()));
            scheduler.run(submit_rx);
        })
        .map_err(|err| anyhow::anyhow!("failed to spawn Kimi-K2 scheduler thread: {err}"))?;
    init_rx
        .recv()
        .map_err(|err| anyhow::anyhow!("Kimi-K2 scheduler init channel closed: {err}"))??;
    Ok(EngineHandle::new_with_join_handle(
        submit_tx,
        scheduler_handle,
    ))
}

fn start_engine_tp1_dp8(
    model_path: &Path,
    options: &EngineLoadOptions,
    parallel: ParallelConfig,
) -> Result<EngineHandle> {
    info!("kimi-k2: starting TP1/DP8 engine");
    ensure!(
        options.ep_backend == EpBackend::Pplx,
        "Kimi-K2 TP1/DP8 requires --ep-backend=pplx"
    );
    let dp_world = parallel.dp_world();
    let config = build_runner_config(
        model_path,
        options,
        parallel,
        KimiK2ParallelShape::tp1_dp8(),
    )?;
    let stop_token_ids = load_stop_token_ids(model_path)?;
    let executors = build_tp1_dp8_executors(&config)?;
    let coordinator = DpCoordinator::new(executors, stop_token_ids, options.seed);
    let lb = DpLoadBalancer::new(dp_world);

    let (submit_tx, submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
    let (init_tx, init_rx) = bounded::<Result<()>>(1);
    let coord_handle = thread::Builder::new()
        .name("kimi-k2-dp-coord".into())
        .spawn(move || {
            let _ = init_tx.send(Ok(()));
            coordinator.run(submit_rx, lb);
        })
        .map_err(|err| anyhow::anyhow!("failed to spawn Kimi-K2 DP coordinator: {err}"))?;
    init_rx
        .recv()
        .map_err(|err| anyhow::anyhow!("Kimi-K2 DP coordinator init failed: {err}"))??;

    info!("kimi-k2: TP1 DP{dp_world} coordinated engine started");
    Ok(EngineHandle::new_with_join_handle(submit_tx, coord_handle))
}

fn build_tp8_dp1_executor(config: &KimiK2RunnerConfig) -> Result<Box<dyn ForwardExecutor + Send>> {
    let started = Instant::now();
    info!("kimi-k2: start build TP8/DP1 executor");
    let workers = spawn_workers(config)?;
    let weight_reports =
        maybe_load_rank_weights(&config.model_path, &config.rank_sliced_load_plans, &workers)?;
    init_tp_nccl(&workers)?;
    if config.ep_backend == EpBackend::Pplx {
        install_pplx_backends(config, &workers)?;
        debug!(
            "kimi-k2: pplx EP backends installed on all {} ranks",
            workers.len()
        );
    }
    let executor: Box<dyn ForwardExecutor + Send> = Box::new(Tp8Dp1ForwardExecutor {
        workers,
        weight_reports,
    });
    info!(
        "kimi-k2: build TP8/DP1 executor cost {:.2}s",
        started.elapsed().as_secs_f64()
    );
    Ok(executor)
}

fn build_tp1_dp8_executors(
    config: &KimiK2RunnerConfig,
) -> Result<Vec<Box<dyn ForwardExecutor + Send>>> {
    let started = Instant::now();
    info!("kimi-k2: start build TP1/DP8 executors");
    let workers = spawn_workers(config)?;
    let weight_reports =
        maybe_load_rank_weights(&config.model_path, &config.rank_sliced_load_plans, &workers)?;
    install_pplx_backends(config, &workers)?;

    let mut executors: Vec<Box<dyn ForwardExecutor + Send>> =
        Vec::with_capacity(config.parallel.dp_world());
    for (worker, weight_report) in workers.into_iter().zip(weight_reports) {
        executors.push(Box::new(Tp1Dp8ForwardExecutor {
            worker,
            weight_report,
        }));
    }
    info!(
        "kimi-k2: build TP1/DP8 executors cost {:.2}s",
        started.elapsed().as_secs_f64()
    );
    Ok(executors)
}

fn maybe_load_rank_weights(
    model_path: &Path,
    load_plans: &[KimiRankSlicedLoadPlan],
    workers: &[KimiRankWorker],
) -> Result<Vec<KimiRankWeightLoadReport>> {
    let started = Instant::now();
    info!("kimi-k2: start load rank weights: ranks={}", workers.len());
    ensure_weight_payload_available(model_path, load_plans)?;
    let receivers = workers
        .iter()
        .map(|worker| worker.load_sliced_weights_async(model_path))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(workers.len());
    for (worker, receiver) in workers.iter().zip(receivers) {
        let rank = worker.placement().rank;
        let report = receiver
            .recv()
            .map_err(|_| {
                anyhow::anyhow!(
                    "Kimi-K2 rank {} dropped weight load response",
                    worker.placement().rank
                )
            })?
            .with_context(|| {
                format!(
                    "Kimi-K2 rank {} sliced weight load failed",
                    worker.placement().rank
                )
            })?;
        debug!(
            "kimi-k2: rank {rank} weights loaded: tensors={}, bytes={}, expert_layers={}",
            report.tensor_count,
            ByteSize(report.total_bytes as u64),
            report.expert_kernel_layers
        );
        reports.push(report);
    }
    info!(
        "kimi-k2: load rank weights cost {:.2}s: ranks={}",
        started.elapsed().as_secs_f64(),
        reports.len()
    );
    Ok(reports)
}

fn spawn_workers(config: &KimiK2RunnerConfig) -> Result<Vec<KimiRankWorker>> {
    let started = Instant::now();
    let n = config.placements.len();
    info!("kimi-k2: start spawn rank workers: ranks={n}");
    ensure!(
        config.rank_weight_names.len() == n && config.rank_sliced_load_plans.len() == n,
        "Kimi-K2 names/sliced counts must match {} placements",
        n
    );
    let contexts = config
        .placements
        .iter()
        .map(|placement| KimiRankGpuContext::new(placement.device_ordinal))
        .collect::<Result<Vec<_>>>()?;
    let collective_barrier = Arc::new(Barrier::new(config.parallel.tp_world()));
    let mut workers = Vec::with_capacity(n);
    for (((&placement, weight_names), sliced_load_plan), ctx) in config
        .placements
        .iter()
        .zip(config.rank_weight_names.iter().cloned())
        .zip(config.rank_sliced_load_plans.iter().cloned())
        .zip(contexts)
    {
        let thread_placement = config.thread_placement.rank(placement.rank)?;
        let worker = KimiRankWorker::spawn(
            placement,
            weight_names,
            sliced_load_plan,
            thread_placement,
            config.local_dims,
            ctx,
            Arc::clone(&collective_barrier),
            config.enable_cuda_graph,
        )?;
        debug_assert_eq!(worker.placement(), placement);
        workers.push(worker);
    }
    info!(
        "kimi-k2: spawn rank workers cost {:.2}s: ranks={}",
        started.elapsed().as_secs_f64(),
        workers.len()
    );
    Ok(workers)
}

fn init_tp_nccl(workers: &[KimiRankWorker]) -> Result<()> {
    let started = Instant::now();
    info!("kimi-k2: start TP NCCL init: ranks={}", workers.len());
    let nccl_id = cudarc::nccl::safe::Id::new()
        .map_err(|err| anyhow::anyhow!("Kimi TP NCCL unique id creation failed: {err:?}"))?;
    let comm_receivers = workers
        .iter()
        .map(|worker| worker.init_tp_comm_async(nccl_id, workers.len()))
        .collect::<Result<Vec<_>>>()?;
    for (rank, receiver) in comm_receivers.into_iter().enumerate() {
        receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("Kimi rank {rank} dropped TP comm init response"))?
            .with_context(|| format!("Kimi rank {rank} TP comm init"))?;
    }
    info!(
        "kimi-k2: TP NCCL init cost {:.2}s: ranks={}",
        started.elapsed().as_secs_f64(),
        workers.len()
    );
    Ok(())
}

fn install_pplx_backends(config: &KimiK2RunnerConfig, workers: &[KimiRankWorker]) -> Result<()> {
    let started = Instant::now();
    info!(
        "kimi-k2: start install PPLX EP backend: ranks={}",
        workers.len()
    );
    let build_started = Instant::now();
    debug!(
        "kimi-k2: start build PPLX EP backends: ranks={}",
        workers.len()
    );
    let ep_shape = pegainfer_comm::bootstrap::EpModelShape {
        n_routed_experts: crate::config::KIMI_K2_ROUTED_EXPERTS,
        n_activated_experts: crate::config::KIMI_K2_TOPK,
        hidden_dim: crate::config::KIMI_K2_HIDDEN,
    };
    let devices: Vec<usize> = config.placements.iter().map(|p| p.device_ordinal).collect();
    let pplx_params = pegainfer_comm::bootstrap::PplxBootstrapParams {
        max_num_tokens: 2048,
        expert_padding: crate::runner::moe_pplx::PPLX_EXPERT_PADDING,
        out_dtype: pegainfer_comm::ScalarType::F32,
        canonicalize_duplicate_sources: config.parallel.tp_world() > 1
            && config.parallel.dp_world() == 1,
        ..pegainfer_comm::bootstrap::PplxBootstrapParams::default()
    };
    let (backends, resources) = pegainfer_comm::bootstrap::build_intra_node_backends(
        ep_shape,
        &devices,
        &config.pplx_thread_placement,
        pplx_params,
    )?;
    std::mem::forget(resources);
    debug!(
        "kimi-k2: build PPLX EP backends cost {:.2}s",
        build_started.elapsed().as_secs_f64()
    );
    let enable_started = Instant::now();
    debug!(
        "kimi-k2: start enable PPLX EP backends: ranks={}",
        workers.len()
    );
    let pplx_receivers = workers
        .iter()
        .zip(backends)
        .map(|(worker, backend)| worker.enable_pplx_async(backend))
        .collect::<Result<Vec<_>>>()?;
    for (rank, receiver) in pplx_receivers.into_iter().enumerate() {
        receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("Kimi rank {rank} dropped PPLX enable response"))?
            .with_context(|| format!("Kimi rank {rank} PPLX EP backend enable"))?;
    }
    debug!(
        "kimi-k2: enable PPLX EP backends cost {:.2}s: ranks={}",
        enable_started.elapsed().as_secs_f64(),
        workers.len()
    );
    info!(
        "kimi-k2: PPLX EP backend install cost {:.2}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn ensure_weight_payload_available(
    model_path: &Path,
    load_plans: &[KimiRankSlicedLoadPlan],
) -> Result<()> {
    let shards = load_plans
        .iter()
        .flat_map(|plan| plan.shards.iter().map(|shard| shard.shard.as_str()))
        .collect::<BTreeSet<_>>();
    let existing = shards
        .iter()
        .filter(|shard| model_path.join(shard).exists())
        .count();
    if existing != shards.len() {
        bail!(
            "Kimi-K2 weight payload under {} is incomplete: found {existing}/{} planned shards",
            model_path.display(),
            shards.len()
        );
    }
    Ok(())
}
