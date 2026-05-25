use std::{
    collections::{BTreeSet, VecDeque},
    path::Path,
    sync::{Arc, Barrier, mpsc as std_mpsc},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use pegainfer_core::{
    engine::{EngineHandle, EngineLoadOptions, FinishReason, GenerateRequest, TokenEvent},
    parallel::ParallelConfig,
};
use tokio::sync::mpsc;

#[cfg(feature = "pplx-ep")]
use crate::runner::{
    engine::DpCoordinator, executor::Tp1Dp8ForwardExecutor, load_balancer::DpLoadBalancer,
};
use crate::{
    config::KimiK2ParallelShape,
    runner::{
        affinity::pin_scheduler_thread,
        config::KimiK2RunnerConfig,
        executor::{ForwardExecutor, Tp8Dp1ForwardExecutor},
        worker::{KimiRankWeightLoadReport, KimiRankWorker, build_placements},
    },
    weights::{KimiRankGpuContext, KimiRankSlicedLoadPlan, ensure_text_only_model_index},
};

const KIMI_RUNNER_MAX_BATCH: usize = 64;
// Prompt-len=1 service TTFT is dominated by microbatch stair-stepping. Larger
// row-wise batches have recorded TP8/NCCL trace drift, so this stays tied to
// the performance ledger rather than treated as exact-token parity.
const KIMI_PROMPT_LEN1_PREFILL_MICROBATCH: usize = 64;
const KIMI_PREFILL_BATCH_COALESCE: Duration = Duration::from_millis(100);
const KIMI_PREFILL_BATCH_POLL: Duration = Duration::from_micros(50);

pub(crate) fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    let parallel = resolve_parallel_config(&options)?;
    ensure!(
        options.device_ordinals.len() == parallel.ep_world,
        "Kimi-K2 {:?} requires {} devices, got {:?}",
        parallel,
        parallel.ep_world,
        options.device_ordinals
    );

    match (parallel.tp_world, parallel.dp_world) {
        (8, 1) => start_engine_tp8_dp1(model_path, options, parallel),
        (1, _) => start_engine_tp1_dp(model_path, options, parallel),
        _ => bail!(
            "Kimi-K2 TP{}/DP{} not yet supported (v1: TP8DP1 or TP1DP8)",
            parallel.tp_world,
            parallel.dp_world
        ),
    }
}

fn resolve_parallel_config(_options: &EngineLoadOptions) -> Result<ParallelConfig> {
    if let Ok(mode) = std::env::var("PEGAINFER_KIMI_PARALLEL") {
        match mode.as_str() {
            "tp8dp1" | "tp8_dp1" => return Ok(ParallelConfig::new(8, 1)),
            "tp1dp8" | "tp1_dp8" => return Ok(ParallelConfig::new(1, 8)),
            other => bail!("PEGAINFER_KIMI_PARALLEL={other}: expected tp8dp1 or tp1dp8"),
        }
    }
    Ok(ParallelConfig::new(8, 1))
}

fn build_runner_config(
    model_path: &Path,
    options: &EngineLoadOptions,
    parallel: ParallelConfig,
    shape: KimiK2ParallelShape,
) -> Result<KimiK2RunnerConfig> {
    let mut weight_manifest = ensure_text_only_model_index(model_path)?;
    weight_manifest = weight_manifest.with_parallel_shape(shape)?;
    let placements = build_placements(&options.device_ordinals)?;
    let thread_placement = crate::runner::affinity::KimiRankThreadPlacementPlan::for_devices(
        &options.device_ordinals,
    )?;
    let rank_weight_plans = (0..placements.len())
        .map(|rank| weight_manifest.rank_plan(rank))
        .collect::<Result<Vec<_>>>()?;
    let rank_weight_names = (0..placements.len())
        .map(|rank| weight_manifest.rank_weight_names(rank))
        .collect::<Result<Vec<_>>>()?;
    let rank_shard_plans = (0..placements.len())
        .map(|rank| weight_manifest.rank_shard_plan(rank))
        .collect::<Result<Vec<_>>>()?;
    let rank_sliced_load_plans = (0..placements.len())
        .map(|rank| weight_manifest.rank_sliced_load_plan(rank))
        .collect::<Result<Vec<_>>>()?;
    #[cfg(feature = "pplx-ep")]
    let pplx_thread_placement = pegainfer_core::cpu_topology::RankThreadPlacementPlan::for_devices(
        &options.device_ordinals,
    )?;
    Ok(KimiK2RunnerConfig {
        model_path: model_path.to_path_buf(),
        parallel,
        local_dims: shape.local_dims(),
        weight_manifest,
        rank_weight_plans,
        rank_weight_names,
        rank_shard_plans,
        rank_sliced_load_plans,
        placements,
        thread_placement,
        #[cfg(feature = "pplx-ep")]
        pplx_thread_placement,
        enable_cuda_graph: options.enable_cuda_graph,
    })
}

fn start_engine_tp8_dp1(
    model_path: &Path,
    options: EngineLoadOptions,
    parallel: ParallelConfig,
) -> Result<EngineHandle> {
    let runtime_config = build_runner_config(
        model_path,
        &options,
        parallel,
        KimiK2ParallelShape::tp8_ep8(),
    )?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
    let (init_tx, init_rx) = std_mpsc::channel::<Result<()>>();
    let scheduler_handle = thread::Builder::new()
        .name("kimi-k2-scheduler".into())
        .spawn(move || {
            pin_scheduler_thread(&runtime_config.thread_placement);
            let mut scheduler = match KimiK2Scheduler::new(runtime_config) {
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

#[cfg(not(feature = "pplx-ep"))]
fn start_engine_tp1_dp(
    _model_path: &Path,
    _options: EngineLoadOptions,
    _parallel: ParallelConfig,
) -> Result<EngineHandle> {
    bail!("Kimi-K2 TP1 DP requires pplx-ep feature (PPLX is the only EP backend for TP1)")
}

#[cfg(feature = "pplx-ep")]
fn start_engine_tp1_dp(
    model_path: &Path,
    options: EngineLoadOptions,
    parallel: ParallelConfig,
) -> Result<EngineHandle> {
    let dp_world = parallel.dp_world;
    let runtime_config = build_runner_config(
        model_path,
        &options,
        parallel,
        KimiK2ParallelShape::tp1_dp8(),
    )?;

    let workers = spawn_workers(&runtime_config)?;
    let weight_reports = maybe_load_rank_weights(
        &runtime_config.model_path,
        &runtime_config.rank_sliced_load_plans,
        &workers,
    )?;
    install_pplx_backends(&runtime_config, &workers)?;

    let mut executors: Vec<Box<dyn ForwardExecutor + Send>> = Vec::with_capacity(dp_world);
    for (worker, weight_report) in workers.into_iter().zip(weight_reports.into_iter()) {
        executors.push(Box::new(Tp1Dp8ForwardExecutor {
            worker,
            weight_report,
        }));
    }

    let coordinator = DpCoordinator::new(executors);
    let lb = DpLoadBalancer::new(dp_world);

    let (submit_tx, submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
    let (init_tx, init_rx) = std_mpsc::channel::<Result<()>>();
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

    eprintln!("kimi-k2: TP1 DP{dp_world} coordinated engine started");
    Ok(EngineHandle::new_with_join_handle(submit_tx, coord_handle))
}

struct KimiK2Scheduler {
    runtime: KimiK2Runtime,
}

struct ActiveKimiRequest {
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    prompt_len: usize,
    completion_tokens: usize,
    max_tokens: usize,
    last_token: u32,
    slot: usize,
    decode_batch_size: usize,
}

impl KimiK2Scheduler {
    fn new(config: KimiK2RunnerConfig) -> Result<Self> {
        let runtime = KimiK2Runtime::spawn(config)?;
        runtime
            .ensure_decode_batch(KIMI_RUNNER_MAX_BATCH)
            .with_context(|| {
                format!("Kimi-K2 warm decode arena bs{KIMI_RUNNER_MAX_BATCH} before serving")
            })?;
        let warm_tokens = (0..KIMI_RUNNER_MAX_BATCH)
            .map(|idx| 100 + (idx % 1000) as u32)
            .collect::<Vec<_>>();
        let warm_slots = (0..KIMI_RUNNER_MAX_BATCH).collect::<Vec<_>>();
        let _ = runtime
            .forward_prompt_len1_batch_next_tokens(warm_tokens, warm_slots, KIMI_RUNNER_MAX_BATCH)
            .with_context(|| {
                format!("Kimi-K2 warm prompt_len1 bs{KIMI_RUNNER_MAX_BATCH} before serving")
            })?;
        Ok(Self { runtime })
    }

    fn run(&mut self, mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>) {
        let mut pending = VecDeque::new();
        loop {
            if pending.is_empty() {
                match submit_rx.blocking_recv() {
                    Some(req) => pending.push_back(req),
                    None => return,
                }
            }

            while let Ok(req) = submit_rx.try_recv() {
                pending.push_back(req);
            }
            let deadline = Instant::now() + KIMI_PREFILL_BATCH_COALESCE;
            while pending.len() < KIMI_RUNNER_MAX_BATCH && Instant::now() < deadline {
                match submit_rx.try_recv() {
                    Ok(req) => pending.push_back(req),
                    Err(mpsc::error::TryRecvError::Empty) => {
                        thread::sleep(KIMI_PREFILL_BATCH_POLL);
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                }
            }

            let mut batch = Vec::with_capacity(KIMI_RUNNER_MAX_BATCH);
            while batch.len() < KIMI_RUNNER_MAX_BATCH {
                let Some(req) = pending.pop_front() else {
                    break;
                };
                batch.push(req);
            }
            if !batch.is_empty() {
                self.handle_request_batch(batch);
            }
        }
    }

    fn handle_request_batch(&mut self, reqs: Vec<GenerateRequest>) {
        let mut prefill_reqs = Vec::with_capacity(reqs.len());
        for req in reqs {
            if let Some(req) = schedule_prefill_candidate(req) {
                prefill_reqs.push(req);
            }
        }
        if prefill_reqs.is_empty() {
            return;
        }

        let decode_batch_size = prefill_reqs.len();
        if let Err(err) = self.runtime.ensure_decode_batch(decode_batch_size) {
            let message = format!(
                "Kimi-K2 decode arena allocation failed for batch size {decode_batch_size} after {}/{} ranks loaded: {err:#}",
                self.runtime.gpu_weight_ready_rank_count(),
                self.runtime.rank_count()
            );
            eprintln!("kimi-k2: {message}");
            for req in prefill_reqs {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
            return;
        }
        let mut active = if prefill_reqs.len() > 1
            && prefill_reqs.iter().all(|req| req.prompt_tokens.len() == 1)
        {
            self.prefill_prompt_len1_batch(prefill_reqs, decode_batch_size)
        } else {
            let mut active = Vec::with_capacity(prefill_reqs.len());
            for (slot, req) in prefill_reqs.into_iter().enumerate() {
                if let Some(active_req) = self.prefill_request(req, slot, decode_batch_size) {
                    active.push(active_req);
                }
            }
            active
        };

        while !active.is_empty() {
            let decode_batch_size = active[0].decode_batch_size;
            debug_assert!(
                active
                    .iter()
                    .all(|req| req.decode_batch_size == decode_batch_size)
            );
            let token_ids = active.iter().map(|req| req.last_token).collect::<Vec<_>>();
            let append_positions = active
                .iter()
                .map(|req| req.prompt_len + req.completion_tokens - 1)
                .collect::<Vec<_>>();
            let slots = active.iter().map(|req| req.slot).collect::<Vec<_>>();
            let reports = match self.runtime.forward_decode_batch_next_tokens(
                token_ids,
                append_positions,
                slots,
                decode_batch_size,
            ) {
                Ok(reports) => reports,
                Err(err) => {
                    let message = format!(
                        "Kimi-K2 batch decode forward failed after {}/{} ranks loaded: {err:#}",
                        self.runtime.gpu_weight_ready_rank_count(),
                        self.runtime.rank_count()
                    );
                    eprintln!("kimi-k2: {message}");
                    for req in active.drain(..) {
                        let _ = req.token_tx.send(TokenEvent::Error {
                            message: message.clone(),
                            prompt_tokens: req.prompt_len,
                            completion_tokens: req.completion_tokens,
                        });
                    }
                    return;
                }
            };
            if reports.len() != active.len() {
                let message = format!(
                    "Kimi-K2 batch decode returned {} reports for {} active requests",
                    reports.len(),
                    active.len()
                );
                for req in active.drain(..) {
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: message.clone(),
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.completion_tokens,
                    });
                }
                return;
            }

            let mut retire = Vec::new();
            for (idx, report) in reports.into_iter().enumerate() {
                let req = &mut active[idx];
                let token_id = report.local_next_token_global_id;
                req.completion_tokens += 1;
                if req
                    .token_tx
                    .send(TokenEvent::Token {
                        id: token_id,
                        logprob: None,
                    })
                    .is_err()
                {
                    retire.push(idx);
                    continue;
                }
                if req.completion_tokens >= req.max_tokens {
                    let _ = req.token_tx.send(TokenEvent::Finished {
                        finish_reason: FinishReason::Length,
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.completion_tokens,
                    });
                    retire.push(idx);
                } else {
                    req.last_token = token_id;
                }
            }
            for idx in retire.into_iter().rev() {
                active.swap_remove(idx);
            }
        }
    }

    fn prefill_request(
        &mut self,
        req: GenerateRequest,
        slot: usize,
        decode_batch_size: usize,
    ) -> Option<ActiveKimiRequest> {
        let completion_tokens = 0usize;
        let last_token = match self.runtime.forward_prompt_next_token_in_slot(
            req.prompt_tokens.clone(),
            slot,
            decode_batch_size,
        ) {
            Ok(report) => {
                let token_id = report.local_next_token_global_id;
                if req
                    .token_tx
                    .send(TokenEvent::Token {
                        id: token_id,
                        logprob: None,
                    })
                    .is_err()
                {
                    return None;
                }
                token_id
            }
            Err(err) => {
                let message = format!(
                    "Kimi-K2 prompt forward failed for slot {slot} after {}/{} ranks loaded: {err:#}",
                    self.runtime.gpu_weight_ready_rank_count(),
                    self.runtime.rank_count()
                );
                let _ = req.token_tx.send(TokenEvent::Error {
                    message,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens,
                });
                return None;
            }
        };
        let completion_tokens = completion_tokens + 1;
        if completion_tokens >= req.max_tokens {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens,
            });
            return None;
        }
        Some(ActiveKimiRequest {
            token_tx: req.token_tx,
            prompt_len: req.prompt_tokens.len(),
            completion_tokens,
            max_tokens: req.max_tokens,
            last_token,
            slot,
            decode_batch_size,
        })
    }

    fn prefill_prompt_len1_batch(
        &mut self,
        prefill_reqs: Vec<GenerateRequest>,
        decode_batch_size: usize,
    ) -> Vec<ActiveKimiRequest> {
        let mut active = Vec::with_capacity(prefill_reqs.len());
        let mut group = Vec::with_capacity(KIMI_PROMPT_LEN1_PREFILL_MICROBATCH);
        for (slot, req) in prefill_reqs.into_iter().enumerate() {
            group.push((slot, req));
            if group.len() == KIMI_PROMPT_LEN1_PREFILL_MICROBATCH {
                self.prefill_prompt_len1_microbatch(
                    std::mem::take(&mut group),
                    decode_batch_size,
                    &mut active,
                );
            }
        }
        if !group.is_empty() {
            self.prefill_prompt_len1_microbatch(group, decode_batch_size, &mut active);
        }
        active
    }

    fn prefill_prompt_len1_microbatch(
        &mut self,
        group: Vec<(usize, GenerateRequest)>,
        decode_batch_size: usize,
        active: &mut Vec<ActiveKimiRequest>,
    ) {
        let token_ids = group
            .iter()
            .map(|(_, req)| req.prompt_tokens[0])
            .collect::<Vec<_>>();
        let slots = group.iter().map(|(slot, _)| *slot).collect::<Vec<_>>();
        let reports = match self.runtime.forward_prompt_len1_batch_next_tokens(
            token_ids,
            slots.clone(),
            decode_batch_size,
        ) {
            Ok(reports) => reports,
            Err(err) => {
                let message = format!(
                    "Kimi-K2 prompt_len1 batch forward failed after {}/{} ranks loaded: {err:#}",
                    self.runtime.gpu_weight_ready_rank_count(),
                    self.runtime.rank_count()
                );
                eprintln!("kimi-k2: {message}");
                for (_, req) in group {
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: message.clone(),
                        prompt_tokens: req.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                }
                return;
            }
        };
        if reports.len() != group.len() {
            let message = format!(
                "Kimi-K2 prompt_len1 batch returned {} reports for {} requests",
                reports.len(),
                group.len()
            );
            for (_, req) in group {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
            return;
        }

        for ((slot, req), report) in group.into_iter().zip(reports.into_iter()) {
            let token_id = report.local_next_token_global_id;
            if req
                .token_tx
                .send(TokenEvent::Token {
                    id: token_id,
                    logprob: None,
                })
                .is_err()
            {
                continue;
            }
            let completion_tokens = 1usize;
            if completion_tokens >= req.max_tokens {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Length,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens,
                });
                continue;
            }
            active.push(ActiveKimiRequest {
                token_tx: req.token_tx,
                prompt_len: req.prompt_tokens.len(),
                completion_tokens,
                max_tokens: req.max_tokens,
                last_token: token_id,
                slot,
                decode_batch_size,
            });
        }
    }
}

fn schedule_prefill_candidate(req: GenerateRequest) -> Option<GenerateRequest> {
    let scheduled_at = unix_now_s();
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s: req.queued_at_unix_s.unwrap_or(scheduled_at),
        scheduled_at_unix_s: scheduled_at,
        prompt_tokens: req.prompt_tokens.len(),
    });
    if req.max_tokens == 0 {
        let _ = req.token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
        return None;
    }
    if req.prompt_tokens.is_empty() {
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message: "Kimi-K2 forward requires at least one prompt token".to_string(),
            prompt_tokens: 0,
            completion_tokens: 0,
        });
        return None;
    }
    Some(req)
}

struct KimiK2Runtime {
    executor: Tp8Dp1ForwardExecutor,
}

impl KimiK2Runtime {
    fn spawn(config: KimiK2RunnerConfig) -> Result<Self> {
        let workers = spawn_workers(&config)?;
        let rank_weight_reports =
            maybe_load_rank_weights(&config.model_path, &config.rank_sliced_load_plans, &workers)?;
        init_tp_nccl(&workers)?;

        #[cfg(feature = "pplx-ep")]
        {
            install_pplx_backends(&config, &workers)?;
            eprintln!(
                "kimi-k2: pplx EP backends installed on all {} ranks",
                workers.len()
            );
        }

        let executor = Tp8Dp1ForwardExecutor {
            workers,
            weight_reports: rank_weight_reports,
        };
        let _ = config;
        Ok(Self { executor })
    }

    fn rank_count(&self) -> usize {
        self.executor.worker_count()
    }

    fn gpu_weight_ready_rank_count(&self) -> usize {
        self.executor.gpu_weight_ready_count()
    }

    fn ensure_decode_batch(&self, decode_batch_size: usize) -> Result<()> {
        self.executor.ensure_decode_batch(decode_batch_size)
    }

    fn forward_prompt_next_token_in_slot(
        &self,
        input_ids: Vec<u32>,
        slot: usize,
        decode_batch_size: usize,
    ) -> Result<crate::runner::worker::KimiOneTokenForwardReport> {
        self.executor
            .forward_prefill(&input_ids, slot, decode_batch_size, 0)
    }

    fn forward_prompt_len1_batch_next_tokens(
        &self,
        token_ids: Vec<u32>,
        slots: Vec<usize>,
        decode_batch_size: usize,
    ) -> Result<Vec<crate::runner::worker::KimiOneTokenForwardReport>> {
        self.executor
            .forward_prompt_len1_batch(&token_ids, &slots, decode_batch_size)
    }

    fn forward_decode_batch_next_tokens(
        &self,
        token_ids: Vec<u32>,
        append_positions: Vec<usize>,
        slots: Vec<usize>,
        decode_batch_size: usize,
    ) -> Result<Vec<crate::runner::worker::KimiOneTokenForwardReport>> {
        self.executor
            .forward_decode_batch(&token_ids, &append_positions, &slots, decode_batch_size)
    }
}

fn unix_now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |duration| duration.as_secs_f64())
}

fn maybe_load_rank_weights(
    model_path: &Path,
    load_plans: &[KimiRankSlicedLoadPlan],
    workers: &[KimiRankWorker],
) -> Result<Vec<KimiRankWeightLoadReport>> {
    ensure_weight_payload_available(model_path, load_plans)?;
    let receivers = workers
        .iter()
        .map(|worker| worker.load_sliced_weights_async(model_path))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(workers.len());
    for (worker, receiver) in workers.iter().zip(receivers) {
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
        reports.push(report);
    }
    Ok(reports)
}

fn spawn_workers(config: &KimiK2RunnerConfig) -> Result<Vec<KimiRankWorker>> {
    let n = config.placements.len();
    ensure!(
        config.rank_weight_plans.len() == n
            && config.rank_weight_names.len() == n
            && config.rank_shard_plans.len() == n
            && config.rank_sliced_load_plans.len() == n,
        "Kimi-K2 plan/names/shard/sliced counts must match {} placements",
        n
    );
    let contexts = config
        .placements
        .iter()
        .map(|placement| KimiRankGpuContext::new(placement.device_ordinal))
        .collect::<Result<Vec<_>>>()?;
    let collective_barrier = Arc::new(Barrier::new(config.parallel.tp_world));
    let mut workers = Vec::with_capacity(n);
    for (((((&placement, weight_plan), weight_names), shard_plan), sliced_load_plan), ctx) in config
        .placements
        .iter()
        .zip(config.rank_weight_plans.iter().cloned())
        .zip(config.rank_weight_names.iter().cloned())
        .zip(config.rank_shard_plans.iter().cloned())
        .zip(config.rank_sliced_load_plans.iter().cloned())
        .zip(contexts.into_iter())
    {
        let thread_placement = config.thread_placement.rank(placement.rank)?;
        let worker = KimiRankWorker::spawn(
            placement,
            weight_plan,
            weight_names,
            shard_plan,
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
    Ok(workers)
}

fn init_tp_nccl(workers: &[KimiRankWorker]) -> Result<()> {
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
    Ok(())
}

#[cfg(feature = "pplx-ep")]
fn install_pplx_backends(config: &KimiK2RunnerConfig, workers: &[KimiRankWorker]) -> Result<()> {
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
        canonicalize_duplicate_sources: config.parallel.tp_world > 1
            && config.parallel.dp_world == 1,
        ..pegainfer_comm::bootstrap::PplxBootstrapParams::default()
    };
    let (backends, resources) = pegainfer_comm::bootstrap::build_intra_node_backends(
        ep_shape,
        &devices,
        &config.pplx_thread_placement,
        pplx_params,
    )?;
    std::mem::forget(resources);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placements_map_rank_to_device() {
        let placements = build_placements(&(0..8).collect::<Vec<_>>()).unwrap();
        assert_eq!(placements[0].rank, 0);
        assert_eq!(placements[7].device_ordinal, 7);
    }
}
