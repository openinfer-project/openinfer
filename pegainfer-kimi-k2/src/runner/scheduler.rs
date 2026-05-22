use std::{
    cmp::Ordering,
    collections::{BTreeSet, VecDeque},
    path::Path,
    sync::{Arc, Barrier, mpsc as std_mpsc},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use pegainfer_core::engine::{
    EngineHandle, EngineLoadOptions, FinishReason, GenerateRequest, TokenEvent,
};
use tokio::sync::mpsc;

use crate::{
    config::{KIMI_K2_DENSE_LAYERS, KIMI_K2_LAYERS, KIMI_K2_MOE_LAYERS, KIMI_K2_VOCAB},
    runner::{
        affinity::pin_scheduler_thread,
        config::KimiK2RunnerConfig,
        worker::{KimiRankWeightLoadReport, KimiRankWorker, build_tp8_ep8_placements},
    },
    weights::{KimiRankGpuContext, KimiRankSlicedLoadPlan, ensure_text_only_model_index},
};

const KIMI_RUNNER_MAX_BATCH: usize = 4;

pub(crate) fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    if options.device_ordinals != (0..8).collect::<Vec<_>>() {
        bail!(
            "Kimi-K2 TP8/EP8 currently requires device_ordinals=0..7, got {:?}",
            options.device_ordinals
        );
    }
    let weight_manifest = ensure_text_only_model_index(model_path)?;
    let placements = build_tp8_ep8_placements(&options.device_ordinals)?;
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
    let runtime_config = KimiK2RunnerConfig {
        model_path: model_path.to_path_buf(),
        weight_manifest,
        rank_weight_plans,
        rank_weight_names,
        rank_shard_plans,
        rank_sliced_load_plans,
        placements,
        thread_placement,
        enable_cuda_graph: options.enable_cuda_graph,
    };

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
        Ok(Self {
            runtime: KimiK2Runtime::spawn(config)?,
        })
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
        let decode_batch_size = reqs.len();
        let mut active = Vec::with_capacity(reqs.len());
        for (slot, req) in reqs.into_iter().enumerate() {
            if let Some(active_req) = self.prefill_request(req, slot, decode_batch_size) {
                active.push(active_req);
            }
        }

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
}

struct KimiK2Runtime {
    config: KimiK2RunnerConfig,
    workers: Vec<KimiRankWorker>,
    rank_weight_reports: Vec<KimiRankWeightLoadReport>,
}

impl KimiK2Runtime {
    fn spawn(config: KimiK2RunnerConfig) -> Result<Self> {
        if config.rank_weight_plans.len() != config.placements.len() {
            bail!(
                "Kimi-K2 rank weight plan count {} must match placements {}",
                config.rank_weight_plans.len(),
                config.placements.len()
            );
        }
        if config.rank_weight_names.len() != config.placements.len() {
            bail!(
                "Kimi-K2 rank weight names count {} must match placements {}",
                config.rank_weight_names.len(),
                config.placements.len()
            );
        }
        if config.rank_shard_plans.len() != config.placements.len() {
            bail!(
                "Kimi-K2 rank shard plan count {} must match placements {}",
                config.rank_shard_plans.len(),
                config.placements.len()
            );
        }
        if config.rank_sliced_load_plans.len() != config.placements.len() {
            bail!(
                "Kimi-K2 rank sliced load plan count {} must match placements {}",
                config.rank_sliced_load_plans.len(),
                config.placements.len()
            );
        }
        let contexts = config
            .placements
            .iter()
            .map(|placement| KimiRankGpuContext::new(placement.device_ordinal))
            .collect::<Result<Vec<_>>>()?;
        let mut workers = Vec::with_capacity(config.placements.len());
        let collective_barrier = Arc::new(Barrier::new(config.placements.len()));
        for (((((&placement, weight_plan), weight_names), shard_plan), sliced_load_plan), ctx) in
            config
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
                ctx,
                Arc::clone(&collective_barrier),
                config.enable_cuda_graph,
            )?;
            debug_assert_eq!(worker.placement(), placement);
            debug_assert_eq!(worker.weight_plan().rank, placement.rank);
            debug_assert_eq!(worker.weight_names().rank, placement.rank);
            debug_assert_eq!(worker.shard_plan().rank, placement.rank);
            debug_assert_eq!(worker.sliced_load_plan().rank, placement.rank);
            debug_assert_eq!(worker.thread_placement().rank, placement.rank);
            workers.push(worker);
        }
        let rank_weight_reports =
            maybe_load_rank_weights(&config.model_path, &config.rank_sliced_load_plans, &workers)?;
        // Temporary NCCL bridge: this communicator is used both for normal
        // TP sums and for Kimi MoE shared/routed combine sums. It is not a
        // PPLX EP backend; production EP dispatch/combine must replace the
        // MoE-side all-reduce call sites explicitly. Initialize each NCCL
        // rank inside its persistent worker thread so the communicator is
        // created under the same CUDA context and stream that will enqueue
        // decode collectives.
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
        Ok(Self {
            config,
            workers,
            rank_weight_reports,
        })
    }

    fn rank_count(&self) -> usize {
        debug_assert_eq!(self.workers.len(), self.config.placements.len());
        debug_assert_eq!(
            self.config.weight_manifest.layers.len(),
            crate::config::KIMI_K2_LAYERS
        );
        self.workers.len()
    }

    fn gpu_weight_ready_rank_count(&self) -> usize {
        self.rank_weight_reports
            .iter()
            .filter(|report| report.loaded_to_gpu && report.typed_view_validated)
            .count()
    }

    fn forward_prompt_next_token_in_slot(
        &self,
        input_ids: Vec<u32>,
        slot: usize,
        decode_batch_size: usize,
    ) -> Result<crate::runner::worker::KimiOneTokenForwardReport> {
        if self.workers.is_empty() {
            bail!("Kimi runtime has no rank workers");
        }
        let responses = self
            .workers
            .iter()
            .map(|worker| {
                worker.forward_prompt_next_token_async(input_ids.clone(), slot, decode_batch_size)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut reports = Vec::with_capacity(responses.len());
        for response in responses {
            reports.push(
                response.recv().map_err(|_| {
                    anyhow::anyhow!("Kimi-K2 rank worker dropped forward response")
                })??,
            );
        }
        let expected_input = input_ids
            .last()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Kimi prompt report validation requires input token"))?;
        for report in &reports {
            validate_one_token_report(report, "prompt", slot, expected_input)?;
        }
        reports
            .into_iter()
            .max_by(|left, right| {
                left.local_top_logit_f32
                    .partial_cmp(&right.local_top_logit_f32)
                    .unwrap_or(Ordering::Equal)
            })
            .ok_or_else(|| anyhow::anyhow!("Kimi runtime produced no rank forward reports"))
    }

    fn forward_decode_batch_next_tokens(
        &self,
        token_ids: Vec<u32>,
        append_positions: Vec<usize>,
        slots: Vec<usize>,
        decode_batch_size: usize,
    ) -> Result<Vec<crate::runner::worker::KimiOneTokenForwardReport>> {
        if self.workers.is_empty() {
            bail!("Kimi runtime has no rank workers");
        }
        if token_ids.is_empty() {
            bail!("Kimi batch decode requires at least one token");
        }
        if token_ids.len() != append_positions.len() || token_ids.len() != slots.len() {
            bail!(
                "Kimi batch decode input mismatch: tokens={}, positions={}, slots={}",
                token_ids.len(),
                append_positions.len(),
                slots.len()
            );
        }
        let responses = self
            .workers
            .iter()
            .map(|worker| {
                worker.forward_decode_batch_next_tokens_async(
                    token_ids.clone(),
                    append_positions.clone(),
                    slots.clone(),
                    decode_batch_size,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut rank_reports = Vec::with_capacity(responses.len());
        for response in responses {
            rank_reports.push(response.recv().map_err(|_| {
                anyhow::anyhow!("Kimi-K2 rank worker dropped batch decode response")
            })??);
        }
        let shard_count = rank_reports.len();
        ensure!(
            shard_count == self.workers.len(),
            "Kimi batch decode received {} rank report sets for {} workers",
            shard_count,
            self.workers.len()
        );
        for (rank_idx, reports) in rank_reports.iter().enumerate() {
            ensure!(
                reports.len() == token_ids.len(),
                "Kimi rank {rank_idx} returned {} decode reports for {} active rows",
                reports.len(),
                token_ids.len()
            );
            for (row, report) in reports.iter().enumerate() {
                validate_one_token_report(report, "decode", slots[row], token_ids[row])?;
            }
        }
        let mut selected = Vec::with_capacity(token_ids.len());
        for row in 0..token_ids.len() {
            let best = rank_reports
                .iter()
                .map(|reports| reports[row].clone())
                .max_by(|left, right| {
                    left.local_top_logit_f32
                        .partial_cmp(&right.local_top_logit_f32)
                        .unwrap_or(Ordering::Equal)
                })
                .ok_or_else(|| anyhow::anyhow!("Kimi runtime produced no report for row {row}"))?;
            selected.push(best);
        }
        Ok(selected)
    }
}

fn validate_one_token_report(
    report: &crate::runner::worker::KimiOneTokenForwardReport,
    phase: &str,
    expected_slot: usize,
    expected_input_token: u32,
) -> Result<()> {
    ensure!(
        report.batch_slot == expected_slot,
        "Kimi {phase} rank {} report slot mismatch: got {}, expected {}",
        report.rank,
        report.batch_slot,
        expected_slot
    );
    ensure!(
        report.input_token_id == expected_input_token,
        "Kimi {phase} rank {} report input token mismatch: got {}, expected {}",
        report.rank,
        report.input_token_id,
        expected_input_token
    );
    ensure!(
        report.vocab_rows > 0 && report.vocab_start + report.vocab_rows <= KIMI_K2_VOCAB,
        "Kimi {phase} rank {} report invalid vocab shard: start={}, rows={}, vocab={}",
        report.rank,
        report.vocab_start,
        report.vocab_rows,
        KIMI_K2_VOCAB
    );
    ensure!(
        (report.local_next_token_id as usize) < report.vocab_rows,
        "Kimi {phase} rank {} local token {} outside shard rows {}",
        report.rank,
        report.local_next_token_id,
        report.vocab_rows
    );
    ensure!(
        report.local_next_token_global_id as usize
            == report.vocab_start + report.local_next_token_id as usize,
        "Kimi {phase} rank {} global token mismatch: got {}, expected {}",
        report.rank,
        report.local_next_token_global_id,
        report.vocab_start + report.local_next_token_id as usize
    );
    ensure!(
        report.local_top_logit_f32.is_finite(),
        "Kimi {phase} rank {} report has non-finite top logit {}",
        report.rank,
        report.local_top_logit_f32
    );
    ensure!(
        report.dense_layers_executed == KIMI_K2_DENSE_LAYERS,
        "Kimi {phase} rank {} dense layer count mismatch: got {}, expected {}",
        report.rank,
        report.dense_layers_executed,
        KIMI_K2_DENSE_LAYERS
    );
    ensure!(
        report.moe_layers_executed == KIMI_K2_MOE_LAYERS,
        "Kimi {phase} rank {} MoE layer count mismatch: got {}, expected {}",
        report.rank,
        report.moe_layers_executed,
        KIMI_K2_MOE_LAYERS
    );
    ensure!(
        report.dense_layers_executed + report.moe_layers_executed == KIMI_K2_LAYERS,
        "Kimi {phase} rank {} executed layer count mismatch: dense {} + moe {} != {}",
        report.rank,
        report.dense_layers_executed,
        report.moe_layers_executed,
        KIMI_K2_LAYERS
    );
    Ok(())
}

impl Drop for KimiK2Runtime {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            let _ = worker.shutdown();
        }
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
    fn placements_require_eight_ranks() {
        let err = build_tp8_ep8_placements(&[0, 1, 2]).unwrap_err();
        assert!(err.to_string().contains("exactly 8"));
    }

    #[test]
    fn placements_map_rank_to_device() {
        let placements = build_tp8_ep8_placements(&(0..8).collect::<Vec<_>>()).unwrap();
        assert_eq!(placements[0].rank, 0);
        assert_eq!(placements[7].device_ordinal, 7);
    }
}
