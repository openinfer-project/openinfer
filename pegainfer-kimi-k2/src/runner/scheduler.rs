use std::{
    collections::VecDeque,
    thread,
    time::{Duration, Instant},
};

pub(in crate::runner) mod dp;
mod lifecycle;

use crate::runner::executor::ForwardExecutor;
use anyhow::{Context, Result};
use lifecycle::schedule_prefill_candidate;
use log::error;
use pegainfer_core::engine::{FinishReason, GenerateRequest, TokenEvent};
use tokio::sync::mpsc;

const KIMI_RUNNER_MAX_BATCH: usize = 64;
// Prompt-len=1 service TTFT is dominated by microbatch stair-stepping. Larger
// row-wise batches have recorded TP8/NCCL trace drift, so this stays tied to
// the performance ledger rather than treated as exact-token parity.
const KIMI_PROMPT_LEN1_PREFILL_MICROBATCH: usize = 64;
const KIMI_PREFILL_BATCH_COALESCE: Duration = Duration::from_millis(100);
const KIMI_PREFILL_BATCH_POLL: Duration = Duration::from_micros(50);

pub(super) struct KimiK2Scheduler {
    executor: Box<dyn ForwardExecutor + Send>,
    stop_token_ids: Vec<u32>,
}

struct ActiveKimiRequest {
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    prompt_len: usize,
    completion_tokens: usize,
    max_tokens: usize,
    ignore_eos: bool,
    last_token: u32,
    slot: usize,
    decode_batch_size: usize,
}

impl KimiK2Scheduler {
    pub(super) fn new(
        executor: Box<dyn ForwardExecutor + Send>,
        stop_token_ids: Vec<u32>,
    ) -> Result<Self> {
        executor
            .ensure_decode_batch(KIMI_RUNNER_MAX_BATCH)
            .with_context(|| {
                format!("Kimi-K2 warm decode arena bs{KIMI_RUNNER_MAX_BATCH} before serving")
            })?;
        let warm_tokens = (0..KIMI_RUNNER_MAX_BATCH)
            .map(|idx| 100 + (idx % 1000) as u32)
            .collect::<Vec<_>>();
        let warm_slots = (0..KIMI_RUNNER_MAX_BATCH).collect::<Vec<_>>();
        let _ = executor
            .forward_prompt_len1_batch(&warm_tokens, &warm_slots, KIMI_RUNNER_MAX_BATCH)
            .with_context(|| {
                format!("Kimi-K2 warm prompt_len1 bs{KIMI_RUNNER_MAX_BATCH} before serving")
            })?;
        Ok(Self {
            executor,
            stop_token_ids,
        })
    }

    pub(in crate::runner) fn run(
        &mut self,
        mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    ) {
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
        if let Err(err) = self.executor.ensure_decode_batch(decode_batch_size) {
            let message = format!(
                "Kimi-K2 decode arena allocation failed for batch size {decode_batch_size} after {}/{} ranks loaded: {err:#}",
                self.executor.gpu_weight_ready_count(),
                self.executor.worker_count()
            );
            error!("kimi-k2: {message}");
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
            let reports = match self.executor.forward_decode_batch(
                &token_ids,
                &append_positions,
                &slots,
                decode_batch_size,
            ) {
                Ok(reports) => reports,
                Err(err) => {
                    let message = format!(
                        "Kimi-K2 batch decode forward failed after {}/{} ranks loaded: {err:#}",
                        self.executor.gpu_weight_ready_count(),
                        self.executor.worker_count()
                    );
                    error!("kimi-k2: {message}");
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
            let mut retire = Vec::new();
            for (idx, report) in reports.into_iter().enumerate() {
                let req = &mut active[idx];
                let token_id = report.local_next_token_global_id;
                req.completion_tokens += 1;
                // EOS outranks the length limit; the stop token itself is not
                // emitted (same contract as the Qwen schedulers).
                if !req.ignore_eos && self.stop_token_ids.contains(&token_id) {
                    let _ = req.token_tx.send(TokenEvent::Finished {
                        finish_reason: FinishReason::Stop,
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.completion_tokens,
                    });
                    retire.push(idx);
                    continue;
                }
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
        let last_token = match self.executor.forward_prefill(
            &req.prompt_tokens,
            slot,
            decode_batch_size,
            0,
        ) {
            Ok(report) => {
                let token_id = report.local_next_token_global_id;
                if !req.params.ignore_eos && self.stop_token_ids.contains(&token_id) {
                    let _ = req.token_tx.send(TokenEvent::Finished {
                        finish_reason: FinishReason::Stop,
                        prompt_tokens: req.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                    return None;
                }
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
                    self.executor.gpu_weight_ready_count(),
                    self.executor.worker_count()
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
            ignore_eos: req.params.ignore_eos,
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
        let reports = match self.executor.forward_prompt_len1_batch(
            &token_ids,
            &slots,
            decode_batch_size,
        ) {
            Ok(reports) => reports,
            Err(err) => {
                let message = format!(
                    "Kimi-K2 prompt_len1 batch forward failed after {}/{} ranks loaded: {err:#}",
                    self.executor.gpu_weight_ready_count(),
                    self.executor.worker_count()
                );
                error!("kimi-k2: {message}");
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
        for ((slot, req), report) in group.into_iter().zip(reports) {
            let token_id = report.local_next_token_global_id;
            if !req.params.ignore_eos && self.stop_token_ids.contains(&token_id) {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                continue;
            }
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
                ignore_eos: req.params.ignore_eos,
                last_token: token_id,
                slot,
                decode_batch_size,
            });
        }
    }
}
