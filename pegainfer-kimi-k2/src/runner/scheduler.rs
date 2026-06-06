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
const KIMI_DECODE_ADMISSION_MICROBATCH: usize = 64;
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
    logprobs: usize,
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
        let warm_positions = vec![0; KIMI_RUNNER_MAX_BATCH];
        let warm_slots = (0..KIMI_RUNNER_MAX_BATCH).collect::<Vec<_>>();
        let warm_logprobs = vec![0; KIMI_RUNNER_MAX_BATCH];
        let _ = executor
            .forward_decode_batch(
                &warm_tokens,
                &warm_positions,
                &warm_slots,
                KIMI_RUNNER_MAX_BATCH,
                &warm_logprobs,
            )
            .with_context(|| {
                format!("Kimi-K2 warm decode admission bs{KIMI_RUNNER_MAX_BATCH} before serving")
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
        let mut active = Vec::with_capacity(prefill_reqs.len());
        let mut decode_admissions = Vec::with_capacity(KIMI_DECODE_ADMISSION_MICROBATCH);
        for (slot, req) in prefill_reqs.into_iter().enumerate() {
            if req.prompt_tokens.len() == 1 {
                decode_admissions.push((slot, req));
                if decode_admissions.len() == KIMI_DECODE_ADMISSION_MICROBATCH {
                    self.decode_admission_microbatch(
                        std::mem::take(&mut decode_admissions),
                        decode_batch_size,
                        &mut active,
                    );
                }
                continue;
            }
            if !decode_admissions.is_empty() {
                self.decode_admission_microbatch(
                    std::mem::take(&mut decode_admissions),
                    decode_batch_size,
                    &mut active,
                );
            }
            if let Some(active_req) = self.prefill_request(req, slot, decode_batch_size) {
                active.push(active_req);
            }
        }
        if !decode_admissions.is_empty() {
            self.decode_admission_microbatch(decode_admissions, decode_batch_size, &mut active);
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
            let logprobs = active.iter().map(|req| req.logprobs).collect::<Vec<_>>();
            let reports = match self.executor.forward_decode_batch(
                &token_ids,
                &append_positions,
                &slots,
                decode_batch_size,
                &logprobs,
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
                        logprob: report.logprob,
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
            req.logprobs,
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
                        logprob: report.logprob,
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
            logprobs: req.logprobs,
        })
    }

    fn decode_admission_microbatch(
        &mut self,
        group: Vec<(usize, GenerateRequest)>,
        decode_batch_size: usize,
        active: &mut Vec<ActiveKimiRequest>,
    ) {
        let token_ids = group
            .iter()
            .map(|(_, req)| req.prompt_tokens[0])
            .collect::<Vec<_>>();
        let append_positions = vec![0; token_ids.len()];
        let slots = group.iter().map(|(slot, _)| *slot).collect::<Vec<_>>();
        let logprobs = group
            .iter()
            .map(|(_, req)| req.logprobs)
            .collect::<Vec<_>>();
        let reports = match self.executor.forward_decode_batch(
            &token_ids,
            &append_positions,
            &slots,
            decode_batch_size,
            &logprobs,
        ) {
            Ok(reports) => reports,
            Err(err) => {
                let message = format!(
                    "Kimi-K2 decode admission forward failed after {}/{} ranks loaded: {err:#}",
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
                    logprob: report.logprob,
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
                logprobs: req.logprobs,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use pegainfer_core::sampler::SamplingParams;

    use crate::runner::worker::KimiOneTokenForwardReport;

    use super::*;

    #[derive(Debug, Eq, PartialEq)]
    enum ForwardCall {
        EnsureDecodeBatch(usize),
        Prefill {
            input_ids: Vec<u32>,
            slot: usize,
            decode_batch_size: usize,
        },
        Decode {
            token_ids: Vec<u32>,
            append_positions: Vec<usize>,
            slots: Vec<usize>,
            decode_batch_size: usize,
        },
    }

    struct RecordingExecutor {
        calls: Arc<Mutex<Vec<ForwardCall>>>,
    }

    impl RecordingExecutor {
        fn new(calls: Arc<Mutex<Vec<ForwardCall>>>) -> Self {
            Self { calls }
        }
    }

    impl ForwardExecutor for RecordingExecutor {
        fn ensure_decode_batch(&self, decode_batch_size: usize) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(ForwardCall::EnsureDecodeBatch(decode_batch_size));
            Ok(())
        }

        fn forward_prefill(
            &self,
            input_ids: &[u32],
            slot: usize,
            decode_batch_size: usize,
            _ep_max_seq_len: usize,
            _logprobs: usize,
        ) -> Result<KimiOneTokenForwardReport> {
            self.calls.lock().unwrap().push(ForwardCall::Prefill {
                input_ids: input_ids.to_vec(),
                slot,
                decode_batch_size,
            });
            Ok(report(slot, *input_ids.last().unwrap(), 1000 + slot as u32))
        }

        fn forward_decode_batch(
            &self,
            token_ids: &[u32],
            append_positions: &[usize],
            slots: &[usize],
            decode_batch_size: usize,
            _logprobs: &[usize],
        ) -> Result<Vec<KimiOneTokenForwardReport>> {
            self.calls.lock().unwrap().push(ForwardCall::Decode {
                token_ids: token_ids.to_vec(),
                append_positions: append_positions.to_vec(),
                slots: slots.to_vec(),
                decode_batch_size,
            });
            Ok(token_ids
                .iter()
                .zip(slots)
                .enumerate()
                .map(|(row, (token_id, slot))| report(*slot, *token_id, 2000 + row as u32))
                .collect())
        }

        fn worker_count(&self) -> usize {
            1
        }

        fn gpu_weight_ready_count(&self) -> usize {
            1
        }
    }

    fn report(slot: usize, input_token_id: u32, next_token_id: u32) -> KimiOneTokenForwardReport {
        KimiOneTokenForwardReport {
            rank: 0,
            batch_slot: slot,
            input_token_id,
            local_next_token_id: next_token_id,
            local_next_token_global_id: next_token_id,
            local_top_logit_f32: 0.0,
            vocab_start: 0,
            vocab_rows: 1,
            dense_layers_executed: 0,
            moe_layers_executed: 0,
            logprob: None,
        }
    }

    fn request(prompt_tokens: Vec<u32>) -> GenerateRequest {
        let (token_tx, _token_rx) = mpsc::unbounded_channel();
        GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens: 1,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        }
    }

    #[test]
    fn mixed_prompt_batch_routes_single_token_requests_to_decode() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let executor = RecordingExecutor::new(Arc::clone(&calls));
        let mut scheduler = KimiK2Scheduler {
            executor: Box::new(executor),
            stop_token_ids: Vec::new(),
        };

        scheduler.handle_request_batch(vec![
            request(vec![11]),
            request(vec![22, 33]),
            request(vec![44]),
        ]);

        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                ForwardCall::EnsureDecodeBatch(3),
                ForwardCall::Decode {
                    token_ids: vec![11],
                    append_positions: vec![0],
                    slots: vec![0],
                    decode_batch_size: 3,
                },
                ForwardCall::Prefill {
                    input_ids: vec![22, 33],
                    slot: 1,
                    decode_batch_size: 3,
                },
                ForwardCall::Decode {
                    token_ids: vec![44],
                    append_positions: vec![0],
                    slots: vec![2],
                    decode_batch_size: 3,
                },
            ]
        );
    }
}
