//! Mixed-request greedy serving for the DeepSeek-V2-Lite EP2 gate.
//!
//! This is the first serving-semantics gate for the model. It keeps one
//! `DecodeCache` per active request, admits only shapes the current runtime can
//! honor exactly, and retires each request independently when validation,
//! disconnect, EOS, length, or request-local decode errors occur.

use std::{collections::VecDeque, mem};

use anyhow::{Result, ensure};
use openinfer_engine::{
    engine::{FinishReason, GenerateRequest, TokenEvent, TokenSink, unix_now_s},
    sampler::SamplingParams,
};
use tokio::sync::mpsc;

use crate::{
    Config,
    attribution::DecodeAttributionProfile,
    host_ops::DecodeCache,
    runtime::{DeepSeekV2LiteEp2Generator, GenerationStats},
};

pub(crate) const DEFAULT_MAX_ACTIVE_REQUESTS: usize = 8;

pub(crate) struct MixedRequestScheduler {
    generator: DeepSeekV2LiteEp2Generator,
    submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    pending: VecDeque<PendingRequest>,
    active: Vec<ActiveRequestState>,
    max_active_requests: usize,
}

struct PendingRequest {
    request_id: Option<String>,
    queued_at_unix_s: Option<f64>,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    max_tokens: usize,
    lora_adapter: Option<String>,
    token_tx: TokenSink,
    logprobs: usize,
    echo: bool,
}

struct ActiveRequestState {
    request_id: Option<String>,
    token_tx: TokenSink,
    prompt_len: usize,
    max_tokens: usize,
    generated: usize,
    last_token: u32,
    finish_policy: FinishPolicy,
    cache: DecodeCache,
    stats: GenerationStats,
}

#[derive(Clone, Copy)]
struct FinishPolicy {
    eos_token_id: u32,
    ignore_eos: bool,
}

struct AdmissionBatch {
    admitted: Vec<PendingRequest>,
    rejected: Vec<(PendingRequest, String)>,
    finished: Vec<PendingRequest>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum AdmissionDecision {
    Admit,
    Reject(String),
    Finish(FinishReason),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DecodeGrouping {
    Empty,
    BatchSamePosition { position: usize, rows: usize },
    SingleRows,
}

impl MixedRequestScheduler {
    pub(crate) fn new(
        generator: DeepSeekV2LiteEp2Generator,
        submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    ) -> Self {
        Self {
            generator,
            submit_rx,
            pending: VecDeque::new(),
            active: Vec::new(),
            max_active_requests: DEFAULT_MAX_ACTIVE_REQUESTS,
        }
    }

    pub(crate) fn run(mut self) {
        while self.block_until_work() {
            self.drain_pending_submissions();
            self.admit_ready_requests();
            if !self.active.is_empty() {
                self.decode_round();
            }
        }
    }

    fn block_until_work(&mut self) -> bool {
        if !self.pending.is_empty() || !self.active.is_empty() {
            return true;
        }

        match self.submit_rx.blocking_recv() {
            Some(req) => {
                self.pending.push_back(PendingRequest::from(req));
                true
            }
            None => false,
        }
    }

    fn drain_pending_submissions(&mut self) {
        while let Ok(req) = self.submit_rx.try_recv() {
            self.pending.push_back(PendingRequest::from(req));
        }
    }

    fn admit_ready_requests(&mut self) {
        let supported_context = self.generator.config().supported_plain_rope_context();
        let batch = take_admission_batch(
            &mut self.pending,
            self.active.len(),
            self.max_active_requests,
            supported_context,
        );

        for (pending, message) in batch.rejected {
            if send_scheduled(&pending) {
                let _ = send_prompt_echo(&pending);
                let _ = pending.token_tx.send(TokenEvent::Rejected {
                    message,
                    prompt_tokens: pending.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
        }

        for pending in batch.finished {
            if send_scheduled(&pending) {
                let _ = send_prompt_echo(&pending);
                let _ = pending.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Length,
                    prompt_tokens: pending.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
        }

        for pending in batch.admitted {
            if self.active.len() >= self.max_active_requests {
                self.pending.push_front(pending);
                break;
            }
            if let Some(active) = self.prefill_request(pending) {
                self.active.push(active);
            }
        }
    }

    fn prefill_request(&mut self, pending: PendingRequest) -> Option<ActiveRequestState> {
        let prompt_len = pending.prompt_tokens.len();
        if !send_scheduled(&pending) {
            return None;
        }

        if !send_prompt_echo(&pending) {
            return None;
        }

        let mut cache = DecodeCache::new(self.generator.config());
        let mut stats = self.generator.new_generation_stats(prompt_len);
        let mut attribution = DecodeAttributionProfile::disabled();
        let next = match self.generator.prefill_next_token(
            &pending.prompt_tokens,
            &mut cache,
            &mut stats,
            &mut attribution,
        ) {
            Ok(token) => token,
            Err(err) => {
                let _ = pending.token_tx.send(TokenEvent::Error {
                    message: err.to_string(),
                    prompt_tokens: prompt_len,
                    completion_tokens: 0,
                });
                return None;
            }
        };

        let mut active = ActiveRequestState {
            request_id: pending.request_id,
            token_tx: pending.token_tx,
            prompt_len,
            max_tokens: pending.max_tokens,
            generated: 0,
            last_token: next,
            finish_policy: FinishPolicy {
                eos_token_id: self.generator.config().eos_token_id,
                ignore_eos: pending.params.ignore_eos,
            },
            cache,
            stats,
        };

        if active.emit_token_or_finish(next) {
            return None;
        }
        Some(active)
    }

    fn decode_round(&mut self) {
        self.retire_bad_cache_positions();
        let positions: Vec<_> = self
            .active
            .iter()
            .map(ActiveRequestState::next_decode_position)
            .collect();
        match decode_grouping_for_positions(&positions) {
            DecodeGrouping::Empty => {}
            DecodeGrouping::BatchSamePosition { position, rows } if rows > 1 => {
                self.decode_batch_round(position);
            }
            DecodeGrouping::BatchSamePosition { .. } | DecodeGrouping::SingleRows => {
                self.decode_single_rows();
            }
        }
    }

    fn retire_bad_cache_positions(&mut self) {
        let config = self.generator.config();
        let mut survivors = Vec::with_capacity(self.active.len());
        for state in self.active.drain(..) {
            match state.cache_position(config) {
                Ok(()) => survivors.push(state),
                Err(message) => state.emit_error(message.to_string()),
            }
        }
        self.active = survivors;
    }

    fn decode_batch_round(&mut self, position: usize) {
        let tokens: Vec<_> = self.active.iter().map(|state| state.last_token).collect();
        let token_index = self
            .active
            .iter()
            .map(|state| state.generated)
            .min()
            .unwrap_or(0);
        let prompt_tokens = self.active.iter().map(|state| state.prompt_len).sum();
        let mut stats = self.generator.new_generation_stats(prompt_tokens);
        let mut attribution = DecodeAttributionProfile::disabled();
        let mut caches: Vec<_> = self
            .active
            .iter_mut()
            .map(|state| mem::take(&mut state.cache))
            .collect();
        let result = self.generator.decode_next_tokens_batch(
            &tokens,
            position,
            &mut caches,
            &mut stats,
            &mut attribution,
            token_index,
        );

        match result {
            Ok(next_tokens) if next_tokens.len() == self.active.len() => {
                for (state, cache) in self.active.iter_mut().zip(caches) {
                    state.cache = cache;
                }
                self.apply_decoded_tokens(next_tokens);
            }
            // The batched path mutates per-row caches as it advances through the
            // model. This gate avoids full-cache rollback clones; a batch decode
            // failure is therefore a shared runtime error for the active rows.
            Ok(next_tokens) => self.retire_active_batch_error(format!(
                "DeepSeek-V2-Lite batched decode returned {} rows for {} active requests",
                next_tokens.len(),
                self.active.len()
            )),
            Err(err) => self.retire_active_batch_error(format!(
                "DeepSeek-V2-Lite batched decode failed for {} active requests: {err}",
                self.active.len()
            )),
        }
    }

    fn decode_single_rows(&mut self) {
        let mut survivors = Vec::with_capacity(self.active.len());
        for mut state in self.active.drain(..) {
            let token = state.last_token;
            let position = state.next_decode_position();
            let token_index = state.generated;
            let result = self.generator.decode_next_token(
                token,
                position,
                &mut state.cache,
                &mut state.stats,
                &mut DecodeAttributionProfile::disabled(),
                token_index,
            );
            match result {
                Ok(next) => {
                    if !state.emit_token_or_finish(next) {
                        survivors.push(state);
                    }
                }
                Err(err) => state.emit_error(err.to_string()),
            }
        }
        self.active = survivors;
    }

    fn apply_decoded_tokens(&mut self, next_tokens: Vec<u32>) {
        let mut survivors = Vec::with_capacity(self.active.len());
        for (mut state, token) in self.active.drain(..).zip(next_tokens) {
            if !state.emit_token_or_finish(token) {
                survivors.push(state);
            }
        }
        self.active = survivors;
    }

    fn retire_active_batch_error(&mut self, message: String) {
        retire_active_requests_with_error(&mut self.active, message);
    }
}

fn retire_active_requests_with_error(active: &mut Vec<ActiveRequestState>, message: String) {
    for state in active.drain(..) {
        state.emit_error(message.clone());
    }
}

impl From<GenerateRequest> for PendingRequest {
    fn from(req: GenerateRequest) -> Self {
        Self {
            request_id: req.request_id,
            queued_at_unix_s: req.queued_at_unix_s,
            prompt_tokens: req.prompt_tokens,
            params: req.params,
            max_tokens: req.max_tokens,
            lora_adapter: req.lora_adapter,
            token_tx: req.token_tx,
            logprobs: req.logprobs,
            echo: req.echo,
        }
    }
}

impl ActiveRequestState {
    fn next_decode_position(&self) -> usize {
        self.prompt_len + self.generated - 1
    }

    fn cache_position(&self, config: &Config) -> Result<()> {
        let expected = self.next_decode_position();
        let actual = self.cache.position(config)?;
        ensure!(
            actual == expected,
            "DeepSeek-V2-Lite request {:?} cache position mismatch: cache_len={}, expected={expected}",
            self.request_id,
            actual
        );
        Ok(())
    }

    fn emit_token_or_finish(&mut self, token: u32) -> bool {
        self.last_token = token;
        if !self.finish_policy.ignore_eos && token == self.finish_policy.eos_token_id {
            let _ = self.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: self.prompt_len,
                completion_tokens: self.generated,
            });
            return true;
        }

        if self
            .token_tx
            .send(TokenEvent::Token {
                id: token,
                logprob: None,
            })
            .is_err()
        {
            return true;
        }
        self.generated += 1;

        if self.generated == self.max_tokens {
            let _ = self.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: self.prompt_len,
                completion_tokens: self.generated,
            });
            return true;
        }
        false
    }

    fn emit_error(self, message: String) {
        let _ = self.token_tx.send(TokenEvent::Error {
            message,
            prompt_tokens: self.prompt_len,
            completion_tokens: self.generated,
        });
    }
}

fn send_scheduled(pending: &PendingRequest) -> bool {
    let now = unix_now_s();
    pending
        .token_tx
        .send(TokenEvent::Scheduled {
            queued_at_unix_s: pending.queued_at_unix_s.unwrap_or(now),
            scheduled_at_unix_s: now,
            prompt_tokens: pending.prompt_tokens.len(),
            cached_tokens: 0,
        })
        .is_ok()
}

fn send_prompt_echo(pending: &PendingRequest) -> bool {
    if !pending.echo {
        return true;
    }
    pending
        .token_tx
        .send(TokenEvent::PromptTokens {
            ids: pending.prompt_tokens.clone(),
            logprobs: vec![None; pending.prompt_tokens.len()],
        })
        .is_ok()
}

fn take_admission_batch(
    pending: &mut VecDeque<PendingRequest>,
    active_len: usize,
    max_active_requests: usize,
    supported_context: usize,
) -> AdmissionBatch {
    let mut batch = AdmissionBatch {
        admitted: Vec::new(),
        rejected: Vec::new(),
        finished: Vec::new(),
    };

    while let Some(pending_req) = pending.pop_front() {
        let can_admit = active_len + batch.admitted.len() < max_active_requests;
        match admission_decision(&pending_req, supported_context) {
            AdmissionDecision::Admit if can_admit => batch.admitted.push(pending_req),
            AdmissionDecision::Admit => {
                pending.push_front(pending_req);
                break;
            }
            AdmissionDecision::Reject(message) => batch.rejected.push((pending_req, message)),
            AdmissionDecision::Finish(FinishReason::Length) => batch.finished.push(pending_req),
            AdmissionDecision::Finish(reason) => {
                batch.rejected.push((
                    pending_req,
                    format!("DeepSeek-V2-Lite unsupported admission finish reason: {reason:?}"),
                ));
            }
        }
    }

    batch
}

fn admission_decision(req: &PendingRequest, supported_context: usize) -> AdmissionDecision {
    let prompt_tokens = req.prompt_tokens.len();
    if !req.params.is_greedy() {
        return AdmissionDecision::Reject(format!(
            "DeepSeek-V2-Lite EP=2 mixed serving gate supports greedy decoding only; requested temperature={}, top_k={}, top_p={}",
            req.params.temperature, req.params.top_k, req.params.top_p
        ));
    }
    if req.logprobs > 0 {
        return AdmissionDecision::Reject(
            "DeepSeek-V2-Lite EP=2 mixed serving gate does not return logprobs yet".to_string(),
        );
    }
    if req.lora_adapter.is_some() {
        return AdmissionDecision::Reject(
            "DeepSeek-V2-Lite EP=2 mixed serving gate does not support LoRA adapters".to_string(),
        );
    }
    if req.prompt_tokens.is_empty() {
        return AdmissionDecision::Reject(
            "DeepSeek-V2-Lite EP=2 mixed serving gate requires a non-empty prompt".to_string(),
        );
    }
    if req.max_tokens == 0 {
        return AdmissionDecision::Finish(FinishReason::Length);
    }

    let Some(requested_context) = prompt_tokens.checked_add(req.max_tokens) else {
        return AdmissionDecision::Reject(format!(
            "DeepSeek-V2-Lite EP=2 mixed serving gate context length overflow: prompt_tokens={prompt_tokens} max_new_tokens={}",
            req.max_tokens
        ));
    };
    if requested_context > supported_context {
        return AdmissionDecision::Reject(format!(
            "DeepSeek-V2-Lite EP=2 mixed serving gate supports plain RoPE context <= {supported_context} tokens; requested prompt_tokens={prompt_tokens} max_new_tokens={} total={requested_context}. YaRN rope_scaling long context is not implemented yet.",
            req.max_tokens
        ));
    }

    AdmissionDecision::Admit
}

fn decode_grouping_for_positions(positions: &[usize]) -> DecodeGrouping {
    let Some((&first, rest)) = positions.split_first() else {
        return DecodeGrouping::Empty;
    };
    if positions.len() > 1 && rest.iter().all(|position| *position == first) {
        return DecodeGrouping::BatchSamePosition {
            position: first,
            rows: positions.len(),
        };
    }
    DecodeGrouping::SingleRows
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::AtomicBool};

    use openinfer_engine::engine::RequestTag;
    use openinfer_engine::sampler::SamplingParams;
    use tokio::sync::mpsc;

    use super::*;
    use crate::config::test_lite_config;

    fn request(
        id: &str,
        prompt_len: usize,
        max_tokens: usize,
    ) -> (
        PendingRequest,
        openinfer_engine::engine::TokenStreamReceiver,
    ) {
        let (token_tx, token_rx) = TokenSink::standalone();
        (
            PendingRequest {
                request_id: Some(id.to_string()),
                queued_at_unix_s: None,
                prompt_tokens: vec![1; prompt_len],
                params: SamplingParams::default(),
                max_tokens,
                lora_adapter: None,
                token_tx,
                logprobs: 0,
                echo: false,
            },
            token_rx,
        )
    }

    fn recv_event(rx: &mut openinfer_engine::engine::TokenStreamReceiver) -> TokenEvent {
        rx.try_recv().expect("expected event").1
    }

    #[test]
    fn admission_rejects_unsupported_shapes() {
        let context = 16;

        let (mut sampling, _rx) = request("sampling", 1, 1);
        sampling.params.temperature = 0.8;
        assert!(matches!(
            admission_decision(&sampling, context),
            AdmissionDecision::Reject(message) if message.contains("greedy")
        ));

        let (mut logprobs, _rx) = request("logprobs", 1, 1);
        logprobs.logprobs = 1;
        assert!(matches!(
            admission_decision(&logprobs, context),
            AdmissionDecision::Reject(message) if message.contains("logprobs")
        ));

        let (mut lora, _rx) = request("lora", 1, 1);
        lora.lora_adapter = Some("adapter-a".to_string());
        assert!(matches!(
            admission_decision(&lora, context),
            AdmissionDecision::Reject(message) if message.contains("LoRA")
        ));

        let (empty, _rx) = request("empty", 0, 1);
        assert!(matches!(
            admission_decision(&empty, context),
            AdmissionDecision::Reject(message) if message.contains("non-empty prompt")
        ));

        let (zero, _rx) = request("zero", 1, 0);
        assert_eq!(
            admission_decision(&zero, context),
            AdmissionDecision::Finish(FinishReason::Length)
        );
    }

    #[test]
    fn context_overflow_is_rejected() {
        let (req, _rx) = request("too-long", 12, 5);

        assert!(matches!(
            admission_decision(&req, 16),
            AdmissionDecision::Reject(message)
                if message.contains("context") && message.contains("total=17")
        ));
    }

    #[test]
    fn active_cap_defers_in_fcfs_order() {
        let mut pending = VecDeque::new();
        pending.push_back(request("first", 2, 1).0);
        pending.push_back(request("second", 2, 1).0);
        pending.push_back(request("third", 2, 1).0);

        let batch = take_admission_batch(&mut pending, 1, 3, 16);

        assert_eq!(
            batch
                .admitted
                .iter()
                .map(|req| req.request_id.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("first"), Some("second")]
        );
        assert!(batch.rejected.is_empty());
        assert!(batch.finished.is_empty());
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].request_id.as_deref(), Some("third"));
    }

    #[test]
    fn terminal_requests_do_not_wait_for_active_capacity() {
        let mut pending = VecDeque::new();
        pending.push_back(request("zero", 2, 0).0);
        let (mut invalid, _rx) = request("invalid", 2, 1);
        invalid.logprobs = 1;
        pending.push_back(invalid);
        pending.push_back(request("valid", 2, 1).0);

        let batch = take_admission_batch(&mut pending, 8, 8, 16);

        assert!(batch.admitted.is_empty());
        assert_eq!(batch.finished.len(), 1);
        assert_eq!(batch.finished[0].request_id.as_deref(), Some("zero"));
        assert_eq!(batch.rejected.len(), 1);
        assert_eq!(batch.rejected[0].0.request_id.as_deref(), Some("invalid"));
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].request_id.as_deref(), Some("valid"));
    }

    #[test]
    fn invalid_request_does_not_block_later_admission_when_cap_has_room() {
        let mut pending = VecDeque::new();
        let (mut invalid, _rx) = request("invalid", 2, 1);
        invalid.logprobs = 1;
        pending.push_back(invalid);
        pending.push_back(request("valid", 2, 1).0);

        let batch = take_admission_batch(&mut pending, 0, 2, 16);

        assert_eq!(batch.rejected.len(), 1);
        assert_eq!(batch.rejected[0].0.request_id.as_deref(), Some("invalid"));
        assert_eq!(batch.admitted.len(), 1);
        assert_eq!(batch.admitted[0].request_id.as_deref(), Some("valid"));
        assert!(pending.is_empty());
    }

    #[test]
    fn terminal_admission_events_keep_scheduler_contract() {
        let (mut zero, mut zero_rx) = request("zero", 2, 0);
        zero.echo = true;
        assert!(send_scheduled(&zero));
        assert!(send_prompt_echo(&zero));
        let _ = zero.token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: zero.prompt_tokens.len(),
            completion_tokens: 0,
        });

        assert!(matches!(
            recv_event(&mut zero_rx),
            TokenEvent::Scheduled { .. }
        ));
        assert!(matches!(
            recv_event(&mut zero_rx),
            TokenEvent::PromptTokens { ids, .. } if ids == vec![1, 1]
        ));
        assert!(matches!(
            recv_event(&mut zero_rx),
            TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                completion_tokens: 0,
                ..
            }
        ));

        let (rejected, mut rejected_rx) = request("rejected", 2, 1);
        assert!(send_scheduled(&rejected));
        let _ = rejected.token_tx.send(TokenEvent::Rejected {
            message: "nope".to_string(),
            prompt_tokens: rejected.prompt_tokens.len(),
            completion_tokens: 0,
        });

        assert!(matches!(
            recv_event(&mut rejected_rx),
            TokenEvent::Scheduled { .. }
        ));
        assert!(matches!(
            recv_event(&mut rejected_rx),
            TokenEvent::Rejected {
                completion_tokens: 0,
                ..
            }
        ));
    }

    #[test]
    fn eos_retirement_is_independent_per_request() {
        let config = test_lite_config();
        let (tx_stop, mut rx_stop) = TokenSink::standalone();
        let (tx_live, mut rx_live) = TokenSink::standalone();
        let mut stop_state = ActiveRequestState {
            request_id: Some("stop".to_string()),
            token_tx: tx_stop,
            prompt_len: 3,
            max_tokens: 4,
            generated: 1,
            last_token: 10,
            finish_policy: FinishPolicy {
                eos_token_id: config.eos_token_id,
                ignore_eos: false,
            },
            cache: DecodeCache::new(&config),
            stats: GenerationStats::default(),
        };
        let mut live_state = ActiveRequestState {
            request_id: Some("live".to_string()),
            token_tx: tx_live,
            prompt_len: 2,
            max_tokens: 4,
            generated: 1,
            last_token: 11,
            finish_policy: FinishPolicy {
                eos_token_id: config.eos_token_id,
                ignore_eos: false,
            },
            cache: DecodeCache::new(&config),
            stats: GenerationStats::default(),
        };

        assert!(stop_state.emit_token_or_finish(config.eos_token_id));
        assert!(!live_state.emit_token_or_finish(12));

        match recv_event(&mut rx_stop) {
            TokenEvent::Finished {
                finish_reason,
                completion_tokens,
                ..
            } => {
                assert_eq!(finish_reason, FinishReason::Stop);
                assert_eq!(completion_tokens, 1);
            }
            _ => panic!("EOS request should finish without emitting EOS"),
        }
        match recv_event(&mut rx_live) {
            TokenEvent::Token { id, .. } => assert_eq!(id, 12),
            _ => panic!("live request should receive its own token"),
        }
        assert!(rx_live.try_recv().is_err());
    }

    #[test]
    fn cancelled_token_sink_retires_request() {
        let config = test_lite_config();
        let (stream_tx, mut stream_rx) = mpsc::unbounded_channel();
        let cancelled = Arc::new(AtomicBool::new(true));
        let sink = TokenSink::new(
            RequestTag::from("cancelled"),
            stream_tx,
            Arc::clone(&cancelled),
        );
        let mut state = ActiveRequestState {
            request_id: Some("cancelled".to_string()),
            token_tx: sink,
            prompt_len: 2,
            max_tokens: 4,
            generated: 1,
            last_token: 11,
            finish_policy: FinishPolicy {
                eos_token_id: config.eos_token_id,
                ignore_eos: false,
            },
            cache: DecodeCache::new(&config),
            stats: GenerationStats::default(),
        };

        assert!(state.emit_token_or_finish(12));
        assert!(stream_rx.try_recv().is_err());
    }

    #[test]
    fn closed_token_sink_retires_request() {
        let config = test_lite_config();
        let (sink, rx) = TokenSink::standalone();
        drop(rx);
        let mut state = ActiveRequestState {
            request_id: Some("closed".to_string()),
            token_tx: sink,
            prompt_len: 2,
            max_tokens: 4,
            generated: 1,
            last_token: 11,
            finish_policy: FinishPolicy {
                eos_token_id: config.eos_token_id,
                ignore_eos: false,
            },
            cache: DecodeCache::new(&config),
            stats: GenerationStats::default(),
        };

        assert!(state.emit_token_or_finish(12));
    }

    #[test]
    fn batch_decode_error_retires_all_active_requests() {
        let config = test_lite_config();
        let (first_tx, mut first_rx) = TokenSink::standalone();
        let (second_tx, mut second_rx) = TokenSink::standalone();
        let mut active = vec![
            ActiveRequestState {
                request_id: Some("first".to_string()),
                token_tx: first_tx,
                prompt_len: 3,
                max_tokens: 8,
                generated: 2,
                last_token: 11,
                finish_policy: FinishPolicy {
                    eos_token_id: config.eos_token_id,
                    ignore_eos: false,
                },
                cache: DecodeCache::new(&config),
                stats: GenerationStats::default(),
            },
            ActiveRequestState {
                request_id: Some("second".to_string()),
                token_tx: second_tx,
                prompt_len: 4,
                max_tokens: 8,
                generated: 1,
                last_token: 12,
                finish_policy: FinishPolicy {
                    eos_token_id: config.eos_token_id,
                    ignore_eos: false,
                },
                cache: DecodeCache::new(&config),
                stats: GenerationStats::default(),
            },
        ];

        retire_active_requests_with_error(&mut active, "batch failed".to_string());

        assert!(active.is_empty());
        match recv_event(&mut first_rx) {
            TokenEvent::Error {
                message,
                prompt_tokens,
                completion_tokens,
            } => {
                assert_eq!(message, "batch failed");
                assert_eq!(prompt_tokens, 3);
                assert_eq!(completion_tokens, 2);
            }
            _ => panic!("first active request should receive batch error"),
        }
        match recv_event(&mut second_rx) {
            TokenEvent::Error {
                message,
                prompt_tokens,
                completion_tokens,
            } => {
                assert_eq!(message, "batch failed");
                assert_eq!(prompt_tokens, 4);
                assert_eq!(completion_tokens, 1);
            }
            _ => panic!("second active request should receive batch error"),
        }
    }

    #[test]
    fn decode_grouping_batches_only_uniform_positions() {
        assert_eq!(decode_grouping_for_positions(&[]), DecodeGrouping::Empty);
        assert_eq!(
            decode_grouping_for_positions(&[5]),
            DecodeGrouping::SingleRows
        );
        assert_eq!(
            decode_grouping_for_positions(&[7, 7, 7]),
            DecodeGrouping::BatchSamePosition {
                position: 7,
                rows: 3
            }
        );
        assert_eq!(
            decode_grouping_for_positions(&[7, 8, 7]),
            DecodeGrouping::SingleRows
        );
    }
}
