use openinfer_engine::engine::{FinishReason, TokenEvent};

use super::{RequestMetrics, RequestOutcome};

pub(crate) struct RequestRecord {
    pub(crate) outcome: RequestOutcome,
    pub(crate) metrics: RequestMetrics,
    pub(crate) trace: Option<serde_json::Value>,
}

pub(crate) struct RequestTrace {
    request_id: String,
    queued_at_unix_s: f64,
    scheduled_at_unix_s: Option<f64>,
    first_token_emit_unix_s: Option<f64>,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
}

impl RequestTrace {
    pub(crate) fn new(request_id: String, queued_at_unix_s: f64, prompt_tokens: usize) -> Self {
        Self {
            request_id,
            queued_at_unix_s,
            scheduled_at_unix_s: None,
            first_token_emit_unix_s: None,
            prompt_tokens,
            cached_tokens: 0,
            completion_tokens: 0,
        }
    }

    pub(crate) fn observe_event(&mut self, event: &TokenEvent) -> Option<RequestOutcome> {
        match event {
            TokenEvent::Scheduled {
                queued_at_unix_s,
                scheduled_at_unix_s,
                prompt_tokens,
                cached_tokens,
            } => {
                self.queued_at_unix_s = *queued_at_unix_s;
                self.scheduled_at_unix_s = Some(*scheduled_at_unix_s);
                self.prompt_tokens = *prompt_tokens;
                self.cached_tokens = *cached_tokens;
                None
            }
            TokenEvent::Token { .. } => {
                if self.first_token_emit_unix_s.is_none() {
                    let first_token_emit_unix_s = now_secs_f64();
                    self.first_token_emit_unix_s = Some(first_token_emit_unix_s);
                }
                self.completion_tokens = self.completion_tokens.saturating_add(1);
                None
            }
            TokenEvent::PromptTokens { .. } => None,
            TokenEvent::Finished {
                finish_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                self.observe_terminal(*prompt_tokens, *completion_tokens);
                Some(outcome_from_finish_reason(*finish_reason))
            }
            TokenEvent::Error {
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                self.observe_terminal(*prompt_tokens, *completion_tokens);
                Some(RequestOutcome::Error)
            }
            TokenEvent::Rejected {
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                self.observe_terminal(*prompt_tokens, *completion_tokens);
                Some(RequestOutcome::Rejected)
            }
        }
    }

    pub(crate) fn finish(
        &self,
        outcome: RequestOutcome,
        terminal_at_unix_s: f64,
        include_trace: bool,
    ) -> RequestRecord {
        let metrics = self.metrics(terminal_at_unix_s);
        RequestRecord {
            outcome,
            metrics,
            trace: include_trace.then(|| self.to_json(outcome, terminal_at_unix_s)),
        }
    }

    fn observe_terminal(&mut self, prompt_tokens: usize, completion_tokens: usize) {
        self.prompt_tokens = prompt_tokens;
        self.completion_tokens = completion_tokens;
    }

    fn metrics(&self, terminal_at_unix_s: f64) -> RequestMetrics {
        RequestMetrics {
            queue_wait_ms: self
                .scheduled_at_unix_s
                .map(|scheduled| ms_between(self.queued_at_unix_s, scheduled)),
            ttft_ms: self
                .first_token_emit_unix_s
                .map(|first| ms_between(self.queued_at_unix_s, first)),
            duration_ms: Some(ms_between(self.queued_at_unix_s, terminal_at_unix_s)),
            prompt_tokens: self.prompt_tokens,
            cached_prompt_tokens: self.cached_tokens,
            completion_tokens: self.completion_tokens,
        }
    }

    fn to_json(&self, outcome: RequestOutcome, terminal_at_unix_s: f64) -> serde_json::Value {
        let mut trace = serde_json::json!({
            "request_id": self.request_id,
            "queued_at_unix_s": self.queued_at_unix_s,
            "terminal_at_unix_s": terminal_at_unix_s,
            "finish_reason": outcome.label(),
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
        });
        if let Some(scheduled_at) = self.scheduled_at_unix_s {
            trace["scheduled_at_unix_s"] = serde_json::json!(scheduled_at);
        }
        if let Some(first_token_at) = self.first_token_emit_unix_s {
            trace["first_token_emit_unix_s"] = serde_json::json!(first_token_at);
            if let Some(scheduled_at) = self.scheduled_at_unix_s {
                trace["prefill_ms"] = serde_json::json!(ms_between(scheduled_at, first_token_at));
            }
        }
        trace
    }
}

fn outcome_from_finish_reason(reason: FinishReason) -> RequestOutcome {
    match reason {
        FinishReason::Length => RequestOutcome::Length,
        FinishReason::Stop => RequestOutcome::Stop,
        FinishReason::Error => RequestOutcome::Error,
    }
}

fn ms_between(start_unix_s: f64, end_unix_s: f64) -> f64 {
    ((end_unix_s - start_unix_s) * 1_000.0).max(0.0)
}

fn now_secs_f64() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}
