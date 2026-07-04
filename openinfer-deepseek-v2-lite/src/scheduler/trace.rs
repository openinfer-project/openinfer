use openinfer_engine::engine::FinishReason;

#[derive(Clone, Debug)]
pub(super) struct RequestTrace {
    pub(super) queued_at_unix_s: f64,
    pub(super) scheduled_at_unix_s: f64,
    pub(super) prefill_done_unix_s: Option<f64>,
    pub(super) first_token_emit_unix_s: Option<f64>,
    pub(super) prefill_ms: Option<f64>,
    pub(super) first_decode_ms: Option<f64>,
    pub(super) active_set_size_max: usize,
    pub(super) pending_queue_size_max: usize,
    pub(super) active_set_size_at_terminal: usize,
    pub(super) pending_queue_size_at_terminal: usize,
    pub(super) decode_batch_size_max: usize,
    pub(super) batch_decode_steps: usize,
}

#[derive(Debug)]
pub(super) struct ScheduledTrace {
    pub(super) queued_at_unix_s: f64,
    pub(super) scheduled_at_unix_s: f64,
}

impl RequestTrace {
    pub(super) fn new(
        queued_at_unix_s: f64,
        scheduled_at_unix_s: f64,
        prefill_done_unix_s: f64,
        prefill_ms: f64,
    ) -> Self {
        Self {
            queued_at_unix_s,
            scheduled_at_unix_s,
            prefill_done_unix_s: Some(prefill_done_unix_s),
            first_token_emit_unix_s: None,
            prefill_ms: Some(prefill_ms),
            first_decode_ms: None,
            active_set_size_max: 1,
            pending_queue_size_max: 0,
            active_set_size_at_terminal: 0,
            pending_queue_size_at_terminal: 0,
            decode_batch_size_max: 0,
            batch_decode_steps: 0,
        }
    }

    pub(super) fn terminal(queued_at_unix_s: f64, scheduled_at_unix_s: f64) -> Self {
        Self {
            queued_at_unix_s,
            scheduled_at_unix_s,
            prefill_done_unix_s: None,
            first_token_emit_unix_s: None,
            prefill_ms: None,
            first_decode_ms: None,
            active_set_size_max: 0,
            pending_queue_size_max: 0,
            active_set_size_at_terminal: 0,
            pending_queue_size_at_terminal: 0,
            decode_batch_size_max: 0,
            batch_decode_steps: 0,
        }
    }

    pub(super) fn note_active_set(&mut self, active_set_size: usize) {
        self.active_set_size_max = self.active_set_size_max.max(active_set_size);
    }

    pub(super) fn note_scheduler_state(
        &mut self,
        active_set_size: usize,
        pending_queue_size: usize,
    ) {
        self.note_active_set(active_set_size);
        self.pending_queue_size_max = self.pending_queue_size_max.max(pending_queue_size);
    }

    pub(super) fn note_decode_step(&mut self, batch_size: usize, decode_ms: f64) {
        self.first_decode_ms.get_or_insert(decode_ms);
        self.decode_batch_size_max = self.decode_batch_size_max.max(batch_size);
        if batch_size > 1 {
            self.batch_decode_steps += 1;
        }
    }

    pub(super) fn note_terminal_state(
        &mut self,
        active_set_size_at_terminal: usize,
        pending_queue_size_at_terminal: usize,
    ) {
        self.active_set_size_at_terminal = active_set_size_at_terminal;
        self.pending_queue_size_at_terminal = pending_queue_size_at_terminal;
        self.note_scheduler_state(active_set_size_at_terminal, pending_queue_size_at_terminal);
    }
}

pub(super) fn http_trace_payload(
    request_id: &str,
    trace: &RequestTrace,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: FinishReason,
    error: Option<&str>,
) -> serde_json::Value {
    let mut payload = serde_json::json!({
        "request_id": request_id,
        "queued_at_unix_s": trace.queued_at_unix_s,
        "scheduled_at_unix_s": trace.scheduled_at_unix_s,
        "prefill_done_unix_s": trace.prefill_done_unix_s,
        "first_token_emit_unix_s": trace.first_token_emit_unix_s,
        "prefill_ms": trace.prefill_ms,
        "first_decode_ms": trace.first_decode_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason_label(finish_reason),
        "terminal_reason": terminal_reason_label(finish_reason, error),
        "active_set_size": trace.active_set_size_max,
        "active_set_size_max": trace.active_set_size_max,
        "pending_queue_size_max": trace.pending_queue_size_max,
        "active_set_size_at_terminal": trace.active_set_size_at_terminal,
        "pending_queue_size_at_terminal": trace.pending_queue_size_at_terminal,
        "healthy_baseline_after_terminal": trace.active_set_size_at_terminal == 0
            && trace.pending_queue_size_at_terminal == 0,
        "decode_batch_size_max": trace.decode_batch_size_max,
        "batch_decode_steps": trace.batch_decode_steps,
    });
    if let Some(error) = error
        && let Some(object) = payload.as_object_mut()
    {
        object.insert(
            "error".to_string(),
            serde_json::Value::String(error.to_string()),
        );
    }
    payload
}

fn finish_reason_label(finish_reason: FinishReason) -> &'static str {
    match finish_reason {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
        FinishReason::Error => "error",
    }
}

fn terminal_reason_label(finish_reason: FinishReason, error: Option<&str>) -> &'static str {
    match (finish_reason, error) {
        (FinishReason::Length, _) => "completed_length",
        (FinishReason::Stop, _) => "completed_stop",
        (FinishReason::Error, Some(message)) if message.contains("disconnected") => "disconnected",
        (FinishReason::Error, Some(message)) if message.contains("cancelled") => "cancelled",
        (FinishReason::Error, Some(message))
            if message.contains("supports greedy")
                || message.contains("logprobs")
                || message.contains("LoRA")
                || message.contains("non-empty prompt")
                || message.contains("context")
                || message.contains("unsupported") =>
        {
            "rejected"
        }
        (FinishReason::Error, _) => "error",
    }
}
