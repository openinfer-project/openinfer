use std::time::{SystemTime, UNIX_EPOCH};

use pegainfer_core::engine::{FinishReason, GenerateRequest, TokenEvent};

pub(in crate::runner) fn schedule_prefill_candidate(
    req: GenerateRequest,
) -> Option<GenerateRequest> {
    send_scheduled(&req);
    if finish_unschedulable(&req) {
        None
    } else {
        Some(req)
    }
}

pub(in crate::runner) fn preflight_prefill_candidate(
    req: GenerateRequest,
) -> Option<GenerateRequest> {
    if finish_unschedulable(&req) {
        send_scheduled(&req);
        None
    } else {
        Some(req)
    }
}

pub(in crate::runner) fn send_scheduled(req: &GenerateRequest) {
    let scheduled_at = unix_now_s();
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s: req.queued_at_unix_s.unwrap_or(scheduled_at),
        scheduled_at_unix_s: scheduled_at,
        prompt_tokens: req.prompt_tokens.len(),
    });
}

fn finish_unschedulable(req: &GenerateRequest) -> bool {
    if req.max_tokens == 0 {
        let _ = req.token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
        return true;
    }
    if req.prompt_tokens.is_empty() {
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message: "Kimi-K2 forward requires at least one prompt token".to_string(),
            prompt_tokens: 0,
            completion_tokens: 0,
        });
        return true;
    }
    if let Err(message) = validate_sampling_params(req) {
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message,
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
        return true;
    }
    false
}

/// Honor-or-reject (#237): a request whose sampling parameters cannot be
/// honored exactly is rejected here, before any forward work.
fn validate_sampling_params(req: &GenerateRequest) -> Result<(), String> {
    let p = &req.params;
    if !p.temperature.is_finite() || p.temperature < 0.0 {
        return Err(format!(
            "temperature must be finite and >= 0, got {}",
            p.temperature
        ));
    }
    if !p.top_p.is_finite() || p.top_p <= 0.0 || p.top_p > 1.0 {
        return Err(format!("top_p must be in (0, 1], got {}", p.top_p));
    }
    if p.top_k < -1 || p.top_k == 0 {
        return Err(format!(
            "top_k must be -1 (disabled) or >= 1, got {}",
            p.top_k
        ));
    }
    Ok(())
}

fn unix_now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before UNIX_EPOCH")
        .as_secs_f64()
}
