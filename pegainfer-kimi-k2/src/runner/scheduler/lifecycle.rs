use std::time::{SystemTime, UNIX_EPOCH};

use pegainfer_core::engine::{FinishReason, GenerateRequest, TokenEvent};

use crate::runner::worker::KIMI_MAX_REQUEST_TOKENS;

/// KV tokens a request can write over its lifetime. The final generated
/// token is returned but never fed back, so its KV is never written (the
/// same dangling-token contract as the qwen3 admission).
pub(in crate::runner) fn request_max_kv_tokens(req: &GenerateRequest) -> usize {
    req.prompt_tokens.len() + req.max_tokens.saturating_sub(1)
}

/// Pool blocks a request may draw over its lifetime. One token more than
/// `request_max_kv_tokens`: kvbm appends the final generated token to the
/// sequence and provisions its block even though its KV is never written,
/// so at boundary alignments the peak draw is `ceil((prompt + max) / bs)`.
pub(in crate::runner) fn request_lifetime_blocks(
    req: &GenerateRequest,
    block_size: usize,
) -> usize {
    (req.prompt_tokens.len() + req.max_tokens)
        .div_ceil(block_size)
        .max(1)
}

/// Honor-or-reject (#239): a request that can never fit — per-request KV
/// capacity, pool size, or a path-specific prompt cap — is rejected at
/// admission with the limit spelled out, instead of poisoning the batch
/// mid-decode when the KV write finally lands out of bounds.
pub(in crate::runner) fn validate_kv_capacity(
    req: &GenerateRequest,
    block_size: usize,
    max_request_blocks: usize,
    max_prompt_tokens: Option<usize>,
) -> Result<(), String> {
    if let Some(max_prompt) = max_prompt_tokens
        && req.prompt_tokens.len() > max_prompt
    {
        return Err(format!(
            "prompt of {} tokens exceeds the per-request prompt cap of {max_prompt} \
             tokens on this serving path",
            req.prompt_tokens.len()
        ));
    }
    let max_kv_tokens = request_max_kv_tokens(req);
    if max_kv_tokens > KIMI_MAX_REQUEST_TOKENS {
        return Err(format!(
            "prompt_tokens ({}) + max_tokens ({}) needs {max_kv_tokens} KV tokens, \
             exceeding the per-request capacity of {KIMI_MAX_REQUEST_TOKENS} tokens",
            req.prompt_tokens.len(),
            req.max_tokens
        ));
    }
    let blocks = request_lifetime_blocks(req, block_size);
    if blocks > max_request_blocks {
        return Err(format!(
            "request needs {blocks} KV blocks ({max_kv_tokens} KV tokens), exceeding \
             the pool capacity of {max_request_blocks} blocks"
        ));
    }
    Ok(())
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
