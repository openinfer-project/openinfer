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
    // The decode path only computes a split-vocab argmax; it cannot sample.
    // Rejecting here keeps the API contract honest — silently returning
    // greedy output for a temperature>0 request is the one forbidden state
    // (issue #237).
    if req.params.temperature > 0.0 {
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message: format!(
                "Kimi-K2 decodes greedy only; non-greedy sampling is not implemented \
                 (got temperature={}, top_k={}, top_p={}). Send temperature=0",
                req.params.temperature, req.params.top_k, req.params.top_p
            ),
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
        return true;
    }
    false
}

fn unix_now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before UNIX_EPOCH")
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use pegainfer_core::sampler::SamplingParams;
    use tokio::sync::mpsc;

    use super::*;

    fn request(params: SamplingParams) -> (GenerateRequest, mpsc::UnboundedReceiver<TokenEvent>) {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        (
            GenerateRequest {
                request_id: None,
                queued_at_unix_s: None,
                prompt_tokens: vec![1, 2, 3],
                params,
                max_tokens: 8,
                lora_adapter: None,
                token_tx,
                logprobs: 0,
                echo: false,
            },
            token_rx,
        )
    }

    #[test]
    fn greedy_request_is_schedulable() {
        let (req, _token_rx) = request(SamplingParams::default());
        assert!(preflight_prefill_candidate(req).is_some());
    }

    #[test]
    fn non_greedy_request_is_rejected() {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.9,
            ignore_eos: false,
        };
        let (req, mut token_rx) = request(params);

        assert!(preflight_prefill_candidate(req).is_none());
        let Ok(TokenEvent::Rejected { message, .. }) = token_rx.try_recv() else {
            panic!("expected Rejected event");
        };
        assert!(message.contains("greedy only"), "message: {message}");
        assert!(message.contains("temperature=0.8"), "message: {message}");
    }
}
