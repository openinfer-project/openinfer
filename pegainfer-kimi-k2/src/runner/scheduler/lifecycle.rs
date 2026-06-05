use std::time::{SystemTime, UNIX_EPOCH};

use pegainfer_core::engine::{FinishReason, GenerateRequest, TokenEvent};

use crate::config::KIMI_K2_SERVING_CONTEXT_TOKENS;

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
    // Per-request KV demand: the prompt plus every generated token except the
    // last, which is emitted without being fed back. The HTTP layer already
    // clamps to the advertised `max_model_len`, so this only fires for direct
    // engine submitters — rejecting here beats erroring mid-batch, where a
    // forward failure takes down every co-scheduled request (issue #239).
    let kv_demand = req.prompt_tokens.len() + req.max_tokens - 1;
    if kv_demand > KIMI_K2_SERVING_CONTEXT_TOKENS {
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message: format!(
                "Kimi-K2 serving context is {KIMI_K2_SERVING_CONTEXT_TOKENS} tokens; \
                 prompt ({}) + max_tokens ({}) needs {kv_demand} KV positions",
                req.prompt_tokens.len(),
                req.max_tokens
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

    fn request(
        prompt_len: usize,
        max_tokens: usize,
        params: SamplingParams,
    ) -> (GenerateRequest, mpsc::UnboundedReceiver<TokenEvent>) {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        (
            GenerateRequest {
                request_id: None,
                queued_at_unix_s: None,
                prompt_tokens: vec![1; prompt_len],
                params,
                max_tokens,
                lora_adapter: None,
                token_tx,
                logprobs: 0,
                echo: false,
            },
            token_rx,
        )
    }

    fn rejection_message(token_rx: &mut mpsc::UnboundedReceiver<TokenEvent>) -> String {
        let Ok(TokenEvent::Rejected { message, .. }) = token_rx.try_recv() else {
            panic!("expected Rejected event");
        };
        message
    }

    #[test]
    fn greedy_request_is_schedulable() {
        let (req, _token_rx) = request(3, 8, SamplingParams::default());
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
        let (req, mut token_rx) = request(3, 8, params);

        assert!(preflight_prefill_candidate(req).is_none());
        let message = rejection_message(&mut token_rx);
        assert!(message.contains("greedy only"), "message: {message}");
        assert!(message.contains("temperature=0.8"), "message: {message}");
    }

    #[test]
    fn request_filling_the_context_exactly_is_schedulable() {
        // KV demand is prompt + max_tokens - 1: the last generated token is
        // emitted without being appended.
        let max_tokens = 8;
        let prompt_len = KIMI_K2_SERVING_CONTEXT_TOKENS + 1 - max_tokens;
        let (req, _token_rx) = request(prompt_len, max_tokens, SamplingParams::default());
        assert!(preflight_prefill_candidate(req).is_some());
    }

    #[test]
    fn request_one_token_over_the_context_is_rejected() {
        let max_tokens = 8;
        let prompt_len = KIMI_K2_SERVING_CONTEXT_TOKENS + 2 - max_tokens;
        let (req, mut token_rx) = request(prompt_len, max_tokens, SamplingParams::default());

        assert!(preflight_prefill_candidate(req).is_none());
        let message = rejection_message(&mut token_rx);
        assert!(message.contains("serving context"), "message: {message}");
    }

    #[test]
    fn over_long_prompt_is_rejected() {
        let (req, mut token_rx) = request(
            KIMI_K2_SERVING_CONTEXT_TOKENS + 1,
            1,
            SamplingParams::default(),
        );

        assert!(preflight_prefill_candidate(req).is_none());
        let message = rejection_message(&mut token_rx);
        assert!(message.contains("serving context"), "message: {message}");
    }
}
