use super::*;

#[test]
fn send_rejection_reports_kv_lifetime_request_tokens() {
    let (token_tx, mut token_rx) = TokenSink::standalone();
    let req = SchedulerRequest {
        request_id: Some("too-large".to_string()),
        queued_at_unix_s: None,
        data_parallel_rank: None,
        prompt_tokens: vec![1; 16],
        params: SamplingParams::default(),
        max_tokens: 65,
        lora_adapter: None,
        token_tx,
        logprobs: 0,
        echo: false,
    };

    send_rejection(&req, RejectReason::KvBudget);

    match token_rx.blocking_recv().map(|(_, event)| event) {
        Some(TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens,
        }) => {
            assert_eq!(prompt_tokens, 16);
            assert_eq!(completion_tokens, 0);
            assert!(
                message.contains("max_request_tokens=80"),
                "rejection should report the full lifetime KV request"
            );
        }
        _ => panic!("expected rejection event"),
    }
}

#[test]
fn send_rejection_reports_context_window_limit() {
    let (token_tx, mut token_rx) = TokenSink::standalone();
    let req = SchedulerRequest {
        request_id: Some("too-long".to_string()),
        queued_at_unix_s: None,
        data_parallel_rank: None,
        prompt_tokens: vec![1; 16],
        params: SamplingParams::default(),
        max_tokens: 17,
        lora_adapter: None,
        token_tx,
        logprobs: 0,
        echo: false,
    };

    send_rejection(&req, RejectReason::ContextLength { limit: 32 });

    match token_rx.blocking_recv().map(|(_, event)| event) {
        Some(TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens,
        }) => {
            assert_eq!(prompt_tokens, 16);
            assert_eq!(completion_tokens, 0);
            assert!(
                message.contains("maximum context length of 32 tokens"),
                "rejection should report the context-window limit"
            );
            assert!(
                message.contains("requested 33"),
                "rejection should report prompt + max_tokens"
            );
        }
        _ => panic!("expected rejection event"),
    }
}
