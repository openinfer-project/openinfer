use super::*;

#[test]
fn send_rejection_reports_kv_lifetime_request_tokens() {
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
    let req = SchedulerRequest {
        request_id: Some("too-large".to_string()),
        queued_at_unix_s: None,
        prompt_tokens: vec![1; 16],
        params: SamplingParams::default(),
        max_tokens: 65,
        lora_adapter: None,
        token_tx,
        logprobs: 0,
        echo: false,
    };

    send_rejection(&req);

    match token_rx.blocking_recv() {
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
