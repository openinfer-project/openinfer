use std::path::Path;

use openinfer_core::engine::EngineLoadOptions;
use openinfer_core::engine::EpBackend;

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
fn tp_scheduler_uses_eager_only_plan() {
    let pending = vec!["prefill"];
    assert!(
        matches!(
            build_eager_only_plan(true, pending),
            Some(ExecutionPlan::Prefill { pending }) if pending == vec!["prefill"]
        ),
        "TP Phase 1 should prefill first instead of choosing unified"
    );
    assert!(
        matches!(
            build_eager_only_plan::<&str>(true, vec![]),
            Some(ExecutionPlan::Decode)
        ),
        "TP Phase 1 should decode only when no prefill chunk is scheduled"
    );
}

#[test]
fn tp_engine_rejects_cuda_graph_before_model_load() {
    let err = match crate::start_engine_with_capacity(
        Path::new("unused"),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0, 1],
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        },
        1,
        1,
    ) {
        Ok(_) => panic!("TP CUDA Graph startup should fail"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("eager execution only"));
}

#[test]
#[ignore = "requires two CUDA devices and Qwen3.5 weights"]
fn tp2_scheduler_chunked_prefill_then_decode_smoke() {
    let model_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
        .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
    let handle =
        start_tp_with_capacity(&model_path, 42, &[0, 1], 1, 1).expect("start Qwen3.5 TP scheduler");
    let (token_tx, mut token_rx) = TokenSink::standalone();

    handle
        .submit(SchedulerRequest {
            request_id: Some("tp2-scheduler-smoke".to_string()),
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens: vec![151_646, 9707],
            params: SamplingParams {
                ignore_eos: true,
                ..SamplingParams::default()
            },
            max_tokens: 3,
            lora_adapter: None,
            token_tx,
            logprobs: 1,
            echo: false,
        })
        .expect("submit TP scheduler request");

    let mut tokens = Vec::new();
    loop {
        match token_rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, logprob }) => {
                let logprob = logprob.expect("TP scheduler smoke should return token logprob");
                assert!(logprob.logprob.is_finite());
                assert_eq!(logprob.top_logprobs.len(), 1);
                tokens.push(id);
            }
            Some(TokenEvent::Finished {
                finish_reason,
                prompt_tokens,
                completion_tokens,
            }) => {
                assert_eq!(finish_reason, FinishReason::Length);
                assert_eq!(prompt_tokens, 2);
                assert_eq!(completion_tokens, 3);
                assert_eq!(tokens.len(), 3);
                break;
            }
            Some(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Error { message, .. }) => {
                panic!("TP scheduler smoke failed: {message}")
            }
            Some(TokenEvent::Rejected { message, .. }) => {
                panic!("TP scheduler smoke rejected: {message}")
            }
            None => panic!("TP scheduler channel closed before Finished"),
        }
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
