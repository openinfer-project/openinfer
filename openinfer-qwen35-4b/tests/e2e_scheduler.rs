/// E2E scheduler integration test for Qwen3.5-4B.
///
/// Tests the Qwen3.5 reduced-capacity scheduler path (batch prefill +
/// CUDA Graph decode) with sequential, concurrent, and consumer-drop requests.
use std::time::Instant;

use log::info;
use tokio::sync::mpsc;

use openinfer_core::engine::FinishReason;
use openinfer_core::engine::{EngineHandle, GenerateRequest, TokenEvent};
use openinfer_core::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");

const CASES: &[TestCase] = &[
    TestCase {
        name: "tell_story",
        prompt: "Tell me a story",
        max_new_tokens: 50,
    },
    TestCase {
        name: "my_name",
        prompt: "My name is",
        max_new_tokens: 50,
    },
    TestCase {
        name: "math",
        prompt: "What is 2 + 2?",
        max_new_tokens: 30,
    },
    TestCase {
        name: "chinese_weather",
        prompt: "The weather is nice today",
        max_new_tokens: 50,
    },
    TestCase {
        name: "chinese_capital",
        prompt: "Introduce the capital city of China",
        max_new_tokens: 50,
    },
    TestCase {
        name: "python_code",
        prompt: "Write a Python function to reverse a string",
        max_new_tokens: 50,
    },
    TestCase {
        name: "kanye_album",
        prompt: "My favorite Kanye West album is",
        max_new_tokens: 50,
    },
    TestCase {
        name: "coldplay_ghost",
        prompt: "Coldplay's Ghost Stories album feels",
        max_new_tokens: 50,
    },
    TestCase {
        name: "oyster_riddle",
        prompt: "An oyster cooked in a pan becomes",
        max_new_tokens: 50,
    },
    TestCase {
        name: "monkey_king_lake",
        prompt: "A clever monkey jumps into a lake and returns as",
        max_new_tokens: 50,
    },
];

fn get_model_path() -> String {
    std::env::var("OPENINFER_TEST_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string())
}

fn max_position_embeddings(model_path: &str) -> usize {
    let config_path = std::path::Path::new(model_path).join("config.json");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", config_path.display())),
    )
    .expect("config.json must be valid JSON");
    config
        .pointer("/text_config/max_position_embeddings")
        .or_else(|| config.pointer("/max_position_embeddings"))
        .and_then(serde_json::Value::as_u64)
        .expect("Qwen3.5 config must expose max_position_embeddings") as usize
}

struct TestCase {
    name: &'static str,
    prompt: &'static str,
    max_new_tokens: usize,
}

fn generate_tokens(
    handle: &EngineHandle,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> (Vec<u32>, FinishReason) {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();

    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut tokens = Vec::new();
    loop {
        match token_rx.blocking_recv() {
            Some(TokenEvent::Token { id, .. }) => tokens.push(id),
            Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
            Some(TokenEvent::Finished { finish_reason, .. }) => {
                return (tokens, finish_reason);
            }
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

fn expect_context_window_rejection(handle: &EngineHandle, max_context_tokens: usize) {
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();

    handle
        .submit(GenerateRequest {
            request_id: Some("over-context-window".to_string()),
            queued_at_unix_s: None,
            prompt_tokens: vec![1; max_context_tokens],
            params: SamplingParams::default(),
            max_tokens: 1,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit over-context request");

    match token_rx.blocking_recv() {
        Some(TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens,
        }) => {
            assert_eq!(prompt_tokens, max_context_tokens);
            assert_eq!(completion_tokens, 0);
            assert!(
                message.contains("maximum context length"),
                "expected context-window rejection, got: {message}"
            );
            assert!(
                message.contains(&(max_context_tokens + 1).to_string()),
                "rejection should report prompt + max_tokens, got: {message}"
            );
        }
        Some(_) => panic!("expected context-window rejection"),
        None => panic!("scheduler channel closed without rejection"),
    }
}

#[test]
fn test_e2e_qwen35_scheduler() {
    // logging intentionally left to the test harness

    let model_path = get_model_path();
    let max_context_tokens = max_position_embeddings(&model_path);

    info!("Loading Qwen3.5 model for scheduler test...");
    let start = Instant::now();
    let model =
        openinfer_qwen35_4b::runtime::Qwen35Model::from_safetensors_with_options(&model_path, true)
            .expect("Failed to load model");
    let tokenizer = common::load_tokenizer(&model_path);
    // Use reduced batch capacity (8) to fit on 16GB GPUs alongside the model.
    let handle = openinfer_qwen35_4b::runtime::start_with_capacity(model, 42, 8)
        .expect("Failed to start Qwen3.5 scheduler");
    info!("scheduler loaded in {:.2?}", start.elapsed());

    // ── 0. Static context-window rejection ─────────────────────────────
    info!("=== Phase 0: Context-window rejection ===");
    expect_context_window_rejection(&handle, max_context_tokens);
    info!("  PASS: over-context request rejected before prefill");

    // ── 1. Sequential scheduler requests ────────────────────────────────
    info!("=== Phase 1: Qwen3.5 sequential scheduler requests ===");
    for case in CASES {
        info!("--- {:?} ---", case.name);
        let start = Instant::now();
        let (tokens, finish_reason) =
            generate_tokens(&handle, &tokenizer, case.prompt, case.max_new_tokens);
        let elapsed = start.elapsed();

        let text = tokenizer.decode(&tokens, true).expect("decode failed");
        let tok_s = tokens.len() as f64 / elapsed.as_secs_f64();

        info!(
            "  {} tokens in {:.2?} ({:.1} tok/s) finish={:?}",
            tokens.len(),
            elapsed,
            tok_s,
            finish_reason
        );

        assert!(!text.is_empty(), "empty output for: {:?}", case.name);
        if tokens.len() >= case.max_new_tokens {
            assert_eq!(finish_reason, FinishReason::Length);
        }

        info!("  PASS: {:?}", case.name);
    }

    // ── 2. Multi-request (scheduler state reuse) ────────────────────────
    info!("=== Phase 2: Multi-request ===");
    for case in CASES {
        let (tokens, _) = generate_tokens(&handle, &tokenizer, case.prompt, case.max_new_tokens);
        let text = tokenizer.decode(&tokens, true).expect("decode failed");
        assert!(
            !text.is_empty(),
            "empty output on second run for: {:?}",
            case.name
        );
        info!("  PASS: {:?} → {} tokens", case.name, tokens.len());
    }

    // ── 3. Concurrent requests ──────────────────────────────────────────
    info!("=== Phase 3: Concurrent requests ===");
    {
        let mut receivers: Vec<(String, mpsc::UnboundedReceiver<TokenEvent>)> = Vec::new();

        // Submit all cases concurrently
        for case in CASES {
            let prompt_tokens = tokenizer.encode(case.prompt, false).expect("encode failed");
            let (token_tx, token_rx) = mpsc::unbounded_channel();
            handle
                .submit(GenerateRequest {
                    request_id: None,
                    queued_at_unix_s: None,
                    prompt_tokens,
                    params: SamplingParams::default(),
                    max_tokens: case.max_new_tokens,
                    lora_adapter: None,
                    token_tx,
                    logprobs: 0,
                    echo: false,
                })
                .expect("submit failed");
            receivers.push((case.name.to_string(), token_rx));
        }

        // Collect all results
        for (name, mut rx) in receivers {
            let mut tokens = Vec::new();
            loop {
                match rx.blocking_recv() {
                    Some(TokenEvent::Token { id, .. }) => tokens.push(id),
                    Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
                    Some(TokenEvent::Finished { .. }) => break,
                    Some(TokenEvent::Error { message, .. }) => {
                        panic!("generation failed: {message}")
                    }
                    Some(TokenEvent::Rejected { message, .. }) => {
                        panic!("generation rejected: {message}")
                    }
                    None => panic!("channel closed for {:?}", name),
                }
            }
            let text = tokenizer.decode(&tokens, true).expect("decode failed");
            assert!(!text.is_empty(), "empty output for concurrent: {:?}", name);
            info!("  PASS: {:?} → {} tokens", name, tokens.len());
        }
    }

    // ── 4. Consumer drop safety ─────────────────────────────────────────
    info!("=== Phase 4: Consumer drop ===");
    {
        let prompt_tokens = tokenizer.encode("Hello", false).expect("encode failed");
        let (token_tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        handle
            .submit(GenerateRequest {
                request_id: None,
                queued_at_unix_s: None,
                prompt_tokens,
                params: SamplingParams::default(),
                max_tokens: 10,
                lora_adapter: None,
                token_tx,
                logprobs: 0,
                echo: false,
            })
            .expect("submit failed");
        std::thread::sleep(std::time::Duration::from_millis(500));
        info!("  PASS: consumer drop handled");
    }

    // Verify scheduler survives
    let (tokens, _) = generate_tokens(&handle, &tokenizer, "Hello", 5);
    let text = tokenizer.decode(&tokens, true).expect("decode failed");
    assert!(!text.is_empty(), "scheduler dead after consumer drop");
    info!("  PASS: scheduler survived consumer drop");

    info!("All Qwen3.5 scheduler tests passed!");
}
