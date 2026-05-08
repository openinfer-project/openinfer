/// E2E scheduler integration test for Qwen3.5-4B.
///
/// Tests the Qwen3.5 reduced-capacity scheduler path (batch prefill +
/// CUDA Graph decode) with sequential, concurrent, and consumer-drop requests.
use std::path::{Path, PathBuf};
use std::time::Instant;

use log::info;
use serde::Deserialize;
use tokio::sync::mpsc;

use pegainfer_core::engine::FinishReason;
use pegainfer_core::engine::{EngineHandle, GenerateRequest, TokenEvent};
use pegainfer_core::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");

fn get_model_path() -> String {
    std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string())
}

fn get_test_data_path(model_path: &str) -> PathBuf {
    let name = Path::new(model_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(model_path);
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../test_data")
        .join(format!("{name}.json"))
}

#[derive(Deserialize)]
struct TestData {
    cases: Vec<TestCase>,
}

#[derive(Deserialize)]
struct TestCase {
    name: String,
    prompt: String,
    max_new_tokens: usize,
}

fn load_test_cases(path: &Path) -> Vec<TestCase> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    let data: TestData = serde_json::from_str(&content).expect("Failed to parse JSON");
    data.cases
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
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut tokens = Vec::new();
    loop {
        match token_rx.blocking_recv() {
            Some(TokenEvent::Token { id, .. }) => tokens.push(id),
            Some(TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { finish_reason, .. }) => {
                return (tokens, finish_reason);
            }
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

#[test]
fn test_e2e_qwen35_scheduler() {
    // logging intentionally left to the test harness

    let model_path = get_model_path();
    let test_data_path = get_test_data_path(&model_path);
    let cases = load_test_cases(&test_data_path);

    info!("Loading Qwen3.5 model for scheduler test...");
    let start = Instant::now();
    let model =
        pegainfer_qwen35_4b::runtime::Qwen35Model::from_safetensors_with_options(&model_path, true)
            .expect("Failed to load model");
    let tokenizer = common::load_tokenizer(&model_path);
    // Use reduced batch capacity (8) to fit on 16GB GPUs alongside the model.
    let handle = pegainfer_qwen35_4b::runtime::start_with_capacity(model, 42, 8)
        .expect("Failed to start Qwen3.5 scheduler");
    info!("Qwen3.5 scheduler loaded in {:.2?}", start.elapsed());

    // ── 1. Sequential scheduler requests ────────────────────────────────
    info!("=== Phase 1: Qwen3.5 sequential scheduler requests ===");
    for case in &cases {
        info!("--- {:?} ---", case.name);
        let start = Instant::now();
        let (tokens, finish_reason) =
            generate_tokens(&handle, &tokenizer, &case.prompt, case.max_new_tokens);
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
    for case in &cases {
        let (tokens, _) = generate_tokens(&handle, &tokenizer, &case.prompt, case.max_new_tokens);
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
        for case in &cases {
            let prompt_tokens = tokenizer
                .encode(&case.prompt, false)
                .expect("encode failed");
            let (token_tx, token_rx) = mpsc::unbounded_channel();
            handle
                .submit(GenerateRequest {
                    prompt_tokens,
                    params: SamplingParams::default(),
                    max_tokens: case.max_new_tokens,
                    token_tx,
                    logprobs: 0,
                    echo: false,
                })
                .expect("submit failed");
            receivers.push((case.name.clone(), token_rx));
        }

        // Collect all results
        for (name, mut rx) in receivers {
            let mut tokens = Vec::new();
            loop {
                match rx.blocking_recv() {
                    Some(TokenEvent::Token { id, .. }) => tokens.push(id),
                    Some(TokenEvent::PromptTokens { .. }) => {}
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
                prompt_tokens,
                params: SamplingParams::default(),
                max_tokens: 10,
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
