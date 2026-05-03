use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use log::info;
use serde::Deserialize;
use tokio::sync::mpsc;

use pegainfer_core::engine::{
    EngineHandle, EngineLoadOptions, FinishReason, GenerateRequest, TokenEvent,
};
use pegainfer_core::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/Qwen3-4B");

fn get_model_path() -> String {
    let model_path =
        std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string());
    info!("Using model path: {}", model_path);
    model_path
}

fn get_test_data_path(model_path: &str) -> PathBuf {
    let model_name = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path);
    let test_data_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(format!("{model_name}.json"));
    info!("Using test data path: {}", test_data_path.display());
    test_data_path
}

// ── Test data types ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TestData {
    cases: Vec<TestCase>,
}

#[derive(Deserialize)]
struct TestCase {
    #[allow(dead_code)]
    name: String,
    prompt: String,
    max_new_tokens: usize,
    output: String,
}

fn load_test_cases(test_data_path: &Path) -> Vec<TestCase> {
    let content = std::fs::read_to_string(test_data_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", test_data_path.display()));
    let data: TestData = serde_json::from_str(&content).expect("Failed to parse test data JSON");
    data.cases
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn init_logging() {}

/// Submit a request and collect all generated tokens (blocking).
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
            params: SamplingParams::default(), // greedy
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
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

// ── Test ─────────────────────────────────────────────────────────────────

#[test]
fn test_e2e_generation() {
    init_logging();
    let model_path = get_model_path();
    let test_data_path = get_test_data_path(&model_path);
    let test_cases = load_test_cases(&test_data_path);

    info!("Loading engine...");
    let handle = pegainfer_qwen3_4b::start_engine(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            seed: 42,
        },
    )
    .expect("Failed to start engine");
    let tokenizer = common::load_tokenizer(&model_path);

    // Build expected-output lookup from JSON
    let expected: HashMap<&str, &str> = test_cases
        .iter()
        .map(|tc| (tc.prompt.as_str(), tc.output.as_str()))
        .collect();

    // Derive (prompt, max_tokens) from JSON — single source of truth
    let cases: Vec<(&str, usize)> = test_cases
        .iter()
        .map(|tc| (tc.prompt.as_str(), tc.max_new_tokens))
        .collect();

    // ── 1. Greedy correctness ────────────────────────────────────────────

    info!("=== Phase 1: Greedy correctness ===");
    for &(prompt, max_tokens) in &cases {
        info!("--- \"{}\" ---", prompt);

        let start = Instant::now();
        let (tokens, finish_reason) = generate_tokens(&handle, &tokenizer, prompt, max_tokens);
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
        info!("  Output: \"{}\"", text);

        assert!(!text.is_empty(), "empty output for: {}", prompt);
        if tokens.len() >= max_tokens {
            assert_eq!(finish_reason, FinishReason::Length);
        }

        // Greedy output must match reference
        let exp = expected[prompt];
        assert_eq!(
            text, exp,
            "greedy output mismatch for: \"{}\"\n  got:      {:?}\n  expected: {:?}",
            prompt, text, exp
        );
        info!("  PASS: matches reference");
    }

    // ── 2. Multi-request (re-run all cases) ──────────────────────────────

    info!("=== Phase 2: Multi-request ===");
    for &(prompt, max_tokens) in &cases {
        let (tokens, _) = generate_tokens(&handle, &tokenizer, prompt, max_tokens);
        let text = tokenizer.decode(&tokens, true).expect("decode failed");
        assert!(
            !text.is_empty(),
            "empty output on second run for: {}",
            prompt
        );
        info!("  PASS: \"{}\" → {} tokens", prompt, tokens.len());
    }

    // ── 3. Consumer drop safety ──────────────────────────────────────────

    info!("=== Phase 3: Consumer drop ===");
    {
        let prompt_tokens = tokenizer.encode("Hello", false).expect("encode failed");
        let (token_tx, rx) = mpsc::unbounded_channel();
        drop(rx); // drop receiver immediately
        // Submit should succeed — scheduler will notice send error and retire the request
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
        // Give scheduler time to process and retire
        std::thread::sleep(std::time::Duration::from_millis(500));
        info!("  PASS: consumer drop handled");
    }

    // Verify scheduler is still alive after consumer drop
    let (tokens, _) = generate_tokens(&handle, &tokenizer, "Hello", 5);
    let text = tokenizer.decode(&tokens, true).expect("decode failed");
    assert!(!text.is_empty(), "scheduler dead after consumer drop");
    info!("  PASS: scheduler survived consumer drop");
}
