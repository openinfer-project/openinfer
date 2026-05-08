/// E2E greedy regression test for Qwen3.5-4B.
///
/// Loads the model, runs greedy decoding on the standard test cases, and
/// compares against golden data in `test_data/Qwen3.5-4B.json`.
use std::path::{Path, PathBuf};

use log::info;
use pegainfer_core::engine::{EngineHandle, EngineLoadOptions, GenerateRequest, TokenEvent};
use pegainfer_core::sampler::SamplingParams;
use serde::Deserialize;
use tokio::sync::mpsc;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");

fn get_model_path() -> String {
    let path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
        .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    info!("Using model path: {}", path);
    path
}

fn get_test_data_path(model_path: &str) -> PathBuf {
    let name = Path::new(model_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(model_path);
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../test_data")
        .join(format!("{name}.json"));
    info!("Using test data path: {}", p.display());
    p
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
    output: String,
}

fn load_test_cases(path: &Path) -> Vec<TestCase> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    let data: TestData = serde_json::from_str(&content).expect("Failed to parse JSON");
    data.cases
}

fn generate_text(
    handle: &EngineHandle,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> String {
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

    let mut out = Vec::new();
    loop {
        match token_rx.blocking_recv() {
            Some(TokenEvent::Token { id, .. }) => out.push(id),
            Some(TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed"),
        }
    }

    tokenizer.decode(&out, true).expect("decode failed")
}

#[test]
fn test_e2e_qwen35_generation() {
    // logging intentionally left to the test harness

    let model_path = get_model_path();
    let test_data_path = get_test_data_path(&model_path);
    let cases = load_test_cases(&test_data_path);

    info!("Loading Qwen3.5 model...");
    let tokenizer = common::load_tokenizer(&model_path);
    let handle = pegainfer_qwen35_4b::start_engine(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            seed: 42,
        },
    )
    .expect("Failed to start engine");
    info!("Model loaded");

    info!("=== Qwen3.5-4B greedy correctness ===");
    let mut pass = 0;
    let mut fail = 0;
    for case in &cases {
        let output = generate_text(&handle, &tokenizer, &case.prompt, case.max_new_tokens);
        if output == case.output {
            info!("  PASS: {:?}", case.name);
            pass += 1;
        } else {
            eprintln!("  FAIL: {:?}", case.name);
            eprintln!("    expected: {:?}", case.output);
            eprintln!("    got:      {:?}", output);
            fail += 1;
        }
    }

    assert_eq!(
        fail,
        0,
        "{fail} / {} cases failed (see output above)",
        pass + fail
    );
    info!("All {} cases passed", pass);
}
