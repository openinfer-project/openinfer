//! Scheduler robustness IT for Qwen3-4B.
//!
//! Numerical regression lives in `hf_golden_gate.rs` (tolerance vs an HF golden);
//! this test owns the one thing that gate does not — that the scheduler keeps
//! running when a client hangs up mid-flight. We submit a request, drop its
//! receiver immediately, and assert the engine retires that request cleanly and
//! still serves the next one. It drives the real engine + `submit` rather than a
//! mocked scheduler, so it exercises the actual send-failure retirement path.
//!
//! Started via the `--batch-invariant` builder, it also checks that the flag sets
//! the pin policy before serving. CUDA-graph pin behavior is covered by
//! `batch_invariance_decode_gemm_graph`.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when the model is
//! absent (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;
use std::time::Duration;

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, GenerateRequest, TokenEvent, TokenSink,
};
use openinfer_core::sampler::SamplingParams;
use openinfer_kernels::ops::{
    NumericPolicy, numeric_policy, pin_served, reset_numeric_policy_counters,
};
use vllm_text::tokenizer::DynTokenizer;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 scheduler_robustness: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Submit `prompt` and block until the request finishes; returns the decoded text.
fn generate_text(
    handle: &EngineHandle,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> String {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
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
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, .. }) => tokens.push(id),
            Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
    tokenizer.decode(&tokens, true).expect("decode failed")
}

/// A client that drops its receiver before the request is scheduled must not
/// wedge the engine: the submit still succeeds, the scheduler retires the
/// orphaned request when its sends start failing, and later requests are served.
#[test]
fn scheduler_survives_consumer_drop() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = openinfer_qwen3::start_engine_with_offload(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            enable_prefill_profile: false,
            device_ordinals: vec![0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
        openinfer_qwen3::Qwen3OffloadOptions::disabled(),
        true,
        openinfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS,
        openinfer_qwen3::Qwen3MemoryOptions::default(),
        openinfer_qwen3::DecodeOverlap::Off,
        true,
        None,
        false,
    )
    .expect("failed to start engine");
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Pin,
        "--batch-invariant did not reach the pin policy before serving"
    );
    let tokenizer = common::load_tokenizer(&model_path);

    // Submit, then drop the receiver immediately — the scheduler should notice
    // the send failures and retire the request rather than spinning on it.
    let prompt_tokens = tokenizer.encode("Hello", false).expect("encode failed");
    let (token_tx, rx) = TokenSink::standalone();
    drop(rx);
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens: 10,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");
    std::thread::sleep(Duration::from_millis(500));

    // Barrier: drain the dropped orphan before the counted runs (else its prefill leaks into prefill_served).
    let _ = generate_text(&handle, &tokenizer, "Hello", 1);

    reset_numeric_policy_counters();
    let _ = generate_text(&handle, &tokenizer, "Hello", 1);
    let prefill_served = pin_served();

    reset_numeric_policy_counters();
    let text = generate_text(&handle, &tokenizer, "Hello", 5);
    let full_served = pin_served();
    eprintln!("[scheduler-robustness] pin served: prefill={prefill_served} full={full_served}");

    assert!(!text.is_empty(), "scheduler dead after consumer drop");
    // The pin ran decode GEMMs beyond prefill, not just prefill's.
    assert!(
        full_served > prefill_served,
        "pin served no decode GEMM beyond prefill (prefill={prefill_served} full={full_served}) — flag→builder→graph-capture may be broken"
    );
}
