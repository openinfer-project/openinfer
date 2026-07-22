//! In-window context regression IT for Qwen3-4B.
//!
//! Companion to `context_window.rs` (which proves *oversized* prompts are
//! rejected). This proves the *positive* side of the same fix: a prompt that is
//! larger than the old hardcoded 4096-entry RoPE table but still inside the
//! position-encoding window (`max_position_embeddings` = 40960) must be served
//! end-to-end.
//!
//! A 4097-token prompt spans positions 0..=4096. Position 4096 is exactly the
//! first index past the old table's bounds (valid 0..=4095) — before the fix it
//! read garbage out of bounds (and after the kernel trap was added, would
//! `__trap` the CUDA context). Generating even one token therefore requires the
//! resized RoPE cache to actually exist and be indexed at 4096; if the table
//! ever silently reverts to 4096 entries this test crashes instead of passing.
//!
//! Lives in its own test binary for the same reason as `context_window.rs`:
//! `cargo test` serializes test binaries but parallelizes `#[test]`s within one
//! binary, so two engines on one GPU would contend. One engine per file.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when the model is
//! absent (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use openinfer_core::engine::EngineLoadOptions;
use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::TokenEvent;
use openinfer_core::engine::TokenSink;
use openinfer_core::sampler::SamplingParams;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 context_window_in_window: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

#[test]
fn in_window_prompt_past_old_rope_table_is_served() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = openinfer_qwen3::start_engine(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
    )
    .expect("failed to start engine");

    // 4097 tokens → positions 0..=4096. Position 4096 is the first index past the
    // old 4096-entry RoPE table; serving this prompt requires the resized cache.
    // Token id 1 is a valid vocab id — a forward pass actually runs here (unlike
    // the rejection test, which never reaches prefill).
    let prompt_tokens = vec![1u32; 4097];
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            trace_parent: None,
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens: 1,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut generated = 0usize;
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { .. }) => generated += 1,
            Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            Some(TokenEvent::Error { message, .. }) => {
                panic!("in-window prompt errored (resized RoPE cache not exercised?): {message}")
            }
            Some(TokenEvent::Rejected { message, .. }) => {
                panic!("in-window prompt was wrongly rejected: {message}")
            }
            None => panic!("scheduler channel closed without Finished"),
        }
    }

    assert_eq!(
        generated, 1,
        "expected exactly one generated token for a 4097-token prompt with max_tokens=1"
    );
}
