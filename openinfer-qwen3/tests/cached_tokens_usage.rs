//! Prefix-cache observability IT for Qwen3-4B (#246).
//!
//! The frontend reports `usage.prompt_tokens_details.cached_tokens` from
//! `TokenEvent::Scheduled`. This test pins the engine half of that contract:
//! a cold prompt reports zero cached tokens, a warm repeat of the same prompt
//! reports a nonzero full-block count, and the count never claims the whole
//! prompt (the last token is always recomputed).
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when the model is
//! absent (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, GenerateRequest, TokenEvent, TokenSink,
};
use openinfer_core::sampler::SamplingParams;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const KV_BLOCK_SIZE: usize = 16;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 cached_tokens_usage: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Submit `prompt_tokens`, drain the stream to `Finished`, and return the
/// `cached_tokens` carried by the `Scheduled` event.
fn run_and_capture_cached(handle: &EngineHandle, prompt_tokens: Vec<u32>) -> usize {
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens: 4,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut cached = None;
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Scheduled { cached_tokens, .. }) => {
                assert!(
                    cached.replace(cached_tokens).is_none(),
                    "Scheduled must be emitted exactly once per request"
                );
            }
            Some(TokenEvent::Token { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
    cached.expect("Scheduled event must precede Finished")
}

#[test]
fn warm_repeat_reports_cached_tokens() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = openinfer_qwen3::start_engine(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            enable_prefill_profile: false,
            device_ordinals: vec![0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
    )
    .expect("failed to start engine");
    let tokenizer = common::load_tokenizer(&model_path);

    let prompt = "The kv cache stores attention keys and values for every \
        generated token so the model never recomputes earlier positions. "
        .repeat(8);
    let prompt_tokens = tokenizer.encode(&prompt, false).expect("encode failed");
    let prompt_len = prompt_tokens.len();
    assert!(
        prompt_len > 2 * KV_BLOCK_SIZE,
        "prompt must span multiple KV blocks for a meaningful hit"
    );

    let cold = run_and_capture_cached(&handle, prompt_tokens.clone());
    assert_eq!(cold, 0, "cold run must report zero cached tokens");

    let warm = run_and_capture_cached(&handle, prompt_tokens);
    assert!(warm > 0, "warm repeat must report a prefix-cache hit");
    assert!(
        warm < prompt_len,
        "at least the last prompt token is always recomputed (warm={warm}, prompt={prompt_len})"
    );
    assert_eq!(
        warm % KV_BLOCK_SIZE,
        0,
        "hits are matched in full blocks (warm={warm})"
    );
    assert_eq!(
        warm,
        (prompt_len - 1) / KV_BLOCK_SIZE * KV_BLOCK_SIZE,
        "warm hit must cover every cacheable full block (prompt={prompt_len})"
    );
}
