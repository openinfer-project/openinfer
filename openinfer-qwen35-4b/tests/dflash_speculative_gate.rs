//! Scheduler-level Qwen3.5 DFlash gate coverage.
//!
//! This test keeps the PR1 contract narrow: DFlash is opt-in, single-active,
//! greedy-only, and lossless against the plain scheduler for the covered path.
//! Multi-active requests are expected to fall back to normal target decode.

use std::path::{Path, PathBuf};

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, FinishReason, GenerateRequest, TokenEvent, TokenSink,
    TokenStreamReceiver,
};
use openinfer_core::sampler::SamplingParams;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
const SINGLE_ACTIVE_MAX_TOKENS: usize = 16;
const CONCURRENT_MAX_TOKENS: usize = 8;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen35 dflash_speculative_gate: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

fn draft_path_or_skip() -> Option<PathBuf> {
    match std::env::var("OPENINFER_DFLASH_TEST_MODEL_PATH") {
        Ok(path) if Path::new(&path).join("config.json").exists() => Some(PathBuf::from(path)),
        Ok(path) => {
            eprintln!("skipping qwen35 dflash_speculative_gate: {path}/config.json is missing");
            None
        }
        Err(_) => {
            eprintln!(
                "skipping qwen35 dflash_speculative_gate: set OPENINFER_DFLASH_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

fn start_engine(model_path: &str, dflash_draft_model_path: Option<PathBuf>) -> EngineHandle {
    openinfer_qwen35_4b::start_engine_with_capacity_and_dflash(
        Path::new(model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            enable_prefill_profile: false,
            device_ordinals: vec![0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
        4,
        openinfer_qwen35_4b::DEFAULT_MAX_PREFILL_TOKENS,
        dflash_draft_model_path,
    )
    .expect("failed to start Qwen3.5 engine")
}

fn set_require_spec(enabled: bool) {
    // Tests that mutate process environment run with `--test-threads=1` in the
    // documented GPU gate. Keep the env change scoped to engine startup.
    unsafe {
        if enabled {
            std::env::set_var("OPENINFER_QWEN35_DFLASH_REQUIRE_SPEC", "1");
        } else {
            std::env::remove_var("OPENINFER_QWEN35_DFLASH_REQUIRE_SPEC");
        }
    }
}

fn generate(
    handle: &EngineHandle,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    params: SamplingParams,
    logprobs: usize,
    name: &str,
) -> (Vec<u32>, FinishReason) {
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: Some(name.to_string()),
            queued_at_unix_s: None,
            prompt_tokens,
            params,
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs,
            echo: false,
        })
        .expect("submit failed");

    collect(&mut rx, name)
}

fn collect(rx: &mut TokenStreamReceiver, name: &str) -> (Vec<u32>, FinishReason) {
    let mut tokens = Vec::new();
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, .. }) => tokens.push(id),
            Some(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { finish_reason, .. }) => return (tokens, finish_reason),
            Some(TokenEvent::Error { message, .. }) => {
                panic!("{name}: generation failed: {message}")
            }
            Some(TokenEvent::Rejected { message, .. }) => {
                panic!("{name}: generation rejected: {message}")
            }
            None => panic!("{name}: scheduler channel closed without Finished"),
        }
    }
}

fn deterministic_params() -> SamplingParams {
    SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    }
}

#[test]
fn dflash_single_active_matches_plain_greedy_scheduler() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let Some(draft_path) = draft_path_or_skip() else {
        return;
    };
    let tokenizer = common::load_tokenizer(&model_path);
    let prompt = concat!(
        "Explain how an inference scheduler preserves request state while doing ",
        "greedy decoding. Include attention cache ownership, recurrent state, ",
        "and why speculative verification must be lossless. Keep the answer concise."
    );
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    assert!(
        prompt_tokens.len() >= 16,
        "test prompt must be long enough for DFlash context capture"
    );

    let plain_tokens = {
        let handle = start_engine(&model_path, None);
        let (tokens, finish) = generate(
            &handle,
            prompt_tokens.clone(),
            SINGLE_ACTIVE_MAX_TOKENS,
            deterministic_params(),
            0,
            "plain-single-active",
        );
        assert_eq!(finish, FinishReason::Length);
        tokens
    };

    let dflash_tokens = {
        set_require_spec(true);
        let handle = start_engine(&model_path, Some(draft_path));
        let (tokens, finish) = generate(
            &handle,
            prompt_tokens,
            SINGLE_ACTIVE_MAX_TOKENS,
            deterministic_params(),
            0,
            "dflash-single-active",
        );
        assert_eq!(finish, FinishReason::Length);
        set_require_spec(false);
        tokens
    };
    set_require_spec(false);

    assert_eq!(
        dflash_tokens, plain_tokens,
        "Qwen3.5 DFlash single-active greedy path must be lossless against the plain scheduler"
    );
}

#[test]
fn dflash_multi_active_and_logprobs_fallback_finish() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let Some(draft_path) = draft_path_or_skip() else {
        return;
    };
    let tokenizer = common::load_tokenizer(&model_path);
    set_require_spec(false);
    let handle = start_engine(&model_path, Some(draft_path));
    let prompts = [
        concat!(
            "Summarize continuous batching with one example. Mention request slots, ",
            "KV cache growth, and scheduler fairness."
        ),
        concat!(
            "Describe greedy decoding in a short paragraph. Mention token sampling, ",
            "EOS handling, and max token limits."
        ),
        concat!(
            "Explain why logprobs require extra output handling. Mention top tokens, ",
            "host copies, and deterministic greedy tokens."
        ),
    ];
    let mut receivers = Vec::new();
    for (idx, prompt) in prompts.iter().enumerate() {
        let (token_tx, token_rx) = TokenSink::standalone();
        let logprobs = usize::from(idx == 2);
        handle
            .submit(GenerateRequest {
                request_id: Some(format!("dflash-fallback-{idx}")),
                queued_at_unix_s: None,
                prompt_tokens: tokenizer.encode(prompt, false).expect("encode failed"),
                params: deterministic_params(),
                max_tokens: CONCURRENT_MAX_TOKENS,
                lora_adapter: None,
                token_tx,
                logprobs,
                echo: false,
            })
            .expect("submit failed");
        receivers.push((idx, token_rx));
    }

    for (idx, mut rx) in receivers {
        let (tokens, finish) = collect(&mut rx, &format!("dflash-fallback-{idx}"));
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(
            tokens.len(),
            CONCURRENT_MAX_TOKENS,
            "fallback request {idx} should finish through the normal scheduler path"
        );
    }
}
