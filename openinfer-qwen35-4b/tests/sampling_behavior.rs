//! Behavioral coverage for Qwen3.5 sampling params on the scheduler path.
//!
//! This mirrors the Qwen3 test: `temperature` / `top_k` / `top_p` must steer
//! real generation, while `top_k=1` and tiny `top_p` collapse to greedy. It
//! guards #284 against silently falling back to greedy or dropping masks after
//! the single-row sampler was removed.
//!
//! Requires a CUDA GPU and Qwen3.5-4B weights; skips cleanly when the model is
//! absent (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, GenerateRequest, TokenEvent, TokenSink,
};
use openinfer_core::sampler::SamplingParams;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
const GENERATED_TOKENS: usize = 32;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen35 sampling_behavior: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

fn params(mut params: SamplingParams) -> SamplingParams {
    params.ignore_eos = true;
    params
}

/// Submit one request and collect the generated token ids until `Finished`.
fn generate(handle: &EngineHandle, prompt_tokens: Vec<u32>, params: SamplingParams) -> Vec<u32> {
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params,
            max_tokens: GENERATED_TOKENS,
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
            Some(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => return tokens,
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

#[test]
fn sampling_params_steer_the_qwen35_sampler() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = openinfer_qwen35_4b::start_engine_with_capacity(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            enable_prefill_profile: false,
            device_ordinals: vec![0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
        4,
        openinfer_qwen35_4b::DEFAULT_MAX_PREFILL_TOKENS,
    )
    .expect("failed to start Qwen3.5 engine");
    let tokenizer = common::load_tokenizer(&model_path);

    let prompt = "Here is a short story about a dragon. Once upon a time";
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");

    let greedy_params = params(SamplingParams::default());

    let greedy = generate(&handle, prompt_tokens.clone(), greedy_params);
    assert_eq!(
        greedy.len(),
        GENERATED_TOKENS,
        "ignore_eos should force a full 32-token generation"
    );
    let greedy_again = generate(&handle, prompt_tokens.clone(), greedy_params);
    assert_eq!(greedy, greedy_again, "greedy decode must be deterministic");

    let top_k_one = generate(
        &handle,
        prompt_tokens.clone(),
        params(SamplingParams {
            temperature: 0.8,
            top_k: 1,
            ..SamplingParams::default()
        }),
    );
    assert_eq!(top_k_one, greedy, "top_k=1 must collapse to greedy");

    let top_p_tiny = generate(
        &handle,
        prompt_tokens.clone(),
        params(SamplingParams {
            temperature: 1.0,
            top_p: 1e-6,
            ..SamplingParams::default()
        }),
    );
    assert_eq!(top_p_tiny, greedy, "top_p=1e-6 must collapse to greedy");

    let hot = params(SamplingParams {
        temperature: 1.5,
        top_k: -1,
        top_p: 1.0,
        ..SamplingParams::default()
    });
    let runs: Vec<Vec<u32>> = (0..4)
        .map(|_| generate(&handle, prompt_tokens.clone(), hot))
        .collect();
    assert!(
        runs.iter().any(|run| *run != runs[0]),
        "4 high-temperature runs were token-identical; sampling params are not reaching the sampler"
    );
}
