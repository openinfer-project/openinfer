//! Qwen3.5 DFlash scheduler losslessness gate.
//!
//! The live DFlash path must be opt-in and greedy-lossless over Qwen3.5's
//! hybrid target state: full-attention KV plus recurrent and convolution state.
//! Exact token equality is the first preference. At a divergence we use the
//! same regret rule as the Qwen3 gate: the speculative token must be either the
//! prefill-kernel greedy pick for the shared context or sit within a small
//! near-tie band in that prefill distribution. This avoids false failures from
//! the known decode-vs-prefill bf16 tie boundary while still catching real
//! verify/commit/capture bugs.

use std::path::{Path, PathBuf};
use std::time::Duration;

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, GenerateRequest, TokenEvent, TokenSink,
};
use openinfer_core::sampler::SamplingParams;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use vllm_text::tokenizer::DynTokenizer;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B-DFlash");
const LOGPROBS: usize = 20;
const MARGIN_TOL: f32 = 0.20;
const MAX_BATCH: usize = 16;
const SYNTHETIC_TOKEN_LO: u32 = 100;
const SYNTHETIC_TOKEN_HI: u32 = 100_000;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Step {
    id: u32,
    top_logprobs: Vec<(u32, f32)>,
}

fn target_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen35 DFlash gate: {MODEL_PATH}/config.json missing; set OPENINFER_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn draft_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_DFLASH_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DRAFT_PATH).join("config.json").exists() => {
            Some(DRAFT_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen35 DFlash gate: {DRAFT_PATH}/config.json missing; set OPENINFER_DFLASH_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn engine_options() -> EngineLoadOptions {
    EngineLoadOptions {
        enable_cuda_graph: true,
        enable_prefill_profile: false,
        device_ordinals: vec![0],
        seed: 42,
        ..EngineLoadOptions::default()
    }
}

fn launch(model_path: &str, draft_path: Option<PathBuf>) -> EngineHandle {
    openinfer_qwen35_4b::start_engine_with_capacity_and_dflash(
        Path::new(model_path),
        engine_options(),
        MAX_BATCH,
        openinfer_qwen35_4b::DEFAULT_MAX_PREFILL_TOKENS,
        draft_path,
    )
    .expect("failed to start Qwen3.5 engine")
}

fn greedy_params() -> SamplingParams {
    SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    }
}

fn generate(
    handle: &EngineHandle,
    prompt_tokens: Vec<u32>,
    logprobs: usize,
    max_tokens: usize,
) -> Vec<Step> {
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: greedy_params(),
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs,
            echo: false,
        })
        .expect("submit failed");

    collect_steps(&mut rx)
}

fn generate_concurrent(handle: &EngineHandle, requests: Vec<(Vec<u32>, usize)>) -> Vec<Vec<Step>> {
    let receivers: Vec<_> = requests
        .into_iter()
        .map(|(prompt_tokens, max_tokens)| {
            let (token_tx, rx) = TokenSink::standalone();
            handle
                .submit(GenerateRequest {
                    request_id: None,
                    queued_at_unix_s: None,
                    prompt_tokens,
                    params: greedy_params(),
                    max_tokens,
                    lora_adapter: None,
                    token_tx,
                    logprobs: 0,
                    echo: false,
                })
                .expect("submit failed");
            rx
        })
        .collect();

    receivers
        .into_iter()
        .map(|mut rx| collect_steps(&mut rx))
        .collect()
}

fn collect_steps(rx: &mut openinfer_core::engine::TokenStreamReceiver) -> Vec<Step> {
    let mut steps = Vec::new();
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, logprob }) => steps.push(Step {
                id,
                top_logprobs: logprob.map(|lp| lp.top_logprobs).unwrap_or_default(),
            }),
            Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
            Some(TokenEvent::Finished { .. }) => return steps,
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

fn prefill_next(handle: &EngineHandle, context: Vec<u32>) -> Step {
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens: context,
            params: greedy_params(),
            max_tokens: 1,
            lora_adapter: None,
            token_tx,
            logprobs: LOGPROBS,
            echo: true,
        })
        .expect("submit failed");

    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, logprob }) => {
                return Step {
                    id,
                    top_logprobs: logprob.map(|lp| lp.top_logprobs).unwrap_or_default(),
                };
            }
            Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
            Some(TokenEvent::Finished { .. }) => panic!("prefill_next finished without a token"),
            Some(TokenEvent::Error { message, .. }) => panic!("prefill_next failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => {
                panic!("prefill_next rejected: {message}")
            }
            None => panic!("scheduler channel closed without prefill_next token"),
        }
    }
}

fn check_lossless(
    handle: &EngineHandle,
    tokenizer: &DynTokenizer,
    label: &str,
    prompt_tokens: &[u32],
    base: &[Step],
    spec: &[Step],
) -> Result<(), String> {
    let matched = base
        .iter()
        .zip(spec.iter())
        .take_while(|(b, s)| b.id == s.id)
        .count();
    if matched == base.len().min(spec.len()) {
        eprintln!("{label}: {matched}/{} tokens identical", base.len());
        return Ok(());
    }
    if matched >= spec.len() {
        return Err(format!(
            "{label}: speculative output ended before baseline at token {matched}"
        ));
    }
    if base[matched].top_logprobs.is_empty() {
        return Err(format!(
            "{label}: missing baseline logprobs at first divergence {matched}"
        ));
    }

    let spec_id = spec[matched].id;
    let decode_argmax = base[matched].top_logprobs[0].0;
    let mut context = prompt_tokens.to_vec();
    context.extend(base[..matched].iter().map(|step| step.id));
    let prefill_ref = prefill_next(handle, context);

    if prefill_ref.id == spec_id {
        eprintln!(
            "{label}: prefill/decode kernel-gap flip at token {matched}; spec matches prefill greedy"
        );
        return Ok(());
    }

    let prefill_regret = prefill_ref
        .top_logprobs
        .iter()
        .find(|(token, _)| *token == spec_id)
        .map(|(_, lp)| prefill_ref.top_logprobs[0].1 - lp);
    if prefill_regret.is_some_and(|regret| regret <= MARGIN_TOL) {
        eprintln!(
            "{label}: near-tie at token {matched}; regret {:.3} <= {MARGIN_TOL}",
            prefill_regret.unwrap()
        );
        return Ok(());
    }

    let decode_regret = base[matched]
        .top_logprobs
        .iter()
        .find(|(token, _)| *token == spec_id)
        .map(|(_, lp)| base[matched].top_logprobs[0].1 - lp);
    let lo = matched.saturating_sub(2);
    let hi = (matched + 4).min(base.len()).min(spec.len());
    let base_ids: Vec<u32> = base[lo..hi].iter().map(|step| step.id).collect();
    let spec_ids: Vec<u32> = spec[lo..hi].iter().map(|step| step.id).collect();
    Err(format!(
        "{label}: real divergence at token {matched}: spec={spec_id}, prefill_argmax={}, decode_argmax={decode_argmax}, prefill_regret={prefill_regret:?}, decode_regret={decode_regret:?}, base_window={:?} ({:?}), spec_window={:?} ({:?})",
        prefill_ref.id,
        base_ids,
        tokenizer.decode(&base_ids, false).unwrap_or_default(),
        spec_ids,
        tokenizer.decode(&spec_ids, false).unwrap_or_default(),
    ))
}

fn long_prompt(seed: &str) -> String {
    format!(
        "{seed}\n{}\n{}\n{}\n{}\n{}\n{}",
        "Explain the implementation carefully, include state ownership, scheduler batching, and why every accepted token must be target-verified.",
        "Use compact technical prose with concrete examples and avoid changing topic.",
        "Then continue with a deterministic continuation that has enough context for a draft model to use hidden-state features.",
        "Repeat the key point: KV cache, recurrent state, and convolution state must move together.",
        "Describe the rollback path, the replay path, the fixed verification buffers, and the reason heterogeneous output budgets can shorten verify spans.",
        "Finally, restate the same mechanism in a second paragraph with slightly different wording so the prompt is long enough to exercise DFlash capture."
    )
}

fn synthetic_random_prompt(len: usize, seed: u64, request_idx: usize) -> Vec<u32> {
    let mut rng =
        StdRng::seed_from_u64(seed ^ (request_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (0..len)
        .map(|_| rng.random_range(SYNTHETIC_TOKEN_LO..SYNTHETIC_TOKEN_HI))
        .collect()
}

#[test]
fn qwen35_dflash_single_and_concurrent_greedy_are_lossless() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let tokenizer = common::load_tokenizer(&model_path);

    let cases: Vec<(String, usize)> = vec![
        (
            long_prompt("Write a Rust function for paged attention."),
            48,
        ),
        (long_prompt("Summarize a GPU scheduler benchmark."), 32),
        (
            long_prompt("Explain speculative decoding for a hybrid model."),
            40,
        ),
        (
            long_prompt("Draft a short guide for CUDA Graph verification."),
            24,
        ),
    ];
    let encoded: Vec<Vec<u32>> = cases
        .iter()
        .map(|(prompt, _)| tokenizer.encode(prompt, false).expect("encode failed"))
        .collect();
    for (idx, tokens) in encoded.iter().enumerate() {
        assert!(
            tokens.len() >= 128,
            "case {idx} prompt must exceed the DFlash capture threshold, got {} tokens",
            tokens.len()
        );
    }

    let baselines: Vec<Vec<Step>> = {
        let handle = launch(&model_path, None);
        let out = encoded
            .iter()
            .zip(cases.iter())
            .map(|(tokens, (_, max_tokens))| {
                generate(&handle, tokens.clone(), LOGPROBS, *max_tokens)
            })
            .collect();
        drop(handle);
        std::thread::sleep(Duration::from_secs(2));
        out
    };

    let handle = launch(&model_path, Some(PathBuf::from(&draft_path)));
    let single_spec = generate(&handle, encoded[0].clone(), 0, cases[0].1);
    let concurrent_specs = generate_concurrent(
        &handle,
        encoded
            .iter()
            .zip(cases.iter())
            .map(|(tokens, (_, max_tokens))| (tokens.clone(), *max_tokens))
            .collect(),
    );

    let mut failures = Vec::new();
    if let Err(err) = check_lossless(
        &handle,
        &tokenizer,
        "single",
        &encoded[0],
        &baselines[0],
        &single_spec,
    ) {
        failures.push(err);
    }
    for (idx, spec) in concurrent_specs.iter().enumerate() {
        if let Err(err) = check_lossless(
            &handle,
            &tokenizer,
            &format!("concurrent-{idx}"),
            &encoded[idx],
            &baselines[idx],
            spec,
        ) {
            failures.push(err);
        }
    }
    drop(handle);

    assert!(
        failures.is_empty(),
        "Qwen3.5 DFlash speculative decode is not lossless:\n{}",
        failures.join("\n")
    );
}

#[test]
fn qwen35_dflash_short_prompt_concurrent_random_is_within_oracle() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let tokenizer = common::load_tokenizer(&model_path);
    let output_len = 256;
    let prompts: Vec<Vec<u32>> = (0..MAX_BATCH)
        .map(|idx| synthetic_random_prompt(1, 0, idx))
        .collect();

    let baselines: Vec<Vec<Step>> = {
        let handle = launch(&model_path, None);
        let out = prompts
            .iter()
            .map(|tokens| generate(&handle, tokens.clone(), LOGPROBS, output_len))
            .collect();
        drop(handle);
        std::thread::sleep(Duration::from_secs(2));
        out
    };

    let handle = launch(&model_path, Some(PathBuf::from(&draft_path)));
    let specs = generate_concurrent(
        &handle,
        prompts
            .iter()
            .map(|tokens| (tokens.clone(), output_len))
            .collect(),
    );

    let mut failures = Vec::new();
    for (idx, spec) in specs.iter().enumerate() {
        if let Err(err) = check_lossless(
            &handle,
            &tokenizer,
            &format!("short-random-c{MAX_BATCH}-{idx}"),
            &prompts[idx],
            &baselines[idx],
            spec,
        ) {
            failures.push(err);
        }
    }
    drop(handle);

    assert!(
        failures.is_empty(),
        "Qwen3.5 DFlash short-prompt concurrent decode is outside the oracle:\n{}",
        failures.join("\n")
    );
}

fn check_random_concurrent_case(
    model_path: &str,
    draft_path: &str,
    tokenizer: &DynTokenizer,
    prompt_len: usize,
    concurrency: usize,
) -> Vec<String> {
    let output_len = 256;
    let prompts: Vec<Vec<u32>> = (0..concurrency)
        .map(|idx| synthetic_random_prompt(prompt_len, 42, idx))
        .collect();

    let baselines: Vec<Vec<Step>> = {
        let handle = launch(model_path, None);
        let out = prompts
            .iter()
            .map(|tokens| generate(&handle, tokens.clone(), LOGPROBS, output_len))
            .collect();
        drop(handle);
        std::thread::sleep(Duration::from_secs(2));
        out
    };

    let handle = launch(model_path, Some(PathBuf::from(draft_path)));
    let specs = generate_concurrent(
        &handle,
        prompts
            .iter()
            .map(|tokens| (tokens.clone(), output_len))
            .collect(),
    );

    let mut failures = Vec::new();
    for (idx, spec) in specs.iter().enumerate() {
        if let Err(err) = check_lossless(
            &handle,
            tokenizer,
            &format!("bench-random-p{prompt_len}-c{concurrency}-{idx}"),
            &prompts[idx],
            &baselines[idx],
            spec,
        ) {
            failures.push(err);
        }
    }
    drop(handle);
    failures
}

#[test]
fn qwen35_dflash_benchmark_random_concurrency_is_within_oracle() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let tokenizer = common::load_tokenizer(&model_path);

    let mut failures = Vec::new();
    for (prompt_len, concurrency) in [(1024, 16), (4096, 8), (4096, 16)] {
        failures.extend(check_random_concurrent_case(
            &model_path,
            &draft_path,
            &tokenizer,
            prompt_len,
            concurrency,
        ));
    }

    assert!(
        failures.is_empty(),
        "Qwen3.5 DFlash benchmark-shaped concurrent decode is outside the oracle:\n{}",
        failures.join("\n")
    );
}
