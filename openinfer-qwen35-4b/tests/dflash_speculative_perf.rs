//! DFlash speculative-decoding single-stream latency A/B for Qwen3.5.
//!
//! This mirrors the Qwen3 DFlash perf harness: fixed 256-token greedy decode,
//! speculative OFF vs ON, same prompts and hardware, one warm-up discarded.
//! Qwen3.5 is a hybrid 24-linear + 8-full-attention model, so this harness is
//! the explicit evidence source for the single-stream boundary instead of
//! inferring it from the concurrent throughput sweep.
//!
//! Requires CUDA, Qwen3.5 target weights, and the Qwen3.5 DFlash drafter. Set
//! `OPENINFER_TEST_MODEL_PATH` and `OPENINFER_DFLASH_TEST_MODEL_PATH`; skips
//! when either model is unavailable. Use `--nocapture` to read the numbers.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, GenerateRequest, TokenEvent, TokenSink,
};
use openinfer_core::sampler::SamplingParams;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B-DFlash");
const GENERATED_TOKENS: usize = 256;
const MAX_BATCH: usize = 16;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn target_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!("skipping qwen35 DFlash perf A/B: set OPENINFER_TEST_MODEL_PATH");
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
            eprintln!("skipping qwen35 DFlash perf A/B: set OPENINFER_DFLASH_TEST_MODEL_PATH");
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

fn timed_generate(handle: &EngineHandle, prompt_tokens: Vec<u32>) -> (usize, Duration) {
    let (token_tx, mut rx) = TokenSink::standalone();
    let start = Instant::now();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams {
                ignore_eos: true,
                ..SamplingParams::default()
            },
            max_tokens: GENERATED_TOKENS,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut count = 0usize;
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { .. }) => count += 1,
            Some(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => return (count, start.elapsed()),
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

fn measure(handle: &EngineHandle, prompts: &[Vec<u32>]) -> f64 {
    let _ = timed_generate(handle, prompts[0].clone());
    let mut tokens = 0usize;
    let mut elapsed = Duration::ZERO;
    for prompt in prompts {
        let (n, dt) = timed_generate(handle, prompt.clone());
        tokens += n;
        elapsed += dt;
    }
    tokens as f64 / elapsed.as_secs_f64()
}

#[test]
fn qwen35_dflash_single_stream_speedup() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let tokenizer = common::load_tokenizer(&model_path);
    let prompts: Vec<Vec<u32>> = [
        "Write a compact explanation of Qwen3.5 hybrid recurrent state.",
        "Explain speculative decoding for a model with recurrent and KV state.",
        "List the tradeoffs of batched verification in a serving scheduler.",
    ]
    .iter()
    .map(|prompt| tokenizer.encode(prompt, false).expect("encode failed"))
    .collect();

    let baseline_tps = {
        let handle = launch(&model_path, None);
        let tps = measure(&handle, &prompts);
        drop(handle);
        std::thread::sleep(Duration::from_secs(2));
        tps
    };

    let spec_tps = {
        let handle = launch(&model_path, Some(PathBuf::from(&draft_path)));
        measure(&handle, &prompts)
    };

    let speedup = spec_tps / baseline_tps;
    eprintln!("──────── Qwen3.5 DFlash single-stream decode A/B (bs=1) ────────");
    eprintln!("  spec OFF (plain decode): {baseline_tps:7.1} tok/s");
    eprintln!("  spec ON  (DFlash):       {spec_tps:7.1} tok/s");
    eprintln!("  speedup:                 {speedup:7.2}×");
    eprintln!("────────────────────────────────────────────────────────────────");

    assert!(
        speedup > 0.8,
        "Qwen3.5 DFlash single-stream is catastrophically slower ({speedup:.2}×)"
    );
}
