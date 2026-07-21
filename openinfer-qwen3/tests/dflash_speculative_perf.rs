//! DFlash speculative-decoding single-stream latency A/B.
//!
//! Speculative decoding's direct win is single-stream (batch=1) decode latency:
//! plain decode is memory-bound (one target forward per token), while spec
//! amortizes that forward over the accepted run. This measures end-to-end
//! wall-clock to generate a fixed token budget, speculative OFF vs ON, on the
//! same prompts and hardware, and reports the speedup.
//!
//! This is a measurement harness, not a pass/fail gate — it asserts only that
//! spec is not catastrophically slower (a guard against the draft mispredicting
//! everything). Read the printed numbers for the real signal. `--nocapture`.
//!
//! Requires a CUDA GPU, Qwen3-4B weights, and the DFlash drafter. Set
//! `OPENINFER_TEST_MODEL_PATH` + `OPENINFER_DFLASH_TEST_MODEL_PATH`; skips when
//! either is absent. Single-stream only — throughput under load is a separate
//! `vllm bench serve` A/B.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use openinfer_core::engine::EngineHandle;
use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::TokenEvent;
use openinfer_core::engine::TokenSink;
use openinfer_core::sampler::SamplingParams;
use openinfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
use openinfer_qwen3::DEFAULT_KV_PAGE_SIZE;
use openinfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS;
use openinfer_qwen3::DecodeOverlap;
use openinfer_qwen3::Qwen3LaunchOptions;
use openinfer_qwen3::Qwen3MemoryOptions;
use openinfer_qwen3::Qwen3OffloadOptions;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B-DFlash-b16");
const GENERATED_TOKENS: usize = 256;

fn target_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => None,
    }
}

fn draft_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_DFLASH_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DRAFT_PATH).join("config.json").exists() => {
            Some(DRAFT_PATH.to_string())
        }
        Err(_) => None,
    }
}

fn launch_options(draft: Option<PathBuf>) -> Qwen3LaunchOptions {
    Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        dump_graph_png: None,
        offload: Qwen3OffloadOptions::disabled(),
        no_prefix_cache: true,
        max_prefill_tokens: DEFAULT_MAX_PREFILL_TOKENS,
        memory: Qwen3MemoryOptions::new(
            0.85,
            DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES,
            DEFAULT_KV_PAGE_SIZE,
        )
        .validate()
        .expect("valid memory options"),
        lora: None,
        decode_overlap: DecodeOverlap::Off,
        batch_invariant: false,
        dflash_draft_model_path: draft,
        enable_kv_events: false,
    }
}

/// Generate `GENERATED_TOKENS` greedily and return (token_count, elapsed).
fn timed_generate(handle: &EngineHandle, prompt_tokens: Vec<u32>) -> (usize, Duration) {
    let (token_tx, mut rx) = TokenSink::standalone();
    let start = Instant::now();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
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

/// Decode tok/s averaged over the prompts (one warm-up run discarded).
fn measure(handle: &EngineHandle, prompts: &[Vec<u32>]) -> f64 {
    // Warm up CUDA-graph capture / allocator on the first prompt.
    let _ = timed_generate(handle, prompts[0].clone());
    let mut tokens = 0usize;
    let mut elapsed = Duration::ZERO;
    for p in prompts {
        let (n, dt) = timed_generate(handle, p.clone());
        tokens += n;
        elapsed += dt;
    }
    tokens as f64 / elapsed.as_secs_f64()
}

#[test]
fn dflash_speculative_single_stream_speedup() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        eprintln!(
            "skipping dflash perf A/B: set OPENINFER_TEST_MODEL_PATH + OPENINFER_DFLASH_TEST_MODEL_PATH"
        );
        return;
    };

    let tokenizer = common::load_tokenizer(&model_path);
    let prompts: Vec<Vec<u32>> = [
        "Write a short essay about the history of the Roman Empire.",
        "Explain how a transformer neural network works, step by step.",
        "List ten facts about the planet Mars and describe each one.",
    ]
    .iter()
    .map(|p| tokenizer.encode(p, false).expect("encode failed"))
    .collect();

    let baseline_tps = {
        let handle = openinfer_qwen3::launch(Path::new(&model_path), launch_options(None))
            .expect("baseline engine");
        let tps = measure(&handle, &prompts);
        drop(handle);
        std::thread::sleep(Duration::from_secs(2));
        tps
    };

    let spec_tps = {
        let handle = openinfer_qwen3::launch(
            Path::new(&model_path),
            launch_options(Some(PathBuf::from(&draft_path))),
        )
        .expect("speculative engine");
        measure(&handle, &prompts)
    };

    let speedup = spec_tps / baseline_tps;
    eprintln!("───────────── DFlash single-stream decode A/B (bs=1) ─────────────");
    eprintln!("  spec OFF (plain decode): {baseline_tps:7.1} tok/s");
    eprintln!("  spec ON  (DFlash):       {spec_tps:7.1} tok/s");
    eprintln!("  speedup:                 {speedup:7.2}×");
    eprintln!("───────────────────────────────────────────────────────────────────────────");

    assert!(
        speedup > 0.8,
        "speculative decode is catastrophically slower ({speedup:.2}×) — draft likely mispredicting"
    );
}
