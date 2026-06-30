//! EAGLE-3 speculative-decode A/B on **real GSM8K prompts** (vs the synthetic
//! prompts in `eagle3_speculative_perf`).
//!
//! GSM8K is in-distribution reasoning text — the regime EAGLE-3 was trained for
//! and where chain acceptance is highest. Each prompt is the standard lm-eval
//! `gsm8k` 8-shot format (8 train Q/A exemplars + the held-out question), so the
//! generated continuation looks like the training distribution. We measure
//! single-stream (bs=1) decode tok/s with speculation OFF vs ON and report the
//! speedup; acceptance itself is logged per round at `debug` (run with
//! `RUST_LOG=openinfer_qwen3_4b=debug` to see `cumulative_accept_rate`).
//!
//! Requires a CUDA GPU, Qwen3-4B weights, the EAGLE-3 drafter, and the GSM8K
//! jsonl. Set `OPENINFER_TEST_MODEL_PATH` + `OPENINFER_EAGLE3_TEST_MODEL_PATH`,
//! and (optionally) `OPENINFER_GSM8K_DIR` (default `<crate>/../models/gsm8k`,
//! holding `train.jsonl` + `test.jsonl`). Skips cleanly when anything is absent.
//!
//! Fetch the data once:
//!   mkdir -p models/gsm8k && for f in train test; do curl -sSL \
//!     "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/$f.jsonl" \
//!     -o "models/gsm8k/$f.jsonl"; done

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use openinfer_core::engine::{EngineHandle, GenerateRequest, TokenEvent, TokenSink};
use openinfer_core::sampler::SamplingParams;
use openinfer_qwen3_4b::{
    DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES, DecodeOverlap, Qwen3LaunchOptions, Qwen3MemoryOptions,
    Qwen3OffloadOptions,
};

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B_eagle3");
const GSM8K_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/gsm8k");
/// Fixed decode budget per prompt (ignore EOS) so the A/B compares pure decode
/// throughput on identical work.
const GENERATED_TOKENS: usize = 256;
const NUM_FEWSHOT: usize = 8;
const NUM_PROMPTS: usize = 20;
/// The 8-shot prompt is ~1.3k tokens; raise the prefill-chunk cap above it so the
/// drafter captures the whole prompt in one chunk (v1 single-chunk requirement).
const MAX_PREFILL_TOKENS: usize = 4096;

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
    match std::env::var("OPENINFER_EAGLE3_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DRAFT_PATH).join("config.json").exists() => {
            Some(DRAFT_PATH.to_string())
        }
        Err(_) => None,
    }
}

fn gsm8k_dir_or_skip() -> Option<PathBuf> {
    let dir = std::env::var("OPENINFER_GSM8K_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(GSM8K_DIR));
    if dir.join("test.jsonl").exists() && dir.join("train.jsonl").exists() {
        Some(dir)
    } else {
        None
    }
}

/// Read the `question`/`answer` fields from a GSM8K jsonl.
fn read_gsm8k(path: &Path) -> Vec<(String, String)> {
    let text = std::fs::read_to_string(path).expect("read gsm8k jsonl");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).expect("parse gsm8k line");
            let q = v["question"].as_str().expect("question").to_string();
            let a = v["answer"].as_str().expect("answer").to_string();
            (q, a)
        })
        .collect()
}

/// lm-eval `gsm8k` 8-shot prefix: `Question: ..\nAnswer: ..` exemplars joined by
/// blank lines, then the held-out question with a trailing `Answer:`.
fn build_prompt(fewshot: &[(String, String)], question: &str) -> String {
    let mut s = String::new();
    for (q, a) in fewshot {
        s.push_str(&format!("Question: {q}\nAnswer: {a}\n\n"));
    }
    s.push_str(&format!("Question: {question}\nAnswer:"));
    s
}

fn launch_options(draft: Option<PathBuf>) -> Qwen3LaunchOptions {
    Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        offload: Qwen3OffloadOptions::disabled(),
        no_prefix_cache: true,
        max_prefill_tokens: MAX_PREFILL_TOKENS,
        memory: Qwen3MemoryOptions::new(0.85, DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES)
            .validate()
            .expect("valid memory options"),
        lora: None,
        decode_overlap: DecodeOverlap::Off,
        batch_invariant: false,
        dflash_draft_model_path: None,
        eagle3_draft_model_path: draft,
        enable_kv_events: false,
    }
}

/// Generate `GENERATED_TOKENS` greedily (ignore EOS) and return (count, elapsed).
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
    let _ = timed_generate(handle, prompts[0].clone()); // warm up CUDA-graph capture
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
fn eagle3_gsm8k_single_stream_speedup() {
    let (Some(model_path), Some(draft_path), Some(gsm8k_dir)) = (
        target_path_or_skip(),
        draft_path_or_skip(),
        gsm8k_dir_or_skip(),
    ) else {
        eprintln!(
            "skipping eagle3 GSM8K A/B: set OPENINFER_TEST_MODEL_PATH + OPENINFER_EAGLE3_TEST_MODEL_PATH \
             and provide models/gsm8k/{{train,test}}.jsonl (or OPENINFER_GSM8K_DIR)"
        );
        return;
    };

    let tokenizer = common::load_tokenizer(&model_path);
    let train = read_gsm8k(&gsm8k_dir.join("train.jsonl"));
    let test = read_gsm8k(&gsm8k_dir.join("test.jsonl"));
    let fewshot: Vec<(String, String)> = train.iter().take(NUM_FEWSHOT).cloned().collect();

    let prompts: Vec<Vec<u32>> = test
        .iter()
        .take(NUM_PROMPTS)
        .map(|(q, _)| {
            let text = build_prompt(&fewshot, q);
            tokenizer.encode(&text, false).expect("encode failed")
        })
        .collect();
    let avg_prompt_len =
        prompts.iter().map(|p| p.len()).sum::<usize>() as f64 / prompts.len() as f64;
    eprintln!(
        "GSM8K A/B: {NUM_PROMPTS} prompts, {NUM_FEWSHOT}-shot, avg prompt {avg_prompt_len:.0} tokens"
    );

    let baseline_tps = {
        let handle = openinfer_qwen3_4b::launch(Path::new(&model_path), launch_options(None))
            .expect("baseline engine");
        let tps = measure(&handle, &prompts);
        drop(handle);
        std::thread::sleep(Duration::from_secs(2));
        tps
    };

    let spec_tps = {
        let handle = openinfer_qwen3_4b::launch(
            Path::new(&model_path),
            launch_options(Some(PathBuf::from(&draft_path))),
        )
        .expect("speculative engine");
        measure(&handle, &prompts)
    };

    let speedup = spec_tps / baseline_tps;
    eprintln!("───────────── EAGLE-3 GSM8K single-stream decode A/B (bs=1) ─────────────");
    eprintln!("  spec OFF (plain decode): {baseline_tps:7.1} tok/s");
    eprintln!("  spec ON  (EAGLE-3):      {spec_tps:7.1} tok/s");
    eprintln!("  speedup:                 {speedup:7.2}×");
    eprintln!("─────────────────────────────────────────────────────────────────────────────");

    assert!(
        speedup > 0.5,
        "speculative decode catastrophically slower ({speedup:.2}×) on GSM8K — investigate"
    );
}
