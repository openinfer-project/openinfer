//! TP=2 launch with CUDA Graph requested — guards `launch` disabling CUDA Graph under TP.
//! The drain loop polls with a deadline so a deadlock fails instead of wedging the run.

use std::mem::ManuallyDrop;
use std::path::Path;
use std::time::{Duration, Instant};

use openinfer_core::engine::{GenerateRequest, TokenEvent, TokenSink};
use openinfer_core::sampler::SamplingParams;
use openinfer_qwen3::{
    DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES, DEFAULT_KV_PAGE_SIZE, DEFAULT_MAX_PREFILL_TOKENS,
    DecodeOverlap, Qwen3LaunchOptions, Qwen3MemoryOptions, Qwen3OffloadOptions,
};
use tokio::sync::mpsc::error::TryRecvError;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const REQUESTS: usize = 16;
const DEADLINE_SECS: u64 = 300;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping tp2 concurrent decode: {MODEL_PATH}/config.json missing; set OPENINFER_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn cuda_device_count() -> usize {
    cudarc::driver::CudaContext::device_count().map_or(0, |n| n.max(0) as usize)
}

#[test]
fn tp2_concurrent_decode_completes() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let gpus = cuda_device_count();
    if gpus < 2 {
        eprintln!("skipping tp2 concurrent decode: needs >=2 GPUs, have {gpus}");
        return;
    }

    let options = Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 2,
        cuda_graph: true,
        offload: Qwen3OffloadOptions::disabled(),
        no_prefix_cache: false,
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
        dflash_draft_model_path: None,
        enable_kv_events: false,
    };
    // Dropping the handle joins the scheduler thread; on a panic the engine may be
    // wedged and the join would hang, so panics leak it — only the happy path drops.
    let handle = ManuallyDrop::new(
        openinfer_qwen3::launch(Path::new(&model_path), options).expect("launch tp2 engine"),
    );

    let tokenizer = common::load_tokenizer(&model_path);
    // Submit all up front so they coexist in the engine and form real decode batches.
    let receivers: Vec<_> = (0..REQUESTS)
        .map(|i| {
            let prompt = format!("Write a few sentences about topic {i}:");
            let prompt_tokens = tokenizer.encode(&prompt, false).expect("encode failed");
            let (token_tx, rx) = TokenSink::standalone();
            handle
                .submit(GenerateRequest {
                    request_id: None,
                    queued_at_unix_s: None,
                    prompt_tokens,
                    params: SamplingParams::default(),
                    max_tokens: 24 + (i % 4) * 24,
                    lora_adapter: None,
                    token_tx,
                    logprobs: 0,
                    echo: false,
                })
                .expect("submit failed");
            rx
        })
        .collect();

    let deadline = Instant::now() + Duration::from_secs(DEADLINE_SECS);
    for (i, mut rx) in receivers.into_iter().enumerate() {
        let mut tokens = 0usize;
        loop {
            match rx.try_recv() {
                Ok((_, TokenEvent::Token { .. })) => tokens += 1,
                Ok((_, TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. })) => {}
                Ok((_, TokenEvent::Finished { .. })) => break,
                Ok((_, TokenEvent::Error { message, .. })) => {
                    panic!("request {i} failed: {message}")
                }
                Ok((_, TokenEvent::Rejected { message, .. })) => {
                    panic!("request {i} rejected: {message}")
                }
                Err(TryRecvError::Empty) => {
                    assert!(
                        Instant::now() < deadline,
                        "request {i}: no progress within {DEADLINE_SECS}s, engine deadlocked"
                    );
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(TryRecvError::Disconnected) => {
                    panic!("request {i}: channel closed without Finished")
                }
            }
        }
        assert!(tokens > 0, "request {i} finished with zero decoded tokens");
    }
    drop(ManuallyDrop::into_inner(handle));
}
