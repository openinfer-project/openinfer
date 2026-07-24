//! Cancellation-sweep integration test for Qwen3-4B (issue #642).
//!
//! Issue #642 added a proactive cancellation sweep to the Qwen3 scheduler that
//! drops cancelled requests (their `token_tx` closed) before they ever reach
//! prefill. The scheduler also republishes load metrics *after* that sweep, so
//! a disconnected request batch is visible as a collapse to zero on the load
//! watch before the loop parks idle. This test proves both: when a whole batch
//! of clients disconnects mid-flight, the engine's load metrics
//! (`num_running_reqs`, `num_waiting_reqs`) collapse to zero promptly, and the
//! engine still serves a fresh request immediately after.
//!
//! It drives the real engine + `submit` rather than a mocked scheduler, so it
//! exercises the actual send-failure retirement and pre-schedule sweep paths.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when the model is
//! absent (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use openinfer_core::engine::EngineHandle;
use openinfer_core::engine::EngineLoadOptions;
use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::TokenEvent;
use openinfer_core::engine::TokenSink;
use openinfer_core::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

/// Number of concurrent requests in the disconnect burst.
const BURST_SIZE: usize = 10;
/// Prompt length (tokens) per burst request — long enough that only ~1 prefills
/// per scheduler step (`DEFAULT_MAX_PREFILL_TOKENS` = 1024), so a mix of running
/// and waiting requests is in flight when the disconnect happens.
const PROMPT_TOKENS: usize = 600;
/// Decode budget per burst request — keeps them active long enough to be in
/// flight at disconnect time.
const BURST_MAX_TOKENS: usize = 128;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 cancellation_sweep: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
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
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
            data_parallel_rank: None,
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

/// A mass client disconnect must not leave cancelled requests occupying running
/// or waiting slots: the prefill cancellation sweep (issue #642) drops them
/// before prefill, and the scheduler retires in-flight ones when their sends
/// fail. We assert the live load metrics collapse to zero within a tight bound,
/// then prove the engine still serves a fresh request.
#[test]
fn cancelled_burst_metrics_collapse_promptly() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = openinfer_qwen3::start_engine_with_offload(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
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
    let tokenizer = common::load_tokenizer(&model_path);

    // Build a long prompt and trim it to a known token length so each request
    // lands in prefill/waiting rather than retiring instantly.
    let long_prompt = "The quick brown fox jumps over the lazy dog while a gentle breeze rustles the autumn leaves. ".repeat(64);
    let prompt_tokens = {
        let mut t = tokenizer
            .encode(&long_prompt, false)
            .expect("encode failed");
        assert!(
            t.len() >= PROMPT_TOKENS,
            "base prompt only tokenized to {} tokens; need >= {PROMPT_TOKENS}",
            t.len()
        );
        t.truncate(PROMPT_TOKENS);
        t
    };

    // Submit a burst of concurrent requests, keeping every receiver alive so the
    // scheduler admits and starts processing them.
    let mut receivers = Vec::with_capacity(BURST_SIZE);
    for _ in 0..BURST_SIZE {
        let (token_tx, rx) = TokenSink::standalone();
        handle
            .submit(GenerateRequest {
                request_id: None,
                queued_at_unix_s: None,
                prompt_tokens: prompt_tokens.clone(),
                params: SamplingParams::default(),
                max_tokens: BURST_MAX_TOKENS,
                lora_adapter: None,
                token_tx,
                logprobs: 0,
                echo: false,
                data_parallel_rank: None,
            })
            .expect("submit failed");
        receivers.push(rx);
    }

    // Let some requests enter prefilling/active so the disconnect hits real
    // in-flight work, not a still-empty queue.
    std::thread::sleep(Duration::from_millis(200));

    let load_rx = handle
        .load_watch()
        .expect("engine did not wire a load feed");
    let pre_drop = *load_rx.borrow();
    eprintln!(
        "[cancellation-sweep] pre-drop: running={} waiting={} kv_used={}",
        pre_drop.num_running_reqs, pre_drop.num_waiting_reqs, pre_drop.kv_used_blocks
    );
    // The burst is large and slow enough that something must be in flight at
    // disconnect time; if not, the test would exercise nothing.
    assert!(
        pre_drop.num_running_reqs + pre_drop.num_waiting_reqs > 0,
        "no burst requests were in flight at disconnect time; test exercised nothing"
    );

    // Mass disconnect: dropping every receiver closes every `token_tx`, so
    // `token_tx.is_closed()` becomes true for all requests.
    drop(receivers);

    // Poll the live load metrics until both running and waiting collapse to
    // zero. The sweep + send-failure retirement (and the post-sweep publish)
    // should make this fast.
    let collapse_start = Instant::now();
    let deadline = collapse_start + Duration::from_secs(10);
    let mut last = *load_rx.borrow();
    while last.num_running_reqs != 0 || last.num_waiting_reqs != 0 {
        assert!(
            Instant::now() < deadline,
            "cancelled-request metrics did not collapse within 10s: running={} waiting={}",
            last.num_running_reqs,
            last.num_waiting_reqs
        );
        std::thread::sleep(Duration::from_millis(50));
        last = *load_rx.borrow();
    }
    eprintln!(
        "[cancellation-sweep] collapsed to zero in {:.0}ms (kv_used={})",
        collapse_start.elapsed().as_secs_f64() * 1000.0,
        last.kv_used_blocks
    );

    // The engine must still serve a fresh, live request immediately after the
    // sweep cleared the cancelled burst.
    let follow_up = generate_text(&handle, &tokenizer, "Hello, how are you today?", 128);
    assert!(
        !follow_up.is_empty(),
        "engine did not serve a follow-up request after cancellation sweep"
    );
    eprintln!("[cancellation-sweep] follow-up reply: {follow_up:?}");
}
