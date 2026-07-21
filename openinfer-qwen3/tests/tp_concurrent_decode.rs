//! TP=2 launch with CUDA Graph on — when rendering tools are available, startup
//! dumps rank 0's pre-captured bs1 graph, then decode replays captured graphs
//! under concurrent serving.
//! The drain loop polls with a deadline so a deadlock fails instead of wedging the run.

use std::mem::ManuallyDrop;
use std::path::Path;
use std::process::Command;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

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

fn graph_render_tools_available() -> bool {
    let succeeds = |program: &str, args: &[&str]| {
        Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    };
    succeeds("dot", &["-Tpng:cairo"]) && succeeds("c++filt", &["--version"])
}

#[test]
fn tp2_graph_dump_when_available_and_concurrent_decode_complete() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let gpus = cuda_device_count();
    if gpus < 2 {
        eprintln!("skipping tp2 concurrent decode: needs >=2 GPUs, have {gpus}");
        return;
    }
    let dump_dir = tempfile::tempdir().expect("create graph dump directory");
    let dump_png = dump_dir.path().join("tp2-decode.png");
    let dump_enabled = graph_render_tools_available();
    if dump_enabled {
        openinfer_core::cuda_graph::validate_graph_dump_request(&dump_png)
            .expect("validate TP graph export request");
    } else {
        eprintln!("TP graph export coverage disabled: Graphviz Cairo or c++filt unavailable");
    }

    let options = Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 2,
        cuda_graph: true,
        dump_graph_png: dump_enabled.then(|| dump_png.clone()),
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
    if dump_enabled {
        let dump_dot = dump_png.with_extension("dot");
        assert!(dump_png.is_file(), "TP graph PNG was not exported");
        assert!(dump_dot.is_file(), "TP graph DOT was not exported");
        let dot = std::fs::read_to_string(&dump_dot).expect("read TP graph DOT");
        assert!(dot.contains("dynamic_shared_mem_bytes="));
        assert!(dot.contains(" -> "), "TP graph DOT has no dependency edges");
    }

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
                    data_parallel_rank: None,
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
