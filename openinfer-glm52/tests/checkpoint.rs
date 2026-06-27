use std::path::Path;
use std::time::Instant;

use openinfer_core::{
    engine::{EpBackend, FinishReason, GenerateRequest, TokenEvent, TokenSink},
    sampler::SamplingParams,
};
use openinfer_glm52::{GLM52_VOCAB, Glm52LaunchOptions, launch};

const JIUZHANG_GLM52_MODEL_PATH: &str = "/data/models/GLM-5.2-FP8";

/// End-to-end bs=1 decode on the real 8-stage checkpoint: load, greedily decode a
/// short prompt, and report TPOT (the inter-token wall-clock past prefill). The
/// per-token target is < 10ms. Runs only with the checkpoint + 8 GPUs present.
///   cargo test --release -p openinfer-glm52 --test checkpoint -- --ignored --nocapture
#[test]
#[ignore]
fn jiuzhang_checkpoint_decodes_bs1() {
    openinfer_core::logging::init_default();

    let model_path = Path::new(JIUZHANG_GLM52_MODEL_PATH);
    assert!(
        model_path.join("model.safetensors.index.json").exists(),
        "GLM5.2 checkpoint missing at {}",
        model_path.display()
    );

    let handle = launch(
        model_path,
        Glm52LaunchOptions {
            tp_size: 1,
            dp_size: 8,
            ep_backend: EpBackend::DeepEp,
            cuda_graph: true,
        },
    )
    .expect("GLM5.2 checkpoint startup");

    // Arbitrary valid prompt token ids; greedy decode, fixed output budget.
    let prompt: Vec<u32> = vec![100, 2048, 9001, 12345, 64, 777, 4096, 31415];
    let max_tokens = 48usize;

    let (token_tx, mut token_rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: Some("glm52-decode-bs1".to_string()),
            queued_at_unix_s: None,
            prompt_tokens: prompt.clone(),
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit GLM5.2 decode request");

    let Some((_, TokenEvent::Scheduled { prompt_tokens, .. })) = token_rx.blocking_recv() else {
        panic!("GLM5.2 decode request was not scheduled");
    };
    assert_eq!(prompt_tokens, prompt.len());

    let mut tokens: Vec<u32> = Vec::new();
    let mut token_arrivals: Vec<f64> = Vec::new();
    let start = Instant::now();
    let mut ttft_ms = None;
    let finish_reason = loop {
        match token_rx.blocking_recv() {
            Some((_, TokenEvent::Token { id, .. })) => {
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                if ttft_ms.is_none() {
                    ttft_ms = Some(elapsed);
                }
                token_arrivals.push(elapsed);
                tokens.push(id);
            }
            Some((_, TokenEvent::Finished { finish_reason, .. })) => break finish_reason,
            Some((_, TokenEvent::Error { message, .. })) => {
                panic!("GLM5.2 decode error: {message}")
            }
            Some((_, TokenEvent::Rejected { message, .. })) => {
                panic!("GLM5.2 decode rejected: {message}")
            }
            Some(_) => continue, // Scheduled / PromptTokens — not part of TPOT
            None => panic!("GLM5.2 token channel closed before finish"),
        }
    };

    assert!(!tokens.is_empty(), "GLM5.2 produced no tokens");
    assert!(
        tokens.iter().all(|&id| (id as usize) < GLM52_VOCAB),
        "GLM5.2 produced an out-of-vocab token id"
    );
    assert!(
        matches!(finish_reason, FinishReason::Length | FinishReason::Stop),
        "GLM5.2 decode finished with {finish_reason:?}"
    );

    // TPOT = inter-token interval past the first generated token (the first
    // interval still carries decode-style prefill; later ones are pure decode).
    let mut itl: Vec<f64> = token_arrivals.windows(2).map(|w| w[1] - w[0]).collect();
    itl.sort_by(|a, b| a.total_cmp(b));
    let median = if itl.is_empty() {
        f64::NAN
    } else {
        itl[itl.len() / 2]
    };
    let p99 = if itl.is_empty() {
        f64::NAN
    } else {
        itl[(itl.len() as f64 * 0.99) as usize % itl.len()]
    };

    println!(
        "GLM5.2 bs=1 decode: prompt={} generated={} finish={finish_reason:?} TTFT={:.1}ms TPOT median={:.2}ms p99={:.2}ms",
        prompt.len(),
        tokens.len(),
        ttft_ms.unwrap_or(f64::NAN),
        median,
        p99,
    );
    println!(
        "GLM5.2 first 16 tokens: {:?}",
        &tokens[..tokens.len().min(16)]
    );

    assert!(
        median < 10.0,
        "GLM5.2 bs=1 TPOT median {median:.2}ms exceeds the 10ms target"
    );
}
