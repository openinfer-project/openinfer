//! PegaInfer-owned deterministic rand/hash regression corpus for Qwen3.5-4B.
//!
//! This is intentionally separate from the HF logits gate. HF owns the small
//! external oracle surface; this corpus is a broad PegaInfer baseline after that
//! oracle surface is accepted. It compares generated token ids first and keeps
//! token SHA256 as the compact review signal.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use pegainfer_core::engine::{EngineHandle, EngineLoadOptions, GenerateRequest, TokenEvent};
use pegainfer_core::sampler::SamplingParams;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
const CORPUS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/qwen35-4b-rand-hash.json"
);

const SEED: u64 = 0x5EED_3535;
const NUM_CASES: usize = 32;
const MIN_PROMPT_LEN: usize = 1;
const MAX_PROMPT_LEN: usize = 192;
const MAX_NEW_TOKENS: usize = 16;
const VOCAB_CEILING: u32 = 100_000;
const ENGINE_SEED: u64 = 42;
const REQUEST_TIMEOUT_SECS: u64 = 120;

#[derive(Debug, Deserialize, Serialize)]
struct Corpus {
    schema_version: u32,
    fixture_kind: String,
    model_name: String,
    producer: String,
    producer_commit: String,
    config_sha256: String,
    model_revision: String,
    seed: String,
    engine_seed: u64,
    prompt_token_vocab_ceiling: u32,
    prompt_len_range: [usize; 2],
    max_new_tokens: usize,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Case {
    name: String,
    prompt_tokens: Vec<u32>,
    generated_token_ids: Vec<u32>,
    token_sha256: String,
}

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen35 rand_hash_regression: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

fn sha256_file(path: impl AsRef<Path>) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut digest = Sha256::new();
    digest.update(bytes);
    Some(
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    )
}

fn model_revision(model_path: &str) -> Option<String> {
    if let Ok(value) = std::env::var("PEGAINFER_TEST_MODEL_REVISION") {
        return Some(value);
    }
    let path = Path::new(model_path);
    let metadata_path = path
        .join(".cache")
        .join("huggingface")
        .join("download")
        .join("config.json.metadata");
    if let Ok(content) = std::fs::read_to_string(metadata_path) {
        if let Some(first) = content.lines().next().map(str::trim) {
            if !first.is_empty() {
                return Some(first.to_string());
            }
        }
    }
    if path.join(".git").exists() {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(path)
            .arg("rev-parse")
            .arg("HEAD")
            .output()
            .ok()?;
        if output.status.success() {
            return Some(String::from_utf8_lossy(&output.stdout).trim().to_string());
        }
    }
    let parts: Vec<_> = path.components().collect();
    for window in parts.windows(2) {
        if window[0].as_os_str() == "snapshots" {
            return Some(window[1].as_os_str().to_string_lossy().to_string());
        }
    }
    None
}

fn token_sha256(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn producer_commit() -> String {
    std::process::Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn random_prompts() -> Vec<Vec<u32>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..NUM_CASES)
        .map(|_| {
            let len = rng.random_range(MIN_PROMPT_LEN..=MAX_PROMPT_LEN);
            (0..len)
                .map(|_| rng.random_range(1..VOCAB_CEILING))
                .collect()
        })
        .collect()
}

fn start_handle(model_path: &str) -> EngineHandle {
    pegainfer_qwen35_4b::start_engine(
        Path::new(model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            enable_prefill_profile: false,
            device_ordinals: vec![0],
            seed: ENGINE_SEED,
        },
    )
    .expect("start Qwen3.5 engine")
}

fn generate_tokens(handle: &EngineHandle, request_id: usize, prompt_tokens: Vec<u32>) -> Vec<u32> {
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
    handle
        .submit(GenerateRequest {
            request_id: Some(format!("rand_hash_{request_id:03}")),
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens: MAX_NEW_TOKENS,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit Qwen3.5 rand/hash request");

    let mut out = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(REQUEST_TIMEOUT_SECS);
    loop {
        match token_rx.try_recv() {
            Ok(TokenEvent::Token { id, .. }) => out.push(id),
            Ok(TokenEvent::PromptTokens { .. }) => {}
            Ok(TokenEvent::Scheduled { .. }) => {}
            Ok(TokenEvent::Finished { .. }) => break,
            Ok(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Ok(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                assert!(
                    Instant::now() < deadline,
                    "Qwen3.5 rand/hash request {request_id} timed out after {REQUEST_TIMEOUT_SECS}s"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                panic!("Qwen3.5 scheduler channel closed")
            }
        }
    }
    out
}

fn build_corpus(model_path: &str) -> Corpus {
    let config_sha256 = sha256_file(PathBuf::from(model_path).join("config.json"))
        .expect("cannot hash Qwen3.5 config.json");
    let model_revision =
        model_revision(model_path).expect("cannot determine Qwen3.5 model revision for corpus");
    let handle = start_handle(model_path);
    let cases = random_prompts()
        .into_iter()
        .enumerate()
        .map(|(idx, prompt_tokens)| {
            let generated_token_ids = generate_tokens(&handle, idx, prompt_tokens.clone());
            let token_sha256 = token_sha256(&generated_token_ids);
            Case {
                name: format!("rand_{idx:03}"),
                prompt_tokens,
                generated_token_ids,
                token_sha256,
            }
        })
        .collect();

    Corpus {
        schema_version: 1,
        fixture_kind: "pegainfer-qwen35-rand-hash-regression".to_string(),
        model_name: "Qwen3.5-4B".to_string(),
        producer: "pegainfer".to_string(),
        producer_commit: producer_commit(),
        config_sha256,
        model_revision,
        seed: format!("0x{SEED:016x}"),
        engine_seed: ENGINE_SEED,
        prompt_token_vocab_ceiling: VOCAB_CEILING,
        prompt_len_range: [MIN_PROMPT_LEN, MAX_PROMPT_LEN],
        max_new_tokens: MAX_NEW_TOKENS,
        cases,
    }
}

fn load_corpus() -> Corpus {
    let content = std::fs::read_to_string(CORPUS).unwrap_or_else(|err| {
        panic!(
            "failed to read Qwen3.5 rand/hash corpus {CORPUS}: {err}; run ignored test regen_qwen35_rand_hash_corpus"
        )
    });
    serde_json::from_str(&content).expect("parse Qwen3.5 rand/hash corpus")
}

fn assert_model_matches_corpus(model_path: &str, corpus: &Corpus) -> bool {
    let actual = sha256_file(PathBuf::from(model_path).join("config.json"))
        .expect("cannot hash Qwen3.5 config.json");
    assert_eq!(
        actual, corpus.config_sha256,
        "Qwen3.5 config hash differs from rand/hash corpus; regenerate the corpus for this model/config"
    );
    assert_ne!(
        corpus.model_revision, "unknown",
        "Qwen3.5 rand/hash corpus must record a pinned model_revision"
    );
    let Some(actual_revision) = model_revision(model_path) else {
        eprintln!(
            "skipping qwen35 rand_hash_regression: corpus requires model_revision={}, but local model revision is unknown",
            corpus.model_revision
        );
        return false;
    };
    assert_eq!(
        actual_revision, corpus.model_revision,
        "Qwen3.5 model revision differs from rand/hash corpus; use the corpus model snapshot or regenerate the corpus"
    );
    true
}

#[test]
fn qwen35_rand_hash_corpus_matches_accepted_pegainfer_baseline() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let corpus = load_corpus();
    if !assert_model_matches_corpus(&model_path, &corpus) {
        return;
    }

    let handle = start_handle(&model_path);
    for (idx, case) in corpus.cases.iter().enumerate() {
        let got = generate_tokens(&handle, idx, case.prompt_tokens.clone());
        let got_hash = token_sha256(&got);
        assert_eq!(
            got_hash, case.token_sha256,
            "{} token hash mismatch",
            case.name
        );
        assert_eq!(
            got, case.generated_token_ids,
            "{} generated token ids changed",
            case.name
        );
    }
}

#[test]
#[ignore = "regenerates the accepted Qwen3.5 PegaInfer rand/hash regression corpus"]
fn regen_qwen35_rand_hash_corpus() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let corpus = build_corpus(&model_path);
    let output = serde_json::to_string_pretty(&corpus).expect("serialize corpus");
    let path = Path::new(CORPUS);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create test_data directory");
    }
    std::fs::write(path, format!("{output}\n")).expect("write corpus");
    eprintln!("wrote {CORPUS}");
}
