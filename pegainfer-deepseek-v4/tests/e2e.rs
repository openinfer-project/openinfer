use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::time::Instant;

use log::{LevelFilter, Metadata, Record, info};
use pegainfer_core::engine::{EngineHandle, EngineLoadOptions, GenerateRequest, TokenEvent};
use pegainfer_core::sampler::SamplingParams;
use serde::Deserialize;
use tokio::sync::mpsc;
use vllm_text::tokenizer::{HuggingFaceTokenizer, Tokenizer};

const DEFAULT_MODEL_PATH: &str = "models/DeepSeek-V4-Flash";
const DEFAULT_GROUND_TRUTH_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/deepseek-v4-ground-truth.json"
);
const DEFAULT_MAX_NEW_TOKENS: usize = 300;

struct TestLogger;

impl log::Log for TestLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: TestLogger = TestLogger;

#[derive(Deserialize)]
struct GroundTruthCase {
    question: String,
    answer: String,
}

fn get_model_path() -> PathBuf {
    let path =
        std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.into());
    info!("Using model path: {path}");
    PathBuf::from(path)
}

fn init_logging() {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(LevelFilter::Info);
    }
}

fn get_ground_truth_path() -> PathBuf {
    let path = std::env::var("PEGAINFER_DEEPSEEK_GT_PATH")
        .unwrap_or_else(|_| DEFAULT_GROUND_TRUTH_PATH.into());
    info!("Using ground truth path: {path}");
    PathBuf::from(path)
}

fn get_case_limit() -> Option<usize> {
    std::env::var("PEGAINFER_DEEPSEEK_GT_LIMIT")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("PEGAINFER_DEEPSEEK_GT_LIMIT must be a usize")
        })
}

fn get_case_offset() -> usize {
    std::env::var("PEGAINFER_DEEPSEEK_GT_OFFSET")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("PEGAINFER_DEEPSEEK_GT_OFFSET must be a usize")
        })
        .unwrap_or(0)
}

fn get_max_new_tokens() -> usize {
    std::env::var("PEGAINFER_DEEPSEEK_GT_MAX_NEW_TOKENS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("PEGAINFER_DEEPSEEK_GT_MAX_NEW_TOKENS must be a usize")
        })
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS)
}

fn load_cases(path: &Path) -> Vec<GroundTruthCase> {
    serde_json::from_reader(
        std::fs::File::open(path)
            .unwrap_or_else(|err| panic!("failed to open {}: {err}", path.display())),
    )
    .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()))
}

fn load_tokenizer(model_path: &Path) -> HuggingFaceTokenizer {
    let tokenizer_path = model_path.join("tokenizer.json");
    HuggingFaceTokenizer::new(&tokenizer_path)
        .unwrap_or_else(|err| panic!("failed to load {}: {err:?}", tokenizer_path.display()))
}

fn encode_dsv4_chat_prompt(question: &str) -> String {
    format!("<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜></think>")
}

fn exact_answer_prefix_possible(generated: &str, expected: &str) -> bool {
    generated == expected || expected.starts_with(generated)
}

fn generate_text(
    handle: &EngineHandle,
    tokenizer: &HuggingFaceTokenizer,
    prompt: &str,
    expected: &str,
    max_tokens: usize,
) -> (String, usize) {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();

    handle
        .submit(GenerateRequest {
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut out = Vec::new();
    loop {
        match token_rx.blocking_recv() {
            Some(TokenEvent::Token { id, .. }) => {
                out.push(id);
                let text = tokenizer.decode(&out, false).expect("decode failed");
                if !exact_answer_prefix_possible(&text, expected) {
                    drop(token_rx);
                    let tokens = out.len();
                    return (text, tokens);
                }
            }
            Some(TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => break,
        }
    }

    let text = tokenizer.decode(&out, false).expect("decode failed");
    let tokens = out.len();
    (text, tokens)
}

#[test]
fn test_e2e_deepseek_v4_generation() {
    init_logging();

    let model_path = get_model_path();
    let ground_truth_path = get_ground_truth_path();
    let all_cases = load_cases(&ground_truth_path);
    let offset = get_case_offset();
    let limit = get_case_limit().unwrap_or_else(|| all_cases.len().saturating_sub(offset));
    let max_new_tokens = get_max_new_tokens();
    let cases = all_cases
        .iter()
        .enumerate()
        .skip(offset)
        .take(limit)
        .collect::<Vec<_>>();
    assert!(
        !cases.is_empty(),
        "no DeepSeek V4 ground-truth cases selected"
    );

    let tokenizer = load_tokenizer(&model_path);
    info!("Loading DeepSeek V4 model...");
    let load_start = Instant::now();
    let handle = ManuallyDrop::new(
        pegainfer_deepseek_v4::start_engine(
            &model_path,
            EngineLoadOptions {
                enable_cuda_graph: false,
                device_ordinals: (0..8).collect(),
                seed: 42,
            },
        )
        .expect("Failed to start DeepSeek V4 engine"),
    );
    info!("Model loaded in {:.2?}", load_start.elapsed());

    let mut pass = 0usize;
    let mut fail = 0usize;
    for (idx, case) in cases {
        let prompt = encode_dsv4_chat_prompt(&case.question);
        let start = Instant::now();
        let (output, generated_tokens) =
            generate_text(&handle, &tokenizer, &prompt, &case.answer, max_new_tokens);
        let elapsed = start.elapsed();
        let tokens_per_second = if elapsed.as_secs_f64() > 0.0 {
            generated_tokens as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        if output == case.answer {
            info!(
                "  PASS case={idx} generated_tokens={generated_tokens} elapsed={elapsed:.2?} tokens_per_second={tokens_per_second:.2}"
            );
            pass += 1;
        } else {
            eprintln!(
                "  FAIL case={idx} generated_tokens={generated_tokens} elapsed={elapsed:.2?} tokens_per_second={tokens_per_second:.2}"
            );
            eprintln!("    question: {:?}", case.question);
            eprintln!("    expected: {:?}", case.answer);
            eprintln!("    got:      {:?}", output);
            fail += 1;
        }
    }

    assert_eq!(
        fail,
        0,
        "{fail} / {} DeepSeek V4 exact cases failed",
        pass + fail
    );
    info!("All {pass} DeepSeek V4 exact cases passed");
}
