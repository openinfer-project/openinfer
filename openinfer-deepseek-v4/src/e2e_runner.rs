use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use log::info;
use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, GenerateRequest, TokenEvent, TokenSink, TokenStreamReceiver,
};
use openinfer_core::sampler::SamplingParams;
use serde::Deserialize;
use vllm_text::tokenizer::{HuggingFaceTokenizer, Tokenizer};

pub const DEFAULT_MODEL_PATH: &str = "models/DeepSeek-V4-Flash";
pub const DEFAULT_GROUND_TRUTH_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/deepseek-v4-ground-truth.json"
);
pub const DEFAULT_MAX_NEW_TOKENS: usize = 300;

#[derive(Debug, Clone)]
pub struct E2eOptions {
    pub model_path: PathBuf,
    pub ground_truth_path: PathBuf,
    pub offset: usize,
    pub limit: Option<usize>,
    pub max_new_tokens: usize,
    pub device_ordinals: Vec<usize>,
    pub seed: u64,
    pub enable_cuda_graph: bool,
}

impl E2eOptions {
    pub fn default_paths() -> Self {
        Self {
            model_path: PathBuf::from(DEFAULT_MODEL_PATH),
            ground_truth_path: PathBuf::from(DEFAULT_GROUND_TRUTH_PATH),
            offset: 0,
            limit: None,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            device_ordinals: (0..8).collect(),
            seed: 42,
            enable_cuda_graph: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct E2eSummary {
    pub pass: usize,
    pub fail: usize,
}

#[derive(Deserialize)]
struct GroundTruthCase {
    question: String,
    answer: String,
}

struct GenerationResult {
    output: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    elapsed: Duration,
    ttft: Option<Duration>,
}

impl GenerationResult {
    fn tpot(&self) -> Option<Duration> {
        let ttft = self.ttft?;
        if self.generated_tokens <= 1 {
            return None;
        }
        self.elapsed
            .checked_sub(ttft)
            .map(|duration| duration / (self.generated_tokens - 1) as u32)
    }
}

pub fn run(options: &E2eOptions) -> Result<E2eSummary> {
    info!("Using model path: {}", options.model_path.display());
    info!(
        "Using ground truth path: {}",
        options.ground_truth_path.display()
    );

    let all_cases = load_cases(&options.ground_truth_path)?;
    let limit = options
        .limit
        .unwrap_or_else(|| all_cases.len().saturating_sub(options.offset));
    let cases = all_cases
        .iter()
        .enumerate()
        .skip(options.offset)
        .take(limit)
        .collect::<Vec<_>>();
    if cases.is_empty() {
        bail!("no DeepSeek V4 ground-truth cases selected");
    }

    let tokenizer = load_tokenizer(&options.model_path)?;
    info!("Loading DeepSeek V4 model...");
    let load_start = Instant::now();
    let handle = ManuallyDrop::new(
        crate::start_engine(
            &options.model_path,
            EngineLoadOptions {
                enable_cuda_graph: options.enable_cuda_graph,
                enable_prefill_profile: false,
                device_ordinals: options.device_ordinals.clone(),
                seed: options.seed,
                ..EngineLoadOptions::default()
            },
        )
        .with_context(|| {
            format!(
                "failed to start DeepSeek V4 engine from {}",
                options.model_path.display()
            )
        })?,
    );
    info!("Model loaded in {:.2?}", load_start.elapsed());

    let mut pass = 0usize;
    let mut fail = 0usize;
    for (idx, case) in cases {
        let prompt = encode_dsv4_chat_prompt(&case.question);
        let result = generate_text(
            &handle,
            &tokenizer,
            &prompt,
            &case.answer,
            options.max_new_tokens,
        )
        .with_context(|| format!("generation failed for ground-truth case {idx}"))?;

        if result.output == case.answer {
            info!(
                "  PASS case={idx} prompt_tokens={} generated_tokens={} ttft={} tpot={} elapsed={:.2?}",
                result.prompt_tokens,
                result.generated_tokens,
                format_optional_duration(result.ttft),
                format_optional_duration(result.tpot()),
                result.elapsed
            );
            pass += 1;
        } else {
            eprintln!(
                "  FAIL case={idx} prompt_tokens={} generated_tokens={} ttft={} tpot={} elapsed={:.2?}",
                result.prompt_tokens,
                result.generated_tokens,
                format_optional_duration(result.ttft),
                format_optional_duration(result.tpot()),
                result.elapsed
            );
            eprintln!("    question: {:?}", case.question);
            eprintln!("    expected: {:?}", case.answer);
            eprintln!("    got:      {:?}", result.output);
            fail += 1;
        }
    }

    if fail == 0 {
        info!("All {pass} DeepSeek V4 exact cases passed");
    }
    Ok(E2eSummary { pass, fail })
}

fn load_cases(path: &Path) -> Result<Vec<GroundTruthCase>> {
    serde_json::from_reader(
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))
}

fn load_tokenizer(model_path: &Path) -> Result<HuggingFaceTokenizer> {
    let tokenizer_path = model_path.join("tokenizer.json");
    HuggingFaceTokenizer::new(&tokenizer_path).map_err(|err| {
        anyhow::anyhow!(
            "failed to load tokenizer {}: {err:?}",
            tokenizer_path.display()
        )
    })
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
) -> Result<GenerationResult> {
    let prompt_tokens = tokenizer
        .encode(prompt, false)
        .map_err(|err| anyhow::anyhow!("encode failed: {err:?}"))?;
    let prompt_token_count = prompt_tokens.len();
    let (token_tx, mut token_rx) = TokenSink::standalone();
    let started = Instant::now();

    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .context("submit generation request")?;

    collect_generation_events(
        &mut token_rx,
        tokenizer,
        expected,
        started,
        prompt_token_count,
    )
}

fn collect_generation_events(
    token_rx: &mut TokenStreamReceiver,
    tokenizer: &HuggingFaceTokenizer,
    expected: &str,
    started: Instant,
    prompt_tokens: usize,
) -> Result<GenerationResult> {
    let mut out = Vec::new();
    let mut ttft = None;
    loop {
        match token_rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, .. }) => {
                ttft.get_or_insert_with(|| started.elapsed());
                out.push(id);
                let text = tokenizer
                    .decode(&out, false)
                    .map_err(|err| anyhow::anyhow!("decode failed: {err:?}"))?;
                if !exact_answer_prefix_possible(&text, expected) {
                    return Ok(GenerationResult {
                        output: text,
                        prompt_tokens,
                        generated_tokens: out.len(),
                        elapsed: started.elapsed(),
                        ttft,
                    });
                }
            }
            Some(TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Scheduled { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            Some(TokenEvent::Error { message, .. }) => bail!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => bail!("generation rejected: {message}"),
            None => break,
        }
    }

    let text = tokenizer
        .decode(&out, false)
        .map_err(|err| anyhow::anyhow!("decode failed: {err:?}"))?;
    Ok(GenerationResult {
        output: text,
        prompt_tokens,
        generated_tokens: out.len(),
        elapsed: started.elapsed(),
        ttft,
    })
}

fn format_optional_duration(duration: Option<Duration>) -> String {
    duration
        .map(|duration| format!("{duration:.2?}"))
        .unwrap_or_else(|| "n/a".to_string())
}
