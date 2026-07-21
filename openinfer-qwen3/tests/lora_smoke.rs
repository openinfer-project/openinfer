use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use openinfer_core::engine::EngineHandle;
use openinfer_core::engine::EngineLoadOptions;
use openinfer_core::engine::FinishReason;
use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::LoadLoraAdapterRequest;
use openinfer_core::engine::TokenEvent;
use openinfer_core::engine::TokenSink;
use openinfer_core::sampler::SamplingParams;
use openinfer_qwen3::lora_fixtures as fixtures;
use serde::Deserialize;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

#[derive(Deserialize)]
struct ModelConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

fn get_model_path() -> String {
    std::env::var("OPENINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
}

fn get_device_ordinal() -> usize {
    std::env::var("OPENINFER_TEST_DEVICE_ORDINAL")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn load_model_config(model_path: &str) -> ModelConfig {
    let config_path = Path::new(model_path).join("config.json");
    let content = fs::read_to_string(&config_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", config_path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", config_path.display()))
}

fn write_zero_lora_adapter(path: &Path, config: &ModelConfig, rank: usize) {
    fixtures::write_adapter_config(path, rank, rank, &["q_proj", "v_proj"]);

    let mut tensors = BTreeMap::new();
    for layer_idx in 0..config.num_hidden_layers {
        fixtures::push_projection(
            &mut tensors,
            layer_idx,
            "self_attn.q_proj",
            rank,
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        );
        fixtures::push_projection(
            &mut tensors,
            layer_idx,
            "self_attn.v_proj",
            rank,
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
        );
    }
    fixtures::write_adapter_tensors(path, tensors);
}

fn load_adapter(handle: &EngineHandle, adapter_name: &str, adapter_path: PathBuf) {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("build runtime")
        .block_on(handle.load_lora_adapter(LoadLoraAdapterRequest {
            lora_name: adapter_name.to_string(),
            lora_path: adapter_path,
            load_inplace: false,
        }))
        .expect("load LoRA adapter");
}

fn generate_tokens(
    handle: &EngineHandle,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
    lora_adapter: Option<String>,
) -> (Vec<u32>, FinishReason) {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let (token_tx, mut token_rx) = TokenSink::standalone();

    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut tokens = Vec::new();
    loop {
        match token_rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, .. }) => tokens.push(id),
            Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
            Some(TokenEvent::Finished { finish_reason, .. }) => return (tokens, finish_reason),
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

#[test]
#[ignore = "requires Qwen3-4B weights and a CUDA GPU"]
fn qwen3_lora_loads_adapter_and_generates() {
    qwen3_lora_loads_rank_and_generates(1, "zero-smoke");
}

#[test]
#[ignore = "requires Qwen3-4B weights and a CUDA GPU"]
fn qwen3_lora_loads_rank64_adapter_and_generates() {
    qwen3_lora_loads_rank_and_generates(64, "zero-rank64-smoke");
}

fn qwen3_lora_loads_rank_and_generates(rank: usize, adapter_name: &str) {
    let model_path = get_model_path();
    let config = load_model_config(&model_path);
    assert!(
        config.intermediate_size > config.hidden_size,
        "unexpected Qwen3 config dimensions"
    );

    let adapter_dir = tempfile::tempdir().expect("create temp adapter dir");
    write_zero_lora_adapter(adapter_dir.path(), &config, rank);

    let handle = openinfer_qwen3::start_engine_with_lora_control(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: vec![get_device_ordinal()],
            seed: 42,
            ..EngineLoadOptions::default()
        },
        openinfer_qwen3::Qwen3LoraOptions::default(),
        openinfer_qwen3::Qwen3OffloadOptions::disabled(),
        false,
        openinfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS,
        openinfer_qwen3::Qwen3MemoryOptions::default(),
        openinfer_qwen3::DecodeOverlap::Off,
        false,
    )
    .expect("start LoRA-capable Qwen3 engine");

    assert!(handle.supports_lora_control());
    load_adapter(&handle, adapter_name, adapter_dir.path().to_path_buf());

    let tokenizer = common::load_tokenizer(&model_path);
    let (tokens, finish_reason) = generate_tokens(
        &handle,
        &tokenizer,
        "Hello",
        4,
        Some(adapter_name.to_string()),
    );
    assert!(
        !tokens.is_empty(),
        "LoRA smoke generation returned no tokens"
    );
    assert_eq!(finish_reason, FinishReason::Length);
}
