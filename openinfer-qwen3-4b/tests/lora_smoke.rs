use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use half::bf16;
use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, FinishReason, GenerateRequest, LoadLoraAdapterRequest,
    TokenEvent,
};
use openinfer_core::sampler::SamplingParams;
use safetensors::Dtype;
use safetensors::tensor::View;
use serde::Deserialize;
use tokio::sync::mpsc;
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

#[derive(Clone)]
struct TestTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for TestTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
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

fn temp_adapter_dir() -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "openinfer-qwen3-lora-smoke-{}-{now}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).expect("create temp adapter dir");
    path
}

fn write_zero_lora_adapter(path: &Path, config: &ModelConfig, rank: usize) {
    fs::write(
        path.join("adapter_config.json"),
        format!(
            r#"{{
  "peft_type": "LORA",
  "r": {rank},
  "lora_alpha": {rank},
  "target_modules": ["q_proj", "v_proj"]
}}"#
        ),
    )
    .expect("write adapter_config.json");

    let mut tensors = BTreeMap::new();
    for layer_idx in 0..config.num_hidden_layers {
        push_zero_tensor(
            &mut tensors,
            tensor_name(layer_idx, "self_attn.q_proj", "lora_A"),
            vec![rank, config.hidden_size],
        );
        push_zero_tensor(
            &mut tensors,
            tensor_name(layer_idx, "self_attn.q_proj", "lora_B"),
            vec![config.num_attention_heads * config.head_dim, rank],
        );
        push_zero_tensor(
            &mut tensors,
            tensor_name(layer_idx, "self_attn.v_proj", "lora_A"),
            vec![rank, config.hidden_size],
        );
        push_zero_tensor(
            &mut tensors,
            tensor_name(layer_idx, "self_attn.v_proj", "lora_B"),
            vec![config.num_key_value_heads * config.head_dim, rank],
        );
    }

    safetensors::serialize_to_file(tensors, None, &path.join("adapter_model.safetensors"))
        .expect("write adapter_model.safetensors");
}

fn push_zero_tensor(tensors: &mut BTreeMap<String, TestTensor>, name: String, shape: Vec<usize>) {
    let elems = shape.iter().product::<usize>();
    tensors.insert(
        name,
        TestTensor {
            dtype: Dtype::BF16,
            shape,
            data: bf16::from_f32(0.0).to_bits().to_le_bytes().repeat(elems),
        },
    );
}

fn tensor_name(layer_idx: usize, path_segment: &str, lora_side: &str) -> String {
    format!("base_model.model.model.layers.{layer_idx}.{path_segment}.{lora_side}.weight")
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
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();

    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
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
        match token_rx.blocking_recv() {
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

    let adapter_path = temp_adapter_dir();
    write_zero_lora_adapter(&adapter_path, &config, rank);

    let handle = openinfer_qwen3_4b::start_engine_with_lora_control(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: false,
            enable_prefill_profile: false,
            device_ordinals: vec![get_device_ordinal()],
            seed: 42,
            ..EngineLoadOptions::default()
        },
        openinfer_qwen3_4b::Qwen3LoraOptions::default(),
        openinfer_qwen3_4b::Qwen3OffloadOptions::disabled(),
        false,
        openinfer_qwen3_4b::DEFAULT_MAX_PREFILL_TOKENS,
    )
    .expect("start LoRA-capable Qwen3 engine");

    assert!(handle.supports_lora_control());
    load_adapter(&handle, adapter_name, adapter_path);

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
