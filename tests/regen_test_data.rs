/// Regenerate model-specific golden data in test_data/<model_name>.json using greedy decoding.
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use pegainfer::model::Qwen35Model;
use pegainfer::sampler::SamplingParams;
use pegainfer::scheduler::{self, SchedulerRequest, TokenEvent};
use pegainfer::scheduler_qwen35;
use tokio::sync::mpsc;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const DEFAULT_QWEN35_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");
static GPU_REGEN_TEST_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum PromptStyle {
    Base,
    Instruct,
}

struct Case {
    name: &'static str,
    prompt: &'static str,
    max_new_tokens: usize,
}

const CASES: &[Case] = &[
    Case {
        name: "tell_story",
        prompt: "Tell me a story",
        max_new_tokens: 50,
    },
    Case {
        name: "my_name",
        prompt: "My name is",
        max_new_tokens: 50,
    },
    Case {
        name: "math",
        prompt: "What is 2 + 2?",
        max_new_tokens: 30,
    },
    Case {
        name: "chinese_weather",
        prompt: "今天天气真好",
        max_new_tokens: 50,
    },
    Case {
        name: "chinese_capital",
        prompt: "请介绍一下中国的首都",
        max_new_tokens: 50,
    },
    Case {
        name: "python_code",
        prompt: "Write a Python function to reverse a string",
        max_new_tokens: 50,
    },
    Case {
        name: "kanye_album",
        prompt: "我最喜欢的 Kanye West 的专辑是",
        max_new_tokens: 50,
    },
    Case {
        name: "coldplay_ghost",
        prompt: "Coldplay 的《Ghost story》专辑真是",
        max_new_tokens: 50,
    },
    Case {
        name: "oyster_riddle",
        prompt: "生蚝煮熟了是熟蚝",
        max_new_tokens: 50,
    },
    Case {
        name: "monkey_king_lake",
        prompt: "孙悟空跳到一个湖里，跳出来变成了六耳猕猴，这个湖的名字是",
        max_new_tokens: 50,
    },
];

fn model_name(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

fn test_data_path(model_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join(format!("{model_name}.json"))
}

fn prompt_style(model_name: &str) -> PromptStyle {
    if model_name.to_ascii_lowercase().contains("instruct") {
        PromptStyle::Instruct
    } else {
        PromptStyle::Base
    }
}

fn wrap_prompt(prompt: &str, style: PromptStyle) -> String {
    match style {
        PromptStyle::Base => prompt.to_string(),
        PromptStyle::Instruct => format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ),
    }
}

/// Generate text via a scheduler handle.
fn generate_text_scheduler(
    handle: &scheduler::SchedulerHandle,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> String {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();

    handle
        .submit(SchedulerRequest {
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut tokens = Vec::new();
    loop {
        match token_rx.blocking_recv() {
            Some(TokenEvent::Token { id, .. }) => tokens.push(id),
            Some(TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            None => panic!("scheduler closed"),
        }
    }

    tokenizer.decode(&tokens, true).expect("decode failed")
}

fn write_golden_json(output_path: &Path, model_name: &str, cases_json: &[serde_json::Value]) {
    let data = serde_json::json!({
        "model_name": model_name,
        "engine": "pegainfer",
        "cases": cases_json,
    });
    let json = serde_json::to_string_pretty(&data).unwrap();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create test data directory");
    }
    std::fs::write(output_path, format!("{json}\n")).expect("Failed to write test data");
    eprintln!("Wrote {}", output_path.display());
}

fn lock_gpu_regen_test() -> MutexGuard<'static, ()> {
    // These tests each load a full model onto the same GPU, so they must not run concurrently.
    GPU_REGEN_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
#[ignore = "regenerates checked-in golden data; run manually when fixtures need refresh"]
fn regen_test_data_qwen35() {
    let _guard = lock_gpu_regen_test();
    pegainfer::logging::init_stderr("info");

    let model_path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
        .unwrap_or_else(|_| DEFAULT_QWEN35_MODEL_PATH.to_string());
    let model_name = model_name(&model_path);
    let prompt_style = prompt_style(&model_name);
    let output_path = test_data_path(&model_name);

    eprintln!(
        "Regenerating golden data: model={}, path={}, prompt_style={:?}, output={}",
        model_name,
        model_path,
        prompt_style,
        output_path.display()
    );

    let model = Qwen35Model::from_safetensors_with_options(&model_path, true)
        .expect("Failed to load model");
    let tokenizer = common::load_tokenizer(&model_path);
    let handle = scheduler_qwen35::start(model, 42).expect("Failed to start scheduler");

    let mut cases_json = Vec::new();
    for case in CASES {
        let prompt = wrap_prompt(case.prompt, prompt_style);
        let output = generate_text_scheduler(&handle, &tokenizer, &prompt, case.max_new_tokens);
        eprintln!(
            "[{}] raw_prompt={:?} prompt={:?} output={:?}",
            case.name, case.prompt, prompt, output
        );
        cases_json.push(serde_json::json!({
            "name": case.name,
            "prompt": prompt,
            "max_new_tokens": case.max_new_tokens,
            "output": output,
        }));
    }

    write_golden_json(&output_path, &model_name, &cases_json);
}
