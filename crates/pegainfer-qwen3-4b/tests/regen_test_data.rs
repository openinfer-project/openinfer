/// Regenerate Qwen3 golden data in the workspace `test_data/` directory.
use std::path::{Path, PathBuf};

use pegainfer_core::engine::{EngineHandle, EngineLoadOptions, GenerateRequest, TokenEvent};
use pegainfer_core::sampler::SamplingParams;
use tokio::sync::mpsc;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/Qwen3-4B");

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
        .join("../../test_data")
        .join(format!("{model_name}.json"))
}

fn generate_text_scheduler(
    handle: &EngineHandle,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> String {
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

#[test]
#[ignore = "regenerates checked-in golden data; run manually when fixtures need refresh"]
fn regen_test_data() {
    let model_path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
        .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    let model_name = model_name(&model_path);
    let output_path = test_data_path(&model_name);

    let handle = pegainfer_qwen3_4b::start_engine(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            seed: 42,
        },
    )
    .expect("Failed to start engine");
    let tokenizer = common::load_tokenizer(&model_path);

    let mut cases_json = Vec::new();
    for case in CASES {
        let output = generate_text_scheduler(&handle, &tokenizer, case.prompt, case.max_new_tokens);
        cases_json.push(serde_json::json!({
            "name": case.name,
            "prompt": case.prompt,
            "max_new_tokens": case.max_new_tokens,
            "output": output,
        }));
    }

    let data = serde_json::json!({
        "model_name": model_name,
        "engine": "pegainfer",
        "cases": cases_json,
    });
    let json = serde_json::to_string_pretty(&data).unwrap();
    std::fs::write(&output_path, format!("{json}\n")).expect("Failed to write test data");
    eprintln!("Wrote {}", output_path.display());
}
