use std::path::Path;

use openinfer_core::{
    engine::{EpBackend, GenerateRequest, TokenEvent, TokenSink},
    sampler::SamplingParams,
};
use openinfer_glm52::{Glm52LaunchOptions, launch};

const JIUZHANG_GLM52_MODEL_PATH: &str = "/data/models/GLM-5.2-FP8";

#[test]
#[ignore]
fn jiuzhang_checkpoint_loads_and_rejects_until_forward_lands() {
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

    let (token_tx, mut token_rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: Some("glm52-checkpoint-smoke".to_string()),
            queued_at_unix_s: None,
            prompt_tokens: vec![1, 2, 3],
            params: SamplingParams::default(),
            max_tokens: 1,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit GLM5.2 smoke request");

    let Some((_, TokenEvent::Scheduled { prompt_tokens, .. })) = token_rx.blocking_recv() else {
        panic!("GLM5.2 smoke request was not scheduled");
    };
    assert_eq!(prompt_tokens, 3);

    let Some((_, TokenEvent::Rejected { message, .. })) = token_rx.blocking_recv() else {
        panic!("GLM5.2 smoke request did not reject while forward is pending");
    };
    assert!(message.contains("forward runtime is not implemented yet"));
}
