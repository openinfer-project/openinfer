use std::path::PathBuf;

use openinfer_core::{
    engine::{GenerateRequest, TokenEvent, TokenSink},
    sampler::SamplingParams,
};
use openinfer_glm52::{Glm52LaunchOptions, launch};

const DEFAULT_GLM52_MODEL_PATH: &str = "models/GLM-5.2-FP8";

#[test]
#[ignore]
fn checkpoint_loads_dp1_ep8_weights() {
    openinfer_core::logging::init_default();

    let model_path = std::env::var_os("OPENINFER_TEST_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_GLM52_MODEL_PATH));
    assert!(
        model_path.join("model.safetensors.index.json").exists(),
        "GLM5.2 checkpoint missing at {}",
        model_path.display()
    );

    let handle = launch(
        &model_path,
        Glm52LaunchOptions {
            tp_size: 1,
            dp_size: 1,
        },
    )
    .expect("GLM5.2 checkpoint startup");

    let (token_tx, mut token_rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: Some("glm52-load-weight-only".to_string()),
            queued_at_unix_s: None,
            prompt_tokens: vec![100, 2048, 9001, 12345],
            params: SamplingParams::default(),
            max_tokens: 1,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit GLM5.2 request");

    let mut saw_scheduled = false;
    loop {
        match token_rx.blocking_recv() {
            Some((_, TokenEvent::Scheduled { prompt_tokens, .. })) => {
                saw_scheduled = true;
                assert_eq!(prompt_tokens, 4);
            }
            Some((_, TokenEvent::Rejected { message, .. })) => {
                assert!(
                    saw_scheduled,
                    "request should be scheduled before rejection"
                );
                assert!(
                    message.contains("load-weight-only"),
                    "unexpected rejection message: {message}"
                );
                break;
            }
            Some((_, TokenEvent::Error { message, .. })) => {
                panic!("GLM5.2 load-only engine returned error: {message}");
            }
            Some((_, _event)) => panic!("unexpected GLM5.2 load-only event kind"),
            None => panic!("GLM5.2 token channel closed before rejection"),
        }
    }
}
