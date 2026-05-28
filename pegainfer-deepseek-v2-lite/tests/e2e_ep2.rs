use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use pegainfer_deepseek_v2_lite::DeepSeekV2LiteEp2Generator;
use pegainfer_engine::engine::{EngineLoadOptions, FinishReason};
use sha2::{Digest, Sha256};
use vllm_text::tokenizer::{HuggingFaceTokenizer, Tokenizer};

const EXPECTED_GENERATED_TOKENS: usize = 16;
const EXPECTED_OUTPUT_TOKEN_SHA256: &str =
    "4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225";
const EXPECTED_OUTPUT_TEXT_SHA256: &str =
    "0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347";
const DSV2_LITE_HIDDEN_SIZE: usize = 2048;
const DSV2_LITE_MOE_LAYERS: usize = 26;
const E2E_JSON_OUT_ENV: &str = "PEGAINFER_DSV2_LITE_E2E_JSON_OUT";

#[test]
fn test_deepseek_v2_lite_ep2_rust_generation() -> Result<()> {
    let model_path_label = env::var("PEGAINFER_TEST_MODEL_PATH")
        .context("PEGAINFER_TEST_MODEL_PATH must point to DeepSeek-V2-Lite weights")?;
    let model_path = resolve_model_path(&model_path_label);
    ensure!(
        model_path.join("config.json").exists(),
        "missing config.json under {}",
        model_path.display()
    );

    let duplicate_ordinal_err = DeepSeekV2LiteEp2Generator::load(
        &model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            enable_prefill_profile: false,
            device_ordinals: vec![0, 0],
            seed: 42,
        },
    )
    .err()
    .context("duplicate CUDA device ordinals unexpectedly loaded")?;
    ensure!(
        format!("{duplicate_ordinal_err:#}").contains("two distinct CUDA device ordinals"),
        "duplicate CUDA ordinal error should mention distinct devices, got {duplicate_ordinal_err:#}"
    );

    run_rust_generation(&model_path_label, &model_path)
}

fn run_rust_generation(model_path_label: &str, model_path: &Path) -> Result<()> {
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = HuggingFaceTokenizer::new(&tokenizer_path).map_err(|err| {
        anyhow::anyhow!(
            "failed to load tokenizer {}: {err:?}",
            tokenizer_path.display()
        )
    })?;
    let prompt = "Hello";
    let prompt_tokens = tokenizer
        .encode(prompt, false)
        .map_err(|err| anyhow::anyhow!("encode prompt failed: {err:?}"))?;
    ensure!(!prompt_tokens.is_empty(), "tokenizer returned empty prompt");

    let mut generator = DeepSeekV2LiteEp2Generator::load(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            enable_prefill_profile: false,
            device_ordinals: vec![0, 1],
            seed: 42,
        },
    )?;
    let result = generator.generate_greedy(&prompt_tokens, 16, false)?;
    ensure!(
        !result.tokens.is_empty(),
        "DeepSeek-V2-Lite Rust generation produced no tokens"
    );
    ensure!(
        result.stats.ep_size == 2,
        "DeepSeek-V2-Lite E2E expected ep_size=2, got {}",
        result.stats.ep_size
    );
    ensure!(
        result.stats.device_ordinals == vec![0, 1],
        "DeepSeek-V2-Lite E2E expected devices [0, 1], got {:?}",
        result.stats.device_ordinals
    );
    ensure!(
        result.stats.generated_tokens == EXPECTED_GENERATED_TOKENS,
        "DeepSeek-V2-Lite E2E generated {} tokens, expected {}",
        result.stats.generated_tokens,
        EXPECTED_GENERATED_TOKENS
    );
    ensure!(
        result.finish_reason == FinishReason::Length,
        "DeepSeek-V2-Lite E2E finish_reason drift: got {:?}, expected Length",
        result.finish_reason
    );
    ensure!(
        result.stats.ep_backend == current_backend(),
        "DeepSeek-V2-Lite E2E backend mismatch: got {}, expected {}",
        result.stats.ep_backend,
        current_backend()
    );
    match result.stats.ep_backend.as_str() {
        "host-staged" => {
            ensure!(
                result.stats.host_dispatch_calls > 0
                    && result.stats.host_combine_calls == result.stats.host_dispatch_calls
                    && result.stats.host_dispatch_elements > 0
                    && result.stats.host_combine_elements == result.stats.host_dispatch_elements,
                "host-staged EP gate did not record dispatch/combine counts"
            );
            ensure!(
                result.stats.host_dispatch_remote_routes > 0,
                "host-staged EP gate did not exercise any remote routed expert"
            );
            ensure!(
                result.stats.host_dispatch_local_routes > 0,
                "host-staged EP gate did not exercise any local routed expert"
            );
            ensure!(
                result.stats.nccl_dense_exchange_calls == 0
                    && result.stats.nccl_combine_calls == 0
                    && result.stats.nccl_dense_exchange_elements == 0
                    && result.stats.nccl_combine_elements == 0,
                "host-staged EP gate unexpectedly recorded NCCL collectives"
            );
        }
        "nccl" => {
            ensure!(
                result.stats.nccl_dispatch_remote_routes > 0,
                "NCCL EP gate did not exercise any remote routed expert"
            );
            ensure!(
                result.stats.nccl_dispatch_local_routes > 0,
                "NCCL EP gate did not exercise any local routed expert"
            );
            ensure!(
                result.stats.nccl_combine_routes
                    == result.stats.nccl_dispatch_local_routes
                        + result.stats.nccl_dispatch_remote_routes,
                "NCCL combine route accounting drift"
            );
            let expected_moe_calls = result.stats.generated_tokens * DSV2_LITE_MOE_LAYERS;
            let expected_collective_elements = expected_moe_calls * DSV2_LITE_HIDDEN_SIZE;
            ensure!(
                result.stats.nccl_dense_exchange_calls == expected_moe_calls,
                "NCCL dense hidden exchange call count drift: got {}, expected {}",
                result.stats.nccl_dense_exchange_calls,
                expected_moe_calls
            );
            ensure!(
                result.stats.nccl_combine_calls == expected_moe_calls,
                "NCCL combine call count drift: got {}, expected {}",
                result.stats.nccl_combine_calls,
                expected_moe_calls
            );
            ensure!(
                result.stats.nccl_dense_exchange_elements == expected_collective_elements,
                "NCCL dense hidden exchange element count drift: got {}, expected {}",
                result.stats.nccl_dense_exchange_elements,
                expected_collective_elements
            );
            ensure!(
                result.stats.nccl_combine_elements == expected_collective_elements,
                "NCCL combine element count drift: got {}, expected {}",
                result.stats.nccl_combine_elements,
                expected_collective_elements
            );
        }
        other => anyhow::bail!("unexpected DeepSeek-V2-Lite EP backend in E2E: {other}"),
    }

    let output_text = tokenizer
        .decode(&result.tokens, false)
        .map_err(|err| anyhow::anyhow!("decode output failed: {err:?}"))?;
    let mut hasher = Sha256::new();
    hasher.update(output_text.as_bytes());
    let output_text_sha256 = hex::encode(hasher.finalize());
    let payload = serde_json::json!({
        "model_path": model_path_label,
        "gpu_count": 2,
        "ep_size": result.stats.ep_size,
        "ep_backend": result.stats.ep_backend,
        "devices": &result.stats.device_ordinals,
        "prompt": prompt,
        "prompt_tokens": result.stats.prompt_tokens,
        "prompt_token_ids": &prompt_tokens,
        "max_new_tokens": 16,
        "generated_tokens": result.stats.generated_tokens,
        "generated_token_ids": &result.tokens,
        "generated_text": &output_text,
        "output_token_sha256": result.stats.output_token_sha256,
        "output_text_sha256": output_text_sha256,
        "token_sha256_algorithm": "sha256 over generated token ids encoded as little-endian u32",
        "text_sha256_algorithm": "sha256 over UTF-8 generated text bytes",
        "host_dispatch_calls": result.stats.host_dispatch_calls,
        "host_dispatch_elements": result.stats.host_dispatch_elements,
        "host_combine_calls": result.stats.host_combine_calls,
        "host_combine_elements": result.stats.host_combine_elements,
        "host_dispatch_local_routes": result.stats.host_dispatch_local_routes,
        "host_dispatch_remote_routes": result.stats.host_dispatch_remote_routes,
        "nccl_dispatch_local_routes": result.stats.nccl_dispatch_local_routes,
        "nccl_dispatch_remote_routes": result.stats.nccl_dispatch_remote_routes,
        "nccl_combine_routes": result.stats.nccl_combine_routes,
        "nccl_dense_exchange_calls": result.stats.nccl_dense_exchange_calls,
        "nccl_combine_calls": result.stats.nccl_combine_calls,
        "nccl_dense_exchange_elements": result.stats.nccl_dense_exchange_elements,
        "nccl_combine_elements": result.stats.nccl_combine_elements,
        "output_text": &output_text,
    });
    let payload_text = serde_json::to_string_pretty(&payload)?;
    if let Ok(path) = env::var(E2E_JSON_OUT_ENV) {
        if !path.is_empty() {
            let path = PathBuf::from(path);
            let path = resolve_workspace_path(path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create {}", parent.display()))?;
            }
            fs::write(&path, format!("{payload_text}\n"))
                .with_context(|| format!("write {}", path.display()))?;
        }
    }
    println!("{payload_text}");
    ensure!(
        result.stats.output_token_sha256 == EXPECTED_OUTPUT_TOKEN_SHA256,
        "DeepSeek-V2-Lite E2E token hash drift: got {}, expected {}",
        result.stats.output_token_sha256,
        EXPECTED_OUTPUT_TOKEN_SHA256
    );
    ensure!(
        output_text_sha256 == EXPECTED_OUTPUT_TEXT_SHA256,
        "DeepSeek-V2-Lite E2E text hash drift: got {}, expected {}",
        output_text_sha256,
        EXPECTED_OUTPUT_TEXT_SHA256
    );
    Ok(())
}

fn current_backend() -> String {
    env::var("PEGAINFER_DSV2_LITE_EP_BACKEND").unwrap_or_else(|_| "host-staged".to_string())
}

fn resolve_model_path(raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.join("config.json").exists() {
        return path;
    }
    let workspace_path = resolve_workspace_path(path.clone());
    if workspace_path.join("config.json").exists() {
        return workspace_path;
    }
    path
}

fn resolve_workspace_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    workspace_root().join(path)
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("model crate must live under the workspace root")
        .to_path_buf()
}
