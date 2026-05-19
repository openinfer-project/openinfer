use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use pegainfer_deepseek_v2_lite::DeepSeekV2LiteEp2Generator;
use pegainfer_engine::engine::{EngineLoadOptions, FinishReason};
use sha2::{Digest, Sha256};
use vllm_text::tokenizer::{HuggingFaceTokenizer, Tokenizer};

const EXPECTED_GENERATED_TOKENS: usize = 16;
const EXPECTED_OUTPUT_TOKEN_SHA256: &str =
    "f39e57d9b3eb949057ada9b3bc92f7f7037dfd19658dbe3ce145d8fad03ded5e";
const EXPECTED_OUTPUT_TEXT_SHA256: &str =
    "a521f6dc9739b4506a46822da7c239ac558a879d571f54a064a4a2fbc3a097b7";

#[test]
fn test_deepseek_v2_lite_ep2_rust_generation() -> Result<()> {
    let model_path = PathBuf::from(
        env::var("PEGAINFER_TEST_MODEL_PATH")
            .context("PEGAINFER_TEST_MODEL_PATH must point to DeepSeek-V2-Lite weights")?,
    );
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

    run_rust_generation(&model_path)
}

fn run_rust_generation(model_path: &Path) -> Result<()> {
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
        result.stats.output_token_sha256 == EXPECTED_OUTPUT_TOKEN_SHA256,
        "DeepSeek-V2-Lite E2E token hash drift: got {}, expected {}",
        result.stats.output_token_sha256,
        EXPECTED_OUTPUT_TOKEN_SHA256
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
                result.stats.host_dispatch_remote_routes > 0,
                "host-staged EP gate did not exercise any remote routed expert"
            );
            ensure!(
                result.stats.host_dispatch_local_routes > 0,
                "host-staged EP gate did not exercise any local routed expert"
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
        }
        other => anyhow::bail!("unexpected DeepSeek-V2-Lite EP backend in E2E: {other}"),
    }

    let output_text = tokenizer
        .decode(&result.tokens, false)
        .map_err(|err| anyhow::anyhow!("decode output failed: {err:?}"))?;
    let mut hasher = Sha256::new();
    hasher.update(output_text.as_bytes());
    let output_text_sha256 = hex::encode(hasher.finalize());
    ensure!(
        output_text_sha256 == EXPECTED_OUTPUT_TEXT_SHA256,
        "DeepSeek-V2-Lite E2E text hash drift: got {}, expected {}",
        output_text_sha256,
        EXPECTED_OUTPUT_TEXT_SHA256
    );
    let payload = serde_json::json!({
        "model_path": model_path,
        "gpu_count": 2,
        "ep_size": result.stats.ep_size,
        "ep_backend": result.stats.ep_backend,
        "devices": result.stats.device_ordinals,
        "prompt": prompt,
        "prompt_tokens": result.stats.prompt_tokens,
        "generated_tokens": result.stats.generated_tokens,
        "output_token_sha256": result.stats.output_token_sha256,
        "output_text_sha256": output_text_sha256,
        "host_dispatch_local_routes": result.stats.host_dispatch_local_routes,
        "host_dispatch_remote_routes": result.stats.host_dispatch_remote_routes,
        "nccl_dispatch_local_routes": result.stats.nccl_dispatch_local_routes,
        "nccl_dispatch_remote_routes": result.stats.nccl_dispatch_remote_routes,
        "nccl_combine_routes": result.stats.nccl_combine_routes,
        "output_text": output_text,
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}

fn current_backend() -> String {
    env::var("PEGAINFER_DSV2_LITE_EP_BACKEND").unwrap_or_else(|_| "host-staged".to_string())
}
