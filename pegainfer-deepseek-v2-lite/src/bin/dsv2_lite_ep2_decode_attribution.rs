use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use pegainfer_deepseek_v2_lite::DeepSeekV2LiteEp2Generator;
use pegainfer_engine::engine::EngineLoadOptions;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use vllm_text::tokenizer::{HuggingFaceTokenizer, Tokenizer};

const PROMPT: &str = "Hello";
const OUTPUT_LEN: usize = 16;
const EXPECTED_OUTPUT_TOKEN_SHA256: &str =
    "4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225";
const EXPECTED_OUTPUT_TEXT_SHA256: &str =
    "0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347";

struct Cli {
    model_path: String,
    out: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = parse_cli()?;
    let model_path = resolve_model_path(&cli.model_path);
    ensure!(
        model_path.join("config.json").exists(),
        "missing config.json under {}",
        model_path.display()
    );

    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = HuggingFaceTokenizer::new(&tokenizer_path).map_err(|err| {
        anyhow::anyhow!(
            "failed to load tokenizer {}: {err:?}",
            tokenizer_path.display()
        )
    })?;
    let prompt_tokens = tokenizer
        .encode(PROMPT, false)
        .map_err(|err| anyhow::anyhow!("encode prompt failed: {err:?}"))?;
    ensure!(!prompt_tokens.is_empty(), "tokenizer returned empty prompt");

    let mut generator = DeepSeekV2LiteEp2Generator::load(
        &model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            enable_prefill_profile: false,
            device_ordinals: vec![0, 1],
            seed: 42,
        },
    )?;
    let (result, attribution) =
        generator.generate_greedy_with_attribution(&prompt_tokens, OUTPUT_LEN, false)?;
    let generated_text = tokenizer
        .decode(&result.tokens, false)
        .map_err(|err| anyhow::anyhow!("decode output failed: {err:?}"))?;
    let text_sha256 = sha256_text(&generated_text);

    ensure!(
        result.stats.output_token_sha256 == EXPECTED_OUTPUT_TOKEN_SHA256,
        "DeepSeek-V2-Lite attribution token hash drift: got {}, expected {}",
        result.stats.output_token_sha256,
        EXPECTED_OUTPUT_TOKEN_SHA256
    );
    ensure!(
        text_sha256 == EXPECTED_OUTPUT_TEXT_SHA256,
        "DeepSeek-V2-Lite attribution text hash drift: got {}, expected {}",
        text_sha256,
        EXPECTED_OUTPUT_TEXT_SHA256
    );

    let by_section = attribution.by_section();
    let by_gpu_section = attribution.by_gpu_section();
    let by_gpu_call_site = attribution.by_gpu_call_site();
    let by_op: Vec<Value> = by_section
        .iter()
        .map(|row| {
            json!({
                "op": row.section,
                "calls": row.calls,
                "total_us": row.total_us,
                "mean_us": row.mean_us,
                "min_us": row.min_us,
                "p50_us": row.p50_us,
                "p95_us": row.p95_us,
                "p99_us": row.p99_us,
                "max_us": row.max_us,
                "pct": row.pct,
            })
        })
        .collect();
    let mut per_output_token_us = Vec::with_capacity(OUTPUT_LEN);
    if let Some(prefill_next_token_us) = attribution.prefill_next_token_us() {
        per_output_token_us.push(prefill_next_token_us);
    }
    per_output_token_us.extend_from_slice(attribution.per_token_decode_us());

    let report = json!({
        "schema": 1,
        "report_type": "deepseek-v2-lite-ep2-decode-attribution",
        "model": "DeepSeek-V2-Lite",
        "phase": "decode",
        "backend": &result.stats.ep_backend,
        "config": {
            "batch_size": 1,
            "prompt": PROMPT,
            "prompt_token_ids": &prompt_tokens,
            "output_len": OUTPUT_LEN,
            "ep_size": result.stats.ep_size,
            "device_ordinals": &result.stats.device_ordinals,
        },
        "accuracy": {
            "generated_token_ids": &result.tokens,
            "generated_text": &generated_text,
            "token_sha256": &result.stats.output_token_sha256,
            "text_sha256": &text_sha256,
            "expected_token_sha256": EXPECTED_OUTPUT_TOKEN_SHA256,
            "expected_text_sha256": EXPECTED_OUTPUT_TEXT_SHA256,
            "token_hash_exact": true,
            "text_hash_exact": true,
        },
        "timing": {
            "source": "CPU-side wall-clock sections around the existing DeepSeek-V2-Lite EP2 greedy path",
            "total_generation_us": attribution.total_generation_us(),
            "prefill_next_token_us": attribution.prefill_next_token_us(),
            "per_output_token_us": per_output_token_us,
            "decode_token_count": attribution.per_token_decode_us().len(),
            "per_token_decode_us": attribution.per_token_decode_us(),
            "per_token_decode_stats": latency_stats(attribution.per_token_decode_us()),
        },
        "attribution_source": "DeepSeekV2LiteEp2Generator::generate_greedy_with_attribution; CPU sections use host wall-clock timers, and selected GPU/NCCL sections also carry CUDA event timing plus optional NVTX ranges.",
        "gpu_timing": {
            "source": "CUDA event timing around selected DeepSeek-V2-Lite EP2 GPU/NCCL stream sections in the explicit attribution path",
            "sample_count": attribution.gpu_sample_count(),
            "failure_count": attribution.gpu_timing_failure_count(),
            "nvtx_enabled": attribution.nvtx_enabled(),
            "nvtx_range_count": attribution.nvtx_range_count(),
            "scope": "selected GPU/NCCL sections only; host routing, host accumulation, and the mixed attention_host_path remain CPU-side attribution rows; GPU timing failures do not replace the token/text hash oracle; NVTX range wall time is only a profiler correlation marker and may include host/event overhead",
        },
        "schedule_source": "fixed DeepSeek-V2-Lite EP2 greedy gate: prompt=Hello, output_len=16, cuda_graph=false, device_ordinals=[0,1]",
        "by_section": by_section,
        "by_op": by_op,
        "by_call_site": attribution.by_call_site(),
        "by_gpu_section": by_gpu_section,
        "by_gpu_call_site": by_gpu_call_site,
        "coverage": coverage_rows(&result.stats.ep_backend, &attribution),
        "ep": ep_report(&result.stats),
        "claim_boundary": "Attribution only for the covered EP2 Hello/16 decode gate. CPU-side section timing, selected CUDA event timing, NVTX ranges, and route/collective counts are not a throughput, sparse-dispatch, multi-node, or production EP readiness claim.",
    });

    let text = serde_json::to_string_pretty(&report)?;
    if let Some(out) = cli.out {
        let path = resolve_workspace_path(out);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        fs::write(&path, format!("{text}\n"))
            .with_context(|| format!("write {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    println!("{text}");
    Ok(())
}

fn parse_cli() -> Result<Cli> {
    let mut model_path = "models/DeepSeek-V2-Lite".to_string();
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-path" => {
                model_path = args
                    .next()
                    .context("--model-path requires a path argument")?;
            }
            "--out" => {
                out = Some(PathBuf::from(
                    args.next().context("--out requires a path argument")?,
                ));
            }
            "-h" | "--help" => {
                println!(
                    "DeepSeek-V2-Lite EP2 decode attribution gate\n\nUSAGE:\n  dsv2_lite_ep2_decode_attribution [--model-path PATH] [--out PATH]\n\nThe gate is intentionally fixed to batch=1, prompt=Hello, output_len=16. Select NCCL with PEGAINFER_DSV2_LITE_EP_BACKEND=nccl."
                );
                std::process::exit(0);
            }
            other => bail!(
                "unsupported argument `{other}`; supported flags: --model-path PATH, --out PATH"
            ),
        }
    }
    Ok(Cli { model_path, out })
}

fn ep_report(stats: &pegainfer_deepseek_v2_lite::GenerationStats) -> Value {
    let (local_route_count, remote_route_count) = match stats.ep_backend.as_str() {
        "host-staged" => (
            stats.host_dispatch_local_routes,
            stats.host_dispatch_remote_routes,
        ),
        "nccl" => (
            stats.nccl_dispatch_local_routes,
            stats.nccl_dispatch_remote_routes,
        ),
        _ => (0, 0),
    };
    json!({
        "dispatch_calls": stats.host_dispatch_calls,
        "dispatch_elements": stats.host_dispatch_elements,
        "combine_calls": stats.host_combine_calls,
        "combine_elements": stats.host_combine_elements,
        "nccl_exchange_calls": stats.nccl_dense_exchange_calls,
        "nccl_exchange_elements": stats.nccl_dense_exchange_elements,
        "nccl_combine_calls": stats.nccl_combine_calls,
        "nccl_combine_elements": stats.nccl_combine_elements,
        "local_route_count": local_route_count,
        "remote_route_count": remote_route_count,
        "host_dispatch_local_routes": stats.host_dispatch_local_routes,
        "host_dispatch_remote_routes": stats.host_dispatch_remote_routes,
        "nccl_dispatch_local_routes": stats.nccl_dispatch_local_routes,
        "nccl_dispatch_remote_routes": stats.nccl_dispatch_remote_routes,
        "nccl_combine_routes": stats.nccl_combine_routes,
    })
}

fn coverage_rows(
    backend: &str,
    attribution: &pegainfer_deepseek_v2_lite::DecodeAttributionProfile,
) -> Vec<Value> {
    vec![
        json!({
            "item": "accuracy.token_text_hash",
            "status": "passed",
            "source": "same token/text hash oracle as the DeepSeek-V2-Lite EP2 HF accuracy gate",
        }),
        json!({
            "item": "cpu_side_sections",
            "status": "measured",
            "source": "Instant-based host-side section timers in the explicit attribution path",
        }),
        json!({
            "item": "ep_route_and_transfer_counts",
            "status": "measured",
            "source": format!("{backend} EP counters recorded by the DeepSeek-V2-Lite EP2 runtime"),
        }),
        json!({
            "item": "gpu_event_timing",
            "status": gpu_timing_status(attribution),
            "source": "CUDA event timing around selected GPU/NCCL stream sections; pure host sections and mixed attention_host_path are not represented as GPU event rows; failures are counted separately from the accuracy oracle",
        }),
        json!({
            "item": "nvtx_ranges",
            "status": if attribution.nvtx_enabled() { "emitted" } else { "available_when_enabled" },
            "source": "set PEGAINFER_DSV2_LITE_NVTX=1 to emit NVTX ranges for the same selected GPU/NCCL attribution sections",
        }),
        json!({
            "item": "throughput_or_production_ep_readiness",
            "status": "not_claimed",
            "source": "single batch=1 prompt=Hello output_len=16 diagnostic gate only",
        }),
    ]
}

fn gpu_timing_status(attribution: &pegainfer_deepseek_v2_lite::DecodeAttributionProfile) -> &str {
    match (
        attribution.gpu_sample_count(),
        attribution.gpu_timing_failure_count(),
    ) {
        (0, 0) => "not_covered",
        (0, _) => "timing_failed",
        (_, 0) => "measured",
        (_, _) => "measured_with_failures",
    }
}

fn latency_stats(samples: &[u64]) -> Value {
    if samples.is_empty() {
        return json!({
            "count": 0,
            "total_us": 0,
            "mean_us": 0.0,
            "min_us": 0,
            "p50_us": 0,
            "p95_us": 0,
            "p99_us": 0,
            "max_us": 0,
        });
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let total_us: u64 = sorted.iter().sum();
    json!({
        "count": sorted.len(),
        "total_us": total_us,
        "mean_us": total_us as f64 / sorted.len() as f64,
        "min_us": sorted[0],
        "p50_us": percentile(&sorted, 0.50),
        "p95_us": percentile(&sorted, 0.95),
        "p99_us": percentile(&sorted, 0.99),
        "max_us": sorted[sorted.len() - 1],
    })
}

fn percentile(sorted: &[u64], quantile: f64) -> u64 {
    let idx = ((sorted.len() as f64 - 1.0) * quantile).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn sha256_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
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
