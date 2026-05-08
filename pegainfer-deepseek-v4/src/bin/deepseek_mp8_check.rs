use std::{env, path::PathBuf};

use anyhow::{Context, Result, bail};
use pegainfer_deepseek_v4::{
    Config, F32Logits, RankGpuContext, RankWeightView, TensorParallelConfig, load_rank_manifest,
    load_rank_subset_to_gpu, load_rank_to_gpu, prefill_logits_group_bf16_hidden,
};
use serde::Deserialize;
use vllm_text::tokenizer::{HuggingFaceTokenizer, Tokenizer};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_path = args
        .get(1)
        .map(PathBuf::from)
        .or_else(|| env::var_os("PEGAINFER_TEST_MODEL_PATH").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("models/DeepSeek-V4-Flash"));

    let rank = parse_flag_usize(&args, "--rank")?.unwrap_or(0);
    let gpu_subset = args.iter().any(|arg| arg == "--gpu-subset");
    let gpu_all = args.iter().any(|arg| arg == "--gpu-all");
    let gpu_all_ranks = args.iter().any(|arg| arg == "--gpu-all-ranks");
    let all_ranks = args.iter().any(|arg| arg == "--all-ranks");
    let prefill_check = args.iter().any(|arg| arg == "--prefill-check");
    let prefill_seq_len = parse_flag_usize(&args, "--seq-len")?.unwrap_or(128);
    let ground_truth_next =
        parse_flag_string(&args, "--ground-truth-next-token")?.map(PathBuf::from);
    let gt_limit = parse_flag_usize(&args, "--gt-limit")?.unwrap_or(1);
    let gt_offset = parse_flag_usize(&args, "--gt-offset")?.unwrap_or(0);

    let config = Config::from_model_dir(&model_path)
        .with_context(|| format!("failed to load config from {}", model_path.display()))?;

    if let Some(ground_truth) = ground_truth_next {
        ground_truth_next_token_check(&model_path, &config, &ground_truth, gt_offset, gt_limit)?;
    } else if prefill_check {
        full_prefill_check(&model_path, &config, prefill_seq_len)?;
    } else if gpu_all_ranks {
        load_all_ranks_to_gpu(&model_path, &config)?;
    } else if all_ranks {
        for rank in 0..8 {
            check_rank(&model_path, &config, rank, false, false)?;
        }
    } else {
        check_rank(&model_path, &config, rank, gpu_subset, gpu_all)?;
    }

    Ok(())
}

#[derive(Deserialize)]
struct GroundTruthCase {
    question: String,
    answer: String,
}

struct FullPrefillRuntime<'a> {
    contexts: Vec<RankGpuContext>,
    views: Vec<RankWeightView<'a>>,
    comms: Vec<cudarc::nccl::safe::Comm>,
}

fn ground_truth_next_token_check(
    model_path: &PathBuf,
    config: &Config,
    ground_truth_path: &PathBuf,
    offset: usize,
    limit: usize,
) -> Result<()> {
    let cases = load_ground_truth_cases(ground_truth_path)?;
    let tokenizer = load_tokenizer(model_path)?;
    let runtime = load_full_prefill_runtime(model_path, config)?;

    for (idx, case) in cases.iter().enumerate().skip(offset).take(limit) {
        let prompt = encode_dsv4_chat_prompt(&case.question);
        let prompt_tokens = tokenizer
            .encode(&prompt, false)
            .map_err(|err| anyhow::anyhow!("tokenize ground-truth case {idx} failed: {err:?}"))?;
        let logits = run_prefill_logits(&runtime, config, &prompt_tokens)?;
        let rank0 = logits[0].to_host(&runtime.contexts[0])?;
        let token = argmax_f32(&rank0) as u32;
        let topk = topk_f32(&rank0, 10);
        let token_text = tokenizer
            .decode(&[token], false)
            .map_err(|err| anyhow::anyhow!("decode next token {token} failed: {err:?}"))?;
        println!(
            "gt_case={} prompt_tokens={} next_token={} next_text={:?} topk={:?} expected_answer={:?}",
            idx,
            prompt_tokens.len(),
            token,
            token_text,
            topk,
            case.answer
        );
    }
    Ok(())
}

fn load_ground_truth_cases(ground_truth_path: &PathBuf) -> Result<Vec<GroundTruthCase>> {
    serde_json::from_reader(
        std::fs::File::open(ground_truth_path)
            .with_context(|| format!("open {}", ground_truth_path.display()))?,
    )
    .with_context(|| format!("parse {}", ground_truth_path.display()))
}

fn load_tokenizer(model_path: &PathBuf) -> Result<HuggingFaceTokenizer> {
    let tokenizer_path = model_path.join("tokenizer.json");
    HuggingFaceTokenizer::new(&tokenizer_path).map_err(|err| {
        anyhow::anyhow!(
            "load tokenizer {} failed: {err:?}",
            tokenizer_path.display()
        )
    })
}

fn load_full_prefill_runtime<'a>(
    model_path: &PathBuf,
    config: &'a Config,
) -> Result<FullPrefillRuntime<'a>> {
    let mut contexts = Vec::with_capacity(8);
    for rank in 0..8 {
        contexts.push(RankGpuContext::new(rank)?);
    }
    let weights = contexts
        .iter()
        .enumerate()
        .map(|(rank, ctx)| {
            load_rank_to_gpu(ctx, model_path, config, TensorParallelConfig::mp8(rank))
        })
        .collect::<Result<Vec<_>>>()?;
    let weights: &'static [_] = Box::leak(weights.into_boxed_slice());
    let views = weights
        .iter()
        .map(|weights| weights.view(config))
        .collect::<Result<Vec<_>>>()?;
    let streams = contexts
        .iter()
        .map(|ctx| ctx.stream.clone())
        .collect::<Vec<_>>();
    let comms = cudarc::nccl::safe::Comm::from_devices(streams)
        .map_err(|err| anyhow::anyhow!("NCCL comm creation failed: {err:?}"))?;
    Ok(FullPrefillRuntime {
        contexts,
        views,
        comms,
    })
}

fn run_prefill_logits(
    runtime: &FullPrefillRuntime<'_>,
    config: &Config,
    token_ids_host: &[u32],
) -> Result<Vec<F32Logits>> {
    let token_ids = runtime
        .contexts
        .iter()
        .map(|ctx| ctx.stream.clone_htod(token_ids_host).map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    for ctx in &runtime.contexts {
        ctx.sync()?;
    }
    let inputs = (0..8)
        .map(|rank| {
            (
                &runtime.contexts[rank],
                &runtime.views[rank],
                &runtime.comms[rank],
                &token_ids[rank],
            )
        })
        .collect::<Vec<_>>();
    prefill_logits_group_bf16_hidden(&inputs, config, token_ids_host.len())
}

fn encode_dsv4_chat_prompt(question: &str) -> String {
    format!("<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜></think>")
}

fn argmax_f32(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best {
            best = value;
            best_idx = idx;
        }
    }
    best_idx
}

fn topk_f32(values: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<_> = values.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|(_, a), (_, b)| b.total_cmp(a));
    indexed[..k.min(indexed.len())]
        .iter()
        .map(|(token, logit)| (*token as u32, *logit))
        .collect()
}

fn parse_flag_usize(args: &[String], name: &str) -> Result<Option<usize>> {
    let Some(index) = args.iter().position(|arg| arg == name) else {
        return Ok(None);
    };
    let value = args
        .get(index + 1)
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))?;
    value
        .parse::<usize>()
        .with_context(|| format!("invalid value for {name}: {value}"))
        .map(Some)
}

fn parse_flag_string(args: &[String], name: &str) -> Result<Option<String>> {
    let Some(index) = args.iter().position(|arg| arg == name) else {
        return Ok(None);
    };
    let value = args
        .get(index + 1)
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))?;
    Ok(Some(value.clone()))
}

fn full_prefill_check(model_path: &PathBuf, config: &Config, seq_len: usize) -> Result<()> {
    if seq_len == 0 {
        bail!("--seq-len must be positive");
    }
    if seq_len < 128 && config.compress_ratios.iter().any(|ratio| *ratio == 128) {
        bail!("--seq-len must be at least 128 for ratio-128 compressed layers");
    }

    let token_ids_host = (0..seq_len).map(|token| token as u32).collect::<Vec<_>>();
    let runtime = load_full_prefill_runtime(model_path, config)?;
    let logits = run_prefill_logits(&runtime, config, &token_ids_host)?;
    let rank0 = logits[0].to_host(&runtime.contexts[0])?;
    let rank7 = logits[7].to_host(&runtime.contexts[7])?;
    println!(
        "prefill_check ranks={} seq_len={} vocab={} rank0_logits={} rank0_rank7_equal={}",
        logits.len(),
        seq_len,
        logits[0].vocab_size,
        rank0.len(),
        rank0 == rank7
    );
    let finite = rank0.iter().all(|value| value.is_finite());
    let nonzero = rank0.iter().any(|value| *value != 0.0);
    println!("rank0_logits_finite={finite} rank0_logits_nonzero={nonzero}");
    if !finite || !nonzero || rank0 != rank7 {
        bail!("full prefill check produced invalid logits");
    }
    Ok(())
}

fn load_all_ranks_to_gpu(model_path: &PathBuf, config: &Config) -> Result<()> {
    let mut loaded_ranks = Vec::new();
    let mut total_bytes = 0usize;
    for rank in 0..8 {
        let tp = TensorParallelConfig::mp8(rank);
        let ctx = RankGpuContext::new(rank)?;
        let weights = load_rank_to_gpu(&ctx, model_path, config, tp)?;
        ctx.sync()?;
        total_bytes += weights.total_bytes;
        println!(
            "gpu_rank={} tensors={} bytes={}",
            rank,
            weights.tensors.len(),
            weights.total_bytes
        );
        loaded_ranks.push((ctx, weights));
    }
    println!(
        "gpu_all_ranks={} total_bytes={}",
        loaded_ranks.len(),
        total_bytes
    );
    Ok(())
}

fn check_rank(
    model_path: &PathBuf,
    config: &Config,
    rank: usize,
    gpu_subset: bool,
    gpu_all: bool,
) -> Result<()> {
    if rank >= 8 {
        bail!("rank must be in 0..8, got {rank}");
    }
    let tp = TensorParallelConfig::mp8(rank);
    let manifest = load_rank_manifest(model_path, config, tp)?;
    println!(
        "rank={} path={} tensors={}",
        manifest.rank,
        manifest.path.display(),
        manifest.tensors.len()
    );
    println!(
        "embed={:?} head={:?}",
        manifest.tensors["embed.weight"].shape, manifest.tensors["head.weight"].shape
    );
    println!(
        "layer0 wq_a={:?}/{:?} expert0 w1={:?}/{:?}",
        manifest.tensors["layers.0.attn.wq_a.weight"].shape,
        manifest.tensors["layers.0.attn.wq_a.weight"].dtype,
        manifest.tensors[&format!("layers.0.ffn.experts.{}.w1.weight", rank * 32)].shape,
        manifest.tensors[&format!("layers.0.ffn.experts.{}.w1.weight", rank * 32)].dtype,
    );

    if gpu_subset || gpu_all {
        let ctx = RankGpuContext::new(rank)?;
        if gpu_all {
            let weights = load_rank_to_gpu(&ctx, model_path, config, tp)?;
            ctx.sync()?;
            println!(
                "gpu_all_tensors={} bytes={}",
                weights.tensors.len(),
                weights.total_bytes
            );
            return Ok(());
        }

        let loaded = load_rank_subset_to_gpu(
            &ctx,
            model_path,
            config,
            tp,
            &[
                "embed.weight",
                "layers.0.attn.wq_a.weight",
                "layers.0.attn.wq_a.scale",
                &format!("layers.0.ffn.experts.{}.w1.weight", rank * 32),
                &format!("layers.0.ffn.experts.{}.w1.scale", rank * 32),
            ],
        )?;
        ctx.sync()?;
        let total_bytes: usize = loaded.values().map(|tensor| tensor.bytes).sum();
        println!("gpu_subset_tensors={} bytes={}", loaded.len(), total_bytes);
    }

    Ok(())
}
