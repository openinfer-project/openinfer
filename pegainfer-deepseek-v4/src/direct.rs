use std::{path::Path, sync::mpsc as std_mpsc, thread};

use anyhow::{Context, Result, bail};
use cudarc::driver::{CudaSlice, DevicePtrMut, result as cuda_result};
use log::{info, warn};
use pegainfer_core::engine::{
    EngineHandle, EngineLoadOptions, FinishReason, GenerateRequest, TokenEvent,
};
use tokio::sync::mpsc;

use crate::{
    Config, DeepSeekRopeCache, F32Logits, LayerDecodeCache, RankGpuContext, RankWeightView,
    TensorParallelConfig, block_decode_group_bf16_hidden, embedding_vocab_parallel_group,
    final_logits_group_bf16_hidden, hc_expand_bf16_hidden, load_rank_to_gpu, precompute_rope_cache,
    prefill_logits_and_decode_cache_group_bf16_hidden,
};

struct FullDirectRuntime<'a> {
    contexts: Vec<RankGpuContext>,
    views: Vec<RankWeightView<'a>>,
    comms: Vec<cudarc::nccl::safe::Comm>,
    caches: Vec<Vec<LayerDecodeCache>>,
    ropes: Vec<Vec<DeepSeekRopeCache>>,
    max_cache_seq_len: usize,
}

pub struct DirectGeneration {
    pub generated: Vec<u32>,
    pub finish_reason: FinishReason,
}

pub struct DeepSeekV4DirectGenerator {
    config: &'static Config,
    runtime: FullDirectRuntime<'static>,
}

impl DeepSeekV4DirectGenerator {
    pub fn from_model_dir(model_path: &Path) -> Result<Self> {
        let config = Box::leak(Box::new(Config::from_model_dir(model_path).with_context(
            || {
                format!(
                    "failed to load DeepSeek V4 config from {}",
                    model_path.display()
                )
            },
        )?));
        let runtime = load_full_direct_runtime(model_path, config)?;
        Ok(Self { config, runtime })
    }

    pub fn eos_token_id(&self) -> usize {
        self.config.eos_token_id
    }

    pub fn generate_greedy<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
        mut on_token: F,
    ) -> Result<DirectGeneration>
    where
        F: FnMut(u32) -> Result<()>,
    {
        if prompt_tokens.is_empty() {
            bail!("DeepSeek V4 request produced an empty prompt");
        }
        if max_new_tokens == 0 {
            return Ok(DirectGeneration {
                generated: Vec::new(),
                finish_reason: FinishReason::Length,
            });
        }

        ensure_direct_decode_caches(
            &mut self.runtime,
            self.config,
            prompt_tokens.len() + max_new_tokens,
        )?;

        let mut next_logits = run_prefill_logits_and_seed_decode_cache(
            &mut self.runtime,
            self.config,
            prompt_tokens,
        )?;
        let mut generated = Vec::with_capacity(max_new_tokens);

        for step in 0..max_new_tokens {
            let rank0 = next_logits[0].to_host(&self.runtime.contexts[0])?;
            let token = argmax_f32(&rank0) as u32;
            if !ignore_eos && token as usize == self.config.eos_token_id {
                return Ok(DirectGeneration {
                    generated,
                    finish_reason: FinishReason::Stop,
                });
            }
            on_token(token)?;
            generated.push(token);
            if step + 1 == max_new_tokens {
                break;
            }
            next_logits = run_direct_decode_logits(
                &mut self.runtime,
                self.config,
                token,
                prompt_tokens.len() + step,
            )?;
        }

        Ok(DirectGeneration {
            generated,
            finish_reason: FinishReason::Length,
        })
    }
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    if options.device_ordinals != (0..8).collect::<Vec<_>>() {
        bail!(
            "DeepSeek V4 MP8 currently requires device_ordinals=0..7, got {:?}",
            options.device_ordinals
        );
    }
    if options.enable_cuda_graph {
        warn!("DeepSeek V4 direct engine does not use CUDA graph yet");
    }
    let model_path = model_path.to_path_buf();
    let (submit_tx, mut submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
    let (init_tx, init_rx) = std_mpsc::channel::<Result<()>>();
    thread::Builder::new()
        .name("deepseek-v4-direct".into())
        .spawn(move || {
            let mut generator = match DeepSeekV4DirectGenerator::from_model_dir(&model_path) {
                Ok(generator) => {
                    let _ = init_tx.send(Ok(()));
                    generator
                }
                Err(err) => {
                    let _ = init_tx.send(Err(err));
                    return;
                }
            };
            info!("DeepSeek V4 direct engine ready");
            while let Some(req) = submit_rx.blocking_recv() {
                handle_request(&mut generator, req);
            }
            info!("DeepSeek V4 direct engine exiting");
        })
        .expect("failed to spawn DeepSeek V4 direct engine thread");
    init_rx
        .recv()
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 engine init channel closed: {err}"))??;
    Ok(EngineHandle::new(submit_tx))
}

fn handle_request(generator: &mut DeepSeekV4DirectGenerator, req: GenerateRequest) {
    let prompt_len = req.prompt_tokens.len();
    if req.echo {
        let _ = req.token_tx.send(TokenEvent::PromptTokens {
            ids: req.prompt_tokens.clone(),
            logprobs: vec![None; prompt_len],
        });
    }
    if req.params.temperature > 0.0 || req.params.top_k != -1 || req.params.top_p < 1.0 {
        reject_request(
            &req,
            prompt_len,
            format!(
                "DeepSeek V4 direct engine currently serves greedy decoding only; requested temperature={}, top_k={}, top_p={}",
                req.params.temperature, req.params.top_k, req.params.top_p
            ),
        );
        return;
    }
    if req.logprobs > 0 {
        reject_request(
            &req,
            prompt_len,
            "DeepSeek V4 direct engine does not return logprobs yet".to_string(),
        );
        return;
    }

    let token_tx = req.token_tx.clone();
    let result = generator.generate_greedy(
        &req.prompt_tokens,
        req.max_tokens,
        req.params.ignore_eos,
        |token| {
            token_tx
                .send(TokenEvent::Token {
                    id: token,
                    logprob: None,
                })
                .map_err(|_| anyhow::anyhow!("request receiver dropped"))?;
            Ok(())
        },
    );
    match result {
        Ok(generation) => {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: generation.finish_reason,
                prompt_tokens: prompt_len,
                completion_tokens: generation.generated.len(),
            });
        }
        Err(err) => {
            let message = format!("DeepSeek V4 direct request failed: {err:#}");
            warn!("{message}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
        }
    }
}

fn reject_request(req: &GenerateRequest, prompt_len: usize, reason: String) {
    warn!("{reason}");
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message: reason,
        prompt_tokens: prompt_len,
        completion_tokens: 0,
    });
}

fn load_full_direct_runtime(
    model_path: &Path,
    config: &'static Config,
) -> Result<FullDirectRuntime<'static>> {
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
    Ok(FullDirectRuntime {
        contexts,
        views,
        comms,
        caches: Vec::new(),
        ropes: Vec::new(),
        max_cache_seq_len: 0,
    })
}

fn allocate_direct_decode_caches(
    contexts: &[RankGpuContext],
    config: &Config,
    max_seq_len: usize,
) -> Result<Vec<Vec<LayerDecodeCache>>> {
    let mut layers = Vec::with_capacity(config.n_layers);
    for layer in 0..config.n_layers {
        let mut rank_caches = Vec::with_capacity(contexts.len());
        for ctx in contexts {
            rank_caches.push(LayerDecodeCache::zeros_with_max_seq(
                ctx,
                config,
                layer,
                max_seq_len,
            )?);
        }
        layers.push(rank_caches);
    }
    Ok(layers)
}

fn allocate_direct_rope_caches(
    contexts: &[RankGpuContext],
    config: &Config,
    max_seq_len: usize,
) -> Result<Vec<Vec<DeepSeekRopeCache>>> {
    let mut layers = Vec::with_capacity(config.n_layers);
    for layer in 0..config.n_layers {
        let mut rank_ropes = Vec::with_capacity(contexts.len());
        for ctx in contexts {
            rank_ropes.push(precompute_rope_cache(ctx, config, layer, max_seq_len)?);
        }
        layers.push(rank_ropes);
    }
    Ok(layers)
}

fn ensure_direct_decode_caches(
    runtime: &mut FullDirectRuntime<'_>,
    config: &Config,
    max_seq_len: usize,
) -> Result<()> {
    if runtime.caches.len() != config.n_layers || runtime.max_cache_seq_len < max_seq_len {
        runtime.caches = allocate_direct_decode_caches(&runtime.contexts, config, max_seq_len)?;
        runtime.ropes = allocate_direct_rope_caches(&runtime.contexts, config, max_seq_len)?;
        runtime.max_cache_seq_len = max_seq_len;
    } else {
        reset_direct_decode_caches(runtime)?;
    }
    Ok(())
}

fn reset_direct_decode_caches(runtime: &mut FullDirectRuntime<'_>) -> Result<()> {
    for rank_caches in runtime.caches.iter_mut() {
        for (ctx, cache) in runtime.contexts.iter().zip(rank_caches.iter_mut()) {
            zero_cuda_slice(ctx, &mut cache.kv.data)?;
            if let Some(compressor) = cache.compressor.as_mut() {
                zero_cuda_slice(ctx, &mut compressor.kv)?;
                fill_f32_cuda_slice(ctx, &mut compressor.score, f32::NEG_INFINITY)?;
            }
            if let Some(indexer_kv) = cache.indexer_kv.as_mut() {
                zero_cuda_slice(ctx, &mut indexer_kv.data)?;
            }
            if let Some(indexer_compressor) = cache.indexer_compressor.as_mut() {
                zero_cuda_slice(ctx, &mut indexer_compressor.kv)?;
                fill_f32_cuda_slice(ctx, &mut indexer_compressor.score, f32::NEG_INFINITY)?;
            }
        }
    }
    Ok(())
}

fn zero_cuda_slice<T>(ctx: &RankGpuContext, slice: &mut CudaSlice<T>) -> Result<()> {
    ctx.set_current()?;
    let bytes = slice.num_bytes();
    if bytes == 0 {
        return Ok(());
    }
    let (ptr, _guard) = slice.device_ptr_mut(&ctx.stream);
    unsafe {
        cuda_result::memset_d8_async(ptr, 0, bytes, ctx.stream.cu_stream())?;
    }
    Ok(())
}

fn fill_f32_cuda_slice(ctx: &RankGpuContext, slice: &mut CudaSlice<f32>, value: f32) -> Result<()> {
    ctx.set_current()?;
    let host = vec![value; slice.len()];
    ctx.stream.memcpy_htod(&host, slice)?;
    Ok(())
}

fn run_direct_decode_logits(
    runtime: &mut FullDirectRuntime<'_>,
    config: &Config,
    token_id: u32,
    start_pos: usize,
) -> Result<Vec<F32Logits>> {
    if runtime.caches.len() != config.n_layers {
        bail!(
            "direct decode caches are not initialized: have {}, need {}",
            runtime.caches.len(),
            config.n_layers
        );
    }
    let token_ids = runtime
        .contexts
        .iter()
        .enumerate()
        .map(|(rank, ctx)| {
            ctx.stream
                .clone_htod(&[token_id])
                .with_context(|| format!("copy token_id to rank {rank}"))
        })
        .collect::<Result<Vec<_>>>()?;
    for ctx in &runtime.contexts {
        ctx.sync().context("sync after token_id copy")?;
    }
    let embedding_inputs = (0..8)
        .map(|rank| {
            (
                &runtime.contexts[rank],
                &runtime.views[rank],
                &runtime.comms[rank],
                &token_ids[rank],
            )
        })
        .collect::<Vec<_>>();
    let hidden = embedding_vocab_parallel_group(&embedding_inputs, config, 1)
        .context("embedding_vocab_parallel_group")?;
    let mut hcs = (0..8)
        .map(|rank| {
            hc_expand_bf16_hidden(&runtime.contexts[rank], &hidden[rank], config.hc_mult)
                .with_context(|| format!("hc_expand rank {rank}"))
        })
        .collect::<Result<Vec<_>>>()?;

    for layer in 0..config.n_layers {
        let rope_refs = runtime.ropes[layer].iter().collect::<Vec<_>>();
        let block_inputs = (0..8)
            .map(|rank| {
                (
                    &runtime.contexts[rank],
                    &runtime.views[rank],
                    &runtime.comms[rank],
                    &hcs[rank],
                    &token_ids[rank],
                )
            })
            .collect::<Vec<_>>();
        hcs = block_decode_group_bf16_hidden(
            &block_inputs,
            config,
            layer,
            &rope_refs,
            start_pos,
            &mut runtime.caches[layer],
        )
        .with_context(|| format!("block_decode_group_bf16_hidden layer {layer} pos {start_pos}"))?;
        for ctx in &runtime.contexts {
            ctx.sync()
                .with_context(|| format!("sync after layer {layer} pos {start_pos}"))?;
        }
    }

    let logits_inputs = (0..8)
        .map(|rank| {
            (
                &runtime.contexts[rank],
                &runtime.views[rank],
                &runtime.comms[rank],
                &hcs[rank],
            )
        })
        .collect::<Vec<_>>();
    final_logits_group_bf16_hidden(&logits_inputs, config).context("final_logits_group_bf16_hidden")
}

fn run_prefill_logits_and_seed_decode_cache(
    runtime: &mut FullDirectRuntime<'_>,
    config: &Config,
    prompt_tokens: &[u32],
) -> Result<Vec<F32Logits>> {
    if runtime.caches.len() != config.n_layers {
        bail!(
            "direct decode caches are not initialized: have {}, need {}",
            runtime.caches.len(),
            config.n_layers
        );
    }
    let seq_len = prompt_tokens.len();
    let token_ids = runtime
        .contexts
        .iter()
        .enumerate()
        .map(|(rank, ctx)| {
            ctx.stream
                .clone_htod(prompt_tokens)
                .with_context(|| format!("copy prompt tokens to rank {rank}"))
        })
        .collect::<Result<Vec<_>>>()?;
    for ctx in &runtime.contexts {
        ctx.sync().context("sync after prompt token copy")?;
    }
    let prefill_inputs = (0..8)
        .map(|rank| {
            (
                &runtime.contexts[rank],
                &runtime.views[rank],
                &runtime.comms[rank],
                &token_ids[rank],
            )
        })
        .collect::<Vec<_>>();
    prefill_logits_and_decode_cache_group_bf16_hidden(
        &prefill_inputs,
        config,
        seq_len,
        &mut runtime.caches,
    )
    .context("prefill_logits_and_decode_cache_group_bf16_hidden")
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
