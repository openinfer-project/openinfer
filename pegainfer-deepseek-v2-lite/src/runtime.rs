use std::{
    env,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use pegainfer_core::{
    ops,
    tensor::{DeviceVec, HiddenStates},
};
use pegainfer_engine::engine::{EngineLoadOptions, FinishReason};
use sha2::{Digest, Sha256};

use crate::{
    Config,
    attribution::DecodeAttributionProfile,
    device::activate,
    ep::ExpertParallelConfig,
    host_ops::{
        DecodeCache, LayerCache, append_kv_and_build_queries,
        append_kv_and_build_queries_decode_batch, compute_attention_host,
        compute_attention_host_decode_batch, gate_logits_host, hidden_from_bf16_host,
        hidden_from_f32_host, hidden_to_bf16, hidden_to_f32, normalize_compressed_kv,
        rms_norm_hidden_host, rms_norm_host, topk_softmax_routes,
    },
    model::{
        AttentionWeights, DriverRankModel, ExpertMlp, ExpertRankModel, MlpWeights, MoeMlp,
        dense_mlp_forward, dense_mlp_forward_per_token,
    },
    nccl_backend::NaiveNcclEp2Backend,
    weights::{ModelManifest, RankLoadPlan},
};

const EP_BACKEND_ENV: &str = "PEGAINFER_DSV2_LITE_EP_BACKEND";
const HOST_STAGED_BACKEND: &str = "host-staged";
const NCCL_BACKEND: &str = "nccl";

#[derive(Clone, Debug, Default)]
pub struct GenerationStats {
    pub model_path: PathBuf,
    pub device_ordinals: Vec<usize>,
    pub ep_backend: String,
    pub ep_size: usize,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub host_dispatch_calls: usize,
    pub host_dispatch_elements: usize,
    pub host_combine_calls: usize,
    pub host_combine_elements: usize,
    pub host_dispatch_local_routes: usize,
    pub host_dispatch_remote_routes: usize,
    pub nccl_dispatch_local_routes: usize,
    pub nccl_dispatch_remote_routes: usize,
    pub nccl_combine_routes: usize,
    pub nccl_dense_exchange_calls: usize,
    pub nccl_combine_calls: usize,
    pub nccl_dense_exchange_elements: usize,
    pub nccl_combine_elements: usize,
    pub output_token_sha256: String,
}

#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    pub stats: GenerationStats,
}

#[derive(Clone, Debug)]
pub struct BatchedGenerationResult {
    pub tokens: Vec<Vec<u32>>,
    pub prefill_next_token_us: Vec<u64>,
    pub per_token_decode_us: Vec<u64>,
    pub total_generation_us: u64,
}

pub struct DeepSeekV2LiteEp2Generator {
    model_path: PathBuf,
    device_ordinals: Vec<usize>,
    config: Config,
    rank0: DriverRankModel,
    rank1: ExpertRankModel,
    backend: EpBackendRuntime,
}

// SAFETY: The generator is driven by exactly one worker thread after load. It
// switches CUDA devices explicitly before every rank-local op and recreates the
// thread-local cuBLAS handle when the active device changes.
unsafe impl Send for DeepSeekV2LiteEp2Generator {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EpBackendKind {
    HostStaged,
    Nccl,
}

impl EpBackendKind {
    fn from_env() -> Result<Self> {
        let raw = env::var(EP_BACKEND_ENV).ok();
        parse_backend(raw.as_deref())
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::HostStaged => HOST_STAGED_BACKEND,
            Self::Nccl => NCCL_BACKEND,
        }
    }
}

enum EpBackendRuntime {
    HostStaged,
    Nccl(NaiveNcclEp2Backend),
}

impl EpBackendRuntime {
    fn new(
        kind: EpBackendKind,
        rank0: &pegainfer_core::tensor::DeviceContext,
        rank1: &pegainfer_core::tensor::DeviceContext,
    ) -> Result<Self> {
        match kind {
            EpBackendKind::HostStaged => Ok(Self::HostStaged),
            EpBackendKind::Nccl => Ok(Self::Nccl(NaiveNcclEp2Backend::new(rank0, rank1)?)),
        }
    }

    fn kind(&self) -> EpBackendKind {
        match self {
            Self::HostStaged => EpBackendKind::HostStaged,
            Self::Nccl(_) => EpBackendKind::Nccl,
        }
    }
}

fn parse_backend(raw: Option<&str>) -> Result<EpBackendKind> {
    match raw.unwrap_or(HOST_STAGED_BACKEND) {
        HOST_STAGED_BACKEND => Ok(EpBackendKind::HostStaged),
        NCCL_BACKEND => Ok(EpBackendKind::Nccl),
        other => bail!(
            "DeepSeek-V2-Lite EP=2 backend '{other}' is not supported; supported backends: {HOST_STAGED_BACKEND}, {NCCL_BACKEND}"
        ),
    }
}

impl GenerationStats {
    fn record_routes(&mut self, backend: EpBackendKind, local_routes: usize, remote_routes: usize) {
        match backend {
            EpBackendKind::HostStaged => {
                self.host_dispatch_local_routes += local_routes;
                self.host_dispatch_remote_routes += remote_routes;
            }
            EpBackendKind::Nccl => {
                self.nccl_dispatch_local_routes += local_routes;
                self.nccl_dispatch_remote_routes += remote_routes;
                self.nccl_combine_routes += local_routes + remote_routes;
            }
        }
    }

    fn record_host_staged_moe(&mut self, hidden_dim: usize, route_count: usize) {
        let elements = hidden_dim * route_count;
        self.host_dispatch_calls += 1;
        self.host_combine_calls += 1;
        self.host_dispatch_elements += elements;
        self.host_combine_elements += elements;
    }

    fn record_nccl_moe_collectives(&mut self, hidden_dim: usize, seq_len: usize) {
        let elements = hidden_dim * seq_len;
        self.nccl_dense_exchange_calls += 1;
        self.nccl_combine_calls += 1;
        self.nccl_dense_exchange_elements += elements;
        self.nccl_combine_elements += elements;
    }
}

impl DeepSeekV2LiteEp2Generator {
    pub fn load(model_path: &Path, options: EngineLoadOptions) -> Result<Self> {
        let config = Config::from_model_dir(model_path)?;
        ensure!(
            !options.enable_cuda_graph,
            "DeepSeek-V2-Lite EP=2 first gate requires cuda_graph disabled"
        );
        let backend_kind = validate_backend_and_devices(&options.device_ordinals)?;

        let rank0_layout = ExpertParallelConfig::ep2(0).validate_for(&config)?;
        let rank1_layout = ExpertParallelConfig::ep2(1).validate_for(&config)?;
        let manifest = ModelManifest::from_model_dir(model_path)?;
        manifest.validate_rank_plan(&RankLoadPlan::for_driver_rank(&config, &rank0_layout))?;
        manifest.validate_rank_plan(&RankLoadPlan::for_expert_rank(&config, &rank1_layout))?;

        let rank0 = DriverRankModel::load(
            model_path,
            &config,
            rank0_layout,
            options.device_ordinals[0],
        )
        .context("load DeepSeek-V2-Lite EP rank 0")?;
        let rank1 = ExpertRankModel::load(
            model_path,
            &config,
            rank1_layout,
            options.device_ordinals[1],
        )
        .context("load DeepSeek-V2-Lite EP rank 1")?;
        let backend = EpBackendRuntime::new(backend_kind, &rank0.ctx, &rank1.ctx)?;

        Ok(Self {
            model_path: model_path.to_path_buf(),
            device_ordinals: options.device_ordinals,
            config,
            rank0,
            rank1,
            backend,
        })
    }

    pub fn generate_greedy(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
    ) -> Result<GenerationResult> {
        let mut attribution = DecodeAttributionProfile::disabled();
        self.generate_greedy_inner(prompt_tokens, max_new_tokens, ignore_eos, &mut attribution)
    }

    pub fn generate_greedy_with_attribution(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
    ) -> Result<(GenerationResult, DecodeAttributionProfile)> {
        let mut attribution = DecodeAttributionProfile::enabled();
        let result = self.generate_greedy_inner(
            prompt_tokens,
            max_new_tokens,
            ignore_eos,
            &mut attribution,
        )?;
        Ok((result, attribution))
    }

    pub fn generate_greedy_batch_same_prompt_with_timings(
        &mut self,
        prompt_tokens: &[u32],
        batch_size: usize,
        max_new_tokens: usize,
        ignore_eos: bool,
    ) -> Result<BatchedGenerationResult> {
        ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
        ensure!(batch_size > 0, "batch_size must be positive");
        ensure!(
            batch_size <= 8,
            "DeepSeek-V2-Lite batched decode benchmark supports batch_size <= 8, got {batch_size}"
        );
        ensure!(max_new_tokens > 0, "max_new_tokens must be positive");
        ensure!(
            ignore_eos,
            "DeepSeek-V2-Lite batched decode benchmark requires ignore_eos=true so every row has the same output length"
        );

        let requested_context = prompt_tokens.len() + max_new_tokens;
        let supported_context = self.config.supported_plain_rope_context();
        ensure!(
            requested_context <= supported_context,
            "DeepSeek-V2-Lite EP=2 first gate supports plain RoPE context <= {supported_context} tokens; requested prompt_tokens={} max_new_tokens={} total={requested_context}. YaRN rope_scaling long context is not implemented yet.",
            prompt_tokens.len(),
            max_new_tokens
        );

        let generation_start = Instant::now();
        let mut attribution = DecodeAttributionProfile::disabled();
        let mut stats = GenerationStats {
            model_path: self.model_path.clone(),
            device_ordinals: self.device_ordinals.clone(),
            ep_backend: self.backend.kind().as_str().to_string(),
            ep_size: 2,
            prompt_tokens: prompt_tokens.len() * batch_size,
            ..GenerationStats::default()
        };
        let mut caches: Vec<_> = (0..batch_size)
            .map(|_| DecodeCache::new(&self.config))
            .collect();
        let mut generated: Vec<Vec<u32>> = (0..batch_size)
            .map(|_| Vec::with_capacity(max_new_tokens))
            .collect();
        let mut prefill_next_token_us = Vec::with_capacity(batch_size);

        for row in 0..batch_size {
            let next = self.prefill_next_token(
                prompt_tokens,
                &mut caches[row],
                &mut stats,
                &mut attribution,
            )?;
            prefill_next_token_us.push(duration_micros(generation_start.elapsed()));
            generated[row].push(next);
        }

        let mut per_token_decode_us = Vec::with_capacity(max_new_tokens.saturating_sub(1));
        for token_index in 1..max_new_tokens {
            let input_tokens: Vec<_> = generated
                .iter()
                .map(|tokens| {
                    *tokens
                        .last()
                        .expect("batched decode rows are seeded by prefill")
                })
                .collect();
            let position = prompt_tokens.len() + token_index - 1;
            let decode_start = Instant::now();
            let next_tokens = self.decode_next_tokens_batch(
                &input_tokens,
                position,
                &mut caches,
                &mut stats,
                &mut attribution,
                token_index,
            )?;
            ensure!(
                next_tokens.len() == batch_size,
                "batched decode returned {} tokens for batch_size={batch_size}",
                next_tokens.len()
            );
            per_token_decode_us.push(duration_micros(decode_start.elapsed()));
            for (row, token) in next_tokens.into_iter().enumerate() {
                generated[row].push(token);
            }
        }

        ensure_same_prompt_batch_rows_match(&generated)?;
        Ok(BatchedGenerationResult {
            tokens: generated,
            prefill_next_token_us,
            per_token_decode_us,
            total_generation_us: duration_micros(generation_start.elapsed()),
        })
    }

    fn generate_greedy_inner(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
        attribution: &mut DecodeAttributionProfile,
    ) -> Result<GenerationResult> {
        ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
        ensure!(max_new_tokens > 0, "max_new_tokens must be positive");
        let generation_start = Instant::now();
        let requested_context = prompt_tokens.len() + max_new_tokens;
        let supported_context = self.config.supported_plain_rope_context();
        ensure!(
            requested_context <= supported_context,
            "DeepSeek-V2-Lite EP=2 first gate supports plain RoPE context <= {supported_context} tokens; requested prompt_tokens={} max_new_tokens={} total={requested_context}. YaRN rope_scaling long context is not implemented yet.",
            prompt_tokens.len(),
            max_new_tokens
        );

        let mut stats = GenerationStats {
            model_path: self.model_path.clone(),
            device_ordinals: self.device_ordinals.clone(),
            ep_backend: self.backend.kind().as_str().to_string(),
            ep_size: 2,
            prompt_tokens: prompt_tokens.len(),
            ..GenerationStats::default()
        };

        let mut cache = DecodeCache::new(&self.config);
        let mut generated = Vec::with_capacity(max_new_tokens);
        let prefill_start = Instant::now();
        let mut next =
            self.prefill_next_token(prompt_tokens, &mut cache, &mut stats, attribution)?;
        attribution.set_prefill_next_token(prefill_start.elapsed());
        let mut finish_reason = FinishReason::Length;

        for step in 0..max_new_tokens {
            if let Some(reason) =
                append_generated_token(&mut generated, next, self.config.eos_token_id, ignore_eos)
            {
                finish_reason = reason;
                break;
            }
            if step + 1 == max_new_tokens {
                break;
            }
            let position = prompt_tokens.len() + generated.len() - 1;
            let token_index = generated.len();
            let decode_start = Instant::now();
            next = self.decode_next_token(
                next,
                position,
                &mut cache,
                &mut stats,
                attribution,
                token_index,
            )?;
            attribution.push_decode_token(decode_start.elapsed());
        }

        stats.generated_tokens = generated.len();
        stats.output_token_sha256 = token_sha256(&generated);
        attribution.set_total_generation(generation_start.elapsed());
        Ok(GenerationResult {
            tokens: generated,
            finish_reason,
            stats,
        })
    }

    fn prefill_next_token(
        &mut self,
        prompt_tokens: &[u32],
        cache: &mut DecodeCache,
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
    ) -> Result<u32> {
        let mut hidden = attribution.record_result(
            "prefill",
            "embedding",
            || "prefill.embedding",
            None,
            None,
            || self.embed_tokens(prompt_tokens),
        )?;
        hidden = self.forward_layers(hidden, 0, cache, stats, attribution, "prefill", Some(0))?;
        attribution.record_result(
            "prefill",
            "sample_last_token",
            || "prefill.sample_last_token",
            None,
            Some(0),
            || self.sample_last_token(&hidden),
        )
    }

    fn decode_next_token(
        &mut self,
        token: u32,
        position: usize,
        cache: &mut DecodeCache,
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
        token_index: usize,
    ) -> Result<u32> {
        let mut hidden = attribution.record_result(
            "decode",
            "embedding",
            || "decode.embedding",
            None,
            Some(token_index),
            || self.embed_tokens(&[token]),
        )?;
        hidden = self.forward_layers(
            hidden,
            position,
            cache,
            stats,
            attribution,
            "decode",
            Some(token_index),
        )?;
        attribution.record_result(
            "decode",
            "sample_last_token",
            || "decode.sample_last_token",
            None,
            Some(token_index),
            || self.sample_last_token(&hidden),
        )
    }

    fn decode_next_tokens_batch(
        &mut self,
        tokens: &[u32],
        position: usize,
        caches: &mut [DecodeCache],
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
        token_index: usize,
    ) -> Result<Vec<u32>> {
        ensure!(
            !tokens.is_empty(),
            "batched decode tokens must not be empty"
        );
        ensure!(
            tokens.len() == caches.len(),
            "batched decode token/cache mismatch: tokens={}, caches={}",
            tokens.len(),
            caches.len()
        );
        let mut hidden = attribution.record_result(
            "decode",
            "embedding",
            || "decode.batch_embedding",
            None,
            Some(token_index),
            || self.embed_tokens(tokens),
        )?;
        hidden = self.forward_layers_decode_batch(
            hidden,
            position,
            caches,
            stats,
            attribution,
            token_index,
        )?;
        attribution.record_result(
            "decode",
            "sample_last_token",
            || "decode.batch_sample_tokens",
            None,
            Some(token_index),
            || self.sample_tokens(&hidden),
        )
    }

    fn embed_tokens(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;
        let token_ids_gpu = self.rank0.ctx.stream.clone_htod(token_ids)?;
        let mut out =
            HiddenStates::zeros(&self.rank0.ctx, self.config.hidden_size, token_ids.len())?;
        ops::embedding_batch(
            &self.rank0.ctx,
            &self.rank0.embed_tokens,
            &token_ids_gpu,
            &mut out,
        )?;
        Ok(out)
    }

    fn forward_layers(
        &mut self,
        mut hidden: HiddenStates,
        start_pos: usize,
        cache: &mut DecodeCache,
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
    ) -> Result<HiddenStates> {
        ensure!(
            cache.layers.len() == self.rank0.layers.len(),
            "decode cache layer count mismatch"
        );
        for layer_idx in 0..self.rank0.layers.len() {
            hidden = self
                .forward_layer(
                    layer_idx,
                    &hidden,
                    start_pos,
                    &mut cache.layers[layer_idx],
                    stats,
                    attribution,
                    phase,
                    token_index,
                )
                .with_context(|| format!("DeepSeek-V2-Lite layer {layer_idx}"))?;
        }
        Ok(hidden)
    }

    fn forward_layers_decode_batch(
        &mut self,
        mut hidden: HiddenStates,
        position: usize,
        caches: &mut [DecodeCache],
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
        token_index: usize,
    ) -> Result<HiddenStates> {
        ensure!(
            caches.len() == hidden.seq_len,
            "batched decode cache count {} must match hidden seq_len {}",
            caches.len(),
            hidden.seq_len
        );
        ensure!(
            caches
                .iter()
                .all(|cache| cache.layers.len() == self.rank0.layers.len()),
            "batched decode cache layer count mismatch"
        );
        for layer_idx in 0..self.rank0.layers.len() {
            let mut layer_caches: Vec<_> = caches
                .iter_mut()
                .map(|cache| &mut cache.layers[layer_idx])
                .collect();
            hidden = self
                .forward_layer_decode_batch(
                    layer_idx,
                    &hidden,
                    position,
                    &mut layer_caches,
                    stats,
                    attribution,
                    token_index,
                )
                .with_context(|| format!("DeepSeek-V2-Lite batched layer {layer_idx}"))?;
        }
        Ok(hidden)
    }

    fn forward_layer(
        &mut self,
        layer_idx: usize,
        hidden: &HiddenStates,
        start_pos: usize,
        cache: &mut LayerCache,
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
    ) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;

        let layer = &self.rank0.layers[layer_idx];
        let normed = attribution.record_result(
            phase,
            "host_rms_norm",
            || format!("layer.{layer_idx}.input_rms_norm"),
            Some(layer_idx),
            token_index,
            || self.rms_norm_hidden(hidden, &layer.input_layernorm_host),
        )?;

        let attn = attribution.record_result(
            phase,
            "attention_host_path",
            || format!("layer.{layer_idx}.attention_host_path"),
            Some(layer_idx),
            token_index,
            || self.attention_forward(&normed, &layer.attention, start_pos, cache),
        )?;
        activate(&self.rank0.ctx)?;
        let attn_projected = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "gpu_o_proj_enqueue",
            || format!("layer.{layer_idx}.attention_o_proj"),
            Some(layer_idx),
            token_index,
            || ops::gemm(&self.rank0.ctx, &layer.attention.o_proj, &attn),
        )?;
        let after_attn = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "gpu_residual_add_enqueue",
            || format!("layer.{layer_idx}.attention_residual_add"),
            Some(layer_idx),
            token_index,
            || ops::add_batch(&self.rank0.ctx, hidden, &attn_projected),
        )?;

        let ffn_norm = attribution.record_result(
            phase,
            "host_rms_norm",
            || format!("layer.{layer_idx}.post_attention_rms_norm"),
            Some(layer_idx),
            token_index,
            || self.rms_norm_hidden(&after_attn, &layer.post_attention_layernorm_host),
        )?;

        let (ffn_out, local_routes, remote_routes) = match &layer.mlp {
            MlpWeights::Dense(dense) => (
                attribution.record_gpu_result(
                    &self.rank0.ctx,
                    phase,
                    "dense_mlp_enqueue",
                    || format!("layer.{layer_idx}.dense_mlp"),
                    Some(layer_idx),
                    token_index,
                    || dense_mlp_forward(&self.rank0.ctx, dense, &ffn_norm),
                )?,
                0,
                0,
            ),
            MlpWeights::Moe(moe) => {
                let (ffn_out, local_routes, remote_routes) = self.moe_forward(
                    layer_idx,
                    &ffn_norm,
                    moe,
                    attribution,
                    phase,
                    token_index,
                    false,
                )?;
                match self.backend.kind() {
                    EpBackendKind::HostStaged => {
                        stats.record_host_staged_moe(
                            ffn_norm.hidden_dim,
                            local_routes + remote_routes,
                        );
                    }
                    EpBackendKind::Nccl => {
                        stats.record_nccl_moe_collectives(ffn_norm.hidden_dim, ffn_norm.seq_len);
                    }
                }
                (ffn_out, local_routes, remote_routes)
            }
        };
        stats.record_routes(self.backend.kind(), local_routes, remote_routes);
        attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "gpu_residual_add_enqueue",
            || format!("layer.{layer_idx}.ffn_residual_add"),
            Some(layer_idx),
            token_index,
            || ops::add_batch(&self.rank0.ctx, &after_attn, &ffn_out),
        )
    }

    fn forward_layer_decode_batch(
        &mut self,
        layer_idx: usize,
        hidden: &HiddenStates,
        position: usize,
        caches: &mut [&mut LayerCache],
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
        token_index: usize,
    ) -> Result<HiddenStates> {
        ensure!(
            hidden.seq_len == caches.len(),
            "batched layer hidden seq_len {} must match cache rows {}",
            hidden.seq_len,
            caches.len()
        );
        activate(&self.rank0.ctx)?;

        let layer = &self.rank0.layers[layer_idx];
        let normed = attribution.record_result(
            "decode",
            "host_rms_norm",
            || format!("layer.{layer_idx}.batch_input_rms_norm"),
            Some(layer_idx),
            Some(token_index),
            || self.rms_norm_hidden(hidden, &layer.input_layernorm_host),
        )?;

        let attn = attribution.record_result(
            "decode",
            "attention_host_path",
            || format!("layer.{layer_idx}.batch_attention_host_path"),
            Some(layer_idx),
            Some(token_index),
            || self.attention_forward_decode_batch(&normed, &layer.attention, position, caches),
        )?;
        activate(&self.rank0.ctx)?;
        let attn_projected = attribution.record_gpu_result(
            &self.rank0.ctx,
            "decode",
            "gpu_o_proj_enqueue",
            || format!("layer.{layer_idx}.batch_attention_o_proj"),
            Some(layer_idx),
            Some(token_index),
            || ops::gemm_per_token(&self.rank0.ctx, &layer.attention.o_proj, &attn),
        )?;
        let after_attn = attribution.record_gpu_result(
            &self.rank0.ctx,
            "decode",
            "gpu_residual_add_enqueue",
            || format!("layer.{layer_idx}.batch_attention_residual_add"),
            Some(layer_idx),
            Some(token_index),
            || ops::add_batch(&self.rank0.ctx, hidden, &attn_projected),
        )?;

        let ffn_norm = attribution.record_result(
            "decode",
            "host_rms_norm",
            || format!("layer.{layer_idx}.batch_post_attention_rms_norm"),
            Some(layer_idx),
            Some(token_index),
            || self.rms_norm_hidden(&after_attn, &layer.post_attention_layernorm_host),
        )?;

        let (ffn_out, local_routes, remote_routes) = match &layer.mlp {
            MlpWeights::Dense(dense) => (
                attribution.record_gpu_result(
                    &self.rank0.ctx,
                    "decode",
                    "dense_mlp_enqueue",
                    || format!("layer.{layer_idx}.batch_dense_mlp"),
                    Some(layer_idx),
                    Some(token_index),
                    || dense_mlp_forward_per_token(&self.rank0.ctx, dense, &ffn_norm),
                )?,
                0,
                0,
            ),
            MlpWeights::Moe(moe) => {
                let (ffn_out, local_routes, remote_routes) = self.moe_forward(
                    layer_idx,
                    &ffn_norm,
                    moe,
                    attribution,
                    "decode",
                    Some(token_index),
                    true,
                )?;
                match self.backend.kind() {
                    EpBackendKind::HostStaged => {
                        stats.record_host_staged_moe(
                            ffn_norm.hidden_dim,
                            local_routes + remote_routes,
                        );
                    }
                    EpBackendKind::Nccl => {
                        stats.record_nccl_moe_collectives(ffn_norm.hidden_dim, ffn_norm.seq_len);
                    }
                }
                (ffn_out, local_routes, remote_routes)
            }
        };
        stats.record_routes(self.backend.kind(), local_routes, remote_routes);
        attribution.record_gpu_result(
            &self.rank0.ctx,
            "decode",
            "gpu_residual_add_enqueue",
            || format!("layer.{layer_idx}.batch_ffn_residual_add"),
            Some(layer_idx),
            Some(token_index),
            || ops::add_batch(&self.rank0.ctx, &after_attn, &ffn_out),
        )
    }

    fn attention_forward(
        &self,
        input: &HiddenStates,
        attn: &AttentionWeights,
        start_pos: usize,
        cache: &mut LayerCache,
    ) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;
        ensure!(
            cache.len(&self.config) == start_pos,
            "attention cache position mismatch: cache_len={}, start_pos={start_pos}",
            cache.len(&self.config)
        );

        let q = ops::gemm(&self.rank0.ctx, &attn.q_proj, input)?;
        let kv_a = ops::gemm(&self.rank0.ctx, &attn.kv_a_proj, input)?;
        let q_host = hidden_to_f32(&self.rank0.ctx, &q)?;
        let kv_a_host = hidden_to_f32(&self.rank0.ctx, &kv_a)?;

        let compressed_norm = normalize_compressed_kv(
            &self.config,
            &kv_a_host,
            &attn.kv_a_norm_host,
            input.seq_len,
        );
        let compressed = hidden_from_bf16_host(
            &self.rank0.ctx,
            &compressed_norm,
            self.config.kv_lora_rank,
            input.seq_len,
        )?;
        activate(&self.rank0.ctx)?;
        let kv_b = ops::gemm(&self.rank0.ctx, &attn.kv_b_proj, &compressed)?;
        let kv_b_host = hidden_to_f32(&self.rank0.ctx, &kv_b)?;

        let mut queries =
            vec![
                0.0f32;
                input.seq_len * self.config.num_attention_heads * self.config.query_head_dim()
            ];
        append_kv_and_build_queries(
            &self.config,
            &q_host,
            &kv_a_host,
            &kv_b_host,
            start_pos,
            input.seq_len,
            &mut queries,
            cache,
        );

        let out_host =
            compute_attention_host(&self.config, &queries, cache, start_pos, input.seq_len);
        hidden_from_f32_host(
            &self.rank0.ctx,
            &out_host,
            self.config.o_proj_cols(),
            input.seq_len,
        )
    }

    fn attention_forward_decode_batch(
        &self,
        input: &HiddenStates,
        attn: &AttentionWeights,
        position: usize,
        caches: &mut [&mut LayerCache],
    ) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;
        ensure!(
            input.seq_len == caches.len(),
            "batched attention input seq_len {} must match cache rows {}",
            input.seq_len,
            caches.len()
        );
        for (row, cache) in caches.iter().enumerate() {
            ensure!(
                cache.len(&self.config) == position,
                "batched attention cache row {row} position mismatch: cache_len={}, position={position}",
                cache.len(&self.config)
            );
        }

        let q = ops::gemm_per_token(&self.rank0.ctx, &attn.q_proj, input)?;
        let kv_a = ops::gemm_per_token(&self.rank0.ctx, &attn.kv_a_proj, input)?;
        let q_host = hidden_to_f32(&self.rank0.ctx, &q)?;
        let kv_a_host = hidden_to_f32(&self.rank0.ctx, &kv_a)?;

        let compressed_norm = normalize_compressed_kv(
            &self.config,
            &kv_a_host,
            &attn.kv_a_norm_host,
            input.seq_len,
        );
        let compressed = hidden_from_bf16_host(
            &self.rank0.ctx,
            &compressed_norm,
            self.config.kv_lora_rank,
            input.seq_len,
        )?;
        activate(&self.rank0.ctx)?;
        let kv_b = ops::gemm_per_token(&self.rank0.ctx, &attn.kv_b_proj, &compressed)?;
        let kv_b_host = hidden_to_f32(&self.rank0.ctx, &kv_b)?;

        let mut queries =
            vec![
                0.0f32;
                input.seq_len * self.config.num_attention_heads * self.config.query_head_dim()
            ];
        append_kv_and_build_queries_decode_batch(
            &self.config,
            &q_host,
            &kv_a_host,
            &kv_b_host,
            position,
            &mut queries,
            caches,
        );

        let cache_views: Vec<_> = caches.iter().map(|cache| &**cache as &LayerCache).collect();
        let out_host =
            compute_attention_host_decode_batch(&self.config, &queries, &cache_views, position);
        hidden_from_f32_host(
            &self.rank0.ctx,
            &out_host,
            self.config.o_proj_cols(),
            input.seq_len,
        )
    }

    fn moe_forward(
        &self,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
        shared_per_token_gemm: bool,
    ) -> Result<(HiddenStates, usize, usize)> {
        match &self.backend {
            EpBackendRuntime::HostStaged => self.moe_forward_host_staged(
                layer_idx,
                input,
                moe,
                attribution,
                phase,
                token_index,
                shared_per_token_gemm,
            ),
            EpBackendRuntime::Nccl(nccl) => self.moe_forward_nccl(
                nccl,
                layer_idx,
                input,
                moe,
                attribution,
                phase,
                token_index,
                shared_per_token_gemm,
            ),
        }
    }

    fn moe_forward_host_staged(
        &self,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
        shared_per_token_gemm: bool,
    ) -> Result<(HiddenStates, usize, usize)> {
        activate(&self.rank0.ctx)?;
        let (input_host, routes) = attribution.record_result(
            phase,
            "ep_route_host",
            || format!("layer.{layer_idx}.host_staged.route"),
            Some(layer_idx),
            token_index,
            || {
                let input_host = hidden_to_bf16(&self.rank0.ctx, input)?;
                let route_logits_host = gate_logits_host(&self.config, &input_host, &moe.gate_host);
                let routes = topk_softmax_routes(&self.config, &route_logits_host, input.seq_len);
                Ok((input_host, routes))
            },
        )?;

        let shared = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_expert_enqueue",
            || format!("layer.{layer_idx}.shared_expert"),
            Some(layer_idx),
            token_index,
            || {
                if shared_per_token_gemm {
                    dense_mlp_forward_per_token(&self.rank0.ctx, &moe.shared, input)
                } else {
                    dense_mlp_forward(&self.rank0.ctx, &moe.shared, input)
                }
            },
        )?;
        let mut rank0_contrib = vec![0.0f32; input.seq_len * self.config.hidden_size];
        let mut rank1_contrib = vec![0.0f32; rank0_contrib.len()];
        let mut local_routes = 0usize;
        let mut remote_routes = 0usize;

        for (token, token_routes) in routes.iter().enumerate() {
            let token_input =
                &input_host[token * self.config.hidden_size..(token + 1) * self.config.hidden_size];
            for &(global_expert, weight) in token_routes {
                let owner_rank = self.rank0.layout.owner_rank(global_expert)?;
                let section = if owner_rank == 0 {
                    "host_staged_local_expert"
                } else {
                    "host_staged_remote_dispatch"
                };
                let expert_ctx = if owner_rank == 0 {
                    &self.rank0.ctx
                } else {
                    &self.rank1.ctx
                };
                let dst = if owner_rank == 0 {
                    &mut rank0_contrib
                } else {
                    &mut rank1_contrib
                };
                let (out, is_remote) = attribution.record_gpu_result(
                    expert_ctx,
                    phase,
                    section,
                    || format!("layer.{layer_idx}.{section}"),
                    Some(layer_idx),
                    token_index,
                    || self.expert_forward_host(layer_idx, global_expert, token_input),
                )?;
                if is_remote {
                    remote_routes += 1;
                } else {
                    local_routes += 1;
                }
                let offset = token * self.config.hidden_size;
                attribution.record_result(
                    phase,
                    "host_staged_combine_accumulate",
                    || format!("layer.{layer_idx}.host_staged.combine_accumulate"),
                    Some(layer_idx),
                    token_index,
                    || {
                        for (dst, value) in dst[offset..offset + self.config.hidden_size]
                            .iter_mut()
                            .zip(out)
                        {
                            *dst += weight * value;
                        }
                        Ok(())
                    },
                )?;
            }
        }
        let routed_accum: Vec<_> = rank0_contrib
            .into_iter()
            .zip(rank1_contrib)
            .map(|(rank0, rank1)| rank0 + rank1)
            .collect();

        let routed = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "host_staged_combine_to_device",
            || format!("layer.{layer_idx}.host_staged.combine_to_device"),
            Some(layer_idx),
            token_index,
            || {
                hidden_from_f32_host(
                    &self.rank0.ctx,
                    &routed_accum,
                    self.config.hidden_size,
                    input.seq_len,
                )
            },
        )?;
        activate(&self.rank0.ctx)?;
        let hidden = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_plus_routed_enqueue",
            || format!("layer.{layer_idx}.shared_plus_routed"),
            Some(layer_idx),
            token_index,
            || ops::add_batch(&self.rank0.ctx, &routed, &shared),
        )?;
        Ok((hidden, local_routes, remote_routes))
    }

    fn moe_forward_nccl(
        &self,
        nccl: &NaiveNcclEp2Backend,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
        shared_per_token_gemm: bool,
    ) -> Result<(HiddenStates, usize, usize)> {
        activate(&self.rank0.ctx)?;
        let routes = attribution.record_result(
            phase,
            "ep_route_host",
            || format!("layer.{layer_idx}.nccl.route"),
            Some(layer_idx),
            token_index,
            || {
                let input_host = hidden_to_bf16(&self.rank0.ctx, input)?;
                let route_logits_host = gate_logits_host(&self.config, &input_host, &moe.gate_host);
                Ok(topk_softmax_routes(
                    &self.config,
                    &route_logits_host,
                    input.seq_len,
                ))
            },
        )?;

        let shared = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_expert_enqueue",
            || format!("layer.{layer_idx}.shared_expert"),
            Some(layer_idx),
            token_index,
            || {
                if shared_per_token_gemm {
                    dense_mlp_forward_per_token(&self.rank0.ctx, &moe.shared, input)
                } else {
                    dense_mlp_forward(&self.rank0.ctx, &moe.shared, input)
                }
            },
        )?;
        let mut rank0_contrib = vec![0.0f32; input.seq_len * self.config.hidden_size];
        let mut rank1_contrib = vec![0.0f32; rank0_contrib.len()];
        // NCCL covers only the dense hidden exchange and final contribution
        // sum in this first gate. Route iteration and expert-output
        // accumulation stay host-side so host-staged remains a simple oracle.
        let rank1_input = attribution.record_gpu_pair_result(
            &self.rank0.ctx,
            &self.rank1.ctx,
            phase,
            "nccl_dense_exchange",
            || format!("layer.{layer_idx}.nccl.dense_exchange"),
            Some(layer_idx),
            token_index,
            || nccl.dense_all_reduce_rank0_hidden_to_rank1(&self.rank0.ctx, &self.rank1.ctx, input),
        )?;
        let mut local_routes = 0usize;
        let mut remote_routes = 0usize;

        for (token, token_routes) in routes.iter().enumerate() {
            for &(global_expert, weight) in token_routes {
                let owner_rank = self.rank0.layout.owner_rank(global_expert)?;
                let (out, dst) = match owner_rank {
                    0 => {
                        local_routes += 1;
                        let expert = self.rank0.routed_expert(layer_idx, global_expert)?;
                        (
                            attribution.record_gpu_result(
                                &self.rank0.ctx,
                                phase,
                                "nccl_local_expert",
                                || format!("layer.{layer_idx}.nccl.local_expert"),
                                Some(layer_idx),
                                token_index,
                                || expert_forward_device(&self.rank0.ctx, expert, input, token),
                            )?,
                            &mut rank0_contrib,
                        )
                    }
                    1 => {
                        remote_routes += 1;
                        let expert = self.rank1.routed_expert(layer_idx, global_expert)?;
                        (
                            attribution.record_gpu_result(
                                &self.rank1.ctx,
                                phase,
                                "nccl_remote_expert",
                                || format!("layer.{layer_idx}.nccl.remote_expert"),
                                Some(layer_idx),
                                token_index,
                                || {
                                    expert_forward_device(
                                        &self.rank1.ctx,
                                        expert,
                                        &rank1_input,
                                        token,
                                    )
                                },
                            )?,
                            &mut rank1_contrib,
                        )
                    }
                    other => {
                        bail!("routed expert {global_expert} maps to unsupported EP rank {other}")
                    }
                };
                let offset = token * self.config.hidden_size;
                attribution.record_result(
                    phase,
                    "nccl_contribution_accumulate",
                    || format!("layer.{layer_idx}.nccl.contribution_accumulate"),
                    Some(layer_idx),
                    token_index,
                    || {
                        for (dst, value) in dst[offset..offset + self.config.hidden_size]
                            .iter_mut()
                            .zip(out)
                        {
                            *dst += weight * value;
                        }
                        Ok(())
                    },
                )?;
            }
        }

        let combined = attribution.record_gpu_pair_result(
            &self.rank0.ctx,
            &self.rank1.ctx,
            phase,
            "nccl_combine",
            || format!("layer.{layer_idx}.nccl.combine"),
            Some(layer_idx),
            token_index,
            || {
                nccl.combine_f32_contributions_to_rank0(
                    &self.rank0.ctx,
                    &self.rank1.ctx,
                    &rank0_contrib,
                    &rank1_contrib,
                )
            },
        )?;
        let routed = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "nccl_combine_to_device",
            || format!("layer.{layer_idx}.nccl.combine_to_device"),
            Some(layer_idx),
            token_index,
            || {
                hidden_from_f32_host(
                    &self.rank0.ctx,
                    &combined,
                    self.config.hidden_size,
                    input.seq_len,
                )
            },
        )?;
        activate(&self.rank0.ctx)?;
        let hidden = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_plus_routed_enqueue",
            || format!("layer.{layer_idx}.shared_plus_routed"),
            Some(layer_idx),
            token_index,
            || ops::add_batch(&self.rank0.ctx, &routed, &shared),
        )?;
        Ok((hidden, local_routes, remote_routes))
    }

    fn expert_forward_host(
        &self,
        layer_idx: usize,
        global_expert: usize,
        token_input: &[bf16],
    ) -> Result<(Vec<f32>, bool)> {
        let owner_rank = self.rank0.layout.owner_rank(global_expert)?;
        let (ctx, expert) = match owner_rank {
            0 => (
                &self.rank0.ctx,
                self.rank0.routed_expert(layer_idx, global_expert)?,
            ),
            1 => (
                &self.rank1.ctx,
                self.rank1.routed_expert(layer_idx, global_expert)?,
            ),
            other => bail!("routed expert {global_expert} maps to unsupported EP rank {other}"),
        };

        let input = hidden_from_bf16_host(ctx, token_input, self.config.hidden_size, 1)?;
        let out = dense_mlp_forward(ctx, &expert.dense, &input)?;
        Ok((hidden_to_f32(ctx, &out)?, owner_rank != 0))
    }

    fn sample_last_token(&self, hidden: &HiddenStates) -> Result<u32> {
        self.sample_token_at(hidden, hidden.seq_len - 1)
    }

    fn sample_tokens(&self, hidden: &HiddenStates) -> Result<Vec<u32>> {
        activate(&self.rank0.ctx)?;
        ensure!(hidden.seq_len != 0, "cannot sample an empty hidden batch");

        let normed = self.rms_norm_hidden(hidden, &self.rank0.norm_host)?;
        activate(&self.rank0.ctx)?;
        let logits = ops::gemm_per_token(&self.rank0.ctx, &self.rank0.lm_head, &normed)?;
        let mut values = self.rank0.ctx.stream.alloc_zeros(hidden.seq_len)?;
        let mut out = self.rank0.ctx.stream.alloc_zeros(hidden.seq_len)?;
        ops::argmax_batch_bf16_into(&self.rank0.ctx, &logits, &mut values, &mut out)?;

        let out_host = self.rank0.ctx.stream.clone_dtoh(&out)?;
        self.rank0.ctx.sync()?;

        out_host
            .into_iter()
            .map(|token| {
                ensure!(token >= 0, "argmax returned negative token id {token}");
                Ok(token as u32)
            })
            .collect()
    }

    fn sample_token_at(&self, hidden: &HiddenStates, row: usize) -> Result<u32> {
        activate(&self.rank0.ctx)?;
        ensure!(
            row < hidden.seq_len,
            "sample row {row} out of range for seq_len {}",
            hidden.seq_len
        );
        let state = ops::extract_vec(&self.rank0.ctx, hidden, row)?;
        let normed = self.rms_norm_vec(&state, &self.rank0.norm_host)?;
        let logits = ops::linear(&self.rank0.ctx, &normed, &self.rank0.lm_head)?;
        ops::argmax(&self.rank0.ctx, &logits)
    }

    fn rms_norm_hidden(&self, hidden: &HiddenStates, weight: &[f32]) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;
        let input_host = hidden_to_f32(&self.rank0.ctx, hidden)?;
        let out = rms_norm_hidden_host(&self.config, &input_host, weight, hidden.seq_len);
        hidden_from_bf16_host(
            &self.rank0.ctx,
            &out,
            self.config.hidden_size,
            hidden.seq_len,
        )
    }

    fn rms_norm_vec(&self, input: &DeviceVec, weight: &[f32]) -> Result<DeviceVec> {
        activate(&self.rank0.ctx)?;
        let input_host = input.to_host(&self.rank0.ctx)?;
        let mut out = vec![bf16::ZERO; input.len];
        rms_norm_host(&input_host, weight, self.config.rms_norm_eps, &mut out);
        DeviceVec::from_host(&self.rank0.ctx, &out)
    }
}

fn expert_forward_device(
    ctx: &pegainfer_core::tensor::DeviceContext,
    expert: &ExpertMlp,
    input: &HiddenStates,
    token_idx: usize,
) -> Result<Vec<f32>> {
    activate(ctx)?;
    let token = ops::extract_vec(ctx, input, token_idx)?;
    let token_hidden = HiddenStates {
        hidden_dim: token.len,
        seq_len: 1,
        data: token.data,
    };
    let out = dense_mlp_forward(ctx, &expert.dense, &token_hidden)?;
    hidden_to_f32(ctx, &out)
}

fn validate_backend_and_devices(device_ordinals: &[usize]) -> Result<EpBackendKind> {
    ensure!(
        device_ordinals.len() == 2,
        "DeepSeek-V2-Lite first EP gate supports exactly 2 CUDA devices for ep_size=2, got {}",
        device_ordinals.len()
    );
    ensure!(
        device_ordinals[0] != device_ordinals[1],
        "DeepSeek-V2-Lite EP=2 requires two distinct CUDA device ordinals, got {:?}",
        device_ordinals
    );
    EpBackendKind::from_env()
}

fn token_sha256(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

fn ensure_same_prompt_batch_rows_match(rows: &[Vec<u32>]) -> Result<()> {
    let Some(first) = rows.first() else {
        return Ok(());
    };
    for (row_idx, row) in rows.iter().enumerate().skip(1) {
        if row != first {
            let first_diff = first
                .iter()
                .zip(row)
                .position(|(lhs, rhs)| lhs != rhs)
                .unwrap_or_else(|| first.len().min(row.len()));
            bail!(
                "same-prompt batched decode row {row_idx} differs from row 0 at generated token index {first_diff}: row0_sha256={}, row_sha256={}",
                token_sha256(first),
                token_sha256(row)
            );
        }
    }
    Ok(())
}

fn duration_micros(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

fn append_generated_token(
    generated: &mut Vec<u32>,
    token: u32,
    eos_token_id: u32,
    ignore_eos: bool,
) -> Option<FinishReason> {
    if !ignore_eos && token == eos_token_id {
        return Some(FinishReason::Stop);
    }
    generated.push(token);
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_token_is_not_appended_when_eos_is_enabled() {
        let mut generated = vec![10, 11];

        let finish_reason = append_generated_token(&mut generated, 100_001, 100_001, false);

        assert_eq!(finish_reason, Some(FinishReason::Stop));
        assert_eq!(generated, vec![10, 11]);
    }

    #[test]
    fn stop_token_is_appended_when_eos_is_ignored() {
        let mut generated = vec![10, 11];

        let finish_reason = append_generated_token(&mut generated, 100_001, 100_001, true);

        assert_eq!(finish_reason, None);
        assert_eq!(generated, vec![10, 11, 100_001]);
    }

    #[test]
    fn same_prompt_batch_rows_must_match() {
        ensure_same_prompt_batch_rows_match(&[
            vec![11, 304, 608],
            vec![11, 304, 608],
            vec![11, 304, 608],
        ])
        .unwrap();

        let err = ensure_same_prompt_batch_rows_match(&[vec![11, 304, 608], vec![11, 463, 608]])
            .unwrap_err();
        assert!(
            err.to_string().contains("generated token index 1"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn duplicate_device_ordinals_are_rejected() {
        let err = validate_backend_and_devices(&[0, 0]).unwrap_err();

        assert!(
            err.to_string()
                .contains("two distinct CUDA device ordinals"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn ep_backend_defaults_to_host_staged() {
        assert_eq!(parse_backend(None).unwrap(), EpBackendKind::HostStaged);
    }

    #[test]
    fn ep_backend_accepts_nccl() {
        assert_eq!(parse_backend(Some("nccl")).unwrap(), EpBackendKind::Nccl);
    }

    #[test]
    fn ep_backend_rejects_unknown_backend() {
        let err = parse_backend(Some("pplx")).unwrap_err();

        assert!(
            err.to_string()
                .contains("supported backends: host-staged, nccl"),
            "unexpected error: {err:#}"
        );
    }
}
