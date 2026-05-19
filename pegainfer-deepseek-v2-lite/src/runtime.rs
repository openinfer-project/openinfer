use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use pegainfer_core::{ops, tensor::HiddenStates};
use pegainfer_engine::engine::{EngineLoadOptions, FinishReason};
use sha2::{Digest, Sha256};

use crate::{
    Config,
    device::activate,
    ep::ExpertParallelConfig,
    host_ops::{
        DecodeCache, LayerCache, append_kv_and_build_queries, compute_attention_host,
        hidden_from_bf16_host, hidden_from_f32_host, hidden_to_bf16, hidden_to_f32,
        normalize_compressed_kv, topk_softmax_routes,
    },
    model::{
        AttentionWeights, DriverRankModel, ExpertMlp, ExpertRankModel, MlpWeights, MoeMlp,
        dense_mlp_forward,
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
    pub host_dispatch_local_routes: usize,
    pub host_dispatch_remote_routes: usize,
    pub nccl_dispatch_local_routes: usize,
    pub nccl_dispatch_remote_routes: usize,
    pub nccl_combine_routes: usize,
    pub output_token_sha256: String,
}

#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    pub stats: GenerationStats,
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
        ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
        ensure!(max_new_tokens > 0, "max_new_tokens must be positive");
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
        let mut next = self.prefill_next_token(prompt_tokens, &mut cache, &mut stats)?;
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
            next = self.decode_next_token(next, position, &mut cache, &mut stats)?;
        }

        stats.generated_tokens = generated.len();
        stats.output_token_sha256 = token_sha256(&generated);
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
    ) -> Result<u32> {
        let mut hidden = self.embed_tokens(prompt_tokens)?;
        hidden = self.forward_layers(hidden, 0, cache, stats)?;
        self.sample_last_token(&hidden)
    }

    fn decode_next_token(
        &mut self,
        token: u32,
        position: usize,
        cache: &mut DecodeCache,
        stats: &mut GenerationStats,
    ) -> Result<u32> {
        let mut hidden = self.embed_tokens(&[token])?;
        hidden = self.forward_layers(hidden, position, cache, stats)?;
        self.sample_last_token(&hidden)
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
                )
                .with_context(|| format!("DeepSeek-V2-Lite layer {layer_idx}"))?;
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
    ) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;

        let layer = &self.rank0.layers[layer_idx];
        let mut normed =
            HiddenStates::zeros(&self.rank0.ctx, self.config.hidden_size, hidden.seq_len)?;
        ops::rms_norm_batch_into(
            &self.rank0.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut normed,
        );

        let attn = self.attention_forward(&normed, &layer.attention, start_pos, cache)?;
        activate(&self.rank0.ctx)?;
        let attn_projected = ops::gemm(&self.rank0.ctx, &layer.attention.o_proj, &attn)?;
        let after_attn = ops::add_batch(&self.rank0.ctx, hidden, &attn_projected)?;

        let mut ffn_norm =
            HiddenStates::zeros(&self.rank0.ctx, self.config.hidden_size, after_attn.seq_len)?;
        ops::rms_norm_batch_into(
            &self.rank0.ctx,
            &after_attn,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut ffn_norm,
        );

        let (ffn_out, local_routes, remote_routes) = match &layer.mlp {
            MlpWeights::Dense(dense) => {
                (dense_mlp_forward(&self.rank0.ctx, dense, &ffn_norm)?, 0, 0)
            }
            MlpWeights::Moe(moe) => self.moe_forward(layer_idx, &ffn_norm, moe)?,
        };
        stats.record_routes(self.backend.kind(), local_routes, remote_routes);
        ops::add_batch(&self.rank0.ctx, &after_attn, &ffn_out)
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

    fn moe_forward(
        &self,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
    ) -> Result<(HiddenStates, usize, usize)> {
        match &self.backend {
            EpBackendRuntime::HostStaged => self.moe_forward_host_staged(layer_idx, input, moe),
            EpBackendRuntime::Nccl(nccl) => self.moe_forward_nccl(nccl, layer_idx, input, moe),
        }
    }

    fn moe_forward_host_staged(
        &self,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
    ) -> Result<(HiddenStates, usize, usize)> {
        activate(&self.rank0.ctx)?;
        let route_logits = ops::gemm(&self.rank0.ctx, &moe.gate, input)?;
        let route_logits_host = hidden_to_f32(&self.rank0.ctx, &route_logits)?;
        let routes = topk_softmax_routes(&self.config, &route_logits_host, input.seq_len);

        let shared = dense_mlp_forward(&self.rank0.ctx, &moe.shared, input)?;
        let mut accum = hidden_to_f32(&self.rank0.ctx, &shared)?;
        let input_host = hidden_to_bf16(&self.rank0.ctx, input)?;
        let mut local_routes = 0usize;
        let mut remote_routes = 0usize;

        for (token, token_routes) in routes.iter().enumerate() {
            let token_input =
                &input_host[token * self.config.hidden_size..(token + 1) * self.config.hidden_size];
            for &(global_expert, weight) in token_routes {
                let (out, is_remote) =
                    self.expert_forward_host(layer_idx, global_expert, token_input)?;
                if is_remote {
                    remote_routes += 1;
                } else {
                    local_routes += 1;
                }
                let offset = token * self.config.hidden_size;
                for (dst, value) in accum[offset..offset + self.config.hidden_size]
                    .iter_mut()
                    .zip(out)
                {
                    *dst += weight * value;
                }
            }
        }

        let hidden = hidden_from_f32_host(
            &self.rank0.ctx,
            &accum,
            self.config.hidden_size,
            input.seq_len,
        )?;
        Ok((hidden, local_routes, remote_routes))
    }

    fn moe_forward_nccl(
        &self,
        nccl: &NaiveNcclEp2Backend,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
    ) -> Result<(HiddenStates, usize, usize)> {
        activate(&self.rank0.ctx)?;
        let route_logits = ops::gemm(&self.rank0.ctx, &moe.gate, input)?;
        let route_logits_host = hidden_to_f32(&self.rank0.ctx, &route_logits)?;
        let routes = topk_softmax_routes(&self.config, &route_logits_host, input.seq_len);

        let shared = dense_mlp_forward(&self.rank0.ctx, &moe.shared, input)?;
        let mut rank0_contrib = hidden_to_f32(&self.rank0.ctx, &shared)?;
        let mut rank1_contrib = vec![0.0f32; rank0_contrib.len()];
        let rank1_input =
            nccl.dispatch_rank0_hidden_to_rank1(&self.rank0.ctx, &self.rank1.ctx, input)?;
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
                            expert_forward_device(&self.rank0.ctx, expert, input, token)?,
                            &mut rank0_contrib,
                        )
                    }
                    1 => {
                        remote_routes += 1;
                        let expert = self.rank1.routed_expert(layer_idx, global_expert)?;
                        (
                            expert_forward_device(&self.rank1.ctx, expert, &rank1_input, token)?,
                            &mut rank1_contrib,
                        )
                    }
                    other => {
                        bail!("routed expert {global_expert} maps to unsupported EP rank {other}")
                    }
                };
                let offset = token * self.config.hidden_size;
                for (dst, value) in dst[offset..offset + self.config.hidden_size]
                    .iter_mut()
                    .zip(out)
                {
                    *dst += weight * value;
                }
            }
        }

        let combined = nccl.combine_f32_contributions_to_rank0(
            &self.rank0.ctx,
            &self.rank1.ctx,
            &rank0_contrib,
            &rank1_contrib,
        )?;
        let hidden = hidden_from_f32_host(
            &self.rank0.ctx,
            &combined,
            self.config.hidden_size,
            input.seq_len,
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
        activate(&self.rank0.ctx)?;
        let last = ops::extract_vec(&self.rank0.ctx, hidden, hidden.seq_len - 1)?;
        let normed = ops::rms_norm(
            &self.rank0.ctx,
            &last,
            &self.rank0.norm,
            self.config.rms_norm_eps,
        )?;
        let logits = ops::linear(&self.rank0.ctx, &normed, &self.rank0.lm_head)?;
        ops::argmax(&self.rank0.ctx, &logits)
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
