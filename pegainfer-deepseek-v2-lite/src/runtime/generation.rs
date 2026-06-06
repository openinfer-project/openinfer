use std::{path::Path, time::Instant};

use anyhow::{Context, Result, ensure};
use pegainfer_engine::engine::{EngineLoadOptions, FinishReason};

use super::{
    DeepSeekV2LiteEp2Generator,
    backend::{EpBackendRuntime, validate_backend_and_devices},
    helpers::{
        append_generated_token, duration_micros, ensure_same_prompt_batch_rows_match, token_sha256,
    },
    types::{BatchedGenerationResult, GenerationResult, GenerationStats},
};
use crate::{
    Config,
    attribution::DecodeAttributionProfile,
    ep::ExpertParallelConfig,
    host_ops::DecodeCache,
    model::{DriverRankModel, ExpertRankModel},
    weights::{ModelManifest, RankLoadPlan},
};

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
        let mut attribution = DecodeAttributionProfile::disabled();
        self.generate_greedy_batch_same_prompt_inner(
            prompt_tokens,
            batch_size,
            max_new_tokens,
            ignore_eos,
            &mut attribution,
        )
    }

    pub fn generate_greedy_batch_same_prompt_with_attribution(
        &mut self,
        prompt_tokens: &[u32],
        batch_size: usize,
        max_new_tokens: usize,
        ignore_eos: bool,
    ) -> Result<(BatchedGenerationResult, DecodeAttributionProfile)> {
        let mut attribution = DecodeAttributionProfile::enabled();
        let result = self.generate_greedy_batch_same_prompt_inner(
            prompt_tokens,
            batch_size,
            max_new_tokens,
            ignore_eos,
            &mut attribution,
        )?;
        Ok((result, attribution))
    }

    fn generate_greedy_batch_same_prompt_inner(
        &mut self,
        prompt_tokens: &[u32],
        batch_size: usize,
        max_new_tokens: usize,
        ignore_eos: bool,
        attribution: &mut DecodeAttributionProfile,
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
            let next =
                self.prefill_next_token(prompt_tokens, &mut caches[row], &mut stats, attribution)?;
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
                attribution,
                token_index,
            )?;
            ensure!(
                next_tokens.len() == batch_size,
                "batched decode returned {} tokens for batch_size={batch_size}",
                next_tokens.len()
            );
            let decode_elapsed = decode_start.elapsed();
            per_token_decode_us.push(duration_micros(decode_elapsed));
            attribution.push_decode_token(decode_elapsed);
            for (row, token) in next_tokens.into_iter().enumerate() {
                generated[row].push(token);
            }
        }

        ensure_same_prompt_batch_rows_match(&generated)?;
        stats.generated_tokens = generated.iter().map(Vec::len).sum();
        stats.output_token_sha256 = token_sha256(&generated[0]);
        let total_generation = generation_start.elapsed();
        attribution.set_total_generation(total_generation);
        Ok(BatchedGenerationResult {
            tokens: generated,
            prefill_next_token_us,
            per_token_decode_us,
            total_generation_us: duration_micros(total_generation),
            stats,
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
}
