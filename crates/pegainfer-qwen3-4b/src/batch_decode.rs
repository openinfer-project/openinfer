//! Batched decode: process N requests' tokens in one forward pass.

use anyhow::Result;

use super::batch_decode_buffers::{BATCH_BUCKETS, BatchDecodeBuffers, bucket_for};
use super::weights::{Qwen3Model, TransformerBlock};
use pegainfer_core::kv_pool::{KvLayout, KvState};
use pegainfer_core::ops;

impl Qwen3Model {
    /// Batch decode step: N requests, 1 new token each, one forward pass.
    ///
    /// When `enable_cuda_graph` is set, pads to the nearest bucket size and
    /// uses per-bucket CUDA Graph capture/replay.
    pub(crate) fn batch_decode(
        &self,
        token_ids: &[u32],
        kv_states: &mut [&mut KvState],
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let bs = token_ids.len();
        assert_eq!(bs, kv_states.len());
        assert!(bs > 0);

        // Grow pages and advance seq_len for each request
        let mut positions = Vec::with_capacity(bs);
        for kv in kv_states.iter_mut() {
            let pos = kv.seq_len();
            kv.ensure_capacity(pos + 1)?;
            kv.advance(1);
            positions.push(pos as i32);
        }

        // Pad to bucket size for CUDA Graph stability
        let padded_bs = if self.enable_cuda_graph {
            bucket_for(bs)
        } else {
            bs
        };

        // Set batch size on all buffers (padded — kernels run at bucket width)
        bufs.set_batch_size(padded_bs);

        // Sync metadata to GPU (pad token_ids/positions with 0 for padding slots)
        let mut token_ids_padded = token_ids.to_vec();
        token_ids_padded.resize(padded_bs, 0);
        positions.resize(padded_bs, 0);

        self.ctx
            .stream
            .memcpy_htod(&token_ids_padded, &mut bufs.token_ids_d)?;
        self.ctx
            .stream
            .memcpy_htod(&positions, &mut bufs.positions_d)?;

        let kv_refs: Vec<&KvState> = kv_states.iter().map(|s| &**s).collect();
        bufs.sync_paged_meta(&self.ctx, &kv_refs, padded_bs)?;

        // Forward pass — with or without CUDA Graph
        let kv_buffer = kv_states[0].buffer();
        let layout = *kv_states[0].layout();
        if self.enable_cuda_graph {
            let bucket_idx = BATCH_BUCKETS.iter().position(|&b| b == padded_bs).unwrap();
            // Take graphs out of bufs to avoid split-borrow conflict with closure
            let mut graphs = std::mem::take(&mut bufs.graphs);
            let result = graphs[bucket_idx].run_or_capture(&self.ctx, || {
                self.batch_decode_kernels(kv_buffer, &layout, padded_bs, bufs)
            });
            bufs.graphs = graphs;
            result?;
        } else {
            self.batch_decode_kernels(kv_buffer, &layout, padded_bs, bufs)?;
        }

        Ok(())
    }

    fn batch_decode_kernels(
        &self,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        bs: usize,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let num_layers = self.layers.len();

        // Embedding: N token_ids → hidden [hidden_dim, bs]
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_d,
            &mut bufs.hidden,
        )?;

        // First layer norm
        ops::rms_norm_batch_into(
            &self.ctx,
            &bufs.hidden,
            &self.layers[0].input_layernorm,
            eps,
            &mut bufs.normed,
        );

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.batch_decode_layer(layer_idx, layer, kv_buffer, layout, bs, bufs)?;

            let next_weight = if layer_idx + 1 < num_layers {
                &self.layers[layer_idx + 1].input_layernorm
            } else {
                &self.norm
            };
            ops::fused_add_rms_norm_batch_into(
                &self.ctx,
                &mut bufs.hidden,
                &bufs.mlp_out,
                next_weight,
                eps,
                &mut bufs.normed,
            );
        }

        // Output projection: logits [vocab_size, bs]
        ops::gemm_into(
            &self.ctx,
            self.output_projection(),
            &bufs.normed,
            &mut bufs.logits,
        );

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_decode_layer(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        bs: usize,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        // Match prefill numerics: compute Q/K/V via row-sliced GEMMs instead of
        // fused qkv GEMM + deinterleave. The fused path is mathematically
        // equivalent but diverges enough under shard-local TP to flip greedy
        // decode in parity tests.
        let q_dim = layer.attention.q_dim;
        let kv_dim = layer.attention.kv_dim;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            0,
            q_dim,
            &bufs.normed,
            &mut bufs.q,
        );
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.k,
        );
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.v,
        );

        // QK norm + RoPE (batched, per-request positions)
        ops::qk_norm_rope_batch_decode_into(
            &self.ctx,
            &mut bufs.q,
            &mut bufs.k,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions_d,
            self.local_num_attention_heads(),
            self.local_num_key_value_heads(),
            self.config.head_dim,
            eps,
        );

        // KV append + paged attention decode (FlashInfer, batched)
        ops::paged_attention_batch_decode_into(
            &self.ctx,
            &bufs.q,
            &bufs.k,
            &bufs.v,
            kv_buffer,
            layout,
            layer_idx,
            &bufs.page_indices_d,
            &bufs.page_indptr_d,
            &bufs.last_page_len_d,
            &bufs.positions_d,
            &bufs.request_indices_d,
            &bufs.kv_tile_indices_d,
            &bufs.kv_chunk_size_d,
            &mut bufs.attn_out,
            self.local_num_attention_heads(),
            bs,
        )?;

        // O projection (GEMM)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.attn_proj,
        );
        self.all_reduce_hidden(&mut bufs.attn_proj)?;

        // Residual + LayerNorm
        ops::fused_add_rms_norm_batch_into(
            &self.ctx,
            &mut bufs.hidden,
            &bufs.attn_proj,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        );

        // MLP: fused gate+up GEMM → silu_mul_fused → down GEMM
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.gate_up_out,
        );
        ops::silu_mul_fused_batch_into(&self.ctx, &bufs.gate_up_out, &mut bufs.mlp_act);
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.mlp_act,
            &mut bufs.mlp_out,
        );
        self.all_reduce_hidden(&mut bufs.mlp_out)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_decode_buffers::BatchDecodeBuffers;
    use crate::weights::ModelRuntimeConfig;
    use pegainfer_core::model::ModelForward;
    use pegainfer_core::sampler::SamplingParams;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/Qwen3-4B");

    fn get_model_path() -> String {
        std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
    }

    fn sample_batch_tokens(
        model: &Qwen3Model,
        bufs: &BatchDecodeBuffers,
        params: &[&SamplingParams],
        rng: &mut StdRng,
    ) -> Vec<u32> {
        let mut scratch_probs = model
            .ctx
            .stream
            .alloc_zeros(model.config.vocab_size)
            .unwrap();
        let mut scratch_top1 = model.ctx.stream.alloc_zeros(1).unwrap();
        let mut scratch_row_states = model
            .ctx
            .stream
            .alloc_zeros(pegainfer_core::ops::flashinfer_topk_row_states_bytes())
            .unwrap();
        let mut scratch_valid = model.ctx.stream.alloc_zeros(1).unwrap();
        let mut scratch_out = model.ctx.stream.alloc_zeros(1).unwrap();
        (0..params.len())
            .map(|i| {
                let logits_i = ops::extract_vec(&model.ctx, &bufs.logits, i).unwrap();
                let random_val: f32 = rand::RngExt::random(rng);
                ops::gpu_sample_into(
                    &model.ctx,
                    &logits_i,
                    &mut scratch_probs,
                    &mut scratch_top1,
                    &mut scratch_row_states,
                    &mut scratch_valid,
                    &mut scratch_out,
                    params[i],
                    random_val,
                )
                .unwrap()
            })
            .collect()
    }

    /// Run single-request decode via batch_decode(bs=1) for a prompt, return generated token IDs.
    ///
    /// Uses batch_decode with bs=1 (not ModelForward::forward) so the decode path
    /// is identical to the multi-request batch — same cuBLAS handle, same FlashInfer
    /// BatchDecode kernel. This avoids FP accumulation differences between BatchPrefill
    /// (used by forward()) and BatchDecode.
    fn sequential_decode(
        model: &Qwen3Model,
        prompt_tokens: &[u32],
        num_decode_steps: usize,
        seed: u64,
    ) -> Vec<u32> {
        let params = SamplingParams::default(); // greedy
        let mut rng = StdRng::seed_from_u64(seed);

        // Prefill via ModelForward (produces first token)
        let mut state = model.create_state().unwrap();
        model.forward(prompt_tokens, &mut state).unwrap();
        let first_token = model.select_token(&mut state, &params, &mut rng).unwrap();

        // Decode remaining tokens via batch_decode(bs=1)
        let mut kv_state = std::mem::replace(&mut state.kv_state, model.kv_pool.alloc());
        let mut bufs = BatchDecodeBuffers::new(
            &model.ctx,
            model.config.hidden_size,
            model.local_q_dim(),
            model.local_kv_dim(),
            model.local_intermediate_size(),
            model.config.vocab_size,
            1,
            model.kv_pool.capacity_pages(),
            model.kv_pool.padding_page_id(),
        )
        .unwrap();

        let mut tokens = vec![first_token];
        for _ in 1..num_decode_steps {
            let token_ids = [*tokens.last().unwrap()];
            let mut kv_refs: Vec<&mut KvState> = vec![&mut kv_state];
            model
                .batch_decode(&token_ids, &mut kv_refs, &mut bufs)
                .unwrap();
            let params_refs: Vec<&SamplingParams> = vec![&params];
            let batch_tokens = sample_batch_tokens(model, &bufs, &params_refs, &mut rng);
            tokens.push(batch_tokens[0]);
        }
        tokens
    }

    /// Run batch decode for multiple prompts, return per-request generated tokens.
    fn batch_decode_run(
        model: &Qwen3Model,
        prompts: &[&[u32]],
        num_decode_steps: usize,
        seed: u64,
    ) -> Vec<Vec<u32>> {
        let bs = prompts.len();
        let params = SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(seed);

        // Create per-request states and prefill each independently
        let mut kv_states: Vec<KvState> = (0..bs).map(|_| model.kv_pool.alloc()).collect();
        let mut first_tokens = Vec::with_capacity(bs);

        for (i, prompt) in prompts.iter().enumerate() {
            let mut state = model.create_state().unwrap();
            model.forward(prompt, &mut state).unwrap();
            let token = model.select_token(&mut state, &params, &mut rng).unwrap();
            first_tokens.push(token);
            std::mem::swap(&mut kv_states[i], &mut state.kv_state);
        }

        let mut all_tokens: Vec<Vec<u32>> = first_tokens.iter().map(|&t| vec![t]).collect();

        // Allocate buffers at bucket size (graph padding may exceed actual bs)
        let max_bs = if model.enable_cuda_graph {
            bucket_for(bs)
        } else {
            bs
        };
        let mut bufs = BatchDecodeBuffers::new(
            &model.ctx,
            model.config.hidden_size,
            model.local_q_dim(),
            model.local_kv_dim(),
            model.local_intermediate_size(),
            model.config.vocab_size,
            max_bs,
            model.kv_pool.capacity_pages(),
            model.kv_pool.padding_page_id(),
        )
        .unwrap();

        for _ in 1..num_decode_steps {
            let token_ids: Vec<u32> = all_tokens.iter().map(|t| *t.last().unwrap()).collect();
            let mut kv_refs: Vec<&mut KvState> = kv_states.iter_mut().collect();
            model
                .batch_decode(&token_ids, &mut kv_refs, &mut bufs)
                .unwrap();
            let params_refs: Vec<&SamplingParams> = (0..bs).map(|_| &params).collect();
            let tokens = sample_batch_tokens(model, &bufs, &params_refs, &mut rng);
            for (i, &tok) in tokens.iter().enumerate() {
                all_tokens[i].push(tok);
            }
        }

        all_tokens
    }

    /// Verify batch prefill, batch decode, and batch decode with CUDA Graph all match
    /// sequential single-request results. Runs as one test to avoid parallel model loads
    /// OOM-ing on a single GPU.
    #[test]
    fn batch_matches_sequential() {
        let model_path = get_model_path();
        let prompt_a: Vec<u32> = vec![9707]; // "Hello"
        let prompt_b: Vec<u32> = vec![3838, 374, 220, 17, 10, 17]; // "What is 2+2"
        let num_steps = 10;
        let seed = 42;

        // --- Phase 1: batch prefill + batch decode (no CUDA Graph) ---
        {
            let model = Qwen3Model::from_safetensors_with_runtime(
                &model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph: false,
                    tensor_parallel: None,
                    device_ordinal: 0,
                },
            )
            .unwrap();

            // Batch prefill: multi-token prompts so both paths use GEMM prefill
            let prefill_a: Vec<u32> = vec![3838, 374, 220, 17, 10, 17];
            let prefill_b: Vec<u32> = (1..65).collect();
            let prefill_c: Vec<u32> = (1..129).collect();
            let params = SamplingParams::default();

            let mut seq_first_tokens = Vec::new();
            for prompt in [&prefill_a, &prefill_b, &prefill_c] {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut state = model.create_state().unwrap();
                model.forward(prompt.as_slice(), &mut state).unwrap();
                let token = model.select_token(&mut state, &params, &mut rng).unwrap();
                seq_first_tokens.push(token);
            }

            let prompts: Vec<&[u32]> = vec![&prefill_a, &prefill_b, &prefill_c];
            let mut kv_states: Vec<KvState> = (0..3).map(|_| model.kv_pool.alloc()).collect();
            let mut kv_refs: Vec<&mut KvState> = kv_states.iter_mut().collect();
            let (logits_vec, _) = model.batch_prefill(&prompts, &mut kv_refs, false).unwrap();

            let mut batch_first_tokens = Vec::new();
            for logits in &logits_vec {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut probs: cudarc::driver::CudaSlice<f32> = model
                    .ctx
                    .stream
                    .alloc_zeros(model.config.vocab_size)
                    .unwrap();
                let mut top1_value: cudarc::driver::CudaSlice<half::bf16> =
                    model.ctx.stream.alloc_zeros(1).unwrap();
                let mut row_states: cudarc::driver::CudaSlice<u8> = model
                    .ctx
                    .stream
                    .alloc_zeros(pegainfer_core::ops::flashinfer_topk_row_states_bytes())
                    .unwrap();
                let mut valid: cudarc::driver::CudaSlice<u8> =
                    model.ctx.stream.alloc_zeros(1).unwrap();
                let mut out: cudarc::driver::CudaSlice<i32> =
                    model.ctx.stream.alloc_zeros(1).unwrap();
                let random_val: f32 = rand::RngExt::random(&mut rng);
                let token = pegainfer_core::ops::gpu_sample_into(
                    &model.ctx,
                    logits,
                    &mut probs,
                    &mut top1_value,
                    &mut row_states,
                    &mut valid,
                    &mut out,
                    &params,
                    random_val,
                )
                .unwrap();
                batch_first_tokens.push(token);
            }

            for (i, (seq_tok, batch_tok)) in seq_first_tokens
                .iter()
                .zip(batch_first_tokens.iter())
                .enumerate()
            {
                assert_eq!(
                    seq_tok, batch_tok,
                    "Prefill first token mismatch for prompt {i}: seq={seq_tok}, batch={batch_tok}"
                );
            }

            // Batch decode (no graph)
            let seq_a = sequential_decode(&model, &prompt_a, num_steps, seed);
            let seq_b = sequential_decode(&model, &prompt_b, num_steps, seed);
            let batch = batch_decode_run(&model, &[&prompt_a, &prompt_b], num_steps, seed);

            assert_eq!(
                batch[0], seq_a,
                "Decode request A mismatch:\n  batch: {:?}\n  seq:   {:?}",
                batch[0], seq_a
            );
            assert_eq!(
                batch[1], seq_b,
                "Decode request B mismatch:\n  batch: {:?}\n  seq:   {:?}",
                batch[1], seq_b
            );
        }
        // model dropped, GPU memory freed

        // --- Phase 2: batch decode with CUDA Graph ---
        {
            let model = Qwen3Model::from_safetensors_with_runtime(
                &model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph: true,
                    tensor_parallel: None,
                    device_ordinal: 0,
                },
            )
            .unwrap();

            let seq_a = sequential_decode(&model, &prompt_a, num_steps, seed);
            let seq_b = sequential_decode(&model, &prompt_b, num_steps, seed);
            let batch = batch_decode_run(&model, &[&prompt_a, &prompt_b], num_steps, seed);

            assert_eq!(
                batch[0], seq_a,
                "Graph request A mismatch:\n  batch: {:?}\n  seq:   {:?}",
                batch[0], seq_a
            );
            assert_eq!(
                batch[1], seq_b,
                "Graph request B mismatch:\n  batch: {:?}\n  seq:   {:?}",
                batch[1], seq_b
            );
        }
    }
}
