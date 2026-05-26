//! Batched decode: process N requests' tokens in one forward pass.

use anyhow::Result;

use super::batch_decode_buffers::{
    BATCH_BUCKETS, BatchDecodeBuffers, DecodeAttentionPath, bucket_for,
};
use super::batch_decode_dag::BatchDecodeDag;
use super::weights::{Qwen3Model, TransformerBlock};
use pegainfer_core::kv_pool::{KvLayout, KvState};
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops;
use pegainfer_kernels::tensor::{KvDim, QDim};

#[cfg(feature = "kernel-call-trace")]
macro_rules! dag_label {
    ($label:expr) => {
        $label.to_string()
    };
}

#[cfg(not(feature = "kernel-call-trace"))]
macro_rules! dag_label {
    ($label:expr) => {
        ()
    };
}

#[cfg(feature = "kernel-call-trace")]
macro_rules! trace_decode_kv_len {
    ($kv_len:expr, $body:block) => {
        ops::call_trace::with_decode_kv_len($kv_len, || $body)
    };
}

#[cfg(not(feature = "kernel-call-trace"))]
macro_rules! trace_decode_kv_len {
    ($kv_len:expr, $body:block) => {{ $body }};
}

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
        let attention_path = bufs.attention_path(padded_bs);
        #[cfg(feature = "kernel-call-trace")]
        let trace_kv_len = kv_refs.iter().map(|kv| kv.seq_len()).max().unwrap_or(0);

        // Forward pass — with or without CUDA Graph
        let kv_buffer = kv_states[0].buffer();
        let layout = *kv_states[0].layout();
        if self.enable_cuda_graph {
            let bucket_idx = BATCH_BUCKETS.iter().position(|&b| b == padded_bs).unwrap();
            let graph_idx = BatchDecodeBuffers::graph_index(bucket_idx, attention_path);
            // Take graphs out of bufs to avoid split-borrow conflict with closure
            let mut graphs = std::mem::take(&mut bufs.graphs);
            let result = graphs[graph_idx].run_or_capture(&self.ctx, || {
                trace_decode_kv_len!(trace_kv_len, {
                    self.batch_decode_kernels(kv_buffer, &layout, padded_bs, attention_path, bufs)
                })
            });
            bufs.graphs = graphs;
            result?;
        } else {
            trace_decode_kv_len!(trace_kv_len, {
                self.batch_decode_kernels(kv_buffer, &layout, padded_bs, attention_path, bufs)
            })?;
        }

        Ok(())
    }

    fn batch_decode_kernels(
        &self,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        bs: usize,
        attention_path: DecodeAttentionPath,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let num_layers = self.layers.len();
        let dag = BatchDecodeDag::new(self, kv_buffer, layout, bs, attention_path);

        // Embedding: N token_ids → hidden [hidden_dim, bs]
        dag.embedding(dag_label!("embedding"), &bufs.token_ids_d, &mut bufs.hidden)?;

        // First layer norm
        dag.rms_norm(
            dag_label!("input.rms_norm"),
            &bufs.hidden,
            &self.layers[0].input_layernorm,
            &mut bufs.normed,
        );

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            Self::batch_decode_layer(layer_idx, layer, &dag, bufs)?;

            let next_weight = if layer_idx + 1 < num_layers {
                &self.layers[layer_idx + 1].input_layernorm
            } else {
                &self.norm
            };
            dag.fused_add_rms_norm(
                if layer_idx + 1 < num_layers {
                    dag_label!(format!("L{layer_idx}.mlp.fused_add_rms_norm"))
                } else {
                    dag_label!("final.rms_norm")
                },
                &mut bufs.hidden,
                &bufs.mlp_out,
                next_weight,
                &mut bufs.normed,
            )?;
        }

        // Output projection: logits [vocab_size, bs]
        dag.lm_head(
            dag_label!("lm_head"),
            self.output_projection(),
            &bufs.normed,
            &mut bufs.logits,
        );

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_decode_layer(
        layer_idx: usize,
        layer: &TransformerBlock,
        dag: &BatchDecodeDag<'_>,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        // Match prefill numerics: compute Q/K/V via row-sliced GEMMs instead of
        // fused qkv GEMM + deinterleave. The fused path is mathematically
        // equivalent but diverges enough under shard-local TP to flip greedy
        // decode in parity tests.
        let q_dim = layer.attention.q_dim;
        let kv_dim = layer.attention.kv_dim;
        dag.gemm_rows::<QDim>(
            dag_label!(format!("L{layer_idx}.attn.q_proj")),
            &layer.attention.qkv_proj,
            0,
            q_dim,
            &bufs.normed,
            &mut bufs.q,
        );
        dag.gemm_rows::<KvDim>(
            dag_label!(format!("L{layer_idx}.attn.k_proj")),
            &layer.attention.qkv_proj,
            q_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.k,
        );
        dag.gemm_rows::<KvDim>(
            dag_label!(format!("L{layer_idx}.attn.v_proj")),
            &layer.attention.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.v,
        );

        // QK norm + RoPE (batched, per-request positions)
        dag.qk_norm_rope(
            dag_label!(format!("L{layer_idx}.attn.qk_norm_rope")),
            &mut bufs.q,
            &mut bufs.k,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &bufs.positions_d,
        );

        // KV append + paged attention decode (FlashInfer, batched)
        dag.paged_decode_attention(
            dag_label!(format!("L{layer_idx}.attn.paged_decode")),
            layer_idx,
            bufs,
        )?;

        // O projection (GEMM)
        dag.o_proj(
            dag_label!(format!("L{layer_idx}.attn.o_proj")),
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.attn_proj,
        );
        dag.all_reduce_hidden(
            dag_label!(format!("L{layer_idx}.attn.all_reduce")),
            &mut bufs.attn_proj,
        )?;

        // Residual + LayerNorm
        dag.fused_add_rms_norm(
            dag_label!(format!("L{layer_idx}.attn.fused_add_rms_norm")),
            &mut bufs.hidden,
            &bufs.attn_proj,
            &layer.post_attention_layernorm,
            &mut bufs.normed,
        )?;

        // MLP: split gate/up GEMMs → silu_mul → down GEMM
        dag.mlp_gate_proj(
            dag_label!(format!("L{layer_idx}.mlp.gate_proj")),
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.gate_out,
        );
        dag.mlp_up_proj(
            dag_label!(format!("L{layer_idx}.mlp.up_proj")),
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.up_out,
        );
        dag.silu_mul_split(
            dag_label!(format!("L{layer_idx}.mlp.silu_mul")),
            &bufs.gate_out,
            &bufs.up_out,
            &mut bufs.mlp_act,
        )?;
        dag.down_proj(
            dag_label!(format!("L{layer_idx}.mlp.down_proj")),
            &layer.mlp.down_proj,
            &bufs.mlp_act,
            &mut bufs.mlp_out,
        );
        dag.all_reduce_hidden(
            dag_label!(format!("L{layer_idx}.mlp.all_reduce")),
            &mut bufs.mlp_out,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_decode_buffers::BatchDecodeBuffers;
    use crate::weights::ModelRuntimeConfig;
    use pegainfer_core::ops;
    use pegainfer_core::sampler::SamplingParams;
    use pegainfer_core::tensor::DeviceVec;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::path::Path;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

    fn get_model_path_or_skip() -> Option<String> {
        match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
            Ok(path) => Some(path),
            Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
                Some(MODEL_PATH.to_string())
            }
            Err(_) => {
                eprintln!(
                    "skipping Qwen3 batch decode model test because {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
                );
                None
            }
        }
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

    fn sample_logits(
        model: &Qwen3Model,
        logits: &DeviceVec,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> u32 {
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
        let mut valid: cudarc::driver::CudaSlice<u8> = model.ctx.stream.alloc_zeros(1).unwrap();
        let mut out: cudarc::driver::CudaSlice<i32> = model.ctx.stream.alloc_zeros(1).unwrap();
        let random_val: f32 = rand::RngExt::random(rng);
        ops::gpu_sample_into(
            &model.ctx,
            logits,
            &mut probs,
            &mut top1_value,
            &mut row_states,
            &mut valid,
            &mut out,
            params,
            random_val,
        )
        .unwrap()
    }

    fn prefill_one(
        model: &Qwen3Model,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> (KvState, u32) {
        let mut kv_state = model.kv_pool.alloc();
        let prompts: Vec<&[u32]> = vec![prompt_tokens];
        let mut kv_refs: Vec<&mut KvState> = vec![&mut kv_state];
        let (logits_vec, _) = model.batch_prefill(&prompts, &mut kv_refs, false).unwrap();
        let first_token = sample_logits(model, &logits_vec[0], params, rng);
        (kv_state, first_token)
    }

    /// Run single-request decode via batch_decode(bs=1) for a prompt, return generated token IDs.
    ///
    /// Uses batch_prefill and batch_decode with bs=1 so the decode path
    /// is identical to the multi-request batch — same cuBLAS handle, same FlashInfer
    /// BatchDecode kernel.
    fn sequential_decode(
        model: &Qwen3Model,
        prompt_tokens: &[u32],
        num_decode_steps: usize,
        seed: u64,
    ) -> Vec<u32> {
        let params = SamplingParams::default(); // greedy
        let mut rng = StdRng::seed_from_u64(seed);

        let (mut kv_state, first_token) = prefill_one(model, prompt_tokens, &params, &mut rng);
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
            model.local_num_attention_heads(),
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

        let mut kv_states: Vec<KvState> = (0..bs).map(|_| model.kv_pool.alloc()).collect();
        let mut kv_refs: Vec<&mut KvState> = kv_states.iter_mut().collect();
        let (logits_vec, _) = model.batch_prefill(prompts, &mut kv_refs, false).unwrap();
        let first_tokens: Vec<u32> = logits_vec
            .iter()
            .map(|logits| sample_logits(model, logits, &params, &mut rng))
            .collect();

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
            model.local_num_attention_heads(),
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
        let Some(model_path) = get_model_path_or_skip() else {
            return;
        };
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
                let (_kv_state, token) = prefill_one(&model, prompt.as_slice(), &params, &mut rng);
                seq_first_tokens.push(token);
            }

            let prompts: Vec<&[u32]> = vec![&prefill_a, &prefill_b, &prefill_c];
            let mut kv_states: Vec<KvState> = (0..3).map(|_| model.kv_pool.alloc()).collect();
            let mut kv_refs: Vec<&mut KvState> = kv_states.iter_mut().collect();
            let (logits_vec, _) = model.batch_prefill(&prompts, &mut kv_refs, false).unwrap();

            let mut batch_first_tokens = Vec::new();
            for logits in &logits_vec {
                let mut rng = StdRng::seed_from_u64(seed);
                let token = sample_logits(&model, logits, &params, &mut rng);
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
