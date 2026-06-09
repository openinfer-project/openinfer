//! Batched decode: process N requests' tokens in one forward pass.

use anyhow::Result;

use cudarc::driver::CudaSlice;
use half::bf16;

use super::batch_decode_buffers::{
    BATCH_BUCKETS, BatchDecodeBuffers, DecodeAttentionPath, bucket_for,
};
use super::batch_decode_dag::BatchDecodeDag;
use super::weights::{PackedLoraProjection, Qwen3Model, TransformerBlock};
use crate::lora::LoraProjectionKind;
use openinfer_core::kv_pool::KvLayout;
use openinfer_core::ops;
use openinfer_kernels::tensor::{KvDim, QDim};
use openinfer_kv_cache::KvView;

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
    /// When `enable_cuda_graph` is set and the batch does not use LoRA, pads
    /// to the nearest bucket size and uses per-bucket CUDA Graph capture/replay.
    /// LoRA batches currently run eager because packed LoRA slots are not part
    /// of the CUDA Graph cache key yet.
    pub(crate) fn batch_decode(
        &self,
        token_ids: &[u32],
        kv_views: &[KvView],
        lora_adapters: &[Option<&str>],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let bs = token_ids.len();
        assert_eq!(bs, kv_views.len());
        assert_eq!(bs, lora_adapters.len());
        assert!(bs > 0);
        let lora_slots = self.decode_lora_slots(lora_adapters)?;
        let use_lora = lora_slots.is_some();
        let use_cuda_graph = self.enable_cuda_graph && lora_slots.is_none();

        // Derive positions from views (seq_len - 1 = position of the new token)
        let mut positions: Vec<i32> = kv_views.iter().map(|v| (v.seq_len() - 1) as i32).collect();

        // Pad to bucket size for CUDA Graph stability
        let padded_bs = if use_cuda_graph { bucket_for(bs) } else { bs };

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
        if let Some(lora_slots) = &lora_slots {
            self.ctx
                .stream
                .memcpy_htod(lora_slots, &mut bufs.lora_token_slots_d)?;
        }

        let kv_refs: Vec<&KvView> = kv_views.iter().collect();
        bufs.sync_paged_meta(&self.ctx, &kv_refs, padded_bs)?;
        let attention_path = bufs.attention_path(padded_bs);
        #[cfg(feature = "kernel-call-trace")]
        let trace_kv_len = kv_views.iter().map(|v| v.seq_len()).max().unwrap_or(0);
        if use_cuda_graph {
            let bucket_idx = BATCH_BUCKETS.iter().position(|&b| b == padded_bs).unwrap();
            let graph_idx = BatchDecodeBuffers::graph_index(bucket_idx, attention_path);
            // Take graphs out of bufs to avoid split-borrow conflict with closure
            let mut graphs = std::mem::take(&mut bufs.graphs);
            let result = graphs[graph_idx].run_or_capture(&self.ctx, || {
                trace_decode_kv_len!(trace_kv_len, {
                    self.batch_decode_kernels(
                        kv_buffer,
                        layout,
                        padded_bs,
                        attention_path,
                        use_lora,
                        bufs,
                    )
                })
            });
            bufs.graphs = graphs;
            result?;
        } else {
            trace_decode_kv_len!(trace_kv_len, {
                self.batch_decode_kernels(
                    kv_buffer,
                    layout,
                    padded_bs,
                    attention_path,
                    use_lora,
                    bufs,
                )
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
        use_lora: bool,
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
            self.batch_decode_layer(layer_idx, layer, &dag, use_lora, bufs)?;

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
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        dag: &BatchDecodeDag<'_>,
        use_lora: bool,
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
        self.apply_decode_lora_projection_group3(
            layer_idx,
            LoraProjectionKind::Q,
            LoraProjectionKind::K,
            LoraProjectionKind::V,
            use_lora,
            &bufs.normed,
            &mut bufs.q,
            &mut bufs.k,
            &mut bufs.v,
            &bufs.lora_token_slots_d,
        )?;

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
        self.apply_decode_lora_projection(
            layer_idx,
            LoraProjectionKind::O,
            use_lora,
            &bufs.attn_out,
            &mut bufs.attn_proj,
            0,
            &bufs.lora_token_slots_d,
        )?;
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
        self.apply_decode_lora_projection_group2(
            layer_idx,
            LoraProjectionKind::Gate,
            LoraProjectionKind::Up,
            use_lora,
            &bufs.normed,
            &mut bufs.gate_out,
            &mut bufs.up_out,
            &bufs.lora_token_slots_d,
        )?;
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
        self.apply_decode_lora_projection(
            layer_idx,
            LoraProjectionKind::Down,
            use_lora,
            &bufs.mlp_act,
            &mut bufs.mlp_out,
            0,
            &bufs.lora_token_slots_d,
        )?;
        dag.all_reduce_hidden(
            dag_label!(format!("L{layer_idx}.mlp.all_reduce")),
            &mut bufs.mlp_out,
        )?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_decode_lora_projection_group2(
        &self,
        layer_idx: usize,
        kind0: LoraProjectionKind,
        kind1: LoraProjectionKind,
        use_lora: bool,
        input: &openinfer_core::tensor::HiddenStates,
        out0: &mut openinfer_core::tensor::HiddenStates,
        out1: &mut openinfer_core::tensor::HiddenStates,
        token_slots: &CudaSlice<i32>,
    ) -> Result<()> {
        if !use_lora {
            return Ok(());
        }

        let p0 = self.decode_grouped_lora_projection(layer_idx, kind0, out0);
        let p1 = self.decode_grouped_lora_projection(layer_idx, kind1, out1);
        ops::lora_decode_fused_delta_group3_into(&self.ctx, token_slots, input, p0, p1, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_decode_lora_projection_group3(
        &self,
        layer_idx: usize,
        kind0: LoraProjectionKind,
        kind1: LoraProjectionKind,
        kind2: LoraProjectionKind,
        use_lora: bool,
        input: &openinfer_core::tensor::HiddenStates,
        out0: &mut openinfer_core::tensor::HiddenStates,
        out1: &mut openinfer_core::tensor::HiddenStates,
        out2: &mut openinfer_core::tensor::HiddenStates,
        token_slots: &CudaSlice<i32>,
    ) -> Result<()> {
        if !use_lora {
            return Ok(());
        }

        let p0 = self.decode_grouped_lora_projection(layer_idx, kind0, out0);
        let p1 = self.decode_grouped_lora_projection(layer_idx, kind1, out1);
        let p2 = self.decode_grouped_lora_projection(layer_idx, kind2, out2);
        ops::lora_decode_fused_delta_group3_into(&self.ctx, token_slots, input, p0, p1, p2)
    }

    fn decode_grouped_lora_projection<'a>(
        &'a self,
        layer_idx: usize,
        kind: LoraProjectionKind,
        out: &'a mut openinfer_core::tensor::HiddenStates,
    ) -> Option<ops::LoraDecodeGroupedProjection<'a>> {
        let packed = self.packed_lora_projection(layer_idx, kind)?;
        Some(grouped_projection_from_packed(packed, out))
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_decode_lora_projection(
        &self,
        layer_idx: usize,
        kind: LoraProjectionKind,
        use_lora: bool,
        input: &openinfer_core::tensor::HiddenStates,
        out: &mut openinfer_core::tensor::HiddenStates,
        row_offset: usize,
        token_slots: &CudaSlice<i32>,
    ) -> Result<()> {
        if !use_lora {
            return Ok(());
        }

        let Some(packed) = self.packed_lora_projection(layer_idx, kind) else {
            return Ok(());
        };
        ops::lora_decode_fused_delta_into(
            &self.ctx,
            &packed.a,
            &packed.b,
            &packed.scales,
            token_slots,
            input,
            out,
            packed.max_loras,
            packed.max_rank,
            packed.rank,
            packed.out_dim,
            row_offset,
        )
    }
}

fn grouped_projection_from_packed<'a>(
    packed: &'a PackedLoraProjection,
    out: &'a mut openinfer_core::tensor::HiddenStates,
) -> ops::LoraDecodeGroupedProjection<'a> {
    ops::LoraDecodeGroupedProjection {
        a_packed: &packed.a,
        b_packed: &packed.b,
        scales: &packed.scales,
        out,
        max_loras: packed.max_loras,
        max_rank: packed.max_rank,
        rank: packed.rank,
        out_dim: packed.out_dim,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_decode_buffers::BatchDecodeBuffers;
    use crate::weights::ModelRuntimeConfig;
    use openinfer_core::ops;
    use openinfer_core::sampler::SamplingParams;
    use openinfer_core::tensor::DeviceVec;
    use openinfer_kv_cache::{KvCacheManager, RequestKv};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::path::Path;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

    fn get_model_path_or_skip() -> Option<String> {
        match std::env::var("OPENINFER_TEST_MODEL_PATH") {
            Ok(path) => Some(path),
            Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
                Some(MODEL_PATH.to_string())
            }
            Err(_) => {
                eprintln!(
                    "skipping Qwen3 batch decode model test because {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
                );
                None
            }
        }
    }

    fn make_kv_mgr(model: &Qwen3Model) -> KvCacheManager {
        let budget = model.kv_budget();
        KvCacheManager::new(
            &model.ctx.stream,
            budget.num_layers,
            budget.num_kv_heads,
            budget.head_dim,
            budget.block_size,
            budget.num_blocks,
        )
        .unwrap()
    }

    fn core_layout(mgr: &KvCacheManager) -> KvLayout {
        let l = mgr.buffer().layout();
        KvLayout::new(l.num_layers, l.num_kv_heads, l.head_dim, l.page_size)
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
            .alloc_zeros(openinfer_core::ops::flashinfer_topk_row_states_bytes())
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
            .alloc_zeros(openinfer_core::ops::flashinfer_topk_row_states_bytes())
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
        mgr: &KvCacheManager,
        layout: &KvLayout,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> (RequestKv, u32) {
        let mut rkv = mgr.pool().new_request(prompt_tokens.to_vec(), 256, None);
        rkv.schedule_prefill(prompt_tokens.len(), mgr.pool())
            .unwrap();
        let view = rkv.prefill_view(prompt_tokens.len());
        let (logits_vec, _) = model
            .batch_prefill(
                &[prompt_tokens],
                &[view],
                &[None],
                mgr.buffer().buffer(),
                layout,
                false,
            )
            .unwrap();
        let first_token = sample_logits(model, &logits_vec[0], params, rng);
        rkv.apply_prefill(first_token, mgr.pool()).unwrap();
        (rkv, first_token)
    }

    fn sequential_decode(
        model: &Qwen3Model,
        mgr: &KvCacheManager,
        layout: &KvLayout,
        prompt_tokens: &[u32],
        num_decode_steps: usize,
        seed: u64,
    ) -> Vec<u32> {
        let params = SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(seed);

        let (mut rkv, first_token) =
            prefill_one(model, mgr, layout, prompt_tokens, &params, &mut rng);

        let mut bufs = BatchDecodeBuffers::new(
            &model.ctx,
            model.config.hidden_size,
            model.local_q_dim(),
            model.local_kv_dim(),
            model.local_intermediate_size(),
            model.config.vocab_size,
            1,
            mgr.pool().total_blocks(),
            mgr.pool().padding_block_id(),
            model.local_num_attention_heads(),
        )
        .unwrap();

        let mut tokens = vec![first_token];
        for _ in 1..num_decode_steps {
            let token_ids = [*tokens.last().unwrap()];
            rkv.schedule_decode(mgr.pool()).unwrap();
            let view = rkv.decode_view();
            model
                .batch_decode(
                    &token_ids,
                    &[view],
                    &[None],
                    mgr.buffer().buffer(),
                    layout,
                    &mut bufs,
                )
                .unwrap();
            let params_refs: Vec<&SamplingParams> = vec![&params];
            let batch_tokens = sample_batch_tokens(model, &bufs, &params_refs, &mut rng);
            rkv.apply_decode(batch_tokens[0], mgr.pool()).unwrap();
            tokens.push(batch_tokens[0]);
        }
        rkv.release().unwrap();
        tokens
    }

    fn batch_decode_run(
        model: &Qwen3Model,
        mgr: &KvCacheManager,
        layout: &KvLayout,
        prompts: &[&[u32]],
        num_decode_steps: usize,
        seed: u64,
    ) -> Vec<Vec<u32>> {
        let bs = prompts.len();
        let params = SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(seed);

        // Prefill all requests
        let mut rkvs: Vec<RequestKv> = Vec::with_capacity(bs);
        for &prompt in prompts {
            let mut rkv = mgr.pool().new_request(prompt.to_vec(), 256, None);
            rkv.schedule_prefill(prompt.len(), mgr.pool()).unwrap();
            rkvs.push(rkv);
        }
        let views: Vec<_> = rkvs
            .iter()
            .zip(prompts.iter())
            .map(|(r, p)| r.prefill_view(p.len()))
            .collect();
        let (logits_vec, _) = model
            .batch_prefill(
                prompts,
                &views,
                &vec![None; prompts.len()],
                mgr.buffer().buffer(),
                layout,
                false,
            )
            .unwrap();
        let first_tokens: Vec<u32> = logits_vec
            .iter()
            .map(|logits| sample_logits(model, logits, &params, &mut rng))
            .collect();
        for (rkv, &tok) in rkvs.iter_mut().zip(&first_tokens) {
            rkv.apply_prefill(tok, mgr.pool()).unwrap();
        }

        let mut all_tokens: Vec<Vec<u32>> = first_tokens.iter().map(|&t| vec![t]).collect();

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
            mgr.pool().total_blocks(),
            mgr.pool().padding_block_id(),
            model.local_num_attention_heads(),
        )
        .unwrap();

        for _ in 1..num_decode_steps {
            let token_ids: Vec<u32> = all_tokens.iter().map(|t| *t.last().unwrap()).collect();
            for rkv in &mut rkvs {
                rkv.schedule_decode(mgr.pool()).unwrap();
            }
            let views: Vec<_> = rkvs.iter().map(|r| r.decode_view()).collect();
            model
                .batch_decode(
                    &token_ids,
                    &views,
                    &vec![None; token_ids.len()],
                    mgr.buffer().buffer(),
                    layout,
                    &mut bufs,
                )
                .unwrap();
            let params_refs: Vec<&SamplingParams> = (0..bs).map(|_| &params).collect();
            let tokens = sample_batch_tokens(model, &bufs, &params_refs, &mut rng);
            for (i, &tok) in tokens.iter().enumerate() {
                rkvs[i].apply_decode(tok, mgr.pool()).unwrap();
                all_tokens[i].push(tok);
            }
        }

        for rkv in &mut rkvs {
            rkv.release().unwrap();
        }
        all_tokens
    }

    #[test]
    fn batch_matches_sequential() {
        let Some(model_path) = get_model_path_or_skip() else {
            return;
        };
        let prompt_a: Vec<u32> = vec![9707];
        let prompt_b: Vec<u32> = vec![3838, 374, 220, 17, 10, 17];
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
                    ..Default::default()
                },
            )
            .unwrap();
            let mgr = make_kv_mgr(&model);
            let layout = core_layout(&mgr);

            // Batch prefill: multi-token prompts
            let prefill_a: Vec<u32> = vec![3838, 374, 220, 17, 10, 17];
            let prefill_b: Vec<u32> = (1..65).collect();
            let prefill_c: Vec<u32> = (1..129).collect();
            let params = SamplingParams::default();

            let mut seq_first_tokens = Vec::new();
            for prompt in [&prefill_a, &prefill_b, &prefill_c] {
                let mut rng = StdRng::seed_from_u64(seed);
                let (_rkv, token) =
                    prefill_one(&model, &mgr, &layout, prompt.as_slice(), &params, &mut rng);
                seq_first_tokens.push(token);
            }

            // Batch prefill all three at once
            let prompts: Vec<&[u32]> = vec![&prefill_a, &prefill_b, &prefill_c];
            let mut rkvs: Vec<RequestKv> = prompts
                .iter()
                .map(|p| {
                    let mut r = mgr.pool().new_request(p.to_vec(), 256, None);
                    r.schedule_prefill(p.len(), mgr.pool()).unwrap();
                    r
                })
                .collect();
            let views: Vec<_> = rkvs
                .iter()
                .zip(&prompts)
                .map(|(r, p)| r.prefill_view(p.len()))
                .collect();
            let (logits_vec, _) = model
                .batch_prefill(
                    &prompts,
                    &views,
                    &vec![None; prompts.len()],
                    mgr.buffer().buffer(),
                    &layout,
                    false,
                )
                .unwrap();

            let mut batch_first_tokens = Vec::new();
            for (i, logits) in logits_vec.iter().enumerate() {
                let mut rng = StdRng::seed_from_u64(seed);
                let token = sample_logits(&model, logits, &params, &mut rng);
                rkvs[i].apply_prefill(token, mgr.pool()).unwrap();
                batch_first_tokens.push(token);
            }
            for rkv in &mut rkvs {
                rkv.release().unwrap();
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

            let seq_a = sequential_decode(&model, &mgr, &layout, &prompt_a, num_steps, seed);
            let seq_b = sequential_decode(&model, &mgr, &layout, &prompt_b, num_steps, seed);
            let batch = batch_decode_run(
                &model,
                &mgr,
                &layout,
                &[&prompt_a, &prompt_b],
                num_steps,
                seed,
            );

            assert_eq!(batch[0], seq_a, "Decode request A mismatch");
            assert_eq!(batch[1], seq_b, "Decode request B mismatch");
        }

        // --- Phase 2: batch decode with CUDA Graph ---
        {
            let model = Qwen3Model::from_safetensors_with_runtime(
                &model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph: true,
                    tensor_parallel: None,
                    device_ordinal: 0,
                    ..Default::default()
                },
            )
            .unwrap();
            let mgr = make_kv_mgr(&model);
            let layout = core_layout(&mgr);

            let seq_a = sequential_decode(&model, &mgr, &layout, &prompt_a, num_steps, seed);
            let seq_b = sequential_decode(&model, &mgr, &layout, &prompt_b, num_steps, seed);
            let batch = batch_decode_run(
                &model,
                &mgr,
                &layout,
                &[&prompt_a, &prompt_b],
                num_steps,
                seed,
            );

            assert_eq!(batch[0], seq_a, "Graph request A mismatch");
            assert_eq!(batch[1], seq_b, "Graph request B mismatch");
        }
    }
}
