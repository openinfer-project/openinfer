//! Batched decode: process N requests' tokens in one forward pass.

use anyhow::Result;

use cudarc::driver::CudaSlice;
use half::bf16;

use super::batch_decode_buffers::{
    BATCH_BUCKETS, BatchDecodeBuffers, DecodeAttentionPath, bucket_for,
};
use super::batch_decode_dag::BatchDecodeDag;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::lora::apply_lora_projection_delta;
use pegainfer_core::kv_pool::KvLayout;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops;
use pegainfer_kernels::tensor::{KvDim, QDim};
use pegainfer_kv_cache::KvView;

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
        kv_views: &[KvView],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let bs = token_ids.len();
        assert_eq!(bs, kv_views.len());
        assert!(bs > 0);

        // Derive positions from views (seq_len - 1 = position of the new token)
        let mut positions: Vec<i32> = kv_views.iter().map(|v| (v.seq_len() - 1) as i32).collect();

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

        let kv_refs: Vec<&KvView> = kv_views.iter().collect();
        bufs.sync_paged_meta(&self.ctx, &kv_refs, padded_bs)?;
        let attention_path = bufs.attention_path(padded_bs);
        #[cfg(feature = "kernel-call-trace")]
        let trace_kv_len = kv_views.iter().map(|v| v.seq_len()).max().unwrap_or(0);
        if self.enable_cuda_graph {
            let bucket_idx = BATCH_BUCKETS.iter().position(|&b| b == padded_bs).unwrap();
            let graph_idx = BatchDecodeBuffers::graph_index(bucket_idx, attention_path);
            // Take graphs out of bufs to avoid split-borrow conflict with closure
            let mut graphs = std::mem::take(&mut bufs.graphs);
            let result = graphs[graph_idx].run_or_capture(&self.ctx, || {
                trace_decode_kv_len!(trace_kv_len, {
                    self.batch_decode_kernels(kv_buffer, layout, padded_bs, attention_path, bufs)
                })
            });
            bufs.graphs = graphs;
            result?;
        } else {
            trace_decode_kv_len!(trace_kv_len, {
                self.batch_decode_kernels(kv_buffer, layout, padded_bs, attention_path, bufs)
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
            self.batch_decode_layer(layer_idx, layer, &dag, bufs)?;

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
        if let Some((lora_layer, scale)) = self.lora_layer(layer_idx)
            && let Some(projection) = &lora_layer.q_proj
        {
            apply_lora_projection_delta(
                &self.ctx,
                projection,
                &bufs.normed,
                &mut bufs.q,
                0,
                scale,
            )?;
        }
        dag.gemm_rows::<KvDim>(
            dag_label!(format!("L{layer_idx}.attn.k_proj")),
            &layer.attention.qkv_proj,
            q_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.k,
        );
        if let Some((lora_layer, scale)) = self.lora_layer(layer_idx)
            && let Some(projection) = &lora_layer.k_proj
        {
            apply_lora_projection_delta(
                &self.ctx,
                projection,
                &bufs.normed,
                &mut bufs.k,
                0,
                scale,
            )?;
        }
        dag.gemm_rows::<KvDim>(
            dag_label!(format!("L{layer_idx}.attn.v_proj")),
            &layer.attention.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.v,
        );
        if let Some((lora_layer, scale)) = self.lora_layer(layer_idx)
            && let Some(projection) = &lora_layer.v_proj
        {
            apply_lora_projection_delta(
                &self.ctx,
                projection,
                &bufs.normed,
                &mut bufs.v,
                0,
                scale,
            )?;
        }

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
        if let Some((lora_layer, scale)) = self.lora_layer(layer_idx)
            && let Some(projection) = &lora_layer.o_proj
        {
            apply_lora_projection_delta(
                &self.ctx,
                projection,
                &bufs.attn_out,
                &mut bufs.attn_proj,
                0,
                scale,
            )?;
        }
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
        if let Some((lora_layer, scale)) = self.lora_layer(layer_idx) {
            if let Some(projection) = &lora_layer.gate_proj {
                apply_lora_projection_delta(
                    &self.ctx,
                    projection,
                    &bufs.normed,
                    &mut bufs.gate_out,
                    0,
                    scale,
                )?;
            }
            if let Some(projection) = &lora_layer.up_proj {
                apply_lora_projection_delta(
                    &self.ctx,
                    projection,
                    &bufs.normed,
                    &mut bufs.up_out,
                    0,
                    scale,
                )?;
            }
        }
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
        if let Some((lora_layer, scale)) = self.lora_layer(layer_idx)
            && let Some(projection) = &lora_layer.down_proj
        {
            apply_lora_projection_delta(
                &self.ctx,
                projection,
                &bufs.mlp_act,
                &mut bufs.mlp_out,
                0,
                scale,
            )?;
        }
        dag.all_reduce_hidden(
            dag_label!(format!("L{layer_idx}.mlp.all_reduce")),
            &mut bufs.mlp_out,
        )?;

        Ok(())
    }
}
