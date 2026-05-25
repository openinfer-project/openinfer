use anyhow::Result;
use cudarc::nccl::safe::{Comm, ReduceOp};
use log::{debug, info};
use std::time::Instant;

use super::config::{Config, TensorParallelConfig};
use crate::lora::{DeviceLoraAdapter, DeviceLoraLayer};
use pegainfer_core::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use pegainfer_core::weight_loader::{
    deserialize_shards, load_shard_info, load_tensor_1d, load_tensor_2d, load_tensor_2d_col_shard,
    load_tensor_2d_row_shard, mmap_shards, precompute_rope,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelRuntimeConfig {
    pub(crate) enable_cuda_graph: bool,
    pub(crate) tensor_parallel: Option<TensorParallelConfig>,
    pub(crate) device_ordinal: usize,
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            tensor_parallel: None,
            device_ordinal: 0,
        }
    }
}

/// Attention layer weights.
/// QKV stored as a single concatenated matrix [q_dim + 2*kv_dim, hidden_size].
/// Individual projections accessed via row offsets (zero extra memory).
pub(super) struct Attention {
    /// Fused [q_proj; k_proj; v_proj] row-major
    pub(super) qkv_proj: DeviceMatrix,
    pub(super) o_proj: DeviceMatrix,
    pub(super) q_norm: DeviceVec,
    pub(super) k_norm: DeviceVec,
    pub(super) q_dim: usize,
    pub(super) kv_dim: usize,
}

/// MLP layer weights.
/// Gate+Up stored as a single concatenated matrix [2*intermediate_size, hidden_size].
#[allow(clippy::upper_case_acronyms, clippy::struct_field_names)]
pub(super) struct MLP {
    /// Fused [gate_proj; up_proj] row-major
    pub(super) gate_up_proj: DeviceMatrix,
    pub(super) down_proj: DeviceMatrix,
}

/// Transformer block
pub(super) struct TransformerBlock {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attention: Attention,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: MLP,
}

/// Qwen3 model — weights and config only. Request state is owned by the executor.
pub(crate) struct Qwen3Model {
    pub(super) ctx: DeviceContext,
    pub(super) config: Config,
    pub(super) embed_tokens: DeviceMatrix,
    pub(super) lm_head: Option<DeviceMatrix>,
    pub(super) layers: Vec<TransformerBlock>,
    pub(super) norm: DeviceVec,
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
    pub(super) enable_cuda_graph: bool,
    pub(super) kv_pool: pegainfer_core::kv_pool::KvPool,
    pub(super) tensor_parallel: TensorParallelConfig,
    pub(super) tp_comm: Option<Comm>,
    pub(super) lora_adapter: Option<DeviceLoraAdapter>,
}

// SAFETY: Each model instance is pinned to a single CUDA device and is only
// driven from one worker thread at a time. The TP path creates one model per
// rank and never shares a single rank-local model concurrently across threads.
unsafe impl Send for Qwen3Model {}
unsafe impl Sync for Qwen3Model {}

impl Qwen3Model {
    pub(crate) fn from_safetensors_with_runtime(
        model_path: &str,
        runtime: ModelRuntimeConfig,
    ) -> Result<Self> {
        info!("Loading model from: {}", model_path);
        debug!("Initializing GPU device {}", runtime.device_ordinal);
        let ctx = DeviceContext::new_with_device(runtime.device_ordinal)?;

        let config = Config::from_file(model_path)?;
        let tensor_parallel = runtime.tensor_parallel.unwrap_or_default();
        tensor_parallel.validate_for(&config)?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        debug!("Loading {} safetensor shard(s)", shard_paths.len());
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let t_gpu = Instant::now();
        debug!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(&ctx, &shards, &weight_map, "model.embed_tokens.weight")?;
        let lm_head = if config.tie_word_embeddings {
            debug!("Using tied input/output embeddings");
            None
        } else {
            debug!("Loading untied LM head to GPU");
            Some(load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                config.lm_head_tensor_name(),
            )?)
        };

        debug!(
            "Loading layers to GPU: num_layers={}, tp_rank={}, tp_world_size={}",
            config.num_hidden_layers, tensor_parallel.rank, tensor_parallel.world_size,
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let (q_row_offset, q_rows) =
            tensor_parallel.shard_range(config.num_attention_heads * config.head_dim);
        let (kv_row_offset, kv_rows) =
            tensor_parallel.shard_range(config.num_key_value_heads * config.head_dim);
        let (inter_row_offset, inter_rows) = tensor_parallel.shard_range(config.intermediate_size);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);

            let q_proj = if tensor_parallel.is_sharded() {
                load_tensor_2d_row_shard(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    q_row_offset,
                    q_rows,
                )?
            } else {
                load_tensor_2d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.self_attn.q_proj.weight", prefix),
                )?
            };
            let k_proj = if tensor_parallel.is_sharded() {
                load_tensor_2d_row_shard(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    kv_row_offset,
                    kv_rows,
                )?
            } else {
                load_tensor_2d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.self_attn.k_proj.weight", prefix),
                )?
            };
            let v_proj = if tensor_parallel.is_sharded() {
                load_tensor_2d_row_shard(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    kv_row_offset,
                    kv_rows,
                )?
            } else {
                load_tensor_2d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.self_attn.v_proj.weight", prefix),
                )?
            };
            let q_dim = q_proj.rows;
            let kv_dim = k_proj.rows;
            let qkv_proj = DeviceMatrix::vstack(&ctx, &[&q_proj, &k_proj, &v_proj])?;
            drop(q_proj);
            drop(k_proj);
            drop(v_proj);

            let gate_proj = if tensor_parallel.is_sharded() {
                load_tensor_2d_row_shard(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.mlp.gate_proj.weight", prefix),
                    inter_row_offset,
                    inter_rows,
                )?
            } else {
                load_tensor_2d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.mlp.gate_proj.weight", prefix),
                )?
            };
            let up_proj = if tensor_parallel.is_sharded() {
                load_tensor_2d_row_shard(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.mlp.up_proj.weight", prefix),
                    inter_row_offset,
                    inter_rows,
                )?
            } else {
                load_tensor_2d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.mlp.up_proj.weight", prefix),
                )?
            };
            let gate_up_proj = DeviceMatrix::vstack(&ctx, &[&gate_proj, &up_proj])?;
            drop(gate_proj);
            drop(up_proj);

            let block = TransformerBlock {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                attention: Attention {
                    qkv_proj,
                    o_proj: if tensor_parallel.is_sharded() {
                        load_tensor_2d_col_shard(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.self_attn.o_proj.weight", prefix),
                            q_row_offset,
                            q_rows,
                        )?
                    } else {
                        load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.self_attn.o_proj.weight", prefix),
                        )?
                    },
                    q_norm: load_tensor_1d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.q_norm.weight", prefix),
                    )?,
                    k_norm: load_tensor_1d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.k_norm.weight", prefix),
                    )?,
                    q_dim,
                    kv_dim,
                },
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                mlp: MLP {
                    gate_up_proj,
                    down_proj: if tensor_parallel.is_sharded() {
                        load_tensor_2d_col_shard(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.mlp.down_proj.weight", prefix),
                            inter_row_offset,
                            inter_rows,
                        )?
                    } else {
                        load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.mlp.down_proj.weight", prefix),
                        )?
                    },
                },
            };
            layers.push(block);
        }

        let norm = load_tensor_1d(&ctx, &shards, &weight_map, "model.norm.weight")?;

        debug!("Precomputing RoPE cache on GPU");
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.head_dim, 4096, config.rope_theta)?;

        ctx.sync()?;
        info!(
            "GPU transfer complete in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );
        info!("GPU model loaded successfully");

        let page_size = 16;
        let layout = pegainfer_core::kv_pool::KvLayout::new(
            config.num_hidden_layers,
            config.local_num_key_value_heads(tensor_parallel),
            config.head_dim,
            page_size,
        );
        let bytes_per_page = layout.page_stride * std::mem::size_of::<half::bf16>();
        let (free_bytes, _total_bytes) = cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow::anyhow!("cuMemGetInfo failed: {e}"))?;
        let kv_budget = (free_bytes as f64 * 0.85) as usize;
        let num_pages = (kv_budget / bytes_per_page).max(64);
        let kv_mb = num_pages * bytes_per_page / (1024 * 1024);
        info!(
            "KV cache: {num_pages} pages ({kv_mb} MB, {:.0}% of {:.0} MB free)",
            kv_budget as f64 / free_bytes as f64 * 100.0,
            free_bytes as f64 / 1024.0 / 1024.0
        );
        let kv_pool = pegainfer_core::kv_pool::KvPool::new(
            &ctx,
            config.num_hidden_layers,
            config.local_num_key_value_heads(tensor_parallel),
            config.head_dim,
            page_size,
            num_pages,
        )?;

        let model = Self {
            ctx,
            config,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            enable_cuda_graph: runtime.enable_cuda_graph,
            kv_pool,
            tensor_parallel,
            tp_comm: None,
            lora_adapter: None,
        };

        if model.enable_cuda_graph {
            debug!("Decode path CUDA Graph is enabled (captures on first decode step)");
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }

        Ok(model)
    }

    pub(super) fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    pub(crate) fn device_ctx(&self) -> &pegainfer_core::tensor::DeviceContext {
        &self.ctx
    }

    pub(crate) fn local_num_attention_heads(&self) -> usize {
        self.config.local_num_attention_heads(self.tensor_parallel)
    }

    pub(crate) fn local_num_key_value_heads(&self) -> usize {
        self.config.local_num_key_value_heads(self.tensor_parallel)
    }

    pub(crate) fn local_intermediate_size(&self) -> usize {
        self.config.local_intermediate_size(self.tensor_parallel)
    }

    pub(crate) fn local_q_dim(&self) -> usize {
        self.config.local_q_dim(self.tensor_parallel)
    }

    pub(crate) fn local_kv_dim(&self) -> usize {
        self.config.local_kv_dim(self.tensor_parallel)
    }

    pub(crate) fn attach_tp_comm(&mut self, comm: Comm) {
        self.tp_comm = Some(comm);
    }

    pub(crate) fn set_lora_adapter(&mut self, adapter: DeviceLoraAdapter) {
        debug!(
            "Activating Qwen3 LoRA adapter {} from {}",
            adapter.name,
            adapter.manifest.path.display()
        );
        self.lora_adapter = Some(adapter);
    }

    pub(crate) fn lora_layer(&self, layer_idx: usize) -> Option<(&DeviceLoraLayer, f32)> {
        self.lora_adapter.as_ref().and_then(|adapter| {
            adapter
                .layers
                .get(layer_idx)
                .map(|layer| (layer, adapter.scale))
        })
    }

    pub(crate) fn all_reduce_hidden(
        &self,
        hidden: &mut pegainfer_core::tensor::HiddenStates,
    ) -> Result<()> {
        #[cfg(feature = "kernel-call-trace")]
        if pegainfer_core::ops::call_trace::is_enabled() {
            let label = pegainfer_core::ops::call_trace::current_label("all_reduce_hidden");
            pegainfer_core::ops::call_trace::record_call(
                pegainfer_core::ops::call_spec::all_reduce_hidden_call(
                    label,
                    hidden.hidden_dim,
                    hidden.seq_len,
                ),
            );
        }
        self.all_reduce_hidden_untraced(hidden)
    }

    pub(crate) fn all_reduce_hidden_untraced(
        &self,
        hidden: &mut pegainfer_core::tensor::HiddenStates,
    ) -> Result<()> {
        if let Some(comm) = &self.tp_comm {
            comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum)
                .map_err(|e| anyhow::anyhow!("nccl all-reduce failed: {e:?}"))?;
        }
        Ok(())
    }

    /// Allocate a fresh (empty) per-request KV state from the shared pool.
    pub(crate) fn alloc_kv(&self) -> pegainfer_core::kv_pool::KvState {
        self.kv_pool.alloc()
    }

    pub(crate) fn kv_pool(&self) -> &pegainfer_core::kv_pool::KvPool {
        &self.kv_pool
    }

    /// Create pre-allocated batch decode buffers.
    pub(crate) fn create_batch_decode_bufs(
        &self,
        max_batch_size: usize,
    ) -> anyhow::Result<super::batch_decode_buffers::BatchDecodeBuffers> {
        super::batch_decode_buffers::BatchDecodeBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            self.local_q_dim(),
            self.local_kv_dim(),
            self.local_intermediate_size(),
            self.config.vocab_size,
            max_batch_size,
            self.kv_pool.capacity_pages(),
            self.kv_pool.padding_page_id(),
            self.local_num_attention_heads(),
        )
    }
}
