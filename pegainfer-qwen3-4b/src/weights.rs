use anyhow::Result;
use cudarc::nccl::safe::{Comm, ReduceOp};
use log::{debug, info};
#[cfg(test)]
use std::path::Path;
use std::time::Instant;

use super::config::{Config, TensorParallelConfig};
use std::collections::HashMap;

use crate::lora::{DeviceLoraAdapter, DeviceLoraLayer};
use pegainfer_core::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use pegainfer_core::weight_loader::{
    deserialize_shards, load_shard_info, load_tensor_1d, load_tensor_2d, load_tensor_2d_col_shard,
    load_tensor_2d_row_shard, mmap_shards, precompute_rope,
};

pub(crate) struct KvBudget {
    pub(crate) num_layers: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) block_size: usize,
    pub(crate) num_blocks: usize,
}

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
    pub(super) tensor_parallel: TensorParallelConfig,
    pub(super) tp_comm: Option<Comm>,
    pub(super) lora_adapters: HashMap<String, DeviceLoraAdapter>,
    pub(super) active_lora_adapter: Option<String>,
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
            tensor_parallel,
            tp_comm: None,
            lora_adapters: HashMap::new(),
            active_lora_adapter: None,
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

    pub(crate) fn install_lora_adapter(
        &mut self,
        adapter: DeviceLoraAdapter,
        load_inplace: bool,
    ) -> Result<()> {
        debug!(
            "Installing Qwen3 LoRA adapter {} from {}",
            adapter.name,
            adapter.manifest.path.display()
        );
        install_lora_adapter_in_registry(&mut self.lora_adapters, adapter, load_inplace)
    }

    pub(crate) fn uninstall_lora_adapter(&mut self, name: &str) -> Result<()> {
        anyhow::ensure!(
            self.lora_adapters.remove(name).is_some(),
            "Qwen3 LoRA adapter {name} is not loaded"
        );
        if self.active_lora_adapter.as_deref() == Some(name) {
            self.active_lora_adapter = None;
        }
        Ok(())
    }

    pub(crate) fn activate_lora_adapter(&mut self, name: Option<&str>) -> Result<()> {
        match name {
            Some(name) => {
                anyhow::ensure!(
                    self.lora_adapters.contains_key(name),
                    "Qwen3 LoRA adapter {name} is not loaded"
                );
                self.active_lora_adapter = Some(name.to_string());
            }
            None => self.active_lora_adapter = None,
        }
        Ok(())
    }

    pub(crate) fn lora_layer(&self, layer_idx: usize) -> Option<(&DeviceLoraLayer, f32)> {
        self.active_lora_adapter.as_ref().and_then(|name| {
            let adapter = self.lora_adapters.get(name)?;
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

    /// KV cache geometry and budget for the executor to create a KvCacheManager.
    pub(crate) fn kv_budget(&self) -> KvBudget {
        let page_size = 16;
        let num_kv_heads = self.local_num_key_value_heads();
        let layout = pegainfer_kv_cache::KvLayout::new(
            self.config.num_hidden_layers,
            num_kv_heads,
            self.config.head_dim,
            page_size,
        );
        let bytes_per_block = layout.page_stride * std::mem::size_of::<half::bf16>();
        let (free_bytes, _) = cudarc::driver::result::mem_get_info().expect("cuMemGetInfo failed");
        let kv_budget_bytes = (free_bytes as f64 * 0.85) as usize;
        let num_blocks = (kv_budget_bytes / bytes_per_block).max(64);
        let kv_mb = num_blocks * bytes_per_block / (1024 * 1024);
        log::info!(
            "KV cache: {num_blocks} blocks ({kv_mb} MB, {:.0}% of {:.0} MB free)",
            kv_budget_bytes as f64 / free_bytes as f64 * 100.0,
            free_bytes as f64 / 1024.0 / 1024.0
        );
        KvBudget {
            num_layers: self.config.num_hidden_layers,
            num_kv_heads,
            head_dim: self.config.head_dim,
            block_size: page_size,
            num_blocks,
        }
    }
}

fn install_lora_adapter_in_registry(
    lora_adapters: &mut HashMap<String, DeviceLoraAdapter>,
    adapter: DeviceLoraAdapter,
    load_inplace: bool,
) -> Result<()> {
    if !load_inplace {
        anyhow::ensure!(
            !lora_adapters.contains_key(&adapter.name),
            "Qwen3 LoRA adapter {} is already loaded",
            adapter.name
        );
    }
    lora_adapters.insert(adapter.name.clone(), adapter);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{DeviceLoraLayer, LoraAdapterManifest};

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "pegainfer-qwen3-lora-{name}-{}",
            std::process::id()
        ))
    }

    fn test_device_adapter(name: &str, path: &Path) -> DeviceLoraAdapter {
        DeviceLoraAdapter {
            name: name.to_string(),
            manifest: LoraAdapterManifest {
                path: path.to_path_buf(),
                rank: 1,
                alpha: 1,
                target_modules: vec!["q_proj".to_string()],
                tensor_count: 0,
            },
            scale: 1.0,
            layers: vec![DeviceLoraLayer::default()],
        }
    }

    #[test]
    fn install_lora_adapter_requires_load_inplace_to_replace_existing_name() {
        let mut adapters = HashMap::new();
        let first_path = temp_path("replace-first");
        let second_path = temp_path("replace-second");

        let first = test_device_adapter("adapter-a", &first_path);
        install_lora_adapter_in_registry(&mut adapters, first, false)
            .expect("install first adapter");
        assert_eq!(
            adapters
                .get("adapter-a")
                .map(|adapter| adapter.manifest.path.as_path()),
            Some(first_path.as_path()),
        );

        let duplicate = test_device_adapter("adapter-a", &second_path);
        let error = install_lora_adapter_in_registry(&mut adapters, duplicate, false)
            .expect_err("duplicate adapter without load_inplace should fail")
            .to_string();
        assert!(error.contains("already loaded"));
        assert_eq!(
            adapters
                .get("adapter-a")
                .map(|adapter| adapter.manifest.path.as_path()),
            Some(first_path.as_path()),
        );

        let replacement = test_device_adapter("adapter-a", &second_path);
        install_lora_adapter_in_registry(&mut adapters, replacement, true)
            .expect("replace adapter");
        assert_eq!(
            adapters
                .get("adapter-a")
                .map(|adapter| adapter.manifest.path.as_path()),
            Some(second_path.as_path()),
        );
    }
}
