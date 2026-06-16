//! Higgs backbone forward — mirrors `openinfer-qwen3-4b` weight loading and
//! prefill path, adapted for `body.*` weight names and `text_config` from
//! Higgs config.json.
//!
//! Compiled only under `#[cfg(feature = "higgs-audio")]`.

use anyhow::{Context, Result};
use cudarc::driver::CudaSlice;
use half::bf16;
use log::{debug, info};
use std::time::Instant;

use crate::config::HiggsConfig;
use openinfer_core::kv_pool::KvLayout;
use openinfer_core::ops;
use openinfer_core::ops::PrefillPagedPlan;
use openinfer_core::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use openinfer_core::weight_loader::{
    deserialize_shards, load_shard_info, load_tensor_1d, load_tensor_2d, mmap_shards,
    precompute_rope,
};

// ── Data structures (identical to qwen3, minus LoRA / TP / CUDA graph) ──────

/// Attention layer weights.
///
/// QKV stored as a single concatenated matrix `[q_dim + 2*kv_dim, hidden_size]`.
/// Individual projections accessed via row offsets (zero extra memory).
pub(super) struct Attention {
    pub(super) qkv_proj: DeviceMatrix,
    pub(super) o_proj: DeviceMatrix,
    pub(super) q_norm: DeviceVec,
    pub(super) k_norm: DeviceVec,
    pub(super) q_dim: usize,
    pub(super) kv_dim: usize,
}

/// MLP layer weights.
///
/// Gate+Up stored as a single concatenated matrix `[2*intermediate_size, hidden_size]`.
pub(super) struct MLP {
    pub(super) gate_up_proj: DeviceMatrix,
    pub(super) down_proj: DeviceMatrix,
}

/// Transformer block.
pub(super) struct TransformerBlock {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attention: Attention,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: MLP,
}

/// Pre-allocated scratch buffers for one prefill forward pass.
/// Created once per forward, eliminating per-layer allocation overhead.
pub(super) struct PrefillBuffers {
    pub(super) hidden_out: HiddenStates,  // hidden_dim × seq_len
    pub(super) normed: HiddenStates,      // hidden_dim × seq_len (reused for normed2)
    pub(super) q_batch: HiddenStates,     // q_dim × seq_len
    pub(super) k_batch: HiddenStates,     // kv_dim × seq_len
    pub(super) v_batch: HiddenStates,     // kv_dim × seq_len
    pub(super) o_buf: HiddenStates,       // hidden_dim × seq_len (reused for mlp_out)
    pub(super) gate_out: HiddenStates,    // inter_dim × seq_len
    pub(super) up_out: HiddenStates,      // inter_dim × seq_len
    pub(super) act_out: HiddenStates,     // inter_dim × seq_len
    pub(super) attn_output: HiddenStates, // q_dim × seq_len
}

impl PrefillBuffers {
    fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, seq_len)?,
        })
    }
}

// ── HiggsBackbone ──────────────────────────────────────────────────────────

/// Higgs backbone model — weights and config only.
///
/// Holds GPU-resident weights for embed_tokens, 36 transformer layers,
/// final RMSNorm, and precomputed RoPE cache.
pub struct HiggsBackbone {
    pub(super) ctx: DeviceContext,
    pub(super) config: crate::config::TextConfig,
    pub(super) embed_tokens: DeviceMatrix,
    pub(super) layers: Vec<TransformerBlock>,
    pub(super) norm: DeviceVec,
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
}

// SAFETY: Each backbone instance is pinned to a single CUDA device and is only
// driven from one thread at a time.
unsafe impl Send for HiggsBackbone {}
unsafe impl Sync for HiggsBackbone {}

impl HiggsBackbone {
    /// Load Higgs backbone weights from safetensors.
    ///
    /// Reads Higgs `config.json` → extracts `text_config` for architecture
    /// params. Loads all `body.*` tensors (36 layers × 11 weights) and both
    /// tied embeddings. Precomputes RoPE cache on GPU.
    pub fn from_safetensors(model_path: &str, device_ordinal: usize) -> Result<Self> {
        info!("Loading Higgs backbone from: {}", model_path);
        debug!("Initializing GPU device {}", device_ordinal);
        let ctx = DeviceContext::new_with_device(device_ordinal)?;

        // Load Higgs config → extract text_config
        let model_path = std::path::Path::new(model_path);
        let higgs_config =
            HiggsConfig::from_path(model_path).context("failed to parse Higgs config.json")?;
        let config = higgs_config.text_config;

        // Validate architecture facts before loading weights
        anyhow::ensure!(
            config.num_hidden_layers == 36,
            "expected 36 hidden layers, got {}",
            config.num_hidden_layers
        );
        anyhow::ensure!(
            config.hidden_size == 2560,
            "expected hidden_size=2560, got {}",
            config.hidden_size
        );

        // Load safetensors
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model_path is not valid UTF-8"))?;
        let (shard_paths, weight_map) = load_shard_info(model_path_str)?;
        debug!("Loading {} safetensor shard(s)", shard_paths.len());
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let t_gpu = Instant::now();

        // Load tied embeddings (used for both embed_tokens and lm_head)
        debug!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(
            &ctx,
            &shards,
            &weight_map,
            "tied.embedding.text_embedding.weight",
        )
        .with_context(|| {
            "failed to load tied.embedding.text_embedding.weight — \
             is this a Higgs checkpoint?"
        })?;

        // Load 36 layers using `body.layers.{i}.{rest}` naming
        let num_layers = config.num_hidden_layers;
        debug!("Loading {} layers to GPU", num_layers);
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let prefix = format!("body.layers.{}", i);

            let q_proj = load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.self_attn.q_proj.weight", prefix),
            )?;
            let k_proj = load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.self_attn.k_proj.weight", prefix),
            )?;
            let v_proj = load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.self_attn.v_proj.weight", prefix),
            )?;
            let q_dim = q_proj.rows;
            let kv_dim = k_proj.rows;
            let qkv_proj = DeviceMatrix::vstack(&ctx, &[&q_proj, &k_proj, &v_proj])?;
            drop(q_proj);
            drop(k_proj);
            drop(v_proj);

            let gate_proj = load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.mlp.gate_proj.weight", prefix),
            )?;
            let up_proj = load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.mlp.up_proj.weight", prefix),
            )?;
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
                    o_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.o_proj.weight", prefix),
                    )?,
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
                    down_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.down_proj.weight", prefix),
                    )?,
                },
            };
            layers.push(block);
        }

        // Load final norm
        let norm = load_tensor_1d(&ctx, &shards, &weight_map, "body.norm.weight")
            .with_context(|| "failed to load body.norm.weight")?;

        // Precompute RoPE cache
        debug!("Precomputing RoPE cache on GPU");
        let rope_theta = config.rope_theta();
        let (cos_cache, sin_cache) = precompute_rope(
            &ctx,
            config.head_dim,
            config.max_position_embeddings,
            rope_theta,
        )?;

        ctx.sync()?;
        info!(
            "Higgs backbone loaded in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );

        Ok(Self {
            ctx,
            config,
            embed_tokens,
            layers,
            norm,
            cos_cache,
            sin_cache,
        })
    }

    /// Return the config used by this backbone (from `text_config`).
    pub fn config(&self) -> &crate::config::TextConfig {
        &self.config
    }

    /// Return the device context for external coordination (KV allocation etc.).
    pub fn device_ctx(&self) -> &DeviceContext {
        &self.ctx
    }

    // ── Forward pass ────────────────────────────────────────────────────

    /// Embed a batch of token IDs into hidden states `[hidden_dim, seq_len]`.
    pub(super) fn get_embeddings_batch(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        let seq_len = token_ids.len();
        let hidden_dim = self.config.hidden_size;

        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(token_ids)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let mut out = HiddenStates::zeros(&self.ctx, hidden_dim, seq_len)?;
        ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids_gpu, &mut out)?;

        Ok(out)
    }

    /// Process a single transformer layer (per-token prefill).
    ///
    /// Steps: RMSNorm → QKV projections → paged attention (norm+RoPE+attend) →
    /// O projection → fused residual+RMSNorm → MLP (gate/up → silu_mul → down) →
    /// residual add.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
        plan: &PrefillPagedPlan,
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        // 1. RMSNorm → bufs.normed
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        // 2. QKV projections from fused qkv_proj
        let q_dim = layer.attention.q_dim;
        let kv_dim = layer.attention.kv_dim;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            0,
            q_dim,
            &bufs.normed,
            &mut bufs.q_batch,
        );
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.v_batch,
        );

        // 3. Paged prefill: norm+RoPE → append K/V to paged → batch attention
        ops::prefill_attention_paged_into(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            kv_buffer,
            layout,
            layer_idx,
            plan,
            &mut bufs.attn_output,
            num_heads,
            num_kv_heads,
            head_dim,
            self.config.rms_norm_eps,
        )?;

        // 4. O projection → bufs.o_buf
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );

        // 5+6. Residual add + MLP RMSNorm (fused)
        openinfer_kernels::ops::fused_add_rms_norm_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        // 7. MLP: split gate/up GEMMs → silu_mul → down
        let inter_dim = self.config.intermediate_size;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            0,
            inter_dim,
            &bufs.normed,
            &mut bufs.gate_out,
        );
        ops::gemm_rows_into(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            inter_dim,
            inter_dim,
            &bufs.normed,
            &mut bufs.up_out,
        );
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );

        // 8. Residual add: hidden + mlp_out → bufs.hidden_out
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }

    /// Process all layers for a single prefill sequence.
    fn process_all_layers(
        &self,
        mut hidden: HiddenStates,
        layout: &KvLayout,
        kv_buffer: &CudaSlice<bf16>,
        plan: &PrefillPagedPlan,
    ) -> Result<HiddenStates> {
        let total_tokens = hidden.seq_len;
        let inter_dim = self.config.intermediate_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;

        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            total_tokens,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer(
                layer_idx,
                layer,
                &mut hidden,
                kv_buffer,
                layout,
                plan,
                &mut bufs,
            )?;
        }

        Ok(hidden)
    }

    /// Compute logits for all positions in hidden states: final RMSNorm + lm_head.
    ///
    /// Returns `HiddenStates` with shape `[vocab_size, seq_len]`.
    pub fn compute_all_position_logits(&self, hidden: &HiddenStates) -> Result<HiddenStates> {
        let mut normed = HiddenStates::zeros(&self.ctx, hidden.hidden_dim, hidden.seq_len)?;
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut normed,
        );
        ops::gemm(&self.ctx, &self.embed_tokens, &normed)
    }

    /// Extract last-token logits: gather the final column of `hidden`,
    /// apply final RMSNorm and lm_head. Returns `[vocab_size, 1]`.
    pub fn last_token_logits(&self, hidden: &HiddenStates) -> Result<HiddenStates> {
        let last_idx = (hidden.seq_len - 1) as i32;
        let indices_d = self
            .ctx
            .stream
            .clone_htod(&[last_idx])
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let mut gathered = HiddenStates::zeros(&self.ctx, hidden.hidden_dim, 1)?;
        ops::gather_hidden_tokens_into(&self.ctx, hidden, &indices_d, 1, &mut gathered)?;

        let mut normed = HiddenStates::zeros(&self.ctx, hidden.hidden_dim, 1)?;
        ops::rms_norm_batch_into(
            &self.ctx,
            &gathered,
            &self.norm,
            self.config.rms_norm_eps,
            &mut normed,
        );

        ops::gemm(&self.ctx, &self.embed_tokens, &normed)
    }

    /// Full prefill forward for a single sequence.
    ///
    /// Takes a `kv_buffer` (allocated by caller with enough pages for this
    /// sequence), a `KvLayout`, and a `PrefillPagedPlan`. Returns
    /// `(hidden_states, logits)` where `hidden_states` has shape
    /// `[hidden_dim, seq_len]` and `logits` has shape `[vocab_size, 1]`
    /// (last-token only).
    ///
    /// For "echo" mode (all-position logits), call
    /// `compute_all_position_logits` on the returned hidden states before
    /// calling `last_token_logits`.
    pub fn forward(
        &self,
        input_ids: &[u32],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
        plan: &PrefillPagedPlan,
    ) -> Result<HiddenStates> {
        let hidden = self.get_embeddings_batch(input_ids)?;
        self.process_all_layers(hidden, layout, kv_buffer, plan)
    }
}
