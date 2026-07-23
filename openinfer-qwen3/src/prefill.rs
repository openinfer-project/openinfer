use std::any::Any;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_core::kv_pool::KvLayout;
use openinfer_core::ops;
use openinfer_core::ops::PrefillPagedPlan;

use super::config::PREFILL_ATTENTION_CTA_TILE_Q;
use super::weights::Qwen3Model;
use super::weights::TransformerBlock;
use crate::lora::DeviceLoraTokenGroup;
use crate::lora::build_lora_token_ranges;
use crate::lora::prepare_lora_token_groups;

// Thread-local deferred-drop queue for decode-overlap mode. Buffers pushed here
// during prefill (under stream override) are dropped later when
// `drain_deferred_drops()` runs after the prefill stream is synchronized.
thread_local! {
    static DEFERRED_DROPS: std::cell::RefCell<Vec<Box<dyn Any>>> = std::cell::RefCell::new(Vec::new());
}

/// Defer an object's drop until `drain_deferred_drops()` is called.
fn defer_drop<T: 'static>(val: T) {
    DEFERRED_DROPS.with(|q| q.borrow_mut().push(Box::new(val)));
}

/// Drop all deferred objects. Call after prefill stream sync.
pub(crate) fn drain_deferred_drops() {
    DEFERRED_DROPS.with(|q| q.borrow_mut().clear());
}
use openinfer_core::tensor::DeviceContext;
use openinfer_core::tensor::HiddenStates;
use openinfer_kv_cache::KvView;

/// Pre-allocated scratch buffers for one prefill forward pass.
/// Created once per prefill pass, eliminating
/// per-layer `cuMemAllocAsync` overhead (~11k calls / 88ms at seq=2048).
///
/// Buffer reuse across steps (all kernels serialized on a single stream):
///   `normed`  reused for `normed2`  (steps 1-4 done before step 8)
///   `o_buf`   reused for `mlp_out`  (step 7 done before step 12)
pub(super) struct PrefillBuffers {
    /// Output ping-pong: layer writes result here; caller swaps with the incoming hidden.
    pub(super) hidden_out: HiddenStates, // hidden_dim × seq_len
    pub(super) normed: HiddenStates, // hidden_dim × seq_len (reused for normed2)
    pub(super) q_batch: HiddenStates, // q_dim × seq_len
    pub(super) k_batch: HiddenStates, // kv_dim × seq_len
    pub(super) v_batch: HiddenStates, // kv_dim × seq_len
    pub(super) o_buf: HiddenStates,  // hidden_dim × seq_len (reused for mlp_out)
    pub(super) gate_out: HiddenStates, // inter_dim × seq_len
    pub(super) up_out: HiddenStates, // inter_dim × seq_len
    pub(super) act_out: HiddenStates, // inter_dim × seq_len
    pub(super) attn_output: HiddenStates, // q_dim × seq_len
}

impl PrefillBuffers {
    pub(super) fn new(
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

    /// Point every scratch buffer's logical row count at `rows` without
    /// reallocating. Used by the fixed-buffer verify path (see
    /// [`crate::verify_graph`]); the buffers must have been allocated for at
    /// least `rows`.
    pub(super) fn set_rows(&mut self, rows: usize) {
        self.hidden_out.seq_len = rows;
        self.normed.seq_len = rows;
        self.q_batch.seq_len = rows;
        self.k_batch.seq_len = rows;
        self.v_batch.seq_len = rows;
        self.o_buf.seq_len = rows;
        self.gate_out.seq_len = rows;
        self.up_out.seq_len = rows;
        self.act_out.seq_len = rows;
        self.attn_output.seq_len = rows;
    }
}

impl Qwen3Model {
    #[fastrace::trace(name = "get_embeddings_batch")]
    pub(super) fn get_embeddings_batch(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        let seq_len = token_ids.len();
        let hidden_dim = self.config.hidden_size;

        // Copy token IDs to GPU
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(token_ids)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let mut out = HiddenStates::zeros(&self.ctx, hidden_dim, seq_len)?;
        ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids_gpu, &mut out)?;

        // Defer drop of token_ids_gpu in SM-partition mode to prevent
        // use-after-free (allocated on ctx.stream, kernel on green stream).
        if openinfer_kernels::tensor::has_stream_override() {
            defer_drop(token_ids_gpu);
        }

        Ok(out)
    }

    /// Embed a device-resident token buffer into a pre-allocated output, with
    /// no host round-trip or allocation — used by the DFlash draft rollout's
    /// graph-stable scratch.
    pub(super) fn get_embeddings_batch_into(
        &self,
        token_ids_gpu: &cudarc::driver::CudaSlice<u32>,
        out: &mut HiddenStates,
    ) -> Result<()> {
        anyhow::ensure!(
            out.hidden_dim == self.config.hidden_size,
            "embedding output hidden_dim {} does not match model hidden_size {}",
            out.hidden_dim,
            self.config.hidden_size
        );
        ops::embedding_batch(&self.ctx, &self.embed_tokens, token_ids_gpu, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_batch_paged(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &openinfer_core::kv_pool::KvLayout,
        plan: &PrefillPagedPlan,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        self.forward_layer_pre_attn(layer_idx, layer, hidden, lora_groups, bufs)?;
        self.forward_layer_attn(layer_idx, layer, kv_buffer, layout, plan, bufs)?;
        self.forward_layer_post_attn(layer_idx, layer, hidden, lora_groups, bufs)?;
        Ok(())
    }

    /// Pre-attention dense ops: input RMSNorm + fused QKV projections (+ LoRA).
    /// Reads `hidden`; writes `bufs.normed` / `bufs.q_batch` / `bufs.k_batch` /
    /// `bufs.v_batch`. Graph-safe — shapes depend only on the fixed row count, not
    /// on KV length — so the verify piecewise CUDA Graph captures it.
    pub(crate) fn forward_layer_pre_attn(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &HiddenStates,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
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
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.q_proj.as_ref(),
            &bufs.normed,
            &mut bufs.q_batch,
            0,
        )?;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.k_proj.as_ref(),
            &bufs.normed,
            &mut bufs.k_batch,
            0,
        )?;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.v_batch,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.v_proj.as_ref(),
            &bufs.normed,
            &mut bufs.v_batch,
            0,
        )?;
        Ok(())
    }

    /// The attention op: q/k norm + RoPE + paged KV append + paged attention.
    /// This is the ONLY part of the layer whose KV iteration count tracks the
    /// (growing) context length, so the verify piecewise graph keeps it EAGER —
    /// capturing it would freeze the KV length at capture time (`num_iterations`
    /// in FlashInfer's prefill kernel is fixed when the graph is recorded).
    pub(crate) fn forward_layer_attn(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &openinfer_core::kv_pool::KvLayout,
        plan: &PrefillPagedPlan,
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        let num_heads = self.local_num_attention_heads();
        let num_kv_heads = self.local_num_key_value_heads();
        let head_dim = self.config.head_dim;

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
        Ok(())
    }

    /// Post-attention dense ops: O projection + residual + MLP + final residual
    /// add. Reads `bufs.attn_output` and `hidden`; writes the layer output back
    /// into `hidden` via the ping-pong buffer swap. Graph-safe (no KV-length
    /// dependence) — captured into the verify piecewise CUDA Graph.
    pub(crate) fn forward_layer_post_attn(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        // 4. O projection → bufs.o_buf (as o_batch)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.o_proj.as_ref(),
            &bufs.attn_output,
            &mut bufs.o_buf,
            0,
        )?;
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // 5+6. Residual add + MLP RMSNorm (fused): hidden += o_buf; normed = rms_norm(hidden)
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        // 7. MLP: split gate/up GEMMs → silu_mul → down → bufs.o_buf
        let inter_dim = self.local_intermediate_size();
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
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.gate_proj.as_ref(),
            &bufs.normed,
            &mut bufs.gate_out,
            0,
        )?;
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.up_proj.as_ref(),
            &bufs.normed,
            &mut bufs.up_out,
            0,
        )?;
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.down_proj.as_ref(),
            &bufs.act_out,
            &mut bufs.o_buf,
            0,
        )?;
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // 8. Residual add: attn_residual + mlp_out → bufs.hidden_out (old hidden_in, free to overwrite)
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        // Swap: hidden = layer output, bufs.hidden_out = attn_residual (free next layer)
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }

    // ── Batch prefill ──────────────────────────────────────────────────

    /// Batch prefill: process multiple prompts in a single forward pass.
    ///
    /// Compute logits for ALL positions in the hidden states.
    ///
    /// Used when `echo=true` to return prompt token log-probabilities.
    /// Applies final RMS norm + lm_head projection in a single batched GEMM.
    /// Returns `HiddenStates` with shape `[vocab_size, total_tokens]`.
    fn compute_all_position_logits(&self, hidden: &HiddenStates) -> Result<HiddenStates> {
        let mut normed = HiddenStates::zeros(&self.ctx, hidden.hidden_dim, hidden.seq_len)?;
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut normed,
        );
        ops::gemm(&self.ctx, self.output_projection(), &normed)
    }

    /// Batched last-token logits: gather the given token columns out of
    /// `hidden`, then apply final RMSNorm and lm_head as single batched ops.
    /// Returns `HiddenStates [vocab_size, n]`, one column per index.
    pub(crate) fn batch_token_logits(
        &self,
        hidden: &HiddenStates,
        token_indices: &[i32],
    ) -> Result<HiddenStates> {
        let n = token_indices.len();
        let indices_d = self.ctx.stream.clone_htod(token_indices)?;
        let mut gathered = HiddenStates::zeros(&self.ctx, hidden.hidden_dim, n)?;
        ops::gather_hidden_tokens_into(&self.ctx, hidden, &indices_d, n, &mut gathered)?;
        let mut normed = HiddenStates::zeros(&self.ctx, hidden.hidden_dim, n)?;
        ops::rms_norm_batch_into(
            &self.ctx,
            &gathered,
            &self.norm,
            self.config.rms_norm_eps,
            &mut normed,
        );
        ops::gemm(&self.ctx, self.output_projection(), &normed)
    }

    /// Concatenates all prompts' tokens, runs one GEMM per layer for the
    /// entire batch, and uses FlashInfer's multi-request causal attention.
    /// Returns batched last-token logits `[vocab_size, batch]`, one column
    /// per request.
    ///
    /// If `echo` is true, also returns all-position logits as a
    /// `HiddenStates [vocab_size, total_tokens]` for prompt logprobs.
    /// Batch prefill forward.
    ///
    /// `capture_layer_ids`, when set, copies the residual-stream hidden states
    /// after the listed (strictly increasing) transformer layers into an extra
    /// `[hidden_size * layers, total_tokens]` buffer returned as the third tuple
    /// element. This feeds the DFlash draft model its target context; `None`
    /// behaves identically to a plain prefill and returns `None` there.
    pub(crate) fn batch_prefill(
        &self,
        prompts: &[&[u32]],
        kv_views: &[KvView],
        lora_adapters: &[Option<&str>],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
        echo: bool,
        capture_layer_ids: Option<&[usize]>,
    ) -> Result<(HiddenStates, Option<HiddenStates>, Option<HiddenStates>)> {
        let batch_size = prompts.len();
        assert_eq!(batch_size, kv_views.len());
        assert_eq!(batch_size, lora_adapters.len());

        let seq_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let lora_ranges =
            build_lora_token_ranges(seq_lens.iter().copied(), lora_adapters.iter().copied());
        let lora_groups = prepare_lora_token_groups(&self.ctx, &lora_ranges)?;
        let start_positions: Vec<usize> = kv_views
            .iter()
            .zip(prompts.iter())
            .map(|(v, p)| v.seq_len() - p.len())
            .collect();

        // Concatenate all tokens
        let all_tokens: Vec<u32> = prompts.iter().flat_map(|p| p.iter().copied()).collect();
        let hidden = self.get_embeddings_batch(&all_tokens)?;

        // Build batch plan from views
        let page_indices: Vec<Vec<i32>> =
            kv_views.iter().map(|v| v.page_indices().to_vec()).collect();
        let last_page_lens: Vec<usize> = kv_views
            .iter()
            .map(openinfer_kv_cache::KvView::last_page_len)
            .collect();
        let plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            &self.ctx,
            &page_indices,
            &last_page_lens,
            &start_positions,
            &seq_lens,
            self.local_num_attention_heads(),
            self.local_num_key_value_heads(),
            self.config.head_dim,
            PREFILL_ATTENTION_CTA_TILE_Q,
        )?;

        // Forward through all layers
        let (hidden, captured_hidden) = self.process_all_layers_batch_multi(
            hidden,
            layout,
            kv_buffer,
            &plan,
            &lora_groups,
            capture_layer_ids,
        )?;

        // All-position logits for echo (before we extract last-token logits)
        let all_logits = if echo {
            Some(self.compute_all_position_logits(&hidden)?)
        } else {
            None
        };

        // Batched last-token logits (one lm_head GEMM for the whole batch)
        let mut last_indices = Vec::with_capacity(batch_size);
        let mut offset = 0usize;
        for &seq_len in &seq_lens {
            last_indices.push((offset + seq_len - 1) as i32);
            offset += seq_len;
        }
        let logits = self.batch_token_logits(&hidden, &last_indices)?;

        // In SM-partition mode (stream override active), defer dropping
        // GPU-backed temp buffers until after the prefill stream is synced.
        // Otherwise cuMemFreeAsync on ctx.stream races with green-stream kernels.
        if openinfer_kernels::tensor::has_stream_override() {
            defer_drop(hidden);
            defer_drop(plan);
        }

        Ok((logits, all_logits, captured_hidden))
    }

    fn process_all_layers_batch_multi(
        &self,
        mut hidden: HiddenStates,
        layout: &KvLayout,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        plan: &PrefillPagedPlan,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        capture_layer_ids: Option<&[usize]>,
    ) -> Result<(HiddenStates, Option<HiddenStates>)> {
        let total_tokens = hidden.seq_len;
        let inter_dim = self.local_intermediate_size();
        let q_dim = self.local_q_dim();
        let kv_dim = self.local_kv_dim();

        let capture_layer_ids = capture_layer_ids.unwrap_or(&[]);
        anyhow::ensure!(
            capture_layer_ids.windows(2).all(|pair| pair[0] < pair[1]),
            "target hidden capture layer ids must be strictly increasing"
        );
        anyhow::ensure!(
            capture_layer_ids
                .iter()
                .all(|&layer| layer < self.layers.len()),
            "target hidden capture layer id out of range"
        );
        let mut captured_hidden = if capture_layer_ids.is_empty() {
            None
        } else {
            Some(HiddenStates::zeros(
                &self.ctx,
                self.config.hidden_size * capture_layer_ids.len(),
                total_tokens,
            )?)
        };
        let mut next_capture = 0usize;

        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            total_tokens,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch_paged(
                layer_idx,
                layer,
                &mut hidden,
                kv_buffer,
                layout,
                plan,
                lora_groups,
                &mut bufs,
            )?;
            if capture_layer_ids.get(next_capture) == Some(&layer_idx) {
                let out = captured_hidden
                    .as_mut()
                    .expect("capture buffer exists when ids are non-empty");
                ops::copy_hidden_rows_into(
                    &self.ctx,
                    &hidden,
                    out,
                    next_capture * self.config.hidden_size,
                )?;
                next_capture += 1;
            }
        }

        // Defer drop of PrefillBuffers in SM-partition mode.
        if openinfer_kernels::tensor::has_stream_override() {
            defer_drop(bufs);
        }

        Ok((hidden, captured_hidden))
    }
}
