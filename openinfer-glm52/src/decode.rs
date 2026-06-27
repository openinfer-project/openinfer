//! GLM5.2 bs=1 decode forward for one pipeline stage.
//!
//! Composes the oracle-gated bricks into the per-stage decode step: embed (stage
//! 0) -> the stage's layers -> final norm + lm_head (last stage). Each layer is
//! the standard pre-norm residual block
//!   `r=h; h=input_ln(h); h=MLA(h); h=r+h; r=h; h=post_ln(h); h=MLP(h); h=r+h`,
//! with the MLP being the dense SwiGLU (layers 0..first_k_dense_replace) or the
//! routed+shared MoE block.
//!
//! The KV cache is the FlashMLA fp8_ds_mla paged cache, one per layer, written in
//! place at the current position each step. The DSA indexer is deferred (Slice 4):
//! at short context (<= MAX_CTX) the top-2048 select is all tokens, so `topk` is
//! just `[0,1,..,position, -1, -1, ...]` (fixed 2048, -1 padded -- the SM90 kernel
//! requires a static topk length).
//!
//! Buffers are allocated per call (wire-first). CUDA-graph capture needs a
//! pre-allocated arena; that is the next slice.

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE, Glm52FlashMlaSparseDecode, add_batch,
    glm52_flashmla_sparse_decode_num_sm_parts, rms_norm_batch_into,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceVec, HiddenStates};

use crate::bookend::{glm52_embed, glm52_final_norm, glm52_lm_head};
use crate::config::GLM52_HIDDEN;
use crate::dense::glm52_dense_mlp_forward;
use crate::mla_decode::glm52_mla_decode_forward;
use crate::model::{Glm52MlpModel, Glm52StageModel};
use crate::moe_decode::glm52_moe_forward;

const RMS_EPS: f32 = 1.0e-5;
const ROPE_THETA: f64 = 8_000_000.0;
const ROPE_HALF: usize = 32; // qk_rope_head_dim / 2 -> cos/sin table width
const SM_SCALE: f32 = 0.0625; // qk_head_dim^-0.5
const FIXED_TOPK: usize = 2048; // SM90 sparse kernel requires a static topk length

/// Max decode context (prompt + generated). `topk` is padded to 2048 and the
/// cache spans `ceil(MAX_CTX / PAGE_SIZE)` pages.
pub(crate) const GLM52_DECODE_MAX_CTX: usize = 2048;
const NUM_BLOCKS: usize = GLM52_DECODE_MAX_CTX.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE);
const CACHE_BYTES_PER_TOKEN: usize = 656; // fp8_ds_mla: 512 ckv + 16 scale + 128 k_pe
const CACHE_BYTES_PER_LAYER: usize =
    NUM_BLOCKS * GLM52_FLASHMLA_SPARSE_PAGE_SIZE * CACHE_BYTES_PER_TOKEN;

/// One pipeline stage's decode runtime: the typed weights plus the per-layer KV
/// caches, the precomputed rotary tables, and the reused topk buffer.
pub(crate) struct Glm52StageDecode {
    model: Glm52StageModel,
    caches: Vec<CudaSlice<u8>>,
    cos_table: CudaSlice<bf16>, // [MAX_CTX, 32]
    sin_table: CudaSlice<bf16>, // [MAX_CTX, 32]
    topk: CudaSlice<i32>,       // [2048], rebuilt per step
    contract: Glm52FlashMlaSparseDecode,
}

impl Glm52StageDecode {
    pub(crate) fn new(ctx: &DeviceContext, model: Glm52StageModel) -> Result<Self> {
        let num_layers = model.layers.len();
        let mut caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            caches.push(ctx.stream.alloc_zeros::<u8>(CACHE_BYTES_PER_LAYER)?);
        }
        let (cos_host, sin_host) = rope_tables();
        let mut cos_table = ctx.stream.alloc_zeros::<bf16>(cos_host.len())?;
        let mut sin_table = ctx.stream.alloc_zeros::<bf16>(sin_host.len())?;
        ctx.stream.memcpy_htod(&cos_host, &mut cos_table)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin_table)?;
        let topk = ctx.stream.alloc_zeros::<i32>(FIXED_TOPK)?;
        let num_sm_parts = glm52_flashmla_sparse_decode_num_sm_parts()?;
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: 1,
            num_blocks: NUM_BLOCKS,
            topk: FIXED_TOPK,
            num_sm_parts,
            sm_scale: SM_SCALE,
        };
        Ok(Self {
            model,
            caches,
            cos_table,
            sin_table,
            topk,
            contract,
        })
    }

    pub(crate) fn stage(&self) -> usize {
        self.model.stage
    }

    pub(crate) fn owns_head(&self) -> bool {
        self.model.lm_head.is_some()
    }

    /// Reset the KV caches for a fresh request (zero every page).
    pub(crate) fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        for cache in &mut self.caches {
            ctx.stream.memset_zeros(cache)?;
        }
        Ok(())
    }

    /// Embed a token id into the residual stream `[HIDDEN]` (stage 0 only).
    pub(crate) fn embed(&self, ctx: &DeviceContext, token_id: u32) -> Result<Vec<bf16>> {
        let embed = self.model.embed.as_ref().ok_or_else(|| {
            anyhow::anyhow!("GLM5.2 stage {} has no embedding table", self.stage())
        })?;
        let mut tid = ctx.stream.alloc_zeros::<u32>(1)?;
        ctx.stream.memcpy_htod(&[token_id], &mut tid)?;
        let hidden = glm52_embed(ctx, embed, &tid)?;
        ctx.stream.synchronize()?;
        Ok(ctx.stream.clone_dtoh(&hidden.data)?)
    }

    /// Run this stage's layers over `hidden_in[HIDDEN]` at `position`, returning the
    /// stage output hidden `[HIDDEN]` (host, for the next stage's input).
    pub(crate) fn run_layers(
        &mut self,
        ctx: &DeviceContext,
        hidden_in: &[bf16],
        position: usize,
    ) -> Result<Vec<bf16>> {
        let hidden = self.run_layers_device(ctx, hidden_in, position)?;
        ctx.stream.synchronize()?;
        Ok(ctx.stream.clone_dtoh(&hidden.data)?)
    }

    /// Run this stage's layers + final norm + lm_head (last stage only), returning
    /// the vocab logits `[VOCAB]` (host).
    pub(crate) fn run_layers_and_head(
        &mut self,
        ctx: &DeviceContext,
        hidden_in: &[bf16],
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.run_layers_device(ctx, hidden_in, position)?;
        let final_norm =
            self.model.final_norm.as_ref().ok_or_else(|| {
                anyhow::anyhow!("GLM5.2 stage {} has no final norm", self.stage())
            })?;
        let lm_head = self
            .model
            .lm_head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GLM5.2 stage {} has no lm_head", self.stage()))?;
        let hidden_dv = DeviceVec {
            data: hidden.data,
            len: GLM52_HIDDEN,
        };
        let normed = glm52_final_norm(ctx, &hidden_dv, final_norm)?;
        let logits = glm52_lm_head(ctx, &normed, lm_head)?;
        ctx.stream.synchronize()?;
        Ok(logits.to_host(ctx)?)
    }

    fn run_layers_device(
        &mut self,
        ctx: &DeviceContext,
        hidden_in: &[bf16],
        position: usize,
    ) -> Result<HiddenStates> {
        ensure!(
            hidden_in.len() == GLM52_HIDDEN,
            "GLM5.2 stage {} hidden input len {} != {GLM52_HIDDEN}",
            self.stage(),
            hidden_in.len()
        );
        ensure!(
            position < GLM52_DECODE_MAX_CTX,
            "GLM5.2 decode position {position} exceeds MAX_CTX {GLM52_DECODE_MAX_CTX}"
        );
        self.write_topk(ctx, position)?;

        let mut data = ctx.stream.alloc_zeros::<bf16>(GLM52_HIDDEN)?;
        ctx.stream.memcpy_htod(hidden_in, &mut data)?;
        let mut hidden = HiddenStates {
            data,
            hidden_dim: GLM52_HIDDEN,
            seq_len: 1,
        };

        // Disjoint borrows: the layer loop needs &mut cache[i] together with the
        // shared rope tables / topk / contract.
        let Glm52StageDecode {
            model,
            caches,
            cos_table,
            sin_table,
            topk,
            contract,
        } = self;
        let cos = cos_table.slice(position * ROPE_HALF..(position + 1) * ROPE_HALF);
        let sin = sin_table.slice(position * ROPE_HALF..(position + 1) * ROPE_HALF);
        // glm52_mla_decode_forward wants owned `&CudaSlice` rope rows; copy the
        // position's row out once (64 bytes), reused across this stage's layers.
        let mut cos_row = ctx.stream.alloc_zeros::<bf16>(ROPE_HALF)?;
        let mut sin_row = ctx.stream.alloc_zeros::<bf16>(ROPE_HALF)?;
        ctx.stream.memcpy_dtod(&cos, &mut cos_row)?;
        ctx.stream.memcpy_dtod(&sin, &mut sin_row)?;

        for (layer, cache) in model.layers.iter().zip(caches.iter_mut()) {
            hidden = layer_forward(
                ctx, layer, hidden, cache, position, &cos_row, &sin_row, topk, *contract,
            )?;
        }
        Ok(hidden)
    }

    fn write_topk(&mut self, ctx: &DeviceContext, position: usize) -> Result<()> {
        // Dense self-attention over all cached tokens 0..=position; -1 pads the
        // fixed 2048 slots (the SM90 kernel skips -1 indices).
        let mut host = vec![-1i32; FIXED_TOPK];
        for (idx, slot) in host.iter_mut().enumerate().take(position + 1) {
            *slot = idx as i32;
        }
        ctx.stream.memcpy_htod(&host, &mut self.topk)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn layer_forward(
    ctx: &DeviceContext,
    layer: &crate::model::Glm52LayerModel,
    hidden: HiddenStates,
    cache: &mut CudaSlice<u8>,
    position: usize,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    topk: &CudaSlice<i32>,
    contract: Glm52FlashMlaSparseDecode,
) -> Result<HiddenStates> {
    // --- attention sub-block: r + MLA(input_ln(r)) ---
    let mut normed = HiddenStates::zeros(ctx, GLM52_HIDDEN, 1)?;
    rms_norm_batch_into(ctx, &hidden, &layer.input_layernorm, RMS_EPS, &mut normed);
    let attn = glm52_mla_decode_forward(
        ctx,
        &layer.mla,
        &normed.data,
        cos,
        sin,
        cache,
        position,
        topk,
        contract,
    )?;
    let attn = HiddenStates {
        data: attn,
        hidden_dim: GLM52_HIDDEN,
        seq_len: 1,
    };
    let hidden = add_batch(ctx, &hidden, &attn)?;

    // --- MLP sub-block: r + MLP(post_attn_ln(r)) ---
    let mut normed = HiddenStates::zeros(ctx, GLM52_HIDDEN, 1)?;
    rms_norm_batch_into(
        ctx,
        &hidden,
        &layer.post_attention_layernorm,
        RMS_EPS,
        &mut normed,
    );
    let mlp = match &layer.mlp {
        Glm52MlpModel::Dense(dense) => glm52_dense_mlp_forward(ctx, dense, &normed.data)?,
        Glm52MlpModel::Moe(moe) => glm52_moe_forward(ctx, moe, &normed.data)?,
    };
    let mlp = HiddenStates {
        data: mlp,
        hidden_dim: GLM52_HIDDEN,
        seq_len: 1,
    };
    Ok(add_batch(ctx, &hidden, &mlp)?)
}

/// Precompute the rotary `cos`/`sin` tables `[MAX_CTX, 32]` (bf16). GLM5.2 default
/// rope: `inv_freq[i] = theta^(-i/32)` (theta = 8e6), `angle = position * inv_freq`.
fn rope_tables() -> (Vec<bf16>, Vec<bf16>) {
    let mut cos = vec![bf16::from_f32(0.0); GLM52_DECODE_MAX_CTX * ROPE_HALF];
    let mut sin = vec![bf16::from_f32(0.0); GLM52_DECODE_MAX_CTX * ROPE_HALF];
    for pos in 0..GLM52_DECODE_MAX_CTX {
        for i in 0..ROPE_HALF {
            let inv = ROPE_THETA.powf(-(i as f64) / ROPE_HALF as f64);
            let angle = pos as f64 * inv;
            cos[pos * ROPE_HALF + i] = bf16::from_f64(angle.cos());
            sin[pos * ROPE_HALF + i] = bf16::from_f64(angle.sin());
        }
    }
    (cos, sin)
}
