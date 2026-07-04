//! GLM5.2 decoder-layer composition for row-batched decode: two-norm residual
//! layout around the MLA/DSA attention and the dense-or-MoE MLP. Every buffer
//! carries the step's `tokens` independent rows.
//!
//! Per-layer math (vllm `DeepseekV2DecoderLayer`, verified for `glm_moe_dsa`):
//!
//! ```text
//! residual = hidden
//! x = rms_norm(hidden, input_layernorm)
//! attn = MLA(x)                       # + DSA indexer or shared top-k
//! hidden = residual + attn
//! residual = hidden
//! x = rms_norm(hidden, post_attention_layernorm)
//! mlp = dense_mlp(x) | moe(x)
//! hidden = residual + mlp
//! ```
//!
//! Cross-layer top-k sharing (the GLM5.2 divergence from DSv3.2): only `full`
//! layers own indexer weights and compute a fresh top-k; `shared` layers reuse
//! the previous full layer's `topk_indices` verbatim. That reuse is sound
//! because the indices are global KV slots and every layer shares one block
//! table / slot mapping. The carry is threaded through `topk_carry`: a full
//! layer overwrites it, a shared layer requires it.

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
#[cfg(test)]
use openinfer_kernels::ops::rms_norm_into;
use openinfer_kernels::ops::{Glm52IndexerCacheLayout, add_into, fused_add_rms_norm_round_into};
use openinfer_kernels::tensor::{DeviceContext, DeviceVec};

use crate::dense::Glm52DenseMlpWeights;
#[cfg(test)]
use crate::dense::glm52_dense_mlp_forward_into;
use crate::indexer::{Glm52IndexerLayerWeights, glm52_indexer_forward_into};
use crate::mla_decode::{
    Glm52MlaLayerWeights, Glm52MlaSchedMetadata, glm52_mla_attend_into, glm52_mla_front_q_into,
    glm52_mla_front_rest_into,
};
#[cfg(test)]
use crate::moe_decode::Glm52MoeLayerWeights;
#[cfg(test)]
use crate::moe_decode::{Glm52MoeExpertPath, glm52_moe_forward};
use crate::moe_ep8::Glm52MoeEp8LayerWeights;
use crate::scratch::Glm52DecodeScratch;

const HIDDEN: usize = 6144;
pub(crate) const GLM52_RMS_EPS: f32 = 1.0e-5;
const RMS_EPS: f32 = GLM52_RMS_EPS;

/// The MLP half of a decoder layer: dense (layers 0..first_k_dense_replace),
/// EP1 routed+shared MoE (all 256 experts local — the oracle-gate path), or
/// the EP8 rank-0 MoE (router + shared + this rank's 32 experts; the expert
/// compute itself runs through the collective driver in `moe_ep8`, so the
/// single-layer forward below rejects it). Boxed: layer weight structs are
/// built once and held in a 78-entry vec — the indirection is free, the enum
/// stays small.
pub(crate) enum Glm52LayerMlp {
    Dense(Box<Glm52DenseMlpWeights>),
    #[cfg(test)]
    Moe(Box<Glm52MoeLayerWeights>),
    MoeEp8(Box<Glm52MoeEp8LayerWeights>),
}

/// The DSA indexer role of a decoder layer (`config.indexer_types[layer]`):
/// `Full` owns indexer weights and computes a fresh top-k; `Shared` reuses the
/// previous full layer's top-k and has no indexer weights in the checkpoint.
pub(crate) enum Glm52LayerIndexer {
    Full(Box<Glm52IndexerLayerWeights>),
    Shared,
}

/// One decoder layer's weights, device-resident.
pub(crate) struct Glm52DecoderLayerWeights {
    pub(crate) input_ln: DeviceVec,     // bf16 [HIDDEN]
    pub(crate) post_attn_ln: DeviceVec, // bf16 [HIDDEN]
    pub(crate) mla: Glm52MlaLayerWeights,
    pub(crate) indexer: Glm52LayerIndexer,
    pub(crate) mlp: Glm52LayerMlp,
}

/// Per-layer mutable caches: the MLA fp8_ds_mla paged cache (656 B/token) and,
/// on full-indexer layers, the DeepGEMM-layout index-K cache.
pub(crate) struct Glm52LayerCaches {
    pub(crate) mla_cache: CudaSlice<u8>,
    pub(crate) index_k_cache: Option<CudaSlice<u8>>,
}

/// Everything one decode step shares across layers: the token position, the two
/// rotary tables (MLA interleaved; indexer half-split — different conventions,
/// same `[32]` cos/sin extent), and the paging plumbing common to every layer's
/// caches.
pub(crate) struct Glm52DecodeStep<'a> {
    pub(crate) mla_cos: &'a CudaSlice<bf16>,
    pub(crate) mla_sin: &'a CudaSlice<bf16>,
    pub(crate) idx_cos: &'a CudaSlice<bf16>,
    pub(crate) idx_sin: &'a CudaSlice<bf16>,
    /// FlashMLA contract + tile-scheduler plan — computed once (the plan only
    /// depends on batch size / SM parts), shared by every layer.
    pub(crate) mla_sched: &'a Glm52MlaSchedMetadata,
    pub(crate) index_cache_layout: Glm52IndexerCacheLayout,
    pub(crate) slot_mapping: &'a CudaSlice<i64>,
    pub(crate) block_table: &'a CudaSlice<i32>,
    pub(crate) seq_lens: &'a CudaSlice<i32>,
}

/// Persistent per-layer composition scratch: the residual-stream boundary
/// buffers shared by all 78 layers (layer N's values are dead once layer N's
/// closing add consumed them), sized for the step's `tokens` rows.
pub(crate) struct Glm52LayerScratch {
    pub(crate) tokens: usize,
    /// input_layernorm output — the MLA/indexer input. Written by the
    /// PREVIOUS layer's closing fused add+norm (layer 0's comes from a
    /// standalone norm of the embedding).
    pub(crate) normed: DeviceVec,
    /// attention outputs, ping-ponged by layer parity: after layer L's
    /// closing fused add, `attn[L % 2]` carries the residual stream INTO
    /// layer L+1 while layer L+1's attention writes `attn[(L + 1) % 2]`.
    pub(crate) attn: [CudaSlice<bf16>; 2],
    /// post_attention_layernorm output — the MLP input.
    pub(crate) normed2: CudaSlice<bf16>,
    /// the MLP half's final contribution (dense out, or routed+shared sum).
    pub(crate) mlp_out: CudaSlice<bf16>,
    /// MoE shared-expert contribution.
    pub(crate) shared_out: CudaSlice<bf16>,
}

impl Glm52LayerScratch {
    pub(crate) fn new(ctx: &DeviceContext, tokens: usize) -> Result<Self> {
        Ok(Self {
            tokens,
            normed: DeviceVec::zeros(ctx, tokens * HIDDEN)?,
            attn: [
                ctx.stream.alloc_zeros::<bf16>(tokens * HIDDEN)?,
                ctx.stream.alloc_zeros::<bf16>(tokens * HIDDEN)?,
            ],
            normed2: ctx.stream.alloc_zeros::<bf16>(tokens * HIDDEN)?,
            mlp_out: ctx.stream.alloc_zeros::<bf16>(tokens * HIDDEN)?,
            shared_out: ctx.stream.alloc_zeros::<bf16>(tokens * HIDDEN)?,
        })
    }
}

/// The attention half of one decoder layer for one token: input norm → MLA
/// front → DSA indexer (or shared top-k carry) → MLA attend → fused
/// add+post-attention-norm. The residual-stream input is `s.hidden`; the
/// results land in `s.layer.attn` (the carried residual) and `s.layer.normed2`
/// (the MLP input).
///
/// The cross-layer top-k carry lives in `s.idx.global_slots`: a `Full` layer
/// overwrites it, a `Shared` layer reuses it. `carry_ready` guards the read —
/// callers must pass a fresh `false` per step (layer 0 is always `Full`, so an
/// in-order full-stack walk refreshes the carry before any read, but a stale
/// buffer from a previous step must not be silently accepted if a walk started
/// at a `Shared` layer).
pub(crate) fn glm52_layer_attention_half(
    ctx: &DeviceContext,
    aux: Option<&DeviceContext>,
    w: &Glm52DecoderLayerWeights,
    caches: &mut Glm52LayerCaches,
    step: &Glm52DecodeStep<'_>,
    s: &mut Glm52DecodeScratch,
    carry_ready: &mut bool,
    parity: usize,
    first_layer: bool,
) -> Result<()> {
    // `s.layer.normed` (this layer's input_layernorm output) is already
    // populated: by the previous layer's closing fused add+norm, or — for
    // the first layer — by the caller's standalone norm of the embedding.
    //
    // On full-indexer layers with an aux stream, the DSA indexer chain runs
    // concurrently with the rest of the MLA front: the indexer only needs
    // `normed` + `q_resid` (the q-phase), while q_b/kv_a are independent of
    // it. Same kernels either way — byte-identical; the fork/join events
    // become graph edges at capture.
    let tokens = step.mla_sched.batch();
    ensure!(
        s.layer.tokens == tokens && s.mla_front.tokens == tokens,
        "GLM5.2 layer scratch rows {} / front rows {} != attend plan batch {tokens}",
        s.layer.tokens,
        s.mla_front.tokens
    );
    glm52_mla_front_q_into(ctx, &w.mla, &s.layer.normed.data, &mut s.mla_front)?;
    let mut topk_ready = None;
    match &w.indexer {
        Glm52LayerIndexer::Full(indexer) => {
            let index_k_cache = caches
                .index_k_cache
                .as_mut()
                .context("GLM5.2 full-indexer layer is missing its index-K cache")?;
            let idx_ctx = if let Some(aux) = aux {
                let q_ready = ctx.stream.record_event(None)?;
                aux.stream.wait(&q_ready)?;
                aux
            } else {
                ctx
            };
            glm52_indexer_forward_into(
                idx_ctx,
                indexer,
                &s.layer.normed.data,
                &s.mla_front.q_resid,
                step.idx_cos,
                step.idx_sin,
                index_k_cache,
                step.index_cache_layout,
                step.slot_mapping,
                step.block_table,
                step.seq_lens,
                step.mla_sched.topk(),
                &mut s.idx,
            )?;
            if let Some(aux) = aux {
                topk_ready = Some(aux.stream.record_event(None)?);
            }
            *carry_ready = true;
        }
        Glm52LayerIndexer::Shared => {
            ensure!(
                caches.index_k_cache.is_none(),
                "GLM5.2 shared-indexer layer unexpectedly owns an index-K cache"
            );
        }
    }
    ensure!(
        *carry_ready,
        "GLM5.2 shared-indexer layer reached before any full indexer ran"
    );
    glm52_mla_front_rest_into(ctx, &w.mla, &s.layer.normed.data, &mut s.mla_front)?;
    if let Some(topk_ready) = &topk_ready {
        // Join before the attend consumes `s.idx.global_slots`.
        ctx.stream.wait(topk_ready)?;
    }
    let (attn_lo, attn_hi) = s.layer.attn.split_at_mut(1);
    let (attn_out, attn_other) = if parity == 0 {
        (&mut attn_lo[0], &attn_hi[0])
    } else {
        (&mut attn_hi[0], &attn_lo[0])
    };
    glm52_mla_attend_into(
        ctx,
        &w.mla,
        &s.mla_front,
        step.mla_cos,
        step.mla_sin,
        &mut caches.mla_cache,
        step.slot_mapping,
        &s.idx.global_slots,
        step.mla_sched,
        &mut s.mla_attend,
        attn_out,
    )?;

    // Fused add+norm at the post-attention boundary (bit-identical to separate
    // add + rms_norm — the `_round` variant rounds the sum to bf16 before the
    // variance, exactly like the plain add would). The residual stream enters
    // in the OTHER parity's attn buffer (written by the previous layer's
    // closing fused add), or in `s.hidden` for the first layer (the embedding).
    let residual: &CudaSlice<bf16> = if first_layer {
        &s.hidden.data
    } else {
        attn_other
    };
    fused_add_rms_norm_round_into(
        ctx,
        attn_out,
        residual,
        &w.post_attn_ln,
        RMS_EPS,
        HIDDEN,
        tokens,
        &mut s.layer.normed2,
    )?;
    Ok(())
}

/// The layer's closing residual add, FUSED with the next layer's
/// input_layernorm (bit-identical to separate add + rms_norm, same `_round`
/// kernel as the mid-layer boundary): `attn[parity] += mlp_out` becomes the
/// residual stream into layer L+1, and `s.layer.normed` becomes L+1's
/// attention input.
pub(crate) fn glm52_layer_finish_fused(
    ctx: &DeviceContext,
    s: &mut Glm52DecodeScratch,
    parity: usize,
    next_input_ln: &DeviceVec,
) -> Result<()> {
    let tokens = s.layer.tokens;
    fused_add_rms_norm_round_into(
        ctx,
        &mut s.layer.attn[parity],
        &s.layer.mlp_out,
        next_input_ln,
        RMS_EPS,
        HIDDEN,
        tokens,
        &mut s.layer.normed.data,
    )
}

/// The LAST layer's closing residual add: `s.hidden = attn[parity] + mlp_out`
/// (the final norm consumes `s.hidden`).
pub(crate) fn glm52_layer_finish(
    ctx: &DeviceContext,
    s: &mut Glm52DecodeScratch,
    parity: usize,
) -> Result<()> {
    add_into(
        ctx,
        &s.layer.attn[parity],
        &s.layer.mlp_out,
        s.layer.tokens * HIDDEN,
        &mut s.hidden.data,
    )
}

/// One full decoder layer for one token (attention half + local MLP half),
/// `s.hidden` → `s.hidden`. EP8 rank MoE layers must go through the
/// collective driver in `moe_ep8` instead — this single-layer path fails
/// closed on them. Oracle-gate path (the production spine drives the halves
/// directly in `model.rs`).
#[cfg(test)]
pub(crate) fn glm52_decoder_layer_forward(
    ctx: &DeviceContext,
    w: &Glm52DecoderLayerWeights,
    caches: &mut Glm52LayerCaches,
    step: &Glm52DecodeStep<'_>,
    moe_path: Glm52MoeExpertPath,
    s: &mut Glm52DecodeScratch,
    carry_ready: &mut bool,
) -> Result<()> {
    // Oracle-gate walk: one layer per call, stream in `s.hidden` — standalone
    // input norm + fixed parity 0 (no cross-layer fusion in this unit).
    rms_norm_into(ctx, &s.hidden, &w.input_ln, RMS_EPS, &mut s.layer.normed)?;
    glm52_layer_attention_half(ctx, None, w, caches, step, s, carry_ready, 0, true)?;
    match &w.mlp {
        Glm52LayerMlp::Dense(dense) => glm52_dense_mlp_forward_into(
            ctx,
            dense,
            &s.layer.normed2,
            &mut s.dense_mlp,
            &mut s.layer.mlp_out,
        )?,
        Glm52LayerMlp::Moe(moe) => {
            // EP1 oracle path: the all-256-expert forward allocates internally;
            // relay its output into the layer scratch.
            let mlp = glm52_moe_forward(ctx, moe, &s.layer.normed2, moe_path)?;
            ctx.stream.memcpy_dtod(&mlp, &mut s.layer.mlp_out)?;
        }
        Glm52LayerMlp::MoeEp8(_) => anyhow::bail!(
            "GLM5.2 EP8 MoE layers require the collective driver (moe_ep8), not the single-layer forward"
        ),
    };
    glm52_layer_finish(ctx, s, 0)
}
