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
use openinfer_kernels::ops::rms_norm_rows_into;
use openinfer_kernels::ops::{add_into, fused_add_rms_norm_round_into};
use openinfer_kernels::tensor::{DeviceContext, DeviceVec};

use crate::config::{GLM52_HIDDEN, GLM52_RMS_EPS as RMS_EPS};
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
use crate::rows::Rows;
use crate::scratch::Glm52DecodeScratch;

const HIDDEN: usize = GLM52_HIDDEN;

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
    /// TP8 topology: the router is the only per-layer MLP weight here — the
    /// routed experts AND the shared expert live in the rank's slice bank
    /// (`Glm52MoeTp8Rank.slices`, shared folded at bank index 256).
    MoeTp8(Box<crate::moe_decode::Glm52MoeRouterWeights>),
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
    pub(crate) slot_mapping: &'a CudaSlice<i64>,
    pub(crate) block_table: &'a CudaSlice<i32>,
    pub(crate) seq_lens: &'a CudaSlice<i32>,
}

/// Persistent per-layer composition scratch: the residual-stream boundary
/// buffers shared by all 78 layers (layer N's values are dead once layer N's
/// closing add consumed them), sized for the step's `tokens` rows.
pub(crate) struct Glm52LayerScratch {
    /// input_layernorm output — the MLA/indexer input. Written by the
    /// PREVIOUS layer's closing fused add+norm (layer 0's comes from a
    /// standalone norm of the embedding).
    pub(crate) normed: Rows<GLM52_HIDDEN>,
    /// attention outputs, ping-ponged by layer parity: after layer L's
    /// closing fused add, `attn[L % 2]` carries the residual stream INTO
    /// layer L+1 while layer L+1's attention writes `attn[(L + 1) % 2]`.
    pub(crate) attn: [Rows<GLM52_HIDDEN>; 2],
    /// post_attention_layernorm output — the MLP input.
    pub(crate) normed2: Rows<GLM52_HIDDEN>,
    /// the MLP half's final contribution (dense out, or routed+shared sum).
    pub(crate) mlp_out: Rows<GLM52_HIDDEN>,
    /// head-sharded o_proj partial, before the attention-TP all-reduce
    /// (unused when the layer holds all 64 heads).
    pub(crate) ar_partial: Rows<GLM52_HIDDEN>,
    /// MoE shared-expert contribution.
    pub(crate) shared_out: Rows<GLM52_HIDDEN>,
}

impl Glm52LayerScratch {
    pub(crate) fn new(ctx: &DeviceContext, tokens: usize) -> Result<Self> {
        Ok(Self {
            normed: Rows::zeros(ctx, tokens)?,
            attn: [Rows::zeros(ctx, tokens)?, Rows::zeros(ctx, tokens)?],
            normed2: Rows::zeros(ctx, tokens)?,
            mlp_out: Rows::zeros(ctx, tokens)?,
            ar_partial: Rows::zeros(ctx, tokens)?,
            shared_out: Rows::zeros(ctx, tokens)?,
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
#[allow(clippy::too_many_arguments)]
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
    tp8_ar: Option<(&mut crate::moe_tp8::Glm52MoeTp8State, usize)>,
) -> Result<()> {
    // Attention-TP: a head-sharded layer (8 of 64 heads) produces an o_proj
    // PARTIAL that must cross the AR brick before the residual add; holding
    // full heads with AR wiring (or a shard without it) is a build bug —
    // crash here, not on silently-wrong hidden states.
    let sharded = w.mla.heads != crate::config::GLM52_HEADS;
    ensure!(
        sharded == tp8_ar.is_some(),
        "GLM5.2 attention-TP wiring mismatch: layer holds {} heads but AR is {}",
        w.mla.heads,
        if tp8_ar.is_some() { "wired" } else { "absent" }
    );
    // `s.layer.normed` (this layer's input_layernorm output) is already
    // populated: by the previous layer's closing fused add+norm, or — for
    // the first layer — by the caller's standalone norm of the embedding.
    //
    // On full-indexer layers with an aux stream, the DSA indexer chain runs
    // concurrently with the rest of the MLA front: the indexer only needs
    // `normed` + `q_resid` (the q-phase), while q_b/kv_a are independent of
    // it. Same kernels either way — byte-identical; the fork/join events
    // become graph edges at capture.
    // The scratch buffers and the attend plan were all built from one row
    // count (`Glm52DecodeScratch::new` / `Glm52BucketState`), so the batch is
    // read from the plan without re-validating the buffers against it.
    let tokens = step.mla_sched.batch();
    glm52_mla_front_q_into(ctx, &w.mla, &s.layer.normed, &mut s.mla_front)?;
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
                &s.layer.normed,
                &s.mla_front.q_resid,
                step.idx_cos,
                step.idx_sin,
                index_k_cache,
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
    glm52_mla_front_rest_into(ctx, &w.mla, &s.layer.normed, &mut s.mla_front)?;
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
    match tp8_ar {
        Some((tp8, layer_slot)) => {
            // Sharded: attend lands the o_proj partial in `ar_partial`; the
            // AR brick sums the 8 ranks' partials into `attn_out`
            // (bit-identical on every rank, fixed source order).
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
                &mut s.layer.ar_partial,
            )?;
            tp8.attn_ar_launch(
                ctx,
                layer_slot,
                tokens,
                s.layer.ar_partial.data(),
                attn_out.data_mut(),
            )?;
        }
        None => glm52_mla_attend_into(
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
        )?,
    }

    // Fused add+norm at the post-attention boundary (bit-identical to separate
    // add + rms_norm — the `_round` variant rounds the sum to bf16 before the
    // variance, exactly like the plain add would). The residual stream enters
    // in the OTHER parity's attn buffer (written by the previous layer's
    // closing fused add), or in `s.hidden` for the first layer (the embedding).
    let residual: &CudaSlice<bf16> = if first_layer {
        s.hidden.data()
    } else {
        attn_other.data()
    };
    fused_add_rms_norm_round_into(
        ctx,
        attn_out.data_mut(),
        residual,
        &w.post_attn_ln,
        RMS_EPS,
        HIDDEN,
        tokens,
        s.layer.normed2.data_mut(),
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
    let tokens = s.layer.normed.tokens();
    fused_add_rms_norm_round_into(
        ctx,
        s.layer.attn[parity].data_mut(),
        s.layer.mlp_out.data(),
        next_input_ln,
        RMS_EPS,
        HIDDEN,
        tokens,
        s.layer.normed.data_mut(),
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
        s.layer.attn[parity].data(),
        s.layer.mlp_out.data(),
        s.hidden.tokens() * HIDDEN,
        s.hidden.data_mut(),
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
    let tokens = s.hidden.tokens();
    rms_norm_rows_into(
        ctx,
        s.hidden.data(),
        &w.input_ln,
        RMS_EPS,
        HIDDEN,
        tokens,
        s.layer.normed.data_mut(),
    )?;
    glm52_layer_attention_half(ctx, None, w, caches, step, s, carry_ready, 0, true, None)?;
    match &w.mlp {
        Glm52LayerMlp::Dense(dense) => glm52_dense_mlp_forward_into(
            ctx,
            dense,
            s.layer.normed2.data(),
            &mut s.dense_mlp,
            s.layer.mlp_out.data_mut(),
        )?,
        Glm52LayerMlp::Moe(moe) => {
            // EP1 oracle path: the all-256-expert forward allocates internally;
            // relay its output into the layer scratch.
            let mlp = glm52_moe_forward(ctx, moe, s.layer.normed2.data(), moe_path)?;
            ctx.stream.memcpy_dtod(&mlp, s.layer.mlp_out.data_mut())?;
        }
        Glm52LayerMlp::MoeEp8(_) | Glm52LayerMlp::MoeTp8(_) => anyhow::bail!(
            "GLM5.2 EP8/TP8 MoE layers require their collective drivers, not the single-layer forward"
        ),
    }
    glm52_layer_finish(ctx, s, 0)
}
