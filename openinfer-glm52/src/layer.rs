//! GLM5.2 decoder-layer composition for bs=1 decode: two-norm residual layout
//! around the MLA/DSA attention and the dense-or-MoE MLP.
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
use openinfer_kernels::ops::{
    Glm52FlashMlaSparseDecode, Glm52IndexerCacheLayout, add_batch,
    fused_add_rms_norm_round_batch_into, rms_norm_into,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceVec, HiddenStates};

use crate::dense::{Glm52DenseMlpWeights, glm52_dense_mlp_forward};
use crate::indexer::{Glm52IndexerLayerWeights, glm52_indexer_forward};
use crate::mla_decode::{Glm52MlaLayerWeights, glm52_mla_attend, glm52_mla_front};
use crate::moe_decode::{Glm52MoeExpertPath, Glm52MoeLayerWeights, glm52_moe_forward};

const HIDDEN: usize = 6144;
const RMS_EPS: f32 = 1.0e-5;

/// The MLP half of a decoder layer: dense (layers 0..first_k_dense_replace) or
/// routed+shared MoE. Boxed: layer weight structs are built once and held in a
/// 78-entry vec — the indirection is free, the enum stays small.
pub(crate) enum Glm52LayerMlp {
    Dense(Box<Glm52DenseMlpWeights>),
    Moe(Box<Glm52MoeLayerWeights>),
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
    pub(crate) position: usize,
    pub(crate) mla_cos: &'a CudaSlice<bf16>,
    pub(crate) mla_sin: &'a CudaSlice<bf16>,
    pub(crate) idx_cos: &'a CudaSlice<bf16>,
    pub(crate) idx_sin: &'a CudaSlice<bf16>,
    pub(crate) contract: Glm52FlashMlaSparseDecode,
    pub(crate) index_cache_layout: Glm52IndexerCacheLayout,
    pub(crate) slot_mapping: &'a CudaSlice<i64>,
    pub(crate) block_table: &'a CudaSlice<i32>,
    pub(crate) seq_lens: &'a CudaSlice<i32>,
    pub(crate) num_sms: usize,
    pub(crate) max_model_len: usize,
    pub(crate) moe_path: Glm52MoeExpertPath,
}

/// `rms_norm(residual, gamma)` without consuming the residual stream.
fn rms(ctx: &DeviceContext, residual: &DeviceVec, gamma: &DeviceVec) -> Result<CudaSlice<bf16>> {
    let mut out = DeviceVec::zeros(ctx, HIDDEN)?;
    rms_norm_into(ctx, residual, gamma, RMS_EPS, &mut out)?;
    Ok(out.data)
}

/// `residual + delta`, consuming both (the residual stream moves forward).
fn residual_add(
    ctx: &DeviceContext,
    residual: DeviceVec,
    delta: CudaSlice<bf16>,
) -> Result<DeviceVec> {
    let a = HiddenStates {
        data: residual.data,
        hidden_dim: HIDDEN,
        seq_len: 1,
    };
    let b = HiddenStates {
        data: delta,
        hidden_dim: HIDDEN,
        seq_len: 1,
    };
    Ok(DeviceVec {
        data: add_batch(ctx, &a, &b)?.data,
        len: HIDDEN,
    })
}

/// One decoder layer for one token. `hidden` is the residual-stream input
/// `[HIDDEN]` (consumed); returns the residual-stream output. `topk_carry`
/// threads the cross-layer top-k: a `Full` layer replaces it, a `Shared` layer
/// requires it.
///
/// The carry is only meaningful WITHIN one decode step: callers must pass a
/// fresh `None` per step (layer 0 is always `Full`, so an in-order full-stack
/// walk refreshes it before any read — but a `Some` left over from a previous
/// step would be silently accepted if a walk started at a `Shared` layer).
/// PR4's 78-layer spine should hold the carry in a step-scoped struct so that
/// misuse is unrepresentable.
pub(crate) fn glm52_decoder_layer_forward(
    ctx: &DeviceContext,
    w: &Glm52DecoderLayerWeights,
    caches: &mut Glm52LayerCaches,
    hidden: CudaSlice<bf16>,
    step: &Glm52DecodeStep<'_>,
    topk_carry: &mut Option<CudaSlice<i32>>,
) -> Result<CudaSlice<bf16>> {
    ensure!(hidden.len() >= HIDDEN, "GLM5.2 layer hidden too small");
    let residual = DeviceVec {
        data: hidden,
        len: HIDDEN,
    };

    // ---- attention half ----
    let normed = rms(ctx, &residual, &w.input_ln)?;
    let front = glm52_mla_front(ctx, &w.mla, &normed)?;
    match &w.indexer {
        Glm52LayerIndexer::Full(indexer) => {
            let index_k_cache = caches
                .index_k_cache
                .as_mut()
                .context("GLM5.2 full-indexer layer is missing its index-K cache")?;
            let topk = glm52_indexer_forward(
                ctx,
                indexer,
                &normed,
                &front.q_resid,
                step.idx_cos,
                step.idx_sin,
                index_k_cache,
                step.index_cache_layout,
                step.slot_mapping,
                step.block_table,
                step.seq_lens,
                step.num_sms,
                step.max_model_len,
            )?;
            *topk_carry = Some(topk);
        }
        Glm52LayerIndexer::Shared => {
            ensure!(
                caches.index_k_cache.is_none(),
                "GLM5.2 shared-indexer layer unexpectedly owns an index-K cache"
            );
        }
    }
    let topk = topk_carry
        .as_ref()
        .context("GLM5.2 shared-indexer layer reached before any full indexer ran")?;
    let attn = glm52_mla_attend(
        ctx,
        &w.mla,
        &front,
        step.mla_cos,
        step.mla_sin,
        &mut caches.mla_cache,
        step.position,
        topk,
        step.contract,
    )?;

    // ---- MLP half ----
    // Fused add+norm at the post-attention boundary (bit-identical to separate
    // add + rms_norm — the `_round` variant rounds the sum to bf16 before the
    // variance, exactly like add_batch would). The input_layernorm boundary
    // spans layers and stays unfused in this single-layer unit.
    let mut attn_hs = HiddenStates {
        data: attn,
        hidden_dim: HIDDEN,
        seq_len: 1,
    };
    let residual_hs = HiddenStates {
        data: residual.data,
        hidden_dim: HIDDEN,
        seq_len: 1,
    };
    let mut normed_hs = HiddenStates {
        data: ctx.stream.alloc_zeros::<bf16>(HIDDEN)?,
        hidden_dim: HIDDEN,
        seq_len: 1,
    };
    fused_add_rms_norm_round_batch_into(
        ctx,
        &mut attn_hs,
        &residual_hs,
        &w.post_attn_ln,
        RMS_EPS,
        &mut normed_hs,
    )?;
    let residual = DeviceVec {
        data: attn_hs.data,
        len: HIDDEN,
    };
    let normed = normed_hs.data;

    let mlp = match &w.mlp {
        Glm52LayerMlp::Dense(dense) => glm52_dense_mlp_forward(ctx, dense, &normed)?,
        Glm52LayerMlp::Moe(moe) => glm52_moe_forward(ctx, moe, &normed, step.moe_path)?,
    };
    Ok(residual_add(ctx, residual, mlp)?.data)
}
