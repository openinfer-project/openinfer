//! GLM5.2 decode bookends for bs=1: the token embedding and the final RMSNorm +
//! lm_head tail that bracket the layer stack.
//!
//! All three are plain bf16 ops (the embedding table, `model.norm.weight`, and
//! `lm_head.weight` are bf16, not fp8). These are thin wrappers over the shared
//! `openinfer-kernels` embed/norm/gemv ops -- the only GLM5.2-specific facts are
//! the dimensions and the 1e-5 RMS epsilon.

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use openinfer_kernels::ops::{embedding_decode_into, gemv, rms_norm_into};
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

use crate::config::{GLM52_HIDDEN, GLM52_VOCAB};

const RMS_EPS: f32 = 1.0e-5;

/// Token embedding lookup: `embed[token_id] -> [HIDDEN]`. `token_id` is a
/// single-element device buffer (read on-device, so the lookup is
/// CUDA-graph-safe -- the scheduler rewrites it in place each decode step).
pub(crate) fn glm52_embed(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_id: &CudaSlice<u32>,
) -> Result<DeviceVec> {
    ensure!(
        embed.rows == GLM52_VOCAB && embed.cols == GLM52_HIDDEN,
        "GLM5.2 embed table shape [{},{}] != [{GLM52_VOCAB},{GLM52_HIDDEN}]",
        embed.rows,
        embed.cols
    );
    let mut out = DeviceVec {
        data: ctx.stream.alloc_zeros::<half::bf16>(GLM52_HIDDEN)?,
        len: GLM52_HIDDEN,
    };
    embedding_decode_into(ctx, embed, token_id, &mut out)?;
    Ok(out)
}

/// Final RMSNorm: `rms_norm(hidden, model.norm.weight, eps=1e-5)`.
pub(crate) fn glm52_final_norm(
    ctx: &DeviceContext,
    hidden: &DeviceVec,
    norm_weight: &DeviceVec,
) -> Result<DeviceVec> {
    ensure!(
        hidden.len == GLM52_HIDDEN && norm_weight.len == GLM52_HIDDEN,
        "GLM5.2 final norm lengths hidden {} / weight {} != {GLM52_HIDDEN}",
        hidden.len,
        norm_weight.len
    );
    let mut out = DeviceVec {
        data: ctx.stream.alloc_zeros::<half::bf16>(GLM52_HIDDEN)?,
        len: GLM52_HIDDEN,
    };
    rms_norm_into(ctx, hidden, norm_weight, RMS_EPS, &mut out)?;
    Ok(out)
}

/// lm_head projection: `lm_head @ normed -> [VOCAB]` logits. The weight is bf16
/// `[VOCAB, HIDDEN]`; the caller feeds the final-normed hidden.
pub(crate) fn glm52_lm_head(
    ctx: &DeviceContext,
    normed: &DeviceVec,
    lm_head: &DeviceMatrix,
) -> Result<DeviceVec> {
    ensure!(
        lm_head.rows == GLM52_VOCAB && lm_head.cols == GLM52_HIDDEN,
        "GLM5.2 lm_head shape [{},{}] != [{GLM52_VOCAB},{GLM52_HIDDEN}]",
        lm_head.rows,
        lm_head.cols
    );
    ensure!(
        normed.len == GLM52_HIDDEN,
        "GLM5.2 lm_head input len {} != {GLM52_HIDDEN}",
        normed.len
    );
    let mut out = DeviceVec {
        data: ctx.stream.alloc_zeros::<half::bf16>(GLM52_VOCAB)?,
        len: GLM52_VOCAB,
    };
    gemv(ctx, lm_head, normed, &mut out)?;
    Ok(out)
}
