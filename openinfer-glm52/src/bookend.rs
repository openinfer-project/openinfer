//! GLM5.2 decode bookends, row-batched: the token embedding and the final
//! RMSNorm + lm_head tail that bracket the layer stack.
//!
//! All three are plain bf16 ops (the embedding table, `model.norm.weight`, and
//! `lm_head.weight` are bf16, not fp8). These are thin wrappers over the shared
//! `openinfer-kernels` embed/norm/gemm ops -- the only GLM5.2-specific facts are
//! the dimensions and the 1e-5 RMS epsilon.

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use openinfer_kernels::ops::{embedding_rows_into, gemm_strided_batched_bf16, rms_norm_rows_into};
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

use crate::config::{GLM52_HIDDEN, GLM52_VOCAB};

const RMS_EPS: f32 = 1.0e-5;

/// Token embedding lookup: `embed[token_ids[r]] -> [T, HIDDEN]`. `token_ids`
/// is a device buffer (read on-device, so the lookup is CUDA-graph-safe --
/// the scheduler rewrites it in place each decode step).
#[cfg(test)]
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
    glm52_embed_into(ctx, embed, token_id, 1, &mut out)?;
    Ok(out)
}

/// [`glm52_embed`] over `tokens` rows into a pre-allocated `[T, HIDDEN]`
/// output (the decode path).
pub(crate) fn glm52_embed_into(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids: &CudaSlice<u32>,
    tokens: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    ensure!(
        embed.rows == GLM52_VOCAB && embed.cols == GLM52_HIDDEN,
        "GLM5.2 embed table shape [{},{}] != [{GLM52_VOCAB},{GLM52_HIDDEN}]",
        embed.rows,
        embed.cols
    );
    ensure!(
        out.len == tokens * GLM52_HIDDEN,
        "GLM5.2 embed out len drifted"
    );
    embedding_rows_into(ctx, embed, token_ids, tokens, &mut out.data)
}

/// Final RMSNorm: `rms_norm(hidden, model.norm.weight, eps=1e-5)`.
#[cfg(test)]
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
    glm52_final_norm_into(ctx, hidden, norm_weight, 1, &mut out)?;
    Ok(out)
}

/// [`glm52_final_norm`] over `tokens` rows into a pre-allocated `[T, HIDDEN]`
/// output (the decode path).
pub(crate) fn glm52_final_norm_into(
    ctx: &DeviceContext,
    hidden: &DeviceVec,
    norm_weight: &DeviceVec,
    tokens: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    ensure!(
        hidden.len == tokens * GLM52_HIDDEN && norm_weight.len == GLM52_HIDDEN,
        "GLM5.2 final norm lengths hidden {} / weight {} drifted",
        hidden.len,
        norm_weight.len
    );
    ensure!(
        out.len == tokens * GLM52_HIDDEN,
        "GLM5.2 final norm out len drifted"
    );
    rms_norm_rows_into(
        ctx,
        &hidden.data,
        norm_weight,
        RMS_EPS,
        GLM52_HIDDEN,
        tokens,
        &mut out.data,
    )
}

/// lm_head projection: `lm_head @ normed -> [VOCAB]` logits. The weight is bf16
/// `[VOCAB, HIDDEN]`; the caller feeds the final-normed hidden.
#[cfg(test)]
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
    glm52_lm_head_into(ctx, normed, lm_head, 1, &mut out)?;
    Ok(out)
}

/// [`glm52_lm_head`] over `tokens` rows into a pre-allocated `[T, VOCAB]`
/// output (the decode path). One cuBLAS GEMM with the rows on the n
/// dimension: the col-major `[VOCAB, T]` output IS the row-major `[T, VOCAB]`
/// layout the argmax consumes, and the 1.9 GB weight is read once for all
/// rows.
pub(crate) fn glm52_lm_head_into(
    ctx: &DeviceContext,
    normed: &DeviceVec,
    lm_head: &DeviceMatrix,
    tokens: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    ensure!(
        lm_head.rows == GLM52_VOCAB && lm_head.cols == GLM52_HIDDEN,
        "GLM5.2 lm_head shape [{},{}] != [{GLM52_VOCAB},{GLM52_HIDDEN}]",
        lm_head.rows,
        lm_head.cols
    );
    ensure!(
        normed.len == tokens * GLM52_HIDDEN,
        "GLM5.2 lm_head input len {} drifted",
        normed.len
    );
    ensure!(
        out.len == tokens * GLM52_VOCAB,
        "GLM5.2 lm_head out len drifted"
    );
    gemm_strided_batched_bf16(
        ctx,
        true,  // lm_head [VOCAB, HIDDEN] row-major -> col-major via transpose
        false, // normed [T, HIDDEN] row-major = [HIDDEN, T] col-major
        GLM52_VOCAB,
        tokens,
        GLM52_HIDDEN,
        &lm_head.data,
        GLM52_HIDDEN,
        0,
        &normed.data,
        GLM52_HIDDEN,
        0,
        &mut out.data,
        GLM52_VOCAB,
        0,
        1,
    )
}
