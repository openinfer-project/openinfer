//! GLM5.2 decode bookends, row-batched: the token embedding and the final
//! RMSNorm + lm_head tail that bracket the layer stack.
//!
//! All three are plain bf16 ops (the embedding table, `model.norm.weight`, and
//! `lm_head.weight` are bf16, not fp8). These are thin wrappers over the shared
//! `openinfer-kernels` embed/norm/gemm ops -- the only GLM5.2-specific facts are
//! the dimensions. Shapes are not re-validated here: the weight matrices are
//! pinned to `[VOCAB, HIDDEN]` at model build, and the `Rows` buffers carry
//! their row count and width by construction.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use openinfer_kernels::ops::{embedding_rows_into, gemm_strided_batched_bf16, rms_norm_rows_into};
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

use crate::config::{GLM52_HIDDEN, GLM52_RMS_EPS, GLM52_VOCAB};
use crate::rows::Rows;

/// Token embedding lookup: `embed[token_ids[r]] -> [T, HIDDEN]`. `token_ids`
/// is a device buffer (read on-device, so the lookup is CUDA-graph-safe --
/// the scheduler rewrites it in place each decode step).
#[cfg(test)]
pub(crate) fn glm52_embed(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_id: &CudaSlice<u32>,
) -> Result<Rows<GLM52_HIDDEN>> {
    let mut out = Rows::zeros(ctx, 1)?;
    glm52_embed_into(ctx, embed, token_id, &mut out)?;
    Ok(out)
}

/// [`glm52_embed`] over `out.tokens()` rows into a pre-allocated `[T, HIDDEN]`
/// output (the decode path).
pub(crate) fn glm52_embed_into(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids: &CudaSlice<u32>,
    out: &mut Rows<GLM52_HIDDEN>,
) -> Result<()> {
    let tokens = out.tokens();
    embedding_rows_into(ctx, embed, token_ids, tokens, out.data_mut())
}

/// Final RMSNorm: `rms_norm(hidden, model.norm.weight, eps=1e-5)`.
#[cfg(test)]
pub(crate) fn glm52_final_norm(
    ctx: &DeviceContext,
    hidden: &Rows<GLM52_HIDDEN>,
    norm_weight: &DeviceVec,
) -> Result<Rows<GLM52_HIDDEN>> {
    let mut out = Rows::zeros(ctx, hidden.tokens())?;
    glm52_final_norm_into(ctx, hidden, norm_weight, &mut out)?;
    Ok(out)
}

/// [`glm52_final_norm`] over the buffers' `tokens()` rows into a
/// pre-allocated `[T, HIDDEN]` output (the decode path).
pub(crate) fn glm52_final_norm_into(
    ctx: &DeviceContext,
    hidden: &Rows<GLM52_HIDDEN>,
    norm_weight: &DeviceVec,
    out: &mut Rows<GLM52_HIDDEN>,
) -> Result<()> {
    let tokens = out.tokens();
    rms_norm_rows_into(
        ctx,
        hidden.data(),
        norm_weight,
        GLM52_RMS_EPS,
        GLM52_HIDDEN,
        tokens,
        out.data_mut(),
    )
}

/// lm_head projection: `lm_head @ normed -> [VOCAB]` logits. The weight is bf16
/// `[VOCAB, HIDDEN]`; the caller feeds the final-normed hidden.
#[cfg(test)]
pub(crate) fn glm52_lm_head(
    ctx: &DeviceContext,
    normed: &Rows<GLM52_HIDDEN>,
    lm_head: &DeviceMatrix,
) -> Result<Rows<GLM52_VOCAB>> {
    let mut out = Rows::zeros(ctx, normed.tokens())?;
    glm52_lm_head_into(ctx, normed, lm_head, &mut out)?;
    Ok(out)
}

/// [`glm52_lm_head`] over the buffers' `tokens()` rows into a pre-allocated
/// `[T, VOCAB]` output (the decode path). One cuBLAS GEMM with the rows on
/// the n dimension: the col-major `[VOCAB, T]` output IS the row-major
/// `[T, VOCAB]` layout the argmax consumes, and the 1.9 GB weight is read
/// once for all rows.
pub(crate) fn glm52_lm_head_into(
    ctx: &DeviceContext,
    normed: &Rows<GLM52_HIDDEN>,
    lm_head: &DeviceMatrix,
    out: &mut Rows<GLM52_VOCAB>,
) -> Result<()> {
    let tokens = out.tokens();
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
        normed.data(),
        GLM52_HIDDEN,
        0,
        out.data_mut(),
        GLM52_VOCAB,
        0,
        1,
    )
}
