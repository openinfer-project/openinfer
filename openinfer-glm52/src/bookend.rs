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
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use openinfer_kernels::ops::embedding_rows_into;
use openinfer_kernels::ops::gemm_strided_batched_bf16;
use openinfer_kernels::ops::rms_norm_rows_into;
use openinfer_kernels::tensor::DeviceContext;
use openinfer_kernels::tensor::DeviceMatrix;
use openinfer_kernels::tensor::DeviceVec;

use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_RMS_EPS;
use crate::config::GLM52_SELECTION_VOCAB;
use crate::config::GLM52_VOCAB;
use crate::rows::Rows;

/// Token embedding lookup over `out.tokens()` rows: `embed[token_ids[r]] ->
/// [T, HIDDEN]`. `token_ids` is a device buffer (read on-device, so the lookup
/// is CUDA-graph-safe -- the scheduler rewrites it in place each decode step).
pub(crate) fn glm52_embed_into(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids: &CudaSlice<u32>,
    out: &mut Rows<GLM52_HIDDEN>,
) -> Result<()> {
    let tokens = out.tokens();
    embedding_rows_into(ctx, embed, token_ids, tokens, out.data_mut())
}

/// Final RMSNorm over the buffers' `tokens()` rows:
/// `rms_norm(hidden, model.norm.weight, eps=1e-5)`.
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

/// lm_head projection over the buffers' `tokens()` rows. EP passes the
/// full checkpoint head but emits only the tokenizer-selectable prefix;
/// attention-TP decode passes this rank's already-trimmed contiguous shard. One
/// cuBLAS GEMM puts tokens on the n dimension, so the col-major
/// `[logit_rows, T]` output is the compact row-major layout argmax consumes.
pub(crate) fn glm52_lm_head_into(
    ctx: &DeviceContext,
    normed: &Rows<GLM52_HIDDEN>,
    lm_head: &DeviceMatrix,
    out: &mut Rows<GLM52_SELECTION_VOCAB>,
) -> Result<usize> {
    let tokens = out.tokens();
    let logit_rows = if lm_head.rows == GLM52_VOCAB {
        GLM52_SELECTION_VOCAB
    } else {
        lm_head.rows
    };
    ensure!(
        logit_rows <= GLM52_SELECTION_VOCAB
            && (lm_head.rows == GLM52_VOCAB || GLM52_SELECTION_VOCAB.is_multiple_of(logit_rows)),
        "GLM5.2 lm_head rows {} are neither the checkpoint head nor a selectable-vocab shard",
        lm_head.rows
    );
    gemm_strided_batched_bf16(
        ctx,
        true,
        false,
        logit_rows,
        tokens,
        GLM52_HIDDEN,
        &lm_head.data,
        GLM52_HIDDEN,
        0,
        normed.data(),
        GLM52_HIDDEN,
        0,
        out.data_mut(),
        logit_rows,
        0,
        1,
    )?;
    Ok(logit_rows)
}
