//! GLM5.2 MTP layer-78 accuracy-oracle bookends.
//!
//! The checkpoint's MTP decoder block is the same concrete decoder-layer
//! implementation as the target stack. This module owns only the math unique
//! to MTP:
//!
//! ```text
//! embed = where(position == 0, 0, embed)
//! decoder_input = eh_proj(cat(enorm(embed), hnorm(previous_hidden)))
//! raw_hidden = decoder_layer_78(decoder_input)
//! recycle_hidden = shared_head.norm(raw_hidden)
//! logits = lm_head(shared_head.norm(raw_hidden))
//! ```
//!
//! `raw_hidden` must remain available for target-head logits. The normalized
//! value is recycled into the next draft iteration; normalizing in place
//! would apply the shared norm twice on the logits path.
//!
//! Production serving owns residency and state in `model::mtp`; the oracle
//! tests call these same bookend operations directly.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::copy_hidden_rows_raw_into;
use openinfer_kernels::ops::gemm_strided_batched_bf16;
use openinfer_kernels::ops::mask_position_zero_rows_into;
use openinfer_kernels::ops::rms_norm_rows_into;
use openinfer_kernels::tensor::DeviceContext;
use openinfer_kernels::tensor::DeviceMatrix;
use openinfer_kernels::tensor::DeviceVec;
use openinfer_kernels::tensor::HiddenStates;

use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_INDEX_HEAD_DIM;
use crate::config::GLM52_RMS_EPS;
use crate::model::GLM52_DECODE_BUCKETS;
use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::model::GLM52_MODEL_LEN_ALIGN;
use crate::model::glm52_pool_blocks;
use crate::rows::Rows;

const MTP_FUSED_INPUT: usize = 2 * GLM52_HIDDEN;
pub(crate) const GLM52_MTP_DRAFTS: usize = 5;

/// Context-scaled device memory owned by the native MTP lane: one layer of
/// MLA + index-K cache and one set of per-bucket indexer logits/block tables.
/// Fixed-size weights and scratch are accounted by the post-build headroom
/// probe; this function is the exact monotone term used to derive the context
/// cap before those arenas are allocated.
pub(crate) fn glm52_mtp_arena_bytes(max_model_len: usize) -> Result<usize> {
    let blocks = glm52_pool_blocks(max_model_len, GLM52_MAX_BATCH_PER_RANK);
    let mla = blocks
        .checked_mul(GLM52_MODEL_LEN_ALIGN)
        .and_then(|v| v.checked_mul(openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN))
        .context("GLM5.2 MTP MLA arena byte count overflow")?;
    let index_k = blocks
        .checked_mul(GLM52_MODEL_LEN_ALIGN)
        .and_then(|v| v.checked_mul(GLM52_INDEX_HEAD_DIM + size_of::<f32>()))
        .context("GLM5.2 MTP index-K arena byte count overflow")?;
    let rows: usize = GLM52_DECODE_BUCKETS.iter().sum();
    let indexer_logits = rows
        .checked_mul(max_model_len.next_multiple_of(256))
        .and_then(|v| v.checked_mul(size_of::<bf16>() + size_of::<f32>()))
        .context("GLM5.2 MTP indexer scratch byte count overflow")?;
    let block_tables = rows
        .checked_mul(max_model_len.div_ceil(GLM52_MODEL_LEN_ALIGN))
        .and_then(|v| v.checked_mul(size_of::<i32>()))
        .context("GLM5.2 MTP block-table byte count overflow")?;
    mla.checked_add(index_k)
        .and_then(|v| v.checked_add(indexer_logits))
        .and_then(|v| v.checked_add(block_tables))
        .context("GLM5.2 MTP arena byte count overflow")
}

/// The four BF16 weights around the ordinary layer-78 decoder block.
pub(crate) struct Glm52MtpBookendWeights {
    enorm: DeviceVec,
    hnorm: DeviceVec,
    eh_proj: DeviceMatrix,
    shared_norm: DeviceVec,
}

impl Glm52MtpBookendWeights {
    pub(crate) fn new(
        enorm: DeviceVec,
        hnorm: DeviceVec,
        eh_proj: DeviceMatrix,
        shared_norm: DeviceVec,
    ) -> Result<Self> {
        ensure!(
            enorm.len == GLM52_HIDDEN,
            "GLM5.2 MTP enorm must be [{GLM52_HIDDEN}], got [{}]",
            enorm.len
        );
        ensure!(
            hnorm.len == GLM52_HIDDEN,
            "GLM5.2 MTP hnorm must be [{GLM52_HIDDEN}], got [{}]",
            hnorm.len
        );
        ensure!(
            eh_proj.rows == GLM52_HIDDEN && eh_proj.cols == MTP_FUSED_INPUT,
            "GLM5.2 MTP eh_proj must be [{GLM52_HIDDEN}, {MTP_FUSED_INPUT}], got [{}, {}]",
            eh_proj.rows,
            eh_proj.cols
        );
        ensure!(
            shared_norm.len == GLM52_HIDDEN,
            "GLM5.2 MTP shared norm must be [{GLM52_HIDDEN}], got [{}]",
            shared_norm.len
        );
        Ok(Self {
            enorm,
            hnorm,
            eh_proj,
            shared_norm,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_host(
        ctx: &DeviceContext,
        enorm: &[u8],
        hnorm: &[u8],
        eh_proj: &[u8],
        shared_norm: &[u8],
    ) -> Result<Self> {
        Self::new(
            DeviceVec::from_safetensors(ctx, enorm)?,
            DeviceVec::from_safetensors(ctx, hnorm)?,
            DeviceMatrix::from_safetensors(ctx, eh_proj, GLM52_HIDDEN, MTP_FUSED_INPUT)?,
            DeviceVec::from_safetensors(ctx, shared_norm)?,
        )
    }
}

/// Persistent MTP-only intermediates for one row bucket.
pub(crate) struct Glm52MtpScratch {
    masked_embed: Rows<GLM52_HIDDEN>,
    normed_embed: Rows<GLM52_HIDDEN>,
    normed_previous: Rows<GLM52_HIDDEN>,
    fused_input: HiddenStates,
}

impl Glm52MtpScratch {
    pub(crate) fn new(ctx: &DeviceContext, tokens: usize) -> Result<Self> {
        Ok(Self {
            masked_embed: Rows::zeros(ctx, tokens)?,
            normed_embed: Rows::zeros(ctx, tokens)?,
            normed_previous: Rows::zeros(ctx, tokens)?,
            fused_input: HiddenStates::zeros(ctx, MTP_FUSED_INPUT, tokens)?,
        })
    }

    #[cfg(test)]
    pub(crate) fn normed_embed(&self) -> &Rows<GLM52_HIDDEN> {
        &self.normed_embed
    }

    #[cfg(test)]
    pub(crate) fn normed_previous(&self) -> &Rows<GLM52_HIDDEN> {
        &self.normed_previous
    }
}

/// Build the ordinary layer-78 decoder input. One GEMM consumes the physical
/// concatenation so its accumulation and BF16 output boundary match vLLM's
/// `nn.Linear(torch.cat(...))`.
pub(crate) fn glm52_mtp_prepare_into(
    ctx: &DeviceContext,
    w: &Glm52MtpBookendWeights,
    positions: &CudaSlice<u32>,
    inputs_embeds: &Rows<GLM52_HIDDEN>,
    previous_hidden: &Rows<GLM52_HIDDEN>,
    s: &mut Glm52MtpScratch,
    decoder_input: &mut Rows<GLM52_HIDDEN>,
) -> Result<()> {
    let tokens = inputs_embeds.tokens();
    ensure!(
        previous_hidden.tokens() == tokens
            && s.masked_embed.tokens() == tokens
            && decoder_input.tokens() == tokens,
        "GLM5.2 MTP row bucket mismatch"
    );
    mask_position_zero_rows_into(
        ctx,
        inputs_embeds.data(),
        positions,
        GLM52_HIDDEN,
        tokens,
        s.masked_embed.data_mut(),
    )?;
    rms_norm_rows_into(
        ctx,
        s.masked_embed.data(),
        &w.enorm,
        GLM52_RMS_EPS,
        GLM52_HIDDEN,
        tokens,
        s.normed_embed.data_mut(),
    )?;
    rms_norm_rows_into(
        ctx,
        previous_hidden.data(),
        &w.hnorm,
        GLM52_RMS_EPS,
        GLM52_HIDDEN,
        tokens,
        s.normed_previous.data_mut(),
    )?;
    copy_hidden_rows_raw_into(
        ctx,
        s.normed_embed.data(),
        GLM52_HIDDEN,
        &mut s.fused_input.data,
        MTP_FUSED_INPUT,
        0,
        tokens,
    )?;
    copy_hidden_rows_raw_into(
        ctx,
        s.normed_previous.data(),
        GLM52_HIDDEN,
        &mut s.fused_input.data,
        MTP_FUSED_INPUT,
        GLM52_HIDDEN,
        tokens,
    )?;
    gemm_strided_batched_bf16(
        ctx,
        true,
        false,
        GLM52_HIDDEN,
        tokens,
        MTP_FUSED_INPUT,
        &w.eh_proj.data,
        MTP_FUSED_INPUT,
        0,
        &s.fused_input.data,
        MTP_FUSED_INPUT,
        0,
        decoder_input.data_mut(),
        GLM52_HIDDEN,
        0,
        1,
    )
}

/// Normalize layer 78's raw residual output for the next MTP iteration.
/// Callers retain `raw_hidden` unchanged for the shared target lm_head path.
pub(crate) fn glm52_mtp_recycle_into(
    ctx: &DeviceContext,
    w: &Glm52MtpBookendWeights,
    raw_hidden: &Rows<GLM52_HIDDEN>,
    recycle_hidden: &mut Rows<GLM52_HIDDEN>,
) -> Result<()> {
    ensure!(
        raw_hidden.tokens() == recycle_hidden.tokens(),
        "GLM5.2 MTP recycle row bucket mismatch"
    );
    rms_norm_rows_into(
        ctx,
        raw_hidden.data(),
        &w.shared_norm,
        GLM52_RMS_EPS,
        GLM52_HIDDEN,
        raw_hidden.tokens(),
        recycle_hidden.data_mut(),
    )
}
