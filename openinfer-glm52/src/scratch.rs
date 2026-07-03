//! Persistent per-rank decode scratch: every intermediate of one decode step
//! lives in a buffer allocated once at model build.
//!
//! Two properties this buys (both prerequisites for the whole-step CUDA
//! graph, PR5c stage 3):
//!
//! - **No per-step allocator traffic.** PR5a made the MoE collective chain
//!   persistent (`Glm52MoeEp8State`); this extends the same treatment to the
//!   MLA/indexer/MLP spine (~5 ms/step of residual `alloc_zeros` at bs=1).
//! - **Pointer stability.** A captured graph replays the recorded device
//!   pointers; every kernel in the step must read/write the same buffers
//!   every step.
//!
//! Safety rule for reuse: every buffer is bound to ONE call-site purpose and
//! shared only ACROSS layers (the 78 layers have identical shapes, and layer
//! N's intermediates are dead once layer N consumed them). Stale content is
//! therefore always "the same semantic value from the previous step/layer" —
//! either fully overwritten before its consumer reads it, or never written by
//! anyone and still holding the build-time zero initialization (e.g. the
//! TMA-relayout pad rows). The one deliberate exception is
//! `idx.global_slots`, the cross-layer top-k carry: full-indexer layers
//! overwrite it, shared layers reuse it — exactly the DSA contract.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;

use openinfer_kernels::ops::{Glm52DeepGemmMqaLogitsShape, Glm52FlashMlaSparseDecode};
use openinfer_kernels::tensor::{DeviceContext, DeviceVec};

use crate::config::{GLM52_HIDDEN, GLM52_VOCAB};
use crate::dense::GLM52_DENSE_INTERMEDIATE;
use crate::fp8::{Glm52MlpScratch, Glm52ProjScratch};
use crate::indexer::Glm52IndexerScratch;
use crate::layer::Glm52LayerScratch;
use crate::mla_decode::{Glm52MlaAttendScratch, Glm52MlaFront};
use crate::moe_decode::{GLM52_SHARED_EXPERT_INTERMEDIATE, Glm52RouterScratch};

/// The widest fp8 projection activation (o_proj: k = 64 heads × 256 v_head).
/// The single shared projection scratch is sized to it.
const MAX_PROJ_K: usize = 16_384;

/// Everything one decode step writes, allocated once per rank.
pub(crate) struct Glm52DecodeScratch {
    pub(crate) proj: Glm52ProjScratch,
    pub(crate) mla_front: Glm52MlaFront,
    pub(crate) mla_attend: Glm52MlaAttendScratch,
    pub(crate) idx: Glm52IndexerScratch,
    pub(crate) dense_mlp: Glm52MlpScratch,
    pub(crate) shared_mlp: Glm52MlpScratch,
    pub(crate) router: Glm52RouterScratch,
    pub(crate) layer: Glm52LayerScratch,
    /// The residual stream: embed writes it, every layer reads and rewrites
    /// it, the final norm consumes it.
    pub(crate) hidden: DeviceVec,
    pub(crate) final_normed: DeviceVec,
    pub(crate) logits: DeviceVec,
    /// Device greedy argmax outputs: the top logit's bf16 value (for the
    /// crash-early non-finite guard) and its index — the step's 6-byte D2H
    /// egress.
    pub(crate) argmax_value: CudaSlice<bf16>,
    pub(crate) argmax_index: CudaSlice<i32>,
}

impl Glm52DecodeScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        contract: &Glm52FlashMlaSparseDecode,
        mqa_shape: Glm52DeepGemmMqaLogitsShape,
    ) -> Result<Self> {
        Ok(Self {
            proj: Glm52ProjScratch::new(ctx, MAX_PROJ_K)?,
            mla_front: Glm52MlaFront::new(ctx)?,
            mla_attend: Glm52MlaAttendScratch::new(ctx, contract)?,
            idx: Glm52IndexerScratch::new(ctx, mqa_shape)?,
            dense_mlp: Glm52MlpScratch::new(ctx, GLM52_DENSE_INTERMEDIATE)?,
            shared_mlp: Glm52MlpScratch::new(ctx, GLM52_SHARED_EXPERT_INTERMEDIATE)?,
            router: Glm52RouterScratch::new(ctx)?,
            layer: Glm52LayerScratch::new(ctx)?,
            hidden: DeviceVec::zeros(ctx, GLM52_HIDDEN)?,
            final_normed: DeviceVec::zeros(ctx, GLM52_HIDDEN)?,
            logits: DeviceVec::zeros(ctx, GLM52_VOCAB)?,
            argmax_value: ctx.stream.alloc_zeros::<bf16>(1)?,
            argmax_index: ctx.stream.alloc_zeros::<i32>(1)?,
        })
    }
}
