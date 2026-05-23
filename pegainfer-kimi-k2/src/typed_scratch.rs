//! Typed decode scratch buffers using `GpuTensor<const DIM>` and `gpu_buffers!`.
//!
//! This is the compile-time dimension-safe replacement for `KimiWorkerDecodeScratch`.
//! All tensor dimensions are encoded in the type — passing a wrong-dim buffer to any
//! op is a compile error.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_kernels::gpu_buffers;
use pegainfer_kernels::tensor::{DeviceContext, GpuTensor};

use crate::config::{
    KIMI_K2_DENSE_INTERMEDIATE, KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_Q_LORA_RANK,
    KIMI_K2_ROUTED_EXPERTS, KIMI_K2_TOPK,
};
use pegainfer_kernels::ops::{
    KIMI_K2_EP_WORLD, KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_LORA_RANK,
    KIMI_K2_MLA_LOCAL_HEADS_TP8, KIMI_K2_MLA_NOPE_DIM, KIMI_K2_MLA_O_LOCAL_IN_TP8,
    KIMI_K2_MLA_Q_LOCAL_OUT_TP8, KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8, KIMI_K2_MLA_QKV_A_OUT,
    KIMI_K2_MLA_ROPE_DIM,
};

// ── Derived dimension constants ──────────────────────────────────────

pub(crate) const DENSE_GATE_UP: usize = KIMI_K2_DENSE_INTERMEDIATE / 4;
pub(crate) const DENSE_ACTIVATED: usize = KIMI_K2_DENSE_INTERMEDIATE / 8;
pub(crate) const SHARED_GATE_UP: usize = KIMI_K2_EXPERT_INTERMEDIATE / 4;
pub(crate) const SHARED_ACTIVATED: usize = KIMI_K2_EXPERT_INTERMEDIATE / 8;
pub(crate) const Q_NOPE_DIM: usize = KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_NOPE_DIM;
pub(crate) const MARLIN_W13_OUT: usize = 2 * KIMI_K2_EXPERT_INTERMEDIATE;

// ── Typed scratch: standard tensors ──────────────────────────────────

gpu_buffers! {
    /// MLA + norm intermediate buffers (all batch_size-indexed).
    pub(crate) struct MlaDecodeScratch {
        pub(crate) hidden:            GpuTensor<{ KIMI_K2_HIDDEN }>,
        pub(crate) normed:            GpuTensor<{ KIMI_K2_HIDDEN }>,
        pub(crate) projected:         GpuTensor<{ KIMI_K2_HIDDEN }>,
        pub(crate) qkv_a:             GpuTensor<{ KIMI_K2_MLA_QKV_A_OUT }>,
        pub(crate) q_a:               GpuTensor<{ KIMI_K2_Q_LORA_RANK }>,
        pub(crate) q_a_normed:        GpuTensor<{ KIMI_K2_Q_LORA_RANK }>,
        pub(crate) q_proj:            GpuTensor<{ KIMI_K2_MLA_Q_LOCAL_OUT_TP8 }>,
        pub(crate) compressed_kv:     GpuTensor<{ KIMI_K2_MLA_KV_LORA_RANK }>,
        pub(crate) k_rope:            GpuTensor<{ KIMI_K2_MLA_ROPE_DIM }>,
        pub(crate) compressed_normed: GpuTensor<{ KIMI_K2_MLA_KV_LORA_RANK }>,
        pub(crate) q_nope:            GpuTensor<{ Q_NOPE_DIM }>,
        pub(crate) q_pe:              GpuTensor<{ KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8 }>,
        pub(crate) append_kpe:        GpuTensor<{ KIMI_K2_MLA_ROPE_DIM }>,
        pub(crate) q_abs_nope:        GpuTensor<{ KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8 }>,
        pub(crate) latent:            GpuTensor<{ KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8 }>,
        pub(crate) attn_out:          GpuTensor<{ KIMI_K2_MLA_O_LOCAL_IN_TP8 }>,
    }
}

gpu_buffers! {
    /// Dense MLP intermediate buffers.
    pub(crate) struct DenseMlpDecodeScratch {
        pub(crate) dense_gate_up:   GpuTensor<{ DENSE_GATE_UP }>,
        pub(crate) dense_activated: GpuTensor<{ DENSE_ACTIVATED }>,
    }
}

gpu_buffers! {
    /// Shared expert intermediate buffers.
    pub(crate) struct SharedExpertDecodeScratch {
        pub(crate) shared_gate_up:   GpuTensor<{ SHARED_GATE_UP }>,
        pub(crate) shared_activated: GpuTensor<{ SHARED_ACTIVATED }>,
    }
}

gpu_buffers! {
    /// Router output buffers.
    pub(crate) struct RouterScratch {
        pub(crate) router_logits:        GpuRawSlice<{ KIMI_K2_ROUTED_EXPERTS }>,
        pub(crate) router_scores:        GpuRawSlice<{ KIMI_K2_ROUTED_EXPERTS }>,
        pub(crate) router_choice_scores: GpuRawSlice<{ KIMI_K2_ROUTED_EXPERTS }>,
        pub(crate) router_topk_weight:   GpuRawSlice<{ KIMI_K2_TOPK }>,
        pub(crate) router_topk_idx:      GpuRawSliceI32<{ KIMI_K2_TOPK }>,
    }
}

/// Marlin expert buffers — allocated with `batch_size * KIMI_K2_TOPK` as the
/// batch dimension (one row per routed token-expert pair).
pub(crate) struct MarlinExpertScratch {
    pub(crate) w13_out: GpuTensor<MARLIN_W13_OUT>,
    pub(crate) activated: GpuTensor<KIMI_K2_EXPERT_INTERMEDIATE>,
    pub(crate) expert_output: GpuTensor<KIMI_K2_HIDDEN>,
}

impl MarlinExpertScratch {
    pub(crate) fn new(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        let route_elems = batch_size * KIMI_K2_TOPK;
        Ok(Self {
            w13_out: GpuTensor::zeros(ctx, route_elems)?,
            activated: GpuTensor::zeros(ctx, route_elems)?,
            expert_output: GpuTensor::zeros(ctx, route_elems)?,
        })
    }
}

/// Communication buffers for EP reduce-scatter and TP all-reduce.
pub(crate) struct CommScratch {
    pub(crate) routed_out_f32: CudaSlice<f32>,
    pub(crate) routed_reduce_scatter_send_f32: CudaSlice<f32>,
    pub(crate) hidden_allreduce_f32: CudaSlice<f32>,
}

impl CommScratch {
    pub(crate) fn new(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        let reduce_scatter_send_rows = batch_size * KIMI_K2_EP_WORLD;
        Ok(Self {
            routed_out_f32: ctx.stream.alloc_zeros(batch_size * KIMI_K2_HIDDEN)?,
            routed_reduce_scatter_send_f32: ctx
                .stream
                .alloc_zeros(reduce_scatter_send_rows * KIMI_K2_HIDDEN)?,
            hidden_allreduce_f32: ctx.stream.alloc_zeros(batch_size * KIMI_K2_HIDDEN)?,
        })
    }
}

/// Sampling scratch buffers.
pub(crate) struct SamplingScratch {
    pub(crate) top1_value_scratch: CudaSlice<half::bf16>,
    pub(crate) top1_out: CudaSlice<i32>,
}

impl SamplingScratch {
    pub(crate) fn new(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        Ok(Self {
            top1_value_scratch: ctx.stream.alloc_zeros(batch_size)?,
            top1_out: ctx.stream.alloc_zeros(batch_size)?,
        })
    }
}

/// Complete typed decode scratch — replaces `KimiWorkerDecodeScratch`.
///
/// Composed from domain-specific sub-scratches. Each has its own `gpu_buffers!`
/// or manual constructor, so the struct and allocation can never drift apart.
pub(crate) struct TypedDecodeScratch {
    pub(crate) mla: MlaDecodeScratch,
    pub(crate) dense_mlp: DenseMlpDecodeScratch,
    pub(crate) shared_expert: SharedExpertDecodeScratch,
    pub(crate) router: RouterScratch,
    pub(crate) marlin: MarlinExpertScratch,
    pub(crate) comm: CommScratch,
    pub(crate) sampling: SamplingScratch,
}

impl TypedDecodeScratch {
    pub(crate) fn new(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        Ok(Self {
            mla: MlaDecodeScratch::new(ctx, batch_size)?,
            dense_mlp: DenseMlpDecodeScratch::new(ctx, batch_size)?,
            shared_expert: SharedExpertDecodeScratch::new(ctx, batch_size)?,
            router: RouterScratch::new(ctx, batch_size)?,
            marlin: MarlinExpertScratch::new(ctx, batch_size)?,
            comm: CommScratch::new(ctx, batch_size)?,
            sampling: SamplingScratch::new(ctx, batch_size)?,
        })
    }

    pub(crate) fn set_batch_size(&mut self, bs: usize) {
        self.mla.set_batch_size(bs);
        self.dense_mlp.set_batch_size(bs);
        self.shared_expert.set_batch_size(bs);
        self.router.set_batch_size(bs);
    }
}

// ── Line count comparison ────────────────────────────────────────────
//
// Original KimiWorkerDecodeScratch:
//   struct definition:  36 fields,  37 lines
//   fn new():           67 lines (manual alloc + marlin workspace)
//   Total:             104 lines
//
// This file:
//   gpu_buffers! declarations:       ~30 lines (4 macros)
//   Manual structs (marlin/comm/sampling): ~40 lines
//   TypedDecodeScratch composition:  ~20 lines
//   Total:             ~90 lines
//
// Net savings are modest in line count, but the real win is:
// 1. Struct fields and new() can never drift apart (macro generates both)
// 2. Every tensor carries its dimension in the type
// 3. Forward functions get compile-time shape checking (see typed_forward.rs)
