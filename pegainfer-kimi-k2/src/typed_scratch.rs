//! Typed Kimi decode scratch buffers.

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
    KIMI_K2_MLA_O_LOCAL_IN_TP8, KIMI_K2_MLA_Q_LOCAL_OUT_TP8, KIMI_K2_MLA_Q_NOPE_LOCAL_OUT_TP8,
    KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8, KIMI_K2_MLA_QKV_A_OUT, KIMI_K2_MLA_ROPE_DIM,
    KimiMarlinRouteWorkspace, KimiMarlinWna16Workspace,
};

pub(crate) const DENSE_GATE_UP_DIM: usize = KIMI_K2_DENSE_INTERMEDIATE / 4;
pub(crate) const DENSE_ACTIVATED_DIM: usize = KIMI_K2_DENSE_INTERMEDIATE / 8;
pub(crate) const SHARED_GATE_UP_DIM: usize = KIMI_K2_EXPERT_INTERMEDIATE / 4;
pub(crate) const SHARED_ACTIVATED_DIM: usize = KIMI_K2_EXPERT_INTERMEDIATE / 8;
pub(crate) const MARLIN_W13_OUT_DIM: usize = 2 * KIMI_K2_EXPERT_INTERMEDIATE;

gpu_buffers! {
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
        pub(crate) q_nope:            GpuTensor<{ KIMI_K2_MLA_Q_NOPE_LOCAL_OUT_TP8 }>,
        pub(crate) q_pe:              GpuTensor<{ KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8 }>,
        pub(crate) append_kpe:        GpuTensor<{ KIMI_K2_MLA_ROPE_DIM }>,
        pub(crate) q_abs_nope:        GpuTensor<{ KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8 }>,
        pub(crate) latent:            GpuTensor<{ KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8 }>,
        pub(crate) attn_out:          GpuTensor<{ KIMI_K2_MLA_O_LOCAL_IN_TP8 }>,
    }
}

gpu_buffers! {
    pub(crate) struct DenseMlpDecodeScratch {
        pub(crate) gate_up:   GpuTensor<{ DENSE_GATE_UP_DIM }>,
        pub(crate) activated: GpuTensor<{ DENSE_ACTIVATED_DIM }>,
    }
}

gpu_buffers! {
    pub(crate) struct SharedExpertDecodeScratch {
        pub(crate) gate_up:   GpuTensor<{ SHARED_GATE_UP_DIM }>,
        pub(crate) activated: GpuTensor<{ SHARED_ACTIVATED_DIM }>,
    }
}

gpu_buffers! {
    pub(crate) struct RouterScratch {
        pub(crate) router_logits:        GpuRawSlice<{ KIMI_K2_ROUTED_EXPERTS }>,
        pub(crate) router_scores:        GpuRawSlice<{ KIMI_K2_ROUTED_EXPERTS }>,
        pub(crate) router_choice_scores: GpuRawSlice<{ KIMI_K2_ROUTED_EXPERTS }>,
        pub(crate) router_topk_weight:   GpuRawSlice<{ KIMI_K2_TOPK }>,
        pub(crate) router_topk_idx:      GpuRawSliceI32<{ KIMI_K2_TOPK }>,
    }
}

pub(crate) struct MarlinExpertScratch {
    pub(crate) w13_out: GpuTensor<MARLIN_W13_OUT_DIM>,
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

pub(crate) struct KimiWorkerDecodeScratch {
    pub(crate) mla: MlaDecodeScratch,
    pub(crate) dense_mlp: DenseMlpDecodeScratch,
    pub(crate) shared_expert: SharedExpertDecodeScratch,
    pub(crate) router: RouterScratch,
    pub(crate) marlin: MarlinExpertScratch,
    pub(crate) marlin_route_workspace: KimiMarlinRouteWorkspace,
    pub(crate) marlin_workspace: KimiMarlinWna16Workspace,
    pub(crate) comm: CommScratch,
    pub(crate) sampling: SamplingScratch,
}
