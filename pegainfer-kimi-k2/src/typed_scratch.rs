//! Typed Kimi decode scratch buffers.

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use pegainfer_kernels::gpu_buffers;
use pegainfer_kernels::tensor::{DeviceContext, GpuTensor, HiddenStates};

use crate::config::{
    KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_Q_LORA_RANK, KIMI_K2_ROUTED_EXPERTS,
    KIMI_K2_TOPK, KIMI_K2_VOCAB, KimiLocalDims,
};
use pegainfer_kernels::ops::{
    KIMI_K2_EP_WORLD, KIMI_K2_MLA_KV_LORA_RANK, KIMI_K2_MLA_QKV_A_OUT, KIMI_K2_MLA_ROPE_DIM,
    KimiMarlinRouteWorkspace, KimiMarlinWna16Workspace, argmax_batch_bf16_split_partials_len,
};

pub(crate) const MARLIN_W13_OUT_DIM: usize = 2 * KIMI_K2_EXPERT_INTERMEDIATE;
const PROMPT_LEN1_MARLIN_BLOCK_SIZE: usize = 8;

pub(crate) struct MlaDecodeScratch {
    // TP-independent (global model dims)
    pub(crate) hidden: GpuTensor<KIMI_K2_HIDDEN>,
    pub(crate) normed: GpuTensor<KIMI_K2_HIDDEN>,
    pub(crate) projected: GpuTensor<KIMI_K2_HIDDEN>,
    pub(crate) qkv_a: GpuTensor<KIMI_K2_MLA_QKV_A_OUT>,
    pub(crate) q_a: GpuTensor<KIMI_K2_Q_LORA_RANK>,
    pub(crate) q_a_normed: GpuTensor<KIMI_K2_Q_LORA_RANK>,
    pub(crate) compressed_kv: GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
    pub(crate) k_rope: GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    pub(crate) compressed_normed: GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
    pub(crate) append_kpe: GpuTensor<KIMI_K2_MLA_ROPE_DIM>,

    // TP-dependent (local_heads × per-head dim)
    pub(crate) q_proj: HiddenStates,
    pub(crate) q_nope: HiddenStates,
    pub(crate) q_pe: HiddenStates,
    pub(crate) q_abs_nope: HiddenStates,
    pub(crate) kv_b: HiddenStates,
    pub(crate) latent: HiddenStates,
    pub(crate) attn_out: HiddenStates,
}

impl MlaDecodeScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        batch_size: usize,
        dims: &KimiLocalDims,
    ) -> Result<Self> {
        Ok(Self {
            hidden: GpuTensor::zeros(ctx, batch_size)?,
            normed: GpuTensor::zeros(ctx, batch_size)?,
            projected: GpuTensor::zeros(ctx, batch_size)?,
            qkv_a: GpuTensor::zeros(ctx, batch_size)?,
            q_a: GpuTensor::zeros(ctx, batch_size)?,
            q_a_normed: GpuTensor::zeros(ctx, batch_size)?,
            compressed_kv: GpuTensor::zeros(ctx, batch_size)?,
            k_rope: GpuTensor::zeros(ctx, batch_size)?,
            compressed_normed: GpuTensor::zeros(ctx, batch_size)?,
            append_kpe: GpuTensor::zeros(ctx, batch_size)?,
            q_proj: HiddenStates::zeros(ctx, dims.q_proj_out, batch_size)?,
            q_nope: HiddenStates::zeros(ctx, dims.q_nope_out, batch_size)?,
            q_pe: HiddenStates::zeros(ctx, dims.q_pe_out, batch_size)?,
            q_abs_nope: HiddenStates::zeros(ctx, dims.abs_q_out, batch_size)?,
            kv_b: HiddenStates::zeros(ctx, dims.kv_b_out, batch_size)?,
            latent: HiddenStates::zeros(ctx, dims.abs_q_out, batch_size)?,
            attn_out: HiddenStates::zeros(ctx, dims.o_proj_in, batch_size)?,
        })
    }
}

pub(crate) struct DenseMlpDecodeScratch {
    pub(crate) gate_up: HiddenStates,
    pub(crate) activated: HiddenStates,
}

impl DenseMlpDecodeScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        batch_size: usize,
        dims: &KimiLocalDims,
    ) -> Result<Self> {
        Ok(Self {
            gate_up: HiddenStates::zeros(ctx, dims.dense_gate_up, batch_size)?,
            activated: HiddenStates::zeros(ctx, dims.dense_activated, batch_size)?,
        })
    }
}

pub(crate) struct SharedExpertDecodeScratch {
    pub(crate) gate_up: HiddenStates,
    pub(crate) activated: HiddenStates,
}

impl SharedExpertDecodeScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        batch_size: usize,
        dims: &KimiLocalDims,
    ) -> Result<Self> {
        Ok(Self {
            gate_up: HiddenStates::zeros(ctx, dims.shared_gate_up, batch_size)?,
            activated: HiddenStates::zeros(ctx, dims.shared_activated, batch_size)?,
        })
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

pub(crate) struct PromptLen1MoeScratch {
    pub(crate) route_workspace: KimiMarlinRouteWorkspace,
    pub(crate) marlin_workspace: KimiMarlinWna16Workspace,
}

impl PromptLen1MoeScratch {
    pub(crate) fn new(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        let route_workspace =
            KimiMarlinRouteWorkspace::new(ctx, batch_size, PROMPT_LEN1_MARLIN_BLOCK_SIZE)?;
        let marlin_workspace = KimiMarlinWna16Workspace::new(
            ctx,
            route_workspace.max_m_blocks,
            KIMI_K2_HIDDEN,
            PROMPT_LEN1_MARLIN_BLOCK_SIZE,
        )?;
        Ok(Self {
            route_workspace,
            marlin_workspace,
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
    pub(crate) top1_partial_values: CudaSlice<f32>,
    pub(crate) top1_partial_indices: CudaSlice<i32>,
}

impl SamplingScratch {
    pub(crate) fn new(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        let partials = argmax_batch_bf16_split_partials_len(batch_size, KIMI_K2_VOCAB);
        Ok(Self {
            top1_value_scratch: ctx.stream.alloc_zeros(batch_size)?,
            top1_out: ctx.stream.alloc_zeros(batch_size)?,
            top1_partial_values: ctx.stream.alloc_zeros(partials)?,
            top1_partial_indices: ctx.stream.alloc_zeros(partials)?,
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
    pub(crate) prompt_len1_moe: PromptLen1MoeScratch,
    pub(crate) comm: CommScratch,
    pub(crate) sampling: SamplingScratch,
}

impl KimiWorkerDecodeScratch {
    pub(crate) fn set_prompt_seq_len(&mut self, seq_len: usize) -> Result<()> {
        set_gpu_tensor_seq_len("mla.hidden", &mut self.mla.hidden, seq_len)?;
        set_gpu_tensor_seq_len("mla.normed", &mut self.mla.normed, seq_len)?;
        set_gpu_tensor_seq_len("mla.projected", &mut self.mla.projected, seq_len)?;
        set_gpu_tensor_seq_len("mla.qkv_a", &mut self.mla.qkv_a, seq_len)?;
        set_gpu_tensor_seq_len("mla.q_a", &mut self.mla.q_a, seq_len)?;
        set_gpu_tensor_seq_len("mla.q_a_normed", &mut self.mla.q_a_normed, seq_len)?;
        set_gpu_tensor_seq_len("mla.compressed_kv", &mut self.mla.compressed_kv, seq_len)?;
        set_gpu_tensor_seq_len("mla.k_rope", &mut self.mla.k_rope, seq_len)?;
        set_gpu_tensor_seq_len(
            "mla.compressed_normed",
            &mut self.mla.compressed_normed,
            seq_len,
        )?;
        set_gpu_tensor_seq_len("mla.append_kpe", &mut self.mla.append_kpe, seq_len)?;
        set_hidden_states_seq_len("mla.q_proj", &mut self.mla.q_proj, seq_len)?;
        set_hidden_states_seq_len("mla.q_nope", &mut self.mla.q_nope, seq_len)?;
        set_hidden_states_seq_len("mla.q_pe", &mut self.mla.q_pe, seq_len)?;
        set_hidden_states_seq_len("mla.q_abs_nope", &mut self.mla.q_abs_nope, seq_len)?;
        set_hidden_states_seq_len("mla.kv_b", &mut self.mla.kv_b, seq_len)?;
        set_hidden_states_seq_len("mla.latent", &mut self.mla.latent, seq_len)?;
        set_hidden_states_seq_len("mla.attn_out", &mut self.mla.attn_out, seq_len)?;
        set_hidden_states_seq_len("dense_mlp.gate_up", &mut self.dense_mlp.gate_up, seq_len)?;
        set_hidden_states_seq_len(
            "dense_mlp.activated",
            &mut self.dense_mlp.activated,
            seq_len,
        )?;
        set_hidden_states_seq_len(
            "shared_expert.gate_up",
            &mut self.shared_expert.gate_up,
            seq_len,
        )?;
        set_hidden_states_seq_len(
            "shared_expert.activated",
            &mut self.shared_expert.activated,
            seq_len,
        )?;
        self.router.set_batch_size(seq_len);
        let route_elems = seq_len
            .checked_mul(KIMI_K2_TOPK)
            .ok_or_else(|| anyhow::anyhow!("Kimi prompt route elem count overflow"))?;
        set_gpu_tensor_seq_len("marlin.w13_out", &mut self.marlin.w13_out, route_elems)?;
        set_gpu_tensor_seq_len("marlin.activated", &mut self.marlin.activated, route_elems)?;
        set_gpu_tensor_seq_len(
            "marlin.expert_output",
            &mut self.marlin.expert_output,
            route_elems,
        )?;
        Ok(())
    }
    pub(crate) fn set_moe_seq_len(&mut self, seq_len: usize) -> Result<()> {
        set_gpu_tensor_seq_len("mla.hidden", &mut self.mla.hidden, seq_len)?;
        set_gpu_tensor_seq_len("mla.normed", &mut self.mla.normed, seq_len)?;
        set_gpu_tensor_seq_len("mla.projected", &mut self.mla.projected, seq_len)?;
        set_hidden_states_seq_len(
            "shared_expert.gate_up",
            &mut self.shared_expert.gate_up,
            seq_len,
        )?;
        set_hidden_states_seq_len(
            "shared_expert.activated",
            &mut self.shared_expert.activated,
            seq_len,
        )?;
        Ok(())
    }
}

fn set_gpu_tensor_seq_len<const DIM: usize>(
    name: &str,
    tensor: &mut GpuTensor<DIM>,
    seq_len: usize,
) -> Result<()> {
    ensure!(
        seq_len > 0 && seq_len * DIM <= tensor.data.len(),
        "{name} seq_len {seq_len} exceeds storage rows {}",
        tensor.data.len() / DIM
    );
    tensor.seq_len = seq_len;
    Ok(())
}

fn set_hidden_states_seq_len(name: &str, states: &mut HiddenStates, seq_len: usize) -> Result<()> {
    ensure!(
        seq_len > 0 && seq_len * states.hidden_dim <= states.data.len(),
        "{name} seq_len {seq_len} exceeds storage rows {}",
        states.data.len() / states.hidden_dim
    );
    states.seq_len = seq_len;
    Ok(())
}
