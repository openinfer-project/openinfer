use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use openinfer_kernels::ops::Glm52DeepEpDispatchScratch;

use crate::{
    arena::Glm52DecodeArenaPlan,
    deepep::{GLM52_DEEPEP_EXPERT_ALIGNMENT, GLM52_EP_WORLD},
    weights::Glm52RankGpuContext,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DeepEpEnableReport {
    pub(crate) rank: usize,
    pub(crate) num_ranks: usize,
    pub(crate) decode_max_tokens_per_rank: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52MoePsumLayoutReport {
    pub(crate) rank: usize,
    pub(crate) local_experts: usize,
    pub(crate) expert_alignment: usize,
    pub(crate) recv_tokens: usize,
    pub(crate) active_experts: usize,
    pub(crate) expanded_rows: usize,
    pub(crate) empty_rank: bool,
    pub(crate) grouped_layout_valid: bool,
}

#[derive(Debug)]
pub(super) struct Glm52MoePsumLayoutSnapshot {
    pub(crate) report: Glm52MoePsumLayoutReport,
    pub(super) psum_expert: Vec<i32>,
}

pub(super) struct Glm52MoePsumLayout<'a> {
    rank: usize,
    plan: Glm52DecodeArenaPlan,
    psum_rank: &'a CudaSlice<i32>,
    psum_expert: &'a CudaSlice<i32>,
}

impl<'a> Glm52MoePsumLayout<'a> {
    pub(super) fn from_scratch(
        rank: usize,
        plan: Glm52DecodeArenaPlan,
        scratch: &'a Glm52DeepEpDispatchScratch,
    ) -> Result<Self> {
        ensure!(
            scratch.psum_rank.len() >= GLM52_EP_WORLD
                && scratch.psum_expert.len() >= plan.local_experts + 1,
            "GLM5.2 rank {rank} MoE psum layout scratch too small: psum_rank={}, psum_expert={}, local_experts={}",
            scratch.psum_rank.len(),
            scratch.psum_expert.len(),
            plan.local_experts
        );
        ensure!(
            plan.expert_alignment == GLM52_DEEPEP_EXPERT_ALIGNMENT,
            "GLM5.2 rank {rank} MoE psum layout alignment drifted: {} != {}",
            plan.expert_alignment,
            GLM52_DEEPEP_EXPERT_ALIGNMENT
        );
        Ok(Self {
            rank,
            plan,
            psum_rank: &scratch.psum_rank,
            psum_expert: &scratch.psum_expert,
        })
    }

    pub(super) fn snapshot(&self, ctx: &Glm52RankGpuContext) -> Result<Glm52MoePsumLayoutSnapshot> {
        let stream = ctx.stream();
        let psum_rank = stream.clone_dtoh(&self.psum_rank.slice(0..GLM52_EP_WORLD))?;
        let psum_expert = stream.clone_dtoh(&self.psum_expert.slice(0..self.plan.local_experts))?;
        Glm52MoePsumLayoutSnapshot::new(self.rank, self.plan, psum_rank, psum_expert)
    }
}

impl Glm52MoePsumLayoutSnapshot {
    fn new(
        rank: usize,
        plan: Glm52DecodeArenaPlan,
        psum_rank: Vec<i32>,
        psum_expert: Vec<i32>,
    ) -> Result<Self> {
        ensure!(
            psum_rank.len() == GLM52_EP_WORLD && psum_expert.len() == plan.local_experts,
            "GLM5.2 rank {rank} MoE psum layout snapshot shape mismatch: psum_rank={}, psum_expert={}, local_experts={}",
            psum_rank.len(),
            psum_expert.len(),
            plan.local_experts
        );

        let recv_tokens = psum_rank.last().copied().unwrap_or_default();
        let mut grouped_layout_valid = recv_tokens >= 0;
        let recv_tokens = recv_tokens.max(0) as usize;
        grouped_layout_valid &= recv_tokens <= plan.deepep_worst_recv_tokens;
        grouped_layout_valid &= psum_rank.windows(2).all(|window| window[0] <= window[1]);

        let mut previous_end = 0usize;
        let mut active_experts = 0usize;
        for (expert, end) in psum_expert.iter().copied().enumerate() {
            grouped_layout_valid &= end >= 0;
            let end = end.max(0) as usize;
            let start = if expert == 0 {
                0
            } else {
                align_up(previous_end, plan.expert_alignment)
            };
            grouped_layout_valid &= end >= start;
            grouped_layout_valid &= end <= plan.deepep_worst_expanded_tokens;
            if end > start {
                active_experts += 1;
            }
            previous_end = end;
        }

        let expanded_rows = align_up(previous_end, plan.expert_alignment);
        grouped_layout_valid &= expanded_rows <= plan.deepep_worst_expanded_tokens;

        ensure!(
            grouped_layout_valid,
            "GLM5.2 rank {rank} DeepEP grouped layout invalid: psum_rank={psum_rank:?}, psum_expert={psum_expert:?}, worst_recv={}, worst_expanded={}",
            plan.deepep_worst_recv_tokens,
            plan.deepep_worst_expanded_tokens
        );

        Ok(Self {
            report: Glm52MoePsumLayoutReport {
                rank,
                local_experts: plan.local_experts,
                expert_alignment: plan.expert_alignment,
                recv_tokens,
                active_experts,
                expanded_rows,
                empty_rank: expanded_rows == 0,
                grouped_layout_valid,
            },
            psum_expert,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DeepEpSmokeReport {
    pub(crate) rank: usize,
    pub(crate) num_tokens: usize,
    pub(crate) topk: usize,
    pub(crate) hidden: usize,
    pub(crate) router_routes_valid: bool,
    pub(crate) router_weights_normalized: bool,
    pub(crate) grouped_layout: Glm52MoePsumLayoutReport,
    pub(crate) recv_quant: Option<Glm52MoeQuantSmokeReport>,
    pub(crate) gemm_metadata: Option<Glm52MoeGemmMetadataSmokeReport>,
    pub(crate) combined_zero: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52MoeGemmMetadataSmokeReport {
    pub(crate) rank: usize,
    pub(crate) local_experts: usize,
    pub(crate) active_experts: usize,
    pub(crate) expanded_rows: usize,
    pub(crate) offsets_valid: bool,
    pub(crate) w13_problem_sizes_valid: bool,
    pub(crate) w2_problem_sizes_valid: bool,
    pub(crate) deepgemm_block_m64_psum_compatible: bool,
    pub(crate) deepgemm_block_m128_psum_compatible: bool,
    pub(crate) trtllm_grouped_offset_scale_rows_required: usize,
    pub(crate) trtllm_grouped_offset_scale_rows_covered: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52MoeQuantSmokeReport {
    pub(crate) rank: usize,
    pub(crate) rows: usize,
    pub(crate) group_size: usize,
    pub(crate) route_weights_applied: bool,
    pub(crate) quant_ran: bool,
    pub(crate) hidden_quant_valid: bool,
    pub(crate) swiglu_quant_valid: bool,
    pub(crate) swiglu_weighted_scale_valid: bool,
    pub(crate) hidden_scale_layout_valid: bool,
    pub(crate) swiglu_scale_layout_valid: bool,
    pub(crate) trtllm_offset_scale_layout_ran: bool,
    pub(crate) trtllm_offset_scale_layout_valid: bool,
    pub(crate) trtllm_offset_scale_rows: usize,
    pub(crate) scale_layout_aligned_rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DecodeGraphSmokeReport {
    pub(crate) rank: usize,
    pub(crate) num_tokens: usize,
    pub(crate) fixed_bucket_tokens: usize,
    pub(crate) worst_expanded_rows: usize,
    pub(crate) router_routes_valid: bool,
    pub(crate) router_weights_normalized: bool,
    pub(crate) route_weights_applied: bool,
    pub(crate) swiglu_weighted_scale_valid: bool,
    pub(crate) trtllm_offset_scale_layout_valid: bool,
    pub(crate) moe_gemm_metadata_valid: bool,
    pub(crate) grouped_layout_valid: bool,
    pub(crate) w13_output_nonzero: bool,
    pub(crate) w2_output_nonzero: bool,
    pub(crate) combined_nonzero: bool,
    pub(crate) capture_and_first_launch_ok: bool,
    pub(crate) replay_ok: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52MoeGemmSmokeReport {
    pub(crate) rank: usize,
    pub(crate) num_tokens: usize,
    pub(crate) layer_idx: usize,
    pub(crate) router_routes_valid: bool,
    pub(crate) router_weights_normalized: bool,
    pub(crate) grouped_layout: Glm52MoePsumLayoutReport,
    pub(crate) gemm_metadata: Glm52MoeGemmMetadataSmokeReport,
    pub(crate) w13_output_nonzero: bool,
    pub(crate) w2_output_nonzero: bool,
    pub(crate) combined_nonzero: bool,
}

pub(super) fn deepgemm_psum_compatible(psum_expert: &[i32], block_m: usize) -> bool {
    let mut previous_end = 0usize;
    for raw_end in psum_expert.iter().copied() {
        if raw_end < 0 {
            return false;
        }
        let end = raw_end as usize;
        let start = align_up(previous_end, block_m);
        if end < start {
            return false;
        }
        previous_end = end;
    }
    true
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}
