use anyhow::{Context, Result, ensure};
use openinfer_kernels::ops::{
    Glm52DeepGemmGroupedFp8Contract, Glm52DeepGemmGroupedFp8Kind, Glm52TrtllmGroupedFp8Contract,
    Glm52TrtllmGroupedFp8Kind, glm52_deepgemm_grouped_fp8_contract_validate,
    glm52_trtllm_grouped_fp8_contract_validate, glm52_trtllm_grouped_fp8_launch,
    glm52_trtllm_grouped_fp8_workspace_size,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::{
    arena::Glm52DecodeArena,
    config::{GLM52_EXPERT_INTERMEDIATE, GLM52_HIDDEN},
    deepep::GLM52_DEEPEP_EXPERT_ALIGNMENT,
    weights::{
        Glm52DeepGemmMGroupedFp8WeightPlan, Glm52MoeLayerExpertFp8Weights,
        Glm52RankExpertFp8Weights,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Glm52MoeGemmBackend {
    ExpandedDeepEpGroupedFp8Contract,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52MoeGemmOperandContract {
    pub(crate) groups: usize,
    pub(crate) m_capacity: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) weight_elems: usize,
    pub(crate) weight_scale_rows: usize,
    pub(crate) weight_scale_cols: usize,
    pub(crate) activation_scale_cols: usize,
    pub(crate) activation_scale_tma_rows: usize,
    pub(crate) activation_scale_trtllm_rows: usize,
    pub(crate) trtllm_workspace_bytes: usize,
    pub(crate) output_elems: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52MoeGemmContractReport {
    pub(crate) rank: usize,
    pub(crate) layer_count: usize,
    pub(crate) first_layer_idx: usize,
    pub(crate) last_layer_idx: usize,
    pub(crate) backend: Glm52MoeGemmBackend,
    pub(crate) psum_layout_entries: usize,
    pub(crate) expert_alignment: usize,
    pub(crate) w13: Glm52MoeGemmOperandContract,
    pub(crate) w2: Glm52MoeGemmOperandContract,
    pub(crate) graph_stable_arena: bool,
}

pub(crate) fn validate_moe_gemm_contract(
    rank: usize,
    expert_weights: &Glm52RankExpertFp8Weights,
    arena: &Glm52DecodeArena,
) -> Result<Glm52MoeGemmContractReport> {
    ensure!(
        expert_weights.rank == rank,
        "GLM5.2 MoE GEMM contract rank mismatch: weights rank {}, worker rank {rank}",
        expert_weights.rank
    );
    expert_weights.validate()?;
    arena.plan.validate()?;

    let first_layer_idx = expert_weights
        .layers
        .first()
        .with_context(|| format!("GLM5.2 rank {rank} has no routed expert package"))?
        .layer_idx;
    let last_layer_idx = expert_weights
        .layers
        .last()
        .expect("validated non-empty GLM5.2 expert package")
        .layer_idx;

    let mut w13 = None;
    let mut w2 = None;
    for layer in &expert_weights.layers {
        let (layer_w13, layer_w2) = validate_layer_plan(rank, layer, arena)?;
        w13 = Some(layer_w13);
        w2 = Some(layer_w2);
    }

    Ok(Glm52MoeGemmContractReport {
        rank,
        layer_count: expert_weights.layers.len(),
        first_layer_idx,
        last_layer_idx,
        backend: Glm52MoeGemmBackend::ExpandedDeepEpGroupedFp8Contract,
        psum_layout_entries: arena.plan.local_experts,
        expert_alignment: arena.plan.expert_alignment,
        w13: w13.expect("validated GLM5.2 W13 contract"),
        w2: w2.expect("validated GLM5.2 W2 contract"),
        graph_stable_arena: true,
    })
}

fn validate_layer_plan(
    rank: usize,
    layer: &Glm52MoeLayerExpertFp8Weights,
    arena: &Glm52DecodeArena,
) -> Result<(Glm52MoeGemmOperandContract, Glm52MoeGemmOperandContract)> {
    ensure!(
        arena.plan.expert_alignment == GLM52_DEEPEP_EXPERT_ALIGNMENT,
        "GLM5.2 rank {rank} MoE GEMM expert alignment drifted: {} != {}",
        arena.plan.expert_alignment,
        GLM52_DEEPEP_EXPERT_ALIGNMENT
    );
    ensure!(
        arena.moe_gemm_expert_offsets.len() == arena.plan.local_experts + 1
            && arena.moe_w13_problem_sizes.len() == arena.plan.local_experts * 3
            && arena.moe_w2_problem_sizes.len() == arena.plan.local_experts * 3,
        "GLM5.2 rank {rank} MoE GEMM metadata buffers drifted: offsets={}, w13={}, w2={}, local_experts={}",
        arena.moe_gemm_expert_offsets.len(),
        arena.moe_w13_problem_sizes.len(),
        arena.moe_w2_problem_sizes.len(),
        arena.plan.local_experts
    );

    let w13_plan = layer.w13.deepgemm_m_grouped_plan()?;
    let w2_plan = layer.down.deepgemm_m_grouped_plan()?;
    let w13 = validate_w13_contract(rank, arena, w13_plan)?;
    let w2 = validate_w2_contract(rank, arena, w2_plan)?;

    Ok((w13, w2))
}

fn validate_w13_contract(
    rank: usize,
    arena: &Glm52DecodeArena,
    plan: Glm52DeepGemmMGroupedFp8WeightPlan,
) -> Result<Glm52MoeGemmOperandContract> {
    ensure!(
        plan.groups == arena.plan.local_experts
            && plan.n == GLM52_EXPERT_INTERMEDIATE * 2
            && plan.k == GLM52_HIDDEN,
        "GLM5.2 rank {rank} W13 DeepGEMM plan does not match decode arena: plan={plan:?}, arena={:?}",
        arena.plan
    );
    ensure!(
        arena.moe_w13_input_fp8.len() >= arena.plan.deepep_worst_expanded_tokens * plan.k
            && arena.moe_w13_input_scale_tma.len()
                >= arena.plan.moe_w13_scale_tma_aligned_rows * arena.plan.moe_w13_scale_cols
            && arena.moe_w13_output_bf16.len() >= arena.plan.deepep_worst_expanded_tokens * plan.n,
        "GLM5.2 rank {rank} W13 DeepGEMM arena buffers are too small"
    );
    let mut contract = Glm52MoeGemmOperandContract {
        groups: plan.groups,
        m_capacity: arena.plan.deepep_worst_expanded_tokens,
        n: plan.n,
        k: plan.k,
        weight_elems: plan.weight_elems,
        weight_scale_rows: plan.scale_rows,
        weight_scale_cols: plan.scale_cols,
        activation_scale_cols: arena.plan.moe_w13_scale_cols,
        activation_scale_tma_rows: arena.plan.moe_w13_scale_tma_aligned_rows,
        activation_scale_trtllm_rows: arena.plan.moe_trtllm_grouped_offset_rows,
        trtllm_workspace_bytes: 0,
        output_elems: arena.plan.deepep_worst_expanded_tokens * plan.n,
    };
    contract.trtllm_workspace_bytes =
        validate_trtllm_kernel_abi_contract(Glm52TrtllmGroupedFp8Kind::W13, contract)?;
    validate_kernel_abi_contract(
        Glm52DeepGemmGroupedFp8Kind::W13,
        contract,
        arena.plan.local_experts,
        arena.plan.expert_alignment,
    )?;
    Ok(contract)
}

fn validate_w2_contract(
    rank: usize,
    arena: &Glm52DecodeArena,
    plan: Glm52DeepGemmMGroupedFp8WeightPlan,
) -> Result<Glm52MoeGemmOperandContract> {
    ensure!(
        plan.groups == arena.plan.local_experts
            && plan.n == GLM52_HIDDEN
            && plan.k == GLM52_EXPERT_INTERMEDIATE,
        "GLM5.2 rank {rank} W2 DeepGEMM plan does not match decode arena: plan={plan:?}, arena={:?}",
        arena.plan
    );
    ensure!(
        arena.moe_w2_input_fp8.len() >= arena.plan.deepep_worst_expanded_tokens * plan.k
            && arena.moe_w2_input_scale_tma.len()
                >= arena.plan.moe_w2_scale_tma_aligned_rows * arena.plan.moe_w2_scale_cols
            && arena.moe_w2_output_bf16.len() >= arena.plan.deepep_worst_expanded_tokens * plan.n,
        "GLM5.2 rank {rank} W2 DeepGEMM arena buffers are too small"
    );
    let mut contract = Glm52MoeGemmOperandContract {
        groups: plan.groups,
        m_capacity: arena.plan.deepep_worst_expanded_tokens,
        n: plan.n,
        k: plan.k,
        weight_elems: plan.weight_elems,
        weight_scale_rows: plan.scale_rows,
        weight_scale_cols: plan.scale_cols,
        activation_scale_cols: arena.plan.moe_w2_scale_cols,
        activation_scale_tma_rows: arena.plan.moe_w2_scale_tma_aligned_rows,
        activation_scale_trtllm_rows: arena.plan.moe_trtllm_grouped_offset_rows,
        trtllm_workspace_bytes: 0,
        output_elems: arena.plan.deepep_worst_expanded_tokens * plan.n,
    };
    contract.trtllm_workspace_bytes =
        validate_trtllm_kernel_abi_contract(Glm52TrtllmGroupedFp8Kind::W2, contract)?;
    validate_kernel_abi_contract(
        Glm52DeepGemmGroupedFp8Kind::W2,
        contract,
        arena.plan.local_experts,
        arena.plan.expert_alignment,
    )?;
    Ok(contract)
}

fn validate_kernel_abi_contract(
    kind: Glm52DeepGemmGroupedFp8Kind,
    contract: Glm52MoeGemmOperandContract,
    psum_entries: usize,
    expert_alignment: usize,
) -> Result<()> {
    glm52_deepgemm_grouped_fp8_contract_validate(
        kind,
        Glm52DeepGemmGroupedFp8Contract {
            groups: contract.groups,
            m_capacity: contract.m_capacity,
            n: contract.n,
            k: contract.k,
            weight_scale_rows: contract.weight_scale_rows,
            weight_scale_cols: contract.weight_scale_cols,
            activation_scale_cols: contract.activation_scale_cols,
            activation_scale_tma_rows: contract.activation_scale_tma_rows,
            psum_entries,
            expert_alignment,
        },
    )
}

fn validate_trtllm_kernel_abi_contract(
    kind: Glm52TrtllmGroupedFp8Kind,
    contract: Glm52MoeGemmOperandContract,
) -> Result<usize> {
    let trtllm_contract = Glm52TrtllmGroupedFp8Contract {
        groups: contract.groups,
        m_capacity: contract.m_capacity,
        n: contract.n,
        k: contract.k,
        weight_scale_rows: contract.weight_scale_rows,
        weight_scale_cols: contract.weight_scale_cols,
        activation_scale_cols: contract.activation_scale_cols,
        activation_scale_trtllm_rows: contract.activation_scale_trtllm_rows,
    };
    glm52_trtllm_grouped_fp8_contract_validate(kind, trtllm_contract)?;
    glm52_trtllm_grouped_fp8_workspace_size(kind, trtllm_contract)
}

pub(crate) fn launch_trtllm_w13_grouped_fp8(
    ctx: &DeviceContext,
    rank: usize,
    layer: &Glm52MoeLayerExpertFp8Weights,
    arena: &mut Glm52DecodeArena,
) -> Result<()> {
    let contract = validate_w13_contract(rank, arena, layer.w13.deepgemm_m_grouped_plan()?)?;
    glm52_trtllm_grouped_fp8_launch(
        ctx,
        Glm52TrtllmGroupedFp8Kind::W13,
        trtllm_contract_from_operand(contract),
        &arena.moe_w13_input_fp8,
        &arena.moe_w13_input_scale_trtllm_offset_tma,
        &layer.w13.weight_e4m3,
        &layer.w13.weight_scale_inv_f32,
        &arena.moe_gemm_expert_offsets,
        &mut arena.moe_w13_output_bf16,
    )
    .with_context(|| {
        format!(
            "GLM5.2 rank {rank} layer {} TRTLLM W13 grouped FP8 launch",
            layer.layer_idx
        )
    })
}

pub(crate) fn launch_trtllm_w2_grouped_fp8(
    ctx: &DeviceContext,
    rank: usize,
    layer: &Glm52MoeLayerExpertFp8Weights,
    arena: &mut Glm52DecodeArena,
) -> Result<()> {
    let contract = validate_w2_contract(rank, arena, layer.down.deepgemm_m_grouped_plan()?)?;
    glm52_trtllm_grouped_fp8_launch(
        ctx,
        Glm52TrtllmGroupedFp8Kind::W2,
        trtllm_contract_from_operand(contract),
        &arena.moe_w2_input_fp8,
        &arena.moe_w2_input_scale_trtllm_offset_tma,
        &layer.down.weight_e4m3,
        &layer.down.weight_scale_inv_f32,
        &arena.moe_gemm_expert_offsets,
        &mut arena.moe_w2_output_bf16,
    )
    .with_context(|| {
        format!(
            "GLM5.2 rank {rank} layer {} TRTLLM W2 grouped FP8 launch",
            layer.layer_idx
        )
    })
}

fn trtllm_contract_from_operand(
    contract: Glm52MoeGemmOperandContract,
) -> Glm52TrtllmGroupedFp8Contract {
    Glm52TrtllmGroupedFp8Contract {
        groups: contract.groups,
        m_capacity: contract.m_capacity,
        n: contract.n,
        k: contract.k,
        weight_scale_rows: contract.weight_scale_rows,
        weight_scale_cols: contract.weight_scale_cols,
        activation_scale_cols: contract.activation_scale_cols,
        activation_scale_trtllm_rows: contract.activation_scale_trtllm_rows,
    }
}
