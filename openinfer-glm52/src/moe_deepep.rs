//! GLM5.2 DeepEP runtime substrate.
//!
//! This module mirrors Kimi's TP1/DP8 DeepEP bootstrap: rank 0 creates one
//! NCCL unique id, every rank worker enters context creation concurrently, and
//! the context plus decode scratch stay owned by that rank thread. It does not
//! run MoE yet; it makes the communication context real and shape-checked.

use anyhow::{Context, Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_kernels::{
    ops::{
        GLM52_MOE_QUANT_GROUP_SIZE, Glm52DeepEp, Glm52DeepEpDispatchScratch,
        Glm52DeepGemmScaleLayout, Glm52MoeQuantShape, Glm52RouterBatch, Glm52RouterConfig,
        Glm52RouterOutput, Glm52TrtllmGroupedOffsetScaleLayout, glm52_deepep_info,
        glm52_deepep_unique_id, glm52_deepgemm_grouped_fp8_metadata_launch,
        glm52_deepgemm_grouped_offset_tma_aligned_f32_launch,
        glm52_deepgemm_mn_major_tma_aligned_f32_launch,
        glm52_fp8_per_token_group_quant_bf16_launch, glm52_router_noaux_tc_launch,
        glm52_silu_and_mul_per_token_group_quant_bf16_launch,
        glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch,
    },
    tensor::DeviceContext,
};

use crate::{
    arena::Glm52DecodeArena,
    deepep::{
        GLM52_DEEPEP_DECODE_BATCH_CAP, GLM52_DEEPEP_EXPERT_ALIGNMENT, GLM52_EP_WORLD,
        GLM52_LOCAL_EXPERTS, Glm52DeepEpShape,
    },
    weights::{Glm52MoeLayerExpertFp8Weights, Glm52RankGpuContext, Glm52RouterGpuWeights},
};

mod types;

pub(crate) use types::{
    Glm52DecodeGraphSmokeReport, Glm52DeepEpEnableReport, Glm52DeepEpSmokeReport,
    Glm52MoeGemmMetadataSmokeReport, Glm52MoeGemmSmokeReport, Glm52MoePsumLayoutReport,
    Glm52MoeQuantSmokeReport,
};

use types::{Glm52MoePsumLayout, Glm52MoePsumLayoutSnapshot, deepgemm_psum_compatible};

pub(crate) fn unique_id() -> Result<[u8; 128]> {
    glm52_deepep_unique_id()
}

pub(crate) fn decode_moe_quant_smoke(
    ctx: &Glm52RankGpuContext,
    rank: usize,
    arena: &mut Glm52DecodeArena,
    rows: usize,
) -> Result<Glm52MoeQuantSmokeReport> {
    ctx.set_current()?;
    arena.seed_moe_quant_smoke_inputs(ctx, rows)?;
    run_moe_quant_smoke(ctx, rank, arena, rows, false, false)
}

fn decode_moe_quant_from_deepep_recv_smoke(
    ctx: &Glm52RankGpuContext,
    rank: usize,
    arena: &mut Glm52DecodeArena,
    layout: &Glm52MoePsumLayoutSnapshot,
) -> Result<Glm52MoeQuantSmokeReport> {
    let rows = layout.report.expanded_rows;
    if rows == 0 {
        return Ok(Glm52MoeQuantSmokeReport {
            rank,
            rows,
            group_size: GLM52_MOE_QUANT_GROUP_SIZE,
            route_weights_applied: false,
            quant_ran: false,
            hidden_quant_valid: true,
            swiglu_quant_valid: true,
            swiglu_weighted_scale_valid: true,
            hidden_scale_layout_valid: true,
            swiglu_scale_layout_valid: true,
            trtllm_offset_scale_layout_ran: false,
            trtllm_offset_scale_layout_valid: true,
            trtllm_offset_scale_rows: 0,
            scale_layout_aligned_rows: 0,
        });
    }
    arena.seed_moe_w13_output_smoke(ctx, rows)?;
    run_moe_quant_smoke(ctx, rank, arena, rows, true, true)
}

fn run_moe_quant_smoke(
    ctx: &Glm52RankGpuContext,
    rank: usize,
    arena: &mut Glm52DecodeArena,
    rows: usize,
    route_weights_applied: bool,
    trtllm_offset_scale_layout: bool,
) -> Result<Glm52MoeQuantSmokeReport> {
    let device_ctx = ctx.as_device_context();
    let (hidden_scale_layout, _) = launch_moe_quant_substrate(
        &device_ctx,
        rank,
        arena,
        rows,
        route_weights_applied,
        trtllm_offset_scale_layout,
    )?;
    ctx.sync()
        .with_context(|| format!("GLM5.2 rank {rank} MoE quant smoke sync"))?;
    let (hidden_quant_valid, swiglu_quant_valid) =
        arena.validate_moe_quant_smoke_outputs(ctx, rows)?;
    let swiglu_weighted_scale_valid = if route_weights_applied {
        arena.validate_weighted_swiglu_scale_output(ctx, rows)?
    } else {
        true
    };
    let (hidden_scale_layout_valid, swiglu_scale_layout_valid) =
        arena.validate_deepgemm_scale_layout_outputs(ctx, rows)?;
    let (trtllm_w13_scale_valid, trtllm_w2_scale_valid) = if trtllm_offset_scale_layout {
        arena.validate_trtllm_grouped_offset_scale_layout_outputs(ctx)?
    } else {
        (true, true)
    };
    let trtllm_offset_scale_layout_valid = trtllm_w13_scale_valid && trtllm_w2_scale_valid;
    ensure!(
        hidden_quant_valid
            && swiglu_quant_valid
            && swiglu_weighted_scale_valid
            && hidden_scale_layout_valid
            && swiglu_scale_layout_valid
            && trtllm_offset_scale_layout_valid,
        "GLM5.2 rank {rank} MoE quant smoke failed: hidden_quant_valid={hidden_quant_valid}, swiglu_quant_valid={swiglu_quant_valid}, swiglu_weighted_scale_valid={swiglu_weighted_scale_valid}, hidden_scale_layout_valid={hidden_scale_layout_valid}, swiglu_scale_layout_valid={swiglu_scale_layout_valid}, trtllm_offset_scale_layout_valid={trtllm_offset_scale_layout_valid}"
    );
    Ok(Glm52MoeQuantSmokeReport {
        rank,
        rows,
        group_size: GLM52_MOE_QUANT_GROUP_SIZE,
        route_weights_applied,
        quant_ran: true,
        hidden_quant_valid,
        swiglu_quant_valid,
        swiglu_weighted_scale_valid,
        hidden_scale_layout_valid,
        swiglu_scale_layout_valid,
        trtllm_offset_scale_layout_ran: trtllm_offset_scale_layout,
        trtllm_offset_scale_layout_valid,
        trtllm_offset_scale_rows: if trtllm_offset_scale_layout {
            arena.plan.moe_trtllm_grouped_offset_rows
        } else {
            0
        },
        scale_layout_aligned_rows: hidden_scale_layout.aligned_rows,
    })
}

fn launch_moe_quant_substrate(
    device_ctx: &DeviceContext,
    rank: usize,
    arena: &mut Glm52DecodeArena,
    rows: usize,
    route_weights_applied: bool,
    trtllm_offset_scale_layout: bool,
) -> Result<(Glm52DeepGemmScaleLayout, Glm52DeepGemmScaleLayout)> {
    let hidden_scale_layout = launch_w13_input_quant_substrate(
        device_ctx,
        rank,
        arena,
        rows,
        trtllm_offset_scale_layout,
    )?;
    let swiglu_scale_layout = launch_w2_input_quant_substrate(
        device_ctx,
        rank,
        arena,
        rows,
        route_weights_applied,
        trtllm_offset_scale_layout,
    )?;
    Ok((hidden_scale_layout, swiglu_scale_layout))
}

fn launch_w13_input_quant_substrate(
    device_ctx: &DeviceContext,
    rank: usize,
    arena: &mut Glm52DecodeArena,
    rows: usize,
    trtllm_offset_scale_layout: bool,
) -> Result<Glm52DeepGemmScaleLayout> {
    glm52_fp8_per_token_group_quant_bf16_launch(
        device_ctx,
        Glm52MoeQuantShape {
            rows,
            width: arena.plan.hidden,
            group_size: GLM52_MOE_QUANT_GROUP_SIZE,
        },
        &arena.deepep_recv_x,
        &mut arena.moe_w13_input_fp8,
        &mut arena.moe_w13_input_scale,
    )
    .with_context(|| format!("GLM5.2 rank {rank} W13 input FP8 quant"))?;
    let hidden_scale_layout = Glm52DeepGemmScaleLayout::f32(rows, arena.plan.moe_w13_scale_cols);
    glm52_deepgemm_mn_major_tma_aligned_f32_launch(
        device_ctx,
        hidden_scale_layout,
        &arena.moe_w13_input_scale,
        &mut arena.moe_w13_input_scale_tma,
    )
    .with_context(|| format!("GLM5.2 rank {rank} W13 input DeepGEMM scale-layout"))?;
    if trtllm_offset_scale_layout {
        let w13_offset_layout = Glm52TrtllmGroupedOffsetScaleLayout::f32(
            arena.plan.deepep_worst_expanded_tokens,
            arena.plan.moe_w13_scale_cols,
            arena.plan.local_experts,
        );
        glm52_deepgemm_grouped_offset_tma_aligned_f32_launch(
            device_ctx,
            w13_offset_layout,
            &arena.moe_w13_input_scale,
            &arena.moe_gemm_expert_offsets,
            &mut arena.moe_w13_input_scale_trtllm_offset_tma,
        )
        .with_context(|| format!("GLM5.2 rank {rank} W13 TRTLLM offset scale-layout"))?;
    }
    Ok(hidden_scale_layout)
}

fn launch_w2_input_quant_substrate(
    device_ctx: &DeviceContext,
    rank: usize,
    arena: &mut Glm52DecodeArena,
    rows: usize,
    route_weights_applied: bool,
    trtllm_offset_scale_layout: bool,
) -> Result<Glm52DeepGemmScaleLayout> {
    let swiglu_shape = Glm52MoeQuantShape {
        rows,
        width: arena.plan.moe_intermediate,
        group_size: GLM52_MOE_QUANT_GROUP_SIZE,
    };
    if route_weights_applied {
        glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch(
            device_ctx,
            swiglu_shape,
            &arena.moe_w13_output_bf16,
            &arena.deepep_recv_topk_weight,
            &mut arena.moe_w2_input_fp8,
            &mut arena.moe_w2_input_scale,
        )
        .with_context(|| format!("GLM5.2 rank {rank} weighted W2 input SiLU FP8 quant"))?;
    } else {
        glm52_silu_and_mul_per_token_group_quant_bf16_launch(
            device_ctx,
            swiglu_shape,
            &arena.moe_w13_output_bf16,
            &mut arena.moe_w2_input_fp8,
            &mut arena.moe_w2_input_scale,
        )
        .with_context(|| format!("GLM5.2 rank {rank} W2 input SiLU FP8 quant"))?;
    }
    let swiglu_scale_layout = Glm52DeepGemmScaleLayout::f32(rows, arena.plan.moe_w2_scale_cols);
    glm52_deepgemm_mn_major_tma_aligned_f32_launch(
        device_ctx,
        swiglu_scale_layout,
        &arena.moe_w2_input_scale,
        &mut arena.moe_w2_input_scale_tma,
    )
    .with_context(|| format!("GLM5.2 rank {rank} W2 input DeepGEMM scale-layout"))?;
    if trtllm_offset_scale_layout {
        let w2_offset_layout = Glm52TrtllmGroupedOffsetScaleLayout::f32(
            arena.plan.deepep_worst_expanded_tokens,
            arena.plan.moe_w2_scale_cols,
            arena.plan.local_experts,
        );
        glm52_deepgemm_grouped_offset_tma_aligned_f32_launch(
            device_ctx,
            w2_offset_layout,
            &arena.moe_w2_input_scale,
            &arena.moe_gemm_expert_offsets,
            &mut arena.moe_w2_input_scale_trtllm_offset_tma,
        )
        .with_context(|| format!("GLM5.2 rank {rank} W2 TRTLLM offset scale-layout"))?;
    }
    Ok(swiglu_scale_layout)
}

fn launch_moe_gemm_metadata_substrate(
    device_ctx: &DeviceContext,
    rank: usize,
    arena: &mut Glm52DecodeArena,
    psum_expert: &CudaSlice<i32>,
) -> Result<()> {
    glm52_deepgemm_grouped_fp8_metadata_launch(
        device_ctx,
        psum_expert,
        &mut arena.moe_gemm_expert_offsets,
        &mut arena.moe_w13_problem_sizes,
        &mut arena.moe_w2_problem_sizes,
    )
    .with_context(|| format!("GLM5.2 rank {rank} MoE GEMM metadata launch"))
}

fn validate_moe_gemm_metadata_smoke(
    ctx: &Glm52RankGpuContext,
    rank: usize,
    arena: &Glm52DecodeArena,
    layout: &Glm52MoePsumLayoutSnapshot,
) -> Result<Glm52MoeGemmMetadataSmokeReport> {
    let validation = arena.validate_moe_gemm_metadata_outputs(ctx, &layout.psum_expert)?;
    let deepgemm_block_m64_psum_compatible = deepgemm_psum_compatible(&layout.psum_expert, 64);
    let deepgemm_block_m128_psum_compatible = deepgemm_psum_compatible(&layout.psum_expert, 128);
    ensure!(
        validation.offsets_valid
            && validation.w13_problem_sizes_valid
            && validation.w2_problem_sizes_valid
            && validation.active_experts == layout.report.active_experts
            && validation.expanded_rows == layout.report.expanded_rows
            && deepgemm_block_m64_psum_compatible,
        "GLM5.2 rank {rank} MoE GEMM metadata smoke failed: validation={validation:?}, layout={:?}, deepgemm_block_m64_psum_compatible={deepgemm_block_m64_psum_compatible}",
        layout.report
    );
    Ok(Glm52MoeGemmMetadataSmokeReport {
        rank,
        local_experts: layout.report.local_experts,
        active_experts: validation.active_experts,
        expanded_rows: validation.expanded_rows,
        offsets_valid: validation.offsets_valid,
        w13_problem_sizes_valid: validation.w13_problem_sizes_valid,
        w2_problem_sizes_valid: validation.w2_problem_sizes_valid,
        deepgemm_block_m64_psum_compatible,
        deepgemm_block_m128_psum_compatible,
        trtllm_grouped_offset_scale_rows_required: validation
            .trtllm_grouped_offset_scale_rows_required,
        trtllm_grouped_offset_scale_rows_covered: validation
            .trtllm_grouped_offset_scale_rows_covered,
    })
}

pub(crate) struct Glm52MoeDeepEpState {
    ep: Glm52DeepEp,
    scratch: Glm52DeepEpDispatchScratch,
    report: Glm52DeepEpEnableReport,
}

impl Glm52MoeDeepEpState {
    pub(crate) fn new(
        ctx: &DeviceContext,
        unique_id: &[u8; 128],
        num_ranks: usize,
        rank_idx: usize,
    ) -> Result<Self> {
        let info = glm52_deepep_info();
        let shape = Glm52DeepEpShape::tp1_dp8_h200();
        ensure!(
            info.num_ranks as usize == shape.ep_world
                && info.num_experts as usize == shape.routed_experts
                && info.num_local_experts as usize == shape.local_experts
                && info.num_topk as usize == shape.topk
                && info.hidden as usize == shape.hidden,
            "GLM5.2 DeepEP shim config does not match model shape: info={info:?}, shape={shape:?}"
        );
        ensure!(
            info.expert_alignment as usize == GLM52_DEEPEP_EXPERT_ALIGNMENT,
            "GLM5.2 DeepEP expert_alignment {} does not match {}",
            info.expert_alignment,
            GLM52_DEEPEP_EXPERT_ALIGNMENT
        );
        ensure!(
            info.decode_max_tokens_per_rank as usize == GLM52_DEEPEP_DECODE_BATCH_CAP,
            "GLM5.2 DeepEP decode cap drifted: {}",
            info.decode_max_tokens_per_rank
        );
        ensure!(
            num_ranks == GLM52_EP_WORLD,
            "GLM5.2 DeepEP requires {GLM52_EP_WORLD} ranks, got {num_ranks}"
        );
        ensure!(
            rank_idx < GLM52_EP_WORLD,
            "GLM5.2 DeepEP rank {rank_idx} out of EP{GLM52_EP_WORLD}"
        );
        ensure!(
            info.num_local_experts as usize == GLM52_LOCAL_EXPERTS,
            "GLM5.2 local expert count drifted: {} != {}",
            info.num_local_experts,
            GLM52_LOCAL_EXPERTS
        );

        let ep = Glm52DeepEp::new(unique_id, num_ranks, rank_idx)
            .with_context(|| format!("GLM5.2 rank {rank_idx} DeepEP context create"))?;
        let scratch = Glm52DeepEpDispatchScratch::new_decode(ctx)
            .with_context(|| format!("GLM5.2 rank {rank_idx} DeepEP decode scratch"))?;
        Ok(Self {
            ep,
            scratch,
            report: Glm52DeepEpEnableReport {
                rank: rank_idx,
                num_ranks,
                decode_max_tokens_per_rank: info.decode_max_tokens_per_rank as usize,
            },
        })
    }

    pub(crate) fn report(&self) -> Glm52DeepEpEnableReport {
        let _ = (&self.ep, &self.scratch);
        self.report
    }

    pub(crate) fn decode_smoke_roundtrip(
        &mut self,
        ctx: &Glm52RankGpuContext,
        arena: &mut Glm52DecodeArena,
        num_tokens: usize,
    ) -> Result<Glm52DeepEpSmokeReport> {
        ensure!(
            self.report.decode_max_tokens_per_rank >= num_tokens && num_tokens > 0,
            "GLM5.2 DeepEP decode smoke tokens {num_tokens} out of 1..={}",
            self.report.decode_max_tokens_per_rank
        );
        ctx.set_current()?;
        arena.seed_deepep_decode_smoke_routes(ctx, num_tokens)?;
        let device_ctx = ctx.as_device_context();
        self.ep
            .decode_dispatch(
                &device_ctx,
                &arena.hidden,
                &arena.topk_idx,
                &arena.topk_weight,
                num_tokens,
                &mut self.scratch,
                &mut arena.deepep_recv_x,
                &mut arena.deepep_recv_topk_weight,
                &mut arena.deepep_recv_src_metadata,
            )
            .with_context(|| format!("GLM5.2 rank {} DeepEP smoke dispatch", self.report.rank))?;
        self.ep
            .decode_combine(
                &device_ctx,
                &arena.moe_w2_output_bf16,
                &self.scratch,
                &arena.deepep_recv_src_metadata,
                &arena.topk_idx,
                num_tokens,
                &mut arena.deepep_combined,
            )
            .with_context(|| format!("GLM5.2 rank {} DeepEP smoke combine", self.report.rank))?;
        ctx.sync()
            .with_context(|| format!("GLM5.2 rank {} DeepEP smoke sync", self.report.rank))?;
        let check_len = num_tokens * arena.plan.hidden;
        let combined = ctx
            .stream()
            .clone_dtoh(&arena.deepep_combined.slice(0..check_len))
            .with_context(|| format!("GLM5.2 rank {} DeepEP smoke D2H", self.report.rank))?;
        let grouped_layout = self.validate_grouped_layout(ctx, arena)?;
        let combined_zero = combined.iter().all(|value| *value == bf16::ZERO);
        ensure!(
            combined_zero,
            "GLM5.2 rank {} DeepEP smoke expected zero combined output",
            self.report.rank
        );
        Ok(Glm52DeepEpSmokeReport {
            rank: self.report.rank,
            num_tokens,
            topk: arena.plan.topk,
            hidden: arena.plan.hidden,
            router_routes_valid: true,
            router_weights_normalized: true,
            grouped_layout,
            recv_quant: None,
            gemm_metadata: None,
            combined_zero,
        })
    }

    pub(crate) fn decode_router_smoke_roundtrip(
        &mut self,
        ctx: &Glm52RankGpuContext,
        router: &Glm52RouterGpuWeights<'_>,
        arena: &mut Glm52DecodeArena,
        num_tokens: usize,
    ) -> Result<Glm52DeepEpSmokeReport> {
        ensure!(
            self.report.decode_max_tokens_per_rank >= num_tokens && num_tokens > 0,
            "GLM5.2 router smoke tokens {num_tokens} out of 1..={}",
            self.report.decode_max_tokens_per_rank
        );
        ctx.set_current()?;
        arena.seed_router_smoke_hidden(ctx, num_tokens)?;
        let device_ctx = ctx.as_device_context();
        {
            let mut output = Glm52RouterOutput {
                topk_weight: &mut arena.topk_weight,
                topk_idx: &mut arena.topk_idx,
            };
            glm52_router_noaux_tc_launch(
                &device_ctx,
                Glm52RouterConfig::glm52(),
                Glm52RouterBatch {
                    active_tokens: num_tokens,
                    padded_tokens: num_tokens,
                },
                &arena.hidden,
                &router.gate_weight.data,
                &router.e_score_correction_bias.data,
                &mut arena.router_logits,
                &mut output,
            )
            .with_context(|| format!("GLM5.2 rank {} router smoke", self.report.rank))?;
        }
        ctx.sync()
            .with_context(|| format!("GLM5.2 rank {} router smoke sync", self.report.rank))?;
        let (router_routes_valid, router_weights_normalized) =
            arena.validate_router_smoke_routes(ctx, num_tokens)?;
        ensure!(
            router_routes_valid && router_weights_normalized,
            "GLM5.2 rank {} router smoke produced invalid routes: routes_valid={}, weights_normalized={}",
            self.report.rank,
            router_routes_valid,
            router_weights_normalized
        );

        self.ep
            .decode_dispatch(
                &device_ctx,
                &arena.hidden,
                &arena.topk_idx,
                &arena.topk_weight,
                num_tokens,
                &mut self.scratch,
                &mut arena.deepep_recv_x,
                &mut arena.deepep_recv_topk_weight,
                &mut arena.deepep_recv_src_metadata,
            )
            .with_context(|| {
                format!(
                    "GLM5.2 rank {} router-route DeepEP smoke dispatch",
                    self.report.rank
                )
            })?;
        ctx.sync().with_context(|| {
            format!(
                "GLM5.2 rank {} router-route DeepEP dispatch sync",
                self.report.rank
            )
        })?;
        let grouped_snapshot = self.snapshot_grouped_layout(ctx, arena)?;
        let grouped_layout = grouped_snapshot.report;
        launch_moe_gemm_metadata_substrate(
            &device_ctx,
            self.report.rank,
            arena,
            &self.scratch.psum_expert,
        )?;
        let recv_quant = decode_moe_quant_from_deepep_recv_smoke(
            ctx,
            self.report.rank,
            arena,
            &grouped_snapshot,
        )?;
        let gemm_metadata =
            validate_moe_gemm_metadata_smoke(ctx, self.report.rank, arena, &grouped_snapshot)?;
        self.ep
            .decode_combine(
                &device_ctx,
                &arena.moe_w2_output_bf16,
                &self.scratch,
                &arena.deepep_recv_src_metadata,
                &arena.topk_idx,
                num_tokens,
                &mut arena.deepep_combined,
            )
            .with_context(|| {
                format!(
                    "GLM5.2 rank {} router-route DeepEP smoke combine",
                    self.report.rank
                )
            })?;
        ctx.sync().with_context(|| {
            format!(
                "GLM5.2 rank {} router-route DeepEP smoke sync",
                self.report.rank
            )
        })?;
        let check_len = num_tokens * arena.plan.hidden;
        let combined = ctx
            .stream()
            .clone_dtoh(&arena.deepep_combined.slice(0..check_len))
            .with_context(|| {
                format!(
                    "GLM5.2 rank {} router-route DeepEP smoke D2H",
                    self.report.rank
                )
            })?;
        let combined_zero = combined.iter().all(|value| *value == bf16::ZERO);
        ensure!(
            combined_zero,
            "GLM5.2 rank {} router-route DeepEP smoke expected zero combined output",
            self.report.rank
        );
        Ok(Glm52DeepEpSmokeReport {
            rank: self.report.rank,
            num_tokens,
            topk: arena.plan.topk,
            hidden: arena.plan.hidden,
            router_routes_valid,
            router_weights_normalized,
            grouped_layout,
            recv_quant: Some(recv_quant),
            gemm_metadata: Some(gemm_metadata),
            combined_zero,
        })
    }

    pub(crate) fn decode_moe_gemm_smoke_roundtrip(
        &mut self,
        ctx: &Glm52RankGpuContext,
        router: &Glm52RouterGpuWeights<'_>,
        layer: &Glm52MoeLayerExpertFp8Weights,
        arena: &mut Glm52DecodeArena,
        num_tokens: usize,
    ) -> Result<Glm52MoeGemmSmokeReport> {
        ensure!(
            self.report.decode_max_tokens_per_rank >= num_tokens && num_tokens > 0,
            "GLM5.2 MoE GEMM smoke tokens {num_tokens} out of 1..={}",
            self.report.decode_max_tokens_per_rank
        );
        ctx.set_current()?;
        arena.seed_router_smoke_hidden(ctx, num_tokens)?;
        let device_ctx = ctx.as_device_context();
        self.launch_decode_moe_layer_substrate(
            &device_ctx,
            router,
            layer,
            arena,
            num_tokens,
            "MoE GEMM smoke",
        )?;
        ctx.sync()
            .with_context(|| format!("GLM5.2 rank {} MoE GEMM smoke sync", self.report.rank))?;

        let (router_routes_valid, router_weights_normalized) =
            arena.validate_router_smoke_routes(ctx, num_tokens)?;
        let grouped_snapshot = self.snapshot_grouped_layout(ctx, arena)?;
        let grouped_layout = grouped_snapshot.report;
        let gemm_metadata =
            validate_moe_gemm_metadata_smoke(ctx, self.report.rank, arena, &grouped_snapshot)?;
        let (w13_output_nonzero, w2_output_nonzero) = arena.validate_moe_gemm_smoke_outputs(ctx)?;
        let check_len = num_tokens * arena.plan.hidden;
        let combined = ctx
            .stream()
            .clone_dtoh(&arena.deepep_combined.slice(0..check_len))
            .with_context(|| {
                format!(
                    "GLM5.2 rank {} MoE GEMM smoke combined D2H",
                    self.report.rank
                )
            })?;
        let combined_nonzero = grouped_layout.empty_rank
            || combined
                .iter()
                .any(|value| value.to_f32().is_finite() && *value != bf16::ZERO);
        ensure!(
            router_routes_valid
                && router_weights_normalized
                && grouped_layout.grouped_layout_valid
                && gemm_metadata.offsets_valid
                && gemm_metadata.w13_problem_sizes_valid
                && gemm_metadata.w2_problem_sizes_valid
                && w13_output_nonzero
                && w2_output_nonzero
                && combined_nonzero,
            "GLM5.2 rank {} MoE GEMM smoke failed: routes_valid={router_routes_valid}, weights_normalized={router_weights_normalized}, grouped_layout={grouped_layout:?}, gemm_metadata={gemm_metadata:?}, w13_output_nonzero={w13_output_nonzero}, w2_output_nonzero={w2_output_nonzero}, combined_nonzero={combined_nonzero}",
            self.report.rank
        );
        Ok(Glm52MoeGemmSmokeReport {
            rank: self.report.rank,
            num_tokens,
            layer_idx: layer.layer_idx,
            router_routes_valid,
            router_weights_normalized,
            grouped_layout,
            gemm_metadata,
            w13_output_nonzero,
            w2_output_nonzero,
            combined_nonzero,
        })
    }

    pub(crate) fn decode_graph_smoke_roundtrip(
        &mut self,
        ctx: &Glm52RankGpuContext,
        router: &Glm52RouterGpuWeights<'_>,
        layer: &Glm52MoeLayerExpertFp8Weights,
        arena: &mut Glm52DecodeArena,
        num_tokens: usize,
    ) -> Result<Glm52DecodeGraphSmokeReport> {
        ensure!(
            num_tokens == arena.plan.batch_capacity
                && num_tokens == self.report.decode_max_tokens_per_rank,
            "GLM5.2 decode graph smoke requires the fixed decode bucket: got {num_tokens}, arena={}, deepep={}",
            arena.plan.batch_capacity,
            self.report.decode_max_tokens_per_rank
        );
        ctx.set_current()?;
        let rows = arena.plan.deepep_worst_expanded_tokens;
        arena.seed_router_smoke_hidden(ctx, num_tokens)?;
        arena.seed_moe_quant_smoke_inputs(ctx, rows)?;
        ctx.sync().with_context(|| {
            format!(
                "GLM5.2 rank {} decode graph smoke seed sync",
                self.report.rank
            )
        })?;

        let device_ctx = ctx.as_device_context();
        let mut graph = CudaGraphState::new();
        graph
            .run_or_capture(&device_ctx, || {
                self.decode_graph_smoke_kernels(&device_ctx, router, layer, arena, num_tokens, rows)
            })
            .with_context(|| {
                format!(
                    "GLM5.2 rank {} decode graph smoke capture/first launch",
                    self.report.rank
                )
            })?;
        graph
            .run_or_capture(&device_ctx, || {
                self.decode_graph_smoke_kernels(&device_ctx, router, layer, arena, num_tokens, rows)
            })
            .with_context(|| {
                format!("GLM5.2 rank {} decode graph smoke replay", self.report.rank)
            })?;
        ctx.sync().with_context(|| {
            format!(
                "GLM5.2 rank {} decode graph smoke replay sync",
                self.report.rank
            )
        })?;

        let (router_routes_valid, router_weights_normalized) =
            arena.validate_router_smoke_routes(ctx, num_tokens)?;
        let swiglu_weighted_scale_valid = arena.validate_weighted_swiglu_scale_output(ctx, rows)?;
        let grouped_snapshot = self.snapshot_grouped_layout(ctx, arena)?;
        let grouped_layout = grouped_snapshot.report;
        let gemm_metadata =
            validate_moe_gemm_metadata_smoke(ctx, self.report.rank, arena, &grouped_snapshot)?;
        let (trtllm_w13_scale_valid, trtllm_w2_scale_valid) =
            arena.validate_trtllm_grouped_offset_scale_layout_outputs(ctx)?;
        let trtllm_offset_scale_layout_valid = trtllm_w13_scale_valid && trtllm_w2_scale_valid;
        let moe_gemm_metadata_valid = gemm_metadata.offsets_valid
            && gemm_metadata.w13_problem_sizes_valid
            && gemm_metadata.w2_problem_sizes_valid
            && gemm_metadata.deepgemm_block_m64_psum_compatible;
        let (w13_output_nonzero, w2_output_nonzero) = arena.validate_moe_gemm_smoke_outputs(ctx)?;
        let check_len = num_tokens * arena.plan.hidden;
        let combined = ctx
            .stream()
            .clone_dtoh(&arena.deepep_combined.slice(0..check_len))
            .with_context(|| {
                format!(
                    "GLM5.2 rank {} decode graph smoke combined D2H",
                    self.report.rank
                )
            })?;
        let combined_nonzero = grouped_layout.empty_rank
            || combined
                .iter()
                .any(|value| value.to_f32().is_finite() && *value != bf16::ZERO);
        ensure!(
            router_routes_valid
                && router_weights_normalized
                && swiglu_weighted_scale_valid
                && trtllm_offset_scale_layout_valid
                && moe_gemm_metadata_valid
                && grouped_layout.grouped_layout_valid
                && w13_output_nonzero
                && w2_output_nonzero
                && combined_nonzero,
            "GLM5.2 rank {} decode graph smoke validation failed: routes_valid={router_routes_valid}, weights_normalized={router_weights_normalized}, swiglu_weighted_scale_valid={swiglu_weighted_scale_valid}, trtllm_offset_scale_layout_valid={trtllm_offset_scale_layout_valid}, moe_gemm_metadata_valid={moe_gemm_metadata_valid}, grouped_layout={grouped_layout:?}, w13_output_nonzero={w13_output_nonzero}, w2_output_nonzero={w2_output_nonzero}, combined_nonzero={combined_nonzero}",
            self.report.rank
        );
        Ok(Glm52DecodeGraphSmokeReport {
            rank: self.report.rank,
            num_tokens,
            fixed_bucket_tokens: arena.plan.batch_capacity,
            worst_expanded_rows: rows,
            router_routes_valid,
            router_weights_normalized,
            route_weights_applied: true,
            swiglu_weighted_scale_valid,
            trtllm_offset_scale_layout_valid,
            moe_gemm_metadata_valid,
            grouped_layout_valid: grouped_layout.grouped_layout_valid,
            w13_output_nonzero,
            w2_output_nonzero,
            combined_nonzero,
            capture_and_first_launch_ok: true,
            replay_ok: true,
        })
    }

    fn launch_decode_moe_layer_substrate(
        &mut self,
        device_ctx: &DeviceContext,
        router: &Glm52RouterGpuWeights<'_>,
        layer: &Glm52MoeLayerExpertFp8Weights,
        arena: &mut Glm52DecodeArena,
        num_tokens: usize,
        label: &str,
    ) -> Result<()> {
        {
            let mut output = Glm52RouterOutput {
                topk_weight: &mut arena.topk_weight,
                topk_idx: &mut arena.topk_idx,
            };
            glm52_router_noaux_tc_launch(
                device_ctx,
                Glm52RouterConfig::glm52(),
                Glm52RouterBatch {
                    active_tokens: num_tokens,
                    padded_tokens: num_tokens,
                },
                &arena.hidden,
                &router.gate_weight.data,
                &router.e_score_correction_bias.data,
                &mut arena.router_logits,
                &mut output,
            )
            .with_context(|| format!("GLM5.2 rank {} {label} router", self.report.rank))?;
        }
        self.ep
            .decode_dispatch(
                device_ctx,
                &arena.hidden,
                &arena.topk_idx,
                &arena.topk_weight,
                num_tokens,
                &mut self.scratch,
                &mut arena.deepep_recv_x,
                &mut arena.deepep_recv_topk_weight,
                &mut arena.deepep_recv_src_metadata,
            )
            .with_context(|| format!("GLM5.2 rank {} {label} DeepEP dispatch", self.report.rank))?;
        launch_moe_gemm_metadata_substrate(
            device_ctx,
            self.report.rank,
            arena,
            &self.scratch.psum_expert,
        )?;
        launch_w13_input_quant_substrate(
            device_ctx,
            self.report.rank,
            arena,
            arena.plan.deepep_worst_expanded_tokens,
            true,
        )?;
        crate::moe_gemm::launch_trtllm_w13_grouped_fp8(device_ctx, self.report.rank, layer, arena)?;
        launch_w2_input_quant_substrate(
            device_ctx,
            self.report.rank,
            arena,
            arena.plan.deepep_worst_expanded_tokens,
            true,
            true,
        )?;
        crate::moe_gemm::launch_trtllm_w2_grouped_fp8(device_ctx, self.report.rank, layer, arena)?;
        self.ep
            .decode_combine(
                device_ctx,
                &arena.moe_w2_output_bf16,
                &self.scratch,
                &arena.deepep_recv_src_metadata,
                &arena.topk_idx,
                num_tokens,
                &mut arena.deepep_combined,
            )
            .with_context(|| format!("GLM5.2 rank {} {label} DeepEP combine", self.report.rank))?;
        Ok(())
    }

    fn decode_graph_smoke_kernels(
        &mut self,
        device_ctx: &DeviceContext,
        router: &Glm52RouterGpuWeights<'_>,
        layer: &Glm52MoeLayerExpertFp8Weights,
        arena: &mut Glm52DecodeArena,
        num_tokens: usize,
        _rows: usize,
    ) -> Result<()> {
        self.launch_decode_moe_layer_substrate(
            device_ctx,
            router,
            layer,
            arena,
            num_tokens,
            "decode graph smoke",
        )
    }

    fn validate_grouped_layout(
        &self,
        ctx: &Glm52RankGpuContext,
        arena: &Glm52DecodeArena,
    ) -> Result<Glm52MoePsumLayoutReport> {
        Ok(self.snapshot_grouped_layout(ctx, arena)?.report)
    }

    fn snapshot_grouped_layout(
        &self,
        ctx: &Glm52RankGpuContext,
        arena: &Glm52DecodeArena,
    ) -> Result<Glm52MoePsumLayoutSnapshot> {
        Glm52MoePsumLayout::from_scratch(self.report.rank, arena.plan, &self.scratch)?.snapshot(ctx)
    }
}
