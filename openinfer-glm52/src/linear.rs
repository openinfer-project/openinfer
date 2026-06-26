use anyhow::{Context, Result, ensure};
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_MOE_QUANT_GROUP_SIZE, Glm52MoeQuantShape, Glm52TrtllmFp8LinearContract,
    glm52_fp8_per_token_group_quant_bf16_launch, glm52_trtllm_fp8_linear_contract_validate,
    glm52_trtllm_fp8_linear_launch, glm52_trtllm_fp8_linear_workspace_size,
};

use crate::{
    arena::Glm52DecodeArena,
    weights::{Glm52AttentionGpuWeights, Glm52Fp8ProjectionGpuWeights, Glm52RankGpuContext},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52LinearSmokeReport {
    pub(crate) rank: usize,
    pub(crate) rows: usize,
    pub(crate) projections: Vec<Glm52ProjectionSmokeReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52ProjectionSmokeReport {
    pub(crate) name: &'static str,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) weight_scale_rows: usize,
    pub(crate) weight_scale_cols: usize,
    pub(crate) activation_scale_cols: usize,
    pub(crate) workspace_bytes: usize,
    pub(crate) activation_quant_valid: bool,
    pub(crate) output_nonzero: bool,
}

pub(crate) fn decode_attention_projection_smoke(
    ctx: &Glm52RankGpuContext,
    rank: usize,
    attention: &Glm52AttentionGpuWeights<'_>,
    arena: &mut Glm52DecodeArena,
    rows: usize,
) -> Result<Glm52LinearSmokeReport> {
    ctx.set_current()?;
    ensure!(
        (1..=arena.plan.batch_capacity).contains(&rows),
        "GLM5.2 attention projection smoke rows {rows} out of 1..={}",
        arena.plan.batch_capacity
    );

    let mut projections = Vec::with_capacity(7);
    projections.push(launch_projection_smoke(
        ctx,
        rank,
        "q_a",
        &attention.q_a_proj,
        rows,
        &mut arena.normed,
        &mut arena.linear_input_fp8,
        &mut arena.linear_input_scale,
        &mut arena.attention_q_a,
    )?);
    projections.push(launch_projection_smoke(
        ctx,
        rank,
        "q_b",
        &attention.q_b_proj,
        rows,
        &mut arena.attention_q_a_normed,
        &mut arena.linear_input_fp8,
        &mut arena.linear_input_scale,
        &mut arena.attention_q_b,
    )?);
    projections.push(launch_projection_smoke(
        ctx,
        rank,
        "kv_a",
        &attention.kv_a_proj_with_mqa,
        rows,
        &mut arena.normed,
        &mut arena.linear_input_fp8,
        &mut arena.linear_input_scale,
        &mut arena.attention_kv_a,
    )?);
    projections.push(launch_projection_smoke(
        ctx,
        rank,
        "kv_b",
        &attention.kv_b_proj,
        rows,
        &mut arena.attention_kv_a_normed,
        &mut arena.linear_input_fp8,
        &mut arena.linear_input_scale,
        &mut arena.attention_kv_b,
    )?);
    projections.push(launch_projection_smoke(
        ctx,
        rank,
        "o_proj",
        &attention.o_proj,
        rows,
        &mut arena.attention_out,
        &mut arena.linear_input_fp8,
        &mut arena.linear_input_scale,
        &mut arena.hidden,
    )?);

    let indexer = attention.indexer.as_ref().ok_or_else(|| {
        anyhow::anyhow!("GLM5.2 layer0 attention projection smoke needs indexer weights")
    })?;
    projections.push(launch_projection_smoke(
        ctx,
        rank,
        "indexer_wk",
        &indexer.wk,
        rows,
        &mut arena.normed,
        &mut arena.linear_input_fp8,
        &mut arena.linear_input_scale,
        &mut arena.indexer_wk,
    )?);
    projections.push(launch_projection_smoke(
        ctx,
        rank,
        "indexer_wq_b",
        &indexer.wq_b,
        rows,
        &mut arena.attention_q_a_normed,
        &mut arena.linear_input_fp8,
        &mut arena.linear_input_scale,
        &mut arena.indexer_wq_b,
    )?);

    Ok(Glm52LinearSmokeReport {
        rank,
        rows,
        projections,
    })
}

fn launch_projection_smoke(
    ctx: &Glm52RankGpuContext,
    rank: usize,
    name: &'static str,
    projection: &Glm52Fp8ProjectionGpuWeights<'_>,
    rows: usize,
    input: &mut cudarc::driver::CudaSlice<bf16>,
    linear_input_fp8: &mut cudarc::driver::CudaSlice<u8>,
    linear_input_scale: &mut cudarc::driver::CudaSlice<f32>,
    output: &mut cudarc::driver::CudaSlice<bf16>,
) -> Result<Glm52ProjectionSmokeReport> {
    let contract = projection_contract(name, rows, projection)?;
    ensure!(
        linear_input_fp8.len() >= rows * contract.k
            && linear_input_scale.len() >= rows * contract.activation_scale_cols,
        "GLM5.2 rank {rank} {name} linear arena too small: fp8={}, scale={}, contract={contract:?}",
        linear_input_fp8.len(),
        linear_input_scale.len()
    );

    seed_linear_smoke_input(ctx, name, input, rows, contract.k)?;
    let device_ctx = ctx.as_device_context();
    glm52_fp8_per_token_group_quant_bf16_launch(
        &device_ctx,
        Glm52MoeQuantShape {
            rows,
            width: contract.k,
            group_size: GLM52_MOE_QUANT_GROUP_SIZE,
        },
        input,
        linear_input_fp8,
        linear_input_scale,
    )
    .with_context(|| format!("GLM5.2 rank {rank} {name} input FP8 quant"))?;

    glm52_trtllm_fp8_linear_contract_validate(contract)
        .with_context(|| format!("GLM5.2 rank {rank} {name} TRTLLM linear contract"))?;
    let workspace_bytes = glm52_trtllm_fp8_linear_workspace_size(contract)
        .with_context(|| format!("GLM5.2 rank {rank} {name} TRTLLM linear workspace"))?;
    ensure!(
        workspace_bytes == 0,
        "GLM5.2 rank {rank} {name} TRTLLM linear unexpected workspace: {workspace_bytes}"
    );
    glm52_trtllm_fp8_linear_launch(
        &device_ctx,
        contract,
        linear_input_fp8,
        linear_input_scale,
        &projection.weight.data,
        &projection.weight_scale_inv.data,
        output,
    )
    .with_context(|| format!("GLM5.2 rank {rank} {name} TRTLLM FP8 linear launch"))?;

    ctx.sync()
        .with_context(|| format!("GLM5.2 rank {rank} {name} linear smoke sync"))?;
    let activation_quant_valid =
        validate_linear_quant_output(ctx, linear_input_scale, linear_input_fp8, rows, contract.k)?;
    let output_nonzero = validate_bf16_nonzero(ctx, output, rows * contract.n)?;
    ensure!(
        activation_quant_valid && output_nonzero,
        "GLM5.2 rank {rank} {name} linear smoke failed: activation_quant_valid={activation_quant_valid}, output_nonzero={output_nonzero}"
    );

    Ok(Glm52ProjectionSmokeReport {
        name,
        n: contract.n,
        k: contract.k,
        weight_scale_rows: contract.weight_scale_rows,
        weight_scale_cols: contract.weight_scale_cols,
        activation_scale_cols: contract.activation_scale_cols,
        workspace_bytes,
        activation_quant_valid,
        output_nonzero,
    })
}

fn projection_contract(
    name: &str,
    rows: usize,
    projection: &Glm52Fp8ProjectionGpuWeights<'_>,
) -> Result<Glm52TrtllmFp8LinearContract> {
    let [n, k] = projection.weight.shape.as_slice() else {
        anyhow::bail!(
            "GLM5.2 {name} projection weight must be rank-2, got {:?}",
            projection.weight.shape
        );
    };
    let [weight_scale_rows, weight_scale_cols] = projection.weight_scale_inv.shape.as_slice()
    else {
        anyhow::bail!(
            "GLM5.2 {name} projection scale must be rank-2, got {:?}",
            projection.weight_scale_inv.shape
        );
    };
    Ok(Glm52TrtllmFp8LinearContract {
        m: rows,
        n: *n,
        k: *k,
        weight_scale_rows: *weight_scale_rows,
        weight_scale_cols: *weight_scale_cols,
        activation_scale_cols: k.div_ceil(GLM52_MOE_QUANT_GROUP_SIZE),
    })
}

fn seed_linear_smoke_input(
    ctx: &Glm52RankGpuContext,
    name: &str,
    input_slice: &mut cudarc::driver::CudaSlice<bf16>,
    rows: usize,
    width: usize,
) -> Result<()> {
    ensure!(
        rows * width <= input_slice.len(),
        "GLM5.2 {name} linear smoke input shape rows={rows}, width={width} exceeds input len {}",
        input_slice.len()
    );
    let elems = rows * width;
    let mut input = Vec::with_capacity(elems);
    for row in 0..rows {
        for col in 0..width {
            let value = ((row + 1) * ((col % 29) + 1)) as f32 * 0.00015;
            input.push(bf16::from_f32(value));
        }
    }
    ctx.stream()
        .memcpy_htod(&input, &mut input_slice.slice_mut(0..elems))?;
    Ok(())
}

fn validate_linear_quant_output(
    ctx: &Glm52RankGpuContext,
    scales_slice: &cudarc::driver::CudaSlice<f32>,
    quant_slice: &cudarc::driver::CudaSlice<u8>,
    rows: usize,
    width: usize,
) -> Result<bool> {
    let scale_cols = width / GLM52_MOE_QUANT_GROUP_SIZE;
    let scales = ctx
        .stream()
        .clone_dtoh(&scales_slice.slice(0..rows * scale_cols))?;
    let quant = ctx
        .stream()
        .clone_dtoh(&quant_slice.slice(0..rows * width))?;
    Ok(scales.iter().all(|scale| scale.is_finite() && *scale > 0.0)
        && quant.iter().any(|value| *value != 0))
}

fn validate_bf16_nonzero(
    ctx: &Glm52RankGpuContext,
    output: &cudarc::driver::CudaSlice<bf16>,
    elems: usize,
) -> Result<bool> {
    let sample_elems = elems.min(8192);
    let values = ctx.stream().clone_dtoh(&output.slice(0..sample_elems))?;
    Ok(values.iter().any(|value| value.to_f32() != 0.0))
}
