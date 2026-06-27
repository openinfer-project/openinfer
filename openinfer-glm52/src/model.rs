//! GLM5.2 PP-stage decode model: the typed, device-resident weights for one
//! pipeline stage, built once from the raw loader output.
//!
//! The loader leaves every tensor as a raw `[u8]` device buffer keyed by HF name
//! (`Glm52StageGpuWeights`) plus the expert-major grouped FP8 packages
//! (`Glm52StageExpertFp8Weights`). This module *drains* those into the typed brick
//! weight structs the decode forward consumes — moving the fp8 projections in
//! place (no re-upload, no 2x peak) and bridging the bf16 bookends / layernorms
//! through host (they are stored as raw `u8` but the ops want `bf16`). The drain
//! is strict: every resident tensor and every expert package must be consumed.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::CudaSlice;
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use safetensors::Dtype;

use crate::config::{GLM52_HIDDEN, GLM52_VOCAB};
use crate::dense::Glm52DenseMlpWeights;
use crate::fp8::ProjWeight;
use crate::mla_decode::Glm52MlaLayerWeights;
use crate::moe_decode::Glm52MoeLayerWeights;
use crate::weights::{
    Glm52AttentionWeightNames, Glm52Fp8ProjectionWeightNames, Glm52GpuRawTensor,
    Glm52LayerWeightKindNames, Glm52StageExpertFp8Weights, Glm52StageGpuWeights,
    Glm52StageWeightNames,
};

/// The MLP half of a decode layer: a dense SwiGLU (layers 0..first_k_dense_replace)
/// or the routed + shared MoE block (the rest).
pub(crate) enum Glm52MlpModel {
    Dense(Glm52DenseMlpWeights),
    Moe(Glm52MoeLayerWeights),
}

/// One decode layer's weights: the two layernorm gammas bracketing attention/MLP,
/// the MLA attention block, and the MLP block.
pub(crate) struct Glm52LayerModel {
    pub(crate) layer_idx: usize,
    pub(crate) input_layernorm: DeviceVec,
    pub(crate) mla: Glm52MlaLayerWeights,
    pub(crate) post_attention_layernorm: DeviceVec,
    pub(crate) mlp: Glm52MlpModel,
}

/// All typed weights resident on one pipeline stage. Bookends are present only on
/// their owning stage (embedding on stage 0, final norm + lm_head on the last).
pub(crate) struct Glm52StageModel {
    pub(crate) stage: usize,
    pub(crate) embed: Option<DeviceMatrix>,
    pub(crate) layers: Vec<Glm52LayerModel>,
    pub(crate) final_norm: Option<DeviceVec>,
    pub(crate) lm_head: Option<DeviceMatrix>,
}

impl Glm52StageModel {
    /// Drain the raw loader output into the typed stage model. Consumes both the
    /// raw tensor map and the expert packages; a leftover tensor or package is a
    /// loader/manifest drift and crashes here.
    pub(crate) fn build(
        ctx: &DeviceContext,
        mut weights: Glm52StageGpuWeights,
        experts: Glm52StageExpertFp8Weights,
        names: &Glm52StageWeightNames,
    ) -> Result<Self> {
        ensure!(
            weights.stage == names.stage && experts.stage == names.stage,
            "GLM5.2 stage model build mismatch: weights {} / experts {} / names {}",
            weights.stage,
            experts.stage,
            names.stage
        );
        // Expert packages keyed by layer index (moved out per MoE layer below).
        let mut expert_by_layer: BTreeMap<usize, _> = experts
            .layers
            .into_iter()
            .map(|layer| (layer.layer_idx, layer))
            .collect();

        let embed = names
            .top
            .token_embedding
            .as_deref()
            .map(|name| take_bf16_matrix(ctx, &mut weights, name, GLM52_VOCAB, GLM52_HIDDEN))
            .transpose()?;
        let final_norm = names
            .top
            .final_norm
            .as_deref()
            .map(|name| take_bf16_vec(ctx, &mut weights, name))
            .transpose()?;
        let lm_head = names
            .top
            .lm_head
            .as_deref()
            .map(|name| take_bf16_matrix(ctx, &mut weights, name, GLM52_VOCAB, GLM52_HIDDEN))
            .transpose()?;

        let mut layers = Vec::with_capacity(names.layers.len());
        for layer in &names.layers {
            let input_layernorm =
                take_bf16_vec(ctx, &mut weights, &layer.attention.input_layernorm)?;
            let mla = build_mla(ctx, &mut weights, &layer.attention)?;
            let post_attention_layernorm =
                take_bf16_vec(ctx, &mut weights, &layer.attention.post_attention_layernorm)?;
            let mlp = match &layer.kind {
                Glm52LayerWeightKindNames::Dense(dense) => {
                    Glm52MlpModel::Dense(Glm52DenseMlpWeights::from_device(
                        take_proj(&mut weights, &dense.gate_proj)?,
                        take_proj(&mut weights, &dense.up_proj)?,
                        take_proj(&mut weights, &dense.down_proj)?,
                    )?)
                }
                Glm52LayerWeightKindNames::Moe(moe) => {
                    let pkg = expert_by_layer.remove(&layer.layer_idx).ok_or_else(|| {
                        anyhow!(
                            "GLM5.2 stage {} missing expert package for MoE layer {}",
                            names.stage,
                            layer.layer_idx
                        )
                    })?;
                    let gate_weight = take_raw_u8(&mut weights, &moe.router.gate_weight)?;
                    let e_score_bias =
                        take_raw_u8(&mut weights, &moe.router.e_score_correction_bias)?;
                    let shared_gate = take_proj(&mut weights, &moe.shared_experts.gate_proj)?;
                    let shared_up = take_proj(&mut weights, &moe.shared_experts.up_proj)?;
                    let shared_down = take_proj(&mut weights, &moe.shared_experts.down_proj)?;
                    Glm52MlpModel::Moe(Glm52MoeLayerWeights::from_resident(
                        gate_weight,
                        e_score_bias,
                        pkg.w13.weight_e4m3,
                        pkg.w13.weight_scale_inv_f32,
                        pkg.down.weight_e4m3,
                        pkg.down.weight_scale_inv_f32,
                        shared_gate,
                        shared_up,
                        shared_down,
                    )?)
                }
            };
            layers.push(Glm52LayerModel {
                layer_idx: layer.layer_idx,
                input_layernorm,
                mla,
                post_attention_layernorm,
                mlp,
            });
        }

        ensure!(
            weights.tensors.is_empty(),
            "GLM5.2 stage {} model build left {} resident tensors unconsumed (e.g. {:?})",
            names.stage,
            weights.tensors.len(),
            weights.tensors.keys().take(8).collect::<Vec<_>>()
        );
        ensure!(
            expert_by_layer.is_empty(),
            "GLM5.2 stage {} model build left {} expert packages unconsumed",
            names.stage,
            expert_by_layer.len()
        );
        Ok(Self {
            stage: names.stage,
            embed,
            layers,
            final_norm,
            lm_head,
        })
    }
}

/// Assemble the MLA attention weights, moving the fp8 projections in and bridging
/// the two internal layernorm gammas. The DSA indexer (Slice 4) is deferred:
/// short-context decode (<= 2048 tokens) makes the top-2048 select a no-op
/// (all tokens), so the indexer's resident tensors are consumed and dropped here.
fn build_mla(
    ctx: &DeviceContext,
    weights: &mut Glm52StageGpuWeights,
    att: &Glm52AttentionWeightNames,
) -> Result<Glm52MlaLayerWeights> {
    let q_a = take_proj(weights, &att.q_a_proj)?;
    let q_a_layernorm = take_bf16_vec(ctx, weights, &att.q_a_layernorm)?;
    let q_b = take_proj(weights, &att.q_b_proj)?;
    let kv_a = take_proj(weights, &att.kv_a_proj_with_mqa)?;
    let kv_a_layernorm = take_bf16_vec(ctx, weights, &att.kv_a_layernorm)?;
    let kv_b = take_proj(weights, &att.kv_b_proj)?;
    let o_proj = take_proj(weights, &att.o_proj)?;
    if let Some(indexer) = &att.indexer {
        for name in indexer.tensor_names() {
            // Loaded but unused until the indexer slice lands; drop to free HBM.
            drop(take_raw(weights, name)?);
        }
    }
    Glm52MlaLayerWeights::from_device(
        ctx,
        q_a,
        q_a_layernorm,
        q_b,
        kv_a,
        kv_a_layernorm,
        kv_b,
        o_proj,
    )
}

fn take_raw(weights: &mut Glm52StageGpuWeights, name: &str) -> Result<Glm52GpuRawTensor> {
    weights
        .tensors
        .remove(name)
        .ok_or_else(|| anyhow!("GLM5.2 missing resident tensor {name}"))
}

/// Move a raw device buffer out as-is (bf16 gate / f32 bias kept as raw `u8`).
fn take_raw_u8(weights: &mut Glm52StageGpuWeights, name: &str) -> Result<CudaSlice<u8>> {
    Ok(take_raw(weights, name)?.data)
}

/// Move an fp8 projection (weight + scale) out into a `ProjWeight`, sourcing
/// `[n,k]` from the resident weight shape.
fn take_proj(
    weights: &mut Glm52StageGpuWeights,
    names: &Glm52Fp8ProjectionWeightNames,
) -> Result<ProjWeight> {
    let weight = take_raw(weights, &names.weight)?;
    let scale = take_raw(weights, &names.weight_scale_inv)?;
    ensure!(
        weight.dtype == Dtype::F8_E4M3 && weight.shape.len() == 2,
        "GLM5.2 fp8 projection {} dtype/shape {:?}/{:?} unexpected",
        weight.name,
        weight.dtype,
        weight.shape
    );
    let (n, k) = (weight.shape[0], weight.shape[1]);
    ProjWeight::from_device(weight.data, scale.data, n, k)
}

/// Bridge a resident bf16 vector (stored as raw `u8`) into a `DeviceVec`. The
/// device buffer is freed before the re-upload so peak HBM does not double.
fn take_bf16_vec(
    ctx: &DeviceContext,
    weights: &mut Glm52StageGpuWeights,
    name: &str,
) -> Result<DeviceVec> {
    let tensor = take_raw(weights, name)?;
    ensure!(
        tensor.dtype == Dtype::BF16,
        "GLM5.2 bf16 tensor {} has dtype {:?}",
        tensor.name,
        tensor.dtype
    );
    let host = ctx.stream.clone_dtoh(&tensor.data)?;
    drop(tensor);
    DeviceVec::from_safetensors(ctx, &host)
}

/// Bridge a resident bf16 matrix (stored as raw `u8`) into a `DeviceMatrix`.
fn take_bf16_matrix(
    ctx: &DeviceContext,
    weights: &mut Glm52StageGpuWeights,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DeviceMatrix> {
    let tensor = take_raw(weights, name)?;
    ensure!(
        tensor.dtype == Dtype::BF16 && tensor.shape == [rows, cols],
        "GLM5.2 bf16 matrix {} dtype/shape {:?}/{:?} != BF16 [{rows},{cols}]",
        tensor.name,
        tensor.dtype,
        tensor.shape
    );
    let host = ctx.stream.clone_dtoh(&tensor.data)?;
    drop(tensor);
    DeviceMatrix::from_safetensors(ctx, &host, rows, cols)
}
