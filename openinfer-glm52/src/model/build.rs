//! Weight-extraction helpers: build one decoder layer's GPU-resident
//! weights (MLA + indexer + dense/MoE MLP + layernorms) from the resident
//! tensor map.

use anyhow::{Result, ensure};
use half::bf16;
use openinfer_kernels::tensor::{DeviceContext, DeviceVec};

use crate::config::{
    GLM52_DENSE_LAYERS, GLM52_HIDDEN, GLM52_INDEX_HEAD_DIM, glm52_layer_has_full_indexer,
};
use crate::dense::Glm52DenseMlpWeights;
use crate::fp8::ProjWeight;
use crate::indexer::Glm52IndexerLayerWeights;
use crate::layer::{Glm52DecoderLayerWeights, Glm52LayerIndexer, Glm52LayerMlp};
use crate::mla_decode::Glm52MlaLayerWeights;
use crate::moe_decode::{Glm52MoeExpertBank, Glm52MoeRouterWeights, Glm52MoeSharedExpert};
use crate::moe_ep8::Glm52MoeEp8LayerWeights;
use crate::weights::{Glm52RankGpuWeights, retype_owned};

/// Take one fp8 projection (weight + scale) out of the resident tensor map.
pub(super) fn take_proj(
    w: &mut Glm52RankGpuWeights,
    stem: &str,
    n: usize,
    k: usize,
) -> Result<ProjWeight> {
    ProjWeight::from_device(
        w.take_tensor(&format!("{stem}.weight"))?,
        w.take_tensor(&format!("{stem}.weight_scale_inv"))?,
        n,
        k,
    )
}

/// Take a bf16 vector (e.g. a layernorm gamma) out of the resident map.
pub(super) fn take_bf16_vec(
    ctx: &DeviceContext,
    w: &mut Glm52RankGpuWeights,
    name: &str,
    len: usize,
) -> Result<DeviceVec> {
    let raw = w.take_tensor(name)?;
    ensure!(
        raw.len() == len * 2,
        "GLM5.2 tensor {name} bytes {} != bf16 [{len}]",
        raw.len()
    );
    Ok(DeviceVec {
        data: retype_owned::<bf16>(&ctx.stream, raw)?,
        len,
    })
}

pub(super) fn build_decoder_layer(
    ctx: &DeviceContext,
    w: &mut Glm52RankGpuWeights,
    layer: usize,
) -> Result<Glm52DecoderLayerWeights> {
    let p = format!("model.layers.{layer}");

    let kv_b = take_proj(w, &format!("{p}.self_attn.kv_b_proj"), 28_672, 512)?;
    let mla = Glm52MlaLayerWeights::from_device(
        ctx,
        take_proj(w, &format!("{p}.self_attn.q_a_proj"), 2048, GLM52_HIDDEN)?,
        take_bf16_vec(ctx, w, &format!("{p}.self_attn.q_a_layernorm.weight"), 2048)?,
        take_proj(w, &format!("{p}.self_attn.q_b_proj"), 16_384, 2048)?,
        take_proj(
            w,
            &format!("{p}.self_attn.kv_a_proj_with_mqa"),
            576,
            GLM52_HIDDEN,
        )?,
        take_bf16_vec(ctx, w, &format!("{p}.self_attn.kv_a_layernorm.weight"), 512)?,
        &kv_b,
        take_proj(w, &format!("{p}.self_attn.o_proj"), GLM52_HIDDEN, 16_384)?,
    )?;
    // kv_b is only dequanted into the absorb factors, not stored — free the
    // fp8 blob before the indexer/MLP uploads below.
    drop(kv_b);

    let indexer = if glm52_layer_has_full_indexer(layer) {
        let ip = format!("{p}.self_attn.indexer");
        let k_norm_w = ctx
            .stream
            .clone_dtoh(&w.take_tensor(&format!("{ip}.k_norm.weight"))?)?;
        let k_norm_b = ctx
            .stream
            .clone_dtoh(&w.take_tensor(&format!("{ip}.k_norm.bias"))?)?;
        let weights_proj = retype_owned::<bf16>(
            &ctx.stream,
            w.take_tensor(&format!("{ip}.weights_proj.weight"))?,
        )?;
        Glm52LayerIndexer::Full(Box::new(Glm52IndexerLayerWeights::from_device(
            ctx,
            take_proj(w, &format!("{ip}.wq_b"), 32 * GLM52_INDEX_HEAD_DIM, 2048)?,
            take_proj(w, &format!("{ip}.wk"), GLM52_INDEX_HEAD_DIM, GLM52_HIDDEN)?,
            weights_proj,
            &k_norm_w,
            &k_norm_b,
        )?))
    } else {
        Glm52LayerIndexer::Shared
    };

    let mp = format!("{p}.mlp");
    let mlp = if layer < GLM52_DENSE_LAYERS {
        Glm52LayerMlp::Dense(Box::new(Glm52DenseMlpWeights::from_device(
            ctx,
            &take_proj(w, &format!("{mp}.gate_proj"), 12_288, GLM52_HIDDEN)?,
            &take_proj(w, &format!("{mp}.up_proj"), 12_288, GLM52_HIDDEN)?,
            take_proj(w, &format!("{mp}.down_proj"), GLM52_HIDDEN, 12_288)?,
        )?))
    } else {
        Glm52LayerMlp::MoeEp8(Box::new(Glm52MoeEp8LayerWeights {
            router: Glm52MoeRouterWeights::new(
                w.take_tensor(&format!("{mp}.gate.weight"))?,
                w.take_tensor(&format!("{mp}.gate.e_score_correction_bias"))?,
            )?,
            shared: Glm52MoeSharedExpert::new(
                ctx,
                &take_proj(
                    w,
                    &format!("{mp}.shared_experts.gate_proj"),
                    2048,
                    GLM52_HIDDEN,
                )?,
                &take_proj(
                    w,
                    &format!("{mp}.shared_experts.up_proj"),
                    2048,
                    GLM52_HIDDEN,
                )?,
                take_proj(
                    w,
                    &format!("{mp}.shared_experts.down_proj"),
                    GLM52_HIDDEN,
                    2048,
                )?,
            )?,
            bank: Glm52MoeExpertBank::from_regions(ctx, w.take_expert_layer(layer)?)?,
        }))
    };

    Ok(Glm52DecoderLayerWeights {
        input_ln: take_bf16_vec(ctx, w, &format!("{p}.input_layernorm.weight"), GLM52_HIDDEN)?,
        post_attn_ln: take_bf16_vec(
            ctx,
            w,
            &format!("{p}.post_attention_layernorm.weight"),
            GLM52_HIDDEN,
        )?,
        mla,
        indexer,
        mlp,
    })
}
