//! GLM5.2 DP8/EP8 full-model decode: every rank owns the whole non-expert
//! path (embed → 78 decoder layers → final norm → lm_head → greedy argmax)
//! plus its 32 local experts, and serves one request at a time (bs=1 per
//! rank; prefill rides decode token-by-token).
//!
//! Every step, all 8 ranks run the forward in lock-step, each dispatching
//! exactly one token (a real request token, or a padding token on an idle
//! rank) into every MoE layer's DeepEP collective — the collectives require
//! all ranks to enter in the same layer order 3..=77.

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE, GLM52_FLASHMLA_SPARSE_TOPK, Glm52FlashMlaSparseDecode,
    Glm52IndexerCacheLayout, add_into, argmax_bf16_into, glm52_flashmla_sparse_decode_num_sm_parts,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

use crate::bookend::{glm52_embed_into, glm52_final_norm_into, glm52_lm_head_into};
use crate::config::{GLM52_DENSE_LAYERS, GLM52_HIDDEN, GLM52_LAYERS, GLM52_VOCAB};
use crate::dense::{Glm52DenseMlpWeights, glm52_dense_mlp_forward_into};
use crate::fp8::ProjWeight;
use crate::indexer::{Glm52IndexerLayerWeights, Glm52IndexerScratch};
use crate::layer::{
    Glm52DecodeStep, Glm52DecoderLayerWeights, Glm52LayerCaches, Glm52LayerIndexer, Glm52LayerMlp,
    glm52_layer_attention_half, glm52_layer_finish,
};
use crate::mla_decode::{Glm52MlaLayerWeights, Glm52MlaSchedMetadata};
use crate::moe_decode::{
    Glm52MoeExpertBank, Glm52MoeRouterWeights, Glm52MoeSharedExpert, run_router_into,
};
use crate::moe_ep8::{Glm52MoeEp8LayerWeights, Glm52MoeEp8State, glm52_moe_ep8_routed_forward};
use crate::scratch::Glm52DecodeScratch;
use crate::weights::{Glm52RankGpuWeights, retype_owned};

/// bs=1 bring-up context cap: `prompt + max_tokens - 1 <= GLM52_MAX_MODEL_LEN`.
/// Sizes the per-layer MLA and index-K caches at build time.
pub(crate) const GLM52_MAX_MODEL_LEN: usize = 4096;

/// The DP8 coordinator's protocol constant: every rank enters each MoE
/// collective having agreed on this many dispatched tokens per step — one
/// token per rank (real or padding), so the global count is the rank count.
/// Every rank's `decode_step` must derive its `global_tokens` from this
/// single definition — a disagreement makes the grouped-GEMM row bound wrong
/// on some rank, which the metadata kernel answers with a device trap.
pub(crate) const GLM52_DECODE_GLOBAL_TOKENS: usize = crate::weights::GLM52_EP_RANKS;

pub(crate) const ROPE_HALF: usize = 32;
const ROPE_THETA: f32 = 8_000_000.0;
/// 1/sqrt(qk_head_dim = 192 + 64).
pub(crate) const SM_SCALE: f32 = 0.0625;
pub(crate) const INDEX_HEAD_DIM: usize = 128;
/// DeepGEMM paged MQA requires BLOCK_KV=64.
pub(crate) const INDEX_CACHE_BLOCK: usize = 64;
pub(crate) const NUM_SMS: usize = 132;

/// `indexer_types[layer]` per the transformers derivation
/// (`index_topk_freq=4`, `index_skip_topk_offset=3`): full iff
/// `max(layer-2, 0) % 4 == 0` → {0,1,2} ∪ {6,10,…,74}, 21 of 78 layers.
pub(crate) fn glm52_layer_has_full_indexer(layer: usize) -> bool {
    layer.saturating_sub(2).is_multiple_of(4)
}

pub(crate) fn rope_tables(position: usize) -> (Vec<bf16>, Vec<bf16>) {
    (0..ROPE_HALF)
        .map(|j| {
            let inv_freq = 1.0 / ROPE_THETA.powf(j as f32 / ROPE_HALF as f32);
            let angle = position as f32 * inv_freq;
            (bf16::from_f32(angle.cos()), bf16::from_f32(angle.sin()))
        })
        .unzip()
}

/// Take one fp8 projection (weight + scale) out of the resident tensor map.
fn take_proj(w: &mut Glm52RankGpuWeights, stem: &str, n: usize, k: usize) -> Result<ProjWeight> {
    ProjWeight::from_device(
        w.take_tensor(&format!("{stem}.weight"))?,
        w.take_tensor(&format!("{stem}.weight_scale_inv"))?,
        n,
        k,
    )
}

/// Take a bf16 vector (e.g. a layernorm gamma) out of the resident map.
fn take_bf16_vec(
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

fn build_decoder_layer(
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
        kv_b,
        take_proj(w, &format!("{p}.self_attn.o_proj"), GLM52_HIDDEN, 16_384)?,
    )?;

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
            take_proj(w, &format!("{ip}.wq_b"), 32 * INDEX_HEAD_DIM, 2048)?,
            take_proj(w, &format!("{ip}.wk"), INDEX_HEAD_DIM, GLM52_HIDDEN)?,
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
            take_proj(w, &format!("{mp}.gate_proj"), 12_288, GLM52_HIDDEN)?,
            take_proj(w, &format!("{mp}.up_proj"), 12_288, GLM52_HIDDEN)?,
            take_proj(w, &format!("{mp}.down_proj"), GLM52_HIDDEN, 12_288)?,
        )?))
    } else {
        Glm52LayerMlp::MoeEp8(Box::new(Glm52MoeEp8LayerWeights {
            router: Glm52MoeRouterWeights::new(
                w.take_tensor(&format!("{mp}.gate.weight"))?,
                w.take_tensor(&format!("{mp}.gate.e_score_correction_bias"))?,
            )?,
            shared: Glm52MoeSharedExpert::new(
                take_proj(
                    w,
                    &format!("{mp}.shared_experts.gate_proj"),
                    2048,
                    GLM52_HIDDEN,
                )?,
                take_proj(
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

/// One DP rank: the full non-expert model plus this rank's expert banks.
pub(crate) struct Glm52RankModel {
    layers: Vec<Glm52DecoderLayerWeights>,
    caches: Vec<Glm52LayerCaches>,
    embed: DeviceMatrix,
    final_norm: DeviceVec,
    lm_head: DeviceMatrix,
    mla_sched: Glm52MlaSchedMetadata,
    index_cache_layout: Glm52IndexerCacheLayout,
    block_table: CudaSlice<i32>,
    slot_mapping: CudaSlice<i64>,
    seq_lens: CudaSlice<i32>,
    /// Device-resident rope tables for every position (`[GLM52_MAX_MODEL_LEN,
    /// ROPE_HALF]`); a step slices its position's row instead of recomputing
    /// on the host and copying up.
    cos_table: CudaSlice<bf16>,
    sin_table: CudaSlice<bf16>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    token_id: CudaSlice<u32>,
    scratch: Glm52DecodeScratch,
}

impl Glm52RankModel {
    pub(crate) fn build(ctx: &DeviceContext, w: &mut Glm52RankGpuWeights) -> Result<Self> {
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: 1,
            num_blocks: GLM52_MAX_MODEL_LEN.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE),
            topk: GLM52_FLASHMLA_SPARSE_TOPK,
            num_sm_parts: glm52_flashmla_sparse_decode_num_sm_parts()?,
            sm_scale: SM_SCALE,
        };
        let index_blocks = GLM52_MAX_MODEL_LEN.div_ceil(INDEX_CACHE_BLOCK);
        let index_cache_layout = Glm52IndexerCacheLayout {
            cache_blocks: index_blocks,
            cache_block_size: INDEX_CACHE_BLOCK,
            cache_block_stride_bytes: INDEX_CACHE_BLOCK * (INDEX_HEAD_DIM + 4),
        };

        let mut layers = Vec::with_capacity(GLM52_LAYERS);
        let mut caches = Vec::with_capacity(GLM52_LAYERS);
        for layer in 0..GLM52_LAYERS {
            layers.push(
                build_decoder_layer(ctx, w, layer)
                    .with_context(|| format!("build GLM5.2 decoder layer {layer}"))?,
            );
            caches.push(Glm52LayerCaches {
                mla_cache: ctx
                    .stream
                    .alloc_zeros::<u8>(contract.packed_kv_cache_len())?,
                index_k_cache: glm52_layer_has_full_indexer(layer)
                    .then(|| {
                        ctx.stream
                            .alloc_zeros::<u8>(index_cache_layout.min_cache_bytes()?)
                            .map_err(anyhow::Error::from)
                    })
                    .transpose()?,
            });
        }

        let embed_raw = w.take_tensor("model.embed_tokens.weight")?;
        let lm_head_raw = w.take_tensor("lm_head.weight")?;
        ensure!(
            embed_raw.len() == GLM52_VOCAB * GLM52_HIDDEN * 2
                && lm_head_raw.len() == GLM52_VOCAB * GLM52_HIDDEN * 2,
            "GLM5.2 embed/lm_head byte lengths unexpected"
        );
        let embed = DeviceMatrix {
            data: retype_owned::<bf16>(&ctx.stream, embed_raw)?,
            rows: GLM52_VOCAB,
            cols: GLM52_HIDDEN,
        };
        let lm_head = DeviceMatrix {
            data: retype_owned::<bf16>(&ctx.stream, lm_head_raw)?,
            rows: GLM52_VOCAB,
            cols: GLM52_HIDDEN,
        };
        let final_norm = take_bf16_vec(ctx, w, "model.norm.weight", GLM52_HIDDEN)?;

        // The MTP layer's experts are loaded (checkpoint-coverage validation)
        // but out of campaign scope — drop this rank's copy.
        let _ = w.take_expert_layer(crate::weights::GLM52_MTP_LAYER)?;

        let block_table_host: Vec<i32> = (0..index_blocks as i32).collect();
        let mut block_table = ctx.stream.alloc_zeros::<i32>(index_blocks)?;
        ctx.stream
            .memcpy_htod(&block_table_host, &mut block_table)?;

        let mut cos_host = Vec::with_capacity(GLM52_MAX_MODEL_LEN * ROPE_HALF);
        let mut sin_host = Vec::with_capacity(GLM52_MAX_MODEL_LEN * ROPE_HALF);
        for position in 0..GLM52_MAX_MODEL_LEN {
            let (cos_row, sin_row) = rope_tables(position);
            cos_host.extend_from_slice(&cos_row);
            sin_host.extend_from_slice(&sin_row);
        }
        let mut cos_table = ctx
            .stream
            .alloc_zeros::<bf16>(GLM52_MAX_MODEL_LEN * ROPE_HALF)?;
        let mut sin_table = ctx
            .stream
            .alloc_zeros::<bf16>(GLM52_MAX_MODEL_LEN * ROPE_HALF)?;
        ctx.stream.memcpy_htod(&cos_host, &mut cos_table)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin_table)?;

        let mqa_shape = Glm52IndexerScratch::decode_shape(
            index_cache_layout,
            index_blocks,
            NUM_SMS,
            GLM52_MAX_MODEL_LEN,
        );
        Ok(Self {
            layers,
            caches,
            embed,
            final_norm,
            lm_head,
            mla_sched: Glm52MlaSchedMetadata::new(ctx, contract)?,
            index_cache_layout,
            block_table,
            slot_mapping: ctx.stream.alloc_zeros::<i64>(1)?,
            seq_lens: ctx.stream.alloc_zeros::<i32>(1)?,
            cos_table,
            sin_table,
            cos: ctx.stream.alloc_zeros::<bf16>(ROPE_HALF)?,
            sin: ctx.stream.alloc_zeros::<bf16>(ROPE_HALF)?,
            token_id: ctx.stream.alloc_zeros::<u32>(1)?,
            scratch: Glm52DecodeScratch::new(ctx, &contract, mqa_shape)?,
        })
    }

    /// One full-model step: feed `token` at `position`, return the greedy
    /// next-token id. Enters 75 MoE collectives — every other rank must be
    /// stepping concurrently (an idle rank steps a padding token).
    pub(crate) fn decode_step(
        &mut self,
        ctx: &DeviceContext,
        ep8: &mut Glm52MoeEp8State,
        token: u32,
        position: usize,
    ) -> Result<u32> {
        ensure!(
            position < GLM52_MAX_MODEL_LEN,
            "GLM5.2 position {position} exceeds the model-length cap {GLM52_MAX_MODEL_LEN}"
        );
        let rope = position * ROPE_HALF..(position + 1) * ROPE_HALF;
        ctx.stream
            .memcpy_dtod(&self.cos_table.slice(rope.clone()), &mut self.cos)?;
        ctx.stream
            .memcpy_dtod(&self.sin_table.slice(rope), &mut self.sin)?;
        ctx.stream
            .memcpy_htod(&[position as i64], &mut self.slot_mapping)?;
        ctx.stream
            .memcpy_htod(&[(position + 1) as i32], &mut self.seq_lens)?;
        ctx.stream.memcpy_htod(&[token], &mut self.token_id)?;

        let step = Glm52DecodeStep {
            mla_cos: &self.cos,
            mla_sin: &self.sin,
            idx_cos: &self.cos,
            idx_sin: &self.sin,
            mla_sched: &self.mla_sched,
            index_cache_layout: self.index_cache_layout,
            slot_mapping: &self.slot_mapping,
            block_table: &self.block_table,
            seq_lens: &self.seq_lens,
        };

        let s = &mut self.scratch;
        glm52_embed_into(ctx, &self.embed, &self.token_id, &mut s.hidden)?;
        let mut carry_ready = false;
        for (layer, (weights, cache)) in self.layers.iter().zip(self.caches.iter_mut()).enumerate()
        {
            glm52_layer_attention_half(ctx, weights, cache, &step, s, &mut carry_ready)
                .with_context(|| format!("GLM5.2 layer {layer} attention half"))?;
            match &weights.mlp {
                Glm52LayerMlp::Dense(dense) => glm52_dense_mlp_forward_into(
                    ctx,
                    dense,
                    &s.layer.normed2,
                    &mut s.proj,
                    &mut s.dense_mlp,
                    &mut s.layer.mlp_out,
                )?,
                Glm52LayerMlp::MoeEp8(moe) => {
                    run_router_into(ctx, &moe.router, &s.layer.normed2, &mut s.router)?;
                    let dispatched = glm52_moe_ep8_routed_forward(
                        ctx,
                        ep8,
                        &moe.bank,
                        Some((&s.layer.normed2, &s.router.route)),
                        GLM52_DECODE_GLOBAL_TOKENS,
                    )
                    .with_context(|| format!("GLM5.2 layer {layer} EP8 MoE"))?;
                    ensure!(
                        dispatched,
                        "EP8 MoE returned no combined output for a dispatched token"
                    );
                    moe.shared.forward_into(
                        ctx,
                        &s.layer.normed2,
                        &mut s.proj,
                        &mut s.shared_mlp,
                        &mut s.layer.shared_out,
                    )?;
                    add_into(
                        ctx,
                        ep8.combined(),
                        &s.layer.shared_out,
                        GLM52_HIDDEN,
                        &mut s.layer.mlp_out,
                    )?;
                }
                Glm52LayerMlp::Moe(_) => {
                    anyhow::bail!("GLM5.2 EP8 spine built an EP1 MoE layer — loader bug")
                }
            };
            glm52_layer_finish(ctx, s)?;
        }

        glm52_final_norm_into(ctx, &s.hidden, &self.final_norm, &mut s.final_normed)?;
        glm52_lm_head_into(ctx, &s.final_normed, &self.lm_head, &mut s.logits)?;
        // Device greedy argmax (same semantics as a host scan: lowest index
        // wins ties, NaN never wins) — the step's egress shrinks from the
        // full vocab row to 6 bytes, and the kernel chain ends on-device
        // (the graph boundary for PR5c stage 3).
        argmax_bf16_into(
            ctx,
            &s.logits.data,
            GLM52_VOCAB,
            &mut s.argmax_value,
            &mut s.argmax_index,
        )?;
        let top_value = ctx.stream.clone_dtoh(&s.argmax_value)?[0].to_f32();
        let top_index = ctx.stream.clone_dtoh(&s.argmax_index)?[0];
        ensure!(
            top_value.is_finite(),
            "GLM5.2 greedy argmax found no finite logit (top = {top_value})"
        );
        ensure!(
            (0..GLM52_VOCAB as i32).contains(&top_index),
            "GLM5.2 greedy argmax index {top_index} outside the vocab"
        );
        Ok(top_index as u32)
    }
}
