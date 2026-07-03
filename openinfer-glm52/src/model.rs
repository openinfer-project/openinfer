//! GLM5.2 DP1/EP8 full-model decode: rank 0 owns the whole non-expert path
//! (embed → 78 decoder layers → final norm → lm_head → greedy argmax) and its
//! 32 local experts; ranks 1..7 hold only their 32 experts per MoE layer and
//! enter the per-layer DeepEP collectives.
//!
//! bs=1, one token per step; prefill rides decode token-by-token. Every MoE
//! layer is one collective (`glm52_moe_ep8_routed_forward`) that all ranks
//! must enter in the same order — rank 0 walks layers 0..=77 and hits the 75
//! MoE layers (3..=77) in ascending order; expert ranks replay exactly that
//! sequence in `expert_step`.

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE, GLM52_FLASHMLA_SPARSE_TOPK, Glm52FlashMlaSparseDecode,
    Glm52IndexerCacheLayout, add_batch, glm52_flashmla_sparse_decode_num_sm_parts,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

use crate::bookend::{glm52_embed, glm52_final_norm, glm52_lm_head};
use crate::config::{GLM52_DENSE_LAYERS, GLM52_HIDDEN, GLM52_LAYERS, GLM52_VOCAB};
use crate::dense::{Glm52DenseMlpWeights, glm52_dense_mlp_forward};
use crate::fp8::ProjWeight;
use crate::indexer::Glm52IndexerLayerWeights;
use crate::layer::{
    Glm52DecodeStep, Glm52DecoderLayerWeights, Glm52LayerCaches, Glm52LayerIndexer, Glm52LayerMlp,
    glm52_layer_attention_half, glm52_layer_finish,
};
use crate::mla_decode::{Glm52MlaLayerWeights, Glm52MlaSchedMetadata};
use crate::moe_decode::{
    Glm52MoeExpertBank, Glm52MoeExpertPath, Glm52MoeRouterWeights, Glm52MoeSharedExpert, run_router,
};
use crate::moe_ep8::{Glm52MoeEp8LayerWeights, Glm52MoeEp8State, glm52_moe_ep8_routed_forward};
use crate::weights::{Glm52RankGpuWeights, retype_owned};

/// bs=1 bring-up context cap: `prompt + max_tokens - 1 <= GLM52_MAX_MODEL_LEN`.
/// Sizes the per-layer MLA and index-K caches at build time.
pub(crate) const GLM52_MAX_MODEL_LEN: usize = 4096;

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

/// Rank 0: the full non-expert model plus this rank's expert banks.
pub(crate) struct Glm52Rank0Model {
    layers: Vec<Glm52DecoderLayerWeights>,
    caches: Vec<Glm52LayerCaches>,
    embed: DeviceMatrix,
    final_norm: DeviceVec,
    lm_head: DeviceMatrix,
    contract: Glm52FlashMlaSparseDecode,
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
}

impl Glm52Rank0Model {
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
        // but out of campaign scope — drop rank 0's copy too.
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

        Ok(Self {
            layers,
            caches,
            embed,
            final_norm,
            lm_head,
            mla_sched: Glm52MlaSchedMetadata::new(ctx, contract)?,
            contract,
            index_cache_layout,
            block_table,
            slot_mapping: ctx.stream.alloc_zeros::<i64>(1)?,
            seq_lens: ctx.stream.alloc_zeros::<i32>(1)?,
            cos_table,
            sin_table,
            cos: ctx.stream.alloc_zeros::<bf16>(ROPE_HALF)?,
            sin: ctx.stream.alloc_zeros::<bf16>(ROPE_HALF)?,
            token_id: ctx.stream.alloc_zeros::<u32>(1)?,
        })
    }

    /// One full-model step: feed `token` at `position`, return the greedy
    /// next-token id. Enters 75 MoE collectives — the expert ranks must be
    /// running `expert_step` concurrently.
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
            position,
            mla_cos: &self.cos,
            mla_sin: &self.sin,
            idx_cos: &self.cos,
            idx_sin: &self.sin,
            contract: self.contract,
            mla_sched: &self.mla_sched,
            index_cache_layout: self.index_cache_layout,
            slot_mapping: &self.slot_mapping,
            block_table: &self.block_table,
            seq_lens: &self.seq_lens,
            num_sms: NUM_SMS,
            max_model_len: GLM52_MAX_MODEL_LEN,
            moe_path: Glm52MoeExpertPath::Grouped,
        };

        let mut hidden = glm52_embed(ctx, &self.embed, &self.token_id)?.data;
        let mut topk_carry: Option<CudaSlice<i32>> = None;
        for (layer, (weights, cache)) in self.layers.iter().zip(self.caches.iter_mut()).enumerate()
        {
            let boundary =
                glm52_layer_attention_half(ctx, weights, cache, hidden, &step, &mut topk_carry)
                    .with_context(|| format!("GLM5.2 layer {layer} attention half"))?;
            let mlp = match &weights.mlp {
                Glm52LayerMlp::Dense(dense) => {
                    glm52_dense_mlp_forward(ctx, dense, &boundary.normed)?
                }
                Glm52LayerMlp::MoeEp8(moe) => {
                    let route = run_router(ctx, &moe.router, &boundary.normed)?;
                    // global_tokens = 1: the bs=1 coordinator steps one token
                    // across the whole EP8 group (expert_step matches).
                    let routed = glm52_moe_ep8_routed_forward(
                        ctx,
                        ep8,
                        &moe.bank,
                        Some((&boundary.normed, &route)),
                        1,
                    )
                    .with_context(|| format!("GLM5.2 layer {layer} EP8 MoE"))?
                    .context("rank-0 EP8 MoE returned no combined output")?;
                    let shared = moe.shared.forward(ctx, &boundary.normed)?;
                    let routed_hs = HiddenStates {
                        data: routed,
                        hidden_dim: GLM52_HIDDEN,
                        seq_len: 1,
                    };
                    let shared_hs = HiddenStates {
                        data: shared,
                        hidden_dim: GLM52_HIDDEN,
                        seq_len: 1,
                    };
                    add_batch(ctx, &routed_hs, &shared_hs)?.data
                }
                Glm52LayerMlp::Moe(_) => {
                    anyhow::bail!("GLM5.2 EP8 spine built an EP1 MoE layer — loader bug")
                }
            };
            hidden = glm52_layer_finish(ctx, boundary.residual, mlp)?;
        }

        let hidden_vec = DeviceVec {
            data: hidden,
            len: GLM52_HIDDEN,
        };
        let normed = glm52_final_norm(ctx, &hidden_vec, &self.final_norm)?;
        let logits = glm52_lm_head(ctx, &normed, &self.lm_head)?;
        let logits_host = ctx.stream.clone_dtoh(&logits.data)?;
        greedy_argmax(&logits_host)
    }
}

/// Host greedy argmax over the bf16 logits (bs=1 bring-up; device sampling is
/// the PR5 scheduler's job). Fails loudly on a non-finite top logit.
fn greedy_argmax(logits: &[bf16]) -> Result<u32> {
    ensure!(logits.len() == GLM52_VOCAB, "GLM5.2 logits length drifted");
    let mut best = f32::NEG_INFINITY;
    let mut best_idx = 0usize;
    for (idx, v) in logits.iter().enumerate() {
        let v = v.to_f32();
        if v > best {
            best = v;
            best_idx = idx;
        }
    }
    ensure!(
        best.is_finite(),
        "GLM5.2 greedy argmax found no finite logit (top = {best})"
    );
    Ok(best_idx as u32)
}

/// Ranks 1..7: one expert bank per MoE layer (3..=77, ascending), nothing else.
pub(crate) struct Glm52ExpertRankModel {
    banks: Vec<Glm52MoeExpertBank>,
}

impl Glm52ExpertRankModel {
    pub(crate) fn build(ctx: &DeviceContext, w: &mut Glm52RankGpuWeights) -> Result<Self> {
        let banks = (GLM52_DENSE_LAYERS..GLM52_LAYERS)
            .map(|layer| {
                Glm52MoeExpertBank::from_regions(ctx, w.take_expert_layer(layer)?)
                    .with_context(|| format!("build GLM5.2 expert bank for layer {layer}"))
            })
            .collect::<Result<Vec<_>>>()?;
        // The MTP layer's experts are loaded (checkpoint-coverage validation)
        // but out of campaign scope — drop them to reclaim ~1.2 GiB.
        let _ = w.take_expert_layer(crate::weights::GLM52_MTP_LAYER)?;
        Ok(Self { banks })
    }

    /// Replay one decode step's 75 MoE collectives in rank-0's layer order.
    pub(crate) fn expert_step(
        &self,
        ctx: &DeviceContext,
        ep8: &mut Glm52MoeEp8State,
    ) -> Result<()> {
        for (idx, bank) in self.banks.iter().enumerate() {
            let combined =
                glm52_moe_ep8_routed_forward(ctx, ep8, bank, None, 1).with_context(|| {
                    format!("GLM5.2 expert rank MoE layer {}", idx + GLM52_DENSE_LAYERS)
                })?;
            ensure!(
                combined.is_none(),
                "GLM5.2 expert rank unexpectedly produced a combined output"
            );
        }
        Ok(())
    }
}
