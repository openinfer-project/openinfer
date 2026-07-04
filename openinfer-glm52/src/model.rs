//! GLM5.2 DP8/EP8 full-model decode: every rank owns the whole non-expert
//! path (embed → 78 decoder layers → final norm → lm_head → greedy argmax)
//! plus its 32 local experts, and forwards one of two batch buckets per step
//! — a single row, or the full `GLM52_MAX_BATCH_PER_RANK` rows (real request
//! tokens in occupied slots, padding rows elsewhere; prefill rides decode
//! token-by-token). Each slot owns a disjoint `GLM52_MAX_MODEL_LEN`-token
//! region of the paged KV/index caches, so pad rows write only their own
//! dead slots.
//!
//! Every step, all 8 ranks run the forward in lock-step with the SAME bucket
//! (the coordinator's global decision), each dispatching exactly that many
//! rows into every MoE layer's DeepEP collective — the collectives require
//! all ranks to enter in the same layer order 3..=77 with the agreed global
//! row count, and the per-bucket fixed row count keeps every step's kernel
//! shapes identical within a bucket (the whole-step CUDA graphs' contract).

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE, GLM52_FLASHMLA_SPARSE_TOPK, Glm52FlashMlaSparseDecode,
    Glm52IndexerCacheLayout, add_into, argmax_bf16_split_into, embedding_rows_into,
    glm52_flashmla_sparse_decode_num_sm_parts, rms_norm_rows_into,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

use crate::bookend::{glm52_embed_into, glm52_final_norm_into, glm52_lm_head_into};
use crate::config::{GLM52_DENSE_LAYERS, GLM52_HIDDEN, GLM52_LAYERS, GLM52_VOCAB};
use crate::dense::{Glm52DenseMlpWeights, glm52_dense_mlp_forward_into};
use crate::fp8::ProjWeight;
use crate::indexer::{Glm52IndexerLayerWeights, Glm52IndexerScratch};
use crate::layer::{
    GLM52_RMS_EPS, Glm52DecodeStep, Glm52DecoderLayerWeights, Glm52LayerCaches, Glm52LayerIndexer,
    Glm52LayerMlp, glm52_layer_attention_half, glm52_layer_finish, glm52_layer_finish_fused,
};
use crate::mla_decode::{Glm52MlaLayerWeights, Glm52MlaSchedMetadata};
use crate::moe_decode::{
    Glm52MoeExpertBank, Glm52MoeRouterWeights, Glm52MoeSharedExpert, run_router_into,
};
use crate::moe_ep8::{Glm52MoeEp8LayerWeights, Glm52MoeEp8State, glm52_moe_ep8_routed_forward};
use crate::scratch::Glm52DecodeScratch;
use crate::weights::{Glm52RankGpuWeights, retype_owned};

/// Per-request context cap: `prompt + max_tokens - 1 <= GLM52_MAX_MODEL_LEN`.
/// Sizes each slot's region of the per-layer MLA and index-K caches at build
/// time.
pub(crate) const GLM52_MAX_MODEL_LEN: usize = 4096;

/// The fixed per-rank decode batch: every step forwards exactly this many
/// rows (request tokens in occupied slots, padding rows elsewhere), so every
/// step has the same kernel shapes by construction — the whole-step CUDA
/// graph's contract. Each slot owns a `GLM52_MAX_MODEL_LEN`-token region of
/// the paged caches. The CUDA side instantiates the batched weight-only GEMV
/// for exactly this batch (`kBatchedGemvBatch` in `glm52_moe_gemv.cu`) — a
/// drift crashes at the launch boundary.
pub(crate) const GLM52_MAX_BATCH_PER_RANK: usize = 8;

/// The step's batch bucket, agreed globally by the coordinator: `One` forwards
/// a single row (the request — or padding — in `slot`), `Full` forwards all
/// `GLM52_MAX_BATCH_PER_RANK` rows. Two buckets, not a continuum: each bucket
/// has its own captured CUDA graphs, and the batched GEMV kernel is
/// instantiated for exactly {1, 8}. An idle or lightly-loaded server (≤ 1
/// request per rank) keeps the 1-row step cost; the 8-row step is only paid
/// when some rank holds ≥ 2 requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Glm52StepShape {
    One { slot: usize },
    Full,
}

/// Short-context attention tier: while `seq_len <= 256` the DSA top-256 IS
/// the full token set (exactly like top-2048 is below 2048 tokens), so the
/// short-tier graph attends the same tokens at 1/8 the FlashMLA padding work
/// — the V3.2 sparse kernel always walks all `topk` index slots (`-1` pads
/// run the full load+GEMM and are only masked in softmax). Must be a multiple
/// of the 64-entry topk block.
pub(crate) const GLM52_MLA_TOPK_SHORT: usize = 256;

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
            ctx,
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
                ctx,
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
    /// FlashMLA plans, indexed `[bucket][tier]` (see the index constants
    /// below): the short tier (topk 256, few SM parts) serves steps with
    /// `seq_len <= GLM52_MLA_TOPK_SHORT`, the full tier (topk 2048)
    /// everything beyond. Same math on the same real tokens — only the padded
    /// index-walk length and the row count differ.
    mla_scheds: [[Glm52MlaSchedMetadata; 2]; 2],
    index_cache_layout: Glm52IndexerCacheLayout,
    /// `[GLM52_MAX_BATCH_PER_RANK, blocks_per_slot]` — row b maps slot b's
    /// context into its disjoint region of the index-K cache. Static: written
    /// once at build.
    block_table: CudaSlice<i32>,
    /// `[1, blocks_per_slot]` — the 1-row bucket's block table. The prologue
    /// rewrites it (dtod from the static table's row for the active slot)
    /// each 1-row step, so the captured b1 graphs address whichever slot the
    /// request lives in through device data, never a baked slot id.
    block_table_b1: CudaSlice<i32>,
    blocks_per_slot: usize,
    slot_mapping: CudaSlice<i64>,
    seq_lens: CudaSlice<i32>,
    /// Device-resident rope tables for every position (`[GLM52_MAX_MODEL_LEN,
    /// ROPE_HALF]` as a gatherable matrix); the prologue gathers each row's
    /// position row instead of recomputing on the host and copying up.
    cos_table: DeviceMatrix,
    sin_table: DeviceMatrix,
    positions: CudaSlice<u32>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    token_ids: CudaSlice<u32>,
    /// Scratch arenas, indexed `[bucket]`.
    scratches: [Glm52DecodeScratch; 2],
    /// The whole-step decode graphs, indexed `[bucket][tier]`: each is
    /// captured on this rank's first step in that shape, replayed every step
    /// after. Valid forever — within a shape every step has the same kernel
    /// sequence and the same (arena) pointers by construction; the per-step
    /// inputs are the device buffers the prologue rewrites.
    graphs: [[CudaGraphState; 2]; 2],
}

/// Index constants for the per-shape arrays: every consumer selects with the
/// same `[BUCKET_*][TIER_*]` pair computed once per step, so a graph can
/// never be taken from one shape slot and restored into another.
const BUCKET_FULL: usize = 0;
const BUCKET_ONE: usize = 1;
const TIER_FULL: usize = 0;
const TIER_SHORT: usize = 1;

impl Glm52RankModel {
    pub(crate) fn build(ctx: &DeviceContext, w: &mut Glm52RankGpuWeights) -> Result<Self> {
        let batch = GLM52_MAX_BATCH_PER_RANK;
        let num_sm_parts = glm52_flashmla_sparse_decode_num_sm_parts()?;
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: batch,
            num_blocks: batch * GLM52_MAX_MODEL_LEN.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE),
            topk: GLM52_FLASHMLA_SPARSE_TOPK,
            num_sm_parts,
            sm_scale: SM_SCALE,
        };
        // The short tier walks only 256/64 = 4 index blocks per row — more
        // SM parts than blocks would just be empty splits for the combine.
        let contract_short = Glm52FlashMlaSparseDecode {
            topk: GLM52_MLA_TOPK_SHORT,
            num_sm_parts: num_sm_parts.min(GLM52_MLA_TOPK_SHORT / 64),
            ..contract
        };
        // The 1-row bucket: same paged cache (num_blocks is cache geometry,
        // not batch), one query row.
        let contract_b1 = Glm52FlashMlaSparseDecode {
            batch_size: 1,
            ..contract
        };
        let contract_b1_short = Glm52FlashMlaSparseDecode {
            batch_size: 1,
            ..contract_short
        };
        // Each slot owns `blocks_per_slot` consecutive index-K cache blocks;
        // the whole cache holds every slot's region back-to-back.
        let blocks_per_slot = GLM52_MAX_MODEL_LEN.div_ceil(INDEX_CACHE_BLOCK);
        let index_cache_layout = Glm52IndexerCacheLayout {
            cache_blocks: batch * blocks_per_slot,
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

        // Row b of the block table maps slot b's context onto its own cache
        // region: block j of slot b is global block b*blocks_per_slot + j.
        let block_table_host: Vec<i32> = (0..batch * blocks_per_slot).map(|i| i as i32).collect();
        let mut block_table = ctx.stream.alloc_zeros::<i32>(batch * blocks_per_slot)?;
        ctx.stream
            .memcpy_htod(&block_table_host, &mut block_table)?;

        let mut cos_host = Vec::with_capacity(GLM52_MAX_MODEL_LEN * ROPE_HALF);
        let mut sin_host = Vec::with_capacity(GLM52_MAX_MODEL_LEN * ROPE_HALF);
        for position in 0..GLM52_MAX_MODEL_LEN {
            let (cos_row, sin_row) = rope_tables(position);
            cos_host.extend_from_slice(&cos_row);
            sin_host.extend_from_slice(&sin_row);
        }
        let mut cos_table_data = ctx
            .stream
            .alloc_zeros::<bf16>(GLM52_MAX_MODEL_LEN * ROPE_HALF)?;
        let mut sin_table_data = ctx
            .stream
            .alloc_zeros::<bf16>(GLM52_MAX_MODEL_LEN * ROPE_HALF)?;
        ctx.stream.memcpy_htod(&cos_host, &mut cos_table_data)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin_table_data)?;

        let mqa_shape = Glm52IndexerScratch::decode_shape(
            batch,
            index_cache_layout,
            blocks_per_slot,
            NUM_SMS,
            GLM52_MAX_MODEL_LEN,
        );
        let mqa_shape_b1 = Glm52IndexerScratch::decode_shape(
            1,
            index_cache_layout,
            blocks_per_slot,
            NUM_SMS,
            GLM52_MAX_MODEL_LEN,
        );
        Ok(Self {
            layers,
            caches,
            embed,
            final_norm,
            lm_head,
            mla_scheds: [
                [
                    Glm52MlaSchedMetadata::new(ctx, contract)?,
                    Glm52MlaSchedMetadata::new(ctx, contract_short)?,
                ],
                [
                    Glm52MlaSchedMetadata::new(ctx, contract_b1)?,
                    Glm52MlaSchedMetadata::new(ctx, contract_b1_short)?,
                ],
            ],
            index_cache_layout,
            block_table,
            block_table_b1: ctx.stream.alloc_zeros::<i32>(blocks_per_slot)?,
            blocks_per_slot,
            slot_mapping: ctx.stream.alloc_zeros::<i64>(batch)?,
            seq_lens: ctx.stream.alloc_zeros::<i32>(batch)?,
            cos_table: DeviceMatrix {
                data: cos_table_data,
                rows: GLM52_MAX_MODEL_LEN,
                cols: ROPE_HALF,
            },
            sin_table: DeviceMatrix {
                data: sin_table_data,
                rows: GLM52_MAX_MODEL_LEN,
                cols: ROPE_HALF,
            },
            positions: ctx.stream.alloc_zeros::<u32>(batch)?,
            cos: ctx.stream.alloc_zeros::<bf16>(batch * ROPE_HALF)?,
            sin: ctx.stream.alloc_zeros::<bf16>(batch * ROPE_HALF)?,
            token_ids: ctx.stream.alloc_zeros::<u32>(batch)?,
            scratches: [
                Glm52DecodeScratch::new(ctx, &contract, mqa_shape)?,
                Glm52DecodeScratch::new(ctx, &contract_b1, mqa_shape_b1)?,
            ],
            graphs: [
                [CudaGraphState::new(), CudaGraphState::new()],
                [CudaGraphState::new(), CudaGraphState::new()],
            ],
        })
    }

    /// One lock-step step: feed each active slot's `(token, position)` row,
    /// return the greedy next-token id per slot. Enters 75 MoE collectives —
    /// every other rank must be stepping concurrently WITH THE SAME BUCKET
    /// (`shape` batch count): the coordinator agrees the bucket globally per
    /// step. `Full` forwards all `GLM52_MAX_BATCH_PER_RANK` rows (unoccupied
    /// slots carry the padding row, whose cache writes land in that slot's
    /// own dead region); `One { slot }` forwards a single row — the request
    /// (or padding) living in `slot` — at the 1-row step cost.
    ///
    /// The step body (embed → 78 layers → lm_head → argmax) is captured into
    /// a CUDA graph on the first call in each (attention tier × bucket) shape
    /// and replayed afterwards: one graph launch instead of ~4155 kernel
    /// launches per rank per step. The prologue rewrites the device input
    /// buffers the captured kernels read (per-row rope rows, slots, seq_lens,
    /// tokens, and — for the 1-row bucket — the active slot's block-table
    /// row), and the epilogue reads back the per-row argmax results — both
    /// outside the graph. Capture-time safety: stream capture records without
    /// executing, and in lock-step all ranks enter a new shape on the same
    /// global step, so the DeepEP collectives first execute together on every
    /// rank's first launch of that shape (the same argument as the kimi
    /// decode graph; the ceiling is the ~100 s DeepEP device timeout against
    /// a capture window of tens of ms — already proven by the mid-serving
    /// tier-crossing capture).
    pub(crate) fn decode_step(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        ep8: &mut Glm52MoeEp8State,
        inputs: &[(u32, usize); GLM52_MAX_BATCH_PER_RANK],
        shape: Glm52StepShape,
    ) -> Result<[u32; GLM52_MAX_BATCH_PER_RANK]> {
        let (batch, base_slot) = match shape {
            Glm52StepShape::One { slot } => {
                ensure!(
                    slot < GLM52_MAX_BATCH_PER_RANK,
                    "GLM5.2 1-row step slot {slot} outside the {GLM52_MAX_BATCH_PER_RANK}-slot batch"
                );
                (1, slot)
            }
            Glm52StepShape::Full => (GLM52_MAX_BATCH_PER_RANK, 0),
        };
        let mut tokens_host = [0u32; GLM52_MAX_BATCH_PER_RANK];
        let mut positions_host = [0u32; GLM52_MAX_BATCH_PER_RANK];
        let mut slots_host = [0i64; GLM52_MAX_BATCH_PER_RANK];
        let mut seq_lens_host = [0i32; GLM52_MAX_BATCH_PER_RANK];
        for row in 0..batch {
            let slot = base_slot + row;
            let (token, position) = inputs[slot];
            ensure!(
                position < GLM52_MAX_MODEL_LEN,
                "GLM5.2 slot {slot} position {position} exceeds the model-length cap {GLM52_MAX_MODEL_LEN}"
            );
            tokens_host[row] = token;
            positions_host[row] = position as u32;
            // Slot b owns cache tokens [b*MAX_LEN, (b+1)*MAX_LEN) — the same
            // global slot id addresses both the MLA and index-K caches.
            slots_host[row] = (slot * GLM52_MAX_MODEL_LEN + position) as i64;
            seq_lens_host[row] = (position + 1) as i32;
        }
        ctx.stream.memcpy_htod(&tokens_host, &mut self.token_ids)?;
        ctx.stream
            .memcpy_htod(&positions_host, &mut self.positions)?;
        ctx.stream
            .memcpy_htod(&slots_host, &mut self.slot_mapping)?;
        ctx.stream.memcpy_htod(&seq_lens_host, &mut self.seq_lens)?;
        // Gather each row's rotary table row (a bit-exact row copy).
        embedding_rows_into(ctx, &self.cos_table, &self.positions, batch, &mut self.cos)?;
        embedding_rows_into(ctx, &self.sin_table, &self.positions, batch, &mut self.sin)?;
        if batch == 1 {
            // Point the 1-row block table at the active slot's cache region —
            // device data, so the captured b1 graphs replay against whichever
            // slot holds the request.
            let src = self
                .block_table
                .slice(base_slot * self.blocks_per_slot..(base_slot + 1) * self.blocks_per_slot);
            ctx.stream.memcpy_dtod(&src, &mut self.block_table_b1)?;
        }

        // Attention tier: while EVERY forwarded row's context fits in the
        // short top-k, top-256 selects exactly the same tokens as top-2048
        // (all of them) — the short graph only walks 1/8 of the padded
        // FlashMLA index slots. Pad rows sit at position 0 and never lift the
        // tier; a mixed-tier batch runs at the full tier (correct for every
        // row, only the padded walk length differs). Each shape has its own
        // captured graph; the lazily captured graph records without
        // executing, so a mid-serving capture holds the other ranks'
        // collectives for tens of ms — far under the ~100 s DeepEP device
        // timeout (same argument as the first-step capture).
        let short_tier = (0..batch)
            .map(|row| inputs[base_slot + row].1 + 1)
            .max()
            .is_some_and(|longest| longest <= GLM52_MLA_TOPK_SHORT);
        // The step's shape indices — computed once; every per-shape array
        // (plans, graphs, scratches) is selected with this same pair.
        let bucket = if batch == 1 { BUCKET_ONE } else { BUCKET_FULL };
        let tier = if short_tier { TIER_SHORT } else { TIER_FULL };
        let step = Glm52DecodeStep {
            mla_cos: &self.cos,
            mla_sin: &self.sin,
            idx_cos: &self.cos,
            idx_sin: &self.sin,
            mla_sched: &self.mla_scheds[bucket][tier],
            index_cache_layout: self.index_cache_layout,
            slot_mapping: &self.slot_mapping,
            block_table: if bucket == BUCKET_ONE {
                &self.block_table_b1
            } else {
                &self.block_table
            },
            seq_lens: &self.seq_lens,
        };
        // Every rank must pass the same global token count into the MoE
        // collectives — guaranteed by the coordinator agreeing the bucket.
        let global_tokens = crate::weights::GLM52_EP_RANKS * batch;

        let mut graph = std::mem::take(&mut self.graphs[bucket][tier]);
        let s = &mut self.scratches[bucket];
        let result = graph.run_or_capture(ctx, || {
            run_step_body(
                ctx,
                aux,
                ep8,
                &self.layers,
                &mut self.caches,
                &self.embed,
                &self.final_norm,
                &self.lm_head,
                &self.token_ids,
                &step,
                s,
                global_tokens,
            )
        });
        self.graphs[bucket][tier] = graph;
        result?;

        let s = &self.scratches[bucket];
        let top_values = ctx.stream.clone_dtoh(&s.argmax_values)?;
        let top_indices = ctx.stream.clone_dtoh(&s.argmax_indices)?;
        let mut outputs = [0u32; GLM52_MAX_BATCH_PER_RANK];
        for row in 0..batch {
            let slot = base_slot + row;
            let top_value = top_values[row].to_f32();
            let top_index = top_indices[row];
            ensure!(
                top_value.is_finite(),
                "GLM5.2 slot {slot} greedy argmax found no finite logit (top = {top_value})"
            );
            ensure!(
                (0..GLM52_VOCAB as i32).contains(&top_index),
                "GLM5.2 slot {slot} greedy argmax index {top_index} outside the vocab"
            );
            outputs[slot] = top_index as u32;
        }
        Ok(outputs)
    }
}

/// The captured region of one decode step: embed → 78 layers → lm_head →
/// device argmax over the step's `batch` rows (read from the attend plan —
/// the single source of truth for the step's row count). Shared verbatim by
/// both batch buckets; only the plan, scratch, block table, and
/// `global_tokens` differ per shape.
#[allow(clippy::too_many_arguments)]
fn run_step_body(
    ctx: &DeviceContext,
    aux: &DeviceContext,
    ep8: &mut Glm52MoeEp8State,
    layers: &[Glm52DecoderLayerWeights],
    caches: &mut [Glm52LayerCaches],
    embed: &DeviceMatrix,
    final_norm: &DeviceVec,
    lm_head: &DeviceMatrix,
    token_ids: &CudaSlice<u32>,
    step: &Glm52DecodeStep<'_>,
    s: &mut Glm52DecodeScratch,
    global_tokens: usize,
) -> Result<()> {
    let batch = step.mla_sched.batch();
    glm52_embed_into(ctx, embed, token_ids, batch, &mut s.hidden)?;
    // Layer 0's input norm is standalone (the embedding is the residual);
    // every later layer's input norm is fused into the previous layer's
    // closing add (`glm52_layer_finish_fused`).
    rms_norm_rows_into(
        ctx,
        &s.hidden.data,
        &layers[0].input_ln,
        GLM52_RMS_EPS,
        GLM52_HIDDEN,
        batch,
        &mut s.layer.normed.data,
    )?;
    let mut carry_ready = false;
    for (layer, (weights, cache)) in layers.iter().zip(caches.iter_mut()).enumerate() {
        let parity = layer % 2;
        glm52_layer_attention_half(
            ctx,
            Some(aux),
            weights,
            cache,
            step,
            s,
            &mut carry_ready,
            parity,
            layer == 0,
        )
        .with_context(|| format!("GLM5.2 layer {layer} attention half"))?;
        match &weights.mlp {
            Glm52LayerMlp::Dense(dense) => glm52_dense_mlp_forward_into(
                ctx,
                dense,
                &s.layer.normed2,
                &mut s.dense_mlp,
                &mut s.layer.mlp_out,
            )?,
            Glm52LayerMlp::MoeEp8(moe) => {
                // Fork: the shared expert only needs `normed2`, so it runs on
                // the aux stream concurrently with the routed path's
                // dispatch/grouped-GEMM/combine — the cooperative collectives
                // occupy a fixed SM slice and mostly wait on peers, leaving
                // the rest of the GPU free. The events recorded here during
                // capture become graph edges; replay keeps the parallel
                // branches.
                let normed_ready = ctx.stream.record_event(None)?;
                aux.stream.wait(&normed_ready)?;
                moe.shared.forward_into(
                    aux,
                    &s.layer.normed2,
                    &mut s.shared_mlp,
                    &mut s.layer.shared_out,
                )?;
                let shared_done = aux.stream.record_event(None)?;

                run_router_into(ctx, &moe.router, &s.layer.normed2, &mut s.router)?;
                let dispatched = glm52_moe_ep8_routed_forward(
                    ctx,
                    ep8,
                    &moe.bank,
                    Some((&s.layer.normed2, &s.router.route, batch)),
                    global_tokens,
                )
                .with_context(|| format!("GLM5.2 layer {layer} EP8 MoE"))?;
                ensure!(
                    dispatched,
                    "EP8 MoE returned no combined output for the dispatched rows"
                );
                // Join: the closing add consumes both branches.
                ctx.stream.wait(&shared_done)?;
                add_into(
                    ctx,
                    ep8.combined(),
                    &s.layer.shared_out,
                    batch * GLM52_HIDDEN,
                    &mut s.layer.mlp_out,
                )?;
            }
            Glm52LayerMlp::Moe(_) => {
                anyhow::bail!("GLM5.2 EP8 spine built an EP1 MoE layer — loader bug")
            }
        }
        if layer + 1 < layers.len() {
            glm52_layer_finish_fused(ctx, s, parity, &layers[layer + 1].input_ln)?;
        } else {
            glm52_layer_finish(ctx, s, parity)?;
        }
    }

    glm52_final_norm_into(ctx, &s.hidden, final_norm, batch, &mut s.final_normed)?;
    glm52_lm_head_into(ctx, &s.final_normed, lm_head, batch, &mut s.logits)?;
    // Device greedy argmax per row (same semantics as a host scan: lowest
    // index wins ties, NaN never wins) — the step's egress shrinks from the
    // full vocab rows to 6 bytes per row, and the kernel chain ends on-device
    // (the graph boundary). Two-stage: per-4096-tile partials in parallel,
    // then one finalize block per row — bit-identical to the single-block
    // scan (the partials carry global indices, same total order), and each
    // row's result is independent of its slot-mates.
    argmax_bf16_split_into(
        ctx,
        &s.logits.data,
        batch,
        GLM52_VOCAB,
        &mut s.argmax_partial_values,
        &mut s.argmax_partial_indices,
        &mut s.argmax_values,
        &mut s.argmax_indices,
    )
}
