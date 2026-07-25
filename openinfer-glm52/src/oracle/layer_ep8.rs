//! EP8 layer-6 MoE oracle gate: the PR3 decoder-layer gate re-run with the
//! MoE half going through the real 8-GPU DeepEP dispatch/combine.
//!
//! Rank 0 walks the full decoder layer (attention + indexer + EP8 MoE +
//! shared expert) over the same seeded input as the EP1 gate; ranks 1..7 hold
//! their 32 local experts and replay one collective per position. The probe
//! constants, tolerance, and router tie-flip allowance are shared verbatim
//! with `layer_oracle_gate` — passing here proves the collective path
//! (dispatch → re-quant → metadata → grouped GEMMs → combine) computes the
//! same layer output as the local EP1 chain.

use std::sync::Arc;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_TOPK;
use openinfer_kernels::ops::Glm52FlashMlaSparseDecode;
use openinfer_kernels::ops::Glm52IndexerCacheLayout;
use openinfer_kernels::ops::add_scaled_bf16_into;
use openinfer_kernels::ops::glm52_ep_deepep_unique_id;
use openinfer_kernels::ops::glm52_flashmla_sparse_decode_num_sm_parts;
use openinfer_kernels::tensor::DeviceContext;

use super::layer::GateLayerMlp;
use super::layer::LayerTensors;
use super::layer::MOE_ORACLE_CTX;
use super::layer::MOE_ORACLE_HIDDEN_DIGEST;
use super::layer::MOE_ORACLE_INPUT_SCALE;
use super::layer::MOE_ORACLE_LAYER;
use super::layer::MOE_ORACLE_LAYER_PROBES;
use super::layer::MOE_ORACLE_LAYER_TOL;
use super::layer::MOE_ORACLE_SEED;
use super::layer::assert_layer_probes;
use super::layer::checked_hidden;
use super::layer::load_decoder_layer;
use super::layer::load_rank_expert_bank;
use super::layer::model_path;
use crate::config::GLM52_INDEX_HEAD_DIM;
use crate::config::GLM52_ROPE_HALF;
use crate::config::GLM52_SM_SCALE;
use crate::indexer::Glm52IndexerScratch;
use crate::layer::Glm52DecodeStep;
use crate::layer::Glm52LayerCaches;
use crate::layer::Glm52LayerMlp;
use crate::layer::glm52_layer_attention_half;
use crate::layer::glm52_layer_finish;
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::model::GLM52_DECODE_BUCKETS;
use crate::model::INDEX_CACHE_BLOCK;
use crate::model::NUM_SMS;
use crate::model::rope_tables;
use crate::moe_decode::HIDDEN;
use crate::moe_decode::run_ep_router;
use crate::moe_ep8::Glm52MoeEp8State;
use crate::moe_ep8::glm52_moe_ep8_routed_forward;
use crate::scratch::Glm52DecodeScratch;

const EP_RANKS: usize = 8;
/// Every global-token protocol value the production coordinator can agree on
/// — one per decode bucket, largest first (the worst-case row bound leads).
/// The gate replays its collectives at each, pinning every bucket's
/// collective row-bound math to the oracle instead of leaving it to e2e
/// parity alone.
const GLOBAL_TOKEN_BUCKETS: [usize; GLM52_DECODE_BUCKETS.len()] = {
    let mut buckets = [0usize; GLM52_DECODE_BUCKETS.len()];
    let mut i = 0;
    while i < GLM52_DECODE_BUCKETS.len() {
        buckets[i] = EP_RANKS * GLM52_DECODE_BUCKETS[GLM52_DECODE_BUCKETS.len() - 1 - i];
        i += 1;
    }
    buckets
};

#[test]
#[ignore = "requires 8×H200 + GLM-5.2-FP8 checkpoint + NCCL >= 2.30.4 + DeepGEMM env"]
fn layer_moe_ep8_oracle_gate() -> Result<()> {
    let hidden_host = checked_hidden(
        MOE_ORACLE_SEED,
        MOE_ORACLE_CTX,
        MOE_ORACLE_INPUT_SCALE,
        MOE_ORACLE_HIDDEN_DIGEST,
    )?;
    let unique_id = glm52_ep_deepep_unique_id(8)?;
    let tensors = Arc::new(LayerTensors::load(&model_path(), MOE_ORACLE_LAYER)?);

    // Expert ranks: pack the 32 local experts, then replay one collective per
    // position. Context creation inside is collective with rank 0's below.
    let handles: Vec<_> = (1..EP_RANKS)
        .map(|rank| {
            let tensors = Arc::clone(&tensors);
            std::thread::Builder::new()
                .name(format!("ep8-gate-rank-{rank}"))
                .spawn(move || -> Result<()> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let bank =
                        load_rank_expert_bank(&ctx, &tensors, MOE_ORACLE_LAYER, rank, EP_RANKS)?;
                    let mut ep8 = Glm52MoeEp8State::new(&ctx, &unique_id, EP_RANKS, rank)?;
                    for global_tokens in GLOBAL_TOKEN_BUCKETS {
                        for _position in 0..MOE_ORACLE_CTX {
                            let dispatched = glm52_moe_ep8_routed_forward(
                                &ctx,
                                &mut ep8,
                                &bank,
                                None,
                                global_tokens,
                            )?;
                            ensure!(!dispatched, "expert rank produced a combined output");
                        }
                    }
                    Ok(())
                })
                .expect("spawn ep8 gate rank thread")
        })
        .collect();

    // Rank 0: full decoder layer with the EP8 MoE half, prefill-via-decode.
    let ctx = DeviceContext::new_with_device(0)?;
    let w = load_decoder_layer(
        &ctx,
        &model_path(),
        MOE_ORACLE_LAYER,
        GateLayerMlp::MoeEp8Rank0,
    )?;
    let mut ep8 = Glm52MoeEp8State::new(&ctx, &unique_id, EP_RANKS, 0)?;
    // Replay the layer once per global-token bucket, in the same order as the
    // expert threads' collective loops.
    let outputs: Result<Vec<LayerEp8Outputs>> = GLOBAL_TOKEN_BUCKETS
        .into_iter()
        .map(|global_tokens| {
            run_layer_prefill_ep8(
                &ctx,
                &w,
                &mut ep8,
                &hidden_host,
                MOE_ORACLE_CTX,
                global_tokens,
            )
        })
        .collect();

    // The DeepEP context drop is collective: the expert threads drop theirs
    // right after their last collective and spin in the destroy barrier, so
    // rank 0 must drop BEFORE joining them (join-then-drop deadlocks until
    // the ~100 s device timeout traps every rank).
    drop(ep8);
    for (rank, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .expect("ep8 gate rank thread panicked")
            .with_context(|| format!("ep8 gate rank {}", rank + 1))?;
    }
    for (outputs, global_tokens) in outputs?.iter().zip(GLOBAL_TOKEN_BUCKETS) {
        assert_layer_probes(
            &format!("layer6/moe/ep8/g{global_tokens}"),
            &outputs.hidden,
            MOE_ORACLE_LAYER_PROBES,
            MOE_ORACLE_LAYER_TOL,
            4,
        );
    }
    Ok(())
}

/// The EP8 variant of the gate's prefill-via-decode walk: same decode
/// environment as `oracle::layer::run_layer_prefill`, with the MLP half
/// driven through the collective.
pub(super) struct LayerEp8Outputs {
    pub(super) hidden: Vec<f32>,
    pub(super) post_attention: Vec<f32>,
    pub(super) mlp: Vec<f32>,
}

fn run_moe_ep8_half(
    ctx: &DeviceContext,
    moe: &crate::moe_ep8::Glm52MoeEp8LayerWeights,
    ep8: &mut Glm52MoeEp8State,
    scratch: &mut Glm52DecodeScratch,
    global_tokens: usize,
) -> Result<()> {
    let route = run_ep_router(ctx, &moe.router, scratch.layer.normed2.data())?;
    let dispatched = glm52_moe_ep8_routed_forward(
        ctx,
        ep8,
        &moe.bank,
        Some((scratch.layer.normed2.data(), &route, 1)),
        global_tokens,
    )?;
    ensure!(dispatched, "rank-0 EP8 MoE returned no combined output");
    moe.shared.forward_into(
        ctx,
        scratch.layer.normed2.data(),
        &mut scratch.shared_mlp,
        scratch.layer.shared_out.data_mut(),
    )?;
    add_scaled_bf16_into(
        ctx,
        ep8.combined(),
        crate::config::GLM52_ROUTED_SCALING_FACTOR as f32,
        scratch.layer.shared_out.data(),
        HIDDEN,
        scratch.layer.mlp_out.data_mut(),
    )
}

pub(super) fn run_layer_prefill_ep8(
    ctx: &DeviceContext,
    w: &crate::layer::Glm52DecoderLayerWeights,
    ep8: &mut Glm52MoeEp8State,
    hidden_host: &[bf16],
    oracle_ctx: usize,
    global_tokens: usize,
) -> Result<LayerEp8Outputs> {
    let Glm52LayerMlp::MoeEp8(moe) = &w.mlp else {
        anyhow::bail!("ep8 gate requires the MoeEp8 layer weights");
    };
    let contract = Glm52FlashMlaSparseDecode {
        batch_size: 1,
        num_blocks: oracle_ctx.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE),
        topk: GLM52_FLASHMLA_SPARSE_TOPK,
        num_sm_parts: glm52_flashmla_sparse_decode_num_sm_parts()?,
        sm_scale: GLM52_SM_SCALE,
    };
    let index_blocks = oracle_ctx.div_ceil(INDEX_CACHE_BLOCK);
    let index_cache_layout = Glm52IndexerCacheLayout {
        cache_blocks: index_blocks,
        cache_block_size: INDEX_CACHE_BLOCK,
        cache_block_stride_bytes: INDEX_CACHE_BLOCK * (GLM52_INDEX_HEAD_DIM + 4),
    };
    let mut caches = Glm52LayerCaches {
        mla_cache: ctx
            .stream
            .alloc_zeros::<u8>(contract.packed_kv_cache_len())?,
        index_k_cache: Some(
            ctx.stream
                .alloc_zeros::<u8>(index_cache_layout.min_cache_bytes()?)?,
        ),
    };

    let block_table_host: Vec<i32> = (0..index_blocks as i32).collect();
    let mut block_table = ctx.stream.alloc_zeros::<i32>(index_blocks)?;
    ctx.stream
        .memcpy_htod(&block_table_host, &mut block_table)?;
    let mut slot_mapping = ctx.stream.alloc_zeros::<i64>(1)?;
    let mut seq_lens = ctx.stream.alloc_zeros::<i32>(1)?;
    let mut cos = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
    let mut sin = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
    let mla_sched = Glm52MlaSchedMetadata::new(ctx, contract, w.mla.heads)?;

    let mqa_shape = Glm52IndexerScratch::paged_mqa_shape(
        1,
        index_cache_layout,
        index_blocks,
        NUM_SMS,
        oracle_ctx,
    );
    let mut scratch =
        Glm52DecodeScratch::new(ctx, &contract, mqa_shape, crate::config::GLM52_HEADS, false)?;

    let mut hidden_outputs = Vec::with_capacity(oracle_ctx * HIDDEN);
    let mut post_attention_outputs = Vec::with_capacity(oracle_ctx * HIDDEN);
    let mut mlp_outputs = Vec::with_capacity(oracle_ctx * HIDDEN);
    for position in 0..oracle_ctx {
        ctx.stream.memcpy_htod(
            &hidden_host[position * HIDDEN..(position + 1) * HIDDEN],
            scratch.hidden.data_mut(),
        )?;
        let (cos_host, sin_host) = rope_tables(position);
        ctx.stream.memcpy_htod(&cos_host, &mut cos)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin)?;
        ctx.stream
            .memcpy_htod(&[position as i64], &mut slot_mapping)?;
        ctx.stream
            .memcpy_htod(&[(position + 1) as i32], &mut seq_lens)?;

        let step = Glm52DecodeStep {
            mla_cos: &cos,
            mla_sin: &sin,
            idx_cos: &cos,
            idx_sin: &sin,
            mla_sched: &mla_sched,
            slot_mapping: &slot_mapping,
            block_table: &block_table,
            seq_lens: &seq_lens,
        };
        let mut carry_ready = false;
        // Gate walk: standalone input norm + fixed parity 0 (one layer per
        // call, stream in scratch.hidden — same shape as the EP1 gate).
        openinfer_kernels::ops::rms_norm_rows_into(
            ctx,
            scratch.hidden.data(),
            &w.input_ln,
            crate::config::GLM52_RMS_EPS,
            HIDDEN,
            1,
            scratch.layer.normed.data_mut(),
        )?;
        glm52_layer_attention_half(
            ctx,
            None,
            w,
            &mut caches,
            &step,
            &mut scratch,
            &mut carry_ready,
            0,
            true,
            None,
            crate::layer::Glm52LayerIndexMode::Normal,
        )?;
        run_moe_ep8_half(ctx, moe, ep8, &mut scratch, global_tokens)?;
        let post_attention_host = ctx.stream.clone_dtoh(scratch.layer.attn[0].data())?;
        post_attention_outputs.extend(post_attention_host.iter().map(|v| v.to_f32()));
        let mlp_host = ctx.stream.clone_dtoh(scratch.layer.mlp_out.data())?;
        mlp_outputs.extend(mlp_host.iter().map(|v| v.to_f32()));
        glm52_layer_finish(ctx, &mut scratch, 0, false)?;
        let out_host = ctx.stream.clone_dtoh(scratch.hidden.data())?;
        hidden_outputs.extend(out_host.iter().map(|v| v.to_f32()));
    }
    Ok(LayerEp8Outputs {
        hidden: hidden_outputs,
        post_attention: post_attention_outputs,
        mlp: mlp_outputs,
    })
}

pub(super) struct MoeEp8RowsOutputs {
    pub(super) mlp: Vec<f32>,
    pub(super) normed: Vec<f32>,
    pub(super) topk_ids: Vec<i32>,
    pub(super) topk_weights: Vec<f32>,
    pub(super) routed: Vec<f32>,
    pub(super) shared_gate_up: Vec<f32>,
    pub(super) shared_silu: Vec<f32>,
    pub(super) shared: Vec<f32>,
}

pub(super) fn run_moe_ep8_rows(
    ctx: &DeviceContext,
    w: &crate::layer::Glm52DecoderLayerWeights,
    ep8: &mut Glm52MoeEp8State,
    post_attention_host: &[bf16],
    reference_normed_host: &[bf16],
    rows: usize,
    global_tokens: usize,
) -> Result<MoeEp8RowsOutputs> {
    ensure!(
        post_attention_host.len() == rows * HIDDEN,
        "EP8 MoE oracle input has {} elements, expected {}",
        post_attention_host.len(),
        rows * HIDDEN
    );
    ensure!(
        reference_normed_host.len() == rows * HIDDEN,
        "EP8 MoE oracle reference norm has {} elements, expected {}",
        reference_normed_host.len(),
        rows * HIDDEN
    );
    let Glm52LayerMlp::MoeEp8(moe) = &w.mlp else {
        anyhow::bail!("EP8 MoE oracle requires MoeEp8 layer weights");
    };
    let contract = Glm52FlashMlaSparseDecode {
        batch_size: 1,
        num_blocks: 1,
        topk: GLM52_FLASHMLA_SPARSE_TOPK,
        num_sm_parts: glm52_flashmla_sparse_decode_num_sm_parts()?,
        sm_scale: GLM52_SM_SCALE,
    };
    let index_cache_layout = Glm52IndexerCacheLayout {
        cache_blocks: 1,
        cache_block_size: INDEX_CACHE_BLOCK,
        cache_block_stride_bytes: INDEX_CACHE_BLOCK * (GLM52_INDEX_HEAD_DIM + 4),
    };
    let mqa_shape =
        Glm52IndexerScratch::paged_mqa_shape(1, index_cache_layout, 1, NUM_SMS, rows.max(1));
    let mut scratch =
        Glm52DecodeScratch::new(ctx, &contract, mqa_shape, crate::config::GLM52_HEADS, false)?;
    let mut outputs = MoeEp8RowsOutputs {
        mlp: Vec::with_capacity(post_attention_host.len()),
        normed: Vec::with_capacity(post_attention_host.len()),
        topk_ids: Vec::with_capacity(rows * crate::config::GLM52_TOPK),
        topk_weights: Vec::with_capacity(rows * crate::config::GLM52_TOPK),
        routed: Vec::with_capacity(post_attention_host.len()),
        shared_gate_up: Vec::with_capacity(
            rows * 2 * crate::moe_decode::GLM52_SHARED_EXPERT_INTERMEDIATE,
        ),
        shared_silu: Vec::with_capacity(rows * crate::moe_decode::GLM52_SHARED_EXPERT_INTERMEDIATE),
        shared: Vec::with_capacity(post_attention_host.len()),
    };
    for (row, reference_normed) in post_attention_host
        .chunks_exact(HIDDEN)
        .zip(reference_normed_host.chunks_exact(HIDDEN))
    {
        ctx.stream.memcpy_htod(row, scratch.hidden.data_mut())?;
        openinfer_kernels::ops::rms_norm_rows_into(
            ctx,
            scratch.hidden.data(),
            &w.post_attn_ln,
            crate::config::GLM52_RMS_EPS,
            HIDDEN,
            1,
            scratch.layer.normed2.data_mut(),
        )?;
        let normed = ctx.stream.clone_dtoh(scratch.layer.normed2.data())?;
        outputs
            .normed
            .extend(normed.iter().map(|value| value.to_f32()));
        // Feed the exact vLLM norm output into all downstream stages. This
        // keeps the RMSNorm delta from contaminating router and expert-kernel
        // diagnostics.
        ctx.stream
            .memcpy_htod(reference_normed, scratch.layer.normed2.data_mut())?;
        let route = run_ep_router(ctx, &moe.router, scratch.layer.normed2.data())?;
        let topk_ids = ctx.stream.clone_dtoh(&route.topk_idx)?;
        outputs
            .topk_ids
            .extend_from_slice(&topk_ids[..crate::config::GLM52_TOPK]);
        let topk_weights = ctx.stream.clone_dtoh(&route.topk_weight)?;
        outputs
            .topk_weights
            .extend_from_slice(&topk_weights[..crate::config::GLM52_TOPK]);
        let dispatched = glm52_moe_ep8_routed_forward(
            ctx,
            ep8,
            &moe.bank,
            Some((scratch.layer.normed2.data(), &route, 1)),
            global_tokens,
        )?;
        ensure!(dispatched, "rank-0 EP8 MoE returned no combined output");
        let routed = ctx.stream.clone_dtoh(ep8.combined())?;
        outputs.routed.extend(routed[..HIDDEN].iter().map(|value| {
            bf16::from_f32(value.to_f32() * crate::config::GLM52_ROUTED_SCALING_FACTOR as f32)
                .to_f32()
        }));
        moe.shared.forward_into(
            ctx,
            scratch.layer.normed2.data(),
            &mut scratch.shared_mlp,
            scratch.layer.shared_out.data_mut(),
        )?;
        let shared_gate_up = ctx.stream.clone_dtoh(scratch.shared_mlp.gate_up())?;
        outputs
            .shared_gate_up
            .extend(shared_gate_up.iter().map(|value| value.to_f32()));
        let shared_silu = ctx.stream.clone_dtoh(scratch.shared_mlp.silu_out())?;
        outputs
            .shared_silu
            .extend(shared_silu.iter().map(|value| value.to_f32()));
        let shared = ctx.stream.clone_dtoh(scratch.layer.shared_out.data())?;
        outputs
            .shared
            .extend(shared.iter().map(|value| value.to_f32()));
        add_scaled_bf16_into(
            ctx,
            ep8.combined(),
            crate::config::GLM52_ROUTED_SCALING_FACTOR as f32,
            scratch.layer.shared_out.data(),
            HIDDEN,
            scratch.layer.mlp_out.data_mut(),
        )?;
        let output = ctx.stream.clone_dtoh(scratch.layer.mlp_out.data())?;
        outputs
            .mlp
            .extend(output.iter().map(|value| value.to_f32()));
    }
    Ok(outputs)
}
