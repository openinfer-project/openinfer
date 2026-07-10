//! TP8 layer-6 MoE oracle gate: the EP8 gate's twin with the MoE half going
//! through the replicated TP8 phase-kernel chain (local union → sliced GEMMs
//! → prob-weighted all-reduce, shared expert folded at bank index 256).
//!
//! Replicated activations, mirrored exactly: ALL 8 ranks run the identical
//! full decoder-layer walk (attention + indexer + router replicated — the
//! per-rank weights are bit-identical, so every rank derives the same
//! normed2/topk locally with zero exchange), and only the MoE partials cross
//! the wire. Passing proves two things at once: the layer output matches the
//! EP1 golden probes (slice loader → union → mma partials → AR sum compute
//! the same math), and every rank's output is BIT-IDENTICAL — the
//! replicated-activations contract the next layer's redundant compute
//! depends on.

use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE, GLM52_FLASHMLA_SPARSE_TOPK, Glm52FlashMlaSparseDecode,
    Glm52IndexerCacheLayout, glm52_flashmla_sparse_decode_num_sm_parts,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::indexer::Glm52IndexerScratch;
use crate::layer::{
    Glm52DecodeStep, Glm52LayerCaches, Glm52LayerMlp, glm52_layer_attention_half,
    glm52_layer_finish,
};
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::scratch::Glm52DecodeScratch;

use super::layer::{
    GateLayerMlp, MOE_ORACLE_CTX, MOE_ORACLE_HIDDEN_DIGEST, MOE_ORACLE_INPUT_SCALE,
    MOE_ORACLE_LAYER, MOE_ORACLE_LAYER_PROBES, MOE_ORACLE_LAYER_TOL, MOE_ORACLE_SEED,
    assert_layer_probes, checked_hidden, load_decoder_layer, model_path,
};
use crate::config::{GLM52_INDEX_HEAD_DIM, GLM52_ROPE_HALF, GLM52_SM_SCALE};
use crate::model::{INDEX_CACHE_BLOCK, NUM_SMS, rope_tables};
use crate::moe_decode::{Glm52RouterScratch, HIDDEN, run_router_into};
use crate::moe_tp8::{Glm52MoeTp8State, Glm52Tp8Exchange, load_tp8_slice_layer};
use crate::weights::Glm52WeightManifest;

const TP_RANKS: usize = 8;

#[test]
#[ignore = "requires 8×H200 + GLM-5.2-FP8 checkpoint"]
fn layer_moe_tp8_oracle_gate() -> Result<()> {
    let hidden_host = Arc::new(checked_hidden(
        MOE_ORACLE_SEED,
        MOE_ORACLE_CTX,
        MOE_ORACLE_INPUT_SCALE,
        MOE_ORACLE_HIDDEN_DIGEST,
    )?);
    let manifest = Arc::new(Glm52WeightManifest::from_model_dir(&model_path())?);
    let exchange = Arc::new(Glm52Tp8Exchange::new());

    // Every rank: the identical full decoder-layer walk (replicated
    // attention/indexer/router weights + this rank's expert slice bank).
    let handles: Vec<_> = (1..TP_RANKS)
        .map(|rank| {
            let manifest = Arc::clone(&manifest);
            let exchange = Arc::clone(&exchange);
            let hidden_host = Arc::clone(&hidden_host);
            std::thread::Builder::new()
                .name(format!("tp8-gate-rank-{rank}"))
                .spawn(move || -> Result<Vec<f32>> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let w = load_decoder_layer(
                        &ctx,
                        &model_path(),
                        MOE_ORACLE_LAYER,
                        GateLayerMlp::MoeEp8Rank0,
                    )?;
                    let bank = load_tp8_slice_layer(
                        &ctx,
                        &model_path(),
                        &manifest,
                        rank,
                        MOE_ORACLE_LAYER,
                    )?;
                    let mut tp8 = Glm52MoeTp8State::new(&ctx, rank, rank, &exchange, 1, 1)?;
                    let outputs = run_layer_prefill_tp8(
                        &ctx,
                        &w,
                        &mut tp8,
                        &bank,
                        &hidden_host,
                        MOE_ORACLE_CTX,
                    );
                    ctx.stream.synchronize()?;
                    outputs
                })
                .expect("spawn tp8 gate rank thread")
        })
        .collect();

    let ctx = DeviceContext::new_with_device(0)?;
    let w = load_decoder_layer(
        &ctx,
        &model_path(),
        MOE_ORACLE_LAYER,
        GateLayerMlp::MoeEp8Rank0,
    )?;
    let bank = load_tp8_slice_layer(&ctx, &model_path(), &manifest, 0, MOE_ORACLE_LAYER)?;
    let mut tp8 = Glm52MoeTp8State::new(&ctx, 0, 0, &exchange, 1, 1)?;
    let outputs = run_layer_prefill_tp8(&ctx, &w, &mut tp8, &bank, &hidden_host, MOE_ORACLE_CTX);
    ctx.stream.synchronize()?;

    let mut peer_outputs = Vec::with_capacity(TP_RANKS - 1);
    for (rank, handle) in handles.into_iter().enumerate() {
        peer_outputs.push(
            handle
                .join()
                .expect("tp8 gate rank thread panicked")
                .with_context(|| format!("tp8 gate rank {}", rank + 1))?,
        );
    }
    let outputs = outputs?;
    // Cross-rank bitwise identity: the replicated-activations contract.
    for (rank, peer) in peer_outputs.iter().enumerate() {
        let diff = peer
            .iter()
            .zip(outputs.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(
            diff == 0,
            "rank {} layer output deviates bitwise from rank 0 in {diff} of {} elements",
            rank + 1,
            outputs.len()
        );
    }
    assert_layer_probes(
        "layer6/moe/tp8/g8",
        &outputs,
        MOE_ORACLE_LAYER_PROBES,
        MOE_ORACLE_LAYER_TOL,
        4,
    );
    Ok(())
}

/// The TP8 variant of the gate's prefill-via-decode walk: same decode
/// environment as `oracle::layer::run_layer_prefill`, with the MLP half
/// driven through the replicated TP8 kernel (which folds the shared expert,
/// so there is no separate `add_into`). The kernel's fixed 8-row shape is
/// staged with row 0 = the walk's token and rows 1..8 zero: a zero row's
/// topk (expert 0, prob 0) joins the union but its zero-prob terms are
/// skipped in the fixed-order sum, so row 0's arithmetic is untouched.
fn run_layer_prefill_tp8(
    ctx: &DeviceContext,
    w: &crate::layer::Glm52DecoderLayerWeights,
    tp8: &mut Glm52MoeTp8State,
    bank: &crate::moe_tp8::Glm52MoeTp8SliceBank,
    hidden_host: &[bf16],
    oracle_ctx: usize,
) -> Result<Vec<f32>> {
    let Glm52LayerMlp::MoeEp8(moe) = &w.mlp else {
        anyhow::bail!("tp8 gate requires the MoeEp8 layer weights (router source)");
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

    let mqa_shape =
        Glm52IndexerScratch::decode_shape(1, index_cache_layout, index_blocks, NUM_SMS, oracle_ctx);
    let mut scratch =
        Glm52DecodeScratch::new(ctx, &contract, mqa_shape, crate::config::GLM52_HEADS)?;
    let mut router_scratch = Glm52RouterScratch::new(ctx, 1)?;

    // 8-row staging for the replicated kernel (rows 1..8 stay zero).
    let mut normed2_all = ctx.stream.alloc_zeros::<bf16>(TP_RANKS * HIDDEN)?;
    let mut topk_idx_all = ctx.stream.alloc_zeros::<i32>(TP_RANKS * 8)?;
    let mut topk_prob_all = ctx.stream.alloc_zeros::<f32>(TP_RANKS * 8)?;
    let mut mlp_out_all = ctx.stream.alloc_zeros::<bf16>(TP_RANKS * HIDDEN)?;

    let mut outputs = Vec::with_capacity(oracle_ctx * HIDDEN);
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
        )?;
        // The production TP8 arm verbatim: router on the real path, then the
        // replicated kernel writes routed + shared into all 8 rows.
        run_router_into(
            ctx,
            &moe.router,
            scratch.layer.normed2.data(),
            &mut router_scratch,
        )?;
        ctx.stream.memcpy_dtod(
            scratch.layer.normed2.data(),
            &mut normed2_all.slice_mut(0..HIDDEN),
        )?;
        ctx.stream.memcpy_dtod(
            &router_scratch.route.topk_idx.slice(0..8),
            &mut topk_idx_all.slice_mut(0..8),
        )?;
        ctx.stream.memcpy_dtod(
            &router_scratch.route.topk_weight.slice(0..8),
            &mut topk_prob_all.slice_mut(0..8),
        )?;
        tp8.advance_epoch(ctx)?;
        tp8.forward(
            ctx,
            0,
            bank,
            &normed2_all,
            &topk_idx_all,
            &topk_prob_all,
            &mut mlp_out_all,
        )?;
        ctx.stream.memcpy_dtod(
            &mlp_out_all.slice(0..HIDDEN),
            scratch.layer.mlp_out.data_mut(),
        )?;
        glm52_layer_finish(ctx, &mut scratch, 0)?;
        let out_host = ctx.stream.clone_dtoh(scratch.hidden.data())?;
        outputs.extend(out_host.iter().map(|v| v.to_f32()));
    }
    Ok(outputs)
}
