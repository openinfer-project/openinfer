//! TP8 layer-6 MoE oracle gate: the EP8 gate's twin with the MoE half going
//! through the whole-layer TP8 kernel (LL allgather → union → sliced GEMMs →
//! prob-weighted reduce-scatter, shared expert folded at bank index 256).
//!
//! Rank 0 walks the full decoder layer (attention + indexer + TP8 MoE) over
//! the same seeded input as the EP1/EP8 gates; ranks 1..7 play the bucket-1
//! pad ranks — each holds its 1/8-I slice of all 257 experts and feeds a
//! fixed pad token through the collective kernel once per position (pad
//! outputs are dropped, exactly like production padding). The probe
//! constants, tolerance, and router tie-flip allowance are shared verbatim
//! with `layer_oracle_gate` — passing here proves the TP8 sharding (slice
//! loader → AG routing broadcast → union → mma partials → RS sum) computes
//! the same layer output as the local EP1 chain.

use std::sync::Arc;

use anyhow::{Context as _, Result};
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
    GateLayerMlp, LayerTensors, MOE_ORACLE_CTX, MOE_ORACLE_HIDDEN_DIGEST, MOE_ORACLE_INPUT_SCALE,
    MOE_ORACLE_LAYER, MOE_ORACLE_LAYER_PROBES, MOE_ORACLE_LAYER_TOL, MOE_ORACLE_SEED,
    assert_layer_probes, checked_hidden, load_decoder_layer, model_path, upload_u8,
};
use crate::config::{GLM52_INDEX_HEAD_DIM, GLM52_ROPE_HALF, GLM52_SM_SCALE};
use crate::model::{INDEX_CACHE_BLOCK, NUM_SMS, rope_tables};
use crate::moe_decode::{Glm52MoeRouterWeights, Glm52RouterScratch, HIDDEN, run_router_into};
use crate::moe_tp8::{Glm52MoeTp8State, Glm52Tp8Exchange, load_tp8_slice_layer};
use crate::weights::Glm52WeightManifest;

const TP_RANKS: usize = 8;

/// Span-8 gate: the SAME 8 rows through both row mappings must be
/// bit-identical. dp8 gathers row j from rank j; span gathers all 8 from the
/// owner — either way the kernel sees the same xg, the same expert union in
/// the same slot order, the same fixed-order partial sums, so any deviation
/// is an addressing bug in the AG/RS source-mode generalization (exactly
/// what this gate exists to catch). Also checks the span pad contract:
/// non-owner ranks' 8-row mlp_out is zero-filled.
#[test]
#[ignore = "requires 8×H200 + GLM-5.2-FP8 checkpoint"]
fn layer_moe_tp8_span_matches_dp8() -> Result<()> {
    let hidden_host = checked_hidden(
        MOE_ORACLE_SEED,
        MOE_ORACLE_CTX,
        MOE_ORACLE_INPUT_SCALE,
        MOE_ORACLE_HIDDEN_DIGEST,
    )?;
    let manifest = Arc::new(Glm52WeightManifest::from_model_dir(&model_path())?);
    let tensors = Arc::new(LayerTensors::load(&model_path(), MOE_ORACLE_LAYER)?);
    let exchange = Arc::new(Glm52Tp8Exchange::new());
    let rows: Arc<Vec<bf16>> = Arc::new(hidden_host[..TP_RANKS * HIDDEN].to_vec());

    // Ranks 1..7: dp8 pass with their own row, then span pass as pad ranks
    // (inputs ignored, mlp_out must come back all-zero). Returns the rank's
    // dp8 row output for the owner-side comparison.
    let handles: Vec<_> = (1..TP_RANKS)
        .map(|rank| {
            let manifest = Arc::clone(&manifest);
            let tensors = Arc::clone(&tensors);
            let exchange = Arc::clone(&exchange);
            let rows = Arc::clone(&rows);
            std::thread::Builder::new()
                .name(format!("tp8-span-gate-rank-{rank}"))
                .spawn(move || -> Result<Vec<bf16>> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let bank = load_tp8_slice_layer(
                        &ctx,
                        &model_path(),
                        &manifest,
                        rank,
                        MOE_ORACLE_LAYER,
                    )?;
                    let mut tp8 = Glm52MoeTp8State::new(&ctx, rank, rank, &exchange, 1)?;
                    let router = gate_router(&ctx, &tensors)?;
                    let mut normed2 = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
                    ctx.stream
                        .memcpy_htod(&rows[rank * HIDDEN..(rank + 1) * HIDDEN], &mut normed2)?;
                    let mut router_scratch = Glm52RouterScratch::new(&ctx, 1)?;
                    run_router_into(&ctx, &router, &normed2, &mut router_scratch)?;
                    let mut dp8_out = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
                    tp8.advance_epoch(&ctx)?;
                    tp8.forward(&ctx, 0, &bank, &normed2, &router_scratch, &mut dp8_out)?;
                    let dp8_host = ctx.stream.clone_dtoh(&dp8_out)?;

                    // Span pass: pad rank. Row inputs are ignored by the
                    // kernel (only the staged owner's are read); poison
                    // mlp_out to prove the zero-fill runs.
                    let dummy_rows = ctx.stream.alloc_zeros::<bf16>(TP_RANKS * HIDDEN)?;
                    let dummy_idx = ctx.stream.alloc_zeros::<i32>(TP_RANKS * 8)?;
                    let dummy_prob = ctx.stream.alloc_zeros::<f32>(TP_RANKS * 8)?;
                    let mut pad_out = ctx.stream.alloc_zeros::<bf16>(TP_RANKS * HIDDEN)?;
                    ctx.stream.memcpy_htod(
                        &vec![bf16::from_f32(f32::NAN); TP_RANKS * HIDDEN],
                        &mut pad_out,
                    )?;
                    tp8.stage_span_owner(&ctx, 0)?;
                    tp8.advance_epoch(&ctx)?;
                    tp8.forward_span(
                        &ctx,
                        0,
                        &bank,
                        &dummy_rows,
                        &dummy_idx,
                        &dummy_prob,
                        &mut pad_out,
                    )?;
                    let pad_host = ctx.stream.clone_dtoh(&pad_out)?;
                    ctx.stream.synchronize()?;
                    anyhow::ensure!(
                        pad_host.iter().all(|v| v.to_f32() == 0.0),
                        "rank {rank} span pad mlp_out not zero-filled"
                    );
                    Ok(dp8_host)
                })
                .expect("spawn tp8 span gate rank thread")
        })
        .collect();

    // Rank 0 (owner): dp8 with row 0, then span with all 8 rows. Per-row
    // routing (batch-1, dtod-copied into the concat topk buffers) keeps the
    // routing bytes identical to what each rank computed in the dp8 pass.
    let ctx = DeviceContext::new_with_device(0)?;
    let bank = load_tp8_slice_layer(&ctx, &model_path(), &manifest, 0, MOE_ORACLE_LAYER)?;
    let mut tp8 = Glm52MoeTp8State::new(&ctx, 0, 0, &exchange, 1)?;
    let router = gate_router(&ctx, &tensors)?;

    let mut normed2_all = ctx.stream.alloc_zeros::<bf16>(TP_RANKS * HIDDEN)?;
    ctx.stream.memcpy_htod(rows.as_slice(), &mut normed2_all)?;
    let mut topk_idx_all = ctx.stream.alloc_zeros::<i32>(TP_RANKS * 8)?;
    let mut topk_prob_all = ctx.stream.alloc_zeros::<f32>(TP_RANKS * 8)?;
    let mut router_scratch = Glm52RouterScratch::new(&ctx, 1)?;
    let mut row_buf = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
    for row in 0..TP_RANKS {
        ctx.stream.memcpy_dtod(
            &normed2_all.slice(row * HIDDEN..(row + 1) * HIDDEN),
            &mut row_buf,
        )?;
        run_router_into(&ctx, &router, &row_buf, &mut router_scratch)?;
        ctx.stream.memcpy_dtod(
            &router_scratch.route.topk_idx.slice(0..8),
            &mut topk_idx_all.slice_mut(row * 8..(row + 1) * 8),
        )?;
        ctx.stream.memcpy_dtod(
            &router_scratch.route.topk_weight.slice(0..8),
            &mut topk_prob_all.slice_mut(row * 8..(row + 1) * 8),
        )?;
    }

    // dp8 pass (row 0 is rank 0's token): re-route row 0 so the batch-1
    // scratch holds row 0's topk, not row 7's from the loop above.
    ctx.stream
        .memcpy_dtod(&normed2_all.slice(0..HIDDEN), &mut row_buf)?;
    run_router_into(&ctx, &router, &row_buf, &mut router_scratch)?;
    let mut dp8_out = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
    tp8.advance_epoch(&ctx)?;
    tp8.forward(&ctx, 0, &bank, &row_buf, &router_scratch, &mut dp8_out)?;
    let dp8_row0 = ctx.stream.clone_dtoh(&dp8_out)?;

    // Span pass: all 8 rows from rank 0 (the staged owner).
    let mut span_out = ctx.stream.alloc_zeros::<bf16>(TP_RANKS * HIDDEN)?;
    tp8.stage_span_owner(&ctx, 0)?;
    tp8.advance_epoch(&ctx)?;
    tp8.forward_span(
        &ctx,
        0,
        &bank,
        &normed2_all,
        &topk_idx_all,
        &topk_prob_all,
        &mut span_out,
    )?;
    let span_host = ctx.stream.clone_dtoh(&span_out)?;
    ctx.stream.synchronize()?;

    let mut dp8_rows: Vec<Vec<bf16>> = vec![dp8_row0];
    for (rank, handle) in handles.into_iter().enumerate() {
        dp8_rows.push(
            handle
                .join()
                .expect("tp8 span gate rank thread panicked")
                .with_context(|| format!("tp8 span gate rank {}", rank + 1))?,
        );
    }
    for (row, dp8_row) in dp8_rows.iter().enumerate() {
        let span_row = &span_host[row * HIDDEN..(row + 1) * HIDDEN];
        let diff = span_row
            .iter()
            .zip(dp8_row.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff, 0,
            "span row {row} deviates from its dp8 output in {diff} of {HIDDEN} elements"
        );
    }
    Ok(())
}

fn gate_router(ctx: &DeviceContext, tensors: &LayerTensors) -> Result<Glm52MoeRouterWeights> {
    let mp = format!("model.layers.{MOE_ORACLE_LAYER}.mlp");
    Glm52MoeRouterWeights::new(
        upload_u8(ctx, tensors.bytes(&format!("{mp}.gate.weight"))?)?,
        upload_u8(
            ctx,
            tensors.bytes(&format!("{mp}.gate.e_score_correction_bias"))?,
        )?,
    )
}

#[test]
#[ignore = "requires 8×H200 + GLM-5.2-FP8 checkpoint"]
fn layer_moe_tp8_oracle_gate() -> Result<()> {
    let hidden_host = checked_hidden(
        MOE_ORACLE_SEED,
        MOE_ORACLE_CTX,
        MOE_ORACLE_INPUT_SCALE,
        MOE_ORACLE_HIDDEN_DIGEST,
    )?;
    let manifest = Arc::new(Glm52WeightManifest::from_model_dir(&model_path())?);
    let tensors = Arc::new(LayerTensors::load(&model_path(), MOE_ORACLE_LAYER)?);
    let exchange = Arc::new(Glm52Tp8Exchange::new());

    // Pad ranks: slice bank + router over a fixed pad token, one collective
    // kernel per position. The pad token's routing is real (its experts join
    // the union and its partials cost compute) but its output is dropped —
    // the production bucket-1 pad semantics.
    let handles: Vec<_> = (1..TP_RANKS)
        .map(|rank| {
            let manifest = Arc::clone(&manifest);
            let tensors = Arc::clone(&tensors);
            let exchange = Arc::clone(&exchange);
            let pad_hidden = hidden_host[rank * HIDDEN..(rank + 1) * HIDDEN].to_vec();
            std::thread::Builder::new()
                .name(format!("tp8-gate-rank-{rank}"))
                .spawn(move || -> Result<()> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let bank = load_tp8_slice_layer(
                        &ctx,
                        &model_path(),
                        &manifest,
                        rank,
                        MOE_ORACLE_LAYER,
                    )?;
                    let mut tp8 = Glm52MoeTp8State::new(&ctx, rank, rank, &exchange, 1)?;
                    let mp = format!("model.layers.{MOE_ORACLE_LAYER}.mlp");
                    let router = Glm52MoeRouterWeights::new(
                        upload_u8(&ctx, tensors.bytes(&format!("{mp}.gate.weight"))?)?,
                        upload_u8(
                            &ctx,
                            tensors.bytes(&format!("{mp}.gate.e_score_correction_bias"))?,
                        )?,
                    )?;
                    let mut normed2 = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
                    ctx.stream.memcpy_htod(&pad_hidden, &mut normed2)?;
                    let mut router_scratch = Glm52RouterScratch::new(&ctx, 1)?;
                    run_router_into(&ctx, &router, &normed2, &mut router_scratch)?;
                    let mut pad_out = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
                    for _position in 0..MOE_ORACLE_CTX {
                        tp8.advance_epoch(&ctx)?;
                        tp8.forward(&ctx, 0, &bank, &normed2, &router_scratch, &mut pad_out)?;
                    }
                    // The LL buffers may be written by peers until every
                    // rank's last kernel retires — sync before dropping.
                    ctx.stream.synchronize()?;
                    Ok(())
                })
                .expect("spawn tp8 gate rank thread")
        })
        .collect();

    // Rank 0: full decoder layer with the TP8 MoE half, prefill-via-decode.
    // MoeEp8Rank0 weights supply the router; the routed+shared math runs
    // entirely in the TP8 kernel from rank 0's own slice bank.
    let ctx = DeviceContext::new_with_device(0)?;
    let w = load_decoder_layer(
        &ctx,
        &model_path(),
        MOE_ORACLE_LAYER,
        GateLayerMlp::MoeEp8Rank0,
    )?;
    let bank = load_tp8_slice_layer(&ctx, &model_path(), &manifest, 0, MOE_ORACLE_LAYER)?;
    let mut tp8 = Glm52MoeTp8State::new(&ctx, 0, 0, &exchange, 1)?;
    let outputs = run_layer_prefill_tp8(&ctx, &w, &mut tp8, &bank, &hidden_host, MOE_ORACLE_CTX);
    ctx.stream.synchronize()?;

    for (rank, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .expect("tp8 gate rank thread panicked")
            .with_context(|| format!("tp8 gate rank {}", rank + 1))?;
    }
    assert_layer_probes(
        "layer6/moe/tp8/g8",
        &outputs?,
        MOE_ORACLE_LAYER_PROBES,
        MOE_ORACLE_LAYER_TOL,
        4,
    );
    Ok(())
}

/// The TP8 variant of the gate's prefill-via-decode walk: same decode
/// environment as `oracle::layer::run_layer_prefill`, with the MLP half
/// driven through the whole-layer TP8 kernel (which folds the shared expert,
/// so there is no separate `add_into`).
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
    let mla_sched = Glm52MlaSchedMetadata::new(ctx, contract)?;

    let mqa_shape =
        Glm52IndexerScratch::decode_shape(1, index_cache_layout, index_blocks, NUM_SMS, oracle_ctx);
    let mut scratch = Glm52DecodeScratch::new(ctx, &contract, mqa_shape)?;
    let mut router_scratch = Glm52RouterScratch::new(ctx, 1)?;

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
        )?;
        // The production TP8 arm verbatim: router on the real path, then the
        // whole-layer kernel writes routed + shared into mlp_out.
        run_router_into(
            ctx,
            &moe.router,
            scratch.layer.normed2.data(),
            &mut router_scratch,
        )?;
        tp8.advance_epoch(ctx)?;
        tp8.forward(
            ctx,
            0,
            bank,
            scratch.layer.normed2.data(),
            &router_scratch,
            scratch.layer.mlp_out.data_mut(),
        )?;
        glm52_layer_finish(ctx, &mut scratch, 0)?;
        let out_host = ctx.stream.clone_dtoh(scratch.hidden.data())?;
        outputs.extend(out_host.iter().map(|v| v.to_f32()));
    }
    Ok(outputs)
}
