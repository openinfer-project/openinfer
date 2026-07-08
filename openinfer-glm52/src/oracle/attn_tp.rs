//! Attention-TP twin gate (M4 K3b): 8 ranks each carry an 8-head shard of
//! layer 0's MLA weights (q_b rows / kv_b rows / o_proj columns, the
//! production `ProjWeight` slice helpers), walk the same decode-as-prefill
//! loop as the MLA oracle over identical replicated inputs, and reduce their
//! o_proj partials through the TP8 AR brick. Every rank must end with the
//! bit-identical reduced hidden, and that hidden must match the full
//! 64-head single-device reference within the MLA oracle's RMS-scaled
//! tolerance class (exact equality is off the table by construction: the
//! sharded o_proj splits the k=16384 sum into 8 bf16-rounded partials).
//!
//! Per-head attention math is untouched by the shard — the FlashMLA kernel
//! computes heads independently (the shard's 8 heads land in query slots
//! 0..8 of the fixed 64-wide tile), the absorb GEMMs batch per head — so a
//! deviation beyond o_proj rounding is an addressing bug in the slice /
//! compact-layout plumbing, exactly what this gate exists to catch.

use std::path::PathBuf;
use std::sync::{Arc, Barrier, Mutex};

use anyhow::{Result, ensure};
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE, GLM52_FLASHMLA_SPARSE_TOPK, GLM52_TP8_AR_CHUNK_PACKETS,
    Glm52FlashMlaSparseDecode, Glm52Tp8LlBuffer, glm52_flashmla_sparse_decode_num_sm_parts,
    glm52_moe_tp8_epoch_advance, glm52_tp8_ar_buffer_bytes, glm52_tp8_ar_launch,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceVec};

use super::mla::{Layer0Tensors, load_layer0, seeded_hidden};
use crate::config::{GLM52_ROPE_HALF, GLM52_SM_SCALE};
use crate::fp8::ProjWeight;
use crate::mla_decode::{Glm52MlaLayerWeights, Glm52MlaSchedMetadata, glm52_mla_decode_forward};
use crate::model::rope_tables;
use crate::rows::Rows;

const RANKS: usize = 8;
const HIDDEN: usize = 6144;
const CTX: usize = 48;
const SEED: u64 = 0x5eed604d;
/// Mean-norm gate. The layer output has near-zero RMS (the 8 rank partials
/// cancel), so per-element deltas from the sanctioned rounding — each rank's
/// o_proj partial is bf16-rounded before the AR's fixed-order f32 sum, half
/// an ulp of the PARTIAL magnitude per rank — reach ~0.14×RMS at the max
/// (measured 2026-07-08) while the mean sits at 0.0011. A head/offset
/// addressing bug replaces at least one rank's whole partial (magnitude ≥
/// the output RMS itself), pushing the mean to O(0.1)+ — so the mean
/// separates bug from rounding by ≥10× in both directions where the max
/// cannot. `MAX_TOL` stays as a loose catastrophe bound.
const MEAN_TOL: f64 = 0.01;
const MAX_TOL: f32 = 1.0;

fn shard_layer0(
    ctx: &DeviceContext,
    t: &Layer0Tensors,
    rank: usize,
) -> Result<Glm52MlaLayerWeights> {
    let p = "model.layers.0.self_attn";
    let q_b_full = ProjWeight::upload(ctx, &t.proj(&format!("{p}.q_b_proj"), 16384, 2048)?)?;
    let kv_b_full = ProjWeight::upload(ctx, &t.proj(&format!("{p}.kv_b_proj"), 28672, 512)?)?;
    let o_full = ProjWeight::upload(ctx, &t.proj(&format!("{p}.o_proj"), HIDDEN, 16384)?)?;
    let q_b = q_b_full.slice_rows(ctx, rank * 2048, 2048)?;
    let kv_b = kv_b_full.slice_rows(ctx, rank * 3584, 3584)?;
    let o_proj = o_full.slice_cols(ctx, rank * 2048, 2048)?;
    Glm52MlaLayerWeights::from_device(
        ctx,
        ProjWeight::upload(ctx, &t.proj(&format!("{p}.q_a_proj"), 2048, HIDDEN)?)?,
        DeviceVec::from_safetensors(ctx, t.bytes(&format!("{p}.q_a_layernorm.weight"))?)?,
        q_b,
        ProjWeight::upload(
            ctx,
            &t.proj(&format!("{p}.kv_a_proj_with_mqa"), 576, HIDDEN)?,
        )?,
        DeviceVec::from_safetensors(ctx, t.bytes(&format!("{p}.kv_a_layernorm.weight"))?)?,
        &kv_b,
        o_proj,
    )
}

/// Decode-as-prefill over `CTX` positions with weight set `w`; returns the
/// per-position output rows. `ar` carries the shard side's allreduce wiring
/// (None = full-weights reference, the o_proj output IS the layer output).
#[allow(clippy::too_many_arguments)]
fn walk_positions(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden_host: &[bf16],
    ar: Option<(
        &mut cudarc::driver::CudaSlice<u64>,
        u64,
        [u64; RANKS],
        usize,
    )>,
) -> Result<Vec<bf16>> {
    let num_sm_parts = glm52_flashmla_sparse_decode_num_sm_parts()?;
    let contract = Glm52FlashMlaSparseDecode {
        batch_size: 1,
        num_blocks: CTX.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE),
        topk: GLM52_FLASHMLA_SPARSE_TOPK,
        num_sm_parts,
        sm_scale: GLM52_SM_SCALE,
    };
    let mut cache = ctx
        .stream
        .alloc_zeros::<u8>(contract.packed_kv_cache_len())?;
    let sched = Glm52MlaSchedMetadata::new(ctx, contract)?;
    let mut reduced = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
    let mut ar = ar;
    let mut outputs = Vec::with_capacity(CTX * HIDDEN);
    for position in 0..CTX {
        let mut hidden = Rows::<HIDDEN>::zeros(ctx, 1)?;
        ctx.stream.memcpy_htod(
            &hidden_host[position * HIDDEN..(position + 1) * HIDDEN],
            hidden.data_mut(),
        )?;
        let (cos_host, sin_host) = rope_tables(position);
        let mut cos = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
        let mut sin = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
        ctx.stream.memcpy_htod(&cos_host, &mut cos)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin)?;
        let mut topk_host = vec![-1i32; contract.topk];
        for (slot, v) in topk_host.iter_mut().enumerate().take(position + 1) {
            *v = slot as i32;
        }
        let mut topk_dev = ctx.stream.alloc_zeros::<i32>(contract.topk)?;
        ctx.stream.memcpy_htod(&topk_host, &mut topk_dev)?;

        let o = glm52_mla_decode_forward(
            ctx, w, &hidden, &cos, &sin, &mut cache, position, &topk_dev, &sched,
        )?;
        let row = match ar.as_mut() {
            Some((epoch_dev, ar_local, peer_ar, rank)) => {
                glm52_moe_tp8_epoch_advance(ctx, epoch_dev)?;
                glm52_tp8_ar_launch(
                    ctx,
                    0,
                    1,
                    o.data(),
                    &mut reduced,
                    *ar_local,
                    *peer_ar,
                    epoch_dev,
                    None,
                    *rank,
                )?;
                ctx.stream.clone_dtoh(&reduced)?
            }
            None => ctx.stream.clone_dtoh(o.data())?,
        };
        outputs.extend_from_slice(&row);
    }
    ctx.stream.synchronize()?;
    Ok(outputs)
}

#[test]
#[ignore = "requires 8xH200 + GLM-5.2-FP8 checkpoint"]
fn attn_tp8_shard_matches_full() -> Result<()> {
    let model_path = std::env::var_os("OPENINFER_TEST_MODEL_PATH")
        .map_or_else(|| PathBuf::from("models/GLM-5.2-FP8"), PathBuf::from);
    let tensors = Arc::new(Layer0Tensors::load(&model_path)?);
    let hidden = Arc::new(seeded_hidden(SEED, CTX * HIDDEN));
    let vas: Arc<Mutex<Vec<Vec<u64>>>> = Arc::new(Mutex::new(vec![Vec::new(); RANKS]));
    let barrier = Arc::new(Barrier::new(RANKS));

    let handles: Vec<_> = (0..RANKS)
        .map(|rank| {
            let tensors = Arc::clone(&tensors);
            let hidden = Arc::clone(&hidden);
            let vas = Arc::clone(&vas);
            let barrier = Arc::clone(&barrier);
            std::thread::Builder::new()
                .name(format!("attn-tp-gate-rank-{rank}"))
                .spawn(move || -> Result<(Vec<bf16>, Option<Vec<bf16>>)> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let w = shard_layer0(&ctx, &tensors, rank)?;
                    ensure!(w.heads == 8, "shard derived {} heads, want 8", w.heads);
                    let ordinals: Vec<usize> = (0..RANKS).collect();
                    let buf = Glm52Tp8LlBuffer::alloc(glm52_tp8_ar_buffer_bytes(1), &ordinals)?;
                    vas.lock().unwrap()[rank] = (0..RANKS).map(|i| buf.addr_for(i)).collect();
                    barrier.wait();
                    let peer_ar: [u64; RANKS] = {
                        let published = vas.lock().unwrap();
                        std::array::from_fn(|dst| {
                            published[dst][rank] + (rank * GLM52_TP8_AR_CHUNK_PACKETS * 16) as u64
                        })
                    };
                    let mut epoch_dev = ctx.stream.alloc_zeros::<u64>(1)?;
                    let reduced = walk_positions(
                        &ctx,
                        &w,
                        &hidden,
                        Some((&mut epoch_dev, buf.addr_for(rank), peer_ar, rank)),
                    )?;
                    // Reference on rank 0's device, after the collective walk
                    // (its own cache, full 64-head weights).
                    let reference = if rank == 0 {
                        let w_full = load_layer0(&ctx, &model_path_from_env())?;
                        Some(walk_positions(&ctx, &w_full, &hidden, None)?)
                    } else {
                        None
                    };
                    ctx.stream.synchronize()?;
                    barrier.wait(); // streams idle before any LL buffer drops
                    Ok((reduced, reference))
                })
                .expect("spawn attn tp gate rank thread")
        })
        .collect();

    let mut all: Vec<(Vec<bf16>, Option<Vec<bf16>>)> = Vec::new();
    for h in handles {
        all.push(h.join().expect("attn tp gate rank thread panicked")?);
    }

    // Cross-rank bit-identity: the replicated-activation topology depends on
    // every rank holding the same post-AR hidden.
    for rank in 1..RANKS {
        ensure!(
            all[rank].0 == all[0].0,
            "rank {rank} reduced output differs bitwise from rank 0"
        );
    }

    // Sharded vs full-width reference, RMS-scaled.
    let reference = all[0].1.as_ref().expect("rank 0 carries the reference");
    let rms =
        (reference.iter().map(|v| v.to_f32().powi(2)).sum::<f32>() / reference.len() as f32).sqrt();
    let mut max_scaled = 0f32;
    let mut sum_scaled = 0f64;
    for (a, b) in all[0].0.iter().zip(reference.iter()) {
        let scaled = (a.to_f32() - b.to_f32()).abs() / rms;
        max_scaled = max_scaled.max(scaled);
        sum_scaled += scaled as f64;
    }
    let mean_scaled = sum_scaled / reference.len() as f64;
    println!(
        "attn_tp8 twin: rms {rms:.6e}, max scaled delta {max_scaled:.4}, mean {mean_scaled:.6}"
    );
    ensure!(
        mean_scaled < MEAN_TOL,
        "sharded attention deviates from the full reference: mean scaled {mean_scaled:.6} >= {MEAN_TOL}"
    );
    ensure!(
        max_scaled < MAX_TOL,
        "sharded attention deviates catastrophically: max scaled {max_scaled:.4} >= {MAX_TOL}"
    );
    Ok(())
}

fn model_path_from_env() -> PathBuf {
    std::env::var_os("OPENINFER_TEST_MODEL_PATH")
        .map_or_else(|| PathBuf::from("models/GLM-5.2-FP8"), PathBuf::from)
}
