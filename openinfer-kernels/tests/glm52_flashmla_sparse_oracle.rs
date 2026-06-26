//! GPU integration gate for the GLM5.2 FlashMLA sparse decode wrapper, validated
//! against an HF MLA oracle. SM90-only (FlashMLA sparse fp8 decode) — runs on the
//! H200 build node, no-ops elsewhere.
//!
//! It packs nothing itself: a numpy prep (tools side) turns the layer-0 oracle npz
//! into raw bins under `$GLM52_FLASHMLA_PROBE_DIR` using the *verified*
//! `fp8_ds_mla` 656-byte convention (512 e4m3 ckv + 4 f32 group scales + 64 bf16
//! rope-key). This test uploads them, runs the real wrapper (num_sm_parts →
//! metadata → decode), and compares the latent `[64,512]` against:
//!   - `latent_fp8ref` — numpy attention over the SAME dequantized fp8 cache, the
//!     tight gate: only kernel arithmetic (bf16 q·k, f32 softmax) differs.
//!   - `latent_expected` — the full-precision oracle, reported to show the fp8
//!     quantization noise floor.
//!
//!   cargo test --release -p openinfer-kernels --features glm52 \
//!       --test glm52_flashmla_sparse_oracle -- --nocapture
#![cfg(feature = "glm52")]

use half::bf16;
use openinfer_kernels::ops::{
    Glm52FlashMlaSparseDecode, glm52_flashmla_sparse_decode_launch,
    glm52_flashmla_sparse_decode_metadata_launch, glm52_flashmla_sparse_decode_num_sm_parts,
};
use openinfer_kernels::tensor::DeviceContext;
use std::path::PathBuf;

fn probe_dir() -> PathBuf {
    std::env::var("GLM52_FLASHMLA_PROBE_DIR")
        .unwrap_or_else(|_| "/data/models/glm52_mla_ref/flashmla_probe".into())
        .into()
}

fn read_bytes(dir: &PathBuf, name: &str) -> Option<Vec<u8>> {
    std::fs::read(dir.join(name)).ok()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn as_i32(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn as_bf16(b: &[u8]) -> Vec<bf16> {
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn cmp(label: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    let (mut maxd, mut sumd) = (0.0f32, 0.0f32);
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (g - w).abs();
        maxd = maxd.max(d);
        sumd += d;
    }
    let mean = sumd / got.len() as f32;
    println!("  vs {label:14}: max|Δ|={maxd:.6} mean|Δ|={mean:.6}");
    (maxd, mean)
}

#[test]
fn flashmla_sparse_decode_matches_mla_oracle() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let dir = probe_dir();
    let Some(cache_b) = read_bytes(&dir, "cache.bin") else {
        eprintln!("no probe fixtures at {dir:?}; skipping (run glm_flashmla_prep.py)");
        return;
    };
    let query = as_bf16(&read_bytes(&dir, "query.bin").unwrap());
    // V3.2 sparse decode has no dynamic per-row length (kernel asserts
    // topk_length == nullptr): topk is the fixed 2048, and short context is
    // expressed purely by -1 padding in topk_indices (the kernel skips -1).
    let topk = as_i32(&read_bytes(&dir, "topk.bin").unwrap());
    let want_fp8 = as_f32(&read_bytes(&dir, "latent_fp8ref.bin").unwrap());
    let want_oracle = as_f32(&read_bytes(&dir, "latent_expected.bin").unwrap());

    let num_sm_parts = glm52_flashmla_sparse_decode_num_sm_parts()
        .expect("num_sm_parts query failed (SM90 required)");
    let contract = Glm52FlashMlaSparseDecode {
        batch_size: 1,
        num_blocks: 1,
        topk: 2048,
        num_sm_parts,
        sm_scale: 0.0625, // 256**-0.5; FlashMLA applies it internally
    };
    assert_eq!(query.len(), contract.q_len());
    assert_eq!(cache_b.len(), contract.packed_kv_cache_len());
    assert_eq!(topk.len(), contract.topk_indices_len());

    // upload
    let mut q_d = ctx.stream.alloc_zeros::<bf16>(query.len()).unwrap();
    let mut cache_d = ctx.stream.alloc_zeros::<u8>(cache_b.len()).unwrap();
    let mut topk_d = ctx.stream.alloc_zeros::<i32>(topk.len()).unwrap();
    ctx.stream.memcpy_htod(&query, &mut q_d).unwrap();
    ctx.stream.memcpy_htod(&cache_b, &mut cache_d).unwrap();
    ctx.stream.memcpy_htod(&topk, &mut topk_d).unwrap();

    // metadata
    let mut sched = ctx
        .stream
        .alloc_zeros::<i32>(contract.tile_scheduler_metadata_len())
        .unwrap();
    let mut splits = ctx
        .stream
        .alloc_zeros::<i32>(contract.num_splits_len())
        .unwrap();
    glm52_flashmla_sparse_decode_metadata_launch(
        &ctx,
        contract.batch_size,
        num_sm_parts,
        None,
        &mut sched,
        &mut splits,
    )
    .unwrap();

    // decode
    let mut latent = ctx
        .stream
        .alloc_zeros::<bf16>(contract.latent_len())
        .unwrap();
    let mut lse = ctx.stream.alloc_zeros::<f32>(contract.lse_len()).unwrap();
    let mut lse_accum = ctx
        .stream
        .alloc_zeros::<f32>(contract.lse_accum_len())
        .unwrap();
    let mut o_accum = ctx
        .stream
        .alloc_zeros::<f32>(contract.o_accum_len())
        .unwrap();
    glm52_flashmla_sparse_decode_launch(
        &ctx,
        contract,
        &q_d,
        &cache_d,
        &topk_d,
        None,
        &sched,
        &splits,
        &mut latent,
        &mut lse,
        &mut lse_accum,
        &mut o_accum,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();

    let got: Vec<f32> = ctx
        .stream
        .clone_dtoh(&latent)
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect();

    let sig = want_oracle.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    println!("flashmla latent [64,512]  signal|max|={sig:.5}");
    let (max_fp8, _) = cmp("fp8ref", &got, &want_fp8);
    cmp("oracle(fullprec)", &got, &want_oracle);

    assert!(got.iter().all(|x| x.is_finite()), "latent has non-finite");
    // Tight gate vs the same-fp8-cache numpy reference: only bf16 kernel arithmetic
    // differs. A wrong layout/scale/rope/index would blow this far past tol.
    assert!(
        max_fp8 < 1.0e-3,
        "FlashMLA latent vs fp8 reference max|Δ|={max_fp8} exceeds 1e-3 (signal {sig})"
    );
}
