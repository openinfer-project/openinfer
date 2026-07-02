//! Smoke test for the DSA indexer forward — verifies kernel launches and
//! output shape, not correctness (that requires the oracle gate + checkpoint).
//!
//! Run (H200 + DeepGEMM env):
//! ```text
//! OPENINFER_DEEPGEMM_ROOT=openinfer-kernels/third_party/DeepGEMM \
//! CUDA_HOME=/usr/local/cuda \
//!   cargo test --release -p openinfer-glm52 --features glm52 --lib indexer_smoke -- --nocapture
//! ```

use anyhow::Result;
use half::bf16;

use openinfer_kernels::ops::{GLM52_INDEXER_TOPK, Glm52IndexerCacheLayout};
use openinfer_kernels::tensor::DeviceContext;

use crate::fp8::{FP8_BLOCK, Glm52ProjBytes};
use crate::indexer::{Glm52IndexerLayerWeights, glm52_indexer_forward};

const HIDDEN: usize = 6144;
const Q_LORA: usize = 2048;
const INDEX_HEADS: usize = 32;
const INDEX_HEAD_DIM: usize = 128;
const CACHE_BLOCK_SIZE: usize = 64; // DeepGEMM paged MQA requires BLOCK_KV=64
const CACHE_BLOCKS: usize = 4;
const MAX_MODEL_LEN: usize = 512;
const NUM_SMS: usize = 132;

fn deepgemm_env_ready() -> bool {
    std::env::var("OPENINFER_DEEPGEMM_ROOT").is_ok() && std::env::var("CUDA_HOME").is_ok()
}

fn synthetic_proj(n: usize, k: usize) -> Glm52ProjBytes<'static> {
    let weight = vec![0u8; n * k];
    // scale_inv must be non-zero (TRTLLM rejects denormals). Use 1.0f32 = [0x3f800000].
    let scale_len = n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK);
    let scale: Vec<u8> = (0..scale_len).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    Glm52ProjBytes {
        weight: weight.leak(),
        scale: scale.leak(),
        n,
        k,
    }
}

#[test]
#[ignore = "requires H200 (SM90) — TRTLLM FP8 blockscale GEMM is SM90-only"]
fn indexer_smoke() -> Result<()> {
    if !deepgemm_env_ready() {
        eprintln!("skip: set OPENINFER_DEEPGEMM_ROOT + CUDA_HOME to run indexer smoke test");
        return Ok(());
    }

    let ctx = DeviceContext::new()?;

    // ---- synthetic weights (all zeros, scale=1.0) ----
    let wq_b = synthetic_proj(INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA);
    let wk = synthetic_proj(INDEX_HEAD_DIM, HIDDEN);
    // weights_proj is bf16 [32, 6144] (not fp8 in checkpoint)
    let weights_proj_bf16 = vec![0u8; INDEX_HEADS * HIDDEN * 2];
    let k_norm_w = vec![0u8; INDEX_HEAD_DIM * 2]; // bf16
    let k_norm_b = vec![0u8; INDEX_HEAD_DIM * 2]; // bf16

    let w = Glm52IndexerLayerWeights::from_host(
        &ctx, &wq_b, &wk, &weights_proj_bf16, &k_norm_w, &k_norm_b,
    )?;

    // ---- inputs ----
    let hidden = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
    let q_resid = ctx.stream.alloc_zeros::<bf16>(Q_LORA)?;
    let cos = ctx.stream.alloc_zeros::<bf16>(32)?;
    let sin = ctx.stream.alloc_zeros::<bf16>(32)?;

    // ---- cache setup ----
    let cache_layout = Glm52IndexerCacheLayout {
        cache_blocks: CACHE_BLOCKS,
        cache_block_size: CACHE_BLOCK_SIZE,
        cache_block_stride_bytes: CACHE_BLOCK_SIZE * (INDEX_HEAD_DIM + 4),
    };
    let cache_bytes = cache_layout.min_cache_bytes()?;
    let mut index_k_cache = ctx.stream.alloc_zeros::<u8>(cache_bytes)?;

    let slot_mapping = ctx.stream.alloc_zeros::<i64>(1)?;
    let block_table_host: Vec<i32> = (0..CACHE_BLOCKS as i32).collect();
    let mut block_table = ctx.stream.alloc_zeros::<i32>(CACHE_BLOCKS)?;
    ctx.stream
        .memcpy_htod(&block_table_host, &mut block_table)?;
    let seq_lens_host = vec![128i32];
    let mut seq_lens = ctx.stream.alloc_zeros::<i32>(1)?;
    ctx.stream.memcpy_htod(&seq_lens_host, &mut seq_lens)?;

    // ---- forward ----
    let topk = glm52_indexer_forward(
        &ctx,
        &w,
        &hidden,
        &q_resid,
        &cos,
        &sin,
        &mut index_k_cache,
        cache_layout,
        &slot_mapping,
        &block_table,
        &seq_lens,
        NUM_SMS,
        MAX_MODEL_LEN,
    )?;

    // ---- verify shape ----
    assert_eq!(
        topk.len(),
        GLM52_INDEXER_TOPK,
        "topk_indices must be [2048]"
    );

    let topk_host = ctx.stream.clone_dtoh(&topk)?;
    for &v in &topk_host {
        assert!(v >= -1, "topk slot {v} < -1 (corrupted)");
        assert!(
            v < (CACHE_BLOCKS * CACHE_BLOCK_SIZE) as i32,
            "topk slot {v} out of range"
        );
    }

    eprintln!(
        "indexer smoke: topk[0..8] = {:?}",
        &topk_host[..8.min(topk_host.len())]
    );
    Ok(())
}
