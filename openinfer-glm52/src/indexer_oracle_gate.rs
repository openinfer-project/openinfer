//! HF-oracle gate for the DSA indexer forward.
//!
//! The oracle side is `tools/accuracy/glm52_oracle.py --stage indexer`
//! (transformers 5.13.0.dev0 local checkout — fixes indexer RoPE interleave).
//! It runs layer-0 attention on a seeded input and captures `topk_indices`,
//! `q_resid`, `cos`, `sin` as probes. This test loads the same checkpoint
//! weights, feeds the same inputs, and asserts set-overlap on the top-k
//! indices (FlashInfer vs torch.topk tie-break on 1-ULP logit ties differs;
//! exact match is impossible, so we allow 1/2048 divergence).
//!
//! Run (H200 + checkpoint):
//! ```text
//! OPENINFER_TEST_MODEL_PATH=/data/models/GLM-5.2-FP8 \
//! OPENINFER_DEEPGEMM_ROOT=openinfer-kernels/third_party/DeepGEMM \
//! CUDA_HOME=/usr/local/cuda \
//!   cargo test --release -p openinfer-glm52 --features glm52 --lib indexer_oracle -- --ignored --nocapture
//! ```

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use half::bf16;
use sha2::{Digest, Sha256};

use openinfer_kernels::ops::Glm52IndexerCacheLayout;
use openinfer_kernels::tensor::DeviceContext;

use crate::fp8::Glm52ProjBytes;
use crate::indexer::{Glm52IndexerLayerWeights, glm52_indexer_forward};

// ---- BEGIN GENERATED: glm52_oracle indexer probes ----
// uv run tools/accuracy/glm52_oracle.py --model-path /data/models/GLM-5.2-FP8 \
//     --ctx 4096 --seed 0x5eed604d --layer 0 --precision fp8sim --stage indexer
// transformers=5.13.0.dev0 torch=2.12.1+cu130
const ORACLE_SEED: u64 = 0x5eed_604d;
const ORACLE_CTX: usize = 4096;
// Input digests — verify before running the indexer forward.
// These will be filled in after the first oracle run on H200.
#[allow(dead_code)]
const ORACLE_Q_RESID_DIGEST: &str = "TODO_RunHarness";
#[allow(dead_code)]
const ORACLE_COS_DIGEST: &str = "TODO_RunHarness";
#[allow(dead_code)]
const ORACLE_SIN_DIGEST: &str = "TODO_RunHarness";
// topk_indices [2048] i32 (provenance only — the gate asserts set-overlap, not bit-equality)
#[allow(dead_code)]
const ORACLE_TOPK_DIGEST: &str = "TODO_RunHarness";
const ORACLE_TOPK_SET: &[i32] = &[];
// ---- END GENERATED ----

const HIDDEN: usize = 6144;
const Q_LORA: usize = 2048;
const INDEX_HEAD_DIM: usize = 128;
const CACHE_BLOCK_SIZE: usize = 128;
const NUM_SMS: usize = 132;

fn bf16_digest(data: &[bf16]) -> String {
    let mut hasher = Sha256::new();
    for v in data {
        hasher.update(v.to_bits().to_le_bytes());
    }
    hex::encode(&hasher.finalize()[..8])
}

fn i32_digest(data: &[i32]) -> String {
    let mut hasher = Sha256::new();
    for v in data {
        hasher.update(v.to_le_bytes());
    }
    hex::encode(&hasher.finalize()[..8])
}

fn seeded_hidden(seed: u64, count: usize) -> Vec<bf16> {
    (0..count)
        .map(|i| {
            let mut z = seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 / (1u64 << 53) as f64;
            bf16::from_f32(((u - 0.5) * 4.0) as f32)
        })
        .collect()
}

fn rope_tables(position: usize) -> (Vec<bf16>, Vec<bf16>) {
    (0..32)
        .map(|j| {
            let inv_freq = 1.0 / 8_000_000.0_f32.powf(j as f32 / 32.0);
            let angle = position as f32 * inv_freq;
            (bf16::from_f32(angle.cos()), bf16::from_f32(angle.sin()))
        })
        .unzip()
}

struct Layer0IndexerTensors {
    by_name: std::collections::BTreeMap<String, Vec<u8>>,
}

impl Layer0IndexerTensors {
    fn load(model_path: &Path) -> Result<Self> {
        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            model_path.join("model.safetensors.index.json"),
        )?)?;
        let weight_map = index["weight_map"]
            .as_object()
            .context("weight_map missing")?;
        let prefix = "model.layers.0.self_attn.indexer";
        let mut by_shard: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::default();
        for (name, shard) in weight_map {
            if name.starts_with(prefix) {
                by_shard
                    .entry(shard.as_str().context("shard not a string")?.to_owned())
                    .or_default()
                    .push(name.clone());
            }
        }
        let mut by_name = std::collections::BTreeMap::new();
        for (shard, names) in by_shard {
            let mmap = crate::weights::mmap_file(&model_path.join(&shard))?;
            let st = safetensors::SafeTensors::deserialize(mmap.as_ref())?;
            for name in names {
                by_name.insert(name.clone(), st.tensor(&name)?.data().to_vec());
            }
        }
        Ok(Self { by_name })
    }

    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.by_name
            .get(name)
            .map(Vec::as_slice)
            .with_context(|| format!("layer-0 indexer tensor {name} not loaded"))
    }

    fn proj(&self, stem: &str, n: usize, k: usize) -> Result<Glm52ProjBytes<'_>> {
        Ok(Glm52ProjBytes {
            weight: self.bytes(&format!("{stem}.weight"))?,
            scale: self.bytes(&format!("{stem}.weight_scale_inv"))?,
            n,
            k,
        })
    }
}

fn load_indexer_layer0(ctx: &DeviceContext, model_path: &Path) -> Result<Glm52IndexerLayerWeights> {
    let t = Layer0IndexerTensors::load(model_path)?;
    let p = "model.layers.0.self_attn.indexer";
    Glm52IndexerLayerWeights::from_host(
        ctx,
        &t.proj(&format!("{p}.wq_b"), 32 * 128, Q_LORA)?,
        &t.proj(&format!("{p}.wk"), 128, HIDDEN)?,
        &t.proj(&format!("{p}.weights_proj"), 32, HIDDEN)?,
        t.bytes(&format!("{p}.k_norm.weight"))?,
        t.bytes(&format!("{p}.k_norm.bias"))?,
    )
}

#[test]
#[ignore = "requires H200 + GLM-5.2-FP8 checkpoint + DeepGEMM env"]
fn indexer_oracle_gate() -> Result<()> {
    let model_path = std::env::var_os("OPENINFER_TEST_MODEL_PATH")
        .map_or_else(|| PathBuf::from("models/GLM-5.2-FP8"), PathBuf::from);

    // Position to test: the last token in the sequence (attends over the full prefix).
    let position = ORACLE_CTX - 1;

    // ---- generate inputs ----
    let hidden_host = seeded_hidden(ORACLE_SEED, ORACLE_CTX * HIDDEN);
    let hidden_digest = bf16_digest(&hidden_host);
    ensure!(
        hidden_digest == ORACLE_HIDDEN_DIGEST,
        "hidden digest {hidden_digest} != oracle {ORACLE_HIDDEN_DIGEST}: \
         PRNG drift or stale probes — regenerate with tools/accuracy/glm52_oracle.py"
    );

    let ctx = DeviceContext::new()?;
    let w = load_indexer_layer0(&ctx, &model_path)?;

    // ---- upload inputs for the last position ----
    let mut hidden = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
    ctx.stream.memcpy_htod(
        &hidden_host[position * HIDDEN..(position + 1) * HIDDEN],
        &mut hidden,
    )?;

    // q_resid: in the full pipeline this is q_a_layernorm(q_a_proj(hidden)).
    // For the oracle gate we need the oracle to emit q_resid as a safetensors
    // dump and load it here. For now, we use the seeded hidden as a placeholder
    // until the oracle dump pipeline is wired.
    // TODO: load q_resid from oracle safetensors dump.
    let q_resid = ctx.stream.alloc_zeros::<bf16>(Q_LORA)?;

    let (cos_host, sin_host) = rope_tables(position);
    let mut cos = ctx.stream.alloc_zeros::<bf16>(32)?;
    let mut sin = ctx.stream.alloc_zeros::<bf16>(32)?;
    ctx.stream.memcpy_htod(&cos_host, &mut cos)?;
    ctx.stream.memcpy_htod(&sin_host, &mut sin)?;

    // ---- cache setup ----
    let cache_blocks = ORACLE_CTX.div_ceil(CACHE_BLOCK_SIZE);
    let cache_layout = Glm52IndexerCacheLayout {
        cache_blocks,
        cache_block_size: CACHE_BLOCK_SIZE,
        cache_block_stride_bytes: CACHE_BLOCK_SIZE * (INDEX_HEAD_DIM + 4),
    };
    let cache_bytes = cache_layout.min_cache_bytes()?;
    let mut index_k_cache = ctx.stream.alloc_zeros::<u8>(cache_bytes)?;

    // slot_mapping: position p → slot p
    let slot_mapping_host = vec![position as i64];
    let mut slot_mapping = ctx.stream.alloc_zeros::<i64>(1)?;
    ctx.stream
        .memcpy_htod(&slot_mapping_host, &mut slot_mapping)?;

    // block_table: identity mapping (block i = i)
    let block_table_host: Vec<i32> = (0..cache_blocks as i32).collect();
    let mut block_table = ctx.stream.alloc_zeros::<i32>(cache_blocks)?;
    ctx.stream
        .memcpy_htod(&block_table_host, &mut block_table)?;

    let seq_lens_host = vec![(position + 1) as i32];
    let mut seq_lens = ctx.stream.alloc_zeros::<i32>(1)?;
    ctx.stream.memcpy_htod(&seq_lens_host, &mut seq_lens)?;

    // ---- run the indexer forward ----
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
        ORACLE_CTX,
    )?;

    // ---- assert set-overlap with oracle ----
    let topk_host = ctx.stream.clone_dtoh(&topk)?;
    let topk_digest = i32_digest(&topk_host);
    eprintln!("rust topk digest: {topk_digest}");
    eprintln!("oracle topk digest: {ORACLE_TOPK_DIGEST}");

    let rust_set: HashSet<i32> = topk_host.iter().copied().filter(|&v| v >= 0).collect();
    let oracle_set: HashSet<i32> = ORACLE_TOPK_SET.iter().copied().collect();

    let overlap = rust_set.intersection(&oracle_set).count();
    let max_allowed = rust_set.len().max(oracle_set.len());
    let min_required = max_allowed.saturating_sub(1); // allow 1 tie-break divergence

    ensure!(
        overlap >= min_required,
        "indexer topk set-overlap {overlap} < required {min_required} \
         (rust set size {}, oracle set size {})",
        rust_set.len(),
        oracle_set.len()
    );

    eprintln!(
        "indexer oracle gate: overlap {overlap}/{} (allowed >= {min_required})",
        max_allowed
    );
    Ok(())
}
