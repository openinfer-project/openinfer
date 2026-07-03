//! HF-oracle gate for the decode bookends (embed / final RMSNorm / lm_head).
//!
//! The oracle side is `tools/accuracy/glm52_oracle.py --stage bookend`. Three
//! assertions, typed by what each op can promise:
//! - embed: a bf16 gather is EXACT — assert the embed-rows digest bit-for-bit.
//! - logits: RMS-scaled probe tolerance (bf16 GEMV vs f32 reference matmul).
//! - argmax: EXACT per position — the only thing greedy decode consumes.
//!
//! Run (GPU + checkpoint):
//! ```text
//! OPENINFER_TEST_MODEL_PATH=/data/models/GLM-5.2-FP8 \
//!   cargo test --release -p openinfer-glm52 --features glm52 --lib bookend_oracle -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use half::bf16;
use sha2::{Digest, Sha256};

use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

use crate::bookend::{glm52_embed, glm52_final_norm, glm52_lm_head};
use crate::config::{GLM52_HIDDEN, GLM52_VOCAB};

// ---- BEGIN GENERATED: glm52_oracle bookend probes ----
// UNGENERATED PLACEHOLDER — run on the H200 node and paste:
// uv run tools/accuracy/glm52_oracle.py --model-path /data/models/GLM-5.2-FP8 \
//     --ctx 8 --seed 0x5eed604d --precision fp8sim --stage bookend
const ORACLE_SEED: u64 = 0x5eed_604d;
const ORACLE_CTX: usize = 8;
const ORACLE_HIDDEN_DIGEST: &str = "UNGENERATED";
const ORACLE_EMBED_ROWS_DIGEST: &str = "UNGENERATED";
const ORACLE_LOGITS_RMS: f32 = 0.0;
const ORACLE_LOGITS_REL_TOL: f32 = 0.05;
const ORACLE_LOGITS_PROBES: &[(usize, f32)] = &[];
const ORACLE_ARGMAX: &[u32] = &[];
// ---- END GENERATED ----

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

/// Mirror of the harness `seeded_token_ids`: splitmix64 uniform on `seed ^
/// 0xB00C`, scaled to the vocab (f64 multiply + truncation, exact both sides).
fn seeded_token_ids(seed: u64, count: usize, vocab: usize) -> Vec<u32> {
    let seed = seed ^ 0xB00C;
    (0..count)
        .map(|i| {
            let mut z = seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 / (1u64 << 53) as f64;
            ((u * vocab as f64) as u64).min(vocab as u64 - 1) as u32
        })
        .collect()
}

fn bf16_digest(data: &[bf16]) -> String {
    let mut hasher = Sha256::new();
    for v in data {
        hasher.update(v.to_bits().to_le_bytes());
    }
    hex::encode(&hasher.finalize()[..8])
}

/// Load one whole bf16 tensor from the checkpoint by name.
fn load_bf16(model_path: &Path, name: &str) -> Result<Vec<u8>> {
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        model_path.join("model.safetensors.index.json"),
    )?)?;
    let shard = index["weight_map"][name]
        .as_str()
        .with_context(|| format!("{name} missing from weight map"))?;
    let mmap = crate::weights::mmap_file(&model_path.join(shard))?;
    let st = safetensors::SafeTensors::deserialize(mmap.as_ref())?;
    Ok(st.tensor(name)?.data().to_vec())
}

#[test]
#[ignore = "requires GPU + GLM-5.2-FP8 checkpoint"]
fn bookend_oracle_gate() -> Result<()> {
    assert!(
        !ORACLE_LOGITS_PROBES.is_empty(),
        "probe block is the ungenerated placeholder — run the harness and paste"
    );
    let model_path = std::env::var_os("OPENINFER_TEST_MODEL_PATH")
        .map_or_else(|| PathBuf::from("models/GLM-5.2-FP8"), PathBuf::from);

    let hidden_host = seeded_hidden(ORACLE_SEED, ORACLE_CTX * GLM52_HIDDEN);
    let digest = bf16_digest(&hidden_host);
    ensure!(
        digest == ORACLE_HIDDEN_DIGEST,
        "input digest {digest} != oracle {ORACLE_HIDDEN_DIGEST}: PRNG drift or stale probes"
    );
    let token_ids = seeded_token_ids(ORACLE_SEED, ORACLE_CTX, GLM52_VOCAB);

    let ctx = DeviceContext::new()?;
    let embed = DeviceMatrix::from_safetensors(
        &ctx,
        &load_bf16(&model_path, "model.embed_tokens.weight")?,
        GLM52_VOCAB,
        GLM52_HIDDEN,
    )?;
    let norm_weight =
        DeviceVec::from_safetensors(&ctx, &load_bf16(&model_path, "model.norm.weight")?)?;
    let lm_head = DeviceMatrix::from_safetensors(
        &ctx,
        &load_bf16(&model_path, "lm_head.weight")?,
        GLM52_VOCAB,
        GLM52_HIDDEN,
    )?;

    // ---- embed: exact gather digest ----
    let mut token_id_buf = ctx.stream.alloc_zeros::<u32>(1)?;
    let mut embed_rows: Vec<bf16> = Vec::with_capacity(ORACLE_CTX * GLM52_HIDDEN);
    for &id in &token_ids {
        ctx.stream.memcpy_htod(&[id], &mut token_id_buf)?;
        let row = glm52_embed(&ctx, &embed, &token_id_buf)?;
        embed_rows.extend(ctx.stream.clone_dtoh(&row.data)?);
    }
    let embed_digest = bf16_digest(&embed_rows);
    ensure!(
        embed_digest == ORACLE_EMBED_ROWS_DIGEST,
        "embed rows digest {embed_digest} != oracle {ORACLE_EMBED_ROWS_DIGEST} (gather is exact — this is a hard bug)"
    );
    println!("bookend embed: digest exact ({ORACLE_CTX} rows)");

    // ---- final norm + lm_head: probes + exact argmax ----
    ensure!(ORACLE_ARGMAX.len() == ORACLE_CTX, "argmax length mismatch");
    let mut logits_all: Vec<f32> = Vec::with_capacity(ORACLE_CTX * GLM52_VOCAB);
    for position in 0..ORACLE_CTX {
        let mut hidden = DeviceVec::zeros(&ctx, GLM52_HIDDEN)?;
        ctx.stream.memcpy_htod(
            &hidden_host[position * GLM52_HIDDEN..(position + 1) * GLM52_HIDDEN],
            &mut hidden.data,
        )?;
        let normed = glm52_final_norm(&ctx, &hidden, &norm_weight)?;
        let logits = glm52_lm_head(&ctx, &normed, &lm_head)?;
        let logits_host: Vec<f32> = ctx
            .stream
            .clone_dtoh(&logits.data)?
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let argmax = logits_host
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        ensure!(
            argmax == ORACLE_ARGMAX[position],
            "argmax mismatch at position {position}: engine {argmax} vs oracle {}",
            ORACLE_ARGMAX[position]
        );
        logits_all.extend(logits_host);
    }
    println!("bookend argmax: exact across {ORACLE_CTX} positions");

    let tol = ORACLE_LOGITS_REL_TOL * ORACLE_LOGITS_RMS;
    let failures: Vec<_> = ORACLE_LOGITS_PROBES
        .iter()
        .filter(|&&(idx, expected)| (logits_all[idx] - expected).abs() > tol)
        .collect();
    println!(
        "bookend logits: {}/{} probes within tol={tol:.6e}",
        ORACLE_LOGITS_PROBES.len() - failures.len(),
        ORACLE_LOGITS_PROBES.len()
    );
    for &&(idx, expected) in failures.iter().take(10) {
        println!(
            "  probe[{idx}]: oracle {expected:.6} vs engine {:.6}",
            logits_all[idx]
        );
    }
    ensure!(
        failures.is_empty(),
        "{} probes out of tolerance",
        failures.len()
    );
    Ok(())
}
