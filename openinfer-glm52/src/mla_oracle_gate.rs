//! HF-oracle gate for the single-layer MLA decode brick.
//!
//! The oracle side is `tools/accuracy/glm52_oracle.py` (pinned transformers
//! `glm_moe_dsa` — the official modeling code): it runs layer-0 attention on a
//! seeded input and emits the probe constants pasted below. This test replays
//! the *same* input through `glm52_mla_decode_forward` position by position
//! (prefill-via-decode, full top-k — DSA-equivalent at ctx <= 2048) and asserts
//! the outputs land on the probes within an RMS-scaled tolerance.
//!
//! Input generation is splitmix64 (integer-only), bit-identical across the two
//! languages; the input digest is asserted first so PRNG drift fails loudly and
//! separately from kernel bugs. Float probes are tolerance-checked, never
//! hash-checked: fp8 GEMM + absorbed-attention accumulation order cannot be
//! bit-equal to the HF decompress path.
//!
//! Run (H200 + checkpoint):
//! ```text
//! OPENINFER_TEST_MODEL_PATH=/data/models/GLM-5.2-FP8 \
//!   cargo test --release -p openinfer-glm52 --features glm52 --lib mla_oracle -- --ignored --nocapture
//! ```
//! Set `OPENINFER_GLM52_ORACLE_DUMP=/path/taps.safetensors` (harness `--emit
//! safetensors`) to additionally diff the whole output tensor, not just probes.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use half::bf16;
use sha2::{Digest, Sha256};

use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE, GLM52_FLASHMLA_SPARSE_TOPK, Glm52FlashMlaSparseDecode,
    glm52_flashmla_sparse_decode_num_sm_parts,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::fp8::Glm52ProjBytes;
use crate::mla_decode::{Glm52MlaLayerWeights, glm52_mla_decode_forward};

// ---- BEGIN GENERATED: glm52_oracle probes ----
// uv run tools/accuracy/glm52_oracle.py --model-path /data/models/GLM-5.2-FP8 \
//     --ctx 200 --seed 0x5eed604d --layer 0 --precision fp8sim
// transformers=5.12.1 torch=2.12.1+cu130
const ORACLE_SEED: u64 = 0x5eed_604d;
const ORACLE_CTX: usize = 200;
// sha256[..16] of the seeded bf16 input — a mismatch means PRNG drift, not a kernel bug.
const ORACLE_HIDDEN_DIGEST: &str = "922e6646a688905e";
// tap `o` [200, 6144] bf16 digest=a673bd68e8c83185 (provenance only, never assert)
const ORACLE_O_RMS: f32 = 1.375840278e-03;
const ORACLE_O_REL_TOL: f32 = 0.05;
const ORACLE_O_PROBES: &[(usize, f32)] = &[
    (8720, -7.019042969e-04),
    (14476, 3.528594971e-04),
    (19280, 1.441955566e-03),
    (23544, 1.053810120e-04),
    (23874, 2.502441406e-03),
    (40579, 6.818771362e-05),
    (43624, 2.288818359e-04),
    (51849, -2.929687500e-03),
    (62323, 1.235961914e-03),
    (79939, -2.960205078e-03),
    (132890, -7.133483887e-04),
    (161812, -6.408691406e-04),
    (218021, 3.128051758e-04),
    (239838, 2.685546875e-03),
    (280089, -2.784729004e-04),
    (362277, 7.400512695e-04),
    (370383, -1.129150391e-03),
    (374208, 1.773834229e-04),
    (378353, 1.497268677e-04),
    (405409, 3.643035889e-04),
    (409140, -1.335144043e-03),
    (427475, -2.887099981e-07),
    (431425, 4.062652588e-04),
    (436346, 6.065368652e-04),
    (466158, 2.059936523e-03),
    (470207, -1.045227051e-03),
    (499911, -4.291534424e-04),
    (558784, -2.120971680e-03),
    (577517, -3.299713135e-04),
    (658200, -1.075744629e-03),
    (665506, 2.960205078e-03),
    (693297, 5.912780762e-04),
    (701374, 2.157688141e-05),
    (740279, -2.395629883e-03),
    (742773, 1.457214355e-03),
    (744036, 8.440017700e-05),
    (775501, -9.727478027e-05),
    (780318, 2.586841583e-05),
    (784846, 8.630752563e-05),
    (789177, 2.920627594e-05),
    (828119, -1.617431641e-03),
    (847914, -2.914428711e-03),
    (870453, -1.136779785e-03),
    (874510, 1.220703125e-03),
    (939675, 4.711151123e-04),
    (943269, -5.989074707e-04),
    (957049, 4.029273987e-05),
    (961775, -1.258850098e-03),
    (967873, 1.518249512e-03),
    (985001, 5.645751953e-04),
    (1018444, -4.825592041e-04),
    (1028723, 4.920959473e-04),
    (1042431, -7.972717285e-04),
    (1050404, -5.378723145e-04),
    (1085060, -2.222061157e-04),
    (1115997, -1.693725586e-03),
    (1117939, -3.032684326e-04),
    (1156020, 1.098632812e-03),
    (1160003, -3.108978271e-04),
    (1163780, -5.226135254e-04),
    (1164135, -9.679794312e-05),
    (1194525, 5.722045898e-04),
    (1200098, 7.247924805e-04),
    (1225370, 2.994537354e-04),
];
// ---- END GENERATED ----

const HIDDEN: usize = 6144;
const ROPE_HALF: usize = 32;
const ROPE_THETA: f32 = 8_000_000.0;
// qk_head_dim(256)^-0.5; rope_type "default" means no yarn mscale correction.
const SM_SCALE: f32 = 0.0625;

/// splitmix64 -> 53-bit uniform -> (u - 0.5) * 4.0 -> f32 -> bf16. Mirror of
/// the Python generator, including the f64 -> f32 -> bf16 double rounding
/// (numpy `.astype(np.float32)` then torch's bf16 cast); a direct f64 -> bf16
/// rounds a handful of values differently and fails the digest.
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

fn bf16_digest(data: &[bf16]) -> String {
    let mut hasher = Sha256::new();
    for v in data {
        hasher.update(v.to_bits().to_le_bytes());
    }
    hex::encode(&hasher.finalize()[..8])
}

/// Per-position rotary table first half `[32]` in the HF pipeline's precision:
/// angles in f32, cos/sin rounded to bf16.
fn rope_tables(position: usize) -> (Vec<bf16>, Vec<bf16>) {
    (0..ROPE_HALF)
        .map(|j| {
            let inv_freq = 1.0 / ROPE_THETA.powf(j as f32 / ROPE_HALF as f32);
            let angle = position as f32 * inv_freq;
            (bf16::from_f32(angle.cos()), bf16::from_f32(angle.sin()))
        })
        .unzip()
}

/// Copy layer-0 attention tensors out of the checkpoint shards. Owned copies
/// (~250 MB total) keep the borrow story trivial; this is a test-only path.
struct Layer0Tensors {
    by_name: BTreeMap<String, Vec<u8>>,
}

impl Layer0Tensors {
    fn load(model_path: &Path) -> Result<Self> {
        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            model_path.join("model.safetensors.index.json"),
        )?)?;
        let weight_map = index["weight_map"]
            .as_object()
            .context("weight_map missing")?;
        let prefix = "model.layers.0.self_attn";
        let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::default();
        for (name, shard) in weight_map {
            if name.starts_with(prefix) && !name.contains(".indexer.") {
                by_shard
                    .entry(shard.as_str().context("shard not a string")?.to_owned())
                    .or_default()
                    .push(name.clone());
            }
        }
        let mut by_name = BTreeMap::new();
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
            .with_context(|| format!("layer-0 tensor {name} not loaded"))
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

fn load_layer0(ctx: &DeviceContext, model_path: &Path) -> Result<Glm52MlaLayerWeights> {
    let t = Layer0Tensors::load(model_path)?;
    let p = "model.layers.0.self_attn";
    Glm52MlaLayerWeights::from_host(
        ctx,
        &t.proj(&format!("{p}.q_a_proj"), 2048, HIDDEN)?,
        t.bytes(&format!("{p}.q_a_layernorm.weight"))?,
        &t.proj(&format!("{p}.q_b_proj"), 16384, 2048)?,
        &t.proj(&format!("{p}.kv_a_proj_with_mqa"), 576, HIDDEN)?,
        t.bytes(&format!("{p}.kv_a_layernorm.weight"))?,
        &t.proj(&format!("{p}.kv_b_proj"), 28672, 512)?,
        &t.proj(&format!("{p}.o_proj"), HIDDEN, 16384)?,
    )
}

#[test]
#[ignore = "requires H200 + GLM-5.2-FP8 checkpoint"]
fn mla_oracle_gate() -> Result<()> {
    let model_path = std::env::var_os("OPENINFER_TEST_MODEL_PATH")
        .map_or_else(|| PathBuf::from("models/GLM-5.2-FP8"), PathBuf::from);

    let hidden_host = seeded_hidden(ORACLE_SEED, ORACLE_CTX * HIDDEN);
    let digest = bf16_digest(&hidden_host);
    ensure!(
        digest == ORACLE_HIDDEN_DIGEST,
        "input digest {digest} != oracle {ORACLE_HIDDEN_DIGEST}: PRNG drift or stale probes — regenerate with tools/accuracy/glm52_oracle.py"
    );

    let ctx = DeviceContext::new()?;
    let w = load_layer0(&ctx, &model_path)?;

    let contract = Glm52FlashMlaSparseDecode {
        batch_size: 1,
        num_blocks: ORACLE_CTX.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE),
        topk: GLM52_FLASHMLA_SPARSE_TOPK,
        num_sm_parts: glm52_flashmla_sparse_decode_num_sm_parts()?,
        sm_scale: SM_SCALE,
    };
    let mut cache = ctx
        .stream
        .alloc_zeros::<u8>(contract.packed_kv_cache_len())?;

    // Prefill via decode: position p writes its token into the cache, then
    // attends over the full prefix [0..=p] via a -1-padded top-k list.
    let mut outputs = Vec::with_capacity(ORACLE_CTX * HIDDEN);
    for position in 0..ORACLE_CTX {
        let mut hidden = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
        ctx.stream.memcpy_htod(
            &hidden_host[position * HIDDEN..(position + 1) * HIDDEN],
            &mut hidden,
        )?;
        let (cos_host, sin_host) = rope_tables(position);
        let mut cos = ctx.stream.alloc_zeros::<bf16>(ROPE_HALF)?;
        let mut sin = ctx.stream.alloc_zeros::<bf16>(ROPE_HALF)?;
        ctx.stream.memcpy_htod(&cos_host, &mut cos)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin)?;

        let mut topk_host = vec![-1i32; GLM52_FLASHMLA_SPARSE_TOPK];
        for (slot, v) in topk_host.iter_mut().enumerate().take(position + 1) {
            *v = slot as i32;
        }
        let mut topk = ctx.stream.alloc_zeros::<i32>(GLM52_FLASHMLA_SPARSE_TOPK)?;
        ctx.stream.memcpy_htod(&topk_host, &mut topk)?;

        let o = glm52_mla_decode_forward(
            &ctx, &w, &hidden, &cos, &sin, &mut cache, position, &topk, contract,
        )?;
        let o_host = ctx.stream.clone_dtoh(&o)?;
        outputs.extend(o_host.iter().map(|v| v.to_f32()));
    }

    assert_probes(&outputs);
    if let Some(dump) = std::env::var_os("OPENINFER_GLM52_ORACLE_DUMP") {
        diff_against_dump(&outputs, Path::new(&dump))?;
    }
    Ok(())
}

/// Probe assertion: |engine - oracle| <= rel_tol * oracle_rms at every sampled
/// index. RMS-scaled (not per-element relative) because near-zero elements have
/// huge relative error at bf16 while being irrelevant to the layer output.
fn assert_probes(outputs: &[f32]) {
    assert!(
        !ORACLE_O_PROBES.is_empty(),
        "probe block is the ungenerated placeholder"
    );
    let tol = ORACLE_O_REL_TOL * ORACLE_O_RMS;
    let failures: Vec<_> = ORACLE_O_PROBES
        .iter()
        .filter(|&&(idx, expected)| (outputs[idx] - expected).abs() > tol)
        .collect();
    println!(
        "oracle gate: {}/{} probes within tol={tol:.6e}",
        ORACLE_O_PROBES.len() - failures.len(),
        ORACLE_O_PROBES.len()
    );
    for &&(idx, expected) in failures.iter().take(10) {
        println!(
            "  probe[{idx}]: oracle {expected:.6} vs engine {:.6}",
            outputs[idx]
        );
    }
    assert!(
        failures.is_empty(),
        "{} probes out of tolerance",
        failures.len()
    );
}

/// Whole-tensor diff against the harness's safetensors dump (`o` tap): a
/// probes-pass/rest-garbage failure mode cannot hide from this. Asserts the
/// coverage-stable statistics (diff RMS and p99) — the absolute max grows with
/// element count (bf16 tail over 1.2M elements) so it is printed, not asserted;
/// same lesson as the qwen3 golden gate.
fn diff_against_dump(outputs: &[f32], dump: &Path) -> Result<()> {
    let mmap = crate::weights::mmap_file(dump)?;
    let st = safetensors::SafeTensors::deserialize(mmap.as_ref())?;
    let oracle: Vec<f32> = st
        .tensor("o")?
        .data()
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    ensure!(
        oracle.len() == outputs.len(),
        "dump `o` has {} elements, engine produced {}",
        oracle.len(),
        outputs.len()
    );
    let mut worst: Vec<(usize, f32)> = outputs
        .iter()
        .zip(&oracle)
        .enumerate()
        .map(|(i, (a, b))| (i, (a - b).abs()))
        .collect();
    worst.sort_by(|a, b| b.1.total_cmp(&a.1));
    let diff_rms = (worst.iter().map(|(_, d)| d * d).sum::<f32>() / worst.len() as f32).sqrt();
    let p99 = worst[worst.len() / 100].1;
    println!(
        "full-tensor diff vs dump: diff_rms={diff_rms:.6e}, p99={p99:.6e}, max={:.6e} (printed, not asserted), top offenders:",
        worst[0].1
    );
    for (i, d) in worst.iter().take(10) {
        println!(
            "  o[{i}]: engine {:.6} oracle {:.6} (|d|={d:.6})",
            outputs[*i], oracle[*i]
        );
    }
    let tol = ORACLE_O_REL_TOL * ORACLE_O_RMS;
    ensure!(
        diff_rms <= tol && p99 <= tol,
        "full-tensor diff stats out of tolerance: rms {diff_rms:.6e} / p99 {p99:.6e} vs tol {tol:.6e}"
    );
    Ok(())
}
