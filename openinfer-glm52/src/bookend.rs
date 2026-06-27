//! GLM5.2 decode bookends for bs=1: the token embedding (PP stage 0) and the
//! final RMSNorm + lm_head tail (last PP stage) that brackets the layer stack.
//!
//! All three are plain bf16 ops (the embedding table, `model.norm.weight`, and
//! `lm_head.weight` are bf16, not fp8). These are thin wrappers over the shared
//! `openinfer-kernels` embed/norm/gemv ops -- the only GLM5.2-specific facts are
//! the dimensions and the 1e-5 RMS epsilon.

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use openinfer_kernels::ops::{embedding_decode_into, linear, rms_norm_into};
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

use crate::config::{GLM52_HIDDEN, GLM52_VOCAB};

const RMS_EPS: f32 = 1.0e-5;

/// Token embedding lookup (PP stage 0): `embed[token_id] -> [HIDDEN]`. `token_id`
/// is a single-element device buffer (read on-device, so the lookup is
/// CUDA-graph-safe -- the scheduler rewrites it in place each decode step).
pub fn glm52_embed(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_id: &CudaSlice<u32>,
) -> Result<DeviceVec> {
    ensure!(
        embed.rows == GLM52_VOCAB && embed.cols == GLM52_HIDDEN,
        "GLM5.2 embed table shape [{},{}] != [{GLM52_VOCAB},{GLM52_HIDDEN}]",
        embed.rows,
        embed.cols
    );
    let mut out = DeviceVec::zeros(ctx, GLM52_HIDDEN)?;
    embedding_decode_into(ctx, embed, token_id, &mut out)?;
    Ok(out)
}

/// Final RMSNorm (last PP stage): `rms_norm(hidden, model.norm.weight, eps=1e-5)`.
pub fn glm52_final_norm(
    ctx: &DeviceContext,
    hidden: &DeviceVec,
    norm_weight: &DeviceVec,
) -> Result<DeviceVec> {
    ensure!(
        hidden.len == GLM52_HIDDEN && norm_weight.len == GLM52_HIDDEN,
        "GLM5.2 final norm lengths hidden {} / weight {} != {GLM52_HIDDEN}",
        hidden.len,
        norm_weight.len
    );
    let mut out = DeviceVec::zeros(ctx, GLM52_HIDDEN)?;
    rms_norm_into(ctx, hidden, norm_weight, RMS_EPS, &mut out)?;
    Ok(out)
}

/// lm_head projection (last PP stage): `lm_head @ normed -> [VOCAB]` logits. The
/// weight is bf16 `[VOCAB, HIDDEN]`; the caller feeds the final-normed hidden.
pub fn glm52_lm_head(
    ctx: &DeviceContext,
    normed: &DeviceVec,
    lm_head: &DeviceMatrix,
) -> Result<DeviceVec> {
    ensure!(
        lm_head.rows == GLM52_VOCAB && lm_head.cols == GLM52_HIDDEN,
        "GLM5.2 lm_head shape [{},{}] != [{GLM52_VOCAB},{GLM52_HIDDEN}]",
        lm_head.rows,
        lm_head.cols
    );
    ensure!(
        normed.len == GLM52_HIDDEN,
        "GLM5.2 lm_head input len {} != {GLM52_HIDDEN}",
        normed.len
    );
    linear(ctx, normed, lm_head)
}

#[cfg(test)]
mod tests {
    //! Bookend decode gates (H200 sm_90):
    //!   embed   -- exact: `glm52_embed(t)` must equal the raw checkpoint row t.
    //!   norm    -- sigma-normalized abs-dev of `rms_norm(hidden)` vs the f32 ref.
    //!   lm_head -- same metric on the `[8, VOCAB]` logits, plus an EXACT argmax
    //!              match (the only thing greedy decode actually consumes).
    //!
    //! No-ops without a CUDA device, the checkpoint, or the bookend probe bins:
    //!   cargo test --release -p openinfer-glm52 --features <model> bookend -- --nocapture

    use super::*;
    use half::bf16;
    use memmap2::MmapOptions;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::{Path, PathBuf};

    fn model_path() -> PathBuf {
        std::env::var("GLM52_MODEL_PATH")
            .unwrap_or_else(|_| "/data/models/GLM-5.2-FP8".into())
            .into()
    }
    fn probe_dir() -> PathBuf {
        std::env::var("GLM52_BOOKEND_PROBE_DIR")
            .unwrap_or_else(|_| "/data/models/glm52_mla_ref/bookend_probe".into())
            .into()
    }
    fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
        std::fs::read(dir.join(name))
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn weight_map(model: &Path) -> HashMap<String, String> {
        let idx: Value = serde_json::from_str(
            &std::fs::read_to_string(model.join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        idx["weight_map"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect()
    }
    fn load_u8(model: &Path, map: &HashMap<String, String>, name: &str) -> Vec<u8> {
        let file =
            File::open(model.join(map.get(name).unwrap_or_else(|| panic!("missing {name}"))))
                .unwrap();
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let st = safetensors::SafeTensors::deserialize(&mmap).unwrap();
        st.tensor(name).unwrap().data().to_vec()
    }
    fn bf16_bytes_to_f32(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    }

    fn sigma(reference: &[f32]) -> f32 {
        let mean: f32 = reference.iter().sum::<f32>() / reference.len() as f32;
        (reference.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / reference.len() as f32).sqrt()
    }
    fn check(label: &str, got: &[f32], reference: &[f32], mean_gate: f32, max_gate: f32) {
        assert_eq!(got.len(), reference.len(), "{label} length mismatch");
        let sig = sigma(reference);
        let (mut sum_abs, mut max_abs) = (0.0f64, 0.0f32);
        for (&g, &r) in got.iter().zip(reference) {
            let d = (g - r).abs();
            sum_abs += d as f64;
            max_abs = max_abs.max(d);
        }
        let meand = (sum_abs / got.len() as f64) as f32;
        println!(
            "bookend {label}: sigma {sig:.4}, mean|d|/sig {:.4}, max|d|/sig {:.4}",
            meand / sig,
            max_abs / sig
        );
        assert!(
            meand / sig < mean_gate,
            "{label} mean dev {meand} / sig {sig} too large"
        );
        assert!(
            max_abs / sig < max_gate,
            "{label} max dev {max_abs} / sig {sig} exceeds floor"
        );
    }

    #[test]
    fn bookend_matches_oracle() {
        let Ok(ctx) = DeviceContext::new() else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        let model = model_path();
        if !model.join("model.safetensors.index.json").exists() {
            eprintln!("no checkpoint; skipping");
            return;
        }
        let probe = probe_dir();
        if !probe.join("logits.bin").exists() {
            eprintln!("no bookend probe bins; skipping");
            return;
        }
        let map = weight_map(&model);
        let stream = &ctx.stream;

        // ---- embed: exact gather against the raw checkpoint rows ----
        let embed_bytes = load_u8(&model, &map, "model.embed_tokens.weight");
        let embed =
            DeviceMatrix::from_safetensors(&ctx, &embed_bytes, GLM52_VOCAB, GLM52_HIDDEN).unwrap();
        let embed_f32 = bf16_bytes_to_f32(&embed_bytes);
        for &tok in &[0u32, 1, 7, 12345, GLM52_VOCAB as u32 - 1] {
            let mut tid = stream.alloc_zeros::<u32>(1).unwrap();
            stream.memcpy_htod(&[tok], &mut tid).unwrap();
            let got = glm52_embed(&ctx, &embed, &tid).unwrap();
            stream.synchronize().unwrap();
            let got: Vec<f32> = stream
                .clone_dtoh(&got.data)
                .unwrap()
                .iter()
                .map(|x| x.to_f32())
                .collect();
            let row = &embed_f32[tok as usize * GLM52_HIDDEN..(tok as usize + 1) * GLM52_HIDDEN];
            let max_d = got
                .iter()
                .zip(row)
                .map(|(&g, &r)| (g - r).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                max_d, 0.0,
                "embed token {tok} mismatch (gather must be exact)"
            );
        }
        println!("bookend embed: exact gather OK");

        // ---- norm + lm_head against the f32 oracle ----
        let norm_w =
            DeviceVec::from_safetensors(&ctx, &load_u8(&model, &map, "model.norm.weight")).unwrap();
        let lm_head_bytes = load_u8(&model, &map, "lm_head.weight");
        let lm_head =
            DeviceMatrix::from_safetensors(&ctx, &lm_head_bytes, GLM52_VOCAB, GLM52_HIDDEN)
                .unwrap();

        let hidden = read_f32(&probe, "hidden.bin");
        let norm_ref = read_f32(&probe, "final_norm_output.bin");
        let logits_ref = read_f32(&probe, "logits.bin");
        let tokens = hidden.len() / GLM52_HIDDEN;
        assert_eq!(tokens * GLM52_HIDDEN, hidden.len());

        let mut norm_got = Vec::with_capacity(tokens * GLM52_HIDDEN);
        let mut logits_got = Vec::with_capacity(tokens * GLM52_VOCAB);
        for t in 0..tokens {
            let row: Vec<bf16> = hidden[t * GLM52_HIDDEN..(t + 1) * GLM52_HIDDEN]
                .iter()
                .map(|&x| bf16::from_f32(x))
                .collect();
            let hd = DeviceVec::from_host(&ctx, &row).unwrap();
            let normed = glm52_final_norm(&ctx, &hd, &norm_w).unwrap();
            let logits = glm52_lm_head(&ctx, &normed, &lm_head).unwrap();
            stream.synchronize().unwrap();
            norm_got.extend(
                stream
                    .clone_dtoh(&normed.data)
                    .unwrap()
                    .iter()
                    .map(|x| x.to_f32()),
            );
            logits_got.extend(
                stream
                    .clone_dtoh(&logits.data)
                    .unwrap()
                    .iter()
                    .map(|x| x.to_f32()),
            );
        }

        check("norm", &norm_got, &norm_ref, 0.02, 0.20);
        check("lm_head", &logits_got, &logits_ref, 0.02, 0.20);

        // Greedy decode only consumes the argmax -- it must match per token.
        for t in 0..tokens {
            let g = &logits_got[t * GLM52_VOCAB..(t + 1) * GLM52_VOCAB];
            let r = &logits_ref[t * GLM52_VOCAB..(t + 1) * GLM52_VOCAB];
            let amax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            assert_eq!(amax(g), amax(r), "lm_head argmax mismatch at token {t}");
        }
        println!("bookend lm_head: argmax exact across {tokens} tokens");
    }
}
