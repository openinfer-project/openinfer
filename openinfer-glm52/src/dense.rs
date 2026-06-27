//! GLM5.2 dense-MLP decode forward for bs=1 (layers 0..first_k_dense_replace).
//!
//! The dense layers replace the MoE block with a plain fp8 SwiGLU MLP
//! `down(silu(gate(x)) * up(x))` -- the same shape as the MoE shared expert, only
//! the intermediate is wider (12288 vs 2048). It reuses the shared `fp8_mlp`
//! helper, so this module is just the weight bundle + a thin forward.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::tensor::DeviceContext;

use crate::fp8::{Glm52ProjBytes, ProjWeight, fp8_mlp};

const HIDDEN: usize = 6144;
const INTERMEDIATE: usize = 12288;

/// The three fp8 projections of one dense MLP layer, resident on device.
pub struct Glm52DenseMlpWeights {
    gate: ProjWeight, // fp8 [INTERMEDIATE, HIDDEN]
    up: ProjWeight,   // fp8 [INTERMEDIATE, HIDDEN]
    down: ProjWeight, // fp8 [HIDDEN, INTERMEDIATE]
}

impl Glm52DenseMlpWeights {
    /// Upload the dense MLP projections, validating every extent against the
    /// GLM5.2 dense-layer architecture (crash-early on a packaging drift). The
    /// `ProjWeight::upload` shape checks plus `fp8_mlp`'s internal cross-checks
    /// pin gate/up/down consistency.
    pub fn from_host(
        ctx: &DeviceContext,
        gate: &Glm52ProjBytes,
        up: &Glm52ProjBytes,
        down: &Glm52ProjBytes,
    ) -> Result<Self> {
        let shape = |label: &str, p: &Glm52ProjBytes, n: usize, k: usize| -> Result<()> {
            anyhow::ensure!(
                p.n == n && p.k == k,
                "GLM5.2 dense MLP {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        shape("gate_proj", gate, INTERMEDIATE, HIDDEN)?;
        shape("up_proj", up, INTERMEDIATE, HIDDEN)?;
        shape("down_proj", down, HIDDEN, INTERMEDIATE)?;
        Ok(Self {
            gate: ProjWeight::upload(ctx, gate)?,
            up: ProjWeight::upload(ctx, up)?,
            down: ProjWeight::upload(ctx, down)?,
        })
    }

    /// Build from already-resident projections (the production loader path),
    /// validating the dense-layer shapes against the moved-in `ProjWeight`s.
    pub fn from_device(gate: ProjWeight, up: ProjWeight, down: ProjWeight) -> Result<Self> {
        let shape = |label: &str, p: &ProjWeight, n: usize, k: usize| -> Result<()> {
            anyhow::ensure!(
                p.n == n && p.k == k,
                "GLM5.2 dense MLP {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        shape("gate_proj", &gate, INTERMEDIATE, HIDDEN)?;
        shape("up_proj", &up, INTERMEDIATE, HIDDEN)?;
        shape("down_proj", &down, HIDDEN, INTERMEDIATE)?;
        Ok(Self { gate, up, down })
    }
}

/// Dense MLP contribution for a single token. `normed_hidden` is the
/// post-attention-layernorm hidden `[HIDDEN]`; returns the MLP output `[HIDDEN]`
/// (the caller adds it to the post-attention residual).
pub fn glm52_dense_mlp_forward(
    ctx: &DeviceContext,
    weights: &Glm52DenseMlpWeights,
    normed_hidden: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    fp8_mlp(
        ctx,
        &weights.gate,
        &weights.up,
        &weights.down,
        normed_hidden,
    )
}

#[cfg(test)]
mod tests {
    //! Dense-MLP decode gate (H200 sm_90): drive the bs=1 dense MLP for each of the
    //! 8 seed-0 oracle tokens through the real fp8 projections and compare against
    //! the float32-dequant reference. Same sigma-normalized metric as the MoE gate:
    //! `mean|d|/sig` is the wiring invariant, the max is the loose fp8 floor.
    //!
    //! No-ops without a CUDA device, the checkpoint, or the dense probe bins:
    //!   cargo test --release -p openinfer-glm52 --features <model> dense_mlp -- --nocapture

    use super::*;
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
        std::env::var("GLM52_DENSE_PROBE_DIR")
            .unwrap_or_else(|_| "/data/models/glm52_mla_ref/dense_probe".into())
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

    /// Sigma-normalized abs-deviation gate of one output set vs its f32 oracle.
    fn check(label: &str, got: &[f32], reference: &[f32], mean_gate: f32, max_gate: f32) {
        assert_eq!(got.len(), reference.len(), "{label} length mismatch");
        let mean_ref: f32 = reference.iter().sum::<f32>() / reference.len() as f32;
        let sig = (reference
            .iter()
            .map(|&x| (x - mean_ref).powi(2))
            .sum::<f32>()
            / reference.len() as f32)
            .sqrt();
        let mut sum_abs = 0.0f64;
        let mut max_abs = 0.0f32;
        for (&g, &r) in got.iter().zip(reference) {
            let d = (g - r).abs();
            sum_abs += d as f64;
            max_abs = max_abs.max(d);
        }
        let meand = (sum_abs / got.len() as f64) as f32;
        println!(
            "dense {label}: sigma {sig:.4}, mean|d|/sig {:.4}, max|d|/sig {:.4}",
            meand / sig,
            max_abs / sig
        );
        assert!(
            meand / sig < mean_gate,
            "{label} mean dev {meand} vs sigma {sig} too large (wiring bug, not fp8 floor)"
        );
        assert!(
            max_abs / sig < max_gate,
            "{label} max dev {max_abs} vs sigma {sig} exceeds fp8 floor"
        );
    }

    #[test]
    fn dense_mlp_matches_oracle() {
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
        if !probe.join("dense_mlp_output.bin").exists() {
            eprintln!("no dense probe bins; skipping");
            return;
        }
        let map = weight_map(&model);
        let p = "model.layers.0.mlp.";
        let stream = &ctx.stream;

        let gate_w = load_u8(&model, &map, &format!("{p}gate_proj.weight"));
        let gate_s = load_u8(&model, &map, &format!("{p}gate_proj.weight_scale_inv"));
        let up_w = load_u8(&model, &map, &format!("{p}up_proj.weight"));
        let up_s = load_u8(&model, &map, &format!("{p}up_proj.weight_scale_inv"));
        let down_w = load_u8(&model, &map, &format!("{p}down_proj.weight"));
        let down_s = load_u8(&model, &map, &format!("{p}down_proj.weight_scale_inv"));

        let weights = Glm52DenseMlpWeights::from_host(
            &ctx,
            &Glm52ProjBytes {
                weight: &gate_w,
                scale: &gate_s,
                n: INTERMEDIATE,
                k: HIDDEN,
            },
            &Glm52ProjBytes {
                weight: &up_w,
                scale: &up_s,
                n: INTERMEDIATE,
                k: HIDDEN,
            },
            &Glm52ProjBytes {
                weight: &down_w,
                scale: &down_s,
                n: HIDDEN,
                k: INTERMEDIATE,
            },
        )
        .unwrap();

        let hidden = read_f32(&probe, "hidden.bin"); // [tokens, 6144] f32
        let out_ref = read_f32(&probe, "dense_mlp_output.bin");
        let tokens = hidden.len() / HIDDEN;
        assert_eq!(tokens * HIDDEN, hidden.len());

        let mut got = Vec::with_capacity(tokens * HIDDEN);
        for t in 0..tokens {
            let row: Vec<bf16> = hidden[t * HIDDEN..(t + 1) * HIDDEN]
                .iter()
                .map(|&x| bf16::from_f32(x))
                .collect();
            let mut hd = stream.alloc_zeros::<bf16>(HIDDEN).unwrap();
            stream.memcpy_htod(&row, &mut hd).unwrap();
            let out = glm52_dense_mlp_forward(&ctx, &weights, &hd).unwrap();
            stream.synchronize().unwrap();
            got.extend(stream.clone_dtoh(&out).unwrap().iter().map(|x| x.to_f32()));
        }

        // Measured fp8 floor on H200 sm_90 (vs the f32-dequant reference): a single
        // 3-linear fp8 SwiGLU MLP, same class as the MoE shared expert (mean ~0.028,
        // max ~0.25). A wiring bug (swapped gate/up, wrong intermediate) is tens of
        // percent. Max gate carries ~1.4x headroom for fp8 ULP wobble.
        check("mlp", &got, &out_ref, 0.04, 0.34);
    }
}
