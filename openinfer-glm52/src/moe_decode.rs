//! GLM5.2 single-layer routed-MoE decode forward for bs=1 (EP1: all 256 routed
//! experts resident on this PP stage; no DeepEP all-to-all).
//!
//! Data flow (one token, normed hidden -> routed contribution):
//!
//!   router (noaux_tc, route_scale=2.5) -> topk_idx[8], topk_weight[8]
//!   quant normed hidden -> fp8 row + per-group scale
//!   route_offsets -> grouped expert_offsets[E+1]   (bs=1: each expert <= 1 row)
//!   scatter        -> replicate the fp8 row into each selected expert's slot,
//!                     write the expert-major route weight
//!   grouped W13 (gate|up) -> weighted SwiGLU quant (folds route weight) ->
//!     grouped W2 (down)
//!   combine        -> sum the selected experts' rows back into routed[H]
//!
//! The route weight is folded into the W2 input by the weighted SwiGLU quant, so
//! combine is a plain sum. The shared expert and the routed+shared add live in a
//! separate brick. Buffers are allocated per call (wire-first; the CUDA-graph
//! arena conversion is a later slice).

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT, GLM52_ROUTED_RESIDUAL_SCALE, Glm52MoeQuantShape,
    Glm52RouterBatch, Glm52RouterConfig, Glm52RouterOutput, Glm52TrtllmGroupedFp8Contract,
    Glm52TrtllmGroupedFp8Kind, Glm52TrtllmGroupedOffsetScaleLayout, add_batch,
    glm52_deepgemm_grouped_offset_tma_aligned_f32_launch,
    glm52_fp8_per_token_group_quant_bf16_launch, glm52_moe_combine_launch,
    glm52_moe_route_offsets_launch, glm52_moe_route_scatter_launch, glm52_router_noaux_tc_launch,
    glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch, glm52_trtllm_grouped_fp8_launch,
};
use openinfer_kernels::tensor::{DeviceContext, HiddenStates};

use crate::fp8::{Glm52ProjBytes, ProjWeight, fp8_mlp};

const HIDDEN: usize = 6144;
const EXPERTS: usize = 256;
const TOPK: usize = 8;
const INTERMEDIATE: usize = 2048;
const QUANT_GROUP: usize = 128;

const W13_N: usize = 2 * INTERMEDIATE; // 4096 (gate|up)
const W13_K: usize = HIDDEN; // 6144
const W2_N: usize = HIDDEN; // 6144
const W2_K: usize = INTERMEDIATE; // 2048

const HIDDEN_SCALE_COLS: usize = HIDDEN / QUANT_GROUP; // 48
const W13_SCALE_ROWS: usize = W13_N / QUANT_GROUP; // 32
const W2_SCALE_COLS: usize = W2_K / QUANT_GROUP; // 16
const W2_SCALE_ROWS: usize = W2_N / QUANT_GROUP; // 48

/// bs=1 expert-major row capacity: each of the top-k distinct experts owns at
/// most one row, padded to the 64-row alignment, so `TOPK * ALIGNMENT` is a tight
/// upper bound on the spanned rows (route_offsets emits `expert_offsets[E] <=` it).
const M_CAPACITY: usize = TOPK * GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT; // 512

/// All weights for one MoE layer: the 256 routed experts as expert-major grouped
/// FP8 (the layout the grouped GEMM consumes directly), plus the single shared
/// expert (a plain fp8 MLP). Built once on device; borrowed by every decode step.
pub struct Glm52MoeLayerWeights {
    gate_weight: CudaSlice<u8>,  // bf16 [EXPERTS, HIDDEN]
    e_score_bias: CudaSlice<u8>, // f32  [EXPERTS]
    w13_weight: CudaSlice<u8>,   // fp8  [EXPERTS, W13_N, W13_K]
    w13_scale: CudaSlice<f32>,   // f32  [EXPERTS, W13_SCALE_ROWS, HIDDEN_SCALE_COLS]
    w2_weight: CudaSlice<u8>,    // fp8  [EXPERTS, W2_N, W2_K]
    w2_scale: CudaSlice<f32>,    // f32  [EXPERTS, W2_SCALE_ROWS, W2_SCALE_COLS]
    shared_gate: ProjWeight,     // fp8  [INTERMEDIATE, HIDDEN]
    shared_up: ProjWeight,       // fp8  [INTERMEDIATE, HIDDEN]
    shared_down: ProjWeight,     // fp8  [HIDDEN, INTERMEDIATE]
}

impl Glm52MoeLayerWeights {
    /// Wrap the pre-assembled grouped routed buffers + upload the shared expert
    /// projections from host bytes (the oracle/test path), validating every extent
    /// against the GLM5.2 MoE architecture (crash-early on a packaging drift).
    #[allow(clippy::too_many_arguments)]
    pub fn from_device(
        ctx: &DeviceContext,
        gate_weight: CudaSlice<u8>,
        e_score_bias: CudaSlice<u8>,
        w13_weight: CudaSlice<u8>,
        w13_scale: CudaSlice<f32>,
        w2_weight: CudaSlice<u8>,
        w2_scale: CudaSlice<f32>,
        shared_gate: &Glm52ProjBytes,
        shared_up: &Glm52ProjBytes,
        shared_down: &Glm52ProjBytes,
    ) -> Result<Self> {
        Self::new(
            gate_weight,
            e_score_bias,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            ProjWeight::upload(ctx, shared_gate)?,
            ProjWeight::upload(ctx, shared_up)?,
            ProjWeight::upload(ctx, shared_down)?,
        )
    }

    /// Wrap already-resident buffers (the production loader path): the grouped
    /// routed buffers come from the expert packager, the shared expert from the
    /// non-expert resident map. Everything is moved in, no copy.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_resident(
        gate_weight: CudaSlice<u8>,
        e_score_bias: CudaSlice<u8>,
        w13_weight: CudaSlice<u8>,
        w13_scale: CudaSlice<f32>,
        w2_weight: CudaSlice<u8>,
        w2_scale: CudaSlice<f32>,
        shared_gate: ProjWeight,
        shared_up: ProjWeight,
        shared_down: ProjWeight,
    ) -> Result<Self> {
        Self::new(
            gate_weight,
            e_score_bias,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            shared_gate,
            shared_up,
            shared_down,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        gate_weight: CudaSlice<u8>,
        e_score_bias: CudaSlice<u8>,
        w13_weight: CudaSlice<u8>,
        w13_scale: CudaSlice<f32>,
        w2_weight: CudaSlice<u8>,
        w2_scale: CudaSlice<f32>,
        shared_gate: ProjWeight,
        shared_up: ProjWeight,
        shared_down: ProjWeight,
    ) -> Result<Self> {
        let check = |name: &str, have: usize, want: usize| -> Result<()> {
            ensure!(
                have == want,
                "GLM5.2 MoE layer weight {name} length {have} != expected {want}"
            );
            Ok(())
        };
        check("gate_weight", gate_weight.len(), EXPERTS * HIDDEN * 2)?;
        check("e_score_bias", e_score_bias.len(), EXPERTS * 4)?;
        check("w13_weight", w13_weight.len(), EXPERTS * W13_N * W13_K)?;
        check(
            "w13_scale",
            w13_scale.len(),
            EXPERTS * W13_SCALE_ROWS * HIDDEN_SCALE_COLS,
        )?;
        check("w2_weight", w2_weight.len(), EXPERTS * W2_N * W2_K)?;
        check(
            "w2_scale",
            w2_scale.len(),
            EXPERTS * W2_SCALE_ROWS * W2_SCALE_COLS,
        )?;
        let shape = |label: &str, p: &ProjWeight, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 MoE shared {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        shape("gate_proj", &shared_gate, INTERMEDIATE, HIDDEN)?;
        shape("up_proj", &shared_up, INTERMEDIATE, HIDDEN)?;
        shape("down_proj", &shared_down, HIDDEN, INTERMEDIATE)?;
        Ok(Self {
            gate_weight,
            e_score_bias,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            shared_gate,
            shared_up,
            shared_down,
        })
    }
}

/// Run the routed-MoE contribution for a single token. `normed_hidden` is the
/// post-attention-layernorm hidden `[HIDDEN]`; returns the routed output `[HIDDEN]`
/// (route weight + routed_scaling already folded in). The shared expert is added
/// by the caller.
pub fn glm52_moe_routed_forward(
    ctx: &DeviceContext,
    weights: &Glm52MoeLayerWeights,
    normed_hidden: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    ensure!(
        normed_hidden.len() >= HIDDEN,
        "GLM5.2 MoE routed forward hidden too small: have {}, need {HIDDEN}",
        normed_hidden.len()
    );
    let stream = &ctx.stream;

    // 1. router: select top-8 experts + normalized x2.5 weights.
    let mut logits = stream.alloc_zeros::<f32>(EXPERTS)?;
    let mut topk_idx = stream.alloc_zeros::<i32>(TOPK)?;
    let mut topk_weight = stream.alloc_zeros::<f32>(TOPK)?;
    let router_cfg = Glm52RouterConfig {
        route_scale: GLM52_ROUTED_RESIDUAL_SCALE,
        ..Glm52RouterConfig::glm52()
    };
    let mut router_out = Glm52RouterOutput {
        topk_weight: &mut topk_weight,
        topk_idx: &mut topk_idx,
    };
    glm52_router_noaux_tc_launch(
        ctx,
        router_cfg,
        Glm52RouterBatch {
            active_tokens: 1,
            padded_tokens: 1,
        },
        normed_hidden,
        &weights.gate_weight,
        &weights.e_score_bias,
        &mut logits,
        &mut router_out,
    )?;

    // 2. quantize the single hidden row -> fp8 + per-group scale.
    let mut hidden_fp8 = stream.alloc_zeros::<u8>(HIDDEN)?;
    let mut hidden_scale = stream.alloc_zeros::<f32>(HIDDEN_SCALE_COLS)?;
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: 1,
            width: HIDDEN,
            group_size: QUANT_GROUP,
        },
        normed_hidden,
        &mut hidden_fp8,
        &mut hidden_scale,
    )?;

    // 3. grouped expert_offsets directly from the top-k ids.
    let mut expert_offsets = stream.alloc_zeros::<i64>(EXPERTS + 1)?;
    glm52_moe_route_offsets_launch(
        ctx,
        EXPERTS,
        TOPK,
        GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT,
        &topk_idx,
        &mut expert_offsets,
    )?;

    // 4. scatter the fp8 row + per-row route weight into the expert-major slots.
    //    Pad rows stay zero (alloc_zeros), so they contribute nothing.
    let mut w13_act = stream.alloc_zeros::<u8>(M_CAPACITY * W13_K)?;
    let mut w13_act_scale = stream.alloc_zeros::<f32>(M_CAPACITY * HIDDEN_SCALE_COLS)?;
    let mut row_weight = stream.alloc_zeros::<f32>(M_CAPACITY)?;
    glm52_moe_route_scatter_launch(
        ctx,
        M_CAPACITY,
        TOPK,
        W13_K,
        HIDDEN_SCALE_COLS,
        &hidden_fp8,
        &hidden_scale,
        &topk_idx,
        &topk_weight,
        &expert_offsets,
        &mut w13_act,
        &mut w13_act_scale,
        &mut row_weight,
    )?;

    // 5. W13 grouped FP8 GEMM (gate|up): relayout the activation scale, then GEMM.
    let w13_out = grouped_gemm(
        ctx,
        Glm52TrtllmGroupedFp8Kind::W13,
        W13_N,
        W13_K,
        HIDDEN_SCALE_COLS,
        W13_SCALE_ROWS,
        &w13_act,
        &w13_act_scale,
        &weights.w13_weight,
        &weights.w13_scale,
        &expert_offsets,
    )?;

    // 6. weighted SwiGLU quant: silu(gate)*up*route_weight -> fp8 W2 input.
    let mut w2_act = stream.alloc_zeros::<u8>(M_CAPACITY * W2_K)?;
    let mut w2_act_scale = stream.alloc_zeros::<f32>(M_CAPACITY * W2_SCALE_COLS)?;
    glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: M_CAPACITY,
            width: W2_K,
            group_size: QUANT_GROUP,
        },
        &w13_out,
        &row_weight,
        &mut w2_act,
        &mut w2_act_scale,
    )?;

    // 7. W2 grouped FP8 GEMM (down).
    let w2_out = grouped_gemm(
        ctx,
        Glm52TrtllmGroupedFp8Kind::W2,
        W2_N,
        W2_K,
        W2_SCALE_COLS,
        W2_SCALE_ROWS,
        &w2_act,
        &w2_act_scale,
        &weights.w2_weight,
        &weights.w2_scale,
        &expert_offsets,
    )?;

    // 8. combine: sum the selected experts' rows -> routed[HIDDEN].
    let mut routed = stream.alloc_zeros::<bf16>(HIDDEN)?;
    glm52_moe_combine_launch(
        ctx,
        M_CAPACITY,
        HIDDEN,
        TOPK,
        &w2_out,
        &topk_idx,
        &expert_offsets,
        &mut routed,
    )?;
    Ok(routed)
}

/// Shared-expert contribution for a single token: a plain fp8 SwiGLU MLP
/// (intermediate 2048). Returns `[HIDDEN]` bf16.
fn glm52_moe_shared_forward(
    ctx: &DeviceContext,
    weights: &Glm52MoeLayerWeights,
    normed_hidden: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    fp8_mlp(
        ctx,
        &weights.shared_gate,
        &weights.shared_up,
        &weights.shared_down,
        normed_hidden,
    )
}

/// Full MoE contribution for a single token: routed experts + shared expert. The
/// caller adds this to the post-attention residual. Returns `[HIDDEN]` bf16.
pub fn glm52_moe_forward(
    ctx: &DeviceContext,
    weights: &Glm52MoeLayerWeights,
    normed_hidden: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    let routed = glm52_moe_routed_forward(ctx, weights, normed_hidden)?;
    let shared = glm52_moe_shared_forward(ctx, weights, normed_hidden)?;
    let routed_hs = HiddenStates {
        data: routed,
        hidden_dim: HIDDEN,
        seq_len: 1,
    };
    let shared_hs = HiddenStates {
        data: shared,
        hidden_dim: HIDDEN,
        seq_len: 1,
    };
    Ok(add_batch(ctx, &routed_hs, &shared_hs)?.data)
}

/// Relayout the plain per-row activation scale into the offset-major TMA layout,
/// then run one grouped FP8 GEMM. Returns the bf16 output `[M_CAPACITY, n]`.
#[allow(clippy::too_many_arguments)]
fn grouped_gemm(
    ctx: &DeviceContext,
    kind: Glm52TrtllmGroupedFp8Kind,
    n: usize,
    k: usize,
    scale_cols: usize,
    weight_scale_rows: usize,
    activation: &CudaSlice<u8>,
    activation_scale_plain: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    expert_offsets: &CudaSlice<i64>,
) -> Result<CudaSlice<bf16>> {
    let stream = &ctx.stream;
    let scale_layout = Glm52TrtllmGroupedOffsetScaleLayout::f32(M_CAPACITY, scale_cols, EXPERTS);
    let mut activation_scale_tma = stream.alloc_zeros::<f32>(scale_layout.output_len()?)?;
    glm52_deepgemm_grouped_offset_tma_aligned_f32_launch(
        ctx,
        scale_layout,
        activation_scale_plain,
        expert_offsets,
        &mut activation_scale_tma,
    )?;

    let contract = Glm52TrtllmGroupedFp8Contract {
        groups: EXPERTS,
        m_capacity: M_CAPACITY,
        n,
        k,
        weight_scale_rows,
        weight_scale_cols: scale_cols,
        activation_scale_cols: scale_cols,
        activation_scale_trtllm_rows: scale_layout.padded_rows,
    };
    let mut out = stream.alloc_zeros::<bf16>(M_CAPACITY * n)?;
    glm52_trtllm_grouped_fp8_launch(
        ctx,
        kind,
        contract,
        activation,
        &activation_scale_tma,
        weight,
        weight_scale,
        expert_offsets,
        &mut out,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! Routed-MoE decode gate (H200 sm_90): drive the bs=1 routed forward for each
    //! of the 8 oracle tokens through the real grouped FP8 expert GEMMs and compare
    //! against the HF `routed_output` (the combined, scaled routed contribution
    //! before the shared expert). The metric is sigma-normalized absolute deviation
    //! (robust to near-zero hidden dims): `meand/sig` is the wiring invariant, the
    //! max is the loose fp8 floor.
    //!
    //! No-ops without a CUDA device, the checkpoint, or the MoE probe bins. Run on
    //! the build node:
    //!   cargo test --release -p openinfer-glm52 --features <model> moe_routed -- --nocapture

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
        std::env::var("GLM52_MOE_PROBE_DIR")
            .unwrap_or_else(|_| "/data/models/glm52_mla_ref/moe_probe".into())
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
    fn load_f32(model: &Path, map: &HashMap<String, String>, name: &str) -> Vec<f32> {
        load_u8(model, map, name)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
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
            "MoE {label}: sigma {sig:.4}, mean|d|/sig {:.4}, max|d|/sig {:.4}",
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
    fn moe_forward_matches_oracle() {
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
        if !probe.join("routed_output.bin").exists() {
            eprintln!("no MoE probe bins; skipping");
            return;
        }
        let map = weight_map(&model);
        let p = "model.layers.3.mlp.";
        let stream = &ctx.stream;

        // Assemble the expert-major grouped FP8 weights from the 256 per-expert
        // checkpoint tensors (mirrors the packager: W13 = [gate; up] rows, scales
        // concatenated likewise; W2 = down). Independent of the production loader.
        let mut w13_host: Vec<u8> = Vec::with_capacity(EXPERTS * W13_N * W13_K);
        let mut w13_scale_host: Vec<f32> =
            Vec::with_capacity(EXPERTS * W13_SCALE_ROWS * HIDDEN_SCALE_COLS);
        let mut w2_host: Vec<u8> = Vec::with_capacity(EXPERTS * W2_N * W2_K);
        let mut w2_scale_host: Vec<f32> =
            Vec::with_capacity(EXPERTS * W2_SCALE_ROWS * W2_SCALE_COLS);
        for e in 0..EXPERTS {
            let ep = format!("{p}experts.{e}.");
            let gw = load_u8(&model, &map, &format!("{ep}gate_proj.weight"));
            let uw = load_u8(&model, &map, &format!("{ep}up_proj.weight"));
            assert_eq!(gw.len(), INTERMEDIATE * W13_K);
            assert_eq!(uw.len(), INTERMEDIATE * W13_K);
            w13_host.extend_from_slice(&gw);
            w13_host.extend_from_slice(&uw);
            let gs = load_f32(&model, &map, &format!("{ep}gate_proj.weight_scale_inv"));
            let us = load_f32(&model, &map, &format!("{ep}up_proj.weight_scale_inv"));
            assert_eq!(gs.len(), (INTERMEDIATE / QUANT_GROUP) * HIDDEN_SCALE_COLS);
            w13_scale_host.extend_from_slice(&gs);
            w13_scale_host.extend_from_slice(&us);
            let dw = load_u8(&model, &map, &format!("{ep}down_proj.weight"));
            assert_eq!(dw.len(), W2_N * W2_K);
            w2_host.extend_from_slice(&dw);
            let ds = load_f32(&model, &map, &format!("{ep}down_proj.weight_scale_inv"));
            assert_eq!(ds.len(), W2_SCALE_ROWS * W2_SCALE_COLS);
            w2_scale_host.extend_from_slice(&ds);
        }
        let gate = load_u8(&model, &map, &format!("{p}gate.weight"));
        let bias = load_u8(&model, &map, &format!("{p}gate.e_score_correction_bias"));

        // Shared expert: plain fp8 gate/up/down projections (intermediate 2048).
        let sp = format!("{p}shared_experts.");
        let sgw = load_u8(&model, &map, &format!("{sp}gate_proj.weight"));
        let sgs = load_u8(&model, &map, &format!("{sp}gate_proj.weight_scale_inv"));
        let suw = load_u8(&model, &map, &format!("{sp}up_proj.weight"));
        let sus = load_u8(&model, &map, &format!("{sp}up_proj.weight_scale_inv"));
        let sdw = load_u8(&model, &map, &format!("{sp}down_proj.weight"));
        let sds = load_u8(&model, &map, &format!("{sp}down_proj.weight_scale_inv"));

        let to_dev_u8 = |h: &[u8]| {
            let mut d = stream.alloc_zeros::<u8>(h.len()).unwrap();
            stream.memcpy_htod(h, &mut d).unwrap();
            d
        };
        let to_dev_f32 = |h: &[f32]| {
            let mut d = stream.alloc_zeros::<f32>(h.len()).unwrap();
            stream.memcpy_htod(h, &mut d).unwrap();
            d
        };
        let weights = Glm52MoeLayerWeights::from_device(
            &ctx,
            to_dev_u8(&gate),
            to_dev_u8(&bias),
            to_dev_u8(&w13_host),
            to_dev_f32(&w13_scale_host),
            to_dev_u8(&w2_host),
            to_dev_f32(&w2_scale_host),
            &Glm52ProjBytes {
                weight: &sgw,
                scale: &sgs,
                n: INTERMEDIATE,
                k: HIDDEN,
            },
            &Glm52ProjBytes {
                weight: &suw,
                scale: &sus,
                n: INTERMEDIATE,
                k: HIDDEN,
            },
            &Glm52ProjBytes {
                weight: &sdw,
                scale: &sds,
                n: HIDDEN,
                k: INTERMEDIATE,
            },
        )
        .unwrap();

        let hidden = read_f32(&probe, "hidden.bin"); // [tokens, 6144] f32
        let routed_ref = read_f32(&probe, "routed_output.bin");
        let shared_ref = read_f32(&probe, "shared_output.bin");
        let moe_ref = read_f32(&probe, "moe_output.bin");
        let tokens = hidden.len() / HIDDEN;
        assert_eq!(tokens * HIDDEN, hidden.len());

        // Drive every token through routed / shared / full (the public forward), and
        // collect the bf16 outputs flat for a global sigma-normalized comparison.
        let mut routed_got = Vec::with_capacity(tokens * HIDDEN);
        let mut shared_got = Vec::with_capacity(tokens * HIDDEN);
        let mut full_got = Vec::with_capacity(tokens * HIDDEN);
        for t in 0..tokens {
            let row: Vec<bf16> = hidden[t * HIDDEN..(t + 1) * HIDDEN]
                .iter()
                .map(|&x| bf16::from_f32(x))
                .collect();
            let mut hd = stream.alloc_zeros::<bf16>(HIDDEN).unwrap();
            stream.memcpy_htod(&row, &mut hd).unwrap();
            let routed = glm52_moe_routed_forward(&ctx, &weights, &hd).unwrap();
            let shared = glm52_moe_shared_forward(&ctx, &weights, &hd).unwrap();
            let full = glm52_moe_forward(&ctx, &weights, &hd).unwrap();
            stream.synchronize().unwrap();
            let pull = |s: &CudaSlice<bf16>| -> Vec<f32> {
                stream
                    .clone_dtoh(s)
                    .unwrap()
                    .iter()
                    .map(|x| x.to_f32())
                    .collect()
            };
            routed_got.extend(pull(&routed));
            shared_got.extend(pull(&shared));
            full_got.extend(pull(&full));
        }

        // Measured fp8 floors on H200 sm_90 (vs the f32-dequant HF oracle):
        //   routed  mean 0.026  max 0.176  (two grouped fp8 GEMMs + weighted re-quant + 8-way sum)
        //   shared  mean 0.028  max 0.248  (a single 3-linear fp8 MLP -- less averaging, fatter max tail)
        //   full    mean 0.026  max 0.213  (routed + shared)
        // A wiring bug (wrong expert, dropped x2.5, swapped gate/up, missing shared)
        // would be tens of percent. Max gates carry ~1.4x headroom for fp8 ULP wobble.
        check("routed", &routed_got, &routed_ref, 0.04, 0.25);
        check("shared", &shared_got, &shared_ref, 0.04, 0.34);
        check("full", &full_got, &moe_ref, 0.04, 0.30);
    }
}
