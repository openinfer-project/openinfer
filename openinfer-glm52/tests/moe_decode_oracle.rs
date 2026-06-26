//! GLM5.2 MoE decode oracle gates (H200 sm_90): validate the MoE FFN block ops
//! against the HF `GlmMoeDsaMoE` oracle (layer 3, the first sparse layer). Built
//! brick-by-brick like the MLA gates; the first brick is the router.
//!
//!   router : glm52_router_noaux_tc (route_scale=2.5) vs the oracle's top-8
//!            selection + normalized weights. Selection runs on sigmoid(gate@h)+
//!            e_score_correction_bias; the weights are the UNBIASED sigmoid probs
//!            at the selected ids, normalized to 1, then x2.5 (so each token's 8
//!            weights sum to 2.5). HF emits ids unsorted -> compare as a set and
//!            match weights by expert id.
//!
//! No-ops without a CUDA device, the checkpoint, or the MoE probe bins (built by
//! tools/glm52/moe_oracle_prep.py -> moe_probe_prep.py). Run on the build node:
//!   cargo test --release -p openinfer-glm52 --test moe_decode_oracle -- --nocapture

use half::bf16;
use memmap2::MmapOptions;
use openinfer_kernels::ops::{
    GLM52_ROUTED_RESIDUAL_SCALE, Glm52RouterBatch, Glm52RouterConfig, Glm52RouterOutput,
    glm52_router_noaux_tc_launch,
};
use openinfer_kernels::tensor::DeviceContext;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

const HIDDEN: usize = 6144;
const EXPERTS: usize = 256;
const TOPK: usize = 8;
const TOKENS: usize = 8; // the oracle batch

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
fn read_i32(dir: &Path, name: &str) -> Vec<i32> {
    std::fs::read(dir.join(name))
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
/// Raw bytes of a checkpoint tensor (mmap'd shard, no dtype conversion).
fn load_tensor(model: &Path, map: &HashMap<String, String>, name: &str) -> Vec<u8> {
    let file = File::open(model.join(map.get(name).unwrap())).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let st = safetensors::SafeTensors::deserialize(&mmap).unwrap();
    st.tensor(name).unwrap().data().to_vec()
}

#[test]
fn moe_router_matches_oracle() {
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
    if !probe.join("topk_indices.bin").exists() {
        eprintln!("no MoE probe bins; skipping");
        return;
    }
    let map = weight_map(&model);
    let gate = load_tensor(&model, &map, "model.layers.3.mlp.gate.weight"); // bf16 [256,6144]
    let bias = load_tensor(
        &model,
        &map,
        "model.layers.3.mlp.gate.e_score_correction_bias",
    ); // f32 [256]
    assert_eq!(gate.len(), EXPERTS * HIDDEN * 2);
    assert_eq!(bias.len(), EXPERTS * 4);

    // hidden [1,8,6144] f32 -> bf16 [8,6144]
    let hidden_h: Vec<bf16> = read_f32(&probe, "hidden.bin")
        .iter()
        .map(|&x| bf16::from_f32(x))
        .collect();
    assert_eq!(hidden_h.len(), TOKENS * HIDDEN);
    let oracle_idx = read_i32(&probe, "topk_indices.bin"); // [8,8]
    let oracle_w = read_f32(&probe, "topk_weights.bin"); // [8,8]

    let mut gate_d = ctx.stream.alloc_zeros::<u8>(gate.len()).unwrap();
    let mut bias_d = ctx.stream.alloc_zeros::<u8>(bias.len()).unwrap();
    let mut hidden_d = ctx.stream.alloc_zeros::<bf16>(hidden_h.len()).unwrap();
    let mut logits = ctx.stream.alloc_zeros::<f32>(TOKENS * EXPERTS).unwrap();
    let mut topk_w = ctx.stream.alloc_zeros::<f32>(TOKENS * TOPK).unwrap();
    let mut topk_i = ctx.stream.alloc_zeros::<i32>(TOKENS * TOPK).unwrap();
    ctx.stream.memcpy_htod(&gate, &mut gate_d).unwrap();
    ctx.stream.memcpy_htod(&bias, &mut bias_d).unwrap();
    ctx.stream.memcpy_htod(&hidden_h, &mut hidden_d).unwrap();

    let config = Glm52RouterConfig {
        route_scale: GLM52_ROUTED_RESIDUAL_SCALE,
        ..Glm52RouterConfig::glm52()
    };
    let batch = Glm52RouterBatch {
        active_tokens: TOKENS,
        padded_tokens: TOKENS,
    };
    let mut output = Glm52RouterOutput {
        topk_weight: &mut topk_w,
        topk_idx: &mut topk_i,
    };
    glm52_router_noaux_tc_launch(
        &ctx,
        config,
        batch,
        &hidden_d,
        &gate_d,
        &bias_d,
        &mut logits,
        &mut output,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let got_idx: Vec<i32> = ctx.stream.clone_dtoh(&topk_i).unwrap();
    let got_w: Vec<f32> = ctx.stream.clone_dtoh(&topk_w).unwrap();

    // Per token: ids must match as a SET; weights match by expert id; sum == 2.5.
    let mut worst_w = 0.0f32;
    for t in 0..TOKENS {
        let g_ids: HashSet<i32> = got_idx[t * TOPK..t * TOPK + TOPK].iter().copied().collect();
        let o_ids: HashSet<i32> = oracle_idx[t * TOPK..t * TOPK + TOPK]
            .iter()
            .copied()
            .collect();
        assert_eq!(
            g_ids, o_ids,
            "token {t} expert set mismatch: got {g_ids:?} want {o_ids:?}"
        );

        let g_map: HashMap<i32, f32> = (0..TOPK)
            .map(|j| (got_idx[t * TOPK + j], got_w[t * TOPK + j]))
            .collect();
        let mut sum = 0.0f32;
        for j in 0..TOPK {
            let id = oracle_idx[t * TOPK + j];
            let ow = oracle_w[t * TOPK + j];
            let gw = g_map[&id];
            worst_w = worst_w.max((gw - ow).abs() / ow.abs());
            sum += gw;
        }
        assert!(
            (sum - 2.5).abs() < 1e-3,
            "token {t} weight sum {sum} != 2.5"
        );
    }
    println!(
        "MoE router: {TOKENS} tokens, expert sets exact, worst weight rel {:.2e}",
        worst_w
    );
    assert!(worst_w < 1e-3, "router weight rel {worst_w} too large");
}
