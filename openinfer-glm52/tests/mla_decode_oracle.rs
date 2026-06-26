//! Back-half MLA decode oracle gate (H200 sm_90): the absorbed latent -> v_up ->
//! o_proj -> o path, validated against the HF MLA oracle. This is the other end of
//! the attention block from `glm52_flashmla_sparse_oracle.rs` (which proves
//! query/cache -> latent). Together they pin everything from the FlashMLA output
//! to the attention block output, leaving only the front-half projections + the
//! rope/cache-pack assembly.
//!
//! Two independent checks, each isolated against an oracle intermediate:
//!   v_up   : oracle latent[64,512] @ W_UV  ==  oracle attn[64,256], the attention
//!            OUTPUT (= weighted sum of v = latent @ W_UV) — NOT value_states, which
//!            is the per-token v (kv_c@W_UV). W_UV = host-dequant of kv_b_proj's
//!            v-part (fp8 e4m3 block-scaled -> bf16): the batched-GEMM absorption.
//!   o_proj : quant(oracle attn[16384]) -> TRTLLM fp8 linear == oracle o[6144].
//!            The TRTLLM blockscale runner wants the activation scale in col-major
//!            [k/128, round_up(m,4)] TMA layout, so the plain per-token-group scale
//!            is relaid via glm52_deepgemm_mn_major_tma_aligned_f32 before the GEMM.
//!
//! SM90-only (TRTLLM grouped fp8 is sm_90a); no-ops without a CUDA device, the
//! checkpoint, or the probe fixtures. Run on the build node:
//!   cargo test --release -p openinfer-glm52 --test mla_decode_oracle -- --nocapture

use half::bf16;
use memmap2::MmapOptions;
use openinfer_kernels::ops::{
    Glm52DeepGemmScaleLayout, Glm52MoeQuantShape, Glm52TrtllmFp8LinearContract,
    gemm_strided_batched_bf16, glm52_deepgemm_mn_major_tma_aligned_f32_launch,
    glm52_fp8_per_token_group_quant_bf16_launch, glm52_trtllm_fp8_linear_launch,
};
use openinfer_kernels::tensor::DeviceContext;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

const HEADS: usize = 64;
const KV_LORA: usize = 512;
const QK_NOPE: usize = 192;
const V_HEAD: usize = 256;
const KV_B_ROWS_PER_HEAD: usize = QK_NOPE + V_HEAD; // 448
const HIDDEN: usize = 6144;
const V_FLAT: usize = HEADS * V_HEAD; // 16384
const FP8_BLOCK: usize = 128;

fn model_path() -> PathBuf {
    std::env::var("GLM52_MODEL_PATH")
        .unwrap_or_else(|_| "/data/models/GLM-5.2-FP8".into())
        .into()
}
fn probe_dir() -> PathBuf {
    std::env::var("GLM52_FLASHMLA_PROBE_DIR")
        .unwrap_or_else(|_| "/data/models/glm52_mla_ref/flashmla_probe".into())
        .into()
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    bytes_to_f32(&std::fs::read(dir.join(name)).unwrap())
}

/// OCP `float8_e4m3fn` decode (bias 7, no inf; subnormals supported).
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if (b >> 7) & 1 == 1 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    let mag = if e == 0 {
        2f32.powi(-6) * (m / 8.0)
    } else {
        2f32.powi(e - 7) * (1.0 + m / 8.0)
    };
    sign * mag
}

fn read_weight_map(model: &Path) -> HashMap<String, String> {
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

/// Raw bytes + shape of a checkpoint tensor (mmap'd shard, no dtype conversion).
fn load_tensor(model: &Path, map: &HashMap<String, String>, name: &str) -> (Vec<u8>, Vec<usize>) {
    let shard = map
        .get(name)
        .unwrap_or_else(|| panic!("index missing {name}"));
    let file = File::open(model.join(shard)).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let st = safetensors::SafeTensors::deserialize(&mmap).unwrap();
    let view = st.tensor(name).unwrap();
    (view.data().to_vec(), view.shape().to_vec())
}

fn report(label: &str, got: &[f32], want: &[f32]) -> f32 {
    let sig = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let (mut maxd, mut sumd) = (0.0f32, 0.0f32);
    for (g, w) in got.iter().zip(want.iter()) {
        maxd = maxd.max((g - w).abs());
        sumd += (g - w).abs();
    }
    let mean = sumd / got.len() as f32;
    println!(
        "{label}: signal|max|={sig:.5} max|Δ|={maxd:.5} mean|Δ|={mean:.6} (rel max {:.3}%)",
        100.0 * maxd / sig
    );
    maxd / sig
}

#[test]
fn mla_back_half_matches_oracle() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let model = model_path();
    if !model.join("model.safetensors.index.json").exists() {
        eprintln!("no checkpoint at {model:?}; skipping");
        return;
    }
    let probe = probe_dir();
    if !probe.join("latent_expected.bin").exists() {
        eprintln!("no probe fixtures at {probe:?}; skipping");
        return;
    }
    let map = read_weight_map(&model);

    // ---- host-dequant kv_b v-part -> W_UV bf16 [64,256,512] (head-major) ----
    let (kvb, kvb_shape) = load_tensor(&model, &map, "model.layers.0.self_attn.kv_b_proj.weight");
    let (kvbs, _) = load_tensor(
        &model,
        &map,
        "model.layers.0.self_attn.kv_b_proj.weight_scale_inv",
    );
    assert_eq!(kvb_shape, vec![HEADS * KV_B_ROWS_PER_HEAD, KV_LORA]);
    let kvb_scale = bytes_to_f32(&kvbs);
    let scale_cols = KV_LORA / FP8_BLOCK; // 4
    let mut w_uv = vec![bf16::from_f32(0.0); HEADS * V_HEAD * KV_LORA];
    for h in 0..HEADS {
        for o in 0..V_HEAD {
            let row = h * KV_B_ROWS_PER_HEAD + QK_NOPE + o; // v-part row in kv_b
            for j in 0..KV_LORA {
                let s = kvb_scale[(row / FP8_BLOCK) * scale_cols + j / FP8_BLOCK];
                let val = e4m3_to_f32(kvb[row * KV_LORA + j]) * s;
                w_uv[(h * V_HEAD + o) * KV_LORA + j] = bf16::from_f32(val);
            }
        }
    }

    // ---- v_up: v[64,256] = W_UV @ latent, batched over heads ----
    let latent: Vec<bf16> = read_f32(&probe, "latent_expected.bin")
        .iter()
        .map(|&x| bf16::from_f32(x))
        .collect();
    assert_eq!(latent.len(), HEADS * KV_LORA);
    let mut w_uv_d = ctx.stream.alloc_zeros::<bf16>(w_uv.len()).unwrap();
    let mut latent_d = ctx.stream.alloc_zeros::<bf16>(latent.len()).unwrap();
    let mut v_d = ctx.stream.alloc_zeros::<bf16>(HEADS * V_HEAD).unwrap();
    ctx.stream.memcpy_htod(&w_uv, &mut w_uv_d).unwrap();
    ctx.stream.memcpy_htod(&latent, &mut latent_d).unwrap();
    gemm_strided_batched_bf16(
        &ctx,
        true,
        false,
        V_HEAD,
        1,
        KV_LORA, // transpose_a: W_UV row-major [m=256,k=512]
        &w_uv_d,
        KV_LORA,
        V_HEAD * KV_LORA,
        &latent_d,
        KV_LORA,
        KV_LORA,
        &mut v_d,
        V_HEAD,
        V_HEAD,
        HEADS,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let v_got: Vec<f32> = ctx
        .stream
        .clone_dtoh(&v_d)
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect();
    // v_up target is the attention OUTPUT (latent @ W_UV), head-major [16384].
    let attn_exp = read_f32(&probe, "attn_expected.bin");
    let rel_v = report("v_up   ", &v_got, &attn_exp);

    // ---- o_proj: quant(oracle attn) -> relayout act scale -> TRTLLM fp8 linear -> o ----
    let attn_in: Vec<bf16> = attn_exp.iter().map(|&x| bf16::from_f32(x)).collect();
    let mut attn_d = ctx.stream.alloc_zeros::<bf16>(attn_in.len()).unwrap();
    ctx.stream.memcpy_htod(&attn_in, &mut attn_d).unwrap();
    let qshape = Glm52MoeQuantShape {
        rows: 1,
        width: V_FLAT,
        group_size: FP8_BLOCK,
    };
    let mut v_fp8 = ctx.stream.alloc_zeros::<u8>(V_FLAT).unwrap();
    let mut v_scale_plain = ctx.stream.alloc_zeros::<f32>(V_FLAT / FP8_BLOCK).unwrap();
    glm52_fp8_per_token_group_quant_bf16_launch(
        &ctx,
        qshape,
        &attn_d,
        &mut v_fp8,
        &mut v_scale_plain,
    )
    .unwrap();
    // TRTLLM activation scale must be col-major [k/128, round_up(m,4)] (M-contiguous,
    // M padded to 4, pad rows zeroed) — the runner consumes it as-is, no relayout.
    let scale_layout = Glm52DeepGemmScaleLayout::f32(1, V_FLAT / FP8_BLOCK);
    let mut v_scale = ctx
        .stream
        .alloc_zeros::<f32>(scale_layout.output_len().unwrap())
        .unwrap();
    glm52_deepgemm_mn_major_tma_aligned_f32_launch(
        &ctx,
        scale_layout,
        &v_scale_plain,
        &mut v_scale,
    )
    .unwrap();

    let (oproj, oproj_shape) = load_tensor(&model, &map, "model.layers.0.self_attn.o_proj.weight");
    let (oprojs, _) = load_tensor(
        &model,
        &map,
        "model.layers.0.self_attn.o_proj.weight_scale_inv",
    );
    assert_eq!(oproj_shape, vec![HIDDEN, V_FLAT]);
    let mut oproj_d = ctx.stream.alloc_zeros::<u8>(oproj.len()).unwrap();
    let mut oproj_scale_d = ctx.stream.alloc_zeros::<u8>(oprojs.len()).unwrap();
    ctx.stream.memcpy_htod(&oproj, &mut oproj_d).unwrap();
    ctx.stream.memcpy_htod(&oprojs, &mut oproj_scale_d).unwrap();
    let contract = Glm52TrtllmFp8LinearContract {
        m: 1,
        n: HIDDEN,
        k: V_FLAT,
        weight_scale_rows: HIDDEN / FP8_BLOCK,
        weight_scale_cols: V_FLAT / FP8_BLOCK,
        activation_scale_cols: V_FLAT / FP8_BLOCK,
    };
    let mut o_d = ctx.stream.alloc_zeros::<bf16>(HIDDEN).unwrap();
    glm52_trtllm_fp8_linear_launch(
        &ctx,
        contract,
        &v_fp8,
        &v_scale,
        &oproj_d,
        &oproj_scale_d,
        &mut o_d,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let o_got: Vec<f32> = ctx
        .stream
        .clone_dtoh(&o_d)
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect();
    let o_exp = read_f32(&probe, "o_expected.bin");
    let rel_o = report("o_proj ", &o_got, &o_exp);

    assert!(v_got.iter().all(|x| x.is_finite()) && o_got.iter().all(|x| x.is_finite()));
    // fp8-weight dequant + bf16 GEMM vs the full-precision oracle: a few % is the
    // fp8 floor; a wrong dequant/orientation/scale blows past 30%.
    assert!(rel_v < 0.10, "v_up rel max {rel_v} too large");
    assert!(rel_o < 0.10, "o_proj rel max {rel_o} too large");
}
