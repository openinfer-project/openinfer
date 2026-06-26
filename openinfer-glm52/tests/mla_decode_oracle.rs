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
    glm52_fp8_per_token_group_quant_bf16_launch, glm52_mla_cache_pack_launch,
    glm52_mla_query_assemble_launch, glm52_trtllm_fp8_linear_launch, rms_norm_into,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceVec};
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

/// One GLM fp8 projection (bs=1): quant(input) -> relayout activation scale to the
/// TRTLLM col-major TMA layout -> blockscale linear. Host in/out (avoids naming
/// cudarc types). `input` is [k] bf16; returns [n] f32.
fn fp8_linear(
    ctx: &DeviceContext,
    model: &Path,
    map: &HashMap<String, String>,
    wname: &str,
    n: usize,
    k: usize,
    input: &[bf16],
) -> Vec<f32> {
    assert_eq!(input.len(), k);
    let (w, wsh) = load_tensor(model, map, &format!("{wname}.weight"));
    let (ws, _) = load_tensor(model, map, &format!("{wname}.weight_scale_inv"));
    assert_eq!(wsh, vec![n, k]);
    let scale_cols = k / FP8_BLOCK;

    let mut in_d = ctx.stream.alloc_zeros::<bf16>(k).unwrap();
    let mut w_d = ctx.stream.alloc_zeros::<u8>(w.len()).unwrap();
    let mut ws_d = ctx.stream.alloc_zeros::<u8>(ws.len()).unwrap();
    ctx.stream.memcpy_htod(input, &mut in_d).unwrap();
    ctx.stream.memcpy_htod(&w, &mut w_d).unwrap();
    ctx.stream.memcpy_htod(&ws, &mut ws_d).unwrap();

    let qshape = Glm52MoeQuantShape {
        rows: 1,
        width: k,
        group_size: FP8_BLOCK,
    };
    let mut a_fp8 = ctx.stream.alloc_zeros::<u8>(k).unwrap();
    let mut a_scale_plain = ctx.stream.alloc_zeros::<f32>(scale_cols).unwrap();
    glm52_fp8_per_token_group_quant_bf16_launch(ctx, qshape, &in_d, &mut a_fp8, &mut a_scale_plain)
        .unwrap();
    let layout = Glm52DeepGemmScaleLayout::f32(1, scale_cols);
    let mut a_scale = ctx
        .stream
        .alloc_zeros::<f32>(layout.output_len().unwrap())
        .unwrap();
    glm52_deepgemm_mn_major_tma_aligned_f32_launch(ctx, layout, &a_scale_plain, &mut a_scale)
        .unwrap();

    let contract = Glm52TrtllmFp8LinearContract {
        m: 1,
        n,
        k,
        weight_scale_rows: n.div_ceil(FP8_BLOCK),
        weight_scale_cols: scale_cols,
        activation_scale_cols: scale_cols,
    };
    let mut out = ctx.stream.alloc_zeros::<bf16>(n).unwrap();
    glm52_trtllm_fp8_linear_launch(ctx, contract, &a_fp8, &a_scale, &w_d, &ws_d, &mut out).unwrap();
    ctx.stream.synchronize().unwrap();
    ctx.stream
        .clone_dtoh(&out)
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect()
}

/// GLM RMSNorm (eps=1e-5) on the GPU via the shared op; host in/out. `weight_bytes`
/// is the raw bf16 layernorm gamma from the checkpoint.
fn rms_gpu(ctx: &DeviceContext, x: &[f32], weight_bytes: &[u8]) -> Vec<f32> {
    let x_bf: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_dv = DeviceVec::from_host(ctx, &x_bf).unwrap();
    let w_dv = DeviceVec::from_safetensors(ctx, weight_bytes).unwrap();
    let mut out_dv = DeviceVec::zeros(ctx, x.len()).unwrap();
    rms_norm_into(ctx, &x_dv, &w_dv, 1.0e-5, &mut out_dv).unwrap();
    out_dv.to_host(ctx).unwrap()
}

/// Absorb back-projection: ql_nope[64,512] = q_pass @ W_UK, W_UK = kv_b nope-part
/// [:,:192,:] host-dequant. `q_full` is the q_b output [64,256] head-major; q_pass
/// is the first 192 of each head (read via stride 256 in the batched GEMM).
fn absorb_ql_nope(
    ctx: &DeviceContext,
    model: &Path,
    map: &HashMap<String, String>,
    q_full: &[f32],
) -> Vec<f32> {
    let (kvb, _) = load_tensor(model, map, "model.layers.0.self_attn.kv_b_proj.weight");
    let (kvbs, _) = load_tensor(
        model,
        map,
        "model.layers.0.self_attn.kv_b_proj.weight_scale_inv",
    );
    let kvb_scale = bytes_to_f32(&kvbs);
    let scale_cols = KV_LORA / FP8_BLOCK;
    let mut w_uk = vec![bf16::from_f32(0.0); HEADS * QK_NOPE * KV_LORA];
    for h in 0..HEADS {
        for p in 0..QK_NOPE {
            let row = h * KV_B_ROWS_PER_HEAD + p; // nope-part row
            for j in 0..KV_LORA {
                let s = kvb_scale[(row / FP8_BLOCK) * scale_cols + j / FP8_BLOCK];
                w_uk[(h * QK_NOPE + p) * KV_LORA + j] =
                    bf16::from_f32(e4m3_to_f32(kvb[row * KV_LORA + j]) * s);
            }
        }
    }
    let q_full_bf: Vec<bf16> = q_full.iter().map(|&x| bf16::from_f32(x)).collect();
    let mut wuk_d = ctx.stream.alloc_zeros::<bf16>(w_uk.len()).unwrap();
    let mut q_d = ctx.stream.alloc_zeros::<bf16>(q_full_bf.len()).unwrap();
    let mut ql_d = ctx.stream.alloc_zeros::<bf16>(HEADS * KV_LORA).unwrap();
    ctx.stream.memcpy_htod(&w_uk, &mut wuk_d).unwrap();
    ctx.stream.memcpy_htod(&q_full_bf, &mut q_d).unwrap();
    // ql_nope[h,l] = Σ_p W_UK[h,p,l] * q_pass[h,p]; no transpose, m=512,n=1,k=192.
    // A=W_UK row-major [64,192,512] (lda=512, stride 192*512); B=q_full at head
    // stride 256, k=192 reads only q_pass; C=ql[64,512].
    gemm_strided_batched_bf16(
        ctx,
        false,
        false,
        KV_LORA,
        1,
        QK_NOPE,
        &wuk_d,
        KV_LORA,
        QK_NOPE * KV_LORA,
        &q_d,
        QK_NOPE,
        V_HEAD,
        &mut ql_d,
        KV_LORA,
        KV_LORA,
        HEADS,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    ctx.stream
        .clone_dtoh(&ql_d)
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect()
}

#[test]
fn mla_front_half_matches_oracle() {
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
    if !probe.join("hidden_input.bin").exists() {
        eprintln!("no front-half fixtures at {probe:?}; skipping");
        return;
    }
    let map = read_weight_map(&model);
    let p = "model.layers.0.self_attn.";

    let hidden: Vec<bf16> = read_f32(&probe, "hidden_input.bin")
        .iter()
        .map(|&x| bf16::from_f32(x))
        .collect();
    assert_eq!(hidden.len(), HIDDEN);

    // q path: q_a_proj -> q_a_layernorm -> q_b_proj -> q_pass
    let q_a = fp8_linear(
        &ctx,
        &model,
        &map,
        &format!("{p}q_a_proj"),
        2048,
        HIDDEN,
        &hidden,
    );
    let q_a_ln = load_tensor(&model, &map, &format!("{p}q_a_layernorm.weight")).0;
    let q_resid = rms_gpu(&ctx, &q_a, &q_a_ln);
    let r_qr = report(
        "q_resid",
        &q_resid,
        &read_f32(&probe, "q_resid_expected.bin"),
    );
    let q_resid_bf: Vec<bf16> = q_resid.iter().map(|&x| bf16::from_f32(x)).collect();
    let q_full = fp8_linear(
        &ctx,
        &model,
        &map,
        &format!("{p}q_b_proj"),
        16384,
        2048,
        &q_resid_bf,
    );
    let mut q_pass = vec![0f32; HEADS * QK_NOPE];
    for h in 0..HEADS {
        for i in 0..QK_NOPE {
            q_pass[h * QK_NOPE + i] = q_full[h * V_HEAD + i];
        }
    }
    let r_qp = report("q_pass ", &q_pass, &read_f32(&probe, "q_pass_expected.bin"));

    // kv path: kv_a_proj_with_mqa -> split -> kv_a_layernorm
    let ckv = fp8_linear(
        &ctx,
        &model,
        &map,
        &format!("{p}kv_a_proj_with_mqa"),
        576,
        HIDDEN,
        &hidden,
    );
    let kv_a_ln = load_tensor(&model, &map, &format!("{p}kv_a_layernorm.weight")).0;
    let kv_c = rms_gpu(&ctx, &ckv[..KV_LORA], &kv_a_ln);
    let r_kv = report("kv_c   ", &kv_c, &read_f32(&probe, "kv_c_expected.bin"));

    // absorb: q_pass @ W_UK -> ql_nope
    let ql_nope = absorb_ql_nope(&ctx, &model, &map, &q_full);
    let r_ql = report(
        "ql_nope",
        &ql_nope,
        &read_f32(&probe, "ql_nope_expected.bin"),
    );

    assert!(ql_nope.iter().all(|x| x.is_finite()) && kv_c.iter().all(|x| x.is_finite()));
    // fp8-activation + fp8-weight projections vs the full-precision oracle: a couple
    // % is the fp8 floor; a wrong layout/scale/orientation blows past 30%.
    assert!(r_qr < 0.05, "q_resid rel max {r_qr}");
    assert!(r_qp < 0.05, "q_pass rel max {r_qp}");
    assert!(r_kv < 0.08, "kv_c rel max {r_kv}");
    assert!(r_ql < 0.05, "ql_nope rel max {r_ql}");
}

const ROPE_DIM: usize = 64;
const QUERY_DIM: usize = KV_LORA + ROPE_DIM; // 576
const CACHE_BYTES: usize = 656;

fn read_bf16_raw(dir: &Path, name: &str) -> Vec<bf16> {
    std::fs::read(dir.join(name))
        .unwrap()
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Dequant one fp8_ds_mla 656-byte token to (ckv[512] f32, k_pe[64] f32). Plain
/// f32 group scale (both the produced and reference tokens store f32), so the
/// comparison reflects the ckv values regardless of the kernel's later bf16
/// scale down-cast.
fn dequant_656(tok: &[u8]) -> (Vec<f32>, Vec<f32>) {
    let scales = bytes_to_f32(&tok[KV_LORA..KV_LORA + 16]); // [4]
    let ckv = (0..KV_LORA)
        .map(|i| e4m3_to_f32(tok[i]) * scales[i / FP8_BLOCK])
        .collect();
    let kpe = tok[528..CACHE_BYTES]
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    (ckv, kpe)
}

#[test]
fn mla_assemble_matches_oracle() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let probe = probe_dir();
    if !probe.join("q_pe_input.bin").exists() {
        eprintln!("no assembly fixtures at {probe:?}; skipping");
        return;
    }
    let cos = read_bf16_raw(&probe, "cos32.bin");
    let sin = read_bf16_raw(&probe, "sin32.bin");
    let mut cos_d = ctx.stream.alloc_zeros::<bf16>(cos.len()).unwrap();
    let mut sin_d = ctx.stream.alloc_zeros::<bf16>(sin.len()).unwrap();
    ctx.stream.memcpy_htod(&cos, &mut cos_d).unwrap();
    ctx.stream.memcpy_htod(&sin, &mut sin_d).unwrap();

    // ---- query assemble: [ql_nope | rope(q_pe)] -> [64,576] vs query.bin ----
    let ql_bf: Vec<bf16> = read_f32(&probe, "ql_nope_expected.bin")
        .iter()
        .map(|&x| bf16::from_f32(x))
        .collect();
    let q_pe = read_bf16_raw(&probe, "q_pe_input.bin");
    let mut ql_d = ctx.stream.alloc_zeros::<bf16>(ql_bf.len()).unwrap();
    let mut qpe_d = ctx.stream.alloc_zeros::<bf16>(q_pe.len()).unwrap();
    let mut query_d = ctx.stream.alloc_zeros::<bf16>(HEADS * QUERY_DIM).unwrap();
    ctx.stream.memcpy_htod(&ql_bf, &mut ql_d).unwrap();
    ctx.stream.memcpy_htod(&q_pe, &mut qpe_d).unwrap();
    glm52_mla_query_assemble_launch(
        &ctx,
        &ql_d,
        &qpe_d,
        0,
        ROPE_DIM,
        &cos_d,
        &sin_d,
        &mut query_d,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let query_got: Vec<f32> = ctx
        .stream
        .clone_dtoh(&query_d)
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect();
    let query_want: Vec<f32> = read_bf16_raw(&probe, "query.bin")
        .iter()
        .map(|x| x.to_f32())
        .collect();
    let rel_q = report("query  ", &query_got, &query_want);

    // ---- cache pack: quant(kv_c) + rope(k_pe) -> 656 token vs cache.bin[T=7] ----
    let kv_c_bf: Vec<bf16> = read_f32(&probe, "kv_c_expected.bin")
        .iter()
        .map(|&x| bf16::from_f32(x))
        .collect();
    let mut kvc_d = ctx.stream.alloc_zeros::<bf16>(kv_c_bf.len()).unwrap();
    ctx.stream.memcpy_htod(&kv_c_bf, &mut kvc_d).unwrap();
    let qshape = Glm52MoeQuantShape {
        rows: 1,
        width: KV_LORA,
        group_size: FP8_BLOCK,
    };
    let mut ckv_fp8 = ctx.stream.alloc_zeros::<u8>(KV_LORA).unwrap();
    let mut ckv_scales = ctx.stream.alloc_zeros::<f32>(KV_LORA / FP8_BLOCK).unwrap();
    glm52_fp8_per_token_group_quant_bf16_launch(
        &ctx,
        qshape,
        &kvc_d,
        &mut ckv_fp8,
        &mut ckv_scales,
    )
    .unwrap();
    let k_pe = read_bf16_raw(&probe, "k_pe_input.bin");
    let mut kpe_d = ctx.stream.alloc_zeros::<bf16>(k_pe.len()).unwrap();
    ctx.stream.memcpy_htod(&k_pe, &mut kpe_d).unwrap();
    let mut token_d = ctx.stream.alloc_zeros::<u8>(CACHE_BYTES).unwrap();
    glm52_mla_cache_pack_launch(
        &ctx,
        &ckv_fp8,
        &ckv_scales,
        &kpe_d,
        &cos_d,
        &sin_d,
        &mut token_d,
        0,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let token = ctx.stream.clone_dtoh(&token_d).unwrap();
    let cache_ref = std::fs::read(probe.join("cache.bin")).unwrap();
    let (ckv_got, kpe_got) = dequant_656(&token);
    let (ckv_ref, kpe_ref) = dequant_656(&cache_ref[7 * CACHE_BYTES..8 * CACHE_BYTES]);
    let rel_ckv = report("cache ckv", &ckv_got, &ckv_ref);
    let rel_kpe = report("cache kpe", &kpe_got, &kpe_ref);

    assert!(query_got.iter().all(|x| x.is_finite()));
    // query nope half is an exact bf16 copy; the rope half round-trips q_rot
    // through the inverse rotation + bf16 twice, so ~1 bf16 ULP. cache kpe is the
    // GPU rope of the pre-rope key. Both are tight.
    assert!(rel_q < 0.02, "query rel max {rel_q}");
    assert!(rel_kpe < 0.02, "cache kpe rel max {rel_kpe}");
    // cache ckv is fp8(bf16 kv_c) vs the numpy fp8(f32 kv_c) reference. On this
    // 0.025-magnitude tensor a single small element's e4m3 step is ~10%, so the
    // worst element lands ~3.5% while the bulk stays at the bf16+fp8 floor. The
    // MEAN is the scale/layout invariant (a wrong scale or layout blows it, the
    // query, and the kpe together); the max is bounded loosely as the fp8 floor.
    let mean_ckv = ckv_got
        .iter()
        .zip(&ckv_ref)
        .map(|(g, r)| (g - r).abs())
        .sum::<f32>()
        / ckv_got.len() as f32;
    let sig_ckv = ckv_ref.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(
        mean_ckv / sig_ckv < 0.005,
        "cache ckv mean rel {}",
        mean_ckv / sig_ckv
    );
    assert!(rel_ckv < 0.05, "cache ckv rel max {rel_ckv}");
}
