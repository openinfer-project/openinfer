//! Single-layer GLM5.2 MLA decode forward (bs=1): `hidden[6144] -> o[6144]`.
//!
//! Composes the oracle-validated GPU ops into one callable forward — the
//! attention half of a PP decode stage. The pieces are each gated against the HF
//! MLA oracle in `tests/mla_decode_oracle.rs` (front projections, the rope/query/
//! cache-pack assembly, FlashMLA sparse decode, the back-half v_up/o_proj); this
//! module wires them with no new math.
//!
//! Weights are taken as raw fp8 bytes (`from_host`) and uploaded once — the module
//! is loader-agnostic (functional core). kv_b is pre-dequantized into the bf16
//! absorb factors W_UK / W_UV at construction; the fp8 projection weights stay
//! as-loaded and every projection relays its activation scale into the TRTLLM
//! col-major TMA layout before the blockscale linear (the documented footgun).

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;

use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE, Glm52FlashMlaSparseDecode, Glm52MoeQuantShape,
    gemm_strided_batched_bf16, glm52_flashmla_sparse_decode_launch,
    glm52_flashmla_sparse_decode_metadata_launch, glm52_fp8_per_token_group_quant_bf16_launch,
    glm52_mla_cache_pack_launch, glm52_mla_query_assemble_launch, rms_norm_into,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceVec};

use crate::fp8::{FP8_BLOCK, Glm52ProjBytes, ProjWeight, bytes_to_f32, e4m3_to_f32, fp8_linear};

const HEADS: usize = 64;
const HIDDEN: usize = 6144;
const Q_LORA: usize = 2048;
const QK_NOPE: usize = 192; // absorbed q nope width per head
const Q_HEAD: usize = 256; // qk_nope(192) + qk_rope(64)
const ROPE_DIM: usize = 64;
const KV_LORA: usize = 512;
const KV_A_OUT: usize = 576; // compressed_kv(512) + k_pe(64)
const V_HEAD: usize = 256;
const KV_B_ROWS_PER_HEAD: usize = QK_NOPE + V_HEAD; // 448
const QUERY_DIM: usize = KV_LORA + ROPE_DIM; // 576
const RMS_EPS: f32 = 1.0e-5;

/// One MLA layer's attention weights, device-resident.
pub(crate) struct Glm52MlaLayerWeights {
    q_a: ProjWeight,
    q_a_ln: DeviceVec,
    q_b: ProjWeight,
    kv_a: ProjWeight,
    kv_a_ln: DeviceVec,
    o_proj: ProjWeight,
    w_uk: CudaSlice<bf16>, // [H, 192, 512]
    w_uv: CudaSlice<bf16>, // [H, 256, 512]
}

impl Glm52MlaLayerWeights {
    /// Build from raw checkpoint bytes: upload the fp8 projections + bf16
    /// layernorm gammas, and host-dequant kv_b into the bf16 absorb factors
    /// W_UK = kv_b[:, :192, :], W_UV = kv_b[:, 192:, :].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_host(
        ctx: &DeviceContext,
        q_a: &Glm52ProjBytes,
        q_a_ln: &[u8],
        q_b: &Glm52ProjBytes,
        kv_a: &Glm52ProjBytes,
        kv_a_ln: &[u8],
        kv_b: &Glm52ProjBytes,
        o_proj: &Glm52ProjBytes,
    ) -> Result<Self> {
        // Pin every projection to the MLA architecture at load time: a checkpoint
        // with the wrong shape would otherwise sail through the self-consistent
        // `upload` len check and only die deep in the forward (a GPU slice panic).
        let check = |label: &str, p: &Glm52ProjBytes, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        check("q_a_proj", q_a, Q_LORA, HIDDEN)?;
        check("q_b_proj", q_b, HEADS * Q_HEAD, Q_LORA)?;
        check("kv_a_proj_with_mqa", kv_a, KV_A_OUT, HIDDEN)?;
        check("kv_b_proj", kv_b, HEADS * KV_B_ROWS_PER_HEAD, KV_LORA)?;
        check("o_proj", o_proj, HIDDEN, HEADS * V_HEAD)?;
        let (w_uk, w_uv) = dequant_kv_b(ctx, kv_b)?;
        Ok(Self {
            q_a: ProjWeight::upload(ctx, q_a)?,
            q_a_ln: DeviceVec::from_safetensors(ctx, q_a_ln)?,
            q_b: ProjWeight::upload(ctx, q_b)?,
            kv_a: ProjWeight::upload(ctx, kv_a)?,
            kv_a_ln: DeviceVec::from_safetensors(ctx, kv_a_ln)?,
            o_proj: ProjWeight::upload(ctx, o_proj)?,
            w_uk,
            w_uv,
        })
    }

    /// Build from already-resident weights (the production loader path). The fp8
    /// projections + layernorm gammas are moved in; `kv_b` is consumed to derive
    /// the bf16 absorb factors (its fp8 bytes are pulled back to host once for the
    /// block-scaled dequant, then dropped — it is not stored). Same architecture
    /// shape checks as `from_host`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_device(
        ctx: &DeviceContext,
        q_a: ProjWeight,
        q_a_ln: DeviceVec,
        q_b: ProjWeight,
        kv_a: ProjWeight,
        kv_a_ln: DeviceVec,
        kv_b: ProjWeight,
        o_proj: ProjWeight,
    ) -> Result<Self> {
        let check = |label: &str, p: &ProjWeight, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        check("q_a_proj", &q_a, Q_LORA, HIDDEN)?;
        check("q_b_proj", &q_b, HEADS * Q_HEAD, Q_LORA)?;
        check("kv_a_proj_with_mqa", &kv_a, KV_A_OUT, HIDDEN)?;
        check("kv_b_proj", &kv_b, HEADS * KV_B_ROWS_PER_HEAD, KV_LORA)?;
        check("o_proj", &o_proj, HIDDEN, HEADS * V_HEAD)?;
        ensure!(
            q_a_ln.len == Q_LORA && kv_a_ln.len == KV_LORA,
            "GLM5.2 MLA layernorm lengths q_a_ln {} / kv_a_ln {} != {Q_LORA}/{KV_LORA}",
            q_a_ln.len,
            kv_a_ln.len
        );
        let kv_b_weight = ctx.stream.clone_dtoh(&kv_b.weight)?;
        let kv_b_scale = ctx.stream.clone_dtoh(&kv_b.scale)?;
        let (w_uk, w_uv) = dequant_kv_b(
            ctx,
            &Glm52ProjBytes {
                weight: &kv_b_weight,
                scale: &kv_b_scale,
                n: kv_b.n,
                k: kv_b.k,
            },
        )?;
        Ok(Self {
            q_a,
            q_a_ln,
            q_b,
            kv_a,
            kv_a_ln,
            o_proj,
            w_uk,
            w_uv,
        })
    }
}

/// Host-dequant kv_b (fp8 e4m3 block-scaled) into bf16 W_UK [H,192,512] (nope) and
/// W_UV [H,256,512] (v) absorb factors, head-major, uploaded to device.
fn dequant_kv_b(
    ctx: &DeviceContext,
    kv_b: &Glm52ProjBytes,
) -> Result<(CudaSlice<bf16>, CudaSlice<bf16>)> {
    // kv_b is indexed raw below (it does not pass through ProjWeight::upload), so
    // self-defend its byte lengths here — a truncated blob must error, not panic.
    ensure!(
        kv_b.weight.len() == kv_b.n * kv_b.k,
        "GLM5.2 kv_b weight bytes {} != n*k {}",
        kv_b.weight.len(),
        kv_b.n * kv_b.k
    );
    ensure!(
        kv_b.scale.len() == kv_b.n.div_ceil(FP8_BLOCK) * kv_b.k.div_ceil(FP8_BLOCK) * 4,
        "GLM5.2 kv_b scale bytes {} unexpected for [{},{}]",
        kv_b.scale.len(),
        kv_b.n,
        kv_b.k
    );
    let scale_cols = KV_LORA / FP8_BLOCK;
    let scale = bytes_to_f32(kv_b.scale);
    let mut w_uk = vec![bf16::from_f32(0.0); HEADS * QK_NOPE * KV_LORA];
    let mut w_uv = vec![bf16::from_f32(0.0); HEADS * V_HEAD * KV_LORA];
    for h in 0..HEADS {
        for r in 0..KV_B_ROWS_PER_HEAD {
            let row = h * KV_B_ROWS_PER_HEAD + r;
            for j in 0..KV_LORA {
                let s = scale[(row / FP8_BLOCK) * scale_cols + j / FP8_BLOCK];
                let val = bf16::from_f32(e4m3_to_f32(kv_b.weight[row * KV_LORA + j]) * s);
                if r < QK_NOPE {
                    w_uk[(h * QK_NOPE + r) * KV_LORA + j] = val;
                } else {
                    w_uv[(h * V_HEAD + (r - QK_NOPE)) * KV_LORA + j] = val;
                }
            }
        }
    }
    let mut uk = ctx.stream.alloc_zeros::<bf16>(w_uk.len())?;
    let mut uv = ctx.stream.alloc_zeros::<bf16>(w_uv.len())?;
    ctx.stream.memcpy_htod(&w_uk, &mut uk)?;
    ctx.stream.memcpy_htod(&w_uv, &mut uv)?;
    Ok((uk, uv))
}

/// RMSNorm (eps 1e-5) of `input[len]` into a fresh buffer.
fn rms(
    ctx: &DeviceContext,
    input: CudaSlice<bf16>,
    len: usize,
    weight: &DeviceVec,
) -> Result<CudaSlice<bf16>> {
    let x = DeviceVec { data: input, len };
    let mut out = DeviceVec::zeros(ctx, len)?;
    rms_norm_into(ctx, &x, weight, RMS_EPS, &mut out)?;
    Ok(out.data)
}

fn slice_copy(
    ctx: &DeviceContext,
    src: &CudaSlice<bf16>,
    start: usize,
    len: usize,
) -> Result<CudaSlice<bf16>> {
    let mut dst = ctx.stream.alloc_zeros::<bf16>(len)?;
    ctx.stream
        .memcpy_dtod(&src.slice(start..start + len), &mut dst)?;
    Ok(dst)
}

/// MLA decode forward for one token (bs=1): runs the projections, assembles the
/// FlashMLA query, writes the new token into the paged cache at `position`,
/// attends over the cached context, and projects back to `o[6144]`.
///
/// `cache` is the fp8_ds_mla paged cache (656 bytes/token); `cos`/`sin` are the
/// position's rotary table first half (`[32]`); `topk` is the (fixed-2048,
/// -1-padded) sparse index list; `contract` carries the FlashMLA launch sizing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm52_mla_decode_forward(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &CudaSlice<bf16>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    position: usize,
    topk: &CudaSlice<i32>,
    contract: Glm52FlashMlaSparseDecode,
) -> Result<CudaSlice<bf16>> {
    ensure!(hidden.len() >= HIDDEN, "GLM5.2 MLA hidden too small");
    // The new token is written to cache slot `position`; the FlashMLA paging then
    // attends over `num_blocks` pages of `PAGE_SIZE` tokens. Couple them so a
    // position past the paged window errors here, not as a silent cache stomp.
    ensure!(
        position < contract.num_blocks * GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
        "GLM5.2 MLA position {position} outside paged cache ({} blocks x {GLM52_FLASHMLA_SPARSE_PAGE_SIZE})",
        contract.num_blocks
    );

    // ---- front projections ----
    let q_a = fp8_linear(ctx, &w.q_a, hidden)?; // [2048]
    let q_resid = rms(ctx, q_a, Q_LORA, &w.q_a_ln)?; // [2048]
    let q_full = fp8_linear(ctx, &w.q_b, &q_resid)?; // [16384] = [64,256]
    let ckv = fp8_linear(ctx, &w.kv_a, hidden)?; // [576]
    debug_assert!(ckv.len() >= KV_A_OUT);
    let compressed_kv = slice_copy(ctx, &ckv, 0, KV_LORA)?; // [512]
    let kv_c = rms(ctx, compressed_kv, KV_LORA, &w.kv_a_ln)?; // [512]
    let k_pe = slice_copy(ctx, &ckv, KV_LORA, ROPE_DIM)?; // [64] pre-rope

    // ---- absorb: ql_nope[64,512] = q_pass @ W_UK ----
    let mut ql_nope = ctx.stream.alloc_zeros::<bf16>(HEADS * KV_LORA)?;
    gemm_strided_batched_bf16(
        ctx,
        false,
        false,
        KV_LORA,
        1,
        QK_NOPE,
        &w.w_uk,
        KV_LORA,
        QK_NOPE * KV_LORA,
        &q_full,
        QK_NOPE,
        Q_HEAD,
        &mut ql_nope,
        KV_LORA,
        KV_LORA,
        HEADS,
    )?;

    // ---- assemble query [64,576] = [ql_nope | rope(q_pe)] (q_pe in q_full @192) ----
    let mut query = ctx.stream.alloc_zeros::<bf16>(HEADS * QUERY_DIM)?;
    glm52_mla_query_assemble_launch(
        ctx, &ql_nope, &q_full, QK_NOPE, Q_HEAD, cos, sin, &mut query,
    )?;

    // ---- pack the new token into the cache: quant(kv_c) + rope(k_pe) ----
    let mut ckv_fp8 = ctx.stream.alloc_zeros::<u8>(KV_LORA)?;
    let mut ckv_scales = ctx.stream.alloc_zeros::<f32>(KV_LORA / FP8_BLOCK)?;
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: 1,
            width: KV_LORA,
            group_size: FP8_BLOCK,
        },
        &kv_c,
        &mut ckv_fp8,
        &mut ckv_scales,
    )?;
    glm52_mla_cache_pack_launch(ctx, &ckv_fp8, &ckv_scales, &k_pe, cos, sin, cache, position)?;

    // ---- FlashMLA sparse decode -> latent[64,512] ----
    let mut sched = ctx
        .stream
        .alloc_zeros::<i32>(contract.tile_scheduler_metadata_len())?;
    let mut splits = ctx.stream.alloc_zeros::<i32>(contract.num_splits_len())?;
    glm52_flashmla_sparse_decode_metadata_launch(
        ctx,
        contract.batch_size,
        contract.num_sm_parts,
        None,
        &mut sched,
        &mut splits,
    )?;
    let mut latent = ctx.stream.alloc_zeros::<bf16>(contract.latent_len())?;
    let mut lse = ctx.stream.alloc_zeros::<f32>(contract.lse_len())?;
    let mut lse_accum = ctx.stream.alloc_zeros::<f32>(contract.lse_accum_len())?;
    let mut o_accum = ctx.stream.alloc_zeros::<f32>(contract.o_accum_len())?;
    glm52_flashmla_sparse_decode_launch(
        ctx,
        contract,
        &query,
        cache,
        topk,
        None,
        &sched,
        &splits,
        &mut latent,
        &mut lse,
        &mut lse_accum,
        &mut o_accum,
    )?;

    // ---- back: v[64,256] = latent @ W_UV, then o_proj ----
    let mut v = ctx.stream.alloc_zeros::<bf16>(HEADS * V_HEAD)?;
    gemm_strided_batched_bf16(
        ctx,
        true,
        false,
        V_HEAD,
        1,
        KV_LORA,
        &w.w_uv,
        KV_LORA,
        V_HEAD * KV_LORA,
        &latent,
        KV_LORA,
        KV_LORA,
        &mut v,
        V_HEAD,
        V_HEAD,
        HEADS,
    )?;
    let o = fp8_linear(ctx, &w.o_proj, &v)?; // [6144]
    Ok(o)
}

#[cfg(test)]
mod tests {
    //! Full single-layer MLA decode forward vs the HF oracle (H200 sm_90). Loads
    //! layer-0 weights + the validated 8-token fp8 cache, decodes token 7, and
    //! compares o[6144] end-to-end. No-ops without a CUDA device, the checkpoint,
    //! or the probe fixtures.
    //!   cargo test --release -p openinfer-glm52 --lib mla_decode_forward -- --nocapture
    use super::*;
    use openinfer_kernels::ops::glm52_flashmla_sparse_decode_num_sm_parts;
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::{Path, PathBuf};

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
    fn weight_map(model: &Path) -> HashMap<String, String> {
        let idx: serde_json::Value = serde_json::from_str(
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
    fn load_tensor(
        model: &Path,
        map: &HashMap<String, String>,
        name: &str,
    ) -> (Vec<u8>, Vec<usize>) {
        let file = File::open(model.join(map.get(name).unwrap())).unwrap();
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let st = safetensors::SafeTensors::deserialize(&mmap).unwrap();
        let v = st.tensor(name).unwrap();
        (v.data().to_vec(), v.shape().to_vec())
    }
    fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
        std::fs::read(dir.join(name))
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn read_bf16(dir: &Path, name: &str) -> Vec<bf16> {
        std::fs::read(dir.join(name))
            .unwrap()
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    #[test]
    fn mla_decode_forward_matches_oracle() {
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
        if !probe.join("hidden_input.bin").exists() {
            eprintln!("no probe fixtures; skipping");
            return;
        }
        let map = weight_map(&model);
        let p = "model.layers.0.self_attn.";
        let proj = |stem: &str| {
            let (w, sh) = load_tensor(&model, &map, &format!("{p}{stem}.weight"));
            let (s, _) = load_tensor(&model, &map, &format!("{p}{stem}.weight_scale_inv"));
            (w, s, sh)
        };
        let (q_a_w, q_a_s, q_a_sh) = proj("q_a_proj");
        let (q_b_w, q_b_s, q_b_sh) = proj("q_b_proj");
        let (kv_a_w, kv_a_s, kv_a_sh) = proj("kv_a_proj_with_mqa");
        let (kv_b_w, kv_b_s, kv_b_sh) = proj("kv_b_proj");
        let (o_w, o_s, o_sh) = proj("o_proj");
        let q_a_ln = load_tensor(&model, &map, &format!("{p}q_a_layernorm.weight")).0;
        let kv_a_ln = load_tensor(&model, &map, &format!("{p}kv_a_layernorm.weight")).0;
        let weights = Glm52MlaLayerWeights::from_host(
            &ctx,
            &Glm52ProjBytes {
                weight: &q_a_w,
                scale: &q_a_s,
                n: q_a_sh[0],
                k: q_a_sh[1],
            },
            &q_a_ln,
            &Glm52ProjBytes {
                weight: &q_b_w,
                scale: &q_b_s,
                n: q_b_sh[0],
                k: q_b_sh[1],
            },
            &Glm52ProjBytes {
                weight: &kv_a_w,
                scale: &kv_a_s,
                n: kv_a_sh[0],
                k: kv_a_sh[1],
            },
            &kv_a_ln,
            &Glm52ProjBytes {
                weight: &kv_b_w,
                scale: &kv_b_s,
                n: kv_b_sh[0],
                k: kv_b_sh[1],
            },
            &Glm52ProjBytes {
                weight: &o_w,
                scale: &o_s,
                n: o_sh[0],
                k: o_sh[1],
            },
        )
        .unwrap();

        // 8-token validated fp8 cache; decode token 7 (forward repacks slot 7).
        let cache_host = std::fs::read(probe.join("cache.bin")).unwrap();
        let mut cache = ctx.stream.alloc_zeros::<u8>(cache_host.len()).unwrap();
        ctx.stream.memcpy_htod(&cache_host, &mut cache).unwrap();

        let hidden_h: Vec<bf16> = read_f32(&probe, "hidden_input.bin")
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let cos_h = read_bf16(&probe, "cos32.bin");
        let sin_h = read_bf16(&probe, "sin32.bin");
        let topk_h: Vec<i32> = std::fs::read(probe.join("topk.bin"))
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut hidden = ctx.stream.alloc_zeros::<bf16>(hidden_h.len()).unwrap();
        let mut cos = ctx.stream.alloc_zeros::<bf16>(cos_h.len()).unwrap();
        let mut sin = ctx.stream.alloc_zeros::<bf16>(sin_h.len()).unwrap();
        let mut topk = ctx.stream.alloc_zeros::<i32>(topk_h.len()).unwrap();
        ctx.stream.memcpy_htod(&hidden_h, &mut hidden).unwrap();
        ctx.stream.memcpy_htod(&cos_h, &mut cos).unwrap();
        ctx.stream.memcpy_htod(&sin_h, &mut sin).unwrap();
        ctx.stream.memcpy_htod(&topk_h, &mut topk).unwrap();

        let num_sm_parts =
            glm52_flashmla_sparse_decode_num_sm_parts().expect("num_sm_parts (SM90 required)");
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: 1,
            num_blocks: 1,
            topk: 2048,
            num_sm_parts,
            sm_scale: 0.0625,
        };
        let o = glm52_mla_decode_forward(
            &ctx, &weights, &hidden, &cos, &sin, &mut cache, 7, &topk, contract,
        )
        .unwrap();
        ctx.stream.synchronize().unwrap();
        let o_got: Vec<f32> = ctx
            .stream
            .clone_dtoh(&o)
            .unwrap()
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let o_exp = read_f32(&probe, "o_expected.bin");
        let sig = o_exp.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let maxd = o_got
            .iter()
            .zip(&o_exp)
            .fold(0.0f32, |m, (g, e)| m.max((g - e).abs()));
        let meand = o_got
            .iter()
            .zip(&o_exp)
            .map(|(g, e)| (g - e).abs())
            .sum::<f32>()
            / o_got.len() as f32;
        println!(
            "MLA forward o[6144]: signal {sig:.5} max|Δ| {maxd:.5} ({:.2}%) mean|Δ| {meand:.6} ({:.3}%)",
            100.0 * maxd / sig,
            100.0 * meand / sig
        );
        assert!(o_got.iter().all(|x| x.is_finite()), "o non-finite");
        // End-to-end: hidden -> fp8 projections -> FlashMLA -> back, vs the
        // full-precision oracle. The fp8 model genuinely deviates from fp32 here:
        // the front fp8 projections alone sit at ~3% and that propagates through
        // the attention into a worst-element ~6% on this 0.006-magnitude output.
        // The MEAN (~0.28%) is the wiring invariant — a wrong layout/orientation/
        // scale blows the mean, not just a tail element — so it carries the gate.
        // The max is only a loose sanity ceiling on the accumulated fp8 floor; it
        // is kept well clear of the observed 5.7% so a single ULP-wobble run can't
        // red CI (a real bug fails the mean, or blows past 12%).
        assert!(meand / sig < 0.01, "MLA forward o mean rel {}", meand / sig);
        assert!(maxd / sig < 0.12, "MLA forward o max rel {}", maxd / sig);
    }
}
