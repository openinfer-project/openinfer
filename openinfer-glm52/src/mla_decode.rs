//! Single-layer GLM5.2 MLA decode forward (bs=1): `hidden[6144] -> o[6144]`.
//!
//! Composes the oracle-validated GPU ops into one callable forward — the
//! attention half of a decode layer. The pieces are each gated against the HF
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

use crate::fp8::{
    FP8_BLOCK, Glm52ProjBytes, Glm52ProjScratch, ProjWeight, bytes_to_f32, e4m3_to_f32,
    fp8_linear_into,
};

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

/// Persistent scratch for the MLA front-half projections of one token:
/// everything derivable from `hidden` alone, before the sparse top-k is
/// known. Written in place by [`glm52_mla_front_into`] every step (pointer-
/// stable for graph capture); one instance is shared across all 78 layers —
/// layer N's values are dead once layer N's attend consumed them. `q_resid`
/// is exposed because the DSA indexer consumes it (`wq_b(q_resid)`) to
/// *produce* the top-k that the attend half then attends over.
pub(crate) struct Glm52MlaFront {
    q_a: DeviceVec,                // [2048] pre q_a_layernorm
    pub(crate) q_resid: DeviceVec, // [2048] post q_a_layernorm
    q_full: CudaSlice<bf16>,       // [16384] = [64,256]
    ckv: CudaSlice<bf16>,          // [576] = compressed_kv | k_pe
    kv_c_raw: DeviceVec,           // [512] pre kv_a_layernorm
    kv_c: DeviceVec,               // [512] post kv_a_layernorm
    k_pe: CudaSlice<bf16>,         // [64] pre-rope
}

impl Glm52MlaFront {
    pub(crate) fn new(ctx: &DeviceContext) -> Result<Self> {
        Ok(Self {
            q_a: DeviceVec::zeros(ctx, Q_LORA)?,
            q_resid: DeviceVec::zeros(ctx, Q_LORA)?,
            q_full: ctx.stream.alloc_zeros::<bf16>(HEADS * Q_HEAD)?,
            ckv: ctx.stream.alloc_zeros::<bf16>(KV_A_OUT)?,
            kv_c_raw: DeviceVec::zeros(ctx, KV_LORA)?,
            kv_c: DeviceVec::zeros(ctx, KV_LORA)?,
            k_pe: ctx.stream.alloc_zeros::<bf16>(ROPE_DIM)?,
        })
    }
}

/// MLA front half (bs=1): fp8 projections + norms from `hidden` into the
/// persistent front scratch.
pub(crate) fn glm52_mla_front_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &CudaSlice<bf16>,
    proj: &mut Glm52ProjScratch,
    front: &mut Glm52MlaFront,
) -> Result<()> {
    ensure!(hidden.len() >= HIDDEN, "GLM5.2 MLA hidden too small");
    fp8_linear_into(ctx, &w.q_a, hidden, proj, &mut front.q_a.data)?; // [2048]
    rms_norm_into(ctx, &front.q_a, &w.q_a_ln, RMS_EPS, &mut front.q_resid)?;
    fp8_linear_into(ctx, &w.q_b, &front.q_resid.data, proj, &mut front.q_full)?; // [64,256]
    fp8_linear_into(ctx, &w.kv_a, hidden, proj, &mut front.ckv)?; // [576]
    ctx.stream
        .memcpy_dtod(&front.ckv.slice(0..KV_LORA), &mut front.kv_c_raw.data)?;
    rms_norm_into(ctx, &front.kv_c_raw, &w.kv_a_ln, RMS_EPS, &mut front.kv_c)?;
    ctx.stream
        .memcpy_dtod(&front.ckv.slice(KV_LORA..KV_A_OUT), &mut front.k_pe)?;
    Ok(())
}

/// Persistent scratch for the MLA attend half: absorb/query-assemble/cache-
/// pack intermediates and the FlashMLA output + split accumulators. Shared
/// across all 78 layers, written in place every step.
pub(crate) struct Glm52MlaAttendScratch {
    ql_nope: CudaSlice<bf16>,
    query: CudaSlice<bf16>,
    ckv_fp8: CudaSlice<u8>,
    ckv_scales: CudaSlice<f32>,
    latent: CudaSlice<bf16>,
    lse: CudaSlice<f32>,
    lse_accum: CudaSlice<f32>,
    o_accum: CudaSlice<f32>,
    v: CudaSlice<bf16>,
}

impl Glm52MlaAttendScratch {
    pub(crate) fn new(ctx: &DeviceContext, contract: &Glm52FlashMlaSparseDecode) -> Result<Self> {
        Ok(Self {
            ql_nope: ctx.stream.alloc_zeros::<bf16>(HEADS * KV_LORA)?,
            query: ctx.stream.alloc_zeros::<bf16>(HEADS * QUERY_DIM)?,
            ckv_fp8: ctx.stream.alloc_zeros::<u8>(KV_LORA)?,
            ckv_scales: ctx.stream.alloc_zeros::<f32>(KV_LORA / FP8_BLOCK)?,
            latent: ctx.stream.alloc_zeros::<bf16>(contract.latent_len())?,
            lse: ctx.stream.alloc_zeros::<f32>(contract.lse_len())?,
            lse_accum: ctx.stream.alloc_zeros::<f32>(contract.lse_accum_len())?,
            o_accum: ctx.stream.alloc_zeros::<f32>(contract.o_accum_len())?,
            v: ctx.stream.alloc_zeros::<bf16>(HEADS * V_HEAD)?,
        })
    }
}

/// MLA decode forward for one token (bs=1): runs the projections, assembles the
/// FlashMLA query, writes the new token into the paged cache at `position`,
/// attends over the cached context, and projects back to `o[6144]`.
///
/// Allocating convenience over the `_into` halves for the oracle-gate/test
/// paths (per-call scratch). `cache` is the fp8_ds_mla paged cache (656
/// bytes/token); `cos`/`sin` are the position's rotary table first half
/// (`[32]`); `topk` is the (fixed-2048, -1-padded) sparse index list; `sched`
/// carries the FlashMLA launch sizing (its contract) plus the precomputed
/// tile-scheduler plan.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn glm52_mla_decode_forward(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &CudaSlice<bf16>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    position: usize,
    topk: &CudaSlice<i32>,
    sched: &Glm52MlaSchedMetadata,
) -> Result<CudaSlice<bf16>> {
    let mut proj = Glm52ProjScratch::new(ctx, HEADS * V_HEAD)?;
    let mut front = Glm52MlaFront::new(ctx)?;
    let mut attend = Glm52MlaAttendScratch::new(ctx, &sched.contract)?;
    let mut o = ctx.stream.alloc_zeros::<bf16>(HIDDEN)?;
    glm52_mla_front_into(ctx, w, hidden, &mut proj, &mut front)?;
    glm52_mla_attend_into(
        ctx,
        w,
        &front,
        cos,
        sin,
        cache,
        position,
        topk,
        sched,
        &mut proj,
        &mut attend,
        &mut o,
    )?;
    Ok(o)
}

/// A FlashMLA sparse decode contract paired with its tile-scheduler plan. The
/// plan depends only on `batch_size` and `num_sm_parts` — not on position,
/// sequence length, or layer — so it is computed once (model build time)
/// instead of per layer per step (78 × ~25 µs/step at bs=1). Owning the
/// contract makes a plan/contract mismatch unrepresentable: every consumer
/// reads both from the same object.
pub(crate) struct Glm52MlaSchedMetadata {
    contract: Glm52FlashMlaSparseDecode,
    tile_scheduler_metadata: CudaSlice<i32>,
    num_splits: CudaSlice<i32>,
}

impl Glm52MlaSchedMetadata {
    pub(crate) fn new(ctx: &DeviceContext, contract: Glm52FlashMlaSparseDecode) -> Result<Self> {
        let mut tile_scheduler_metadata = ctx
            .stream
            .alloc_zeros::<i32>(contract.tile_scheduler_metadata_len())?;
        let mut num_splits = ctx.stream.alloc_zeros::<i32>(contract.num_splits_len())?;
        glm52_flashmla_sparse_decode_metadata_launch(
            ctx,
            contract.batch_size,
            contract.num_sm_parts,
            &mut tile_scheduler_metadata,
            &mut num_splits,
        )?;
        Ok(Self {
            contract,
            tile_scheduler_metadata,
            num_splits,
        })
    }
}

/// MLA attend half (bs=1): consumes the front projections + the sparse top-k,
/// packs the new token into the paged cache, runs FlashMLA sparse decode, and
/// projects back into `out[6144]`. Every intermediate lives in the persistent
/// attend scratch — the chain is allocation-free.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm52_mla_attend_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    front: &Glm52MlaFront,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    position: usize,
    topk: &CudaSlice<i32>,
    sched: &Glm52MlaSchedMetadata,
    proj: &mut Glm52ProjScratch,
    s: &mut Glm52MlaAttendScratch,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let contract = sched.contract;
    // The new token is written to cache slot `position`; the FlashMLA paging then
    // attends over `num_blocks` pages of `PAGE_SIZE` tokens. Couple them so a
    // position past the paged window errors here, not as a silent cache stomp.
    ensure!(
        position < contract.num_blocks * GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
        "GLM5.2 MLA position {position} outside paged cache ({} blocks x {GLM52_FLASHMLA_SPARSE_PAGE_SIZE})",
        contract.num_blocks
    );

    // ---- absorb: ql_nope[64,512] = q_pass @ W_UK ----
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
        &front.q_full,
        QK_NOPE,
        Q_HEAD,
        &mut s.ql_nope,
        KV_LORA,
        KV_LORA,
        HEADS,
    )?;

    // ---- assemble query [64,576] = [ql_nope | rope(q_pe)] (q_pe in q_full @192) ----
    glm52_mla_query_assemble_launch(
        ctx,
        &s.ql_nope,
        &front.q_full,
        QK_NOPE,
        Q_HEAD,
        cos,
        sin,
        &mut s.query,
    )?;

    // ---- pack the new token into the cache: quant(kv_c) + rope(k_pe) ----
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: 1,
            width: KV_LORA,
            group_size: FP8_BLOCK,
        },
        &front.kv_c.data,
        &mut s.ckv_fp8,
        &mut s.ckv_scales,
    )?;
    glm52_mla_cache_pack_launch(
        ctx,
        &s.ckv_fp8,
        &s.ckv_scales,
        &front.k_pe,
        cos,
        sin,
        cache,
        position,
    )?;

    // ---- FlashMLA sparse decode -> latent[64,512] ----
    glm52_flashmla_sparse_decode_launch(
        ctx,
        contract,
        &s.query,
        cache,
        topk,
        &sched.tile_scheduler_metadata,
        &sched.num_splits,
        &mut s.latent,
        &mut s.lse,
        &mut s.lse_accum,
        &mut s.o_accum,
    )?;

    // ---- back: v[64,256] = latent @ W_UV, then o_proj ----
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
        &s.latent,
        KV_LORA,
        KV_LORA,
        &mut s.v,
        V_HEAD,
        V_HEAD,
        HEADS,
    )?;
    fp8_linear_into(ctx, &w.o_proj, &s.v, proj, out) // [6144]
}
