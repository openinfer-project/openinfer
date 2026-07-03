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
    FP8_BLOCK, Fp8LinearScratch, Glm52ProjBytes, ProjWeight, bytes_to_f32, e4m3_to_f32, fp8_linear,
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
    pub(crate) fn q_a(&self) -> &ProjWeight {
        &self.q_a
    }

    pub(crate) fn q_b(&self) -> &ProjWeight {
        &self.q_b
    }

    pub(crate) fn kv_a(&self) -> &ProjWeight {
        &self.kv_a
    }

    pub(crate) fn o_proj(&self) -> &ProjWeight {
        &self.o_proj
    }

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

/// The MLA front-half projections for one token: everything derivable from
/// `hidden` alone, before the sparse top-k is known. `q_resid` is exposed
/// because the DSA indexer consumes it (`wq_b(q_resid)`) to *produce* the top-k
/// that the attend half then attends over.
pub(crate) struct Glm52MlaFront {
    pub(crate) q_resid: CudaSlice<bf16>, // [2048] post q_a_layernorm
    q_full: CudaSlice<bf16>,             // [16384] = [64,256]
    kv_c: CudaSlice<bf16>,               // [512] post kv_a_layernorm
    k_pe: CudaSlice<bf16>,               // [64] pre-rope
}

/// MLA front half (bs=1): fp8 projections + norms from `hidden`.
pub(crate) fn glm52_mla_front(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &CudaSlice<bf16>,
) -> Result<Glm52MlaFront> {
    ensure!(hidden.len() >= HIDDEN, "GLM5.2 MLA hidden too small");
    let q_a = fp8_linear(ctx, &w.q_a, hidden)?; // [2048]
    let q_resid = rms(ctx, q_a, Q_LORA, &w.q_a_ln)?; // [2048]
    let q_full = fp8_linear(ctx, &w.q_b, &q_resid)?; // [16384] = [64,256]
    let ckv = fp8_linear(ctx, &w.kv_a, hidden)?; // [576]
    debug_assert!(ckv.len() >= KV_A_OUT);
    let compressed_kv = slice_copy(ctx, &ckv, 0, KV_LORA)?; // [512]
    let kv_c = rms(ctx, compressed_kv, KV_LORA, &w.kv_a_ln)?; // [512]
    let k_pe = slice_copy(ctx, &ckv, KV_LORA, ROPE_DIM)?; // [64] pre-rope
    Ok(Glm52MlaFront {
        q_resid,
        q_full,
        kv_c,
        k_pe,
    })
}

/// MLA decode forward for one token (bs=1): runs the projections, assembles the
/// FlashMLA query, writes the new token into the paged cache at `position`,
/// attends over the cached context, and projects back to `o[6144]`.
///
/// `cache` is the fp8_ds_mla paged cache (656 bytes/token); `cos`/`sin` are the
/// position's rotary table first half (`[32]`); `topk` is the (fixed-2048,
/// -1-padded) sparse index list; `sched` carries the FlashMLA launch sizing
/// (its contract) plus the precomputed tile-scheduler plan.
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
    sched: &Glm52MlaSchedMetadata,
) -> Result<CudaSlice<bf16>> {
    let front = glm52_mla_front(ctx, w, hidden)?;
    glm52_mla_attend(ctx, w, &front, cos, sin, cache, position, topk, sched)
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
/// projects back to `o[6144]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm52_mla_attend(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    front: &Glm52MlaFront,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    position: usize,
    topk: &CudaSlice<i32>,
    sched: &Glm52MlaSchedMetadata,
) -> Result<CudaSlice<bf16>> {
    let contract = sched.contract;
    // The new token is written to cache slot `position`; the FlashMLA paging then
    // attends over `num_blocks` pages of `PAGE_SIZE` tokens. Couple them so a
    // position past the paged window errors here, not as a silent cache stomp.
    ensure!(
        position < contract.num_blocks * GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
        "GLM5.2 MLA position {position} outside paged cache ({} blocks x {GLM52_FLASHMLA_SPARSE_PAGE_SIZE})",
        contract.num_blocks
    );
    let Glm52MlaFront {
        q_resid: _,
        q_full,
        kv_c,
        k_pe,
    } = front;

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
        q_full,
        QK_NOPE,
        Q_HEAD,
        &mut ql_nope,
        KV_LORA,
        KV_LORA,
        HEADS,
    )?;

    // ---- assemble query [64,576] = [ql_nope | rope(q_pe)] (q_pe in q_full @192) ----
    let mut query = ctx.stream.alloc_zeros::<bf16>(HEADS * QUERY_DIM)?;
    glm52_mla_query_assemble_launch(ctx, &ql_nope, q_full, QK_NOPE, Q_HEAD, cos, sin, &mut query)?;

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
        kv_c,
        &mut ckv_fp8,
        &mut ckv_scales,
    )?;
    glm52_mla_cache_pack_launch(ctx, &ckv_fp8, &ckv_scales, k_pe, cos, sin, cache, position)?;

    // ---- FlashMLA sparse decode -> latent[64,512] ----
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
        &sched.tile_scheduler_metadata,
        &sched.num_splits,
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

/// Every intermediate of one MLA decode forward, allocated once. The plain
/// `glm52_mla_decode_forward` allocates ~20 device buffers per call — each a
/// synchronous `cudaMalloc` — which is the dominant host-side cost per layer
/// per token; this scratch plus `glm52_mla_decode_forward_into` is the
/// zero-allocation variant (same ops, same math, buffers reused).
pub(crate) struct Glm52MlaDecodeScratch {
    /// The FlashMLA contract paired with its pre-computed tile-scheduler plan
    /// ([`Glm52MlaSchedMetadata`], #535). Owning it here means a forward can
    /// never pair this scratch's buffers with a different split count — the
    /// plan is only meaningful under the exact `batch_size` + `num_sm_parts`
    /// it was generated with.
    sched: Glm52MlaSchedMetadata,
    fp8: Fp8LinearScratch,
    q_a: DeviceVec,
    q_resid: DeviceVec,
    q_full: CudaSlice<bf16>,
    ckv: CudaSlice<bf16>,
    compressed_kv: DeviceVec,
    kv_c: DeviceVec,
    k_pe: CudaSlice<bf16>,
    ql_nope: CudaSlice<bf16>,
    query: CudaSlice<bf16>,
    ckv_fp8: CudaSlice<u8>,
    ckv_scales: CudaSlice<f32>,
    latent: CudaSlice<bf16>,
    lse: CudaSlice<f32>,
    lse_accum: CudaSlice<f32>,
    o_accum: CudaSlice<f32>,
    v: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
}

impl Glm52MlaDecodeScratch {
    pub(crate) fn new(ctx: &DeviceContext, contract: Glm52FlashMlaSparseDecode) -> Result<Self> {
        Ok(Self {
            sched: Glm52MlaSchedMetadata::new(ctx, contract)?,
            fp8: Fp8LinearScratch::new(ctx, HEADS * V_HEAD)?,
            q_a: DeviceVec::zeros(ctx, Q_LORA)?,
            q_resid: DeviceVec::zeros(ctx, Q_LORA)?,
            q_full: ctx.stream.alloc_zeros::<bf16>(HEADS * Q_HEAD)?,
            ckv: ctx.stream.alloc_zeros::<bf16>(KV_A_OUT)?,
            compressed_kv: DeviceVec::zeros(ctx, KV_LORA)?,
            kv_c: DeviceVec::zeros(ctx, KV_LORA)?,
            k_pe: ctx.stream.alloc_zeros::<bf16>(ROPE_DIM)?,
            ql_nope: ctx.stream.alloc_zeros::<bf16>(HEADS * KV_LORA)?,
            query: ctx.stream.alloc_zeros::<bf16>(HEADS * QUERY_DIM)?,
            ckv_fp8: ctx.stream.alloc_zeros::<u8>(KV_LORA)?,
            ckv_scales: ctx.stream.alloc_zeros::<f32>(KV_LORA / FP8_BLOCK)?,
            latent: ctx.stream.alloc_zeros::<bf16>(contract.latent_len())?,
            lse: ctx.stream.alloc_zeros::<f32>(contract.lse_len())?,
            lse_accum: ctx.stream.alloc_zeros::<f32>(contract.lse_accum_len())?,
            o_accum: ctx.stream.alloc_zeros::<f32>(contract.o_accum_len())?,
            v: ctx.stream.alloc_zeros::<bf16>(HEADS * V_HEAD)?,
            o: ctx.stream.alloc_zeros::<bf16>(HIDDEN)?,
        })
    }

    /// The layer output written by the last `forward_into`.
    pub(crate) fn output(&self) -> &CudaSlice<bf16> {
        &self.o
    }
}

/// [`glm52_mla_decode_forward`] with all intermediates in `scratch`: the same
/// op sequence with zero per-call allocations. The FlashMLA contract lives in
/// the scratch (buffers and the pre-computed tile schedule were built for it),
/// so a mismatched contract is unrepresentable at this call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm52_mla_decode_forward_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &CudaSlice<bf16>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    position: usize,
    topk: &CudaSlice<i32>,
    scratch: &mut Glm52MlaDecodeScratch,
) -> Result<()> {
    let contract = scratch.sched.contract;
    ensure!(hidden.len() >= HIDDEN, "GLM5.2 MLA hidden too small");
    ensure!(
        position < contract.num_blocks * GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
        "GLM5.2 MLA position {position} outside paged cache ({} blocks x {GLM52_FLASHMLA_SPARSE_PAGE_SIZE})",
        contract.num_blocks
    );

    // ---- front projections ----
    fp8_linear_into(ctx, &w.q_a, hidden, &mut scratch.fp8, &mut scratch.q_a.data)?;
    rms_norm_into(ctx, &scratch.q_a, &w.q_a_ln, RMS_EPS, &mut scratch.q_resid)?;
    fp8_linear_into(
        ctx,
        &w.q_b,
        &scratch.q_resid.data,
        &mut scratch.fp8,
        &mut scratch.q_full,
    )?;
    fp8_linear_into(ctx, &w.kv_a, hidden, &mut scratch.fp8, &mut scratch.ckv)?;
    ctx.stream.memcpy_dtod(
        &scratch.ckv.slice(0..KV_LORA),
        &mut scratch.compressed_kv.data,
    )?;
    rms_norm_into(
        ctx,
        &scratch.compressed_kv,
        &w.kv_a_ln,
        RMS_EPS,
        &mut scratch.kv_c,
    )?;
    ctx.stream.memcpy_dtod(
        &scratch.ckv.slice(KV_LORA..KV_LORA + ROPE_DIM),
        &mut scratch.k_pe,
    )?;

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
        &scratch.q_full,
        QK_NOPE,
        Q_HEAD,
        &mut scratch.ql_nope,
        KV_LORA,
        KV_LORA,
        HEADS,
    )?;

    // ---- assemble query ----
    glm52_mla_query_assemble_launch(
        ctx,
        &scratch.ql_nope,
        &scratch.q_full,
        QK_NOPE,
        Q_HEAD,
        cos,
        sin,
        &mut scratch.query,
    )?;

    // ---- pack the new token into the cache ----
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: 1,
            width: KV_LORA,
            group_size: FP8_BLOCK,
        },
        &scratch.kv_c.data,
        &mut scratch.ckv_fp8,
        &mut scratch.ckv_scales,
    )?;
    glm52_mla_cache_pack_launch(
        ctx,
        &scratch.ckv_fp8,
        &scratch.ckv_scales,
        &scratch.k_pe,
        cos,
        sin,
        cache,
        position,
    )?;

    // ---- FlashMLA sparse decode ----
    // The tile schedule was computed once in `Glm52MlaDecodeScratch::new`
    // (data-independent); the per-token path only runs the decode itself.
    glm52_flashmla_sparse_decode_launch(
        ctx,
        contract,
        &scratch.query,
        cache,
        topk,
        &scratch.sched.tile_scheduler_metadata,
        &scratch.sched.num_splits,
        &mut scratch.latent,
        &mut scratch.lse,
        &mut scratch.lse_accum,
        &mut scratch.o_accum,
    )?;

    // ---- back: v = latent @ W_UV, then o_proj ----
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
        &scratch.latent,
        KV_LORA,
        KV_LORA,
        &mut scratch.v,
        V_HEAD,
        V_HEAD,
        HEADS,
    )?;
    fp8_linear_into(ctx, &w.o_proj, &scratch.v, &mut scratch.fp8, &mut scratch.o)?;
    Ok(())
}
