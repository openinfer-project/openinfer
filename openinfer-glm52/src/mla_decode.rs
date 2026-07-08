//! Single-layer GLM5.2 MLA decode forward, row-batched:
//! `hidden[T, 6144] -> o[T, 6144]` (each row is an independent token).
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

#[cfg(test)]
use openinfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use openinfer_kernels::ops::{
    GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW, GLM52_SPARSE_MLA_HEAD_SLOTS, Glm52FlashMlaSparseDecode,
    Glm52MoeQuantShape, Glm52SparseMlaDecode, gemm_strided_batched_bf16,
    glm52_flashmla_sparse_decode_launch, glm52_flashmla_sparse_decode_metadata_launch,
    glm52_fp8_per_token_group_quant_bf16_launch, glm52_mla_cache_pack_launch,
    glm52_mla_ckv_split_launch, glm52_mla_query_assemble_launch, glm52_sparse_mla_decode_launch,
    rms_norm_rows_into,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceVec};

use crate::config::{
    GLM52_HEADS, GLM52_HIDDEN, GLM52_KV_A_OUT, GLM52_KV_LORA_RANK, GLM52_Q_LORA_RANK,
    GLM52_QK_HEAD_DIM, GLM52_QK_NOPE_HEAD_DIM, GLM52_QK_ROPE_HEAD_DIM, GLM52_RMS_EPS as RMS_EPS,
    GLM52_V_HEAD_DIM,
};
use crate::fp8::{
    FP8_BLOCK, Glm52ProjBytes, ProjWeight, bytes_to_f32, e4m3_to_f32, fp8_linear_into,
};
use crate::rows::Rows;

// Local short names for the config-owned architecture constants (the module
// is dense with shape math; the values live in one place).
const HEADS: usize = GLM52_HEADS;
const HIDDEN: usize = GLM52_HIDDEN;
const Q_LORA: usize = GLM52_Q_LORA_RANK;
const QK_NOPE: usize = GLM52_QK_NOPE_HEAD_DIM; // absorbed q nope width per head
const Q_HEAD: usize = GLM52_QK_HEAD_DIM; // qk_nope(192) + qk_rope(64)
const ROPE_DIM: usize = GLM52_QK_ROPE_HEAD_DIM;
const KV_LORA: usize = GLM52_KV_LORA_RANK;
const KV_A_OUT: usize = GLM52_KV_A_OUT; // compressed_kv(512) + k_pe(64)
const V_HEAD: usize = GLM52_V_HEAD_DIM;
const KV_B_ROWS_PER_HEAD: usize = QK_NOPE + V_HEAD; // 448
const QUERY_DIM: usize = KV_LORA + ROPE_DIM; // 576

/// One MLA layer's attention weights, device-resident. `heads` is the number
/// of q/v heads THIS instance carries: the full 64, or an attention-TP head
/// shard (8 of 64 per rank) — q_b/kv_b/o_proj arrive pre-sliced and every
/// head-indexed shape below follows `heads`. The FlashMLA query/latent stay
/// full-width regardless (see `Glm52MlaAttendScratch`).
pub(crate) struct Glm52MlaLayerWeights {
    q_a: ProjWeight,
    q_a_ln: DeviceVec,
    q_b: ProjWeight,
    kv_a: ProjWeight,
    kv_a_ln: DeviceVec,
    o_proj: ProjWeight,
    w_uk: CudaSlice<bf16>, // [heads, 192, 512]
    w_uv: CudaSlice<bf16>, // [heads, 256, 512]
    pub(crate) heads: usize,
}

impl Glm52MlaLayerWeights {
    /// Build from raw checkpoint bytes: upload the fp8 projections + bf16
    /// layernorm gammas, and host-dequant kv_b into the bf16 absorb factors
    /// W_UK = kv_b[:, :192, :], W_UV = kv_b[:, 192:, :].
    #[cfg(test)]
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
        let heads = mla_heads(q_b.n, kv_b.n, o_proj.k)?;
        check("q_a_proj", q_a, Q_LORA, HIDDEN)?;
        check("q_b_proj", q_b, heads * Q_HEAD, Q_LORA)?;
        check("kv_a_proj_with_mqa", kv_a, KV_A_OUT, HIDDEN)?;
        check("kv_b_proj", kv_b, heads * KV_B_ROWS_PER_HEAD, KV_LORA)?;
        check("o_proj", o_proj, HIDDEN, heads * V_HEAD)?;
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
            heads,
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
        kv_b: &ProjWeight,
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
        let heads = mla_heads(q_b.n, kv_b.n, o_proj.k)?;
        check("q_a_proj", &q_a, Q_LORA, HIDDEN)?;
        check("q_b_proj", &q_b, heads * Q_HEAD, Q_LORA)?;
        check("kv_a_proj_with_mqa", &kv_a, KV_A_OUT, HIDDEN)?;
        check("kv_b_proj", kv_b, heads * KV_B_ROWS_PER_HEAD, KV_LORA)?;
        check("o_proj", &o_proj, HIDDEN, heads * V_HEAD)?;
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
            heads,
        })
    }
}

/// Derive the head count carried by a (possibly attention-TP-sharded) weight
/// set from the q_b/kv_b/o_proj shapes; the three must agree and the FlashMLA
/// full width (64) is the ceiling.
fn mla_heads(q_b_n: usize, kv_b_n: usize, o_proj_k: usize) -> Result<usize> {
    ensure!(
        q_b_n.is_multiple_of(Q_HEAD) && q_b_n > 0,
        "GLM5.2 q_b_proj n {q_b_n} is not a positive multiple of {Q_HEAD}"
    );
    let heads = q_b_n / Q_HEAD;
    ensure!(
        heads <= HEADS,
        "GLM5.2 MLA heads {heads} exceeds the architecture's {HEADS}"
    );
    ensure!(
        kv_b_n == heads * KV_B_ROWS_PER_HEAD && o_proj_k == heads * V_HEAD,
        "GLM5.2 MLA head-count mismatch: q_b says {heads} heads, kv_b n {kv_b_n} (want {}), o_proj k {o_proj_k} (want {})",
        heads * KV_B_ROWS_PER_HEAD,
        heads * V_HEAD
    );
    Ok(heads)
}

/// Host-dequant kv_b (fp8 e4m3 block-scaled) into bf16 W_UK [heads,192,512]
/// (nope) and W_UV [heads,256,512] (v) absorb factors, head-major, uploaded
/// to device. The head count follows kv_b.n (full 64 or an attention-TP
/// shard whose rows were sliced upstream).
fn dequant_kv_b(
    ctx: &DeviceContext,
    kv_b: &Glm52ProjBytes,
) -> Result<(CudaSlice<bf16>, CudaSlice<bf16>)> {
    let heads = kv_b.n / KV_B_ROWS_PER_HEAD;
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
    let mut w_uk = vec![bf16::from_f32(0.0); heads * QK_NOPE * KV_LORA];
    let mut w_uv = vec![bf16::from_f32(0.0); heads * V_HEAD * KV_LORA];
    for h in 0..heads {
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

/// Persistent scratch for the MLA front-half projections of one step's
/// `tokens` rows: everything derivable from `hidden` alone, before the sparse
/// top-k is known. Written in place by [`glm52_mla_front_into`] every step
/// (pointer-stable for graph capture); one instance is shared across all 78
/// layers — layer N's values are dead once layer N's attend consumed them.
/// `q_resid` is exposed because the DSA indexer consumes it (`wq_b(q_resid)`)
/// to *produce* the top-k that the attend half then attends over. The row
/// count is baked here so every front kernel derives it from one place.
pub(crate) struct Glm52MlaFront {
    q_a: CudaSlice<bf16>,             // [T, 2048] pre q_a_layernorm
    pub(crate) q_resid: Rows<Q_LORA>, // [T, 2048] post q_a_layernorm
    q_full: CudaSlice<bf16>,          // [T, heads, 256] (compact head shard)
    ckv: CudaSlice<bf16>,             // [T, 576] = compressed_kv | k_pe
    kv_c_raw: CudaSlice<bf16>,        // [T, 512] pre kv_a_layernorm
    kv_c: CudaSlice<bf16>,            // [T, 512] post kv_a_layernorm
    k_pe: CudaSlice<bf16>,            // [T, 64] pre-rope
    heads: usize,
    // Owned mma partial buffer for the front projections (q_a/q_b/kv_a). One
    // per scratch struct: the ctx/aux stream overlap must never share one.
    gemv_partial: CudaSlice<f32>,
}

impl Glm52MlaFront {
    pub(crate) fn new(ctx: &DeviceContext, tokens: usize, heads: usize) -> Result<Self> {
        ensure!(
            heads >= 1 && heads <= HEADS,
            "GLM5.2 MLA front heads {heads} out of 1..={HEADS}"
        );
        Ok(Self {
            q_a: ctx.stream.alloc_zeros::<bf16>(tokens * Q_LORA)?,
            q_resid: Rows::zeros(ctx, tokens)?,
            q_full: ctx.stream.alloc_zeros::<bf16>(tokens * heads * Q_HEAD)?,
            ckv: ctx.stream.alloc_zeros::<bf16>(tokens * KV_A_OUT)?,
            kv_c_raw: ctx.stream.alloc_zeros::<bf16>(tokens * KV_LORA)?,
            kv_c: ctx.stream.alloc_zeros::<bf16>(tokens * KV_LORA)?,
            k_pe: ctx.stream.alloc_zeros::<bf16>(tokens * ROPE_DIM)?,
            heads,
            gemv_partial: ctx
                .stream
                .alloc_zeros::<f32>(tokens * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW)?,
        })
    }

    /// The step row count every front buffer was sized for.
    pub(crate) fn tokens(&self) -> usize {
        self.q_resid.tokens()
    }
}

/// The q-phase of the MLA front: `q_a` projection + q_a_layernorm over the
/// front's `tokens` rows. Split out because `q_resid` is everything the DSA
/// indexer needs — the caller can fork the indexer onto an aux stream right
/// after this and run [`glm52_mla_front_rest_into`] concurrently.
pub(crate) fn glm52_mla_front_q_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &Rows<HIDDEN>,
    front: &mut Glm52MlaFront,
) -> Result<()> {
    let t = front.tokens();
    fp8_linear_into(
        ctx,
        &w.q_a,
        t,
        hidden.data(),
        Some(&mut front.gemv_partial),
        &mut front.q_a,
    )?; // [T, 2048]
    rms_norm_rows_into(
        ctx,
        &front.q_a,
        &w.q_a_ln,
        RMS_EPS,
        Q_LORA,
        t,
        front.q_resid.data_mut(),
    )
}

/// The remainder of the MLA front: q_b + kv_a projections and the kv_c/k_pe
/// unpacking, over the front's `tokens` rows. Independent of the indexer.
pub(crate) fn glm52_mla_front_rest_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &Rows<HIDDEN>,
    front: &mut Glm52MlaFront,
) -> Result<()> {
    let t = front.tokens();
    ensure!(
        w.heads == front.heads,
        "GLM5.2 MLA front sized for {} heads but weights carry {}",
        front.heads,
        w.heads
    );
    fp8_linear_into(
        ctx,
        &w.q_b,
        t,
        front.q_resid.data(),
        Some(&mut front.gemv_partial),
        &mut front.q_full,
    )?; // [T, heads, 256]
    fp8_linear_into(
        ctx,
        &w.kv_a,
        t,
        hidden.data(),
        Some(&mut front.gemv_partial),
        &mut front.ckv,
    )?; // [T, 576]
    glm52_mla_ckv_split_launch(ctx, t, &front.ckv, &mut front.kv_c_raw, &mut front.k_pe)?;
    rms_norm_rows_into(
        ctx,
        &front.kv_c_raw,
        &w.kv_a_ln,
        RMS_EPS,
        KV_LORA,
        t,
        &mut front.kv_c,
    )
}

/// MLA front half: fp8 projections + norms from `hidden[T, 6144]` into the
/// persistent front scratch.
#[cfg(test)]
pub(crate) fn glm52_mla_front_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &Rows<HIDDEN>,
    front: &mut Glm52MlaFront,
) -> Result<()> {
    glm52_mla_front_q_into(ctx, w, hidden, front)?;
    glm52_mla_front_rest_into(ctx, w, hidden, front)
}

/// Sparse-attention backend scratch. A head shard (attention-TP, <= 16 head
/// slots) runs the right-sized sparse MLA kernel — fixed 16-split grid plus a
/// deterministic combine; the 64-head EP8 path stays on FlashMLA with its
/// tile-scheduler split accumulators.
enum Glm52SparseAttend {
    Rightsize {
        // Unnormalized split partials + (m, l) pairs for the combine.
        o_part: CudaSlice<f32>,
        ml_part: CudaSlice<f32>,
    },
    Flash {
        lse: CudaSlice<f32>,
        lse_accum: CudaSlice<f32>,
        o_accum: CudaSlice<f32>,
    },
}

/// Persistent scratch for the MLA attend half: absorb/query-assemble/cache-
/// pack intermediates and the sparse-attention output + backend accumulators,
/// sized for the contract's `batch_size` rows. Shared across all 78 layers,
/// written in place every step.
pub(crate) struct Glm52MlaAttendScratch {
    // Compact head-shard buffers ([T, heads, .]): the absorb GEMM output and
    // the W_UV output feeding o_proj.
    ql_nope: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    // Full-width sparse-attention buffers ([T, 64, .]): both backends keep
    // the 64-slot query/latent shape. Under a head shard the real heads
    // occupy slots 0..heads and the pad slots keep this alloc's zero fill
    // forever (never read back).
    query: CudaSlice<bf16>,
    latent: CudaSlice<bf16>,
    attend: Glm52SparseAttend,
    ckv_fp8: CudaSlice<u8>,
    ckv_scales: CudaSlice<f32>,
    heads: usize,
    // Owned mma partial buffer for the o_proj projection (see Glm52MlaFront).
    gemv_partial: CudaSlice<f32>,
}

impl Glm52MlaAttendScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        contract: &Glm52FlashMlaSparseDecode,
        heads: usize,
    ) -> Result<Self> {
        ensure!(
            heads >= 1 && heads <= HEADS,
            "GLM5.2 MLA attend heads {heads} out of 1..={HEADS}"
        );
        let t = contract.batch_size;
        let attend = if heads <= GLM52_SPARSE_MLA_HEAD_SLOTS {
            let rightsize = rightsize_contract(contract, heads);
            rightsize.validate()?;
            Glm52SparseAttend::Rightsize {
                o_part: ctx.stream.alloc_zeros::<f32>(rightsize.o_part_len())?,
                ml_part: ctx.stream.alloc_zeros::<f32>(rightsize.ml_part_len())?,
            }
        } else {
            Glm52SparseAttend::Flash {
                lse: ctx.stream.alloc_zeros::<f32>(contract.lse_len())?,
                lse_accum: ctx.stream.alloc_zeros::<f32>(contract.lse_accum_len())?,
                o_accum: ctx.stream.alloc_zeros::<f32>(contract.o_accum_len())?,
            }
        };
        Ok(Self {
            ql_nope: ctx.stream.alloc_zeros::<bf16>(t * heads * KV_LORA)?,
            v: ctx.stream.alloc_zeros::<bf16>(t * heads * V_HEAD)?,
            query: ctx.stream.alloc_zeros::<bf16>(t * HEADS * QUERY_DIM)?,
            latent: ctx.stream.alloc_zeros::<bf16>(contract.latent_len())?,
            attend,
            ckv_fp8: ctx.stream.alloc_zeros::<u8>(t * KV_LORA)?,
            ckv_scales: ctx.stream.alloc_zeros::<f32>(t * (KV_LORA / FP8_BLOCK))?,
            heads,
            gemv_partial: ctx
                .stream
                .alloc_zeros::<f32>(t * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW)?,
        })
    }
}

fn rightsize_contract(contract: &Glm52FlashMlaSparseDecode, heads: usize) -> Glm52SparseMlaDecode {
    Glm52SparseMlaDecode {
        batch_size: contract.batch_size,
        num_blocks: contract.num_blocks,
        topk: contract.topk,
        heads,
        sm_scale: contract.sm_scale,
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
    hidden: &Rows<HIDDEN>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    position: usize,
    topk: &CudaSlice<i32>,
    sched: &Glm52MlaSchedMetadata,
) -> Result<Rows<HIDDEN>> {
    ensure!(
        position < sched.contract.num_blocks * GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
        "GLM5.2 MLA position {position} outside paged cache ({} blocks x {GLM52_FLASHMLA_SPARSE_PAGE_SIZE})",
        sched.contract.num_blocks
    );
    // Front, attend scratch, and output all sized from the plan's contract —
    // the same one-construction-point coherence the production bucket state
    // provides.
    let mut front = Glm52MlaFront::new(ctx, sched.batch(), w.heads)?;
    let mut attend = Glm52MlaAttendScratch::new(ctx, &sched.contract, w.heads)?;
    let mut slot_mapping = ctx.stream.alloc_zeros::<i64>(1)?;
    ctx.stream
        .memcpy_htod(&[position as i64], &mut slot_mapping)?;
    let mut o = Rows::zeros(ctx, sched.batch())?;
    glm52_mla_front_into(ctx, w, hidden, &mut front)?;
    glm52_mla_attend_into(
        ctx,
        w,
        &front,
        cos,
        sin,
        cache,
        &slot_mapping,
        topk,
        sched,
        &mut attend,
        &mut o,
    )?;
    Ok(o)
}

/// A FlashMLA sparse decode contract paired with its tile-scheduler plan. The
/// plan depends only on `batch_size`, `topk` and `num_sm_parts` — not on
/// position, sequence length, or layer — so it is computed once (model build
/// time) instead of per layer per step (78 × ~25 µs/step at bs=1). Owning the
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
            contract.topk,
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

    /// The sparse index-list length this plan was built for. The DSA indexer
    /// must produce its top-k with the same k — reading it from the plan makes
    /// an indexer/attend mismatch unrepresentable.
    pub(crate) fn topk(&self) -> usize {
        self.contract.topk
    }

    /// The decode row count this plan was built for — the single source of
    /// truth for a step's batch shape (every consumer reads it from here).
    pub(crate) fn batch(&self) -> usize {
        self.contract.batch_size
    }
}

/// MLA attend half over the plan's `batch()` rows: consumes the front
/// projections + the per-row sparse top-k, packs each row's new token into
/// its paged-cache slot, runs FlashMLA sparse decode, and projects back into
/// `out[T, 6144]`. Every intermediate lives in the persistent attend scratch
/// — the chain is allocation-free. `cos`/`sin` carry one `[32]` row per token
/// (each row sits at its own position); `slot_mapping`/`topk` are `[T]` /
/// `[T, topk]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm52_mla_attend_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    front: &Glm52MlaFront,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    slot_mapping: &CudaSlice<i64>,
    topk: &CudaSlice<i32>,
    sched: &Glm52MlaSchedMetadata,
    s: &mut Glm52MlaAttendScratch,
    out: &mut Rows<HIDDEN>,
) -> Result<()> {
    let contract = sched.contract;
    let t = contract.batch_size;
    ensure!(
        w.heads == front.heads && w.heads == s.heads,
        "GLM5.2 MLA attend head mismatch: weights {}, front {}, scratch {}",
        w.heads,
        front.heads,
        s.heads
    );
    let heads = w.heads;
    // Each row's new token is written to cache slot `slot_mapping[row]`
    // (device data, so the launch replays under CUDA graph capture); the
    // cache-pack kernel traps on a slot outside the paged window. The
    // every-step host guard is the caller's position bound (`decode_step`
    // prologue: position < max_model_len and each row confined to its
    // own slot region by construction).

    // ---- absorb: ql_nope[T,heads,512] = q_pass @ W_UK ----
    // cuBLAS batches over this instance's heads (the full 64 or an
    // attention-TP shard); the T rows ride the GEMM's n dimension — column t
    // of head h reads q_full[t, h, 0..192] (ldb = the compact [T,heads,256]
    // token stride) and writes ql_nope[t, h, 0..512] (ldc = the compact
    // [T,heads,512] token stride).
    gemm_strided_batched_bf16(
        ctx,
        false,
        false,
        KV_LORA,
        t,
        QK_NOPE,
        &w.w_uk,
        KV_LORA,
        QK_NOPE * KV_LORA,
        &front.q_full,
        heads * Q_HEAD,
        Q_HEAD,
        &mut s.ql_nope,
        heads * KV_LORA,
        KV_LORA,
        heads,
    )?;

    // ---- assemble query [T,64,576] = [ql_nope | rope(q_pe)] (q_pe in q_full @192) ----
    // Compact shard in, full-width query out (FlashMLA's fixed shape).
    glm52_mla_query_assemble_launch(
        ctx,
        t,
        heads,
        &s.ql_nope,
        &front.q_full,
        QK_NOPE,
        Q_HEAD,
        cos,
        sin,
        &mut s.query,
    )?;

    // ---- pack each row's new token into the cache: quant(kv_c) + rope(k_pe) ----
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: t,
            width: KV_LORA,
            group_size: FP8_BLOCK,
        },
        &front.kv_c,
        &mut s.ckv_fp8,
        &mut s.ckv_scales,
    )?;
    glm52_mla_cache_pack_launch(
        ctx,
        t,
        &s.ckv_fp8,
        &s.ckv_scales,
        &front.k_pe,
        cos,
        sin,
        cache,
        slot_mapping,
    )?;

    // ---- sparse decode -> latent[T,64,512] ----
    match &mut s.attend {
        Glm52SparseAttend::Rightsize { o_part, ml_part } => {
            glm52_sparse_mla_decode_launch(
                ctx,
                rightsize_contract(&contract, heads),
                &s.query,
                cache,
                topk,
                o_part,
                ml_part,
                &mut s.latent,
            )?;
        }
        Glm52SparseAttend::Flash {
            lse,
            lse_accum,
            o_accum,
        } => {
            glm52_flashmla_sparse_decode_launch(
                ctx,
                contract,
                &s.query,
                cache,
                topk,
                &sched.tile_scheduler_metadata,
                &sched.num_splits,
                &mut s.latent,
                lse,
                lse_accum,
                o_accum,
            )?;
        }
    }

    // ---- back: v[T,heads,256] = latent @ W_UV, then o_proj ----
    // latent stays full-width [T,64,512] (FlashMLA output); the batch count
    // reads only the shard's head slots. v is compact — exactly the sliced
    // o_proj's k = heads * 256 input.
    gemm_strided_batched_bf16(
        ctx,
        true,
        false,
        V_HEAD,
        t,
        KV_LORA,
        &w.w_uv,
        KV_LORA,
        V_HEAD * KV_LORA,
        &s.latent,
        HEADS * KV_LORA,
        KV_LORA,
        &mut s.v,
        heads * V_HEAD,
        V_HEAD,
        heads,
    )?;
    fp8_linear_into(
        ctx,
        &w.o_proj,
        t,
        &s.v,
        Some(&mut s.gemv_partial),
        out.data_mut(),
    ) // [T, 6144]
}
