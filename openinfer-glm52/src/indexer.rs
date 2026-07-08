//! GLM5.2 DSA indexer decode forward, row-batched: produces per-row
//! `topk_indices[T, topk]`.
//!
//! Aligned to vllm `DeepseekV32Indexer` (the production reference). The
//! indexer computes per-token similarity against an index-K cache, selects
//! sparse top-k=2048 slots, and returns global KV cache slot indices for the
//! FlashMLA sparse decode to attend over.
//!
//! Data flow (see `docs/models/glm52/indexer-forward.md` for the vllm
//! cross-reference):
//!
//! ```text
//! q_resid[2048]  (from q_a_layernorm(q_a_proj(hidden)) — produced by the MLA layer)
//!   |
//!   +-- wq_b (fp8 linear) -> q[32, 128]
//!   |     +-- layer_norm (FlashInfer, eps=1e-6, has bias) -> k[128]
//!   |     +-- RoPE (non-interleaved/half-split, q[:64], k[:64], cos/sin[32])
//!   |     +-- q per-token-group fp8 quant -> q_fp8[32*128], q_scale[32]
//!   |     +-- weights fold: weights * q_scale * softmax_scale * n_heads^-0.5
//!   |
//! hidden[6144]
//!   +-- wk (fp8 linear) -> k_raw[128]
//!   +-- weights_proj (bf16 GEMM) -> weights[32]
//!   +-- k quant + cache write (glm52_indexer_k_quant_and_cache)
//!   |
//!   +-- DeepGEMM paged MQA logits (fuses per-head ReLU + weighting)
//!   +-- bf16→f32 cast
//!   +-- FlashInfer deterministic top-k K=2048
//!   +-- local top-k offsets -> global KV slots
//! ```

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;

use openinfer_kernels::ops::{
    GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW, GLM52_INDEXER_HEAD_DIM, GLM52_INDEXER_TOPK,
    Glm52DeepGemmMqaLogitsShape, Glm52IndexerCacheInsert, Glm52IndexerCacheLayout,
    Glm52IndexerLocalTopKToSlots, Glm52IndexerScaleFormat, Glm52IndexerTopK, Glm52MoeQuantShape,
    bf16_bytes_to_f32_into, gemm_strided_batched_bf16, glm52_deepgemm_paged_mqa_logits_launch,
    glm52_deepgemm_paged_mqa_metadata_launch, glm52_flashinfer_topk_2048_launch,
    glm52_fp8_per_token_group_quant_bf16_launch, glm52_indexer_k_quant_and_cache_launch,
    glm52_indexer_local_topk_to_slots_launch, glm52_indexer_rope_launch,
    glm52_indexer_weights_fold_launch, layer_norm_into,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::config::{GLM52_HIDDEN, GLM52_INDEX_HEAD_DIM, GLM52_INDEX_HEADS, GLM52_Q_LORA_RANK};
use crate::fp8::{FP8_BLOCK, ProjWeight, fp8_linear_into};
#[cfg(test)]
use crate::fp8::{Glm52ProjBytes, fp8_linear};
use crate::rows::Rows;

const HIDDEN: usize = GLM52_HIDDEN;
const Q_LORA: usize = GLM52_Q_LORA_RANK;
const INDEX_HEADS: usize = GLM52_INDEX_HEADS;
const INDEX_HEAD_DIM: usize = GLM52_INDEX_HEAD_DIM;
// vllm: softmax_scale = head_dim ** -0.5 = 128 ** -0.5
const SOFTMAX_SCALE: f32 = 0.088_388_35; // 1.0 / 128.0f32.sqrt()
// vllm: n_heads ** -0.5 = 32 ** -0.5
const N_HEADS_SCALE: f32 = 0.176_776_7; // 1.0 / 32.0f32.sqrt()
const K_NORM_EPS: f32 = 1.0e-6;

/// One DSA indexer layer's weights, device-resident.
pub(crate) struct Glm52IndexerLayerWeights {
    wq_b: ProjWeight,              // [32*128, 2048]
    wk: ProjWeight,                // [128, 6144]
    weights_proj: CudaSlice<bf16>, // [32, 6144] — bf16 GEMM (transformers _keep_in_fp32_modules)
    k_norm_w: CudaSlice<f32>,      // [128] — LayerNorm gamma (f32 for FlashInfer)
    k_norm_b: CudaSlice<f32>,      // [128] — LayerNorm beta  (f32 for FlashInfer)
}

impl Glm52IndexerLayerWeights {
    /// Build from raw checkpoint bytes (the test path). Same pattern as
    /// `Glm52MlaLayerWeights::from_host`. `weights_proj` is a bf16 `[32, 6144]`
    /// tensor (transformers keeps it in fp32 via `_keep_in_fp32_modules`, but
    /// the checkpoint stores bf16).
    #[cfg(test)]
    pub(crate) fn from_host(
        ctx: &DeviceContext,
        wq_b: &Glm52ProjBytes,
        wk: &Glm52ProjBytes,
        weights_proj_bf16: &[u8],
        k_norm_w: &[u8],
        k_norm_b: &[u8],
    ) -> Result<Self> {
        let check = |label: &str, p: &Glm52ProjBytes, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 indexer {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        check("wq_b", wq_b, INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA)?;
        check("wk", wk, INDEX_HEAD_DIM, HIDDEN)?;
        ensure!(
            weights_proj_bf16.len() == INDEX_HEADS * HIDDEN * 2,
            "GLM5.2 indexer weights_proj bytes {} != {} (bf16 [32, 6144])",
            weights_proj_bf16.len(),
            INDEX_HEADS * HIDDEN * 2
        );
        ensure!(
            k_norm_w.len() == INDEX_HEAD_DIM * 2,
            "GLM5.2 indexer k_norm_w bytes {} != {}",
            k_norm_w.len(),
            INDEX_HEAD_DIM * 2
        );
        ensure!(
            k_norm_b.len() == INDEX_HEAD_DIM * 2,
            "GLM5.2 indexer k_norm_b bytes {} != {}",
            k_norm_b.len(),
            INDEX_HEAD_DIM * 2
        );

        let w = ProjWeight::upload(ctx, wq_b)?;
        let k = ProjWeight::upload(ctx, wk)?;
        let proj_bf16: &[bf16] = unsafe {
            std::slice::from_raw_parts(
                weights_proj_bf16.as_ptr().cast::<bf16>(),
                INDEX_HEADS * HIDDEN,
            )
        };
        let mut weights_proj = ctx.stream.alloc_zeros::<bf16>(INDEX_HEADS * HIDDEN)?;
        ctx.stream.memcpy_htod(proj_bf16, &mut weights_proj)?;
        let norm_w = upcast_bf16_to_f32(ctx, k_norm_w)?;
        let norm_b = upcast_bf16_to_f32(ctx, k_norm_b)?;
        Ok(Self {
            wq_b: w,
            wk: k,
            weights_proj,
            k_norm_w: norm_w,
            k_norm_b: norm_b,
        })
    }

    /// Build from already-resident weights (the production loader path). The
    /// fp8 projections and the bf16 `weights_proj` are moved in; the two
    /// 128-element k_norm tensors come as host bytes because the checkpoint
    /// stores bf16 and FlashInfer LayerNorm needs f32 gamma/beta.
    pub(crate) fn from_device(
        ctx: &DeviceContext,
        wq_b: ProjWeight,
        wk: ProjWeight,
        weights_proj: CudaSlice<bf16>,
        k_norm_w: &[u8],
        k_norm_b: &[u8],
    ) -> Result<Self> {
        let check = |label: &str, p: &ProjWeight, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 indexer {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        check("wq_b", &wq_b, INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA)?;
        check("wk", &wk, INDEX_HEAD_DIM, HIDDEN)?;
        ensure!(
            weights_proj.len() == INDEX_HEADS * HIDDEN,
            "GLM5.2 indexer weights_proj len {} != {} (bf16 [32, 6144])",
            weights_proj.len(),
            INDEX_HEADS * HIDDEN
        );
        Ok(Self {
            wq_b,
            wk,
            weights_proj,
            k_norm_w: upcast_bf16_to_f32(ctx, k_norm_w)?,
            k_norm_b: upcast_bf16_to_f32(ctx, k_norm_b)?,
        })
    }
}

/// Copy bf16 bytes from a checkpoint tensor and upcast to f32 on host, then
/// upload to device. Used for k_norm weight/bias (FlashInfer LayerNorm
/// requires f32 gamma/beta).
#[allow(clippy::cast_ptr_alignment)]
fn upcast_bf16_to_f32(ctx: &DeviceContext, src: &[u8]) -> Result<CudaSlice<f32>> {
    ensure!(
        src.len() == INDEX_HEAD_DIM * 2,
        "GLM5.2 indexer k_norm bytes {} != {}",
        src.len(),
        INDEX_HEAD_DIM * 2
    );
    let bf16_vals: &[bf16] =
        unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<bf16>(), INDEX_HEAD_DIM) };
    let f32_vals: Vec<f32> = bf16_vals.iter().map(|v| v.to_f32()).collect();
    let mut dst = ctx.stream.alloc_zeros::<f32>(INDEX_HEAD_DIM)?;
    ctx.stream.memcpy_htod(&f32_vals, &mut dst)?;
    Ok(dst)
}

/// Cache-fill phase: compute k for one token and write it into the index_k_cache.
/// Used during prefill to populate the cache for all positions before the
/// topk query. Does NOT compute logits or topk — only wk + LayerNorm + RoPE(k)
/// + quant + cache-write.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn glm52_indexer_cache_fill(
    ctx: &DeviceContext,
    w: &Glm52IndexerLayerWeights,
    hidden: &CudaSlice<bf16>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    index_k_cache: &mut CudaSlice<u8>,
    cache_layout: Glm52IndexerCacheLayout,
    slot_mapping: &CudaSlice<i64>,
) -> Result<()> {
    ensure!(
        hidden.len() >= HIDDEN,
        "GLM5.2 indexer cache_fill hidden too small"
    );

    let k_raw = fp8_linear(ctx, &w.wk, hidden)?; // [128]
    let mut k = ctx.stream.alloc_zeros::<bf16>(INDEX_HEAD_DIM)?;
    layer_norm_into(
        ctx,
        &k_raw,
        &w.k_norm_w,
        &w.k_norm_b,
        K_NORM_EPS,
        INDEX_HEAD_DIM,
        1,
        &mut k,
    )?;

    // RoPE: the kernel applies to both q and k; use a dummy q buffer.
    let mut q_dummy = ctx
        .stream
        .alloc_zeros::<bf16>(INDEX_HEADS * INDEX_HEAD_DIM)?;
    glm52_indexer_rope_launch(ctx, &mut q_dummy, &mut k, INDEX_HEADS, 1, cos, sin)?;

    glm52_indexer_k_quant_and_cache_launch(
        ctx,
        Glm52IndexerCacheInsert {
            tokens: 1,
            layout: cache_layout,
            scale_format: Glm52IndexerScaleFormat::F32,
        },
        &k,
        index_k_cache,
        slot_mapping,
    )?;
    Ok(())
}

/// Persistent scratch for the DSA indexer forward: every intermediate plus
/// the DeepGEMM MQA logits shape (fixed at build — the decode batch, fixed
/// paged layout and logits stride; `shape.batch_size` is the row capacity
/// every buffer is sized for). `global_slots` doubles as the cross-layer
/// top-k carry: a full-indexer layer writes it, the following shared layers
/// read it until the next full layer overwrites it.
pub(crate) struct Glm52IndexerScratch {
    shape: Glm52DeepGemmMqaLogitsShape,
    q: CudaSlice<bf16>,
    k_raw: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    weights_bf16: CudaSlice<bf16>,
    q_fp8: CudaSlice<u8>,
    q_scale: CudaSlice<f32>,
    weights_folded: CudaSlice<f32>,
    schedule_meta: CudaSlice<i32>,
    context_lens: CudaSlice<i32>,
    logits: CudaSlice<u8>,
    logits_f32: CudaSlice<f32>,
    topk_offsets: CudaSlice<i32>,
    topk_values: CudaSlice<f32>,
    pub(crate) global_slots: CudaSlice<i32>,
    topk_lens: CudaSlice<i32>,
    // Owned mma partial buffer (wq_b/wk). The indexer chain runs on the AUX
    // stream concurrently with the ctx-side MLA front — this buffer being
    // owned here (not shared per device) is what makes that overlap safe.
    gemv_partial: CudaSlice<f32>,
}

impl Glm52IndexerScratch {
    pub(crate) fn new(ctx: &DeviceContext, shape: Glm52DeepGemmMqaLogitsShape) -> Result<Self> {
        let t = shape.batch_size;
        let logits_elems = t * shape.next_n * shape.logits_stride;
        Ok(Self {
            q: ctx
                .stream
                .alloc_zeros::<bf16>(t * INDEX_HEADS * INDEX_HEAD_DIM)?,
            k_raw: ctx.stream.alloc_zeros::<bf16>(t * INDEX_HEAD_DIM)?,
            k: ctx.stream.alloc_zeros::<bf16>(t * INDEX_HEAD_DIM)?,
            weights_bf16: ctx.stream.alloc_zeros::<bf16>(t * INDEX_HEADS)?,
            q_fp8: ctx
                .stream
                .alloc_zeros::<u8>(t * INDEX_HEADS * INDEX_HEAD_DIM)?,
            q_scale: ctx.stream.alloc_zeros::<f32>(t * INDEX_HEADS)?,
            weights_folded: ctx.stream.alloc_zeros::<f32>(t * INDEX_HEADS)?,
            schedule_meta: ctx
                .stream
                .alloc_zeros::<i32>(shape.schedule_metadata_len())?,
            context_lens: ctx.stream.alloc_zeros::<i32>(t)?,
            logits: ctx.stream.alloc_zeros::<u8>(logits_elems * 2)?, // bf16
            logits_f32: ctx.stream.alloc_zeros::<f32>(logits_elems)?,
            topk_offsets: ctx.stream.alloc_zeros::<i32>(t * GLM52_INDEXER_TOPK)?,
            topk_values: ctx.stream.alloc_zeros::<f32>(t * GLM52_INDEXER_TOPK)?,
            global_slots: ctx.stream.alloc_zeros::<i32>(t * GLM52_INDEXER_TOPK)?,
            topk_lens: ctx.stream.alloc_zeros::<i32>(t)?,
            gemv_partial: ctx
                .stream
                .alloc_zeros::<f32>(t * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW)?,
            shape,
        })
    }

    /// The build-time MQA logits shape for a decode step: the decode batch,
    /// next_n=1, the paged index-K cache layout, and the fixed logits stride.
    pub(crate) fn decode_shape(
        batch: usize,
        cache_layout: Glm52IndexerCacheLayout,
        block_table_stride: usize,
        num_sms: usize,
        max_model_len: usize,
    ) -> Glm52DeepGemmMqaLogitsShape {
        Glm52DeepGemmMqaLogitsShape {
            batch_size: batch,
            next_n: 1,
            num_heads: INDEX_HEADS,
            head_dim: GLM52_INDEXER_HEAD_DIM,
            num_kv_blocks: cache_layout.cache_blocks,
            block_kv: cache_layout.cache_block_size,
            kv_cache_stride_bytes: cache_layout.cache_block_stride_bytes,
            is_context_lens_2d: false,
            is_varlen: false,
            logits_stride: max_model_len.next_multiple_of(256),
            block_table_stride,
            num_sms,
        }
    }
}

/// DSA indexer decode forward over the scratch's `shape.batch_size` rows:
/// computes each row's sparse top-k slot indices for the FlashMLA sparse
/// decode into `s.global_slots` (`[T, topk]`).
///
/// - `q_resid` is the MLA layer's q_a_layernorm output (`[T, 2048]`).
/// - `hidden` is the step's hidden states (`[T, 6144]`).
/// - `cos`/`sin` carry one indexer RoPE `[32]` row per token.
/// - `index_k_cache` is the paged fp8 indexer key cache (mutable — each row's
///   new k is quantized and written into it at `slot_mapping[row]`). Its
///   layout is read from the scratch's build-time shape (one source of
///   truth — a shape/layout mismatch is unrepresentable).
/// - `block_table` (`[T, block_table_stride]`) / `seq_lens` (`[T]`) describe
///   each row's paged KV region for logits + slot conversion.
///
/// `s.global_slots` ends as `topk_indices[T, topk]` (i32, `-1`-padded for
/// short context). `topk` is the attend plan's index-list length (≤ 2048): a
/// short-context step selects top-`topk` instead of top-2048 — identical
/// selection whenever `seq_len <= topk` (both are "all tokens"), which is
/// exactly the regime the caller's graph tiering guarantees.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm52_indexer_forward_into(
    ctx: &DeviceContext,
    w: &Glm52IndexerLayerWeights,
    hidden: &Rows<HIDDEN>,
    q_resid: &Rows<Q_LORA>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    index_k_cache: &mut CudaSlice<u8>,
    slot_mapping: &CudaSlice<i64>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    topk: usize,
    s: &mut Glm52IndexerScratch,
) -> Result<()> {
    let shape = s.shape;
    let t = shape.batch_size;
    // The paged cache layout the scratch shape was built from
    // (`Glm52IndexerScratch::decode_shape` copies these three fields in).
    let cache_layout = Glm52IndexerCacheLayout {
        cache_blocks: shape.num_kv_blocks,
        cache_block_size: shape.block_kv,
        cache_block_stride_bytes: shape.kv_cache_stride_bytes,
    };
    // `topk` comes from the attend plan; its 1..=GLM52_INDEXER_TOPK range is
    // pinned at compile time against the attention topk (model.rs const
    // asserts).

    // ---- projections ----
    fp8_linear_into(
        ctx,
        &w.wq_b,
        t,
        q_resid.data(),
        Some(&mut s.gemv_partial),
        &mut s.q,
    )?; // [T, 32*128]
    fp8_linear_into(
        ctx,
        &w.wk,
        t,
        hidden.data(),
        Some(&mut s.gemv_partial),
        &mut s.k_raw,
    )?; // [T, 128]
    // weights_proj: bf16 GEMM (transformers keeps weights_proj in fp32 via
    // _keep_in_fp32_modules; checkpoint stores bf16, so bf16 GEMM is the
    // closest match without a dedicated f32 GEMM path).
    // cuBLAS column-major: weights [32, 6144] row-major = [6144, 32]^T,
    // hidden [T, 6144] row-major = [6144, T] col-major. So m=32, n=T, k=6144,
    // op_a=T, op_b=N; the col-major [32, T] output IS the row-major [T, 32]
    // layout the fold consumes.
    gemm_strided_batched_bf16(
        ctx,
        true,        // transpose_a: weights [32, 6144] row-major → col-major
        false,       // transpose_b: hidden [6144, T] col-major
        INDEX_HEADS, // m = 32
        t,           // n = batch rows
        HIDDEN,      // k = 6144
        &w.weights_proj,
        HIDDEN, // lda = k (row stride of transposed weights)
        0,      // stride_a (batch=1, unused)
        hidden.data(),
        HIDDEN, // ldb = k
        0,      // stride_b
        &mut s.weights_bf16,
        INDEX_HEADS, // ldc = m
        0,           // stride_c
        1,           // batch
    )?;
    // ---- k LayerNorm (eps=1e-6, with bias), one CTA per row ----
    layer_norm_into(
        ctx,
        &s.k_raw,
        &w.k_norm_w,
        &w.k_norm_b,
        K_NORM_EPS,
        INDEX_HEAD_DIM,
        t,
        &mut s.k,
    )?;

    // ---- interleave RoPE (q[:64] per head, k[:64]; per-row position) ----
    glm52_indexer_rope_launch(ctx, &mut s.q, &mut s.k, INDEX_HEADS, t, cos, sin)?;

    // ---- q per-token-group fp8 quant ----
    // q is [T, 32, 128] flattened; quant per 128-group (one group per head).
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: t * INDEX_HEADS,
            width: INDEX_HEAD_DIM,
            group_size: FP8_BLOCK,
        },
        &s.q,
        &mut s.q_fp8,
        &mut s.q_scale,
    )?;

    // ---- weights fold: weights * q_scale * softmax_scale * n_heads^-0.5 ----
    // On-device (bit-identical multiply order to the retired host fold): the
    // two D2H readbacks + H2D here were the only mid-step stream syncs, and a
    // captured graph cannot contain them. Pure elementwise over [T, 32].
    glm52_indexer_weights_fold_launch(
        ctx,
        &s.weights_bf16,
        &s.q_scale,
        SOFTMAX_SCALE,
        N_HEADS_SCALE,
        &mut s.weights_folded,
    )?;

    // ---- k quant + cache write ----
    glm52_indexer_k_quant_and_cache_launch(
        ctx,
        Glm52IndexerCacheInsert {
            tokens: t,
            layout: cache_layout,
            scale_format: Glm52IndexerScaleFormat::F32,
        },
        &s.k,
        index_k_cache,
        slot_mapping,
    )?;

    // ---- DeepGEMM paged MQA logits ----
    // The indexer cache layout interleaves fp8 keys and f32 scales per block:
    //   [block_size * 128 fp8][block_size * 4 f32 scale] per block.
    // DeepGEMM reads both from this single buffer — the TMA descriptors
    // use kv_cache_stride_bytes to jump over the scale region between blocks,
    // and the scales pointer is computed as kv_cache + block_kv * head_dim.
    // (Matches vllm's decode-path API — no separate scales buffer needed.)
    ctx.stream
        .memcpy_dtod(&seq_lens.slice(0..t), &mut s.context_lens)?;
    glm52_deepgemm_paged_mqa_metadata_launch(
        ctx,
        shape,
        &mut s.context_lens,
        &mut s.schedule_meta,
        None,
    )?;

    // kv_cache_scales are embedded in the interleaved cache buffer — the CUDA
    // wrapper computes the scales pointer internally from kv_cache + offset.
    // No separate scales allocation needed.
    glm52_deepgemm_paged_mqa_logits_launch(
        ctx,
        shape,
        &s.q_fp8,
        index_k_cache,
        &s.weights_folded,
        &s.context_lens,
        &mut s.logits,
        block_table,
        None,
        &mut s.schedule_meta,
    )?;

    // DeepGEMM outputs bf16 logits; FlashInfer top-k expects f32.
    // The sm90 kernel already fuses per-head ReLU (fmaxf(score, 0) * weight)
    // matching transformers' F.relu(scores) — no extra ReLU needed here.
    bf16_bytes_to_f32_into(ctx, &s.logits, &mut s.logits_f32)?;

    glm52_flashinfer_topk_2048_launch(
        ctx,
        Glm52IndexerTopK {
            num_rows: t,
            top_k: topk,
            max_len: shape.logits_stride,
        },
        &s.logits_f32,
        &s.context_lens,
        &mut s.topk_offsets,
        &mut s.topk_values,
    )?;

    // ---- local top-k offsets -> global KV slots (per row) ----
    glm52_indexer_local_topk_to_slots_launch(
        ctx,
        Glm52IndexerLocalTopKToSlots {
            num_tokens: t,
            topk,
            block_size: cache_layout.cache_block_size,
            block_table_cols: shape.block_table_stride,
        },
        &s.topk_offsets,
        &s.context_lens,
        block_table,
        &mut s.global_slots,
        &mut s.topk_lens,
    )?;

    Ok(())
}

/// Allocating convenience over [`glm52_indexer_forward_into`] for the
/// oracle-gate/test paths. Returns `topk_indices[2048]` (i32, `-1`-padded).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn glm52_indexer_forward(
    ctx: &DeviceContext,
    w: &Glm52IndexerLayerWeights,
    hidden: &Rows<HIDDEN>,
    q_resid: &Rows<Q_LORA>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    index_k_cache: &mut CudaSlice<u8>,
    cache_layout: Glm52IndexerCacheLayout,
    slot_mapping: &CudaSlice<i64>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    num_sms: usize,
    max_model_len: usize,
) -> Result<CudaSlice<i32>> {
    let shape = Glm52IndexerScratch::decode_shape(
        1,
        cache_layout,
        block_table.len(),
        num_sms,
        max_model_len,
    );
    let mut s = Glm52IndexerScratch::new(ctx, shape)?;
    glm52_indexer_forward_into(
        ctx,
        w,
        hidden,
        q_resid,
        cos,
        sin,
        index_k_cache,
        slot_mapping,
        block_table,
        seq_lens,
        GLM52_INDEXER_TOPK,
        &mut s,
    )?;
    Ok(s.global_slots)
}
