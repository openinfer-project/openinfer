//! GLM5.2 DSA indexer decode forward (bs=1): produces `topk_indices[2048]`.
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
//!   |     +-- interleave RoPE (q[:64], k[:64], cos/sin[32])
//!   |     +-- q per-token-group fp8 quant -> q_fp8[32*128], q_scale[32]
//!   |     +-- weights fold: weights * q_scale * softmax_scale * n_heads^-0.5
//!   |
//! hidden[6144]
//!   +-- wk (fp8 linear) -> k_raw[128]
//!   +-- weights_proj (fp8 linear) -> weights[32]
//!   +-- k quant + cache write (glm52_indexer_k_quant_and_cache)
//!   |
//!   +-- DeepGEMM paged MQA logits (fuses ReLU + per-head weighting)
//!   +-- FlashInfer deterministic top-k K=2048
//!   +-- local top-k offsets -> global KV slots
//! ```

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;

use openinfer_kernels::ops::{
    GLM52_INDEXER_HEAD_DIM, GLM52_INDEXER_TOPK, Glm52DeepGemmMqaLogitsShape,
    Glm52IndexerCacheInsert, Glm52IndexerCacheLayout, Glm52IndexerLocalTopKToSlots,
    Glm52IndexerScaleFormat, Glm52IndexerTopK, Glm52MoeQuantShape, bf16_bytes_to_f32_into,
    glm52_deepgemm_paged_mqa_logits_launch, glm52_deepgemm_paged_mqa_metadata_launch,
    glm52_flashinfer_topk_2048_launch, glm52_fp8_per_token_group_quant_bf16_launch,
    glm52_indexer_k_quant_and_cache_launch, glm52_indexer_local_topk_to_slots_launch,
    glm52_indexer_rope_launch, layer_norm_into,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::fp8::{FP8_BLOCK, Glm52ProjBytes, ProjWeight, fp8_linear};

const HIDDEN: usize = 6144;
const Q_LORA: usize = 2048;
const INDEX_HEADS: usize = 32;
const INDEX_HEAD_DIM: usize = 128;
// vllm: softmax_scale = head_dim ** -0.5 = 128 ** -0.5
const SOFTMAX_SCALE: f32 = 0.088_388_35; // 1.0 / 128.0f32.sqrt()
// vllm: n_heads ** -0.5 = 32 ** -0.5
const N_HEADS_SCALE: f32 = 0.176_776_7; // 1.0 / 32.0f32.sqrt()
const K_NORM_EPS: f32 = 1.0e-6;

/// One DSA indexer layer's weights, device-resident.
pub(crate) struct Glm52IndexerLayerWeights {
    wq_b: ProjWeight,         // [32*128, 2048]
    wk: ProjWeight,           // [128, 6144]
    weights_proj: ProjWeight, // [128, 6144] — padded from [32, 6144] (TRTLLM rejects n=32)
    weights_proj_bf16: Vec<bf16>, // host-side [32, 6144] — f32 matmul for oracle parity
    k_norm_w: CudaSlice<f32>, // [128] — LayerNorm gamma (f32 for FlashInfer)
    k_norm_b: CudaSlice<f32>, // [128] — LayerNorm beta  (f32 for FlashInfer)
}

impl Glm52IndexerLayerWeights {
    /// Build from raw checkpoint bytes (the test path). Same pattern as
    /// `Glm52MlaLayerWeights::from_host`. `weights_proj` is a bf16 `[32, 6144]`
    /// tensor (transformers keeps it in fp32/bf16, NOT fp8 block-scaled). It
    /// is host-side quantized to fp8 per-128-block and padded to [128, 6144]
    /// because TRTLLM rejects n=32.
    #[allow(clippy::too_many_arguments)]
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
        let proj = quant_and_pad_weights_proj(ctx, weights_proj_bf16)?;
        let proj_bf16: Vec<bf16> = unsafe {
            std::slice::from_raw_parts(
                weights_proj_bf16.as_ptr().cast::<bf16>(),
                INDEX_HEADS * HIDDEN,
            )
        }
        .to_vec();
        let norm_w = upcast_bf16_to_f32(ctx, k_norm_w)?;
        let norm_b = upcast_bf16_to_f32(ctx, k_norm_b)?;
        Ok(Self {
            wq_b: w,
            wk: k,
            weights_proj: proj,
            weights_proj_bf16: proj_bf16,
            k_norm_w: norm_w,
            k_norm_b: norm_b,
        })
    }
}

/// Quantize bf16 `weights_proj` [32, 6144] to fp8 per-128-block, then pad to
/// [128, 6144] (TRTLLM rejects n=32). The extra 96 rows are zero-padded;
/// the forward slices the first 32 outputs. This matches the checkpoint's
/// fp8 quant contract: scale = group_amax / 448, fp8 e4m3.
fn quant_and_pad_weights_proj(ctx: &DeviceContext, bf16_bytes: &[u8]) -> Result<ProjWeight> {
    let n_orig = INDEX_HEADS; // 32
    let n = INDEX_HEAD_DIM; // 128 (padded)
    let k = HIDDEN;
    let bf16_vals: &[bf16] =
        unsafe { std::slice::from_raw_parts(bf16_bytes.as_ptr().cast::<bf16>(), n_orig * k) };

    let mut fp8_weights = vec![0u8; n * k];
    let scale_rows = n.div_ceil(FP8_BLOCK); // 1
    let scale_cols = k.div_ceil(FP8_BLOCK); // 48
    let mut scales = vec![0.0f32; scale_rows * scale_cols];

    // The TRTLLM fp8 format has scale shape [n/128, k/128] = [1, 48].
    // All 32 original rows must share a single scale per 128-element column
    // group. Use the max amax across all rows to avoid clipping.
    for col_group in 0..scale_cols {
        let start = col_group * FP8_BLOCK;
        let end = (start + FP8_BLOCK).min(k);
        let mut amax = 0.0f32;
        for row in 0..n_orig {
            for j in start..end {
                amax = amax.max(bf16_vals[row * k + j].to_f32().abs());
            }
        }
        let scale = amax.max(1e-4) / 448.0;
        scales[col_group] = scale;
        for row in 0..n_orig {
            for j in start..end {
                let val = bf16_vals[row * k + j].to_f32();
                let q = (val / scale).clamp(-448.0, 448.0);
                fp8_weights[row * k + j] = float_to_fp8_e4m3(q);
            }
        }
    }

    let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();

    let mut weight_dev = ctx.stream.alloc_zeros::<u8>(n * k)?;
    let mut scale_dev = ctx.stream.alloc_zeros::<u8>(scale_bytes.len())?;
    ctx.stream.memcpy_htod(&fp8_weights, &mut weight_dev)?;
    ctx.stream.memcpy_htod(&scale_bytes, &mut scale_dev)?;
    Ok(ProjWeight {
        weight: weight_dev,
        scale: scale_dev,
        n,
        k,
    })
}

/// Convert a float to fp8 e4m3 (saturating). Mirrors __nv_cvt_float_to_fp8
/// with SV_INF_NAN_MODE_OVERFLOW and __NV_E4M3.
fn float_to_fp8_e4m3(val: f32) -> u8 {
    let clamped = val.clamp(-448.0, 448.0);
    let bits = clamped.to_bits();
    let sign = (bits >> 31) & 1;
    let abs = bits & 0x7FFF_FFFF;
    // f32 to e4m3: exponent bias 7 (f32) vs 7 (e4m3), but e4m3 has 3 mantissa bits
    let f32_exp = (abs >> 23) as i32;
    let f32_man = abs & 0x7F_FFFF;
    if f32_exp == 0 {
        // subnormal or zero
        if f32_man == 0 {
            return (sign << 7) as u8;
        }
        // subnormal: value = 2^-6 * (man / 2^23)
        let val = f32::from_bits(abs);
        let scaled = val * (1u32 << 6) as f32 * 8.0 / 448.0 * 448.0;
        let q = scaled.round() as i32;
        let q = q.clamp(0, 7);
        return ((sign << 7) | q as u32) as u8;
    }
    if f32_exp >= 128 + 8 {
        // overflow → saturate to max (448.0 = 0b0_1110_111)
        return ((sign << 7) | 0b1110_111) as u8;
    }
    let e4m3_exp = f32_exp - 127 + 7;
    if e4m3_exp < 0 {
        // underflow → zero
        return (sign << 7) as u8;
    }
    let e4m3_man = (f32_man >> 20) & 0x7;
    ((sign << 7) | ((e4m3_exp as u32) << 3) | e4m3_man) as u8
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
    ensure!(hidden.len() >= HIDDEN, "GLM5.2 indexer cache_fill hidden too small");

    let k_raw = fp8_linear(ctx, &w.wk, hidden)?; // [128]
    let mut k = ctx.stream.alloc_zeros::<bf16>(INDEX_HEAD_DIM)?;
    layer_norm_into(ctx, &k_raw, &w.k_norm_w, &w.k_norm_b, K_NORM_EPS, &mut k)?;

    // RoPE: the kernel applies to both q and k; use a dummy q buffer.
    let mut q_dummy = ctx.stream.alloc_zeros::<bf16>(INDEX_HEADS * INDEX_HEAD_DIM)?;
    glm52_indexer_rope_launch(ctx, &mut q_dummy, &mut k, INDEX_HEADS, cos, sin)?;

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

/// DSA indexer decode forward for one token (bs=1): computes sparse top-k
/// slot indices for the FlashMLA sparse decode.
///
/// - `q_resid` is the MLA layer's q_a_layernorm output (`[2048]`).
/// - `hidden` is the current token's hidden state (`[6144]`).
/// - `cos`/`sin` are the indexer RoPE table first half (`[32]`).
/// - `index_k_cache` is the paged fp8 indexer key cache (mutable — the new
///   token's k is quantized and written into it at `slot_mapping[0]`).
/// - `block_table` / `seq_lens` describe the paged KV layout for logits +
///   slot conversion.
///
/// Returns `topk_indices[2048]` (i32, `-1`-padded for short context).
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm52_indexer_forward(
    ctx: &DeviceContext,
    w: &Glm52IndexerLayerWeights,
    hidden: &CudaSlice<bf16>,
    q_resid: &CudaSlice<bf16>,
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
    ensure!(hidden.len() >= HIDDEN, "GLM5.2 indexer hidden too small");
    ensure!(q_resid.len() >= Q_LORA, "GLM5.2 indexer q_resid too small");

    // ---- projections ----
    let q = fp8_linear(ctx, &w.wq_b, q_resid)?; // [32*128 = 4096]
    let k_raw = fp8_linear(ctx, &w.wk, hidden)?; // [128]
    // weights_proj: f32 host matmul (matches oracle precision — transformers
    // keeps weights_proj in fp32 via _keep_in_fp32_modules).
    let hidden_host = ctx.stream.clone_dtoh(&hidden.slice(0..HIDDEN))?;
    let weights_raw: Vec<f32> = (0..INDEX_HEADS)
        .map(|h| {
            let mut dot = 0.0f32;
            let row = &w.weights_proj_bf16[h * HIDDEN..(h + 1) * HIDDEN];
            for j in 0..HIDDEN {
                dot += hidden_host[j].to_f32() * row[j].to_f32();
            }
            dot
        })
        .collect();

    // ---- k LayerNorm (eps=1e-6, with bias) ----
    let mut k = ctx.stream.alloc_zeros::<bf16>(INDEX_HEAD_DIM)?;
    layer_norm_into(ctx, &k_raw, &w.k_norm_w, &w.k_norm_b, K_NORM_EPS, &mut k)?;

    // ---- interleave RoPE (q[:64] per head, k[:64]) ----
    let mut q = q; // mut for in-place RoPE
    glm52_indexer_rope_launch(ctx, &mut q, &mut k, INDEX_HEADS, cos, sin)?;

    // ---- q per-token-group fp8 quant ----
    // q is [32, 128] flattened; quant per 128-group (one group per head).
    let mut q_fp8 = ctx.stream.alloc_zeros::<u8>(INDEX_HEADS * INDEX_HEAD_DIM)?;
    let mut q_scale = ctx.stream.alloc_zeros::<f32>(INDEX_HEADS)?;
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: INDEX_HEADS,
            width: INDEX_HEAD_DIM,
            group_size: FP8_BLOCK,
        },
        &q,
        &mut q_fp8,
        &mut q_scale,
    )?;

    // ---- weights fold: weights * q_scale * softmax_scale * n_heads^-0.5 ----
    // 32 elements — host-side math is cheaper than a kernel launch.
    let q_scale_host = ctx.stream.clone_dtoh(&q_scale)?;
    let mut weights_folded = vec![0.0f32; INDEX_HEADS];
    for h in 0..INDEX_HEADS {
        weights_folded[h] =
            weights_raw[h] * q_scale_host[h] * SOFTMAX_SCALE * N_HEADS_SCALE;
    }
    let mut weights_out = ctx.stream.alloc_zeros::<f32>(INDEX_HEADS)?;
    ctx.stream.memcpy_htod(&weights_folded, &mut weights_out)?;

    // ---- k quant + cache write ----
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

    // ---- DeepGEMM paged MQA logits ----
    // The indexer cache layout interleaves fp8 keys and f32 scales per block:
    //   [block_size * 128 fp8][block_size * 4 f32 scale] per block.
    // DeepGEMM reads both from this single buffer — the TMA descriptors
    // use kv_cache_stride_bytes to jump over the scale region between blocks,
    // and the scales pointer is computed as kv_cache + block_kv * head_dim.
    // (Matches vllm's decode-path API — no separate scales buffer needed.)
    let shape = Glm52DeepGemmMqaLogitsShape {
        batch_size: 1,
        next_n: 1,
        num_heads: INDEX_HEADS,
        head_dim: GLM52_INDEXER_HEAD_DIM,
        num_kv_blocks: cache_layout.cache_blocks,
        block_kv: cache_layout.cache_block_size,
        kv_cache_stride_bytes: cache_layout.cache_block_stride_bytes,
        is_context_lens_2d: false,
        is_varlen: false,
        logits_stride: max_model_len.next_multiple_of(256),
        block_table_stride: block_table.len(),
        num_sms,
    };
    let mut schedule_meta = ctx
        .stream
        .alloc_zeros::<i32>(shape.schedule_metadata_len())?;
    let mut context_lens = ctx.stream.alloc_zeros::<i32>(1)?;
    ctx.stream
        .memcpy_dtod(&seq_lens.slice(0..1), &mut context_lens)?;
    glm52_deepgemm_paged_mqa_metadata_launch(
        ctx,
        shape,
        &mut context_lens,
        &mut schedule_meta,
        None,
    )?;

    // kv_cache_scales are embedded in the interleaved cache buffer — the CUDA
    // wrapper computes the scales pointer internally from kv_cache + offset.
    // No separate scales allocation needed.

    let logits_elems = shape.batch_size * shape.next_n * shape.logits_stride;
    let mut logits = ctx.stream.alloc_zeros::<u8>(logits_elems * 2)?; // bf16
    glm52_deepgemm_paged_mqa_logits_launch(
        ctx,
        shape,
        &q_fp8,
        index_k_cache,
        &weights_out,
        &context_lens,
        &mut logits,
        block_table,
        None,
        &mut schedule_meta,
    )?;

    // DeepGEMM outputs bf16 logits; FlashInfer top-k expects f32.
    // Cast bf16 logits buffer → f32 before top-k.
    let mut logits_f32 = ctx.stream.alloc_zeros::<f32>(logits_elems)?;
    bf16_bytes_to_f32_into(ctx, &logits, &mut logits_f32)?;

    // Apply ReLU: transformers applies F.relu(scores) before topk.
    // sm100 DeepGEMM fuses this via cvt.relu, but sm90 does not.
    let logits_host = ctx.stream.clone_dtoh(&logits_f32)?;
    let relu_host: Vec<f32> = logits_host.iter().map(|&v| v.max(0.0)).collect();
    ctx.stream.memcpy_htod(&relu_host, &mut logits_f32)?;

    let mut topk_offsets = ctx.stream.alloc_zeros::<i32>(GLM52_INDEXER_TOPK)?;
    let mut topk_values = ctx.stream.alloc_zeros::<f32>(GLM52_INDEXER_TOPK)?;
    glm52_flashinfer_topk_2048_launch(
        ctx,
        Glm52IndexerTopK {
            num_rows: 1,
            top_k: GLM52_INDEXER_TOPK,
            max_len: shape.logits_stride,
        },
        &logits_f32,
        &context_lens,
        &mut topk_offsets,
        &mut topk_values,
    )?;

    // ---- local top-k offsets -> global KV slots ----
    let mut global_slots = ctx.stream.alloc_zeros::<i32>(GLM52_INDEXER_TOPK)?;
    let mut topk_lens = ctx.stream.alloc_zeros::<i32>(1)?;
    glm52_indexer_local_topk_to_slots_launch(
        ctx,
        Glm52IndexerLocalTopKToSlots {
            num_tokens: 1,
            topk: GLM52_INDEXER_TOPK,
            block_size: cache_layout.cache_block_size,
            block_table_cols: block_table.len(),
        },
        &topk_offsets,
        &context_lens,
        block_table,
        &mut global_slots,
        &mut topk_lens,
    )?;

    Ok(global_slots)
}
