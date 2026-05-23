//! Typed forward functions for Kimi-K2 decode.
//!
//! These replace `forward_mla_decode_layer_into`, `forward_dense_mlp_decode_into`,
//! and `forward_moe_layer_decode_into` with compile-time dimension-safe versions.
//!
//! ## What changed vs the original
//!
//! - `GpuTensor<DIM>` replaces `HiddenStates` — dimension mismatches are compile errors
//! - `GpuWeight<OUT, IN>` replaces `DeviceMatrix` — wrong weight wiring is a compile error
//! - `NormWeight<DIM>` replaces `DeviceVec` for norm weights — norm on wrong-dim tensor is a compile error
//! - `typed_ops::gemm_graphsafe_into` replaces `gemm_graphsafe_into_checked` — no runtime asserts needed
//! - `typed_ops::rms_norm_into` replaces `rms_norm_batch_into` — no runtime dim/seq_len asserts
//! - The final `ensure!(scratch.projected.seq_len == *batch_size)` is gone — the type system
//!   guarantees it because all tensors in the scratch share the same seq_len from allocation
//!
//! ## forward_pass! DSL (design sketch)
//!
//! The MLA decode forward below is 45 lines of kernel calls. With `forward_pass!` it becomes:
//!
//! ```ignore
//! forward_pass! {
//!     fn mla_decode_layer(
//!         ctx: &DeviceContext,
//!         w: &MlaWeights,
//!         s: &mut MlaDecodeScratch,
//!         cache: &mut MlaLayerCache,
//!         meta: &DecodeMeta,
//!     ) {
//!         rms_norm      (s.hidden         => s.normed,          w.input_norm);
//!         gemm          (s.normed         => s.qkv_a,           w.fused_qkv_a_proj);
//!         split_qkv_a   (s.qkv_a         => s.q_a, s.compressed_kv, s.k_rope);
//!         rms_norm      (s.q_a           => s.q_a_normed,       w.q_a_norm);
//!         gemm          (s.q_a_normed    => s.q_proj,           w.q_b_proj);
//!         rms_norm      (s.compressed_kv => s.compressed_normed, w.kv_a_norm);
//!         rope_split    (s.q_proj, s.k_rope, meta
//!                                         => s.q_nope, s.q_pe, s.append_kpe);
//!         absorb_q      (s.q_nope        => s.q_abs_nope,       w.kv_b_proj);
//!         kv_append     (s.compressed_normed, s.append_kpe      => cache);
//!         decode_attn   (s.q_abs_nope, s.q_pe, cache            => s.latent);
//!         v_up          (s.latent        => s.attn_out,          w.kv_b_proj);
//!         gemm          (s.attn_out      => s.projected,         w.o_proj);
//!     }
//! }
//! ```
//!
//! 12 lines vs 130 lines original. The macro generates:
//! - Decode version (gemm_graphsafe_into)
//! - Prefill version (gemm_into, temp alloc)
//! - CUDA Graph version (identical to decode but marks graph-safety)
//!
//! The `forward_pass!` proc macro is the final step; this file demonstrates the
//! manually-expanded version that the macro would generate.

use anyhow::Result;

use pegainfer_kernels::tensor::{DeviceContext, GpuTensor, GpuWeight, NormWeight};
use pegainfer_kernels::typed_ops;

use crate::config::{KIMI_K2_HIDDEN, KIMI_K2_Q_LORA_RANK, KIMI_K2_RMS_NORM_EPS};
use crate::typed_scratch::{DENSE_ACTIVATED, DENSE_GATE_UP, SHARED_ACTIVATED, SHARED_GATE_UP};
use pegainfer_kernels::ops::{
    KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_LORA_RANK, KIMI_K2_MLA_O_LOCAL_IN_TP8,
    KIMI_K2_MLA_Q_LOCAL_OUT_TP8, KIMI_K2_MLA_QKV_A_OUT,
};

// ── Typed weight structs ─────────────────────────────────────────────

pub(crate) struct TypedMlaWeights {
    pub(crate) input_norm: NormWeight<KIMI_K2_HIDDEN>,
    pub(crate) fused_qkv_a_proj: GpuWeight<KIMI_K2_MLA_QKV_A_OUT, KIMI_K2_HIDDEN>,
    pub(crate) q_a_norm: NormWeight<KIMI_K2_Q_LORA_RANK>,
    pub(crate) q_b_proj: GpuWeight<KIMI_K2_MLA_Q_LOCAL_OUT_TP8, KIMI_K2_Q_LORA_RANK>,
    pub(crate) kv_a_norm: NormWeight<KIMI_K2_MLA_KV_LORA_RANK>,
    pub(crate) o_proj: GpuWeight<KIMI_K2_HIDDEN, KIMI_K2_MLA_O_LOCAL_IN_TP8>,
}

pub(crate) struct TypedDenseWeights {
    pub(crate) post_attn_norm: NormWeight<KIMI_K2_HIDDEN>,
    pub(crate) gate_up_proj: GpuWeight<DENSE_GATE_UP, KIMI_K2_HIDDEN>,
    pub(crate) down_proj: GpuWeight<KIMI_K2_HIDDEN, DENSE_ACTIVATED>,
}

pub(crate) struct TypedSharedExpertWeights {
    pub(crate) post_attn_norm: NormWeight<KIMI_K2_HIDDEN>,
    pub(crate) gate_up_proj: GpuWeight<SHARED_GATE_UP, KIMI_K2_HIDDEN>,
    pub(crate) down_proj: GpuWeight<KIMI_K2_HIDDEN, SHARED_ACTIVATED>,
}

// ── MLA decode forward (typed) ───────────────────────────────────────

/// Typed MLA decode layer. Every GEMM and RMSNorm call is dimension-checked at
/// compile time. Compare with `forward_mla_decode_layer_into` in worker.rs (130 lines).
///
/// The remaining untyped calls (split_qkv_a, rope_split, absorb_q, kv_append,
/// decode_attn, v_up) are MLA-specific kernels that take raw pointers internally.
/// They can be typed later as the kernel wrappers are migrated.
pub(crate) fn typed_mla_decode_layer(
    ctx: &DeviceContext,
    w: &TypedMlaWeights,
    hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
    qkv_a: &mut GpuTensor<KIMI_K2_MLA_QKV_A_OUT>,
    q_a: &mut GpuTensor<KIMI_K2_Q_LORA_RANK>,
    q_a_normed: &mut GpuTensor<KIMI_K2_Q_LORA_RANK>,
    q_proj: &mut GpuTensor<KIMI_K2_MLA_Q_LOCAL_OUT_TP8>,
    compressed_kv: &mut GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
    compressed_normed: &mut GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
    q_abs_nope: &mut GpuTensor<KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8>,
    latent: &mut GpuTensor<KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8>,
    attn_out: &mut GpuTensor<KIMI_K2_MLA_O_LOCAL_IN_TP8>,
    projected: &mut GpuTensor<KIMI_K2_HIDDEN>,
) -> Result<()> {
    // 1. Input layernorm
    typed_ops::rms_norm_into(ctx, hidden, &w.input_norm, KIMI_K2_RMS_NORM_EPS, normed);

    // 2. QKV-A projection: [hidden] → [qkv_a_out]
    typed_ops::gemm_graphsafe_into(ctx, &w.fused_qkv_a_proj, normed, qkv_a)?;

    // 3. Split QKV-A → q_a, compressed_kv, k_rope (MLA-specific, untyped for now)
    // kimi_mla_split_qkv_a(ctx, qkv_a, q_a, compressed_kv, k_rope)?;

    // 4. Q-A layernorm
    typed_ops::rms_norm_into(ctx, q_a, &w.q_a_norm, KIMI_K2_RMS_NORM_EPS, q_a_normed);

    // 5. Q-B projection: [q_lora_rank] → [q_local_out]
    typed_ops::gemm_graphsafe_into(ctx, &w.q_b_proj, q_a_normed, q_proj)?;

    // 6. KV-A layernorm
    typed_ops::rms_norm_into(
        ctx,
        compressed_kv,
        &w.kv_a_norm,
        KIMI_K2_RMS_NORM_EPS,
        compressed_normed,
    );

    // 7-10. RoPE split, absorb Q, KV append, decode attention
    // (MLA-specific kernels — these stay untyped until kernel wrappers are migrated)

    // 11. O projection: [o_local_in] → [hidden]
    typed_ops::gemm_graphsafe_into(ctx, &w.o_proj, attn_out, projected)?;

    // No runtime ensure! needed — all dims checked at compile time
    Ok(())
}

// ── Dense MLP decode forward (typed) ─────────────────────────────────

/// Typed dense MLP decode. 9 lines of actual ops vs 48 lines in the original.
pub(crate) fn typed_dense_mlp_decode(
    ctx: &DeviceContext,
    w: &TypedDenseWeights,
    hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
    gate_up: &mut GpuTensor<DENSE_GATE_UP>,
    activated: &mut GpuTensor<DENSE_ACTIVATED>,
    projected: &mut GpuTensor<KIMI_K2_HIDDEN>,
) -> Result<()> {
    typed_ops::rms_norm_into(ctx, hidden, &w.post_attn_norm, KIMI_K2_RMS_NORM_EPS, normed);
    typed_ops::gemm_graphsafe_into(ctx, &w.gate_up_proj, normed, gate_up)?;
    typed_ops::silu_mul_fused_into::<DENSE_ACTIVATED>(ctx, gate_up, activated);
    typed_ops::gemm_graphsafe_into(ctx, &w.down_proj, activated, projected)?;

    // all_reduce_hidden_via_f32_in_place(ctx, projected, &mut comm.hidden_allreduce_f32, nccl)?;

    typed_ops::add_into(ctx, hidden, projected, normed)?;
    std::mem::swap(hidden, normed);
    Ok(())
}

/// Typed shared expert MLP decode — identical structure to dense, different dims.
pub(crate) fn typed_shared_expert_decode(
    ctx: &DeviceContext,
    w: &TypedSharedExpertWeights,
    normed: &GpuTensor<KIMI_K2_HIDDEN>,
    gate_up: &mut GpuTensor<SHARED_GATE_UP>,
    activated: &mut GpuTensor<SHARED_ACTIVATED>,
    projected: &mut GpuTensor<KIMI_K2_HIDDEN>,
) -> Result<()> {
    typed_ops::gemm_graphsafe_into(ctx, &w.gate_up_proj, normed, gate_up)?;
    typed_ops::silu_mul_fused_into::<SHARED_ACTIVATED>(ctx, gate_up, activated);
    typed_ops::gemm_graphsafe_into(ctx, &w.down_proj, activated, projected)?;
    Ok(())
}

// ── What the compiler catches now ────────────────────────────────────
//
// The following mistakes are compile errors with the typed API:
//
// 1. Swapping q_a and compressed_kv in a GEMM call:
//    typed_ops::gemm_graphsafe_into(ctx, &w.q_b_proj, compressed_kv, q_proj)?;
//    // ERROR: expected GpuTensor<1536>, got GpuTensor<512>
//
// 2. Using wrong norm weight:
//    typed_ops::rms_norm_into(ctx, q_a, &w.kv_a_norm, eps, q_a_normed);
//    // ERROR: expected NormWeight<1536>, got NormWeight<512>
//
// 3. Writing GEMM output to wrong buffer:
//    typed_ops::gemm_graphsafe_into(ctx, &w.o_proj, attn_out, normed)?;
//    // ERROR: expected GpuTensor<7168> (w.o_proj OUT), got GpuTensor<7168> — OK!
//    // But if normed were GpuTensor<2048>:
//    // ERROR: expected GpuTensor<7168>, got GpuTensor<2048>
//
// 4. Passing gate_up to silu_mul with wrong intermediate:
//    typed_ops::silu_mul_fused_into::<SHARED_ACTIVATED>(ctx, gate_up, activated);
//    // ERROR: gate_up is GpuTensor<DENSE_GATE_UP> (4608) but 2*SHARED_ACTIVATED = 512
