//! Shared GPU operation wrappers and kernel-crate re-exports.

mod attention;
pub mod call_spec;
#[cfg(feature = "kernel-call-trace")]
pub mod call_trace;
mod paged_plan;
#[cfg(feature = "kernel-call-trace")]
mod traced;

pub use attention::{
    paged_attention_batch_decode_hd256_into, paged_attention_batch_decode_into,
    paged_attention_batch_decode_split_kv_into, prefill_attention_paged_into,
};
pub use openinfer_kernels::ops::{
    GEMM_LT_MAX_N, LoraDecodeGroupedProjection, RaggedPrefillPlan,
    accumulate_bf16_token_scaled_to_f32_into, add_batch, add_batch_into, argmax,
    argmax_batch_bf16_into, batch_prefill_ragged_nhd_noncausal_into, bf16_hidden_to_f32_into,
    embedding_decode_into, extract_vec, extract_vec_into, extract_vec_ref, extract_vec_ref_into,
    f32_to_bf16_hidden_into, fused_add_rms_norm_into, gather_hidden_tokens_into, gemm,
    gemm_graphsafe_into_checked, gemm_graphsafe_ref_into_checked, gemm_into_checked, gemm_lt_tune,
    gemm_per_token, gemv, k_norm_rope_batch_decode_into, linear,
    lora_decode_fused_delta_group3_into, lora_decode_fused_delta_into, pack_lora_b_rows_into,
    qk_norm_partial_rope_batched_decode_hd256_into, rms_norm, rms_norm_batch_offset_into,
    rms_norm_gated_batch_into, rms_norm_into, rms_norm_offset_into, scale_f32_in_place,
    scaled_add_batch_into, scaled_add_rows_indexed_into, scaled_add_rows_into,
    scaled_add_rows_token_range_into, silu_mul_batch, silu_mul_batch_into,
    single_prefill_nhd_noncausal_into, strided_segment_copy_into, write_vec_into,
};
#[cfg(not(feature = "kernel-call-trace"))]
pub use openinfer_kernels::ops::{
    embedding_batch, fused_add_rms_norm_batch_into, gemm_into, gemm_rows_into,
    gemm_token_range_into_checked, qk_norm_rope_batch_decode_into, rms_norm_batch_into,
    silu_mul_fused_batch_into,
};
pub use paged_plan::PrefillPagedPlan;
#[cfg(feature = "kernel-call-trace")]
pub use traced::{
    embedding_batch, fused_add_rms_norm_batch_into, gemm_into, gemm_rows_into,
    gemm_token_range_into_checked, qk_norm_rope_batch_decode_into, rms_norm_batch_into,
    silu_mul_fused_batch_into,
};
