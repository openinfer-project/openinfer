//! Shared GPU operation wrappers and kernel-crate re-exports.

mod attention;
pub mod call_spec;
#[cfg(feature = "kernel-call-trace")]
pub mod call_trace;
mod paged_plan;
mod sampling;
#[cfg(feature = "kernel-call-trace")]
mod traced;

pub use attention::{
    paged_attention_batch_decode_hd256_into, paged_attention_batch_decode_into,
    paged_attention_batch_decode_split_kv_into, prefill_attention_paged_into,
};
pub use paged_plan::PrefillPagedPlan;
pub use pegainfer_kernels::ops::{
    LoraDecodeGroupedProjection, accumulate_bf16_token_scaled_to_f32_into, add_batch,
    add_batch_into, bf16_hidden_to_f32_into, embedding_decode_into, extract_vec, extract_vec_into,
    f32_to_bf16_hidden_into, fused_add_rms_norm_into, gather_hidden_tokens_into, gemm,
    gemm_into_checked, gemm_per_token, gemv, linear, lora_decode_fused_delta_group3_into,
    lora_decode_fused_delta_into, pack_lora_b_rows_into,
    qk_norm_partial_rope_batched_decode_hd256_into, rms_norm, rms_norm_batch_offset_into,
    rms_norm_gated_batch_into, rms_norm_into, rms_norm_offset_into, scale_f32_in_place,
    scaled_add_batch_into, scaled_add_rows_indexed_into, scaled_add_rows_into,
    scaled_add_rows_token_range_into, silu_mul_batch, silu_mul_batch_into, write_vec_into,
};
#[cfg(not(feature = "kernel-call-trace"))]
pub use pegainfer_kernels::ops::{
    embedding_batch, fused_add_rms_norm_batch_into, gemm_into, gemm_rows_into,
    gemm_token_range_into_checked, qk_norm_rope_batch_decode_into, rms_norm_batch_into,
    silu_mul_fused_batch_into,
};
pub use sampling::{
    argmax, argmax_batch_bf16_indexed_into, argmax_batch_bf16_into,
    flashinfer_topk_row_states_bytes, gpu_sample, gpu_sample_into, select_batch_tokens_into,
};
#[cfg(feature = "kernel-call-trace")]
pub use traced::{
    embedding_batch, fused_add_rms_norm_batch_into, gemm_into, gemm_rows_into,
    gemm_token_range_into_checked, qk_norm_rope_batch_decode_into, rms_norm_batch_into,
    silu_mul_fused_batch_into,
};
