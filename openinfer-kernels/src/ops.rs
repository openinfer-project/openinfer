//! GPU operations on device tensors.

mod attention;
#[cfg(feature = "moe")]
mod deepep;
#[cfg(feature = "deepseek-v2-lite")]
mod deepseek_v2_lite;
mod elementwise;
mod embedding;
#[cfg(feature = "glm52")]
mod glm52;
#[cfg(feature = "kimi-k2")]
mod kimi_k2;
mod linear;
mod lora;
mod norm;
mod sampling;

pub use attention::{
    PrefillPagedPlan, SUPPORTED_GQA_GROUP_SIZES, dflash_qk_norm_rope_into,
    paged_attention_batch_decode_hd256_into, paged_attention_batch_decode_into,
    paged_attention_batch_decode_split_kv_into, prefill_attention_paged_into,
    qk_norm_partial_rope_batched_decode_hd256_into, qk_norm_rope_batch_decode_into,
    single_prefill_nhd_noncausal_into,
};
#[cfg(feature = "moe")]
pub use deepep::{
    DeepEp, DeepEpDispatchScratch, DeepEpPrefillCounts, deepep_info, deepep_unique_id,
};
#[cfg(feature = "glm52")]
pub use deepep::{Glm52DeepEp, glm52_deepep_info, glm52_deepep_unique_id};
#[cfg(feature = "deepseek-v2-lite")]
pub use deepseek_v2_lite::*;
pub use elementwise::{
    accumulate_bf16_token_scaled_to_f32_into, add_batch, add_batch_into, add_into,
    bf16_bytes_to_f32_into, bf16_hidden_to_f32_into, copy_hidden_rows_into,
    copy_hidden_rows_raw_into, copy_hidden_token_range_into, extract_vec, extract_vec_into,
    extract_vec_ref, extract_vec_ref_into, f32_to_bf16_hidden_into, gather_hidden_tokens_into,
    repeat_f32_for_reduce_scatter_into, scale_f32_in_place, scaled_add_batch_into,
    scaled_add_rows_indexed_into, scaled_add_rows_into, scaled_add_rows_token_range_into,
    silu_mul_batch, silu_mul_batch_into, silu_mul_fused_batch_into, write_vec_into,
};
pub use embedding::{
    embedding_batch, embedding_batch_vocab_shard, embedding_decode_into, embedding_rows_into,
};
#[cfg(feature = "glm52")]
pub use glm52::*;
#[cfg(feature = "kimi-k2")]
pub use kimi_k2::*;
pub use linear::{
    GEMM_LT_MAX_N, NumericPolicy, gemm, gemm_graphsafe_into_checked,
    gemm_graphsafe_ref_into_checked, gemm_into, gemm_into_checked, gemm_lt_pin_check,
    gemm_lt_pin_into_checked, gemm_lt_pin_tune, gemm_lt_pin_warmup, gemm_lt_tune, gemm_per_token,
    gemm_per_token_into_checked, gemm_rows_into, gemm_rows_into_checked, gemm_strided_batched_bf16,
    gemm_token_range_into_checked, gemv, linear, numeric_policy, per_token_served, pin_served,
    reset_numeric_policy_counters, set_numeric_policy,
};
pub use lora::{
    LoraDecodeGroupedProjection, lora_decode_fused_delta_group3_into, lora_decode_fused_delta_into,
    pack_lora_b_rows_into,
};
pub use norm::{
    fused_add_rms_norm_batch_into, fused_add_rms_norm_into, fused_add_rms_norm_round_batch_into,
    fused_add_rms_norm_round_into, layer_norm_into, rms_norm, rms_norm_batch_into,
    rms_norm_batch_offset_into, rms_norm_gated_batch_into, rms_norm_into, rms_norm_offset_into,
    rms_norm_rows_into,
};
pub use sampling::{
    BatchSamplingRow, BatchSamplingScratch, argmax, argmax_batch_bf16_into,
    argmax_batch_bf16_split_indexed_into, argmax_batch_bf16_split_partials_len, argmax_bf16_into,
    argmax_bf16_split_into, flashinfer_top1_batch_into, flashinfer_top1_row_states_bytes,
    gpu_sample_batch_into, markov_step_argmax_into,
};

/// Calling thread's last FFI exception message, ready to append to an error;
/// empty unless `result` is the -1 sentinel set by the C++ guard. Public for
/// crates that call guarded FFI entries directly instead of through a wrapper.
pub fn ffi_exception_message(result: i32) -> String {
    if result != -1 {
        return String::new();
    }
    let ptr = unsafe { crate::ffi::openinfer_kernels_last_error() };
    if ptr.is_null() {
        return String::new();
    }
    let text = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy();
    if text.is_empty() {
        String::new()
    } else {
        format!(": {text}")
    }
}
