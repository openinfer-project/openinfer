//! GPU operations on device tensors.

mod attention;
#[cfg(feature = "kimi-k2")]
mod deepep;
mod elementwise;
mod embedding;
#[cfg(feature = "kimi-k2")]
mod kimi_k2;
mod linear;
mod lora;
mod norm;
mod sampling;

pub use attention::{
    PrefillPagedPlan, paged_attention_batch_decode_hd256_into, paged_attention_batch_decode_into,
    paged_attention_batch_decode_split_kv_into, prefill_attention_paged_into,
    qk_norm_partial_rope_batched_decode_hd256_into, qk_norm_rope_batch_decode_into,
};
#[cfg(feature = "kimi-k2")]
pub use deepep::{
    DeepEp, DeepEpDispatchScratch, DeepEpPrefillCounts, deepep_info, deepep_unique_id,
};
pub use elementwise::{
    accumulate_bf16_token_scaled_to_f32_into, add_batch, add_batch_into, bf16_hidden_to_f32_into,
    extract_vec, extract_vec_into, f32_to_bf16_hidden_into, gather_hidden_tokens_into,
    repeat_f32_for_reduce_scatter_into, scale_f32_in_place, scaled_add_batch_into,
    scaled_add_rows_indexed_into, scaled_add_rows_into, scaled_add_rows_token_range_into,
    silu_mul_batch, silu_mul_batch_into, silu_mul_fused_batch_into, write_vec_into,
};
pub use embedding::{embedding_batch, embedding_batch_vocab_shard, embedding_decode_into};
#[cfg(feature = "kimi-k2")]
pub use kimi_k2::*;
pub use linear::{
    gemm, gemm_graphsafe_into_checked, gemm_into, gemm_into_checked, gemm_per_token,
    gemm_per_token_into_checked, gemm_rows_into, gemm_rows_into_checked,
    gemm_token_range_into_checked, gemv, linear,
};
pub use lora::{
    LoraDecodeGroupedProjection, lora_decode_fused_delta_group3_into, lora_decode_fused_delta_into,
    pack_lora_b_rows_into,
};
pub use norm::{
    fused_add_rms_norm_batch_into, fused_add_rms_norm_into, fused_add_rms_norm_round_batch_into,
    rms_norm, rms_norm_batch_into, rms_norm_batch_offset_into, rms_norm_gated_batch_into,
    rms_norm_into, rms_norm_offset_into,
};
pub use sampling::{
    BatchSamplingRow, BatchSamplingScratch, argmax, argmax_batch_bf16_indexed_into,
    argmax_batch_bf16_into, argmax_batch_bf16_split_partials_len, flashinfer_top1_batch_into,
    flashinfer_topk_row_states_bytes, gpu_sample, gpu_sample_batch_into, gpu_sample_into,
};
