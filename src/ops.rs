//! GPU operations on device tensors.

mod recurrent;

#[cfg(test)]
mod tests;

pub(crate) use pegainfer_core::ops::{PrefillPagedPlan, paged_attention_batch_decode_hd256_into};
pub use pegainfer_core::ops::{
    add_batch, add_batch_into, argmax, embedding_batch, embedding_decode_into, extract_vec,
    extract_vec_into, flashinfer_topk_row_states_bytes, fused_add_rms_norm_batch_into,
    fused_add_rms_norm_into, gemm, gemm_into, gemm_rows_into, gemv, gpu_sample, gpu_sample_into,
    linear, qk_norm_partial_rope_batched_decode_hd256_into, qk_norm_rope_batch_decode_into,
    rms_norm, rms_norm_batch_into, rms_norm_batch_offset_into, rms_norm_gated_batch_into,
    rms_norm_into, rms_norm_offset_into, silu_mul_batch, silu_mul_batch_into,
    silu_mul_fused_batch_into, write_vec_into,
};
pub use recurrent::gated_delta_rule_prefill_chunkwise_into;
pub(crate) use recurrent::{
    conv1d_decode_into, conv1d_prefill_batch_into, gated_delta_rule_decode_vec_into,
};
