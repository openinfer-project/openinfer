//! Shared GPU operation wrappers and kernel-crate re-exports.

mod attention;
mod paged_plan;
mod sampling;

pub use attention::{
    paged_attention_batch_decode_hd256_into, paged_attention_batch_decode_into,
    prefill_attention_paged_into,
};
pub use paged_plan::PrefillPagedPlan;
pub use pegainfer_kernels::ops::{
    add_batch, add_batch_into, embedding_batch, embedding_decode_into, extract_vec,
    extract_vec_into, fused_add_rms_norm_batch_into, fused_add_rms_norm_into, gemm, gemm_into,
    gemm_rows_into, gemv, linear, qk_norm_partial_rope_batched_decode_hd256_into,
    qk_norm_rope_batch_decode_into, rms_norm, rms_norm_batch_into, rms_norm_batch_offset_into,
    rms_norm_gated_batch_into, rms_norm_into, rms_norm_offset_into, silu_mul_batch,
    silu_mul_batch_into, silu_mul_fused_batch_into, write_vec_into,
};
pub use sampling::{argmax, flashinfer_topk_row_states_bytes, gpu_sample, gpu_sample_into};
