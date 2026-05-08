//! GPU operations on device tensors.

mod attention;
mod elementwise;
mod embedding;
mod linear;
mod norm;
mod sampling;

pub use attention::{
    PrefillPagedPlan, paged_attention_batch_decode_hd256_into, paged_attention_batch_decode_into,
    paged_attention_batch_decode_split_kv_into, prefill_attention_paged_into,
    qk_norm_partial_rope_batched_decode_hd256_into, qk_norm_rope_batch_decode_into,
};
pub use elementwise::{
    add_batch, add_batch_into, extract_vec, extract_vec_into, silu_mul_batch, silu_mul_batch_into,
    silu_mul_fused_batch_into, write_vec_into,
};
pub use embedding::{embedding_batch, embedding_batch_vocab_shard, embedding_decode_into};
pub use linear::{gemm, gemm_into, gemm_rows_into, gemv, linear};
pub use norm::{
    fused_add_rms_norm_batch_into, fused_add_rms_norm_into, rms_norm, rms_norm_batch_into,
    rms_norm_batch_offset_into, rms_norm_gated_batch_into, rms_norm_into, rms_norm_offset_into,
};
pub use sampling::{argmax, flashinfer_topk_row_states_bytes, gpu_sample, gpu_sample_into};
