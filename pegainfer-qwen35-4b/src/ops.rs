//! Qwen3.5 GPU operation wrappers.

pub(crate) use pegainfer_core::ops::PrefillPagedPlan;
pub(crate) use pegainfer_core::ops::{
    add_batch, add_batch_into, embedding_batch, extract_vec, extract_vec_into,
    flashinfer_topk_row_states_bytes, gemm, gemm_into, gpu_sample_into, linear,
    paged_attention_batch_decode_hd256_into, qk_norm_partial_rope_batched_decode_hd256_into,
    rms_norm_gated_batch_into, select_batch_tokens_into, silu_mul_batch, silu_mul_batch_into,
    write_vec_into,
};
pub use pegainfer_core::ops::{rms_norm_batch_offset_into, rms_norm_offset_into};
pub use recurrent::gated_delta_rule_prefill_chunkwise_into;
pub(crate) use recurrent::{
    conv1d_decode_into, conv1d_prefill_batch_into, gated_delta_rule_decode_vec_into,
};

use crate::recurrent;
