//! Qwen3.5 GPU operation wrappers.

pub(crate) use openinfer_core::ops::PrefillPagedPlan;
pub(crate) use openinfer_core::ops::{
    GEMM_LT_MAX_N, add_batch, add_batch_into, copy_hidden_token_range_into, embedding_batch,
    extract_vec, extract_vec_into, gemm, gemm_into, gemm_into_checked, gemm_lt_tune,
    gemm_rows_into_checked, paged_attention_batch_decode_hd256_into,
    paged_attention_batch_decode_via_prefill_hd256_into, paged_attention_batch_prefill_hd256_into,
    qk_norm_partial_rope_batched_decode_hd256_into, rms_norm_gated_batch_into,
    silu_mul_fused_batch_into, write_vec_into,
};
pub use openinfer_core::ops::{rms_norm_batch_offset_into, rms_norm_offset_into};
pub use recurrent::gated_delta_rule_prefill_chunkwise_into;
pub(crate) use recurrent::{
    conv1d_decode_batch_into, conv1d_prefill_batch_into, gated_delta_rule_decode_batch_into,
};

use crate::recurrent;
