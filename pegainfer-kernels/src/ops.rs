//! GPU operations on device tensors.

mod attention;
mod elementwise;
mod embedding;
mod kimi_experts;
mod kimi_mla;
mod kimi_router;
mod linear;
mod norm;
mod sampling;

pub use attention::{
    PrefillPagedPlan, paged_attention_batch_decode_hd256_into, paged_attention_batch_decode_into,
    paged_attention_batch_decode_split_kv_into, prefill_attention_paged_into,
    qk_norm_partial_rope_batched_decode_hd256_into, qk_norm_rope_batch_decode_into,
};
pub use elementwise::{
    add_batch, add_batch_into, bf16_hidden_to_f32_into, extract_vec, extract_vec_into,
    f32_to_bf16_hidden_into, repeat_f32_for_reduce_scatter_into, scale_f32_in_place,
    silu_mul_batch, silu_mul_batch_into, silu_mul_fused_batch_into, write_vec_into,
};
pub use embedding::{embedding_batch, embedding_batch_vocab_shard, embedding_decode_into};
pub use kimi_experts::{
    KIMI_K2_EP_WORLD, KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_INT4_GROUP_SIZE,
    KIMI_K2_LOCAL_EXPERTS, KIMI_K2_ROUTED_EXPERTS, KIMI_K2_TOPK, KimiExpertMajorRoute,
    KimiExpertMajorRouteWorkspace, KimiExpertMajorRouting, KimiInt4Encoding, KimiInt4ExpertRole,
    KimiInt4ExpertWeights, KimiInt4LogicalShape, KimiInt4NibbleOrder, KimiInt4TensorShape,
    KimiInt4Weight, KimiInt4WeightManifest, KimiMarlinFusedW13Int4Weight,
    KimiMarlinInt4ExpertWeights, KimiMarlinInt4Weight, KimiMarlinRouteWorkspace, KimiMarlinRouting,
    KimiMarlinWna16Workspace, KimiSwiGluPlan, kimi_add_f32_bf16_to_bf16, kimi_int4_metadata_probe,
    kimi_marlin_int4_fuse_w13, kimi_marlin_int4_reorder_scale, kimi_marlin_int4_reorder_weight,
    kimi_marlin_sum_topk_rows_f32, kimi_marlin_w13_swiglu, kimi_marlin_wna16_w2_gemm,
    kimi_marlin_wna16_w13_gemm, kimi_moe_build_expert_major_route, kimi_moe_expand_to_expert_major,
    kimi_moe_marlin_align_block_size, kimi_moe_reduce_expert_major_f32,
    kimi_scaled_add_f32_bf16_to_bf16, kimi_swiglu_silu_mul, packed_int4_cols, validate_ep_rank,
};
pub use kimi_mla::{
    KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_A_OUT, KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
    KIMI_K2_MLA_KV_LORA_RANK, KIMI_K2_MLA_LOCAL_HEADS_TP8, KIMI_K2_MLA_NOPE_DIM,
    KIMI_K2_MLA_O_LOCAL_IN_TP8, KIMI_K2_MLA_Q_HEAD_DIM, KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
    KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8, KIMI_K2_MLA_QKV_A_OUT, KIMI_K2_MLA_ROPE_DIM,
    KIMI_K2_MLA_V_HEAD_DIM, KimiMlaPagedKvLayout, kimi_flashinfer_batch_decode_mla,
    kimi_flashinfer_single_prefill_mla, kimi_mla_absorb_q_nope, kimi_mla_paged_kv_append,
    kimi_mla_rope_apply_kpe, kimi_mla_rope_assemble_prefill, kimi_mla_rope_split_decode,
    kimi_mla_split_qkv_a, kimi_mla_v_up,
};
pub use kimi_router::{
    KIMI_K2_ROUTER_EXPERTS, KIMI_K2_ROUTER_HIDDEN, KIMI_K2_ROUTER_N_GROUP, KIMI_K2_ROUTER_SCALE,
    KIMI_K2_ROUTER_TOPK, KIMI_K2_ROUTER_TOPK_GROUP, KimiRouterBatch, KimiRouterConfig,
    KimiRouterOutput, KimiRouterScratch, kimi_router_noaux_tc_launch, validate_kimi_router_shapes,
};
pub use linear::{
    gemm, gemm_graphsafe_into_checked, gemm_into, gemm_into_checked, gemm_rows_into,
    gemm_rows_into_checked, gemv, linear,
};
pub use norm::{
    fused_add_rms_norm_batch_into, fused_add_rms_norm_into, rms_norm, rms_norm_batch_into,
    rms_norm_batch_offset_into, rms_norm_gated_batch_into, rms_norm_into, rms_norm_offset_into,
};
pub use sampling::{
    argmax, flashinfer_top1_batch_into, flashinfer_topk_row_states_bytes, gpu_sample,
    gpu_sample_into,
};
