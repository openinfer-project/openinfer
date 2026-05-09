mod config;
mod direct;
pub mod e2e_runner;
mod model;
mod runtime;
mod weights;

pub use config::{Config, RopeScaling, TensorParallelConfig};
pub use direct::{DeepSeekV4DirectGenerator, DirectGeneration, start_engine};
pub use model::{
    AttentionWeightNames, AttentionWeights, BlockWeightNames, BlockWeights, CompressorWeightNames,
    CompressorWeights, DeepSeekRankModel, ExpertWeightNames, ExpertWeights, FfnWeightNames,
    FfnWeights, IndexerWeightNames, IndexerWeights, QuantLinearNames, QuantLinearRef,
    RankWeightView, TensorRef, TopLevelWeightNames,
};
pub use runtime::{
    AttentionProjections, Bf16Cache, Bf16HiddenStates, CompressorDecodeState, DeepSeekRopeCache,
    F32HiddenStates, F32Logits, HcHiddenStates, HcPreState, LayerDecodeCache, MoeFusedRoutePlan,
    RoutedExperts, add_bf16_hidden, add_f32_bf16_to_bf16_hidden, all_gather_logits_group,
    all_reduce_f32_hidden_in_place, all_reduce_hidden_fp32_in_place, all_reduce_hidden_group,
    all_reduce_hidden_group_bf16, all_reduce_hidden_group_fp32, all_reduce_hidden_in_place,
    apply_rope_attention_projections, apply_rope_hidden_in_place,
    apply_rope_hidden_strided_in_place, attention_decode_group_bf16_hidden,
    attention_decode_rank_local_bf16_hidden, attention_output_project_bf16_hidden,
    attention_prefill_compressed_nonoverlap_group_bf16_hidden,
    attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden,
    attention_prefill_compressed_overlap_group_bf16_hidden,
    attention_prefill_compressed_overlap_rank_local_bf16_hidden,
    attention_prefill_group_bf16_hidden, attention_prefill_rank_local_bf16_hidden,
    attention_project_bf16_hidden, bf16_linear_bf16_hidden, block_decode_group_bf16_hidden,
    block_decode_group_rank_threads_bf16_hidden, block_decode_rank_lane_bf16_hidden,
    block_prefill_group_bf16_hidden, block_prefill_rank_local_bf16_hidden,
    build_moe_fused_route_plan, compress_topk_indices, compress_topk_indices_decode,
    compressor_nonoverlap_decode_bf16_hidden, compressor_nonoverlap_prefill_bf16_hidden,
    compressor_overlap_decode_bf16_hidden, compressor_overlap_decode_bf16_hidden_with_dim,
    compressor_overlap_prefill_bf16_hidden, compressor_overlap_prefill_bf16_hidden_with_dim,
    concat_seq_bf16_hidden, concat_topk_indices, copy_bf16_rows_to_cache,
    copy_window_prefill_to_ring_cache, embedding_rank_local, embedding_vocab_parallel_group,
    expand_moe_fused_input, final_logits_group_bf16_hidden, final_logits_rank_local_bf16_hidden,
    fp4_linear_bf16_hidden, fp8_act_quant_nope_bf16_hidden_in_place, fp8_linear_bf16_hidden,
    hadamard_fp4_quant_bf16_hidden_in_place, hash_route_bf16_hidden,
    hash_routed_moe_group_bf16_hidden, hash_routed_moe_rank_local_bf16_hidden,
    hc_expand_bf16_hidden, hc_head_bf16_hidden, hc_post_bf16_hidden, hc_pre_bf16_hidden,
    head_rms_norm_bf16_hidden, indexed_attention_cache_bf16_hidden,
    indexed_attention_prefill_bf16_hidden, indexer_scores_decode_bf16_hidden,
    indexer_scores_prefill_bf16_hidden, indexer_topk_indices_decode, indexer_topk_indices_prefill,
    local_expert_forward_bf16_hidden, local_experts_forward_packed_bf16_hidden,
    moe_group_bf16_hidden, moe_rank_local_bf16_hidden, precompute_rope_cache,
    prefill_logits_and_decode_cache_group_bf16_hidden, prefill_logits_group_bf16_hidden,
    rank_local_logits_from_hidden, reduce_moe_fused_output_f32, rms_norm_bf16_hidden,
    score_route_bf16_hidden, shared_expert_forward_bf16_hidden,
    sparse_attention_prefill_bf16_hidden, swiglu_clamp_bf16_hidden,
    window_and_compress_topk_indices, window_topk_indices, window_topk_indices_decode,
};
pub use weights::{
    GpuRawTensor, RankGpuContext, RankManifest, RankWeights, TensorInfo, load_rank_manifest,
    load_rank_subset_to_gpu, load_rank_to_gpu, mp_rank_path,
};
