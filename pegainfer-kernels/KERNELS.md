# PegaInfer Kernels Index

**Scope**: this crate owns CUDA/Triton build output, FFI declarations, kernel ABI tensor helpers, paged-KV layout metadata, and Rust operator wrappers. Runtime policy objects such as `KvPool`, `PagePool`, and `SamplingParams` stay outside this crate.

Use this file as the LLM entrypoint before editing kernels. Start from `op_id`, then jump to the Rust wrapper, FFI symbol, and source file.

## Qwen3-4B Dense Full-Attention Path

Qwen3-4B uses bf16 dense full attention with `hidden_size=2560`, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128`, and GQA group size 4. TP shards these head/intermediate dimensions per rank; the kernel IDs remain the same.

| op_id | Phase | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3_4b.embedding.batch` | prefill/unified | `ops::embedding_batch` | `embedding_batched_cuda` | `csrc/elementwise.cu` | CUDA | token ids u32, output `HiddenStates` column-major `[hidden, tokens]` |
| `qwen3_4b.norm.rms_batch` | prefill/decode/unified | `ops::rms_norm_batch_into` | `rms_norm_batched_cuda` | `csrc/flashinfer_norm.cu` | FlashInfer CUDA | bf16 hidden states, one row per token |
| `qwen3_4b.norm.rms_vec` | logits | `ops::rms_norm` / `ops::rms_norm_into` | `rms_norm_cuda` | `csrc/flashinfer_norm.cu` | FlashInfer CUDA | bf16 vector |
| `qwen3_4b.linear.gemm_rows` | qkv projection | `ops::gemm_rows_into` | `gemm_cuda` | `csrc/linear.cu` | cuBLAS | row slices from fused QKV matrix |
| `qwen3_4b.linear.gemm` | o/mlp/lm_head | `ops::gemm_into` / `ops::gemm` | `gemm_cuda` | `csrc/linear.cu` | cuBLAS | weight row-major, hidden column-major |
| `qwen3_4b.attn.qk_norm_rope` | attention prep | `ops::qk_norm_rope_batch_decode_into` or direct FFI in unified path | `qk_norm_rope_batched_decode_cuda` | `csrc/prefill_attention.cu` | CUDA | full RoPE, `head_dim=128`, per-token positions |
| `qwen3_4b.kv.scatter` | prefill/decode/unified | direct FFI from model paths | `paged_kv_scatter_cuda` | `csrc/paged_attention.cu` | FlashInfer-layout CUDA wrapper | page-first `KvLayout`, NHD K/V blocks |
| `qwen3_4b.attn.prefill_paged` | prefill/unified | `ops::prefill_attention_paged_into` or direct FFI in unified path | `batch_prefill_paged_cuda` | `csrc/paged_attention.cu` | FlashInfer CUDA | `HEAD_DIM=128`, causal, paged KV |
| `qwen3_4b.attn.decode_paged` | decode/unified | `ops::paged_attention_batch_decode_into` or direct FFI in unified path | `paged_attention_decode_cuda` | `csrc/paged_attention.cu` | FlashInfer CUDA | `HEAD_DIM=128`, no partition-KV |
| `qwen3_4b.norm.fused_add_rms` | residual | `ops::fused_add_rms_norm_batch_into` | `fused_add_rms_norm_batched_cuda` | `csrc/flashinfer_norm.cu` | FlashInfer CUDA | residual add plus RMSNorm over batch |
| `qwen3_4b.mlp.silu_mul_fused` | MLP | `ops::silu_mul_fused_batch_into` | `silu_mul_fused_cuda` | `csrc/fused_proj.cu` | CUDA | input `[2 * intermediate, batch]`, output `[intermediate, batch]` |
| `qwen3_4b.elementwise.add` | residual/unified | `ops::add_batch_into` | `add_cuda` | `csrc/elementwise.cu` | CUDA | same-shape `HiddenStates` |
| `qwen3_4b.sampling.greedy` | decode output | `ops::gpu_sample_into` | `flashinfer_top1_cuda` | `csrc/flashinfer_top1.cu` | FlashInfer CUDA | top-1 path, uses row-state scratch |
| `qwen3_4b.sampling.random` | decode output | `ops::gpu_sample_into` | `gpu_sample_flashinfer_cuda` | `csrc/flashinfer_sampling.cu` | FlashInfer CUDA | temperature/top-k/top-p path |

## DeepSeek V4 MP8 Path

DeepSeek V4 uses the `deepseek-v4` Cargo feature. The server feature forwards
through `pegainfer-deepseek-v4/deepseek-v4` to `pegainfer-kernels/deepseek-v4`.
Runtime call sites live in `pegainfer-deepseek-v4/src/runtime/` and call these
symbols directly through `pegainfer_kernels::ffi`.

| op_id | Runtime owner | FFI symbols | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- |
| `deepseek_v4.quant.fp8_linear` | `runtime/core.rs` | `deepseek_fp8_linear_cuda` | `csrc/deepseek_v4/deepseek_quant.cu`, `tools/tilelang/deepseek_v4/generate.py` | TileLang-generated CUDA with CUDA fallback | TileLang shapes: `N,K` = `512,4096`, `1024,4096`, `2048,4096`, `4096,1024`, `1024,1024`, `4096,2048`; E4M3 activations/weights and E8M0 scales. |
| `deepseek_v4.quant.fp4_linear` | `runtime/core.rs` | `deepseek_fp4_linear_cuda` | `csrc/deepseek_v4/deepseek_quant.cu`, `tools/tilelang/deepseek_v4/generate.py` | TileLang-generated CUDA with serial CUDA fallback | TileLang shapes: `N,K` = `2048,4096`, `4096,2048`; E2M1 weights and E8M0 scales. |
| `deepseek_v4.quant.nope_act` | `runtime/attention_base.rs` | `deepseek_fp8_act_quant_nope_bf16_cuda` | `csrc/deepseek_v4/deepseek_quant.cu` | CUDA | Quantizes the non-RoPE head slice in-place for attention compatibility. |
| `deepseek_v4.copy.rows` | `runtime/state.rs` | `deepseek_bf16_copy_rows_cuda` | `csrc/deepseek_v4/deepseek_quant.cu` | CUDA | BF16 row copy helper for request/state buffers. |
| `deepseek_v4.attn.prep` | `runtime/attention_base.rs`, `runtime/core.rs` | `deepseek_fill_rope_cache_cuda`, `deepseek_head_rms_norm_cuda`, `deepseek_apply_rope_q_kv_cuda` | `csrc/deepseek_v4/deepseek_attention.cu` | CUDA | RoPE cache fill, per-head RMSNorm, and Q/KV RoPE for BF16 attention tensors. |
| `deepseek_v4.attn.indexed_prefill` | `runtime/attention.rs` | `deepseek_indexed_attention_prefill_cuda` | `csrc/deepseek_v4/deepseek_attention.cu`, `tools/tilelang/deepseek_v4/generate.py` | TileLang sparse attention with CUDA glue | TileLang sparse attention shape currently `local_h16_d512`; wrapper pads scratch where needed. |
| `deepseek_v4.collectives.cast` | `runtime/collectives.rs` | `deepseek_bf16_to_f32_cuda`, `deepseek_f32_to_bf16_cuda` | `csrc/deepseek_v4/deepseek_attention.cu` | CUDA | BF16/F32 conversion around NCCL reduction paths. |
| `deepseek_v4.indexer.scores` | `runtime/indexer.rs` | prefill: `deepseek_cutedsl_indexer_scores_exact_bf16_cuda`; decode: `deepseek_indexer_scores_decode_cuda` | `csrc/deepseek_v4/deepseek_indexer.cu`, CuTeDSL AOT artifacts from `tools/cutedsl/deepseek_v4/` | prefill: CuTeDSL exact score; decode: CUDA serial score | Scores compressed KV blocks for sparse/indexed attention. The prefill runtime path uses the exact CuTeDSL score kernel under the default `deepseek-v4` feature; decode remains on the serial CUDA score kernel. Diagnostic CuTeDSL dot helpers are not runtime owners. |
| `deepseek_v4.indexer.topk` | `runtime/indexer.rs`, `runtime/compressor.rs` | `deepseek_indexer_topk_prefill_cuda`, `deepseek_indexer_topk_decode_cuda`, `deepseek_concat_topk_indices_cuda` | `csrc/deepseek_v4/deepseek_indexer.cu` | CUDA | Selects and merges top-k compressed-block indices. Prefill top-k uses a threaded per-token block-reduction selector. It preserves the existing strict `>` semantics by keeping the lower candidate index when scores tie; this candidate-order behavior is part of the runtime contract. |
| `deepseek_v4.indexer.hadamard_fp4` | `runtime/indexer.rs` | `deepseek_hadamard_fp4_quant_bf16_cuda` | `csrc/deepseek_v4/deepseek_indexer.cu`, `tools/tilelang/deepseek_v4/generate.py` | CUDA Hadamard + TileLang FP4 quant | TileLang FP4 quant shape currently `n128`. |
| `deepseek_v4.compressor.rope` | `runtime/attention_base.rs` | `deepseek_apply_rope_hidden_cuda`, `deepseek_apply_rope_hidden_strided_cuda` | `csrc/deepseek_v4/deepseek_compressor.cu` | CUDA | Hidden-state RoPE for plain and strided compressed-state positions. |
| `deepseek_v4.compressor.linear` | `runtime/core.rs` | `deepseek_bf16_linear_cuda` | `csrc/deepseek_v4/deepseek_compressor.cu` | cuBLAS-backed CUDA wrapper | BF16 dense linear used by compressor and small projections. |
| `deepseek_v4.compressor.prefill` | `runtime/compressor.rs` | `deepseek_compressor_nonoverlap_prefill_cuda`, `deepseek_compressor_overlap_prefill_cuda` | `csrc/deepseek_v4/deepseek_compressor.cu` | CUDA | Compressor weighted prefill and normalization for non-overlap/overlap layer variants. |
| `deepseek_v4.compressor.decode` | `runtime/compressor.rs` | `deepseek_compressor_nonoverlap_decode_cuda`, `deepseek_compressor_overlap_decode_cuda` | `csrc/deepseek_v4/deepseek_compressor.cu` | CUDA | Decode projection, state update, weighted compression, and overlap shifting. |
| `deepseek_v4.compressor.concat` | `runtime/attention_base.rs` | `deepseek_concat_seq_bf16_cuda` | `csrc/deepseek_v4/deepseek_compressor.cu` | CUDA | Concatenates BF16 sequence fragments for attention/compressor flow. |
| `deepseek_v4.hc` | `runtime/core.rs` | `deepseek_hc_expand_cuda`, `deepseek_hc_mixes_cuda`, `deepseek_hc_split_sinkhorn_cuda`, `deepseek_hc_pre_output_cuda`, `deepseek_hc_pre_from_mixes_cuda`, `deepseek_hc_pre_norm_from_mixes_cuda`, `deepseek_hc_head_pre_cuda`, `deepseek_hc_post_cuda`, `deepseek_hc_post_f32_branch_cuda` | `csrc/deepseek_v4/deepseek_hc.cu`, `tools/tilelang/deepseek_v4/generate.py` | CUDA + TileLang sinkhorn helper + cuBLAS wrapper | HC split sinkhorn TileLang shape currently `hc4_i20`; decode can fuse split sinkhorn plus pre-output, split sinkhorn plus pre-output plus following RMSNorm, and attention all-reduce F32 branch rounding plus HC post. |
| `deepseek_v4.logits.last_token` | `runtime/core.rs` | `deepseek_last_token_bf16_logits_cuda` | `csrc/deepseek_v4/deepseek_hc.cu` | cuBLAS-backed CUDA wrapper | Computes final logits from the last BF16 hidden token; preserves the FP32 SGEMV path and caches the converted rank-local head weight in per-device scratch. |
| `deepseek_v4.moe.route` | `runtime/moe.rs` | `deepseek_hash_gate_cuda`, `deepseek_score_gate_cuda`, `deepseek_score_gate_debug_cuda` | `csrc/deepseek_v4/deepseek_moe.cu` | CUDA + cuBLAS wrapper | Hash/score gate routing and debug score extraction; score routing uses BF16 cuBLAS projection plus parallel top-k select. |
| `deepseek_v4.moe.experts` | `runtime/core.rs`, `runtime/moe.rs` | `deepseek_fp8_w2_swiglu_with_workspace_cuda`, `deepseek_moe_fp4_grouped_w2_swiglu_with_workspace_cuda` | `csrc/deepseek_v4/deepseek_quant.cu`, `tools/tilelang/deepseek_v4/generate.py` | CUDA + TileLang-generated CUDA | Shared and routed experts use the W13 GEMM plus fused SwiGLU/W2 activation-quant plus W2 GEMM path; the old standalone SwiGLU clamp entry point is intentionally removed. |
| `deepseek_v4.moe.fused_layout` | `runtime/moe.rs` | `deepseek_moe_local_mapping_cuda`, `deepseek_moe_expand_to_fused_cuda`, `deepseek_moe_reduce_fused_f32_cuda`, `deepseek_add_f32_bf16_to_bf16_cuda` | `csrc/deepseek_v4/deepseek_moe.cu` | CUDA | GPU-resident local expert mapping, expert-major input expansion, packed routed output reduction, and residual conversion helpers. |

## Kimi-K2 Text TP8/EP8 Path

Kimi-K2 uses the `pegainfer-kimi-k2` model crate. The first kernel-crate surface
is text-only and targets TP8/EP8 with bs > 1 from the start. Shared BF16 ops
reuse existing PegaInfer wrappers. Kimi-specific MoE router and routed INT4
expert entry points live under model-specific ops modules. Kimi router uses the
existing graph-safe GEMM path plus a device-side top8 selector. The current
CUTLASS example69 INT4 launcher is retained only as a launch/limitation probe:
H20 focused tests show it cannot express Kimi `group_size=32` BF16 scale
semantics, even though its signed nibble conversion is correct. Kimi routed
expert correctness requires a replacement W4A16 backend before full-model
parity claims. Scale metadata now separates checkpoint `[expert,out,group]`,
CUTLASS example69 group-major `[expert,group,out]`, and vLLM Marlin
group-major+perm64 `[expert,group,out]`; packed-weight metadata likewise separates
checkpoint offset-binary, CUTLASS signed reordered, and Marlin uint4b8
no-actorder. The final Marlin runtime package is fused W13 (`gate_then_up`)
plus W2: W13 uses `[expert,K/16,4096*2]` packed weight and
`[expert,K/32,4096]` scale, both in vLLM layout. `kimi_marlin_int4_reorder_weight_cuda`,
`kimi_marlin_int4_reorder_scale_cuda`, and `kimi_marlin_int4_fuse_w13_cuda`
implement the vLLM Marlin package layouts for the future WNA16 backend.
Kimi EP dispatch/combine is planned for
pplx-garden EP rather than NCCL AG/RS.

| op_id | Runtime owner | Rust wrapper | FFI symbols | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `kimi_k2.norm.rms_batch` | `pegainfer-kimi-k2` | `ops::rms_norm_batch_into` | `rms_norm_batched_cuda` | `csrc/flashinfer_norm.cu` | FlashInfer CUDA | BF16 hidden states, one row per token; Kimi hidden `7168`, q LoRA `1536`, and kv LoRA `512` all use the parameterized wrapper. This is not a fallback path. |
| `kimi_k2.norm.rms_vec` | `pegainfer-kimi-k2` | `ops::rms_norm_into` | `rms_norm_cuda` | `csrc/flashinfer_norm.cu` | FlashInfer CUDA | BF16 single vector path; exposed in `pegainfer-kimi-k2` headers as `RmsNormBackend::FlashInferVec`. |
| `kimi_k2.norm.fused_add_rms` | `pegainfer-kimi-k2` | `ops::fused_add_rms_norm_batch_into` | `fused_add_rms_norm_batched_cuda` | `csrc/flashinfer_norm.cu` | FlashInfer CUDA | Residual add plus RMSNorm over bs > 1 token batches. |
| `kimi_k2.linear.dense_bf16` | `pegainfer-kimi-k2` | `ops::gemm_into` / `ops::gemm_rows_into` | `gemm_cuda` | `csrc/linear.cu` | cuBLAS | BF16 attention, dense MLP, shared expert, router gate, and lm_head shard projections. |
| `kimi_k2.attn.mla_fused_qkv_a` | `pegainfer-kimi-k2` | `ops::gemm_graphsafe_into_checked` | `gemm_graphsafe_cuda` | `csrc/linear.cu` | graph-safe cuBLAS GEMM | Load-time `DeviceMatrix::vstack(q_a_proj, kv_a_proj_with_mqa)` creates weight `[2112,7168]`; decode writes `qkv_a [B,2112]` without D2H or step-time allocation. |
| `kimi_k2.attn.mla_split_qkv_a` | `pegainfer-kimi-k2` | `ops::kimi_mla_split_qkv_a` | `kimi_mla_split_qkv_a_cuda` | `csrc/kimi_k2/kimi_mla.cu` | CUDA | Splits fused `qkv_a [B,2112]` into `q_a [B,1536]`, compressed KV `[B,512]`, and raw `k_rope [B,64]`. This replaces the old separate `kv_a` split path. |
| `kimi_k2.attn.mla_rope_split_decode` | `pegainfer-kimi-k2` | `ops::kimi_mla_rope_split_decode` | `kimi_mla_rope_split_decode_cuda` | `csrc/kimi_k2/kimi_mla.cu` | CUDA | Decode-step split+RoPE prep: `q_proj [B,8,192]` and current `k_rope [B,64]` plus device positions produce `q_nope [B,8,128]`, `q_pe [B,8,64]`, and `append_kpe [B,64]` in Kimi split-half RoPE layout. |
| `kimi_k2.attn.mla_absorb_q` | `pegainfer-kimi-k2` | `ops::kimi_mla_absorb_q_nope` | `kimi_mla_absorb_q_nope_cuda` | `csrc/kimi_k2/kimi_mla.cu` | graph-safe cuBLAS strided-batched GEMM | Uses the `W_UK` slice inside `kv_b_proj [8,256,512]` directly: `q_nope [B,8,128] -> q_abs_nope [B,8,512]`, one cuBLAS batch per local head, no weight repack. |
| `kimi_k2.attn.mla_paged_append` | `pegainfer-kimi-k2` | `ops::kimi_mla_paged_kv_append` | `kimi_mla_paged_kv_append_cuda` | `csrc/kimi_k2/kimi_mla.cu` | FlashInfer MLA page helper | Appends compressed MLA KV step tensors into paged cache: `append_ckv [nnz,512]`, `append_kpe [nnz,64]`, device `batch_indices/positions`, page table CSR, and explicit ckv/kpe strides. Runtime may use separate ckv/kpe buffers or strided views into concat storage. |
| `kimi_k2.attn.mla_decode_paged` | `pegainfer-kimi-k2` | `ops::kimi_flashinfer_batch_decode_mla` | `kimi_flashinfer_batch_decode_mla_cuda` | `csrc/kimi_k2/kimi_mla.cu` | FlashInfer BatchDecode MLA | Consumes absorbed `q_abs_nope [B,8,512]`, `q_pe [B,8,64]`, paged compressed KV, and decode plan arrays; writes latent attention output `[B,8,512]`. `W_UK_T [H,128,512]` absorption and `W_UV [H,512,128]` v-up stay model-side. |
| `kimi_k2.attn.mla_v_up` | `pegainfer-kimi-k2` | `ops::kimi_mla_v_up` | `kimi_mla_v_up_cuda` | `csrc/kimi_k2/kimi_mla.cu` | graph-safe cuBLAS strided-batched GEMM | Uses the `W_UV` slice inside `kv_b_proj [8,256,512]` directly: FlashInfer latent `[B,8,512] -> attn_out [B,8,128]`, one cuBLAS batch per local head, no D2H. |
| `kimi_k2.moe.router_noaux_tc` | `pegainfer-kimi-k2` | `ops::kimi_router_noaux_tc_launch` | `kimi_k2_router_noaux_tc_cuda` | `csrc/kimi_k2/kimi_router.cu` | graph-safe GEMM + CUDA selector | BF16 hidden `[padded_tokens,7168]`, gate `[384,7168]`, correction bias `[384]`, output top8 route weights/indices for active tokens; logits projection uses library GEMM, selection stays device-resident. H20 rank0 gate covers real K2.5 layer1 gate/bias. |
| `kimi_k2.moe.expert_major_route` | `pegainfer-kimi-k2` | `ops::kimi_moe_build_expert_major_route` | `kimi_moe_expert_major_route_cuda` | `csrc/kimi_k2/kimi_experts.cu` | CUDA routing metadata | Device-resident `topk_idx[active_tokens,8]` to local expert-major `u32 expert_indptr[49]`, `pos_to_token`, `token_topk_to_pos`, and `local_count`; no host route readback in the hot path. |
| `kimi_k2.moe.marlin_align_block_size` | `pegainfer-kimi-k2` | `ops::kimi_moe_marlin_align_block_size` | `kimi_moe_marlin_align_block_size_cuda` | `csrc/kimi_k2/kimi_experts.cu` | CUDA routing metadata | Device-resident vLLM Marlin/WNA16 alignment: `sorted_token_ids`, `expert_ids`, and `num_tokens_post_padded` for local EP experts. It ignores non-local experts like vLLM `ignore_invalid_experts=True`, pads each local expert to block size `8/16/32/48/64`, uses sentinel `active_tokens * topk`, and performs no D2H or allocation in the decode step. |
| `kimi_k2.moe.expert_major_expand` | `pegainfer-kimi-k2` | `ops::kimi_moe_expand_to_expert_major` | `kimi_moe_expand_to_expert_major_cuda` | `csrc/kimi_k2/kimi_experts.cu` | CUDA copy/scatter | BF16 token-major hidden `[active_tokens,7168]` to expert-major activation scratch `[routed_capacity,7168]` using device `pos_to_token`. |
| `kimi_k2.moe.expert_major_reduce_f32` | `pegainfer-kimi-k2` | `ops::kimi_moe_reduce_expert_major_f32` | `kimi_moe_reduce_expert_major_f32_cuda` | `csrc/kimi_k2/kimi_experts.cu` | CUDA weighted reduce | BF16 expert-major output + f32 `topk_weight` + device `token_topk_to_pos` to f32 token-major routed output `[active_tokens,7168]`. |
| `kimi_k2.moe.int4_metadata` | `pegainfer-kimi-k2` | `ops::kimi_int4_metadata_probe` | `kimi_int4_expert_metadata_probe_cuda` | `csrc/kimi_k2/kimi_experts.cu` | CUDA metadata stub | Shape contract for compressed-tensors `weight_packed` U8 `[48,out,in/2]`, checkpoint `weight_scale` BF16 `[48,out,in/32]`, and `weight_shape` I32 `[96]`; Rust manifest records signed symmetric INT4, low-then-high nibble order, and separate checkpoint/CUTLASS/Marlin group-major+perm64 scale layout specs. |
| `kimi_k2.moe.int4_marlin_package` | `pegainfer-kimi-k2` | `ops::kimi_marlin_int4_reorder_weight`, `ops::kimi_marlin_int4_reorder_scale`, `ops::kimi_marlin_int4_fuse_w13` | `kimi_marlin_int4_reorder_weight_cuda`, `kimi_marlin_int4_reorder_scale_cuda`, `kimi_marlin_int4_fuse_w13_cuda` | `csrc/kimi_k2/kimi_marlin_int4.cu`, `csrc/kimi_k2/kimi_cutlass_int4_sm90a.cu` | CUDA load-time package helpers | Weight package preserves vLLM `uint4b8` bias=8 nibbles. Single projections repack checkpoint `[expert,out,K/8] int32` into Marlin no-actorder `[expert,K/16,N*2] int32`; scale package converts checkpoint `[expert,out,K/32]` into vLLM Marlin group-major+perm64 `[expert,K/32,out]`. Final runtime package fuses gate/up into W13 `[expert,K/16,4096*2]` and W13 scale `[expert,K/32,4096]`; W2 remains `[expert,2048/16,7168*2]` and `[expert,2048/32,7168]`. These are load/package helpers, not decode hot-path kernels. |
| `kimi_k2.moe.int4_grouped_w1_w3` | `pegainfer-kimi-k2` | `ops::kimi_int4_grouped_w1_w3` | `kimi_int4_grouped_w1_w3_cuda`, `kimi_cutlass_int4_grouped_w1_w3_sm90a_cuda` | `csrc/kimi_k2/kimi_experts.cu`, `csrc/kimi_k2/kimi_cutlass_int4_sm90a.cu` | CUTLASS example69 limitation probe; backend replacement required | Expert-major BF16 input `[routed_tokens,7168]`, local experts `48`, topk `8`, group size `32`; W1/W3 fuse into one N=4096 grouped GEMM. Example69 launch can run on H20 but does not implement Kimi per32 BF16 scale semantics. |
| `kimi_k2.moe.swiglu` | `pegainfer-kimi-k2` | `ops::kimi_swiglu_silu_mul` | `silu_mul_triton_aot_cuda` | Triton AOT elementwise wrapper | Triton AOT | External `silu(gate) * up` BF16 scratch between W1/W3 and W2; keeps W2 as a plain CUTLASS grouped GEMM. |
| `kimi_k2.moe.int4_grouped_w2` | `pegainfer-kimi-k2` | `ops::kimi_int4_grouped_w2_swiglu` | `kimi_int4_grouped_w2_swiglu_cuda`, `kimi_cutlass_int4_grouped_w2_sm90a_cuda` | `csrc/kimi_k2/kimi_experts.cu`, `csrc/kimi_k2/kimi_cutlass_int4_sm90a.cu` | CUTLASS example69 limitation probe; backend replacement required | Expert-major BF16 activated input `[routed_tokens,2048]` plus INT4 W2 to BF16 output `[routed_tokens,7168]`; bs > 1 remains explicit through `batch_size`, `active_tokens`, `routed_tokens`, and `expert_indptr[49]`. Example69 launch can run on H20 but does not implement Kimi per32 BF16 scale semantics. |

## Non-Qwen3 Compatibility

The crate still builds CUDA/Triton symbols needed by the current root binary:

- Qwen3.5 HD256 full-attention kernels: `prefill_attention_hd256.cu`, `paged_attention.cu`.
- Qwen3.5 linear-attention decode kernels: `conv1d.cu`, `gated_delta_rule.cu`.
- Qwen3.5 chunk-wise GDR prefill Triton AOT kernels: `tools/triton/gated_delta_rule_chunkwise_kernels.py`.

These are preserved for build compatibility. They are not part of the Qwen3-4B Phase 1 API surface.

## Editing Rule

When adding or replacing a kernel used by Qwen3-4B or DeepSeek V4, update this
routing table.

Do not add model-specific machine-readable manifests here. The kernels crate
owns reusable operator implementations; model crates should own model DAG
metadata. If a Qwen3-4B manifest becomes useful for tracing or simulation, put
it beside the Qwen3-4B model crate and generate or validate it from code.
