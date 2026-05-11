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
| `deepseek_v4.indexer.scores` | `runtime/indexer.rs` | `deepseek_indexer_scores_prefill_cuda`, `deepseek_indexer_scores_decode_cuda` | `csrc/deepseek_v4/deepseek_indexer.cu` | CUDA | Scores compressed KV blocks for sparse/indexed attention. |
| `deepseek_v4.indexer.topk` | `runtime/indexer.rs`, `runtime/compressor.rs` | `deepseek_indexer_topk_prefill_cuda`, `deepseek_indexer_topk_decode_cuda`, `deepseek_concat_topk_indices_cuda` | `csrc/deepseek_v4/deepseek_indexer.cu` | CUDA | Selects and merges top-k compressed-block indices. |
| `deepseek_v4.indexer.hadamard_fp4` | `runtime/indexer.rs` | `deepseek_hadamard_fp4_quant_bf16_cuda` | `csrc/deepseek_v4/deepseek_indexer.cu`, `tools/tilelang/deepseek_v4/generate.py` | CUDA Hadamard + TileLang FP4 quant | TileLang FP4 quant shape currently `n128`. |
| `deepseek_v4.compressor.rope` | `runtime/attention_base.rs` | `deepseek_apply_rope_hidden_cuda`, `deepseek_apply_rope_hidden_strided_cuda` | `csrc/deepseek_v4/deepseek_compressor.cu` | CUDA | Hidden-state RoPE for plain and strided compressed-state positions. |
| `deepseek_v4.compressor.linear` | `runtime/core.rs` | `deepseek_bf16_linear_cuda` | `csrc/deepseek_v4/deepseek_compressor.cu` | cuBLAS-backed CUDA wrapper | BF16 dense linear used by compressor and small projections. |
| `deepseek_v4.compressor.prefill` | `runtime/compressor.rs` | `deepseek_compressor_nonoverlap_prefill_cuda`, `deepseek_compressor_overlap_prefill_cuda` | `csrc/deepseek_v4/deepseek_compressor.cu` | CUDA | Compressor weighted prefill and normalization for non-overlap/overlap layer variants. |
| `deepseek_v4.compressor.decode` | `runtime/compressor.rs` | `deepseek_compressor_nonoverlap_decode_cuda`, `deepseek_compressor_overlap_decode_cuda` | `csrc/deepseek_v4/deepseek_compressor.cu` | CUDA | Decode projection, state update, weighted compression, and overlap shifting. |
| `deepseek_v4.compressor.concat` | `runtime/attention_base.rs` | `deepseek_concat_seq_bf16_cuda` | `csrc/deepseek_v4/deepseek_compressor.cu` | CUDA | Concatenates BF16 sequence fragments for attention/compressor flow. |
| `deepseek_v4.hc` | `runtime/core.rs` | `deepseek_hc_expand_cuda`, `deepseek_hc_mixes_cuda`, `deepseek_hc_split_sinkhorn_cuda`, `deepseek_hc_pre_output_cuda`, `deepseek_hc_pre_from_mixes_cuda`, `deepseek_hc_head_pre_cuda`, `deepseek_hc_post_cuda` | `csrc/deepseek_v4/deepseek_hc.cu`, `tools/tilelang/deepseek_v4/generate.py` | CUDA + TileLang sinkhorn helper + cuBLAS wrapper | HC split sinkhorn TileLang shape currently `hc4_i20`; decode can fuse split sinkhorn and pre-output from already scaled mixes before the following RMSNorm. |
| `deepseek_v4.logits.last_token` | `runtime/core.rs` | `deepseek_last_token_bf16_logits_cuda` | `csrc/deepseek_v4/deepseek_hc.cu` | cuBLAS-backed CUDA wrapper | Computes final logits from the last BF16 hidden token. |
| `deepseek_v4.moe.route` | `runtime/moe.rs` | `deepseek_hash_gate_cuda`, `deepseek_score_gate_cuda`, `deepseek_score_gate_debug_cuda` | `csrc/deepseek_v4/deepseek_moe.cu` | CUDA + cuBLAS wrapper | Hash/score gate routing and debug score extraction. |
| `deepseek_v4.moe.activation` | `runtime/core.rs` | `deepseek_swiglu_clamp_cuda` | `csrc/deepseek_v4/deepseek_moe.cu` | CUDA | SwiGLU clamp for shared and packed routed expert paths. |
| `deepseek_v4.moe.fused_layout` | `runtime/moe.rs` | `deepseek_moe_local_mapping_cuda`, `deepseek_moe_expand_to_fused_cuda`, `deepseek_moe_reduce_fused_f32_cuda`, `deepseek_add_f32_bf16_to_bf16_cuda` | `csrc/deepseek_v4/deepseek_moe.cu` | CUDA | GPU-resident local expert mapping, expert-major input expansion, packed routed output reduction, and residual conversion helpers. |

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
