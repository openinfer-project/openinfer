# Kimi-K2 vLLM Path Comparison

> **TL;DR:** vLLM Kimi/DeepSeekV3 decode 和 PegaInfer 当前 decode 的最大结构差异在 MLA attention：vLLM 用 `fused_qkv_a_proj` 合并 `q_a + kv_a`，并用 `concat_and_cache_mla + FlashMLA + W_UK/W_UV bmm` 的 data-movement-friendly decode；PegaInfer 目前把 `q_a`、`kv_a` 拆成两个 GEMM。本文档创建时发现 `batch_decode_trace.rs` 漏记 `split_compressed_kv`、`kv_a_norm`、`paged_kv_append` 且多记一个 decode 不存在的 `kv_b` GEMM；该 trace 漂移已修正，修正后 bs4/kv1024 静态 trace 为 `1947` calls。下一步按差异表评估 fused qkv_a、MLA metadata/cache append、F32 bridge collective 三类优化。
>
> **Last touched:** 2026-05

## Source Map

- vLLM checkouts:
  - `/root/develop/yingshan/vllm` on `h20-100` was readable from the main shell and matches the vLLM V0 files already used for Kimi fixture work.
  - A sub-agent also read `/data/code/pega-ci/vllm`, whose V1 layout places MLA code under `vllm/model_executor/layers/mla.py` and `vllm/model_executor/layers/attention/mla_attention.py`. The operator structure is consistent, but file names differ.
- Kimi text model uses vLLM `DeepseekV2Model` / `DeepseekV3ForCausalLM`; Kimi-VL only wraps the language model and is out of scope.
- Main vLLM files:
  - `/root/develop/yingshan/vllm/vllm/model_executor/models/deepseek_v2.py`
  - `/root/develop/yingshan/vllm/vllm/attention/backends/mla/common.py`
  - `/root/develop/yingshan/vllm/vllm/attention/backends/flashmla.py`
  - `/root/develop/yingshan/vllm/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`
  - `/root/develop/yingshan/vllm/vllm/model_executor/layers/fused_moe/layer.py`
  - `/root/develop/yingshan/vllm/csrc/cache_kernels.cu`
  - `/root/develop/yingshan/vllm/csrc/moe/*`
- PegaInfer files:
  - `pegainfer-kimi-k2/src/direct/worker.rs`
  - `pegainfer-kimi-k2/src/batch_decode_trace.rs`
  - `pegainfer-kernels/src/ops/kimi_mla.rs`
  - `pegainfer-kernels/src/ops/kimi_router.rs`
  - `pegainfer-kernels/src/ops/kimi_experts.rs`

## vLLM Decode Operator List

This is the source-level list for Kimi/DeepSeekV3 decode, not an nsys trace. PyTorch, CUDA graph, and vLLM custom-op wrappers can fuse or hide individual CUDA kernels at runtime.

| Section | vLLM operator path | Source evidence |
| --- | --- | --- |
| Embedding | `get_input_embeddings(input_ids)` then model layers; TP vocab-parallel reduction is handled by vLLM parallel layers. | `deepseek_v2.py:704-716` |
| Attention input | `input_layernorm(hidden_states, residual)`; residual is carried by vLLM layer contract. | `deepseek_v2.py:609-616` |
| MLA q/kv down projection | `fused_qkv_a_proj = MergedReplicatedLinear(hidden_size, [q_lora_rank, kv_lora_rank + rope_dim])`; forward does one projection and splits into `q_c` and `kv_lora`. V1 small-batch code can route this through `min_latency_fused_qkv_a_proj` / `dsv3_fused_a_gemm`. | `deepseek_v2.py:410-417`, `deepseek_v2.py:505-510`; V1: `layers/mla.py`, `dsv3_fused_a_gemm` |
| MLA q branch | `q_a_layernorm(q_c)` then `q_b_proj(q_c)`; `q` is reshaped to local heads. | `deepseek_v2.py:425-433`, `deepseek_v2.py:511-522` |
| MLA kv branch | `kv_lora.split([kv_lora_rank, qk_rope_head_dim])`; `kv_a_layernorm(kv_c)`; `k_pe` goes through RoPE. | `deepseek_v2.py:517-526` |
| MLA cache append | `ops.concat_and_cache_mla(k_c_normed, k_pe, kv_cache, slot_mapping, ...)` writes latent KV and RoPE PE into MLA paged cache. | `common.py:1276-1285` |
| MLA q absorb | Decode path splits `q_nope/q_pe`, transposes `q_nope`, and runs `torch.bmm(decode_q_nope, W_UK_T)` to form `decode_ql_nope`. | `common.py:1297-1308` |
| MLA attention | FlashMLA path calls `_flashmla_C.fwd_kvcache_mla` via `flash_mla_with_kvcache`, passing block table, seq lens, tile scheduler metadata, and num splits. | `flashmla.py:212-225` |
| MLA v up | FlashMLA returns latent output; `_v_up_proj` runs per-head `torch.bmm(x, W_UV)` and reshapes to `num_heads * v_head_dim`. | `flashmla.py:227`, `common.py:1021-1027` |
| MLA output projection | `o_proj(attn_out)` is a row-parallel linear; TP reduction is handled inside vLLM parallel layer. | `deepseek_v2.py:528-534` |
| Dense layer 0 MLP | `DeepseekV2MLP`: vLLM V1 uses fused `gate_up_proj` GEMM where available, then SiLU multiply, then row-parallel down projection. The V0 path is still gate/up/down at the module level. | `deepseek_v2.py:590-645`; V1: `deepseek_v2.py:190-235` |
| MoE shared expert | `shared_experts = DeepseekV2MLP(..., reduce_results=self.experts.must_reduce_shared_expert_outputs())`. | `deepseek_v2.py:166-176` |
| MoE router | `gate(hidden_states)` then `grouped_topk(..., scoring_func=sigmoid, renormalize=True, num_expert_group, topk_group)` returns normalized top-k weights and ids. V1 has small-batch router GEMM specializations such as `dsv3_router_gemm` before grouped top-k. | `deepseek_v2.py:179-190`, `layer.py:1447-1461`; V1: `GateLinear`, `dsv3_router_gemm` |
| MoE route align | `moe_align_block_size(topk_ids, block_size_m, global_num_experts, expert_map)` produces `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded`. | `fused_marlin_moe.py:99-109` |
| MoE W13 | `ops.moe_wna16_marlin_gemm(..., top_k=topk, mul_topk_weights=apply_router_weight_on_input, use_fp32_reduce=True)`; Kimi path uses WNA16 INT4 experts. | `fused_marlin_moe.py:133-159` |
| MoE activation | `torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, 2 * N))`. | `fused_marlin_moe.py:161-163`, `csrc/activation_kernels.cu` |
| MoE W2 | `ops.moe_wna16_marlin_gemm(..., top_k=1, mul_topk_weights=not apply_router_weight_on_input, use_fp32_reduce=True)`. | `fused_marlin_moe.py:175-201` |
| MoE route sum | `torch.sum(intermediate_cache3.view(...), dim=1, out=output)` sums the top-k route rows. | `fused_marlin_moe.py:203-205` |
| MoE scale and TP reduce | For BF16, routed output is multiplied by `routed_scaling_factor`, added with shared output, then `maybe_all_reduce_tensor_model_parallel`. | `deepseek_v2.py:187-208` |
| Final logits | Final RMSNorm then LM head; sampling/logprobs live in vLLM sampling path rather than model file. | `deepseek_v2.py:724-725` |

## PegaInfer Current Decode Operator List

This list follows the current worker implementation. The static trace is now source-aligned for these high-level operators after the MLA trace fix below.

| Section | PegaInfer actual operator path | Source evidence |
| --- | --- | --- |
| Embedding | `embedding_batch_vocab_shard` then TP all-reduce through BF16-via-F32 bridge. | `batch_decode_trace.rs:49-63` |
| Attention input | `rms_norm_batch_into(hidden, input_norm)`. | `worker.rs:1777-1783` |
| MLA q down | `gemm_graphsafe(q_a_proj)` then `rms_norm_batch(q_a_norm)` then `gemm_graphsafe(q_b_proj)`. | `worker.rs:1784-1808` |
| MLA kv down | `gemm_graphsafe(kv_a_proj_with_mqa)` then `kimi_mla_split_compressed_kv` then `rms_norm_batch(kv_a_norm)`. | `worker.rs:1809-1827` |
| MLA RoPE split | `kimi_mla_rope_split_decode(q_proj, k_rope, cos, sin, positions)` produces `q_nope`, `q_pe`, and `append_kpe`. | `worker.rs:1839-1849` |
| MLA q absorb | `kimi_mla_absorb_q_nope(kv_b_proj, q_nope)` uses preloaded `kv_b_proj` weight; this is the PegaInfer equivalent of vLLM `q_nope @ W_UK_T`. | `worker.rs:1850-1855` |
| MLA cache append | `kimi_mla_paged_kv_append(compressed_normed, append_kpe, page tables, positions)` writes worker-owned paged MLA KV. | `worker.rs:1856-1868` |
| MLA attention | `kimi_flashinfer_batch_decode_mla(q_abs_nope, q_pe, ckv_cache, kpe_cache, page tables, request_indices, kv metadata)`. | `worker.rs:1880-1895` |
| MLA v up | `kimi_mla_v_up(kv_b_proj, latent)`; this is the PegaInfer equivalent of vLLM `_v_up_proj`. | `worker.rs:1907-1912` |
| MLA output projection | `gemm_graphsafe(o_proj)` then TP all-reduce through BF16-via-F32 bridge, then residual add. | `worker.rs:1913-1934`, `batch_decode_trace.rs:279-291` |
| Dense layer 0 MLP | post-attn RMSNorm, separate gate/up GEMMs, `silu_mul_batch`, down GEMM, BF16-via-F32 TP all-reduce, residual add. | `batch_decode_trace.rs:294-327` |
| MoE shared expert | post-attn RMSNorm; shared gate/up GEMMs, `silu_mul_batch`, shared down GEMM, BF16-via-F32 TP all-reduce. | `worker.rs:2201-2238` |
| MoE router | `kimi_router_noaux_tc_launch` with Kimi config, producing `router_topk_weight` and `router_topk_idx`. | `worker.rs:2262-2285` |
| MoE route align | `kimi_moe_marlin_align_block_size` builds local EP route metadata. | `worker.rs:2118-2127`, `batch_decode_trace.rs:360-377` |
| MoE W13 | `kimi_marlin_wna16_w13_gemm` using vLLM Marlin WNA16 package. | `worker.rs:2143-2153` |
| MoE activation | `kimi_marlin_w13_swiglu`. | `worker.rs:2154-2155` |
| MoE W2 | `kimi_marlin_wna16_w2_gemm` with top-k weights. | `worker.rs:2157-2166` |
| MoE route sum | `kimi_marlin_sum_topk_rows_f32`. | `worker.rs:2168-2169` |
| MoE combine | Current decode path uses `repeat_f32_for_reduce_scatter` + NCCL `reduce_scatter` for routed F32 bridge, then `scale_f32_in_place`, shared residual add, and `kimi_add_f32_bf16_to_bf16`. Older non-decode helper still has F32 all-reduce; decode trace should describe the decode path. | `batch_decode_trace.rs:410-460` |
| Final logits/top1 | final RMSNorm, LM head shard GEMM, `top1_batch`; worker reads local top1 ids/values back to host after graph replay and scheduler selects global max across ranks. | `batch_decode_trace.rs:74-96`, `worker.rs:797-824`, `scheduler.rs:528-604` |

## Count Snapshot

Current static trace regenerated locally after fixing MLA trace drift:

```text
calls 1947
428 gemm_graphsafe
245 rms_norm_batch
123 all_reduce
122 add_batch
120 kimi_marlin_wna16_gemm
61  kimi_mla_split_compressed_kv
61  kimi_mla_rope_split_decode
61  kimi_mla_absorb_q_nope
61  kimi_mla_paged_kv_append
61  kimi_flashinfer_batch_decode_mla
61  kimi_mla_v_up
61  silu_mul_batch
60  kimi_router_noaux_tc
60  kimi_moe_marlin_align_block_size
60  kimi_marlin_w13_swiglu
60  kimi_marlin_sum_topk_rows_f32
60  repeat_f32_for_reduce_scatter
60  reduce_scatter
60  scale_f32_in_place
60  kimi_add_f32_bf16_to_bf16
1   embedding_batch_vocab_shard
1   top1_batch
```

This count is now source-aligned for the high-level worker operators. It still folds BF16-via-F32 collectives into one logical `all_reduce` and does not count CUDA memset/memcpy nodes.

## Trace Drift Fixed In This Session

`pegainfer-kimi-k2/src/batch_decode_trace.rs` differed from `worker.rs` in the first draft of this document:

| Trace item | Current trace | Actual worker path | Effect |
| --- | --- | --- | --- |
| q/kv down projection | records `q_a` GEMM and `kv_a` GEMM separately | same | Real structural difference from vLLM; vLLM fuses these into `fused_qkv_a_proj`. Kept. |
| compressed KV split | missing | `kimi_mla_split_compressed_kv` | Fixed: now counted once per layer. |
| `kv_a_norm` | missing | `rms_norm_batch_into(compressed_kv, kv_a_norm)` | Fixed: RMSNorm count increased by 61. |
| decode `kv_b` GEMM | records `L*.attn.kv_b` as `gemm_graphsafe` | no full `kv_b` GEMM in decode; worker uses `kv_b_proj` weight in `absorb_q` and `v_up` custom kernels | Fixed: fake GEMM removed, GEMM count decreased by 61. |
| MLA cache append | missing | `kimi_mla_paged_kv_append` | Fixed: now counted once per layer. |
| all-reduce bridge | folded into one `all_reduce(dtype=bf16_via_f32)` | actual path is BF16-to-F32 kernel, NCCL F32 collective, F32-to-BF16 kernel | Fine for high-level op count, wrong for kernel launch count and CUDA graph node count. |
| top1 | `top1_batch` | kernel is `argmax_batch_bf16_cuda`; `ctx.sync()` + D2H id/value readback happen after graph body | The GPU op is counted, but graph-external host boundary is hidden. |

Patch range: `push_attention_layer` in `batch_decode_trace.rs` removed the fake `kv_b` GEMM and added `kimi_mla_split_compressed_kv`, `rms_norm_batch` for `kv_a_norm`, and `kimi_mla_paged_kv_append`.

Validation:

```bash
cargo fmt --all --check
PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins
PEGAINFER_CUDA_SM=90a cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_kernel_report -- \
  trace --source static --batch-size 4 --kv-len 1024 --out /tmp/kimi_decode_trace_fixed_bs4_kv1024.json
```

H20 validation used the same `cargo check` and static trace command under `/root/develop/xingming/pegainfer-kimi-k2-main` with `PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer-kimi-k2-main/.venv-kimi/bin/python`; output was `calls=1947` with the same op counts as above. `nvidia-smi --query-compute-apps` printed no active process after the trace parse.

## Path Differences That Matter

| Difference | vLLM | PegaInfer | Why it matters |
| --- | --- | --- | --- |
| MLA first projection | One `MergedReplicatedLinear` for `[q_lora_rank, kv_lora_rank + rope_dim]`. | Two GEMMs: `q_a_proj` and `kv_a_proj_with_mqa`. | PegaInfer pays one extra GEMM launch per layer and rereads hidden twice. This is the cleanest operator-list delta. |
| Dense gate/up | V1 can use fused `gate_up_proj`; V0 module-level path still exposes gate/up. | Separate gate and up GEMMs for dense layer and shared expert. | One dense layer only matters little; shared expert repeats 60 times and is a real candidate after trace is fixed. |
| Router GEMM | V1 has small-batch `dsv3_router_gemm` / `router_gemm_bf16_fp32` path before grouped top-k. | `kimi_router_noaux_tc_launch` is a single custom router/top-k kernel path. | Need compare microbench, not assume; router was ~3.7ms/step in old strong-sync profile. |
| MLA cache append and metadata | vLLM uses `concat_and_cache_mla`; FlashMLA prepares tile scheduler metadata and graph buffers. | PegaInfer uses `kimi_mla_paged_kv_append` and precomputed decode arena arrays. | Need compare metadata/cache append cost before changing attention kernels; trace currently hides this. |
| MLA q absorb/v up | vLLM uses `torch.bmm` with preprocessed `W_UK_T/W_UV`. | PegaInfer custom kernels `kimi_mla_absorb_q_nope` and `kimi_mla_v_up` over `kv_b_proj`. | Semantically aligned, but microbench should decide whether custom kernels or cuBLAS batched GEMM wins for bs1..4. |
| MoE WNA16 | Both use Marlin WNA16 route align, W13, SiLU, W2, sum. | PegaInfer has persistent workspace and explicit local EP route metadata. | Main MoE kernel choice is already aligned; next work is route histogram/tail and combine, not replacing WNA16. |
| Routed combine | vLLM EP path maps local experts via `expert_map`; final tensor-parallel reduce happens through vLLM distributed path. | PegaInfer currently uses NCCL bridge: local sum -> repeat -> reduce-scatter -> scale -> residual. | This is not PPLX EP; it is graph-capturable but likely still extra data movement. |
| TP collectives | vLLM parallel layers hide TP reductions; BF16 path does not visibly use our BF16-via-F32 bridge. | PegaInfer uses BF16-via-F32 bridge for hidden all-reduces because BF16 collective changed greedy output. | This is correctness-driven overhead; replacing it needs external vLLM greedy/top-k gate. |
| Sampling/top1 | vLLM sampling/logprobs is integrated with its sampler path. | PegaInfer graph body ends at local top1; worker D2H reads local top1 and scheduler CPU-selects across ranks. | This graph-external boundary is real, but prior profile says it is not the largest item; fix after trace/accounting is accurate. |

## Next Actions

1. Run corrected `kimi_model_report decode --source runtime` on H20 so the measured perf ledger uses the fixed trace shape, not the old `1825`-call shape.
2. Evaluate `fused_qkv_a_proj` as the first structural MLA optimization candidate: it is bs1..4-general, preserves graph-readiness, and removes one per-layer GEMM launch plus one hidden read.
3. After corrected H20 model report exists, evaluate shared expert `gate/up` fusion and router microbench against vLLM-style small-batch kernels.
4. Keep MoE WNA16 kernel path unchanged until the corrected report shows a measured win candidate; current vLLM/PegaInfer MoE compute path is already structurally close.
