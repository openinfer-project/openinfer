# GLM5.2 Decode — Kernel Reuse Map

> **TL;DR:** A 7-op survey (ultracode workflow, 2026-06-27) across vendored FlashMLA/DeepGEMM/FlashInfer, sibling vLLM, and the repo's own kimi/deepseek/glm52 kernels found that **5 of the 7 GLM5.2 decode op-families are already CUDA-implemented and Rust-wrapped** in `openinfer-kernels` (built during the now-dropped DP/EP work and reusable as-is): MLA sparse decode, MoE grouped FP8 GEMM, router noaux_tc, dense/shared/MLA-proj FP8 linear, and embed/norm/lmhead/sampling. **Only two pieces need new CUDA:** the DSA **indexer top-2048** (port vLLM `persistent_topk_kernel`) and the **interleaved-RoPE** wiring. The rest of the work is **glue + orchestration**: build the 576-d absorbed query, pack the 656-byte V3.2 KV cache, and sequence the per-layer forward across the PP8 spine. So the forward is far less greenfield than feared — the heavy GEMMs/attention exist.
>
> **Last touched:** 2026-06

## Reuse table (per decode op)

| Op | Reuse | Wrapped? | Entry symbol | Notes / gaps |
|---|---|---|---|---|
| **MLA sparse decode** | vendored FlashMLA V3.2 `run_flash_splitkv_mla_fp8_sparse_kernel<V32,64>` | ✅ FFI+ops | `Glm52FlashMlaSparseDecode` / `glm52_flashmla_sparse_decode_launch_cuda` (`csrc/glm52/glm52_flashmla_sparse.cu`) | q`[B,1,64,576]`bf16, packed_kv`[blocks,64,1,656]`fp8, topk`[B,1,2048]`, out`[B,1,64,512]`bf16, `sm_scale=1/√256`. Sched-meta kernel already wrapped. **Glue:** build 576 query + 656 cache (below). |
| **MoE grouped FP8 GEMM** | TRTLLM Cutlass FP8 blockscale grouped | ✅ FFI+ops, **complete** | `glm52_trtllm_grouped_fp8_launch` (`src/ops/glm52/trtllm_grouped.rs`) | W13: act`[m,6144]`fp8 @ wt`[G,4096,6144]` → `[m,4096]`bf16; W2: n=6144,k=2048. `expert_offsets[G+1]`. DeepGEMM path is a **stub** (`NOT_SUPPORTED`) — use TRTLLM. |
| **Router noaux_tc** | repo glm52 router (kimi-derived) | ✅ FFI+ops | `glm52_router_noaux_tc_launch` (`src/ops/glm52/router.rs`, `csrc/glm52/glm52_router.cu`) | gate`[256,6144]`bf16 → sigmoid+bias → group top-8 → norm → ×scale. **Pass `route_scale=2.5`** (routed_scaling_factor), NOT the 1.0 placeholder const. Not yet called from any forward. |
| **Dense/shared/MLA-proj FP8 linear** | TRTLLM `CutlassFp8BlockScaleGemmRunner<fp8,fp8,bf16>` | ✅ FFI+ops, **complete** | `Glm52TrtllmFp8LinearContract` / `glm52_trtllm…linear` (`src/ops/glm52/trtllm_linear.rs`) | bs=1 `m=1`. Shapes: dense (12288,6144)/(6144,12288); shared (6144,2048); q_a(2048,6144), kv_a(576,6144), q_b(16384,2048), kv_b(28672,512), o_proj(6144,16384); indexer wk(128,6144), wq_b(4096,2048). |
| **Embed / final norm / lm_head / sample** | repo shared ops | ✅ exist | `embedding_batch` + `rms_norm_into` + `gemm_into` + `flashinfer_top1_batch_into` | vocab 154880, hidden 6144. **Glue:** sequence the 4 + 1 MiB sampling row-state scratch. |
| **Indexer top-2048 (DSA)** | vLLM `persistent_topk_kernel<2048>` (`vllm/csrc/libtorch_stable/persistent_topk.cuh`) | ❌ **new CUDA** | port into `csrc/glm52/glm52_indexer.cu:232` `glm52_indexer_topk_2048_cuda` (stub exists; Rust ops already expect it) | ~250 lines (CG + bitonic + tree select). Plus the index-logits GEMM (wq_b/wk/k_norm, 32 heads × 128). **Hardest slice.** |
| **RMSNorm + RoPE + FP8 quant** | norm `rms_norm_batch_into` ✅ + quant `glm52_fp8_per_token_group_quant_bf16_cuda` ✅ + RoPE ❌ | partial | compose | **CRITICAL: `rope_interleave=True`** — GLM5.2 uses INTERLEAVED RoPE (`q[::2]`/`q[1::2]` per cos/sin), NOT GPT-J. `deepseek_apply_rope_pair` is GPT-J → need the interleaved variant. `rope_theta=8e6`. |

## Two correctness notes (verified against official config 2026-06-27)

1. **Router scale = `routed_scaling_factor` = 2.5.** The kernel's `route_scale` is applied as `scale = route_scale / selected_sum` (normalize top-8 to 1, then ×scale) — correct for `norm_topk_prob=True` + `scoring_func=sigmoid`. The `GLM52_ROUTER_WEIGHT_SCALE=1.0` in `router.rs` is a test placeholder; the forward must feed the parsed 2.5 (`config.rs` already reads `routed_scaling_factor`).
2. **Interleaved RoPE.** `rope_interleave=True`. Do not reuse a GPT-J/non-interleaved rope. HF reference: `transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py` `rotate_half`/`apply_rotary_pos_emb` (the `x1=x[..., ::2]` interleave).

## Remaining work, mapped to the build Slices (`pp-decode.md`)

- **Slice 2 (weights load-plan):** invert the DP8 32-expert-per-rank packaging (`weights/package.rs` `local_experts = experts/ep_world`) to PP8 **all-256-experts per sparse-layer on its stage**; verify all FP8 shapes against the official standard layout (the `mla_layout_probe` IT).
- **Slice 3 (MLA attention):** wire `Glm52FlashMlaSparseDecode`. Glue to build: q absorption `q_nope@W_UK_T` → concat `rope(q_pe)` → `[B,1,64,576]`; KV pack `fp8(kv_c)+scales+bf16(rope(k_pe))` → 656B; v_up `latent@W_UV` → o_proj. All linears via the wrapped TRTLLM FP8 linear.
- **Slice 4 (indexer):** new `persistent_topk` CUDA + index-logits GEMM. Validate top-2048 indices against the HF `GlmMoeDsaIndexer` on one layer.
- **Slice 5 (MoE):** router (`route_scale=2.5`) → local top-8 route/permute → `glm52_trtllm_grouped_fp8_launch` W13 → SwiGLU+quant → W2 → combine. EP1/bs=1 collapses to G=8 active groups (no all-to-all).
- **Slice 6 (dense+bookends):** dense MLP (layers 0-2) + shared expert via wrapped FP8 linear; embed/final-norm/lm_head/top-1 sample; residual/norm sequencing.
- **Slice 7 (full PP8 graph + TPOT):** replace `runner.rs` rejecting stub (`run_rejecting_dp_coordinator:208`) with the PP8 coordinator over the Slice-0 spine; capture the full per-stage graph; measure TPOT<10ms.

## Sources

Survey transcript: workflow `glm52-kernel-align` (run `wf_bf977f38-dbf`). Numeric oracle: `transformers/.../glm_moe_dsa/modeling_glm_moe_dsa.py`. vLLM call patterns: `vllm/v1/attention/backends/mla/flashmla_sparse.py`, `deepseek_v2.py` (`Indexer`/`DeepseekV32IndexerBackend`).
