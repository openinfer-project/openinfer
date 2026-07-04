# GLM5.2 DP1 EP8 Decode Plan

> **TL;DR:** Five sub-PRs on top of the merged load-weight scaffold (PR #476) to reach an EP8 DSA decode serving path. **All five are merged**: #477 MLA brick, #489+#521 DSA indexer, #499 oracle harness, #530 EP1 decoder-layer forward, #531 DeepEP EP8 MoE + first e2e generation (`ep8-deepep-moe.md`), #535 PR5a perf pass (`bs1-decode-serial-overhead.md`), #537 PR5b DP8 scheduler (`dp8-scheduler.md`), #543 PR5c whole-step decode CUDA graph — 200 → 22.5 ms/step (`whole-step-decode-graph.md`). The campaign continued past this plan into dynamic batching: D1 fixed 8-row batch (#546, `batched-decode.md`) and D2 continuous batching ({1,8} bucket graphs + multi-slot admission, `continuous-batching.md`). Prefill rides decode kernels token-by-token; the standing decision is prefill-by-vLLM (P/D disaggregation), so no dedicated prefill path is planned in this line.
>
> **Last touched:** 2026-07

## Current state on main

- PR #476 load-weight scaffold (+ #478 feature gating): EP8 rank-sliced manifest, coalesced H2D into rank-local slabs, fail-closed rejecting coordinator.
- PR #477 (PR1): MLA decode brick — projection/absorb/cache-pack/FlashMLA sparse decode, full top-k.
- PR #499: self-contained accuracy oracle harness (`tools/accuracy/glm52_oracle.py` + probe gates); MLA gate green on jz38 H200.
- PR #489 + #521 (PR2): DSA indexer kernel ops + `glm52_indexer_forward`; indexer oracle gate green (2013/2048 slot overlap).
- Still missing: dense/MoE/bookend forward, layer composition, executor, scheduler. `glm52_trtllm_grouped_fp8` CUDA compute is compiled but has no safe Rust wrapper; `deepgemm_grouped` GEMM entry is a `NOT_SUPPORTED` stub (metadata kernel is real).

The old `feat/glm52-dp8-ep8` remote branch (27 commits, PP8 pivot) is retained as a cherry-pick source for per-layer bricks that are PP-independent. The PP spine, stage coordinator, and stage-sliced weight loading are discarded — EP8 needs rank-axis expert slicing, which main already has.

## GLM5.2 DSA attention data flow

GLM5.2 uses DeepSeek Sparse Attention: an indexer computes per-token similarity against a separate index-K cache, selects top-k=2048 sparse slots, and the MLA decode attends only over those slots. The full per-layer decode path:

```
hidden[6144]
  │
  ├── input_layernorm
  ├── MLA projection (q_a -> q_b -> q; kv_a -> kv_b absorb)
  │     ├── q_a_proj: fp8 linear [2048, 6144]          (TRTLLM CUTLASS)
  │     ├── q_a RMSNorm
  │     ├── q_b_proj: fp8 linear [16384, 2048]         (TRTLLM CUTLASS)
  │     ├── kv_a_proj: fp8 linear [576, 6144]          (TRTLLM CUTLASS)
  │     ├── kv_a RMSNorm (compressed_kv [512])
  │     ├── absorb: ql_nope = q_nope @ W_UK            (cuBLAS strided batched)
  │     ├── query assemble: [ql_nope | rope(q_pe)]     (hand-written)
  │     └── cache pack: write new token to paged cache  (hand-written)
  │
  ├── DSA indexer (selects sparse top-k for this token)
  │     ├── index query quant: fp8 per-token-group     (hand-written, reuse moe_quant)
  │     ├── index-K cache: already written per token    (hand-written)
  │     ├── DeepGEMM paged MQA logits                  (vendored DeepGEMM, new C ABI wrapper)
  │     ├── FlashInfer deterministic top-k (K=2048)    (vendored FlashInfer, new wrapper)
  │     └── offset -> KV slot conversion               (hand-written, new)
  │
  ├── FlashMLA sparse decode (attend over top-k slots) (vendored, on main)
  ├── back: v = latent @ W_UV, o_proj                  (cuBLAS + TRTLLM CUTLASS)
  └── residual add
```

PR1 covers the MLA projection/absorb/cache/decode substrate with **full top-k** (all cached tokens 0..position, -1-padded to 2048). At context <= 2048 this is bit-identical to DSA — the indexer is a no-op selector. PR2 replaces full top-k with the real DSA indexer chain.

## Sub-PR breakdown

### PR1 — `feat/glm52-mla-decode-brick`

Single-layer MLA decode forward (`hidden[6144] -> o[6144]`), bs=1, full top-k (context <= 2048). This validates the MLA projection / absorb / cache-pack / FlashMLA sparse decode path — the attention correctness foundation.

**Scope:**

- `fp8.rs` — shared FP8 block-scaled projection primitive: `ProjWeight` (device-resident fp8 weight + scale), `fp8_linear` (quant activation -> relay scale into TRTLLM col-major TMA layout -> launch), `dequant_kv_b` (host-side fp8->bf16 for W_UK/W_UV absorb factors). MLA/dense/MoE share this; not MLA-specific.
- `mla_decode.rs` — `Glm52MlaLayerWeights` (6 ProjWeight + 2 ln gamma + w_uk/w_uv bf16) + `glm52_mla_decode_forward(ctx, w, hidden, cos, sin, cache, position, topk, contract) -> o[6144]`. Only `from_host` constructor (test path); `from_device` (production, against the existing EP8 slab) deferred to PR4.
- Kernel ops cherry-picked from the old branch (see "Kernel inventory" below).

**Not in PR1:**

- DSA indexer chain (MQA logits, deterministic top-k, slot conversion) — PR2.
- `decode_meta.rs` / batch geometry / page table / slot mapping — PR1 takes `num_blocks`/`position`/`topk` as direct contract args.
- Runner/coordinator changes — still rejecting.
- Dense MLP / MoE / bookends — PR3.
- Server wiring — unchanged.

**Test:** Oracle gate deferred — the prototype's fixture pipeline (HF forward dump → `layer0.npz` → probe bins → Rust test) was not self-contained (the dump script was never in the repo). A follow-up will design a reproducible oracle pipeline before claiming correctness. See `mla-decode-brick.md`.

**Kernel inventory (cherry-pick from `feat/glm52-dp8-ep8`):**

| op | file | backend | hand-written CUDA? |
|---|---|---|---|
| `gemm_strided_batched_bf16` | `ops/linear.rs` | cuBLAS `cublasGemmStridedBatchedEx` | no |
| `glm52_trtllm_fp8_linear` | `ops/glm52/trtllm_linear.rs` + `csrc/glm52/glm52_trtllm_grouped_fp8.cu` (m=1 path) | TRTLLM CUTLASS `CutlassFp8BlockScaleGemmRunner` | no (vendored) |
| `glm52_flashmla_sparse` | already on main | vendored FlashMLA V32 | no |
| `glm52_fp8_per_token_group_quant` | `ops/glm52/moe_quant.rs` + `csrc/glm52/glm52_moe_quant.cu` | **hand-written** (180 lines) | **yes** |
| `glm52_mla_query_assemble` / `glm52_mla_cache_pack` | `ops/glm52/mla_assembly.rs` + `csrc/glm52/glm52_mla_assembly.cu` | **hand-written** (142 lines) | **yes** |

**Hand-written kernel note (perf debt):**

`glm52_moe_quant.cu` and `glm52_mla_assembly.cu` are hand-written CUDA. Both are memory-bound elementwise/reduce kernels (per-128-group amax -> e4m3 quant; query concat + interleave RoPE + cache pack), not GEMM/attention compute — the tile-schedule risk is low. `mla_assembly`'s RoPE device function mirrors `openinfer-kimi-k2`'s `rope_out` and was bit-for-bit oracle-validated in the old branch. `moe_quant` implements the standard DeepGEMM/FlashInfer per-token-group FP8 contract (group=128, f32 scale, e4m3, amax/448).

**These two kernels are correct but not tuned.** They are single-issue-per-element launchers with no vectorized load/store, no shared-memory coalescing beyond the reduce tree, and no occupancy targeting. When decode TPOT is measured (PR5), they are the first candidates for an ncu pass. If a fused alternative appears in vendored FlashInfer/TRTLLM (e.g. `per_token_group_quant` with a C ABI), the hand-written version should be replaced, not optimized in place. See the local `docs/private/glm52/tokenspeed-kernel-gap.md` `DSA FP8 token-group quant` entry for the upstream contract reference.

**Gap-doc cross-reference:**

- `glm52_fp8_per_token_group_quant` -> gap-doc `DSA FP8 token-group quant` (P0 #1). PR1 covers the quant kernel itself; the DSA indexer caller (quant q for MQA logits) lands in PR2.
- `glm52_mla_cache_pack` -> gap-doc `DSA index-K cache set/gather` (P0 #2), MLA KV cache-write half (656-byte fp8_ds_mla token). The separate index-K cache (128-dim, DeepGEMM paged MQA logits layout) lands in PR2.
- `glm52_mla_query_assemble` -> not separately listed in gap doc; subsumed under `Blackwell sparse MLA` query contract. PR1 uses the SM90 FlashMLA sparse path (already on main), not the Blackwell TRTLLM sparse MLA that gap-doc P0 #6 calls for.
- `glm52_trtllm_fp8_linear` -> gap-doc `Dense FP8 block-scale GEMM` (P1 #1). PR1 uses it for MLA projections; the dense-MLP and MoE-expert callers land in PR3/PR4.

### PR2 — `feat/glm52-dsa-indexer`

The DSA indexer chain that replaces PR1's full top-k with sparse top-k=2048. This is the PR that makes GLM5.2 attention actually DSA.

**Scope:**

- `indexer.rs` (model crate) — `Glm52IndexerLayerWeights` (indexer q/k projections + Hadamard + RoPE) + `glm52_indexer_forward(ctx, w, hidden, cos, sin, index_k_cache, position, block_table, seq_lens) -> topk_indices[2048]`.
- Kernel ops (see inventory below).

**Kernel inventory:**

| op | file | backend | hand-written CUDA? |
|---|---|---|---|
| `glm52_indexer_k_quant_and_cache` | `ops/glm52/indexer.rs` + `csrc/glm52/glm52_indexer.cu` (quant+cache insert half) | **hand-written** (258 lines, cherry-pick) | **yes** |
| `glm52_indexer_k_gather_quant_cache` | same file (gather half) | **hand-written** (same file) | **yes** |
| `glm52_deepgemm_paged_mqa_logits` | `ops/glm52/deepgemm_mqa.rs` + `csrc/glm52/glm52_deepgemm_mqa.cu` (new) | vendored DeepGEMM `sm90_fp8_paged_mqa_logits` | no (vendored, new C ABI wrapper) |
| `glm52_flashinfer_topk_2048` | `ops/glm52/topk.rs` + `csrc/glm52/glm52_topk.cu` (new) | vendored FlashInfer `TopKDispatch` | no (vendored, new C wrapper) |
| `glm52_indexer_local_topk_to_slots` | `ops/glm52/indexer.rs` + `csrc/glm52/glm52_indexer.cu` (new kernel) | **hand-written** (new) | **yes** |

**Vendored wrapper notes:**

- **DeepGEMM paged MQA logits**: the vendored entry (`sm90_fp8_paged_mqa_logits` in `third_party/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp`) is a JIT-compiled kernel launched through DeepGEMM's `device_runtime`/`compiler`/`launch_kernel` path with `torch::Tensor` TMA descriptors. The new C ABI wrapper must either (a) replicate the TMA descriptor construction + JIT launch without torch, or (b) call the existing torch-bound entry through a thin C shim. Option (a) is preferred (no torch dependency at runtime) but is the main engineering risk of this PR. Fail-closed if the JIT runtime is not initialized.
- **FlashInfer deterministic top-k**: `TopKDispatch` (in `third_party/flashinfer/include/flashinfer/topk.cuh`) is a C++ template callable directly from a `.cu` file — no torch dependency. The existing `csrc/shared/flashinfer_top1.cu` already wraps `TopKDispatch` for K=1; the new wrapper extends it to K=2048 with `deterministic=true`, `tie_break=TopKTieBreak::Small`, `dsa_graph_safe=true` (matching TokenSpeed's `deterministic_decode_topk` contract from gap-doc `Decode deterministic top-k`). `FILTERED_TOPK_MAX_K=2048` so K=2048 is the max supported.

**Hand-written kernel note (perf debt):**

`glm52_indexer.cu` (quant + cache insert + gather) is hand-written CUDA, cherry-picked from the old branch. It is memory-bound (fp8 quant + scatter write / gather read into the DeepGEMM block-split paged layout: `[block_size * 128 fp8][block_size * 4 f32 scale]` per block). Same perf-debt classification as PR1's hand-written kernels — correct, not tuned, first ncu candidate. The `local_topk_to_slots` conversion is a small int32 index-remap kernel (block table lookup + page stride arithmetic), also hand-written and not tuned.

The old branch's `glm52_indexer_topk_2048_cuda` was a **stub** returning `NOT_SUPPORTED` — PR2 replaces it with the FlashInfer `TopKDispatch` wrapper. Do not cherry-pick the stub.

**Gap-doc cross-reference:**

- `glm52_deepgemm_paged_mqa_logits` -> gap-doc `DSA decode indexer logits` (P0 #3).
- `glm52_flashinfer_topk_2048` -> gap-doc `Decode deterministic top-k` (P0 #4).
- `glm52_indexer_local_topk_to_slots` -> gap-doc `top-k offset to KV slot` (P0 #5).
- `glm52_indexer_k_quant_and_cache` -> gap-doc `DSA index-K cache set/gather` (P0 #2).
- Hadamard rotate -> gap-doc `DSA Hadamard rotate` (P1 #4). PR2 includes a host-side or simple-GPU Hadamard for correctness; the Dao-AILab `fast-hadamard-transform` CUDA port is a follow-up if the naive version is a measured bottleneck.

**Not in PR2:**

- Blackwell TRTLLM sparse MLA (gap-doc P0 #6) — PR2 still uses the SM90 FlashMLA sparse path from PR1, now fed with real sparse top-k instead of full top-k. The TRTLLM sparse MLA launcher is in vendored FlashInfer (`trtllm_paged_attention_decode` with `sparse_mla_top_k`) but is not yet compiled into the build; that is a separate PR after PR5 measures whether SM90 FlashMLA sparse is the bottleneck.
- Prefill indexer logits (contiguous MQA logits, `fp8_mqa_logits` non-paged) — decode-first; prefill rides the decode path token-by-token.

**Test:** `tests/indexer_oracle.rs` — load layer-0 indexer weights + index-K cache fixture, run indexer forward at a context where sparse top-k != full top-k (e.g. context=4096, top-k=2048), compare the selected slot indices against HF oracle. Also a short-context regression: at context <= 2048 the indexer top-k must equal full top-k (validates PR1 compatibility).

### PR3 — `feat/glm52-ep1-forward`

**Superseded by `ep1-forward.md`** (corrections: brick-level oracle gates instead of full-model e2e — the model doesn't fit on one GPU; cross-layer top-k sharing threads through the layer loop; MoE chain shaped to the DeepEP v2 elastic shim contract with graph capturability as the bar). Original scope kept below for history.

Dense MLP (first 3 layers) + bookends (embed / final RMSNorm / lm_head) + routed/shared MoE decode (EP1, local experts). Full EP1 decode forward with DSA attention, e2e bs=1 generation gate. Prefill rides decode kernels token-by-token.

Reuses `fp8.rs` from PR1 and `indexer.rs` from PR2. Adds `dense.rs`, `bookend.rs`, `moe_decode.rs` (EP1 path — grouped FP8 GEMM with all 256 experts local). `glm52_trtllm_grouped` kernel op comes in here (first MoE caller).

### PR4 — `feat/glm52-ep8-deepep-moe`

**Superseded by `ep8-deepep-moe.md`** (corrections there: the shim is instantiated per model config rather than reached through `openinfer-comm`; expert weights are placed into their packed layout at H2D time because post-load repacking cannot fit HBM; the e2e generation gate lands here with a minimal bs=1 coordinator). Original scope kept below for history.

Replace EP1 MoE with EP8 DeepEP all-to-all. Rank 0 owns experts 0..31, ranks 1..7 own 32 each. `from_device` constructors land here (first production consumer of the resident weights). Gate: multi-rank MoE layer oracle + full-model e2e generation.

### PR5 — `feat/glm52-server-wiring`

Replace rejecting coordinator with real `EngineHandle` request flow + scheduler + decode CUDA Graph capture (pointer-stable pre-allocated buffers) + prefill-via-decode path. e2e scheduler liveness test (aligns with qwen35 `e2e_scheduler`).

## Why this order

PR1 first because MLA projection/absorb/cache-pack is where layout and RoPE mistakes hide — isolating it with full top-k (DSA-equivalent at short context) gives a clean correctness floor. PR2 adds the DSA indexer chain as a separate diff so MQA logits / top-k / slot conversion bugs are debugged against PR1's known-good MLA decode, not mixed with projection bugs. PR3 composes the full EP1 forward so PR4's EP8 swap has a known-good EP1 baseline. PR4 is isolated to the MoE all-to-all boundary. PR5 is pure runtime plumbing once forward is proven.

## Preparation

- **Read:**
  - `docs/private/glm52/tokenspeed-kernel-gap.md` — full TokenSpeed GLM5.2 kernel DAG, hand-written-vs-vendored source tags, P0/P1/P2 priority index. This plan follows the gap-doc P0 order for the DSA indexer chain. (Local-only, not in PR.)
  - `docs/models/glm52/load-weights-dp1-ep8.md` — PR #476 execution record; the EP8 slab layout PR4 will consume.
  - `openinfer-glm52/src/lib.rs`, `src/runner.rs`, `src/weights.rs` — current load-only surface.
  - `openinfer-kernels/src/ops/glm52/flashmla_sparse.rs` — the SM90 sparse decode wrapper already on main.
  - `openinfer-kernels/third_party/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp` — vendored paged MQA logits entry PR2 wraps.
  - `openinfer-kernels/third_party/flashinfer/include/flashinfer/topk.cuh` — vendored `TopKDispatch` template PR2 wraps.
  - `openinfer-kernels/csrc/shared/flashinfer_top1.cu` — existing `TopKDispatch` wrapper for K=1; the pattern to extend for K=2048.
  - `openinfer-kimi-k2/src/runner/executor/tp1_dp8.rs`, `moe_deepep.rs` — EP8 executor + DeepEP MoE shape to mirror in PR4.
  - `feat/glm52-dp8-ep8:openinfer-glm52/src/mla_decode.rs`, `fp8.rs` — cherry-pick source for PR1 bricks (PP-independent).
  - `feat/glm52-dp8-ep8:openinfer-glm52/src/indexer.rs` (if present), `openinfer-kernels/src/ops/glm52/indexer.rs`, `csrc/glm52/glm52_indexer.cu` — cherry-pick source for PR2 bricks. Note: the old branch's top-k was a stub; do not cherry-pick `glm52_indexer_topk_2048_cuda`.
  - `feat/glm52-dp8-ep8:openinfer-kernels/csrc/glm52/glm52_mla_assembly.cu`, `glm52_moe_quant.cu`, `glm52_indexer.cu` — the hand-written kernels.
- **Relevant history:**
  - The old branch was PP8 (serial 8-stage), not EP8. Its MLA/MoE/dense/indexer per-layer bricks are PP-independent and reusable; its PP spine, stage coordinator, and stage-sliced weight loading are not.
  - The old branch's DSA indexer was incomplete: `glm52_indexer_topk_2048_cuda` returned `NOT_SUPPORTED`, and `decode.rs` used `write_topk` (full top-k `[0,1,..,position,-1,...]`) as a placeholder. PR2 must implement the real top-k via FlashInfer `TopKDispatch`, not cherry-pick the stub.
  - Gap doc already classified every TokenSpeed GLM5.2 kernel by source (public GitHub / vendored / wheel-only) and flagged which are hand-written vs wrapper.
