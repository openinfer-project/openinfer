# GLM5.2 bs=1 decode serial-overhead pass (PR5a)

> **TL;DR:** First measured perf pass on the PR4 bring-up path: **101–103 → 46–50 ms/step (~2.2×)** on jz-38 8×H200, output byte-identical, all PR1–PR4 oracle gates green. nsys showed rank-0 kernel-busy was only 56% of step wall and 35% of *all* GPU work was the re-quant/SiLU chain running at the fixed 10240-row worst case; the fixes are (1) row coverage bounded by the coordinator's global token count (512 rows at bs=1) with a **device trap** in the metadata kernel if a real segment ever ends past the bound, (2) the MoE chain's buffers made persistent in `Glm52MoeEp8State` (was ~11.6k `cuMemAllocAsync`+memset+free per step across ranks — 73% of CUDA API time), (3) FlashMLA tile-scheduler metadata computed once at build instead of 78×/step, rope tables device-resident. Remaining decode wall is ~46% launch-overhead gap (4155 kernels/step) plus ~5 ms/step of residual small `alloc_zeros` on the MLA/indexer spine — both are the PR5c CUDA-graph target.
>
> **Last touched:** 2026-07

## A/B (jz-38 8×H200, bs=1 greedy, 133-step requests)

| | main `d5cb244` | PR5a | |
|---|---|---|---|
| ms/step | 103.2 / 100.8 / 100.9 | 47.1 / 46.7 / 46.2 | −54% |
| byte-determinism (2 runs) | PASS | PASS | |
| output text | " Paris. Distance from Paris to Lyon is 391 km…" | identical | numerics unchanged |

Gates: MLA / dense / MoE-EP1 / bookend / indexer / EP8 all green; EP8 62/64 with the two outlier probes' engine values bit-identical to the PR4 record (0.021606 / 0.019165) — the row-bound change never touches a real data row.

## Where the 100 ms went (nsys, rank-0 median step)

Before (119.6 ms wall under nsys): kernel busy 66.8 ms (56%), gap ~53 ms. Busy was dominated by capacity-proportional waste, not math:

| kernel | before | after | fix |
|---|---|---|---|
| `fp8_per_token_group_quant` | 26.9 ms | 2.3 ms | rows 10240 → `bound_rows` (512 at bs=1) |
| `silu_and_mul…quant` | 10.75 ms | 0.78 ms | same |
| scale TMA relayout | 2.23 ms | 0.79 ms | same (GEMM `m_capacity` too) |
| memset (from `alloc_zeros`) | 10.6 ms | 5.1 ms | MoE chain buffers persistent |
| `get_mla_metadata` | 1.98 ms | — | computed once at model build |
| grouped/plain FP8 GEMM | 13.5 ms | 13.4 ms | real work, untouched |

After: 53.4 ms wall, busy 28.7 ms (54%) — the remaining gap is host launch overhead (4155 kernels/step ≈ 150k CUDA API calls) and the residual `alloc_zeros` on the MLA/indexer spine (`latent/lse/lse_accum/o_accum/query` per layer). Both are what CUDA-graph capture (PR5c) removes; the graph prerequisite (pointer-stable persistent workspace, host-quiet chain) is laid here.

## The row bound and its enforcement

`bound_rows = min(expanded, g·TOPK + (ALIGN−1)·min(g·TOPK, n_local))` where `g` is the step's global dispatched-token count — the per-`g` instantiation of the shim's own `kDecodeWorstExpandedTokens` formula. Every rank must agree on `g` (`GLM52_DECODE_GLOBAL_TOKENS`, single definition in `model.rs`); the grouped-GEMM metadata kernel receives `bound_rows` as its capacity and **`__trap()`s if any aligned expert segment ends past it** — it is the only kernel in the chain that sees the real psum, and every downstream consumer would otherwise silently multiply stale activations from the previous layer into real outputs. A cross-rank `g` disagreement is therefore a crash at the offending layer, not deterministic garbage.

## Next

- PR5b: DP8 scheduler + batching (the DP1 coordinator's `g=1` becomes the scheduler's per-step batch; `bound_rows` already scales).
- PR5c: decode CUDA-graph capture — kills the ~25 ms/step launch gap; extend the persistent-workspace treatment to the MLA/indexer spine as part of it.
