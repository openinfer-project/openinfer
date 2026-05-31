# kimi_router_noaux_tc Report

> **TL;DR:** `decode.moe.router` is control/launch limited on H20, not memory-bandwidth-bound. Baseline TP1 PPLX `bs=8,ctx=1` is `60.91us/call` (`3.655ms` per 60 MoE layers). The obvious fast path, changing router logits GEMM from pedantic BF16->F32 to tensor-op `CUBLAS_COMPUTE_32F`, improves the row to `28.12us/call` but changes TP1 DP8 generated token traces (`30/64` mismatches), so no router optimization is adopted.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's `wiki/patterns/low-sm-utilization.md` matches this row: the router launches tiny grids (`48`, `12`, and `8` CTAs on a `78`-SM H20), so the bottleneck is wave/launch geometry rather than HBM peak. `sources/prs/sglang/PR-20755.md` is directly relevant as a directional lead: SM90+ MoE routers have used small-GEMM/tinygemm paths for router GEMM speedups.

For Kimi, that lead has a correctness caveat. The router logits participate in top-k routing and downstream generated tokens. A faster math mode is not acceptable unless it preserves the TP1 DP8 token trace or the project explicitly changes the accuracy contract.

## NCU Conclusion

Profiled `decode.moe.router` through `kimi_tp1_pplx_decode_bench` with per-rank `bs=8`, `ctx=1`.

| Item | Value |
|---|---:|
| Shape | hidden rows `8`, hidden dim `7168`, experts `384`, topk `8` |
| Calls per decode step | `60` MoE layers |
| Event timing | `60.910 us/call`, `3.655 ms/step` |

Sub-kernel NCU:

| Kernel | Duration | Grid / block | Waves/SM | DRAM | Compute | No eligible |
|---|---:|---:|---:|---:|---:|---:|
| `gemmSN_TN_kernel` logits GEMM | `81.7-82.7 us` | `48` x `128` | `0.12` | `1.39-1.40%` | `8.85-8.97%` | `89.08-89.12%` |
| `router_scores_kernel` sigmoid+bias | `2.94-3.01 us` | `12` x `256` | `0.02` | `0.13%` | `0.42-0.45%` | `95.08-95.83%` |
| `router_topk_normalize_kernel` | `10.08-10.24 us` | `8` x `512` | `0.03` | `0.04%` | `1.96-1.99%` | `76.29-76.45%` |

NCU full-set replay inflates absolute kernel durations versus CUDA-event timing, but the relative diagnosis is stable: the row is small-grid/control limited. The logits GEMM dominates and uses pedantic BF16->F32 math, which avoids the fast tensor-op path.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Baseline router | TP1 PPLX `bs=8,ctx=1`: `60.910 us/call`, `3.655 ms` per step. | Baseline. |
| cuBLASLt logits GEMM with current pedantic semantics | Standalone harness: cuBLAS `79.26 us/call`, cuBLASLt `79.28 us/call`. | Rejected; no speedup when preserving pedantic math. |
| cuBLAS logits GEMM with `CUBLAS_COMPUTE_32F` tensor-op path | Standalone harness: `9.61 us/call`; production router row: `28.122 us/call`, `1.687 ms` per step (`2.17x` row speedup). | Rejected; TP1 DP8 bs64/o5 token trace has `30/64` mismatches versus baseline. |
| cuBLASLt logits GEMM with `CUBLAS_COMPUTE_32F` tensor-op path | Standalone harness: `13.72 us/call`. | Rejected; slower than cuBLAS fast32 and shares the same correctness risk. |

Correctness evidence for the rejected fast32 path:

| Check | Baseline | Fast32 |
|---|---:|---:|
| TP1 DP8 PPLX bs64/o5 completion | `64/64` | `64/64` |
| First decode p50 | `38.47 ms` | `34.03 ms` |
| Steady TPOT p50 | `37.21 ms` | `32.72 ms` |
| Token trace mismatch count | - | `30/64` |

## Final Conclusion

Do not adopt a router logits math-mode change. The row is the largest measured local-compute bottleneck after the accepted `o_proj` optimization, but the fastest tested path changes generated tokens. Keep `CUBLAS_COMPUTE_32F_PEDANTIC` for router logits until a correctness-preserving small-GEMM path is available.

Next viable directions:

1. Compare FlashInfer/tinygemm-style router GEMM only if it can run with Kimi's required accuracy contract.
2. Investigate fusing `router_scores_kernel` and `router_topk_normalize_kernel`; the upper bound is much smaller than the logits GEMM, but it may reduce launches without touching GEMM math.
