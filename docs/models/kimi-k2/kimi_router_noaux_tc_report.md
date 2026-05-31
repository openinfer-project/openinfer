# kimi_router_noaux_tc Report

> **TL;DR:** `decode.moe.router` is control/launch limited on H20, not memory-bandwidth-bound. Baseline TP1 PPLX `bs=8,ctx=1` was `60.91us/call` (`3.655ms` per 60 MoE layers). The adopted post-GEMM fusion keeps the pedantic BF16->F32 logits GEMM and fuses sigmoid+bias+topk+normalize into one CUDA selector, improving the row to `58.57us/call` (`3.514ms`, `1.04x`) with TP1 DP8 bs64/o5 token A/B `0/64` mismatches. The faster tensor-op logits GEMM remains rejected because it changed generated token traces (`30/64` mismatches).
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

Baseline sub-kernel NCU:

| Kernel | Duration | Grid / block | Waves/SM | DRAM | Compute | No eligible |
|---|---:|---:|---:|---:|---:|---:|
| `gemmSN_TN_kernel` logits GEMM | `81.7-82.7 us` | `48` x `128` | `0.12` | `1.39-1.40%` | `8.85-8.97%` | `89.08-89.12%` |
| `router_scores_kernel` sigmoid+bias | `2.94-3.01 us` | `12` x `256` | `0.02` | `0.13%` | `0.42-0.45%` | `95.08-95.83%` |
| `router_topk_normalize_kernel` | `10.08-10.24 us` | `8` x `512` | `0.03` | `0.04%` | `1.96-1.99%` | `76.29-76.45%` |

NCU full-set replay inflates absolute kernel durations versus CUDA-event timing, but the relative diagnosis is stable: the row is small-grid/control limited. The logits GEMM dominates and uses pedantic BF16->F32 math, which avoids the fast tensor-op path.

Accepted fused selector NCU:

| Kernel | Duration | Grid / block | Waves/SM | DRAM | Memory throughput | Compute | No eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| `router_scores_topk_normalize_kernel` | `10.56 us` | `8` x `512` | `0.03` | `0.05%` | `2.25 GB/s` | `1.96%` | `76.77%` |

The fused selector keeps the same one-block-per-token shape, writes compatible `scores` / `choice_scores` scratch for padded rows, and only removes the standalone score launch plus global selector reloads. It remains control/low-wave limited; the accepted win comes from deleting a small launch and avoiding extra global traffic, not from approaching an H20 roofline.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Baseline router | TP1 PPLX `bs=8,ctx=1`: `60.910 us/call`, `3.655 ms` per step. | Baseline. |
| Post-GEMM score/topk fusion | TP1 PPLX `bs=8,ctx=1`: `58.566 us/call`, `3.514 ms` per step (`1.04x`). TP1 DP8 bs64/o5 token A/B vs argmax-split baseline: `0/64` mismatches; steady TPOT avg `36.095ms -> 35.806ms`. | Accepted; this keeps `CUBLAS_COMPUTE_32F_PEDANTIC` logits GEMM and preserves router top-k semantics. |
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

Adopt the post-GEMM fused selector and keep `CUBLAS_COMPUTE_32F_PEDANTIC` for router logits. The row remains the largest measured local-compute bottleneck, but the fastest tested logits GEMM path changes generated tokens and stays rejected.

Next viable directions:

1. Compare FlashInfer/tinygemm-style router GEMM only if it can run with Kimi's required accuracy contract.
2. Further selector work should be lower priority than the remaining large GEMM rows; the fused selector is still only `~10us/call` under NCU replay.
