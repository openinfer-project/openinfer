# pplx_marlin_compute Report

> **TL;DR:** PPLX routed local compute is now measurable in `kimi_tp1_pplx_decode_bench` without timing EP communication. For TP1 PPLX `bs=8,ctx=1`, the synthetic expected-local-route provider uses `recv_capacity=848`, `64` expected local routes per EP rank, and `400` expected padded work rows. H20 event timing is `pplx_build_marlin_routing=9.49us/call`, `pplx_marlin_w13=436.43us/call`, `pplx_swiglu=14.13us/call`, and `pplx_marlin_w2=236.80us/call`. W13/W2 sit near the H20 ridge (`AI ~= 29 flop/byte`) and are memory-bound by the bench roofline, while NCU still shows substantial SM pipe use (`56.8-58.7%`). This remains a synthetic local-compute baseline, not an all-rank route-imbalance claim.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's `wiki/kernels/fused-moe.md` and `wiki/kernels/grouped-gemm.md` pages match the direction: MoE local compute should be treated as grouped/masked expert GEMM with variable per-expert M, where launch count, padding, and load imbalance are first-class performance variables. The relevant caution is that decode uses small per-expert M; padding and masked layouts can waste compute, while grouped scheduling helps only when the route distribution is represented honestly.

For this report, the bench provider deliberately excludes EP dispatch/combine transport and only times the local PPLX compute kernels after synthetic recv counts have been materialized. That makes the rows measurable for NCU and master-table accounting, but the final optimization target still needs an all-rank histogram to confirm actual per-expert counts.

## NCU Conclusion

Filtered H20 bench:

| Row | Provider | Mean/call | Step latency | Roofline read |
|---|---|---:|---:|---|
| `decode.moe.pplx_build_marlin_routing` | `kimi_pplx_build_marlin_routing_on_stream` | `9.489us` | `569.3us` | control |
| `decode.moe.pplx_marlin_w13` | `kimi_marlin_wna16_pplx_w13_gemm` | `436.432us` | `26.186ms` | memory, `53.82 TF/s`, `1.837 TB/s`, `38.3%` HBM |
| `decode.moe.pplx_swiglu` | `kimi_marlin_w13_swiglu_pplx` | `14.135us` | `848.1us` | memory, `347.7 GB/s`, `7.2%` HBM |
| `decode.moe.pplx_marlin_w2` | `kimi_marlin_wna16_pplx_w2_gemm` | `236.797us` | `14.208ms` | memory, `49.60 TF/s`, `1.705 TB/s`, `35.5%` HBM |

NCU summary:

| Kernel | Duration | Grid / block | Waves/SM | DRAM | Memory throughput | SM throughput | Occupancy | L2 hit | No eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `kimi_pplx_build_marlin_routing_kernel` | `5.28-5.31us` | `1 x 64` | `0.00` | `0.04%` | `1.7 GB/s` | `0.07-0.08%` | `2.3%` | `72.6-82.8%` | `87-88%` |
| `Marlin<...>` W13 | `467.10us` | `234 x 128` | `1.00` | `34.73%` | `1.71 TB/s` | `58.72%` | `17.51%` | `4.88%` | `40.64%` |
| `swiglu_w13_pplx_kernel` | `10.62us` | `6784 x 256` | `10.87` | `6.32%` | `309 GB/s` | `55.40%` | `76.05%` | `38.25%` | `34.20%` |
| `Marlin<...>` W2 | `250.05us` | `234 x 128` | `1.00` | `32.55%` | `1.60 TB/s` | `56.80%` | `17.63%` | `5.29%` | `42.23%` |

Conclusion: W13/W2 are the dominant PPLX local-compute rows in this synthetic baseline. They are not HBM-saturated; they run with a single wave per SM, low L2 hit rate, and moderate SM throughput. Routing is launch/control dominated. PPLX SwiGLU is not the main cost after the current device-side row limiting.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Add TP1 PPLX local-compute providers to `kimi_tp1_pplx_decode_bench` | Rows 24-27 now measure through existing Rust/CUDA wrappers with synthetic recv counts: `64` expected local routes, `400` expected padded rows, and `recv_capacity=848` at `bs=8`. | Kept as baseline coverage. This is bench/report infrastructure, not an optimization. |
| Worst-case synthetic counts using global `512` routes on one EP rank | W13 `905.95us/call`, W2 `485.03us/call`; this filled the local rank to capacity and contradicted the target global `bs~=64` expected load. | Rejected as the default provider shape. It remains a useful stress case, but not the anchor workload. |
| Expected-local-route synthetic counts | W13 `436.43us/call`, W2 `236.80us/call`; NCU captures the Marlin kernels cleanly. | Adopted for the bench baseline until an all-rank route histogram replaces it. |

## Final Conclusion

The immediate gap for PPLX routed local compute was measurement coverage, not a code optimization. The master table should no longer treat W13/SwiGLU/W2/routing as invisible estimate-only rows.

Do not commit an `opt(...)` change from this report: no faster kernel was adopted. The next optimization step should use an all-rank harness or route histogram to replace synthetic counts, then target W13/W2 if they remain top bottlenecks under the real distribution.
