# attention_o_proj Report

> **TL;DR:** `decode.attention.o_proj` is a memory-bound BF16 skinny GEMM on H20 (`AI ~= 7.98 flop/byte`, below the `30.83 flop/byte` ridge). Adopted a Kimi TP1 cuBLASLt exact-shape path for `W[7168,8192] x X[8192,batch_size]`, improving TP1 PPLX `bs=8,ctx=1` from `2.715ms` to `2.374ms` per 61 attention layers (`1.14x`).
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's `wiki/patterns/memory-bound.md` and `wiki/patterns/low-sm-utilization.md` match this row: low-batch decode GEMMs stream large weights, have arithmetic intensity below the H20 ridge point, and can underfill the GPU when the tile grid is smaller than the SM count. The useful direction is a strong library/small-GEMM baseline first, then custom work only if it beats that baseline in the full TP1 PPLX bench.

Directional small-GEMM references such as `sources/prs/sglang/PR-20755.md` justify checking tiny-GEMM style alternatives on SM90+, but they are not direct evidence for Kimi `M=8,N=7168,K=8192`. For this row, cuBLASLt is the strongest measured H20 baseline so far.

## NCU Conclusion

Profiled the adopted cuBLASLt provider through `kimi_tp1_pplx_decode_bench` with label `decode.attention.o_proj`.

| Item | Value |
|---|---:|
| Shape | `W[7168,8192] x X[8192,8] -> Y[7168,8]` |
| Dtype | BF16 input/output, FP32 accumulate |
| Calls per decode step | `61` attention layers |
| cuBLASLt kernel | `nvjet_tst_128x8_64x12_4x1_v_bz_TNT` |
| NCU duration | `32.16-33.09 us` |
| Grid / block | `56` CTAs x `384` threads |
| H20 SMs / waves | `78` SMs, `0.93` waves/SM |
| DRAM throughput | `73.87-75.92%` |
| Memory throughput | `3.63-3.73 TB/s` |
| Compute throughput | `21.20-22.40%` |
| Achieved occupancy | `14.11-14.13%` |
| L2 hit rate | `1.32-1.35%` |
| Scheduler no eligible | `81.79-82.54%` |

Conclusion: the row is memory-bound and limited by streamed weight traffic plus skinny-grid wave quantization. The cuBLASLt kernel is materially better than the prior graphsafe provider, but it is still not at H20 peak bandwidth and is far from compute-bound.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Existing TP1 path through `gemm_dm_hs_to_typed_graphsafe` | Master baseline row: `2.715 ms` for 61 calls, `44.5 us/call`, `21.11 TFLOP/s`, `55.1%` HBM roofline. | Baseline only. |
| Standalone cuBLAS vs cuBLASLt sweep | Same shape, 61 calls: cuBLAS `37.225 us/call` (`2.271 ms` step-equivalent), cuBLASLt `32.376 us/call` (`1.975 ms` step-equivalent). | cuBLASLt is the best local library baseline. |
| Production Kimi cuBLASLt path | TP1 PPLX bench `bs=8,ctx=1`: `2.374 ms` for 61 calls, `38.91 us/call`, `24.15 TFLOP/s`, `3.024 TB/s`, `63.0%` HBM roofline. | Adopted. |

## Final Conclusion

Adopt the Kimi-specific cuBLASLt path for `decode.attention.o_proj` when the shape is exactly TP1 Kimi K2 (`out=7168,in=8192`) and `batch_size=1..64`; keep the old typed GEMM as fallback for other TP shapes or larger batches.

Current accepted result:

| Workload | Before | After | Speedup | Bound |
|---|---:|---:|---:|---|
| H20 TP1/DP8/EP8 PPLX decode, per-rank `bs=8`, global `bs~=64`, `ctx=1` | `2.715 ms` | `2.374 ms` | `1.14x` | memory |

Stop condition for this row: cuBLASLt is the active baseline. Further work only makes sense if a custom small-M GEMM or fused adjacent operator path beats this cuBLASLt row in the full TP1 PPLX bench, not only in standalone GEMM timing.
