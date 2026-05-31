# shared_gate_up Report

> **TL;DR:** `decode.moe.shared_gate_up` is a memory-bound BF16 skinny GEMM on H20 (`AI ~= 7.98 flop/byte`, below the `30.83 flop/byte` ridge). Adopted Kimi-specific cuBLASLt for exact shape `M=4096,K=7168,batch_size=1..64`, improving TP1 PPLX `bs=8,ctx=1` from `1.818ms` to `1.505ms` per 60 MoE layers (`1.21x`).
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's relevant Hopper guidance matches this shape: small-batch decode GEMMs with large streamed weights are usually memory-bound, and low-SM-utilization can hide inside library GEMM choices when the grid is smaller than the GPU. For SM90/H20, the useful leads were FlashInfer `tinygemm`/`tinygemm2` and a strong cuBLASLt baseline; SM100-only ideas such as CLC are not an H20 fix.

The practical rule for this row is: beat the library baseline with measured H20 data, or keep the library path. Custom/CuTe work is only worth keeping if it improves the `bs=8/rank, global~=64` TP1 PPLX bench beyond noise.

## NCU Conclusion

Profiled generic cuBLAS on the same shape as production shared expert gate/up:

| Item | Value |
|---|---:|
| Shape | `W[4096,7168] x X[7168,batch_size] -> Y[4096,batch_size]` |
| Dtype | BF16 input/output, FP32 accumulate |
| Calls per decode step | `60` MoE layers |
| Main cuBLAS kernel | `nvjet_tst_128x8_64x12_2x1_v_bz_splitK_TNT` |
| Main duration | `19.360 us` |
| Split-K reduce duration | `3.424 us` |
| Main grid | `64` blocks on `78` SMs |
| Main DRAM read BW | `3.04 TB/s`, `61.97%` read peak |
| Main SM throughput | `18.42%` |
| Tensor pipe activity | `4.61%` |

Conclusion: the row is not compute-bound and is not at the H20 memory limit. It streams the BF16 weight matrix, has low L2 reuse, leaves SMs idle because the launch geometry is small, and pays split-K reduce overhead.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Existing TP1 path through `gemm_dm_typed_to_hs_graphsafe` | Master baseline row: `1.818 ms` for 60 calls, `15.50 TF/s`, `40.5%` of the H20 memory roofline used by the report. | Baseline only. |
| Standalone same-shape cuBLAS harness | `22.007 us` per call, `1.320 ms` for 60 calls, `2.676 TB/s`, `55.76%` HBM peak. | Useful lower-level reference, but still leaves bandwidth and split-K overhead on the table. |
| FlashInfer `tinygemm2` internal C++ smoke | Roughly `30.6 us` at `N=8`, slower than cuBLAS for this shape on H20. | Rejected for now; no stable public C++ interface in the repo submodule and measured slower than cuBLAS. |
| Standalone cuBLASLt first heuristic, zero workspace | `18.673 us` per call, `1.120 ms` for 60 calls, `3.153 TB/s`, `65.69%` HBM peak. | Best local baseline. |
| Production Kimi cuBLASLt path | TP1 PPLX bench `bs=8,ctx=1`: `1.505 ms` for 60 calls, `18.72 TF/s`, `2.348 TB/s`, `48.9%` HBM peak. | Adopted. |
| Non-power-of-two batch check | TP1 PPLX bench `bs=3,ctx=1`: `1.524 ms`, provider `kimi_shared_gate_up_cublaslt`. | Confirms `batch_size=1..64` support; no power-of-two fallback bug. |
| CUTLASS example 45 dual-GEMM + fused SwiGLU prototype | `profile/kimi-shared-gated-dual-gemm-h20-prototype/`: default `128x64x32` tile measured `207.5us/call`; tuned `16x64x64` measured `69.2us/call`; tuned `16x32x64` measured `68.7us/call`. Current same-harness cuBLASLt + SwiGLU baseline is `21.1-21.3us/call`. NCU for `16x32x64` shows `73.22us`, `64` CTAs, `0.12` waves/SM, `17.06%` DRAM, `14.69%` compute, and `77.12%` no eligible. | Rejected. The stock CUTLASS dual-GEMM path is not shaped for Kimi decode `M=8`; it loses badly to the adopted cuBLASLt baseline despite eliminating the activation launch. |

## Final Conclusion

Adopt the Kimi-specific cuBLASLt path for `shared_gate_up` when the shape is exactly `M=4096,K=7168,batch_size=1..64`; keep the old typed GEMM as fallback for other shapes. The implementation prebuilds one zero-workspace cuBLASLt plan per supported batch size on the rank thread, uses `batch_size` naming throughout, and destroys the thread-local cuBLASLt handle with the existing cuBLAS guard.

Current accepted result:

| Workload | Before | After | Speedup | Bound |
|---|---:|---:|---:|---|
| H20 TP1/DP8/EP8 PPLX decode, per-rank `bs=8`, global `bs~=64`, `ctx=1` | `1.818 ms` | `1.505 ms` | `1.21x` | memory |

Stop condition for this row: do not continue tuning standalone GEMM unless the next attempt beats cuBLASLt in the full TP1 PPLX bench. The stock CUTLASS dual-GEMM/SwiGLU route is also stopped for now: at the Kimi decode shape it is `~3.2x` slower than cuBLASLt + current SwiGLU in the standalone harness. A future row 21/22 fusion would need a decode-specific small-M schedule, not the example 45 schedule.
