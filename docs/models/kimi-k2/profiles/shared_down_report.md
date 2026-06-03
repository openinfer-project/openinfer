# shared_down Report

> **TL;DR:** `decode.moe.shared_down` is a memory-bound BF16 skinny GEMM on H20: `W[7168,2048] x X[2048,batch_size] -> Y[7168,batch_size]`, FP32 accumulate, BF16 output. TP1 PPLX `bs=8,ctx=1` currently measures `14.952us/call`, `897.1us` per 60 MoE layers, `1.974 TB/s`, or `41.1%` of the H20 HBM roofline. The exact-shape cuBLASLt standalone sweep was rejected (`11.000us -> 10.995us`, `~1.0005x`), so keep the generic cuBLAS path unless a true `shared_swiglu -> shared_down` fusion beats it in the full TP1 PPLX bench.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's memory-bound and low-SM-utilization guidance matches this row. The arithmetic intensity is `7.96 flop/byte`, below the H20 roofline ridge used by the TP1 report (`30.83 flop/byte`), so the useful question is weight bandwidth and grid occupancy, not raw tensor-core FLOPS.

For this decode shape, `batch_size=8` gives a small GEMM launch with large streamed weights. KernelWiki's relevant rule is to compare against a strong library baseline first, then only keep custom/fused work if it beats that baseline with measured H20 data. The standalone cuBLASLt provider does not do that here. A future attempt should target a real activation prologue fusion from row 22 into row 23, not another isolated GEMM provider swap.

## NCU Conclusion

Profiled current production provider for `decode.moe.shared_down` on H20:

| Item | Value |
|---|---:|
| Shape | `W[7168,2048] x X[2048,batch_size=8] -> Y[7168,batch_size=8]` |
| Dtype | BF16 input/output, FP32 accumulate |
| Calls per decode step | `60` MoE layers |
| Bench mean | `14.9519 us/call`, `897.112 us/step` |
| Bench throughput | `15.709 TF/s`, `1.974 TB/s` |
| Bench HBM roofline | `41.115%` |
| Main cuBLAS kernel | `nvjet_tst_128x8_64x12_4x1_v_bz_TNT` |
| NCU main duration | `10.78 us` |
| NCU grid | `56` blocks |
| NCU block | `384` threads |
| NCU waves/SM | `0.93` |
| NCU memory throughput | `2.73 TB/s` |
| NCU DRAM throughput | `55.94%` |
| NCU SM throughput | `15.74%` |
| NCU achieved occupancy | `14.25%` |
| NCU L2 hit rate | `2.37%` |
| NCU no eligible | `82.37%` |
| NCU eligible warps/scheduler | `0.19` |

Conclusion: the main kernel is memory-bound and small-grid limited. It streams the down-projection weights with very low L2 reuse, does not fill the H20 (`56` CTAs on `78` SMs, `0.93` waves/SM), and spends most scheduler cycles with no eligible warp. This is not an isolated compute-bound kernel.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current TP1 PPLX provider `gemm_dm_hs_to_typed_graphsafe` | Filtered H20 bench at `bs=8,ctx=1`: `14.9519us/call`, `897.112us` for 60 calls, `1.974 TB/s`, `41.115%` HBM roofline. | Current baseline. |
| Standalone cuBLASLt exact-shape sweep | `11.000260us -> 10.995026us`, `0.05%` faster in `profile/kimi-shared-gated-dual-gemm-h20-prototype/analysis/gemm_cublaslt_sweep.csv`. | Rejected; far below the noise threshold and below the `>3%` commit bar. |
| Row 22/23 fusion via stock dual-GEMM direction | Not implemented for down projection. Standard cuBLASLt does not provide an arbitrary `SwiGLU` prologue into this GEMM, and the stock CUTLASS dual-GEMM path already lost badly on row 21/22 for Kimi decode `M=8`. | Future work only if a decode-specific fused kernel can beat the current full TP1 PPLX bench. |

## Final Conclusion

Stop standalone `shared_down` provider replacement for now. The kernel is clearly memory-bound, but the exact-shape cuBLASLt swap is a no-op at the target shape, and the current cuBLAS kernel already reaches a stronger lower-level NCU bandwidth number than the end-to-end row percentage suggests.

The only remaining plausible path for this row is a correctness-preserving `shared_swiglu -> shared_down` fusion that avoids materializing the activation output and wins in the full TP1 PPLX `bs=8/rank, global~=64` decode bench. Until then, keep the current cuBLAS provider.
