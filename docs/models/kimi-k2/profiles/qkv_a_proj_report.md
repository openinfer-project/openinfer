# qkv_a_proj Report

> **TL;DR:** `decode.attention.qkv_a` is a memory-bound BF16 skinny GEMM on H20: `W[2112,7168] x X[7168,batch_size] -> Y[2112,batch_size]`, FP32 accumulate, BF16 output. TP1 PPLX `bs=8,ctx=1` baseline is `20.407us/call`, `1.245ms` per 61 attention layers, `1.491 TB/s`, or `31.1%` HBM roofline. Standalone cuBLASLt improved a contiguous loop (`15.119us -> 14.052-14.179us`) but the temporary TP1 bench provider only measured `20.407us -> 20.242us` at 256 iters (`0.8%`), so standalone cuBLASLt is rejected. Keep row 6/7 in the fusion queue only for a true RMSNorm-prologue/custom GEMM path.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's memory-bound and low-SM-utilization patterns both apply. The row's arithmetic intensity is `7.96 flop/byte`, below the H20 report ridge (`30.83 flop/byte`), and the current cuBLAS main kernel launches only `72` CTAs for `78` SMs. This is a small-batch streamed-weight GEMM, not a tensor-core-throughput problem.

The adjacent `decode.attention.input_norm` row is also tiny-grid limited (`8` CTAs, `0.05` waves/SM). That makes row 6/7 a plausible fusion target, but only if the fused/custom GEMM keeps qkv_a at least as fast as cuBLAS while removing the RMSNorm launch and normalized-hidden write/read. Tuning RMSNorm alone or swapping qkv_a to a library path with sub-noise gains is not worth keeping.

## NCU Conclusion

Profiled row 6/7 on H20 because qkv_a consumes the normalized hidden state directly:

| Item | `decode.attention.input_norm` | `decode.attention.qkv_a` |
|---|---:|---:|
| Shape | `hidden=7168,batch=8` | `W[2112,7168] x X[7168,8] -> Y[2112,8]` |
| Calls per decode step | `61` | `61` |
| Bench mean | `8.008us/call` | `20.407us/call` |
| Step latency | `488.5us` | `1.245ms` |
| Bench throughput | `57.3 GB/s` | `11.87 TF/s`, `1.491 TB/s` |
| Bench HBM roofline | `1.19%` | `31.1%` |
| Main kernel | `RMSNormKernel<8,bf16>` | `nvjet_tst_128x8_64x12_2x1_v_bz_splitK_TNT` |
| Main duration | `3.97-4.22us` | `11.84-12.22us` |
| Grid / block | `8 x 896` | `72 x 384` |
| Waves/SM | `0.05` | `0.92` |
| DRAM throughput | `0.70-0.74%` | `51-53%` |
| SM throughput | `2.4-2.5%` | `15-16%` |
| No eligible | `60-61%` | `84-85%` |

The qkv_a cuBLAS path also launches a `cublasLt::splitKreduce_kernel<...>` taking `3.04-3.14us` with `66 x 512` threads, `0.21` waves/SM, and only `1.8-1.9%` DRAM. The pair is limited by launch count, intermediate memory traffic, low waves, and split-K reduce overhead. It is not at the H20 roofline.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current TP1 PPLX provider `gemm_graphsafe` | Filtered H20 bench at `bs=8,ctx=1`: `20.407us/call`, `1.245ms` for 61 calls, `11.87 TF/s`, `1.491 TB/s`, `31.1%` HBM roofline. | Current baseline. |
| Standalone cuBLASLt exact-shape loop | cuBLAS `15.119us`; cuBLASLt tuned `14.179us`; cuBLASLt first heuristic `14.052us`. | Useful library probe, but not enough by itself. |
| Temporary TP1 bench provider `kimi_qkv_a_cublaslt` | 64 iters: `20.407us -> 20.070us` (`1.7%`); 256 iters: `20.407us -> 20.242us` (`0.8%`). | Rejected; below the `>3%` adoption threshold. No production code kept. |
| Standalone RMSNorm rewrite | Not attempted. NCU shows row 6 is tiny-grid/launch-latency limited, and changing it alone cannot address the qkv_a GEMM or intermediate traffic. | Rejected as a direction; keep only the fused row 6/7 path. |

## Final Conclusion

Stop standalone qkv_a provider replacement. The current qkv_a GEMM is memory-bound and underfills H20, but exact-shape cuBLASLt does not produce a reproducible full-bench win at the target `batch_size=8`.

The only remaining plausible row 6/7 path is a correctness-preserving RMSNorm-prologue/custom GEMM that removes the standalone `input_norm` launch and intermediate normalized hidden buffer while preserving qkv_a speed. Adoption requires a full TP1 PPLX `bs=8/rank, global~=64` bench win and unchanged token/correctness gates.
