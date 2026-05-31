# q_b_proj Report

> **TL;DR:** `decode.attention.q_b` is a memory-bound TP1 skinny BF16 GEMM: `Y=[12288,batch_size] = W[12288,1536] @ X[1536,batch_size]`, FP32 accumulate, BF16 output. H20 NCU shows the current cuBLAS kernel is low-wave but already reaches `~59-61%` DRAM throughput. A zero-workspace cuBLASLt exact-shape sweep was rejected: at `batch_size=8` it improved only `8.899us -> 8.746us` per call (`1.0175x`), below the adoption threshold.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki pages used:

| Page | Conclusion | Application |
|---|---|---|
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Small grids and tail waves can dominate when the grid is smaller than the SM count; for non-persistent kernels, a grid much larger than SM count is preferred. | The current q_b cuBLAS kernel launches `64` CTAs on `78` H20 SMs, so wave quantization is real. |
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | For low arithmetic-intensity decode GEMM/GEMV-like kernels, first maximize memory bandwidth and use profile evidence before compute-side tuning. | q_b streams a large weight matrix for `batch_size=8`; NCU shows DRAM dominates compute. |
| `sources/prs/sglang/PR-20755.md` (`pr-sglang-20755`) | FlashInfer tinygemm is used for small SM90+ BF16 GEMM in a router-like path. | Directional only: it justifies checking small-GEMM alternatives, not adopting them without Kimi-shape measurement. |

Practical conclusion: q_b should be treated as a memory/low-wave skinny GEMM. Library baselines are the first filter; a custom fused/prologue path is only worth writing if it also removes the preceding `split_qkv_a_norm` launch or beats the current cuBLAS kernel by more than noise.

## NCU Conclusion

Existing H20 NCU run:

- Event artifact: `profile/kimi-attention-row8-row9-h20-baseline/row8_row9_event.json`
- NCU CSV: `profile/kimi-attention-row8-row9-h20-baseline/analysis/row9_details.csv`
- Label: `decode.attention.q_b`
- Workload: `active_rows=8`, `ctx=1`, TP1 PPLX bench path.

Key profile evidence:

| Metric | Value |
|---|---:|
| Kernel | `nvjet_tst_192x8_64x8_2x1_v_bz_TNT` |
| Duration | `12.99us` |
| Grid / block | `64` CTAs x `384` threads |
| Waves/SM | `0.82` |
| DRAM throughput | `60.70%`, `2.98 TB/s` |
| Compute throughput | `17.21%` |
| L2 hit rate | `2.74%` |
| Eligible warps / scheduler | `0.15` |
| No eligible | `85.70%` |

Diagnosis: q_b is not compute-bound on H20. It is a skinny GEMM that streams weights with poor cache reuse, limited grid size, and low eligible warps. The current cuBLAS kernel is already a strong memory-side baseline for this shape.

## Attempts

| Attempt | Evidence | Result | Decision |
|---|---|---|---|
| cuBLASLt exact-shape sweep | `profile/kimi-q-b-cublaslt-h20-baseline/analysis/q_b_gemm_bench.csv` | At `batch_size=8`, current cuBLAS measured `8.899us/call`; the best cuBLASLt heuristic measured `8.746us/call` (`1.0175x`). Several nearby batch sizes were also below the adoption threshold: `bs=3` `1.0258x`, `bs=4` `1.0171x`, `bs=5` `1.0356x`, `bs=6` `1.0439x`, `bs=7` `1.0386x`. | Rejected for production. |

Matrix layout verified by the harness:

- Logical math: `Y = W @ X`.
- `W` is stored row-major `[M,K] = [12288,1536]`.
- cuBLAS/cuBLASLt see `A=[K,M]`, `lda=K`, `op(A)=T`.
- `X=[K,batch_size]` and `Y=[M,batch_size]` are column-major with `ld=K` / `ld=M`.
- Accumulation and output match the existing generic path: BF16 inputs, `CUBLAS_COMPUTE_32F`, BF16 output, `beta=0`.

No production code was kept because the exact-shape library swap is below the `>3%` project threshold at the target `batch_size=8`.

## Final Conclusion

Stop condition for the standalone q_b cuBLASLt path: rejected. The measured cuBLASLt win at the target shape is inside noise and would add another special-case production wrapper without a meaningful decode-step improvement.

Future q_b work should be scoped as row 8/9 fusion, not standalone q_b GEMM replacement: the plausible win is deleting or absorbing `decode.attention.qkv_a_split_norm` while preserving `q_a_normed`, `ckv_normed`, and `k_rope` consumers, and keeping q_b at least as fast as the current cuBLAS kernel.
