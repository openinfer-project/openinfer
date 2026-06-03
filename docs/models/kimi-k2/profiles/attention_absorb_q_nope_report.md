# attention_absorb_q_nope Report

> **TL;DR:** `decode.attention.absorb_q_nope` is a memory-bound TP1 MLA strided-batched BF16 GEMM on H20: `64 x (512x128 @ 128xbatch_size)`, FP32 accumulate, BF16 output. Adopted a cuBLASLt strided-batched path for TP1 `local_heads=64,batch_size<=8`, improving `bs=8,ctx=1` from `973.6us` to `748.5us` per 61 attention layers (`1.30x`) with `0/64` token-trace mismatches in the cuBLAS fallback A/B smoke.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's `wiki/patterns/low-sm-utilization.md` matches this row: the useful first move for small decode GEMMs is to compare strong library baselines, because small grids and tail waves can dominate before custom kernels have enough work to fill the GPU. The page calls out grid-too-small and tail-effect causes; this shape has at most one wave on H20 after cuBLASLt.

The query also surfaced small-GEMM references such as `sources/prs/sglang/PR-20755.md`, but those are directional only. For this Kimi TP1 shape, cuBLASLt strided-batched GEMM is the measured baseline to beat.

## NCU Conclusion

Profiled the adopted path through `kimi_tp1_pplx_decode_bench` with label `decode.attention.absorb_q_nope`, `active_rows=8`, `ctx=1`, and one profiled cuBLASLt launch.

| Item | Value |
|---|---:|
| Shape | `64 x (512x128 @ 128x8)` |
| Dtype | BF16 input/output, FP32 accumulate |
| Calls per decode step | `61` attention layers |
| cuBLASLt kernel | `nvjet_tst_256x8_64x6_2x1_v_bz_NNT` |
| NCU duration | `6.112 us` |
| Grid / block | `78` CTAs x `384` threads |
| Cluster / waves | cluster size `2`, `1.00` waves/SM |
| DRAM read | `1.397 TB/s`, `28.44%` peak |
| SM throughput | `9.99%` peak |
| GMMA instruction active | `11.80%` peak active |
| Tensor pipe elapsed | `1.90%` peak |
| Active warps | `14.87%` |
| Eligible warps | `0.181` per cycle |
| L2 hit / read hit | `9.83%` / `0.91%` |

Conclusion: after cuBLASLt, this row is still memory/low-wave limited rather than compute-bound. The grid barely reaches one H20 wave, the weight stream has almost no L2 reuse, and tensor utilization is low.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Baseline cuBLAS strided-batched GEMM | TP1 PPLX `bs=8,ctx=1`: `973.6us` per 61 calls, `15.96us/call`. | Baseline. |
| Standalone exact-shape microbench | `bs=8`: cuBLAS `8.539us/call`, cuBLASLt `5.810us/call` for the same `64 x (512x128 @ 128x8)` semantics. Batch sweep showed the cuBLASLt win is concentrated at `batch_size<=8`; larger batches are not blindly switched. | Use cuBLASLt only for target small batches. |
| cuBLASLt candidate correctness harness | All tested `CUBLAS_COMPUTE_32F` cuBLASLt candidates matched cuBLAS BF16 output bitwise for the exact bs8 harness (`0/262144` mismatches). | No standalone numerical mismatch. |
| Production cuBLASLt path | Filtered 128-iter TP1 PPLX bench after moving direct providers to CUDA-side lazy init: `748.5us` per 61 calls, `12.27us/call`, `5.47 TFLOP/s`, `727 GB/s`, `15.1%` HBM peak. | Adopted. |
| Serving token A/B | Same code with only MLA cuBLASLt init disabled vs enabled: TP1 DP8 bs64/o5 generated token traces had `0/64` mismatches. | Correctness gate passed. |

Artifacts:

- `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json`
- `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-mla-cublaslt-bs8.json`
- `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-mla-cublaslt-bs8-lazy128.json`
- `profile/kimi-mla-absorb-vup-h20-baseline/analysis/mla_batched_gemm_bench.csv`
- `profile/kimi-mla-cublaslt-h20/analysis/metrics_key_summary.txt`
- `profile/kimi-mla-cublaslt-correctness-h20/analysis/mla_batched_gemm_correctness.csv`

## Final Conclusion

Adopt cuBLASLt for `decode.attention.absorb_q_nope` only when the runtime shape is the TP1 decode target: `local_heads=64` and `batch_size<=8`. Other head counts or larger batches keep the original cuBLAS strided-batched path.

Current accepted result:

| Workload | Before | After | Speedup | Bound |
|---|---:|---:|---:|---|
| H20 TP1/DP8/EP8 PPLX decode, per-rank `bs=8`, global `bs~=64`, `ctx=1` | `973.6 us` | `748.5 us` | `1.30x` | memory |

Stop condition for this row: cuBLASLt is the current target-shape baseline. Future work should be a custom grouped/persistent GEMM only if it beats this library path in the full TP1 PPLX bench while preserving the token A/B gate.
