# attention_v_up Report

> **TL;DR:** `decode.attention.v_up` is the second TP1 MLA strided-batched BF16 GEMM: `64 x (128x512 @ 512xbatch_size)`, FP32 accumulate, BF16 output. The same cuBLASLt strided-batched path improves `bs=8,ctx=1` from `781.0us` to `738.5us` per 61 attention layers (`1.06x`) with `0/64` token-trace mismatches in the cuBLAS fallback A/B smoke.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's `wiki/patterns/low-sm-utilization.md` applies here too: this is a small-grid decode GEMM, so library baselines and batch-specific shape checks should come before custom kernels. The pattern page's grid-too-small and tail-effect diagnosis matches the NCU result: the adopted cuBLASLt kernel launches only `64` CTAs on a `78`-SM H20.

Small-GEMM references such as `sources/prs/sglang/PR-20755.md` are useful leads, but not direct evidence for this Kimi MLA shape. The accepted path is based on H20 measurement, not on upstream intuition.

## NCU Conclusion

Profiled the adopted path through `kimi_tp1_pplx_decode_bench` with label `decode.attention.v_up`, `active_rows=8`, `ctx=1`, and one profiled cuBLASLt launch.

| Item | Value |
|---|---:|
| Shape | `64 x (128x512 @ 512x8)` |
| Dtype | BF16 input/output, FP32 accumulate |
| Calls per decode step | `61` attention layers |
| cuBLASLt kernel | `nvjet_tst_128x8_64x12_1x1_v_bz_TNT` |
| NCU duration | `6.048 us` |
| Grid / block | `64` CTAs x `384` threads |
| Waves | `0.82` waves/SM |
| DRAM read | `1.476 TB/s`, `30.26%` peak |
| SM throughput | `8.67%` peak |
| GMMA instruction active | `14.39%` peak active |
| Tensor pipe elapsed | `2.01%` peak |
| Active warps | `14.69%` |
| Eligible warps | `0.177` per cycle |
| L2 hit / read hit | `5.37%` / `0.78%` |

Conclusion: this row remains memory/low-wave limited. It is not close to H20 compute peak, and the CTAs do not fill a full wave across the GPU.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Baseline cuBLAS strided-batched GEMM | TP1 PPLX `bs=8,ctx=1`: `781.0us` per 61 calls, `12.80us/call`. | Baseline. |
| Standalone exact-shape microbench | `bs=8`: cuBLAS `6.501us/call`, cuBLASLt `5.712us/call` for the same `64 x (128x512 @ 512x8)` semantics. | Worth trying in production. |
| cuBLASLt candidate correctness harness | All tested `CUBLAS_COMPUTE_32F` cuBLASLt candidates matched cuBLAS BF16 output bitwise for the exact bs8 harness (`0/65536` mismatches). | No standalone numerical mismatch. |
| Production cuBLASLt path | Filtered 128-iter TP1 PPLX bench after moving direct providers to CUDA-side lazy init: `738.5us` per 61 calls, `12.11us/call`, `5.54 TFLOP/s`, `704 GB/s`, `14.7%` HBM peak. | Adopted. |
| Serving token A/B | Same code with only MLA cuBLASLt init disabled vs enabled: TP1 DP8 bs64/o5 generated token traces had `0/64` mismatches. | Correctness gate passed. |

Artifacts:

- `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json`
- `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-mla-cublaslt-bs8.json`
- `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-mla-cublaslt-bs8-lazy128.json`
- `profile/kimi-mla-absorb-vup-h20-baseline/analysis/mla_batched_gemm_bench.csv`
- `profile/kimi-mla-cublaslt-h20/analysis/metrics_key_summary.txt`
- `profile/kimi-mla-cublaslt-correctness-h20/analysis/mla_batched_gemm_correctness.csv`

## Final Conclusion

Adopt cuBLASLt for `decode.attention.v_up` only for the target TP1 decode shape: `local_heads=64` and `batch_size<=8`. Other head counts or larger batches keep the original cuBLAS strided-batched path.

Current accepted result:

| Workload | Before | After | Speedup | Bound |
|---|---:|---:|---:|---|
| H20 TP1/DP8/EP8 PPLX decode, per-rank `bs=8`, global `bs~=64`, `ctx=1` | `781.0 us` | `738.5 us` | `1.06x` | memory |

Stop condition for this row: the accepted gain is small but repeatable above the noise threshold in the 128-iter filtered bench. Further work should target a stronger custom/grouped path only after it beats this cuBLASLt row and keeps the token A/B gate clean.
