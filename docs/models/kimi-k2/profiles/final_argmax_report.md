# final_argmax Report

> **TL;DR:** `decode.final.argmax` was a low-SM-utilization BF16 top1 scan over `batch_size=8, vocab=163840`: the old one-CTA-per-row kernel measured `125.3us` in TP1 PPLX bench (`20.9 GB/s`, `0.4%` of H20 HBM peak). The accepted split-vocab path uses `4096`-element tiles, partial `(value,index)` scratch, and a finalize reduction; TP1 PPLX `bs=8,ctx=1` now measures `12.724us` (`206.0 GB/s`, `4.3%` of HBM peak), with TP1 DP8 bs64/o5 token A/B `0/64` mismatches.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki pages used:

| Page | Conclusion | Application |
|---|---|---|
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Small grids and partial waves can dominate when total CTAs are below or near SM count. | The old batched argmax launched only `rows=8` CTAs, one CTA per output row, so H20 had almost no parallelism for a full-vocab scan. |
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | Decode-side low-AI scans/GEMV-like work should be judged by achieved bandwidth and profile counters, not compute peak. | Argmax reads BF16 logits and does a reduction; the optimization target is more memory parallelism and less underfilled scheduling, not tensor FLOPS. |

Practical conclusion: for `batch_size=8`, split the vocab dimension across CTAs, keep tie-breaking deterministic by lower index, and use explicit scratch instead of launching only one block per row. FlashInfer top1 was not adopted in this pass because the split CUDA path keeps the existing argmax ABI/semantics and already had direct standalone plus TP1 DP8 token A/B proof.

## NCU Conclusion

NCU artifact:

- Report: `profile/kimi-final-argmax-h20-baseline/reports/argmax_split_full.ncu-rep`
- CSV: `profile/kimi-final-argmax-h20-baseline/analysis/argmax_split_details.csv`
- Workload: `kimi_tp1_pplx_decode_bench --active-rows 8 --ctx-lens 1 --labels decode.final.argmax`

Key counters:

| Kernel | Duration | Grid | Waves/SM | Memory throughput | DRAM throughput | SM throughput | Occupancy | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `argmax_batch_bf16_partial_kernel` | `10.11us` | `320` CTAs | `0.51` | `259.65 GB/s` | `5.31%` | `16.41%` | `49.51%` | Main scan; still low-wave and memory-latency limited. |
| `argmax_batch_bf16_finalize_kernel` | `3.17us` | `8` CTAs | `0.01` | `2.02 GB/s` | `0.04%` | `0.45%` | `12.10%` | Tiny finalize over `40` partials per row. |

Diagnosis: the new path is much faster because it raises grid size from `8` CTAs to `320 + 8` CTAs and exposes the vocab scan parallelism. It is still not near H20 HBM peak because the reduction has low waves, little reuse, and a small finalize launch; after this change it is no longer a top decode-step bottleneck.

## Attempts

| Attempt | Evidence | Result | Decision |
|---|---|---|---|
| Baseline one-CTA-per-row kernel | Master Phase 1 row 19 | TP1 PPLX `bs=8,ctx=1`: `125.3us`, `20.9 GB/s`, `0.4%` HBM peak. | Replaced for Kimi decode top1. |
| Standalone split-vocab sweep | `profile/kimi-final-argmax-h20-baseline/analysis/argmax_bf16_bench.csv` | Old kernel `116.816us`; split tile `1024` `9.590us`, tile `2048` `9.130us`, tile `4096` `8.580us`. | Chose tile `4096`. |
| Production TP1 PPLX bench | `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-argmax-split-bs8.json` | `12.724us`, `206.0 GB/s`, `4.29%` HBM peak for `bs=8,ctx=1`. | Accepted. |
| TP1 DP8 token A/B | Baseline `/tmp/kimi-tp1dp8-mla-lt/prompt1_bs64_o5.json` vs new `/tmp/kimi-tp1dp8-argmax-split/prompt1_bs64_o5.json` | `0/64` generated token trace mismatches. Steady TPOT avg moved `36.359ms -> 36.095ms`. | Correctness gate passed. |

## Final Conclusion

Adopt `argmax_batch_bf16_split_cuda` for Kimi K2 local top1. The implementation adds per-batch partial `(value,index)` scratch and keeps the old lower-index tie-break semantics. The public generic `argmax_batch_bf16_into` remains available for callers that do not provide scratch.

Stop condition for this pass: accepted and committed as a decode-path win. Further tuning should wait until larger rows above it are addressed; the remaining `12.7us` is mostly low-wave reduction and launch overhead, not a dominant TP1 DP8 bottleneck.
