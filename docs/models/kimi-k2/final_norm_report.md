# Final Norm Report

> **TL;DR:** `decode.final.norm` is the final FlashInfer RMSNorm over the fixed TP1 decode arena (`rows=8, hidden=7168`, BF16) before the full-vocab LM head. H20 timing from `tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` is `8.01us/call`, `57.3GB/s`, and only `~1.2%` H20 HBM by the bench payload model. The same provider and shape were profiled as `decode.attention.input_norm`: NCU shows `8` CTAs, `0.05` waves/SM, `0.70-0.74%` DRAM, and `60-61%` scheduler no eligible. Stop standalone final RMSNorm tuning.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Low SM utilization can come from tail effect, static scheduling, or a grid smaller than the SM count. | The exact-shape FlashInfer RMSNorm NCU launches only `8` CTAs on H20, so final norm is a tiny-grid/control row, not a bandwidth row. |
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | Memory-bound diagnosis requires measured high DRAM throughput, not only low arithmetic intensity. | The row's payload model reaches only `~57GB/s` and NCU on the same kernel/shape reports `<1%` DRAM, so standalone bandwidth work is not the right target. |
| `sources/prs/flashinfer/PR-3014.md` (`pr-flashinfer-3014`) | Small-batch decode helper overhead is best handled by removing unnecessary helper work or launch overhead. | Directional only: there is no empty work here, so useful improvement would have to come from fusing final norm into the LM-head prologue, not from retuning RMSNorm alone. |

Practical conclusion: the final norm is a launch-sized helper before a much larger LM head. Since `decode.final.lm_head` is already near the H20 HBM roofline in the current BF16 path, fusing RMSNorm into that GEMM is only worth revisiting with a library/custom GEMM plan that preserves the LM-head bandwidth.

## NCU Conclusion

Exact `decode.final.norm` NCU has not been collected as a separate label. However, `decode.attention.input_norm` uses the same `rms_norm_batch` provider, the same `hidden=7168,batch=8` shape, and the same FlashInfer `RMSNormKernel<8,bf16>` launch. The row-6 H20 NCU is therefore valid shape evidence for the final norm kernel body.

Existing same-shape H20 NCU:

```bash
/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -c 10 -o profile/kimi-attention-row6-row7-h20-baseline/reports/discover_row6 \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.attention.input_norm \
  --out profile/kimi-attention-row6-row7-h20-baseline/row6_ncu_discover.json
```

Evidence:

| Metric | Value |
|---|---:|
| Provider | FlashInfer `RMSNormKernel<8,bf16>` via `rms_norm_batch` |
| Shape | hidden=`7168`, batch=`8`, BF16 |
| NCU duration | `3.97-4.22us` |
| Grid / block | `8` CTAs x `896` threads |
| Waves / SM | `0.05` |
| Memory throughput | `34-36GB/s`, `1.1-1.3%` max bandwidth |
| DRAM throughput | `0.70-0.74%` |
| Compute throughput | `2.38-2.53%` |
| Achieved occupancy | `41-42%` |
| Scheduler no eligible | `60-61%` |
| NCU top rule | grid too small to fill H20; only about `0.1` full waves |

Current `h20-100` status for fresh NCU:

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# exits 124 with no output
```

If a final-label rerun is needed after NCU recovers, collect:

```bash
/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -k regex:RMSNormKernel \
  -o profile/kimi-final-norm-h20/reports/final_norm_full \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.final.norm \
  --out profile/kimi-final-norm-h20/final_norm_ncu.json
```

Diagnosis: final norm is not H20 HBM-bound or compute-bound in the actionable sense. It is the same tiny FlashInfer RMSNorm launch already profiled for attention input norm, but only once per decode step instead of `61` times.

## Bench Evidence

Runtime path:

| Item | Value |
|---|---|
| Runtime call | `typed_ops::rms_norm_into` in `pegainfer-kimi-k2/src/runner/worker/forward.rs` |
| Bench op | `rms_norm_batch` / `decode.final.norm` |
| Shape | `rows=8, hidden=7168`, BF16 |
| Calls per decode step | `1` |
| Consumer | `decode.final.lm_head` |

H20 timing from `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json`:

| Op | Calls | Step latency | Per call | TFLOP/s | Payload GB/s | H20 HBM pct |
|---|---:|---:|---:|---:|---:|---:|
| `decode.attention.input_norm` | 61 | `490.93us` | `8.05us` | `0.036` | `57.00` | `1.19%` |
| `decode.final.norm` | 1 | `8.01us` | `8.01us` | `0.036` | `57.27` | `1.19%` |

The final norm row is one call of the same kernel shape as attention input norm. Its total contribution is below the major decode bottlenecks and below the threshold where a standalone rewrite is likely to move TP1 PPLX TPOT.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current FlashInfer RMSNorm path | `8.01us/call`, `57.3GB/s` payload-equivalent; same-shape NCU says tiny-grid/control. | Current baseline. |
| Standalone RMSNorm rewrite | Not attempted. Same-shape NCU says the row is launch/grid limited and only one call per decode step. | Stop standalone direction. |
| Fuse into LM-head prologue | Not implemented. The following LM head is a BF16 full-vocab GEMM already at `~90%` H20 HBM, so any fused path must preserve that bandwidth. | Future-only direction, gated by a stronger LM-head GEMM plan. |

## Final Conclusion

Keep the current FlashInfer `rms_norm_batch` final norm provider. Reclassify the master row as `control/tiny-grid`; do not treat the `1.2%` HBM payload number as evidence for memory-bound optimization. Reopen only if a future LM-head path can absorb RMSNorm without slowing the `542.7us` BF16 LM-head baseline, or if final-label NCU contradicts the same-shape row-6 profile.
