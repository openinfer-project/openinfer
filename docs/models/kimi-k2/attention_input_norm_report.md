# Attention Input Norm Report

> **TL;DR:** `decode.attention.input_norm` is a standalone FlashInfer RMSNorm row, but the H20 profile says it is tiny-grid/launch limited rather than close to HBM peak: `8` CTAs on `78` SMs, `0.05` waves/SM, `0.70-0.74%` DRAM, and `60-61%` scheduler no eligible. At the TP1 PPLX target shape (`bs=8/rank`, `ctx=1`) it costs `8.008us/call` or `488.5us` per 61-layer decode step. Stop standalone RMSNorm tuning; future work should only revisit this row as an RMSNorm -> `qkv_a` prologue/custom skinny-GEMM fusion that beats the current cuBLAS qkv_a path in the full TP1 PPLX bench.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Low SM utilization can come from tail effect, static scheduling, or a grid smaller than the SM count; for non-persistent kernels, the grid should be much larger than SM count. | This row launches `8` CTAs on H20's `78` SMs, so standalone tuning is bounded by tiny-grid/launch behavior before HBM bandwidth. |
| `sources/prs/sglang/PR-20755.md` (`pr-sglang-20755`) | Small decode GEMMs sometimes need specialized small-GEMM kernels instead of generic library paths. | Directional only: the useful follow-up is not rewriting RMSNorm, but making sure any RMSNorm prologue fusion preserves or improves the downstream skinny `qkv_a` GEMM. |
| `sources/prs/flashinfer/PR-3014.md` (`pr-flashinfer-3014`) | Small-batch decode helper kernels benefit from removing unnecessary helper work and early exits around empty work. | Directional only: it supports avoiding standalone helper overhead, but this FlashInfer RMSNorm row has no empty-work loop to trim at the TP1 target shape. |

Practical conclusion: do not chase a standalone RMSNorm kernel for this row. The row's gross upside is deleting the launch and intermediate normalized-hidden traffic before `qkv_a`; that is a prologue-fusion problem, not a faster RMSNorm-in-isolation problem.

## NCU Conclusion

Workload: Kimi K2 TP1 DP8 EP8 + PPLX decode, per-rank `bs=8`, global `bs~=64`, `ctx=1`.

Runtime path: `pegainfer-kimi-k2/src/runner/worker/forward.rs` runs `rms_norm_batch_into` for `attention.input_norm` immediately before `fused_qkv_a_proj`. The bench provider maps `decode.attention.input_norm` to `rms_norm_batch`, which calls `rms_norm_batched_cuda`; `pegainfer-kernels/KERNELS.md` records this as the FlashInfer CUDA RMSNorm path.

Existing H20 run:

```bash
cargo run --release -p pegainfer-kimi-k2 --features kernel-report \
  --bin kimi_tp1_pplx_decode_bench -- \
  --active-rows 8 --ctx-lens 1 --iters 64 --format json \
  --labels decode.attention.input_norm,decode.attention.qkv_a \
  --out profile/kimi-attention-row6-row7-h20-baseline/row6_row7_event.json

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
| Event timing | `8.008us/call`, `488.5us/step` for 61 layers |
| Bench throughput | `57.3GB/s`, `1.19%` of H20 HBM roofline |
| NCU duration | `3.97-4.22us` |
| Grid / block | `8` CTAs x `896` threads |
| Waves / SM | `0.05` |
| Memory throughput | `34-36GB/s`, `1.1-1.3%` max bandwidth |
| DRAM throughput | `0.70-0.74%` |
| Compute throughput | `2.38-2.53%` |
| Achieved occupancy | `41-42%` |
| Scheduler no eligible | `60-61%` |
| NCU top rule | grid too small to fill H20; only about `0.1` full waves |

Diagnosis: this is not HBM-bound in the actionable sense. The arithmetic intensity formula classifies it as memory-side work, but NCU shows actual DRAM use is below `1%` and the launch has only one CTA per active decode row. A standalone rewrite could reduce a few microseconds, but it cannot turn this row into a high-bandwidth kernel at `bs=8`.

The adjacent `qkv_a` GEMM matters for any fusion decision:

| Row | H20 evidence |
|---|---|
| `decode.attention.qkv_a` | `20.407us/call`, `1.245ms/step`; main cuBLAS kernel launches `72` CTAs, reaches `51-53%` DRAM, and has a separate `~3us` split-K reduce. |

Any RMSNorm prologue fusion must therefore beat the current `20.407us/call` qkv_a path while deleting the `8.008us/call` RMSNorm launch. Losing more than about `8us/call` in the GEMM erases the entire row-6 benefit.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Standalone RMSNorm rewrite | Not attempted. Existing provider is already FlashInfer, and NCU says grid/launch dominates before HBM. | Stop standalone direction. |
| Tune current FlashInfer launch | Not attempted. Increasing CTAs would require splitting one row across multiple CTAs with a cross-CTA reduction, adding complexity and likely extra synchronization for only `8us/call` gross cost. | Reject as low leverage. |
| Fuse RMSNorm into `qkv_a` GEMM prologue | Not implemented in Phase 2. Profile says it is the only plausible route because it can delete the RMSNorm launch and normalized-hidden write/read. | Keep as future custom prologue/skinning-GEMM work; must beat full TP1 PPLX bench, not standalone timing only. |
| qkv_a cuBLASLt exact-shape baseline | Tried and rejected in `qkv_a_proj_report.md`: full TP1 bench improved only `20.407us -> 20.242us` (`0.8%`). | Do not swap qkv_a to cuBLASLt as a prerequisite. |

## Final Conclusion

Stop `decode.attention.input_norm` as a standalone Phase 3 target. The row is small enough that optimizing it in isolation is mostly launch/scheduling work, and the NCU profile does not support a claim that it is near any H20 memory or compute roofline. Keep the current FlashInfer provider. Reopen only for a true RMSNorm -> `qkv_a` prologue/custom GEMM that preserves TP1 DP8 token correctness and shows `>3%` full-bench improvement at `bs=8/rank`, global `bs~=64`.
