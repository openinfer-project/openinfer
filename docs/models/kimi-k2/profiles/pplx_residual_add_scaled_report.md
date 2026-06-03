# pplx_residual_add_scaled Report

> **TL;DR:** `decode.moe.residual_add_scaled` / `kimi_residual_add_scaled_f32` is a small fused PPLX post-combine elementwise row, not a useful standalone HBM target. At TP1/DP8/PPLX `bs=8,ctx=1`, it launches `224 x 256` threads per MoE layer for `rows=8, hidden=7168` and costs `408.3-410.1us/step` across `60` calls (`6.81-6.83us/call`, about `84GB/s` payload-equivalent). H20 selected NCU confirms low utilization (`224` CTAs, `0.36` waves/SM, `8.37%` SM, `3.33%` DRAM, `91.25%` no-eligible). Keep the current fused scaled-add kernel and only revisit it as part of a launch-removing epilogue/prologue fusion that preserves Kimi's BF16 rounding boundary.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's `wiki/patterns/memory-bound.md` says to call a kernel memory-bound only after profile evidence shows high DRAM throughput with low compute utilization. This row moves only `573,440` payload bytes per call in the bench model and reaches about `84GB/s`, so the old `1.8% HBM` number is evidence that the row is not a bandwidth-saturation target. It is better described as a small launch/control row with simple streaming accesses.

KernelWiki's `wiki/patterns/low-sm-utilization.md` matches the likely failure mode: a grid that is not much larger than the SM count, short runtime, and limited independent work per launch. The concrete advice for this class is not to tune arithmetic, but to remove launches or combine adjacent work when correctness allows.

KernelWiki's `wiki/techniques/epilogue-fusion.md` lists residual add as a typical epilogue operation. For this exact row, the plausible future direction is not a faster standalone add kernel; it is fusing the residual/scaled add into an upstream GEMM epilogue or downstream prologue while keeping the current arithmetic order:

```text
round_bf16(hidden + projected) + routed_f32 * KIMI_K2_ROUTER_SCALE
```

That BF16 rounding point is part of the production math and must be preserved by any fused variant.

## NCU Conclusion

Event-timing evidence from H20 artifacts:

| Artifact | Step latency | Per call | Payload throughput | Notes |
|---|---:|---:|---:|---|
| `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json` | `408.3us` | `6.805us` | `84.27GB/s` | Original Phase 1 baseline, no roofline fields. |
| `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` | `410.1us` | `6.834us` | `83.90GB/s` | Older bench model labeled this memory-bound at `1.75%` HBM; that label is now considered misleading. |

Source geometry:

| Field | Value |
|---|---|
| Call site | `pegainfer-kimi-k2/src/runner/moe_pplx.rs` after PPLX `combine_recv` |
| Rust wrapper | `pegainfer-kernels/src/ops/kimi_k2/experts.rs`, `kimi_residual_add_scaled_f32` |
| CUDA kernel | `pegainfer-kernels/csrc/kimi_k2/kimi_experts.cu`, `kimi_residual_add_scaled_f32_kernel` |
| Elements per call | `8 * 7168 = 57344` |
| Launch | `224` CTAs x `256` threads |
| Payload bytes per call | `57344 * (BF16 hidden + BF16 projected + F32 routed + BF16 out) = 573440B` |
| Calls per decode step | `60` MoE layers |

The kernel performs one linear pass:

```cuda
rounded = bf16(hidden[idx] + projected[idx]);
scaled = routed_f32[idx] * scale;
out[idx] = bf16(scaled + float(rounded));
```

Selected H20 NCU:

```bash
/usr/local/cuda/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section SchedulerStats --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --launch-skip 3 --launch-count 1 \
  -k regex:kimi_residual_add_scaled_f32_kernel \
  -o /dev/shm/kimi-residual-add-scaled-ncu/reports/residual_add_scaled_selected \
  --force-overwrite /dev/shm/pegainfer-kimi-partition-target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.moe.residual_add_scaled \
  --out /dev/shm/kimi-residual-add-scaled-ncu/residual_add_scaled_ncu.json
```

| Metric | Value |
|---|---:|
| NCU kernel | `kimi_residual_add_scaled_f32_kernel` |
| NCU duration | `2.85us` |
| Grid / block | `224` CTAs x `256` threads |
| Waves / SM | `0.36` |
| Registers/thread | `16` |
| Dynamic shared memory/block | `0 B` |
| Memory throughput | `162.16GB/s`, `3.33%` DRAM throughput |
| Compute throughput | `8.37%` |
| Achieved occupancy | `34.03%` |
| L1/TEX / L2 hit rate | `8.25%` / `31.76%` |
| Scheduler no eligible | `91.25%` |
| Top rule | Grid/workload too small; issue slot utilization local speedup estimate `91.25%` |

The payload-equivalent bandwidth is low because the launch is small and the arithmetic is sparse. NCU confirms this is not HBM- or compute-bound; it is dominated by fixed launch/control work and limited independent work.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current fused PPLX residual path | One kernel already combines residual hidden, shared projection, routed F32 output, router scale, and BF16 output rounding. H20 `bs=8,ctx=1` is `6.81-6.83us/call`. | Keep. This is already the local fused helper for the PPLX combine boundary. |
| Treat row as memory-bound roofline target | The bench model reported only `~1.75%` of H20 HBM, but the row is a small `224`-CTA elementwise launch with `573KB` modeled payload per call. | Reject the memory-bound classification. Use `control/elementwise` in the master table. |
| Standalone CUDA retune | Selected NCU points to small-grid/control behavior, not a specific memory or instruction ceiling. | Reject. |

## Final Conclusion

Stop standalone optimization for `decode.moe.residual_add_scaled`. The current kernel is the right baseline for this phase: one exact-preserving fused elementwise pass after PPLX `combine_recv`, with a small fixed launch cost.

Reopen only under one of these conditions:

- A future upstream/downstream fusion removes the launch and preserves `round_bf16(hidden + projected)` before adding scaled routed F32.
- The PPLX combine boundary changes so routed output can be accumulated directly into the final hidden-state format without a separate post-combine pass.
