# pplx_swiglu Report

> **TL;DR:** `decode.moe.pplx_swiglu` / `kimi_marlin_w13_swiglu_pplx` is an elementwise/SFU row between PPLX routed W13 and W2. On H20 trace replay p95 (`recv=67`, `padded=224`, `recv_capacity=848`), it costs `12.66us/call` (`759.7us/step` across `60` MoE layers). Existing NCU shows `swiglu_w13_pplx_kernel` at `10.62us`, launch `6784 x 256`, DRAM `6.32%`, SM throughput `55.40%`, occupancy `76.05%`, and scheduler no eligible `34.20%`. Reclassify it from memory-bound to `compute/elementwise`; standalone tuning is stopped unless a route-aware launch bound or W13/W2 fusion removes the row.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's `wiki/patterns/memory-bound.md` does not match this row: DRAM is low (`6.32%` in NCU) and the row spends work on scalar activation math (`expf`, BF16 rounding, multiply). The old bench output's `4.5% HBM` gap should not be read as a memory-bandwidth opportunity.

KernelWiki's `wiki/techniques/epilogue-fusion.md` and `wiki/kernels/gated-dual-gemm.md` are the relevant directions. SwiGLU is a classic GEMM epilogue candidate: if W13 can produce the activated `INTER` output directly, this row disappears. In the current PPLX Marlin path, however, W13 and W2 are quantized WNA16 Marlin kernels with routing metadata, so a real fusion is a new routed-Marlin design rather than a small edit to this standalone kernel.

KernelWiki's MoE load-imbalance guidance still applies around this row: the launch currently uses a capacity bound and the actual padded row count is read on-device. For p95 replay, launch capacity is `848` rows but actual work is `224` rows, so better route-aware scheduling or tighter graph buckets could reduce wasted launch work. That is the next plausible direction, not scalar instruction tinkering inside the activation formula.

## NCU Conclusion

Existing NCU evidence is from `profile/kimi-pplx-marlin-compute-h20-baseline/` and is recorded in `pplx_marlin_compute_report.md`:

| Kernel | Duration | Grid / block | Waves/SM | DRAM | Memory throughput | SM throughput | Occupancy | L2 hit | No eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `swiglu_w13_pplx_kernel` | `10.62us` | `6784 x 256` | `10.87` | `6.32%` | `309GB/s` | `55.40%` | `76.05%` | `38.25%` | `34.20%` |

Trace replay event timing from `target/kernel_reports/kimi-k2/pplx-marlin-replay-bs64-kv2-varied.json`:

| Quantile | Source | recv / padded / active experts | Mean/call | Step latency | Payload throughput |
|---|---|---:|---:|---:|---:|
| p50 | `L31` rank1 | `56 / 96 / 8` | `11.26us` | `675.6us` | `104.8GB/s` |
| p95 | `L11` rank7 | `67 / 224 / 28` | `12.66us` | `759.7us` | `217.4GB/s` |
| p100 | `L17` rank5 | `207 / 336 / 26` | `13.51us` | `810.7us` | `305.6GB/s` |

Source geometry:

| Field | Value |
|---|---|
| Call sites | `pegainfer-kimi-k2/src/runner/moe_pplx.rs`, after PPLX W13 and before PPLX W2 |
| Rust wrapper | `pegainfer-kernels/src/ops/kimi_k2/experts.rs`, `kimi_marlin_w13_swiglu_pplx` |
| CUDA kernel | `pegainfer-kernels/csrc/kimi_k2/kimi_marlin_wna16.cu`, `swiglu_w13_pplx_kernel` |
| Launch bound | `max_rows * intermediate_dim / 256`, where `max_rows = recv_capacity` |
| p95 launch | `848 * 2048 / 256 = 6784` CTAs x `256` threads |
| p95 actual work | `num_tokens_post_padded[0] * 2048 = 224 * 2048` elements |

The kernel reads `num_tokens_post_padded[0]` on-device, so CTAs past actual rows return early without a host sync. That avoids the old D2H boundary, but the grid is still sized by capacity. The row is therefore a small but real capacity-bound elementwise launch; NCU does not show HBM saturation.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Previous PPLX row limiting | Historical PPLX decode work changed this path to read actual row count on-device, avoiding the earlier full-capacity activation work and removing D2H. | Keep. This is the current correct baseline. |
| Trace replay provider | Replays p50/p95/p100 route histograms into the local provider. P95 is `12.66us/call`, and the row is not a dominant PPLX local-compute cost compared with W13/W2. | Use p95 replay in the master table. |
| Treat as memory-bound | Bench payload model reported `217.4GB/s`, only `4.5%` HBM, while NCU reports `6.32%` DRAM and `55.40%` SM throughput. | Reject memory-bound classification; use `compute/elementwise`. |
| Standalone CUDA rewrite | No NCU evidence points to a memory coalescing or bandwidth issue. The obvious waste is capacity-sized launch versus actual padded rows. | Do not retune standalone. Reopen only for route-aware launch bounds, W13 epilogue fusion, or W2 prologue fusion with full-bench proof. |

## Final Conclusion

Stop standalone optimization for `decode.moe.pplx_swiglu`. The current kernel is already the no-D2H PPLX activation helper, and its p95 cost is about `0.76ms/step`.

The next meaningful work is architectural: either make routed Marlin W13 produce activated output, make W2 consume W13 gate/up with an activation prologue, or reduce the launch bound through route-aware graph buckets/scheduling. Any accepted variant must beat the current W13+SwiGLU+W2 replay and full TP1 PPLX bench by more than noise.
