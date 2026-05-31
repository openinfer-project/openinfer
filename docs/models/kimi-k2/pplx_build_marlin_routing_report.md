# pplx_build_marlin_routing Report

> **TL;DR:** `decode.moe.pplx_build_marlin_routing` / `kimi_pplx_build_marlin_routing_on_stream` is a one-block PPLX metadata builder for Marlin, not a memory or compute roofline target. On the H20 trace replay p95 shape (`recv=67`, `padded=224`, `active_experts=28`, `recv_capacity=848`), it costs `9.87us/call` (`592.3us/step` across `60` MoE layers). NCU from the existing PPLX Marlin baseline shows `1 x 64`, `0.00` waves/SM, `0.04%` DRAM, and `87-88%` scheduler no eligible. Keep it as a control row; reopen only for launch-removing fusion or a route-aware Marlin scheduler that consumes counts directly.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's `wiki/patterns/low-sm-utilization.md` directly matches this row: the kernel launches fewer CTAs than H20 SMs and performs a small amount of metadata work, so utilization counters are structurally low. The page's useful direction for this class is to make the grid large enough or remove the launch by combining adjacent work; arithmetic tuning is not the lever.

KernelWiki's `wiki/patterns/tail-effect.md` and `wiki/patterns/moe-load-imbalance.md` explain why route histograms matter for the downstream Marlin W13/W2 rows, but this builder itself has a different bottleneck: it serializes a tiny prefix sum and fills a small routing table. Changing its math cannot fix PPLX load imbalance; it only prepares the metadata that lets the replay provider measure that imbalance honestly.

KernelWiki's grouped/fused MoE guidance still applies at the design boundary: the better long-term shape is for grouped/persistent Marlin scheduling to consume expert counts or compact metadata without an extra launch where possible. That is a new scheduler/kernel design, not a standalone retune of this `1 x 64` helper.

## NCU Conclusion

Existing NCU evidence is from `profile/kimi-pplx-marlin-compute-h20-baseline/` and is recorded in `pplx_marlin_compute_report.md`:

| Kernel | Duration | Grid / block | Waves/SM | DRAM | Memory throughput | SM throughput | Occupancy | L2 hit | No eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `kimi_pplx_build_marlin_routing_kernel` | `5.28-5.31us` | `1 x 64` | `0.00` | `0.04%` | `1.7GB/s` | `0.07-0.08%` | `2.3%` | `72.6-82.8%` | `87-88%` |

Trace replay event timing from `target/kernel_reports/kimi-k2/pplx-marlin-replay-bs64-kv2-varied.json`:

| Quantile | Source | recv / padded / active experts | Mean/call | Step latency | Bound |
|---|---|---:|---:|---:|---|
| p50 | `L31` rank1 | `56 / 96 / 8` | `11.27us` | `676.1us` | control |
| p95 | `L11` rank7 | `67 / 224 / 28` | `9.87us` | `592.3us` | control |
| p100 | `L17` rank5 | `207 / 336 / 26` | `10.18us` | `611.0us` | control |

Source geometry:

| Field | Value |
|---|---|
| Call sites | `pegainfer-kimi-k2/src/runner/moe_pplx.rs`, decode and prefill PPLX routing setup |
| Rust wrapper | `pegainfer-kernels/src/ops/kimi_k2/experts.rs`, `kimi_pplx_build_marlin_routing_on_stream` |
| CUDA kernel | `pegainfer-kernels/csrc/kimi_k2/kimi_experts.cu`, `kimi_pplx_build_marlin_routing_kernel` |
| Launch | `1` CTA x `64` threads |
| Target local experts | `48` |
| p95 replay metadata | `recv_capacity=848`, `expert_padding=8`, `block_size=8`, `padded_rows=224` |

The kernel does three small jobs: compute per-expert padded route counts, prefix-sum them in shared memory, then fill `sorted_token_ids`, `expert_ids`, and `num_tokens_post_padded`. The source includes a serial prefix loop on `tid == 0`, but the loop is over `48` local experts, so the measured cost is launch/control dominated rather than a scalable compute loop.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Synthetic PPLX provider | Original provider used expected-local-route counts and measured routing at `9.489us/call`, `569.3us/step`. | Kept as a stress/check provider, but not the serving-shape row. |
| Trace replay provider | Runtime route histograms feed real recv counts into the same local provider. P95 routing is `9.87us/call`, close to synthetic timing and insensitive to padded-row quantile. | Use trace replay p95 in the master table. |
| Standalone CUDA rewrite | The kernel is already one tiny metadata launch; NCU shows almost no DRAM/SM usage. | Reject until a concrete NCU finding appears. A rewrite that still launches one tiny kernel will mostly move overhead around. |

## Final Conclusion

Stop standalone optimization for `decode.moe.pplx_build_marlin_routing`. The current helper is a clear, bounded metadata kernel, and its cost is dominated by launch/control structure.

Future work should only revisit this row if it removes the launch or removes the metadata format itself, for example by folding route metadata generation into a route-aware grouped/persistent Marlin path. Keep EP dispatch/combine transport out of this report; that remains explicitly excluded from the decode optimization scope.
