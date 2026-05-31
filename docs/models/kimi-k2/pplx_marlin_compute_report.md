# pplx_marlin_compute Report

> **TL;DR:** PPLX routed local compute is measurable without timing EP communication. `kimi_tp1_pplx_decode_bench` keeps the synthetic stress point (`recv_capacity=848`, `64` expected local routes, `400` padded rows/rank), while `kimi_pplx_marlin_replay` now consumes runtime `kimi_pplx_route_histogram` rows directly. On H20, replaying `tp1-dp8-pplx-route-hist-bs64-kv2-varied.json` with `iters=16` gives padded-row p50/p95/max `96/224/336`; W13 is `114.5/250.6/368.6us` per call and W2 is `66.4/138.5/200.3us` per call. The p95 row (`recv=67`, `padded=224`, active experts `28`) is memory-bound by the H20 roofline model at `38.9%`/`35.4%` HBM for W13/W2, but NCU says the kernel is not at the device limit: W13/W2 are `1` wave/SM, `76KB` dynamic shared memory, `18.8%` theoretical occupancy, `4-5%` L2 hit, and only `13-14%` tensor-pipe elapsed utilization. Synthetic `400` rows remain a stress case, not the serving-shape baseline.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's `wiki/kernels/fused-moe.md` and `wiki/kernels/grouped-gemm.md` pages match the direction: MoE local compute should be treated as grouped/masked expert GEMM with variable per-expert M, where launch count, padding, and load imbalance are first-class performance variables. The relevant caution is that decode uses small per-expert M; padding and masked layouts can waste compute, while grouped scheduling helps only when the route distribution is represented honestly.

For this report, the bench provider deliberately excludes EP dispatch/combine transport and only times the local PPLX compute kernels after synthetic recv counts have been materialized. That makes the rows measurable for NCU and master-table accounting. Runtime decode traces now record `kimi_pplx_route_histogram` after `dispatch_recv`, including `recv_counts`, `recv_total_routes`, `active_local_experts`, `max_count_per_expert`, `padded_rows`, `num_tokens_post_padded`, `recv_capacity`, `expert_padding`, and `block_size`; the final optimization target still needs an H20 all-rank artifact using those fields.

The first all-rank H20 trace artifact is `target/kernel_reports/kimi-k2/tp1-dp8-pplx-route-hist-bs64-kv2-varied.json`. It uses deterministic varied prompt token ids rather than all-zero prompts, because all-zero prompts collapse routes into a few experts and are not a useful optimization target. The trace still includes scheduler admission effects: two `active_rows=1` waves plus two near-target waves where rank0 has `active_rows=7` and ranks1-7 have `active_rows=8` (`504` routes/wave instead of ideal `512`). Replay filters `active_rows>=7` and non-empty local routes, then measures p0/p50/p90/p95/p99/p100 histograms with the same local CUDA providers. Treat replay as local compute evidence; it still excludes EP transport.

## NCU Conclusion

Filtered H20 bench:

| Row | Provider | Mean/call | Step latency | Roofline read |
|---|---|---:|---:|---|
| `decode.moe.pplx_build_marlin_routing` | `kimi_pplx_build_marlin_routing_on_stream` | `9.489us` | `569.3us` | control |
| `decode.moe.pplx_marlin_w13` | `kimi_marlin_wna16_pplx_w13_gemm` | `436.432us` | `26.186ms` | memory, `53.82 TF/s`, `1.837 TB/s`, `38.3%` HBM |
| `decode.moe.pplx_swiglu` | `kimi_marlin_w13_swiglu_pplx` | `14.135us` | `848.1us` | memory, `347.7 GB/s`, `7.2%` HBM |
| `decode.moe.pplx_marlin_w2` | `kimi_marlin_wna16_pplx_w2_gemm` | `236.797us` | `14.208ms` | memory, `49.60 TF/s`, `1.705 TB/s`, `35.5%` HBM |

NCU summary:

| Kernel | Duration | Grid / block | Waves/SM | DRAM | Memory throughput | SM throughput | Occupancy | L2 hit | No eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `kimi_pplx_build_marlin_routing_kernel` | `5.28-5.31us` | `1 x 64` | `0.00` | `0.04%` | `1.7 GB/s` | `0.07-0.08%` | `2.3%` | `72.6-82.8%` | `87-88%` |
| `Marlin<...>` W13 | `467.10us` | `234 x 128` | `1.00` | `34.73%` | `1.71 TB/s` | `58.72%` | `17.51%` | `4.88%` | `40.64%` |
| `swiglu_w13_pplx_kernel` | `10.62us` | `6784 x 256` | `10.87` | `6.32%` | `309 GB/s` | `55.40%` | `76.05%` | `38.25%` | `34.20%` |
| `Marlin<...>` W2 | `250.05us` | `234 x 128` | `1.00` | `32.55%` | `1.60 TB/s` | `56.80%` | `17.63%` | `5.29%` | `42.23%` |

Conclusion: W13/W2 are the dominant PPLX local-compute rows in this synthetic baseline. They are not HBM-saturated; they run with a single wave per SM, low L2 hit rate, and moderate SM throughput. Routing is launch/control dominated. PPLX SwiGLU is not the main cost after the current device-side row limiting.

Trace replay H20 bench (`target/kernel_reports/kimi-k2/pplx-marlin-replay-bs64-kv2-varied.json`, `iters=16`):

| Quantile | Source | recv / padded / experts | Routing | W13 | SwiGLU | W2 | Roofline read |
|---|---|---:|---:|---:|---:|---:|---|
| p50 | L31 rank1 | `56 / 96 / 8` | `11.27us` | `114.52us` | `11.26us` | `66.39us` | W13/W2 compute-bound by AI, `33.3%` / `28.7%` BF16 peak |
| p95 | L11 rank7 | `67 / 224 / 28` | `9.87us` | `250.64us` | `12.66us` | `138.51us` | W13/W2 memory-bound, `38.9%` / `35.4%` HBM |
| p100 | L17 rank5 | `207 / 336 / 26` | `10.18us` | `368.57us` | `13.51us` | `200.31us` | W13/W2 compute-bound by AI, `36.2%` / `33.3%` BF16 peak |

The roofline label flips between memory and compute because the actual active-expert count changes the weight-byte term. This is exactly why PPLX Marlin must be optimized against a route histogram, not only against padded-row count.

Trace replay p95 NCU (`profile/kimi-pplx-marlin-replay-p95-h20/`, full + source reports):

| Kernel | Duration | Grid / block | Waves/SM | Dynamic shared memory | Theoretical / achieved occupancy | SM throughput | Tensor pipe | DRAM read BW | L2 hit | Main NCU rules |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| W13 p95 | `265.6us` | `234 x 128` | `1.00` | `76,458 B` | `18.75% / 17.50%` | `58.07%` | `14.08%` | `1.75 TB/s` | `4.45%` | latency issue; theoretical occupancy limited by shared memory, est. `41.1%`; non-fused FP32, est. `22.3%` |
| W2 p95 | `144.3us` | `234 x 128` | `1.00` | `76,458 B` | `18.75% / 17.67%` | `55.94%` | `13.00%` | `1.60 TB/s` | `5.07%` | latency issue; theoretical occupancy limited by shared memory, est. `42.5%`; non-fused FP32, est. `21.3%`; shared-store bank conflicts, est. `5.5%` |

Source-counter reports were collected, but the current release build does not include CUDA line info for this Marlin TU, so `action.source_info(pc)` has no file/line mapping. The aggregate source counters still matter: W13 reports `11,360` excessive shared wavefronts; W2 reports `61,552` excessive shared wavefronts and `1,523` excessive global sectors. The actionable NCU conclusion is therefore at kernel-structure level: the p95 replay shape is limited by low occupancy / one wave per SM / low L2 reuse, not by saturated HBM.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Add TP1 PPLX local-compute providers to `kimi_tp1_pplx_decode_bench` | Rows 24-27 now measure through existing Rust/CUDA wrappers with synthetic recv counts: `64` expected local routes, `400` expected padded rows, and `recv_capacity=848` at `bs=8`. | Kept as baseline coverage. This is bench/report infrastructure, not an optimization. |
| Worst-case synthetic counts using global `512` routes on one EP rank | W13 `905.95us/call`, W2 `485.03us/call`; this filled the local rank to capacity and contradicted the target global `bs~=64` expected load. | Rejected as the default provider shape. It remains a useful stress case, but not the anchor workload. |
| Expected-local-route synthetic counts | W13 `436.43us/call`, W2 `236.80us/call`; NCU captures the Marlin kernels cleanly. | Adopted for the bench baseline until an all-rank route histogram replaces it. |
| Add runtime `kimi_pplx_route_histogram` trace | `kimi_kernel_report` / `kimi_model_report` runtime traces can be run with `--tp-world 1 --dp-world 8 --ep-backend pplx` and record real per-layer recv histograms without timing EP transport. | Kept as diagnostic infrastructure. No `opt(...)` commit: it does not change kernel latency. |
| H20 varied-prompt route histogram | `batch_size=64`, `kv_len=2`, TP1/DP8/PPLX, artifact `tp1-dp8-pplx-route-hist-bs64-kv2-varied.json`. Near-target active8 rows: `recv_total_routes` p50/p95/max `63/161/282`, `padded_rows` p50/p95/max `80/216/336`, active local experts p50/p95/max `3/24/32`. Worst near-target layer padded sum was `1744` rows across ranks for `504` routes. | Kept as the source histogram for local-compute replay. It still includes admission waves, so replay filters active rows and non-empty local routes explicitly. |
| Add trace-driven replay provider | `kernel_report` PPLX providers accept `pplx_recv_counts`; `kimi_pplx_marlin_replay` selects p0/p50/p90/p95/p99/p100 non-empty active rows from the H20 trace and replays routing/W13/SwiGLU/W2 locally. H20 p95 W13/W2 are `250.64us` / `138.51us`, much lower than synthetic `436.43us` / `236.80us`. | Kept as baseline infrastructure. No `opt(...)` commit: it improves measurement truth, not kernel latency. |
| Add replay filters for NCU isolation | `kimi_pplx_marlin_replay --quantiles p95 --ops w13,w2` isolates one route histogram and one/two Marlin providers without relying on NCU launch-skip over unrelated samples. | Kept as profiling infrastructure. |
| NCU p95 W13/W2 replay | Full/source NCU collected under `profile/kimi-pplx-marlin-replay-p95-h20/`. Both kernels are one-wave and shared-memory-limited; source line mapping is absent because the release TU has no line info. | Use as the current p95 diagnosis. No optimization adopted yet. |
| Force small-batch 256-thread Marlin config | Hard-coded `thread_m_blocks==1` to pick the existing `{thread_k=128, thread_n=128, num_threads=256}` config instead of the current `{64,128,128}`. H20 p95 replay regressed W13 `250.64us -> 303.50us` and W2 `138.51us -> 187.31us`. | Rejected and reverted. The current 128-thread config remains better for the p95 route histogram. |

## Final Conclusion

The immediate gap for PPLX routed local compute was measurement coverage, not a code optimization. The master table should use the trace replay p95 row for PPLX W13/SwiGLU/W2/routing and keep the synthetic `400` padded-row result as a stress reference.

Do not commit an `opt(...)` change from this report: no faster kernel was adopted. The next optimization step is to test a targeted Marlin scheduling/tile variant only if it attacks the measured structure: lower shared-memory footprint, more CTAs/waves for small-M experts, or better route-aware grouping. If a variant cannot improve p95 W13/W2 over the replay baseline by more than noise, stop this kernel and move focus to the next decode-path row.
