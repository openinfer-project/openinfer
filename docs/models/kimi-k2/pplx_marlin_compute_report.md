# pplx_marlin_compute Report

> **TL;DR:** PPLX routed local compute is measurable in `kimi_tp1_pplx_decode_bench` without timing EP communication, and runtime decode tracing now emits per-layer `kimi_pplx_route_histogram` rows after `dispatch_recv`. For TP1 PPLX `bs=8,ctx=1`, the synthetic expected-local-route provider uses `recv_capacity=848`, `64` expected local routes per EP rank, and `400` expected padded work rows/rank. H20 event timing is `pplx_build_marlin_routing=9.49us/call`, `pplx_marlin_w13=436.43us/call`, `pplx_swiglu=14.13us/call`, and `pplx_marlin_w2=236.80us/call`. A H20 varied-prompt trace at global `batch_size=64`, `kv_len=2` produced near-target waves with `504` routes/wave; active8 ranks had padded rows/rank p50 `80`, p95 `216`, max `336`, so the synthetic `400`-row latency is conservative and should not be treated as real serving shape.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

KernelWiki's `wiki/kernels/fused-moe.md` and `wiki/kernels/grouped-gemm.md` pages match the direction: MoE local compute should be treated as grouped/masked expert GEMM with variable per-expert M, where launch count, padding, and load imbalance are first-class performance variables. The relevant caution is that decode uses small per-expert M; padding and masked layouts can waste compute, while grouped scheduling helps only when the route distribution is represented honestly.

For this report, the bench provider deliberately excludes EP dispatch/combine transport and only times the local PPLX compute kernels after synthetic recv counts have been materialized. That makes the rows measurable for NCU and master-table accounting. Runtime decode traces now record `kimi_pplx_route_histogram` after `dispatch_recv`, including `recv_counts`, `recv_total_routes`, `active_local_experts`, `max_count_per_expert`, `padded_rows`, `num_tokens_post_padded`, `recv_capacity`, `expert_padding`, and `block_size`; the final optimization target still needs an H20 all-rank artifact using those fields.

The first all-rank H20 trace artifact is `target/kernel_reports/kimi-k2/tp1-dp8-pplx-route-hist-bs64-kv2-varied.json`. It uses deterministic varied prompt token ids rather than all-zero prompts, because all-zero prompts collapse routes into a few experts and are not a useful optimization target. The current trace still includes scheduler admission effects: two `active_rows=1` waves plus two near-target waves where rank0 has `active_rows=7` and ranks1-7 have `active_rows=8` (`504` routes/wave instead of ideal `512`). Use it to bound synthetic pessimism, not yet to replace the latency rows.

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

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Add TP1 PPLX local-compute providers to `kimi_tp1_pplx_decode_bench` | Rows 24-27 now measure through existing Rust/CUDA wrappers with synthetic recv counts: `64` expected local routes, `400` expected padded rows, and `recv_capacity=848` at `bs=8`. | Kept as baseline coverage. This is bench/report infrastructure, not an optimization. |
| Worst-case synthetic counts using global `512` routes on one EP rank | W13 `905.95us/call`, W2 `485.03us/call`; this filled the local rank to capacity and contradicted the target global `bs~=64` expected load. | Rejected as the default provider shape. It remains a useful stress case, but not the anchor workload. |
| Expected-local-route synthetic counts | W13 `436.43us/call`, W2 `236.80us/call`; NCU captures the Marlin kernels cleanly. | Adopted for the bench baseline until an all-rank route histogram replaces it. |
| Add runtime `kimi_pplx_route_histogram` trace | `kimi_kernel_report` / `kimi_model_report` runtime traces can be run with `--tp-world 1 --dp-world 8 --ep-backend pplx` and record real per-layer recv histograms without timing EP transport. | Kept as diagnostic infrastructure. No `opt(...)` commit: it does not change kernel latency. |
| H20 varied-prompt route histogram | `batch_size=64`, `kv_len=2`, TP1/DP8/PPLX, artifact `tp1-dp8-pplx-route-hist-bs64-kv2-varied.json`. Near-target active8 rows: `recv_total_routes` p50/p95/max `63/161/282`, `padded_rows` p50/p95/max `80/216/336`, active local experts p50/p95/max `3/24/32`. Worst near-target layer padded sum was `1744` rows across ranks for `504` routes. | Kept as evidence that synthetic `400` padded rows/rank is conservative. Need a clean steady-state full `active_rows=8` trace before changing the master latency row. |

## Final Conclusion

The immediate gap for PPLX routed local compute was measurement coverage, not a code optimization. The master table should no longer treat W13/SwiGLU/W2/routing as invisible estimate-only rows.

Do not commit an `opt(...)` change from this report: no faster kernel was adopted. The next optimization step is to replace the synthetic PPLX Marlin provider with a trace-driven or replay-driven provider using real `recv_tokens_per_expert` histograms, then remeasure W13/W2. If the measured rows fall below the current top bottlenecks under real route counts, move the optimization focus accordingly instead of tuning Marlin against the conservative synthetic shape.
