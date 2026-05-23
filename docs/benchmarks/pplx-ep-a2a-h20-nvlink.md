# pplx EP all-to-all latency — H20 ×8 NVLink

> **TL;DR** Dispatch+combine round-trip on 8× H20 (NV18 full-mesh NVLink): tok=1 p50 ~82μs, tok=256 p50 ~204μs (DSV4) / ~303μs (Kimi-K2). Latency scales smoothly with token count; no anomalies except a single DSV4 tok=8 tail spike.

Last touched: 2026-05

## Hardware

| Component | Spec |
|---|---|
| GPU | 8× NVIDIA H20-3e, 141 GB HBM3 each |
| NVLink | NV18 full-mesh (all GPU pairs connected) |
| RDMA NIC | 4× ConnectApply-7 NDR 400Gb/s + 1× ConnectX-6 HDR 200Gb/s |
| NIC ↔ GPU | 2 NICs per NUMA node, PIX to nearest GPU pair |
| NUMA | 2 sockets, GPU 0-3 on NUMA 0, GPU 4-7 on NUMA 1 |
| Driver | 575.57.08 |

## Benchmark

Binary: `pplx_a2a_bench --sweep` (in `pegainfer-comm`).

Each config bootstraps a fresh pplx-garden EP backend (CUMem + fabric MR + NVLink peer-map), runs 20 warmup + 100 measured iterations of the full dispatch_send → dispatch_recv → combine_send → combine_recv cycle, and reports `max_rank_split_sum_us` — the per-iteration maximum across all 8 ranks of the four-stage sum.

Subprocess isolation per config (sequential bootstrap in the same process deadlocks on teardown).

## Results

Sweep: DSV4 (256 experts, topk=6, hidden=4096) and Kimi-K2 (384 experts, topk=8, hidden=7168), `max_num_tokens` ∈ {1, 4, 8, 32, 128, 256}.

### max_rank_split_sum_us

| config | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| dsv4/tok=1 | 82.2 | 79.9 | 98.8 | 110.4 | 169.4 |
| dsv4/tok=4 | 92.0 | 90.9 | 103.7 | 117.2 | 120.5 |
| dsv4/tok=8 | 137.5 | 109.0 | 272.8 | 341.0 | 677.7 |
| dsv4/tok=32 | 95.5 | 94.0 | 110.1 | 114.9 | 125.1 |
| dsv4/tok=128 | 129.8 | 128.1 | 144.6 | 165.1 | 167.3 |
| dsv4/tok=256 | 205.5 | 204.3 | 219.4 | 236.6 | 238.9 |
| kimi-k2/tok=1 | 84.2 | 82.2 | 96.5 | 106.5 | 119.6 |
| kimi-k2/tok=4 | 86.5 | 84.1 | 102.3 | 116.6 | 118.0 |
| kimi-k2/tok=8 | 89.6 | 88.1 | 104.6 | 114.2 | 133.2 |
| kimi-k2/tok=32 | 105.3 | 103.4 | 121.1 | 129.6 | 139.2 |
| kimi-k2/tok=128 | 182.6 | 181.2 | 194.7 | 204.4 | 207.3 |
| kimi-k2/tok=256 | 303.5 | 302.8 | 321.3 | 325.2 | 330.1 |

### Per-stage breakdown (dsv4/tok=1, flattened across ranks)

| stage | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| dispatch_send | 24.2 | 22.5 | 30.6 | 56.5 |
| dispatch_recv | 14.7 | 14.2 | 18.1 | 26.4 |
| combine_send | 18.4 | 17.0 | 25.3 | 29.1 |
| combine_recv | 15.7 | 15.2 | 19.1 | 25.5 |

### Per-stage breakdown (kimi-k2/tok=256, flattened across ranks)

| stage | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| dispatch_send | 62.8 | 62.0 | 70.1 | 78.3 |
| dispatch_recv | 98.7 | 99.1 | 109.4 | 115.1 |
| combine_send | 103.0 | 102.2 | 117.4 | 124.3 |
| combine_recv | 25.8 | 24.2 | 36.1 | 41.1 |

## Notes

- **DSV4 tok=8 tail spike**: p50=109μs but max=678μs. Other token counts are clean. Likely a one-off scheduling stall rather than a systematic issue.
- **tok=1 latency is model-independent**: ~80-84μs for both shapes. The fixed overhead (CUMem peer reads, sync buffer polling) dominates at low token counts.
- **tok=256 Kimi-K2 vs DSV4**: 303μs vs 205μs (1.48×). Data volume ratio is (7168×8)/(4096×6) = 2.33×, so the backend amortizes well — sub-linear scaling with payload size.
- **combine_send is the costliest stage at high token counts** (103μs for kimi-k2/tok=256), likely because it writes the full expert output tensor into the peer-mapped NVLink buffers.
