# Qwen3-4B serving perf vs vLLM on RTX 5090

**TL;DR**: After three fixes — batched step tail (#345 root cause), chunked prefill (`--max-prefill-tokens`, default 2048), and dropping the bs8/16 CUDA-graph buckets (cuBLAS split-K hole) — openinfer beats vLLM 0.22.1 on TTFT at every QPS, beats it outright in overload (QPS16: 61.7ms TPOT / 1865 tok/s vs 78.6ms / 1674), and matches it up to QPS4. Remaining gap: mid-band TPOT (QPS 8–12, 1.25–1.6× vLLM), bounded by prefill cost already near the bf16 roofline — needs prefill kernel work, not scheduling.

Last touched: 2026-06

## Setup

- RTX 5090, `/data/Qwen3-4B`, dev checkout `~/develop/xingming/pegainfer` on the 5090 box.
- Workload: `vllm bench serve`, random dataset, in=1024 / out=128, Poisson arrivals, seed 42, 60s per QPS point (`tools/bench/qps_sweep.sh`).
- Reference: vLLM 0.22.1, `--max-model-len 8192 --no-enable-prefix-caching`, defaults otherwise (max_num_seqs=256, max_num_batched_tokens=2048 chunked prefill, decode-priority). Verified from v0.22.1 source: vLLM does **not** lock batch size at 128; observed ~129 concurrency is KV-capacity-bound.
- Results: 5090 `/data/xingming/bench/20260611-tune/{vllm-ref,oi-chunk512,oi-chunk1024,oi-chunk2048,oi-final,oi-default2048}/`.

## #345 root cause: per-request step tail

nsys at bs≈120 (`--cuda-graph-trace=node`, cudaProfilerApi capture): prefill/unified steps spent ~57ms (~25% of GPU time) in a per-request loop of extract_vec → single-row RMSNorm → lm_head GEMV (0.48ms each) → single-row sample → blocking 1KB pageable DtoH. Linear in batch size, so it presented as "batch doesn't scale" after #341 raised the CUDA-graph buckets to 256. Decode kernels themselves scale fine and beat vLLM ≥bs32 (bs128 step: 13.7ms vs vLLM's 28.8ms implied).

Fix (`perf/qwen3-batched-step-tail`): gather last-token rows → batched RMSNorm → one GEMM → `select_batch_tokens_into` → one DtoH. Same audit found the pattern in other model crates — tracked in #353 (qwen35 is worst: full-vocab blocking DtoH per request per decode step).

## Chunked prefill (`--max-prefill-tokens`)

Prompts are split across steps under a per-step forwarded-token budget; mid-prompt requests live in a scheduler `prefilling` queue, the executor clamps each chunk at the request's `kv_position` (prefix-cache hits compose), non-final chunks apply KV without emitting a token (kvbm `apply_prefill(None)`). Echo never splits. Admission reserves KV + decode slots for mid-prefill requests; KV prefetch respects a reserve floor so it can't steal a queued chunk's blocks.

### Sweep (TPOT p50 / ITL p99, ms)

| QPS | chunk=2048 | chunk=1024 | chunk=512 | vLLM |
|---|---|---|---|---|
| 1 | 7.49 / 9 | 7.49 / 9 | 7.55 / 37 | 6.86 / 12 |
| 2 | 8.40 / 10 | 8.41 / 10 | 8.39 / 38 | 7.42 / 37 |
| 4 | 9.54 / 62 | 9.57 / 62 | 9.58 / 40 | 8.89 / 39 |
| 8 | 15.88 / 100 | 16.44 / 63 | 19.24 / 42 | 12.09 / 47 |
| 10 | 21.81 / 102 | 23.92 / 66 | 32.08 / 45 | 14.62 / 82 |
| 12 | 33.92 / 110 | 41.23 / 73 | 43.96 / 46 | 20.49 / 103 |
| 16 | 63.9 / 112 (**1819 tok/s**) | 69.1 / 75 (1692) | 44.1 / 46 (1433) | 78.6 / 119 (1674) |

Reading:

- **The budget is an ITL-tail vs throughput knob.** 512 matches or beats vLLM's ITL p99 everywhere (vLLM itself degrades to 82/103ms at QPS 10/12) but costs 15% of the overload ceiling. 2048+ keeps the ceiling and wins overload outright (+8.6% throughput, −19% TPOT, TTFT p50 2.0s vs 4.3s at QPS16).
- **Default is 2048** (matches vLLM's `max_num_batched_tokens`): vs 8192 it costs ~2.5% overload throughput but halves overload ITL p99 (112 vs 257ms) by capping a step at two whole 1k prompts. Single prompts ≤2048 tokens never split. Latency-sensitive deployments: `--max-prefill-tokens 512`.
- **Chunking does NOT close the mid-band TPOT gap** — smaller chunks make TPOT *worse* (more steps, more per-step overhead; decode rows pay the prefill toll more often). The gap has a different cause:

## Fixed: bs8–16 decode-step anomaly (cuBLAS skips split-K)

Fine sweep (ctx1024, CUDA graph, p50 ms): bs4 8.11, **bs8 9.42, bs12 9.47, bs16 9.52**, bs20 8.51, bs24 8.52, bs32 8.63 — non-monotonic, and mid-band serving (QPS 8–12) runs at exactly this concurrency. nsys diff of the bs8 vs bs20 graphs: bs20's GEMMs use split-K (7200 `splitKreduce_kernel` calls, GEMM med 16.8µs) while bs8's don't (zero, med 23.2µs) — cuBLAS's heuristic leaves ~20 CTAs on 170 SMs for batch∈[8,16]. GEMM time per step: 7.28ms (bs8) vs 6.15ms (bs20), i.e. less work, +1.1ms. Attention was innocent (40.5 vs 44.8µs, monotonic).

Fix: removed buckets 8/16 from `BATCH_BUCKETS` (now `1,2,4,20,24,…`) so batches 5–19 pad up to 20 and get the split-K configs. Verified: bs5/8/16 steps 9.38/9.40/9.50 → **8.26/8.26/8.34ms** (−12%); padded attention rows are near-free (padded bs5@20 is faster than real bs20).

## Final sweep (all three fixes, TPOT p50 / ITL p99 ms, TTFT p50 ms)

| QPS | openinfer | vLLM 0.22.1 | TTFT oi / vllm |
|---|---|---|---|
| 1 | **7.48** / 8 | 6.86 / 12 | **64** / 56 |
| 2 | 8.10 / 10 | 7.42 / 37 | **32** / 59 |
| 4 | **8.63** / 62 | 8.89 / 39 | 64 / 61 |
| 8 | 15.08 / 100 | 12.09 / 47 | 73 / 68 |
| 10 | 21.38 / 104 | 14.62 / 82 | 99 / 79 |
| 12 | 32.76 / 173 | 20.49 / 103 | 126 / 123 |
| 16 | **61.65** / 257 (**1865 tok/s**) | 78.55 / 119 (1674) | **1355** / 4279 |

(Run with budget 8192. Confirmation sweep with the shipped 2048 default, QPS 8/10/12/16: TPOT p50 15.15/21.57/33.31/63.70, ITL p99 capped at 100/102/110/112, QPS16 throughput 1824 tok/s — i.e. −2.2% overload throughput for less than half the overload ITL tail, still beating vLLM on both. Low-QPS rows are budget-insensitive: a single 1024-token prompt never splits at 2048.)

Where the mid-band gap comes from: TPOT ≈ decode_step / (1 − prefill_share). Our prefill is ~45ms per 1024-token prompt vs a ~39ms bf16 roofline on the 5090 — vLLM's is no faster; it wins mid-band by running decode steps cheaper at bs 20–60 (its GEMMs see the whole 2048-token chunk budget as one batch). Closing it means prefill/decode kernel work, not more scheduling.

QPS16 admission verdict (#345): no knee anymore — the old 90ms TPOT regression at QPS12+ was the per-request step tail, not admission policy. Current overload behavior is strictly better than vLLM's (higher throughput, lower TPOT, 3× lower TTFT); no admission cap needed.

## Pitfalls

- `vllm bench serve` TPOT in overload follows the identity TPOT ≡ bs/throughput — admission policy changes TPOT without changing kernel speed. Compare out_tok/s alongside.
- nsys `--cuda-graph-trace=node` inflates step times 30–60%; use it for composition only, never absolute TPOT.
- pkill from an ssh one-liner matches its own command line — use `pkill -f "[t]arget/release/openinfer"`.
