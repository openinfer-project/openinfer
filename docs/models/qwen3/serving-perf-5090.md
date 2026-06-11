# Qwen3-4B serving perf vs vLLM on RTX 5090

**TL;DR**: openinfer beats vLLM 0.22.1 at QPS1 on every metric (TPOT 6.74 vs 6.86, TTFT 49 vs 56), beats it outright in overload (QPS16: 61.7ms TPOT / 1865 tok/s vs 78.6ms / 1674), beats TTFT everywhere, and matches it up to QPS4. Five fixes got here: batched step tail (#345), chunked prefill (default 2048), dropping bs8/16 buckets (cuBLAS split-K hole), **linking against cuBLAS ≥ 13** (12.9 has a 50–100% GEMM cliff at N=1025 — build with `CUDA_HOME=/usr/local/cuda-13.x`), and decode-path tuning (cublasLt per-shape algos, split-KV to bs4 @ 64-token chunks, two-stage argmax). Remaining gap: mid-band TPOT (QPS 8–12, 1.1–1.6× vLLM) — the unified step's split attention section (+4.5ms vs pure prefill) is the next lever.

Last touched: 2026-06

## Setup

- RTX 5090, `/data/Qwen3-4B`, dev checkout `~/develop/xingming/pegainfer` on the 5090 box.
- Workload: `vllm bench serve`, random dataset, in=1024 / out=128, Poisson arrivals, seed 42, 60s per QPS point (`tools/bench/qps_sweep.sh`).
- Reference: vLLM 0.22.1, `--max-model-len 8192 --no-enable-prefix-caching`, defaults otherwise (max_num_seqs=256, max_num_batched_tokens=2048 chunked prefill, decode-priority). Verified from v0.22.1 source: vLLM does **not** lock batch size at 128; observed ~129 concurrency is KV-capacity-bound.
- Results: 5090 `/data/xingming/bench/20260611-tune/{vllm-ref,oi-chunk512,oi-chunk1024,oi-chunk2048,oi-final,oi-default2048}/`.

## Fixed: cuBLAS 12.9 GEMM cliff at N=1025 — deploy with CUDA ≥ 13

The unified (prefill+decode) step ran 60.8ms where pure prefill of the same tokens took 42.3ms. nsys attribution: +14.6ms was GEMM alone. Root cause: the unified step's GEMMs run at N=1024+decode_bs, and **cuBLAS 12.9's kernel selection collapses at N=1025** (standalone GemmEx microbench, bf16/COMPUTE_32F: gate 9728×2560 234→347µs, oproj 2560×4096 94→185µs, down 2560×9728 223→299µs going from N=1024 to N=1025; qproj immune). cuBLAS 13.2 has no cliff.

The trap: `openinfer-kernels/build.rs` derives both nvcc and the cublas link path from `CUDA_HOME` → `CUDA_PATH` → `/usr/local/cuda`. On a box where `/usr/local/cuda` symlinks 12.9, PATH exports do nothing — the server silently links `libcublas.so.12`. **Build with `CUDA_HOME=/usr/local/cuda-13.x`** and verify with `ldd target/release/openinfer | grep cublas`.

After the CUDA 13.1 rebuild: unified step 60.8 → 47.1ms, QPS1 TTFT 64 → 50ms (arrivals landing mid-decode prefill faster). This also invalidated the doc's earlier claim that mid-band was "bounded by prefill near the bf16 roofline" — the measured 45ms prefill included the 12.9 cliff.

## Fixed: QPS1 decode-path tuning (PR #366)

At QPS1 the TPOT integrand is the decode step itself (43% of steps run bs≥2 from Poisson overlap). Three fixes, decode step (ctx1024, p50): bs1 6.51→5.96ms, bs2 7.02→6.53, bs4 8.11→6.72:

- **cublasLt per-shape algo tuning**: cuBLAS's default heuristic leaves 4–6% bandwidth on the table for every small-N decode GEMM (in-graph kernel times match Lt heuristic[0] exactly; the best candidate is 1.40–1.52 TB/s vs default's 1.28–1.48). `gemm_lt_tune` times all candidates at executor startup — on real weights rotated across all 36 layers so the loop stays L2-cold — and caches the winner per (M,N,K); N≤4 GEMMs consult the cache, untuned shapes fall back to the old paths.
- **Split-KV decode attention bs≤2 → bs≤4, chunk 256 → 64 tokens**: the non-partitioned kernel runs one CTA per request×head = 8 CTAs at bs1 on 170 SMs. 64-token chunks measured fastest (32 is past the merge-overhead knee).
- **Two-stage batched greedy argmax**: tile-parallel partials + per-row finalize; the single-block-per-row kernel cost 91–191µs/step over the 151936 vocab.

## #345 root cause: per-request step tail

nsys at bs≈120 (`--cuda-graph-trace=node`, cudaProfilerApi capture): prefill/unified steps spent ~57ms (~25% of GPU time) in a per-request loop of extract_vec → single-row RMSNorm → lm_head GEMV (0.48ms each) → single-row sample → blocking 1KB pageable DtoH. Linear in batch size, so it presented as "batch doesn't scale" after #341 raised the CUDA-graph buckets to 256. Decode kernels themselves scale fine and beat vLLM ≥bs32 (bs128 step: 13.7ms vs vLLM's 28.8ms implied).

Fix (`perf/qwen3-batched-step-tail`): gather last-token rows → batched RMSNorm → one GEMM → `select_batch_tokens_into` → one DtoH. Same audit found the pattern in other model crates — tracked in #353 (qwen35 is worst: full-vocab blocking DtoH per request per decode step).

## Chunked prefill (`--max-prefill-tokens`)

Prompts are split across steps under a per-step forwarded-token budget; mid-prompt requests live in a scheduler `prefilling` queue, the executor clamps each chunk at the request's `kv_position` (prefix-cache hits compose), non-final chunks apply KV without emitting a token (kvbm `apply_prefill(None)`). Echo never splits. Admission reserves KV + decode slots for mid-prefill requests; KV prefetch respects a reserve floor so it can't steal a queued chunk's blocks.

- **The budget is an ITL-tail vs throughput knob.** 512 matches or beats vLLM's ITL p99 everywhere but costs 15% of the overload ceiling. 2048+ keeps the ceiling and wins overload outright (+8.6% throughput, −19% TPOT, TTFT p50 2.0s vs 4.3s at QPS16).
- **Default is 2048** (matches vLLM's `max_num_batched_tokens`): vs 8192 it costs ~2.5% overload throughput but halves overload ITL p99 (112 vs 257ms). Single prompts ≤2048 tokens never split. Latency-sensitive deployments: `--max-prefill-tokens 512`.
- **Chunking does NOT close the mid-band TPOT gap** — smaller chunks make TPOT *worse* (more steps; decode rows pay the prefill toll more often).

## Fixed: bs8–16 decode-step anomaly (cuBLAS skips split-K)

Fine sweep (ctx1024, CUDA graph, p50 ms): bs4 8.11, **bs8 9.42, bs12 9.47, bs16 9.52**, bs20 8.51 — non-monotonic, exactly the mid-band serving concurrency. nsys diff: bs20's GEMMs use split-K (GEMM med 16.8µs) while bs8's don't (med 23.2µs) — cuBLAS's heuristic leaves ~20 CTAs on 170 SMs for batch∈[8,16]. Re-verified on cuBLAS 13.2: still present (bs8/16 steps 9.2/9.3ms vs bs20 7.9ms).

Fix: removed buckets 8/16 from `BATCH_BUCKETS` so batches 5–19 pad up to 20 and get the split-K configs; padded attention rows are near-free.

## Current sweep (all fixes, cuBLAS 13.2; TPOT mean / ITL p99 ms, TTFT mean ms)

| QPS | openinfer | vLLM 0.22.1 | TTFT oi / vllm |
|---|---|---|---|
| 1 | **6.74** / 8.4 | 6.86 / 12 | **49** / 56 |
| 8 | 13.74 / 88 | 12.09 / 47 | 71 / 68 |
| 10 | 19.63 / 94 | 14.62 / 82 | 87 / 79 |
| 12 | 33.14 / 104 | 20.49 / 103 | 118 / 123 |
| 16 | **61.65** / 257 (**1865 tok/s**) | 78.55 / 119 (1674) | **1355** / 4279 |

(QPS 2/4 unchanged from the pre-Lt sweep: 8.10 / **8.63** vs vLLM 7.42 / 8.89 — openinfer already ahead at 4. QPS16 row predates the cuBLAS-13 rebuild; overload is decode-throughput-bound so the cliff barely moves it.)

Mid-band decomposition (STEPLOG, QPS1→generalizes): inter-step host gap is 12µs — scheduling is free. TPOT = token-weighted decode-step mix + unified-step rides (~47ms each). The remaining mid-band lever is the unified step's split attention section: prefill+decode attention run as separate kernels with their own rope/scatter, +4.5ms vs pure prefill's fused path.

QPS16 admission verdict (#345): no knee anymore — the old 90ms TPOT regression at QPS12+ was the per-request step tail, not admission policy. Current overload behavior is strictly better than vLLM's; no admission cap needed.

## Pitfalls

- **Stale `/usr/local/cuda` symlink silently links cuBLAS 12** — see the N=1025 cliff section. Always `CUDA_HOME=/usr/local/cuda-13.x cargo build` and check `ldd`.
- Microbenching GEMMs with a single weight buffer measures L2-hot timings (96MB L2 on GB202 holds entire decode weights) — rotate ≥4 weight copies or the ranking is fiction. Same for algo tuning, hence the all-layer rotation in `gemm_lt_tune`.
- `vllm bench serve` TPOT in overload follows the identity TPOT ≡ bs/throughput — admission policy changes TPOT without changing kernel speed. Compare out_tok/s alongside.
- nsys `--cuda-graph-trace=node` inflates step times 30–60%; use it for composition only, never absolute TPOT.
- pkill from an ssh one-liner matches its own command line — use `pkill -f "[t]arget/release/openinfer"`.
