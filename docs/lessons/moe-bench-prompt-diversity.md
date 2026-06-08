# MoE Decode Benchmarks Need Diverse Prompts

> **TL;DR:** MoE decode TPOT depends on **token content**, because token content decides expert routing and routing decides grouped-GEMM tile efficiency. Benchmarking a concurrent batch with identical prompts under-measures decode TPOT by **~7–15%** (measured on Kimi-K2, below) — not the ~30% an earlier version of this note claimed. Bench any MoE+EP model with **seeded distinct per-request prompts** (`bench_serving --distinct-prompts`). The whole effect is the **Marlin expert GEMM** (per-launch time ~doubles K=1→K=64); the DeepEP all-to-all does **not** grow with diversity — so the lever is grouped-expert GEMM efficiency, **not** all-to-all overlap (#228). This is evidence, not inference: a `--distinct-prompts` sweep + an nsys kernel diff, both reproduced 2026-06 on 8×H200.

Companion to [moe-dplb-decode-imbalance.md](moe-dplb-decode-imbalance.md) (routing imbalance as a *serving* problem) and the profiling-discipline lesson [profile-diff-before-blaming-transport.md](profile-diff-before-blaming-transport.md), which came out of the same #225 misfire.

## The principle

For a **dense** model, decode cost is a function of *shape* alone — sequence length and batch size — so any prompt of the right length measures the truth; token content is a don't-care.

**MoE breaks this.** Each decoded token selects top-k experts (Kimi-K2: top-8 of 384, EP8 → 48 experts/rank). With identical prompts + greedy decode, all concurrent streams stay byte-identical and route to the *same* narrow set of experts every step; with diverse prompts they spread across many experts. So for MoE, decode cost is a function of `(shape, token distribution)`, and the token distribution is a **load parameter** a benchmark must reproduce.

## Evidence 1 — the sweep: TPOT vs routing diversity

`bench_serving --distinct-prompts K` tiles K unique random prompts across a 64-way in-process batch (K=1 → all identical, K=64 → all distinct). **Transport is held constant — in-process, no HTTP — so K is the only variable.** 8×H200, TP1/DP8/EP8, bs64, DeepEP graph, K2.6:

| K (distinct/64) | first-decode-step p50 | steady-TPOT p50 | steady-TPOT p99 |
| ---: | ---: | ---: | ---: |
| 1 (identical) | **26.11** | 28.65 | 32.72 |
| 2 | 25.96 | 29.23 | 33.89 |
| 4 | 26.00 | 30.34 | 34.58 |
| 8 | 25.82 | 30.26 | 34.66 |
| 16 | 26.26 | 31.04 | 33.70 |
| 32 | 27.12 | 31.74 | 35.65 |
| 64 (diverse) | **28.05** | **32.62** | 35.72 |

Two metrics, on purpose. **first-decode-step** runs at uniform minimal context, so it isolates routing with zero context-growth confound: K=1→64 moves it **26.1 → 28.1 ms = +7%**. **steady-TPOT** averages all 128 decode steps (context grows identically across K, so the K-delta is still routing): **28.7 → 32.6 ms = +14%**. The effect is **non-linear** — flat below K≈16, emerging at K=32/64 — because 64 tokens × top-8 only spread across many *distinct* experts once enough prompts differ. (The steady effect exceeds first-step because greedy+identical streams stay degenerate forever while diverse streams diverge *further* each step, widening the breadth gap over the run.)

So routing diversity is worth **~7–15%** of decode TPOT — real, but not the ~30% originally claimed. **The original "30%" was a metric-mismatch artifact: it compared the identical-prompt *first-decode-step* (26 ms) against the diverse-prompt *steady-TPOT* (32 ms)**, folding a metric change and context growth into the "routing" bucket. (See the companion profiling lesson.)

## Evidence 2 — the kernel diff: it's the expert GEMM, not the all-to-all

nsys `cuda_gpu_kern_sum` over the decode capture, K=1 vs K=64, same in-process binary (so any kernel-time delta is GPU compute/comm, never transport):

| bucket | K=1 (µs) | K=64 (µs) | Δ | share of +Δ |
| --- | ---: | ---: | ---: | ---: |
| **Marlin expert GEMM** | 1 490 140 | **2 905 005** | **+94.9%** | **+118%** |
| MoE dispatch (a2a) | 921 977 | 903 323 | −2.0% | −2% |
| MoE combine (a2a) | 1 429 153 | 1 223 124 | −14.4% | −17% |
| dense GEMM | 3 160 744 | 3 158 835 | −0.1% | 0% |
| MLA attention | 216 693 | 216 622 | 0.0% | 0% |
| **TOTAL GPU** | 7 685 201 | 8 884 677 | **+15.6%** | |

The entire +15.6% is the **Marlin INT4 expert GEMM, which nearly doubles**. Per-kernel: **identical 32 640 launches in both K**, average **45.7 µs → 89.0 µs** (median 40.3 → 79.3). Same launch count, ~2× per launch — the textbook grouped-GEMM tile signature: narrow routing → few experts × many rows → fat efficient tiles; broad routing → many experts × 1–2 rows → thin, weight-load-bound tiles wasting the Marlin tile. Same FLOPs, ~2× wall time.

The DeepEP **all-to-all does not grow** with diversity — dispatch flat, combine *down* 14%. The hypothesis that "diverse prompts move more all-to-all data" is **refuted**; DeepEP buffers are worst-case-sized, so combine is ~fixed. **Lever: grouped-expert GEMM efficiency (thin-tile Marlin), not #228 (a2a overlap).** #228 still helps the absolute floor (a2a is ~30% of GPU time), but it is *not* what makes diverse cost more than identical.

## Transport is still ≈0

Like-for-like (diverse, steady): in-process **32.6 ms** vs HTTP `vllm bench serve` **33.9 ms** — +4%, within the random dataset's richer-diversity margin. The serving bridge / scheduler are not the bottleneck (host 0.1 ms/step, ranks balanced 8/8; see the profiling-discipline lesson). The headline #225 "+51% HTTP" never existed at like-for-like.

## How to apply

- **Any MoE serving benchmark uses seeded distinct per-request prompts.** `bench_serving` does this by default when `--prompt-len` is set (one random prompt per request); `--distinct-prompts K` controls breadth for a routing sweep. Single-stream paths (matrix/curve/snapshot) keep the deterministic prompt for baseline stability.
- **Label single-prompt concurrent batches as kernel microbenches** — they under-report decode TPOT by ~15% and pin routing to its cheapest corner.
- **The trap is MoE-specific.** Kimi-K2 and the DeepSeek-V2-Lite/V4 EP lines are exposed; Qwen3 / Qwen3.5 (dense) are immune (no routing → content-independent decode, the flat dense-GEMM/MLA rows above).
- **Optimization lever:** grouped-expert GEMM tile efficiency under broad routing, not the all-to-all.

## Reproduce

```
# sweep (per-K model reload; ~1 min each)
for K in 1 2 4 8 16 32 64; do
  bench_serving --model-path <K2.6> --tp-size 1 --dp-size 8 --ep-backend deepep --cuda-graph true --format json \
    request --prompt-len 1 --output-len 128 --concurrency 64 --distinct-prompts $K --warmup 1 --iters 3 --seed 42
done
# kernel diff (nsys, K=1 vs K=64): cuda_gpu_kern_sum:base, bucket by Marlin / dispatch / combine / gemm / MLA
```
