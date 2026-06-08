# Kimi-K2 Roadmap

> **TL;DR:** Kimi-K2 decode leads vLLM on the same 8×H200 hardware (active TP1+DP8+EP8 **DeepEP** line, bs64 in-process graph TPOT **26.30 ms** p50 / **30.50 ms** p99 — the −12% win from the #227 CUDA-graph capture). The M1 serving-contract wave has landed — sampling honor-or-reject (#237), EOS (#238), KV admission (#239), prefix cache (#230) — and the git-versioned accuracy gate (#223) is green teacher-forced. The live frontier is **serving performance**: the HTTP path adds **+51% TPOT** over in-process (#225, server-side, *not* a TCP_NODELAY/Nagle bug), the MoE all-to-all runs strictly serial at ~28% of GPU time (#228), and short-prompt TTFT is still 4.5×/31× behind vLLM (#224). See **Measured serving profile** below for first-hand numbers. Re-verified 2026-06-08 on 8×H200.
>
> **Last touched:** 2026-06

Status ledgers: [tp1-dp8-ep8-performance.md](tp1-dp8-ep8-performance.md) (active perf line), [deepep-migration.md](deepep-migration.md) (DeepEP backend + A/B), [accuracy-gate.md](accuracy-gate.md) (the committed gate), [optimization.md](optimization.md) (model card + TP8 history). This doc owns the cross-cutting plan: what's missing, what blocks what, and in which order.

## Measured serving profile (2026-06-08, 8×H200, DeepEP graph path)

First-hand bs64 decode profile on the active TP1+DP8+EP8 DeepEP line (`--cuda-graph true`, prompt 1 / output 128 / concurrency 64). Confirms the published graph numbers and sharpens three open issues.

**Decode TPOT (in-process `bench_serving`, full-bucket graph replay):** p50 **26.30 ms** / p95 27.40 / p99 **30.50 ms**; in-process TTFT p50 56.4 ms. Matches the published 26.03 ms graph figure — the −12% graph win over eager (29.6 ms) holds.

**nsys decode composition** (graph-replay, 8-rank aggregate; `--cuda-graph-trace=node`, so read *proportions*, not absolute times):

| Kernel family | % GPU | median | note |
| --- | ---: | ---: | --- |
| `combine_impl` (DeepEP combine a2a) | 18.7% | 87.9 µs | **#228 target** — 87.9 µs vs ~37 µs NVLink theory (2.4×); systematic tail p99/p50 2.7× |
| `gemmSN_TN` (dense GEMV) | 18.4% | 91.9 µs | rock-steady (σ 0.7 µs) |
| `Marlin` (INT4 expert GEMM, ×2/layer) | 18.2% | 39.7 µs | route-imbalance tail p99/p50 3.2× |
| `nvjet_*` cuBLASLt GEMMs (MLA/proj) | ~22% | — | the #204 skinny-GEMM picks |
| `dispatch_impl` (DeepEP dispatch a2a) | 9.1% | 14.7 µs | rank-skew tail: p99 311 µs, **max 15 ms** one-off (p99/p50 21×) |
| `BatchDecode…MLA` (attention) | 3.5% | 16.5 µs | small at 2k ctx; grows with KV (#231) |

**Host side is clean:** `cuStreamSynchronize` 65% + `cuGraphLaunch` 34%, `cudaLaunchKernel` < 0.5%. Decode is **GPU-bound — the per-step host enqueue overhead is already eliminated by the #227 graph capture.** The remaining decode lever is GPU kernel time: the MoE all-to-all (**dispatch + combine ≈ 28% of GPU time, run strictly serial dispatch→Marlin→combine with no overlap → #228**), not host work.

**HTTP vs in-process serving overhead (#225)** — `vllm bench serve` (async client, random in 1 / out 128, 256 prompts, max-concurrency 64, ignore-eos, greedy) against the OpenAI endpoint, cross-checked with a second client:

| | in-process | HTTP (`vllm bench serve`) | Δ |
| --- | ---: | ---: | ---: |
| TPOT p50 | 26.30 ms | **39.67 ms** | **+13.4 ms (+51%)** |
| TPOT p99 | 30.50 ms | 39.86 ms | — |
| TTFT p50 | 56.4 ms | 52.94 ms | ~none |
| ITL p50 | — | 40.06 ms | tight (p99 39.86) |

The +51% TPOT gap is **real and server-side** (an independent threaded client agrees at 41.2 ms — not a client artifact). It is **not** the Qwen3 Nagle trap: the vendored vLLM Rust frontend already sets `TCP_NODELAY` on every accepted connection (`vllm-server` `lib.rs:153`). TTFT has **no** HTTP penalty (an earlier 504 ms reading was a Python-threaded-client GIL artifact, discarded). The tight ~40 ms ITL pin points to a **fixed per-decode-step host cost in the serving bridge** (engine → IPC output socket → vLLM SSE frontend over 64 concurrent streams) that serialises with — rather than overlaps — the 26 ms GPU step. Exact attribution needs a server-side trace (`pegainfer_http_trace` / nsys of the server process); that is the #225 next step, and a bigger user-facing lever than the issue's stale "~33%".

**Correctness re-run (#223 gate, current DeepEP + graph checkout):** teacher-forced sweep **0 violations** (384 positions, 375 exact = 97.7%, |Δlogprob| mean 0.032 / p99 0.324) — the authoritative claim holds. bs=1 greedy parity **0 violations**. Concurrent free-greedy: **1 violation at `translation` pos 31** (picked a token vLLM scores 1.375 nat below argmax) — the known **#286** final-position concurrent-batch near-tie, reproduced identically to the migration record, not a regression. Det contract (#293) green. Net: the strict gate is green; the only red is the tracked #286 concurrent marginal.

## Capability contract (current state)

| Capability | State | Evidence / PR |
| --- | --- | --- |
| EOS / stop-token | ✓ both paths, honors `ignore_eos` | #238; `runner/scheduler{,/dp}.rs` |
| Sampling (temp/top_k/top_p) | ✓ honor-or-reject, batched FlashInfer (TP1/DP8); TP8 rejects non-greedy | #237/#285 |
| Prompt-length guard | ✓ admission rejects `prompt+max_tokens−1 > 8192` | #239/#292; `worker.rs:65` |
| KV admission | ✓ full-lifetime paged `BlockPool` budget | #239/#292 |
| Prefix cache | ✓ gather ckv→decompress, kpe post-RoPE; warm TTFT 4.8× | #230/#292 |
| Continuous batching | ✓ TP1/DP8 (`DpCoordinator`); TP8/DP1 still batch-then-drain | `runner/scheduler/dp.rs` |
| Accuracy gate in git | ✓ teacher-forced golden + committed K2.6 fixture; green 2026-06-08 | #223/#269; `tests/vllm_golden_gate.rs` |
| logprobs | ✓ exact on TP1/DP8; ✗ TP8 (sharded vocab) | #236; `worker/state.rs:410` |
| echo | rejected before forward (honor-or-reject) | #236; `scheduler/lifecycle.rs:125` |
| CUDA graph decode | ✓ DeepEP full-bucket capture, −12% TPOT | #227/#298 |
| Bench-regression snapshot | ✓ `bench_snapshots/h200/kimi-k2.6.json` | #232 |
| Lint gate (kernels + comm) | ✓ scoped `-D warnings` hook | #233 |
| LoRA | N/A — server rejects cleanly | `pegainfer-server/src/main.rs` |

## Claim boundaries

- **TP1+DP8+EP8 DeepEP (active line):** bs64 in-process graph TPOT p50/p99 `26.30/30.50 ms` (eager `29.6 ms`), measured 2026-06-08 on 8×H200. Service-level H20 history (`1336 tok/s`, `47.3 ms`) lives in [tp1-dp8-ep8-performance.md](tp1-dp8-ep8-performance.md). Greedy parity is gated teacher-forced (#223); free-greedy concurrent has the tracked #286 marginal.
- **TP8+EP8 NCCL:** reference/history. The backend is now a hard architectural split — `bringup.rs` enforces TP8→NCCL and TP1/DP8→DeepEP; there is no PPLX fallback (PPLX deleted in #298/#301).
- **HTTP serving:** +51% TPOT over in-process (#225), server-side, *not* TCP_NODELAY. TTFT has no HTTP penalty.
- **TTFT:** short-prompt still ~4.5×/31× behind vLLM p50/p99 (#224).
- **KV ceiling:** `prompt + max_tokens − 1 ≤ 8192` (`worker.rs:65`). No claim of long-context decode correctness (>8k), TP8 non-greedy / logprobs, or multi-node.

## Sequencing — what blocks what

```
M1 (serving contract) + M2 (accuracy gate) ── DONE → unblock all decode/serving opt + K2.6 (now the active model)
#286 concurrent-decode corruption ─→ trustworthy free-greedy gate ─→ #300 graph-replay numerics gate
#225 HTTP overhead ─ (independent, server-side bridge)
#228 MoE a2a overlap ─→ TP1/DP8 decode TPOT
shared block table (done #292) ─→ MLA split-KV (#231 long-context) ─→ DP prefix-affinity routing (#229)
```

## Roadmap

### Shipped — M1 serving contract + M2 accuracy gate (2026-06)

Closed: sampling honor-or-reject (#237/#285), EOS/stop (#238), KV admission + paged pool (#239/#292), prefix cache (#230/#292), accuracy gate in git (#223/#269), CUDA-graph decode on DeepEP (#227/#298), bench snapshot (#232), lint gate (#233), dead-code + doc sweeps (#234/#235). The PPLX→DeepEP migration (#298/#301) replaced the MoE EP backend underneath all of this. **K2.6 is now the live model** — the gate fixture is K2.6-vLLM-golden, so the old "#16 K2.6 readiness" is effectively satisfied.

### Open — correctness debt (finish what M1/M2 started)

- **#222 `tests/` surface.** `vllm_golden_gate.rs` now exists (teacher-forced sweep + bs1/concurrent parity + det + prefix-cache), but the M1 features it should protect have **no engine-through IT**: EOS (#238), sampling (#237), admission (#239) are guarded only by in-src unit tests. No CPU/single-GPU scheduler-robustness test like qwen3's. Stale `PPLX` strings linger at lines 19/35/309/519.
- **#236 logprobs / echo.** ~85% done — exact logprobs on TP1/DP8 (`eb4255c`), echo rejected before forward (`28f3749`). Remaining: TP8 logprobs (sharded-vocab logsumexp) and echo-via-prefill-logits (prefill runs last-token-only `lm_head`).
- **#286 concurrent-decode corruption.** The real bug behind the "near-tie" title: teacher-forced + bs1 are clean, but concurrent free-greedy mispicks at final positions (`translation` pos 31, 1.4–7.4 nat below argmax, reproduced 2026-06-08), on **both** backends. Split into the cheap near-tie exemption in `check_pick()` and the actual corruption hunt (MLA reduction / combine arrival order / final-step batch recompose).
- **#300 graph-replay numerics gate.** The prod graph path (full-bucket replay) has **no** numerics gate — `vllm_golden_gate` runs `enable_cuda_graph: false` and peaks at ~2 active/rank. Add a graph-enabled, concurrent, full-bucket variant comparing replay vs eager to the bf16 ULP floor.
- **#293 det wobble.** Tolerance gate landed (`b51a53b`, 0.25 nat); per-layer same-page-vs-different-page root-cause isolation still open. Lowest of the band — formalize or fix.

### Next — serving performance

- **#225 HTTP overhead.** +51% TPOT measured (39.67 vs 26.30 ms). *Not* TCP_NODELAY (already set in the vendored frontend). Attribute the fixed per-step serving-bridge cost (engine→IPC output socket→vLLM SSE for 64 streams) with `pegainfer_http_trace` / a server-process nsys; the bridge cost serialises with the GPU step.
- **#224 TTFT.** Decompose short-prompt ~2 s (4.5×/31× vs vLLM): embedding / MLA prefill / MoE prefill / first-collective drain / per-layer scratch alloc. Prime suspects from [deepep-migration.md](deepep-migration.md): ~5 prefill buffers/layer + per-layer host spins. Add a TTFT bench snapshot.
- **#228 MoE a2a overlap.** dispatch + combine = ~28% of GPU time, strictly serial; combine `87.9 µs` vs `~37 µs` NVLink theory. Double-buffer layer N+1 dispatch against layer N combine (recv/combined buffers are persistent and worst-case-sized). ⚠️ Issue body still says "PPLX path" — reframe to DeepEP.
- **#229 DP8 routing.** Greedy free-slot pick, duplicated single-/multi-token; unify into `DpLoadBalancer` with length-skew + (later) prefix-affinity awareness; evaluate on mixed-arrival.

### Later — structural

- **#226 TP8 decision.** Already architecturally NCCL-only (`bringup.rs` enforces it). Just *formalize* supported-vs-reference in docs, and close the two gaps: TP8 is not exercised by the accuracy gate, and #204's cuBLASLt picks are TP1-shaped only (TP8 falls back to old GEMMs).
- **#231 long-context.** Ceiling is now 8192 (#292), but no correctness harness at 4K/32K/128K and `partition_kv=false` (one CTA scans full KV serially, `kimi_mla.cu`). Harness first (correctness gate), then MLA split-KV.
- **Multi-node DP/EP** per [dp-design.md](dp-design.md) §10.

## Cleanup ledger

- ✓ Done: lint gate (#233), dead expert-major/CUTLASS cluster + `weight_shape` tensor (#234), doc refresh + consolidation (#235, lessons lifted to [kimi-bringup-numerics.md](../../lessons/kimi-bringup-numerics.md)).
- Open: `KERNELS.md` stale rows (references a deleted `.cu` + two zero-reference ops); stale `PPLX` strings in `tests/vllm_golden_gate.rs` (lines 19/35/309/519) and the #228 issue body.

## Done criteria

This roadmap is healthy when:

- ✓ a temperature/top_p request is sampled or explicitly rejected, never silently greedy; generation stops at EOS (#237/#238).
- ✓ a fresh clone + an 8×H200 node re-runs the accuracy gate from committed code and the K2.6 fixture alone (#223).
- TTFT p50/p99 has a measured decomposition, a gate, and is within striking distance of the vLLM class (#224, open).
- the HTTP path's per-token cost is attributed and closed toward the in-process number (#225, open).
- the free-greedy gate is trustworthy at full concurrent batch — #286 fixed-or-formally-bounded and the graph-replay path gated (#300).
- `docs/models/kimi-k2/` describes the engine that exists, not the bring-up that happened.
