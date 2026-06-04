# Kimi-K2 Roadmap

> **TL;DR:** Kimi-K2 decode performance is ahead of vLLM on the same H20 ×8 hardware (TP1+DP8+EP8 service bs64: output `1336 tok/s` vs vLLM `594 tok/s`, TPOT p50 `47.3ms` vs `107.2ms`), but the crate is far behind Qwen3-4B on the serving contract and correctness surface: every request runs greedy regardless of sampling params, never stops at EOS, prompts over 2048 tokens overrun the KV arena unchecked, and no accuracy gate is reproducible from a fresh clone. The roadmap sequence is: serving-contract correctness + a git-versioned accuracy gate first, then the TTFT/prefill and HTTP-overhead perf gaps, then the continuous-batching→KV-lifecycle→prefix-cache chain and the PPLX/decode-throughput chain. Findings verified 2026-06-04 against `6ee9247`.
>
> **Last touched:** 2026-06

Status ledgers: [tp1-dp8-ep8-performance.md](tp1-dp8-ep8-performance.md) (active perf line), [optimization.md](optimization.md) (model card + TP8 history). This doc owns the cross-cutting plan: what's missing, what blocks what, and in which order.

## Capability contract (Qwen3-4B as the maturity bar)

| Capability | Qwen3-4B | Kimi-K2 | Evidence |
| --- | --- | --- | --- |
| EOS / stop-token detection | ✓ per-step `is_stop_token` | ✗ always `FinishReason::Length`, generates exactly `max_tokens` | no `eos` handling in `pegainfer-kimi-k2/src`; qwen3 `scheduler/resolve.rs:50` |
| Sampling params | ✓ FlashInfer greedy + non-greedy | ✗ `req.params` never read — temperature/top_k/top_p **silently ignored**, always argmax | `runner/worker/forward.rs:113` top1 only |
| Prompt-length guard | ✓ KV-budget admission rejects impossible requests | ✗ 2048-token arena cap (`worker.rs:60-61`), `append_position` unchecked → silent overrun | `runner/scheduler.rs:148-151` |
| KV admission control | ✓ full-lifetime budget, deferral, rejection (issue #85 fix) | ✗ slot-count only on both paths | qwen3 `scheduler.rs:478-510` |
| Continuous batching | ✓ | TP1/DP8: ✓ (`DpCoordinator` per-step admission, waiting queue); TP8/DP1: ✗ strict batch-then-drain | `runner/scheduler/dp.rs:168,189-236` vs `runner/scheduler.rs:140-207` |
| Prefix cache | ✓ kvbm full-block matching (PR #216) | ✗ fixed per-slot arena, no block table / free pool | `runner/worker/cache.rs:391-396` |
| Accuracy gate in git | ✓ `tests/hf_golden_gate.rs` + committed golden | ✗ no `tests/` dir (only model crate without one); fixtures + A/B harness live off-repo | see Accuracy section |
| logprobs / echo | ✓ | ✗ `logprob: None` always, `PromptTokens` never emitted | `runner/scheduler/dp.rs:345` |
| Bench-regression snapshot | ✓ `bench_snapshots/*/qwen3-4b.json` | ✗ no kimi snapshot, no h20 dir | `docs/conventions/bench-regression.md` |
| CUDA Graph state | shared `pegainfer-core` type | ✓ same shared type + multi-rank synchronized capture | `runner/worker.rs:17` |
| LoRA | ✓ | intentionally N/A, server rejects cleanly | `pegainfer-server/src/main.rs:101-103` |

## Claim boundaries

- **TP1+DP8+EP8 PPLX (active line):** service bs64 `256/256` success, output `1336 tok/s`, TPOT p50/p99 `47.3/47.7ms` — beats the vLLM out-of-box sustained baseline (`594 tok/s` / `107.2ms`, itself depressed by vLLM's DPLB/CUDA-graph bucket cliff; vLLM's balanced pinned capability is ~`48-50ms`). Greedy-only; token parity vs a reference trace is **not yet a gate** on this shape.
- **TP8+EP8 NCCL (graph path):** bs4 TPOT `14.39ms` synthetic. Wave-batched, reference/history path.
- **TTFT is the open perf gap:** HTTP decode-heavy bs4 TTFT p50 `313ms` / p99 `4240ms` vs vLLM `69.6/135.4ms` (4.5×/31× worse). Short-prompt streaming TTFT ~`2.0s`.
- **HTTP vs in-process:** `19.13ms` vs `14.39ms` TPOT at bs4 — ~33% serving overhead, unattributed.
- No claim of: non-greedy sampling, EOS-terminated generation, >2048-token prompts, long-context decode correctness, prefix reuse, multi-node.

## Sequencing — what blocks what

```
Correctness milestone (M1) ──┬─→ any further kernel/decode opt (needs regression gate)
Accuracy gate in git (M2) ───┘    K2.6 weight swap validation
Lint-gate fix ───→ dead-code removal (cleanup ledger)
TP8 continuous batching ─→ shared block table + free pool ─→ MLA physical backend
                                ─→ nonzero-position prefill ─→ prefix cache ─→ DP prefix-affinity routing
PPLX graph capturability / MoE pipelining ─→ TP1+DP8 decode throughput targets
```

## Roadmap

### Now — M1: serving-contract correctness

The engine is fast but does not honor the OpenAI contract it serves. All items are crash-early-or-honor, none are perf work:

1. **EOS/stop-token handling.** Parse `eos_token_id`/`stop_token_ids` in `config.rs`, check per decode step on both scheduler paths, honor `ignore_eos`, emit `FinishReason::Stop`. Frontend mapping already exists.
2. **Sampling params: honor or reject.** `req.params` is never read. Either route non-greedy rows through the shared FlashInfer sampling ops, or reject non-greedy requests explicitly. Audit the full OpenAI param surface (temperature/top_k/top_p, n, seed, penalties, stop strings) — each one: honored / rejected / documented-ignored. Silent-wrong is the only forbidden state.
3. **Prompt-length guard.** Reject prompts whose `prompt + max_tokens` exceeds the 2048-token per-slot arena instead of overrunning KV pages.
4. **KV admission + abort-on-disconnect.** Port the qwen3 admission pattern (budget, deferral, rejection) to both paths; wire disconnect-abort when the server-wide #215 lands.
5. **`tests/` directory.** Scheduler-robustness ITs (CPU-runnable: admission, coalesce, slot-lifecycle edges) so the above gets regression coverage. Kimi-K2 is the only model crate without integration tests.

### Now — M2: accuracy gate in git

Today no accuracy claim is reproducible from a fresh clone: the vLLM top-20 fixture gate's reference *and* candidate fixtures live only on the H20 box, the PegaInfer-side candidate dumper was retired in PR #158, and the prefill-logits A/B harness that gated PR #204's kernel picks was never committed. The HF-loading recipe for K2.5 (trust_remote_code + vision-tower stub) is already solved in `pegainfer-kernels/tools/kimi_k2/hf_logits_reference.py`.

1. Define the gate invariant per `docs/subsystems/correctness/logits-golden-gate.md`: teacher-forced per-position logprob delta vs a committed golden — not top-20 membership.
2. Commit the reference fixture (top-K logprobs over a few sequences — small) under `test_data/`, plus a maintained candidate-dump entry point in the crate.
3. The gate must replay **bs>1 and CUDA-graph** surfaces — every historical kimi accuracy bug was batch/row-state, exactly what one-shot bs1 runs miss.
4. 8×H20-only model ⇒ the gate is a committed script run manually pre-release, with thresholds encoded in the script. It cannot live in CPU CI; it must still live in git.
5. Commit the base-vs-opt prefill-logits A/B harness so future kernel picks repeat the PR #204 validation instead of re-inventing it.

This milestone gates: further decode-kernel opts, the K2.6 weight swap, and any TP1/DP8 parity claim.

### Next — serving performance

6. **Prefill / TTFT milestone.** The largest user-visible gap vs vLLM (p50 4.5×, p99 31×). Decompose short-prompt TTFT (~2s): embedding / MLA prefill / MoE prefill / first-collective stream drain / per-layer scratch allocation; then fix the dominant term and add a TTFT gate. Target: vLLM's `69.6/135.4ms` class.
7. **HTTP-overhead isolation.** ~33% (4.74ms/token) between in-process and HTTP at bs4. Cross-check the three causes already found on qwen3: TCP_NODELAY/Nagle on SSE, frontend bridge, zombie decode from missing abort.
8. **TP8 continuous batching** (if TP8 stays a supported shape): running/waiting split as in dp.rs; recompute batch shape from live rows so retired rows stop paying collective width.
9. **Bench-regression snapshot.** `bench_snapshots/h20/kimi-k2.json` via the existing `bench_serving` wiring, plugged into the convention's compare gate.
10. **PPLX + CUDA Graph.** The EP all-to-all progresses via a host worker thread — non-capturable; graphs are hard-disabled on the PPLX path, leaving ~3ms/token of host enqueue on the table. Either device-side signaling to make a2a stream-capturable, or capture rank-local compute segments around it.
11. **MoE layer pipelining on PPLX.** dispatch→Marlin→combine runs strictly serial per layer (~1.6ms/token structural bubble); the EpBackend already exposes the four phases separately for overlap.
12. **DP8 routing quality.** `DpLoadBalancer::pick_rank` is greedy free-slot-count (duplicated in dp.rs); needs load- and (later) prefix-affinity-aware routing.

### Later — structural

13. **Prefix cache chain.** (a) shared block table + global free pool replacing the fixed per-slot arena; (b) MLA physical backend for kvbm — the logical layer (sequence hashing, block matching) is layout-agnostic and reusable, but `KvLayout`/`KvBuffer` are FullAttention-only and can't express the dual `ckv[512]+kpe[64]` segments; (c) nonzero-position suffix prefill — `configure_slot_prefill` hardcodes positions `0..seq_len`, and kpe is stored RoPE-applied, so cache-hit suffix prefill would reproduce qwen3's start-position drift bug unless positions are threaded through; gate with a cached-replay logits A/B; (d) DP8 prefix-affinity routing.
14. **MLA split-KV decode.** `partition_kv=false`, one CTA scans the full KV serially per row — fine at 2k context, cannot saturate the GPU at long context. Prerequisite: a long-context decode harness (long-context correctness is currently not claimed at all).
15. **TP8 cuBLASLt parity or formal demotion.** PR #204's cuBLASLt GEMM picks are shape-gated to TP1 (`local_heads==64`, `o_proj cols==8192`); TP8 silently falls back to the old GEMMs. Either add TP8-shaped plans or declare TP8 reference-only and stop maintaining two backends.
16. **K2.6 readiness.** Same-architecture weights pending. Confirm drop-in swap (no manifest/config/kernel changes), and re-run the accuracy gate against K2.6 — impossible until M2 exists.
17. **Multi-node DP/EP** per [dp-design.md](dp-design.md) §10.

## Cleanup ledger

Order matters: fix the lint gate first, or the dead code regrows.

1. **Lint-gate hole.** The global clippy hook lints only `default-members` (pegainfer-server); the kimi hook is `--no-deps` scoped to `^pegainfer-kimi-k2/`. Net: `pegainfer-kernels/src/ops/kimi_k2/*`, kimi csrc, and `pegainfer-comm` are never `-D warnings` checked. Add a scoped hook for the kernels kimi slice (and comm).
2. **Dead expert-major/CUTLASS-era cluster.** `experts.rs` (2682 lines, >1k redline) hides a retired API behind a whole-file `#![allow(dead_code, unreachable_pub)]`: 4 dead fns + dead types with zero external callers, plus 4 compiled-but-uncalled CUDA launchers in `kimi_experts.cu` and their FFI decls. Delete surgically (live Marlin kernels are interleaved in the same file), drop the file-level allow.
3. **`weight_shape` tensor.** Loaded to GPU for 60 layers × 384 experts × 3 projections, validated to `[2]`, then never consumed by any kernel (dims come from manifest constants). Drop from the load path after the dead probe (its only consumer) is removed.
4. **`KERNELS.md` stale rows.** References a deleted `.cu` file and two ops with zero code references.
5. **Doc refresh.** `operator-todo.md` still asserts a `KIMI_RUNNER_MAX_BATCH = 4` hard-cap contract — code says 64 with buckets `[1..64]`; three docs still give repro commands using the removed `kimi-k2-pplx-ep` feature and `PEGAINFER_KIMI_PARALLEL` env (replaced by `--tp-size/--dp-size/--ep-backend`). `optimization.md` still headlines the TP8 bs4 path as mainline.
6. **Doc consolidation.** Delete `dp1-tp8-ep8-performance.md` (superseded ledger) after lifting its lessons; collapse the bring-up trio (`support-analysis.md` / `changelog.md` / `operator-todo.md`) into one history doc, lifting to `docs/lessons/`: BF16 bulk all-reduce breaks greedy (keep the F32 bridge), merging shared+routed reduce breaks cold-batch greedy, and full-percentile reporting discipline.

## Done criteria

This roadmap is healthy when:

- a temperature/top_p request is either correctly sampled or explicitly rejected — never silently greedy; generation stops at EOS;
- a fresh clone + an 8×H20 node can re-run the accuracy gate from committed code and fixtures alone;
- TTFT p50/p99 has a measured decomposition, a gate, and is within striking distance of the vLLM class;
- prefix-cache work consumes the shared KV roadmap (#203 §1) rather than a kimi one-off;
- `docs/models/kimi-k2/` describes the engine that exists, not the bring-up that happened.
