# Qwen3-4B Roadmap

> **TL;DR:** Qwen3-4B is the project's maturity bar: continuous batching, TP=2, default-on prefix cache (#216), HF and real-adapter LoRA golden gates, truthful per-request cached-token usage (#246), batched sampling (#307/#284), and in-process KV offload (#316) are live. The main open set is TP numerical qualification, aggregate prefix-cache telemetry, LoRA cache-salt isolation, and YaRN #8 for rope-scaled checkpoints.
>
> **Last touched:** 2026-07

Tracking issue: see the `[Model] Qwen3-4B roadmap` GitHub issue. Cross-model items stay in `docs/roadmap/execution.md`; this doc owns the qwen3 line.

## Where the line stands

| Area | State | Evidence |
| --- | --- | --- |
| Batching | ✓ continuous, KV full-lifetime admission, rejection (#85 fix) | `scheduler.rs`, `scheduler/tests.rs` |
| Prefix cache | ✓ default-on full-block kvbm matching (#216); 4 cache-hit replay passes in the golden gate | `executor.rs`, `tests/hf_golden_gate.rs` |
| Accuracy gate | ✓ HF bf16 golden, bs=1/batched/graph + cached replays; single-GPU, ≤256-token prompts | `tests/hf_golden_gate.rs` |
| Long context | ✓ fixed: RoPE cache sized from `max_position_embeddings`, admission rejects past the window, kernel traps OOB; gated by reject + in-window >4096 ITs. YaRN #8 still open for scaled checkpoints | `weights.rs`, `tests/context_window.rs`, `tests/context_window_in_window.rs` |
| Batch sampling | ✓ #307/#284: greedy rows use indexed batched argmax; non-greedy rows compact into one FlashInfer batched sampling call per step; top1 scratch sizing now comes from the kernel | `openinfer-kernels/src/ops/sampling.rs`, `openinfer-kernels/csrc/shared/flashinfer_sampling.cu` |
| KV offload (L2) | ✓ in-process pegaflow host-tier save/restore (#316); CLI `--kv-offload`/`--no-prefix-cache`, plain + LoRA; pure-L2 TTFT 195→40ms | `subsystems/runtime/pegaflow-offload-integration.md` |
| TP correctness | ⚠ TP=2 graph startup/concurrent completion/deadlock coverage exists; TP-vs-TP=1 logits parity is still ungated | `tests/tp_concurrent_decode.rs`, `tp-design.md` |
| LoRA | ✓ load/unload/request-level/TP built; non-zero PEFT golden covers bs=1, mixed base/LoRA batches, TP=2 sharding, and rejects a silently inactive adapter | `lora.rs`, `tests/lora_golden_gate.rs` |
| Cache usage | ✓ executor hit count reaches `TokenEvent::Scheduled` and frontend prefill/usage stats; cold/warm and terminal-first-output regressions cover propagation | `tests/cached_tokens_usage.rs`, `openinfer-vllm-frontend/src/bridge/tests.rs` |
| Non-greedy sampling | ⚠ qwen3/qwen35 model behavior gates now cover `temperature` / `top_k` / `top_p`; penalties/min_p are still absent from `SamplingParams` | `tests/sampling_behavior.rs`, `openinfer-kernels/src/ops/sampling.rs` |
| Bench snapshots | ⚠ exist but remain stale relative to current main; not refreshed by #216; no mixed-load ITL profile | `bench_snapshots/` |
| PP | greenfield (aspiration only) | — |

## Roadmap

### Now

1. **YaRN for rope-scaled checkpoints (#8).** The #220 RoPE OOB fix landed scope (a): the cos/sin cache is sized from `config.max_position_embeddings`, admission crash-early rejects past the window (distinct context-length vs KV-budget reasons), the kernel `__trap`s an out-of-range position as a last-resort backstop, and the gate now covers both an oversized reject and an in-window >4096 case (`tests/context_window.rs`, `tests/context_window_in_window.rs`). That precompute is correct *only because this checkpoint has `rope_scaling: null`*. Scope (b) remains open: #8 YaRN is the prerequisite for any rope-scaled checkpoint — the precompute length must come from the scaled schedule, coordinated with the qwen3.5 sibling fix so both crates share the pattern.
2. **Batched decode sampling.** ✓ #307/#284 route all-greedy batches through indexed batched argmax and mixed batches through one compact FlashInfer batched sampling pass for non-greedy rows. Shared `openinfer-kernels` work covers qwen35 too; keep the HF gate and nsys no-per-row-sampler check as the regression surface.
3. **Sampling correctness coverage.** Shared sampler tests cover seed determinism and temperature/top_k/top_p behavior, and qwen3/qwen35 each now have a model-level non-greedy behavior gate. Keep auditing the frontend for silently-dropped params (penalties, min_p are absent from `SamplingParams` entirely) — the kimi-k2 silent-greedy bug (#237) shows this class is real.
4. **Aggregate prefix-cache telemetry.** ✓ #246 carries executor-computed `cached_tokens` through `TokenEvent::Scheduled` into frontend prefill/usage stats, with a real cold/warm Qwen3 integration test and bridge regressions for first-token and terminal-only outputs. The remaining observability gap is scheduler-native query/hit counters and a cache hit-rate panel; request-derived cached/computed token counters already reach `/metrics`. Adjacent #78 (streaming completion-token accounting) is a separate usage surface.

### Next

5. **Mixed-load ITL profile, then the chunked-prefill decision.** A long prompt admitted mid-decode runs as one unbounded prefill in the unified step (no per-step token budget exists) — the documented +38% ITL p99 tail vs vLLM. Maintainer stance on chunked prefill is *conditional* (`scheduler.md`: varied-length workloads break waves naturally); so the gate is a tracked mixed-load benchmark profile first, implementation only if the tail matters for a real workload. Refresh the stale bench snapshots in the same pass.
6. **TP correctness pass.** Extend the golden gate to TP=2 (skip when fewer than two GPUs) so its tolerances guard sharding and all-reduce, then qualify TP=8 systematically. `tp_concurrent_decode` now protects graph startup and concurrent progress, but a reduction-order or shard-offset bug remains invisible to numerical gates.
7. **LoRA prefix-cache salt isolation.** The non-zero PEFT gate is implemented in `tests/lora_golden_gate.rs`: it teacher-forces a real adapter against the PEFT reference under golden-gate tolerances, covers mixed base/LoRA batches and TP=2, and fails if the adapter does not change the base result. The remaining #173 coverage gap is a pinning test proving adapter A's cached blocks cannot hit for adapter B.
8. **Eviction behavioral test.** Evict-then-remiss is never exercised: register a prefix, release it, pressure the pool until eviction, assert truncated/zero match + correct recompute. kvbm-logical layer needs no GPU.
9. **Disconnect block-pinning.** After #216, a disconnected request pins its cache blocks (strong Arcs) until the next failed send — #215 is now also a KV-budget problem. Scheduler half: proactive `token_tx.is_closed()` sweep per step; folds into the server-wide #215.

### Later

- **Pipeline parallelism** — greenfield, no code; revisit when a multi-node driver appears.
- **Vocab-parallel embedding/lm_head** — revisit after TP numerical qualification; TP decode CUDA Graph pre-capture is already implemented.

## Cleanup ledger

- **Issue hygiene:** #188 references a test target deleted in #194 — close as superseded by the golden gate. #203 §1 still claims qwen3 has no prefix reuse — stale since #216.
- **File size:** `executor.rs`, `scheduler.rs`, and `kernel_bench.rs` remain above the 1k-line redline.
- **Docs/dead code:** ✓ #248 refreshed the crate/TP docs, moved the full-lifetime KV admission lesson to `docs/lessons/`, and removed the verified-unused Qwen3 constants and model-probe API. `execution.md` Done list still predates #216.

## Done criteria

- No admitted request can read past the RoPE cache; long-context behavior is gated.
- A mixed greedy/non-greedy decode step issues one indexed argmax pass for greedy rows and one batched FlashInfer sampling pass for non-greedy rows, not O(batch) per-row sampling.
- TP sits under the same numerical tolerances as the single-GPU path; the existing LoRA single-GPU/TP=2 golden gate stays green.
- Cached-token usage stays truthful; streaming completion-token accounting is tracked separately in #78.
- The docs describe the crate that exists.
