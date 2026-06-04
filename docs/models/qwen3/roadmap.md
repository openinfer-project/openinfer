# Qwen3-4B Roadmap

> **TL;DR:** Qwen3-4B is the maturity bar of the project — continuous batching, TP=2, default-on prefix cache (#216), and the HF logits golden gate are all live — so its roadmap is sharpening, not bring-up. The verified open set: one real correctness bug (#220 RoPE cache 4096 vs 40960 admitted, silent OOB), per-row batch-decode sampling (O(batch) launches + syncs per step despite a production-proven batched primitive in-tree), zero TP correctness coverage, LoRA built but gated only by a zero-adapter smoke, prefix-cache observability dropped at the scheduler boundary, and a docs layer that describes deleted tooling. Findings verified 2026-06-04 against `6ee9247`.
>
> **Last touched:** 2026-06

Tracking issue: see the `[Model] Qwen3-4B roadmap` GitHub issue. Cross-model items stay in `docs/roadmap/execution.md`; this doc owns the qwen3 line.

## Where the line stands

| Area | State | Evidence |
| --- | --- | --- |
| Batching | ✓ continuous, KV full-lifetime admission, rejection (#85 fix) | `scheduler.rs:478-510` |
| Prefix cache | ✓ default-on full-block kvbm matching (#216); 4 cache-hit replay passes in the golden gate | `executor.rs:750-751`, `tests/hf_golden_gate.rs` |
| Accuracy gate | ✓ HF bf16 golden, bs=1/batched/graph + cached replays; single-GPU, ≤256-token prompts | `tests/hf_golden_gate.rs:451` |
| Long context | ✗ **#220 live**: RoPE cache built for 4096 positions, 40960 admitted, kernel reads unbounded | `weights.rs:314`, `prefill_attention.cu:75-76` |
| Batch sampling | ✗ per-row: O(batch) launches + O(batch) D2H syncs per decode step; 1MB scratch literal | `executor.rs:159-179,212-214`, `ops/sampling.rs:7,204-208` |
| TP correctness | ✗ zero automated coverage — every test runs `device_ordinals: vec![0]` | grep `tests/` |
| LoRA | ⚠ load/unload/TP/request-level all built; only test uses a **zero adapter** | `lora.rs`, `tests/lora_smoke.rs:91-130` |
| Non-greedy sampling | ✗ zero correctness coverage (all tests greedy); penalties/min_p absent from `SamplingParams` | grep `tests/` |
| Bench snapshots | ⚠ exist but ~187 commits stale; not refreshed by #216; no mixed-load ITL profile | `bench_snapshots/` |
| PP | greenfield (aspiration only) | — |

## Roadmap

### Now

1. **#220 RoPE OOB fix.** Two scopes, deliberately split: (a) immediate — size the cos/sin cache from config and crash-early reject at admission past cache length; correct as a plain precompute *only because this checkpoint has `rope_scaling: null`*; (b) #8 YaRN is the prerequisite for any rope-scaled checkpoint — the precompute length must come from the scaled schedule. Add a >4096-position case to the gate once sized.
2. **Batched greedy decode sampling.** Phase 1: route all-greedy batches through `argmax_batch_bf16_into` — one launch + one D2H per step; this primitive is production-proven in deepseek-v2-lite (`runtime.rs:1379`). `flashinfer_top1_batch_into` has *no* production caller and needs its own validation before use. Phase 2: batched random path with per-row params; source the 1MB FlashInfer row-state scratch from the kernel instead of the literal. Shared `pegainfer-core/kernels` work — covers qwen35 too. Gated by the existing golden gate.
3. **Sampling correctness coverage.** Every test in both qwen crates is greedy. Add seed-determinism + temperature/top_k/top_p behavioral tests, and audit the frontend for silently-dropped params (penalties, min_p are absent from `SamplingParams` entirely) — the kimi-k2 silent-greedy bug (#237) shows this class is real and currently nothing would catch it here.
4. **Prefix-cache observability.** `cached_tokens` is computed (`executor.rs:751`) and dies at the scheduler boundary; the frontend hardcodes `num_cached_tokens: 0`. Thread it through `TokenEvent::Scheduled` into usage; log hit rate. Adjacent: #78 (streaming usage discards completion_tokens) — same usage-accounting surface.

### Next

5. **Mixed-load ITL profile, then the chunked-prefill decision.** A long prompt admitted mid-decode runs as one unbounded prefill in the unified step (no per-step token budget exists) — the documented +38% ITL p99 tail vs vLLM. Maintainer stance on chunked prefill is *conditional* (`scheduler.md`: varied-length workloads break waves naturally); so the gate is a tracked mixed-load benchmark profile first, implementation only if the tail matters for a real workload. Refresh the stale bench snapshots in the same pass.
6. **TP correctness pass.** Run the golden gate over `device_ordinals [0,1]` (skip when <2 GPUs) so the tolerances also guard sharding + all-reduce; TP=8 systematic pass after. A reduction-order or shard-offset bug is currently invisible to every gate.
7. **LoRA real-adapter accuracy gate.** The last open #173 acceptance criterion: teacher-force one real PEFT adapter against an HF reference with the golden-gate tolerances. Today base==(base+zero·LoRA) is all that's proven. The salt-isolation of the prefix cache also deserves a pinning test (adapter A's blocks must not hit for adapter B).
8. **Eviction behavioral test.** Evict-then-remiss is never exercised: register a prefix, release it, pressure the pool until eviction, assert truncated/zero match + correct recompute. kvbm-logical layer needs no GPU.
9. **Disconnect block-pinning.** After #216, a disconnected request pins its cache blocks (strong Arcs) until the next failed send — #215 is now also a KV-budget problem. Scheduler half: proactive `token_tx.is_closed()` sweep per step; folds into the server-wide #215.

### Later

- **Pipeline parallelism** — greenfield, no code; revisit when a multi-node driver appears.
- **Vocab-parallel embedding/lm_head + TP CUDA-graph** — the real open remainder of `tp-design.md`.

## Cleanup ledger

- **Issue hygiene:** #188 references a test target deleted in #194 — close as superseded by the golden gate. #203 §1 still claims qwen3 has no prefix reuse — stale since #216.
- **Dead code:** `batch_decode_trace.rs` `HIDDEN_SIZE`/`INTERMEDIATE_SIZE` consts (pub, zero readers); `probe_model()`+`ModelInfo` uncalled (server inlines its own detection — same in qwen35; wire it or delete it).
- **File size:** `executor.rs` (1435), `scheduler.rs` (1420, ~826 of them inline tests), `kernel_bench.rs` (1112) breach the 1k-line redline.
- **Docs:** `model-crate.md` TL;DR advertises a deleted `qwen3_kernel_snapshot` bench and, with `kernels-crate.md`, uses the obsolete `crates/` layout in every command — collapse both into one slim layout doc. `tp-design.md` describes the implemented controller/worker runtime as future direction — rewrite to past tense, promote the 3 real open items. `kv-pressure-hang.md` — lift the KV-lifetime-reservation lessons to `docs/lessons/`, then delete. `execution.md` Done list predates #216.

## Done criteria

- No admitted request can read past the RoPE cache; long-context behavior is gated.
- A bs=32 greedy decode step issues one sampling launch, not 32.
- TP and LoRA paths sit under the same golden-gate tolerances as the single-GPU path.
- Usage reporting (cached tokens, streaming completion tokens) is truthful.
- The docs describe the crate that exists.
