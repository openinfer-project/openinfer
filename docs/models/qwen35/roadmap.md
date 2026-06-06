# Qwen3.5-4B Roadmap

> **TL;DR:** Qwen3.5-4B is fast and decode-correct — GDR kernels optimized, CUDA-graph decode, TTFT/TPOT at vLLM parity, current bench snapshots — and now has a long-prompt logits gate over the old 4096-position RoPE cache boundary. The current #250 slice fixes the Qwen3.5 RoPE cache to use `max_position_embeddings`, adds fail-closed cache coverage checks, verifies 4097/8192-token HF bf16 logits replay, and recovers full GSM8K 8-shot within 0.15 percentage points of the HF baseline (`strict-match` 79.38%, `flexible-extract` 79.30% vs HF 79.45%). Remaining structural items: monolithic HND prefill staging (~640MB transient per request, now fail-closed at the hard 20k-token cap) and prompt-only admission with no `Rejected` event path; the current admission/slot/compaction scheduler decisions are now covered by CPU-level lib tests. Findings originally verified 2026-06-04 against `6ee9247`; #186 gate status updated 2026-06-05, #250 long-prompt and GSM8K status updated 2026-06-05, #255 scheduler seam updated 2026-06-06.
>
> **Last touched:** 2026-06

Tracking issue: see the `[Model] Qwen3.5-4B roadmap` GitHub issue. Sibling doc: `docs/models/qwen3/roadmap.md` — batched sampling and non-greedy coverage are shared items owned there.

## Where the line stands

| Area | State | Evidence |
| --- | --- | --- |
| Decode perf | ✓ GDR fused recurrent optimized; CUDA-graph decode; parity with vLLM | `docs/projects/qwen35-4b-optimization.md` |
| Bench snapshots | ✓ current (unlike qwen3's) | `bench_snapshots/` |
| **Long-prompt accuracy** | Recovered for the measured path: the 4097/8192-token HF logits replay passes after the RoPE cache fix; full GSM8K 8-shot at `batch_size=1` recovers to `strict-match` 79.38% / `flexible-extract` 79.30% vs HF 79.45% | `tests/hf_golden_gate.rs`, `test_data/qwen35-4b-hf-long-golden.safetensors`, `docs/benchmarks/accuracy-eval-results.md`, issue #250 |
| Accuracy gate | ✓ small and long HF bf16 logits gates for pinned Qwen3.5-4B; exact-text e2e/regen retired; broader rand/hash corpus deferred until cross-arch policy exists | `tests/hf_golden_gate.rs`, `test_data/qwen35-4b-hf-golden.safetensors`, `test_data/qwen35-4b-hf-long-golden.safetensors`, `docs/models/qwen35/accuracy.md` |
| Teacher forcing | ✓ model-local test executor can force fixed token IDs through prefill + graph decode; serving scheduler still free-runs user requests | `src/executor.rs`, `tests/hf_golden_gate.rs` |
| Prefill memory | ✗ monolithic HND staging ≈640MB transient per request; `MAX_SEQ = 20000` hard cap | `prefill.rs` |
| Long context | Partial: current #250 slice sizes the RoPE cache from `max_position_embeddings`; decode checks the cache before use, and prefill now fails closed at the active `MAX_SEQ = 20000` HND staging cap | `config.rs`, `weights.rs`, `prefill.rs`, `batch_decode.rs` |
| Admission | ✗ prompt-only KV sizing, no `Rejected` event, KV exhaustion mid-decode aborts the whole batch — pre-#85-fix semantics | `scheduler.rs` |
| Scheduler tests | Partial: current plan selection, prompt-only admission, slot assignment, and slot-compaction decisions are CPU-tested; GPU execution remains coupled to the production scheduler | `src/scheduler/plan.rs` |
| TP | ✗ absent (single GPU only) | — |
| Prefix cache | ✗ absent; recurrent GDR state (~48MB per boundary snapshot) makes "prefix hit" itself a design question | — |

## Roadmap

### Now

1. **Keep #250's score evidence attached to the PR.** The current #250 slice proves a concrete long-prompt logits gate at 4097/8192 tokens, fixes the RoPE cache boundary, and passes full GSM8K 8-shot against `/v1/completions`: `strict-match` 79.38%, `flexible-extract` 79.30%, compared with the HF reference 79.45%.
2. **HF gate widening after the long-prompt root cause.** #186 provides the teacher-forced HF logits gate and qwen35 replay surfaces: sequential graph decode, bucket-straddling graph decode, and slot-compaction replay. #250 adds the first long-prompt case. Future widening should add recurrent-state handoff coverage once prefix work creates that surface.
3. **RoPE cache sibling follow-through.** Qwen3.5 now follows the qwen3 #220 shape: cache length comes from config and runtime checks fail closed before prefill/decode uses a missing position. Keep the YaRN #8 caveat for scaled checkpoints when porting or comparing model families.

### Next

4. **Admission overhaul.** Three coupled defects, fixed together as the qwen35 analog of the #85 work: size admission on full lifetime (prompt + max_tokens), add the `Rejected` event path the engine contract already defines, and on KV exhaustion fail the offending request — not the batch. The #255 scheduler seam gives this work a CPU-testable policy surface.
5. **Prefill full-paged migration.** Replace the HND staging copy with direct paged writes: removes the ~640MB transient, the `MAX_SEQ=20000` cap, and the extra D2D pass. Chain dependency: paged-direct prefill → per-token position plumbing → (3) RoPE cache → opens the door to 7.
6. **Scheduler logic seam follow-through.** The current admission/slot/compaction decisions have a CPU-tested seam. When 4 lands, keep the new lifetime-admission and rejection behavior in that seam instead of re-embedding it in GPU execution.
7. **Prefix-cache design note.** Linear-attention layers carry recurrent state, not KV blocks — a "prefix hit" must restore both the full-attention KV *and* a recurrent-state snapshot at a block boundary (~48MB per boundary at bf16). Whether to snapshot per block, per N blocks, or only at request end is an open trade; write the design note before any code. Depends on 5.
8. **kernel_plan port.** qwen3's `kernel_plan.rs` (runtime kernel selection + plan dump) has no qwen35 counterpart; decode kernel picks are hardwired. Mechanical port, community-friendly.

### Later

- **TP** — no sharding design exists for the hybrid stack (GDR state sharding is the open question). Design-first, no driver today.
- **CUDA-graph prefill** — prefill is eager and serial; revisit after 6 changes the memory layout.

## Cleanup ledger

- **Dead code:** `probe_model()`+`ModelInfo` and `start_with_model()` — zero callers (the server inlines detection; same dead pair in qwen3). Wire or delete.
- **Docs:** `accuracy.md` now owns the #186 HF logits gate and the retired exact-text baseline note. Several qwen35 docs still carry `Status:` enum headers (against repo convention) and `crates/` paths. Parity numbers drifted across docs (225ms/11.81ms vs the refreshed 234ms/11.77ms) — reconcile to one ledger. The e2e-gibberish debugging story should be lifted to `docs/lessons/` (it's a lesson about exact-match gates, not a qwen35 doc).
- **Shared with qwen3 (owned there):** batched greedy decode sampling (`batch_decode.rs` has the same per-row pattern), non-greedy sampling correctness coverage, frontend usage accounting (#78).

## Done criteria

- GSM8K 8-shot within a few points of the HF reference, and a logits-level gate that would have caught the divergence.
- The exact-text e2e baseline-regeneration ritual is retired (#186 gate work).
- A 30k-token prompt is either served or rejected at admission — never a crash, never a silent cap.
- One request's KV exhaustion never kills its batch-mates.
- Scheduler admission logic runs under `cargo test` without a GPU.
