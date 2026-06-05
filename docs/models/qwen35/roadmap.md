# Qwen3.5-4B Roadmap

> **TL;DR:** Qwen3.5-4B is fast and decode-correct — GDR kernels optimized, CUDA-graph decode, TTFT/TPOT at vLLM parity, current bench snapshots — but it still has a **live long-prompt accuracy failure** (GSM8K 8-shot scores ~2% vs ~79% on HF; the current HF logits gate is a short fixed-sequence oracle). The #186 gate now adds a model-local teacher-forcing executor, HF bf16 logits golden, graph-decode replay, slot-compaction replay, and retires the brittle exact-text e2e / baseline-regeneration ritual. Remaining structural items: long-prompt root cause, monolithic HND prefill staging (~640MB transient per request, hard 20k-token cap), prompt-only admission with no rejection path, and zero CPU-testable scheduler logic. Findings originally verified 2026-06-04 against `6ee9247`; #186 gate status updated 2026-06-05.
>
> **Last touched:** 2026-06

Tracking issue: see the `[Model] Qwen3.5-4B roadmap` GitHub issue. Sibling doc: `docs/models/qwen3/roadmap.md` — batched sampling and non-greedy coverage are shared items owned there.

## Where the line stands

| Area | State | Evidence |
| --- | --- | --- |
| Decode perf | ✓ GDR fused recurrent optimized; CUDA-graph decode; parity with vLLM | `docs/projects/qwen35-4b-optimization.md` |
| Bench snapshots | ✓ current (unlike qwen3's) | `bench_snapshots/` |
| **Long-prompt accuracy** | ✗ **GSM8K 8-shot ≈2% vs HF ≈79%** — catastrophic divergence on long prompts; current gate is short fixed-sequence logits coverage | eval run 2026-06; long-prompt case still follow-up |
| Accuracy gate | ✓ small HF bf16 logits gate for pinned Qwen3.5-4B; exact-text e2e/regen retired; broader rand/hash corpus deferred until cross-arch policy exists | `tests/hf_golden_gate.rs`, `test_data/qwen35-4b-hf-golden.safetensors`, `docs/models/qwen35/accuracy.md` |
| Teacher forcing | ✓ model-local test executor can force fixed token IDs through prefill + graph decode; serving scheduler still free-runs user requests | `src/executor.rs`, `tests/hf_golden_gate.rs` |
| Prefill memory | ✗ monolithic HND staging ≈640MB transient per request; `MAX_SEQ = 20000` hard cap | `prefill.rs` |
| Long context | ✗ RoPE cache 4096 positions vs `max_position_embeddings: 262144` — sibling of qwen3 #220 | `weights.rs:297` |
| Admission | ✗ prompt-only KV sizing, no `Rejected` event, KV exhaustion mid-decode aborts the whole batch — pre-#85-fix semantics | `scheduler.rs` |
| Scheduler tests | ✗ zero CPU-level tests; all logic behind GPU-coupled paths | — |
| TP | ✗ absent (single GPU only) | — |
| Prefix cache | ✗ absent; recurrent GDR state (~48MB per boundary snapshot) makes "prefix hit" itself a design question | — |

## Roadmap

### Now

1. **Long-prompt accuracy bug.** The single most important item: bisect where the prefill diverges as prompt length grows (RoPE indexing past some boundary, GDR chunk recurrence at long T, staging-path numeric, and the 4096 RoPE cache are all candidates — the cache alone can't explain failures *below* 4096 if they exist, so first establish the onset length). Reproducible on one GPU with lm-eval against the OpenAI endpoint. The fix lands with a long-prompt case in whatever gate exists at that point.
2. **HF gate widening after the long-prompt root cause.** #186 now provides the teacher-forced HF logits gate and qwen35 replay surfaces: sequential graph decode, bucket-straddling graph decode, and slot-compaction replay. The next accuracy-gate work is to add the long-prompt case once #250 is root-caused, then add recurrent-state handoff coverage once prefix work creates that surface.
3. **RoPE cache sibling fix.** Same shape as qwen3 #220: cache built for 4096, config admits 262144. Size from config + crash-early at admission. Community-friendly; coordinate with the qwen3 fix so both crates use the same pattern (and both inherit the YaRN #8 caveat for scaled checkpoints).

### Next

4. **Admission overhaul.** Three coupled defects, fixed together as the qwen35 analog of the #85 work: size admission on full lifetime (prompt + max_tokens), add the `Rejected` event path the engine contract already defines, and on KV exhaustion fail the offending request — not the batch. CPU-testable once 6 lands.
5. **Prefill full-paged migration.** Replace the HND staging copy with direct paged writes: removes the ~640MB transient, the `MAX_SEQ=20000` cap, and the extra D2D pass. Chain dependency: paged-direct prefill → per-token position plumbing → (3) RoPE cache → opens the door to 7.
6. **Scheduler logic seam.** Extract admission/eviction/slot decisions behind a GPU-free boundary and put CPU tests on them, mirroring qwen3's scheduler-test layout. Prerequisite for testing 4 without a GPU.
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
