# Qwen3.5-4B Roadmap

> **TL;DR:** Qwen3.5-4B is fast and decode-correct — GDR kernels optimized, CUDA-graph decode, TTFT/TPOT at vLLM parity, current bench snapshots — and now has a long-prompt logits gate over the old 4096-position RoPE cache boundary. The #250/#253 RoPE slice sizes the Qwen3.5 cache from `max_position_embeddings`, fails closed before prefill/decode can read a missing position, verifies 4097/8192-token HF bf16 logits replay, and recovers full GSM8K 8-shot within 0.15 percentage points of the HF baseline (`strict-match` 79.38%, `flexible-extract` 79.30% vs HF 79.45%). Remaining structural items include the HND prefill staging footprint; full-lifetime KV admission and context-window rejection now live in the scheduler plan/e2e tests. Findings originally verified 2026-06-04 against `6ee9247`; #186 gate status updated 2026-06-05, #250 long-prompt and GSM8K status updated 2026-06-05, #255 scheduler seam updated 2026-06-06, #253 context-window admission updated 2026-06-11.
>
> **Last touched:** 2026-06

Tracking issue: see the `[Model] Qwen3.5-4B roadmap` GitHub issue. Sibling doc: `docs/models/qwen3/roadmap.md` — batched sampling and non-greedy coverage are shared items owned there.

## Where the line stands

| Area | State | Evidence |
| --- | --- | --- |
| Decode perf | ✓ GDR fused recurrent optimized; CUDA-graph decode; parity with vLLM | `docs/models/qwen35/optimization.md` |
| Bench snapshots | ✓ current (unlike qwen3's) | `bench_snapshots/` |
| **Long-prompt accuracy** | Recovered for the measured path: the 4097/8192-token HF logits replay passes after the RoPE cache fix; full GSM8K 8-shot at `batch_size=1` recovers to `strict-match` 79.38% / `flexible-extract` 79.30% vs HF 79.45% | `tests/hf_golden_gate.rs`, `test_data/qwen35-4b-hf-long-golden.safetensors`, `docs/benchmarks/accuracy-eval-results.md`, issue #250 |
| Accuracy gate | ✓ small and long HF bf16 logits gates for pinned Qwen3.5-4B; exact-text e2e/regen retired; broader rand/hash corpus deferred until cross-arch policy exists | `tests/hf_golden_gate.rs`, `test_data/qwen35-4b-hf-golden.safetensors`, `test_data/qwen35-4b-hf-long-golden.safetensors`, `docs/models/qwen35/accuracy.md` |
| Teacher forcing | ✓ model-local test executor can force fixed token IDs through prefill + graph decode; serving scheduler still free-runs user requests | `src/executor.rs`, `tests/hf_golden_gate.rs` |
| Prefill memory | Partial: prefill is chunked at `PREFILL_CHUNK_LEN = 20000`, but each chunk still carries the large HND staging footprint | `prefill.rs` |
| Long context | Partial: #250/#253 size the RoPE cache from `max_position_embeddings`; prefill/decode check cache coverage before use; scheduler admission rejects `prompt + max_tokens` past the position window and exposes the servable cap to the frontend; the scheduler e2e now covers the over-window rejection path | `config.rs`, `weights.rs`, `prefill.rs`, `batch_decode.rs`, `scheduler.rs`, `tests/e2e_scheduler.rs`, `src/scheduler/plan.rs` |
| Admission | ✓ existing full-lifetime KV admission and explicit `Rejected` events cover impossible KV requests; #253 adds the context-window rejection reason before prefill/decode | `scheduler.rs`, `src/scheduler/plan.rs`, `docs/models/qwen35/kv-admission.md` |
| Scheduler tests | Partial: current plan selection, full-lifetime admission, context-window rejection, slot assignment, and slot-compaction decisions are CPU-tested; GPU execution remains coupled to the production scheduler | `src/scheduler/plan.rs` |
| TP | ✗ absent (single GPU only) | — |
| Prefix cache | ✗ absent; recurrent GDR state (~48MB per boundary snapshot) makes "prefix hit" itself a design question | — |

## Roadmap

### Now

1. **Keep #250's score evidence attached to the PR.** The current #250 slice proves a concrete long-prompt logits gate at 4097/8192 tokens, fixes the RoPE cache boundary, and passes full GSM8K 8-shot against `/v1/completions`: `strict-match` 79.38%, `flexible-extract` 79.30%, compared with the HF reference 79.45%.
2. **HF gate widening after the long-prompt root cause.** #186 provides the teacher-forced HF logits gate and qwen35 replay surfaces: sequential graph decode, bucket-straddling graph decode, and slot-compaction replay. #250 adds the first long-prompt case. Future widening should add recurrent-state handoff coverage once prefix work creates that surface.
3. **RoPE cache sibling follow-through.** Qwen3.5 now follows the qwen3 #220 shape for the unscaled checkpoint: cache length comes from config, runtime checks fail closed before prefill/decode uses a missing position, and admission rejects requests that would run past the position window. Keep the YaRN #8 caveat for scaled checkpoints when porting or comparing model families.

### Next

4. **Prefill full-paged migration.** Replace the HND staging copy with direct paged writes: removes the ~640MB transient and the extra D2D pass. Chain dependency: paged-direct prefill → per-token position plumbing → RoPE/context-window invariants → opens the door to prefix-cache design.
5. **Scheduler logic seam follow-through.** The current admission/slot/compaction decisions have a CPU-tested seam. Keep future admission and rejection changes in that seam instead of re-embedding them in GPU execution.
6. **Prefix-cache design note.** Linear-attention layers carry recurrent state, not KV blocks — a "prefix hit" must restore both the full-attention KV *and* a recurrent-state snapshot at a block boundary (~48MB per boundary at bf16). Whether to snapshot per block, per N blocks, or only at request end is an open trade; write the design note before any code. Depends on 4.
7. **kernel_plan port.** qwen3's `kernel_plan.rs` (runtime kernel selection + plan dump) has no qwen35 counterpart; decode kernel picks are hardwired. Mechanical port, community-friendly.

### Later

- **TP** — no sharding design exists for the hybrid stack (GDR state sharding is the open question). Design-first, no driver today.
- **CUDA-graph prefill** — prefill is eager and serial; revisit after 6 changes the memory layout.

## Cleanup ledger

- **Dead code:** ✓ qwen35 `probe_model()`+`ModelInfo` and the `start_with_model` entry point removed (#258); the same dead pair still exists in qwen3 (owned there).
- **Docs:** ✓ qwen35 docs cleaned (#258): `Status:` enum headers dropped, obsolete `crates/` paths corrected to top-level, parity numbers reconciled to one ledger (234ms/11.77ms), and the e2e-gibberish story lifted to `docs/lessons/exact-match-gate-thread-cublas.md`. #186 then added the HF logits gate and retired the exact-text baseline.
- **Shared with qwen3 (owned there):** batched greedy decode sampling (`batch_decode.rs` has the same per-row pattern), non-greedy sampling correctness coverage, frontend usage accounting (#78).

## Done criteria

- GSM8K 8-shot within a few points of the HF reference, and a logits-level gate that would have caught the divergence.
- The exact-text e2e baseline-regeneration ritual is retired (#186 gate work).
- A 30k-token prompt is either served or rejected at admission — never a crash, never a silent cap.
- One request's KV exhaustion never kills its batch-mates.
- Scheduler admission logic runs under `cargo test` without a GPU.
