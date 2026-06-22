# DFlash Speculative Decoding (Qwen3-4B)

**TL;DR:** Qwen3-4B gains DFlash speculative decoding behind `--dflash-draft-model-path`. Speculative decode is modelled as an **optimistic transaction**: the DFlash drafter proposes K tokens, one target "verify" forward over the K+1 span confirms them, and we commit the longest argmax-matching prefix + 1 bonus token (rolling back the rest of the speculative KV). Greedy decode is **lossless up to bf16 numerical tie-flips** — the same non-determinism that already affects plain greedy decode at genuine bifurcation points. Measured single-stream decode A/B on RTX 5070 Ti (bs=1): **93.4 → 170.0 tok/s, 1.82×**.

Last touched: 2026-06

## The abstraction: speculative decode = optimistic transaction

Every speculative method is the same transaction; only *propose* differs.

1. **Propose** — a method-specific drafter emits K candidate tokens. (DFlash is the only proposer today; an n-gram / EAGLE proposer would slot in here.)
2. **Verify** — the target model runs ONE prefill-style forward over the K+1 span `[current_token, draft_1, …, draft_K]` and reports its argmax at every position (echo=true).
3. **Accept** — `accept_greedy` keeps the longest prefix where each draft matches the target argmax, then appends the target's own token at the first mismatch (the "bonus"). A verify step therefore always commits `1..=K+1` tokens — at least one token of progress even when every draft is rejected.
4. **Commit / roll back** — accepted tokens' KV is committed (`apply_speculative`); the unused tail of the speculative reservation is LIFO-dropped (`revert_schedule`).

The draft↔verify boundary is a **pure token span**. Hidden states never cross it — they stay inside the proposer (`dflash.rs` / `dflash_lane.rs`). This is what lets the shared core (`speculative.rs`: `accept_greedy`, `build_verify_results`) be method-agnostic, and it is why there is deliberately **no proposer trait yet**: a trait with one impl is premature. Add it when a second method lands.

Shared core lives in `openinfer-qwen3-4b/src/speculative.rs`; the transaction wiring is in `openinfer-qwen3-4b/src/executor/spec.rs`; the KV transaction primitives (`schedule_speculative` / `apply_speculative` / `speculative_view` / `revert_schedule`) are in `openinfer-kv-cache/src/pool.rs` delegating to kvbm `scheduled.rs`.

## Key invariant: readiness comes from prefill capture, not a handshake

Speculative-on **forces prefix caching off**. With no prefix reuse, every eligible request's target hidden context is captured during its own prefill — so there is no per-request "drafter ready" handshake. Readiness is derived from the prefill capture-set plus the completed flag. This is the load-bearing simplification; if prefix caching were allowed, a cache-hit request would skip the prefill that seeds the drafter, and the invariant would break.

Draft-seed alignment: after a verify accepting `m` drafts, the committed span positions with *known* hidden states are `[current_token, draft_1..draft_m]` = `m+1` rows. The next current_token (`target_argmax[m]`) is freshly predicted and its hidden was never forwarded — exactly like the post-prefill first token. `record_verify_dflash_context` appends `accepted_tokens.len()` (= `m+1`) rows and advances `kv_position` by the same. This is the subtlest part of the feature; it is defended by crash-early asserts (`append_pending_context` overflow/dim/range checks).

## Losslessness: lossless up to bf16 ties

The greedy oracle is "spec-on equals spec-off token-for-token". In practice the two diverge only at genuine bifurcation points, and only because of bf16 numerical noise — **not** a logic bug. Evidence:

- The verify forward builds committed KV incrementally across batched spans (M=16-ish), while a one-shot prefill is a single M=1 forward. cuBLAS picks different GEMM tilings → ULP-scale reduction-order differences → an argmax flip at a near-tie. This is the same class as the `hf_golden_gate` `MARGIN_TOL = 0.20` tolerance.
- Multi-token acceptance (runs observed up to 11 tokens) is **bit-identical** to baseline at every non-tie position — proving the span-position-≥1 path is correct.
- Re-running flips **different** prompts each time (non-deterministic ⇒ bf16, not a deterministic logic error); one observed flip had regret 0.000 (an exact tie).
- The spec pick is always the #1 or #2 token of the prefill kernel's own distribution, within 0.20 nat of #1.

### The losslessness gate (`tests/dflash_speculative_gate.rs`)

The gate runs a baseline (spec off, logprobs on) and a spec engine on the same prompts, then at the first token where they diverge it measures the spec pick's **regret** against the prefill kernel's own distribution. Within `MARGIN_TOL = 0.20` nat ⇒ benign tie-flip (pass); clearly worse ⇒ real bug (fail). The realistic bug classes (KV misalignment, mask leak) push the pick far outside 0.20 nat, so the gate has teeth — it is not tautological.

**Known scope limit:** the gate regret-checks only the *first* divergence position per prompt, then continues. A bug that stays within the tie band at the first bifurcation but corrupts later positions could slip. Acceptable for v1; the next iteration should re-anchor and regret-check the following few positions after a benign-tie classification.

## Performance

Single-stream decode is where speculative decoding pays off directly: plain decode is memory-bound (one target forward per token); spec amortizes that forward over the accepted run. A/B harness: `tests/dflash_speculative_perf.rs` (bs=1, 256 tokens, `ignore_eos`, one warm-up discarded).

| Config | RTX 5070 Ti, bs=1 |
| --- | --- |
| spec OFF (plain decode) | 93.4 tok/s |
| spec ON (DFlash) | 170.0 tok/s |
| **speedup** | **1.82×** |

Throughput under concurrent load is a separate axis (`vllm bench serve` A/B) and is best measured on the 5090 — pending.

## Constraints & open follow-ups

- **TP=1, primary rank only.** The DFlash lane runs on the worker thread of the primary rank; the launch path gates `!(dflash && lora)` and `tp_size == 1`. The server fails loud (`--dflash-draft-model-path` is rejected for non-Qwen3 model lines rather than silently ignored).
- **Second proposer → introduce the trait.** Until then the proposer is concrete on purpose.
- **File size.** `executor.rs` (~2.8k lines) and `scheduler/tests.rs` (~1.2k lines) exceed the 1k guideline; the spec arms in `execute_step_on_lane` are candidates to move into `executor/spec.rs` in a follow-up.
- **5090 validation** — golden gate + `vllm bench serve` throughput A/B on the company 5090 (same sm_120 arch as the 5070 Ti, so correctness carries; the 5090 is for representative throughput numbers).
