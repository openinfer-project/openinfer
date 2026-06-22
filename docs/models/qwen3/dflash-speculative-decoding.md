# DFlash Speculative Decoding (Qwen3-4B)

**TL;DR:** Qwen3-4B gains DFlash speculative decoding behind `--dflash-draft-model-path`. Speculative decode is modelled as an **optimistic transaction**: the DFlash drafter proposes K tokens, one target "verify" forward over the K+1 span confirms them, and we commit the longest argmax-matching prefix + 1 bonus token (rolling back the rest of the speculative KV). Greedy decode is **lossless up to bf16 numerical tie-flips** — the same non-determinism that already affects plain greedy decode at genuine bifurcation points; lm-eval gsm8k strict-match is identical with spec on vs off. Measured single-stream decode A/B: **1.82× on RTX 5070 Ti** (93.4 → 170.0 tok/s), **1.56× on RTX 5090** (168.9 → 263.2 tok/s). The drafter's out-of-pool footprint (weights + per-request KV/scratch) is reserved during memory profiling so the KV pool shrinks to fit instead of OOMing under load, and requests landing in the draft's final in-fill block are rejected cleanly at admission rather than crashing mid-prefill.

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

| Config | RTX 5070 Ti, bs=1 | RTX 5090, bs=1 |
| --- | --- | --- |
| spec OFF (plain decode) | 93.4 tok/s | 168.9 tok/s |
| spec ON (DFlash) | 170.0 tok/s | 263.2 tok/s |
| **speedup** | **1.82×** | **1.56×** |

The speedup is smaller on the 5090: its higher memory bandwidth makes the baseline decode less memory-bound, so amortizing the target forward buys less. Both builds use CUDA 13.x (the 5090's default 12.9 has the documented cuBLAS N=1025 cliff).

Throughput under concurrent load is a separate axis (`vllm bench serve` A/B). Speculative decoding's win shrinks — and can invert — as batch concurrency rises and the GPU turns compute-bound, so the crossover point is the interesting number. Still pending.

## Task-level accuracy parity (lm-eval gsm8k)

Token-level losslessness should imply task-level parity. Confirmed on the 5090 with `lm-eval` gsm8k (5-shot, greedy, `local-completions` against the openinfer server, 50 questions):

| | flexible-extract | strict-match |
| --- | --- | --- |
| spec OFF | 0.86 | 0.86 |
| spec ON (DFlash) | 0.88 | 0.86 |

`strict-match` is identical; `flexible-extract` differs by one question (within the ±0.05 stderr), the same single bf16 tie-flip the losslessness gate sees. DFlash does not change task accuracy.

Harness note: openinfer's `/v1/completions` rejects a per-request `seed` field (`"per-request seed is not supported by this engine"` → 400), which the OpenAI/lm-eval client sends by default. For the eval the client was patched to drop `seed`; making the frontend accept-and-ignore `seed` under greedy is a separate, unrelated improvement (not part of this change).

## GPU memory budget & context limit

DFlash's draft model and its per-request KV/scratch live **outside** the paged KV pool (`KvCacheManager`), so they must be reserved *before* the pool is sized or they silently steal from it and OOM under concurrency / long contexts. `DFlashMemoryReservation::from_config` (cheap — reads the draft `config.json`) splits the footprint by how it scales and the budget charges each in the matching place:

- **Per-token, pool-scaling** (draft KV + the prompt-tracking context scratch + the in-fill tail scratch + pending context, **65 536 B/token**) → folded into `effective_bytes_per_block`, so the *target* block count shrinks while the pool itself stays allocated at the target-only `bytes_per_block`. This is a safe upper bound: the scheduler reserves pool blocks for each request's whole lifetime (`prompt + max_tokens`), and the draft attends at most that many tokens, so billing the draft per pool-token over-covers. The draft KV and tail scratch are sized one in-fill block past that lifetime, so the fixed term also reserves `max_decode_batch × block_size` of that per-request headroom.
- **Fixed, pool-independent** (draft weights ~1.1 GiB + the block-sized per-request scratch across the decode batch — logits-dominated, ~6.5 MiB × 256) → added to the KV `margin`.

Measured on the RTX 5070 Ti (16 GB, util 90%), the pool correctly makes room for the draft:

| | margin | KV budget | KV blocks |
| --- | --- | --- | --- |
| DFlash OFF | 150 MiB | 4113 MiB | 1828 |
| DFlash ON | 2972 MiB | 1389 MiB | 427 |

The fixed reservation lands exactly in the margin (+2822 MiB) and the per-token factor (≈1.44×) shrinks the remaining blocks. Before this, those ~2.8 GiB loaded on top of the full 1828-block pool → OOM under load.

**Context limit.** The drafter's fixed-width in-fill block writes `block_size` positions past the committed length each step, so the DFlash-effective context is `max_position_embeddings − block_size`. `max_context_tokens()` returns that when DFlash is on, so a request that fits the target window but lands in the draft's final block is **rejected cleanly at admission** instead of crashing mid-prefill (`tests/dflash_speculative_gate.rs::dflash_request_in_draft_headroom_is_rejected_not_panicked`).

## Constraints & open follow-ups

- **TP=1, primary rank only.** The DFlash lane runs on the worker thread of the primary rank; the launch path gates `!(dflash && lora)` and `tp_size == 1`. The server fails loud (`--dflash-draft-model-path` is rejected for non-Qwen3 model lines rather than silently ignored).
- **Second proposer → introduce the trait.** Until then the proposer is concrete on purpose.
- **File size.** `executor.rs` (~2.8k lines) and `scheduler/tests.rs` (~1.2k lines) exceed the 1k guideline; the spec arms in `execute_step_on_lane` are candidates to move into `executor/spec.rs` in a follow-up.
- **5090 validation — done** for correctness and single-stream: `hf_golden_gate` passes (bs1 / batched / cuda-graph / tp2), the losslessness gate passes, single-stream A/B is 1.56×, and lm-eval gsm8k parity holds (above). Still pending: `vllm bench serve` concurrent-throughput A/B (the spec-helps-vs-hurts crossover under load).
- **Frontend `seed`.** `/v1/completions` 400s on a per-request `seed` instead of accepting-and-ignoring it under greedy — surfaced while wiring lm-eval; orthogonal to spec decode, worth a small separate fix.
- **Reclaim the per-request reservation.** The fixed term is dominated by the block-sized scratch billed per decode-batch slot (~6.5 MiB × 256 ≈ 1.6 GiB), most of it the transient logits buffer (`vocab × block`). The per-token term carries the context/pending scratch, which today reallocs to prompt length on the first draft and never shrinks (sized for `[0, context_len)` but only `[committed_len, +block]` is live). Both are billed conservatively because they're per-request; sharing the transient scratch across the serial per-request draft loop (`execute_dflash_draft`) and collapsing the prompt-persistent buffers would cut the reservation to roughly the draft weights + draft KV, reclaiming most of the ~2.8 GiB. Deserves its own validated PR (touches the draft forward).
