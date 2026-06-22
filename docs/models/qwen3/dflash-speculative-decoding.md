# DFlash Speculative Decoding (Qwen3-4B)

**TL;DR:** Qwen3-4B gains DFlash speculative decoding behind `--dflash-draft-model-path`. Speculative decode is modelled as an **optimistic transaction**: the DFlash drafter proposes K tokens, one target "verify" forward over the K+1 span confirms them, and we commit the longest argmax-matching prefix + 1 bonus token (rolling back the rest of the speculative KV). Greedy decode is **lossless up to bf16 numerical tie-flips** — the same non-determinism that already affects plain greedy decode at genuine bifurcation points; lm-eval gsm8k strict-match is identical with spec on vs off. Measured single-stream decode A/B: **1.82× on RTX 5070 Ti** (93.4 → 170.0 tok/s), **1.56× on RTX 5090** (168.9 → 263.2 tok/s). The drafter's out-of-pool footprint (weights + per-request KV/scratch) is reserved during memory profiling so the KV pool shrinks to fit instead of OOMing under load, and requests landing in the draft's final in-fill block are rejected cleanly at admission rather than crashing mid-prefill. **Concurrent throughput is now competitive after batching the draft forward.** The draft used to run a per-request **serial** `for` loop — launch-bound (a skip-attention A/B showed attention compute <2%, so 24.8 ms/step at batch 16 was almost all kernel-launch overhead), which inverted the single-stream win (c16 −59%). Batching the dense ops into one N×block pass drops draft@batch16 to **5.6 ms** and lifts 5090 greedy throughput to **c8 1346 / c16 1868 tok/s — both now beating vLLM's 1240 / 1846**. Single-stream (c1 237 vs vLLM 278) still trails, but **not from accept**: measured 9.1% vs vLLM's 8.85% with the *same* drafter rules out draft quality. The c1 gap is the draft's ~1 ms kernel-launch overhead — openinfer's spec path, unlike base decode, isn't CUDA-Graph captured. **CUDA-Graph draft is the tracked next step.** See Performance § for the A/B tables.

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

### Concurrent throughput: the draft loop is serial and launch-bound

Under concurrent load the single-stream win inverts. Same greedy harness (`temperature=0`, sharegpt prompts, 128 out tokens), openinfer vs vLLM with the **same** DFlash-b16 drafter, RTX 5090:

| concurrency | OI plain | OI DFlash serial | **OI DFlash batched** | vLLM DFlash |
| --- | --- | --- | --- | --- |
| c1 | 170 | 245 | 237 | 278 |
| c8 | 1180 | 831 | **1346** | 1240 |
| c16 | 2277 | 1013 | **1868** | 1846 |

(tok/s, sharegpt out128, same-session serial-vs-batched A/B. The serial draft inverted the win — vLLM degraded gracefully while openinfer nearly halved; batching the draft restores it, c8/c16 now beat vLLM. c1 is unchanged — batch=1 has no batching win; see "Single-stream gap" below.)

#### Root cause (serial draft) and the fix (batched draft, landed)

Accept length is **not** the cause — both engines accept ~equally (same drafter + greedy = longest-prefix match). The cause was the **per-request serial draft loop**: `execute_dflash_draft` (`dflash_lane.rs`) called `draft_logits` once per request in a `for` loop, while verify runs ONE batched target forward over all spans. Per-step draft timing by batch size, **before vs after batching** (5090, instrumented):

| batch | draft serial | draft batched | verify | draft serial scaling | draft batched scaling |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.56 ms | 1.55 ms | 8.1 ms | 1.00× | 1.00× |
| 8 | 12.45 ms | 3.17 ms | 9.6 ms | 7.99× | 2.04× |
| 16 | 24.86 ms | **5.62 ms** | 13.6 ms | **15.96×** | **3.62×** |

The serial draft scaled **exactly linearly** (`draft_x` 16.00 at batch 16); batched draft scales sub-linearly (3.62×). At batch 16 the draft drops from 65% of the step to ~29%, and step time roughly halves → c16 throughput doubles.

**Why it was launch-bound, not compute-bound.** A skip-attention A/B (short-circuit `single_prefill` to a cheap copy, keep every other kernel) barely moved the serial draft — batch 16: 24.36 ms vs 24.81 ms, so attention compute is **<2%**. The 24.8 ms was almost entirely per-request serial **kernel-launch overhead**: each request's draft is ~90 tiny 16-token kernels (5 layers × 35 dense GEMMs + varlen copy/rope/KV), compute ≈ 0.

**The fix that landed: batch the whole draft forward (N requests in one pass), killing the N× launch.** `draft_logits_batched` (`dflash.rs`) runs the dense ops (rms_norm / GEMM / silu / MLP / embedding / logits) **once over an N×block buffer** — free, since cuBLAS takes any M and the ops are already row-batched (N×35 → 35 launches, no CUDA-kernel change). The varlen ops (tail concat / k·v GEMM / rope / KV-copy / attention) stay a per-request loop slicing the batched buffers at `row_offset = i·block_size` (the two DFlash-exclusive ops `dflash_qk_norm_rope_into` / `single_prefill_nhd_noncausal_into` advance the device pointer to the slice). A lane-level `DFlashBatchScratch` (sized for the largest batch bucket) replaces the per-request scratch — folding in the reservation-reclaim follow-up. Losslessness gate still passes (bf16 tie-flips only).

#### Single-stream gap (c1): launch-bound draft, not accept

c1 trails vLLM (237 vs 278) and batching doesn't help it — batch=1 has nothing to batch. The cause is **not accept**: with the *same* drafter and spec config (vLLM `--speculative-config '{"method":"dflash",…,"num_speculative_tokens":16}'`), measured greedy accept is **9.1% (ours) vs 8.85% (vLLM)** — mean 1.29 vs 1.42 draft tokens/step, and our pos-0 accept (60%) is actually higher. 9% is the drafter's floor on sharegpt free-text (bimodal: structured spans accept up to 15, free text accepts 0), shared by both engines.

The real gap is in **step time**: normalized per output token, ours is 4.22 ms vs vLLM's 3.60 ms. c1 step = draft 1.55 ms + verify 7.9 ms. Verify is memory-bound (one 4B target forward, reads ~8 GB of weights) — irreducible and the same for vLLM. But the draft's 1.55 ms is **pure kernel-launch** (85 tiny 16-token kernels, ~18 µs/launch CPU enqueue, compute <2%), while vLLM's draft is CUDA-Graph captured (~0.5 ms). That ~1 ms/step is the c1 gap.

**Next step: CUDA-Graph the draft forward.** openinfer's whole spec path (`SpeculativeDraft` / `SpeculativeVerify`, `executor.rs`) is *not* CUDA-Graph captured, whereas base decode is (`execute_decode` split-graph cache). Capturing the draft is an architecture-level change — the per-request draft KV (`DFlashLayerCache`) needs pointer-stable buffers, the variable context length (0–16) needs a fixed/padded window, plus capture/replay wiring — predicted to drop draft@batch1 1.55→~0.3 ms (c1 → ~272, ≈ vLLM) and draft@batch16 5.6→~1.5 ms (c16 → ~2200). Tracked as its own PR after this one lands.

The EAGLE proposer trait is still deferred to when EAGLE actually lands (see "no proposer trait yet" above); both the batched draft and the future CUDA-Graph draft are DFlash-internal changes behind the unchanged `DraftPlan→DraftResult` seam, so neither is thrown away by EAGLE.

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
- **Second proposer (EAGLE) → introduce the trait then.** A proposer trait wants two real methods to fix its shape; DFlash stays concrete until EAGLE lands. The batched-draft perf work (Performance §) is orthogonal — it's a DFlash-internal change behind the unchanged `DraftPlan→DraftResult` seam, so it can land before or with the trait without being redone, and EAGLE's autoregressive attention won't reuse DFlash's block attention anyway.
- **File size.** `executor.rs` (~2.8k lines) and `scheduler/tests.rs` (~1.2k lines) exceed the 1k guideline; the spec arms in `execute_step_on_lane` are candidates to move into `executor/spec.rs` in a follow-up.
- **5090 validation — done** for correctness, single-stream, and concurrent throughput: `hf_golden_gate` passes (bs1 / batched / cuda-graph / tp2), the losslessness gate passes (before and after the draft batching), single-stream A/B is 1.56×, and lm-eval gsm8k parity holds (above). Concurrent throughput is **fixed** (Performance §): the serial draft inverted the win (c16 −59%), batching the draft restores **c8 1346 / c16 1868 — both beating vLLM**. Single-stream c1 (237 vs vLLM 278) still trails on launch-bound draft overhead; CUDA-Graph draft is tracked next.
- **Frontend `seed`.** `/v1/completions` 400s on a per-request `seed` instead of accepting-and-ignoring it under greedy — surfaced while wiring lm-eval; orthogonal to spec decode, worth a small separate fix.
- **Reclaim the per-request reservation.** The fixed term is dominated by the block-sized scratch billed per decode-batch slot (~6.5 MiB × 256 ≈ 1.6 GiB), most of it the transient logits buffer (`vocab × block`). The per-token term carries the context/pending scratch, which today reallocs to prompt length on the first draft and never shrinks (sized for `[0, context_len)` but only `[committed_len, +block]` is live). Both are billed conservatively because they were per-request. **Partly reclaimed by the batched-draft PR:** the per-request scratch is gone — `DFlashBatchScratch` is allocated once at lane level, sharing the transient logits/dense buffers across the whole batch instead of per request. The reservation accounting (`from_config`) still bills the old per-request upper bound (a safe over-estimate, so no OOM); tightening it to the now-smaller lane-level footprint is the remaining follow-up.

### Review blockers (correctness/usability, independent of the perf work)

Issues surfaced in PR review, largely independent of the draft batching (#2's DFlash wrappers were fixed alongside it):

1. **Unified path silently skips DFlash readiness.** `StepCommand::Unified` (`executor.rs`) captures no DFlash hidden state, and only the plain `execute_prefill` post-step marks a request draft-ready (`executor.rs` ~1766). So when active + pending fuse into a Unified step (`scheduler/plan.rs` — the normal mixed-load path), greedy requests routed through Unified never become draft-ready and never recover: DFlash silently no-ops for them. No wrong tokens, but the feature quietly disables itself under mixed load. Crash-early or capture-in-Unified, don't degrade silently.
2. **Stream-override race (partly fixed).** The DFlash `qk_norm_rope` / `single_prefill` wrappers (`attention.rs`) now use `active_cu_stream(ctx)` (fixed in the batched-draft PR). `copy_hidden_rows_into` (`elementwise.rs:209`) still uses `ctx.stream.cu_stream()` instead of the repo convention (`tensor.rs:43`) — under Green-Context / split-stream decode overlap this remains a planted race.
3. **`gemm_lt_pin_tune` is not a real warmup.** It only pins the heuristic (`linear.cu:497`); it never executes a `cublasLtMatmul` the way the old `gemm_lt_tune_cuda` (`linear.cu:431`) did, so the first real matmul can land inside CUDA-graph capture. `batch_invariance_decode_gemm_graph` backstops it, but the warmup should actually run the matmul.
