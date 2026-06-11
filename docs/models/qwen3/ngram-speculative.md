# Qwen3-4B n-gram speculative decoding (design)

**TL;DR**: Draft-model-free speculative decoding for Qwen3-4B. An n-gram
(prompt-lookup) proposer suggests `K` continuation tokens from the running
context; the target model verifies them in one forward pass and commits the
longest greedy-agreed prefix plus one model token. Greedy verification is
**lossless** — output is bit-for-bit identical to plain greedy decode, just
fewer forward passes on repetitive / structured text (code, quoting, JSON).
The KV layer (kvbm) already supports speculative scheduling natively, so
rejected drafts need no manual rollback.

Last touched: 2026-06 · Status: **design + building blocks landed; GPU verify
forward + executor/scheduler wiring pending.**

## Why this is cheap to add here

Three of the four pieces already exist or are trivial:

- **Proposer** — `openinfer-qwen3-4b/src/ngram.rs` (`NgramProposer`). Done, unit-tested.
- **Acceptance** — `openinfer-qwen3-4b/src/speculative.rs` (`accept_greedy`,
  `num_accepted`, `SpeculativeConfig`). Done, unit-tested.
- **KV scheduling / rollback** — kvbm's `SchedulableSequence` already implements
  `schedule_speculative` / `apply_speculative` with **automatic LIFO release of
  the blocks pre-allocated for rejected drafts**. Exposed on `RequestKv`
  (`openinfer-kv-cache/src/pool.rs`). Done, unit-tested.
- **GPU verify forward + orchestration** — the only remaining work (below).

## Per-step data flow (single request, greedy)

Layering mirrors vLLM V1 (proposer in the runner/scheduler layer that owns
token history; the executor reserves KV, runs the target forward, and accepts):

```
scheduler owns the request's token history (prompt + generated)
  1. drafts = NgramProposer.propose(history)        # K candidates; empty -> plain decode
  2. verify_inputs = [d0, c0, .., c_{K-1}]           # d0 = last committed token (dangling)
  3. RequestKv.schedule_speculative(K + 1)           # room for drafts + bonus token
  4. argmax[0..K] = verify_forward(verify_inputs, prefill_view(1 + K))
  5. committed = accept_greedy(drafts, argmax)        # m accepted drafts + 1 model token
  6. RequestKv.apply_speculative(committed)           # kvbm releases rejected blocks
  7. scheduler appends committed to history; applies stop / max_tokens;
     committed.last() becomes the next step's d0
```

### Token / KV accounting (matches kvbm's verified contract)

- The verify forward computes KV for `1 + K` positions (`d0` + `K` drafts) at
  `base_pos = kv_position`. Structurally this is a prefill of `1 + K` tokens at
  the current position, so the `KvView` is `RequestKv::prefill_view(1 + K)` and
  the forward reuses the existing paged-prefill attention path.
- `argmax[i]` is the model's greedy token *after* consuming `verify_inputs[i]`:
  `argmax[0]` follows `d0` (the true next token), `argmax[i]` follows `c_{i-1}`,
  `argmax[K]` is the bonus continuation. This index convention is exactly what
  `accept_greedy(proposed = drafts, target_argmax = argmax)` expects.
- `apply_speculative(committed)` advances `kv_position` by `committed.len()`
  (`m + 1`); kvbm LIFO-drops the over-allocated blocks. `committed.last()` (the
  model token) becomes the new dangling token. `schedule_speculative(K + 1)`
  guarantees `m + 1 <= K + 1`.

### Why it is lossless

`accept_greedy` only keeps the prefix where `draft[i] == argmax[i]` and then
appends one of the model's own tokens, so the committed sequence is identical
to what plain greedy decode would have produced one token at a time. This gives
a free correctness oracle: **speculative-on must equal speculative-off** for the
same prompt under greedy params.

## Remaining work (the GPU verify forward + wiring)

1. **`verify_forward(verify_tokens, kv_view) -> Vec<u32>`** on `Qwen3Model` — a
   standalone method (kept separate so it does not pollute `batch_prefill` /
   `batch_decode`). Internally reuses the paged-prefill attention + the existing
   all-position-logits path, then takes a per-position argmax on the GPU via the
   existing batch sampler (`select_batch_tokens_into` with greedy params over
   the `K + 1` positions). Only the `K + 1` token ids are copied to host — never
   the `[vocab, K + 1]` logits — mirroring vLLM's GPU-side rejection sampler.
2. **`StepCommand::SpeculativeVerify` + `execute_speculative`** in the executor:
   `schedule_speculative` → `prefill_view(1 + K)` → run verify step → argmax →
   `accept_greedy` → `apply_speculative` → return committed tokens.
3. **Scheduler wiring**: invoke the proposer over per-request history when
   `SpeculativeConfig.enabled`; commit loop applies stop / max-token handling to
   each committed token. (vLLM proposes the *next* step's drafts right after a
   forward and stores them on the request for pipelining; the first cut may
   propose at step start for simplicity.)

## Scope / deferred

First cut is **single-request, greedy, non-CUDA-graph**. Deferred: batched
speculation (ragged verify across requests), sampling (non-greedy) acceptance,
CUDA-graph-captured verify, interaction with the unified prefill+decode step,
and pipelined ahead-of-time proposal.

## Open questions for review

1. `verify_forward` as a standalone model method (preferred) vs. reusing
   `batch_prefill(echo = true)`.
2. Confirm the GPU-side argmax via `select_batch_tokens_into` over the `K + 1`
   verify positions is acceptable (vs. a dedicated argmax kernel later).
3. Proposer placement in the scheduler layer (owns token history), matching
   vLLM's runner / `request.spec_token_ids` split.

## Prior art

- vLLM V1 n-gram / prompt-lookup spec decode (`NgramProposer` in the runner,
  GPU rejection sampler, scheduler reserves KV for `k` draft tokens).
- kvbm `SchedulableSequence` speculative lifecycle (`schedule_speculative` /
  `apply_speculative`, LIFO block release).
