# Qwen3-4B n-gram speculative decoding (design)

**TL;DR**: Draft-model-free speculative decoding for Qwen3-4B. An n-gram
(prompt-lookup) proposer suggests `K` continuation tokens from the running
context; the target model verifies them in one forward pass and commits the
longest greedy-agreed prefix plus one model token. Greedy verification is
**lossless** — output is bit-for-bit identical to plain greedy decode, just
fewer forward passes on repetitive / structured text (code, quoting, JSON).
The KV layer (kvbm) already supports speculative scheduling natively, so
rejected drafts need no manual rollback.

Last touched: 2026-06 · Status: **end-to-end implemented and gated behind a
(default-off) `SpeculativeConfig`: proposer, greedy acceptance, KV speculative
pass-throughs, the executor verify forward (`execute_speculative`,
GPU-validated lossless), and the scheduler serving-loop wiring
(`speculative_decode_step`, mock-tested). Remaining: a public config knob to
turn it on, and a speedup measurement.**

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

## Landed: GPU verify forward + executor step

- **`LocalQwen3Lane::execute_verify(verify_tokens, kv_view, lora)`** — reuses
  `batch_prefill(echo = true)` for per-position logits over the existing paged
  KV, then a GPU argmax (`argmax_batch_bf16_into`) returns one token per verify
  position. Only the position ids cross to host, never the `[vocab, n]` logits
  (vLLM-style). Kept additive: `batch_prefill` / `batch_decode` are untouched.
- **`StepCommand::SpeculativeVerify` + `Qwen3Executor::execute_speculative`** —
  `schedule_speculative(K + 1)` → `prefill_view(1 + K)` → verify step → argmax →
  `accept_greedy` → `apply_speculative`, returning the committed tokens. The
  rank-worker channel carries it (TP-safe).

## Validated

- **Lossless on GPU** — `tests/ngram_speculative.rs` runs real Qwen3-4B and
  asserts greedy n-gram speculative decode is token-identical to plain greedy
  decode (prefix cache off; repetitive prompt). Confirms the full pipeline
  (proposer → schedule_speculative → verify forward → accept_greedy →
  apply_speculative) end-to-end.

## Landed: scheduler serving-loop wiring

- `ActiveRequestState` carries `token_history` (prompt + generated), maintained
  on promote and each committed decode token.
- `scheduler/speculative.rs::speculative_decode_step` proposes per active
  request, runs `execute_speculative` (or a single decode when no draft), and
  streams the committed tokens with stop / max-token handling — isolated from
  the one-token-per-step plan/resolve/effects pipeline. `scheduler_loop` routes
  pure-decode ticks through it when `SpeculativeConfig.enabled`.
- Mock-tested (`FakeExecutor::execute_speculative`): streams every committed
  token + advances state; commits past `max_tokens` truncate, finish, retire.

## Enabling it

`scheduler_loop` builds the config via `SpeculativeConfig::from_env()`
(default-off). Operational switch:

- `OPENINFER_QWEN3_NGRAM_SPEC=1` — turn it on.
- `OPENINFER_QWEN3_NGRAM_SPEC_TOKENS=K` — draft count (default 4).
- `OPENINFER_QWEN3_NGRAM_SPEC_MAX_NGRAM=N` — longest matched suffix (default 3).

Only the non-LoRA `scheduler_loop` reads it; the unified prefill+decode tick
still uses plain decode.

## Remaining work

1. **First-class config knob**: env var is the current switch; thread a typed
   knob through `start_qwen3*` / `start_engine*` (and a server flag) so it shows
   up in the engine config rather than the environment.
2. **Speedup measurement** (initial numbers below — generalize to realistic
   prompts and the scheduler loop).
3. **Batched / vLLM-style verify** (perf): fold the verify tokens into the
   unified batched forward (FlashInfer varlen) with a GPU rejection step,
   instead of the current per-request `batch_prefill`-based verify.

## Measured speedup (best case)

`tests/ngram_speculative.rs::ngram_speculative_speedup` (ignored; needs GPU +
weights) times greedy vs. speculative on Qwen3-4B (eager, single request, 192
tokens). On the perfectly periodic synthetic prompt:

| metric            | greedy   | speculative |
| ----------------- | -------- | ----------- |
| forward passes    | 191      | 39          |
| ms / token        | 9.99     | 2.52        |
| accepted / verify | —        | 5.00 (max with K=4) |
| wall-clock        | 1908 ms  | 481 ms (**3.96x**) |

This is the ceiling: the prompt is exactly periodic so every draft is accepted.
Real prompts accept a fraction of drafts, so expect smaller wins; the benchmark
exists to track acceptance-rate / TPOT as the proposer and verify path evolve.
Run: `cargo test -p openinfer-qwen3-4b --release --test ngram_speculative \
ngram_speculative_speedup -- --ignored --nocapture` (`OPENINFER_BENCH_TOKENS`
overrides the 192-token default).

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
