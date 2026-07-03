# GLM5.2 DP8 scheduler (PR5b)

> **TL;DR:** The bs=1 rejecting/greedy coordinator is replaced by a DP8 lock-step scheduler: every rank holds the full non-expert stack (replicated at load, ~19.6 GiB/rank on top of its 85 GiB expert slab) and serves **one request per rank**; every global step all 8 ranks run the full 78-layer forward simultaneously, each dispatching exactly one token (real or padding) into the per-MoE-layer DeepEP collectives (`GLM52_DECODE_GLOBAL_TOKENS` 1 → 8). DP1 is now just the `active_ranks = 1` special case of the same protocol. Per-request decisions live in `Glm52SlotState` as a pure state machine (unit-tested, no fakes); the coordinator is a thin channel shell. **Verified locally: build + clippy clean on both feature configs, 6 scheduler unit tests green. jz-38 e2e (single-request byte-parity vs the PR5a record + multi-request concurrency) is the open next step.**
>
> **Last touched:** 2026-07

## Why this shape

- The DeepEP shim's capacity formula was always baked for all 8 ranks dispatching (`decode_worst_expanded_tokens`); the PR4/PR5a path used it at `g=1` with rank 0 as the only dispatcher. Writing the scheduler directly in the DP8 shape avoids a second scheduler rewrite when concurrency arrives — the kimi `DpCoordinator` precedent, minus its BlockPool/prefix-cache/batching machinery that GLM5.2 bring-up doesn't have yet.
- Non-expert replication costs ~19.6 GiB × 7 extra ranks at load time and ~105 GiB/rank resident (fits H200 141 GB, proven by the PP8-era loads which had the same per-GPU expert-layer count).

## Protocol

One global step = all 8 ranks each forward exactly one token through the full model:

- **Active rank, prompt not fully fed**: feed `prompt[fed]` at position `fed` (prefill rides decode); the model output is discarded except for the last prompt token's step, which yields the first generated token. Different ranks' prefill/decode advance concurrently in the same lock-step.
- **Active rank, decoding**: feed `last_token` at `prompt_len + completion − 1`.
- **Idle rank**: feed the padding input (token 0, position 0). Its KV/index-cache writes land in the idle rank's own dead cache slots and are overwritten when a request is admitted there; its MoE dispatch adds rows to expert segments but every chain kernel is row-independent and combine addresses slots per source token, so real tokens are unaffected.

`GLM52_DECODE_GLOBAL_TOKENS = GLM52_EP_RANKS` stays the single protocol definition; `bound_rows` becomes 2080 (vs 512 at `g=1`, capacity 10240), and the grouped-GEMM metadata kernel still device-traps if a real segment ends past it.

## What changed

| area | change |
|---|---|
| `weights.rs` | every rank's manifest includes the non-expert names (`loads_non_expert` field deleted — always true); coverage validation unchanged (set-union) |
| `model.rs` | `Glm52Rank0Model` → `Glm52RankModel`, `Glm52ExpertRankModel` deleted (every rank runs `decode_step`) |
| `runner.rs` | `Rank0Step`/`ExpertStep` → one `Step { token, position }` command; every rank builds the full model |
| `scheduler.rs` (new) | `Glm52SlotState` pure state machine (`next_input`/`advance`) + `run_dp8_coordinator` lock-step shell; fast-reject at intake, FIFO queue, one request per free rank |
| `lib.rs` / server | launch contract `--dp-size` 1 → 8 (or omitted); coordinator swap |

Semantics preserved from the bs=1 coordinator: EOS suppressed but counted, EOS outranks the length cap, greedy-only/no-logprobs/no-echo/no-LoRA rejections, `prompt + max_tokens − 1 ≤ 4096` cap, fatal teardown on any step error (a failed step permanently desyncs the EP8 collective group — see `fail_step`). New behavior: a client disconnect frees the rank (send failure during decode, `TokenSink::is_closed` probe during prefill — the bs=1 coordinator decoded to completion into a dead channel). On a failed step the coordinator joins ALL ranks and logs every rank's error before tearing down — the first rank to answer often reports the ~100 s DeepEP device-timeout trap, not the root cause (toxic-review finding).

## Verification

- `cargo test -p openinfer-glm52 --features glm52 --lib` — 9 pass (6 new scheduler state-machine tests; local run needs `LD_LIBRARY_PATH=/data/opt/nccl-2.30.4/lib` and `OPENINFER_NCCL_ROOT=/data/opt/nccl-2.30.4`).
- clippy clean (no new lints) with and without the `glm52` feature; dead-code warning count identical to main (15, all pre-existing EP1-path).
- toxic-review pass done: no fatal findings; the root-cause-masking hard finding (sequential recv dropped the failing rank's error) and the prefill-disconnect zombie are fixed; padding-row bit-parity flagged as a stronger claim than PR3 row-isolation → must be proven by the e2e gate, not asserted.
- **Open (jz-38 8×H200) — the gate list, in order:**
  1. Single request byte-parity vs the PR5a record ("… Paris. Distance from Paris to Lyon is 391 km…", 133 steps) — proves padding ranks' extra expert-segment rows don't change real tokens' bits.
  2. 8 identical prompts concurrently → all outputs identical (only e2e proof that the replicated non-expert stacks + independent caches are correct; ranks 1..7 never ran a real request before).
  3. bs=1 ms/step incl. p99 vs the 46–50 ms PR5a baseline — `bound_rows` 512→2080 rescans 4× rows in the quant/SiLU/metadata chain; "≈ unchanged" is a hypothesis, measure it.
  4. Slot reuse: finish → padding steps → new request on the same rank must match a cold solo run byte-for-byte.
  5. Disconnect mid-decode and mid-prefill → rank freed, next request on it correct.
  6. Mixed 2–8 clients, staggered lengths → per-request determinism, aggregate throughput ≈ linear in active ranks.
  7. Invalid requests mixed into live traffic → no effect on valid streams.
  8. Teardown: close the submit channel mid-decode → clean exit, no 100 s hang; pending queue gets Error events.

## Next

- jz-38 e2e gate above, then PR.
- PR5c: whole-step decode CUDA-graph capture on top of this stable per-rank shape (kills the ~46% launch-gap wall + residual MLA/indexer `alloc_zeros`); coordinate with PR #533's zero-alloc MLA scratch after its rebase.
