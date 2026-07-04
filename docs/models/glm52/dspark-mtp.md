# GLM5.2 DSpark speculative decoding (greedy)

**TL;DR:** Replace the token-by-token decode cadence with DSpark speculative rounds, using the community `RedHatAI/GLM-5.2-speculator.dspark` checkpoint (NOT the native MTP layer — its tensors stay dropped at load). The draft is a **qwen3-architecture** 5-layer dense backbone at GLM's hidden 6144 (64-head MHA, head_dim 64, q/k-norm) + rank-256 Markov head — the same DSpark shape we already ship for Qwen3-4B (`docs/models/qwen3/dspark-integration.md`), so `dspark.rs`/the Markov kernel/the verify-accept seam port over. The verify step maps 1:1 onto the D2.5 multi-row decode bucket: **span 8 = anchor + 7 drafts = one bucket-8 step where all 8 rows are consecutive positions of ONE slot** — no new attention kernels, no graph shape changes, just relaxed row→slot mapping. Checkpoint facts verified on jz-38: draft `embed_tokens`/`lm_head` are **byte-identical to the target's** (sha256-compared) → skip loading both (~3.8 GB loaded per rank instead of 7.6). Milestones: **M1 span-steps DONE** (all jz-38 gates green: 1621-token prompt byte-identical to main's token-by-token walk, **TTFT 35.8 s → 9.25 s = 3.9×**, solo decode unchanged, bucket-8 solo step measured 45.6 ms → projected spec solo ≈ 1.8×), **M2** draft backbone + Markov propose (rank-local) — next, **M3** verify/accept round loop + c1 A/B. Greedy only; confidence head parsed but unused (Phase 2).

Last touched: 2026-07

## Checkpoint (verified on jz-38, 2026-07-04)

`/data/models/GLM-5.2-speculator.dspark` (7.6 GB `model.safetensors`, BF16, MIT). Trained with
[vllm-project/speculators](https://github.com/vllm-project/speculators) (online, hidden states
streamed from a live vLLM GLM-5.2-FP8 server) — **not** DeepSpec; anchor/block layout must be
pinned against the speculators source, not the DeepSpec repo (see Open questions).

| fact | value |
| --- | --- |
| draft backbone | `model_type: qwen3`, 5 layers, hidden **6144** (= target hidden), 64 q-heads / 64 kv-heads (MHA), head_dim 64, q/k-norm `[64]`, inter 12288, rope_theta 8e6, rms_eps 1e-5 |
| block / drafts | `block_size=8`, deployment `num_speculative_tokens: 7` → verify span 8 (= our decode bucket 8) |
| aux target layers | `[8, 23, 39, 55, 70]` → `fc.weight [6144, 30720]` context projection + `hidden_norm` |
| Markov head | `markov_w1/w2 [154880, 256]` (~158 MB), `markov_head_type: vanilla` |
| confidence head | `proj [1, 6400]` (6144+256) + bias — Phase 2, parsed but not loaded |
| embed / lm_head | `[154880, 6144]` each, **sha256 == target's `model.embed_tokens.weight` / `lm_head.weight`** → skip, reuse target head (saves 2×1.9 GB/rank) |
| mask token | 154856 |
| quality (their val) | mean accepted length **3.967** (incl. bonus), per-position accept 0.83→0.46 across 7 positions |

Loaded per rank: 5 backbone layers ≈ 3.3 GB + fc 0.38 GB + markov 0.16 GB ≈ **3.8 GB bf16**,
replicated on all 8 ranks (draft is dense + rank-local; DP over slots, no collectives).

Draft KV: 5 layers × 2 × 4096 dims × bf16 = **80 KiB/token** → 320 MiB per 4096-token slot,
2.6 GiB/rank at 8 slots. Fits H200 alongside the FP8 expert bank.

## Why the D2.5 bucket infra is exactly the verify step

The engine is decode-only ("prompt tokens ride the decode path one position at a time" —
`scheduler.rs`), and a step already carries **per-row** `(token, position)` inputs and returns
**per-row** argmax. A DSpark verify of one request is a step whose 8 rows are
`[(anchor, p), (d1, p+1), …, (d7, p+7)]` — all on the same slot:

- **KV/causality**: per layer, the KV-write kernel covers all rows before attention launches;
  row k's `seq_len = p+k+1` includes rows `< k` and excludes rows `> k`. Same for the indexer
  k-cache. No new kernels.
- **Block table**: the partial-bucket dtod-gather path already maps row→slot arbitrarily; verify
  gathers slot s's block-table row 8 times. Only the full-bucket identity assert (a decode-mode
  invariant) is bypassed in span mode.
- **Rejection rollback is free**: rejected rows' KV entries sit above the committed length —
  `seq_lens` caps what attention reads, and the next step overwrites those positions. Paged-KV
  overwrite semantics replace qwen3's explicit KV transaction.
- **DeepEP protocol**: a verify step is a normal bucket-8 step (global rows = 8×8); ranks without
  a verifying slot carry padding rows, exactly like today.
- **Bucket ladder = span ladder**: buckets {1,2,4,8} let the scheduler verify spans {1,3,7}+anchor,
  or split a rank's 8 rows across 2 slots (span 4 each) when multiple requests share a rank. The
  D2.5 planner generalizes from "one row per slot" to "rows = Σ per-slot spans ≤ bucket".

New step surface (contained): allow repeated slots with strictly increasing per-slot positions,
per-row block-table gather for bucket 8, and an **aux-hidden capture buffer** — 5 dtod copies of
the residual stream after layers {8,23,39,55,70} into a pre-allocated `[8, 30720]` buffer inside
the whole-step graph (~480 KB/step; pointer-stable, graph-safe).

## Round cadence (M3)

Per spec round, per rank (all ranks in lock-step on the global step; draft is between steps):

1. **Verify** (global step, bucket 8): feed span, read 8 argmax rows + capture buffer.
2. **Accept** (host, coordinator): longest matching prefix + bonus/correction — port
   `openinfer-qwen3/src/speculative.rs` verbatim (`accept_greedy`, `build_verify_results`).
3. **Draft** (rank-local command, no collectives): append accepted rows' captured hidden to the
   request's pending context → fc projection → 5-layer backbone over `[anchor, mask×7]` →
   Markov sample loop (port `dspark.rs::sample_block` + the `markov_step_argmax` kernel at
   V=154880) → reply proposed span to the coordinator.

Coordinator sees two command round-trips per round (Step, then Draft); channel latency is µs
against a ~30–70 ms step. Draft forward ≈ 1–3 ms (5 dense layers × 8 rows + 8 sequential
rank-256 GEMV+argmax micro-steps). Draft/verify overlap is a later optimization, not Phase 1.

**Projected win (bucket-8 step now measured by M1 = 45.6 ms solo):** spec solo =
(45.6 ms verify + ~2–3 ms draft) per ~3.97 committed tokens ≈ **12.2 ms/token vs 22.4 plain ≈
1.8×**. Span 4 (bucket 4) is the counter-hypothesis to A/B in M3: fewer drafts per round but a
cheaper verify step. Measure, don't assume.

## Milestones

- **M1 — span steps for prompt ingestion** (no draft model) — **DONE, all jz-38 gates green
  (2026-07-04, `62defc6` + fairness fix)**. Measured (1621-token prompt, ×3 stable):
  **TTFT 35.2–36.6 s → 9.25 s (3.9×)**, 1621 → 203 steps; solo decode 22.5–22.6 ms/step
  (main 22.4–22.5, no regression); **bucket-8 one-real-slot step = 45.6 ms** — the number
  M3's round math needed. Correctness: 1621-token prompt output **byte-identical** span-path
  vs main's token-by-token walk (+ determinism ×2); short-prompt outputs identical to main;
  9-way mutual + ==solo; queue-80 / drain / mixed long-prefill-during-8-way-decode (full
  lengths; text diverges from solo = the known cross-shape FP association property) / SIGTERM
  all PASS. Toxic review: approve, no blockers; fairness finding fixed (leftover rows now
  round-robin across co-resident prefills — ascending-slot greed starved the later prefill
  to 1 row/step). Two notes left open: re-check the now-always-on full-bucket block-table
  gather (8×256 B dtod/step) at the next c64 bench; the repetitive gate prompt makes GLM
  emit EOS as its first token, and the frontend then reports `completion_tokens=0` with the
  prompt echoed in `text` — pre-existing (main identical), frontend accounting, not engine.
  TTFT is 3.9× rather than 8× because a bucket-8 step costs 45.6 ms vs 22.4 — which also
  says a span-4 verify (bucket 4) is worth measuring against span-8 in M3.
- **M2 — draft lane** (rank-local). Loader (skip embed/lm_head, reuse target head via the
  target's lm_head GEMM), backbone forward (port qwen3 `dflash.rs` shape: qk-norm + rope +
  KV-injection tail concat), Markov propose, aux capture plumbed from M1's capture buffer.
  Gate: draft forward runs and proposes plausible spans on jz-38 (no losslessness claim yet —
  greedy verify makes bad drafts a perf bug, not a correctness bug).
- **M3 — round loop + gates + A/B**. Scheduler slot states (prompt-span / spec-round), accept
  seam, cache-cap span truncation near 4096. Gates: greedy spec output == plain greedy output
  on the D2.5 gate prompts (cross-bucket near-tie divergence is a known FP property — same
  diagnostic treatment as D2), accepted-length telemetry. **A/B = the qwen3 dspark methodology
  (`docs/models/qwen3/dspark-integration.md` Results section), c1 only** (user call): `vllm
  bench serve --temperature 0 --ignore-eos` on sharegpt + code prompts against the HTTP server,
  spec vs plain output tok/s, plus the accepted-draft histogram from the server's debug accept
  trace. qwen3 Bug 1 lesson applies verbatim: the bench's default temperature is non-greedy and
  silently disables speculation — `--temperature 0` is mandatory.

## Layout pinned against speculators source (2026-07-04)

Read from `vllm-project/speculators` `src/speculators/models/{dflash,dspark}` (main). The four
bring-up-critical semantics, so M2 doesn't rediscover qwen3's Bug 2:

1. **Anchor-drop, 7 drafts, span 8.** `_build_base_config_kwargs` sets
   `GreedyTokenProposalConfig(speculative_tokens=block_size - 1)` with the comment "First block
   position is the anchor, not emitted during gen", and training zeroes the loss at block
   position 0 (`aligned_loss_mask[:, ::block_size] = 0`). Block input = `[anchor, mask×7]`;
   position k (1..7) predicts the token at `anchor_pos + k`. Our `drafts_start = 1` /
   `verify_span = block_size = 8` — the qwen3 *anchor-drop* path, NOT the DeepSpec anchor-first
   one. Crash-early if a config ever arrives with DeepSpec's `num_anchors` marker.
2. **Intra-block attention is bidirectional; context is strictly `< anchor`.**
   `create_anchor_block_mask_mod`: full-attention layers use `non_causal=True` within the block
   (matches qwen3's `single_prefill_nhd_noncausal_into`), and base-context visibility is
   `kv_base_pos < q_anchor` — the anchor's own captured hidden row is NOT attended. Consistent
   with the inference bookkeeping: the next round's anchor is the bonus token, whose hidden is
   only captured when it is fed in the next verify, so pending context always ends one row
   before the anchor.
3. **Block position ids = `anchor_pos + k`** (`get_base_indices_for_anchored_blocks`), rope
   from the flat `rope_parameters.rope_theta = 8e6`. Same committed+context position walk as
   qwen3; the draft never sees positions > 4096 under the current cap.
4. **Markov loop: prev(k) = block token k−1, applied at positions 1..7.** Training builds
   `prev_token_ids = [b0, b0..b6]` and biases the whole block, but position 0 is dropped, so
   inference is 7 sequential steps: `prev = anchor → bias → argmax(position 1) = d1 → prev = d1
   → …`. Vanilla head: `bias = markov_w2(markov_w1[prev])`, exactly our qwen3
   `sample_block`/`markov_step_argmax` math — port with the loop starting at position 1
   (anchor-drop) instead of 0 (anchor-first).

Remaining unpinned: the vLLM-side aux-capture tensor (residual stream after layer k, pre-norm)
is the family convention qwen3's byte-lossless gate already validated; confirm once against
`speculators` `launch_vllm.py`'s capture hook when wiring M2's capture buffer.

## Decisions

- **dspark over native MTP** (user call): the native MTP-layer tensors stay dropped at
  `build_model`. DSpark gives a trained, measured (3.97 accepted) drafter without bringing up
  MTP-layer training parity.
- **Greedy only** (Phase 1): our engine is greedy; `accept_greedy` is the shared seam. Markov
  head is required (it *is* the dspark draft quality); confidence head deferred.
- **Reuse target embed + lm_head**: proven byte-identical; draft logits = target lm_head GEMM
  over draft final hidden (bf16 dense GEMM `[8, 6144] × [6144, 154880]`, rank-local).
- **Verify span 8 default**: matches checkpoint training width AND the bucket ladder's top.
  Span is a scheduler knob, not a constant — smaller spans are the load/latency fallback.

## Next action

M2: draft lane — loader (skip embed/lm_head), qwen3-shape backbone forward + Markov propose at
V=154880, aux-hidden capture buffer (5 dtod copies at layers {8,23,39,55,70}) added to the step
graph. Pin the vLLM-side capture-tensor convention against `speculators` `launch_vllm.py` first.
