# GLM5.2 batched decode step (dynamic-batching D1)

> **TL;DR:** Execution record of the batched-executor substrate (PR-D1): every rank's decode step now forwards a fixed batch of `GLM52_MAX_BATCH_PER_RANK = 8` token rows (request tokens in occupied slots, padding rows elsewhere), each slot owning a disjoint 4096-token region of the paged MLA/index-K caches; `GLM52_DECODE_GLOBAL_TOKENS` is now 64 (DeepEP shim cap is 128/rank — zero shim changes). jz-38 8×H200: **all 4 oracle gates green, solo output byte-identical to the PR5c bs=1 record, determinism/8-way row-isolation/mixed-tier/disconnect/SIGTERM all PASS.** Measured pad-row cost: **39.2 ms/step single request (1 real + 7 pad rows) vs PR5c's 22.5** — the plan's >5% threshold fires, so **D2 (multi-slot admission) must add {1, 8} batch-bucket graphs** instead of the single fixed shape. The scheduler still admits one request per rank; D2 fills the other 7 slots with real requests, at which point the same ~40 ms step serves up to 64 rows.
>
> **Last touched:** 2026-07

## What changed

One commit on `feat/glm52-batched-decode-step`: 27 files, +1060/−472.

- **CUDA (4 files):** `glm52_moe_gemv.cu` gains a weight-stationary batched GEMV (`kBatchedGemvBatch = 8`, one warp per output row carrying 8 accumulators — the weight is still read once and each row's FMA order is identical to the m=1 kernel, so per-row bit-parity holds by construction; the launcher rejects any other batch, which is the crash-early guard against a Rust-const drift). `glm52_mla_assembly.cu` batches query-assemble/cache-pack (per-token rope rows, per-token cache slots) and adds a `ckv_split` kernel replacing the dtod slice copies. `glm52_indexer_rope.cu` and `flashinfer_norm.cu`'s LayerNorm gain a tokens/rows grid dimension (one CTA per row — per-row bit-identical).
- **Already batch-ready, no kernel changes:** FlashMLA sparse (batch cap 128), DeepGEMM MQA AOT (runtime batch ≤ 32), FlashInfer top-k, `local_topk_to_slots`, index-K cache insert, MoE quant/SiLU, router, fused add+norm, argmax split FFI, and the whole DeepEP shim (`kDecodeMaxTokens = 128/rank`; `bound_rows` grows 2080 → 2528 at g=64 by the existing formula).
- **Model crate:** every scratch struct sized by the batch; absorb/back GEMMs and lm_head ride the cuBLAS n dimension (the col-major `[N, T]` output IS the row-major `[T, N]` layout the next op consumes); per-row rope rows come from an `embedding_rows` gather over the resident rope table; slot b's cache tokens live at `[b*max_model_len, (b+1)*max_model_len)` with a static block table (4096 at the time; launch-time VRAM-derived since #579).
- **Scheduler:** unchanged one-request-per-rank; the request rides slot 0 and slots 1..8 carry the padding row (token 0, position 0) into that slot's own dead cache region — the same row-isolation argument as PR5b's idle-rank padding, now per-slot.

## jz-38 gates (2026-07-03, `glm52_d1_gates.sh`, log `glm52_d1_gates.log`)

| gate | result |
|---|---|
| oracle: MLA full tier / short tier / layer (grouped+gemv) / EP8 layer | all PASS |
| determinism ×2 (24-tok + 128-tok) | PASS |
| solo output vs PR5c v7 record (short + long) | **byte-identical** — cuBLAS n=8 and FlashMLA batch=8 did not move the greedy outputs |
| tier-crossing 320-tok ×2 + prefix + post-short | PASS |
| 8-way identical prompts (row isolation across ranks) | PASS |
| 8-way 128-tok + mixed tiers | PASS |
| disconnect mid-stream → next request | PASS |
| SIGTERM mid-decode | PASS (no DeepEP hang) |

## Measured pad-row cost (the D2 decision datum)

| shape | ms/step | note |
|---|---|---|
| PR5c record (1 row/rank, global 8) | 22.5 | single request |
| D1 fixed batch (8 rows/rank, global 64), 1 real request | **39.2** (×5 runs, dead stable) | +74% — every step pays 64 global rows |
| D1, 8 concurrent (one per rank) | 40.7 (189 tok/s aggregate) | same step shape; PR5c served this at 22.3 / 346 tok/s |

+16.7 ms for 7 pad rows ≈ +2.4 ms/row: the weight stream amortizes, but the row-proportional terms (batched-GEMV FMA lanes, FlashMLA padded index walks ×8, indexer MQA rows ×8, MoE `bound_rows` 2080→2528) do not. **Decision: D2 adds {1, 8} batch buckets (2 tiers × 2 buckets = 4 graphs, lazily captured — the tier-crossing capture-mid-serving argument already holds), so an idle server keeps the 22.5-class step and the 39–40 ms step is only paid when the slots hold real requests.** Projected full-slot upside: ~40 ms serving 64 rows ≈ 1.6k tok/s aggregate — to be measured in D2, not claimed.

## Next step

**Landed as D2** — multi-slot admission + {1, 8} bucket graphs, solo back to 22.4 ms/step, scaling table and soaks measured, and the slot-k parity acceptance item (pinned slot-3/7, both buckets) proven. See `continuous-batching.md`.
