# GLM5.2 continuous batching (D2)

> **TL;DR:** Execution record of PR-D2: multi-slot admission (up to 8 requests per rank, least-loaded rank first) + {1, 8} batch-bucket graphs (2 attention tiers × 2 buckets = 4 whole-step CUDA graphs, lazily captured). **Solo decode is back to 22.4 ms/step** (D1's fixed 8-row batch was 39.2; the PR5c record was 22.5); **diverse-prompt scaling: c8 256 tok/s → c64 911 tok/s (62.8 ms/step p50 / 63.6 p99)**; poisson soaks (rate 4/8, random output 32–224) 600/600 zero errors. jz-38: all oracle gates green (now covering both global-token buckets); solo byte-identical to the PR5c and D1 records; 8/9/64-way, 80-way queueing, mixed tiers, disconnect, SIGTERM all PASS; pinned slot-3/7 parity PASS. **Known numerics property (root-caused, not a bug): the two buckets are distinct FP associations** — a request spanning a bucket switch can greedy-diverge from its solo replay at a near-tie (~token 215 on the probe prompt); deterministic given the same occupancy timeline. **Known perf cliff: c16 ≈ c8 throughput** (2 real rows/rank pay the full 8-row step) — an intermediate bucket is the obvious follow-up, to be justified by its own A/B.
>
> **Last touched:** 2026-07

## What changed

One commit on `feat/glm52-continuous-batching`: `openinfer-glm52` only, zero CUDA changes.

- **Scheduler (`scheduler.rs`):** one-request-per-rank → up to `GLM52_MAX_BATCH_PER_RANK = 8` requests per rank, each owning one slot (and that slot's disjoint 4096-token cache region). Admission is least-loaded rank first, lowest free slot; requests join/leave at step boundaries; beyond 64 active the queue holds. The admission and bucket decisions are pure functions over the occupancy (`admission_target` / `step_bucket`, unit-tested).
- **Batch bucket ({1, 8}):** the coordinator agrees a global bucket per step — 1 row per rank while every rank holds ≤ 1 request, the full 8-row batch as soon as any rank holds two. The MoE collectives require every rank to enter with the same global row count (8 vs 64), so the bucket is a coordinator decision, never per-rank. Attention tier stays per-rank (attention is rank-local).
- **Model (`model.rs`):** four whole-step CUDA graphs (2 tiers × 2 buckets), lazily captured — the mid-serving capture-safety argument from the PR5c tier crossing carries over. The 1-row bucket gets batch-1 FlashMLA plans, a batch-1 scratch arena (~1/8 the batch-8 arena), and a 1-row block table the prologue rewrites (dtod of the static table's row for the active slot) — the captured b1 graphs address whichever slot holds the request through device data, never a baked slot id. The 78-layer step body is shared verbatim by both buckets (`run_step_body`); only the plan, scratch, block table, and `global_tokens` differ.
- **Why zero CUDA:** the weight-only GEMV already supports exactly rows ∈ {1, 8} (the m=1 kernel and the D1 batched kernel), and every other kernel takes the row count as a launch parameter.

## Gates (jz-38 8×H200, 2026-07-04, `glm52_d2_gates.sh` / `glm52_d2_pin_slot.sh`, logs alongside)

| gate | result |
|---|---|
| oracle: MLA full / short tier / layer (grouped+gemv) / EP8 layer (replayed at BOTH global-token buckets, 64 and 8) | all PASS |
| solo determinism ×2 (24 + 128 tok), vs PR5c v7 record, vs D1 record | PASS — byte-identical to both |
| tier crossing 320-tok ×2 + prefix + post-short (b1 graphs) | PASS |
| 8-way identical (one per rank — stays bucket 1) | PASS |
| 9-way identical (first two-slot rank — bucket 8) | PASS |
| 64-way full occupancy | PASS (identical prompts, 1590 tok/s aggregate) |
| 80-way (> 64 slots → queueing) | PASS |
| mixed tiers + mixed lengths (12-way) | PASS |
| post-concurrency solo drain (slot reuse, back to b1) | PASS |
| disconnect mid-stream → 9-way traffic | PASS |
| SIGTERM mid-decode at bucket 8 | PASS |
| **pinned slot-3 (b1 bucket)** — hard-coded admission pin, byte vs slot-0 refs | PASS |
| **pinned slot-3 / slot-7 (forced 8-row bucket)** — the D1 acceptance item | PASS |

Slot-stride evidence beyond the pins: in the 9-way 320-tok probe, rank 0's two real rows (slots 0+1) and the single-row ranks produced **mutually byte-identical** outputs — co-resident real rows don't perturb each other and slot-1 addressing equals slot-0.

## The bucket-switch numerics finding

`GATE-BUCKET-CROSS` (a 320-tok request whose lifetime spans 1→8→1 bucket switches) showed the crossing request diverging from its solo replay while the 8 co-resident 128-tok requests stayed byte-identical to theirs. Root-caused with two discriminators:

- 8-way 320-tok (bucket 1 throughout): **identical to solo** → the b1 path and bucket switching corrupt no state.
- 9-way 320-tok (bucket 8 throughout): all 9 **mutually identical**, all differ from solo starting at output token 215 (position 221, inside the short tier); a strict-short-tier 240-tok replay reproduces it.

So batch-1 and batch-8 logits differ in bits from the start (cuBLAS picks different kernels for n=1 vs n=8; FlashMLA distributes its SM parts over 1 vs 8 rows → different split-combine association), and greedy first flips at a near-tie ~215 tokens in. D1's "byte-identical" evidence (24/128 tok) was argmax-visible equality inside the flip-free region — there was never a cross-batch bit-parity guarantee. This is the standard batch-variant-numerics property every batching engine has (vLLM included); the guarantees that DO hold, all gate-proven: determinism given the same occupancy timeline, slot/rank placement invariance within a bucket, and state integrity across switches.

## Measured performance (jz-38 8×H200, 2026-07-04)

Solo probe (133-step greedy, ×5): **22.4 ms/step dead stable** — D1 was 39.2, the PR5c 1-row record 22.5.

Closed-loop scaling, **diverse prompts** (random ~15-token prompts, 128-token outputs, non-streaming `/v1/completions`, `d2_engine_bench.py`):

| concurrency | ms/step p50 | ms/step p99 | aggregate out tok/s | note |
|---|---|---|---|---|
| 1 | 21.5 | 21.6 | 42 | 1-row bucket |
| 2 | 24.1 | 24.3 | 75 | 1-row bucket — per-step cost climbs gently with active ranks (more real dispatch in the collectives + request-boundary stagger) |
| 4 | 26.6 | 26.8 | 135 | |
| 7 | 29.3 | 29.7 | 215 | |
| 8 | 28.1 | 28.5 | 256 | last 1-row point; +25% over the identical-prompt 22.3 = routing diversity |
| **9** | **47.1** | **49.2** | **171** | **the cliff is non-monotonic**: the 9th request drops TOTAL throughput below c7 — everyone pays the 64-row step for 9 real rows |
| 12 | 49.0 | 49.5 | 220 | still below c8; break-even ≈ c14 |
| 16 | 51.1 | 52.3 | 280 | |
| 32 | 57.1 | 57.7 | 502 | |
| 64 | 62.8 | 63.6 | 911 | 22× the solo rate |

Poisson-arrival soak (random output 32–224, non-streaming): rate 4 → 600/600 ok, 493 tok/s, request e2e p50 8.8 s / p99 14.2 s; rate 8 (mild overload) → 600/600 ok, 782 tok/s, p50 15.6 s / p99 27.0 s (queueing as designed). Post-traffic solo parity holds.

Two measurement lessons this table encodes:

- **Identical prompts overstate MoE throughput badly at scale**: the 64-way identical-prompt gate reads 1586 tok/s vs 911 diverse — a 74% inflation (degenerate routing collapses the expert segments). The known ~7-15% figure from bs=1-per-rank measurements grows with rows per step. D1's "~40 ms full-slot step" projection was made with pad rows (token 0 → degenerate routing); the real diverse 64-row step is ~63 ms.
- **`vllm bench serve` (0.24) measurements against this server are client-distorted**: its streaming path reported ~39 ms/token at concurrency 1 where a streamed curl of the same request measures 23.0 (non-streamed 22.5 — so the server's SSE transport is fine, refuting an initial Nagle/TCP_NODELAY suspicion), and at concurrency 8 the client hung with every request complete and balanced on the server side (160/160 finished, all sockets drained, client parked in epoll). It also sends a non-zero default temperature, which the engine fast-rejects — pass `--temperature 0.0`. Engine numbers here therefore come from the non-streaming closed-loop harness; the vllm-bench client interaction is filed as a frontend follow-up.

## Next step

- **Bucket tuning**: the cliff is non-monotonic — c9 (171 tok/s) drops below c7 (215) and break-even with c8 is ≈ c14, so the 9–14 concurrency band is strictly worse than rejecting the 9th request would be. A middle bucket (e.g. 4: two more graphs + a `kBatchedGemvBatch` instantiation for batch 4 + one more FP-association regime) would serve c9–c32 at a ~30 ms-class step; justify with its own diverse-prompt A/B, not projection.
- Expert-GEMM M-tile work (#542's swapAB lead) re-opens now that real multi-row steps exist — re-measure at the 64-row diverse shape, where the 64-row M-tile is no longer 8× padding.
- P/D decode-node work (KV ingestion from a vLLM prefill, true paged block table, >4096 context) is the next campaign; prefill stays out of scope per the standing prefill-by-vLLM decision.
