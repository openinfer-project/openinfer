# GLM5.2 P/D disaggregation

> **TL;DR:** GLM5.2 target-only P/D is merged in OpenInfer #657: vLLM 0.24.0 TP8 performs prefill, OpenInfer EP8 performs decode, and PegaFlow transfers the 99 target-cache arenas per rank. The measured path preserves token output, admits every request at `suffix == 1`, matches vLLM on GSM8K and NIAH, and trades first-turn TTFT/raw throughput for stable TPOT tails and better later-turn TTFT. DSpark state is not transferred and remains incompatible with this path.
>
> **Last touched:** 2026-07

## Supported contract

This is the one supported GLM5.2 P/D configuration:

| Component | Contract |
| --- | --- |
| Prefill worker | vLLM 0.24.0, TP8, GLM-5.2-FP8 |
| Decode worker | OpenInfer EP8 |
| Data plane | PegaFlow P2P, `pegaflow-core` v0.23.3 rev `1473c53` |
| vLLM cache | `fp8_ds_mla`, page size 64, page-first registration |
| Hashing | vLLM canonical-CBOR block hash with an identical fixed `PYTHONHASHSEED` on P and D |
| Handoff | P forwards its first token; D restores the complete prompt and admits only at `suffix == 1` |
| Speculation | Target cache only; no MTP or DSpark state transfer |

The normal Qwen3 vLLM 0.23.0 NHD layout and the GLM5.2 vLLM 0.24.0 layout are both single-segment block payloads. PegaFlow #382 is therefore not a dependency for either supported configuration.

P and D must use the same checkpoint revision. Cache keys identify token history and layout, not model weights; mismatched weights can still produce a cache hit and silently invalidate the result. Deployment validation must compare representative weight digests before serving traffic.

## Why prefill stays in vLLM

OpenInfer's GLM5.2 engine is intentionally decode-oriented:

- sparse MLA is instantiated for decode rows;
- the indexer uses paged MQA logits;
- the DeepEP shim is the latency-oriented decode protocol;
- the scheduler can ingest a prompt through decode steps, but that is not an efficient production prefill path.

Building native prefill would require a new attention-parallel path, sparse-prefill kernels, normal-mode MoE communication, and chunked-prefill scheduling. The current product path delegates that work to vLLM and treats OpenInfer as the decode worker.

## State and ownership

Each OpenInfer rank registers 99 target arenas under one PegaFlow instance:

- 78 MLA arenas at 656 bytes per token;
- 21 index-K arenas at 132 bytes per token for full-indexer layers.

The combined target state is 53,940 bytes per token per rank. One pool block identifies the matching pages across both cache families, so a restore is atomic at the request-prefix boundary.

There are two uses of the same mechanism:

1. **Host-tier offload:** OpenInfer saves sealed target blocks on release and restores them before local prefix matching. A measured 1,466-token warm restore reduced TTFT from 5,371 ms to 157.6 ms while preserving bytes.
2. **Cross-engine P/D:** vLLM writes target pages under vLLM-compatible hashes and names; OpenInfer derives the same keys, restores the pages, applies the layout fixup, and begins decode.

The cross-engine boundary has four invariants:

1. namespace, hash seed, page size, and layer names match;
2. page-first arena order and per-token byte layouts match;
3. the partial prompt tail is saved under its derived tail key and restored into the request's private page;
4. D never computes a prompt position locally in strict mode.

The router returns P's first generated token to the client and appends that token to D's token-id prompt. This makes D's first forward a real decode step over the forwarded token. A miss or incomplete restore is rejected instead of silently falling back to local prefill.

## Layout compatibility

The compatible target payload was checked at both source and byte level:

- MLA page rows contain 512 FP8 NoPE bytes, four FP32 scales, and 64 BF16 RoPE values per token;
- index-K pages use the DeepGEMM block-split FP8-plus-scale layout;
- both engines use 64-token pages;
- OpenInfer registers the vLLM layer names and page-first order in compatibility mode.

One required conversion remains at the boundary: vLLM stores the rotated MLA and indexer dimensions in interleaved order, while OpenInfer's decode kernels consume the block order used by its native cache. D deinterleaves newly restored pages on the rank stream before replay.

The original vLLM 0.23.0 GLM path allocated indexer state for all 78 layers and produced 156 cache regions. vLLM 0.24.0 allocates index-K state only for the 21 full-indexer layers, matching OpenInfer's 99-arena contract. This is why the supported producer version is fixed rather than expressed as `>= 0.24.0`.

## Readiness and failure semantics

P's HTTP response can arrive before its asynchronous save is visible through the metaserver. D therefore performs bounded, throttled queries for the complete prefix. Partial hits remain parked; strict-mode timeout rejects the request and lets the router retry the complete P/D flow.

Failure injection established these behaviors:

- killing P produces a prompt upstream failure without D-side fallback or engine damage;
- metaserver loss produces bounded request failure rather than a hang;
- the miss breaker uses a short probe window while open, allowing a recovered async fetch to complete and close the breaker;
- PegaFlow clients reconnect after a metaserver restart and new saves become discoverable.

The unresolved recovery gap is catalog reconstruction. The metaserver keeps its directory in memory, and a restarted metaserver does not learn about blocks that data nodes already hold. Existing sessions therefore remain unavailable until P recomputes and republishes them, or until PegaFlow implements catalog re-publication after reconnect.

## Correctness gates

The merged path passed:

- aligned and unaligned prompt transfer with token-for-token equality against the vLLM producer;
- strict zero-prefill checks: every D admission had `suffix == 1`;
- repeated and multi-turn requests using content-addressed delta reuse;
- GSM8K 5-shot, 200-example comparison;
- NIAH at 4k, 8k, and 16k contexts;
- failure and recovery injection.

### GSM8K

| Endpoint | Strict exact match |
| --- | ---: |
| P/D router, run 1 | 0.960 ± 0.014 |
| P/D router, run 2 | 0.970 ± 0.012 |
| vLLM direct | 0.955 ± 0.015 |

The two P/D runs differ on two examples because EP8 bucket composition is not batch-invariant at near-tied logits. Both remain within the measured baseline noise.

### Long-context retrieval

| Endpoint | 4k | 8k | 16k | Total |
| --- | ---: | ---: | ---: | ---: |
| P/D router | 12/12 | 12/12 | 12/12 | 36/36 |
| vLLM direct | 12/12 | 12/12 | 12/12 | 36/36 |

This exercises real sparse top-k selection above 2,048 tokens and large remote restores. It reduces, but does not close, the indexer-reference risk tracked by #541.

## Equal-card serving A/B

The acceptance workload used 16 GPUs on each side:

- P/D: one 8-GPU vLLM TP8 prefill worker plus one 8-GPU OpenInfer EP8 decode worker;
- mixed baseline: two independent 8-GPU vLLM TP8 workers with session affinity;
- 32 chat sessions, five turns, first input 8,192 tokens, then +2,048 tokens per turn, 128 output tokens, greedy, concurrent clients split across two seeds.

Both sides completed 160/160 requests. Every P/D admission remained at `suffix == 1`.

### TTFT median, milliseconds

| Turn | P/D, two clients | Mixed, two clients |
| --- | ---: | ---: |
| 1 | 11,621 / 11,647 | 4,130 / 4,233 |
| 2 | 1,524 / 1,403 | 1,272 / 3,020 |
| 3 | 1,057 / 1,103 | 1,668 / 1,841 |
| 4 | 564 / 974 | 1,664 / 1,686 |
| 5 | 1,030 / 679 | 1,683 / 1,706 |

### TPOT median / p99, milliseconds

| Turn | P/D, two clients | Mixed, two clients |
| --- | ---: | ---: |
| 1 | 23.3/26.4 · 23.7/33.6 | 42.8/67.3 · 53.4/88.9 |
| 2 | 34.1/35.0 · 33.9/36.2 | 23.0/31.6 · 25.4/45.6 |
| 3 | 35.2/36.6 · 34.7/35.6 | 21.0/31.4 · 22.3/65.3 |
| 4 | 35.2/36.7 · 35.6/36.6 | 21.1/31.5 · 21.1/31.6 |
| 5 | 35.4/38.1 · 34.6/38.0 | 21.0/31.5 · 21.0/31.7 |

Mixed finished at 340 output tokens/s; P/D finished at 271 output tokens/s.

The evidence supports a trade-off, not a universal win:

- P/D isolates decode from prefill, keeping TPOT p99 close to its median and improving turn 3+ TTFT in this workload;
- the 1P:1D split under-provisions prefill for the input-heavy first turn, so mixed wins first-turn TTFT and total throughput;
- later-turn TPOT median remains lower on mixed, while its tail is more variable;
- a deployment must choose the P:D ratio from its input/output mix and SLO rather than copying 1:1.

## DSpark boundary

OpenInfer #657 transfers only target MLA and index-K state. DSpark additionally owns five layers of BF16 draft K/V and consumes target auxiliary hidden states while constructing its context.

vLLM's model-based P/D path runs the same cache-owning speculator on P and D and transfers the draft K/V pages with the target cache. For GLM5.2 that draft state is about 80 KiB/token, making target plus draft transfer about 2.52× the target-only payload. vLLM has generic EAGLE3/MTP acceptance gates, but no GLM5.2 DSpark P/D result.

OpenInfer therefore keeps DSpark mutually exclusive with prefix caching, offload, and P/D. Issue #590 owns the first experiment: cold-start the drafter at the restored boundary, preserve absolute positions, and measure first-round and steady-state acceptance. Draft pages must never be fabricated or marked valid merely because target verification can reject bad proposals.

## Remaining risks

- **Indexer reference:** #541 must establish a reproducible long-context indexer oracle. NIAH is an end-to-end probe, not a replacement for that gate.
- **Catalog recovery:** PegaFlow data nodes need to republish their existing block directory after metaserver restart.
- **Version drift:** any vLLM cache-layout or hashing change requires rerunning the byte-layout and aligned/unaligned prompt gates before expanding the supported version range.
- **Speculative state:** DSpark remains outside the P/D contract until #590 produces measured acceptance evidence or a draft-KV transfer protocol is gated.
