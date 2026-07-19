# GLM5.2 serving status

> **TL;DR:** GLM5.2 now has complete decode-serving paths across EP8/TP8 on Hopper, EP4/TP4 on Blackwell, and EP-N within a single NVLink/IMEX domain. Continuous batching, whole-step CUDA Graphs, sampling, DSpark, paged KV, prefix caching, host offload, and target-only vLLM→OpenInfer P/D are implemented. The line remains in the project’s Bring-up tier until the long-context indexer oracle is reproducible and the remaining reliability boundary is closed.
>
> **Last touched:** 2026-07

## Current shape

GLM5.2 is no longer an initial model bring-up. It is a model-owned distributed serving engine with several launch-time topologies and different latency/throughput goals. The project tier remains **Bring-up**, rather than Maturing or Stable, because the correctness and reliability contracts below are not yet continuously enforceable.

### Topologies

| `--moe-topo` | Intended use | Evidence boundary |
| --- | --- | --- |
| `ep8` | Default high-throughput path on 8×H200 | Strongest feature coverage: bucketed continuous batching, DSpark, prefix cache, offload, and P/D |
| `tp8` | Low-latency path on 8×H200 | Attention TP plus TP-sharded MoE; fixed replicated-request shape; DeepEP is still initialized but unused (#608) |
| `tp4` | Low-latency path on 4×GB300 | Whole-step graphs, sparse MLA, vocabulary-parallel tail, and topology-specific kernel tuning |
| `ep4` | Throughput path on 4×GB300 | Functional and oracle-gated; equal-topology decode remains about 15% behind the measured vLLM reference (#668) |
| `ep16` / `ep32` / `ep64` | Scale within one NVLink/IMEX domain | Rank-host control plane and per-width DeepEP shims exist; the strongest end-to-end evidence is still within one rack, not general IB/RoCE scale-out |

See `moe-tp8-low-latency.md`, `tp4-gb300-bringup.md`, `ep4-gb300.md`, and `cross-node-scaling.md` for the measured topology records.

## Serving capabilities

| Area | Current contract |
| --- | --- |
| Scheduling | Up to 8 slots per logical EP rank; `{1,2,4,8}` whole-step graph buckets; least-loaded admission |
| Attention | DSA indexer plus sparse MLA decode; per-request context limit sized from free VRAM |
| Sampling | `temperature`, `top_p`, `top_k`, `min_p`, and engine-level `seed`, honor-or-reject |
| Speculation | DSpark greedy and sampled verify; span 4 default; verify spans reuse decode buckets |
| KV | 64-token paged pool, full-lifetime admission, prefix cache on by default |
| Offload | PegaFlow host-tier save/restore behind `--kv-offload` |
| P/D | vLLM 0.24.0 TP8 prefill → OpenInfer EP8 decode, strict zero-prefill, merged in #657 |
| Observability | Per-logical-partition running/waiting/KV gauges and decode graph export |
| Remote ranks | Framed-TCP rank-host control plane; local and remote workers share one typed command contract |

The P/D support matrix and acceptance data live in `pd-m2-execution.md`. It transfers target state only; DSpark draft state is not part of that protocol.

## Sampling and API limits

The model engine supports `temperature`, `top_p`, `top_k`, `min_p`, and `seed` on both plain and speculative paths. Engine-level seeded replay is deterministic for the same occupancy timeline.

The following surfaces are not part of the GLM5.2 contract:

- `logprobs`, prompt logprobs, and `n > 1`;
- presence, frequency, and repetition penalties;
- GLM-specific guarantees for stop strings, stop token IDs, or `min_tokens` beyond the shared frontend behavior.

HTTP `seed` is still lost in the shared frontend before reaching the engine. Bucket changes can also alter floating-point association, so a greedy request may diverge at a near-tied token when its occupancy timeline changes. Runs with the same request and bucket timeline remain deterministic.

## Promotion blockers

### 1. Reproducible long-context correctness

Issue #541 is the main tier blocker. The indexer oracle once passed against a moving Transformers development reference, but that reference changed and the result is not reproducible. The current engine has passed end-to-end 4k/8k/16k NIAH, yet that probe cannot replace a pinned sparse-index selection gate.

The padded-vocabulary contract is also under repair in #680/#698. The checkpoint contains token IDs the frontend tokenizer cannot decode; every EP, TP, sampling, and DSpark token-producing path must be structurally bounded to the decodable prefix.

### 2. Request lifecycle reliability

Issue #551 records one request that entered the frontend but never reached a terminal engine event. More than 3,500 later requests and extended soaks did not reproduce it. It remains a background reliability boundary until a trace identifies the cause or a sufficiently strong retained soak demotes it.

### 3. Feature composition

DSpark is mutually exclusive with prefix caching, host offload, and P/D. A prefix hit skips the target forwards that normally produce DSpark's historical auxiliary state. Issue #590 must first measure a position-correct boundary cold start before the project considers transferring the additional draft K/V payload.

Remote rank-host mode also remains incompatible with KV offload. Cross-node request execution and cross-node KV residency are separate protocols today.

## Performance work

Measured open work is topology-specific:

- #668: right-size the Blackwell EP4 masked expert kernel for bucket 1; the measured kernel reaches 29% of its byte roofline and leaves about 3 ms/step at equal topology.
- #625: replace the TP8 sparse-MLA static split count with per-row work planning for solo long contexts.
- #608: stop allocating and initializing unused DeepEP state in TP8 mode.
- #582: graph the DSpark draft round only after its fixed launch cost matters; it is currently a small fraction of the verify round.
- #542/#559/#569: older Hopper EP8/bucket/PDL investigations remain evidence, but should be re-baselined before implementation because the active topology and kernels have moved.

No optimization should be carried forward from these records without a matched A/B on the current topology.

## Background work

- #587: expose active slots, current bucket, and queue depth in addition to the scheduler gauges already shipped.
- PegaFlow metaserver recovery: republish the existing block catalog after reconnect; new saves recover today, old remote prefixes do not.
- General scale-out beyond a single NVLink/IMEX domain: preserve the rank-host contract, but use a data plane designed and measured for IB/RoCE rather than treating the one-rack result as universal.
