# GLM5.2 × pegaflow: host-tier KV offload → P/D disaggregation

> **TL;DR:** M1 is implemented by [#600](https://github.com/openinfer-project/openinfer/pull/600): shared `OffloadHost` + 8 rank instances on one namespace, 99 target-model arenas/rank (78 MLA + 21 index-K), restore at admission / save on release, behind `--kv-offload` (+`--kv-offload-hugepages`). Target-only M2 = cross-engine P/D (vLLM prefill → openinfer decode) via the #540 hash-compat pattern plus a byte-dump drift gate. Preserving any model-based speculative decoder with its own KV is a separate state-transfer problem. The dependency intentionally remains at pegaflow-core **v0.23.2 rev d46fd16**: #395 is merged, but stacked #540 also requires still-open #382; the next pin must be a Pegaflow `master` revision containing both.
>
> **Last touched:** 2026-07

## Why pegaflow first

The standing decision is prefill-by-vLLM (no dedicated GLM5.2 prefill path). P/D disaggregation means the decode engine ingests KV that someone else produced — and the ingestion mechanism (query a shared tier by block hash, lease, load into pool pages, commit as matched prefix) is the same mechanism as host-tier offload restore. M1 built and gated that mechanism against OpenInfer-produced blocks; target-only M2 changes who wrote them. Preserving a stateful speculative decoder across the handoff additionally requires a drafter-state contract.

## Current seam map

- `openinfer-kv-offload` is an **in-process** bridge. `OffloadHost` owns one pinned host pool and `PegaEngine`; per-rank `OffloadEngine::with_arenas_on` instances register device arenas against it. No server or metaserver is required unless `P2pConfig` enables RDMA.
- GLM5.2 drives the complete loop in `scheduler/offload.rs`: admission probes and restores host-resident sealed blocks before prefix matching, while release saves newly sealed blocks asynchronously. `BlockPool` exposes the split point (`probe_prefix` → `cpu_query_hashes` → `commit_loaded_blocks`).
- Keying: openinfer serializes its xxh3 `PositionalLineageHash` u128 as 16 big-endian bytes (`pool.rs:602`); pegaflow treats `BlockKey { namespace, hash }` as opaque bytes and never re-hashes.

## M1 — implemented host-tier offload (GLM5.2 ↔ GLM5.2)

1. **Registration geometry**: each rank registers 99 arenas under one instance: 78 MLA arenas (`64 × 656 B` pages) plus 21 index-K arenas (`64 × 132 B`, full-indexer layers only). One save/load entry moves both cache families atomically.
2. **Shared tier**: one `OffloadHost` backs 8 rank instances in a single namespace. Replicated MLA means any rank can restore a block saved by another rank instead of partitioning the host hit rate eight ways.
3. **Restore/save hooks**: admission restores host hits before `match_and_add_prefix`; release saves newly sealed blocks while guards keep source pages alive through D2H. Blocking restore at admission is the accepted M1 behavior.
4. **Policy boundary**: only sealed target blocks enter the tier. DSpark still disables prefix matching/offload because the 99 arenas exclude its five draft K/V layers; #590 owns lifting that restriction.
5. **Measured gates** (2026-07-06, all pass): a 1466-token prompt remained byte-identical cold → warm-restored (22 blocks) → no-offload; TTFT was 5371 ms cold versus **157.6 ms warm-restored (~34×)**; a 16-way mixed rerun after eviction completed with 15 full-prefix restores and zero errors. Hugepage allocation and a 160-prompt warning-free churn were also verified.

## M2 — cross-engine P/D (vLLM prefill → openinfer decode)

The target-only handoff has two hard problems, both with prior art:

- **Hash compat**: openinfer xxh3-lineage vs vLLM SHA-256 `block_hashes` never collide onto the same keys. PR #540 (qwen3) already built the pattern: compute vLLM-compatible hashes for the prompt at admission and query the vLLM-written namespace. Port that, plus the namespace derivation (vLLM side = SHA-256 over model/dtype/tp/heads/... — `connector/common.py:210`; ours must reproduce it byte-for-byte).
- **Layout parity**: verified at source level 2026-07-06 against vLLM revision `cdab28319` — byte-for-byte MATCH on everything device-side, because our kernels are ports of vLLM's: 656 B/token field order `[512 fp8 NoPE][4×f32 scale, ÷448, group=128][64×bf16 RoPE]` (vLLM `cache_kernels.cu:447-547` ↔ ours `glm52_mla_assembly.cu:33-34`), index-K block-split `[64×128 fp8][64×4 B f32 scale]` (vLLM `cache_kernels.cu:550-607` ↔ ours `glm52_indexer.cu:69-87`, which cites the vLLM kernel by name), page=64, layer-outermost page-contiguous arenas both sides. Remaining gate before e2e: dump one block per arena from each engine for the same prefix and byte-compare (guards silent drift on either side's bumps). One structural note: vLLM keeps MLA and indexer caches as separate KV groups; we index both off one pool block id — irrelevant to byte parity, matters only for block-id translation.

### Model-based speculative state at the P/D boundary (vLLM audit, 2026-07-10)

vLLM `0206f10871` does not cold-start a model-based speculator whose draft model owns KV. The generic fix landed in `90f3c01fa4d` / [vLLM #35158](https://github.com/vllm-project/vllm/pull/35158): connector finalization is deferred until after the draft forward so P saves both target and drafter KV. Current Model Runner V2 follows the same ordering: it loads the draft model before collecting `AttentionLayerBase` cache specs, registers every resulting target/draft cache tensor with the connector, runs `speculator.propose()`, then calls connector `post_forward()`. This applies by state ownership to EAGLE/EAGLE3, DFlash/DSpark, and MTP variants with draft attention; ngram/suffix-style methods have no drafter KV payload.

The NIXL P/D + speculative gate makes P and D load the same speculator. P uses one speculative token and the proxy drives it with `max_tokens=1`; D may use a wider proposal span. On a full remote prompt hit, vLLM deliberately changes `num_computed_tokens` from `N` to `N-1` and reruns the final prompt token to recover sampling logits. That is one boundary target forward, not a full target prefill and not reconstruction of historical drafter KV. Static inspection of the Qwen3.5 MTP cache geometry shows that P-without-MTP/D-with-MTP fails NIXL's matching-region validation; the default load-failure policy fails the request, while explicit recomputation falls back to local D prefill. The checked-in runtime gate covers symmetric EAGLE3 and MTP configurations, not that asymmetric case. DSpark was added later (2026-07-01) and has no GLM5.2 P/D gate, so the generic contract is established while GLM cross-engine compatibility is not.

Applied to OpenInfer's DSpark integration, this leaves two honest paths:

1. **Target-only handoff + explicit draft cold-start.** Transfer the existing 99 target arenas, including #657's restored partial target tail, then start DSpark from only the boundary token's aux-hidden capture. The target side now has `suffix == 1`; the draft side still has no historical KV. Never mark absent draft pages as valid. Target verification preserves output correctness, but acceptance is unknown. The current OpenInfer drafter encodes `anchor_pos == committed_len + pending_len`; boundary-only startup therefore needs a separate absolute-position base (or equivalent compact-KV mapping), not a fabricated `committed_len` over empty pages.
2. **vLLM-style stateful handoff.** Run the same DSpark checkpoint on P and add its five layers of BF16 K/V state to the protocol. Draft KV costs 80 KiB/token versus about 52.7 KiB/token for OpenInfer's target MLA + index-K state, so the payload becomes about 2.52× target-only. This path also needs an independent byte-layout/position/checkpoint compatibility gate; target-cache parity does not prove drafter-cache parity.

A third cross-engine representation is the five target aux-hidden rows (60 KiB/token), from which D can construct DSpark KV. vLLM does not use it; it trades somewhat less wire data for a full drafter context precompute on D and a new connector payload type.

## Resolved target-tail constraint (identified 2026-07-07, closed by #395/#657)

The original sealed-block protocol capped reuse at `cacheable = (prompt−1)/64`, leaving up to 64 prompt positions for D to recompute. M1 measured that floor at roughly 150 ms for a 1466-token prompt even though the sealed-block restore itself took only ~3–5 ms. Pegaflow #395 and OpenInfer #657 close it without teaching kvbm to seal partial blocks: P stores the completed partial page under a derived tail key, the router appends P's first generated token to D's context, and D restores the tail into the request's private page before admitting only at `suffix == 1`. The partial page never enters the radix as a falsely sealed content block.

## Pitfalls pinned during the survey

- pegaflow v0.22.6 **lacks the `*_inproc` and `_strided` APIs** this integration uses; inspect the pinned v0.23.2 revision `d46fd16` instead. Do not advance to #395 alone: GLM's page-first payload is single-segment, but the stacked Qwen3 path requires #382 to load split host K/V into a contiguous device page safely.
- The PyO3 package is a gRPC client — reference contract only, not our path.
- vLLM connector requires `storage_offset()==0` and registers CUDA-IPC handles; our in-process path passes raw device pointers and skips IPC entirely.

## Next action

Target-only M2 is now implemented and gated in #657; its next action is the dependency landing sequence recorded in `pd-m2-execution.md`. Speculative-state ownership remains separate: its first empirical gate is a #590 suffix-only DSpark A/B, comparing cold-start acceptance with the 2.52× stateful target+draft payload before selecting draft-KV transfer.
