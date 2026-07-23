# Qwen3.5-4B prefix cache

> **TL;DR:** A Qwen3.5-4B prefix hit is valid only when full-attention KV and a complete recurrent/conv snapshot exist at the same 256-token boundary. `Qwen35PrefixCache` checks and restores both together, so the scheduler sees either one valid hit or a miss. The first version keeps snapshots on GPU and requires Qwen3.5 KV to move from `KvPool`/`KvState` to the content-hashed `BlockPool`/`RequestKv` cache.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` - identified the current Qwen3.5 roadmap and the related Qwen3 cache and scheduler docs.
  - `docs/models/qwen35/roadmap.md` - direct-paged writes and bounded chunked prefill are complete; issue #257 now needs a joint KV/recurrent/conv cache design.
  - Maintainer RFC discussion for issue #257 - narrowed the first version to a GPU snapshot cache with one consistency rule for KV and recurrent state.
  - `docs/models/qwen3/prefix-cache.md` - provides the existing rules for block hashes, adapter isolation, final-token recompute, and keeping matched KV alive.
  - `docs/subsystems/runtime/qwen3-kvbm-integration-spec.md` - describes the content-hashed `BlockPool`/`RequestKv` cache that Qwen3.5 does not yet use.
  - `openinfer-qwen35/src/{scheduler.rs,prefill.rs,prefill_buffers.rs,recurrent.rs,recurrent_state.rs,weights.rs}` - confirmed the current request state flow, valid prefill boundaries, snapshot layout, and GPU memory reservation.
  - `openinfer-core/src/kv_pool.rs` and `openinfer-kv-cache/src/pool.rs` - confirmed that Qwen3.5 still uses anonymous RAII pages while Qwen3 can register, match, and pin content-hashed blocks.
- **Relevant history**:
  - The first draft focused on CPU offload but did not say clearly who keeps KV and snapshots consistent. Review narrowed the first version to GPU allocation, lookup, lifetime, and whole-model snapshot creation.
  - The first draft also treated the 64-token GDR tile as a correctness boundary. Current resumed-prefill coverage uses 16-token scheduler chunks successfully, so a completed whole-model chunk, not an internal GDR tile, is the state boundary.
- **Plan**:
  1. Replace the two-tier proposal with a GPU-only cache that restores KV and recurrent state together.
  2. Base snapshot creation and restore on the current chunked-prefill flow and the future `BlockPool`/`RequestKv` migration.
  3. Define snapshot contents, capacity, publication order, lookup, pinning, eviction, failure behavior, and follow-on validation.
- **Risks / open questions**:
  - The current `RequestKv::match_and_add_prefix` immediately changes a new request to use the longest KV-only match. Qwen3.5 instead needs the exact-boundary lookup and attach behavior described below.
  - Snapshot copy and cold-prefill costs have not been measured for the current 32-value-head 4B configuration. The initial 256-token interval is therefore a starting policy, not a proven optimum.

## Decisions

The first version is deliberately narrow:

- GPU-only recurrent snapshot allocator with a fixed load-time byte budget.
- One complete snapshot contains all 24 linear layers' f32 GDR state, bf16 conv state, and the token position.
- Snapshots are published every 256 prompt tokens. Non-aligned request ends are not cached in the first version.
- A reusable boundary must have both registered full-attention KV and a GPU-resident recurrent snapshot for the same token prefix.
- `Qwen35PrefixCache` is the only interface used by the scheduler for prefix creation and restore.
- Restored state is copied into the request's own `RecurrentState`; active requests never modify or directly share a cache slot.
- Echo and prompt-logprob requests stay on cold prefill because cached positions would not produce their required logits.

CPU offload, a second LRU tier, snapshot compression, request-end snapshots, and cross-process transfer are deferred. They must keep the same rule that KV and recurrent state are restored together.

## Current state and prerequisite

Qwen3.5 prefill already has a safe point at which it can create a snapshot. A `PrefillingRequest35` owns:

```rust
struct PrefillingRequest35 {
    req: SchedulerRequest,
    kv: KvState,
    rec: RecurrentState,
    cursor: usize,
    step_chunk: usize,
}
```

For each scheduled window, `prefill_chunk_forward` writes full-attention K/V directly into paged storage and advances all linear layers' recurrent and conv state. When the whole-model call returns successfully:

```text
kv.seq_len() == rec.seq_len == cursor + step_chunk
```

If the prompt is incomplete, the scheduler keeps these states for the next step. If it is complete, it copies `rec` into a stable decode graph slot. Direct-paged prefill and scheduler chunking are therefore no longer blockers.

The missing prerequisite is content-based KV reuse. Qwen3.5 still uses `openinfer_core::kv_pool::{KvPool, KvState}`: it allocates and returns pages, but it cannot identify their token content, register completed blocks, or match a new request against them. Qwen3 uses `openinfer_kv_cache::{BlockPool, RequestKv}`, which provides those operations and keeps matched blocks alive while they are being attached to a request.

Before prefix reuse can be implemented, Qwen3.5 full-attention KV must move to that cache API while preserving its current page-first memory layout and kernels. Most required operations already exist. Joint lookup adds these requirements:

1. Probe the longest contiguous registered KV prefix without changing the new request.
2. Keep the probed KV blocks pinned while the snapshot cache is checked.
3. Expose complete KV-block boundaries and their canonical `SequenceHash` values in descending order.
4. After a joint boundary is selected, transfer only the blocks through that boundary to the new request and release any longer KV-only tail.
5. Advance the new request's KV position to the selected boundary as part of the same attach operation; a partial attach must not be visible.
6. Keep at least one prompt token uncached so prefill can produce the first generated token.

The existing `schedule_prefill`, `apply_prefill_chunk`, and `revert_schedule` operations then handle suffix prefill and failed forwards.

## Valid cache hit

Qwen3.5 stores KV and recurrent snapshots separately, but a request may reuse a boundary `N` only when all of the following are true:

1. Full-attention KV for `[0, N)` is registered and still on GPU.
2. A complete recurrent snapshot for `[0, N)` is still on GPU.
3. Both use the same `SequenceHash`, which includes the token lineage and adapter/LoRA salt.
4. The KV position, snapshot position, recurrent `seq_len`, and scheduler cursor all equal `N`.

KV without a matching snapshot is not a Qwen3.5 prefix hit. A snapshot without matching KV is also not a hit. The scheduler never sees either one as partial reuse.

## Snapshot interval

A snapshot boundary is the token position after a scheduled prefill window has completed all 32 layers. The interval for the first slice is:

```text
SNAPSHOT_STRIDE = 256 tokens
```

The stride is a multiple of the 16-token KV block size, so every snapshot key can reference the lineage hash of a complete registered KV block. The scheduler must clamp each request's next window to the next snapshot boundary. With a 900-token prompt, the resulting positions are `256 -> 512 -> 768 -> 900`; only the first three are snapshot candidates.

The GDR implementation internally tiles work in 64-token chunks, but this is not a snapshot correctness constraint. It handles a partial final tile and commits the final recurrent state for arbitrary positive sequence lengths. The existing scheduler-level resumed-prefill gate uses 16-token windows and exercises this behavior. The invariant is therefore "after a successful whole-model window," not `position % 64 == 0`.

The 256-token interval limits how many large snapshots one prompt creates. Prefixes shorter than 256 tokens intentionally remain cold. This is a starting policy, not a measured optimum; any later interval must still align to complete KV blocks.

## Snapshot contents and capacity

Each slot uses the same device layout as request-local `RecurrentState`:

- for every linear layer, `state: [num_value_heads, key_head_dim, value_head_dim]` f32;
- for every linear layer, `conv_state: [linear_attn_qkv_dim, conv_kernel_dim - 1]` bf16;
- host metadata recording the exact `seq_len` represented by the slot.

Capacity must be derived from `recurrent_state::bytes_per_request(config)`, not a hard-coded model label. For Qwen3.5-4B (24 linear layers, 16 key heads, 32 value heads, 128x128 state, conv kernel 4), one slot is:

```text
per-layer GDR state = 32 * 128 * 128 * 4             = 2,097,152 bytes
per-layer conv      = 8,192 * (4 - 1) * 2            =    49,152 bytes
all 24 layers       = 24 * (2,097,152 + 49,152)      = 51,511,296 bytes
                                                        49.1 MiB
```

The allocator reserves a fixed number of whole slots at model load:

```rust
let bytes_per_slot = bytes_per_request(config);
let max_slots = snapshot_budget_bytes / bytes_per_slot;
```

The reservation participates in the same load-time budget as prefill scratch, recurrent request/decode slots, and KV pages. The current loader reserves two recurrent states per decode capacity slot before sizing KV; snapshot bytes are additional and must also be subtracted before the KV pool is allocated. Snapshot allocation must not opportunistically consume memory that admission assumes belongs to KV.

If the configured budget produces zero slots, Qwen3.5 prefix reuse is disabled and serving retains its current cold behavior.

## Cache ownership and pinning

`Qwen35PrefixCache` owns the existing full-attention KV manager and the recurrent snapshot cache. `KvCacheManager` keeps the logical `BlockPool` and physical GPU `KvBuffer` together:

```rust
struct Qwen35PrefixCache {
    kv: KvCacheManager,
    snapshots: RecurrentSnapshotCache,
    stride: usize,
}

struct RecurrentSnapshotCache {
    slots: Vec<SnapshotSlot>,
    index: HashMap<SnapshotKey, SlotId>,
    free: Vec<SlotId>,
    lru: LruList<SlotId>,
}

struct SnapshotSlot {
    state: RecurrentState,
    key: Option<SnapshotKey>,
    pin_count: usize,
}

#[derive(Clone, Hash, Eq, PartialEq)]
struct SnapshotKey {
    sequence_hash: SequenceHash,
    boundary: usize,
}
```

`sequence_hash` is the canonical hash returned by the KV cache for the block ending at `boundary`. It already includes the earlier block lineage and adapter salt, so the snapshot cache does not maintain a second adapter identity. Storing `boundary` explicitly prevents a snapshot from being reused at the wrong token position.

There is no third stored copy that combines KV and snapshot data. A valid internal hit records the selected boundary and holds both a KV guard and a snapshot guard. `Qwen35PrefixCache` consumes those guards during restore, so neither resource can be evicted in the meantime. The scheduler never sees the separate guards.

## Creating a snapshot

A snapshot is created after a whole-model window, not inside per-layer GDR scratch. The order is:

1. Clamp the request's scheduled window to the next 256-token boundary or prompt end.
2. `RequestKv::schedule_prefill` reserves the KV blocks for that window.
3. Run the full model, updating full-attention KV and request-local recurrent/conv state.
4. On failure, revert the KV schedule, fail the request, and publish nothing.
5. On success, commit the KV request state (`apply_prefill_chunk` or final `apply_prefill`) so complete blocks are registered.
6. Assert that committed KV position and `rec.seq_len` equal the candidate boundary.
7. If the boundary is snapshot-eligible, allocate an unpinned slot and D2D-copy the complete `RecurrentState` into it.
8. Publish `SnapshotKey -> SlotId` only after the copy has been successfully enqueued under the scheduler stream's ordering contract.

The GDR `chunk_state` scratch is per linear-layer call and cannot represent a whole-model snapshot. Conv state is updated separately. Only request-local `RecurrentState` after all layers have finished contains the complete pair required for publication.

Running out of snapshot slots is a soft cache event: skip insertion and continue the request. A CUDA copy failure is an execution error, not a cache-capacity miss; no key is published.

Insertion of an already-resident key reuses the existing immutable slot and refreshes its LRU position; it does not allocate or copy a duplicate. When replacing an unpinned victim, the cache removes the victim's old index entry before starting the copy. If that copy fails, the slot returns to the free list with no published key.

## Lookup and restore

The scheduler calls one operation:

```rust
let cached_tokens = prefix_cache.restore_prefix(
    &mut prefix_request,
    &mut request_recurrent,
)?;
```

Inside `Qwen35PrefixCache`:

1. Ask the KV cache for the longest registered prefix while keeping the candidate KV blocks alive.
2. Enumerate eligible 256-token boundaries in descending order, subject to the usual rule that at least one prompt token remains to run.
3. Build `SnapshotKey` from the candidate's canonical `SequenceHash` and position.
4. Try to pin the corresponding snapshot slot.
5. The first boundary with both guards becomes the selected hit; if none exists, return `0` without changing request state.
6. Attach exactly that KV boundary to request-local `RequestKv`.
7. D2D-copy the immutable snapshot into request-local `RecurrentState`.
8. Verify all positions equal the selected boundary, then set the scheduler cursor and return it as `cached_tokens`.

For example, a 768-token KV match with snapshots at 256 and 512 restores 512 tokens. The scheduler never receives "KV hit 768, snapshot hit 512" as separate facts.

`Qwen35PrefixCache` must acquire both guards before changing the request. If restore then fails, it releases the request KV and snapshot guard and reports an error. It must not expose a partly restored request or treat a restore error as a normal cache miss.

After restore, suffix prefill operates normally on `tokens[cached_tokens..]`. When prefill finishes, the existing copy from request-local recurrent state into the decode graph slot remains unchanged.

## Lifetime and eviction

The KV and snapshot caches keep their own allocation policies, but `Qwen35PrefixCache` decides whether a boundary can be reused:

- KV candidates are held by strong immutable-block guards from probe until exact-boundary attachment or abandonment.
- Snapshot slots are immutable while indexed and can be evicted only when `pin_count == 0`.
- A snapshot guard is needed only until its D2D copy into request-local state completes. It is not held for the full request lifetime.
- Restored suffix prefill and decode mutate request-owned state, never the cached slot.
- If no free or unpinned snapshot slot exists, insertion is skipped rather than blocking or failing the request.

Eviction does not require synchronized callbacks between the physical pools:

- If KV is evicted first, the snapshot remains indexed but cannot be used. Lookup cannot acquire the KV guard, so snapshot LRU may reclaim it later.
- If the snapshot is evicted first, the KV blocks may remain reusable by the physical pool, but the boundary is KV-only and ineligible for Qwen3.5 restore.
- If either side disappears between candidate discovery and pinning, lookup continues to the next shorter joint boundary.

This cannot produce a partial hit: only `Qwen35PrefixCache` can declare a hit, and it checks both resources on every lookup.

LRU is the first victim policy for unpinned snapshot slots. Correctness depends on pinning and joint validation, not on LRU itself.

## Correctness rules

- `RequestKv::kv_position() == RecurrentState::seq_len == SnapshotKey::boundary` after insertion and restore.
- A snapshot contains both state tensors for every linear layer; GDR-only or conv-only snapshots are invalid.
- Snapshot contents are immutable after publication.
- KV and snapshot must use the same canonical `SequenceHash`.
- A prefix hit always leaves at least one prompt token uncached so final prefill can emit the first generated token.
- Echo and prompt-logprob requests never use prefix matching in the first slice.
- Allocation pressure or no evictable snapshot slot changes hit rate only, not request output.
- Snapshot insertion failure never converts an otherwise valid cold request into a cache hit.
- A KV-only or snapshot-only boundary is never reported as cached tokens.
- Disabling the feature or configuring zero slots preserves current cold-serving behavior.

## Implementation order

Implementation is a follow-on to this design and should land in this order:

1. Move Qwen3.5 KV management to `BlockPool`/`RequestKv` while keeping the existing KV memory layout, admission behavior, direct-paged kernels, and accuracy gates.
2. Add the fixed GPU snapshot allocator, config-derived slot sizing, pin guards, and unpinned LRU eviction.
3. Make scheduler chunk planning boundary-aware and publish snapshots after committed whole-model windows.
4. Add exact-boundary KV lookup/attach and expose the single `Qwen35PrefixCache::restore_prefix` operation.
5. Add metrics for joint hit length, KV-only fallback, snapshot miss, skipped insertion, eviction, and restore latency.
6. Run correctness, pressure, and warm-TTFT gates before enabling the feature by default.

## Validation

The implementation acceptance surface should include:

- cache-management tests for snapshot keys, shorter-boundary fallback, duplicate insertion, pinning, and unpinned eviction;
- scheduler tests for prompts below 256, exactly on boundaries, across multiple boundaries, and with non-aligned tails;
- availability tests for KV-only, snapshot-only, and shorter valid fallback;
- adapter salt isolation;
- mixed cold and warm requests in the same prefill/unified step;
- pool-full behavior proving insertion skip preserves cold output;
- real GPU cold-vs-warm HF logits gates, including resumed suffix prefill and decode-slot promotion;
- retained metrics for snapshot D2D copy time, cold insertion overhead, warm TTFT, joint hit length, and slot occupancy.

Those measurements determine whether 256 remains the right stride. They must not be replaced by the old RTX 4090 CPU-transfer estimates, which measured a deferred design and a different snapshot shape.

## Deferred work

- pinned-CPU snapshot offload and two-tier LRU;
- request-end snapshots for non-aligned prompt lengths;
- workload-adaptive or per-model snapshot stride;
- snapshot compression or reduced-precision state;
- cross-worker/P-D transfer of hybrid state;
- integration with speculative rollback state;
- sharing snapshot infrastructure across other hybrid model lines.

Each of these must preserve the same logical rule: one reusable boundary restores all model state at one token position.
