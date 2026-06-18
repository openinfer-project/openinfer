# Qwen3.5-4B prefix cache

> **TL;DR:** Qwen3.5 prefix reuse restores both full-attention KV (8 layers, paged) and linear-attention recurrent state (24 layers, snapshots). Snapshot every 256 tokens (4×GDR chunks), GPU+CPU two-tier pool (LRU), break-even at ~68 tokens (2× margin over H2D cost). GPU tier holds ~29 snapshots (~7,424 cached tokens) on RTX 4090; CPU tier extends capacity at +2.18ms restore cost. 
>
> **Last touched:** 2026-06

## Problem

Qwen3-4B is 32 full-attention layers. Its prefix cache only needs KV blocks (512 KB each, 16-token granularity). A cache hit will restore KV pages, and suffix prefill continues.

Qwen3.5-4B has **24 linear-attention + 8 full-attention layers**. Linear layers carry recurrent state (updated every 64-token GDR chunk). A prefix cache hit must restore:

1. **KV blocks** for 8 full-attention layers (paged, same as Qwen3)
2. **Recurrent-state snapshot** for 24 linear layers (~52 MB per boundary)

Key constraint: snapshots can only be taken at GDR chunk boundaries (64-token multiples) where linear-attention kernels emit complete recurrent state. This drives coarser granularity than Qwen3 (256 vs 16 tokens) and larger memory footprint per cache entry (52 MB vs 512 KB).

| Aspect | Qwen3-4B | Qwen3.5-4B |
| --- | --- | --- |
| Cache unit | KV blocks only | KV blocks + recurrent-state snapshot |
| Hit types | full hit or miss | full hit (GPU) / CPU hit (+2.18 ms) / miss |
| Pool | single KV block pool | KV pool + two-tier snapshot pool (GPU + CPU) |
| Restore boundary | KV block boundary (16 tokens) | snapshot boundary (256 tokens) |
| Eviction unit | 1 block = 512 KB | 1 snapshot = ~52 MB |

## Design overview

**Core idea:** Checkpoint recurrent state at fixed 256-token intervals during prefill. Store snapshots in a two-tier pool (GPU primary, CPU eviction backup). On cache hit, restore both KV and snapshot at the matched boundary, then prefill only the suffix.

**Key components:**
1. **Snapshot checkpointing**: at every 256-token boundary, D2H-copy the 52 MB recurrent-state snapshot
2. **Two-tier pool**: GPU slots (fast D2D restore) + CPU backup (slower H2D restore but extends capacity)
3. **Joint lookup**: radix tree returns a match only if both KV and snapshot exist at the same boundary
4. **LRU eviction**: independent per-tier, protects in-flight requests

**Trade-offs:**
- ✅ Enables prefix reuse for Qwen3.5's hybrid architecture
- ✅ GPU-tier hits nearly as fast as Qwen3 (sub-ms D2D copy)
- ❌ Coarser granularity than Qwen3 (256 vs 16 tokens)
- ❌ Large memory footprint (52 MB per snapshot vs 512 KB per KV block)

## Quantitative analysis

Measured on RTX 4090 (24 GB), PCIe Gen4 x16.

| Item | Value |
| --- | --- |
| Snapshot size | ~52 MB (24 layers × [32×128×128 f32 recurrent state + conv state]) |
| KV block size | 512 KB (8 layers × 16 tokens × 8 KV heads × 128 dim × bf16 × 2) |
| 52MB D2H (GPU to pinned CPU) | 2.03 ms |
| 52MB H2D (pinned CPU to GPU) | 2.18 ms |
| Prefill throughput (>=1024 tok) | ~15,400 tok/s (~0.065 ms/tok) |

**Break-even threshold:** For a CPU-tier restore (H2D 2.18 ms) to be no worse than recomputing the skipped prefix, the snapshot needs to accommodate at least `2.18 ms / 0.065 ms/tok ≈ 34 tokens`; with 2× safety margin ~68 tokens.

### Memory budget

Measured on RTX 4090 (24 GB VRAM):

| Quantity | Value |
| --- | --- |
| GPU total VRAM | 24,564 MB |
| Model weights | 9,320 MB |
| KV cache (8 layers) | 10,457 MB |
| Prefill scratch | 3,248 MB |
| Free | ~1,539 MB |
| Single snapshot | ~52 MB |
| **Max GPU snapshots** | **~29** |
| Pinned CPU tier | unlimited or configured (system RAM) |

At 256-token intervals, 29 GPU slots = **~7,424 cached tokens** before eviction to CPU. A 4096-token prompt uses 16 snapshots (55% of pool). GPU tier is a hot cache for frequent prefixes; CPU tier extends capacity at +2.18ms H2D cost per hit.

## Design details

### Snapshot interval: why 256 tokens?

**Constraints:**
- Must align to 64-token GDR chunks (architectural — the granularity at which linear-attention kernels emit complete recurrent state; mid-chunk state is undefined)
- Break-even threshold ~68 tokens (H2D 2.18ms cost vs recompute 0.065ms/tok)
- Cold prefill overhead: each snapshot = ~2ms D2H

**Alternatives considered:**
- **64 tokens** (1 chunk): too fine, 4096-token prefill = 64 snapshots = +128ms cold overhead
- **128 tokens** (2 chunks): 32 snapshots = +64ms, marginal over break-even
- **256 tokens** (4 chunks): **current choice** — 16 snapshots = +32ms, 3.7× over break-even
- **512+ tokens**: reduces cold overhead but hurts hit rate on 512-1024 token shared prefixes

256 tokens is chosen as a balanced starting point. Tuning after observing real workload hit-rate distribution is follow-up work.

### Two-tier pool architecture

The snapshot pool owns recurrent-state snapshots. Full-attention KV remains owned by the paged KV cache. Lookup joins the two owners: a boundary is restorable only if both the KV cache and snapshot pool still hold the matching state.

The pool has two tiers:

- **GPU tier (primary)** — pre-allocated N slots on GPU (N = configurable, scales with VRAM). Each slot holds one 52 MB snapshot. LRU eviction: when all slots are full, the least-recently-used snapshot is evicted to CPU before the slot is reused.
- **CPU tier (eviction backup)** — pinned host memory. Evicted GPU snapshots are D2H-copied here. On cache hit, the snapshot is H2D-restored to a GPU slot. Independent LRU with a byte-budget cap; entries beyond the cap are dropped.

**Tier rules:**
- GPU miss may restore from CPU
- CPU hit may promote to GPU when space is available
- GPU eviction may drop only the GPU copy when CPU still has the snapshot
- CPU eviction removes recurrent restore eligibility for that boundary

Snapshot keys must include the token-hash lineage, adapter identity if applicable, and boundary position. A snapshot computed for one token lineage or adapter must never be reused for another.

### Match policy

Radix lookup may find KV beyond the latest available snapshot. Qwen3.5 can only restore to boundaries with *both* KV and snapshot.

**Rules:**

- Matches shorter than one interval (256 tokens) → cold prefill
- Longer matches → restore to nearest joint boundary, prefill suffix only
- GPU-tier hits: D2D restore (sub-ms)
- CPU-tier hits: H2D transfer (~2.18 ms)

After restore, suffix prefill starts from the restored boundary.

### Eviction

Eviction is LRU. The GPU tier and CPU tier each maintain independent LRU ordering.

**Events that refresh LRU state:**
- Successful snapshot restore
- Snapshot insertion
- Promotion from CPU to GPU

A Qwen3.5 prefix hit requires both KV and snapshot availability. If KV exists but the snapshot was evicted, the boundary is not restorable. If the snapshot exists but KV was evicted, the boundary is not restorable.

GPU eviction removes the least-recently-used GPU snapshot. The CPU copy may remain valid. CPU eviction removes the least-recently-used CPU snapshot and removes that boundary from recurrent restore eligibility.

Active requests must hold or copy restored state so LRU eviction cannot corrupt in-flight work.

```
Request arrives

 ├─ Radix tree lookup → match at boundary P (KV + snapshot)

 ├─ Snapshot on GPU? → full hit: restore D2D, prefill suffix only

 ├─ Snapshot on CPU? → CPU hit: H2D restore (2.18 ms), prefill suffix only

 └─ No match → cold prefill: full pass, checkpoint snapshots every 256 tokens
```

## Pitfalls

- **GDR chunk boundaries are non-negotiable**: snapshots can only be taken at 64-token multiples (GDR chunk size). Mid-chunk the recurrent state is undefined because the kernel has partially consumed a chunk but not emitted a complete state update. Attempting finer granularity will produce garbage restores.
- **Joint availability requirement**: a prefix hit needs *both* KV blocks and snapshot at the same boundary. KV eviction without snapshot eviction (or vice versa) breaks restore. The radix lookup must consult both pools before declaring a hit.
- **Snapshot key must include adapter identity**: base-model and LoRA snapshots at the same token positions are different recurrent states. Reusing a snapshot across adapters will corrupt decode. Follow Qwen3's `compute_salt_hash` pattern (adapter name → salt → hash chain).
- **Active request protection**: LRU eviction must not reclaim snapshots in use by in-flight requests. Hold GPU snapshots as strong Arc references or copy to request-local scratch before suffix prefill begins.
- **CPU-tier H2D is blocking**: a CPU-tier hit pays 2.18ms synchronous transfer on the request thread. Don't count it as a "full hit" in TTFT metrics — it's a partial win over cold prefill, not a GPU-tier win.
- **Cold prefill overhead is non-negotiable**: every 256-token boundary pays ~2ms D2H snapshot copy during cold prefill. A 4096-token cold prompt = +32ms baseline overhead. This is the price for enabling future hits.

## Next

**Blocker: paged-prefill migration**. Snapshot checkpointing requires pausing at GDR chunk boundaries to D2H-copy recurrent state. This is only feasible after prefill writes KV directly into pages in chunks—the same refactor that removes the ~640 MB HND staging footprint.

**After paged-prefill lands:**
1. Implement two-tier snapshot pool (GPU + CPU, LRU, Arc-based active-request protection)
2. Wire snapshot checkpoints into prefill executor at 256-token intervals (D2H at chunk boundaries)
3. Extend `RequestKv::match_and_add_prefix` to require joint KV+snapshot availability
4. Add cache-hit metrics: GPU-tier / CPU-tier / miss rates, per-request restore latency breakdown
5. Validate warm TTFT win on repeated long prompts (target: comparable to Qwen3's 8.7× speedup, accounting for H2D cost on CPU-tier hits)
6. Tune interval (256 → 128 or 512?) after observing real workload hit-rate distribution vs cold overhead trade-off
