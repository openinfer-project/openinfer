# GLM5.2 × pegaflow: host-tier KV offload → P/D disaggregation

> **TL;DR:** M1 is implemented (`feat/kv-offload-shared-host`): shared `OffloadHost` + 8 rank instances on one namespace, 99 arenas/rank (78 MLA + 21 index-K), restore at admission / save on release, behind `--kv-offload` (+`--kv-offload-hugepages`). Device-side KV layout is verified byte-identical to vLLM's (our kernels are ports); the M2 gap is block hashing only. M2 = cross-engine P/D (vLLM prefill → openinfer decode) via the #540 hash-compat pattern. Design pins to pegaflow-core **v0.23.2 rev d46fd16** (the vendored dep), not the local v0.22.6 checkout.
>
> **Last touched:** 2026-07

## Why pegaflow first

The standing decision is prefill-by-vLLM (no dedicated GLM5.2 prefill path). P/D disaggregation means the decode engine ingests KV that someone else produced — and the ingestion mechanism (query a shared tier by block hash, lease, load into pool pages, commit as matched prefix) is byte-for-byte the same mechanism as host-tier offload restore. M1 builds and gates that mechanism against ourselves; M2 only changes who wrote the blocks.

## What already exists (seam map, 2026-07-06)

- `openinfer-kv-offload` is an **in-process** bridge: `OffloadEngine::new(config, &KvBuffer, stream)` builds a `PegaEngine` in the same process (`engine.rs:299`); no server, no metaserver unless `P2pConfig` (RDMA) is set. Minimal deployment = host pinned-memory tier only.
- qwen3 drives the full loop today: save sealed blocks (`executor.rs:1504`), CPU-tier query (`:1665`), lease + load + `commit_loaded_blocks` (`:1724`). GLM5.2 stops at the GPU-only `match_and_add_prefix` (`scheduler/mod.rs` admission); the CPU-tier leg is the missing wiring, and `BlockPool` already exposes the split point (`probe_prefix` → `cpu_query_hashes` → `commit_loaded_blocks`, `openinfer-kv-cache/src/pool.rs:169-253`).
- Keying: openinfer serializes its xxh3 `PositionalLineageHash` u128 as 16 big-endian bytes (`pool.rs:602`); pegaflow treats `BlockKey { namespace, hash }` as opaque bytes and never re-hashes.

## M1 — single-engine host-tier offload (GLM5.2 ↔ GLM5.2)

1. **Registration geometry**: GLM5.2 has *two* per-layer arenas sharing pool block ids — MLA fp8_ds_mla (64-token page × 656 B/token) and index-K (64 × (128+4) B, full-indexer layers only). pegaflow's `KVCacheRegistration` models one arena per "layer", so register **two pegaflow layers per model layer** (`glm52.L{n}.mla`, `glm52.L{n}.idxk`), both blocks-first single-segment with their own `block_stride_bytes` via `register_context_layer_batch_strided` (v0.23.2, `lib.rs:301`). Save/load must move both or neither — a page whose MLA restored but whose index-K didn't is silent corruption; make the save entry per (block, both-arenas) atomic at our bridge layer.
2. **One shared host tier, one namespace, 8 rank instances** (user input 2026-07-07): MLA is naturally pooled — the latent has no TP/head sharding and the non-expert weights are replicated, so the same token prefix produces byte-identical KV on every DP rank. Therefore do NOT build 8 isolated engines (which would silo the host tier and inherit the admission "hit rate ÷8" problem): build **one `PegaEngine`** (one pinned pool) and register each rank's arenas as its own pegaflow *instance* (`glm52-rank{r}`, per-rank `device_id`) under a **single shared namespace**. Any rank then restores what any rank saved — the ÷8 placement problem dissolves at the host tier (GPU-tier placement, S3, remains). `OffloadEngine` owns its `PegaEngine` 1:1 today; M1 needs a shared-engine constructor (`Arc<PegaEngine>` in, per-rank instance registration).
3. **Host pool = hugepages on jz-38**: the box has RAM to spare and `PegaEngine::new_with_config(pool_size, use_hugepages, …)` supports it directly; bench the tier with hugepages on (note the jiuzhang platform reclaims node-38 hugepage reservations at reboot — re-check `HugePages_Total` before runs).
4. **Hook**: admission's `match_and_add_prefix` grows the CPU leg exactly like qwen3's executor — GPU match first, then `cpu_query_hashes` against pegaflow, lease + async load into freshly-reserved pool pages, `commit_loaded_blocks`, and only then report `cached_tokens`. Loads must complete before the request's first step (admission is a step boundary; block on the `LoadHandle` like qwen3 does).
5. **Save policy**: sealed (content-addressed) blocks flow to the host tier on release, same as qwen3's `save_sealed_blocks`. dspark × prefix-cache exclusivity carries over unchanged — drafter on ⇒ no prefix matching ⇒ offload restore off (#590 owns lifting that).
6. **Gates** (jz-38, run 2026-07-06, ALL PASS): evict-restore byte parity — a 1466-token prompt's greedy completion byte-identical cold → warm-restored (22 blocks from host) → no-offload server; TTFT 5371 ms cold vs **157.6 ms warm-restored (~34×)**, offload-on miss path 5064 ms (within single-sample noise of no-offload); 16-way mixed re-run after eviction = 15 full-prefix restores (42-43 blocks), 0 errors, 2.2 s wall; hugepage pool verified (`HugePages_Free` 17408 → 1024 at launch); zero host-tier warns across the 160-prompt churn.

## M2 — cross-engine P/D (vLLM prefill → openinfer decode)

Blocked on M1. Two hard problems, both with prior art:

- **Hash compat**: openinfer xxh3-lineage vs vLLM SHA-256 `block_hashes` never collide onto the same keys. PR #540 (qwen3) already built the pattern: compute vLLM-compatible hashes for the prompt at admission and query the vLLM-written namespace. Port that, plus the namespace derivation (vLLM side = SHA-256 over model/dtype/tp/heads/... — `connector/common.py:210`; ours must reproduce it byte-for-byte).
- **Layout parity**: verified at source level 2026-07-06 against the sibling vLLM checkout (`/data/code/workspace-rustllm/vllm`, `cdab28319`) — byte-for-byte MATCH on everything device-side, because our kernels are ports of vLLM's: 656 B/token field order `[512 fp8 NoPE][4×f32 scale, ÷448, group=128][64×bf16 RoPE]` (vLLM `cache_kernels.cu:447-547` ↔ ours `glm52_mla_assembly.cu:33-34`), index-K block-split `[64×128 fp8][64×4 B f32 scale]` (vLLM `cache_kernels.cu:550-607` ↔ ours `glm52_indexer.cu:69-87`, which cites the vLLM kernel by name), page=64, layer-outermost page-contiguous arenas both sides. Remaining gate before e2e: dump one block per arena from each engine for the same prefix and byte-compare (guards silent drift on either side's bumps). One structural note: vLLM keeps MLA and indexer caches as separate KV groups; we index both off one pool block id — irrelevant to byte parity, matters only for block-id translation.

## Open problem: the tail blocks P/D can't ship (2026-07-07)

The content-addressed tier only ever holds *sealed* 64-token blocks, and the reuse cap is `cacheable = (prompt−1)/64` — the tail partial page plus the prompt's final token never enter it (the last token must forward to produce first-step logits). M1 measured the cost: with 96% of a 1466-token prompt restored, warm TTFT is still 157.6 ms, ~150 ms of which is prefilling the ≤64 uncached tail tokens; the restore itself (query + 76 MB H2D) is ~3-5 ms. Harmless for host-tier caching, but in M2 P/D over pegaflow P2P this becomes a **per-request floor on the decode side**: every handed-off request re-prefills up to 64 tokens no matter how perfect the transfer is. Directions if/when it matters: partial-block entries keyed by `(hash, valid_len)` (needs pegaflow + kvbm contract changes — kvbm deliberately refuses to register unsealed blocks), or the prefill side shipping the boundary step's sampled token + KV so decode starts at step 1 (a protocol change, not a cache entry). Deferred until M2 shows the 150 ms floor actually hurts the target workload.

## Pitfalls pinned during the survey

- The local `/data/code/pegaworkspace/pegaflow` checkout is v0.22.6 and **lacks the `*_inproc` and `_strided` APIs** the integration uses; read the vendored `~/.cargo/git/checkouts/pegaflow-*/d46fd16/` instead.
- The PyO3 package is a gRPC client — reference contract only, not our path.
- vLLM connector requires `storage_offset()==0` and registers CUDA-IPC handles; our in-process path passes raw device pointers and skips IPC entirely.

## M1 implementation notes (2026-07-06, `feat/kv-offload-shared-host`)

- `OffloadHost` (openinfer-kv-offload) owns the pinned pool + runtime + P2P lifecycle; `OffloadEngine::with_arenas_on(host, ...)` registers a rank as one more pegaflow instance over the shared pool. `use_hugepages` is a `HostConfig`/CLI knob.
- Each rank registers 99 arenas in one instance (78 `glm52.L{n}.mla` + 21 `glm52.L{n}.idxk` — full-indexer layers are `{0,1,2} ∪ {6,10,…,74}`, NOT a 5-layer set), so one save/load entry per block moves both caches atomically. `Glm52RankModel::kv_arenas` asserts allocation sizes against the geometry at registration.
- Restore leg (`scheduler/offload.rs`): at admission — a step boundary, all ranks joined — probe → query → load into `reserve_loaded_blocks` pages → blocking `wait()` → `commit_loaded_blocks`; the probe is held across `match_and_add_prefix` so the committed blocks can't evict before the re-match. Save leg: on release, fire-and-forget, `assigned_block_guards` pin the pages through the D2H copy; prefix-matched head skipped (already host-resident).
- Launch rejects `--kv-offload` with `--no-prefix-cache` or the DSpark drafter. Blocking the coordinator on the load stalls all 8 ranks for the restore duration — the accepted M1 cost; qwen3-style per-tick polling is the follow-up lever if restore latency shows up in ITL.
- jz-38 hugepages: `echo N > /proc/sys/vm/nr_hugepages` works live as root (platform still zeroes it at reboot).

## Next action

Land the M1 branch once the jz-38 gates pass (evict-restore byte parity via churn eviction, mixed load with restores, hugepage pool, no-offload parity anchor). Then M2: #540-pattern vLLM hash compat + the byte-dump layout gate below.
