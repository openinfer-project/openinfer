# GLM5.2 × pegaflow: host-tier KV offload → P/D disaggregation

> **TL;DR:** Design record (no code yet). The pegaflow in-process bridge already exists and serves qwen3 (`openinfer-kv-offload` embeds `PegaEngine` directly); GLM5.2 integration is an extension, not greenfield. M1 = single-engine host-tier offload wired into the admission prefix-match; M2 = cross-engine P/D (vLLM prefill → openinfer decode) via the #540 hash-compat pattern. Design pins to pegaflow-core **v0.23.2 rev d46fd16** (the vendored dep), not the local v0.22.6 checkout.
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
6. **Gates** (jz-38): byte parity — evict-then-restore a 1200-token varied prompt and require the warm output byte-identical to the resident-prefix run; step-bench flat vs the D5/D10 anchors (save is off the step path, restore is admission-side); 17-way mixed load with restores zero-error; `--no-prefix-cache` disables the whole leg.

## M2 — cross-engine P/D (vLLM prefill → openinfer decode)

Blocked on M1. Two hard problems, both with prior art:

- **Hash compat**: openinfer xxh3-lineage vs vLLM SHA-256 `block_hashes` never collide onto the same keys. PR #540 (qwen3) already built the pattern: compute vLLM-compatible hashes for the prompt at admission and query the vLLM-written namespace. Port that, plus the namespace derivation (vLLM side = SHA-256 over model/dtype/tp/heads/... — `connector/common.py:210`; ours must reproduce it byte-for-byte).
- **Layout parity**: vLLM's GLM5.2 runs the same FlashMLA-sparse fp8_ds_mla + DeepGEMM indexer contracts, but byte-level identity of both arenas (656 B token layout, index-K packing, scale placement) must be *proven*, not assumed — gate: dump one block from each engine for the same prefix and byte-compare before any e2e. Also reconcile page granularity (our fixed 64-token page vs vLLM scheduler-block × dcp/pcp virtual blocks).

## Pitfalls pinned during the survey

- The local `/data/code/pegaworkspace/pegaflow` checkout is v0.22.6 and **lacks the `*_inproc` and `_strided` APIs** the integration uses; read the vendored `~/.cargo/git/checkouts/pegaflow-*/d46fd16/` instead.
- The PyO3 package is a gRPC client — reference contract only, not our path.
- vLLM connector requires `storage_offset()==0` and registers CUDA-IPC handles; our in-process path passes raw device pointers and skips IPC entirely.

## Next action

M1 step 1 (done, `feat/kv-offload-arena-registration`): `KvArena` + `OffloadEngine::with_arenas` extend the registration to explicit multi-arena geometry; qwen3's `from_buffer` delegates to it. Step 2: the shared-`PegaEngine` constructor (one host pool, 8 rank instances, one namespace), then the GLM5.2 arena exposure + admission CPU leg.
