# External PegaFlow server for KV offload

> **TL;DR:** OpenInfer is a CUDA-IPC/RPC client selected by `--kv-offload-server`; the external PegaFlow process owns the host, SSD, and RDMA tiers. Native registration, terminal Load completion, and Flush are implemented for Qwen3 and GLM5.2, with final-head cross-process replay remaining before deployment.
>
> **Last touched:** 2026-07

## Ownership boundary

- External mode is the only PegaFlow integration. OpenInfer no longer embeds a `PegaEngine` or constructs a second pinned-memory pool.
- OpenInfer owns model KV allocations and exports them through CUDA IPC. PegaFlow imports those allocations, owns the storage hierarchy, and performs GPU↔host transfers.
- `--kv-offload-server` enables the client. `--kv-offload-namespace` identifies the checkpoint/deployment content domain; vLLM compatibility uses the connector namespace.
- Server capacity, SSD, RDMA, routing, and topology configuration stay outside OpenInfer.

## Wire contract

- Each registered layer carries one CUDA IPC allocation handle plus view offset, view size, and block stride. This avoids positionally coupling a second stride array to the layer list.
- The server validates the allocation device and view bounds, opens each allocation once per registration batch, and keeps the mapping alive until unregister.
- Native registration has an exact capability version. Old and new clients fail before importing memory instead of silently assuming a dense layout.
- Native `Load` returns only after the GPU transfer completes. The Python connector retains its shared-memory completion path.
- `Flush` waits for previously submitted saves to become cache-visible and for queued MetaServer registrations to be attempted.
- External offload uses exportable CUDA allocations; the default offload-disabled path retains the stream-ordered allocator.

## Model layouts

- Qwen3 registers one page-first fused allocation with strided per-layer views.
- GLM5.2 registers rank-local MLA and index-K arenas by name: 78 MLA plus 21 index-K arenas per EP8 rank.
- Both clients preserve PegaFlow query leases and load only after a host-tier hit has produced a valid lease.

## Validation

Current PR-head local gates are green:

- PegaFlow server: 23 unit tests; core: 124 passed with one GPU-only test ignored; CUDA 12/13 checks and Python wheel builds pass.
- OpenInfer KV offload: 9 unit tests; workspace CPU tests, simulated frontend E2E, SM80 CUDA compile, and SM80 CUDA clippy pass.

Cross-process evidence collected during bring-up established the layout and transfer contract:

- Qwen page-first save/load restored three blocks into different HBM block IDs byte-for-byte; an untouched block remained zero.
- Qwen3 forced CPU-only and combined GPU/CPU prefix restores without a material head-logprob shift.
- GLM5.2 EP8 registered 99 arenas per rank and restored one 64-token host block with the expected first output tokens.

Those runs predate the final PR heads. Replay the Qwen and GLM5.2 gates after PegaFlow #407 and OpenInfer #729 are merged; do not treat the historical runs as deployment evidence.
