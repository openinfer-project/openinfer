# External PegaFlow server for KV offload

> **TL;DR:** OpenInfer is an RPC client selected by `--kv-offload-server`; the external PegaFlow process **allocates and owns the GPU KV arena** and the host/SSD/RDMA tiers. OpenInfer registers its KV layout, imports the arena over CUDA IPC, and runs attention kernels on the imported mapping. Implemented for Qwen3; GLM5.2's multi-arena layout is not supported by the v1 contract.
>
> **Last touched:** 2026-07

## Ownership boundary

- External mode is the only PegaFlow integration. OpenInfer no longer embeds a `PegaEngine` or constructs a second pinned-memory pool.
- **PegaFlow owns the fused GPU KV arena.** Registration sends the layout (per-layer offsets, sizes, block strides, total size); the server allocates the arena on the requested device, zeroes it, registers the raw layer pointers into its engine, and returns a 64-byte CUDA IPC handle in the registration response. OpenInfer imports the handle and builds its `KvBuffer` as a non-owning view.
- Why this direction: GPUDirect RDMA registration (`ibv_reg_mr`/dma-buf) only works on memory the registering process owns — an IPC-imported pointer can never back it. Server-side allocation makes the arena NIC-registerable in the server without any fd side-channel; the IPC handle is plain bytes and rides the existing gRPC response.
- `--kv-offload-server` enables the client. `--kv-offload-namespace` identifies the checkpoint/deployment content domain; vLLM compatibility uses the connector namespace.
- Server capacity, SSD, RDMA, routing, and topology configuration stay outside OpenInfer.

## Wire contract

- One fused arena per `(instance, device)`. `RegisterContextRequest.native_kv_tensors` carries per-layer `{offset_bytes, size_bytes, block_stride_bytes}` views into it, and `native_alloc_size` the total size; `RegisterContextResponse.arena_ipc_handle` returns the CUDA IPC handle.
- Native registration has an exact capability version (`+native-arena-v1`). Old and new clients fail before any memory changes hands.
- Native `Load` sets `wait_for_completion` and returns only after the GPU transfer completes. The Python connector retains its shared-memory completion path.
- `Flush` waits for previously submitted saves to become cache-visible and for queued MetaServer registrations to be attempted.

## Lifetime and failure policy

- The arena lives exactly as long as the registration: unregister (or the server's session/HTTP cleanup, if the client dies) frees it. Teardown order on the client is workers → close IPC mapping → unregister.
- The client treats the server as load-bearing for its own GPU memory: a broken session stream, a failed save/load transport, or an unregister failure exits the process. There is no reconnect.

## Model layouts

- Qwen3 registers one page-first fused arena with strided per-layer views.
- GLM5.2's rank-local MLA and index-K arenas (78 + 21 per EP8 rank) need multiple allocations per instance, which the v1 contract does not cover; `OffloadEngine::with_arenas_on` fails before touching the server.

## Validation

- PegaFlow `native_arena_rpc_e2e` (real GPU): register → child process imports the handle and writes a pattern → save → wipe → `wait_for_completion` load → bit-exact restore → unregister frees the arena.
- OpenInfer `cpu_roundtrip` and `kv_offload_cpu_hit` run against a live server via `OPENINFER_PEGAFLOW_SERVER` (same host; CUDA IPC is host-local).
