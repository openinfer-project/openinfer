# External PegaFlow server for KV offload

> **TL;DR:** KV offload now has one ownership boundary: OpenInfer is a CUDA-IPC/RPC client selected by `--kv-offload-server`, while the external PegaFlow server owns storage and acknowledges transfer/drain barriers before exported GPU memory can be reused; Qwen3 and GLM5.2 cross-process restore gates pass on 8×H200.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — routes this cross-model change to the runtime subsystem.
  - `docs/subsystems/runtime/pegaflow-offload-integration.md` — records the current in-process ownership model, transfer ordering, leases, and Qwen3 CPU-hit gate.
  - `docs/models/qwen3/prefix-cache.md` — establishes full-block matching, the one-token recompute boundary, and cache-lifetime invariants.
  - `docs/models/qwen3/serving-performance.md` — retains the current host-tier offload baseline and pure-L2 measurement shape.
  - `docs/models/glm52/pd-m2-execution.md` — establishes GLM5.2's 99-arena page-first layout, PegaFlow version boundary, and strict restore semantics.
  - `openinfer-kv-offload/src/{lib.rs,engine.rs}` — `OffloadEngine` currently owns an in-process `PegaEngine`; the scheduler-facing save/query/load API is already shared by the model integrations.
  - `openinfer-qwen3/src/{lib.rs,executor.rs}` — Qwen3 constructs one offload engine around its fused page-first `KvBuffer` and already has asynchronous prefetch admission.
  - `openinfer-glm52/src/lib.rs` — GLM5.2 shares one `OffloadHost` across eight rank instances and registers 78 MLA plus 21 index-K arenas per rank.
- **Relevant history**:
  - PegaFlow PRs #331/#333 made strided page-first registration and in-process completion usable by OpenInfer.
  - OpenInfer #316 proved Qwen3 host-tier save/load correctness with an embedded pool.
  - OpenInfer #600/#657 proved GLM5.2 host offload and cross-node target-KV restore with an embedded pool.
  - The development cluster already runs a PegaFlow DaemonSet on selected nodes; the first remote step is read-only inventory, and the production daemon must not be modified for bring-up.
- **Plan**:
  1. On the reserved 8×H200 development node, inspect the PegaFlow DaemonSet endpoint, version, server flags, CUDA IPC registration contract, available model paths, and existing OpenInfer/PegaFlow worktrees without changing the running daemon.
  2. Replace the embedded `PegaEngine` with an external PegaFlow gRPC client while preserving the scheduler-visible query/lease/load/save semantics.
  3. Reuse the server's CUDA IPC registration when it can represent Qwen3's strided fused buffer and GLM5.2's named multi-arena/page-first layout. If it cannot, extend PegaFlow's protocol and server with the smallest layout-general registration primitive, verify its own integration tests, and prepare an upstream PegaFlow PR.
  4. Expose one explicit OpenInfer server configuration for the external endpoint; fail at startup on incompatible combinations, version/layout mismatch, missing server health, or registration failure.
  5. Gate Qwen3 with an external-server register → save → HBM eviction → load path, byte/logit parity, and a repeated-prompt serving check that proves a host-tier hit rather than a GPU-prefix hit.
  6. Gate GLM5.2 with all eight ranks registered against one external server, restore of both MLA and index-K arenas, token parity after forced host-tier restore, and a repeated-prompt serving check with no local GPU-prefix hit.
  7. Run formatting, focused unit/integration tests, and cross-process correctness/latency measurements before review. Record deployment commands without internal hostnames or private paths, then submit the required OpenInfer and PegaFlow PRs.
- **Risks / open questions**:
  - Cross-process registration requires CUDA IPC handles rather than raw device pointers; handle lifetime, device identity, and the server's embedded Python/CUDA context must be verified on the real daemon.
  - The existing gRPC schema may assume vLLM tensor metadata and may not expose GLM5.2's named arena ordering. A protocol change is acceptable, but silent layout coercion is not.
  - Qwen3 uses one engine per device while GLM5.2 uses one host shared across eight rank registrations. The external client must preserve that ownership distinction without introducing a generic abstraction above the transfer layer.
  - The daemon's pinned pool is shared infrastructure. Bring-up needs isolated namespaces and must not evict, stop, or reconfigure unrelated cache contents.
  - Correctness evidence must force HBM misses; a warm request that hits OpenInfer's local prefix cache does not prove external offload.

## Execution Log

### Step 1: development-cluster inventory

- Connected to the reserved 8×H200 development node and inspected the cluster without modifying workloads.
- The cluster runs a host-networked PegaFlow DaemonSet with host IPC, all GPUs initialized, and a MetaServer connection. Its deployed protocol predates the native Rust registration added here, so compatibility must be checked rather than assumed.
- The reserved node was occupied by unrelated inference processes, which were left untouched. Verification moved to an isolated 8×H200 development pod with local Qwen3 and GLM5.2 checkpoints.
- The production daemon already owns CUDA contexts and host hugepages. External mode must use that pool instead of creating another embedded pool.

### Step 2: external-only ownership boundary

- Removed the embedded `PegaEngine`, local pinned-pool construction, local P2P service, and their CLI knobs. `--kv-offload-server` is now the sole enablement switch; server resource configuration stays outside OpenInfer.
- Added a native PegaFlow registration payload carrying CUDA IPC handles, per-view offsets/sizes, and optional block strides. This represents both Qwen3's page-interleaved fused allocation and GLM5.2's MLA/index-K arenas without Python tensor wrappers.
- Added a PegaFlow `Flush` RPC for the existing save/registration visibility barrier. Native loads now await the server's GPU worker through the Load RPC; the Python connector retains its shared-memory completion path.
- KV allocations use legacy-exportable CUDA allocation only when external offload is enabled. The default path retains the stream-ordered allocator. A direct GPU probe confirmed that `cudaIpcGetMemHandle` rejects `cudaMallocAsync` allocations and accepts `cudaMalloc` allocations.
- Local gates passed: PegaFlow server unit tests (26), OpenInfer KV-offload unit tests (8), Qwen3 unit tests (75), default-feature server compile/tests, and a GLM5.2 feature compile using NCCL 2.30.7. The local GPU is not SM90, so real GLM5.2 kernels and behavior still require the 8×H200 target.

### Step 3: 8×H200 cross-process gates

- Built the PegaFlow server without RDMA and built the GLM5.2 OpenInfer feature on the target SM90a system. The build exercised TileLang 0.1.12 code generation, FlashMLA, DeepGEMM, NCCL 2.30.7, and the final server link.
- The raw Qwen page-first gate saved three blocks through a separate PegaFlow process, loaded them into different HBM block ids, and compared every layer/segment byte-for-byte. The negative-control block stayed zero. The test body took 0.44 s; a hot build plus test took 1.20 s.
- The Qwen3-4B executor gate forced both a CPU-only prefix restore and a combined GPU+CPU restore. Warm/cold head-logprob mean deltas were 0.0105 and 0.0242 nat; maxima were 0.0652 and 0.0632. The test body took 5.61 s.
- GLM5.2 EP8 registered eight rank instances with 99 arenas each (78 MLA plus 21 index-K) through native CUDA IPC. A 66-token cold request saved one logical block across all 99 arenas: 3,452,160 bytes, with the PegaFlow D2H worker reporting 0.50 ms.
- Nine distinct 3,900-token requests pinned to the same data-parallel rank displaced the original prefix from its 520-block HBM LRU. Replaying the original request restored one block from PegaFlow, returned `cached_tokens=64`, reproduced the same first two output tokens, and logged `GLM5.2 host-tier restore: 1 blocks committed`. PegaFlow performed 99 H2D copies (3,452,160 bytes) in 0.62 ms. End-to-end request time was 0.256 s cold and 0.044 s after host-tier restore; this is a correctness-gate observation, not a serving benchmark.
- The isolated OpenInfer and PegaFlow processes were stopped after the gate; all eight GPUs returned to zero allocated memory. The development pod and source mounts remain available.

### Step 4: failure and multi-tenant review

- Scoped PegaFlow's asynchronous deeper-tier prefetch state by `(instance_id, req_id)`. A regression test keeps two equal request IDs in separate active entries and proves that polling one instance cannot consume the other's result.
- Replaced process-ID instance suffixes with UUIDv4 and made native content namespaces explicit. Native Qwen3 and GLM5.2 launches now require `--kv-offload-namespace`, a stable checkpoint/deployment identity; vLLM compatibility continues to use the connector namespace.
- Deadlines remain on health, query, release, and flush. Ownership RPCs do not carry a cancelable gRPC deadline because an accepted CUDA DMA cannot be canceled with its handler; a 30-second local watchdog or an indeterminate transport failure is fail-stop. Native load completion is returned by the Load RPC instead of a shared-memory poll.
- Added an owned session guard and explicit client shutdown. Shutdown waits for accepted saves, sends `UnregisterContext`, and releases the session only after the server acknowledges both GPU queue drain and CUDA mapping close. Failure to obtain that ownership acknowledgement is fail-stop.
- PegaFlow unregister keeps a closing instance addressable to concurrent cleanup calls, rejects new transfers, places barriers behind every accepted load/save on every registered device, and removes the instance only after the shared drain completes. The worker threads retain join handles for deterministic teardown.
- Registration and cleanup are serialized per instance and by session generation. A canceled registry reply rolls back a newly published CUDA mapping; session cleanup still closes registry state when engine registration never completed.
- Kept legacy-exportable allocation behind the offload option so an offload-disabled launch uses the original allocator.
- Failure-path local gates pass: PegaFlow server 28/28, PegaFlow core 126/126 with one GPU-only test ignored, and OpenInfer KV-offload 9/9.
- Re-ran the cross-process gates after the lifecycle fixes. The raw Qwen round trip remained byte-exact at 0.43 s versus 0.44 s before the fixes. The hot Qwen3 executor gate took 3.32 s; CPU-only and combined GPU/CPU restores kept mean head-logprob deltas at 0.0105 and 0.0272 nat.
- Re-ran GLM5.2 with a 520-block rank-local HBM pool. After rank-pinned displacement, the original 66-token request restored one host block, returned `cached_tokens=64`, reproduced the same first two tokens, and completed in 0.0438 s versus 0.257 s cold. A graceful stop drained all eight rank instances, closed 99 CUDA mappings per rank, emptied the server instance list, and returned all eight GPUs to zero allocated memory after the PegaFlow process exited.

## Debrief

The embedded and external modes represented two owners for the same host-tier state and doubled every lifecycle decision. Removing the embedded mode left a stable boundary: model crates describe GPU arenas, the client exports allocation-relative CUDA IPC views and preserves lease semantics, and PegaFlow owns storage resources and process-level cleanup.

The runtime gates caught environment assumptions that are worth keeping outside the product interface: a server build needs a full protobuf toolchain, GLM5.2 code generation needs the validated TileLang version, and a NIC-less EP8 development node must disable DeepEP GIN. None of these became OpenInfer offload flags.

Independent failure-path review is READY. Merge the PegaFlow protocol/server PR before the dependent OpenInfer client PR, then deploy matching revisions together.
