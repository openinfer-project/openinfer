# Server startup-to-ready time

**TL;DR**: Qwen3-4B warm (page-cache-hot) startup-to-HTTP-ready cut 3.25s → ~1.45s by (1) loading the engine on a blocking thread so the vLLM frontend's ~1.15s tokenizer/chat-template load runs concurrently, and (2) keeping the source safetensors mmap alive for the model's lifetime instead of paying ~0.4s of munmap page-table teardown on the critical path. Readiness semantics unchanged: HTTP binds only after the engine bridge registers.

Last touched: 2026-06

## Warm-start breakdown (RTX 5070 Ti, Qwen3-4B, 8GB bf16)

Baseline 3.25s was three serial phases:

| Phase | Cost | Fix |
| --- | --- | --- |
| Engine load (CUDA ctx 0.18s + H2D upload ~1.1s @ ~7GB/s pageable) | 1.3s | kept (pageable `clone_htod`; pinned staging judged not worth the complexity — see below) |
| munmap of the 8GB touched source mmap | 0.42s | mmap now lives in `Qwen3Model::_weight_source` until model drop |
| vLLM frontend: tokenizer (11MB tokenizer.json) + text/chat backends | 1.15s | runs concurrently with engine load |

After: engine path ~1.35s and frontend path ~1.25s overlap; ready ≈ max of the two + process start ≈ 1.45s.

## The munmap trap

munmap of ~8GB of touched pages costs ~0.4s of kernel page-table teardown holding the process mmap lock. Dropping the mmaps "in the background" just moves the stall to whatever allocates next — cudaMalloc (KV cache), cuBLAS init on the worker thread, or first-decode graph capture all take the same lock. We measured the 0.4s gap migrating from `kv_budget` → `KvCacheManager::new` → `RankWorker::spawn` as the drop point moved. Keeping the mapping is free: the pages are clean file-backed page cache the kernel reclaims under pressure, so the only cost is RSS accounting.

## Concurrent startup invariant (openinfer-vllm-frontend)

`serve()` takes `impl Future<Output = Result<EngineHandle>>`. vllm-server starts immediately (loads tokenizer, then waits in `EngineCoreClient::connect` for the engine to register); a spawned task awaits the engine, sets the shared `ServableCap`, and runs the `LocalEngineBridge`. Invariant chain: `ServableCap::set` → bridge registers → vllm-server builds the router (guard included) → HTTP binds. So a reachable port still means the engine can serve, and the sampling guard never reads an unset cap (unset → loud 503, dead path by construction).

Consequences to keep in mind:

- The bootstrapped-transport `ready_timeout` now bounds the **whole engine load** (it used to start after load). It is 30min so multi-GPU MoE loads and cold starts fit; load *failure* cancels the server immediately via the engine task, the timeout only catches a truly hung load.
- The graceful ctrl-c handler is installed only **after** the engine load resolves (`cancel_token_on_ctrl_c`). The blocking load can't be cancelled, so during load SIGINT keeps its default kill behavior — same as before the change. (When testing this: background jobs of non-interactive shells inherit SIGINT=SIG_IGN; use a spawner that restores default dispositions.)
- LoRA mode stays sequential: the LoRA routes need the handle when the router is built.

## Rejected: pinned-staging H2D upload

Upload runs at ~7GB/s from pageable mmap memory. A pinned double-buffer pipeline would reach CPU-memcpy speed (~14GB/s, ~0.55s) but the frontend path (~1.25s, external vllm crates) is the floor — e2e gain ≲0.1s for an unsafe staging pipeline. Revisit only if the tokenizer load gets faster or engine-only load time (tests/benches) becomes the pain.

## Next

None active. The remaining floor is the external vllm tokenizer/backend load (~1.15s); revisit if upstream gets faster or if cold-start (bandwidth-bound, per roadmap out of scope) becomes a target.
