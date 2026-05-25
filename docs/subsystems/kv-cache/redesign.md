# KV Cache Redesign

TL;DR: Qwen3 KV 边界已落地：scheduler 只提交 token 级请求生命周期决策，executor-side `Qwen3KvCache` 把这些决策翻译成 TP 物理 KV 状态和 `KvExecView`，worker 只执行、不分配、不释放、不持有、不 advance KV。Release checks 和 serving smoke 通过；c16 random 记录为 forward workspace admission 问题，不属于 KV ownership 边界。

Last touched: 2026-05

## 动机

旧 paged KV 能跑，但状态所有权、资源分配、执行描述混在 worker 里，随着 p/d 分离和 offloading 的需求会持续放大维护成本。

## 改造前架构

```
Scheduler (单线程)
  ├─ admission control: 按 worst-case KV residency 预留
  ├─ 只知道 page_size / available_pages / max_request_pages
  └─ 不持有任何 KV 状态

Executor (per-rank worker 线程)
  ├─ RequestStateStore: HashMap<RequestId, KvState>
  ├─ KvState 自己持有 OwnedPagePermit + KvPool Arc clone
  ├─ ensure_capacity() 时自行去 pool 抢页
  └─ forward 直接 mutate KvState 并 advance seq_len
```

## 问题

### 1. 两次决策，语义不一致

旧 scheduler 做 admission accounting，worker 做实际 `try_grow`。两层之间没有关联，worker 分配 KV 时没有一个统一资源边界。旧代码能 work 是因为 scheduler loop 单线程串行，但这个隐含假设一旦打破就出问题。

### 2. KvState 持有 KvPool 的 Arc

- 每个请求都持有池的引用，形成 "请求知道池" 的反向依赖
- `ensure_capacity` 自己去 pool 抢页 — 这个行为应该由 KV cache 模块控制

### 3. take/restore dance

`RequestStateStore` 的 `take_batch` / `restore_batch`：每次 forward 前从 HashMap move out KvState，forward 后塞回。这是因为 KvState 是 `!Copy` 且需要 `&mut`。开销不大但：
- 如果 forward panic，KvState 泄漏（页最终通过 permit drop 归还，但状态丢了）
- KV 状态的所有权搅在 worker 里，scheduler 完全摸不到

### 4. p/d 分离和 offloading 不友好

系统需要能回答 "这个请求的 KV 在哪、多大、能不能 offload/迁移"。但旧 KV 状态被锁在各 rank worker 的 `RequestStateStore` 里，调度层只能通过间接接口感知。

## 设计决策

## Preparation

- **Read**:
  - `docs/index.md` — confirms this belongs under `subsystems/kv-cache`.
  - `docs/subsystems/kv-cache/redesign.md` — current KV ownership problem and pending decisions.
  - `pegainfer-core/src/kv_pool.rs` — current `KvPool` / `KvState` / `KvDesc` ownership and grow behavior.
  - `pegainfer-core/src/page_pool.rs` — `OwnedPagePermit` grow and RAII release semantics.
  - `pegainfer-qwen3-4b/src/executor.rs` — worker-local `RequestStateStore` and `take_batch` / `restore_batch`.
  - `pegainfer-qwen3-4b/src/scheduler.rs` and `src/scheduler/*.rs` — scheduler admission, execution, failure, and release flow.
  - `pegainfer-qwen3-4b/src/prefill.rs`, `batch_decode.rs`, `unified_forward.rs` — places where worker-side forward currently grows and advances KV.
- **Relevant history**:
  - No older KV redesign task doc exists; this document is the active record.
- **Plan**:
  1. Establish the responsibility contract in this doc: scheduler speaks request/token, KV cache translates to physical state, worker only executes.
  2. Move Qwen3 per-request KV storage out of worker threads into an executor-side KV cache module, so workers no longer allocate, release, or privately store request KV state.
  3. Make the executor-side KV cache pre-grow capacity for prefill/decode/unified steps and build `KvExecView` values for workers.
  4. Change Qwen3 forward paths to consume `KvExecView` instead of mutating `KvState`.
  5. Run focused checks for Qwen3 executor/core changes and record results.
- **Risks / open questions**:
  - TP 下一个 request 的逻辑 KV 状态会映射到多个物理 KV pool；KV cache 必须把同一个 token-level 决策一次性落到所有物理 pool 上。

## Contract First

Three roles own three different languages:

- **Scheduler** owns request lifecycle decisions. It speaks in `RequestId`, prompt tokens, generated tokens, limits, admission, retirement, and cancellation. It should not know physical page IDs, buffer pointers, padding pages, or per-rank permits.
- **KV cache module** owns resource translation. It answers whether token-level work can be admitted, prepares physical KV state for a forward step, and releases request KV state. Internally it may allocate pages, maintain per-rank state, handle padding pages, or later route to offload / p-d storage.
- **Worker** owns forward execution. It receives a complete execution payload, launches kernels, samples results, and reports success or failure. It does not admit, allocate, grow, release, or privately store request KV state.

The stable contract is:

```text
Scheduler -> KV cache:
  admit_requests(active_token_states, pending_token_requests) -> admission decisions
  prepare_prefill / prepare_decode / prepare_unified(appends) -> per-worker execution payloads
  commit_prefill / commit_decode / commit_unified(appends)
  release(request_id)

KV cache -> Worker:
  execution payload with buffer/page/layout/length metadata

Worker -> Scheduler:
  forward success/failure and generated tokens
```

The code keeps `KvState` as the owner of page permits inside `Qwen3KvCache`, but workers receive only `KvExecView`: a value-type execution descriptor with cloned buffer access, page IDs, committed length, and execution-visible length. That removes the worker-private `RequestStateStore` and puts allocate/grow/release/commit behind one translation module.

## Execution Log

### Step 1: Establish the contract

- Added the contract-first section above: scheduler speaks request/token lifecycle, KV cache translates to resources, worker executes a complete payload.
- Recorded the boundary: `KvState` owns pages inside the KV cache; `KvExecView` is the worker payload.
- Result: success.

### Step 2: Move request KV storage out of worker threads

- Modified `pegainfer-qwen3-4b/src/executor.rs`.
- Added executor-side `Qwen3KvCache` with internal `RankKvStateStore`s and moved `drop_request` plus token-level admission behind it.
- `RankWorker` no longer owns a `RequestStateStore`; `StepCommand` carries the per-rank KV batch for that forward step.
- Worker responses no longer carry KV state back; `Qwen3KvCache` remains the owner throughout execution.
- Result: success.

### Step 3: Move grow out of forward execution

- Added `KvState::capacity_tokens` and `KvState::ensure_prepared_capacity` in `pegainfer-core/src/kv_pool.rs`.
- `Qwen3KvCache` now maps each token append request onto every TP physical KV pool before sending work to workers for prefill, decode, and unified steps.
- Updated Qwen3 `prefill`, `batch_decode`, and `unified_forward` paths to consume `KvExecView` and build paged metadata from execution-visible lengths instead of advancing `seq_len` during forward.
- `Qwen3KvCache` commits `seq_len` only after all ranks return success and the primary worker returns the expected payload type.
- `Qwen3KvCache` now treats physical state presence as an internal invariant and builds `KvExecView` only after the scheduler-thread token decision has been prepared in every TP KV pool.
- Result: success.

### Step 4: Cleanup

- Removed dead `Qwen3Model::alloc_kv`.
- Updated `batch_decode_trace` to build trace KV views through `Qwen3KvCache.prepare_*`.
- Initialized `pegainfer-kernels/third_party/flashinfer` submodule so release checks can find FlashInfer headers.
- Result: success.

### Step 5: Review fixes

- Ran a sub-agent review focused on KV ownership, error paths, rank consistency, and commit semantics.
- Fixed the high-severity finding by moving logical `seq_len` advance out of worker forward paths and into `Qwen3KvCache` commit after rank success.
- Removed the ownership lease from the worker protocol and kept TP physical allocation inside `Qwen3KvCache`; rank remains an implementation detail of KV storage, not a scheduling decision point.
- Result: success.

### Step 6: Serve benchmark smoke

- Started the OpenAI-compatible server with `uv run --with triton cargo run --release --bin pegainfer -- --model-path <Qwen3-4B> --served-model-name Qwen3-4B --port 8000`.
- Verified `/v1/completions` with `max_tokens=4` before load.
- Ran `vllm bench serve` against the local server with random prompts across short decode, long decode, and long prefill shapes.
- Saved raw result JSON under `target/profiling/kv-cache-redesign/`.
- Result: success, 140 completed / 0 failed.

### Step 7: Code quality cleanup

- Extracted executor-side KV ownership from `executor.rs` into `pegainfer-qwen3-4b/src/kv_cache.rs`.
- Introduced `KvExecViewBatch` as the worker payload so `executor.rs` no longer depends on the internal `(RequestId, KvState)` storage shape.
- Collapsed prefill/decode/unified KV grow, view construction, and commit logic around a shared `KvAppend` model, so the KV cache consumes only `RequestId + append_tokens` instead of executor step items.
- `executor.rs` dropped below the 1k-line review threshold: `1278` lines before cleanup, `956` after cleanup.
- Result: success.

### Step 8: Sub-agent review fixes

- Ran a `toxic-reviewer`-standard sub-agent review over the full local diff.
- Closed the public `KvState::desc_with_seq_len` backdoor; callers now use committed `desc()` or checked `exec_view()`.
- Moved `batch_decode_trace` off direct `KvState` allocation/grow/advance and onto `Qwen3KvCache.prepare_*`.
- Changed `ModelExecutor::admit_requests` to return `Result<Vec<KvAdmission>>` and added release-path length validation so malformed admission batches error pending requests instead of dropping them through `zip`.
- Changed rank-step execution to drain every dispatched worker response before returning errors, so `Qwen3KvCache` does not release pages while a worker may still hold a `KvExecView`.
- Added regression tests for malformed admission batch length and all-physical-pool preflight without partial grow.
- Result: success.

### Verification

- `cargo fmt --check` — passed.
- `git diff --check` — passed.
- `uv run --with triton cargo check --release -p pegainfer-qwen3-4b` — passed.
- `uv run --with triton cargo check --release -p pegainfer-qwen3-4b --features kernel-call-trace` — passed.
- `uv run --with triton cargo test --release -p pegainfer-qwen3-4b --lib` — passed: 9 tests.
- `uv run --with triton cargo test --release -p pegainfer-core --lib` — passed: 12 tests.
- `uv run --with triton cargo test --release --workspace --lib --exclude pegainfer-comm --exclude pegainfer-comm-a2a-kernels --exclude pegainfer-comm-cuda-lib --exclude pegainfer-comm-cuda-sys --exclude pegainfer-comm-cudart-sys --exclude pegainfer-comm-fabric-debug --exclude pegainfer-comm-fabric-lib --exclude pegainfer-comm-libibverbs-sys --exclude pegainfer-comm-logging-lib --exclude pegainfer-comm-p2p-all-to-all --exclude pegainfer-comm-proc-lib --exclude pegainfer-comm-python-ext --exclude pegainfer-comm-thread-lib --exclude pegainfer-comm-torch-lib -- --test-threads=1` — passed across non-comm workspace packages.
- The same non-comm workspace command without `--test-threads=1` hit `pegainfer-server::ops::tests::test_rms_norm_batch_multi_tile` once with a FlashInfer launch error; that exact test passed when rerun alone, so the recorded workspace gate uses serial GPU tests.
- `uv run --with triton cargo test --release --workspace --lib` — blocked in `pegainfer-comm-a2a-kernels` build because this shell cannot find `nvcc`.

### Serve Benchmark

Single RTX 5070 Ti, Qwen3-4B, `--backend openai`, random dataset, `--request-rate inf`, `--max-concurrency 1`, `--ignore-eos`, 20 prompts per shape. `requested out tok/s` is computed as `completed * requested_output_len / duration`; raw benchmark JSON is retained because the reported generated-token counts differ from requested lengths for some streamed responses.

| input | output | ok/fail | duration s | req/s | requested out tok/s | TTFT med/p99 ms | TPOT med/p99 ms | ITL med/p99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 20/0 | 0.27 | 75.07 | 75.07 | 13.05/15.22 | 0.00/0.00 | 0.00/0.01 |
| 1 | 64 | 20/0 | 14.06 | 1.42 | 91.05 | 14.10/30.11 | 10.91/10.92 | 10.91/11.16 |
| 128 | 32 | 20/0 | 7.24 | 2.76 | 88.36 | 18.24/50.72 | 11.01/11.06 | 11.02/11.25 |
| 512 | 64 | 20/0 | 14.97 | 1.34 | 85.48 | 51.62/52.23 | 11.65/11.67 | 11.63/11.89 |
| 1024 | 128 | 20/0 | 29.35 | 0.68 | 87.22 | 99.34/100.06 | 11.34/11.38 | 11.33/11.60 |
| 2048 | 32 | 20/0 | 10.77 | 1.86 | 59.42 | 200.11/201.66 | 11.46/11.48 | 11.44/11.81 |
| 4096 | 64 | 20/0 | 22.56 | 0.89 | 56.74 | 420.27/429.66 | 11.80/11.84 | 11.79/12.07 |

### Serve Benchmark: Random C16

`vllm bench serve`, random dataset, `--request-rate inf`, `--max-concurrency 16`, `--num-prompts 1000`, `--ignore-eos`, `--temperature 0`. Raw artifacts live under `target/profiling/kv-cache-redesign-random-c16/`.

| center input | center output | range ratio | sampled input | sampled output | ok/fail | duration s | req/s | output tok/s | TTFT med/p99 ms | TPOT med/p99 ms | ITL med/p99 ms | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | 128 | 0.5 | 512-1536 | 64-192 | 1000/0 | 10.14 | 98.63 | 95839.36 | 114.16/223.33 | -0.00/0.00 | 13.99/19.99 | Not a clean pass: server logs showed `CUDA_ERROR_OUT_OF_MEMORY` and streamed completions with `finish_reason="error"`; this vLLM bench run still counted them as completed. |
| 1024 | 128 | 0.5 | 512-1536 | 64-192 | 64/0 | 0.75 | 85.33 | 84853.50 | 123.84/428.81 | -0.00/0.00 | 4.55/5.01 | Saved-log reproduction: first failure is activation allocation, `gate_up_out` shape `19456x15291`, after a unified c16 prefill batch with `15290` prompt tokens. |
| 512 | 64 | 0.5 | 256-768 | 32-96 | 1000/0 | 114.05 | 8.77 | 614.46 | 98.72/839.06 | 26.97/32.03 | 14.20/125.35 | Completed without the inflated output-token accounting seen in the larger run. |

The c16 OOM is not KV page exhaustion. The saved server log shows the chain:

```text
unified plan failed: prefill_requests=15, decode_requests=1, prefill_tokens_total=15290, prefill_tokens_min=528, prefill_tokens_max=1507
primary unified worker rank 0 failed
unified scratch allocation failed: total_tokens=15291, total_prefill=15290, decode_tokens=1
alloc prefill gate_up_out failed: dims=19456x15291
Alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

Startup sized KV at `2513` pages, about `5654 MB` or `85%` of the reported free GPU memory. That leaves too little activation/workspace headroom for c16 random prefill batches with roughly 13k-18k total prompt tokens. KV cache admission now exposes a token-level API, but it accounts for KV residency only; forward workspace remains a separate budget.

## Debrief

- **Outcome**: Qwen3 worker threads no longer privately own request KV storage or allocate/release/advance KV. The executor-side KV cache now owns TP physical request state in `kv_cache.rs`, accepts token-level admission inputs from scheduler, prepares all physical KV pools before execution through a shared append model, builds value-type `KvExecView` payloads for workers, and commits logical length after successful worker execution.
- **Pitfalls encountered**:
  - `cargo check` initially failed because the FlashInfer submodule was absent; initializing `pegainfer-kernels/third_party/flashinfer` fixed the header lookup.
  - Build then required Triton for AOT generation; `uv run --with triton ...` provided the expected Python environment without adding project files.
- **Lessons learned**:
  - The useful boundary is ownership plus payload shape: `KvState` owns permits in the KV cache, while `KvExecView` is a disposable worker descriptor.
  - Existing paged metadata can be built from execution-visible lengths without mutating `KvState`; that gives us commit-after-success while preserving current kernel interfaces.
  - Tests should use `Qwen3KvCache.prepare_*` to produce `KvExecView`; hand-constructing `KvState` in model tests reintroduces the old boundary.
  - Keep `Qwen3KvCache` on the token/resource boundary. Passing executor step items into it leaks sampling/logprob concerns into the KV layer.

## Final Architecture

```text
Scheduler
  speaks RequestId + token counts + lifecycle
  calls ModelExecutor::admit_requests with token-level budget inputs
  never reads page IDs, buffer pointers, or TP physical state

Qwen3KvCache
  owns Vec<KvPool> and one RankKvStateStore per TP physical KV pool
  turns token append requests into physical capacity in every KV pool
  builds per-worker KvExecViewBatch values
  commits seq_len only after worker execution succeeds
  releases all physical request state on finish / cancel / error

Worker
  receives KvExecViewBatch as a disposable execution descriptor
  launches prefill/decode/unified kernels
  returns sampled tokens or errors
  never owns KvState and never mutates committed KV length
```

`RankKvStateStore` is deliberately private to `kv_cache.rs`. Its rank dimension is a storage implementation detail for tensor parallelism. The public boundary above it is token-level admission plus `RequestId + append_tokens`; the public boundary below it is `KvExecView`.

## Invariants

- Scheduler is the request/token lifecycle authority. It decides which requests are active, deferred, rejected, finished, or cancelled.
- `Qwen3KvCache` is the KV resource authority. Scheduler does not compute physical KV capacity, and worker does not allocate KV.
- Physical TP storage must be prepared as one transaction: every physical KV pool is checked before any grow is applied.
- Commit is post-success. Failed forward leaves committed KV length unchanged.
- Model tests that need KV descriptors should go through `Qwen3KvCache.prepare_*`, not direct `KvState` construction.
