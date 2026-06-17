# Worker Fatal Containment

> **TL;DR:** Worker-fatal containment landed at the shared engine boundary: typed `ExecutionError` variants describe the cause and expose `recovery()` as a policy property, `EngineHealth` drives `/health` and admission gating, Qwen3 catches worker panics as fatal, preserves recoverable recovery, and rejects post-fatal work explicitly instead of wedging. Follow-up issues are needed for fully typed model errors and retry/reschedule after domain failure.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routed the task to Qwen3 plus scheduler/frontend/engine boundaries.
  - `docs/subsystems/scheduler/scheduler.md` - current scheduler design and the high-concurrency worker-panic wedge note; the containment fault is that worker death is not promoted to engine readiness/fatal state.
  - `docs/models/qwen3/model-crate.md` - Qwen3 owns worker threads, executor, scheduler, and tests; root/frontend should stay on `EngineHandle`.
  - `openinfer-qwen3-4b/src/executor.rs` - `RankWorker` owns `qwen3-tp-rank-*` threads; panic currently closes the worker response/channel and is surfaced as ordinary execution errors.
  - `openinfer-qwen3-4b/src/scheduler.rs` - `execute_plan` errors call `fail_touched_requests`, which sends request errors and clears active state but does not mark engine/worker fatal.
  - `openinfer-engine/src/engine.rs` - `EngineHandle` exposes submission/control/capacity but no worker health or fatal state.
  - `openinfer-vllm-frontend/src/lib.rs` and `openinfer-sim/tests/frontend_e2e.rs` - frontend readiness is currently tied to server reachability/engine load; health tests only require a successful `/health` response.
- **Relevant history**:
  - `docs/subsystems/scheduler/scheduler.md` - records the same containment shape: worker panic causes channel-closed errors and `/health` remains green.
- **Plan**:
  1. Add a deterministic worker-panic test hook under `#[cfg(test)]` in the Qwen3 executor path, so a unit/integration test can force the worker thread to panic without needing the intermittent cudarc shape.
  2. Split execution failures into recovery tiers at the shared engine/runtime boundary instead of inside Qwen3 only. Request-local errors should fail only that request. Step-local recoverable errors should fail only touched requests and preserve unrelated active/deferred long-running work. Execution-domain-fatal errors include worker panic, worker response channel closed, worker command channel closed, or any CUDA-worker state that makes future execution untrustworthy.
  3. Write a failing regression test that starts the real Qwen3 scheduler with a fake/test executor or a controlled Qwen3 worker path, triggers the panic, observes the channel-closed state, and proves future requests currently keep being admitted/fail incorrectly.
  4. Add engine-level execution error and fatal/readiness primitives to `openinfer-engine` that can be reused by every model crate and shared with `EngineHandle` clones.
  5. Teach the Qwen3 scheduler as the first adopter: handle recoverable errors with bounded blast radius and preserve unaffected long-running requests where state remains trustworthy, but handle execution-domain-fatal failures by failing work bound to the dead domain, stopping future admission to that domain, and making subsequent submissions receive a clear fatal error rather than entering the dead worker loop.
  6. Expose the fatal state to frontend readiness. Prefer an OpenInfer-owned `/health` override if the vLLM router extension allows it; otherwise keep the engine-side fatal API and add direct tests now, then route-level health as the next patch.
  7. Verify with targeted release tests: error-classification tests, engine fatal-state tests, Qwen3 worker-panic containment test, and frontend health behavior if route override lands. Use `--release` for Qwen3/CUDA-bound tests per repo convention.
- **Risks / open questions**:
  - If vLLM's `/health` route cannot be overridden cleanly from `openinfer-vllm-frontend`, this patch may prove containment through `EngineHandle`/scheduler tests first and leave HTTP health wiring as a follow-up.
  - Restarting a CUDA worker in-process is out of scope for the first containment patch unless the restart boundary can rebuild a clean CUDA/model/KV execution domain. The first safe target is best-effort request preservation for recoverable errors, and fail-closed only for state-unsafe worker failures.
  - The deterministic panic hook must be test-only and must not add runtime overhead or user-triggerable behavior in release serving.
  - Avoid stringly typed fatal detection if possible; prefer a typed error boundary from worker/executor to scheduler. The type should live in shared engine/runtime code, not as a Qwen3-only convention.
  - Transparent retry/reschedule after an execution-domain fatal is a separate capability, not part of this containment patch. It needs a defined retry boundary: requests with no emitted tokens may be recomputable; streaming requests with emitted tokens need deterministic replay or explicit client-visible restart semantics; cross-domain migration needs KV/page ownership transfer or cold recompute.

## Execution Log

### Step 1: Current error boundary
- Current Qwen3 executor/scheduler path uses `anyhow::Result` for all execution failures:
  - `ModelExecutor::{execute_prefill, execute_decode, execute_unified}` returns `anyhow::Result`.
  - `scheduler::plan::execute_plan` propagates that `Err`.
  - scheduler loops call `fail_touched_requests` for every `Err` and then continue.
- Worker panic surfaces as channel closure:
  - primary response drop maps to `primary worker dropped step response`.
  - command-channel closure maps to `tensor-parallel worker step channel closed`.
  - TP peer response drop maps to `tensor-parallel <op> worker dropped`.
- Design adjustment from review:
  - Use `thiserror` for typed executor errors.
  - Treat worker channel/response closure and TP protocol violations as worker-fatal.
  - Treat worker-returned step `Err` as recoverable for the first patch because the worker thread is still alive; this preserves long-running work where state is still trustworthy.

### Step 2: Typed containment path
- Added shared engine/runtime primitives in `openinfer-engine`:
  - `ExecutionError` typed variants in `openinfer-engine/src/engine/error.rs` plus `ExecutionResult<T>`.
  - `ExecutionRecovery` as a property returned by `ExecutionError::recovery()`; recoverability is not encoded as the error variant itself.
  - `EngineReadiness::{Healthy, Degraded, Unhealthy}`.
  - `EngineHealth`, shared across `EngineHandle` clones.
- Qwen3 is the first adopter:
  - `ModelExecutor::{execute_prefill, execute_decode, execute_unified}` now returns `ExecutionResult`.
  - Worker command/response channel closure maps to `DomainFatal`.
  - Worker-returned execution errors map to `Recoverable`.
  - Recoverable scheduler errors fail touched requests and continue.
  - Domain-fatal scheduler errors mark engine unhealthy, fail active/deferred/loading/prefilling work, and keep the scheduler alive to reject future requests with a clear `TokenEvent::Error`.
- Tests run:
  - `cargo test --release -p openinfer-engine --lib engine_ -- --nocapture` (health clone test passed; one test matched the filter).
  - `cargo test --release -p openinfer-engine --lib execution_error_separates_recoverable_from_domain_fatal -- --nocapture`
  - `cargo test --release -p openinfer-qwen3-4b --lib fatal_worker_error_marks_engine_unhealthy_and_rejects_future_work -- --nocapture`
  - `cargo test --release -p openinfer-qwen3-4b --lib decode_error_drops_request_state_and_scheduler_recovers -- --nocapture`

### Step 3: Frontend readiness surface
- Added `openinfer-vllm-frontend/src/health.rs`:
  - Middleware intercepts `/health`.
  - `Healthy` returns HTTP 200 with `{"status":"ok"}`.
  - `Degraded` returns HTTP 200 with `{"status":"degraded","reason":...}`.
  - `Unhealthy` returns HTTP 503 with `{"status":"unhealthy","reason":...}`.
- `serve_model_on_host_with_router_extension` now stores the loaded engine's shared `EngineHealth` in the middleware state before the bridge starts.
- Tests run:
  - `cargo test --release -p openinfer-vllm-frontend health_guard --lib -- --nocapture`
  - Re-ran the Qwen3 fatal containment test and engine health clone test after wiring the frontend.
  - `cargo test --release -p openinfer-vllm-frontend --lib`
  - `cargo test --release -p openinfer-sim --test frontend_e2e simulated_engine_serves_openai_completions_over_http -- --nocapture`

### Step 4: Broader verification
- `cargo fmt --check` initially reported formatting diffs; ran `cargo fmt`.
- `cargo fmt --check`
- `cargo test --release -p openinfer-engine --lib`
- `cargo test --release -p openinfer-qwen3-4b --lib scheduler -- --nocapture`
- `cargo test --release -p openinfer-qwen3-4b --lib`

### Step 5: Error typing cleanup
- Moved shared execution error definitions out of `engine.rs` into `openinfer-engine/src/engine/error.rs`.
- Replaced policy-shaped variants (`Recoverable` / `DomainFatal`) with cause-shaped variants:
  - `StepFailed`
  - `WorkerCommandChannelClosed`
  - `WorkerResponseDropped`
  - `WorkerPanic`
  - `UnexpectedWorkerResponse`
- `ExecutionError::recovery()` is now the policy property that returns `ExecutionRecovery::{Recoverable, DomainFatal}`.
- Qwen3 scheduler now branches on `e.recovery()` instead of matching a recoverable/fatal variant name.
- Tests/checks run after the cleanup:
  - `cargo test --release -p openinfer-engine --lib`
  - `cargo test --release -p openinfer-qwen3-4b --lib scheduler -- --nocapture`
  - `cargo fmt --check`
  - `git diff --check`

### Step 6: Fail-safe infrastructure pass
- Added an `EngineHandle` admission gate:
  - `submit` checks shared readiness before enqueueing work.
  - unhealthy engines send a request-local `TokenEvent::Error` immediately and do not enqueue into the scheduler.
  - LoRA control calls also reject before entering a dead engine.
- Moved Qwen3 worker step responses from `anyhow::Result` to `ExecutionResult`, so the worker boundary itself is typed.
- Wrapped Qwen3 worker step execution in `catch_unwind`:
  - a real worker panic is converted to `ExecutionError::WorkerPanic`;
  - the worker reports the fatal error once and exits;
  - later dispatch observes `WorkerCommandChannelClosed`.
- Added a no-GPU worker lifecycle test that triggers an actual panic inside a worker thread and verifies it returns `DomainFatal` then exits.
- Tests/checks run after this pass:
  - `cargo test --release -p openinfer-engine --lib`
  - `cargo test --release -p openinfer-qwen3-4b --lib worker_panic_is_reported_as_domain_fatal_then_worker_exits -- --nocapture`
  - `cargo test --release -p openinfer-qwen3-4b --lib scheduler -- --nocapture`
  - `cargo fmt --check`
  - `git diff --check`

### Step 7: Worker hot-path panic audit
- Audited Qwen3 worker step paths for `panic!` / `assert!` / `unwrap` / `expect` that could turn recoverable execution failures into worker death.
- Converted the obvious worker-hot-path GEMM launch points from unchecked wrappers to checked `Result` propagation:
  - Qwen3 prefill, decode DAG, unified forward, and LoRA projection deltas now use checked GEMM calls where the checked kernel API already exists.
  - These failures now flow back through the worker `ExecutionResult` path instead of panicking first.
- Converted state/protocol boundary panics into typed execution errors:
  - zero-token prefill chunks and missing local `RequestKv` now return step errors;
  - worker results for unknown request ids or mismatched result sets return `UnexpectedWorkerResponse`, which is domain-fatal because applying those results would corrupt scheduler state.
- Remaining panic surface:
  - several low-level kernel/shape wrappers still use assertions or unchecked `()` APIs, especially elementwise/attention helpers. Those need a separate typed-error cleanup instead of opportunistic conversion in this containment patch.
- Tests/checks run after this audit:
  - `cargo check --release -p openinfer-qwen3-4b --lib`
  - `cargo test --release -p openinfer-engine --lib`
  - `cargo test --release -p openinfer-qwen3-4b --lib worker_panic_is_reported_as_domain_fatal_then_worker_exits -- --nocapture`
  - `cargo test --release -p openinfer-qwen3-4b --lib scheduler -- --nocapture`
  - `cargo test --release -p openinfer-qwen3-4b --lib`
  - `cargo test --release -p openinfer-vllm-frontend --lib`
  - `cargo fmt --check`
  - `git diff --check`

### Step 8: Complexity review cleanup
- Applied the over-engineering review cuts that preserved behavior:
  - `EngineHealth` is now a one-way fatal latch backed by `OnceLock<String>` instead of a mutable `Mutex<EngineReadiness>`.
  - Removed the unused `Degraded` readiness state and frontend degraded `/health` response.
  - Replaced the frontend `HealthProbe` newtype with `Arc<OnceLock<EngineHealth>>` directly.
  - Removed scheduler-local `fatal_reason`; the scheduler now reads the shared `EngineHealth` fatal latch.
  - Collapsed duplicate execute/resolve error handling branches in both scheduler loops.
- Tests/checks run after this cleanup:
  - `cargo check --release -p openinfer-engine --lib`
  - `cargo check --release -p openinfer-qwen3-4b --lib`
  - `cargo test --release -p openinfer-engine --lib`
  - `cargo test --release -p openinfer-vllm-frontend --lib`
  - `cargo test --release -p openinfer-qwen3-4b --lib scheduler -- --nocapture`
  - `cargo test --release -p openinfer-qwen3-4b --lib`
  - `cargo fmt --check`
  - `git diff --check`

## Follow-up Issues

### Replace state-machine `anyhow::Result` with typed model/runtime errors
- **Problem**: `anyhow::Result` is still present inside Qwen3 executor internals and some model/kernel glue. It is acceptable for tests, CLI/startup wiring, and outer diagnostics, but it makes scheduler/worker state-machine contracts stale quickly because callers cannot match causes without string inspection.
- **Reference shape**:
  - Databend's current `databend_common_exception` uses a workspace-level `ErrorCode` with stable numeric codes, names, display text, backtrace/context frames, and explicit `map_err_to_code` conversion at foreign-error boundaries.
  - Databend's own error-handling RFC calls out that a single flat error layer is not enough for high-level reasoning and proposes layered error types plus explicit context conversion (`change_context`) instead of implicit `From` propagation.
  - OpenInfer should copy the discipline, not the exact shape: stable shared boundary errors for engine/frontend contracts, smaller domain errors inside model/runtime/kernel crates, explicit conversion between layers, and typed metadata for recovery/admission decisions.
- **Scope**:
  - introduce model/runtime error enums for Qwen3 execution internals and convert them into shared `ExecutionError` only at the engine boundary;
  - remove `anyhow::Result` from scheduler plan/resolve paths and worker-step hot-path helpers where recovery policy matters;
  - keep recoverability as an error property (`recovery()`), not as the variant shape;
  - convert kernel launch/shape failures that already return `Result` into typed variants instead of wrapping everything as generic `StepFailed`;
  - attach operation context as typed fields or explicit frames (`op`, `request_id`, `rank`, `domain`, `layer`, `kernel`) instead of concatenating strings early.
- **Acceptance**:
  - no `anyhow::Result` crosses a worker/executor/scheduler boundary where the caller must decide recover/reject/fatal;
  - tests cover at least one typed recoverable model-step error and one typed domain-fatal protocol/state error;
  - remaining `anyhow` uses are intentionally limited to startup, CLI/test glue, or leaf code that is immediately mapped into a typed error;
  - user-facing errors can be rendered from the typed error without losing machine-readable code/category/recovery policy.

### Retry/reschedule work after execution-domain failure
- **Problem**: current containment is fail-closed. It prevents wedging and preserves unrelated work for recoverable errors, but it does not retry or migrate requests bound to a failed execution domain.
- **Scope**:
  - define domain identity for TP groups and future DP lanes;
  - specify which requests are retryable: no emitted tokens can be recomputed, streamed requests need deterministic replay or explicit client-visible restart semantics;
  - define whether KV can be migrated, cold-recomputed, or must be discarded;
  - add scheduler APIs for requeueing retryable work and degrading capacity when only one DP lane dies.
- **Acceptance**:
  - a domain-fatal failure can reschedule eligible work without double-emitting tokens;
  - unretryable work receives a clear terminal error;
  - TP group failure remains fail-closed unless a clean group restart boundary exists;
  - DP-lane failure can degrade capacity while healthy lanes keep serving when the model line supports DP isolation.

## Debrief

- **Outcome**:
  - Added shared `openinfer-engine` primitives for typed execution errors, recovery classification, and engine readiness.
  - Added a frontend-facing admission gate so unhealthy engines do not continue queueing work.
  - Added worker panic capture in the Qwen3 rank worker loop, with typed fatal reporting and worker exit.
  - Qwen3 scheduler now treats recoverable execution errors as request/step-local and continues serving.
  - Qwen3 scheduler now treats worker-domain fatal errors as engine-unhealthy, fails bound work, and keeps the scheduler alive only to reject future submissions with a clear error.
  - Frontend `/health` now reflects `EngineHealth`: healthy 200, degraded 200, unhealthy 503.
- **Pitfalls encountered**:
  - The first plan was too Qwen3-local. The error/readiness concepts belong in shared engine/runtime code; Qwen3 is only the first adopter.
  - `anyhow::Result` was too weak at the scheduler boundary because the caller needed to classify recovery policy. Lower-level code still uses `anyhow` internally, but the execution boundary is now typed.
  - Encoding recoverable/fatal directly as error variants was also too weak: it hid the actual cause. The shared error now uses cause-shaped variants and exposes recoverability through `recovery()`.
  - A domain-fatal scheduler should not exit immediately; staying alive lets it reject future submissions explicitly instead of turning them into generic channel-closed failures.
- **Lessons learned**:
  - Worker panic containment is not transparent recovery. Requests already streaming from a failed domain still need an explicit error unless a future retry/reschedule design defines deterministic replay or KV migration.
  - Shared readiness should be model-agnostic; `/health` should only consume aggregate engine state, not know TP/DP/Qwen3 details.
- **Follow-ups**:
  - File and implement "Replace state-machine `anyhow::Result` with typed model/runtime errors" from the follow-up issue draft above.
  - File and implement "Retry/reschedule work after execution-domain failure" from the follow-up issue draft above.
  - Adopt `ExecutionError`/`EngineHealth` in other model schedulers so containment semantics are consistent beyond Qwen3.
