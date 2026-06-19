# Frontend Telemetry System

> **TL;DR:** Telemetry now has Grafana/Prometheus `/metrics`, configurable metric prefix/buckets, sparse request lifecycle `tracing` spans/events, a lightweight `metrics` facade at the shared engine boundary, optional OpenTelemetry OTLP payload sink, and opt-in structured trace buffers/log lines.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routes telemetry to frontend/runtime/scheduler; no existing telemetry subsystem doc.
  - `docs/subsystems/frontend/cpu-profiling-baseline.md` - frontend latency needs phase timestamps, especially TTFT decomposition; perf alone cannot explain async wall time.
  - `docs/subsystems/scheduler/output-dispatch.md` - token dispatch already carries request-tagged events and prefill stats through the bridge; this is the narrowest place to count request lifecycle metrics.
  - `docs/subsystems/runtime/runtime.md` - shared runtime should stay small; per-model execution details should not leak into the frontend.
  - `docs/roadmap/direction.md` - long-term tracing is desired, but should grow from shared infrastructure, not a universal model abstraction.
  - `docs/subsystems/kernels/openinfer-kernels-boundary.md` - request tracing should eventually line up with simulator/kernel ledger identities; near-term bridge payloads are the natural integration point.
  - `openinfer-vllm-frontend/src/lib.rs` - router extension is already the only clean place to add frontend-owned routes.
  - `openinfer-vllm-frontend/src/bridge.rs` - bridge sees `Scheduled`, `Token`, terminal events, request ids, prompt/completion counts, cache stats, and first-token emit timestamps.
  - `openinfer-vllm-frontend/src/bridge/tests.rs` - demux tests can validate lifecycle accounting without GPU or sockets.
  - `openinfer-engine/src/engine.rs` - shared `TokenEvent` timestamps and counts are already sufficient for request-level metrics; no engine API expansion is needed for phase 1.
  - `scripts/bench_http_serving.py` and `tests/test_bench_http_serving.py` - benchmark tooling already parses `openinfer_http_trace` JSON lines.
  - `openinfer-sim/src/lib.rs` and `openinfer-sim/src/main.rs` - simulator gives a CPU-only verification path through the real frontend.
- **Relevant history**:
  - `docs/subsystems/frontend/cpu-profiling-baseline.md` - explicitly calls for bridge timestamps before attributing the ~145 ms TTFT gap.
  - `docs/subsystems/scheduler/output-dispatch.md` - bridge demux is already the performance-sensitive shared choke point, so telemetry must be O(events) and low allocation.
- **Plan**:
  1. Add a small `openinfer-vllm-frontend::telemetry` module with atomics and manual Prometheus text rendering; avoid global registries on the hot path.
  2. Surface OpenInfer metrics through the existing vLLM `/metrics` route by appending a middleware response, avoiding route conflicts and preserving vLLM metrics.
  3. Thread one telemetry dispatcher into `LocalEngineBridge` and update only request lifecycle points: start, abort, scheduled metadata, first token, terminal finish/error/reject.
  4. Emit structured request logs only when `OPENINFER_TELEMETRY_LOG=1`; keep benchmark-compatible `openinfer_http_trace {json}` only when `OPENINFER_HTTP_TRACE=1`.
  5. Add an opt-in trace buffer enabled by `OPENINFER_TRACE_BUFFER=N`, exposed at `/openinfer/traces`.
  6. Extend focused bridge tests to cover metrics increments and trace-compatible state shape; add frontend route tests for `/metrics` and traces.
  7. Run the smallest useful checks: `cargo fmt --check`, `cargo test --release -p openinfer-vllm-frontend --lib`, and if build time permits `cargo test --release -p openinfer-sim --test frontend_e2e simulated_engine_serves_openai_completions_over_http -- --nocapture`.
- **Risks / open questions**:
  - The bridge can measure queue wait and first-token/request duration, but not true model prefill/decode split without per-model scheduler timing. Ponytail choice: expose honest available fields now; add scheduler hooks later only when a benchmark needs them.
  - Logs and traces can become high-volume at serving load, so they are opt-in. `/metrics` stays always available and uses only a few atomic updates per request.
  - Main lacks the worker-fatal `/health` code from the other worktree, so this task does not depend on it.

## Execution Log

### Step 1: Worktree and route point
- Switched the current checkout to branch `feat/telemetry-system-review` from `main`.
- Added `openinfer-vllm-frontend/src/telemetry.rs` with an in-process telemetry handle and Prometheus text renderer.
- Wired `openinfer-vllm-frontend/src/lib.rs` so every served frontend appends OpenInfer metrics to the existing vLLM `/metrics` response and the local bridge receives the same telemetry handle.

### Step 2: Bridge lifecycle telemetry
- Added request lifecycle state to `openinfer-vllm-frontend/src/bridge.rs`.
- Counts are recorded once per request at terminal/abort, not globally per token.
- Structured request log construction is gated by `OPENINFER_TELEMETRY_LOG`; benchmark-compatible `openinfer_http_trace` logs remain gated by `OPENINFER_HTTP_TRACE`; default serving only pays the in-memory metrics cost.
- Added bridge tests for finished and aborted request accounting.

### Step 3: Logs and trace buffer
- Added `OPENINFER_TELEMETRY_LOG=1` for one structured `openinfer_request_log {json}` line per terminated request.
- Kept `OPENINFER_HTTP_TRACE=1` as a compatibility path for `scripts/bench_http_serving.py`.
- Added `OPENINFER_TRACE_BUFFER=N` to retain the last N request traces in memory and expose them at `GET /openinfer/traces`.
- Trace/log JSON is built only when one of those three trace surfaces is enabled.

### Step 4: Integration fix and verification
- First sim e2e run failed with `Overlapping method route. Handler for GET /metrics already exists`; vLLM already owns that route.
- Replaced the attempted route merge with middleware that runs only on `/metrics`, reads the existing vLLM metrics body, and appends OpenInfer metrics.
- Extended `openinfer-sim/tests/frontend_e2e.rs` to fetch real HTTP `/metrics` after non-streaming and streaming completions and assert OpenInfer counters; it also verifies `/openinfer/traces` is present and disabled by default.
- Checks run:
  - `cargo fmt --check`
  - `cargo test --release -p openinfer-vllm-frontend --lib`
  - `cargo test --release -p openinfer-sim --test frontend_e2e simulated_engine_serves_openai_completions_over_http -- --nocapture`
  - `git diff --check`

### Step 5: Minimal extension point
- Exposed `openinfer_vllm_frontend::telemetry` plus root re-exports for `Telemetry`, `TelemetryOptions`, `RequestMetrics`, and `RequestOutcome`.
- Added `serve_with_telemetry` and `serve_model_with_lora_routes_and_telemetry` so embedding users can pass a configured telemetry handle instead of forking router setup.
- `Telemetry` owns the built-in metrics/log/trace-buffer configuration; enterprise extensions should subscribe to standard `tracing` spans/events rather than implement an OpenInfer-specific callback trait.

### Step 6: Built-in OpenTelemetry and Grafana surfaces
- Treated `/metrics` as the built-in Grafana/Prometheus interface; no extra Grafana-specific server state is needed.
- Added `OpenTelemetryOptions` and `OpenTelemetrySink`, re-exported from `openinfer_vllm_frontend`.
- `OpenTelemetrySink` converts request traces into OTLP JSON and pushes them into a caller-provided bounded channel.
- No HTTP client/exporter is built into the frontend. Embedders can wire the OTLP payload to their preferred exporter/client through `TelemetryOptions.opentelemetry_sink`, or use a normal `tracing_subscriber` OpenTelemetry layer.
- The OTLP sink reserves queue capacity before building the payload, so a saturated caller queue does not keep paying JSON construction cost.
- Re-ran `cargo fmt --check`, `cargo test --release -p openinfer-vllm-frontend --lib`, `cargo test --release -p openinfer-sim --test frontend_e2e simulated_engine_serves_openai_completions_over_http -- --nocapture`, and `git diff --check`.

### Step 7: Drop frontend reqwest dependency
- Removed `reqwest` from `openinfer-vllm-frontend`.
- Kept the OpenTelemetry interface as an OTLP payload sink rather than a transport opinion.
- Updated tests to cover the caller-queue sink instead of HTTP endpoint normalization.
- Re-ran `cargo fmt --check`, `cargo test --release -p openinfer-vllm-frontend --lib`, `cargo test --release -p openinfer-sim --test frontend_e2e simulated_engine_serves_openai_completions_over_http -- --nocapture`, and `git diff --check`.

### Step 8: Reviewability split
- Split `openinfer-vllm-frontend/src/telemetry.rs` into focused submodules:
  - `telemetry/otlp.rs` owns the OTLP payload sink.
  - `telemetry/trace.rs` owns request trace observation, metrics conversion, and trace JSON construction.
- Simplified `bridge.rs`: `TokenEvent` handling now calls `RequestTrace::observe_event(&event)` once before the output match, so terminal metrics are not repeated across `Finished` / `Error` / `Rejected` branches.

### Step 9: Customizable metric surface
- Kept Prometheus/OTLP JSON field strings as protocol surface: changing those per request would make exporters and dashboards harder to reason about.
- Added `TelemetryOptions.metric_prefix` so embedders can namespace metrics without forking the frontend.
- Added `TelemetryOptions.latency_buckets_ms`; buckets are filtered, sorted, and deduplicated at construction time, so request handling still only does atomic histogram updates.
- Existing `TelemetryOptions.opentelemetry_sink` remains only the built-in OTLP helper; custom export/log/trace integrations should attach standard `tracing_subscriber` layers.

### Step 10: Intermediate dispatcher split
- Refactored request state so it carries only a lightweight `RequestTrace`, analogous to a span's local fields.
- `RequestTrace::finish` now returns a typed request record; it no longer receives or calls the telemetry dispatcher.
- `dispatch_burst` records finished request records through the single outer `Telemetry` dispatcher, keeping collection policy outside per-request state.
- An OpenInfer-specific layer trait was considered here, then removed in Step 11 in favor of the real `tracing` crate.

### Step 11: Direct tracing integration
- Added `tracing` to `openinfer-vllm-frontend`.
- Removed the OpenInfer-specific `TelemetryLayer`/`TelemetryEvent` extension trait.
- `RequestTrace` now creates an `openinfer.request` span with request id, token counts, outcome, queue wait, TTFT, and duration fields.
- Request lifecycle emits standard `tracing` events from the shared engine boundary: `openinfer_request_scheduled` and `openinfer_request_finished`.
- The opt-in benchmark/log trace lines now use `tracing::info!`; the existing log bridge can still route them into the configured logger.

### Step 12: Tracing macro and level policy
- Moved repeated span/event field lists into local macros in `openinfer-engine::TokenSink`, the shared event emitter used by all schedulers.
- Level policy:
  - `DEBUG` span/event: successful request lifecycle, useful while debugging.
  - `WARN` terminal event: rejected requests.
  - `ERROR` terminal event: execution failures.
- High-volume success counts and token totals belong to `metrics`; the shared serving path intentionally does not emit per-token tracing events.

### Step 13: Keep reduce_request pure
- Removed the `include_trace` parameter from `reduce_request`; it now only folds token events into wire output plus a terminal outcome.
- `Telemetry::finish_request` owns the decision to build optional JSON traces, keeping telemetry policy out of the bridge reducer signature.

### Step 14: Lightweight non-frontend tracing
- Added `tracing` to `openinfer-engine`.
- `TokenSink` now owns the request span and emits lifecycle events when schedulers send existing `TokenEvent`s.
- This touches the shared event boundary once instead of adding per-model scheduler/executor hooks.
- Frontend `RequestTrace` no longer emits tracing spans/events; it only computes Prometheus metrics and optional JSON traces, avoiding duplicate request lifecycle events on HTTP serving.

### Step 15: metrics facade
- Added the `metrics` crate as a workspace dependency and to `openinfer-engine`.
- `TokenSink` emits low-cardinality facade metrics at the shared event boundary:
  - `openinfer_engine_requests_submitted_total`
  - `openinfer_engine_requests_scheduled_total`
  - `openinfer_engine_requests_finished_total{outcome=...}`
  - `openinfer_engine_prompt_tokens_total`
  - `openinfer_engine_cached_prompt_tokens_total`
  - `openinfer_engine_completion_tokens_total`
  - `openinfer_engine_queue_wait_ms`
- Per-token metrics and tracing events are intentionally omitted; request lifecycle traces stay sparse enough for production debugging.
- No `metrics` exporter was added. Embedders can install their own recorder; the frontend `/metrics` route keeps the existing atomic renderer for now.
- Cargo initially refreshed unrelated `prost`/`itertools` resolution while adding `metrics`; the lockfile was narrowed back so only `metrics`, `rapidhash`, and crate dependency edges remain in this patch.
- Re-ran `cargo test --release -p openinfer-engine --lib`, `cargo test --release -p openinfer-vllm-frontend --lib`, `cargo test --release -p openinfer-sim --test frontend_e2e simulated_engine_serves_openai_completions_over_http -- --nocapture`, `cargo fmt --check`, and `git diff --check`.

### Step 16: sparse tracing levels
- Lowered successful request lifecycle spans/events to `DEBUG`; default `INFO` serving logs should not get one line per successful request.
- Kept rejected requests at `WARN` and execution failures at `ERROR`.
- Removed shared-path per-token tracing events entirely. Token volume is accounted through metrics counters; deep token debugging should use explicit benchmark/debug tooling, not production tracing.
- Re-ran `cargo test --release -p openinfer-engine --lib`, `cargo test --release -p openinfer-vllm-frontend --lib`, `cargo fmt --check`, and `git diff --check`.

## Debrief

- **Outcome**:
  - Added lightweight frontend telemetry on the `feat/telemetry-system-review` branch from `main`.
  - `/metrics` now includes OpenInfer frontend metrics without replacing vLLM's own metrics.
  - The bridge records active/started/finished request counts, outcome counters, prompt/cache/completion token counters, and queue/TTFT/duration histograms.
  - Optional `OPENINFER_TELEMETRY_LOG=1` emits one structured `openinfer_request_log` JSON line per terminated request.
  - Optional `OPENINFER_HTTP_TRACE=1` emits one `openinfer_http_trace` JSON line per terminated request, compatible with the existing benchmark parser.
  - Optional `OPENINFER_TRACE_BUFFER=N` keeps a small in-memory trace ring and exposes it at `/openinfer/traces`.
  - Optional `OpenTelemetrySink` builds OTLP trace payloads and hands them to caller-owned export code.
  - Custom integrations can set `TelemetryOptions.metric_prefix`, `TelemetryOptions.latency_buckets_ms`, and attach normal `tracing_subscriber` layers in their binary.
  - Request tracing uses local macros to keep span/event field names stable and keep the level policy centralized.
  - Successful lifecycle tracing is `DEBUG` only; `WARN/ERROR` are reserved for rejected and failed requests.
  - `reduce_request` stays focused on demux/output reduction; telemetry recording happens in the outer dispatch loop.
  - Shared scheduler/model tracing is centralized at `TokenSink`; no model executor hot path hooks were added.
  - `metrics` facade support is available at the engine boundary without forcing a global recorder or exporter.
- **Pitfalls encountered**:
  - vLLM already registers `/metrics`; adding another route panicked during sim e2e. Middleware append avoids the conflict.
  - Always-on logs/traces would add avoidable serialization, locking, and log volume on hot serving paths, so only counters/histograms are on by default.
  - OpenTelemetry SDK initialization would fight the existing logforth setup; OTLP payload export avoids global subscriber ownership.
  - Frontend should not pick an HTTP client just to offer telemetry integration. The OTLP payload sink and standard tracing subscribers let downstream users pick `reqwest`, `hyper`, tonic, an agent sidecar, or an internal client.
- **Lessons learned**:
  - The frontend bridge already has enough request lifecycle data for useful serving telemetry without widening the shared engine contract.
  - Integration tests should hit real HTTP `/metrics`; route-level unit tests alone missed the vLLM route collision.
  - Standard `tracing` is the right extension point; an OpenInfer-specific layer trait adds API surface without buying anything the ecosystem does not already provide.
  - Metric names and buckets need a small DI surface because downstream dashboards often impose naming and histogram standards.
  - Per-token tracing in the shared serving path is a production log-volume trap; keep token detail out of default tracing and rely on counters plus explicit debug tooling.
  - True prefill/decode phase attribution still needs per-model scheduler timing. That belongs in a later model-side hook, not in this frontend patch.
  - No known required follow-ups for the frontend baseline; scheduler phase metrics can be added later when a concrete benchmark needs them.
