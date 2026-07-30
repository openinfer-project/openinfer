# Router → OpenInfer E2E Request Tracing

> **TL;DR:** A single trace for client → vllm-router → openinfer → prefill/decode is working and verified on the local Tempo/Grafana stack. The router already emits spans and injects `traceparent` when OTel is on; on the openinfer side, `openinfer-vllm-frontend/src/trace_context.rs` stashes the incoming `traceparent`, correlates it via `X-Request-Id → external_req_id` (tolerating vllm-server's `cmpl-`/`chatcmpl-` prefixes), and the bridge uses it as the parent of the `request` root span. Once upstream PR vllm-project/vllm#50370 (HTTP-layer `trace_headers` population) merges, migrate per openinfer#790 and delete the middleware.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — routing table; no existing tracing subsystem doc (prior tracing work lived only in deploy/tracing + code)
  - `deploy/tracing/docker-compose.yml` / `tempo.yaml` — existing local Tempo+Grafana stack, OTLP gRPC on 4317; point `OPENINFER_TRACE_OTLP_ENDPOINT` at it and go. Spans already emitted: `request → {queue, prefill, decode}`
  - `openinfer-core/src/tracing.rs` — fastrace → OTLP reporter init; zero cost when `OPENINFER_TRACE_OTLP_ENDPOINT` is unset
  - `openinfer-vllm-frontend/src/bridge.rs` — `Span::root("request", SpanContext::random())` (bridge.rs:396): **always opened a fresh trace** — this was the break point; `EngineCoreRequest` arrives from vllm-server over ZMQ
  - `openinfer-qwen3/src/scheduler/phase_trace.rs` — queue/prefill/decode spans are children of `request` (via `GenerateRequest.trace_parent`)
  - `openinfer-vllm-frontend/src/lib.rs` — `vllm_server::serve_with_router_extension(config, shutdown, extend_router)` exposes a `FnOnce(Router) -> Router` hook, so middleware can be added without patching the git dependency
  - vllm-server (verified on both the pinned rev 8e61b64 and upstream main) — `resolve_request_context` extracts only `X-Request-Id`/`X-data-parallel-rank`; the `EngineCoreRequest.trace_headers` protocol field exists but the HTTP layer never populates it; **bumping the pin would not fix this**
  - vllm-router (local checkout /data/code/workspace-rustllm/router) — `--enable-trace` + `--otlp-traces-endpoint`; server span `http_request` (extracts the client's traceparent as parent) + client span `http_client_request`, injecting span context into `traceparent`/`tracestate` on worker-bound requests; with OTel off it still forwards client trace headers verbatim; all client headers (incl. `X-Request-Id`) forwarded as-is; `service.name=vllm-router`; a bare host:port endpoint gets `http://` prepended
  - fastrace 0.7.17 — `SpanContext::decode_w3c_traceparent(&str) -> Option<SpanContext>` (pub, `collector/id.rs:281`); `EngineCoreRequest.external_req_id` works as the correlation key
  - `openinfer-engine/src/tracing_state.rs` — global AtomicBool gate; tests must not flip it (parallel-test races), so the middleware core was written as a pure function that never reads the global flag

- **Relevant history**:
  - `docs/roadmap/roadmap-2026-h2.md` — "observability wiring" is on the H2 plan; this task is part of it
  - `docs/subsystems/router/kv-aware-routing.md` — earlier Dynamo-router multi-turn routing experiment (different router, but the e2e measurement approach carried over)
  - No past tracing task docs found

- **Plan**:
  1. Add W3C trace-context intake on the openinfer side (the only code change in this repo):
     - New module `openinfer-vllm-frontend/src/trace_context.rs`: axum middleware reads `traceparent`; if present, take `X-Request-Id` (generate and inject a short id when absent — vllm-server resolves it into `external_req_id`) and stash into a bounded shared map (`external_req_id → traceparent`, TTL/capacity eviction)
     - Mount the layer for all serving variants inside `serve_model_on_host_with_router_extension`
     - `bridge.rs` add_request: when tracing is on, pop the map by `external_req_id`; on successful `decode_w3c_traceparent`, use it as the parent of the `request` root span; otherwise fall back to `SpanContext::random()` (current behavior)
     - Unit tests: middleware injection/no-header paths + bridge parent resolution (TestReporter pattern from phase_trace.rs)
  2. Bring up the local stack: `docker compose -f deploy/tracing/docker-compose.yml up -d` (Tempo 4317 / Grafana 3000)
  3. Start openinfer: `OPENINFER_TRACE_OTLP_ENDPOINT=http://127.0.0.1:4317 cargo run --release -- --model-path models/Qwen3-4B --port 8000` (confirm weights and GPU first)
  4. Build and start the router (/data/code/workspace-rustllm/router): `cargo build --release`, then `vllm-router --worker-urls http://127.0.0.1:8000 --port 8090 --enable-trace --otlp-traces-endpoint <4317>` (verify flag format at execution)
  5. Verify one chain: send `/v1/completions` through the router; assert via Tempo HTTP API (`:3200/api/search` + `/api/traces/<id>`) that **one trace** contains router `http_request` → `http_client_request` → openinfer `request` → `queue`/`prefill`/`decode` with correct parenting
  6. Observe e2e behavior: small concurrent load (vllm-bench or multi-turn curl) — router forwarding overhead, queue wait, prefill/decode breakdown; also verify the client-supplied-traceparent case (the whole chain should hang under the client's span)
  7. Wrap up: update deploy/tracing comments and docs/index.md, write the Debrief

- **Decision (2026-07-30 review)**: dual track — land the local MVP (middleware workaround) to validate the chain now, while opening an upstream PR against vllm-project/vllm (Rust server extracts traceparent into `trace_headers`, Python parity); track the "middleware → upstream" migration with an issue in this repo. After upstream merges and the pin is bumped, delete the middleware and read `EngineCoreRequest.trace_headers` in the bridge (the bridge change is shared by both tracks, so nothing is wasted).

- **Risks / open questions**:
  - Correlation relies on `X-Request-Id → external_req_id`: the middleware must inject the id before vllm-server reads headers (axum layer order) — verify empirically
  - Router `--otlp-traces-endpoint` format (host:port vs URL) and its `service.name` — check at execution
  - Single-machine single-worker means the router policy is irrelevant, but router retries/circuit-breaking may add extra spans under load — watch for them
  - `models/Qwen3-4B` weights and GPU availability unconfirmed

## Execution Log

### Step 1: W3C trace-context intake on the openinfer side (MVP)
- Added `openinfer-vllm-frontend/src/trace_context.rs`: `TraceContextStash` (Arc<Mutex<HashMap>>, TTL 120s / cap 4096, one-shot `take`) + `stash_trace_context` axum middleware; the core `stash_from_headers` avoids the global flag so unit tests stay race-free
- `lib.rs`: `mod trace_context`; the stash is created at the top of `serve_model_on_host_with_router_extension`, cloned into the engine task (bridge field) and into the extend_router wrapper (`from_fn_with_state` layer as the outermost wrapper, guaranteeing it runs before vllm-server reads headers)
- `bridge.rs`: `LocalEngineBridge` gains a `trace_stash` field; `start_request` destructures `external_req_id`; with tracing on, `take(id) → decode_w3c_traceparent → Span::root parent`, falling back to `SpanContext::random()` on miss/invalid (previous behavior)
- 3 unit tests (roundtrip + W3C decode, id injection, no-header no-op); `cargo test --release -p openinfer-vllm-frontend --lib` 30/30; clippy clean (fixed one single-pattern match → if let)
- Result: success

### Step 2: Environment checks
- docker OK; Tempo/Grafana up via `deploy/tracing/docker-compose.yml up -d` (4317/3000)
- GPU: RTX 5070 Ti 16GB; model: `/data/models/Qwen3-4B` (no models/ dir in the repo on this machine)
- router and openinfer release builds running in parallel in the background

### Step 3: Stack bring-up + first verification (fail → fix → pass)
- GPU was fully held by an old openinfer from pegainfer-2 (PID 725807, 13.5GB) → killed after user confirmation; server starts fine (`OPENINFER_TRACE_OTLP_ENDPOINT=http://127.0.0.1:4317`, :8000)
- Router on :8090: `vllm-router --worker-urls http://127.0.0.1:8000 --policy round_robin --enable-trace --otlp-traces-endpoint 127.0.0.1:4317` (built with zero changes, system libzmq present)
- First verification: the router's two spans chained correctly, but openinfer opened a separate random trace — **join failed**
- Diagnosis: direct-to-openinfer with `traceparent` + `X-Request-Id: dbg12345` still failed; the openinfer trace showed `request_id=cmpl-dbg12345-2b7dfbb4` → vllm-server prepends **`cmpl-`/`chatcmpl-`** to X-Request-Id before it becomes `external_req_id` (llm/request.rs: prepare() reuses the route-prefixed id), so the stash key (bare header value) never matched
- Fix: `TraceContextStash::take_for_external_req_id` — exact lookup first, then prefix-stripped; regression test `lookup_tolerates_vllm_api_prefixes`; 31/31 pass, clippy clean
- Re-verify (trace id `aaaa1111…`): **full chain in a single trace** — client(5555eeee) → router `http_request`(334.79ms) → router `http_client_request` → openinfer `request`(319.89ms) → `queue`(0.04ms) → `prefill`(72.06ms) → `decode`(247.67ms), parenting correct at every level
- Result: success

### Unexpected
- The router's `http_client_request` span once showed a ~5.3s duration (vs the 334ms server span) — looks like a span-lifetime bug on its non-streaming path; a router-side instrumentation issue, doesn't affect chain verification; worth reporting upstream
- Tempo `/api/traces` returns span ids base64-encoded, and spans from multiple requests sharing one trace id get compacted together — use a fresh trace id per debug iteration

### Step 4: Load + both endpoints (pass)
- 16 req / conc=8 / distinct prompts through the router: client p50 756ms (includes Python urllib per-request connection setup); Tempo shows 16/16 complete chains
  - Phase p50: queue 0.02ms / prefill 25.1ms (max 101.7, batching) / decode 359.9ms (32 tok, bs≈8, ~11ms/tok @5070 Ti) / request 466.3ms
  - Sampled single trace: router server span 388.50ms vs openinfer request 386.14ms, start offset 1.02ms → **router overhead is ~1ms**; another ~6.8ms sits in vllm-server HTTP/tokenize before the bridge opens its span (no spans there — shows as a gap)
- `/v1/chat/completions` (chatcmpl- prefix) verified the same way: client span → router's two spans → openinfer's four spans, parenting correct throughout
- No-client-traceparent case: the router's `http_request` becomes the trace root and openinfer attaches normally (the load test exercised exactly this)
- Result: success

### Step 4.5: Tempo query caveats
- `/api/traces/<id>` may return only the spans that have reached a block (ingester flushes on a ~5s window) — refetch and you get everything; not span loss

### Step 5: Upstream PR + tracking issue
- Upstream PR: https://github.com/vllm-project/vllm/pull/50370 "[Rust Frontend] Propagate W3C trace headers to engine-core requests" (xiaguan fork, DCO signed, off upstream/main e5f48dfda)
  - Prior-art check found **#44567** (same goal) closed unmerged after a maintainer objected to its "new protocol surface"; this PR deliberately takes the minimal route: no handshake, no gating, pure extraction into `trace_headers`, stated explicitly in the PR body
  - 10 files changed: `resolve_request_context` extracts traceparent/tracestate → `ResolvedRequestContext.trace_headers` → completions/chat/generate converts → additive pass-through fields in vllm-text/vllm-chat into the llm `GenerateRequest` (llm/engine-core-client untouched); grpc/tokenize only got compile-required `None` fill-ins
  - Gates: fmt/clippy (-D warnings) clean; nextest vllm-text+chat 321 passed, vllm-server 328 passed (incl. 12 new tests)
- Tracking issue: https://github.com/openinfer-project/openinfer/issues/790 — records the MVP state, the upstream PR, and the three migration steps (bump pin → bridge reads `trace_headers["traceparent"]` → delete trace_context.rs and its lib.rs wiring)
- Grafana access: the host's port 3000 collided with a Grafana on the user's mac, so the container was rebound to 4000 (repo compose file untouched); the user's browser initially landed on a different Grafana 12 instance (Sign in/Bookmarks visible) — `curl localhost:4000/api/health` returning 11.3.0 confirmed the right port-forward
- Anonymous Viewer is denied `datasources:explore` on Grafana 11.3 (Access denied in the server log; Explore renders but returns nothing) → recreated the container with `GF_AUTH_ANONYMOUS_ORG_ROLE=Editor`; the repo compose file is fixed in the same PR (fix(deploy))
- PR follow-up: only bot placeholder reviews, no maintainer response; per the user's judgement (maintainers may want to build it themselves), posted a direction/timeline question on the PR (#issuecomment-5126068994) offering to reshape or close in favor of upstream's own implementation

### Step 6: MVP packaged as a pegainfer PR
- Branch `feat/frontend-trace-context`, 3 Commitizen commits (feat(frontend) / fix(deploy) / docs(tracing)); the prek fmt hook reformatted on the first attempt, re-staged and passed
- Pushed to origin (GitHub); PR: https://github.com/openinfer-project/openinfer/pull/791; #790 commented with the link
- The Viewer→Editor compose fix rides along in fix(deploy) (anonymous Viewers lack `datasources:explore` on Grafana 11.3, contradicting the file's own usage comment)
- Review follow-ups: Codex bot flagged a real P2 (`take` ignored entry timestamps — a reused X-Request-Id could join a stale trace) → fixed in `d513dde` with a TTL check on take + regression test; CI's DCO sign-off check failed (repo convention — main's commits all carry Signed-off-by) → all branch commits signed via `git rebase --signoff` and force-pushed

## Debrief

- **Outcome**: the client → vllm-router → openinfer → prefill/decode chain renders as one trace, verified in Tempo (both endpoints, with and without a client traceparent, and under 16-way concurrency); router overhead is ~1ms. The local MVP (middleware + stash + bridge parent resolution) is implemented, tested, and up as PR #791; upstream PR vllm-project/vllm#50370 is open; migration is tracked by openinfer#790.
- **Pitfalls encountered**:
  - Root cause of the first join failure: vllm-server prepends `cmpl-`/`chatcmpl-` to `X-Request-Id` before it becomes `external_req_id` — this protocol fact could only be pinned down from live trace attributes (`request_id=cmpl-dbg12345-…`); the route-layer prefix step was missed when reading code
  - The direct-to-openinfer control experiment (bypassing the router) was the key bisection: it first narrowed the fault to the openinfer side, then span attributes pinned the prefix issue
  - A same-goal upstream PR (#44567) was rejected for adding protocol surface/gating — always check rejection history before proposing upstream; minimal pass-through is the only acceptable shape
  - Tempo `/api/traces` partial returns (ingester flush window) and same-trace-id block compaction each cost a debugging round
- **Lessons learned**:
  - Verify trace chains with deterministic traceparents (carry your own trace id) + Tempo HTTP API assertions — faster and reproducible, unlike UI spelunking
  - A missing feature in a git dependency doesn't force a fork: framework hooks like `serve_with_router_extension` + middleware can close the gap; but a workaround needs a tracking issue and an upstream PR, or it rots in the tree
- **Follow-ups**:
  - openinfer#790: after the upstream PR merges and the pin is bumped, delete the middleware
  - vllm-router `http_client_request` span occasionally ~5s inflated (non-streaming path) — suspected router instrumentation bug; candidate for a separate issue to vllm-project/router
  - vllm-server's own OTel instrumentation (the HTTP/tokenize segment is currently a blank gap in the trace) — upstream has related PRs in flight (#39438, #39905); track them
