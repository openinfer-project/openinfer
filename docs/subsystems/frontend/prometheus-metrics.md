# Prometheus /metrics via the vLLM frontend

**TL;DR:** `/metrics` exposes request histograms for every model and engine stats for schedulers that publish `LoadSnapshot`: Qwen3 reports real prefix-cache query/hit counters plus gauges, while GLM5.2 reports rank-local gauges. Monotonic scheduler totals are differenced by the bridge so coalesced watch updates remain delta-safe.

Last touched: 2026-07

## How the numbers flow

Two independent paths feed the upstream Prometheus registry (`vllm-metrics`, served by `vllm-server` at `/metrics` with its HTTP middleware counters):

1. **Per-request path (works for every model crate).** The bridge stamps each request's first output with `Queued`/`Scheduled` timestamps and `PrefillStats` (prompt/computed/cached token split). The upstream `RequestMetricsTracker` turns those into `time_to_first_token_seconds`, `inter_token_latency_seconds`, `request_queue_time_seconds`, `prompt_tokens_total`, `generation_tokens_total`, `request_success_total`, `prompt_tokens_by_source_total`, … unconditionally — `disable_log_stats` only gates the periodic *text* logger, not Prometheus.
2. **Engine-stats path (needs one `LoadSnapshot` watch per scheduler partition).** The scheduler publishes occupancy gauges at scheduler boundaries; Qwen3 also includes monotonic `prefix_cache_queries_total` / `prefix_cache_hits_total` token counts. One bridge identity per partition forwards a stats-only `RequestBatchOutputs`. The bridge differences the prefix totals against its previous observation before filling vLLM's interval-valued `SchedulerStats`, so a watch update that coalesces several steps loses no increments and Prometheus `inc_by` never replays a cumulative total. The enclosing `engine_index` is both the routing identity and the Prometheus `engine` label. The scheduler's final idle publish settles the gauges back to 0.

For a single-partition model, `EngineHandle::with_load_watch` keeps the original one-engine contract. Qwen3.5 uses that contract for both its single-GPU backend and its TP backend because both execute one logical request stream through one scheduler. A partitioned scheduler uses `with_load_watches`, and the frontend launch declares the same engine count; a mismatch fails startup. GLM5.2 EP8 therefore registers engines 0–7, each bound to its own pending queue and KV pool. TP8 registers only engine 0 because its eight workers mirror one logical request stream.

Measured cost is noise in both covered configurations:

- Qwen3 TPOT: 10.6387 ms (main) vs 10.6395 ms (metrics branch) over 828 tokens.
- GLM5.2 EP8, three-run median at concurrency 64: 1268.58 vs 1264.82 output tok/s (-0.30%); TPOT p50 41.76 vs 41.35 ms.

## What deliberately reads zero (state at capture time)

- The by-reason waiting split (`reason="deferred"` is driven by a skipped-request counter we don't report; all waiting shows as `reason="capacity"`).
- Spec-decode counters, per-GPU FLOPs/bytes estimates, KV-block residency histograms, cudagraph stats — the bridge sends `SchedulerStats::default()` for these fields.
- Every model crate whose scheduler doesn't publish a `LoadSnapshot` watch (currently deepseek and kimi) gets path 1 only; its engine gauges are absent, not lying-zero — the bridge skips the stats task for that partition when no watch exists.

## Validated coverage and next step

Wire `LoadSnapshot` publishing into the qwen35 scheduler (the other schedulers can follow the same recipe). A future partitioned model must expose its logical scheduler partitions instead of averaging them behind engine 0.

## Preparation — issue #603

- **Read**:
  - `docs/index.md` — routed the work to the existing frontend metrics record and the Qwen3 prefix-cache record.
  - `docs/subsystems/frontend/prometheus-metrics.md` — the bridge already publishes scheduler gauges from a coalescing `LoadSnapshot` watch, while prefix-cache counters are deliberately zero.
  - `docs/models/qwen3/prefix-cache.md` — Qwen3 matches 16-token full blocks, skips matching for echo requests, and always leaves at least one prompt token uncached.
  - `docs/subsystems/scheduler/scheduler.md` — the scheduler owns request state and publishes load at step boundaries from one GPU thread.
  - `openinfer-engine/src/engine.rs`, `openinfer-qwen3/src/scheduler.rs`, `openinfer-qwen3/src/executor.rs`, and `openinfer-vllm-frontend/src/bridge.rs` — traced the snapshot, first-prefill result, and stats-only bridge paths.
  - vLLM's pinned `rust/src/engine-core-client/src/{protocol/stats.rs,metrics.rs}` and `vllm/v1/{metrics/stats.py,core/kv_cache_manager.py}` — upstream records prompt-token queries and cached-token hits as interval deltas consumed with `inc_by`; skipped cache reads are not counted.
  - [GitHub issue #603](https://github.com/openinfer-project/openinfer/issues/603) and parent #602 — require real Qwen3 query/hit counters, a bridge assertion, and a repeated-prompt live `/metrics` gate on one consumer GPU.
- **Relevant history**:
  - `docs/subsystems/frontend/prometheus-metrics.md` — the stats watch intentionally coalesces scheduler steps, so raw per-step deltas would be lossy.
  - `docs/models/qwen3/prefix-cache.md` — the existing `cached_tokens` first-prefill result is authoritative for local hits.
- **Risks / open questions**:
  - A watch receiver may skip intermediate values; differencing monotonic totals is required to avoid losing per-step deltas.
  - Live validation depends on a compatible CUDA GPU and locally available Qwen3-4B weights.

## Execution Log — issue #603

### Steps 1–3: scheduler counters and bridge transport

- Extended `openinfer_engine::LoadSnapshot` with explicitly cumulative prefix-cache query/hit token totals; unrelated schedulers retain zero through `Default`.
- Tagged actual Qwen3 first-prefill cache reads in the executor, excluding echo, later chunks, and cache-disabled execution.
- Accumulated upstream-compatible `queries = prompt_tokens` and `hits = cached_tokens` in the plain and LoRA-control scheduler loops.
- Differenced the cumulative values in `openinfer-vllm-frontend/src/bridge.rs` before filling `SchedulerStats.prefix_cache_stats`, preserving increments across coalesced watch updates without replaying earlier totals.
- Added focused Qwen3 scheduler and bridge assertions; validation commands are recorded after they run.

### Step 4: release validation

- `cargo test --release -p openinfer-vllm-frontend --lib`: 22 passed.
- `cargo test --release -p openinfer-engine --lib`: 10 passed.
- `cargo test --release -p openinfer-qwen3 --lib`: 73 passed, including `load_snapshot_accumulates_prefix_cache_query_and_hit_tokens`.
- `cargo test --release -p openinfer-sim --test frontend_e2e --no-run`: compiled successfully, covering every updated simulated `LoadSnapshot` producer.
- The host's libclang 22 makes pinned `rdma-mummy-sys` 0.2.4 generate opaque verbs structs; Qwen validation used a build-only `/tmp` source override with bindgen 0.72.1 and pthread pointer compatibility casts. No dependency or lockfile change was retained.

### Step 5: live gate

- Host GPU: NVIDIA GeForce RTX 2080 Ti, compute capability 7.5, 11,264 MiB total / 10,561 MiB free.
- Model search: no repository `models/` directory and no local Qwen `config.json` found under the usual `/home/zxh`, `/data`, `/mnt`, or `/models` roots.
- Result: not launched. This is not a suitable Qwen3 BF16 live-gate environment, and the required model weights are absent.

## Debrief — issue #603

- **Outcome**: Qwen3 now reports real local prefix-cache prompt-token queries and cached-token hits through `/metrics`. The scheduler publishes monotonic totals, while the bridge emits interval deltas compatible with upstream vLLM Prometheus semantics.
- **Pitfalls encountered**:
  - Publishing raw per-step deltas on a watch channel would lose counts when updates coalesce; cumulative producer state plus consumer differencing avoids that loss.
  - The pinned RDMA binding crate is incompatible with this host's libclang 22 generation behavior, independent of the metrics change.
- **Lessons learned**:
  - Prefix-cache query counters use prompt-token units, not block units; hits use cached-token units.
  - Record only actual cache reads: echo requests, cache-disabled requests, and later prefill chunks do not enter the denominator.
- **Follow-ups**:
  - Run the two-identical-prompt `/metrics` scrape on an Ampere-or-newer consumer GPU with local Qwen3-4B weights.
