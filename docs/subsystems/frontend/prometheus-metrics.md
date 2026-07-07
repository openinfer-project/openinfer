# Prometheus /metrics via the vLLM frontend

**TL;DR:** `/metrics` works out of the box for the Qwen3 line — request histograms (TTFT/ITL/queue time, token counters) are derived by the upstream frontend from the `Queued`/`Scheduled` events and `PrefillStats` the bridge already sends; engine gauges (`num_requests_running/waiting`, `kv_cache_usage_perc`) ride the scheduler's `LoadSnapshot` watch, forwarded by the bridge as stats-only batches. Anything not listed below reads 0 by design, not by accident.

Last touched: 2026-07

## How the numbers flow

Two independent paths feed the upstream Prometheus registry (`vllm-metrics`, served by `vllm-server` at `/metrics` with its HTTP middleware counters):

1. **Per-request path (works for every model crate).** The bridge stamps each request's first output with `Queued`/`Scheduled` timestamps and `PrefillStats` (prompt/computed/cached token split). The upstream `RequestMetricsTracker` turns those into `time_to_first_token_seconds`, `inter_token_latency_seconds`, `request_queue_time_seconds`, `prompt_tokens_total`, `generation_tokens_total`, `request_success_total`, `prompt_tokens_by_source_total`, … unconditionally — `disable_log_stats` only gates the periodic *text* logger, not Prometheus.
2. **Engine-gauge path (needs a `LoadSnapshot` watch).** The scheduler publishes `LoadSnapshot { kv_used_blocks, kv_total_blocks, num_running_reqs, num_waiting_reqs }` at the top of every loop iteration; the bridge's `publish_scheduler_stats` task forwards each snapshot as a stats-only `RequestBatchOutputs` whose `SchedulerStats` the frontend records (`num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`). The watch coalesces to ≤1 message per scheduler step, and the scheduler's final idle publish settles the gauges back to 0. Measured cost: TPOT 10.6387 ms (main) vs 10.6395 ms (branch) over 828 tokens — noise.

## What deliberately reads zero (state at capture time)

- `prefix_cache_queries/hits` and the by-reason waiting split (`reason="deferred"` is driven by a skipped-request counter we don't report; all waiting shows as `reason="capacity"`).
- Spec-decode counters, per-GPU FLOPs/bytes estimates, KV-block residency histograms, cudagraph stats — the bridge sends `SchedulerStats::default()` for these fields.
- Every model crate whose scheduler doesn't publish a `LoadSnapshot` watch (qwen35, deepseek, kimi, glm52) gets path 1 only; its engine gauges are absent, not lying-zero — the bridge skips the stats task when `EngineHandle::load_watch()` is `None`.

## Next step

Wire `LoadSnapshot` publishing into the qwen35 scheduler (the other schedulers can follow the same recipe), and report real prefix-cache query/hit counters instead of zeros.
