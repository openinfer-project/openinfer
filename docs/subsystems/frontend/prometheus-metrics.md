# Prometheus /metrics via the vLLM frontend

**TL;DR:** `/metrics` exposes request histograms for every model and engine gauges for schedulers that publish `LoadSnapshot`: Qwen3 and Qwen3.5 use one logical engine, while GLM5.2 EP8/DP8 uses eight rank-local engines and GLM5.2 TP8 uses one logical engine. The bridge forwards each partition's stats under the same identity the vLLM frontend uses for least-load routing. Qwen3 DFlash also feeds live `vllm:spec_decode_*` acceptance counters.

Last touched: 2026-07

## How the numbers flow

Two independent paths feed the upstream Prometheus registry (`vllm-metrics`, served by `vllm-server` at `/metrics` with its HTTP middleware counters):

1. **Per-request path (works for every model crate).** The bridge stamps each request's first output with `Queued`/`Scheduled` timestamps and `PrefillStats` (prompt/computed/cached token split). The upstream `RequestMetricsTracker` turns those into `time_to_first_token_seconds`, `inter_token_latency_seconds`, `request_queue_time_seconds`, `prompt_tokens_total`, `generation_tokens_total`, `request_success_total`, `prompt_tokens_by_source_total`, … unconditionally — `disable_log_stats` only gates the periodic *text* logger, not Prometheus.
2. **Engine-gauge path (needs one `LoadSnapshot` watch per scheduler partition).** The scheduler publishes `LoadSnapshot { kv_used_blocks, kv_total_blocks, num_running_reqs, num_waiting_reqs }` at scheduler boundaries; one bridge identity per partition forwards its snapshot as a stats-only `RequestBatchOutputs`. The enclosing `engine_index` is both the routing identity and the Prometheus `engine` label. Watches coalesce to ≤1 message per scheduler step, and the scheduler's final idle publish settles the gauges back to 0.

For a single-partition model, `EngineHandle::with_load_watch` keeps the original one-engine contract. Qwen3.5 uses that contract for both its single-GPU backend and its TP backend because both execute one logical request stream through one scheduler. A partitioned scheduler uses `with_load_watches`, and the frontend launch declares the same engine count; a mismatch fails startup. GLM5.2 EP8 therefore registers engines 0–7, each bound to its own pending queue and KV pool. TP8 registers only engine 0 because its eight workers mirror one logical request stream.

### Spec-decode counters ride the engine-gauge path (Qwen3 DFlash)

`vllm:spec_decode_num_drafts` / `_num_draft_tokens` / `_num_accepted_tokens` (Prometheus **counters**, incremented by each step's delta) and the per-position `_num_accepted_tokens_per_pos` are wired through the same `LoadSnapshot` watch, but with one twist the queue gauges don't need. The frontend's counters want *per-step deltas*, yet the watch channel coalesces (a reader only sees the latest snapshot). So `LoadSnapshot.spec_decode` carries **cumulative** `SpecDecodeCounters` (monotone since the draft model loaded), and `publish_scheduler_stats` diffs each snapshot against the last one it forwarded — telescoping deltas that survive coalescing without under- or double-counting. This mirrors the delta-safe discipline the per-request prefix-cache path uses (report each contribution exactly once).

The counters live on `Qwen3Executor`, accumulated in `execute_speculative_verify_impl` from each committed step's per-request `matched_draft_tokens` (accepted) and verify-span length (`K` proposed) — only after every request's KV is applied, so a rolled-back verify never inflates them. Kept executor-side, not read back from the worker lane, so the top-of-loop `publish_load` never round-trips the worker thread. The bridge attaches `spec_decoding_stats` only on intervals that actually drafted (delta `num_drafts > 0`), matching vLLM's own scheduler and avoiding NaN acceptance-rate log spam over idle windows.

`num_accepted_tokens_per_pos` is a fixed `[u64; MAX_SPEC_TOKENS]` (16 — an anchor-first `block_size` 16 checkpoint is the widest `K` we ship), and `num_spec_tokens` is how much of it carries meaning. Fixed width is what keeps `LoadSnapshot` `Copy`, so the scheduler's top-of-loop publish never allocates. The bridge emits only `[..num_spec_tokens]`, which keeps the frontend's `position` label set stable across publishes — both consumers on the Rust side (the `vllm:spec_decode_num_accepted_tokens_per_pos_total` family and the interval log accumulator) size themselves to whatever arrives, so a varying length would churn label series rather than fail.

Measured cost is noise in both covered configurations:

- Qwen3 TPOT: 10.6387 ms (main) vs 10.6395 ms (metrics branch) over 828 tokens.
- GLM5.2 EP8, three-run median at concurrency 64: 1268.58 vs 1264.82 output tok/s (-0.30%); TPOT p50 41.76 vs 41.35 ms.

## What deliberately reads zero (state at capture time)

- `prefix_cache_queries/hits` and the by-reason waiting split (`reason="deferred"` is driven by a skipped-request counter we don't report; all waiting shows as `reason="capacity"`).
- Per-GPU FLOPs/bytes estimates, KV-block residency histograms, cudagraph stats — the bridge sends `SchedulerStats::default()` for these fields.
- Spec-decode counters read zero for any model line *without* a DFlash draft model loaded (the executor returns `None`), and for the non-DFlash proposers once they land. Qwen3 DFlash reports them live (see above).
- Every model crate whose scheduler doesn't publish a `LoadSnapshot` watch (currently deepseek and kimi) gets path 1 only; its engine gauges are absent, not lying-zero — the bridge skips the stats task for that partition when no watch exists.

## Validating the spec-decode acceptance rate (#604)

The frontend's PromQL acceptance rate is `rate(vllm:spec_decode_num_accepted_tokens_total[$i]) / rate(vllm:spec_decode_num_draft_tokens_total[$i])` — the same `accepted / drafted` ratio DFlash logs per request as `cumulative_accept_rate` (`RUST_LOG=openinfer_qwen3=debug`, from `dflash_lane.rs`). Note the `_total` suffix: the counters are *registered* as `vllm:spec_decode_num_drafts` etc., and `prometheus-client` appends `_total` at exposition — querying the registered name returns no data. To confirm the two agree end-to-end (needs a GPU + a draft model):

1. Launch Qwen3 with a DFlash draft model and the vLLM frontend configured with a matching `speculative_config` (its `num_speculative_tokens` must be ≤ the drafter's `K`, or the per-position series the frontend publishes will not line up with the drafter's).
2. Drive a steady workload, then read `/metrics`: `num_accepted_tokens / num_draft_tokens` should equal the tail `cumulative_accept_rate` from the debug log, and `1 + num_accepted_tokens/num_drafts` the mean acceptance length the DFlash perf A/B (`tests/dflash_speculative_perf.rs`) implies from its speedup.
3. The unit side is pinned without a GPU: `spec_delta_telescopes_to_cumulative` + `idle_intervals_omit_spec_decoding_stats` (bridge) and `spec_counters_*` (engine) cover the delta arithmetic and idle suppression.

## Validated coverage and next step

Qwen3.5 single-GPU live RTX 5090 validation confirmed that running and KV gauges rise during generation, waiting rises under batch-slot pressure, and all three return to zero after drain and recovery. The commands and metric samples are recorded in [Qwen3.5 Scheduler LoadSnapshot](../../models/qwen35/load-snapshot.md#validation-boundary). TP uses the same scheduler publication path but was not part of that live run.

Next, wire the DeepSeek-V2-Lite and Kimi-K2 schedulers using the same recipe, and report real prefix-cache query/hit counters instead of zeros. A future partitioned model must expose its logical scheduler partitions instead of averaging them behind engine 0. Extend the spec-decode counters to the DSpark/EAGLE proposers as they land, and to the DFlash TP path if speculative decoding ever leaves the single-GPU gate.
