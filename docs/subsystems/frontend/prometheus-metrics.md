# Prometheus `/metrics` via the vLLM frontend

**TL;DR:** `/metrics` exposes request metrics for every model. Schedulers that publish `LoadSnapshot` also expose load and KV gauges; Qwen3 additionally reports real prefix-cache query and hit token counters.

Last touched: 2026-07

## Metric sources

- Request metrics come from frontend request events and include latency histograms, prompt/generated token totals, and request outcomes.
- Scheduler metrics come from `LoadSnapshot`. They are present only for model schedulers that publish a load watch.
- Each logical scheduler partition has its own `engine` label.

Qwen3 publishes monotonic prefix-cache totals. The frontend converts them to interval deltas before passing them to vLLM's Prometheus collector, so coalesced load-watch updates do not lose or double-count increments.

`vllm:prefix_cache_queries_total` counts prompt tokens submitted to an actual first-prefix lookup. `vllm:prefix_cache_hits_total` counts tokens restored from matching full cache blocks. Echo requests, cache-disabled requests, and later chunks of the same prefill do not add queries. Qwen3 cache blocks contain 16 tokens.

## Check prefix-cache counters

Send the same prompt of at least 16 tokens twice, then scrape `/metrics` and inspect `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total`. Both counters are cumulative. Queries should increase for each request; hits should increase when the repeated prompt reuses cached blocks.

## Coverage limits

Unsupported scheduler fields remain at their upstream defaults, including speculative-decoding counters, GPU FLOP/byte estimates, KV-residency histograms, and CUDA Graph statistics. Models without a `LoadSnapshot` watch expose request metrics but no scheduler gauges.
