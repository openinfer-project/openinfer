# Qwen3.5 Scheduler LoadSnapshot

> **TL;DR:** Qwen3.5 publishes one logical `LoadSnapshot` stream from its shared single-GPU/TP scheduler: running counts active and prefilling requests, waiting counts deferred requests, and KV usage is request-page capacity minus available pages.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — located the Qwen3.5 model and frontend observability documentation.
  - `docs/models/qwen35/model-crate.md` — confirmed that the model crate owns the scheduler and exposes it through the generic `EngineHandle`.
  - `docs/models/qwen35/roadmap.md` — confirmed the serving and lifecycle observability context.
  - `docs/subsystems/frontend/prometheus-metrics.md` — confirmed the existing `LoadSnapshot` bridge contract.
  - `openinfer-qwen35-4b/src/scheduler.rs` before and after the metrics change — compared the wiring with the shared single-GPU/TP scheduler flow.
- **Relevant history**:
  - Qwen3 established the `LoadSnapshot` watch path consumed by the frontend bridge.
  - Qwen3.5 shares one scheduler loop between single-GPU and TP backends, so KV accounting must come from `SchedulerBackend`, not directly from `Qwen35Model`.
- **Plan**:
  1. Publish backend-neutral snapshots from existing Qwen3.5 scheduler boundaries.
  2. Attach one load watch to the single-GPU and TP engine handles.
  3. Validate field accounting and idle reset with the existing scheduler E2E, HTTP benchmark, and raw `/metrics` sampling.

## Design

The data path reuses the existing frontend contract:

```text
Qwen3.5 SchedulerBackend
  -> LoadSnapshot watch
  -> EngineHandle
  -> LocalEngineBridge
  -> SchedulerStats
  -> /metrics
```

Both Qwen3.5 execution modes own one logical request stream, so single-GPU and TP each attach one `EngineHandle::with_load_watch` receiver. The frontend bridge, metric names, labels, and scheduler-stat conversion remain unchanged.

The scheduler publishes at the top of its existing loop. At that point, work retired by the previous step has been removed and its KV pages have been released, so the next snapshot can settle to idle before `blocking_recv()` waits for new work.

Snapshot accounting is:

| Metric field | Existing Qwen3.5 state |
| --- | --- |
| `num_running_reqs` | `active.len() + prefilling.len()` |
| `num_waiting_reqs` | `deferred.len()` |
| `kv_used_blocks` | request KV capacity minus currently available request pages |
| `kv_total_blocks` | backend request KV capacity, excluding the CUDA Graph padding page |

Publication reads the scheduler's existing queues and KV allocator at the settled boundary.

The live gate uses `scripts/bench_http_serving.py` to create overlapping HTTP traffic and a 100 ms `curl /metrics` sampler to retain the three labeled gauges.

## Execution Log

- Added load watches to `start_with_capacity` and `start_tp_with_capacity` and attached each receiver to its engine handle.
- Added direct, backend-neutral `LoadSnapshot` publication at the top of the shared scheduler loop, following Qwen3's instrumentation shape.
- Derived KV capacity and availability through `SchedulerBackend`, so the same publication logic serves single-GPU and TP.
- Validated the single-GPU path with the existing scheduler E2E and live HTTP pressure: running and KV usage rose during generation, waiting reached three at `--max-batch 1`, and every gauge returned to zero after drain and recovery.
- Updated the shared Prometheus documentation for Qwen3.5's one-logical-engine contract.

## Validation Boundary

The single-GPU NVIDIA run was captured at `a033258c1de1944469d6c6335d4a36d4a80192cf` on an RTX 5090 with driver `580.105.08`, CUDA toolkit `12.8.93`, Rust nightly `1.99.0`, Triton `3.6.0`, and model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Commit `6c9b7d2b8464a846414605fdbde9020887f18ee7` was not rerun; its scheduler diff only inlines the snapshot construction and preserves its fields and publication point.

Build and existing scheduler E2E:

```bash
export OPENINFER_CUDA_SM=120
export OPENINFER_TRITON_PYTHON="$PWD/.venv/bin/python"
export OPENINFER_TEST_MODEL_PATH="$PWD/models/Qwen3.5-4B"

cargo build --release -p openinfer-server --features qwen35-4b
cargo test --release -p openinfer-qwen35-4b --features qwen35-4b \
  --test e2e_scheduler test_e2e_qwen35_scheduler -- --exact --nocapture
```

The release build passed and the existing E2E reported `1 passed; 0 failed`.

Server and 100 ms metric sampler:

```bash
RUST_LOG=info target/release/openinfer \
  --model-path models/Qwen3.5-4B \
  --served-model-name qwen35-metrics \
  --port 18080 --device-ordinal 0 --tp-size 1 \
  --cuda-graph=true --max-batch 1 --max-prefill-tokens 1024

while :; do
  date -Ins
  curl -fsS http://127.0.0.1:18080/metrics \
    | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)\{' \
    | grep -F 'engine="0"' \
    | grep -F 'model_name="qwen35-metrics"'
  sleep 0.1
done > metrics-pressure.log
```

Real batch-slot pressure used the repository's existing benchmark. The provenance arguments below are preserved exactly as recorded in the raw artifact:

```bash
python3 scripts/bench_http_serving.py \
  --base-url http://127.0.0.1:18080 \
  --model qwen35-metrics \
  --num-requests 4 --concurrency 4 --warmup 0 \
  --prompt-words 32 --max-tokens 512 \
  --temperature 0 --top-k 0 --top-p 1 --ignore-eos \
  --timeout 300 \
  --model-path models/Qwen3.5-4B \
  --commit a033258c1de1944469d6c6335d4a36d4a80192cf \
  --source-revision a033258c1de1944469d6c6335d4a36d4a80192cf \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --server-binary target/release/openinfer \
  --claim-boundary "Qwen3.5 LoadSnapshot live metrics pressure and recovery only" \
  --out pressure.json
```

Traffic completed without failures or timeouts:

```text
completed=4
failed=0
timeouts=0
wall_s=12.9674
```

One pressure sample and the observed peaks were:

```text
vllm:num_requests_running{model_name="qwen35-metrics",engine="0"} 1
vllm:num_requests_waiting{model_name="qwen35-metrics",engine="0"} 3
vllm:kv_cache_usage_perc{model_name="qwen35-metrics",engine="0"} 0.00008846686915750052

max_running=1
max_waiting=3
max_kv_cache_usage_perc=0.0010026245171183392
```

After the workload drained, all three gauges returned to zero. A follow-up request then validated recovery:

```bash
curl -fsS http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"qwen35-metrics","prompt":"Hello","max_tokens":8,"temperature":0,"top_k":0,"top_p":1,"ignore_eos":true,"stream":false}'
```

```text
usage={"completion_tokens":8,"prompt_tokens":1,"total_tokens":9}

vllm:num_requests_running{model_name="qwen35-metrics",engine="0"} 0
vllm:num_requests_waiting{model_name="qwen35-metrics",engine="0"} 0
vllm:kv_cache_usage_perc{model_name="qwen35-metrics",engine="0"} 0.0
```

The server exited cleanly and the metric sampler reported no errors.

## Debrief

- **Outcome**: Qwen3.5 now feeds one logical `LoadSnapshot` stream to the frontend for both single-GPU and TP. Single-GPU live validation confirms running, waiting, KV usage, idle reset, and recovery; TP uses the same scheduler path but was not part of this live run.
- **Pitfalls encountered**:
  - The TP scheduler rebase required KV accounting through `SchedulerBackend`; retaining model-specific `model.kv_pool()` access would not compile against the shared loop.
- **Lessons learned**:
  - A shared scheduler loop should expose observability through `SchedulerBackend` so one implementation covers both execution topologies.
  - Top-of-loop publication captures settled state, including the idle zero after KV pages return.
  - The existing HTTP benchmark plus raw metric sampling covers the live gauge contract.
