# Frontend CPU Profiling Baseline (pegainfer-sim)

**Created**: 2026-06-05
**Last touched**: 2026-06
**TL;DR**: CPU-side profiling of the vLLM/OpenAI frontend path using `pegainfer-sim` with fixed TTFT=5ms / TPOT=12ms. At 200 req / concurrency=16 / prompt=128 words / output=64 tokens the frontend adds ~140ms TTFT overhead above the 5ms simulated floor and shows no throughput bottleneck (QPS=18.2, 0 failures). Top hotspots: heap allocation (malloc/realloc ~10%), stream polling (~5%), clock_gettime (~2%), JSON serialization (~1%). No single frontend bottleneck dominates — the overhead is distributed across tokio runtime, IPC bridge, and HTTP framing.

## Reproducible Benchmark

### Prerequisites

```bash
# Build sim binary (requires protoc)
cargo build --release -p pegainfer-sim
```

### Create a tiny local model dir (avoids HF download)

```bash
mkdir -p /tmp/pegainfer-sim-model

cat > /tmp/pegainfer-sim-model/tokenizer.json << 'EOF'
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    { "id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": { "type": "WordLevel", "vocab": { "<unk>": 0, "alpha": 1, "beta": 2 }, "unk_token": "<unk>" }
}
EOF

cat > /tmp/pegainfer-sim-model/tokenizer_config.json << 'EOF'
{ "unk_token": "<unk>", "tokenizer_class": "PreTrainedTokenizerFast" }
EOF

cat > /tmp/pegainfer-sim-model/config.json << 'EOF'
{ "model_type": "pegainfer_sim", "max_position_embeddings": 8192 }
EOF
```

### Start server

```bash
cargo run --release -p pegainfer-sim -- \
  --model-id /tmp/pegainfer-sim-model \
  --port 8732 \
  --base-ttft-ms 5 \
  --tpot-ms 12 \
  --prefill-tokens-per-ms 100 \
  --max-model-len 8192
```

### Run benchmark

```bash
python3 scripts/bench_http_serving.py \
  --base-url http://127.0.0.1:8732 \
  --model /tmp/pegainfer-sim-model \
  --num-requests 200 \
  --concurrency 16 \
  --prompt-words 128 \
  --max-tokens 64 \
  --warmup 4 \
  --out /tmp/sim-bench-result.json
```

### Run with perf profiling

```bash
SIM_PID=$(pgrep -f "target/release/pegainfer-sim")

perf stat -p $SIM_PID \
  -e cycles,instructions,cache-references,cache-misses,branch-misses,task-clock,context-switches,cpu-migrations \
  -- <benchmark-command>

perf record -g -p $SIM_PID -o /tmp/sim-perf.data -- <benchmark-command>
perf report -i /tmp/sim-perf.data --stdio --no-children --percent-limit 1
```

## Results

### Baseline: 100 req, concurrency=8, prompt=64 words, output=32 tokens

| Metric | Value |
|---|---|
| Requests | 100 completed, 0 failed |
| QPS | 18.4 |
| Wall time | 5.4s |
| TTFT avg / p50 / p95 / p99 | 151ms / 153ms / 284ms / 297ms |
| TPOT avg / p50 / p95 / p99 | 51.8ms / 13.5ms / 157ms / 158ms |
| ITL avg / p50 / p95 / p99 | 76.7ms / 13.3ms / 303ms / 305ms |
| Input tok/s | 1180 |
| Output tok/s | 590 |

Simulated TTFT floor = 5ms + 2 tokens / 100 tok/ms ≈ 5ms. Observed TTFT ~150ms, so **frontend overhead is ~145ms** at concurrency=8.

The p50 TPOT is 13ms (matching the 12ms simulated TPOT + ~1ms jitter), but the avg/max are inflated by ~300ms ITL spikes. These spikes appear when a request's first token arrives during a batch wave — the request waits for the next token-emission cycle in the stream. This is an artifact of the IPC bridge batching, not a CPU cost.

### High-concurrency: 200 req, concurrency=16, prompt=128 words, output=64 tokens

| Metric | Value |
|---|---|
| Requests | 200 completed, 0 failed |
| QPS | 18.2 |
| Wall time | 11.0s |
| TTFT avg / p50 / p95 / p99 | 153ms / 155ms / 290ms / 303ms |
| TPOT avg / p50 / p95 / p99 | 126ms / 129ms / 158ms / 159ms |
| ITL avg / p50 / p95 / p99 | 128ms / 14ms / 306ms / 308ms |
| Input tok/s | 2326 |
| Output tok/s | 1163 |
| perf task-clock | 2599ms over 12s wall |
| IPC | 0.25 (737M instructions / 2939M cycles) |
| Cache miss rate | 58% (94M misses / 161M refs) |
| Branch mispredictions | 15.3M |

Frontend overhead at concurrency=16 is similar (~150ms TTFT), indicating the overhead is per-request, not queueing-bound.

## CPU Hotspot Breakdown (perf, self %)

From `perf record -g` during the 200-req run:

| Category | Self % | Function(s) |
|---|---|---|
| **Heap allocation** | ~10% | `malloc` (3.2%), `cfree` (1.3%), `realloc` chains (3.9%) |
| **Stream polling** | ~5% | `futures_util::stream::StreamExt::poll_next_unpin` (4.7%), `Instrumented::poll_next` (2.8%) |
| **Clock / timing** | ~2% | `__vdso_clock_gettime` (1.3%), `Timespec::now` (1.8%) |
| **Tokio runtime** | ~3% | `Context::run` (1.0%), `process_at_time` (1.2%), `Steal::steal_into` (0.7%) |
| **HTTP framing** | ~2% | `hyper::Dispatcher::poll_catch` (1.3%), `http_body_util::MapErr::poll_frame` (0.9%), `hyper::Buffered::poll_flush` (0.5%), `ChunkSize::new` (1.0%) |
| **Serialization** | ~1.5% | `serde_json::format_escaped_str_contents` (1.0%), `rmp_serde::Decoder::any_inner` (0.5%) |
| **Vec growth** | ~2.5% | `RawVecInner::finish_grow` (1.5%), `bytes::shared_to_vec` (1.2%) |
| **IPC bridge** | ~1% | `PushSocket::send` (0.7%), `mpsc::Tx::push` (0.7%) |
| **Tokenizer** | ~1% | `ModelWrapper::id_to_token` (0.6%), `AddedVocabulary::simple_id_to_token` (0.5%) |
| **Simulated engine** | ~1% | `run_simulated_request` (0.9%) |

### Observations

1. **No dominant hotspot.** The top single function (`malloc`) is only 3.2%. The cost is spread across many small contributors typical of async Rust / tokio workloads.

2. **Heap allocation is the largest category** (~10% combined). `RawVecInner::finish_grow` and `bytes::shared_to_vec` suggest growing buffers in the streaming response path. This is a known Rust async pattern — per-chunk Vec/Bytes allocation during SSE framing.

3. **Stream polling overhead** (~5%) comes from the `mpsc::UnboundedReceiver` → vLLM `GenerateOutputStream` chain. Each token event requires multiple `poll_next` calls through `Instrumented` wrappers.

4. **TTFT overhead decomposition**: The simulated TTFT floor is ~5ms, but observed is ~150ms. Given the perf profile, the ~145ms overhead likely breaks down as:
   - Tokenization + request parsing: ~10-20ms
   - IPC bridge (msgpack encode/decode + ZMQ round-trip): ~30-50ms
   - Scheduler queueing + tokio task wake latency: ~50-80ms
   - Stream setup + first token emit: ~10-20ms

5. **IPC is visible but not dominant** (~1% CPU). The `PushSocket::send` + `mpsc` cost is proportional to token count, not request count. The ZMQ IPC bridge adds latency but not throughput bottleneck at this QPS level.

6. **Low IPC (0.25)** indicates heavy memory-bound workload — cache miss rate of 58% confirms this. The tokio runtime with many small heap allocations and pointer-chasing is expected to be cache-unfriendly.

## Open Questions

- The ~300ms ITL spikes in the profile are caused by request scheduling waves under concurrency. Is this inherent to the IPC bridge's batch-at-a-time pattern, or can it be smoothed?
- The TTFT floor of ~5ms vs ~150ms observed — how much of that 145ms is vLLM-frontend bookkeeping (tokenizer, sampling params) vs IPC bridge vs tokio scheduling?
- Would replacing the ZMQ IPC bridge with a direct in-process channel (for the single-engine case) reduce the TTFT overhead meaningfully?

## Next Steps

1. Add instrumentation timestamps inside `LocalEngineBridge::start_request` and `run_request_stream` to decompose the 145ms TTFT overhead.
2. Profile with `perf record -g --call-graph dwarf` for better symbol resolution (many `[unknown]` frames in the current profile).
3. Compare with the real Qwen3-4B engine to see if the frontend overhead is similar or if the sim path has unique costs.
