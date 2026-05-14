# DeepSeek V4 HTTP Serving Benchmark Gate

**Created**: 2026-05-14
**Status**: active
**Canonical task**: task #18; P1 TTFT/sweep extension in `#Dsv4性能调优` task #4

## Purpose

This gate measures the OpenAI-compatible HTTP endpoint under concurrent load. It
does not use the in-process `bench_serving request` path as serving evidence.

The benchmark client sends streaming `/v1/completions` requests and records:

- QPS and completed/failed/timeout counts.
- TTFT from request send to first streamed text chunk.
- ITL and TPOT from streamed text chunks.
- End-to-end latency percentiles.
- Per-request output hashes plus a combined output hash for reproducibility.

The P1 sweep extension adds server-log trace attribution for successful
requests:

- `frontend_to_queue_ms`: client request start to engine queue timestamp. This
  includes HTTP ingress, tokenization, and vLLM request submission.
- `admission_queue_ms`: engine queued to scheduled.
- `prefill_ms`: DSV4 direct prefill plus decode-cache seeding.
- `first_decode_ms`: first decode step after the first streamed token. In the
  current direct path, the first streamed token is sampled from prefill logits,
  so this phase explains early TPOT rather than TTFT.
- `stream_flush_ms`: server first-token emission to client first text chunk.

## Reproducible Commands

Build the server on the target GPU host:

```bash
cd /path/to/pegainfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.1
export PEGAINFER_TILELANG_PYTHON=/path/to/venv/bin/python
export PEGAINFER_TRITON_PYTHON=/path/to/venv/bin/python
export PEGAINFER_NVCC_JOBS=8
export CARGO_TARGET_DIR=/path/to/pegainfer-target

cargo build --release -p pegainfer-server --features deepseek-v4 --bin pegainfer
```

Start the OpenAI-compatible HTTP endpoint:

```bash
$CARGO_TARGET_DIR/release/pegainfer \
  --model-path /data/DeepSeek-V4-Flash \
  --port 18118 2>&1 | tee /tmp/dsv4_http_server.log
```

For prefill phase attribution, start the endpoint with profiling enabled:

```bash
$CARGO_TARGET_DIR/release/pegainfer \
  --model-path /data/DeepSeek-V4-Flash \
  --port 18118 \
  --deepseek-prefill-profile 2>&1 | tee /tmp/dsv4_http_server_profile.log
```

Profiling inserts CUDA synchronization around rank-0 prefill phases. Use it to
rank phases and kernel families; use the non-profile server for serving numbers.

Verify the model endpoint:

```bash
curl -sS http://127.0.0.1:18118/v1/models
```

Run the HTTP serving benchmark:

```bash
python3 scripts/bench_http_serving.py \
  --base-url http://127.0.0.1:18118 \
  --model /data/DeepSeek-V4-Flash \
  --warmup 2 \
  --num-requests 8 \
  --concurrency 2 \
  --prompt-words 16 \
  --max-tokens 16 \
  --timeout 240 \
  --server-log /tmp/dsv4_http_server.log \
  --out /tmp/dsv4_http_bench_task18.json
```

Run the P1 concurrency / max-token sweep:

```bash
python3 scripts/bench_http_sweep.py \
  --base-url http://127.0.0.1:18118 \
  --model /data/DeepSeek-V4-Flash \
  --warmup 2 \
  --num-requests 8 \
  --concurrency 1,2,4,8 \
  --max-tokens 16 \
  --repeats 3 \
  --prompt-words 16 \
  --timeout 240 \
  --server-log /tmp/dsv4_http_server.log \
  --out-dir /tmp/dsv4_http_sweep_task4
```

Run the prompt-length sweep that feeds the prefill optimization target:

```bash
python3 scripts/bench_http_sweep.py \
  --base-url http://127.0.0.1:18118 \
  --model /data/DeepSeek-V4-Flash \
  --warmup 1 \
  --num-requests 1 \
  --concurrency 1 \
  --max-tokens 1 \
  --repeats 1 \
  --prompt-words 16,128,512,2048 \
  --timeout 600 \
  --server-log /tmp/dsv4_http_server_profile.log \
  --out-dir /tmp/dsv4_http_prompt_profile_task4
```

The script is intentionally model-server agnostic at the HTTP layer. It only
requires an OpenAI-compatible `/v1/completions` endpoint that supports streaming
responses.

The server trace columns are pegainfer-specific and require a pegainfer server
log containing `pegainfer_http_trace` lines. The sweep fails when any cell has
request failures/timeouts or per-request output hashes that change across
repeats.

## Current Evidence

Evidence below was collected on the internal 8-GPU DeepSeek-V4-Flash validation
host. It describes only this commit, machine, endpoint, and harness.

| Field | Value |
| --- | --- |
| Commit | PR body records the validated head; tracked docs avoid self-referential commit hashes. |
| Endpoint | OpenAI-compatible `/v1/completions`, streaming |
| Model | `/data/DeepSeek-V4-Flash` |
| Workload | warmup `2`, measured requests `8`, concurrency `2`, prompt words `16`, max tokens `16`, temperature `0`, ignore EOS `true`, timeout `240s` |
| Result | completed `8`, failed `0`, timeout `0`, error rate `0.0` |
| QPS | `1.6869` completed requests/s |
| Latency | avg `1112.19ms`, p50 `1179.70ms`, p95 `1207.23ms`, p99 `1207.23ms` |
| TTFT | avg `680.38ms`, p50 `746.66ms`, p95 `775.06ms`, p99 `775.06ms` |
| TPOT | avg `28.78ms`, p50 `28.81ms`, p95 `28.88ms`, p99 `28.88ms` |
| ITL | avg `28.78ms`, p50 `28.28ms`, p95 `30.57ms`, p99 `30.74ms` |
| Output stability | output chunks `128`, unique output hashes `8`, combined output hash `22706877075acde0` |

### P1 TTFT Trace And Concurrency Sweep

The P1 sweep was collected after the HTTP correctness gate stabilized. It keeps
the same prompt shape and `max_tokens=16`, repeats each concurrency point three
times, and treats per-request hash drift as a correctness failure before any
performance interpretation.

| Concurrency | Correctness | QPS (3 runs) | TTFT avg ms (3 runs) | TPOT avg ms (3 runs) | Trace attribution |
| --- | --- | --- | --- | --- | --- |
| `1` | pass; failed `0`, timeout `0`, combined hash `22706877075acde0` in all runs | `1.609`, `1.613`, `1.617` | `180.7`, `177.6`, `177.7` | `29.37`, `29.49`, `29.37` | admission queue avg `~0.0ms`; prefill avg `~178ms` |
| `2` | pass; failed `0`, timeout `0`, combined hash `22706877075acde0` in all runs | `1.620`, `1.620`, `1.622` | `717.4`, `717.9`, `716.5` | `29.34`, `29.35`, `29.33` | admission queue avg `~540ms`; prefill avg `~177ms` |
| `4` | pass; failed `0`, timeout `0`, combined hash `22706877075acde0` in all runs | `1.618`, `1.610`, `1.620` | `1568.9`, `1574.0`, `1566.5` | `29.36`, `29.58`, `29.35` | admission queue avg `~1392ms`; prefill avg `~177ms` |
| `8` | pass; failed `0`, timeout `0`, combined hash `22706877075acde0` in all runs | `1.618`, `1.619`, `1.620` | `2338.6`, `2342.3`, `2339.4` | `29.38`, `29.36`, `29.38` | admission queue avg `~2162ms`; prefill avg `~177ms` |

Trace attribution separates two different bottlenecks:

- Absolute single-request service time is still high. At concurrency `1`, TTFT
  is already `~178ms` and is dominated by DSV4 direct prefill plus decode-cache
  seeding. Each request then spends roughly `15 * 29ms` on remaining decode
  tokens, so the current endpoint tops out near `1.6` requests/s before any
  queueing analysis.
- Increasing concurrency mostly adds admission queue time because HTTP serving
  currently uses a single-request scheduler turn after the correctness fallback.
  The sweep shows queue time growing from `0ms` at c1 to `~2162ms` at c8, while
  prefill, first-decode, stream flush, and TPOT stay roughly stable.

The evidence therefore points to prefill/decode kernel service time as the next
performance target, with admission queue time as the concurrency amplification
of that baseline. This PR only adds the trace/sweep gate; it is not a throughput
optimization result.

### P1 Prompt-Length And Prefill Profile

Prompt-length sweep, measured with profiling enabled and `max_tokens=1`, shows
TTFT tracking DSV4 direct prefill time:

| Prompt words | Prompt tokens | QPS | TTFT avg ms | Prefill avg ms | failed / timeout |
| --- | ---: | ---: | ---: | ---: | --- |
| `16` | `22` | `5.433` | `183.6` | `181.7` | `0 / 0` |
| `128` | `164-165` | `4.243` | `235.3` | `234.5` | `0 / 0` |
| `512` | `660-661` | `2.637` | `378.9` | `378.0` | `0 / 0` |
| `2048` | `2644-2645` | `0.758` | `1318.2` | `1316.9` | `0 / 0` |
| `8192` | `10580` | `0.111` | `9029.0` | `9014.0` | `0 / 0` |

Rank-0 prefill profile records show `block_prefill` dominates the profiled
prefill window. The table groups block-prefill time by DSV4 compress ratio:

| Prompt tokens | Profiled prefill ms | `block_prefill` ms | ratio `0` ms | ratio `4` ms | ratio `128` ms | Top block-prefill layers |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `22` | `150.4` | `149.6` | `5.1` | `80.9` | `63.6` | ratio-4 layers around `4-5ms` each |
| `165` | `203.2` | `202.3` | `8.7` | `100.2` | `93.4` | ratio-4/128 layers around `5-6ms` each |
| `661` | `343.0` | `342.2` | `15.3` | `200.7` | `126.2` | ratio-4 layers around `10ms` each |
| `2645` | `1266.2` | `1260.3` | `37.9` | `818.6` | `403.9` | ratio-4 layers around `39-40ms` each |
| `10580` | `8956.7` | `8940.6` | `134.8` | `6902.4` | `1903.3` | ratio-4 layers dominate, top layer `~500ms` |

This points the first optimization target at ratio-4 `block_prefill`, especially
the compressed attention / indexer / compressor path. A candidate reuse of the
ratio-4 indexer compressed KV was evaluated and rejected because it changed the
HTTP output hash gate; it is not included in this PR.

### P2 Ratio-4 Block-Prefill Optimization

The first accepted optimization removes repeated generation of the same prefill
window top-k index table from the layer loop. The table depends only on
`seq_len` and `sliding_window`, not on the layer. The direct prefill path now
builds it once per request and reuses it for non-compressed sparse attention and
ratio-4 compressed attention layers.

HTTP correctness remains the gate. The same c1/c2/c4/c8 sweep still passes with
`failed=0`, `timeout=0`, and combined hash `22706877075acde0` across all three
repeats:

| Concurrency | QPS (3 runs) | TTFT avg ms (3 runs) | TPOT avg ms (3 runs) |
| --- | --- | --- | --- |
| `1` | `1.737`, `1.730`, `1.744` | `156.5`, `154.6`, `153.2` | `27.92`, `28.20`, `28.00` |
| `2` | `1.746`, `1.748`, `1.747` | `653.3`, `652.9`, `653.9` | `28.01`, `27.98`, `27.97` |
| `4` | `1.746`, `1.746`, `1.748` | `1441.7`, `1442.6`, `1441.2` | `27.97`, `28.00`, `27.98` |
| `8` | `1.745`, `1.746`, `1.747` | `2162.5`, `2159.2`, `2157.3` | `28.01`, `27.97`, `27.99` |

Rank-0 profile shows the clearest long-prompt improvement in the ratio-4 bucket:

| Prompt tokens | P1 ratio-4 `block_prefill` ms | P2 ratio-4 `block_prefill` ms | Delta |
| ---: | ---: | ---: | ---: |
| `661` | `200.7` | `207.3` | noisy / no win |
| `2645` | `818.6` | `815.4` | `-3.2ms` |
| `10580` | `6902.4` | `6729.9` | `-172.5ms` |

Profile-mode TTFT is still a synchronized diagnostic measurement, not a serving
number. The serving-number evidence remains the non-profile HTTP sweep above.

## Boundary

This PR establishes a benchmark gate and one real HTTP run. It does not claim
vLLM parity, production serving stability, larger batch scalability, paged or
prefix KV, or P/D handoff behavior.

`bench_serving request` remains the in-process direct regression path. It is not
used as a substitute for HTTP serving metrics in this document.
