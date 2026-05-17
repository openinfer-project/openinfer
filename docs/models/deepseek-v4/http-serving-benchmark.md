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

### P3 CuTeDSL Indexer-Score Runtime Observation

After the CuTeDSL exact indexer-score path passed diagnostic equivalence, the
next observation compared the default serial score path against the temporary
CuTeDSL score runtime feature:

- default serial: `--features deepseek-v4`;
- experimental: `--features deepseek-v4-cutedsl-indexer-score`.

Both runs used the same head, model, and workload:

| Field | Value |
| --- | --- |
| Commit | `bf9e7d9` |
| Model | `/data/DeepSeek-V4-Flash` |
| HTTP workload | prompt words `8192` (`10580` tokens), `max_tokens=1`, concurrency `1`, measured request `1`, warmup `0`, timeout `900s` |
| Direct workload | `bench_serving request --prompt-len 10580 --output-len 1 --warmup 0 --iters 1 --seed 42` under `nsys profile` |

Correctness stayed on the same hash gate:

| Gate | Default serial | Experimental CuTeDSL score |
| --- | --- | --- |
| 10k HTTP output hash | `eea187c414579fd7` | `eea187c414579fd7` |
| 10k direct generated-token hash | `39a863e299d2b187` | `39a863e299d2b187` |
| c1/c2/c4/c8 HTTP sweep | existing main gate | `failed=0`, `timeout=0`, per-request hashes stable across 3 repeats; combined hash `22706877075acde0` |

The first HTTP profile request after server start is kept separate from warm
repeats:

| Mode | Run | TTFT ms | Prefill ms | `block_prefill` ms | ratio-4 `block_prefill` ms |
| --- | --- | ---: | ---: | ---: | ---: |
| serial | first | `9489.99` | `9471.88` | `9006.45` | `6730.78` |
| CuTeDSL score | first | `9562.18` | `9544.12` | `9082.79` | `6813.58` |
| serial | warm repeat 1 | `9505.90` | `9488.77` | `9035.94` | `6762.69` |
| serial | warm repeat 2 | `8761.66` | `8757.71` | `8733.20` | `6740.94` |
| serial | warm repeat 3 | `8753.81` | `8750.20` | `8729.85` | `6739.70` |
| CuTeDSL score | warm repeat 1 | `8379.36` | `8361.01` | `7953.76` | `5677.60` |
| CuTeDSL score | warm repeat 2 | `7872.12` | `7868.20` | `7843.73` | `5676.17` |
| CuTeDSL score | warm repeat 3 | `7695.51` | `7691.90` | `7672.28` | `5679.06` |

The stable warm repeats show the experimental path reducing the ratio-4
profile bucket by roughly `1.06s` on this workload. The bucket is still coarse:
it covers the whole ratio-4 block, not only indexer-score execution.

`nsys` kernel summaries explain which part moved. Times below are summed kernel
durations across captured GPU launches, so they are attribution data rather
than end-to-end serving latency.

| Kernel family | Serial total ms / calls | CuTeDSL total ms / calls | Observation |
| --- | ---: | ---: | --- |
| overlap compressor | `20785.98 / 504` | `20789.04 / 504` | unchanged |
| indexer top-k | `12971.09 / 168` | `12975.70 / 168` | unchanged; now the largest remaining indexer-side bucket |
| indexer score | `10394.65 / 168` | `1815.79 / 168` | CuTeDSL exact score removes most score-kernel time |
| NCCL all-reduce | `7698.93 / 864` | `7590.09 / 864` | unchanged within noise |
| non-overlap compressor | `7145.96 / 160` | `7150.87 / 160` | unchanged |
| indexed attention | `731.49 / 344` | `731.38 / 344` | unchanged |

Direct `bench_serving` under `nsys` moved from `9126.05ms` TTFT on the default
serial path to `8036.03ms` with the experimental feature. This is consistent
with the HTTP warm-repeat direction, but it is an observation for this explicit
feature only. It does not change the default `deepseek-v4` feature, and it is
not a production throughput conclusion.

The next likely target is no longer the indexer-score kernel itself. After this
feature, the dominant captured prefill families are overlap compressor and
indexer top-k, while indexed attention is comparatively small in this 10k
profile.

Task #28 later folds the exact CuTeDSL score path into the default
`deepseek-v4` feature and removes the public experimental feature after the
default-path hash/profile gates stay clean.

### P4 Indexer Top-K Attribution And Equivalence Gate

The next indexer-side gate focuses only on the prefill top-k step under the
CuTeDSL score runtime. It does not touch overlap compressor, cache reuse,
scheduler, or HTTP behavior.

The current code path is:

- `indexer_topk_indices_prefill` calls `deepseek_indexer_topk_prefill_cuda`;
- the launch is one CUDA block per token:
  `<<<seq_len, 256, compressed_len * sizeof(float)>>>`;
- each block copies that token's score row into shared memory, masks invalid
  compressed positions with `valid = (token + 1) / 4`, and thread 0 performs
  the current serial top-k selection;
- ties are kept in candidate order because the selector only replaces on
  strict `score > best_score`;
- output indices are `offset + compressed_idx`, where the ratio-4 prefill path
  uses the raw window length as `offset`.

For the 10k prefill workload from the P3 observation, this gives the concrete
shape:

| Field | Value |
| --- | --- |
| `seq_len` | `10580` |
| `compressed_len` | `2645` |
| `topk` | `32` |
| `ratio` | `4` |
| `offset` | `10580` |
| shared memory per block | `10580` bytes |

P3 `nsys` attribution, still under the explicit CuTeDSL score feature, showed
top-k as the largest remaining indexer-side bucket:

| Kernel family | Total ms / calls | Average ms / call | Note |
| --- | ---: | ---: | --- |
| indexer top-k | `12975.70 / 168` | `77.236` | unchanged after indexer-score replacement |
| indexer score | `1815.79 / 168` | `10.808` | no longer the indexer-side bottleneck |
| indexed attention | `731.38 / 344` | `2.126` | comparatively small |

The equivalence gate is a GPU test against the current selector semantics:

```bash
cargo test --release -p pegainfer-kernels \
  --features deepseek-v4 \
  --test deepseek_indexer_topk -- --ignored --nocapture
```

It covers two cases:

- odd compressed length with pseudo-random scores:
  `seq_len=257`, `compressed_len=129`, `topk=32`, `ratio=4`, `offset=777`;
- the 10k profile shape:
  `seq_len=10580`, `compressed_len=2645`, `topk=32`, `ratio=4`,
  `offset=10580`.

The first case checks exact output indices against a CPU reference. The second
case uses monotonic scores so the expected top-k sequence is fully derivable
while still exercising the real 10k launch shape, valid-position mask, top-k
width, and offset behavior.

Hash evidence remains the P3 gate because this PR does not change runtime code:
direct 10k generated-token hash `39a863e299d2b187`, 10k HTTP output hash
`eea187c414579fd7`, and c1/c2/c4/c8 repeated HTTP hash `22706877075acde0`.

This gate decides where a future top-k kernel rewrite or launch adjustment
starts. It is not itself a serving throughput claim.

### P5 Indexer Top-K Rewrite And Feature Convergence

The next implementation keeps the exact CuTeDSL indexer score path as the
default `deepseek-v4` prefill score path instead of keeping a long-lived
experimental feature. The diagnostic feature remains test-only, and the public
`deepseek-v4-cutedsl-indexer-score` feature is removed.

The top-k change is scoped to `deepseek_indexer_topk_prefill_cuda`: each token
still launches one block, but candidate scoring is now split across block
threads and reduced per selected route. The selector preserves the existing
strict `>` semantics by keeping the lower candidate index when scores tie.
Score, overlap compressor, cache reuse, scheduler, HTTP behavior, and decode
top-k are unchanged.

The ignored GPU gate now covers three cases:

- odd compressed length: `seq_len=257`, `compressed_len=129`, `topk=32`,
  `ratio=4`, `offset=777`;
- 10k profile shape: `seq_len=10580`, `compressed_len=2645`, `topk=32`,
  `ratio=4`, `offset=10580`;
- tie-heavy rows: `seq_len=17`, `compressed_len=33`, `topk=12`, `ratio=4`,
  `offset=4096`, with equal score groups verifying that candidate order is
  preserved.

Validation at this PR head:

| Gate | Result |
| --- | --- |
| default `deepseek-v4` server/bench build | passed; CuTeDSL score artifacts generated in the default feature |
| top-k ignored GPU gate | 3 passed / 0 failed |
| direct decode hash | `6346f03343d75a65` across 3 measured iterations |
| 10k direct generated-token hash | `39a863e299d2b187` |
| HTTP c1/c2/c4/c8 repeated hash | failed `0`, timeout `0`, per-request hashes stable; combined hash `22706877075acde0` |

The 10k `nsys` observation after the rewrite:

| Kernel family | P3 CuTeDSL score total ms / calls | P5 default total ms / calls | Observation |
| --- | ---: | ---: | --- |
| overlap compressor | `20789.04 / 504` | `20785.88 / 504` | unchanged |
| indexer score | `1815.79 / 168` | `1815.95 / 168` | unchanged; now default path |
| indexer top-k | `12975.70 / 168` | `1196.05 / 168` | top-k bucket drops after the threaded selector rewrite |
| indexed attention | `731.38 / 344` | `731.40 / 344` | unchanged |
| NCCL all-reduce | `7590.09 / 864` | `7379.37 / 856` | same order; capture differs in launch count |

Direct `bench_serving` under `nsys` for the 10k prompt reports TTFT
`6542.36ms` and generated-token hash `39a863e299d2b187`. These are attribution
observations for this workload, not production throughput claims.

### P6 Overlap Compressor Route Unroll

After P5, the remaining largest captured family in the 10k prefill profile is
the ratio-4 overlap compressor. This step keeps the existing overlap compressor
kernel shape and only asks the compiler to unroll its fixed 8-route loops in
`deepseek_compressor_overlap_weighted_kernel`. It does not change cache reuse,
cross-layer reuse, scheduler behavior, HTTP behavior, the normalizer kernel, or
decode compressor paths.

The ignored GPU equivalence gate exercises the prefill overlap compressor core
directly through `deepseek_compressor_overlap_prefill_cuda`. It covers both
ratio-4 call sites used by the prefill runtime:

- indexer overlap compressor shape, represented by `head_dim=128`;
- main overlap compressor shape, represented by `head_dim=512`;
- small sanity shapes, an odd boundary shape, and the 10k launch shape
  `seq_len=10580`.

The tested kernel consumes hidden states, compressor weights, gate weights,
absolute positional embeddings, and the RMS norm vector; it produces the
weighted compressed buffer and normalized compressed BF16 output. Current
ratio-4 prefill callers only support `start_pos=0`; RoPE is applied by the
runtime after this prefill compressor call, and decode tail/overlap state is
handled by the separate decode compressor path. This change therefore does not
alter `start_pos`, RoPE, or tail-state semantics.

Validation at this PR head:

| Gate | Result |
| --- | --- |
| default `deepseek-v4` server/bench build | passed |
| overlap compressor ignored GPU gate | 3 passed / 0 failed |
| direct decode hash | `6346f03343d75a65` across 3 measured iterations |
| 10k direct generated-token hash | `39a863e299d2b187` |
| HTTP c1/c2/c4/c8 repeated hash | failed `0`, timeout `0`, per-request hashes stable; combined hash `22706877075acde0` |

The 10k `nsys` observation:

| Kernel family | P5 default total ms / calls | P6 route-unroll total ms / calls | Observation |
| --- | ---: | ---: | --- |
| overlap compressor | `20785.88 / 504` | `20574.75 / 504` | small, about 1% lower in this capture |
| indexer score | `1815.95 / 168` | `1814.80 / 168` | unchanged |
| indexer top-k | `1196.05 / 168` | `1195.57 / 168` | unchanged |
| indexed attention | `731.40 / 344` | `730.98 / 344` | unchanged |

Direct `bench_serving` under `nsys` for the 10k prompt reports TTFT
`6513.94ms` and generated-token hash `39a863e299d2b187`. This is a small
workload-specific attribution observation, not a generalized serving throughput
or production performance claim.

## Boundary

This PR establishes a benchmark gate and one real HTTP run. It does not claim
vLLM parity, production serving stability, larger batch scalability, paged or
prefix KV, or P/D handoff behavior.

`bench_serving request` remains the in-process direct regression path. It is not
used as a substitute for HTTP serving metrics in this document.
