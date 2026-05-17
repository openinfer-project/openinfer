# DeepSeek V4 Serving Baseline

**Created**: 2026-05-13
**Status**: active
**Canonical task**: task #14

## TL;DR

DeepSeek V4 currently supports a single-request OpenAI-compatible HTTP smoke path and an in-process direct TPOT/hash regression gate. It does not yet support bs>1 serving, continuous batching, or a service-level KV cache manager.

Use this document as the baseline contract before changing the DeepSeek V4 scheduler:

- Keep HTTP single-request smoke passing.
- Keep direct `bench_serving request` TPOT/hash reproducible.
- Keep concurrency claims explicit: current HTTP requests enter a serial DeepSeek direct scheduler.
- Do not describe this state as vLLM `bench serve` readiness.

## Current Capability Contract

| Capability | Status | Evidence |
| --- | --- | --- |
| DeepSeek V4 engine load behind the OpenAI HTTP facade | Available for smoke testing | `pegainfer-server --features deepseek-v4 --bin pegainfer` starts an OpenAI server for `/data/DeepSeek-V4-Flash` on 8x RTX 5090 |
| `/v1/models` | Available | The returned model id is the full model path: `/data/DeepSeek-V4-Flash` |
| `/v1/completions` single-request greedy smoke | Available | Prompt `hello`, `max_tokens=4`, `temperature=0` returned a text completion and usage accounting |
| Direct single-request TPOT/hash regression | Available | `bench_serving request --prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42` is the retained DeepSeek V4 decode gate |
| HTTP multi-request serving benchmark | Not available as a correctness/perf claim | The HTTP facade can receive requests, but the DeepSeek V4 engine behind it is a single scheduler thread that handles one complete request at a time |
| bs>1 decode serving | Not available | The current direct decode path drives one request and one token per decode step |
| Service-level KV cache manager | Not available | KV cache is request-local direct decode state that is reset before each request |

## Reproducible Commands

Run these commands from any checkout at or after PR #101's merge commit `d6d2cee`. Keep build artifacts outside the repository checkout.

Build the HTTP server on the 5090 host:

```bash
cd /path/to/pegainfer
export PATH=/usr/local/cuda-13.1/bin:$PWD/.venv/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.1
export PEGAINFER_TILELANG_PYTHON=$PWD/.venv/bin/python
export PEGAINFER_TRITON_PYTHON=$PWD/.venv/bin/python
export PEGAINFER_NVCC_JOBS=8
export CARGO_TARGET_DIR=/path/to/pegainfer-target

cargo build --release -p pegainfer-server --features deepseek-v4 --bin pegainfer
```

Start the HTTP endpoint:

```bash
$CARGO_TARGET_DIR/release/pegainfer \
  --model-path /data/DeepSeek-V4-Flash \
  --port 18103
```

Verify model registration:

```bash
curl -sS http://127.0.0.1:18103/v1/models
```

Expected shape:

```json
{"object":"list","data":[{"id":"/data/DeepSeek-V4-Flash","object":"model","created":0,"owned_by":"vllm-frontend-rs"}]}
```

Verify single-request completion:

```bash
cat >/tmp/dsv4_completion.json <<'JSON'
{"model":"/data/DeepSeek-V4-Flash","prompt":"hello","max_tokens":4,"temperature":0}
JSON

curl -sS http://127.0.0.1:18103/v1/completions \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/dsv4_completion.json
```

Observed smoke result:

```json
{
  "id": "cmpl-54f70147",
  "object": "text_completion",
  "model": "/data/DeepSeek-V4-Flash",
  "choices": [
    {
      "index": 0,
      "text": ", world!');\n",
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "total_tokens": 5,
    "completion_tokens": 4
  }
}
```

Run the direct single-request decode regression gate:

```bash
cargo run --release -p pegainfer-server \
  --bin bench_serving \
  --features deepseek-v4 \
  -- \
  --model-path /data/DeepSeek-V4-Flash \
  --format json \
  --out /tmp/dsv4_direct_request.json \
  request \
  --prompt-len 1 \
  --output-len 160 \
  --warmup 2 \
  --iters 3 \
  --seed 42
```

Current retained PR #101 reviewer sweep:

| Run | Aggregate steady TPOT avg | Generated-token hash |
| --- | ---: | --- |
| 1 | `28.505793ms` | `6346f03343d75a65` |
| 2 | `28.087102ms` | `6346f03343d75a65` |
| 3 | `29.755957ms` | `6346f03343d75a65` |
| 4 | `27.552965ms` | `6346f03343d75a65` |
| 5 | `29.371630ms` | `6346f03343d75a65` |

## Concurrency Boundary

The HTTP layer and DeepSeek V4 engine have different concurrency semantics today:

- The OpenAI-compatible HTTP facade can accept request messages and stream outputs through the local vLLM engine bridge.
- The DeepSeek V4 direct engine has one scheduler thread.
- That scheduler receives `GenerateRequest` values from an unbounded channel and calls `handle_request` synchronously.
- `handle_request` runs the whole request through prefill and greedy decode before the scheduler receives the next request.

Therefore, HTTP smoke proves endpoint integration, tokenizer/model registration, and single-request generation. It does not prove concurrent serving, batching, admission, or fairness.

## KV Cache Boundary

Current DeepSeek V4 KV state is direct-runtime request state:

- `ensure_direct_decode_caches` ensures per-rank cache capacity for `prompt_len + max_new_tokens`.
- `reset_caches` clears those caches before the request.
- Each layer owns a `LayerDecodeCache` with sliding-window BF16 KV plus compressed/indexer state where needed.
- Decode writes the current token at `start_pos % sliding_window` and uses compressed slots for compressed attention layers.

This is not yet a service-level KV manager. The current path has no paged allocator, no prefix cache, no cross-request reuse, no eviction policy, no request-owned KV handle, and no P/D connector handoff.

## Next Gates

The canonical follow-up tasks are:

- task #15: split DeepSeek V4 direct generation into request state plus externally driven prefill/decode-step.
- task #16: introduce service-level KV ownership baseline before bs>1 decode.
- task #17: add bs>1 greedy decode scheduling after request state and KV ownership are explicit.
- task #18: add HTTP/vLLM-compatible concurrent serving benchmarks after the bs>1 path exists.
- task #19: defer prefix/paged KV and P/D handoff design until the single-node bs>1 path stabilizes.
