# Qwen3-4B serving: openinfer vs vLLM on RTX 5090

**Created**: 2026-06-10
**TL;DR**: TP1 Qwen3-4B on one RTX 5090, both engines driven by `vllm bench serve`
(Poisson arrivals, seed 42). Low load (QPS ≤ 8): openinfer wins TTFT p50 by ~10%,
vLLM wins TPOT at every level (gap grows +11% → +27% with batch size). Saturation:
openinfer knees at ~QPS 10, vLLM at ~QPS 12–14; vLLM's overloaded peak throughput is
~12% higher (1690 vs 1511 out tok/s). Warm prefix-cache-hit TTFT: openinfer leads at
every input length, 3× at 16k tokens (30.3 ms vs 90.8 ms p50).

Source benchmark for the README performance section (issue #327).

## Setup

| Item | Value |
| --- | --- |
| GPU | 1× NVIDIA GeForce RTX 5090 (32 GB), driver 590.48.01, same GPU for both engines (sequential runs) |
| Model | Qwen3-4B, BF16 safetensors (7.6 GB), TP1 |
| openinfer | commit `6901965` (main, 2026-06-10), release build, CUDA Graph on (default) |
| vLLM | 0.22.1 (PyPI), torch 2.11.0+cu130, Python 3.12, default engine config |
| Client | `vllm bench serve` 0.22.1 on localhost (same host) |

Server commands:

```bash
# vLLM (test 1: prefix cache off for random-prompt fairness)
CUDA_VISIBLE_DEVICES=1 vllm serve /data/Qwen3-4B --port 8100 \
  --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching

# vLLM (test 2: prefix cache on — the default; longer context for the sweep)
CUDA_VISIBLE_DEVICES=1 vllm serve /data/Qwen3-4B --port 8100 --max-model-len 20480

# openinfer (test 1: prefix cache off)
CUDA_VISIBLE_DEVICES=1 target/release/openinfer --model-path /data/Qwen3-4B \
  --port 8100 --no-prefix-cache

# openinfer (test 2: prefix cache on — the default)
CUDA_VISIBLE_DEVICES=1 target/release/openinfer --model-path /data/Qwen3-4B --port 8100
```

Both engines got an unrecorded 8-request warmup (`--num-prompts 8 --request-rate inf --seed 7`)
before measurement, so vLLM torch.compile cold-start does not pollute the sweep.

## Test 1 — QPS sweep, Poisson arrivals

`tools/bench/qps_sweep.sh`: random dataset, `input_len=1024`, `output_len=128`,
`--ignore-eos --temperature 0`, Poisson arrivals (`--burstiness 1.0`), `--seed 42`,
`num_prompts = 60 × QPS` (≈60 s of arrivals per level).

```bash
MODEL=/data/Qwen3-4B PORT=8100 ENGINE=<openinfer|vllm> \
RESULT_DIR=<dir> QPS_LIST="1 2 4 8 10 12 16" INPUT_LEN=1024 OUTPUT_LEN=128 SEED=42 \
VLLM=.venv/bin/vllm tools/bench/qps_sweep.sh
```

### openinfer

| QPS | req/s | out tok/s | TTFT p50 (ms) | TTFT p99 | TPOT p50 | TPOT p99 | ITL p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.98 | 126 | 50.7 | 69.2 | 7.36 | 9.34 | 9.5 |
| 2 | 1.97 | 252 | 52.9 | 98.8 | 8.95 | 11.04 | 49.1 |
| 4 | 3.92 | 502 | 57.3 | 139.2 | 11.09 | 13.34 | 52.9 |
| 8 | 7.85 | 1005 | 69.5 | 211.6 | 14.98 | 23.65 | 97.7 |
| 10 | 9.77 | 1251 | 109.4 | 432.2 | 22.98 | 42.99 | 148.2 |
| 12 | 11.05 | 1415 | 1753.8 | 4199.6 | 44.17 | 44.65 | 171.0 |
| 16 | 11.80 | 1511 | 8898.4 | 19716.1 | 41.12 | 42.01 | 252.0 |

### vLLM 0.22.1

| QPS | req/s | out tok/s | TTFT p50 (ms) | TTFT p99 | TPOT p50 | TPOT p99 | ITL p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.99 | 126 | 57.8 | 73.2 | 6.65 | 7.77 | 12.8 |
| 2 | 1.97 | 252 | 58.8 | 103.7 | 7.17 | 8.92 | 35.3 |
| 4 | 3.93 | 504 | 61.8 | 115.8 | 8.57 | 10.60 | 38.2 |
| 8 | 7.88 | 1008 | 68.1 | 192.4 | 11.82 | 15.80 | 46.7 |
| 10 | 9.80 | 1254 | 75.7 | 243.3 | 13.99 | 20.90 | 80.0 |
| 12 | 11.72 | 1501 | 119.5 | 409.3 | 19.41 | 49.21 | 102.5 |
| 16 | 13.20 | 1690 | 3933.9 | 10684.2 | 76.21 | 80.17 | 118.5 |

### Reading

- **Low load (QPS ≤ 8): TTFT favors openinfer (~10% lower p50), TPOT favors vLLM.**
  The TPOT gap grows with concurrency: +11% at QPS 1 (7.36 vs 6.65 ms) to +27% at
  QPS 8 (14.98 vs 11.82 ms). The growth pattern points at batched decode step cost,
  not sampling — greedy token selection is already batched (#307, in this build);
  isolating the kernel-level cause needs an nsys diff at fixed batch size.
- **Saturation: openinfer knees at ~QPS 10, vLLM holds to ~QPS 12.** At QPS 12
  openinfer's queue diverges (TTFT p50 1.75 s) while vLLM still serves at 119.5 ms.
  Both are overloaded at QPS 16; vLLM's saturated throughput is ~12% higher
  (1690 vs 1511 out tok/s).
- **openinfer's saturated throughput is pinned by the bs=64 decode cap** (largest
  CUDA-graph bucket, `BATCH_BUCKETS` ends at 64). Implied decode concurrency
  (`out_tok/s × TPOT`) sits at 62 in both overloaded openinfer runs — exactly the
  cap — giving a hard ceiling of `64 / TPOT(bs64) ≈ 64/42 ms ≈ 1520 tok/s`, which
  matches the measured 1511. vLLM keeps admitting past 64 (implied concurrency ~129
  at QPS 16) and converts that into its +12%. The low-load TPOT gap is unrelated to
  the cap (implied concurrency ≤ 16 at QPS ≤ 8).
- ITL p99 is consistently worse for openinfer under load (97.7 vs 46.7 ms at QPS 8) —
  scheduling jitter, same tail issue noted in `subsystems/scheduler/scheduler.md`.

## Test 2 — warm (prefix-cache-hit) TTFT vs input length

`tools/bench/warm_ttft_sweep.py`: one fixed random-token prompt per length
(seed 42), sent once cold to populate the GPU prefix cache, then 20 warm samples
of the identical prompt with `max_tokens=1`. Every sample drains the SSE stream
to `[DONE]` before the next request. Prefix cache ON for both engines.

```bash
.venv/bin/python tools/bench/warm_ttft_sweep.py \
  --base-url http://localhost:8100 --model /data/Qwen3-4B \
  --tokenizer /data/Qwen3-4B --lengths 256,512,1024,2048,4096,8192,16384 \
  --samples 20 --seed 42 --output <out>.json
```

### openinfer

| Input len | Cold TTFT (ms) | Warm p50 | Warm p99 |
| --- | --- | --- | --- |
| 256 | 34.4 | 9.5 | 10.5 |
| 512 | 23.7 | 9.9 | 10.1 |
| 1024 | 44.8 | 10.5 | 10.7 |
| 2048 | 88.4 | 11.9 | 12.4 |
| 4096 | 199.1 | 14.5 | 15.1 |
| 8192 | 415.0 | 24.6 | 25.6 |
| 16384 | 1003.0 | 30.3 | 31.4 |

### vLLM 0.22.1

| Input len | Cold TTFT (ms) | Warm p50 | Warm p99 |
| --- | --- | --- | --- |
| 256 | 41.5 | 13.3 | 14.2 |
| 512 | 28.5 | 14.3 | 14.5 |
| 1024 | 50.3 | 16.1 | 16.4 |
| 2048 | 95.2 | 19.8 | 20.5 |
| 4096 | 197.9 | 27.3 | 28.3 |
| 8192 | 449.6 | 46.9 | 49.3 |
| 16384 | 1106.8 | 90.8 | 100.2 |

### Reading

- **openinfer wins warm TTFT at every length, and the gap widens with length:
  1.4× at 256 tokens (9.5 vs 13.3 ms), 3× at 16k (30.3 vs 90.8 ms).** vLLM's warm
  TTFT grows ~7× from 256 → 16k while openinfer grows ~3×.
- Cold TTFT (full prefill) is near parity, openinfer slightly ahead at most
  lengths (16k: 1003 vs 1107 ms).
- p99 stays tight on both engines (single client, no contention) — the warm p99
  column shows jitter, not load.

## Caveats

- `vllm bench serve` is the unified client for test 1; both engines see identical
  request streams (same seed → same prompts and same Poisson arrival schedule).
- The historical openinfer caveat about overreported streaming
  `usage.completion_tokens` (see `playbooks/bench-vs-vllm.md`) did **not** reproduce:
  every openinfer run has `total_output_tokens == completed × 128` exactly, so the
  `output_throughput` field is trusted as-is.
- QPS levels past the knee (12, 16) measure overload behavior, not steady state:
  `req/s < QPS` means the arrival window stretched and TTFT includes queue time.
- Raw result JSONs (one per engine × QPS level, plus the two TTFT sweeps) live on
  the 5090 host under `/data/xingming/bench/20260610-qwen3-5090/`; the tables
  above are transcribed from them.
