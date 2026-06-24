# DeepSeek-V2-Lite Status And Benchmark Ledger

> **TL;DR:** DeepSeek-V2-Lite is a feature-gated EP2 correctness and serving-semantics target. HF / host-staged / NCCL exactness is guarded by a committed case set; #281 adds the first greedy mixed-request serving gate; #280 adds HTTP trace evidence, position-subgroup decode batching, and chunked NCCL collectives so 128-word and mixed 16/128-word HTTP runs complete. The subgroup fast-path lifts short-shape throughput, but CUDA Graph, direct batch, vLLM rows, and long-prompt latency remain diagnostic or unclaimed as production serving parity.

Last touched: 2026-06

## Capability Contract

| Capability | Status | Evidence |
| --- | --- | --- |
| EP2 correctness bring-up | Available | PR #149 adds the model crate, EP2 expert ownership, rank1 expert-only loading, and the host-staged dispatch/combine baseline. |
| Naive NCCL backend | Available | PR #150 adds a dense correctness-first NCCL path. Host-staged remains the transport oracle. |
| HF token/text/hash gate | Available | PR #154 establishes the HF / host-staged / NCCL comparison; PR #176 refreshes it to Transformers `generate(..., use_cache=true)`. |
| HF widened case set | Available | Issue #274 adds a committed case set that keeps the HF / host-staged / NCCL oracle strict while adding additional prompts and diagnostic batch sizes `4` and `8`; the 2026-06-20 2x RTX 5090 run classified all 5 cases as `all_token_text_exact`. |
| Decode attribution | Available | PR #162 and PR #169 add CPU/GPU attribution, route counts, NCCL counters, CUDA event timing, and optional NVTX correlation. |
| Direct same-prompt diagnostic batch | Available | PR #184 and PR #196 cover batch sizes `1`, `4`, and `8` for the fixed same-prompt direct path. |
| Startup observability | Available | Load logs report safetensor shard count, mmap/deserialization timing, per-rank GPU model-load timing, backend, devices, and total EP2 startup time. |
| Device-resident NCCL combine | Available | Issue #275 keeps NCCL combine contributions/results on reusable f32 device scratch and preserves the HF / host-staged / NCCL exact gate on 2x RTX 5090. |
| Device-resident NCCL dense exchange | Available | Issue #276 reuses backend-owned bf16 dense-exchange scratch, clears rank1 zero-send every exchange, removes dense-exchange stream sync from the backend call, and preserves HF / host-staged / NCCL exactness on 2x RTX 5090. |
| NCCL route-plan replay | Available | Issue #277 builds a token-major host route plan once after top-k routing, replays that plan for NCCL expert launches and device contribution accumulation, keeps route counters visible, and preserves HF / host-staged / NCCL exactness on 2x RTX 5090. This remains the eager NCCL oracle path. |
| NCCL CUDA Graph readiness | Covered-shape diagnostic | Schema-2 `cuda_graph_readiness` now includes a fail-closed `full_decode_graph_probe`. The 2026-06-20 run reports capture, instantiate, replay, and verification success with `8/8` verified replays for the retained batch-1 NCCL decode step. |
| First mixed-request serving gate | Available | Issue #281 adds greedy-only request admission, FCFS deferral, explicit request-local rejection/error/finish events, and one owned `DecodeCache` per active request. The 2026-06-23 2x RTX 5090 run passed HF / host-staged / NCCL exactness and the mixed-serving E2E for host-staged and NCCL. |
| Long-shape NCCL collectives | Available | Issue #280 chunks large bf16 dense-exchange and f32 combine all-reduces. The 2026-06-24 2x RTX 5090 NCCL checks preserve HF / host-staged / NCCL exactness and complete 24/64/128-word direct long-shape probes. |
| HTTP trace and position-subgroup decode batching | HTTP evidence | Issue #280 logs DeepSeek-V2-Lite `openinfer_http_trace` records and batches same-position decode subgroups while letting singleton or lagging positions decode independently. The 2026-06-24 2x RTX 5090 NCCL HTTP sweeps below complete short same-shape, 128-word smoke, and mixed 16/128-word cells with full trace coverage. |
| vLLM production parity | Not claimed | The vLLM TP2 / TP2+EP2 snapshot below is a runnable comparison from a documented validation environment, not serving parity or a stock-install claim. |

## Correctness Contract

The retained correctness gate is deliberately narrow:

- model: DeepSeek-V2-Lite;
- devices: single-node EP2 with two local GPUs;
- committed cases: `test_data/deepseek-v2-lite-ep2-cases.json` keeps the original `Hello` / 16-token case and widens the oracle with a few additional prompts plus batch sizes `4` and `8`;
- generation mode: greedy;
- backends: host-staged and `OPENINFER_DSV2_LITE_EP_BACKEND=nccl`.

The comparison gate must be run on the same model snapshot for HF, host-staged, and NCCL outputs. Same-host comparison remains strict: HF, host-staged, and NCCL must be token-exact and text-exact for every committed case and every diagnostic batch row. Host-staged remains the baseline oracle for NCCL transport changes. The latest retained evidence is the 2026-06-24 2x RTX 5090 case-set run with `case_count=5`, top-level `classification=all_token_text_exact`, no comparison warnings, token hash `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225`, and text hash `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347`.

The mixed-request serving E2E computes sequential greedy token-id oracles with `DeepSeekV2LiteEp2Generator::generate_greedy`, then submits concurrent requests through `start_engine`. The retained 2026-06-23 run covers same-length mixed prompts for same-position batch decode, different-length mixed prompts for single-row decode fallback, and a valid request submitted beside an invalid `logprobs` request to prove explicit rejection does not poison the valid stream. Host-staged and NCCL both passed the mixed-serving E2E.

The Rust E2E accepts the known HF-confirmed RTX 5090 and A800 hash pairs for this narrow shape, because the same model snapshot has produced different exact greedy text on those hosts while still matching HF on each host. Do not use the static hash pair list as a substitute for the same-host HF comparison when changing accuracy-sensitive code.

## Benchmark Ledger

### Direct Same-Prompt Diagnostic Batch

This path is useful for attribution and for avoiding the earlier row-loop TPOT measurement. It is separate from the first mixed-request serving gate and is not production continuous batching:

- every row uses the same prompt;
- prefill remains conservative;
- the direct benchmark path is not `/v1/completions` serving;
- it does not prove request admission, per-request KV ownership, fairness, or mixed-request scheduling.

Current retained direct snapshot from the issue #277 branch (`2f52ed6`, 2026-06-15, 2x RTX 5090 / SYS interconnect). Shape: `prompt="Hello"`, `output_len=16`, `warmup=5`, `iters=20`; every row produced token trace hash `ed0eab52473991fc`. `decode tok/s` is the benchmark report's aggregate `metrics.decode_tok_s`. This refresh replaces the older PR #184 row values for the current branch ledger, but it should not be read as an isolated route-plan speedup because the retained snapshot was rerun on a different validation environment.

| Batch | Backend | steady TPOT p50 ms | steady TPOT avg ms | decode tok/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | host-staged | 55.727 | 57.313 | 17.486 |
| 1 | NCCL | 181.795 | 188.420 | 5.321 |
| 4 | host-staged | 193.954 | 198.905 | 20.106 |
| 4 | NCCL | 303.389 | 311.621 | 12.821 |
| 8 | host-staged | 385.013 | 394.908 | 20.270 |
| 8 | NCCL | 472.045 | 483.538 | 16.517 |

PR #196 extends attribution for the same direct diagnostic shapes. The retained A800 attribution gate keeps `batch-size=1/4/8`, `prompt="Hello"`, `output_len=16`, host-staged, and NCCL exact against the same-host HF gate.

### HTTP Concurrency Pressure

The issue #277 branch was also run through `/v1/completions` with `vllm bench serve` used only as the common HTTP client. Shape: random input length `2`, output length `16`, `24` prompts, `temperature=0`, `ignore_eos`, `--max-concurrency 1/4/8`, OpenInfer `--cuda-graph=false`.

OpenInfer streaming currently makes the client-side TPOT fields near-zero in this shape, so this table reports output throughput and throughput-derived milliseconds per output token computed as `duration / total_output_tokens`. `--max-concurrency` should be read as concurrent request pressure, not as proof of true internal OpenInfer batch size.

| Backend | conc | completed | output tok/s | throughput-derived ms/output token | mean TTFT ms | median TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| host-staged | 1 | 24/24 | 20.912 | 47.820 | 764.471 | 740.296 |
| host-staged | 4 | 24/24 | 21.030 | 47.552 | 2838.390 | 3036.649 |
| host-staged | 8 | 24/24 | 20.964 | 47.700 | 5198.935 | 5991.506 |
| NCCL | 1 | 24/24 | 6.302 | 158.689 | 2538.216 | 2553.374 |
| NCCL | 4 | 24/24 | 6.326 | 158.083 | 9491.680 | 10097.244 |
| NCCL | 8 | 24/24 | 6.341 | 157.710 | 17242.941 | 20110.121 |

### Issue #280 HTTP Trace, Subgroup Decode, And Long Prompts

Retained 2026-06-24 evidence on 2x RTX 5090, NCCL EP2 with chunked large collectives, release `openinfer-server --features deepseek-v2-lite`, `/v1/completions`, `temperature=0`, `ignore_eos=true`, `max_tokens=16`, `num_requests=8`, `repeats=3`, with server logs consumed by `scripts/bench_http_serving.py`.

This is HTTP serving evidence for request-level trace attribution, completed/failed/timeout accounting, output hash stability, and same-position decode subgroups. It does not prove vLLM parity, production EP readiness, or acceptable long-prompt latency.

Long-shape NCCL direct smoke after chunking:

| prompt words | prompt tokens | generated | token hash |
| ---: | ---: | ---: | --- |
| 24 | 32 | 16 | `78dfd3123da2ed54829027384682c6eb562a6d29b2a92ee96a7b26d7acc4e226` |
| 64 | 86 | 16 | `920f24edd016e8e16973f304e5cb909303812a930a9c6608694d0b47f2c48918` |
| 128 | 172 | 16 | `5fd2f30c1f1c4e4477791f233c30ce6c0148dba91737c358cb357a2065482861` |

HTTP 128-word smoke: `prompt_words=128`, `concurrency=1`, `num_requests=2`, `warmup=0`, actual prompt tokens `170,171`.

| completed | failed/timeouts | QPS | output tok/s | TTFT avg ms | TPOT/ITL avg ms | active max | decode batch max | traces | output hash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2/2 | 0/0 | 0.331 | 5.295 | 2535.3 | 32.3 | 1 | 1 | 2/2 | `2299c1c50f50e819` |

Same-shape sweep: `prompt_words=16`, actual prompt tokens `20..23`.

| conc | completed | failed/timeouts | QPS avg | output tok/s avg | TTFT avg ms | TPOT/ITL avg ms | active max | decode batch max | traces | output hash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 8/8 x3 | 0/0 | 2.068 | 33.083 | 240.5 | 16.2 | 1 | 1 | 8/8 x3 | `0989a10c5d842d8b` |
| 2 | 8/8 x3 | 0/0 | 2.248 | 35.967 | 332.8 | 36.8 | 2 | 2 | 8/8 x3 | `0989a10c5d842d8b` |
| 4 | 8/8 x3 | 0/0 | 2.243 | 35.890 | 481.1 | 85.1 | 4 | 2 | 8/8 x3 | `0989a10c5d842d8b` |
| 8 | 8/8 x3 | 0/0 | 2.290 | 36.635 | 985.9 | 163.1 | 8 | 4 | 8/8 x3 | `0989a10c5d842d8b` |

Interpretation: short-shape throughput improved at every concurrency point, while TTFT stayed queue-sensitive and moved a little in both directions. The trace fields still prove the scheduler did batch live decode rows (`decode_batch_size_max=4` at concurrency 8), so `--max-concurrency` is no longer being inferred as batch size from the client alone.

Mixed-shape proof: `prompt_words=16,128`, `num_requests=8` per repeat, four short and four long requests, `warmup=2`.

| conc | completed | failed/timeouts | QPS avg | output tok/s avg | TTFT avg ms | TPOT/ITL avg ms | active max | decode batch max | traces | output hash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 4 | 8/8 x3 | 0/0 | 0.597 | 9.545 | 2756.2 | 260.2 | 4 | 2 | 8/8 x3 | `d53e286068a7cd5e` |
| 8 | 8/8 x3 | 0/0 | 0.602 | 9.634 | 5251.0 | 527.1 | 8 | 2 | 8/8 x3 | `d53e286068a7cd5e` |

Interpretation: the old long-prompt prefill failure is fixed for this HTTP contract, and the post-fastpath rerun lifts mixed 16/128 throughput a bit, but the row is still dominated by long-prompt prefill and admission queueing. The c4/c8 rows prove subgroup batching can happen with mixed prompt lengths (`decode_batch_size_max=2` here), yet the latency profile is not a production serving claim.

### vLLM Repaired-Environment Comparison

In response to issue #170's request for a vLLM TP2+EP2 or pure TP2 comparison, the 2026-06-15 2x RTX 5090 run now has a successful vLLM baseline for the same DeepSeek-V2-Lite model/tokenizer snapshot and short HTTP benchmark shape as the OpenInfer table above.

This is a validation environment with documented package/toolchain fixes, not a stock vLLM install claim. The working run used vLLM `0.22.1`, Torch `2.11.0+cu130`, Transformers `5.12.0`, FlashInfer `0.6.12`, `FLASHINFER_CUDA_ARCH_LIST=12.0f`, CUDA 13.3-aligned Python CUDA packages, and FastAPI `0.115.14` / Starlette `0.46.2` / `prometheus-fastapi-instrumentator` `7.1.0`. Those fixes were needed after earlier SM120 FlashInfer JIT/linking and HTTP middleware failures.

The server was launched eager bf16 with `--max-model-len 512`, `--gpu-memory-utilization 0.70`, `--tensor-parallel-size 2`; TP2+EP2 additionally used `--enable-expert-parallel`, and the server log confirmed expert parallelism with `32/64` local/global experts.

Client shape: `vllm bench serve --backend openai --endpoint /v1/completions --model DeepSeek-V2-Lite --tokenizer models/DeepSeek-V2-Lite --dataset-name random --random-input-len 2 --random-output-len 16 --num-prompts 24 --temperature 0 --ignore-eos`, run separately with `--max-concurrency 1`, `4`, and `8`.

| Runtime | conc | completed | output tok/s | mean TTFT ms | median TTFT ms | mean TPOT ms | median TPOT ms | mean ITL ms | median ITL ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM TP2 | 1 | 24/24 | 19.140 | 217.484 | 48.451 | 41.193 | 41.193 | 41.193 | 40.312 |
| vLLM TP2 | 4 | 24/24 | 84.790 | 125.694 | 125.623 | 41.678 | 41.635 | 41.678 | 41.247 |
| vLLM TP2 | 8 | 24/24 | 167.600 | 129.826 | 125.069 | 41.933 | 41.810 | 41.933 | 41.033 |
| vLLM TP2+EP2 | 1 | 24/24 | 23.834 | 62.245 | 46.911 | 40.576 | 40.596 | 40.576 | 40.502 |
| vLLM TP2+EP2 | 4 | 24/24 | 87.078 | 121.439 | 121.553 | 40.641 | 40.409 | 40.641 | 40.253 |
| vLLM TP2+EP2 | 8 | 24/24 | 174.097 | 125.144 | 119.399 | 40.348 | 40.272 | 40.348 | 40.117 |

Earlier failed vLLM attempts are still useful as environment-discovery evidence:

| Environment | Modes tried | Result |
| --- | --- | --- |
| vLLM `0.9.2`, Torch `2.7.0+cu128`, Transformers `4.53.3` | TP2, TP2+EP2, V1/V0, default/eager fallback, plus one low-memory smoke | server never reached readiness; failures were `CUBLAS_STATUS_NOT_INITIALIZED` during worker profile plus one low-memory illegal-memory-access smoke |
| vLLM `0.22.1`, Torch `2.11.0+cu130`, Transformers `5.12.0` | TP2 and TP2+EP2 default/eager fallback | server never reached readiness until FlashInfer SM120f, CUDA 13.3 headers/libs, unversioned CUDA linker names, and FastAPI middleware compatibility were repaired |
| vLLM `0.22.1` controlled backend fallback | TP2 and TP2+EP2 with `--attention-backend TRITON_ATTN --moe-backend triton` | server never reached readiness; `TRITON_ATTN` is not valid for DeepSeek-V2 MLA (`MLA not supported`) |

Interpretation:

- direct same-prompt diagnostics show NCCL is still much slower than host-staged, although aggregate decode throughput improves with larger diagnostic batch size;
- NCCL remains a correctness-first backend and is still significantly slower than host-staged;
- the #280 HTTP trace proves active request sets and subgroup decode batches, but throughput still scales only weakly on NCCL EP2 and long prompts have high TTFT;
- vLLM TP2 and TP2+EP2 both show higher aggregate output tok/s under this short HTTP pressure shape than the current OpenInfer HTTP pressure rows;
- TP2+EP2 is slightly ahead of TP2 in this short snapshot, but this is one workload and one documented validation environment;
- standalone Torch bf16 GEMM passed on both GPUs in the failed and repaired vLLM environments, so the earlier failures were specific to the vLLM DeepSeek-V2 server/model startup and Python package/toolchain path rather than a blanket CUDA outage.

## Claim Boundaries

Use these labels consistently:

| Label | Meaning | Do not infer |
| --- | --- | --- |
| `direct single-row` | In-process batch `1` decode. | HTTP serving throughput. |
| `direct same-prompt diagnostic batch` | Fixed same-prompt direct batch sizes `1/4/8`. | Production continuous batching or mixed-request scheduling. |
| `first mixed-request serving gate` | Greedy-only EP2 scheduler path with explicit admission/rejection/deferral, per-request host-side decode `DecodeCache`, active cap `8`, and exact sequential-oracle E2E. | vLLM parity, sparse dispatch, production EP readiness, HTTP throughput scaling, non-greedy sampling, or logprobs support. |
| `HTTP trace/subgroup evidence` | `/v1/completions` requests have per-request `openinfer_http_trace` rows, and HTTP sweeps show non-1 `active_set_size` and `decode_batch_size_max`. | Fair vLLM parity, long-prompt latency readiness, or a before/after percentage unless a paired baseline run is recorded. |
| `covered NCCL decode graph probe` | Probe-only batch-1 `Hello` decode step captured, instantiated, replayed, and token-verified under CUDA Graph. | Default serving graph coverage, multi-step graph replay, batch `4/8` graph coverage, or performance improvement. |
| `HTTP concurrency pressure` | `vllm bench serve --max-concurrency N` against an HTTP endpoint. | True OpenInfer batch size unless the engine path proves it. |
| `vLLM comparison from documented environment` | vLLM TP2 / TP2+EP2 after target-environment package/toolchain fixes. | Stock vLLM install support, OpenInfer serving parity, or production readiness. |

Do not claim:

- production EP readiness;
- sparse dispatch readiness;
- multi-node EP support;
- vLLM serving parity;
- performance improvement from the status tables alone.

## Next Gates

Issue #205 records the model roadmap. Maintainer feedback there calls out NCCL plus CUDA Graph as the likely best decode direction, with host staging possibly deprecated later. Treat that as a future direction, not as current evidence.

The graph-readiness diagnostic is fail-closed. `full_decode_capture_ready=true` is valid only when `full_decode_graph_probe` captures, instantiates, replays, and verifies the covered shape. The optional f32 NCCL graph smoke remains collective-only evidence. HF, host-staged, and NCCL still need token/text exactness for the committed case set.

The next implementation should be chosen from measured evidence:

1. Keep the widened HF / host-staged / NCCL case set current.
   - keep the committed cases and row-level comparison shape in sync with the accuracy docs;
   - treat the widened oracle as correctness evidence only, not serving evidence;
   - keep host-staged as the baseline oracle while it exists.

2. Decide whether to productize the probe-only CUDA Graph path.
   - keep HF / host-staged / NCCL exact before and after;
   - keep host-staged as the correctness baseline while it exists;
   - preserve attribution before and after the change;
   - keep the eager NCCL route-plan path as the serving oracle until the graph path is widened and measured;
   - keep the graph claim at batch `1`, `prompt="Hello"`, `output_len=16` until another probe covers more;
   - treat any future `failure_stage` as fail-closed evidence.

3. Keep a fair serving benchmark contract around future performance work.
   - OpenInfer host-staged.
   - OpenInfer NCCL.
   - vLLM TP2.
   - vLLM TP2+EP2 when supported.
   - default vLLM configuration plus a controlled configuration with cache/flag choices recorded.

4. Widen the first mixed-request serving gate before broader throughput claims.
   - keep the fixed EP2 path and exact sequential oracle until a wider oracle replaces it;
   - keep greedy-only admission explicit until sampling/logprobs have their own gate;
   - keep direct same-prompt batch labeled diagnostic;
   - reduce long-prompt prefill and admission-queue TTFT before claiming long-prompt serving readiness;
   - add paired baseline runs before claiming a percentage speedup from subgroup batching.

5. Keep MoE internals readable.
   - routing, dispatch, expert execution, and combine should remain distinguishable in code and attribution;
   - avoid introducing a generic EP framework before the DeepSeek-V2-Lite EP2 path has a measured reason to need it.
