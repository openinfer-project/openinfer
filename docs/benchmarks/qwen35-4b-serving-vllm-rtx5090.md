# Qwen3.5-4B serving: openinfer vs vLLM on RTX 5090

**Created**: 2026-06-15

**Updated**: 2026-06

**TL;DR**: This Qwen3.5 decode-tuning branch improves openinfer's same-host
direct benchmark by `2.1-3.2%` steady TPOT on decode-heavy shapes and `2.2%`
TTFT on 2048/1. Against vLLM 0.23.0 through the same `vllm bench serve` HTTP
client, openinfer is close on 1-token-prompt decode (`+1.1%` TPOT on 1/256,
`+2.6%` on 1/512), but still trails vLLM on 1024/256 (`+12.0%` TPOT at
concurrency 1) and the gap widens at higher HTTP concurrency. TTFT rows are a
fixed-client contract only; 1024/256 and 2048/1 report different prompt-token
totals between the servers.

Source benchmark for the README Qwen3.5 performance rows.

## Setup

| Item | Value |
| --- | --- |
| GPU | 1x NVIDIA GeForce RTX 5090 (32 GB), driver 580.105.08, same GPU for each sequential run |
| Model | Qwen3.5-4B, BF16 safetensors, TP1, text-only serving |
| openinfer | branch based on upstream/main `a1846ca`, release build with `--features qwen35-4b`, CUDA Graph decode on by default |
| vLLM | 0.23.0 from PyPI, latest stable release checked in June 2026 |
| openinfer serve flags | `--no-prefix-cache`, CUDA Graph decode on by default |
| vLLM serve flags | `--language-model-only`, `--no-enable-prefix-caching`, `--max-model-len 8192`, `--gpu-memory-utilization 0.9` |
| vLLM env | `VLLM_USE_FLASHINFER_SAMPLER=0`; this SM120/CUDA 12.8 host hit a FlashInfer sampler startup error otherwise. Attention selected FlashAttention 2. |
| Client | `vllm bench serve` 0.23.0 on localhost, OpenAI `/v1/completions` backend |
| Profiler | Nsight Systems 2025.3.2 for OpenInfer direct measured-range and full HTTP traces. Nsight Compute was not used for claims on this host. |

Client flags for both HTTP engines:

| Field | Value |
| --- | --- |
| Dataset | random |
| Request count | `--num-prompts 64` |
| Warmup | `--num-warmups 2` |
| Request rate | `--request-rate inf` |
| Length control | `--random-range-ratio 0.0` |
| Decoding | `--temperature 0`, `--ignore-eos` |
| Seed | `--seed 20260618` |

## OpenInfer Direct A/B

Same binary interface, same model, same GPU. Baseline is upstream/main before
the Qwen3.5 gate/up MLP fusion and decode cublasLt tuning in this branch.

| Workload | Metric | upstream/main | tuned branch | Delta |
| --- | --- | ---: | ---: | ---: |
| 1 input / 256 output | steady TPOT avg | 6.524 ms | 6.386 ms | -2.1% |
| 1 input / 512 output | steady TPOT avg | 6.603 ms | 6.397 ms | -3.1% |
| 1024 input / 256 output | steady TPOT avg | 7.338 ms | 7.100 ms | -3.2% |
| 2048 input / 1 output | TTFT avg | 97.978 ms | 95.855 ms | -2.2% |

## HTTP Fixed Shapes

| Workload | Metric | openinfer | vLLM 0.23.0 | Read |
| --- | --- | ---: | ---: | --- |
| 1 input / 256 output | completed | 64/64 | 64/64 | both clean |
| 1 input / 256 output | TTFT mean | 11.83 ms | 15.46 ms | openinfer lower on this client path |
| 1 input / 256 output | TPOT mean | 6.282 ms | **6.214 ms** | vLLM 1.1% lower |
| 1 input / 256 output | output tok/s | 158.58 | **159.95** | vLLM 0.9% higher |
| 1 input / 512 output | completed | 64/64 | 64/64 | both clean |
| 1 input / 512 output | TTFT mean | 11.55 ms | 16.23 ms | openinfer lower on this client path |
| 1 input / 512 output | TPOT mean | 6.381 ms | **6.221 ms** | vLLM 2.5% lower |
| 1 input / 512 output | output tok/s | 156.45 | **160.22** | vLLM 2.4% higher |
| 1024 input / 256 output | completed | 64/64 | 64/64 | both clean |
| 1024 input / 256 output | reported input tokens | 63,459 (991.5/request) | 65,536 (1,024.0/request) | prompt-token totals differ |
| 1024 input / 256 output | TTFT mean | 55.29 ms | 66.34 ms | fixed-client timing, not token-normalized prefill |
| 1024 input / 256 output | TPOT mean | 7.110 ms | **6.346 ms** | vLLM 10.8% lower |
| 1024 input / 256 output | output tok/s | 136.98 | **151.92** | vLLM 10.9% higher |
| 2048 input / 1 output | completed | 64/64 | 64/64 | both clean |
| 2048 input / 1 output | reported input tokens | 126,957 (1,983.7/request) | 131,072 (2,048.0/request) | prompt-token totals differ |
| 2048 input / 1 output | TTFT mean | 97.41 ms | 101.93 ms | fixed-client timing, not token-normalized prefill |
| 2048 input / 1 output | output tok/s | 10.24 | 9.78 | client-contract throughput; prompt-token totals differ |

## HTTP Concurrency Sweep

Workload: 1024 input / 256 output, `num_prompts=64`, random fixed-length
client probes, prefix cache disabled on both servers.

| Max concurrency | openinfer TTFT mean | vLLM TTFT mean | openinfer TPOT mean | vLLM TPOT mean | openinfer output tok/s | vLLM output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 55.29 ms | 66.34 ms | 7.110 ms | **6.346 ms** | 136.98 | **151.92** |
| 2 | 82.07 ms | 97.64 ms | 8.146 ms | **7.148 ms** | 237.06 | **266.55** |
| 4 | 167.04 ms | 165.88 ms | 9.263 ms | **7.459 ms** | 404.76 | **494.85** |
| 8 | 352.18 ms | 232.45 ms | 11.333 ms | **8.650 ms** | 631.33 | **839.05** |
| 16 | 741.95 ms | 358.21 ms | 15.566 ms | **9.823 ms** | 868.63 | **1425.72** |

Extra diagnostic: an in-process openinfer 1024/256 concurrency-16 direct run
reported steady TPOT avg `9.202 ms`, far below openinfer's HTTP concurrency-16
TPOT `15.566 ms`. That points to a serving/scheduler/client interaction gap to
investigate before treating the high-concurrency HTTP delta as purely a model
kernel problem.

## Profiling Notes

Nsight Systems confirms the direct OpenInfer model path and the HTTP serving
path should be read separately:

- Direct `bench_serving request` with CUDA profiler capture around the measured
  1024/256 concurrency-16 iteration reported steady TPOT avg `9.320 ms`
  (`9.201 ms` p50, `9.300 ms` p99). The top GPU kernels were the tuned
  cublasLt/CUTLASS GEMMs: 256x128 tile `50.9%` of kernel time, 128x256
  `15.4%`, and 64x256 `13.5%`. Qwen3.5 GDR, conv, SiLU, and FlashInfer
  attention were much smaller in that trace.
- The full HTTP concurrency-16 trace measured TPOT avg `16.010 ms` for the
  benchmarked requests, close to the standalone HTTP sweep row. Because the
  trace includes server startup and warmup, its kernel-time totals are not a
  measured-only attribution table. Still, the CUDA API/OS runtime summaries
  showed large `cudaEventSynchronize`, `cuMemcpyHtoDAsync`, `futex`, `poll`,
  and `epoll_wait` time, which points at serving/scheduler/event synchronization
  work rather than one obvious missing model kernel.
- `bench_serving decode` was not usable as a pure Qwen3.5 decode profile yet:
  it requires cached-token accounting to prove prefill was excluded, and Qwen3.5
  does not currently report that surface.

## Caveats

- This is a same-host synthetic benchmark, not a production traffic trace.
- The 1-token-prompt decode rows have equal reported input/output token totals.
  The 1024/256 and 2048/1 rows do not, so TTFT is a fixed-client workload
  timing rather than token-normalized prefill throughput.
- Prefix cache was disabled on both servers for this refresh.
- vLLM startup was made serviceable on this host by disabling the FlashInfer
  sampler path. The measured decode was greedy (`temperature=0`); attention
  still selected FlashAttention 2.
- Nsight Systems was used for OpenInfer direct and HTTP diagnostics. The direct
  trace has a measured CUDA profiler range; the HTTP trace is full-process and
  should be treated as coarse attribution only.
- The current honest claim is narrower than vLLM parity: this branch improves
  openinfer Qwen3.5 decode TPOT by a few percent, closes most of the prompt-len
  1 HTTP decode gap, and leaves the 1024/256 plus high-concurrency HTTP gap as
  follow-up work.
