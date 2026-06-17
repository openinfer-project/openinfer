# Qwen3.5-4B serving: openinfer vs vLLM on RTX 5090

**Created**: 2026-06-15

**TL;DR**: TP1 Qwen3.5-4B on one RTX 5090, measured through the same
`vllm bench serve` HTTP client against vLLM 0.23.0, the latest stable release
checked on 2026-06-15. On fixed-length single-concurrency client-contract
probes, openinfer reports lower TTFT, but it also reports fewer prompt tokens
per request (`1,944` vs `2,048` on 2048/1, `971` vs `1,024` on 1024/256), so
TTFT is not a token-normalized prefill comparison. vLLM keeps the steady decode
edge: `6.32 ms` TPOT p50 vs openinfer `7.31 ms`, or `152.3` vs `133.6` output
tok/s.

Source benchmark for the README Qwen3.5 performance rows.

## Setup

| Item | Value |
| --- | --- |
| GPU | 1x NVIDIA GeForce RTX 5090 (32 GB), driver 580.105.08, same GPU for both engines (sequential runs) |
| Model | Qwen3.5-4B, BF16 safetensors, TP1, text-only serving |
| openinfer | main @ `f3dcdf4`, release build with `--features qwen35-4b`, CUDA Graph on by default |
| vLLM | 0.23.0 from PyPI, checked as the latest stable release on 2026-06-15 ([PyPI](https://pypi.org/project/vllm/), [GitHub releases](https://github.com/vllm-project/vllm/releases)) |
| vLLM serve flags | `--language-model-only`, `--no-enable-prefix-caching`, `--max-model-len 8192`, `--gpu-memory-utilization 0.9` |
| vLLM env | `VLLM_USE_FLASHINFER_SAMPLER=0`; this SM120/CUDA 12.8 host hit a FlashInfer sampler startup error otherwise. Attention still selected FlashAttention 2. |
| Client | `vllm bench serve` 0.23.0 on localhost, OpenAI `/v1/completions` backend |

Client flags for both engines:

| Field | Value |
| --- | --- |
| Dataset | random |
| Request count | `--num-prompts 30` |
| Warmup | `--num-warmups 1` |
| Concurrency | `--max-concurrency 1`, `--request-rate inf` |
| Length control | `--random-range-ratio 0.0` |
| Decoding | `--temperature 0`, `--ignore-eos` |

## Results

| Workload | Metric | openinfer | vLLM 0.23.0 | Read |
| --- | --- | ---: | ---: | --- |
| 2048 input / 1 output | completed | 30/30 | 30/30 | both clean |
| 2048 input / 1 output | reported input tokens | 58,324 (1,944.1/request) | 61,440 (2,048.0/request) | openinfer 5.1% fewer |
| 2048 input / 1 output | TTFT p50 | 101.77 ms | 115.23 ms | reported lower; prompt-token counts differ |
| 2048 input / 1 output | TTFT p99 | 108.69 ms | 123.73 ms | reported lower; prompt-token counts differ |
| 2048 input / 1 output | request/output tok/s | 9.94 | 8.61 | client-contract throughput; prompt-token counts differ |
| 1024 input / 256 output | completed | 30/30 | 30/30 | both clean |
| 1024 input / 256 output | reported input tokens | 29,123 (970.8/request) | 30,720 (1,024.0/request) | openinfer 5.2% fewer |
| 1024 input / 256 output | TTFT p50 | 53.75 ms | 67.38 ms | reported lower; prompt-token counts differ |
| 1024 input / 256 output | TPOT p50 | 7.31 ms | **6.32 ms** | vLLM 13.4% lower |
| 1024 input / 256 output | TPOT p99 | 7.36 ms | **6.35 ms** | vLLM 13.7% lower |
| 1024 input / 256 output | output tok/s | 133.57 | **152.28** | vLLM 14.0% higher |

Raw result JSONs were kept on the 5090 validation host under these filenames:

- `openinfer-qwen35-prefill-2048x1-n30-warm1.json`
- `openinfer-qwen35-decode-1024x256-n30-warm1.json`
- `vllm023-qwen35-prefill-2048x1-n30-warm1.json`
- `vllm023-qwen35-decode-1024x256-n30-warm1.json`

## Caveats

- This is a fixed-length, single-concurrency HTTP serving probe. It is not a
  QPS sweep, saturation result, prefix-cache result, or production-load claim.
- The table uses the client-requested input/output lengths as the workload
  contract. Response `usage.prompt_tokens` totals differ between the two
  servers on these random text prompts: `58,324` vs `61,440` for 2048/1 and
  `29,123` vs `30,720` for 1024/256. Read the TTFT rows as fixed-client
  workload timings, not token-normalized prefill throughput.
- vLLM startup was made serviceable on this host by disabling the FlashInfer
  sampler path. That changes sampling implementation, not the measured
  attention backend; the server log selected FlashAttention 2.
- vLLM still has the decode TPOT edge on this shape. The openinfer result to
  carry into the README is narrower: lower reported TTFT on both fixed-client
  shapes, with a prompt-token-count caveat and a decode throughput gap still
  visible.
