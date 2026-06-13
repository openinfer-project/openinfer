# Qwen3-4B serving: openinfer vs vLLM on RTX 5090

**Created**: 2026-06-13 (supersedes the 2026-06-10 run)

**TL;DR**: TP1 Qwen3-4B on one RTX 5090. openinfer's resident footprint is ~5× smaller
(771 MB vs 3814 MB) and it reaches HTTP-ready in ~3 s vs vLLM's 70 s cold / 32.7 s warm — no
`torch.compile`. Warm prefix-cache-hit TTFT leads at every length, 3.6× at 16k (26.3 vs
95.6 ms). QPS sweep: low load comparable, vLLM keeps a TPOT edge at mid load (QPS 8–12), and
openinfer now edges ahead at saturation (1794 vs 1692 out tok/s) after batched lm_head +
sampling (#362) lifted the prior ~1511 cap. pegaflow KV-offload restores prefixes from host
DRAM at 2.6–9.1× over cold prefill.

Source benchmark for the README performance section.

## Setup

| Item | Value |
| --- | --- |
| GPU | 1× NVIDIA GeForce RTX 5090 (32 GB), driver 590.48.01, CUDA 13.1 build, same GPU for both engines (sequential runs) |
| Model | Qwen3-4B, BF16 safetensors (7.6 GB), TP1 |
| openinfer | main @ `0b42ed3` (#377), release build, CUDA Graph on (default), prefix cache on |
| vLLM | 0.22.1 (PyPI), prefix cache on (default) |
| Client | `vllm bench serve` 0.22.1 on localhost (same host), GPU 0 |

Both engines: prefix cache ON, same Poisson stream (seed 42), `input_len=1024` /
`output_len=128`. Each got an unrecorded 8-request warmup before the QPS sweep, so vLLM's
`torch.compile` cold start does not pollute the latency sweep.

## Footprint (the headline)

| metric | openinfer | vLLM | ratio |
| --- | --- | --- | --- |
| RSS before stress (idle, loaded) | 771 MB | 3814 MB | 4.9× less |
| RSS after stress | 1064 MB | 3863 MB | 3.6× less |
| RSS peak (load transient, HWM) | 8156 MB | — | — |
| startup to HTTP ready | 2.99 s | 70.0 s | 23× faster |
| GPU mem (default util) | 28832 MiB | 30290 MiB | — |

- openinfer is a single process; vLLM RSS is summed over its process tree.
- The 8156 MB peak is transient — weights are read through `mmap` during the H2D copy; the
  mmap is dropped after load (#377), so steady-state RSS settles at 771 MB.

## QPS sweep (Poisson, in1024/out128, seed 42)

| QPS | oi out tok/s | vllm out tok/s | oi TTFT p50 | vllm TTFT p50 | oi TPOT p50 | vllm TPOT p50 | oi TPOT p99 | vllm TPOT p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1  | 126.1 | 126.0 | 60.9 | 57.1 | 6.89 | 6.90 | 8.21 | 8.02 |
| 2  | 252.3 | 252.3 | 31.1 | 39.4 | 6.87 | 7.05 | 9.07 | 9.18 |
| 4  | 503.8 | 503.4 | 59.9 | 42.4 | 7.50 | 7.91 | 10.93 | 10.30 |
| 8  | 1008.3 | 1007.5 | 67.7 | 69.1 | 14.61 | 12.09 | 22.16 | 16.14 |
| 10 | 1249.6 | 1253.6 | 90.2 | 79.4 | 21.05 | 14.44 | 32.16 | 21.08 |
| 12 | 1489.8 | 1499.9 | 134.9 | 119.5 | 33.08 | 19.75 | 61.71 | 49.10 |
| 16 | 1794.1 | 1692.6 | 2591.1 | 3712.4 | 65.02 | 78.22 | 65.22 | 81.46 |

- Low load (QPS ≤ 4) is comparable on both throughput and TTFT.
- Mid load (QPS 8–12): vLLM wins TPOT (12.09 vs 14.61 ms at QPS 8, 19.75 vs 33.08 at QPS 12).
- Saturation (QPS 16, both overloaded): openinfer edges ahead on throughput (1794 vs 1692 out
  tok/s) and req/s (14.0 vs 13.2). This is the change from the 2026-06-10 run, where openinfer
  was capped at ~1511 by the bs=64 decode bucket — batched lm_head + sampling (#362) lifted it.

## Warm prefix-cache-hit TTFT (HBM hit) vs input length

| len | oi cold | oi warm p50 | oi warm p99 | vllm cold | vllm warm p50 | vllm warm p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 16.2 | 8.5 | 8.8 | 24.0 | 14.5 | 19.1 |
| 512 | 24.6 | 8.6 | 8.8 | 30.5 | 16.0 | 16.4 |
| 1024 | 44.0 | 9.2 | 9.5 | 52.5 | 18.4 | 19.0 |
| 2048 | 92.0 | 10.4 | 10.8 | 97.8 | 23.7 | 24.4 |
| 4096 | 211.5 | 12.7 | 13.4 | 200.4 | 34.1 | 36.2 |
| 8192 | 460.0 | 21.6 | 22.8 | 451.9 | 58.6 | 59.9 |
| 16384 | 1143.9 | 26.3 | 27.9 | 1115.4 | 95.6 | 98.2 |

- openinfer wins warm TTFT at every length; the gap widens with length, reaching 3.6× at 16k
  (26.3 vs 95.6 ms p50).
- Cold (full-prefill) TTFT is near parity (16k: 1143.9 vs 1115.4 ms).
- openinfer's warm p99 stays within ~1–2 ms of p50 at every length.

## pegaflow KV offload — pure-L2 TTFT (cold full-prefill vs host-tier restore)

`--kv-offload --kv-offload-host-gib 16 --no-prefix-cache` (evict-before-probe, so every prefix
is restored from host DRAM rather than HBM).

| len | cold (full prefill) | L2 warm p50 (host restore) | speedup |
| --- | --- | --- | --- |
| 256 | 25.4 | 9.8 | 2.6× |
| 512 | 25.6 | 11.6 | 2.2× |
| 1024 | 45.3 | 15.4 | 2.9× |
| 2048 | 92.5 | 22.9 | 4.0× |
| 4096 | 211.1 | 37.5 | 5.6× |
| 8192 | 461.3 | 71.4 | 6.5× |
| 16384 | 1140.5 | 125.5 | 9.1× |

Tiering picture at 16k: HBM hit 26 ms < host-tier L2 restore 126 ms ≪ cold prefill 1140 ms.

## Startup: warm vs cold

vLLM startup is dominated by `torch.compile`. Measured with identical env/command, only the
cache state differs:

| | openinfer | vLLM 0.22.1 |
| --- | --- | --- |
| cold (compile from scratch / first start) | 2.99 s | 70.0 s |
| warm (compile cache hit, 15 GB `~/.cache/vllm`) | ~3.0 s (no compile step) | 32.7 s (warm2; warm1 37.8) |

openinfer has no compilation cache to warm up — cold ≈ warm ≈ 3 s. The fair steady-state
comparison is openinfer 3.0 s vs vLLM warm 32.7 s ≈ 11×; cold-vs-cold is 23×.

## Caveats

- `vllm bench serve` is the unified client; both engines see identical request streams (same
  seed → same prompts and the same Poisson arrival schedule).
- QPS levels past the knee (12, 16) measure overload behavior, not steady state: `req/s < QPS`
  means the arrival window stretched and TTFT includes queue time.
- The README chart (`qwen3-4b-5090-perf.png`) plots the QPS-throughput and warm-TTFT columns
  above; regenerate it from these tables if the numbers change.
- Raw result logs (per engine × QPS level, the TTFT sweeps, and `vllm_warmstart_warm{1,2}.log`)
  live on the 5090 host; the tables above are transcribed from them.
