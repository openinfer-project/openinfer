# DFlash Speculative Decoding (Qwen3.5-4B)

> **TL;DR:** Qwen3.5-4B DFlash speculative decoding is implemented behind `--dflash-draft-model-path`, default-off, greedy-only, single-GPU only, and now supports multi-active decode batches with a fixed-buffer batched verifier. Same-host RTX 5090 A/B on `output_len=256` shows clear throughput wins at c4/c8/c16: decode-heavy `prompt_len=1` improves `+15.4%/+19.8%/+12.7%`, medium `prompt_len=1024` improves `+236.6%/+192.9%/+46.0%`, and long `prompt_len=4096` improves `+188.1%/+41.3%/+24.9%`.

Last touched: 2026-07

## How To Enable

Use a Qwen3.5 target model with a matching DFlash draft checkpoint:

```bash
cargo run --release -p openinfer-server --features qwen35-4b -- \
  --model-path <Qwen3.5-4B> \
  --dflash-draft-model-path <Qwen3.5-4B-DFlash>
```

The flag is rejected for unsupported model lines. Qwen3.5 DFlash is incompatible with LoRA, KV offload, tensor parallelism, and decode-overlap modes. Non-greedy requests and logprobs use normal decode.

## Runtime Contract

- The drafter emits `[current_token, draft...]`; the target verifies that span and commits the longest greedy-matching prefix plus one bonus token.
- Verification uses preallocated `VerifyBuffers35` storage for token ids, hidden/logit buffers, GDR scratch, full-attention scratch, paged prefill plans, and sampling scratch. Decode steps reuse fixed buffers instead of allocating on the hot path.
- Qwen3.5 verification is a hybrid transaction over full-attention KV, recurrent state, convolution state, and sequence length. Verify writes to scratch state; commit preserves full-span accepts directly and replays only truncated accepted spans after rolling KV back to the canonical boundary.
- Batched verify handles active batches up to the scheduler bucket size. Complete fixed shapes can use captured graph-compatible paths; truncated or heterogeneous spans use eager verify.
- The scheduler captures target hidden context only on DFlash-eligible prefill paths. If a request falls back to normal decode, its DFlash state is dropped because normal decode does not capture the hidden context needed by the drafter.
- Per-request low-acceptance statistics disable DFlash after enough poor draft tokens, so incompatible prompts return to baseline decode.
- DFlash reserves memory for draft weights, draft KV/cache, verify buffers, and batch scratch before target KV sizing. Admission also reserves draft block headroom, so a near-window request accepted without DFlash can be rejected when DFlash is enabled.

## Validation

All commands below passed on an RTX 5090 validation host with driver `580.105.08`, CUDA 13.3, Nsight Systems 2026.1.3, Triton Python, and `OPENINFER_CUDA_SM=120`. The source snapshot was `8cd46cb`.

```bash
cargo fmt --all --check
git diff --check
OPENINFER_TRITON_PYTHON=<triton-python> OPENINFER_TEST_MODEL_PATH=<Qwen3.5-4B> \
  OPENINFER_DFLASH_TEST_MODEL_PATH=<Qwen3.5-4B-DFlash> \
  cargo test --release -p openinfer-qwen35-4b --features qwen35-4b \
  --test dflash_speculative_gate -- --nocapture --test-threads=1
OPENINFER_TRITON_PYTHON=<triton-python> OPENINFER_TEST_MODEL_PATH=<Qwen3.5-4B> \
  OPENINFER_DFLASH_TEST_MODEL_PATH=<Qwen3.5-4B-DFlash> \
  cargo test --release -p openinfer-qwen35-4b --features qwen35-4b \
  --test speculative_verify -- --nocapture --test-threads=1
OPENINFER_TRITON_PYTHON=<triton-python> OPENINFER_TEST_MODEL_PATH=<Qwen3.5-4B> \
  cargo test --release -p openinfer-qwen35-4b --features qwen35-4b \
  --test hf_golden_gate -- --nocapture
OPENINFER_TRITON_PYTHON=<triton-python> OPENINFER_TEST_MODEL_PATH=<Qwen3.5-4B> \
  cargo test --release -p openinfer-qwen35-4b --features qwen35-4b \
  --test e2e_scheduler -- --nocapture
```

The DFlash scheduler gate checks single request, multi-active batch, heterogeneous `max_tokens`, and mixed concurrent requests against plain greedy decode. The run reported exact generated-token parity for all checked requests, including `48/48`, `32/32`, `40/40`, and `24/24` token spans.

## Benchmark

Same host, same source snapshot (`8cd46cb`), in-process `bench_serving request`, greedy synthetic distinct prompts, `output_len=256`, warmup `3`, iters `8`.

| Prompt | Concurrency | Baseline tok/s | DFlash tok/s | Delta | Baseline effective TPOT p50 | DFlash effective TPOT p50 | Baseline raw ITL p99 | DFlash raw ITL p99 | Baseline TTFT p50 | DFlash TTFT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 152.225 | 150.792 | -0.94% | 6.569 ms | 6.632 ms | 6.642 ms | 6.663 ms | 9.126 ms | 9.223 ms |
| 1 | 4 | 112.028 | 129.303 | +15.42% | 8.905 ms | 8.523 ms | 8.986 ms | 23.210 ms | 39.734 ms | 33.406 ms |
| 1 | 8 | 92.073 | 110.319 | +19.82% | 10.836 ms | 8.996 ms | 10.908 ms | 34.054 ms | 71.238 ms | 64.958 ms |
| 1 | 16 | 66.781 | 75.236 | +12.66% | 14.936 ms | 14.351 ms | 15.117 ms | 62.012 ms | 135.142 ms | 128.442 ms |
| 1024 | 1 | 138.754 | 138.032 | -0.52% | 7.207 ms | 7.244 ms | 7.281 ms | 7.275 ms | 46.991 ms | 46.906 ms |
| 1024 | 4 | 102.211 | 344.014 | +236.57% | 9.875 ms | 3.015 ms | 9.693 ms | 19.854 ms | 154.986 ms | 138.428 ms |
| 1024 | 8 | 82.904 | 242.829 | +192.90% | 12.165 ms | 4.125 ms | 54.483 ms | 28.213 ms | 266.918 ms | 231.595 ms |
| 1024 | 16 | 59.390 | 86.722 | +46.02% | 16.978 ms | 11.717 ms | 60.756 ms | 55.041 ms | 497.441 ms | 417.980 ms |
| 4096 | 1 | 110.688 | 109.782 | -0.82% | 9.035 ms | 9.101 ms | 9.110 ms | 9.113 ms | 191.241 ms | 191.210 ms |
| 4096 | 4 | 80.516 | 231.923 | +188.05% | 12.792 ms | 4.676 ms | 57.662 ms | 23.121 ms | 644.821 ms | 573.805 ms |
| 4096 | 8 | 63.302 | 89.473 | +41.34% | 16.204 ms | 11.663 ms | 60.615 ms | 37.011 ms | 1113.506 ms | 951.100 ms |
| 4096 | 16 | 44.292 | 55.315 | +24.89% | 23.181 ms | 17.808 ms | 65.180 ms | 66.158 ms | 2078.919 ms | 1708.289 ms |

`effective_tpot_ms` is the amortized per-request decode time. Raw token-event ITL can spike under speculative decode because accepted spans emit multiple tokens in one scheduler step; keep both metrics visible when reviewing tails.

## Profile

Profiles used `nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node` and `nsys stats` on the same host. The final c8/c16 traces show that the previous per-request verifier bottleneck is gone: DFlash uses batched prefill verify kernels and partial-only replay instead of singleton target-prefill verification.

| Shape | Baseline dominant work | DFlash dominant work | Profile conclusion |
| --- | --- | --- | --- |
| `prompt=1,c=8` | `gated_delta_rule_decode_kernel` `2.04s`, batch decode attention `72.6ms` | GDR verify kernels plus lower target decode counts; batch decode attention `71.2ms` | Draft/verify overhead is below the throughput saved by multi-token accepts. |
| `prompt=1024,c=8` | `gated_delta_rule_decode_kernel` `2.06s`, batch decode attention `550.2ms` | GDR verify kernels, `SinglePrefillWithKVCacheKernel` `75.3ms`, batch prefill verify `49.6ms`, batch decode attention `71.8ms` | Verifier no longer runs target prefill per request; c8 decode throughput improves `+192.90%`. |
| `prompt=4096,c=16` | `gated_delta_rule_decode_kernel` `4.41s`, batch decode attention `2.44s`, batch prefill `398.5ms` | batch decode attention `1.68s`, batch prefill verify `537.9ms`, GDR verify kernels visible but not dominant | Commit/replay/copy is not the leading bottleneck; long c16 still improves `+24.89%`. |

## Claim Boundaries

- This is an opt-in Qwen3.5 DFlash path with real c4/c8/c16 in-process benchmark wins and exact token-hash sanity gates.
- HTTP startup and mixed request smokes passed with DFlash enabled, but the performance table is in-process benchmark evidence.
- Single-concurrency random synthetic prompts remain flat to slightly slower. The multi-active path is the supported performance claim for this slice.
- Multi-GPU, LoRA, KV offload, decode overlap, non-greedy sampling, and logprobs intentionally use normal decode or fail closed.
