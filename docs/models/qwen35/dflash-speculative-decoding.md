# DFlash Speculative Decoding (Qwen3.5-4B)

> **TL;DR:** Qwen3.5-4B DFlash speculative decoding is implemented behind `--dflash-draft-model-path`, default-off, greedy-only, single-GPU only, and now supports multi-active decode batches with a fixed-buffer batched verifier. Same-host RTX 5090 A/B on `output_len=256` shows clear throughput wins at c4/c8/c16: decode-heavy `prompt_len=1` improves `+16.7%/+15.4%/+14.0%`, medium `prompt_len=1024` improves `+209.9%/+168.6%/+45.3%`, and long `prompt_len=4096` improves `+135.9%/+35.7%/+25.6%`.

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

Commands below passed on an RTX 5090 validation host with driver `580.105.08`, CUDA 13.3, Triton Python `3.7.1`, and `OPENINFER_CUDA_SM=120`. The source snapshot is the PR branch after the benchmark-shaped gate cleanup.

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

The DFlash scheduler gates check single request, multi-active batch, heterogeneous `max_tokens`, mixed concurrent requests, and the benchmark-shaped synthetic cases that exposed hash differences in the raw sweep (`1024/c16`, `4096/c8`, `4096/c16`). The benchmark-shaped follow-up passed: `1024/c16` was exact for 16/16 requests; `4096/c8` and `4096/c16` were exact except for near-ties accepted by the regret oracle (`regret 0.000` and `0.125 <= 0.20`).

## Benchmark

Same host, same PR branch snapshot, in-process `bench_serving request`, greedy synthetic distinct prompts, `output_len=256`, warmup `3`, iters `8`.

| Prompt | Concurrency | Baseline tok/s | DFlash tok/s | Delta | Baseline effective TPOT p50 | DFlash effective TPOT p50 | Baseline raw ITL p99 | DFlash raw ITL p99 | Baseline TTFT p50 | DFlash TTFT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 151.214 | 149.808 | -0.9% | 6.593 ms | 6.645 ms | 6.682 ms | 6.699 ms | 9.122 ms | 9.374 ms |
| 1 | 4 | 110.906 | 129.388 | +16.7% | 8.907 ms | 8.682 ms | 8.988 ms | 21.412 ms | 39.045 ms | 32.756 ms |
| 1 | 8 | 89.977 | 103.856 | +15.4% | 10.889 ms | 9.679 ms | 10.969 ms | 33.776 ms | 69.832 ms | 63.528 ms |
| 1 | 16 | 64.930 | 73.990 | +14.0% | 14.925 ms | 13.851 ms | 15.054 ms | 57.610 ms | 131.570 ms | 125.421 ms |
| 1024 | 1 | 135.543 | 134.699 | -0.6% | 7.220 ms | 7.270 ms | 7.295 ms | 7.297 ms | 46.715 ms | 46.797 ms |
| 1024 | 4 | 97.293 | 301.482 | +209.9% | 9.911 ms | 2.980 ms | 9.695 ms | 19.231 ms | 153.577 ms | 137.400 ms |
| 1024 | 8 | 76.916 | 206.606 | +168.6% | 12.217 ms | 4.062 ms | 54.032 ms | 27.162 ms | 263.906 ms | 229.206 ms |
| 1024 | 16 | 53.404 | 77.602 | +45.3% | 16.963 ms | 11.597 ms | 60.353 ms | 52.408 ms | 492.746 ms | 414.550 ms |
| 4096 | 1 | 102.477 | 101.745 | -0.7% | 9.039 ms | 9.122 ms | 9.139 ms | 9.134 ms | 189.535 ms | 189.955 ms |
| 4096 | 4 | 68.916 | 162.581 | +135.9% | 12.830 ms | 4.635 ms | 57.351 ms | 22.665 ms | 640.507 ms | 567.550 ms |
| 4096 | 8 | 50.473 | 68.502 | +35.7% | 16.238 ms | 11.677 ms | 61.075 ms | 35.653 ms | 1106.224 ms | 941.275 ms |
| 4096 | 16 | 32.875 | 41.304 | +25.6% | 23.239 ms | 17.710 ms | 65.017 ms | 63.604 ms | 2070.313 ms | 1696.621 ms |

`effective_tpot_ms` is the amortized per-request decode time. Raw token-event ITL can spike under speculative decode because accepted spans emit multiple tokens in one scheduler step; keep both metrics visible when reviewing tails.

## Profile

Profiles used `nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node` and `nsys stats` on the same host. The final c8/c16 traces show that the previous per-request verifier bottleneck is gone: DFlash uses batched prefill verify kernels and partial-only replay instead of singleton target-prefill verification.

| Shape | Baseline dominant work | DFlash dominant work | Profile conclusion |
| --- | --- | --- | --- |
| `prompt=1,c=8` | `gated_delta_rule_decode_kernel` `2.04s`, batch decode attention `72.6ms` | GDR verify kernels plus lower target decode counts; batch decode attention `71.2ms` | Draft/verify overhead is below the throughput saved by multi-token accepts. |
| `prompt=1024,c=8` | `gated_delta_rule_decode_kernel` `2.06s`, batch decode attention `550.2ms` | GDR verify kernels, `SinglePrefillWithKVCacheKernel` `75.3ms`, batch prefill verify `49.6ms`, batch decode attention `71.8ms` | Verifier no longer runs target prefill per request; c8 decode throughput improves `+192.90%`. |
| `prompt=4096,c=16` | `gated_delta_rule_decode_kernel` `4.41s`, batch decode attention `2.44s`, batch prefill `398.5ms` | batch decode attention `1.68s`, batch prefill verify `537.9ms`, GDR verify kernels visible but not dominant | Commit/replay/copy is not the leading bottleneck; long c16 still improves `+24.89%`. |

## Claim Boundaries

- This is an opt-in Qwen3.5 DFlash path with real c4/c8/c16 in-process benchmark wins. Token sanity is exact where stable; prompt-length-1 and a few long high-concurrency synthetic cases are covered by the same regret oracle used by the scheduler gate for bf16 near-tie / prefill-vs-decode boundary flips.
- The performance table is in-process benchmark evidence. Do not read it as an HTTP serving pressure claim.
- Single-concurrency random synthetic prompts remain flat to slightly slower. The multi-active path is the supported performance claim for this slice.
- Multi-GPU, LoRA, KV offload, decode overlap, non-greedy sampling, and logprobs intentionally use normal decode or fail closed.

## Remaining Risks And Follow-ups

- No blocker-level implementation risk is known from the current local, GPU, benchmark, and profile evidence. Keep CI state and new reviewer comments as the final merge gate because they can change after the local evidence snapshot.
- Single-request `c1` runs are flat to slightly slower (`-0.6%` to `-0.9%` in the benchmark table). DFlash should be described as a multi-active throughput path, not as an all-shape latency win.
- Raw token-event ITL p99 can increase under short-prompt speculative decode because accepted spans emit multiple tokens in one scheduler step. Keep raw ITL visible next to effective TPOT and output throughput when reviewing tail latency.
- The benchmark table is in-process serving evidence. A production-style HTTP pressure sweep remains useful before making broader OpenAI-compatible serving claims.
- A few synthetic high-concurrency shapes are validated by the regret oracle or prefill-vs-decode boundary check instead of exact raw hash equality. This is covered by scheduler gates, but reviewer discussion should keep the oracle boundary explicit.
- DFlash reserves extra memory for draft weights, draft KV/cache, verify buffers, and batch scratch. Near-window or near-memory requests may be admitted by baseline decode and rejected with DFlash enabled.
- Unsupported modes remain intentional scope boundaries: tensor parallelism, LoRA, KV offload, decode overlap, non-greedy sampling, and logprobs use normal decode or fail closed.
