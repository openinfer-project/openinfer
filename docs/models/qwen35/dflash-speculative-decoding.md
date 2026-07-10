# Qwen3.5 DFlash Speculative Decoding

> **TL;DR:** Qwen3.5 DFlash is an opt-in, single-active greedy path behind `--dflash-draft-model-path`. On RTX 5090, the verified path improves direct output throughput by 2.08x at prompt 64 / output 128 and 2.40x at prompt 1024 / output 256, with matching output hashes. Multi-active and unsupported request shapes use normal decode.

Last touched: 2026-07

## Design

Qwen3.5 can load a DFlash draft model beside the target model. The default path is unchanged when no draft path is provided.

Speculative verification is a transaction over all Qwen3.5 decode state:

- paged full-attention KV;
- linear-attention recurrent state;
- convolution state and sequence length.

Verification writes recurrent and convolution state into scratch buffers. A fully accepted span keeps the verified KV and copies the verified recurrent state into the live slot. A partial acceptance truncates KV, restores the backed-up recurrent state, and replays only the accepted span. Partial replay disables timing-selected cuBLASLt algorithms because those algorithms caused rare greedy hash drift on this shape; normal decode and batched verification keep the tuned cuBLASLt path.

## Enable

```bash
OPENINFER_TRITON_PYTHON=<triton-python> \
cargo run --release --features qwen35-4b -- \
  --model-path <Qwen3.5-4B> \
  --dflash-draft-model-path <Qwen3.5-DFlash>
```

The speculative path requires:

- one active request;
- greedy sampling with `logprobs=0`;
- captured DFlash prompt context;
- a complete verify span within the 2048-token validated context;
- at least two output tokens remaining.

Multi-active, non-greedy, logprobs, missing-context, and longer-context requests continue through normal target decode. Once normal decode takes ownership, the captured draft state is discarded so a request cannot later resume speculation with stale context. LoRA, KV offload, tensor parallel, and decode overlap are rejected when DFlash is enabled.

## Validation

The GPU-only tests are ignored by default, so `--ignored` is required:

```bash
OPENINFER_TRITON_PYTHON=<triton-python> \
OPENINFER_TEST_MODEL_PATH=<Qwen3.5-4B> \
cargo test --release -p openinfer-qwen35-4b --features qwen35-4b \
  --test speculative_verify -- --ignored --nocapture --test-threads=1

OPENINFER_TRITON_PYTHON=<triton-python> \
OPENINFER_TEST_MODEL_PATH=<Qwen3.5-4B> \
OPENINFER_DFLASH_TEST_MODEL_PATH=<Qwen3.5-DFlash> \
cargo test --release -p openinfer-qwen35-4b --features qwen35-4b \
  --test dflash_speculative_gate -- --ignored --nocapture --test-threads=1
```

RTX 5090 results:

- `speculative_verify`: 6 passed;
- `dflash_speculative_gate`: 3 passed;
- `e2e_scheduler`: 1 passed;
- pinned short and long `hf_golden_gate`: 2 passed;
- independent-process stability: eager 20/20 and CUDA Graph 20/20, all with output hash `5a71ac0dfe1cd1e5`.

## Direct Benchmark

Same GPU, model revision, source tree, CUDA Graph setting, and benchmark client. The prompt 64 / output 128 row is the median of three alternating-order A/B runs with warmup 5 and 20 measured iterations per run.

Environment: RTX 5090, driver 580.76.05, CUDA 12.8, Rust 1.96.1, Triton 3.7.1, target revision `851bf6e806ef`, and `z-lab/Qwen3.5-4B-DFlash` config SHA-256 prefix `6fa9ca0d10d2`.

| Shape | Path | Baseline output tok/s | DFlash flag output tok/s | Delta | Output |
| --- | --- | ---: | ---: | ---: | --- |
| prompt 64 / output 128 / c1 | speculative | 158.33 | 329.90 | +108.36% | hash match |
| prompt 1024 / output 256 / c1 | speculative | 136.85 | 327.90 | +139.60% | hash match |
| prompt 1 / output 256 / c1 | normal fallback | 159.22 | 158.57 | -0.40% | hash match |
| prompt 4096 / output 256 / c1 | normal fallback | 101.14 | 100.69 | -0.44% | hash match |

For prompt 64 / output 128, median end-to-end latency falls from `808.48 ms` to `388.03 ms`; TTFT remains flat at `11.68 ms` versus `11.28 ms`, and steady TPOT falls from `6.27 ms` to `2.86 ms`.

The single-request reservation costs 1,817 MB of fixed GPU memory on this host. Target KV capacity remains 31,098 pages versus 34,188 without DFlash, retaining about 91% of the baseline page pool. The earlier pool-scaled draft reservation retained only 10,438 pages and was removed.

Nsight Systems on the same shape attributes the gain to less target work:

- total GPU kernel time: about `790 ms` to `323 ms`;
- device-to-host API time: `810 ms` to `249 ms`;
- partial-commit GemmEx replay: `6.58 ms`, about `2%` of DFlash GPU kernel time.

The partial replay path is not the leading GPU or tail bottleneck. The remaining DFlash GPU time is dominated by the expected batched target-verification GEMMs.

## Claim Boundary

These are direct single-active results, not HTTP serving or vLLM parity results. c4/c8/c16 currently use normal decode fallback, so this implementation does not claim concurrent speculative throughput improvement.
