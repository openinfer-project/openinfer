# Qwen3.5 Batched Step Tail

> **TL;DR:** Qwen3.5 issue #353 is implemented in the local branch: prefill gathers per-request last hidden rows, runs batched offset RMSNorm + one lm_head GEMM, and scheduler/executor decode sample from batched logits. Full-vocab host copies now happen only for `logprobs > 0`. HF logits + scheduler e2e pass; benchmark evidence supports a first-token/short-output TTFT claim only.
>
> **Last touched:** 2026-06

## Scope

- Changed Qwen3.5 only; no new CUDA kernel.
- Kept serial recurrent/full-attention prefill, but batched the final tail across requests.
- Kept request row order as input slice order for prefill and active-slot order for decode.
- Did not change `EngineHandle`, `/v1/completions`, scheduler admission, prefix cache, or chunked prefill design.

## Code Shape

- `openinfer-qwen35-4b/src/prefill.rs` returns each request's last hidden state and batches final `rms_norm_batch_offset_into` + `gemm`.
- `openinfer-qwen35-4b/src/unified_forward.rs` returns batched prefill logits and leaves decode logits in the graph buffer.
- `openinfer-qwen35-4b/src/batch_decode.rs`, `src/scheduler.rs`, and `src/executor.rs` sample from batched logits.
- `openinfer-qwen35-4b/src/logprobs.rs` is the single helper for requested-logprobs snapshots and CPU logprob formatting.
- `openinfer-qwen35-4b/tests/e2e_scheduler.rs` checks greedy `logprobs=0/1` token parity, logprob payload shape, and mixed concurrent logprob/no-logprob requests.

## Verification

```bash
cargo fmt --all --check
git diff --check
OPENINFER_CUDA_SM=120 OPENINFER_TRITON_PYTHON=<python-with-triton> \
  cargo check --release -p openinfer-qwen35-4b --features qwen35-4b --tests
OPENINFER_CUDA_SM=120 OPENINFER_TRITON_PYTHON=<python-with-triton> \
  OPENINFER_TEST_MODEL_PATH=<Qwen3.5-4B> \
  cargo test --release -p openinfer-qwen35-4b --features qwen35-4b --test hf_golden_gate -- --nocapture
OPENINFER_CUDA_SM=120 OPENINFER_TRITON_PYTHON=<python-with-triton> \
  OPENINFER_TEST_MODEL_PATH=<Qwen3.5-4B> \
  cargo test --release -p openinfer-qwen35-4b --features qwen35-4b --test e2e_scheduler -- --nocapture
```

Results:

- `cargo check`: passed.
- `hf_golden_gate`: `2 passed; 0 failed`; short, batched, slot-compaction, and 4097/8192-token long replays stayed within tolerance.
- `e2e_scheduler`: `1 passed; 0 failed`; includes the no-logprobs/logprobs parity guard.

## Benchmark

Environment: single RTX 5090 32GB, driver 580.76.05, CUDA 12.8, Rust 1.96.0, Triton 3.6.0, Qwen3.5-4B local weights.

Final short-output serving A/B:

```bash
cargo run --release --features qwen35-4b --bin bench_serving -- \
  --model-path <Qwen3.5-4B> --format json --label <label> --out <label>.json \
  request --prompt-len 1 --output-len 1 --concurrency 4 --warmup 5 --iters 20 --seed 353
```

Clean paired runs exited `0`, had 80 requests per run, and matched token hashes across main/branch:

| Metric | main avg | branch avg | delta |
| --- | ---: | ---: | ---: |
| TTFT avg | 30.531 ms | 29.221 ms | -1.310 ms (-4.29%) |
| E2E avg | 30.533 ms | 29.224 ms | -1.310 ms (-4.29%) |
| request tok/s | 32.751 | 34.219 | +4.48% |

Boundary: this supports the first-token / short-output public serving path. It is not a long-decode TPOT claim.

Long-output diagnostic:

- Clean `prompt_len=1`, `output_len=128`, `concurrency=8`, `warmup=2`, `iters=8`, `seed=353` paired runs exited `0` and matched token hashes. Steady TPOT was `9.0303 ms` on main and `9.0247 ms` on the branch (`-0.061%`).
- Higher-pressure `concurrency=64`, `warmup=2`, `iters=5` also matched token hashes and exited `0` after the scheduler join-handle fix. Steady TPOT was `9.0412 ms` on main and `9.0494 ms` on the branch (`+0.091%`), so no meaningful long-decode TPOT speedup is claimed.

## Review Notes

- Read-only DeepSeek diff review found no blocker.
- Follow-up review cleaned duplicate logprobs helpers into `src/logprobs.rs` and kept docs public-facing with placeholders rather than machine paths.
- Fail-fast logprobs snapshot errors are intentional; silent logprobs loss is worse than an explicit request error in this correctness-sensitive path.
- Final review flagged a `concurrency=64` shutdown SIGSEGV in `cublas_destroy`; qwen35 now returns an `EngineHandle` that joins the scheduler thread before process teardown.
