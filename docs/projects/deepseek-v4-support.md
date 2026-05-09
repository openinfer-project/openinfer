# DeepSeek V4 Support

**Created**: 2026-05-07
**Status**: Draft PR open - initial DeepSeek V4 support is wired through the native engine, TileLang build-time kernels, exact E2E, and OpenAI-compatible HTTP paths. Prompt prefill now reuses the direct runtime's request-scope RoPE caches instead of rebuilding them per layer/rank. Bring-up Python probes were removed from the PR and archived locally outside the repository; deeper profiling work is deferred.

## Scope

This document is the single project record for the initial DeepSeek V4 PR. It replaces the earlier MP8 bring-up notes and the separate TileLang kernel notes.

The PR scope is:

- add `pegainfer-deepseek-v4` as the model crate for the DeepSeek V4 Flash MP8 checkpoint;
- wire DeepSeek V4 into `pegainfer-server` model detection and `bench_serving`;
- build official-style DeepSeek V4 TileLang kernels at compile time;
- keep runtime Python-free;
- provide exact text, operator, and HTTP service validation;
- keep official-model comparison probes out of the initial PR;
- leave deeper NCCL/TileLang GEMM and final-logits optimization for follow-up work.

## Build Requirements

DeepSeek V4 currently requires the `deepseek-v4` Cargo feature and TileLang at build time. Default Qwen builds do not run the DeepSeek TileLang generator.

The kernels build script probes:

- `PEGAINFER_TILELANG_PYTHON`, if set;
- `../.venv/bin/python`;
- `.venv/bin/python`;
- `python3`;
- `python`.

The interpreter must import both `tilelang` and the CUDA/Triton stack used by the generator. The validated environment used:

- `tilelang 0.1.9`
- `triton 3.6.0`

Minimal setup:

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install "tilelang==0.1.9"
export PEGAINFER_TILELANG_PYTHON=.venv/bin/python
```

The generated CUDA is linked into `pegainfer-kernels` when the feature is enabled; Python is not needed at runtime.

## Implementation Summary

### Model Crate

`pegainfer-deepseek-v4` owns:

- config parsing for DeepSeek V4 MP8;
- per-rank weight manifests and GPU loading;
- runtime ops for block prefill/decode, HC, sparse attention, routing, compressor state, and final logits;
- direct `EngineHandle` integration used by server and tests;
- exact E2E tests driven by `test_data/deepseek-v4-ground-truth.json`, with `PEGAINFER_DEEPSEEK_GT_PATH` available for regenerations.

The direct engine seeds decode cache from prompt prefill instead of replaying prompt tokens through decode. This made exact validation practical enough for PR use.

### TileLang Kernels

`pegainfer-kernels/tools/tilelang/deepseek_v4/generate.py` generates CUDA sources for official-style DeepSeek V4 kernels:

- `act_quant_kernel`
- `fp8_gemm_kernel`
- `fp4_gemm_kernel`
- `fp4_quant_kernel`
- `sparse_attn_kernel`
- `hc_split_sinkhorn_kernel`

TileLang is used as a CUDA source generator rather than a C ABI generator because its generic wrapper cannot marshal E8M0 scale tensors. The generated source is wrapped by stable C ABI launchers with unique symbols.

### CUDA Layout

DeepSeek CUDA glue is split by subsystem:

| File | Role |
| --- | --- |
| `csrc/deepseek_v4/deepseek_quant.cu` | FP8/FP4 quantized linear dispatch and TileLang linear wrappers |
| `csrc/deepseek_v4/deepseek_attention.cu` | head norm, RoPE, sparse/indexed attention, BF16/F32 conversion |
| `csrc/deepseek_v4/deepseek_indexer.cu` | indexer scoring/top-k and Hadamard plus FP4 quant |
| `csrc/deepseek_v4/deepseek_compressor.cu` | hidden RoPE, compressor prefill/decode, concat, BF16 linear |
| `csrc/deepseek_v4/deepseek_hc.cu` | HC expand/mixes/sinkhorn/post and final logits |
| `csrc/deepseek_v4/deepseek_moe.cu` | routing, SwiGLU, and expert accumulation |
| `csrc/deepseek_v4/deepseek_common.cuh` | shared device helpers |

The split keeps each DeepSeek CUDA translation unit under about one thousand lines.

## Validation

### Exact Text

All 20 ground-truth cases pass exact text validation as four 5-case slices with max new tokens 64:

| Offset | Case request times |
| --- | --- |
| 0 | `1.48s`, `1.24s`, `696.83ms`, `733.97ms`, `4.38s` |
| 5 | `1.62s`, `2.18s`, `2.51s`, `2.82s`, `4.37s` |
| 10 | `3.35s`, `1.19s`, `3.31s`, `963.60ms`, `1.05s` |
| 15 | `1.70s`, `1.01s`, `746.25ms`, `3.17s`, `812.70ms` |

Command shape:

```bash
PEGAINFER_DEEPSEEK_GT_OFFSET=<offset> \
PEGAINFER_DEEPSEEK_GT_LIMIT=5 \
PEGAINFER_DEEPSEEK_GT_MAX_NEW_TOKENS=64 \
PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V4-Flash \
cargo test --release -p pegainfer-deepseek-v4 --features deepseek-v4 --test e2e -- --nocapture --exact test_e2e_deepseek_v4_generation
```

### Operator Guards

The full DeepSeek V4 `mp8_manifest` release test passes:

```bash
PEGAINFER_TEST_MODEL_PATH=/data/DeepSeek-V4-Flash \
PEGAINFER_NVCC_JOBS=8 \
cargo test --release -p pegainfer-deepseek-v4 --features deepseek-v4 --test mp8_manifest -- --nocapture
```

Result: `23 passed`, `0 failed`.

Coverage includes MP8 layout accessors, RoPE formula checks, TileLang FP8/FP4 linear guards, Hadamard FP4 quant guard, NCCL pair reduction, rank-local GPU load, and representative layer/block GPU execution paths.

### HTTP Service

With `--features deepseek-v4`, `pegainfer-server` detects `model_type="deepseek_v4"` and starts DeepSeek V4 with eight devices and CUDA graph disabled.

The initial service path is intentionally greedy-only. Requests that ask for sampling or logprobs are rejected before generation and surfaced through `stop_reason` instead of being silently coerced to greedy. This is a temporary compatibility choice in the vLLM frontend path; a later API cleanup should reject unsupported DeepSeek V4 request parameters during request validation instead of representing them as a completed generation.

Server command used for HTTP validation:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --features deepseek-v4 -- \
  --model-path /data/DeepSeek-V4-Flash --port 18080
```

Verified `/v1/chat/completions`:

- case 0 exact content: `346`
- case 4 exact content: `2024 年是闰年，2024 年 2 月有 29 天。`

Representative chat request:

```bash
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/DeepSeek-V4-Flash","messages":[{"role":"user","content":"请只输出最终答案，不要解释。2024年2月有多少天？"}],"temperature":0,"max_tokens":64}'
```

Verified `/v1/completions` with the raw official prompt string:

- prompt shape: `<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜></think>`
- case 4 exact content: `2024 年是闰年，2024 年 2 月有 29 天。`
- `prompt_tokens=32`, `completion_tokens=21`

Representative completions request:

```bash
curl -s http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/DeepSeek-V4-Flash","prompt":"<｜begin▁of▁sentence｜><｜User｜>请只输出最终答案，不要解释。2024年2月有多少天？<｜Assistant｜></think>","temperature":0,"max_tokens":64}'
```

Unsupported non-greedy/logprobs requests are terminated before generation with a visible `stop_reason` instead of being silently coerced to greedy. Validation command shape:

```bash
curl -s http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/DeepSeek-V4-Flash","prompt":"<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>","temperature":0.7,"max_tokens":8}'
```

Verified response status is `200`, `completion_tokens=0`, `finish_reason="stop"`, and `stop_reason` contains the greedy-only rejection message.

The temporary validation server on port `18080` was stopped after validation.

### Performance Sanity

The major validation bottlenecks were fixed with nsys-guided changes:

- prompt prefill now seeds decode cache;
- prompt prefill reuses direct-runtime RoPE caches instead of rebuilding per request/layer/rank;
- decode RoPE tables are cached at request scope instead of rebuilt for every token/layer/rank;
- compressor decode projection was parallelized;
- compressor prefill weighting was parallelized;
- MoE gate, HC mixes, and compressor BF16 linear reuse per-device scratch/handles.

The 2026-05-09 prefill RoPE reuse pass changed `prefill_logits_and_decode_cache_group_bf16_hidden` to require caller-provided RoPE caches. The direct engine passes `runtime.ropes`, so cache-seeding prompt prefill no longer calls `precompute_rope_cache` inside the per-layer loop. Local release format/check passed, and exact E2E validation was verified after the change with a low single-digit percent speedup.

The follow-up optimization pass removes two redundant stream synchronizations after pure same-stream kernels (`hc_expand_bf16_hidden` and `bf16_linear_bf16_hidden`), removes the direct decode per-layer host synchronization, reuses thread-local per-device scratch buffers for `all_reduce_hidden_group_fp32`, and generates fixed attention top-k index tensors on GPU instead of building host `Vec<i32>` buffers and synchronizing after H2D copies. Local release format/check passed, and E2E plus speed validation showed a substantial improvement. MoE route-index D2H synchronization remains the next higher-risk target because it changes CPU expert-activation control flow.

Earlier exact-request profiling removed the large synchronous `cudaMalloc/cudaFree` cliff. The current decode-heavy profile shows the remaining allocation issue is many small async allocations and frees inside decode, alongside kernel launch count. The next GPU buckets are NCCL all-reduce and TileLang FP4/FP8 GEMM.

The current synthetic decode-heavy baseline on 5090-dev is:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- \
  --model-path /data/DeepSeek-V4-Flash --format json \
  request --prompt-len 1 --output-len 32 --warmup 1 --iters 1
```

Latest result after request-scope RoPE caching:

- model load: `34.99s`;
- request e2e: `5.79s`;
- steady TPOT: `181.74ms/token`.

Two nsys decode traces were captured with `--cuda-graph-trace=node --delay=36 --duration=15`:

| Trace | steady TPOT under nsys | Kernel/API headline |
| --- | --- | --- |
| `target/profiling/dsv4_decode_1x32.sqlite` | `265.70ms/token` | `43,312` f32 NCCL all-reduce kernels, `1,081,612` `cudaLaunchKernel`, `810,784` `cuMemAllocAsync` |
| `target/profiling/dsv4_decode_1x32_rope_cached.sqlite` | `257.69ms/token` | `41,496` f32 NCCL all-reduce kernels, `1,021,396` `cudaLaunchKernel`, `727,954` `cuMemAllocAsync` |

The remaining TPOT problem is structural: decode still launches hundreds of thousands of tiny kernels and allocates/frees scratch in hot loops. The biggest buckets are f32 NCCL all-reduce, TileLang FP4/FP8 GEMMs, and repeated temporary allocations. The next performance pass should prioritize per-rank scratch reuse for decode intermediates and `all_reduce_hidden_group_fp32`, then reassess whether all-reduce count or TileLang GEMM efficiency is the dominant limiter.

## Workspace Isolation

DeepSeek V4 is a workspace member, but its DeepSeek-specific bins, integration tests, and `pegainfer-kernels/deepseek-v4` dependency are gated behind the `deepseek-v4` feature. This keeps default Qwen-oriented workspace checks from requiring TileLang.

Verified:

- `PEGAINFER_NVCC_JOBS=8 cargo check --release --workspace` passed with DeepSeek TileLang disabled in `pegainfer-kernels`.
- `PEGAINFER_NVCC_JOBS=8 cargo test --release --workspace --lib` reached existing Qwen3 lib tests and failed because the default local `models/Qwen3-4B` path is absent; it did not trigger DeepSeek TileLang generation.

## Known Follow-ups

These are intentionally out of the initial PR scope:

- reintroduce official-model comparison probes only if they have a maintained owner, provenance notes, and a clear diagnostic workflow;
- add strict native-vs-official `attn.wkv` fixture coverage;
- add arbitrary-value per-shape TileLang FP8/FP4 parity tests beyond the current power-of-two guards;
- profile final logits only if nsys shows it matters;
- profile NCCL all-reduce and TileLang FP4 GEMM after the initial PR lands;
- narrow the current public diagnostic surface in `pegainfer-deepseek-v4` after bring-up bins/tests are either retired or moved behind a dedicated test-helper boundary;
- move unsupported DeepSeek V4 request handling from generation-time `stop_reason` compatibility into frontend request validation;
- add an explicit non-panicking shutdown path for NCCL communicator teardown.

## PR Gate

Before opening the PR, keep the required gate focused:

- `cargo fmt --check -p pegainfer-deepseek-v4`
- `cargo check --release -p pegainfer-server`
- `PEGAINFER_NVCC_JOBS=8 cargo check --release -p pegainfer-server --features deepseek-v4`
- `PEGAINFER_TEST_MODEL_PATH=/data/DeepSeek-V4-Flash PEGAINFER_NVCC_JOBS=8 cargo test --release -p pegainfer-deepseek-v4 --features deepseek-v4 --test mp8_manifest -- --nocapture`
- four exact E2E slices over `test_data/deepseek-v4-ground-truth.json`, using:

```bash
PEGAINFER_TEST_MODEL_PATH=/data/DeepSeek-V4-Flash \
PEGAINFER_DEEPSEEK_GT_OFFSET=<0|5|10|15> \
PEGAINFER_DEEPSEEK_GT_LIMIT=5 \
PEGAINFER_DEEPSEEK_GT_MAX_NEW_TOKENS=64 \
PEGAINFER_NVCC_JOBS=8 \
cargo test --release -p pegainfer-deepseek-v4 --features deepseek-v4 --test e2e -- --nocapture --exact test_e2e_deepseek_v4_generation
```

- one `/v1/completions` or `/v1/chat/completions` validation through `pegainfer-server`, plus one unsupported-parameter request checking `stop_reason`.

Broader workspace checks are valuable, but failures outside the DeepSeek V4 diff should be separated from this initial support PR.
