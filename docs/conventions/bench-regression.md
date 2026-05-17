# Benchmark Regression Tracking

> **TL;DR:** One JSON snapshot per model **per GPU** in `bench_snapshots/{gpu-slug}/`, git history is the timeline. `snapshot` generates (auto-detects GPU), `compare` diffs against git. Thresholds: TPOT p50 >2%, TTFT p50 >3%.
>
> **Status:** Active.

## Concept

Each model has one snapshot file per GPU (`bench_snapshots/{gpu-slug}/{model}.json`), always the latest. The GPU slug is derived automatically from `nvidia-smi` (e.g. `NVIDIA GeForce RTX 5070 Ti` → `rtx-5070-ti`). Git is the history — `git log -p bench_snapshots/` is the timeline. Both `snapshot` and `compare` run inference in-process, no server needed.

Regressions are only meaningful when comparing the **same GPU** across commits. Cross-GPU differences are expected (e.g. RTX 4090 decode is 10–16% faster than RTX 5070 Ti).

> **Migration note (PR #66):** Snapshots moved from `bench_snapshots/{model}.json` to `bench_snapshots/{gpu-slug}/{model}.json`. Since `compare` uses `git show {ref}:{path}`, baselines from before PR #66 are not reachable under the new path. After merging, run `snapshot` once to establish a fresh baseline for each GPU.

## Standard Profiles

| Name | prompt_len | output_len | Key metric |
|------|-----------|------------|------------|
| prefill_heavy | model-dependent | 1 | TTFT |
| decode_heavy | 1024 | 256 | TPOT (steady, excluding first decode step) |

Prefill prompt length is model-dependent: Qwen3 uses 10000 tokens, Qwen3.5 uses 4000 (HD256 attention needs ~4x working memory vs HD128, OOMs at 10k on 16 GB GPUs). `compare` checks shape consistency within the same model — if you change the constants, it will error until you re-baseline.

`prefill_heavy` with `output_len=1` produces no steady decode steps: `steady_tpot_ms` is `null` in the JSON. This is expected.

## Workflow

### Bootstrapping (first time)

```bash
cargo run -r --bin bench_serving -- --model-path models/Qwen3-4B snapshot --warmup 5 --iters 20
git add bench_snapshots/rtx-5070-ti/qwen3-4b.json  # path auto-detected from GPU
git commit -m "chore: establish benchmark baseline for Qwen3-4B"
```

### Before merging a PR

```bash
# 1. Generate snapshot (writes to bench_snapshots/{gpu-slug}/qwen3-4b.json)
cargo run -r --bin bench_serving -- --model-path models/Qwen3-4B snapshot --warmup 5 --iters 20

# 2. Compare against last committed version (exits non-zero if no baseline)
cargo run -r --bin bench_serving -- compare bench_snapshots/rtx-5070-ti/qwen3-4b.json --baseline HEAD

# 3. If clean, commit with the PR
git add bench_snapshots/rtx-5070-ti/qwen3-4b.json
```

Qwen3.5-4B:
```bash
cargo run -r --bin bench_serving -- --model-path models/Qwen3.5-4B snapshot --warmup 5 --iters 20
cargo run -r --bin bench_serving -- compare bench_snapshots/rtx-5070-ti/qwen3.5-4b.json --baseline HEAD
```

Compare against older ref:
```bash
cargo run -r --bin bench_serving -- compare bench_snapshots/rtx-5070-ti/qwen3-4b.json --baseline HEAD~5
cargo run -r --bin bench_serving -- compare bench_snapshots/rtx-5070-ti/qwen3-4b.json --baseline main
```

## Regression Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| TPOT p50 | >2% | Decode is the hot path; measurement noise ~1% at iters=20 |
| TTFT p50 | >3% | Prefill has higher variance from kernel launch jitter |

Thresholds trigger on **p50 only**. The comparison table also shows p99 for manual inspection. A threshold firing means "investigate", not "reject" — run twice if it barely fires, thermal variance accounts for 1–2%.

## Investigating a Regression

1. Which metric regressed? Check the `compare` output.
2. TPOT: likely decode kernels, CUDA graph, MLP/GEMV. Profile with `nsys` using the decode-heavy shape (see [profiling-guide](../playbooks/profiling-guide.md)).
3. TTFT: likely prefill, cuBLAS, or Triton kernels. Profile with the prefill-heavy shape.
4. Both: suspect a fundamental change (memory layout, kernel launch, data flow).

## Snapshot JSON Schema

Filename: `bench_snapshots/{gpu-slug}/{model}.json`. GPU slug is derived from `nvidia-smi` output (strip `NVIDIA GeForce ` prefix, lowercase, spaces to dashes). Model name is the directory basename, lowercased (`models/Qwen3.5-4B` → `qwen3.5-4b.json`).

```json
// bench_snapshots/rtx-5070-ti/qwen3-4b.json
{
  "commit": "117a963",
  "date": "2026-03-30",
  "model": "Qwen3-4B",
  "gpu": "NVIDIA GeForce RTX 5070 Ti",
  "prefill_heavy": {
    "prompt_len": 10000,
    "output_len": 1,
    "metrics": {
      "ttft_ms":              { "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0, "samples": 20 },
      "first_decode_step_ms": null,
      "steady_tpot_ms":       null,
      "e2e_ms":               { "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0, "samples": 20 },
      "generated_tokens":     { "min": 1, "max": 1, "avg": 1.0, "samples": 20 },
      "request_tok_s":        0.0,
      "decode_tok_s":         null
    }
  },
  "decode_heavy": {
    "prompt_len": 1024,
    "output_len": 256,
    "metrics": {
      "ttft_ms":              { "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0, "samples": 20 },
      "first_decode_step_ms": { "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0, "samples": 20 },
      "steady_tpot_ms":       { "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0, "samples": 20 },
      "e2e_ms":               { "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0, "samples": 20 },
      "generated_tokens":     { "min": 256, "max": 256, "avg": 256.0, "samples": 20 },
      "request_tok_s":        0.0,
      "decode_tok_s":         0.0
    }
  }
}
```
