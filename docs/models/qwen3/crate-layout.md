# Qwen3-4B Crate Layout

> **TL;DR:** `pegainfer-qwen3-4b` owns Qwen3 config, weights, executor, scheduler, LoRA, tests, and the kernel routing plan; the kernel surface (CUDA/Triton/FlashInfer source, build, FFI, reusable ops) lives in `pegainfer-kernels`; the server sees only `start_engine() -> EngineHandle`. Replaces the 2026-05 `model-crate.md`/`kernels-crate.md` extraction records — bring-up history lives in git, this doc describes what exists.
>
> **Last touched:** 2026-06

## Crate boundary

Dependency direction: `pegainfer-qwen3-4b` → `pegainfer-core` + `pegainfer-kernels` + `pegainfer-kv-cache`. The server (`pegainfer-server`) depends on the model crate only at registry/startup glue; it never sees `Qwen3Model`, KV state, TP rank workers, or prefill/decode plans.

| What | Where |
| --- | --- |
| Config / weights / RoPE cache | `pegainfer-qwen3-4b/src/{config,weights}.rs` |
| Executor (single-GPU + TP rank workers, CUDA graphs, split-K gate) | `src/executor.rs`, `src/batch_decode*.rs`, `src/prefill.rs`, `src/unified_forward.rs` |
| Scheduler (admission, plan → resolve → effects) | `src/scheduler.rs` + `src/scheduler/{plan,resolve,effects}.rs` |
| LoRA load/unload/activation | `src/lora.rs` |
| Kernel routing index (model DAG phase → reusable kernel) | `src/kernel_plan.rs` (typed Rust, not a hand-maintained manifest) |
| Reusable kernel wrappers, FFI, tensor helpers | `pegainfer-kernels/src/` |
| CUDA source, Triton AOT, FlashInfer submodule, nvcc build | `pegainfer-kernels/csrc/`, `tools/triton/`, `build.rs` |
| Human/LLM kernel routing table | `pegainfer-kernels/KERNELS.md` |
| Tests | `tests/{hf_golden_gate,prefix_cache,lora_smoke,scheduler_robustness}.rs` |
| Report binaries (feature `kernel-report`) | `src/bin/{qwen3_kernel_report,qwen3_model_report}.rs`; `qwen3_decode_context` is the fixed-context decode probe |

Build and test commands are in the repo-root `CLAUDE.md`; per-op report tooling is documented in `docs/subsystems/kernels/kernel-op-reports.md`.

## Public surface

- `start_engine(model_path, EngineLoadOptions) -> EngineHandle` — the only entry the server uses.
- `start_engine_with_lora_control(...)` — same, plus LoRA load/unload control.
- `pegainfer_qwen3_4b::runtime` — deliberate low-level escape hatch re-exporting `Qwen3Executor` and the prefill/decode/unified plan types. It is the production phase boundary used by the scheduler and model-local tools; the server must not use it.
- `kernel_plan()` — model-owned index from DAG phases to reusable kernels.

There is no Criterion bench target in this crate (`autobenches = false`). The old `qwen3_runtime`, `qwen3_attention`, and `qwen3_kernel_snapshot` benches were retired; kernel measurement goes through the `kernel-report` binaries instead.

## Decode split-K gate (load-bearing perf facts)

Low-batch long-context decode under-fills the GPU on the non-partition FlashInfer paged decode path (grid is `(batch, num_kv_heads)`, so `bs=1` launches 8 CTAs scanning the whole KV context — ~7% of peak DRAM bandwidth, CUPTI-verified). The runtime therefore gates FlashInfer split-K decode:

- split-K when `padded_bs <= 2 && max_seq_len >= 1024`, otherwise non-partition;
- tuned to `SPLIT_KV_CHUNK_TOKENS=256`, `SPLIT_KV_MAX_CHUNKS_PER_REQUEST=64` (cold-L2 CUPTI sweep, 2026-05, RTX 5090);
- CUDA graph cache is keyed by `(batch_bucket, attention_path)` — a request can cross the split-K threshold mid-decode and needs a separate graph capture.

Effect at the time of tuning: 4k/64 serving steady TPOT p50 `11.7ms → 6.46ms` on RTX 5090. The batch sweep is why the gate is conservative: at `kv_len=1024`, split-K only wins for `bs<=2`.

## Gotchas worth keeping

- **CUPTI Range Profiler crashes on verbose range names** (`NVPW_CUDA_Profiler_DecodeCounters` inside `libnvperf_host.so`). Use compact range names like `qk/non_partition/b1/k1024`, keep metadata in the JSON output. The first profiled launch also needs an unprofiled warmup launch or CUDA lazy init pollutes its GPU time.
- **FlashInfer C++ objects need `stdc++` linked for test binaries** — owned by `pegainfer-kernels/build.rs`; symptom is link failures only in test targets.
- **Single-layer synthetic kernel benches lie about DRAM** — the working set fits in L2 (RTX 5090: 96MiB), so event-timer "effective bandwidth" can exceed 100% of peak. Use CUPTI `dram__bytes_*` counters for utilization claims.
