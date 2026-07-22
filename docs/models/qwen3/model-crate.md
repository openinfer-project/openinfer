# Qwen3 Crate Layout

> **TL;DR:** `openinfer-qwen3` owns Qwen3 model policy and execution. It composes generic engine contracts from `openinfer-engine`, runtime primitives from `openinfer-core`, KV management from `openinfer-kv-cache`, and reusable GPU operations from `openinfer-kernels`. `openinfer-server` selects and launches the model through `EngineHandle`; model-specific scheduler, executor, weights, and tests stay in the Qwen3 crate.
>
> **Last touched:** 2026-07

## Ownership boundaries

| Path | Owns | Does not own |
| --- | --- | --- |
| `openinfer-qwen3/` | Qwen3 config and weights, launch policy, scheduler, executor, prefill/decode/unified DAGs, LoRA/speculative paths, model tests, kernel benchmark manifest and reports | HTTP serving, reusable kernel implementations |
| `openinfer-kernels/` | Reusable Rust kernel wrappers, CUDA/Triton sources, FFI, tensor operation implementations | Qwen3 scheduling and request policy |
| `openinfer-engine/` | Generic engine handle, requests/events, sampling contracts, parallel configuration | Model-specific execution |
| `openinfer-core/` | Shared runtime helpers, tensor/device context, CUDA Graph and tracing support | Qwen3 weights or scheduler decisions |
| `openinfer-kv-cache/` | Paged KV allocation and request KV state | Admission policy |
| `openinfer-server/` | CLI, model detection/selection, frontend integration | Qwen3 GPU execution details |

The physical directory is the crate boundary. Historical `crates/openinfer-qwen3` and `crates/openinfer-kernels` paths no longer exist.

## Request path

1. `openinfer-server` maps CLI options into `Qwen3LaunchOptions` and calls `openinfer_qwen3::launch`.
2. `launch` validates model-specific option combinations and creates the Qwen3 scheduler/executor.
3. The scheduler owns request admission and batching. It produces prefill, decode, or unified plans.
4. `Qwen3Executor` owns GPU execution and per-request KV state. Tensor-parallel execution fans each step out to rank-local workers.
5. Qwen3 DAG call sites route each phase to operations exported by `openinfer-kernels`; `kernel_manifests/qwen3.toml` and the model report binaries describe benchmark and reporting coverage.
6. Results return through the generic `EngineHandle` event contract, so the frontend does not depend on Qwen3 internals.

## Where to make changes

- Change Qwen3 weight layout, batching, KV policy, or forward composition in `openinfer-qwen3/`.
- Change a reusable CUDA/Triton operator or its Rust wrapper in `openinfer-kernels/`; update Qwen3 call sites only when the operator contract changes.
- Change generic request/event or parallel configuration contracts in `openinfer-engine/` and audit every model crate.
- Change OpenAI-compatible HTTP behavior or model selection in `openinfer-server/`.
- Add Qwen3 integration coverage under `openinfer-qwen3/tests/`; keep policy-only unit tests beside their owner modules.

Do not add a second root-level Qwen3 execution path. Low-level tools that intentionally bypass serving should use `openinfer_qwen3::runtime`, while normal callers use `launch` and `EngineHandle`.

## Current verification commands

Run the affected-crate gates from the repository root:

```bash
cargo fmt --all --check
cargo clippy --release -p openinfer-engine -p openinfer-qwen3 --all-targets -- -D warnings
cargo test --release -p openinfer-engine --lib
cargo test --release -p openinfer-qwen3 --lib
```

GPU/model integration tests are narrower and require their documented fixtures. For example, `openinfer-qwen3/tests/tp_concurrent_decode.rs` requires a Qwen3 checkpoint and at least two visible CUDA devices; it self-skips when either is unavailable.

## Change discipline

- Keep model policy near the model even if another model currently has similar code.
- Extract shared code only after its contract is genuinely model-independent.
- Keep performance records out of this layout document; use the Qwen3 performance documents and benchmark snapshots.
- When paths or ownership change, update this document and `docs/index.md` in the same change.
