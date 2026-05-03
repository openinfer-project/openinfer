# Core Entry Crate

**Created**: 2026-05-03
**Status**: ready for diff review
**TL;DR**: `pegainfer-core` now holds the shared runtime/API entry for future model crates. Root `pegainfer` remains the product entrypoint and vLLM frontend host. Trace is intentionally not part of the active public entry; the old file reporter stays as archived source only. Full 5090 release build, clippy, Qwen3 e2e, and `bench_serving snapshot` pass.

## Preparation

- **Read**:
  - `docs/index.md` - identified the Phase 1 kernels split and per-model boundary docs.
  - `docs/projects/qwen3-kernels-crate.md` - Phase 1 follow-up says the Qwen3 model crate should sit on top of `pegainfer-kernels`, with machine model metadata owned by the model crate.
  - `docs/projects/pegainfer-kernels-boundary.md` - establishes the target shape: root/frontend plus shared runtime/data-plane layers plus per-model engines.
  - `src/lib.rs`, `src/main.rs`, `src/model.rs`, `src/vllm_frontend.rs`, `src/model_executor.rs`, and `src/model/qwen3/*` - mapped the current root-owned runtime APIs that would otherwise force future model crates to depend back on root.
- **Relevant history**:
  - `docs/projects/model-forward-trait.md` and `docs/projects/runtime-complexity-paydown.md` tried to share model execution too broadly. This split should share runtime contracts and leave model DAGs model-owned.
  - `docs/projects/qwen3-kernels-crate.md` showed that moving kernels without moving runtime APIs still leaves model extraction blocked by root dependencies.
- **Plan**:
  1. Add `crates/pegainfer-core` to the workspace.
  2. Move shared runtime entry APIs into it: sampling params, tensor/FFI re-exports, paged KV/page pool, weight loading, CUDA graph state, simple KV cache, shared op adapters, and direct-model traits.
  3. Keep root `pegainfer` source paths as compatibility re-exports so existing tests, benches, Qwen3, and Qwen3.5 code continue to compile.
  4. Remove trace from active CLI/lib surface. Keep `src/trace_reporter.rs` as archived source, but do not export it from `lib.rs`.
  5. Verify locally with format/metadata, then on 5090 with release build/test/e2e/full benchmark snapshot.
- **Risks / open questions**:
  - Visibility must become public in `pegainfer-core` where future model crates need the APIs. This broadens the API surface, but it is the point of the core crate.
  - Root compatibility re-exports should be temporary; once `pegainfer-qwen3` exists, root can depend on model crates directly instead of carrying old paths.

## Execution Log

### Step 1: Extract core runtime API
- Added `crates/pegainfer-core`.
- Moved shared runtime files from root into the core crate:
  - `sampler`
  - `tensor` / `ffi` re-exports
  - `page_pool`
  - `kv_pool`
  - `weight_loader`
  - `cuda_graph`
  - `kv_cache`
  - shared `ops` adapters except Qwen3.5 recurrent kernels
  - `ModelForward` / `GenerationState`
- Replaced root modules with compatibility re-exports where needed.

### Step 2: Remove trace from active public entry
- Removed `--trace-output-path` from the CLI.
- Removed `trace_reporter` from `src/lib.rs`.
- Kept `src/trace_reporter.rs` as archived source, not compiled or exported.
- Kept `fastrace` dependency for now because existing Qwen3 prefill code still has `#[fastrace::trace]` annotations.

### Step 3: Local verification
- `cargo fmt --all --check` passes.
- `cargo metadata --no-deps --format-version 1` passes.

### Step 4: 5090 verification
- Used the isolated build directory from `docs/resources/5090.md`: `/root/develop/xingming/pegainfer-kernels-crate-build`.
- `PEGAINFER_CUDA_SM=120 cargo build --release` passes.
- `PEGAINFER_CUDA_SM=120 cargo clippy --release --all-targets -- -D warnings` passes.
- `PEGAINFER_CUDA_SM=120 cargo build --release && PEGAINFER_CUDA_SM=120 cargo test --release --no-run` passes.
- `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/Qwen3-4B cargo test --release --test e2e -- --nocapture` passes.
- `RUST_LOG=warn PEGAINFER_CUDA_SM=120 cargo run --release --bin bench_serving -- --model-path /data/Qwen3-4B snapshot` passes:
  - `prefill_heavy (10000,1)`: TTFT p50 `502.60ms`, p99 `503.94ms`
  - `decode_heavy (1024,256)`: TPOT p50 `7.39ms`, p99 `7.45ms`
- Snapshot pulled back to `bench_snapshots/rtx-5090/qwen3-4b.json`.

## Debrief

The core split is intentionally conservative: it moves shared runtime/data-plane APIs out of the root crate while preserving root compatibility re-exports for the current Qwen3 and Qwen3.5 code. That gives the next Qwen3-4B model crate a public entry to depend on without forcing it back through the product binary/frontend crate.

Next step: review this diff, then split the Qwen3 model crate on top of `pegainfer-core` and `pegainfer-kernels`.
