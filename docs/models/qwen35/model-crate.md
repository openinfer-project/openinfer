# Qwen3.5-4B Model Crate

**Created**: 2026-05-05
**Status**: complete
**TL;DR**: `crates/pegainfer-qwen35-4b` now owns Qwen3.5 config, weights, prefill/decode/unified forward, recurrent state, scheduler, recurrent op wrappers, e2e tests, regen test, and Qwen3.5 op benches. Root `pegainfer` loads Qwen3.5 through `pegainfer_qwen35_4b::start_engine(...)` / generic `EngineHandle`; root no longer exposes `pegainfer::model::Qwen35Model` or `pegainfer::scheduler_qwen35`. Build/check/clippy, root `bench_serving` smoke, Qwen3.5 e2e, and Qwen3.5 scheduler e2e pass.

## Preparation

- **Read**:
  - `docs/index.md` - identified the existing core split, Qwen3 model crate split, and Qwen3.5 accuracy/optimization docs.
  - `docs/models/qwen3/model-crate.md` - Qwen3 already owns its scheduler, executor/runtime API, tests, benches, and root-facing `EngineHandle` entry.
  - `docs/models/qwen35/accuracy.md` - Qwen3.5 e2e tests are regression guards against `test_data/Qwen3.5-4B.json`; parity tooling mentioned there is partly absent in the current tree.
  - `docs/models/qwen35/optimization.md` - Qwen3.5 should keep its hybrid linear/full-attention scheduler/state architecture.
  - GitHub issue #79 - acceptance criteria require `crates/pegainfer-qwen35-4b`, removal of root `pegainfer::model::Qwen35Model` and `pegainfer::scheduler_qwen35`, generic root `bench_serving`, and CUDA validation.
  - `Cargo.toml`, `src/lib.rs`, `src/main.rs`, `src/ops.rs`, `src/scheduler.rs`, `src/model/qwen35.rs`, and `crates/pegainfer-qwen3-4b/src/lib.rs` - mapped the current root Qwen3.5 surface and the Qwen3 crate interface to copy.
- **Relevant history**:
  - `docs/models/qwen3/model-crate.md` - root should load model crates through `EngineHandle`; model-owned execution details should move behind crate-local modules.
- **Plan**:
  1. Add `crates/pegainfer-qwen35-4b` to the workspace with dependencies mirroring the Qwen3 crate plus the root dependencies Qwen3.5 currently uses.
  2. Move `src/model/qwen35.rs`, `src/model/qwen35/*`, `src/scheduler_qwen35.rs`, and Qwen3.5 recurrent op wrappers into the new crate, keeping CUDA/Triton kernel sources and FFI in `pegainfer-kernels`.
  3. Rewrite imports so the new crate depends on `pegainfer-core` and `pegainfer-kernels`, not on root `pegainfer`.
  4. Expose `probe_model`, `start_engine`, and a deliberate `runtime` module from `pegainfer-qwen35-4b`.
  5. Update root `main.rs` and `src/bin/bench_serving.rs` to call `pegainfer_qwen35_4b::start_engine`.
  6. Move Qwen3.5 e2e tests and regen test into the model crate; adjust model/test-data paths after the move.
  7. Remove root Qwen3.5 modules and compatibility exports, then audit root with `rg`.
  8. Verify with `cargo fmt --all --check`, `cargo metadata --no-deps --format-version 1`, and the CUDA-capable build/test commands available on this machine.
- **Risks / open questions**:
  - Some root operator tests cover Qwen3.5 recurrent wrappers; they may need to move with the wrappers or be split so root no longer imports model-specific scratch types.
  - Accuracy docs reference historical `qwen35_dump_*` and `tools/accuracy/*` files that are not present in the current tree; this migration can document the current test locations but cannot move absent tools.

## Execution Log

### Step 1: Add model crate and move Qwen3.5 runtime
- Added `crates/pegainfer-qwen35-4b` to the workspace and root dependencies.
- Moved Qwen3.5-owned runtime files out of root:
  - `src/model/qwen35.rs`
  - `src/model/qwen35/*`
  - `src/scheduler_qwen35.rs`
  - `src/ops/recurrent.rs`
- The new crate exposes:
  - `probe_model(model_path) -> Result<Option<ModelInfo>>`
  - `start_engine(model_path, EngineLoadOptions) -> Result<EngineHandle>`
  - `start_engine_with_capacity(...)` for root benchmark capacity control
  - `runtime::{Qwen35Model, start_with_capacity, start_with_model, MAX_BATCH}` for model-local tests/debugging
  - `runtime_ops` for Qwen3.5-local operator benches.

### Step 2: Move tests and benches
- Moved root Qwen3.5 tests to the model crate:
  - `crates/pegainfer-qwen35-4b/tests/e2e.rs`
  - `crates/pegainfer-qwen35-4b/tests/e2e_scheduler.rs`
  - `crates/pegainfer-qwen35-4b/tests/regen_test_data.rs`
- Moved Qwen3.5-specific op benches to `crates/pegainfer-qwen35-4b/benches/qwen35_ops.rs`.
- Moved the `conv1d_prefill_handoff_matches_single_prefill` operator test into `crates/pegainfer-qwen35-4b/src/recurrent.rs`, next to the wrapper it validates.
- Removed Qwen3.5-specific GEMV shapes from the root generic `ops_bench`; the model-specific benches now live with Qwen3.5.

### Step 3: Remove root Qwen3.5 compatibility surface
- Removed root exports/modules:
  - `pub mod model`
  - `pub mod scheduler_qwen35`
  - `src/model.rs`
  - `src/ffi.rs`
  - `src/kv_pool.rs`
- Root `main.rs` now calls `pegainfer_qwen35_4b::start_engine(...)` for Qwen3.5.
- Root `bench_serving` now calls `pegainfer_qwen35_4b::start_engine_with_capacity(...)` and still benchmarks via generic `EngineHandle`.
- The Qwen3.5 engine entry honors a single `EngineLoadOptions.device_ordinals` value and rejects multi-device input, matching the current single-GPU implementation instead of silently ignoring the option.
- `rg` confirms there are no root references to `pegainfer::model::Qwen35Model`, `pegainfer::scheduler_qwen35`, or `src/model/qwen35`.

### Step 4: Validation
- Passed:
  - `cargo metadata --no-deps --format-version 1`
  - `cargo fmt --all --check`
  - `PEGAINFER_CUDA_SM=120 cargo check --release --workspace --all-targets`
  - `PEGAINFER_CUDA_SM=120 cargo clippy --release --workspace --all-targets -- -D warnings`
  - `PEGAINFER_CUDA_SM=120 cargo build --release`
  - `PEGAINFER_CUDA_SM=120 cargo test --release -p pegainfer-qwen35-4b recurrent::tests::conv1d_prefill_handoff_matches_single_prefill -- --nocapture`
  - `PEGAINFER_CUDA_SM=120 cargo run --release --bin bench_serving -- --model-path /data/code/workspace-rustllm/pegainfer/models/Qwen3.5-4B request --prompt-len 1 --output-len 1 --warmup 0 --iters 1`
- Initial Qwen3.5 e2e failure:
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/code/workspace-rustllm/pegainfer/models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35-4b --test e2e -- --nocapture`
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/code/workspace-rustllm/pegainfer/models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35-4b --test e2e_scheduler -- --nocapture`
  - Both initially produced all-case gibberish-output mismatches.
- Control run:
  - A temporary old-HEAD worktree at `/tmp/pegainfer-head` ran `PEGAINFER_CUDA_SM=120 PEGAINFER_TRITON_PYTHON=/data/code/workspace-rustllm/pegainfer/.venv/bin/python PEGAINFER_TEST_MODEL_PATH=/data/code/workspace-rustllm/pegainfer/models/Qwen3.5-4B CARGO_TARGET_DIR=/tmp/pegainfer-head-target cargo test --release --test e2e_qwen35 -- --nocapture`.
  - Old HEAD failed the same way on all 10 Qwen3.5 cases, so the e2e mismatch predated this crate split.
- Follow-up fix:
  - `docs/models/qwen35/e2e-gibberish.md` identified the first gibberish commit as `6a5b826`, fixed Qwen3.5 scheduler thread CUDA/cuBLAS binding, kept greedy sampling on FlashInfer top1, and refreshed the exact Qwen3.5 golden for the default engine shape.
  - After that fix, both Qwen3.5 e2e commands above pass.

## Debrief

- **Outcome**: Qwen3.5 is now an independent model crate with the same root-facing engine style as Qwen3-4B. Root retains model detection/frontend/bench orchestration, but not Qwen3.5 model internals. The follow-up e2e corruption fix restored Qwen3.5 e2e and scheduler e2e.
- **Pitfalls encountered**:
  - The first e2e run used a relative `PEGAINFER_TEST_MODEL_PATH`; package tests execute with a crate-oriented working directory, so absolute model paths are safer for crate-local tests.
  - Qwen3.5 e2e initially looked like a crate-split regression, but git history showed the corruption started earlier when cuBLAS handles became thread-local without equivalent Qwen3.5 scheduler thread binding.
  - Moving recurrent wrappers out of root exposed stale root compatibility re-exports (`src/ffi.rs`, `src/kv_pool.rs`, and root Qwen3.5 ops bench shapes), which were removed.
- **Lessons learned**:
  - Model-local benches need a deliberate public surface. `runtime_ops` is intentionally narrow and only exposes the Qwen3.5 operator wrappers needed by Qwen3.5 benches.
  - Qwen3.5 test docs should use absolute `PEGAINFER_TEST_MODEL_PATH` examples when run from the workspace, because package test working directories can make relative paths misleading.
