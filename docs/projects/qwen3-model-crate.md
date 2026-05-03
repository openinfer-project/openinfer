# Qwen3-4B Model Crate

**Created**: 2026-05-03
**Status**: ready for diff review
**TL;DR**: `crates/pegainfer-qwen3-4b` now owns Qwen3 config, weights, execution, scheduler, tests, benches, and kernel plan. Root `pegainfer` loads Qwen3 through a generic `EngineHandle` and no longer contains `Qwen3Model`, `Qwen3Executor`, `ModelRuntimeConfig`, root Qwen3 tests, or `src/model/qwen3/*`. 5090 release build, workspace test-target compile, clippy, Qwen3 crate e2e, and root `bench_serving snapshot` pass. Qwen3 Criterion benches use the production `runtime::Qwen3Executor` phase API; the old `ModelForward` path has been removed; decode length-limit now emits the final token before `Finished`.

## Preparation

- **Read**:
  - `docs/index.md` - identified the kernels/core crate split and per-model boundary docs.
  - `docs/projects/core-entry-crate.md` - `pegainfer-core` now owns shared runtime/API pieces and exists so model crates do not depend back on root.
  - `docs/projects/qwen3-kernels-crate.md` - Qwen3 kernel source/build ownership and human kernel index already live in `pegainfer-kernels`; model-owned DAG metadata should live with the model crate.
  - `docs/projects/pegainfer-kernels-boundary.md` - records the per-model engine direction and says root should be reusable frontend/control-plane infrastructure, not a universal model abstraction.
  - `src/main.rs`, `src/lib.rs`, `src/server_engine.rs`, `src/scheduler.rs`, `src/model_executor.rs`, `src/model/qwen3/*`, `src/bin/bench_serving.rs`, and Qwen3 tests - mapped what root currently knows about Qwen3.
- **Relevant history**:
  - `docs/projects/model-forward-trait.md` and `docs/projects/runtime-complexity-paydown.md` were useful simplifications, but the next boundary should not make `ModelForward` the long-term universal engine API.
  - `docs/projects/core-entry-crate.md` intentionally kept root compatibility re-exports only as a transition step before the Qwen3 crate split.
- **Plan**:
  1. Define the model crate/root interface before moving code.
  2. Move the generic text-generation handle/request/event types into `pegainfer-core` so root and model crates can communicate without model crates depending on root.
  3. Create `crates/pegainfer-qwen3-4b` and move Qwen3 config, weights, forward paths, decode buffers, `Qwen3Executor`, Qwen3 scheduler internals, Qwen3 correctness tests, and Qwen3-specific benches into it.
  4. Keep root `pegainfer` as frontend plus model registry. The registry can know crate names, but `main`, `vllm_frontend`, and generic benchmark code should only see `EngineHandle`, `ModelInfo`, and tokenizer path.
  5. Add a model-owned `kernel_plan.rs` in the Qwen3 crate as the LLM/human index from model DAG phases to reusable kernels. Do not add a hand-maintained public TOML in `pegainfer-kernels`.
  6. Verify locally with format/metadata, then on 5090 with release build, clippy, Qwen3 crate e2e, and root `bench_serving snapshot`. Keep microbench timing in Criterion benches instead of duplicating it as a test.
- **Risks / open questions**:
  - If the scheduler stays in root, root still knows Qwen3's execution shape. To meet the stated goal, the Qwen3 scheduler should move into the Qwen3 crate and expose only a generic handle.
  - `bench_serving` previously had a direct `ModelForward` path for Qwen3 and a scheduler path for Qwen3.5. It needed to become generic over `EngineHandle`, while Qwen3 crate-local benches should use the model executor phase API.
  - Qwen3.5 remains in root for this phase. The registry may temporarily wrap root-local Qwen3.5, but new Qwen3 code should not depend on that temporary shape.

## Interface Proposal

The root-visible interface should be request/response oriented, not prefill/decode oriented.

```rust
// pegainfer-core
pub struct EngineLoadOptions {
    pub enable_cuda_graph: bool,
    pub device_ordinals: Vec<usize>,
    pub seed: u64,
}

pub struct ModelInfo {
    pub id: &'static str,
    pub display_name: String,
    pub max_model_len: Option<u32>,
}

pub struct GenerateRequest {
    pub prompt_tokens: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub token_tx: tokio::sync::mpsc::UnboundedSender<TokenEvent>,
    pub logprobs: usize,
    pub echo: bool,
}

pub enum TokenEvent {
    Token { id: u32, logprob: Option<TokenLogprob> },
    PromptTokens { ids: Vec<u32>, logprobs: Vec<Option<TokenLogprob>> },
    Finished { finish_reason: FinishReason, prompt_tokens: usize, completion_tokens: usize },
}

#[derive(Clone)]
pub struct EngineHandle {
    submit_tx: tokio::sync::mpsc::UnboundedSender<GenerateRequest>,
}
```

```rust
// pegainfer-qwen3-4b
pub fn probe_model(model_path: &std::path::Path) -> anyhow::Result<Option<ModelInfo>>;
pub fn start_engine(
    model_path: &std::path::Path,
    options: EngineLoadOptions,
) -> anyhow::Result<EngineHandle>;
pub fn kernel_plan() -> &'static KernelPlan;
```

`Qwen3Model`, `BatchDecodeBuffers`, and `KvState` should not be root-facing APIs. The deliberate low-level escape hatch is `pegainfer_qwen3_4b::runtime`, which exposes `Qwen3Executor` plus prefill/decode/unified plan types. That is the production phase boundary used by the scheduler and by model-local benches; root should still use `start_engine`.

## Execution Log

### Step 1: Add generic engine API to core
- Added `pegainfer_core::engine` with:
  - `EngineLoadOptions`
  - `ModelInfo`
  - `TokenLogprob`
  - `FinishReason`
  - `GenerateRequest`
  - `TokenEvent`
  - `EngineHandle`
- Root `server_engine` now re-exports `FinishReason` and `TokenLogprob` for compatibility.
- Root `scheduler.rs` is reduced to compatibility re-exports for `SchedulerHandle`, `SchedulerRequest`, and `TokenEvent`.

### Step 2: Extract Qwen3 crate
- Added `crates/pegainfer-qwen3-4b`.
- Moved Qwen3-owned code into the crate:
  - config/weights/forward/prefill/decode/unified forward
  - batch decode buffers
  - `Qwen3Executor`
  - Qwen3 scheduler internals
  - Qwen3 e2e and paged-attention correctness tests
  - Qwen3 regression data generator
  - Qwen3 prefill Criterion bench
- Added `kernel_plan.rs` as the model-owned kernel routing index. It is typed Rust metadata, not a hand-maintained public TOML.

### Step 3: Remove root Qwen3 execution knowledge
- Root no longer has:
  - `src/model/qwen3.rs`
  - `src/model/qwen3/*`
  - `src/model_executor.rs`
  - Qwen3 root tests: `tests/e2e.rs`, `tests/paged_attention.rs`, `tests/bench_prefill.rs`
- Root `main.rs` starts Qwen3 through `pegainfer_qwen3_4b::start_engine(...)`.
- Root `vllm_frontend.rs` accepts a generic `EngineHandle`.
- Root `bench_serving` uses the same generic scheduler bench path for Qwen3 instead of constructing `Qwen3Model` directly.
- Checked root with `rg` and confirmed no hits for `Qwen3Model`, `Qwen3Executor`, `ModelRuntimeConfig`, `model_executor`, `src/model/qwen3`, or stale "Qwen3 continuous" comments under root source/tests/benches/README.

### Step 4: Link and validation fixes
- Added explicit `stdc++` link output in `pegainfer-kernels` build script. Once Qwen3 became an independent crate with its own tests, the FlashInfer C++ CUDA objects needed the C++ runtime linked for test binaries as well as root binaries.
- Fixed the Qwen3 crate prefill test to respect `PEGAINFER_TEST_MODEL_PATH`.
- The isolated 5090 build directory still has no `.git`, so `bench_serving snapshot` writes `commit: unknown`; after pulling it back with `rsync -e 'ssh -S none'`, the local snapshot commit field was set to current local `HEAD` short hash `0f54a1d`.

### Step 5: Verification
- Local:
  - `cargo fmt --all --check` passes.
  - `cargo metadata --no-deps --format-version 1` passes.
- 5090:
  - `PEGAINFER_CUDA_SM=120 cargo clippy --release --all-targets -- -D warnings` passes.
  - `PEGAINFER_CUDA_SM=120 cargo build --release` passes.
  - `PEGAINFER_CUDA_SM=120 cargo test --release --workspace --no-run` passes.
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test e2e -- --nocapture` passes.
  - `RUST_LOG=warn PEGAINFER_CUDA_SM=120 cargo run --release --bin bench_serving -- --model-path /data/Qwen3-4B snapshot` passes:
    - `prefill_heavy (10000,1)`: TTFT p50 `500.90ms`, p99 `503.30ms`
    - `decode_heavy (1024,256)`: TPOT p50 `7.57ms`, p99 `7.74ms`
    - This run exposed a scheduler length-limit bug: `max_tokens=256` emitted only `255` token events because the limit path finished without emitting the final decoded token. It was fixed in Step 7.
- Snapshot pulled back to `bench_snapshots/rtx-5090/qwen3-4b.json`.

### Step 6: Bench Boundary Cleanup
- Removed the duplicate Qwen3 `tests/bench_prefill.rs`; performance timing belongs under Criterion benches, while tests keep correctness/e2e coverage.
- Rejected a bench-only support API and also rejected using `ModelForward` as the benchmark entry.
- Added an explicit `runtime` module that re-exports the scheduler's real `Qwen3Executor` phase API: `PrefillPlan`, `DecodePlan`, `UnifiedPlan`, request items, and result types.
- Removed top-level public `Qwen3Model`, `ModelRuntimeConfig`, and `Qwen3State` re-exports. External low-level tools must opt into `runtime`; root continues to use `start_engine`.
- Replaced `crates/pegainfer-qwen3-4b/benches/qwen3_prefill.rs` with `benches/qwen3_runtime.rs`. It measures executor prefill TTFT over `128`, `512`, `1024`, `2048`, `4096`, and `10000` token prompts, plus executor decode TPOT for batch sizes `1`, `2`, `4`, `8`, `16`, and `32` at a `1024` token context.
- Updated `tests/paged_attention.rs` to use the same executor phase API: prefill once to create KV state, then decode through `execute_decode`.
- Verification after the cleanup:
  - Local `cargo fmt --all --check` and `cargo metadata --no-deps --format-version 1` pass.
  - Local `cargo check --release -p pegainfer-qwen3-4b --benches --tests` cannot run on the Mac without CUDA/nvcc; with `PEGAINFER_CUDA_SM=120` it still fails at local `nvcc`.
  - 5090 `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-qwen3-4b --benches --tests` passes.
  - 5090 `PEGAINFER_CUDA_SM=120 cargo clippy --release --all-targets -- -D warnings` passes.
  - 5090 `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test paged_attention -- --nocapture` passes.
  - 5090 full Criterion bench passes with `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/Qwen3-4B cargo bench -p pegainfer-qwen3-4b --bench qwen3_runtime`:
    - Prefill TTFT: `128 -> 11.804ms`, `512 -> 23.200ms`, `1024 -> 44.114ms`, `2048 -> 87.327ms`, `4096 -> 179.60ms`, `10000 -> 505.55ms`.
    - Decode one-step batch time at 1024-token context: `bs1 -> 9.3095ms`, `bs2 -> 9.3207ms`, `bs4 -> 9.4059ms`, `bs8 -> 10.960ms`, `bs16 -> 11.718ms`, `bs32 -> 13.196ms`.

### Step 7: Retire ModelForward and Fix Length Limit
- Deleted `pegainfer_core::model::{ModelForward, GenerationState}` and removed the root `src/model.rs` re-export.
- Deleted the Qwen3 `forward.rs` compatibility path. Qwen3 tests that used it now build their baselines from `batch_prefill(bs=1)` plus `batch_decode(bs=1)`, so they exercise the same phase APIs as production.
- Fixed Qwen3 decode length-limit handling by adding `DecodeEffect::EmitAndFinish`. EOS behavior is unchanged: EOS finishes without emitting the stop token. Length limit now emits the sampled final token, then sends `Finished { finish_reason: Length }`.
- Regenerated `test_data/Qwen3-4B.json` because every length-limited golden output now includes the final requested token.
- Re-ran `bench_serving snapshot` on 5090 and pulled back `bench_snapshots/rtx-5090/qwen3-4b.json`; `decode_heavy (1024,256)` now records `generated_tokens min=max=avg=256`.
- Performance stayed within noise on RTX 5090:
  - `prefill_heavy (10000,1)`: TTFT p50 `501.69ms`, p99 `503.16ms`.
  - `decode_heavy (1024,256)`: TPOT p50 `7.56ms`, p99 `7.73ms`.
- Final verification after this step:
  - Local `cargo fmt --all --check`, `cargo metadata --no-deps --format-version 1`, and `git diff --check` pass.
  - 5090 `PEGAINFER_CUDA_SM=120 cargo clippy --release --all-targets -- -D warnings` passes.
  - 5090 `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test e2e -- --nocapture` passes.

## Debrief

The Qwen3 split now enforces the intended dependency direction: model execution code depends on `pegainfer-core` and `pegainfer-kernels`; root depends on the model crate only at registry/startup glue points. Root still has a `ModelType::Qwen3` enum and default Qwen3 model path because the product needs a loader choice, but it no longer sees Qwen3 layers, KV state, TP rank workers, or prefill/decode/unified plans.

Next cleanup should be a generic model registry module so `main.rs` and `bench_serving.rs` stop matching model crate names directly. Qwen3.5 should then get the same crate treatment.
