# Qwen3.5 E2E Gibberish

**Created**: 2026-05-05
**Status**: complete
**TL;DR**: Fixed Qwen3.5 e2e gibberish. The regression first appears at `6a5b826` after cuBLAS handles became thread-local: Qwen3.5 loaded the model on one thread but ran scheduler prefill/decode on another without rebinding the CUDA context or initializing that thread's cuBLAS handles. Qwen3.5 greedy remains on FlashInfer top1, and the default Qwen3.5 e2e remains an exact golden-text regression against `test_data/Qwen3.5-4B.json`.

## Preparation

- **Read**:
  - `docs/index.md` - Qwen3.5 accuracy and optimization docs are the relevant references.
  - `docs/models/qwen35/model-crate.md` - confirmed the model-crate split reproduced the same Qwen3.5 e2e failure on old HEAD.
  - `git log -- src/model/qwen35 src/scheduler_qwen35.rs src/ops/recurrent.rs crates/pegainfer-kernels ...` - identified Qwen3.5 and sampling-related commits since the last accuracy work.
- **Relevant history**:
  - `docs/models/qwen35/model-crate.md` - old HEAD and the model-crate split both fail all 10 Qwen3.5 e2e cases with similar gibberish.
  - Commit history has a suspicious sampling change: `020970b refactor(sampling): switch greedy decode to flashinfer top1 (#73)`.
- **Plan**:
  1. Use a temporary clean worktree so the current model-crate diff stays untouched.
  2. Copy or reuse the local FlashInfer third-party tree and set `PEGAINFER_TRITON_PYTHON`.
  3. Run the Qwen3.5 e2e at selected commits around sampling/token-id/scheduler changes.
  4. Once the first bad commit is identified, inspect the diff and run a smaller targeted test or patch.
  5. Apply the fix back on the active branch, update docs, and rerun focused validation.
- **Risks / open questions**:
  - Older commits may not build against the current local third-party/kernel tree without small environment fixes.
  - A full Qwen3.5 e2e loads the 4B model every time; the first phase will use selected commits before full bisect.

## Execution Log

### Step 1: Reproduce and bisect through history
- Created a temporary worktree at `/tmp/pegainfer-q35-debug` so the active model-crate diff stayed untouched.
- Older commits needed the current local FlashInfer third-party tree copied into `third_party/flashinfer` and `PEGAINFER_TRITON_PYTHON=/data/code/workspace-rustllm/pegainfer/.venv/bin/python`.
- Results:
  - `24be186 refactor(embedding): keep token ids unsigned end-to-end (#71)` passed Qwen3.5 e2e.
  - `020970b refactor(sampling): switch greedy decode to flashinfer top1 (#73)` failed a few cases with normal text, matching baseline drift rather than gibberish.
  - `902b725 fix(scheduler): gate prefill admission on kv budget (#74)` still produced normal text.
  - `6a5b826 feat: Add Qwen3 tensor parallel runtime (#75)` failed all cases with repeated multilingual garbage tokens.
- The first gibberish commit also changed cuBLAS handles to `thread_local`, added `cuda_set_device`, and added Qwen3 worker-thread binding, but Qwen3.5 scheduler did not receive equivalent binding.

### Step 2: Prove first-token corruption
- Added a temporary local `debug_tokens` test, then removed it after diagnosis.
- Before the fix, the first generated tokens for `What is the capital of France?` were `[207248, 207344, 83168, 165952, ...]`, decoding to fragments like `"단은"`, `" персонала"`, `"Mbps"`, and `"فيروس"`.
- That showed logits/sampling were already wrong at the first sampled token after prefill; decode KV accumulation was not the primary cause.

### Step 3: Fix scheduler thread binding
- Updated `crates/pegainfer-qwen35-4b/src/scheduler.rs` so the scheduler thread:
  - calls `cuda_set_device` for the model device,
  - binds the existing `CudaContext` to the scheduler thread,
  - initializes thread-local cuBLAS handles on that thread,
  - destroys those handles on scheduler thread exit,
  - reports startup failures back to `start_with_capacity`.
- After this change, the same diagnostic prompt produced `"\n\nThe capital of France is"` instead of garbage.

### Step 4: Keep FlashInfer top1 as the greedy selector
- With thread binding fixed, Qwen3.5 generated normal text but a few cases drifted between valid continuations.
- A direct logits check found `Tell me a story` step 15 had two maximum bf16 logits: token `198` (`"\n"`) and token `271` (`"\n\n"`) were both `20.875`; equal-logit token choice can differ between selection implementations.
- A temporary Qwen3.5-side `argmax_into` branch made exact baselines deterministic, but it added model-side maintenance surface.
- The chosen fix keeps Qwen3.5 greedy on the existing FlashInfer `TopKDispatch(..., k=1)` path. A per-dispatch `cudaMemsetAsync` of `RadixRowState` was tested and removed because FlashInfer's wrapper zero-initializes the cached scratch and the radix top-k kernel resets its row state at the end of the launch.
- Two-run same-seed regen checks on the default engine produced byte-identical JSON in one sampled run (`FLASH_RESET_REGEN_DETERMINISTIC`), while reduced-capacity scheduler runs can still pick a different equal-logit branch because the engine shape changes.
- `test_data/Qwen3.5-4B.json` was refreshed to the current FlashInfer top1 output and is the hard golden for the default Qwen3.5 e2e.
- Qwen3.5 `e2e` compares every prompt against exact text in `test_data/Qwen3.5-4B.json`. `e2e_scheduler` remains a scheduler integration test for the reduced-capacity path (`max_batch=8`) and checks liveness/finish behavior rather than replacing the default exact regression.

### Step 5: Validation
- Passed:
  - `cargo fmt --all --check`
  - `PEGAINFER_CUDA_SM=120 cargo check --release --workspace --all-targets`
  - `PEGAINFER_CUDA_SM=120 cargo clippy --release --workspace --all-targets -- -D warnings`
  - Two-run same-seed regen comparison with temporary model alias `/tmp/Qwen35DetTest` while evaluating FlashInfer top1 behavior.
  - `PEGAINFER_CUDA_SM=120 cargo test --release -p pegainfer test_gpu_sample -- --nocapture`
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/code/workspace-rustllm/pegainfer/models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35-4b --test e2e -- --nocapture`
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/code/workspace-rustllm/pegainfer/models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35-4b --test e2e_scheduler -- --nocapture`
  - `git diff --check`

## Debrief

- **Outcome**: Qwen3.5 e2e and scheduler e2e pass again on the crate-split branch.
- **Pitfalls encountered**:
  - The first control run against an older worktree was misleading until the historical `third_party/flashinfer` layout and Triton Python environment were repaired.
  - The visible symptom had two layers: thread-local cuBLAS misuse caused true gibberish, while FlashInfer top1 caused deterministic-baseline instability after the main corruption was fixed.
- **Lessons learned**:
  - Any runtime that moves a model onto a worker thread must bind the CUDA context and initialize thread-local CUDA library handles inside that worker thread.
  - Greedy e2e baselines can be sensitive to equal-logit top1 choices. Keeping FlashInfer as the selector is lower maintenance, and the exact text comparison is tied to the default engine shape used by `tests/e2e.rs`.
