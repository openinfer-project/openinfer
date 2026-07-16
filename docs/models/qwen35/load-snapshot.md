# Qwen3.5 Scheduler LoadSnapshot

> **TL;DR:** Rebase issue #605 onto the Qwen3.5 TP scheduler architecture and publish logical running, waiting, and KV load through `EngineHandle::with_load_watch` for both single-GPU and TP backends.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — located the Qwen3.5 model documentation and current roadmap.
  - `docs/models/qwen35/model-crate.md` — confirmed that the model crate owns the scheduler and exposes it through the generic `EngineHandle`.
  - `docs/models/qwen35/roadmap.md` — confirmed the serving and lifecycle observability context.
  - `openinfer-qwen35-4b/src/scheduler.rs` on the feature branch and `origin/main` — identified the conflict between the old `Qwen35Model` scheduler arguments and the new shared `SchedulerBackend` abstraction.
- **Relevant history**:
  - Qwen3 already publishes `LoadSnapshot` at scheduler iteration boundaries.
  - Main now shares one Qwen3.5 scheduler loop between single-GPU and TP backends, so KV accounting must come from `SchedulerBackend`, not directly from `Qwen35Model`.
- **Plan**:
  1. Rebase the feature commit onto current `origin/main`, preserving the TP backend refactor.
  2. Publish snapshots from the shared scheduler loop using backend capacity and availability accounting.
  3. Register a load watch for both single-GPU and TP engine handles and preserve the idle wake-up publication boundary.
  4. Add scheduler accounting coverage, run formatting and targeted/full available checks, then review the final diff.
  5. Commit only issue #605 files, push the feature branch to the fork, and open a Draft PR against `openinfer-project/openinfer:main`.
- **Risks / open questions**:
  - Real `/metrics` validation requires a CUDA-capable host and Qwen3.5 weights; local checks may prove compilation and scheduler accounting without proving live GPU gauges.
  - The unrelated untracked `docs/models/qwen35/source-walkthrough.md` must remain outside this change.

## Execution Log

### Step 1: Rebase onto the TP scheduler architecture

- Fetched `origin/main` at `f5d2bf2` and rebased the issue #605 commit.
- Resolved the `scheduler_loop` call conflict by retaining `SchedulerBackend::Single(backend)` and adding the load sender to the new signature.
- Preserved the unrelated untracked `docs/models/qwen35/source-walkthrough.md` outside the rebase and change set.
- Result: success.

### Step 2: Publish backend-neutral scheduler load

- Added load watches to both `start_with_capacity` and `start_tp_with_capacity`.
- Changed publication to derive total and available KV pages from `SchedulerBackend`.
- Kept the top-of-loop publication point and the idle wake-up `continue`, so a newly received idle request is observable as waiting before admission and idle gauges settle after request teardown.
- Added pure accounting coverage for `active + prefilling` running requests, `deferred` waiting requests, normal KV usage, and saturating inconsistent capacity input.
- Result: success.

### Step 3: Local validation

- `cargo fmt --all --check`: passed with the repository-pinned `nightly-2026-07-10` toolchain.
- `git diff --check`: passed.
- Targeted `cargo test --release -p openinfer-qwen35-4b --features qwen35-4b --lib load_snapshot -- --nocapture`: dependency compilation started, then stopped because this Apple Silicon host lacks `protoc` and NVIDIA `nvcc`; no test assertion ran.
- Result: formatting passed; CUDA-dependent compilation and tests require CI or an NVIDIA development host.

### Step 4: Metrics documentation

- Updated `docs/subsystems/frontend/prometheus-metrics.md` to list Qwen3.5's single logical engine for both single-GPU and TP execution.
- Kept live GPU `/metrics` validation as the next step rather than claiming runtime proof from this host.
- Result: success.

### Step 5: Final review

- Standards review found no remaining documented-standard violation after the accounting test was changed to own the scheduler queue mapping.
- Spec review confirmed the watch wiring, queue mapping, idle publication boundary, backend accounting, and documentation; the only remaining acceptance gap is live GPU `/metrics` validation.
- `cargo metadata --no-deps --format-version 1`, `cargo fmt --all --check`, `git diff --check origin/main...HEAD`, and a conflict-free merge-tree check passed.
- Result: implementation ready for Draft PR and GPU validation.

### Step 6: Publish for community review

- Force-pushed the rebased feature branch to `BreezyB1n/openinfer` with lease protection.
- Updated upstream Draft PR #692 with the backend-neutral implementation, local validation results, and the remaining NVIDIA `/metrics` checklist.
- Result: Draft PR ready for review.

## Debrief

- **Outcome**: Draft PR #692 implements issue #605 against the post-TP scheduler architecture. Single-GPU and TP engines expose the same load-watch path, while each backend owns its KV availability semantics.
- **Pitfalls encountered**:
  - Git reported one textual conflict, but a literal resolution would have left the old `model.kv_pool()` reference inside the backend-based scheduler loop.
  - The first local Rust check required installing the repository-pinned toolchain and downloading large Git dependencies before reaching the expected CUDA host boundary.
- **Lessons learned**:
  - Scheduler observability should consume the scheduler backend contract rather than model-specific state, especially when one loop serves multiple execution topologies.
  - Merge validation must inspect automatically merged code for semantic conflicts, not only conflict markers.
- **Follow-ups**:
  - CI or an NVIDIA host must run the Qwen3.5 feature build and scheduler tests.
  - Live `/metrics` pressure validation must still prove the idle wake-up boundary, non-zero waiting under pressure, idle-zero settlement, and a healthy follow-up completion.
