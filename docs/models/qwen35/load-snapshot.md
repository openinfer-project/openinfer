# Qwen3.5 Scheduler LoadSnapshot

> **TL;DR:** Issue #605 now keeps every Rust change in `openinfer-qwen35-4b/src/scheduler.rs`, reuses the existing HTTP benchmark for RTX 5090 proof, and carries no Qwen3.5-specific runner or test files.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — located the Qwen3.5 model and frontend observability documentation.
  - `docs/models/qwen35/model-crate.md` — confirmed that the model crate owns the scheduler and exposes it through the generic `EngineHandle`.
  - `docs/models/qwen35/roadmap.md` — confirmed the serving and lifecycle observability context.
  - `docs/subsystems/frontend/prometheus-metrics.md` — confirmed the existing `LoadSnapshot` bridge contract.
  - `openinfer-qwen35-4b/src/scheduler.rs` on this branch and `origin/main` — compared the metrics wiring with the shared single-GPU/TP scheduler flow.
- **Relevant history**:
  - Qwen3 already publishes `LoadSnapshot`, but its deferred-plus-continue idle transition predates metrics and is model scheduler behavior, not part of the shared observability recipe.
  - Qwen3.5 shares one scheduler loop between single-GPU and TP backends, so KV accounting must come from `SchedulerBackend`, not directly from `Qwen35Model`.
- **Plan**:
  1. Publish backend-neutral snapshots from existing Qwen3.5 scheduler boundaries.
  2. Attach one load watch to the single-GPU and TP engine handles without adding scheduler transitions or iterations.
  3. Use the existing Qwen3.5 scheduler E2E, generic HTTP benchmark, and raw `/metrics` sampling for validation; retain commands and results in this document and the PR body.
- **Risks / open questions**:
  - The cleaned scheduler implementation is an inline, Qwen3-shaped form of the code tested at `a033258`; run the retained NVIDIA gate against the final code commit before marking the PR ready.
  - The unrelated untracked `docs/models/qwen35/source-walkthrough.md` must remain outside this change.

## Design

The data path reuses the existing frontend contract:

```text
Qwen3.5 SchedulerBackend
  -> LoadSnapshot watch
  -> EngineHandle
  -> LocalEngineBridge
  -> SchedulerStats
  -> /metrics
```

Both Qwen3.5 execution modes own one logical request stream, so single-GPU and TP each attach one `EngineHandle::with_load_watch` receiver. The frontend bridge, metric names, labels, and scheduler-stat conversion remain unchanged.

The scheduler publishes at the top of its existing loop. At that point, work retired by the previous step has been removed and its KV pages have been released, so the next snapshot can settle to idle before `blocking_recv()` waits for new work.

Snapshot accounting is:

| Metric field | Existing Qwen3.5 state |
| --- | --- |
| `num_running_reqs` | `active.len() + prefilling.len()` |
| `num_waiting_reqs` | `deferred.len()` |
| `kv_used_blocks` | request KV capacity minus currently available request pages |
| `kv_total_blocks` | backend request KV capacity, excluding the CUDA Graph padding page |

Instrumentation only reads these states. It does not move newly received requests into `deferred`, force a request to appear as waiting, or alter admission, prefill, decode, and idle wake-up control flow.

The live gate uses the repository's existing `scripts/bench_http_serving.py` to create real overlapping HTTP traffic and a 100 ms `curl /metrics` sampler to retain the three labeled gauges. A Qwen3.5-specific runner is not required.

## Execution Log

- Added load watches to `start_with_capacity` and `start_tp_with_capacity` and attached each receiver to its engine handle.
- Added backend-neutral snapshot publication to the shared scheduler loop.
- Kept the original Qwen3.5 idle receive and same-iteration admission flow; removed the draft-only `deferred = pending; continue;` transition after maintainer review.
- Validated `a033258c1de1944469d6c6335d4a36d4a80192cf` on one RTX 5090 with exact Qwen3.5-4B model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Used the existing generic HTTP benchmark with `--max-batch 1`, four concurrent 512-token completions, and raw 100 ms metric sampling. No scheduler transition or Qwen3.5-specific test runner was needed to expose waiting.
- Removed `scripts/validate_qwen35_load_metrics.py` and `tests/test_validate_qwen35_load_metrics.py`; the final diff contains no new runner or test framework.
- Kept the runtime implementation confined to `openinfer-qwen35-4b/src/scheduler.rs`, matching Qwen3's direct `LoadSnapshot` construction inside `publish_load`.
- Updated the shared Prometheus documentation for Qwen3.5's one-logical-engine contract.
- Preserved the unrelated `docs/models/qwen35/source-walkthrough.md` outside the change set.

## Validation Boundary

Local checks completed:

- `cargo fmt --all --check`: passed with `nightly-2026-07-10`.
- `cargo metadata --no-deps --format-version 1`: passed.
- `git diff --check`: passed.

RTX 5090 checks completed against the exact metrics-only commit `a033258`:

- Release Qwen3.5 server build: passed.
- Existing `test_e2e_qwen35_scheduler`: `1 passed; 0 failed`.
- Real HTTP pressure: `4 completed; 0 failed; 0 timeouts`.
- Peaks: running `1`, waiting `3`, KV usage ratio `0.0010026245171183392`.
- Idle before pressure, after drain, and after recovery: all three gauges were zero.
- Follow-up completion: returned eight tokens successfully.

The raw commands, environment, model hashes, server logs, benchmark JSON, and metric samples are retained locally under `docs/private/qwen35-load-metrics-evidence/`.

The dedicated runner and tests are removed in the local working tree. The same GPU gate must now be associated with the final cleaned code commit before the branch is pushed or the PR is marked ready.

If real pressure never retains requests in `deferred`, investigate the actual parked state and track any scheduler-policy change in a separate issue. Do not add a scheduler transition as a metrics workaround.

## Debrief

- **Outcome**: The metrics-only implementation exposes Qwen3.5 scheduler gauges for single-GPU and TP without changing scheduler behavior; the PR surface is reduced to one Rust implementation file plus documentation.
- **Pitfalls encountered**:
  - The TP scheduler rebase required KV accounting through `SchedulerBackend`; retaining model-specific `model.kv_pool()` access would not compile against the shared loop.
  - Copying Qwen3's deferred-plus-continue transition would mix scheduler policy into an observability PR.
- **Lessons learned**:
  - Reuse Qwen3's watch contract, not model-specific control flow.
  - Observability should consume the scheduler backend contract when one loop serves multiple execution topologies.
  - Existing repository E2E and HTTP tooling is sufficient for one-off GPU evidence; a model-specific runner would add more maintenance cost than coverage.
- **Follow-ups**:
  - Run the retained RTX 5090 gate against the cleaned code commit.
  - Correct the PR description's stale idle-wake-up wording.
  - Paste the retained commands and raw metric output before marking the PR ready.
