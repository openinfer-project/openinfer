# Qwen3.5 Scheduler LoadSnapshot

> **TL;DR:** Issue #605 publishes Qwen3.5 logical running, waiting, and KV load through `EngineHandle::with_load_watch` for single-GPU and TP backends without changing scheduler behavior; NVIDIA `/metrics` proof remains required.
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
  3. Cover the scheduler-state mapping with pure accounting tests and document the live NVIDIA acceptance gate.
- **Risks / open questions**:
  - Real `/metrics` validation requires a CUDA-capable host and Qwen3.5 weights.
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

## Execution Log

- Added load watches to `start_with_capacity` and `start_tp_with_capacity` and attached each receiver to its engine handle.
- Added backend-neutral snapshot publication to the shared scheduler loop.
- Kept the original Qwen3.5 idle receive and same-iteration admission flow; removed the draft-only `deferred = pending; continue;` transition after maintainer review.
- Added pure accounting coverage for active plus prefilling running requests, deferred waiting requests, normal KV subtraction, and saturating inconsistent capacity input.
- Updated the shared Prometheus documentation for Qwen3.5's one-logical-engine contract.
- Preserved the unrelated `docs/models/qwen35/source-walkthrough.md` outside the change set.

## Validation Boundary

Local checks completed:

- `cargo fmt --all --check`: passed with `nightly-2026-07-10`.
- `cargo metadata --no-deps --format-version 1`: passed.
- `git diff --check`: passed.
- The targeted `load_snapshot` test did not reach assertions on this Apple Silicon host: `vllm-server` could not find `protoc`, and `openinfer-kernels` could not run NVIDIA `nvcc`.
- `cargo test --release --workspace --lib` did not reach tests because the workspace `moe` build requires NCCL 2.30.4 or newer, which is not installed on this host.

Before Draft PR #692 is ready, an NVIDIA endpoint must provide commands and raw metric samples proving:

1. Running requests and KV usage become non-zero during real generation.
2. Waiting becomes non-zero under real batch-slot or KV pressure.
3. Running, waiting, and KV usage return to zero after the workload drains.
4. A follow-up completion succeeds after the pressure test.

If real pressure never retains requests in `deferred`, investigate the actual parked state and track any scheduler-policy change in a separate issue. Do not add a scheduler transition as a metrics workaround.

## Debrief

- **Outcome**: The local implementation exposes Qwen3.5 scheduler gauges for single-GPU and TP without changing scheduler behavior.
- **Pitfalls encountered**:
  - The TP scheduler rebase required KV accounting through `SchedulerBackend`; retaining model-specific `model.kv_pool()` access would not compile against the shared loop.
  - Copying Qwen3's deferred-plus-continue transition would mix scheduler policy into an observability PR.
- **Lessons learned**:
  - Reuse Qwen3's watch contract, not model-specific control flow.
  - Observability should consume the scheduler backend contract when one loop serves multiple execution topologies.
- **Follow-ups**:
  - Run the Qwen3.5 feature build and scheduler tests on an NVIDIA development host.
  - Attach live active, pressure, idle-zero, and recovery evidence before marking the PR ready.
