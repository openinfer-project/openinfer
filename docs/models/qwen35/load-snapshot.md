# Qwen3.5 Scheduler LoadSnapshot

> **TL;DR:** Issue #605 publishes Qwen3.5 logical running, waiting, and KV load without changing scheduler behavior; a repository-native HTTP runner now produces the required NVIDIA `/metrics` evidence, but the real GPU run remains open.
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
  3. Construct `LoadSnapshot` directly in `publish_load`, matching Qwen3, and rely on shared bridge/sim coverage plus the live NVIDIA acceptance gate.
  4. Add a repository-native HTTP `/metrics` runner that retains machine-readable results and paste-ready community evidence without depending on an external benchmark client.
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

`scripts/validate_qwen35_load_metrics.py` is the acceptance runner. It drives deterministic non-streaming completions through the public endpoint, samples the three scheduler gauges by exact `model_name` and `engine` labels, and requires concurrency to exceed the recorded server `--max-batch`. It writes a complete JSON artifact plus paste-ready Markdown containing the server and runner commands, hardware/toolchain metadata, model revision and fingerprint, request counts, peaks, and raw metric lines. It has no external benchmark-client dependency.

## Execution Log

- Added load watches to `start_with_capacity` and `start_tp_with_capacity` and attached each receiver to its engine handle.
- Added backend-neutral snapshot publication to the shared scheduler loop.
- Kept the original Qwen3.5 idle receive and same-iteration admission flow; removed the draft-only `deferred = pending; continue;` transition after maintainer review.
- Constructed `LoadSnapshot` directly in `publish_load`, matching Qwen3 instead of adding a test-only mapping helper.
- Added the HTTP metrics runner and regression coverage for Prometheus parsing, acceptance failures, evidence selection/rendering, server-command pressure validation, and the complete traffic-to-drain-to-recovery flow against a local fake endpoint.
- Updated the shared Prometheus documentation for Qwen3.5's one-logical-engine contract.
- Preserved the unrelated `docs/models/qwen35/source-walkthrough.md` outside the change set.

## Validation Boundary

Local checks completed:

- `cargo fmt --all --check`: passed with `nightly-2026-07-10`.
- `cargo metadata --no-deps --format-version 1`: passed.
- `git diff --check`: passed.
- `python3 -m unittest tests.test_validate_qwen35_load_metrics -v`: `7/7` passed, including the local HTTP orchestration test.
- `python3 -m py_compile scripts/validate_qwen35_load_metrics.py tests/test_validate_qwen35_load_metrics.py`: passed.
- `python3 -m unittest discover -s tests -p 'test_*.py'`: `151/151` passed.
- The targeted `load_snapshot` test did not reach assertions on this Apple Silicon host: `vllm-server` could not find `protoc`, and `openinfer-kernels` could not run NVIDIA `nvcc`.
- `cargo test --release --workspace --lib` did not reach tests because the workspace `moe` build requires NCCL 2.30.4 or newer, which is not installed on this host.

Before Draft PR #692 is ready, an NVIDIA endpoint must provide commands and raw metric samples proving:

1. Running requests and KV usage become non-zero during real generation.
2. Waiting becomes non-zero under real batch-slot or KV pressure.
3. Running, waiting, and KV usage return to zero after the workload drains.
4. A follow-up completion succeeds after the pressure test.

If real pressure never retains requests in `deferred`, investigate the actual parked state and track any scheduler-policy change in a separate issue. Do not add a scheduler transition as a metrics workaround.

## Debrief

- **Outcome**: The local implementation exposes Qwen3.5 scheduler gauges for single-GPU and TP without changing scheduler behavior, and includes a reproducible runner for the remaining community acceptance evidence.
- **Pitfalls encountered**:
  - The TP scheduler rebase required KV accounting through `SchedulerBackend`; retaining model-specific `model.kv_pool()` access would not compile against the shared loop.
  - Copying Qwen3's deferred-plus-continue transition would mix scheduler policy into an observability PR.
- **Lessons learned**:
  - Reuse Qwen3's watch contract, not model-specific control flow.
  - Observability should consume the scheduler backend contract when one loop serves multiple execution topologies.
- **Follow-ups**:
  - Run the Qwen3.5 feature build, scheduler tests, and `scripts/validate_qwen35_load_metrics.py` on an NVIDIA development host with the server constrained to `--max-batch 1`.
  - Attach the runner-generated Markdown evidence before marking the PR ready.
