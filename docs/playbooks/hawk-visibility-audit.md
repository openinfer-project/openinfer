# Hawk Visibility Audit Playbook

> **TL;DR:** [hawk](https://github.com/astral-sh/hawk) audits workspace-wide Rust visibility against the shipped binaries. **Always run it with all features**: under a single default-feature profile, 61% of `dead_public` findings are false positives from uncompiled feature-gated consumers (2026-07-22 baseline: 345 under the default profile vs 134 under all-features).

hawk treats the workspace as a closed world and walks reachability from the production targets declared in `hawk.toml`. It reports three lints:

- `hawk::dead_public` — unreachable from both production and every non-production target (tests/benches/examples/doctests). **Report-only**: `--fix` never touches it. The safe landing path is to restrict to `pub(crate)`, let rustc's own `dead_code` confirm, then delete.
- `hawk::unnecessary_public` — `pub` that can become `pub(crate)` (including a "needed only by tests" subclass).
- `hawk::unnecessary_restricted_visibility` — restricted visibility (`pub(crate)` and friends) that can become private.

## Install

hawk uses `rustc_private` and is pinned to Rust 1.97.1; the prebuilt release does not need `rustc-dev`:

```sh
rustup toolchain install 1.97.1
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/astral-sh/hawk/releases/latest/download/cargo-hawk-installer.sh | sh
```

## Run

```sh
RUSTC_BOOTSTRAP=1 \
OPENINFER_NCCL_ROOT=/data/opt/nccl-2.30.4 \
OPENINFER_TRITON_PYTHON=.venv/bin/python \
cargo +1.97.1 hawk check
```

All three environment variables are required:

- `RUSTC_BOOTSTRAP=1` — the repo pins nightly (`generic_const_exprs`); hawk only accepts the 1.97.1 stable compiler, so feature gates must be unlocked.
- `OPENINFER_NCCL_ROOT` — under all-features, `openinfer-kernels/moe` builds the DeepEP shim, which needs NCCL >= 2.30.4.
- `OPENINFER_TRITON_PYTHON` — the `qwen35-4b` build-time Triton AOT codegen (build.rs also falls back to `.venv/bin/python` on its own).

Expected runtime: ~20 minutes cold (Triton AOT + Marlin nvcc dominate); artifacts are cached under `/tmp/cargo-hawk-target/<workspace>-<hash>`, so reruns take a few minutes.

## Reading results / gotchas

- `hawk.toml` declares `openinfer` + `bench_serving` as production targets and a single all-features profile. **Plain bins are not part of the non-production surface** — APIs used only by an undeclared bin get misreported as dead, so add a `[[production]]` entry whenever a new shipped bin appears.
- kvbm-logical is a fork of upstream dynamo: whether to act on its findings (~1/3 of the 2026-07 baseline) depends on how much API divergence from upstream is acceptable. Do not batch-process it.
- `#[macro_export]` macros escape module-visibility reachability: hawk flagged `openinfer-kernels`' `pub mod forward_pass` as dead, but the `typed_pipeline!` macro it defines is exported crate-wide and used by kimi-k2. Verify macro exports before deleting a flagged module.
- `openinfer-engine` is an external boundary: the excluded `openinfer-dynamo-backend` workspace consumes it by path (e.g. `EngineHandle::load_watch`/`take_kv_events`). hawk cannot see that workspace — cross-check engine findings against `openinfer-dynamo-*/src` before acting.
- `--fix` applies visibility reductions mechanically (via `cargo fix`) but only works with a single feature profile.
- hawk's all-features, all-target compilation doubles as a CI blind-spot detector: the first run here surfaced the three feature-gated compile errors fixed in #741.

## Baseline (2026-07-22, all-features)

946 findings: 666 `unnecessary_public` / 134 `dead_public` / 146 `unnecessary_restricted_visibility`. `dead_public` by crate: kernels 33, qwen3 25, kvbm 25, core 23, qwen35 13, remainder in single digits.
