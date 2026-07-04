# Coding Style

## Testing

Don't test for the sake of testing. Prefer integration tests over unit tests — if the E2E test catches it, a unit test is ceremony. Unit tests earn their place for silent failures: GPU kernels, tricky data-structure invariants, edge-case-rich pure logic.

## Test runner

`cargo nextest run --workspace` (config in `.config/nextest.toml`) is the sweep: it runs lib unit tests, excludes hardware gates via the config's `default-filter`, and *reports* everything it excluded as skipped — selection is explicit, never silent. Hardware gates run per package with `--ignore-default-filter`:

```bash
OPENINFER_TEST_MODEL_PATH=/path/to/Qwen3-4B \
  cargo nextest run --release -p openinfer-qwen3 --ignore-default-filter -E 'binary(hf_golden_gate)'
```

Why nextest over bare `cargo test`: process-per-test isolation (thread-local cuBLAS/CUDA-context poisoning can't leak between tests — see `lessons/exact-match-gate-thread-cublas.md`), hang-kill via `slow-timeout` (a wedged NCCL collective goes red instead of hanging the session), and GPU concurrency groups (gates strictly serial, CUDA-touching lib tests bounded). It also *compiles* every test target where `cargo test --lib` compiles none of the integration tests — that alone caught a gate that had silently rotted against a renamed API.

Two rules encoded in the config that must not regress:

- `retries = 0`. Flaky = bug. Exact-match greedy gates are this repo's core asset; a retry culture rots them.
- No env-probing skips inside tests. If a test can't run on a machine class, exclude it in `default-filter` with a comment (visible in config + counted as skipped) — never auto-skip from inside the test, which reports a green that guards nothing.

nextest does not run doctests. The workspace has 3 runnable ones today; `cargo test --release --doc --workspace` covers them when doc examples change.

## Logging

Log through `openinfer-core::logging`. The text layout already prints each record's module target, so don't prefix messages with a module or model name — no `kimi-k2:`, no `Qwen3.5 `. Error messages in `anyhow!` / `bail!` keep their prefix; they surface to callers without a target.
