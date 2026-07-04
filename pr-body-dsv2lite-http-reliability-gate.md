## Summary

- Adds a replayable DeepSeek-V2-Lite EP2 HTTP reliability gate for real streaming `/v1/completions` requests.
- Extends the request lifecycle contract across the vLLM bridge, `TokenSink`, and DSV2-Lite scheduler so cancel, disconnect, rejection, error, and completion are machine-distinguishable.
- Adds strict runner JSON, focused tests, live host-staged/NCCL evidence, no-regression benchmark rows, and status-doc claim boundaries.

Fixes #453.

## Extra Risks Found And Fixed

### Cancel and disconnect were not machine-distinguishable

- Risk: a dropped per-request stream and an explicit frontend abort both looked like a send failure, so the HTTP gate could not prove whether it exercised cancel versus disconnect.
- Why it affects #453: the issue asks for cancel and disconnect coverage as separate failure modes.
- Minimal fix: add request-local abort state in `TokenSink` / bridge handling, expose `is_cancelled()`, `is_disconnected()`, and `abort_reason()`, and map send failures to `cancelled` versus `disconnected` terminal reasons in the DSV2-Lite scheduler.
- Evidence: `cargo test --release -p openinfer-engine --lib token_sink -- --nocapture`, `cargo test --release -p openinfer-vllm-frontend --lib abort -- --nocapture`, and the live `cancel_disconnect` scenario on both backends.

### Early disconnect could be misclassified after non-token bridge output

- Risk: the bridge could mark a request as having emitted output before a client-visible token, making an early connection close look like cancel instead of disconnect.
- Why it affects #453: disconnect-before-token and cancel-after-token are different reliability cases.
- Minimal fix: track `has_emitted_tokens` at the bridge boundary and classify aborts using client-visible token output, not internal metadata progress.
- Evidence: `abort_reason_tracks_first_output_boundary` and live `cancel_disconnect` artifacts showing one `cancelled` and one `disconnected` terminal reason per backend.

### Terminal traces could not prove state retirement

- Risk: prior traces had active/decode maxima but no terminal reason or terminal active/pending state, so the gate could not fail on leaked active or pending state.
- Why it affects #453: clean follow-up success alone is weaker than direct scheduler-state evidence.
- Minimal fix: add `terminal_reason`, `pending_queue_size_max`, `active_set_size_at_terminal`, `pending_queue_size_at_terminal`, and `healthy_baseline_after_terminal` to `openinfer_http_trace`.
- Evidence: scheduler lifecycle tests plus the reliability runner's strict trace checks and live artifacts with healthy final baselines after every scenario.

### Overload pressure needed pending-queue recovery evidence

- Risk: an overload run could complete some requests while still hiding bad pending/active cleanup.
- Why it affects #453: active-cap pressure must prove queued work does not poison active requests and the queue can recover.
- Minimal fix: thread active/pending state through batch decode, single-row fallback, cache-position errors, and terminal trace logging; require pending pressure in the runner.
- Evidence: `overload_active_cap` passed on host-staged and NCCL with `active_set_size_max=8`, `pending_queue_size_max=4`, `decode_batch_size_max=7`, `completed=13`, and healthy final baseline.

### Cross-group terminal accounting could overstate active state

- Risk: when different decode-position groups retired rows in the same scheduler round, terminal traces could use a per-group active count instead of a shared round-level active count.
- Why it affects #453: the reliability gate must prove active state retires after failures, so stale terminal active counts weaken the cleanup claim.
- Minimal fix: carry one shared active-remaining counter across same-position batch groups and single-row fallback paths.
- Evidence: `cross_group_terminal_accounting_uses_shared_active_remaining` and live terminal traces with healthy baselines after mixed failure scenarios.

### Benchmark/gate could pass without terminal trace coverage

- Risk: a completed HTTP benchmark can miss failed, cancelled, rejected, or disconnected terminal traces.
- Why it affects #453: the acceptance criteria require counts, terminal reasons, trace coverage, active/pending/decode maxima, and clean follow-up recovery.
- Minimal fix: add `scripts/bench_dsv2lite_http_reliability.py` with strict per-scenario failure rules. HTTP/frontend guard rejections may lack scheduler traces; scheduler-level terminal requests must have traces.
- Evidence: `python3 -m unittest tests.test_bench_dsv2lite_http_reliability`, host-staged reliability artifact `832d65a8e8b2b3a6ad0100c4a35f38475f040d6ffc192ec38a3b7384167187a5`, and NCCL reliability artifact `53bedd98f19c5241df588a1ade8756e84a5e8c99225a589c4ed303e90fba38fa`.

## Evidence

Local checks completed:

```bash
cargo fmt --all --check
git diff --check
cargo test --release -p openinfer-engine --lib token_sink -- --nocapture
cargo test --release -p openinfer-vllm-frontend --lib abort -- --nocapture
python3 -m unittest tests.test_bench_dsv2lite_http_reliability
python3 -m py_compile scripts/bench_dsv2lite_http_reliability.py tests/test_bench_dsv2lite_http_reliability.py
```

Remote 2-GPU checks completed:

```bash
cargo build --release -p openinfer-server --no-default-features --features deepseek-v2-lite
cargo test --release -p openinfer-vllm-frontend --lib abort -- --nocapture
cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --lib scheduler -- --nocapture
OPENINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite OPENINFER_DSV2_LITE_EP_BACKEND=host-staged cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture
OPENINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite OPENINFER_DSV2_LITE_EP_BACKEND=nccl cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture
```

Results:

- `openinfer-engine` token sink tests passed: 3 passed, 0 failed.
- `openinfer-vllm-frontend` abort tests passed: 2 passed, 0 failed.
- DSV2-Lite scheduler lifecycle tests passed: 23 passed, 0 failed.
- Host-staged `e2e_ep2` passed: 1 passed, 0 failed.
- NCCL `e2e_ep2` passed: 1 passed, 0 failed. The SM120 validation used an NCCL 2.30.7 runtime library.
- Python reliability runner tests passed: 3 passed, 0 failed.
- Python syntax check passed.
- Server release build passed.

Live HTTP reliability artifacts:

| Backend | Artifact | SHA-256 | Result |
| --- | --- | --- | --- |
| host-staged | `http_reliability_host_staged.json` | `832d65a8e8b2b3a6ad0100c4a35f38475f040d6ffc192ec38a3b7384167187a5` | passed |
| NCCL | `http_reliability_nccl.json` | `53bedd98f19c5241df588a1ade8756e84a5e8c99225a589c4ed303e90fba38fa` | passed |

Scenario summary:

| Backend | Scenario | Counts | Trace maxima | Final baseline |
| --- | --- | --- | --- | --- |
| host-staged | `cancel_disconnect` | completed `2`, cancelled `1`, disconnected `1`, failed/rejected/timeout `0` | active `2`, pending `0`, decode `1` | healthy |
| host-staged | `invalid_requests` | completed `2`, rejected `4`, failed/timeout `0` | active `1`, pending `0`, decode `1` | healthy |
| host-staged | `overload_active_cap` | completed `13`, failed/rejected/timeout `0` | active `8`, pending `4`, decode `7` | healthy |
| host-staged | `mixed_short_long_with_failures` | completed `5`, cancelled `1`, rejected `1`, failed/timeout `0` | active `4`, pending `0`, decode `2` | healthy |
| NCCL | `cancel_disconnect` | completed `2`, cancelled `1`, disconnected `1`, failed/rejected/timeout `0` | active `2`, pending `0`, decode `1` | healthy |
| NCCL | `invalid_requests` | completed `2`, rejected `4`, failed/timeout `0` | active `1`, pending `0`, decode `1` | healthy |
| NCCL | `overload_active_cap` | completed `13`, failed/rejected/timeout `0` | active `8`, pending `4`, decode `7` | healthy |
| NCCL | `mixed_short_long_with_failures` | completed `5`, cancelled `1`, rejected `1`, failed/timeout `0` | active `5`, pending `0`, decode `2` | healthy |

## No-Regression Benchmark

The no-regression benchmark used real `/v1/completions` traffic after the reliability gate. Every row passed with `failed=0`, `timeouts=0`, and `missing_traces=[]`.

| Backend | Shape | Completed | Failed/timeouts | Output tok/s | TTFT avg ms | TPOT/ITL avg ms | active max | decode batch max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host-staged | short `64/64`, c1 | 8/8 | 0/0 | 22.886 | 1226.8 | 24.9 | 1 | 1 |
| host-staged | short `64/64`, c4 | 8/8 | 0/0 | 21.698 | 2612.8 | 145.1 | 4 | 3 |
| host-staged | short `64/64`, c8 | 8/8 | 0/0 | 21.830 | 5439.8 | 284.2 | 8 | 5 |
| host-staged | mixed `16/128`, c4 | 8/8 | 0/0 | 8.281 | 4543.3 | 210.0 | 4 | 2 |
| host-staged | mixed `16/128`, c8 | 8/8 | 0/0 | 8.288 | 7897.1 | 496.9 | 8 | 3 |
| NCCL | short `64/64`, c1 | 8/8 | 0/0 | 24.873 | 1034.2 | 24.4 | 1 | 1 |
| NCCL | short `64/64`, c4 | 8/8 | 0/0 | 23.678 | 2210.0 | 135.8 | 4 | 3 |
| NCCL | short `64/64`, c8 | 8/8 | 0/0 | 23.872 | 4594.4 | 266.1 | 8 | 6 |
| NCCL | mixed `16/128`, c4 | 8/8 | 0/0 | 9.186 | 2216.6 | 314.9 | 4 | 1 |
| NCCL | mixed `16/128`, c8 | 8/8 | 0/0 | 9.269 | 6920.1 | 453.0 | 8 | 3 |

This is a no-regression and trace-coverage record. It is not a throughput optimization claim.

## Claim Boundary

- This PR claims a stronger HTTP reliability gate and auditable failure isolation for DeepSeek-V2-Lite EP2 serving.
- It does not claim production EP readiness.
- It does not claim vLLM parity.
- It does not claim sparse dispatch, multi-node EP, CUDA Graph productization, or host-staged deprecation.
- It does not claim throughput improvement from the reliability runner or no-regression benchmark.

## Remaining Risks

No #453 reliability blocker remains after the gates above.

Separate follow-ups remain outside this PR's claim boundary:

- #452 remains the mixed/long-prompt latency follow-up.
- #465 remains the sustained soak benchmark follow-up.
- #466 remains the retained SLO report follow-up.
- #467 remains the benchmark artifact manifest follow-up.

## Commands Run And Exact Result

| Command | Result |
| --- | --- |
| `cargo fmt --all --check` | Passed locally. |
| `git diff --check` | Passed locally. |
| `python3 -m py_compile scripts/bench_dsv2lite_http_reliability.py tests/test_bench_dsv2lite_http_reliability.py` | Passed locally. |
| `python3 -m unittest tests.test_bench_dsv2lite_http_reliability` | Passed locally; 3 tests verify dry-run success plus false-positive failures for bad terminal traces and missing trace fields. |
| `cargo test --release -p openinfer-engine --lib token_sink -- --nocapture` | Passed locally; 3 tests verify explicit cancel, disconnect, and closed receiver behavior. |
| `cargo test --release -p openinfer-vllm-frontend --lib abort -- --nocapture` | Passed locally and remotely; 2 tests verify abort handling. |
| `cargo build --release -p openinfer-server --no-default-features --features deepseek-v2-lite` | Passed remotely. |
| `cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --lib scheduler -- --nocapture` | Passed remotely; 23 passed, 0 failed. |
| `OPENINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite OPENINFER_DSV2_LITE_EP_BACKEND=host-staged cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture` | Passed remotely; 1 passed, 0 failed. |
| `OPENINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite OPENINFER_DSV2_LITE_EP_BACKEND=nccl cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture` | Passed remotely; 1 passed, 0 failed. |
| host-staged live HTTP reliability runner | Passed; artifact `http_reliability_host_staged.json`, SHA-256 `832d65a8e8b2b3a6ad0100c4a35f38475f040d6ffc192ec38a3b7384167187a5`. |
| NCCL live HTTP reliability runner | Passed; artifact `http_reliability_nccl.json`, SHA-256 `53bedd98f19c5241df588a1ade8756e84a5e8c99225a589c4ed303e90fba38fa`. |
| host-staged/NCCL no-regression HTTP benchmark | Passed; all 10 rows completed 8/8 requests with failed/timeouts `0/0` and `missing_traces=[]`. |
