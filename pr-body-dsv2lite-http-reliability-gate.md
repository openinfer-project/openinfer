## Summary

Adds a replayable DeepSeek-V2-Lite EP2 HTTP reliability gate for real streaming `/v1/completions` traffic.

This PR also tightens the request lifecycle contract across the vLLM bridge, `TokenSink`, and the DSV2-Lite scheduler so completion, rejection, cancellation, disconnect, and error retirement are machine-distinguishable in traces.

Fixes #453.

## What Changed

- Added `scripts/bench_dsv2lite_http_reliability.py` for live HTTP reliability scenarios:
  - client cancel / disconnect during streaming;
  - invalid or unsupported request parameters;
  - active-cap overload pressure;
  - mixed short/long prompts with adjacent failures;
  - clean follow-up request after every scenario.
- Added DSV2-Lite scheduler lifecycle instrumentation:
  - terminal reason;
  - active/pending/decode maxima;
  - terminal active/pending state;
  - healthy final baseline evidence.
- Added request abort reason tracking so cancel and disconnect are distinguishable.
- Added focused unit coverage for token sink abort state, vLLM bridge abort behavior, scheduler retirement, active-cap recovery, and terminal accounting.
- Updated the DeepSeek-V2-Lite status ledger with the reliability evidence and claim boundary.

## Extra Risks Fixed

While implementing #453, the gate exposed a few reliability risks that would have weakened the result:

- Cancel and disconnect previously collapsed into similar send-failure behavior, so the gate could not prove both cases independently.
- Early disconnect could be misclassified after non-token bridge output.
- Terminal traces lacked enough state to prove active/pending retirement.
- Overload runs could complete requests while hiding pending-queue cleanup issues.
- Cross-group terminal accounting could overstate active state within one scheduler round.
- A benchmark could pass while missing terminal trace coverage for failed/cancelled/disconnected requests.

Each fix is scoped to the DSV2-Lite HTTP/scheduler reliability path.

## Evidence

Local checks:

```bash
cargo fmt --all --check
git diff --check
cargo test --release -p openinfer-engine --lib token_sink -- --nocapture
cargo test --release -p openinfer-vllm-frontend --lib abort -- --nocapture
python3 -m unittest tests.test_bench_dsv2lite_http_reliability
python3 -m py_compile scripts/bench_dsv2lite_http_reliability.py tests/test_bench_dsv2lite_http_reliability.py
```

Remote 2-GPU checks:

```bash
cargo build --release -p openinfer-server --no-default-features --features deepseek-v2-lite
cargo test --release -p openinfer-vllm-frontend --lib abort -- --nocapture
cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --lib scheduler -- --nocapture
OPENINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite OPENINFER_DSV2_LITE_EP_BACKEND=host-staged cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture
OPENINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite OPENINFER_DSV2_LITE_EP_BACKEND=nccl cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture
```

Results:

- `openinfer-engine` token sink tests: 3 passed.
- `openinfer-vllm-frontend` abort tests: 2 passed.
- DSV2-Lite scheduler lifecycle tests: 23 passed.
- Host-staged `e2e_ep2`: 1 passed.
- NCCL `e2e_ep2`: 1 passed.
- Python reliability runner tests: 3 passed.
- Server release build passed.

## HTTP Reliability Gate

Live reliability artifacts passed on both backends:

| Backend | Artifact | SHA-256 |
| --- | --- | --- |
| host-staged | `http_reliability_host_staged.json` | `832d65a8e8b2b3a6ad0100c4a35f38475f040d6ffc192ec38a3b7384167187a5` |
| NCCL | `http_reliability_nccl.json` | `53bedd98f19c5241df588a1ade8756e84a5e8c99225a589c4ed303e90fba38fa` |

Both backends passed:

- `cancel_disconnect`
- `invalid_requests`
- `overload_active_cap`
- `mixed_short_long_with_failures`

Every scenario ended with a healthy scheduler baseline and a successful clean follow-up request.

## No-Regression Benchmark

The no-regression HTTP benchmark ran after the reliability gate.

Coverage:

- short same-shape: `input_len=64`, `output_len=64`, concurrency `1/4/8`;
- mixed short/long: `prompt_words=16,128`, `max_tokens=16`, concurrency `4/8`;
- host-staged and NCCL.

Result:

- all 10 rows completed `8/8` requests;
- `failed=0`;
- `timeouts=0`;
- `missing_traces=[]`;
- trace maxima captured active request pressure and decode batch activity.

This benchmark shows that the reliability changes did not regress normal HTTP serving behavior under the covered concurrent workloads. It is not a throughput optimization claim.

## Claim Boundary

This PR claims stronger DeepSeek-V2-Lite EP2 HTTP reliability evidence and auditable failure isolation.

It does not claim:

- production EP readiness;
- vLLM parity;
- sparse dispatch readiness;
- multi-node EP support;
- CUDA Graph productization;
- throughput improvement.

## Remaining Risks

No #453 reliability blocker remains after the gates above.

Separate follow-ups remain outside this PR:

- #452: mixed/long-prompt latency;
- #465: sustained soak benchmark;
- #466: retained SLO report;
- #467: benchmark artifact manifest.
