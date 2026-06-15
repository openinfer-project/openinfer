# DeepSeek-V2-Lite Status And Benchmark Ledger

> **TL;DR:** DeepSeek-V2-Lite is a feature-gated EP2 correctness and attribution target. The original `Hello` / 16 greedy gate is now widened through a committed small case set for HF / host-staged / NCCL comparison; NCCL decode combine and dense exchange use reusable device scratch, and NCCL replay now uses a precomputed route plan while full decode graph capture remains blocked. Current batch, HTTP, and vLLM data remain diagnostic and do not claim production serving parity.

Last touched: 2026-06

## Capability Contract

| Capability | Status | Evidence |
| --- | --- | --- |
| EP2 correctness bring-up | Available | PR #149 adds the model crate, EP2 expert ownership, rank1 expert-only loading, and the host-staged dispatch/combine baseline. |
| Naive NCCL backend | Available | PR #150 adds a dense correctness-first NCCL path. Host-staged remains the transport oracle. |
| HF token/text/hash gate | Available | PR #154 establishes the HF / host-staged / NCCL comparison; PR #176 refreshes it to Transformers `generate(..., use_cache=true)`. |
| HF widened case set | Available | Issue #274 adds a committed case set that keeps the HF / host-staged / NCCL oracle strict while adding additional prompts and diagnostic batch sizes `4` and `8`; the 2026-06-14 2x RTX 5090 run classified all 5 cases as `all_token_text_exact`. |
| Decode attribution | Available | PR #162 and PR #169 add CPU/GPU attribution, route counts, NCCL counters, CUDA event timing, and optional NVTX correlation. |
| Direct same-prompt diagnostic batch | Available | PR #184 and PR #196 cover batch sizes `1`, `4`, and `8` for the fixed same-prompt direct path. |
| Device-resident NCCL combine | Available | Issue #275 keeps NCCL combine contributions/results on reusable f32 device scratch and preserves the HF / host-staged / NCCL exact gate on 2x RTX 5090. |
| Device-resident NCCL dense exchange | Available | Issue #276 reuses backend-owned bf16 dense-exchange scratch, clears rank1 zero-send every exchange, removes dense-exchange stream sync from the backend call, and preserves HF / host-staged / NCCL exactness on 2x RTX 5090. |
| NCCL route-plan replay | Available | Issue #277 builds a token-major host route plan once after top-k routing, replays that plan for NCCL expert launches and device contribution accumulation, keeps route counters visible, and preserves HF / host-staged / NCCL exactness on 2x RTX 5090. |
| NCCL CUDA Graph readiness | Diagnostic only | The attribution binary emits `cuda_graph_readiness`. Current NCCL full decode capture remains blocked by host route-plan construction/replay; the removed dense-exchange allocation/sync and old per-token route-iteration blockers should stay absent. |
| Production continuous batching | Not available | The direct diagnostic batch path is not mixed-request HTTP serving. |
| vLLM production parity | Not claimed | The vLLM startup-failure evidence below keeps the issue #170 comparison matrix honest without claiming serving parity. |

## Correctness Contract

The retained correctness gate is deliberately narrow:

- model: DeepSeek-V2-Lite;
- devices: single-node EP2 with two local GPUs;
- committed cases: `test_data/deepseek-v2-lite-ep2-cases.json` keeps the original `Hello` / 16-token case and widens the oracle with a few additional prompts plus batch sizes `4` and `8`;
- generation mode: greedy;
- backends: host-staged and `OPENINFER_DSV2_LITE_EP_BACKEND=nccl`.

The comparison gate must be run on the same model snapshot for HF, host-staged, and NCCL outputs. Same-host comparison remains strict: HF, host-staged, and NCCL must be token-exact and text-exact for every committed case and every diagnostic batch row. Host-staged remains the baseline oracle for NCCL transport changes. The latest retained evidence is the 2026-06-14 2x RTX 5090 case-set run with `case_count=5`, top-level `classification=all_token_text_exact`, and no comparison warnings.

The Rust E2E accepts the known HF-confirmed RTX 5090 and A800 hash pairs for this narrow shape, because the same model snapshot has produced different exact greedy text on those hosts while still matching HF on each host. Do not use the static hash pair list as a substitute for the same-host HF comparison when changing accuracy-sensitive code.

## Benchmark Ledger

### Direct Same-Prompt Diagnostic Batch

This path is useful for attribution and for avoiding the earlier row-loop TPOT measurement. It is not production continuous batching:

- every row uses the same prompt;
- prefill remains conservative;
- the direct benchmark path is not `/v1/completions` serving;
- it does not prove request admission, per-request KV ownership, fairness, or mixed-request scheduling.

Current retained direct snapshot from the issue #277 branch (`2f52ed6`, 2026-06-15, 2x RTX 5090 / SYS interconnect). Shape: `prompt="Hello"`, `output_len=16`, `warmup=5`, `iters=20`; every row produced token trace hash `ed0eab52473991fc`. `decode tok/s` is the benchmark report's aggregate `metrics.decode_tok_s`. This refresh replaces the older PR #184 row values for the current branch ledger, but it should not be read as an isolated route-plan speedup because the retained snapshot was rerun on a different validation environment.

| Batch | Backend | steady TPOT p50 ms | steady TPOT avg ms | decode tok/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | host-staged | 55.727 | 57.313 | 17.486 |
| 1 | NCCL | 181.795 | 188.420 | 5.321 |
| 4 | host-staged | 193.954 | 198.905 | 20.106 |
| 4 | NCCL | 303.389 | 311.621 | 12.821 |
| 8 | host-staged | 385.013 | 394.908 | 20.270 |
| 8 | NCCL | 472.045 | 483.538 | 16.517 |

PR #196 extends attribution for the same direct diagnostic shapes. The retained A800 attribution gate keeps `batch-size=1/4/8`, `prompt="Hello"`, `output_len=16`, host-staged, and NCCL exact against the same-host HF gate.

### HTTP Concurrency Pressure

The issue #277 branch was also run through `/v1/completions` with `vllm bench serve` used only as the common HTTP client. Shape: random input length `2`, output length `16`, `24` prompts, `temperature=0`, `ignore_eos`, `--max-concurrency 1/4/8`, OpenInfer `--cuda-graph=false`.

OpenInfer streaming currently makes the client-side TPOT fields near-zero in this shape, so this table reports output throughput and throughput-derived milliseconds per output token computed as `duration / total_output_tokens`. `--max-concurrency` should be read as concurrent request pressure, not as proof of true internal OpenInfer batch size.

| Backend | conc | completed | output tok/s | throughput-derived ms/output token | mean TTFT ms | median TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| host-staged | 1 | 24/24 | 20.912 | 47.820 | 764.471 | 740.296 |
| host-staged | 4 | 24/24 | 21.030 | 47.552 | 2838.390 | 3036.649 |
| host-staged | 8 | 24/24 | 20.964 | 47.700 | 5198.935 | 5991.506 |
| NCCL | 1 | 24/24 | 6.302 | 158.689 | 2538.216 | 2553.374 |
| NCCL | 4 | 24/24 | 6.326 | 158.083 | 9491.680 | 10097.244 |
| NCCL | 8 | 24/24 | 6.341 | 157.710 | 17242.941 | 20110.121 |

### vLLM Startup Failures

In response to issue #170's request for a vLLM TP2+EP2 or pure TP2 comparison, keep vLLM in the matrix when the selected validation environment can bring the server to readiness. The 2026-06-15 attempts on the same 2x RTX 5090 host failed before `/v1/models` readiness for both the older retained vLLM environment and a freshly installed current vLLM environment. Earlier manual vLLM rows should not be mixed with this current branch/hardware snapshot; they only show that vLLM may be viable in other environments.

| Environment | Modes tried | Result |
| --- | --- | --- |
| vLLM `0.9.2`, Torch `2.7.0+cu128`, Transformers `4.53.3` | TP2, TP2+EP2, V1/V0, default/eager fallback, plus one low-memory smoke | server never reached readiness; failures were `CUBLAS_STATUS_NOT_INITIALIZED` during worker profile plus one low-memory illegal-memory-access smoke |
| vLLM `0.22.1`, Torch `2.11.0+cu130`, Transformers `5.12.0` | TP2 and TP2+EP2 default/eager fallback | server never reached readiness; failures were FlashInfer / SM12x backend selection errors (`No supported CUDA architectures found for major versions [12]`, `FlashInfer requires GPUs with sm75 or higher`) |
| vLLM `0.22.1` controlled backend fallback | TP2 and TP2+EP2 with `--attention-backend TRITON_ATTN --moe-backend triton` | server never reached readiness; `TRITON_ATTN` is not valid for DeepSeek-V2 MLA (`MLA not supported`) |

Interpretation:

- direct same-prompt diagnostics show NCCL is still much slower than host-staged, although aggregate decode throughput improves with larger diagnostic batch size;
- NCCL remains a correctness-first backend and is still significantly slower than host-staged;
- OpenInfer HTTP throughput did not scale with concurrency in this snapshot, so serving batching remains open;
- vLLM comparison rows should remain failed/missing for this environment until a selected vLLM build reaches server readiness; standalone Torch bf16 GEMM passed on both GPUs in both tested vLLM environments, so the retained failure is specific to the vLLM DeepSeek-V2 server/model startup path rather than a blanket CUDA outage.

## Claim Boundaries

Use these labels consistently:

| Label | Meaning | Do not infer |
| --- | --- | --- |
| `direct single-row` | In-process batch `1` decode. | HTTP serving throughput. |
| `direct same-prompt diagnostic batch` | Fixed same-prompt direct batch sizes `1/4/8`. | Production continuous batching or mixed-request scheduling. |
| `HTTP concurrency pressure` | `vllm bench serve --max-concurrency N` against an HTTP endpoint. | True OpenInfer batch size unless the engine path proves it. |

Do not claim:

- production EP readiness;
- sparse dispatch readiness;
- multi-node EP support;
- vLLM serving parity;
- performance improvement from the status tables alone.

## Next Gates

Issue #205 records the model roadmap. Maintainer feedback there calls out NCCL plus CUDA Graph as the likely best decode direction, with host staging possibly deprecated later. Treat that as a future direction, not as current evidence.

The current graph-readiness diagnostic is intentionally fail-closed: `full_decode_capture_ready=false` for NCCL. Issue #275 removed the old NCCL combine H2D/D2H/allocation/sync blockers, issue #276 removed the dense-exchange allocation/sync blockers, and issue #277 narrows the remaining NCCL route work into a precomputed host route plan plus host-directed replay. The old per-token route-iteration and host-directed expert-accumulation blocker IDs should stay absent from the current readiness report. The optional f32 NCCL graph smoke is a separate collective-only diagnostic and is not #276/#277 evidence. HF, host-staged, and NCCL remain token/text exact for the committed case set.

The next implementation should be chosen from measured evidence:

1. Keep the widened HF / host-staged / NCCL case set current.
   - keep the committed cases and row-level comparison shape in sync with the accuracy docs;
   - treat the widened oracle as correctness evidence only, not serving evidence;
   - keep host-staged as the baseline oracle while it exists.

2. Move the remaining NCCL decode path toward CUDA Graph coverage.
   - keep HF / host-staged / NCCL exact before and after;
   - keep host-staged as the correctness baseline while it exists;
   - preserve attribution before and after the change;
   - keep narrowing host route-plan construction/replay before claiming full decode capture;
   - avoid broad generic EP or multi-node work;
   - judge issue #170 by whether it reduces NCCL decode overhead and makes the path more graph-friendly.

3. Keep a fair serving benchmark contract around future performance work.
   - OpenInfer host-staged.
   - OpenInfer NCCL.
   - vLLM TP2.
   - vLLM TP2+EP2 when supported.
   - default vLLM configuration plus a controlled configuration with cache/flag choices recorded.

4. Add real request batching / serving semantics before broader throughput claims.
   - request admission;
   - per-request KV ownership;
   - mixed request state;
   - decode iterations that carry multiple live `/v1/completions` requests.

5. Keep MoE internals readable.
   - routing, dispatch, expert execution, and combine should remain distinguishable in code and attribution;
   - avoid introducing a generic EP framework before the DeepSeek-V2-Lite EP2 path has a measured reason to need it.
