# DeepSeek-V2-Lite Status And Benchmark Ledger

> **TL;DR:** DeepSeek-V2-Lite is a feature-gated EP2 correctness and attribution target. The original `Hello` / 16 greedy gate is now widened through a committed small case set for HF / host-staged / NCCL comparison; NCCL decode combine and dense exchange use reusable device scratch, while host-directed routing/expert accumulation still block full decode graph capture. Current batch and vLLM data remain diagnostic and do not claim production serving parity.

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
| NCCL CUDA Graph readiness | Diagnostic only | The attribution binary emits `cuda_graph_readiness`. Current NCCL full decode capture remains blocked by host route iteration and host-directed expert accumulation; the removed dense-exchange allocation/sync blockers should stay absent. |
| Production continuous batching | Not available | The direct diagnostic batch path is not mixed-request HTTP serving. |
| vLLM production parity | Not claimed | The manual vLLM snapshot below is for understanding the gap requested in issue #170. |

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

Current retained direct snapshot from PR #184:

| Batch | Backend | steady TPOT p50 ms | steady TPOT avg ms | decode tok/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | host-staged | 58.558 | 62.009 | 16.144 |
| 1 | NCCL | 193.650 | 201.276 | 4.982 |
| 4 | host-staged | 202.186 | 210.409 | 19.124 |
| 4 | NCCL | 333.321 | 344.764 | 11.528 |
| 8 | host-staged | 394.753 | 411.348 | 19.423 |
| 8 | NCCL | 522.917 | 539.643 | 14.874 |

PR #196 extends attribution for the same direct diagnostic shapes. The retained A800 attribution gate keeps `batch-size=1/4/8`, `prompt="Hello"`, `output_len=16`, host-staged, and NCCL exact against the same-host HF gate.

### Manual vLLM Snapshot

In response to issue #170's request for a vLLM TP2+EP2 or pure TP2 comparison, a manual same-model snapshot was collected with `vllm bench serve` concurrency pressure `1`, `4`, and `8`.

This table is retained only to document the current gap. It is not evidence of a complete, fair production-serving parity comparison, and `--max-concurrency` should be read as concurrent request pressure, not as proof of true internal OpenInfer batch size.

| Engine | Mode | conc=1 TPOT ms | conc=4 TPOT ms | conc=8 TPOT ms | Output tok/s at 1/4/8 |
| --- | --- | ---: | ---: | ---: | --- |
| OpenInfer | host-staged | 49.95 | 51.30 | 51.22 | 19.84 / 19.53 / 19.56 |
| OpenInfer | NCCL | 178.31 | 173.22 | 174.46 | 5.59 / 5.77 / 5.73 |
| vLLM | TP2 default | 35.61 | 36.43 | 36.37 | 27.54 / 97.72 / 195.28 |
| vLLM | TP2+EP2 default | 34.15 | 34.97 | 34.88 | 28.87 / 101.52 / 204.08 |

Interpretation:

- at single-concurrency TPOT, host-staged is closer to vLLM than the current NCCL backend;
- NCCL remains a correctness-first backend and is still significantly slower than host-staged;
- OpenInfer HTTP throughput did not scale with concurrency in this snapshot, so serving batching remains open;
- vLLM TP2+EP2 worked in this environment and should stay in future comparison matrices.

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

The current graph-readiness diagnostic is intentionally fail-closed: `full_decode_capture_ready=false` for NCCL. Issue #275 removed the old NCCL combine H2D/D2H/allocation/sync blockers, and issue #276 removed the dense-exchange allocation/sync blockers from the retained 2x RTX 5090 attribution gate. Those removed dense-exchange blockers are absent from the current readiness report. The remaining NCCL blockers are host route iteration and host-directed expert accumulation. The optional f32 NCCL graph smoke is a separate collective-only diagnostic and is not #276 evidence. HF, host-staged, and NCCL remain token/text exact for the committed case set.

The next implementation should be chosen from measured evidence:

1. Keep the widened HF / host-staged / NCCL case set current.
   - keep the committed cases and row-level comparison shape in sync with the accuracy docs;
   - treat the widened oracle as correctness evidence only, not serving evidence;
   - keep host-staged as the baseline oracle while it exists.

2. Move the remaining NCCL decode path toward CUDA Graph coverage.
   - keep HF / host-staged / NCCL exact before and after;
   - keep host-staged as the correctness baseline while it exists;
   - preserve attribution before and after the change;
   - attack host route iteration and host-directed expert accumulation next;
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
