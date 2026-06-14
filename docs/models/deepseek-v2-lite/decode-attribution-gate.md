# DeepSeek-V2-Lite EP2 Decode Attribution Gate

> **TL;DR:** DeepSeek-V2-Lite has a narrow EP2 decode attribution report for the retained diagnostic shape: `prompt="Hello"`, `output_len=16`, `batch-size=1/4/8`, host-staged backend, and NCCL backend. The wider HF accuracy gate now covers more cases, while this report stays focused on CPU-side attribution, selected CUDA event timing, optional NVTX ranges, route/transfer counts, and a fail-closed CUDA Graph readiness section. It is evidence for the next bottleneck decision, not a throughput or production EP claim.
>
> **Status:** Passing for the covered EP2 `Hello` / 16-token host-staged and NCCL attribution gate. The batch attribution mode is diagnostic and uses the same-prompt, fixed-length shape as the direct benchmark path.

## Scope

This gate deliberately stays model-specific and shape-specific:

- Model: DeepSeek-V2-Lite.
- Shape: batch size `1`, `4`, or `8`, prompt `Hello`, prompt token ids `[17464]`, output length `16`.
- Backends: default host-staged EP2 and `OPENINFER_DSV2_LITE_EP_BACKEND=nccl`.
- Accuracy oracle: the generated token/text/hash comparison from `hf-accuracy-gate.md`; attribution itself remains on the `Hello` / 16-token diagnostic shape.
- Attribution source: `DeepSeekV2LiteEp2Generator::generate_greedy_with_attribution` for `batch-size=1`, and `DeepSeekV2LiteEp2Generator::generate_greedy_batch_same_prompt_with_attribution` for `batch-size>1`.
- GPU attribution source: CUDA events around selected stream sections in the explicit attribution path.
- NVTX source: set `OPENINFER_DSV2_LITE_NVTX=1` to emit matching ranges for those selected sections during a profiler run.

Out of scope:

- sparse dispatch;
- openinfer-comm / NVLink backend;
- multi-node or generic EP topology;
- production continuous batching or broader prompts;
- performance improvement or throughput claims.

## Report Shape

`dsv2_lite_ep2_decode_attribution` emits structured JSON:

- `report_type`, `model`, `phase`, `backend`, and fixed-shape `config`;
- nested `accuracy` with generated token ids, generated text, token sha256, and text sha256; batch reports also include per-row token/text/hash fields and require same-prompt rows to stay exact;
- CPU-side `timing` with total generation, the prefill-produced first output token, `per_output_token_us`, the 15 true decode-token samples for `output_len=16`, and latency stats;
- `gpu_timing`, `by_gpu_section`, and `by_gpu_call_site` with CUDA event timing for selected GPU/NCCL stream sections, plus a `failure_count` for event-timing failures that did not replace the token/text hash oracle;
- `by_section`, `by_op`, and `by_call_site` rollups in the same vocabulary family as the Qwen3 model report;
- `coverage` rows that distinguish CPU section timing, selected GPU event timing, optional NVTX ranges, and unclaimed throughput;
- `ep` counters for host-staged dispatch/combine and NCCL dense exchange/combine plus local/remote route counts;
- `cuda_graph_readiness` with backend, batch size, fail-closed blockers, route/collective metrics, and an optional NCCL graph smoke result.

Host-staged `dispatch_calls` / `combine_calls` count MoE layer invocations in the fixed greedy run. Host-staged `dispatch_elements` / `combine_elements` count selected routed hidden vectors, so the value is route count times hidden size. NCCL `exchange` and `combine` counters count the dense all-reduce calls and elements used by the current naive NCCL gate.

The GPU event rows are intentionally narrower than the CPU rows. They cover sections that enqueue device work or NCCL work on known streams, including projections, dense/shared/routed experts, NCCL dense exchange, NCCL combine clear, device contribution accumulation, and NCCL combine. They do not relabel pure host routing or host-directed route iteration as GPU work, and the mixed `attention_host_path` stays CPU-side because it includes host attention assembly as well as internal GPU projections.

## Commands

Run the accuracy gate first, because attribution is not allowed to weaken the HF / host-staged / NCCL oracle:

```bash
mkdir -p target/accuracy/dsv2-lite-ep2

python tools/accuracy/hf_dump_dsv2_lite_ep2_greedy.py \
  --model-path models/DeepSeek-V2-Lite \
  --case-set-json test_data/deepseek-v2-lite-ep2-cases.json \
  --out target/accuracy/dsv2-lite-ep2/hf.json

OPENINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
OPENINFER_DSV2_LITE_E2E_CASE_SET=test_data/deepseek-v2-lite-ep2-cases.json \
OPENINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/host-staged.json \
  cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

OPENINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
OPENINFER_DSV2_LITE_E2E_CASE_SET=test_data/deepseek-v2-lite-ep2-cases.json \
OPENINFER_DSV2_LITE_EP_BACKEND=nccl \
OPENINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/nccl.json \
  cargo test --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

python tools/accuracy/compare_dsv2_lite_ep2_outputs.py \
  --hf target/accuracy/dsv2-lite-ep2/hf.json \
  --host-staged target/accuracy/dsv2-lite-ep2/host-staged.json \
  --nccl target/accuracy/dsv2-lite-ep2/nccl.json \
  --out target/accuracy/dsv2-lite-ep2/comparison.json \
  --require-all-exact
```

Then collect attribution for the same two openinfer backends. Use `--batch-size 1` for the original single-row gate, and `--batch-size 4` / `--batch-size 8` for the true-batch benchmark attribution shape. If the NCCL runtime should come from a Python CUDA wheel rather than the system install, set `OPENINFER_NCCL_PYTHON` or `OPENINFER_TRITON_PYTHON` to that environment's Python for the attribution command too.

```bash
cargo run --release -p openinfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path models/DeepSeek-V2-Lite \
  --batch-size 1 \
  --out target/accuracy/dsv2-lite-ep2/host-staged-attribution.json

OPENINFER_DSV2_LITE_EP_BACKEND=nccl \
  cargo run --release -p openinfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path models/DeepSeek-V2-Lite \
  --batch-size 1 \
  --out target/accuracy/dsv2-lite-ep2/nccl-attribution.json

for batch in 4 8; do
  cargo run --release -p openinfer-deepseek-v2-lite \
    --features deepseek-v2-lite \
    --bin dsv2_lite_ep2_decode_attribution \
    -- --model-path models/DeepSeek-V2-Lite \
    --batch-size "$batch" \
    --out "target/accuracy/dsv2-lite-ep2/host-staged-batch${batch}-attribution.json"

  OPENINFER_DSV2_LITE_EP_BACKEND=nccl \
    cargo run --release -p openinfer-deepseek-v2-lite \
    --features deepseek-v2-lite \
    --bin dsv2_lite_ep2_decode_attribution \
    -- --model-path models/DeepSeek-V2-Lite \
    --batch-size "$batch" \
    --out "target/accuracy/dsv2-lite-ep2/nccl-batch${batch}-attribution.json"
done
```

For an Nsight Systems pass, run the same attribution command under the profiler and set `OPENINFER_DSV2_LITE_NVTX=1`; the JSON `coverage` row then records `nvtx_ranges=emitted`. The NVTX labels are correlation markers for the selected GPU/NCCL sections, not timing evidence by themselves. Their wall-clock span can include CPU-side wrapper work, event setup, and synchronization around the section, so compare JSON `by_gpu_*` rows only with CUDA event timing, not with raw NVTX range duration.

To inspect the CUDA Graph readiness boundary for the current NCCL backend, run the attribution binary with the optional smoke flag:

```bash
OPENINFER_DSV2_LITE_EP_BACKEND=nccl \
  cargo run --release -p openinfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path models/DeepSeek-V2-Lite \
  --batch-size 1 \
  --nccl-graph-smoke \
  --out target/accuracy/dsv2-lite-ep2/nccl-graph-smoke.json
```

The smoke uses one preallocated f32 NCCL all-reduce over the existing two rank streams. With `--nccl-graph-smoke`, the command exits non-zero unless capture, replay, and verification all pass. Passing proves only basic collective capture/replay in this runtime. It does not prove full decode CUDA Graph coverage.

## Environment Notes

The NCCL path depends on a runtime that supports the selected GPU. On newer GPUs, older NCCL runtimes may fail communicator initialization before the model-level comparison runs, for example with a shared-memory init error like:

```text
ncclMaxSharedMem 82240 exceeds device/fn maxSharedMem 79856
NCCL WARN Cuda failure 1 'invalid argument'
```

The NCCL loader now tries explicit overrides first (`OPENINFER_NCCL_LIB`, then `OPENINFER_NCCL_LIB_DIR` / `OPENINFER_NCCL_LIBRARY_PATH`), then Python wheel NCCL directories discoverable from `OPENINFER_NCCL_PYTHON`, `OPENINFER_TRITON_PYTHON`, `VIRTUAL_ENV`, or `CONDA_PREFIX`, and finally the system `libnccl.so.2` / `libnccl.so`. This keeps the code path unchanged while avoiding a stale system NCCL when the validation environment already has a newer CUDA wheel runtime.

The HF oracle needs a Python environment that can load DeepSeek-V2-Lite with `trust_remote_code=True`. The helper script tolerates the model file's optional `flash_attn` import check when FlashAttention is not installed, but the HF environment remains separate from the Rust runtime claim: it is only the truth-source generator for the comparison JSON.

## Latest Validation

The issue #276 refresh was rerun on 2026-06-10 with DeepSeek-V2-Lite snapshot `604d5664dddd88a0433dbae533b7fe9472482de0`, `prompt="Hello"`, `output_len=16`, and 2x RTX 5090. HF, host-staged, and NCCL were dumped from the same model directory and compared with `--require-all-exact`. The Rust path loaded NCCL `2.30.7+cuda12.9` from the Python CUDA wheel path because the system NCCL `2.25.1+cuda12.8` failed the init smoke on this Blackwell host before model-level validation.

- HF / host-staged / NCCL comparison: `all_token_text_exact`.
- Token SHA256: `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225`.
- Text SHA256: `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347`.
- Generated text: `, I am a 19 year old girl from the UK. I am`.
- Candidate NCCL attribution: `gpu_timing.sample_count=8384`, `failure_count=0`.

The candidate readiness report still has `full_decode_capture_ready=false`. Compared with the issue #275 candidate, it removes the dense-exchange allocation/sync blockers. The remaining blockers are `nccl_route_iteration_on_host` and `nccl_expert_accumulation_host_directed`.

Current NCCL attribution for the issue #276 gate:

| Batch | GPU event samples | GPU failures | NCCL exchange/combine calls | Route counters | Readiness blockers |
| ---: | ---: | ---: | --- | --- | --- |
| 1 | 8384 | 0 | `416 / 416` | `local=1284`, `remote=1212`, `combine=2496` | `nccl_route_iteration_on_host`, `nccl_expert_accumulation_host_directed` |
| 4 | 23996 | 0 | `494 / 494` | `local=5136`, `remote=4848`, `combine=9984` | `nccl_route_iteration_on_host`, `nccl_expert_accumulation_host_directed` |
| 8 | 44812 | 0 | `598 / 598` | `local=10272`, `remote=9696`, `combine=19968` | `nccl_route_iteration_on_host`, `nccl_expert_accumulation_host_directed` |

The previous A800 strict same-host accuracy gate was rerun on 2026-06-04 with DeepSeek-V2-Lite snapshot `604d5664dddd88a0433dbae533b7fe9472482de0`, `prompt="Hello"`, `output_len=16`, and 2x A800-SXM4-80GB. The token/text oracle is confirmed by a real HF `AutoModelForCausalLM.generate(..., do_sample=false, use_cache=true)` run on the same model directory as the Rust E2E gate.

The Rust E2E accepts the known HF-confirmed RTX 5090 and A800 hash pairs for this narrow shape, but the comparison gate remains stricter: HF, host-staged, and NCCL JSON must be dumped from the same model/runtime and compared with `--require-all-exact`. This refresh does not claim a model-runtime improvement, a manual-loop root cause, or a transport issue.

- HF / host-staged / NCCL comparison: `all_token_text_exact`.
- Token SHA256: `d05a7b0f0ac6435fb51040582a337d8b6d72844dd61194daa1b3090fa0e16ce8`.
- Text SHA256: `4aaafbe4b3a46bc5b9ab5ea8d09d5fad71225006c2e234e87a928e3265b387c6`.
- Generated text: `, I am a 20 year old female and I have been having a`.

The historical graph-readiness diagnostic before the #275/#276 device-scratch work was rerun on 2026-06-04 on the same model snapshot and 2x A800-SXM4-80GB:

- `full_decode_capture_ready=false`;
- `status=blocked_full_decode_path`;
- NCCL blockers reported: per-call dense-exchange allocation/sync, host-side route iteration, host-side contribution accumulation, combine H2D copy, combine allocation, combine sync, and combine D2H copy;
- optional `nccl_cuda_graph_smoke=captured_replayed_verified`;
- smoke capture mode: `thread_local`;
- smoke result: `captured=true`, `replayed=true`, `verified=true`, `rank0_value=3.0`, `rank1_value=3.0`, with no `capture_error`, `replay_error`, or `verification_error`.

The attribution table below is the retained 2026-05-30 batch `1/4/8` diagnostic snapshot.

| Backend | Batch | Decode steps | Mean shared decode us | GPU event samples | GPU failures | Route / collective counters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| host-staged | 1 | 15 | 69128.2 | 5056 | 0 | `dispatch_calls=416`, `combine_calls=416`, `local=1284`, `remote=1212` |
| NCCL | 1 | 15 | 592081.8 | 5888 | 0 | `nccl_exchange_calls=416`, `nccl_combine_calls=416`, `local=1284`, `remote=1212` |
| host-staged | 4 | 15 | 256487.9 | 13024 | 0 | `dispatch_calls=494`, `combine_calls=494`, `local=5136`, `remote=4848` |
| NCCL | 4 | 15 | 985367.7 | 14012 | 0 | `nccl_exchange_calls=494`, `nccl_combine_calls=494`, `local=5136`, `remote=4848` |
| host-staged | 8 | 15 | 502502.3 | 23648 | 0 | `dispatch_calls=598`, `combine_calls=598`, `local=10272`, `remote=9696` |
| NCCL | 8 | 15 | 1560818.4 | 24844 | 0 | `nccl_exchange_calls=598`, `nccl_combine_calls=598`, `local=10272`, `remote=9696` |

For batch `4` and `8`, every same-prompt row reported `same_prompt_rows_exact=true` and the same token/text hashes as the HF comparison run.

## Claim Boundary

This report proves only that the covered DeepSeek-V2-Lite EP2 greedy path still produces the expected token/text hashes and that the current runtime observed the listed CPU-side sections, selected CUDA event sections, NVTX markers when enabled, route counts, and dense collective counts. It does not prove serving throughput, sparse dispatch readiness, multi-node behavior, or production EP readiness.
