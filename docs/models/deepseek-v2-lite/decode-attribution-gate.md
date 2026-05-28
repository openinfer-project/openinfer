# DeepSeek-V2-Lite EP2 Decode Attribution Gate

> **TL;DR:** DeepSeek-V2-Lite now has a narrow EP2 decode attribution report for the same correctness shape as the HF gate: `batch=1`, `prompt="Hello"`, `output_len=16`, host-staged backend, and NCCL backend. The report combines CPU-side attribution, selected CUDA event timing, optional NVTX ranges, and route/transfer counts; it is evidence for the next bottleneck decision, not a throughput or production EP claim.
>
> **Status:** Passing for the covered EP2 `Hello` / 16-token host-staged and NCCL attribution gate.

## Scope

This gate deliberately stays model-specific and shape-specific:

- Model: DeepSeek-V2-Lite.
- Shape: batch `1`, prompt `Hello`, prompt token ids `[17464]`, output length `16`.
- Backends: default host-staged EP2 and `PEGAINFER_DSV2_LITE_EP_BACKEND=nccl`.
- Accuracy oracle: the same generated token/text/hash gate used by `hf-accuracy-gate.md`.
- Attribution source: `DeepSeekV2LiteEp2Generator::generate_greedy_with_attribution`.
- GPU attribution source: CUDA events around selected stream sections in the explicit attribution path.
- NVTX source: set `PEGAINFER_DSV2_LITE_NVTX=1` to emit matching ranges for those selected sections during a profiler run.

Out of scope:

- sparse dispatch;
- pegainfer-comm / NVLink backend;
- multi-node or generic EP topology;
- batch > 1 or broader prompts;
- performance improvement or throughput claims.

## Report Shape

`dsv2_lite_ep2_decode_attribution` emits structured JSON:

- `report_type`, `model`, `phase`, `backend`, and fixed-shape `config`;
- nested `accuracy` with generated token ids, generated text, token sha256, and text sha256;
- CPU-side `timing` with total generation, the prefill-produced first output token, `per_output_token_us`, the 15 true decode-token samples for `output_len=16`, and latency stats;
- `gpu_timing`, `by_gpu_section`, and `by_gpu_call_site` with CUDA event timing for selected GPU/NCCL stream sections, plus a `failure_count` for event-timing failures that did not replace the token/text hash oracle;
- `by_section`, `by_op`, and `by_call_site` rollups in the same vocabulary family as the Qwen3 model report;
- `coverage` rows that distinguish CPU section timing, selected GPU event timing, optional NVTX ranges, and unclaimed throughput;
- `ep` counters for host-staged dispatch/combine and NCCL dense exchange/combine plus local/remote route counts.

Host-staged `dispatch_calls` / `combine_calls` count MoE layer invocations in the fixed greedy run. Host-staged `dispatch_elements` / `combine_elements` count selected routed hidden vectors, so the value is route count times hidden size. NCCL `exchange` and `combine` counters count the dense all-reduce calls and elements used by the current naive NCCL gate.

The GPU event rows are intentionally narrower than the CPU rows. They cover sections that enqueue device work or NCCL work on known streams, including projections, dense/shared/routed experts, host-to-device combine, NCCL dense exchange, and NCCL combine. They do not relabel pure host routing or host accumulation as GPU work, and the mixed `attention_host_path` stays CPU-side because it includes host attention assembly as well as internal GPU projections.

## Commands

Run the accuracy gate first, because attribution is not allowed to weaken the HF / host-staged / NCCL oracle:

```bash
mkdir -p target/accuracy/dsv2-lite-ep2

python tools/accuracy/hf_dump_dsv2_lite_ep2_greedy.py \
  --model-path models/DeepSeek-V2-Lite \
  --prompt Hello \
  --output-len 16 \
  --out target/accuracy/dsv2-lite-ep2/hf.json

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/host-staged.json \
  cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/nccl.json \
  cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

python tools/accuracy/compare_dsv2_lite_ep2_outputs.py \
  --hf target/accuracy/dsv2-lite-ep2/hf.json \
  --host-staged target/accuracy/dsv2-lite-ep2/host-staged.json \
  --nccl target/accuracy/dsv2-lite-ep2/nccl.json \
  --out target/accuracy/dsv2-lite-ep2/comparison.json \
  --require-all-exact
```

Then collect attribution for the same two pegainfer backends:

```bash
cargo run --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path models/DeepSeek-V2-Lite \
  --out target/accuracy/dsv2-lite-ep2/host-staged-attribution.json

PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
  cargo run --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path models/DeepSeek-V2-Lite \
  --out target/accuracy/dsv2-lite-ep2/nccl-attribution.json
```

For an Nsight Systems pass, run the same attribution command under the profiler and set `PEGAINFER_DSV2_LITE_NVTX=1`; the JSON `coverage` row then records `nvtx_ranges=emitted`. The NVTX labels are correlation markers for the selected GPU/NCCL sections, not timing evidence by themselves. Their wall-clock span can include CPU-side wrapper work, event setup, and synchronization around the section, so compare JSON `by_gpu_*` rows only with CUDA event timing, not with raw NVTX range duration.

## Environment Notes

The NCCL path depends on a runtime that supports the selected GPU. On newer GPUs, older NCCL runtimes may fail communicator initialization before the model-level comparison runs, for example with a shared-memory init error like:

```text
ncclMaxSharedMem 82240 exceeds device/fn maxSharedMem 79856
NCCL WARN Cuda failure 1 'invalid argument'
```

Use a newer NCCL runtime through the normal library path if the system runtime fails this way. The project code path should not change just to work around local NCCL installation age.

The HF oracle needs a Python environment that can load DeepSeek-V2-Lite with `trust_remote_code=True`. The helper script tolerates the model file's optional `flash_attn` import check when FlashAttention is not installed, but the HF environment remains separate from the Rust runtime claim: it is only the truth-source generator for the comparison JSON.

## Latest Validation

The full gate was last rerun on 2026-05-27 with DeepSeek-V2-Lite snapshot `604d5664dddd88a0433dbae533b7fe9472482de0`, `prompt="Hello"`, `output_len=16`, and 2x RTX 5090. The refreshed token/text oracle is confirmed by a real HF `AutoModelForCausalLM.generate(..., do_sample=false, use_cache=true)` run on the same model snapshot as the Rust E2E gate.

The earlier 2026-05-26 hash is superseded for the active gate because it was produced from a different local model snapshot. Treat only the hash below as the active oracle for snapshot `604d5664dddd88a0433dbae533b7fe9472482de0`. This refresh does not claim a model-runtime improvement, a manual-loop root cause, or a transport issue.

- HF / host-staged / NCCL comparison: `all_token_text_exact`.
- Token SHA256: `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225`.
- Text SHA256: `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347`.
- Host-staged attribution: `dispatch_calls=416`, `combine_calls=416`, `dispatch_elements=5111808`, `combine_elements=5111808`, `local_route_count=1284`, `remote_route_count=1212`.
- NCCL attribution: `nccl_exchange_calls=416`, `nccl_combine_calls=416`, `nccl_exchange_elements=851968`, `nccl_combine_elements=851968`, `local_route_count=1284`, `remote_route_count=1212`.
- GPU event timing: selected GPU/NCCL attribution sections are reported separately from CPU-side section timing; host-staged emitted `gpu_timing.sample_count=5056`, NCCL emitted `gpu_timing.sample_count=5888`, and both reported `gpu_timing.failure_count=0`.
- NVTX smoke: with `PEGAINFER_DSV2_LITE_NVTX=1`, host-staged emitted `nvtx_range_count=5056` and `coverage.nvtx_ranges=emitted`.

## Claim Boundary

This report proves only that the covered DeepSeek-V2-Lite EP2 greedy path still produces the expected token/text hashes and that the current runtime observed the listed CPU-side sections, selected CUDA event sections, NVTX markers when enabled, route counts, and dense collective counts. It does not prove serving throughput, sparse dispatch readiness, multi-node behavior, or production EP readiness.
