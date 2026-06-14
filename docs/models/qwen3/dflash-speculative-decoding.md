# DFlash Speculative Decoding

> **TL;DR:** Qwen3-4B DFlash is wired end-to-end for greedy TP1 serving with native Rust/CUDA drafter, target verifier, INFO acceptance logs with `committed_tokens`, profiling-only timing, verifier-span/full-draft HF regret gate, request-local draft scratch, reusable pending-context buffer, config-derived DFlash memory reserve, DFlash side-state byte-budget admission, transactional speculative KV rollback, per-request DFlash prefill capture, chunked-prefill DFlash context continuity checks, and DFlash small-N cublasLt tuning. Speculative protocol types live in `openinfer-qwen3-4b/src/speculative.rs`, and DFlash lane state lives in `openinfer-qwen3-4b/src/executor/dflash_lane.rs`. Draft PR: #380. Latest local 5070 Ti PR-head `vllm bench serve` results: Spec-Bench bs=1 149.32 tok/s (1.66x), Spec-Bench c4 330.42 tok/s (1.09x), random 1024/128 bs=1 136.09 tok/s (1.57x), random c4 349.50 tok/s (1.29x). Post-hardening smoke: Spec-Bench c4 n=12 completed 12/12 at 368.71 tok/s and logged four concurrent DFlash requests in one wave. 5090 OpenInfer bs=1 reaches 251.48 tok/s on Spec-Bench (1.50x); upstream vLLM 0.22.1 supports Qwen3 DFlash and reaches 289.57 tok/s on the same Spec-Bench (1.78x vs vLLM baseline), with native `vllm bench serve` acceptance metrics. The current OpenInfer DFlash path is not CUDA-graph captured: local nsys shows draft eager at ~2.7-2.9 ms/step, ~98-106 kernel launches/step, dominated by GEMM/GEMV. Multi-active DFlash is enabled for all eligible greedy requests, but the draft side is still per-request serial; target verification is batched.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - Qwen3 model-line docs and scheduler/runtime docs are the relevant routing points.
  - `docs/models/qwen3/dflash-model-download.md` - confirms `z-lab/Qwen3-4B-DFlash-b16` is local at `/data/models/Qwen3-4B-DFlash-b16`.
  - `docs/models/qwen3/roadmap.md` - Qwen3 is the mature line; new decode behavior must sit under correctness/perf gates.
  - `docs/subsystems/scheduler/scheduler.md` - current runtime is single GPU-owner thread, FCFS prefill priority, paged KV, bucket CUDA Graph decode, unified prefill+decode.
  - `/data/models/Qwen3-4B-DFlash-b16/config.json` - DFlash drafter shape: `block_size=16`, 5 layers, hidden 2560, 32 Q heads, 8 KV heads, selected target layers `[1, 9, 17, 25, 33]`, mask token `151669`.
  - `/data/models/Qwen3-4B-DFlash-b16/dflash.py` - generation contract: prefill target, use target hidden states, draft a block, verify with target, accept matching prefix plus one posterior token, crop target/drafter KV.
  - `openinfer-qwen3-4b/src/scheduler/plan.rs` - current plan space is `Prefill`, `Decode`, `Unified`; pure decode calls `execute_decode`.
  - `openinfer-qwen3-4b/src/unified_forward.rs` - decode rows already use prefill-style attention metadata as `qo_len=1`, which is the shape to generalize for verifier spans.
  - `kvbm/kvbm-logical/src/integrations/scheduled.rs` - lower layer already supports `schedule_speculative` and `apply_speculative`, including partial accept and LIFO release of excess capacity.
- **Relevant history**:
  - `docs/subsystems/scheduler/scheduler.md` records the FlashInfer paged metadata invariants; speculative views must continue to expose exact page rows, never raw assigned blocks with surplus generation capacity.
  - `docs/models/qwen3/accuracy-gate.md` is the existing truth pattern for Qwen3 logits; DFlash should add a verifier-span logits/regret gate rather than rely on exact spec-on/spec-off greedy text identity.
- **Plan**:
  1. Add a shared speculative KV lifecycle surface in `openinfer-kv-cache`, backed by kvbm's existing `schedule_speculative` / `apply_speculative`.
  2. Add a model-owned speculative verifier API in Qwen3 executor that forwards verify spans and returns accepted tokens plus logits/hidden features needed by drafters.
  3. Port the DFlash drafter loader/forward path natively from the downloaded safetensors/custom Python, reusing target embeddings/lm head.
  4. Wire scheduler policy conservatively: enable DFlash first for pure greedy decode requests with no LoRA and no pending prefill, then broaden only after gates prove correctness and throughput.
  5. Add gates: EOS/max_tokens/context-window behavior, prefix-cache interaction, acceptance histogram, target/drafter/verifier timing, and serving throughput vs baseline.
- **Risks / open questions**:
  - DFlash's Python implementation depends on target hidden states for selected layers; Qwen3 forward currently only returns logits, so hidden-state taps must be added without polluting normal decode hot paths.
  - DFlash drafter has its own KV cache. The first correct native path may run eager before CUDA Graph capture; graphing belongs after verifier-span correctness and acceptance are proven.
  - The README speedups are for specific stacks/workloads. Local expected speedup must be measured against current OpenInfer Qwen3-4B baseline, not copied from the paper.
  - Exact spec-on/spec-off greedy text parity is not a stable gate for this implementation because the verifier uses multi-token target prefill while baseline decode uses the single-token decode path. Use target logits/regret gates plus scheduler/KV lifecycle gates for correctness; exact text mismatches at low-margin choices need separate investigation before becoming a release blocker.

## Design Notes

- **Verify span**: a speculative verifier step forwards `N` token positions for a request. In DFlash, `N=block_size`: the current dangling token plus `block_size - 1` draft candidates. The target posterior at the first mismatch supplies one bonus token.
- **Accepted tokens**: the sequence passed to `apply_speculative` is the newly produced output tokens for that step: matched draft tokens followed by the posterior bonus token. Its length equals the number of target KV positions that are now valid, leaving the last token dangling for the next step.
- **Abstraction boundary**: KV cache exposes generic speculative schedule/view/apply. Qwen3 executor owns verifier forward and sampling/acceptance. DFlash owns draft generation. Scheduler owns policy.
- **Code boundary**:
  - `openinfer-qwen3-4b/src/speculative.rs` owns speculative draft/verify protocol types and acceptance-prefix construction.
  - `openinfer-qwen3-4b/src/executor/dflash_lane.rs` owns DFlash per-lane request state, target-hidden context append, draft execution, INFO acceptance logs, and profiling timing.
  - `openinfer-qwen3-4b/src/executor/dflash_prefill.rs` owns DFlash prefill eligibility and chunk-finalization policy.
  - `openinfer-qwen3-4b/src/executor/speculative_exec.rs` owns Qwen3 executor-side speculative draft/verify scheduling and KV apply.
  - `openinfer-qwen3-4b/src/executor/{lifecycle,model_executor,worker}.rs` split executor construction/offload lifecycle, runtime trait implementation, and worker-lane plumbing; every `src/executor/*.rs` file is now under 1k lines.
  - `openinfer-qwen3-4b/src/dflash.rs` owns the DFlash model weights, request-local draft scratch, reusable pending-context buffer, DFlash-specific small-N cublasLt tuning, and GPU forward kernels.
- **Initial policy**: DFlash is opt-in and greedy-only. Unsupported combinations fail closed or route clearly to baseline, not silently change semantics.
- **Concurrency boundary**: when DFlash is configured, every eligible active greedy request may enter speculative decode. Pending prefill chunks still run before the next draft step so new requests can capture DFlash hidden context and join the speculative set. DFlash side-state admission is byte-budgeted from the config-derived reserve, so short concurrent requests can all enter while long requests queue instead of OOMing. The current multi-active implementation drafts serially per request, then batches the target verifier; this is correct and measurable, but not the final throughput shape.

## Execution Log

### Step 1: Download and inspect DFlash artifact
- Artifact: `/data/models/Qwen3-4B-DFlash-b16`.
- `config.json` confirms Qwen3-compatible geometry and DFlash-specific fields:
  - `block_size = 16`
  - `dflash_config.mask_token_id = 151669`
  - `dflash_config.target_layer_ids = [1, 9, 17, 25, 33]`
  - `architectures = ["DFlashDraftModel"]`
- `model.safetensors` has 58 BF16 tensors: 5 Qwen3-like layers plus `fc.weight`, `hidden_norm.weight`, and `norm.weight`. It has no lm-head; DFlash uses the target lm-head.
- Python `spec_generate` verifies the contract: target prefill → DFlash block draft → target block verify → prefix match → posterior bonus → crop caches.

### Step 2: Expose speculative KV lifecycle
- Added `RequestKv::schedule_speculative`, `RequestKv::speculative_view`, and `RequestKv::apply_speculative` in `openinfer-kv-cache`.
- `speculative_view(num_draft_tokens)` uses the same exact page-row contract as `prefill_view` / `decode_view`: it covers `kv_position + num_draft_tokens` and does not expose extra eagerly-held generation blocks.
- Added lifecycle tests:
  - `speculative_view_covers_verify_span`
  - `speculative_partial_accept_releases_excess_capacity`
- Verification:
  ```bash
  cargo fmt --all --check
  cargo test --release -p openinfer-kv-cache --test lifecycle speculative -- --nocapture
  ```
  Both commands passed.

### Step 3: Add Qwen3 target verifier API
- Added low-level runtime types:
  - `SpeculativeVerifyStepItem`
  - `SpeculativeVerifyPlan`
  - `SpeculativeVerifyRequestResult`
  - `SpeculativeVerifyResult`
- Added `Qwen3Executor::execute_speculative_verify`.
- Implementation:
  - schedules each request with `RequestKv::schedule_speculative(verify_span_len)`
  - builds exact `speculative_view` rows
  - runs a prefill-style target forward with `echo=true` to get all-position logits
  - greedily samples target posterior tokens
  - computes the accepted draft prefix plus posterior bonus
  - commits via `RequestKv::apply_speculative(&accepted_tokens)`
- Added unit tests for acceptance semantics:
  - `speculative_verify_accepts_matching_prefix_plus_posterior_bonus`
  - `speculative_verify_all_match_still_adds_block_end_posterior`
- Verification:
  ```bash
  cargo fmt --all --check
  cargo test --release -p openinfer-qwen3-4b executor::tests::speculative_verify --lib -- --nocapture
  cargo check --release -p openinfer-qwen3-4b --tests
  ```
  All commands passed.

### Step 4: Native DFlash drafter and scheduler wiring
- Added `openinfer-qwen3-4b/src/dflash.rs`:
  - loads `fc`, `hidden_norm`, `norm`, and the 5 DFlash transformer layers from `model.safetensors`
  - keeps a per-request DFlash KV cache and pending target hidden-state context
  - reuses the target embedding/lm-head path for block logits
- Added selected hidden-state capture in Qwen3 prefill/verify for DFlash target layer IDs.
- Added scheduler/executor speculative decode stages:
  - DFlash-ready request tracking after prefill hidden capture
  - `SpeculativeDraft` to build the draft block
  - `SpeculativeVerify` to run the target verifier and apply accepted tokens
  - `DecodeEffect::{EmitManyAndContinue, EmitManyAndFinish}` for multi-token emission
- Server CLI:
  - `--dflash-draft-model-path <PATH>`
  - rejects tensor parallel use with DFlash
  - disables prefix cache while DFlash is enabled, because hidden-state capture currently needs the full target suffix.
- Memory reservation: DFlash weights plus a config-derived request-state/scratch reserve are held out before Qwen3 KV budgeting. The reserve-polish local startup reserved 4034.1 MB total for DFlash and left 1028 Qwen3 KV blocks on the 16 GiB 5070 Ti.

### Step 5: Edge-case fixes and local gates
- Fixed speculative verification near `max_tokens`: draft token IDs are clamped to the remaining output budget before scheduling the target verify span. The crash signature before the fix was `KvView pages must exactly cover seq_len=49`.
- Tightened DFlash startup/config invariants:
  - DFlash draft/target config checks now reject mismatched hidden/vocab/head geometry, non-increasing target layer IDs, too-small block sizes, and out-of-vocab mask token IDs.
  - The downloaded-model config test is gated by `OPENINFER_TEST_MODEL_PATH` and `OPENINFER_DFLASH_TEST_MODEL_PATH`, and skips cleanly on machines without local weights.
- Tightened serving behavior:
  - DFlash prefill context capture requires exactly one supported request in the prefill step.
  - DFlash hidden-state append now checks that captured rows exactly match the scheduled token span.
  - Speculative verify context recording now checks request/result count, request IDs, and missing DFlash state before returning a successful worker result.
  - Multi-request prefill can capture DFlash hidden context per request. Unsupported rows are dropped from the DFlash side path without poisoning supported greedy rows in the same prefill batch.
  - Pending prefill chunks take priority over speculative decode, so a long speculative active request does not block new request prefill.
  - DFlash request drop removes side-channel metric/timing maps as well as drafter request state.
- Fixed chunked-prefill DFlash context handling:
  - `dflash_prefill_supported` now rejects prefix-cache hits (`cached_tokens > 0`) even though the server disables prefix cache when DFlash is enabled. This keeps the path fail-closed if a future config accidentally re-enables cache hits without full hidden-state replay.
  - Non-final prefill chunks keep the DFlash request state pending instead of dropping it. The final chunk marks the request DFlash-ready.
  - DFlash pending context append validates `pending_context_len == chunk_start` before accepting a captured hidden-state chunk. A discontinuous chunk sequence now fails early instead of silently drafting from an incomplete target context.
  - The DFlash HF golden gate now also forces a chunked-prefill path and checks 144 verifier positions.
- Added/ran gates:
  ```bash
  cargo fmt --all --check
  cargo check --release -p openinfer-server
  cargo test --release -p openinfer-qwen3-4b executor::tests::speculative_verify --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b scheduler::plan::tests::speculative_verify_items_clamp_to_remaining_output_budget --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b dflash_prefill_capture_requires_single_supported_request --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b speculative_plan_runs_only_after_pending_prefill_is_drained --lib -- --nocapture
  cargo test --release -p openinfer-kv-cache --test lifecycle speculative -- --nocapture
  OPENINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/models/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b dflash::tests::downloaded_dflash_config_matches_qwen3_4b --lib -- --nocapture
  OPENINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/models/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate dflash_speculative_verify_matches_hf_argmax_regret_gate -- --nocapture
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo build --release -p openinfer-server
  ```
  All commands passed locally.
- Latest local DFlash HF regret gate after the chunked-prefill fix:
  ```bash
  OPENINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/models/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate dflash_speculative_verify_matches_hf_argmax_regret_gate -- --nocapture
  ```
  It passed and checked 144 target positions.
- Added a core sampling equivalence gate for the all-greedy contiguous argmax fast path:
  ```bash
  cargo test --release -p openinfer-core all_greedy_contiguous_argmax_matches_indexed_greedy_rows --lib -- --nocapture
  ```
  It builds 8193-vocab logits, places maxima across multiple 4096-token tiles plus a cross-tile tie, runs the all-greedy contiguous path, then forces the mixed indexed path and checks greedy-row equality.

### Step 5.5: Speculative abstraction cleanup
- Moved speculative draft/verify protocol types and acceptance construction out of `executor.rs` into `openinfer-qwen3-4b/src/speculative.rs`.
- Moved DFlash lane state and per-request lifecycle into `openinfer-qwen3-4b/src/executor/dflash_lane.rs`.
- `SpeculativeDraftStepItem` has a public constructor for external runtime/integration callers, while `execute_speculative_draft` and `execute_speculative_verify` fail early when DFlash is not loaded or the request has not completed DFlash prefill context capture.
- `execute_speculative_verify` validates every request before any speculative KV scheduling, so an unsupported mixed request cannot dirty KV state before failing.
- Split the executor surface after the DFlash integration:
  - `executor.rs` now keeps shared request/result types, worker step dispatch, low-level sampling/logprob helpers, and tuning helpers.
  - `executor/lifecycle.rs` holds constructors, public wrappers, KV-offload save/load helpers, and `run_step`.
  - `executor/model_executor.rs` holds the `ModelExecutor` impl and LoRA load/unload policy.
  - `executor/worker.rs` holds `LocalQwen3Lane`, `StepCommand`, `WorkerStepOutcome`, and `RankWorker`.
  - `executor/dflash_prefill.rs` and `executor/speculative_exec.rs` hold the DFlash-specific policy added by this work.
  - File sizes after the split: `executor.rs` 830 lines, `model_executor.rs` 668, `lifecycle.rs` 584, `worker.rs` 469, `dflash_lane.rs` 355.
- Added a DFlash verifier-span HF regret gate in `openinfer-qwen3-4b/tests/hf_golden_gate.rs`:
  - Builds a real DFlash-enabled executor.
  - Runs prefill hidden capture for fixed HF-golden prompts.
  - Runs `execute_speculative_verify` with 16-token teacher-forced spans.
  - Checks 144 target verifier positions against HF top-K using the same regret tolerance as the main Qwen3 logits gate, including one forced chunked-prefill request.
  - Also runs 8 native DFlash draft → target verify steps, checks the first verifier posterior for each against HF top-K, validates `accepted_tokens` against the produced draft span, and requires at least one accepted draft candidate. This guards the DFlash draft column-offset contract without freezing exact draft token IDs across GPUs.
- Verification:
  ```bash
  cargo fmt --all --check
  cargo check --release -p openinfer-qwen3-4b --lib
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  OPENINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/models/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate dflash_speculative_verify_matches_hf_argmax_regret_gate -- --nocapture
  ```
  All commands passed locally.
- After adding the native full draft → verify assertions, the same DFlash HF gate passed and checked 152 target positions.
- The full serial HF gate also passed with the full-path DFlash assertions:
  ```bash
  OPENINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/models/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate -- --nocapture --test-threads=1
  ```
  It covered DFlash full draft → verify, DFlash verifier-span, Qwen3 eager/cached replay, and Qwen3 CUDA Graph padded/cached replay buckets.
- The same full serial HF gate passed again after the config-derived DFlash memory reserve and effective-context admission changes; the DFlash verifier checked 152 target positions, and Qwen3 eager/cached/CUDA-graph padded replay gates all remained within tolerance.

### Step 6: bs=1 serving performance on local 5070 Ti
- Baseline server command:
  ```bash
  ./target/release/openinfer --model-path /data/models/Qwen3-4B --served-model-name Qwen3-4B --port 8010 --max-prefill-tokens 1024 --no-prefix-cache
  ```
- DFlash server command:
  ```bash
  ./target/release/openinfer --model-path /data/models/Qwen3-4B --served-model-name Qwen3-4B --port 8010 --dflash-draft-model-path /data/models/Qwen3-4B-DFlash-b16 --max-prefill-tokens 1024
  ```
- `vllm bench serve`, greedy bs=1 / `--max-concurrency 1`, measured output throughput:

| Dataset | Prompt count | Output len | Baseline tok/s | DFlash tok/s | Speedup | Baseline mean TPOT | DFlash mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spec-Bench | 20 | 64 | 89.97 | 144.69-147.02 | 1.61-1.63x | 10.81 ms | 6.42-6.47 ms |
| ShareGPT sample | 20 | 64 | 88.23 | 126.00 | 1.43x | 10.99 ms | 7.50 ms |
| Random | 20 | 128 | 85.69 | 130.78 | 1.53x | 10.99 ms | 6.93 ms |

- Final current Spec-Bench run after the executor split and 8193-vocab argmax equivalence gate (`target/dflash-bench/dflash-final-current-specbench-n20-out64.json`, server log `target/dflash-bench/dflash-server-final-current.log`):
  - completed requests: 20 / 20
  - output throughput: 146.46 tok/s
  - mean/median/p99 TPOT: 6.41 / 6.66 / 8.88 ms
  - mean TTFT: 32.92 ms
  - total input/output tokens: 3918 / 1280
  - this matches the previous chunkfix run within run-to-run noise, so the later cleanup and gate hardening did not regress bs=1 Spec-Bench throughput.
- Hotpath-cleanup Spec-Bench run after removing per-draft params/random allocation and reusing the DFlash block-token host buffer (`target/dflash-bench/dflash-hotpath-cleanup-specbench-n20-out64.json`, server log `target/dflash-bench/dflash-server-hotpath-cleanup.log`):
  - completed requests: 20 / 20
  - output throughput: 147.02 tok/s
  - mean/median/p99 TPOT: 6.42 / 6.67 / 8.90 ms
  - mean TTFT: 30.59 ms
  - total input/output tokens: 3918 / 1280
  - DFlash verify logs and acceptance were byte-for-byte aligned at the aggregate level with the final-current run: 491 rows, 769 / 6524 accepted/verified draft tokens, 1260 committed tokens.
- Host-buffer-cleanup Spec-Bench run after reusing `SamplingScratch::out_host` for DFlash greedy argmax D2H (`target/dflash-bench/dflash-host-buffer-cleanup-specbench-n20-out64.json`, server log `target/dflash-bench/dflash-server-host-buffer-cleanup.log`):
  - completed requests: 20 / 20
  - output throughput: 145.75 tok/s
  - mean/median/p99 TPOT: 6.42 / 6.67 / 8.90 ms
  - mean TTFT: 34.51 ms
  - total input/output tokens: 3918 / 1280
  - DFlash verify logs: 491 rows, accepted/verified draft tokens 769 / 6524 = 0.1179, committed 1260 tokens, timing fields stayed `-1.000`.
- Reserve-polish Spec-Bench run after deriving DFlash request-state memory reserve from the draft config and reducing the effective admission context limit by one DFlash block (`target/dflash-bench/dflash-reserve-polish-specbench-n20-out64.json`, server log `target/dflash-bench/dflash-server-reserve-polish.log`):
  - startup reserve: weights 1074.9 MB, request/scratch extra 2959.3 MB, total 4034.1 MB.
  - resulting local KV cache: 1028 blocks / 2313 MB; frontend `max_model_len=16432`.
  - DFlash effective target context limit: `40944` tokens (`40960 - block_size 16`), so requests that would overflow the drafter cache are rejected at admission instead of failing mid-prefill.
  - completed requests: 20 / 20
  - output throughput: 146.33 tok/s
  - mean/median/p99 TPOT: 6.42 / 6.67 / 8.89 ms
  - mean TTFT: 32.51 ms
  - total input/output tokens: 3918 / 1280
  - DFlash verify logs: 491 rows, accepted/verified draft tokens 769 / 6524 = 0.1179, committed 1260 tokens, timing fields stayed `-1.000`.
- Latest post chunked-prefill-fix Spec-Bench run (`target/dflash-bench/dflash-chunkfix-specbench-n20-out64.json`, server log `target/dflash-bench/dflash-server-chunkfix.log`):
  - completed requests: 20 / 20
  - output throughput: 146.90 tok/s
  - mean/median/p99 TPOT: 6.42 / 6.67 / 8.90 ms
  - request 19 used a 1188-token prompt context in DFlash after chunked prefill and completed 64 output tokens.

- Result files live under `target/dflash-bench/`, including:
  - `baseline-specbench-n20-out64.json`
  - `dflash-specbench-n20-out64.json`
  - `dflash-final-specbench-n20-out64.json`
  - `dflash-nosync-specbench-n20-out64.json`
  - `dflash-post-abstraction-specbench-n20-out64.json`
  - `dflash-scratch-specbench-n20-out64.json`
  - `dflash-tuned-specbench-n20-out64.json`
  - `dflash-pending-buffer-specbench-n20-out64.json`
  - `dflash-contiguous-argmax-specbench-n20-out64.json`
  - `dflash-chunkfix-specbench-n20-out64.json`
  - `dflash-final-current-specbench-n20-out64.json`
  - `dflash-hotpath-cleanup-specbench-n20-out64.json`
  - `dflash-host-buffer-cleanup-specbench-n20-out64.json`
  - `dflash-reserve-polish-specbench-n20-out64.json`
  - `baseline-sharegpt-n20-out64.json`
  - `dflash-sharegpt-n20-out64.json`
  - `baseline-random-in1024-out128-n20.json`
  - `dflash-random-in1024-out128-n20.json`

### Step 7: Acceptance, timing, and graph/profiling status
- Added INFO logs for every DFlash verify step:
  - accepted draft tokens
  - verified draft tokens
  - committed tokens (`committed_tokens`, before scheduler stop-token suppression)
  - cumulative acceptance rate
  - last/average draft ms in profiling mode, otherwise `-1.000`
  - last/average verify ms in profiling mode, otherwise `-1.000`
  - pending and committed DFlash context lengths
- Default serving does not synchronize for DFlash timing. Profiling timings and NVTX ranges require:
  ```bash
  OPENINFER_QWEN3_DFLASH_NVTX=1
  ```
- No-sync local Spec-Bench bs=1 run (`target/dflash-bench/dflash-nosync-specbench-n20-out64.json` and `target/dflash-bench/dflash-server-nosync.log`) showed:
  - completed requests: 20 / 20
  - output throughput: 144.69 tok/s
  - mean/median/p99 TPOT: 6.47 / 6.70 / 9.71 ms
  - DFlash verify log rows: 494
  - accepted/verified draft tokens: 766 / 6544 = 0.117
  - committed tokens: 1260, or 2.55 tokens/speculative step
  - timing fields: `draft_ms=-1.000`, `verify_ms=-1.000`, confirming default INFO logs do not force sync.
- Post-abstraction no-sync Spec-Bench bs=1 run (`target/dflash-bench/dflash-post-abstraction-specbench-n20-out64.json` and `target/dflash-bench/dflash-server-post-abstraction.log`) showed no throughput regression:
  - completed requests: 20 / 20
  - output throughput: 144.60 tok/s
  - mean/median/p99 TPOT: 6.47 / 6.69 / 9.72 ms
  - DFlash verify log rows: 494
  - accepted/verified draft tokens: 766 / 6544 = 0.117
  - timing fields remained `draft_ms=-1.000`, `verify_ms=-1.000`.
- Draft scratch cleanup:
  - `DFlashRequestState` now owns reusable draft buffers for context projection, block hidden states, tail K/V, MLP intermediates, and output logits.
  - The first draft for a long prompt can grow the scratch to that prompt-context length; steady speculative steps reuse the same device buffers and only update `seq_len` metadata.
  - The scratch-only Spec-Bench bs=1 run (`target/dflash-bench/dflash-scratch-specbench-n20-out64.json`) completed 20/20 requests at 144.89 tok/s with mean/median/p99 TPOT 6.47 / 6.69 / 9.71 ms, 494 log rows, acceptance 766 / 6544 = 0.117, and no timing sync.
- DFlash cublasLt tuning:
  - Worker bind now extends target decode GEMM tuning with DFlash-specific shapes: `fc` `[M=2560, K=12800, N=1..16]` and DFlash K/V projection tails `[M=1024, K=2560, N=17..32]`.
  - Startup logs the tuned ranges:
    `Qwen3 DFlash cublasLt tuned: fc M=2560 K=12800 N=1..16, kv M=1024 K=2560 N=17..32`.
  - Tuned Spec-Bench bs=1 run (`target/dflash-bench/dflash-tuned-specbench-n20-out64.json` and `target/dflash-bench/dflash-server-tuned.log`) completed 20/20 requests at 146.05 tok/s with mean/median/p99 TPOT 6.47 / 6.68 / 9.68 ms.
  - Tuned DFlash verify logs: 495 rows, accepted/verified draft tokens 765 / 6559 = 0.1166, committed 1260 tokens, timing fields `-1.000`.
- Pending-context buffer cleanup:
  - `DFlashRequestState` now keeps target hidden context in a reusable `DFlashPendingContext` instead of allocating a fresh `HiddenStates` on each append and allocating a merged buffer when chunks accumulate.
  - The buffer tracks active length separately from capacity, grows to the first long prompt/chunk sequence, clears by resetting active length, and reuses the same device allocation for steady speculative steps.
  - This removes a request-local allocation/merge source and makes the pending-context pointer stable after its first growth. Other graph blockers remain: K/V cache write offsets, RoPE positions, and attention `kv_len` still vary by step.
  - Pending-buffer Spec-Bench bs=1 run (`target/dflash-bench/dflash-pending-buffer-specbench-n20-out64.json` and `target/dflash-bench/dflash-server-pending-buffer.log`) completed 20/20 requests at 145.97 tok/s with mean/median/p99 TPOT 6.46 / 6.67 / 9.67 ms.
  - Pending-buffer DFlash verify logs: 495 rows, accepted/verified draft tokens 765 / 6559 = 0.1166, committed 1260 tokens, timing fields stayed `-1.000`.
- Contiguous greedy argmax cleanup:
  - Added a Rust wrapper for the existing `argmax_batch_bf16_split_cuda` non-indexed path and taught `select_batch_tokens_into` to use it whenever every row is greedy and logits rows are contiguous.
  - This removes the per-call row-index vector construction and H2D row-index copy from all-greedy paths, including DFlash draft and DFlash verifier token selection. It does not change the two argmax kernels themselves, so the launch-heavy draft profile is expected to remain dominated by GEMM/GEMV.
  - Added `openinfer-core` GPU equivalence coverage comparing contiguous all-greedy output against indexed greedy rows from the mixed path at 8193 vocab, including cross-tile maxima and a cross-tile tie.
  - Contiguous-argmax Spec-Bench bs=1 run (`target/dflash-bench/dflash-contiguous-argmax-specbench-n20-out64.json` and `target/dflash-bench/dflash-server-contiguous-argmax.log`) completed 20/20 requests at 145.98 tok/s with mean/median/p99 TPOT 6.47 / 6.68 / 9.69 ms.
  - Contiguous-argmax DFlash verify logs: 495 rows, accepted/verified draft tokens 765 / 6559 = 0.1166, committed 1260 tokens, timing fields stayed `-1.000`.
- Chunked-prefill fix no-sync run:
  - `target/dflash-bench/dflash-chunkfix-specbench-n20-out64.json` and `target/dflash-bench/dflash-server-chunkfix.log` completed 20/20 Spec-Bench requests at 146.90 tok/s with mean/median/p99 TPOT 6.42 / 6.67 / 8.90 ms.
  - DFlash verify logs: 491 rows, accepted/verified draft tokens 769 / 6524 = 0.1179, committed 1260 tokens, or 2.57 tokens/speculative step.
  - The longest prompt row had `draft_context_tokens=1188` followed by `draft_committed_context=1188` on the next speculative step, confirming the chunked prefill context was preserved across chunks.
  - Timing fields stayed `draft_ms=-1.000`, `verify_ms=-1.000`.
- Final current no-sync run:
  - `target/dflash-bench/dflash-final-current-specbench-n20-out64.json` and `target/dflash-bench/dflash-server-final-current.log` completed 20/20 Spec-Bench requests at 146.46 tok/s with mean/median/p99 TPOT 6.41 / 6.66 / 8.88 ms.
  - DFlash verify logs: 491 rows, accepted/verified draft tokens 769 / 6524 = 0.1179, committed 1260 tokens, or 2.57 tokens/speculative step.
  - Max observed `draft_context_tokens=1188`; max `draft_committed_context=1245`.
  - Timing fields stayed `draft_ms=-1.000`, `verify_ms=-1.000`, so default INFO logging remains no-sync.
- Hotpath-cleanup no-sync run:
  - `target/dflash-bench/dflash-hotpath-cleanup-specbench-n20-out64.json` and `target/dflash-bench/dflash-server-hotpath-cleanup.log` completed 20/20 Spec-Bench requests at 147.02 tok/s with mean/median/p99 TPOT 6.42 / 6.67 / 8.90 ms.
  - DFlash verify logs: 491 rows, accepted/verified draft tokens 769 / 6524 = 0.1179, committed 1260 tokens, or 2.57 tokens/speculative step.
  - Max observed `draft_context_tokens=1188`; max `draft_committed_context=1245`.
  - Timing fields stayed `draft_ms=-1.000`, `verify_ms=-1.000`.
- Host-buffer-cleanup no-sync run:
  - Reused `SamplingScratch::out_host` for the greedy contiguous argmax D2H path, removing the last per-DFlash-draft host allocation in token selection.
  - `target/dflash-bench/dflash-host-buffer-cleanup-specbench-n20-out64.json` and `target/dflash-bench/dflash-server-host-buffer-cleanup.log` completed 20/20 Spec-Bench requests at 145.75 tok/s with mean/median/p99 TPOT 6.42 / 6.67 / 8.90 ms.
  - DFlash verify logs: 491 rows, accepted/verified draft tokens 769 / 6524 = 0.1179, committed 1260 tokens, or 2.57 tokens/speculative step.
  - Max observed `draft_context_tokens=1188`; max `draft_committed_context=1245`.
  - Timing fields stayed `draft_ms=-1.000`, `verify_ms=-1.000`.
- DFlash memory/admission polish:
  - The old fixed 1 GiB DFlash extra reserve only covered short serving probes. The reserve is now derived from the DFlash config and covers one request's worst-case drafter KV cache, pending target-hidden context, long-context projection scratch, tail K/V scratch, block scratch, logits scratch, and a 256 MiB allocator margin, with 1 GiB as a floor.
  - Local startup after this change logged `weights=1074.9 MB, extra=2959.3 MB, total=4034.1 MB`; the resulting KV pool has 1028 Qwen3 blocks and still runs the standard bs=1 Spec-Bench profile at 146.33 tok/s.
  - Qwen3 executor metadata now exposes an effective max context of `target_max_position_embeddings - dflash_block_size` when DFlash is loaded. This keeps DFlash cache overflow as an admission-time rejection instead of a GPU prefill failure.
- Toxic-reviewer follow-up caught a real shared-path regression in the first hotpath-cleanup patch: `select_batch_tokens_into` briefly required `params.len() == logits.seq_len`, which conflicts with Qwen3 CUDA Graph decode where logits are bucket-padded (`bs=5 -> bucket 8`, `bs=9 -> bucket 16`).
  - Fixed the sampling contract so `logits.seq_len` may include padding rows while `params.len()` names the real rows to consume; `params.len() > logits.seq_len` still fails early.
  - `LocalQwen3Lane::select_step_tokens` now sizes sampling scratch from `max(logits.seq_len, params.len())`, preserving padded-bucket capacity.
  - Added `all_greedy_contiguous_argmax_ignores_cuda_graph_padding_rows`, which uses 8193-vocab logits with 8 padded rows and 5 real greedy params.
  - Verification:
    ```bash
    cargo test --release -p openinfer-core all_greedy_contiguous_argmax --lib -- --nocapture
    OPENINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/models/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate -- --nocapture --test-threads=1
    ```
    Both passed. The serial HF gate covered the DFlash verifier-span gate plus Qwen3 `batched cuda-graph (9 padded)`, `batched cuda-graph (5 padded)`, and `batched cuda-graph cached replay (5)`.
- Local timing smoke (`target/dflash-bench/dflash-server-timing.log`, Spec-Bench bs=1) showed:
  - 94 DFlash verify steps
  - accepted/verified draft tokens: 158 / 1297 = 0.1218
  - committed tokens: 252, or 2.68 tokens/speculative step
  - draft mean/median/p90/max: 2.740 / 2.692 / 2.740 / 4.907 ms
  - verify mean/median/p90/max: 13.546 / 13.414 / 13.870 / 14.271 ms
  - steady draft rows (`draft_context_tokens <= 16`) averaged 2.697 ms; cold prompt-context rows averaged 3.728 ms.
- Added optional NVTX ranges with `OPENINFER_QWEN3_DFLASH_NVTX=1`:
  - `qwen3.dflash.draft`
  - `qwen3.dflash.verify`
- Local nsys trace (`target/dflash-bench/nsys-dflash-full-with-request.nsys-rep`) on 2 Spec-Bench prompts:
  - 47 draft ranges: mean 2.874 ms, min 2.717 ms, max 5.020 ms
  - 47 verify ranges: mean 13.938 ms, min 13.616 ms, max 14.375 ms
  - all captured kernels had null `graphId`; the DFlash speculative path is eager, not CUDA-graph replayed.
  - draft range kernel sum: 117.6 ms total, 2.50 ms/range, ~106 kernel launches/range
  - draft range kernel time split:
    - GEMM/GEMV: 110.7 ms, 94.1%
    - single-prefill attention: 2.8 ms, 2.4%
    - norm/MLP elementwise: 1.5 ms, 1.3%
    - hidden copy/embedding: 0.8 ms, 0.7%
    - cublasLt split-K reduce: 0.8 ms, 0.7%
    - DFlash qk-norm+RoPE: 0.5 ms, 0.4%
    - argmax: 0.5 ms, 0.4%
- Pending-buffer nsys trace (`target/dflash-bench/nsys-dflash-pending-buffer.nsys-rep` / `.sqlite`) on 2 Spec-Bench prompts:
  - 47 draft ranges: mean/median/min/max 2.758 / 2.677 / 2.616 / 5.137 ms by NVTX range.
  - 47 verify ranges: mean/median/min/max 13.953 / 14.100 / 13.606 / 14.424 ms by NVTX range.
  - all kernels inside both DFlash draft and DFlash verify ranges had null `graphId`; DFlash speculative draft and target verify are still eager.
  - draft range kernel sum: 117.0 ms total, 2.49 ms/range, ~98 kernel launches/range.
  - draft range kernel time split:
    - CUTLASS/cublasLt GEMM/GEMV: 104.4 ms, 89.3%
    - FlashInfer single-prefill attention: 2.8 ms, 2.4%
    - SM120 nvjet GEMM variants: 2.4 ms, 2.0%
    - DFlash qk-norm+RoPE: 0.5 ms, 0.4%
    - hidden/context copies: 0.8 ms, 0.7%
    - argmax: 0.5 ms, 0.4%
  - profiling-mode server timings from the same run: 47 rows, draft mean/median/min/max 2.895 / 2.796 / 2.742 / 4.854 ms, verify mean/median/min/max 13.760 / 13.751 / 13.636 / 13.953 ms. Steady rows with `draft_context_tokens <= 16` averaged 2.852 ms; the 2 cold prompt-context rows averaged 3.858 ms.
- KernelWiki references checked for the graph/fill question:
  - `sources/prs/vllm/PR-17484.md` and `sources/prs/vllm/PR-17668.md` show the relevant upstream pattern: move q/o projections and rotary work into the CUDA-graph region to reduce CPU overhead.
  - `sources/prs/vllm/PR-25984.md` records a speculative-decode attention path that refactors metadata for optimized spec-as-decode FlashInfer-MLA kernels.
  - `sources/prs/flashinfer/PR-969.md` and `sources/prs/flashinfer/PR-2244.md` reinforce the same constraint for our DFlash path: graph readiness requires removing host API/sync overhead and making all per-step metadata capturable or graph-updatable.
- Local GPU metrics/occupancy could not be collected: nsys reported `ERR_NVGPUCTRPERM` on the local 5070 Ti, and `ncu` is not installed locally. This means local evidence proves eager launch structure and time split, but not SM occupancy percentage.

### Step 8: 5090 validation and vLLM comparison
- Remote machine: `5090` (`host-192-168-172-86`), 8x RTX 5090, idle at probe time.
- Applied the patch to remote branch `feat/qwen3-dflash-mtp-codex` in `~/develop/xingming/pegainfer`.
- Remote build/test commands:
  ```bash
  export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-13.1
  cargo fmt --all --check
  cargo check --release -p openinfer-server
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo test --release -p openinfer-kv-cache --test lifecycle speculative -- --nocapture
  cargo build --release -p openinfer-server
  ```
  All commands passed on 5090.
- After adding DFlash draft scratch and DFlash-specific cublasLt tuning, the same patch was re-synced to 5090 and the same fmt/check/lib/KV/build command set passed again. The DFlash config/model-weight gate skipped there because `OPENINFER_DFLASH_TEST_MODEL_PATH` is not available on the box.
- After adding the pending-context reusable buffer and contiguous greedy argmax fast path, the final patch was re-synced to 5090 and this command set passed:
  ```bash
  export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-13.1
  cargo fmt --all --check
  cargo check --release -p openinfer-server
  cargo test --release -p openinfer-kernels ops::sampling --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo test --release -p openinfer-kv-cache --test lifecycle speculative -- --nocapture
  cargo build --release -p openinfer-server
  git diff --check
  ```
  The remote DFlash config/model-weight gate still skipped because `OPENINFER_DFLASH_TEST_MODEL_PATH` is not available on the box.
- After the chunked-prefill DFlash context fix, the patch was re-synced to 5090 again and the same fmt/check/sampling/Qwen3-lib/KV/build/diff command set passed on `host-192-168-172-86` with CUDA 13.1. At that pre-download point, real-weight DFlash checks skipped.
- After splitting the executor files under 1k lines and adding the core contiguous-vs-indexed greedy argmax equivalence test, the patch was re-synced to 5090 again and this command set passed:
  ```bash
  export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-13.1
  cargo fmt --all --check
  cargo check --release -p openinfer-server
  cargo test --release -p openinfer-core all_greedy_contiguous_argmax_matches_indexed_greedy_rows --lib -- --nocapture
  cargo test --release -p openinfer-kernels ops::sampling --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo test --release -p openinfer-kv-cache --test lifecycle speculative -- --nocapture
  cargo build --release -p openinfer-server
  git diff --check
  ```
  At that pre-download point, real-weight DFlash checks skipped.
- After removing per-DFlash-draft CPU allocations for params/random vectors and reusing the block-token host buffer, the patch was re-synced to 5090 again and this command set passed:
  ```bash
  export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-13.1
  cargo fmt --all --check
  cargo check --release -p openinfer-server
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo build --release -p openinfer-server
  git diff --check
  ```
  At that pre-download point, real-weight DFlash checks skipped.
- After fixing the padded-logits sampling contract from the toxic-reviewer follow-up, the patch was re-synced to 5090 again and this command set passed:
  ```bash
  export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-13.1
  cargo fmt --all --check
  cargo test --release -p openinfer-core all_greedy_contiguous_argmax --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo build --release -p openinfer-server
  git diff --check
  ```
  At that pre-download point, real-weight DFlash checks skipped.
- After adding the full native draft → verify assertions to the DFlash HF gate, the patch was re-synced to 5090 again and this command set passed:
  ```bash
  export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-13.1
  cargo fmt --all --check
  cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate dflash_speculative_verify_matches_hf_argmax_regret_gate -- --nocapture
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo build --release -p openinfer-server
  git diff --check
  ```
  The remote integration test compiled; at that pre-download point, real-weight checks skipped because the default relative model paths did not exist on 5090.
- After reusing `SamplingScratch::out_host` for DFlash greedy argmax D2H, the patch was re-synced to 5090 again and this command set passed:
  ```bash
  export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-13.1
  cargo fmt --all --check
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo build --release -p openinfer-server
  git diff --check
  ```
  At that pre-download point, real-weight DFlash checks skipped.
- After adding config-derived DFlash memory reserve and DFlash-aware effective context admission, the patch was re-synced to 5090 again and this command set passed:
  ```bash
  export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-13.1
  cargo fmt --all --check
  cargo test --release -p openinfer-qwen3-4b dflash_context_limit_reserves_one_draft_block --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b dflash_request_state_reserve_covers_long_context_scratch --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo build --release -p openinfer-server
  git diff --check
  ```
  At that pre-download point, real-weight DFlash checks skipped.
- After the user explicitly approved placing the DFlash artifact on 5090, downloaded `z-lab/Qwen3-4B-DFlash-b16` to `/data/Qwen3-4B-DFlash-b16` using the box proxy from `.bashrc` (`http://172.17.0.1:1081`).
  - `model.safetensors`: 1074860568 bytes
  - `config.json`, `README.md`, `dflash.py`, `modeling_dflash.py`, `utils.py`, and `assets/` are present.
- Real-weight 5090 gates now pass:
  ```bash
  OPENINFER_TEST_MODEL_PATH=/data/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b dflash::tests::downloaded_dflash_config_matches_qwen3_4b --lib -- --nocapture
  OPENINFER_TEST_MODEL_PATH=/data/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate dflash_speculative_verify_matches_hf_argmax_regret_gate -- --nocapture
  ```
  Both passed; the DFlash HF gate checked 152 target positions.
- 5090 OpenInfer bs=1 `vllm bench serve`, greedy, `--max-concurrency 1`:

| Dataset | Prompt count | Output len | Baseline tok/s | DFlash tok/s | Speedup | Baseline mean TPOT | DFlash mean TPOT | Result files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spec-Bench | 20 | 64 | 167.34 | 251.48 | 1.50x | 5.83 ms | 3.79 ms | `/data/dflash-bench/{baseline,dflash}-5090-specbench-n20-out64-c1.json` |
| ShareGPT sample | 20 | 64 | 144.56 | 144.50 | 1.00x | 6.67 ms | 6.67 ms | `/data/dflash-bench/{baseline,dflash}-5090-sharegpt-n20-out64-c1.json` |
| Random 1024/128 | 20 | 128 | 154.00 | 168.83 | 1.10x | 6.18 ms | 5.59 ms | `/data/dflash-bench/{baseline,dflash}-5090-random-in1024-out128-n20-c1.json` |

- OpenInfer concurrency probe: Spec-Bench `--max-concurrency 4`, 40 prompts, output 64:
  - baseline: 558.18 tok/s, mean TPOT 6.83 ms
  - DFlash: 540.23 tok/s, mean TPOT 6.91 ms
  - This was measured before the multi-active default policy landed, and is retained as historical evidence that the first bs=1-only policy left batch-concurrency behavior under-optimized.
- 5090 DFlash INFO logs:
  - `/data/dflash-bench/dflash-server-5090.log`: 485 rows, accepted/verified draft tokens 775 / 6478 = 0.1196, committed 1260 tokens, 2.60 committed tokens/step, no old `emitted=` field.
  - `/data/dflash-bench/dflash-server-5090-r2.log`: 477 rows, accepted/verified draft tokens 669 / 6795 = 0.0985, committed 1146 tokens, 2.40 committed tokens/step, no old `emitted=` field.
- 5090 nsys DFlash Spec-Bench c1 trace:
  - report: `/data/dflash-bench/nsys-5090-dflash-specbench.nsys-rep`
  - sqlite: `/data/dflash-bench/nsys-5090-dflash-specbench.sqlite`
  - result: `/data/dflash-bench/dflash-5090-nsys-specbench-n4-out64-c1.json`
  - nsys-wrapped run completed 4/4 at 253.73 tok/s, mean TPOT 3.71 ms.
  - Profiling-mode INFO rows show steady draft around 1.52-1.59 ms and verify around 8.0-8.6 ms, with tail verify rows near 9.3-9.9 ms.
- Upstream vLLM support check on 5090:
  - `.venv/bin/vllm --version` reports `0.22.1`.
  - `vllm serve --help=all` lists `--spec-method dflash`.
  - Python import finds `vllm.model_executor.models.qwen3_dflash`.
  - vLLM DFlash startup logs resolve `DFlashDraftModel`, build `SpeculativeConfig(method='dflash', model='/data/Qwen3-4B-DFlash-b16', num_spec_tokens=16)`, and warn that DFlash uses the v1 model runner because runner v2 does not yet support DFlash parallel drafting.
- 5090 vLLM 0.22.1 bs=1 `vllm bench serve`, greedy, `--max-num-seqs 1`, `--max-model-len 40944`, `--max-concurrency 1`:

| Dataset | Prompt count | Output len | vLLM baseline tok/s | vLLM DFlash tok/s | Speedup | Baseline mean TPOT | DFlash mean TPOT | DFlash acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spec-Bench | 20 | 64 | 162.48 | 289.57 | 1.78x | 5.97 ms | 3.07 ms | 2.62 |
| ShareGPT sample | 20 | 64 | 159.96 | 226.71 | 1.42x | 5.98 ms | 4.01 ms | 2.06 |
| Random 1024/128 | 20 | 128 | 156.85 | 258.13 | 1.65x | 6.03 ms | 3.49 ms | 2.64 |

  Result files:
  - `/data/dflash-bench/vllm-baseline-5090-specbench-n20-out64-c1-r2.json`
  - `/data/dflash-bench/vllm-dflash-5090-specbench-n20-out64-c1-r2.json`
  - `/data/dflash-bench/vllm-baseline-5090-sharegpt-n20-out64-c1.json`
  - `/data/dflash-bench/vllm-dflash-5090-sharegpt-n20-out64-c1.json`
  - `/data/dflash-bench/vllm-baseline-5090-random-in1024-out128-n20-c1.json`
  - `/data/dflash-bench/vllm-dflash-5090-random-in1024-out128-n20-c1.json`
  Server logs:
  - `/data/dflash-bench/vllm-dflash-5090-server.log`
  - `/data/dflash-bench/vllm-baseline-5090-server.log`
- 5090 profiler state:
  - `nsys` is installed and supports GPU metrics on all 8 GPUs.
  - `ncu` is not installed.

### Step 9: Review follow-up fixes
- Toxic reviewer follow-up found two real issues after the reserve/admission polish:
  - A mid-prompt chunk could start a fresh DFlash state after an earlier mixed/chunked prefill step had dropped state, then fail on the hidden-context continuity check. Worker capture eligibility now uses real lane state: `chunk_start == 0` may start DFlash state, while `chunk_start > 0` captures hidden state only when a pending DFlash request state already exists. The worker returns `PrefillResult::dflash_context_captured_requests`, and the main executor uses the per-request result when deciding `MarkReady` / `KeepPending` / `Drop`.
  - DFlash INFO logs used the field name `emitted` for `accepted_tokens.len()`. That value is committed to request state before scheduler stop-token suppression, so the log field is now `committed_tokens`.
- Local validation after this fix:
  ```bash
  cargo fmt --all --check
  cargo test --release -p openinfer-qwen3-4b dflash_prefill --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b servable_len_reports_the_tighter_context_or_kv_limit --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo build --release -p openinfer-server
  OPENINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/models/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate dflash_speculative_verify_matches_hf_argmax_regret_gate -- --nocapture
  git diff --check
  ```
  All commands passed locally. The DFlash HF gate checked 152 target positions after this fix.
- Local Step 9 Spec-Bench bs=1 run (`target/dflash-bench/dflash-step9-specbench-n20-out64.json`, server log `target/dflash-bench/dflash-server-step9.log`) completed 20/20 requests:
  - output throughput: 148.34 tok/s
  - mean/median/p99 TPOT: 6.34 / 6.60 / 8.80 ms
  - mean TTFT: 31.65 ms
  - total input/output tokens: 3918 / 1280
  - DFlash verify logs: 491 rows, accepted/verified draft tokens 769 / 6524 = 0.1179, committed 1260 tokens, or 2.57 committed tokens/speculative step.
  - Default timing fields stayed `draft_ms=-1.000`, `verify_ms=-1.000`, and no DFlash verify row used the old `emitted=` field.

### Step 10: Draft PR and local multi-active validation
- Draft PR opened: <https://github.com/openinfer-project/openinfer/pull/380>.
- After user direction to remove the single-active restriction, scheduler policy now lets all eligible active greedy requests enter DFlash when the draft model is configured. The executor drafts each active request and batches target verification.
- Local Spec-Bench dataset materialized for vLLM 0.21:
  ```bash
  curl -L --fail --show-error https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl -o target/dflash-bench/spec_bench_question.jsonl
  wc -l target/dflash-bench/spec_bench_question.jsonl
  ```
  The file has 480 JSONL rows with the expected `turns` field.
- Latest local PR-head serving results on RTX 5070 Ti, greedy `vllm bench serve`, `--temperature 0`, OpenAI `/v1/completions` endpoint:

| Dataset | Concurrency | Prompts | Output len | Baseline tok/s | DFlash tok/s | Speedup | Baseline mean TPOT | DFlash mean TPOT | Result files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spec-Bench | 1 | 20 | 64 | 89.72 | 149.32 | 1.66x | 10.77 ms | 6.33 ms | `baseline-local-pr380-specbench-n20-out64-c1.json`, `dflash-local-pr380-specbench-n20-out64-c1-r2.json` |
| Spec-Bench | 4 | 40 | 64 | 303.92 | 330.42 | 1.09x | 12.09 ms | 10.95 ms | `baseline-local-pr380-specbench-n40-out64-c4.json`, `dflash-local-pr380-specbench-n40-out64-c4-r2.json` |
| Random 1024/128 | 1 | 20 | 128 | 86.61 | 136.09 | 1.57x | 10.86 ms | 6.63 ms | `baseline-local-pr380-random-in1024-out128-n20-c1.json`, `dflash-local-pr380-random-in1024-out128-n20-c1.json` |
| Random 1024/128 | 4 | 40 | 128 | 270.93 | 349.50 | 1.29x | 13.41 ms | 10.18 ms | `baseline-local-pr380-random-in1024-out128-n40-c4.json`, `dflash-local-pr380-random-in1024-out128-n40-c4.json` |

- DFlash server log for the local multi-active session: `target/dflash-bench/dflash-local-pr380-multiactive-server.log`.
  - The log contains DFlash rows for 180 internal request IDs (`0..179`) and shows multiple request IDs logging DFlash acceptance in the same decode wave, e.g. request `0/1/2/3` immediately after the first c4 batch.
  - Aggregate over the session log: 5529 DFlash rows, accepted/verified draft tokens 9651 / 75682 = 0.1275, committed 15180 tokens, or 2.75 committed tokens/speculative row.
  - The aggregate includes the discarded overlapping Spec-Bench c1/c4 attempt, so it is only evidence of multi-active coverage and acceptance logging, not a per-dataset performance number.

### Step 11: Multi-active hardening after review
- Toxic review of the multi-active change found two release blockers:
  - `Prefill` failure while DFlash-ready active requests exist only targeted pending requests, then cleared `active`, so active requests could disappear without an error event or executor cleanup.
  - Removing the single-active restriction let each eligible request allocate DFlash side state without aggregate memory admission.
- Fixes:
  - `failure_targets_for(Prefill)` now includes active requests as well as pending requests, so a failed active+pending prefill step errors and drops every touched request.
  - `RequestKv::revert_schedule` exposes kvbm's scheduled-state rollback. `execute_speculative_verify_impl` now rolls back every request whose speculative KV was scheduled if later scheduling, worker execution, or result-shape validation fails.
  - DFlash memory reserve is split into total reserved bytes and request-state budget bytes. Scheduler admission estimates each DFlash-supported request's true side-state footprint from the DFlash config and `prompt + max_tokens + block_size`; the global allocator margin/floor is reserved once, not charged per request. Requests that exceed the remaining side-state budget stay deferred instead of OOMing after admission.
  - DFlash prefill hidden capture is now per request. A mixed prefill batch can still capture target hidden states; unsupported rows drop only their own DFlash side state, while supported greedy rows keep progressing toward `dflash_ready`.
- Local validation:
  ```bash
  cargo fmt --all --check
  cargo test --release -p openinfer-kv-cache --test lifecycle speculative -- --nocapture
  cargo test --release -p openinfer-qwen3-4b dflash_prefill --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b admission_respects_speculative_state_budget_for_supported_requests --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b admission_counts_only_active_requests_with_speculative_state --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b dflash_short_request_footprint_does_not_include_global_allocator_floor --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b prefill_failure_with_active_targets_active_and_pending_requests --lib -- --nocapture
  cargo test --release -p openinfer-qwen3-4b --lib -- --nocapture
  cargo build --release -p openinfer-server
  OPENINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/models/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate dflash_speculative_verify_matches_hf_argmax_regret_gate -- --nocapture
  ```
  These passed locally; the HF gate checked 152 DFlash verifier target positions.
- Post-hardening serving smoke on RTX 5070 Ti after fixing per-request side-state footprint:
  - Server log: `target/dflash-bench/dflash-pr380-hardening-smoke-r2-server.log`
  - Result file: `target/dflash-bench/dflash-pr380-hardening-smoke-r2-specbench-n12-c4.json`
  - Command shape: Spec-Bench, greedy, output 64, `--num-prompts 12`, `--max-concurrency 4`, OpenAI `/v1/completions`, served model `Qwen3-4B`.
  - Startup log: `Qwen3 DFlash memory reserve: weights=1074.9 MB, extra=2959.3 MB, state_budget=2690.8 MB, total=4034.1 MB`.
  - Result: 12/12 successful, 768 generated tokens, 368.71 tok/s, mean/median/p99 TPOT 9.54 / 9.44 / 11.36 ms, mean TTFT 70.73 ms.
  - Log evidence: DFlash INFO rows include four live request IDs in the same wave (for example `0/1/2/3` at the start and `8/9/10/11` later) with cumulative acceptance around 0.12.

## Debrief

- **Outcome**: DFlash is integrated end-to-end for the Qwen3-4B TP1 greedy path, with native drafter loading/forward, reusable draft scratch, per-draft CPU allocation cleanup, reusable D2H host output buffer for greedy token selection, config-derived request-state memory reserve, DFlash-aware effective context admission, DFlash side-state byte admission, transactional speculative KV rollback, per-request DFlash prefill capture, DFlash-specific small-N cublasLt tuning, speculative scheduler stages, hidden-state capture, acceptance/timing logs, NVTX profiling ranges, local and 5090 standard bs=1 serving measurements, local multi-active serving measurements, verifier-span plus full draft→verify HF gates, executor files split under 1k lines, 5090 real-weight gates, a vLLM 0.22.1 DFlash baseline for comparison, and draft PR #380.
- **Review**: toxic reviewer previously returned Ready for review after the executor split and 8193-vocab argmax equivalence test, then follow-ups caught the chunked-prefill lifecycle / `emitted` log-field issues fixed in Step 9 and the multi-active failure/admission issues fixed in Step 11. Non-blockers called out in review are intentionally documented here: DFlash is not in CUDA Graph, draft execution is not batched/fused, and HTTP/scheduler-level DFlash E2E coverage can be broadened later.
- **Pitfalls encountered**:
  - Draft tokens must be clamped to remaining `max_tokens` before building the verify span; otherwise speculative KV views can request a span larger than the committed output budget.
  - `vllm bench serve` no longer forces greedy requests by default. DFlash only engages when requests are greedy, so benchmark commands must include `--temperature 0` when the server default is not enough.
  - Per-step timing must not synchronize the hot path by default. Use no-sync INFO logs for acceptance/shape observability and `OPENINFER_QWEN3_DFLASH_NVTX=1` when profiling timing/graphs.
  - Multi-active DFlash is correct enough to benchmark, but the draft path is still per-request serial. This preserves the user's "DFlash on means eligible requests use DFlash" semantics, while leaving real batched/fused draft execution as the next performance target.
  - Chunked prefill needs continuity checks on the hidden-state side channel. Dropping state on a non-final chunk or accepting a mismatched `chunk_start` corrupts the drafter context without touching the normal target KV path.
  - DFlash request memory is not just the 1.1 GB safetensors file. Long prompts can grow drafter KV, pending hidden-state context, and projection scratch by multiple GiB, so the KV pool reserve must be config-derived, the admission context limit must leave one draft block of headroom, and scheduler admission must budget DFlash side state across concurrent requests.
  - Local NVTX capture-range triggering did not match reliably; a bounded full nsys capture plus sqlite post-processing produced the useful draft/verify breakdown.
  - Local nsys GPU metrics are blocked by permissions. The 5090 supports nsys GPU metrics and now has the draft model, so exact graph/kernel attribution should be collected there before final bs=1 optimization claims.
  - vLLM DFlash materially outperforms the current OpenInfer DFlash path on ShareGPT and random bs=1, not just Spec-Bench. That points to a real implementation/policy gap rather than benchmark noise.
- **Lessons learned**:
  - DFlash's `acceptance_length + 1` maps naturally to kvbm's `apply_speculative(&accepted_tokens)`: accepted draft prefix plus posterior bonus.
  - The current speedup is from amortizing one target verifier step over ~2.7 committed tokens, not from a fully optimized drafter. The drafter is eager, launch-heavy, and GEMM/GEMV dominated.
  - DFlash draft graph capture requires more than wrapping the current function in `CudaGraphState`: pending context pointers, K/V cache write offsets, RoPE positions, and attention `kv_len` are dynamic today. They need stable scratch/device metadata or graph-exec parameter updates before capture would be correct.
  - DFlash verifier correctness must be tested through a DFlash-enabled executor. A bare Qwen3 executor does not own DFlash hidden-context state and should fail closed on speculative verify.
  - A production-quality next step is DFlash graph capture or a fused/smaller-launch draft path, plus aligning OpenInfer's request-state policy with vLLM's proven bs=1 behavior. Tuning acceptance policy alone will not answer the draft-kernel efficiency question.
- **Follow-ups**:
  - Optimize bs=1 first against the 5090 vLLM DFlash reference: inspect why OpenInfer ShareGPT barely enters or barely benefits from DFlash, then profile draft/verify graph capture and kernel fill.
  - Re-run the latest multi-active PR head on 5090 once the host is convenient again; local 5070 Ti shows positive c4 throughput, but 5090 is still the release performance gate.
  - Replace the per-request serial DFlash draft loop with a batched/fused draft path or graph-captured draft replay, then re-run the same Spec-Bench/random c1/c4 gates.
