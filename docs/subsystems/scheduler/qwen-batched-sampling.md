# Qwen Batched Sampling

> **TL;DR:** Issue #284 migrates Qwen3/Qwen3.5 mixed greedy/non-greedy token selection to one batched FlashInfer sampling call per step, while keeping greedy/effectively-greedy rows on batched argmax; Qwen gates, nsys composition proof, and same-host Qwen3 greedy TPOT A/B pass.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routed this task to Qwen model docs, scheduler docs, kernels docs, and profiling playbook.
  - `docs/models/qwen3/roadmap.md` - records #284 as phase 2 after batched greedy sampling.
  - `docs/models/qwen35/roadmap.md` - treats sampling as shared with Qwen3 and gated by HF logits tests.
  - `docs/subsystems/scheduler/scheduler.md` - identifies the old O(batch) per-row sampler as the scheduler hot-path issue.
  - `docs/playbooks/profiling-guide.md` - nsys trace must use release builds and `--cuda-graph-trace=node` for composition proof.
- **Relevant history**:
  - `docs/models/kimi-k2/sampling.md` - Kimi already uses the desired split: greedy rows stay on argmax, non-greedy rows use one batched FlashInfer pass.
  - Memory reminder: benchmark evidence should be labeled as correctness gate, diagnostic snapshot, HTTP pressure, or production-serving claim.
- **Plan**:
  1. Rework shared Qwen token selection to compact non-greedy rows and call `gpu_sample_batch_into` once per step with a u64 seed.
  2. Remove legacy single-row sampling wrappers and the custom single-block softmax kernel.
  3. Update Qwen3/Qwen3.5 scratch, seed flow, tests, kernel plans, and roadmap docs.
  4. Validate with local Rust/CUDA checks, then run remote HF gates and an nsys composition proof.
- **Risks / open questions**:
  - Qwen3.5 still has single-logit prefill/unified helper paths; they need one-row batched sampling without changing request lifecycle behavior.
  - Kimi still uses FlashInfer top1 helpers, so top1 scratch sizing must remain available even after removing the single-row sampler.

## Execution Log

### Step 1: Shared sampling API and kernel cleanup

- Removed the legacy single-row FlashInfer wrapper surface from Rust exports.
- Deleted the custom single-block `logits_to_probs_kernel` and `gpu_sample_flashinfer_cuda` entry point from `openinfer-kernels/csrc/shared/flashinfer_sampling.cu`.
- Kept the batched `gpu_sample_batch_flashinfer_cuda` path and moved top1 row-state scratch sizing to `flashinfer_top1_row_states_bytes_cuda()`.

### Step 2: Qwen decode token selection

- Qwen3 plan/executor paths now carry one `sample_seed: u64` per step instead of per-row `f32` seed material.
- Qwen3 and Qwen3.5 batch decode use `select_batch_tokens_into`: greedy rows go through indexed batched argmax, non-greedy rows compact into one batched FlashInfer sampling call.
- Qwen3.5 single-logit prefill/unified helpers now use `argmax` for greedy params and a one-row `gpu_sample_batch_into` call for non-greedy params.
- `top_p <= 1 / vocab` is treated as effectively greedy in the shared selector. That preserves the mathematical single-token nucleus semantics and avoids FlashInfer top-p boundary behavior on high-entropy rows.

### Step 3: Public wrappers, tests, and docs

- Server ops/tests/benchmarks no longer re-export or call the deleted single-row sampler.
- Kimi top1 scratch-size calls were renamed to `flashinfer_top1_row_states_bytes` while leaving Kimi sampling behavior unchanged.
- Kernel plans, roadmap docs, and the scheduler index now describe #284 as the batched Qwen sampling path.
- `gpu_sample_batch_flashinfer_cuda` chooses the matching FlashInfer primitive for a compacted batch: plain sampling when no top-k/top-p filter is active, TopP when only top-p is active, and TopKTopP when any row uses top-k.
- A read-only Codex review caught an earlier Qwen3.5 unified decode gap: active decode rows were still sampled through per-row extracted logits. The final path leaves decode logits in `graph_state.buffers.logits` and calls `select_tokens_batch_varied` once for the active decode batch.

### Step 4: Validation

- Local static gates passed: `cargo fmt --all --check`, `git diff --check`, and the old-symbol grep for `gpu_sample_flashinfer_cuda`, `logits_to_probs_kernel`, `u64::from(...to_bits)`, and `FLASHINFER_TOPK_ROW_STATES_BYTES`.
- Remote RTX 5090 / CUDA 12.8 gates passed:
  - `cargo check --release -p openinfer-core -p openinfer-kernels`
  - `cargo check --release -p openinfer-qwen3-4b`
  - `cargo check --release -p openinfer-qwen35-4b --features qwen35-4b`
  - `cargo check --release -p openinfer-server --bench ops_bench`
  - `cargo test --release -p openinfer-kernels batch_sampling_top_p_only_small_nucleus_collapses_to_argmax -- --nocapture`
  - `cargo test --release -p openinfer-qwen3-4b --tests --no-run`
  - `cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate -- --nocapture`
  - `cargo test --release -p openinfer-qwen3-4b --test sampling_behavior -- --nocapture`
  - `cargo test --release -p openinfer-qwen35-4b --features qwen35-4b --test hf_golden_gate -- --nocapture`
  - `cargo test --release -p openinfer-qwen35-4b --features qwen35-4b --test sampling_behavior -- --nocapture`
  - `cargo test --release -p openinfer-qwen35-4b --features qwen35-4b unified_step_decode_matches_graph_decode -- --nocapture`
  - `cargo test --release -p openinfer-qwen35-4b --features qwen35-4b --test e2e_scheduler -- --nocapture`
- Qwen3.5 scheduler e2e now alternates greedy and sampled requests in the concurrent phase, so the real decode batch covers mixed token-selection rows.
- Qwen3.5 `sampling_behavior` now proves `temperature` / `top_k` / `top_p` still steer the scheduler path, `top_k=1` and tiny `top_p` collapse to greedy, and hot sampling still varies across runs.
- `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --export=sqlite` on the Qwen3.5 mixed scheduler test showed:
  - `logits_to_probs_kernel`: 0
  - `gpu_sample_flashinfer_cuda`: 0
  - `gather_cast_logits_f32_kernel`: 103
  - `OnlineSoftmaxMapKernel`: 103
  - `OnlineSoftmaxReduceKernel`: 103
  - `TopKTopPSamplingFromProbKernel`: 103
- Qwen3 greedy TPOT A/B on the same validation host, same model, same `request --prompt-len 4096 --output-len 64 --warmup 5 --iters 20 --seed 42` workload:
  - upstream/main steady TPOT p50: `6.492853ms`; p99: `6.590101ms`; output hash: `83f4c3f2614d57b5`.
  - patched steady TPOT p50: `6.523550ms`; p99: `6.686876ms`; output hash: `83f4c3f2614d57b5`.
  - p50 delta: `+0.47%`, below the 2% regression gate.

## Debrief

- **Outcome**: The legacy single-row FlashInfer sampler and custom one-block softmax are removed from the runtime surface. Qwen3/Qwen3.5 batch decode now share the compacted non-greedy row path through `gpu_sample_batch_into`, and greedy rows remain on indexed batched argmax.
- **Pitfalls encountered**:
  - The validation host initially had Qwen3.5 weights but no Qwen3-4B weights. Direct Hugging Face download failed on the host, so the Qwen3-4B snapshot was fetched through ModelScope before running the Qwen3 gates.
  - Qwen3.5 on the validation GPU required a Triton 3.4 Python for feature builds; older Triton rejected the SM target before the patch was exercised.
  - A Qwen3 test helper initially missed the `RngExt` import needed for `rng.random()`. The full Qwen3 integration compile caught it.
  - Qwen3 `sampling_behavior` caught a real regression: using TopKTopP for a top-p-only tiny nucleus did not match greedy on high-entropy rows. The final patch routes top-p-only batches through the matching FlashInfer primitive and treats `top_p <= 1 / vocab` as argmax.
  - Qwen3.5 unified decode needed a second pass after review because it still extracted decode logits row-by-row before sampling. The final version samples active decode rows from the graph logits buffer in one batched selector call.
- **Lessons learned**:
  - For scheduler hot-path changes, make the e2e test exercise mixed params directly; otherwise a passing greedy e2e does not prove the #284 acceptance path.
  - Keep nsys evidence as kernel composition proof, not a throughput claim. The current evidence proves the old per-row softmax wrapper is gone and the batched FlashInfer sequence is used for mixed Qwen3.5 decode.
