# DeepSeek V4 Decode Kernel Audit

**Created**: 2026-05-11
**Status**: complete

## TL;DR

Decode launch audit is complete for the current non-MoE slice. Kept changes cache the final logits F32 head-weight scratch and fuse the attention FP32 all-reduce tail into HC post while preserving the old BF16 rounding boundary. Exact DeepSeek V4 E2E remains `20/20`; the best comparable 1x32 clean bench improved from `98.80ms` to `91.84ms` with token hash `5f6c64b667f2abf5`. Several plausible launch-reduction experiments were rejected because exact parity alone did not produce stable TPOT wins.

## Preparation

- **Read**:
  - `docs/index.md` - found DeepSeek V4 decode work history and the active MoE AG/RS record.
  - `docs/projects/deepseek-moe-ag-rs.md` - confirmed decode MoE is not the focus for this slice and that HC prenorm fusion currently benchmarks at `98.80ms`.
  - `pegainfer-deepseek-v4/src/direct/worker.rs` - decode token flow is rank workers: token copy, embedding, all-reduce, HC expand, 43 block decodes, final logits, logits all-gather.
  - `pegainfer-deepseek-v4/src/runtime/block.rs` - per-layer decode structure is HC prenorm, attention, FP32 all-reduce, HC post, FFN HC prenorm, MoE AG/RS, HC post.
  - `pegainfer-deepseek-v4/src/runtime/core.rs` - final logits calls `hc_head`, final RMSNorm, and `deepseek_last_token_bf16_logits_cuda`.
  - `pegainfer-kernels/csrc/deepseek_v4/deepseek_hc.cu` - final logits currently allocates temporary F32 buffers, converts the full head weight every call, creates a cuBLAS handle, runs SGEMV, then frees everything.
- **Relevant history**:
  - `docs/projects/deepseek-moe-ag-rs.md` Step 12 rejected a hand-written HC mixes decode kernel because exact E2E passed but clean TPOT regressed to `115.10ms`.
- **Plan**:
  1. Add a launch-granularity decode checklist to this document so future work does not collapse back to local wrapper names.
  2. Change `deepseek_last_token_bf16_logits_cuda` to reuse per-device scratch for the F32 hidden vector, converted F32 head weight, and cuBLAS handle.
  3. Preserve the existing FP32 SGEMV math path: BF16 input/weight are still converted to F32 before dot product.
  4. Run local format/check, sync to 5090, run exact E2E, then run longer decode benches with `output_len >= 160`, `warmup >= 2`, `iters >= 3`.
- **Risks / open questions**:
  - The first token after load still pays the head-weight conversion; steady TPOT should improve more than first decode.
  - 5090 is temporarily returning `no available gateway`; remote E2E and bench will be run once the alias is reachable.

## Decode Launch Checklist

- **Per token, before layers**:
  - H2D token id copy
  - `embedding_batched_vocab_shard_cuda`
  - embedding NCCL all-reduce
  - `deepseek_hc_expand_cuda`
- **Per layer, attention prelude**:
  - `deepseek_hc_mixes_cuda`: BF16->F32 kernel, cuBLAS SGEMV, HC scale kernel
  - `deepseek_hc_pre_norm_from_mixes_cuda`
  - attention projection: FP8 wq_a act+GEMM, q RMSNorm, FP8 wq_b act+GEMM, head RMSNorm, FP8 wkv act+GEMM, kv RMSNorm
  - Q RoPE, KV RoPE, KV nope quant, KV cache row copy
- **ratio 0 attention layers**:
  - window top-k decode
  - indexed cache attention
  - output inverse RoPE, wo_a BF16 linear, wo_b FP8 act+GEMM
  - BF16->F32, NCCL all-reduce, F32->BF16
  - HC post
- **ratio 128 attention layers**:
  - non-overlap compressor decode every token
  - window top-k decode
  - after compressed history exists: compress top-k decode and concat top-k
  - on `(start_pos + 1) % 128 == 0`: compressed KV strided RoPE, nope quant, compressed KV cache copy
  - indexed cache attention, output projection, FP32 all-reduce wrapper, HC post
- **ratio 4 attention layers**:
  - overlap compressor decode every token
  - indexer q FP8 act+GEMM, indexer q RoPE, indexer q Hadamard FP4 quant
  - indexer overlap compressor decode every token
  - after compressed history exists: indexer weights BF16 linear, indexer score kernel, NCCL score all-reduce, indexer top-k, concat top-k
  - on `(start_pos + 1) % 4 == 0`: main compressed KV RoPE/quant/copy plus indexer compressed KV RoPE/Hadamard/copy
  - indexed cache attention, output projection, FP32 all-reduce wrapper, HC post
- **Per layer, FFN/MoE branch**:
  - FFN HC mixes and HC prenorm
  - MoE hidden all-gather; hash layers also gather token ids
  - hash or score gate
  - local mapping, expert-major expand
  - grouped FP4 w1 act+GEMM, grouped FP4 w3 act+GEMM, SwiGLU, grouped FP4 w2 act+GEMM
  - fused output reduce, F32 reduce-scatter
  - shared expert FP8 w1 act+GEMM, FP8 w3 act+GEMM, SwiGLU, FP8 w2 act+GEMM
  - routed/shared add, HC post
- **Per token, final logits**:
  - HC head mixes, head pre, head pre-output
  - final RMSNorm
  - final logits GEMV
  - logits NCCL all-gather
  - rank0 logits D2H

## Execution Log

### Step 1: Cache final logits F32 head weight
- Changed `deepseek_last_token_bf16_logits_cuda` in `pegainfer-kernels/csrc/deepseek_v4/deepseek_hc.cu`:
  - Reuses `DeepseekHcScratch::x_f32` for the last hidden vector conversion.
  - Adds per-device `logits_weight_f32` scratch keyed by source weight pointer and valid element count.
  - Reuses the existing per-device cuBLAS handle instead of creating/destroying one per token.
  - Keeps the original FP32 math path: BF16 hidden and BF16 head weight are converted to F32, then cuBLAS SGEMV computes F32 logits.
- Updated `pegainfer-kernels/KERNELS.md` to record the final logits scratch behavior.
- Local verification:
  - `cargo fmt --check` passed.
  - `git diff --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- 5090 exact verification:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- --model-path /data/DeepSeek-V4-Flash
```

- Result: `All 20 DeepSeek V4 exact cases passed`.
- Comparable short bench:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 32 --warmup 1 --iters 1 --seed 42
```

- Result: `steady_tpot_ms.avg = 93.14ms`, `p50 = 92.11ms`, `p95 = 99.18ms`, `first_decode_step_ms = 90.98ms`, `decode_tok_s = 10.74`.
- Baseline for the same 1x32 shape after HC prenorm fuse was `steady_tpot_ms.avg = 98.80ms`, so this is a `5.66ms/token` improvement on the short benchmark.

### Step 2: Longer decode reproducibility bench
- New longer bench shape:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42
```

- Reason: `output_len=160` crosses the ratio-128 compressed-attention boundary, so it is a better stability probe than 1x32.
- Round 1 result: `steady_tpot_ms.avg = 106.19ms`, `p50 = 107.07ms`, `p95 = 114.40ms`, `p99 = 116.75ms`, `first_decode_step_ms.avg = 100.54ms`, `decode_tok_s = 9.42`.
- Round 2 result: `steady_tpot_ms.avg = 102.89ms`, `p50 = 98.18ms`, `p95 = 118.52ms`, `p99 = 127.17ms`, `first_decode_step_ms.avg = 93.72ms`, `decode_tok_s = 9.72`.
- Observation: average TPOT differs by about `3.1%` across two clean runs and p99 varies more than the mean. This bench shape is useful for stability tracking, but more runs are needed before treating a single measurement as reproducible.
- Both long bench runs printed a rank-7 `NcclError` panic while aborting the communicator after scheduler exit. The process returned success and metrics were emitted, but shutdown cleanup should be investigated separately.
- Bench rigor note: always pass `--seed 42` explicitly, but do not treat seed equality as workload equality. DeepSeek V4 direct decode is greedy today; the generated token trace still must be compared because token ids affect hash routing, EPLB/expert balance, and therefore TPOT.

### Step 3: Fuse decode final HC head plus RMSNorm
- Tried a decode-only final-head fused kernel for `seq_len=1`.
- The experiment kept `deepseek_hc_mixes_cuda` unchanged so final head dot products still used the existing FP32 cuBLAS path, then fused `deepseek_hc_head_pre_cuda`, `deepseek_hc_pre_output_cuda`, and the following RMSNorm into one kernel.
- The fused kernel preserved the old BF16 rounding boundary between HC head output and RMSNorm, and non-`seq_len=1` shapes stayed on the old path.
- Verification:
  - local `cargo fmt --check` passed
  - local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed
  - 5090 full exact E2E passed: `All 20 DeepSeek V4 exact cases passed`
- Bench result:
  - 1x32 run 1: `steady_tpot_ms.avg = 100.69ms`
  - 1x32 run 2: `steady_tpot_ms.avg = 107.65ms`
- Result: reverted the code. Reducing two small launches did not offset replacing FlashInfer RMSNorm with the custom fused reduction kernel.

### Step 4: Reuse decode window top-k indices across layers
- Tried generating `window_topk_indices_decode` once per rank decode token in `run_decode_on_rank_lane`, then passing the shared `window_idxs/window_topk` through every decode block.
- Kept compressed/indexer top-k generation layer-local because those depend on ratio, compressed length, and scores.
- Expected launch change was up to 43 window-index launches/token/rank down to 1.
- Verification:
  - local `cargo fmt --check` passed
  - local `git diff --check` passed
  - local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed
  - 5090 full exact E2E passed: `All 20 DeepSeek V4 exact cases passed`
- Bench result:
  - 1x32: `steady_tpot_ms.avg = 98.80ms`
  - 1x160: `steady_tpot_ms.avg = 110.27ms`
- Result: reverted the runtime code. The launch reduction did not translate into TPOT improvement, likely because the saved window-index kernels are tiny and allocation/lifetime changes plus normal variance dominate.

### Step 5: Record generated token traces in bench output
- Added `generated_token_traces` to `bench_serving` metrics. Each measured iteration records:
  - `hash`: stable FNV-1a hash of generated token ids
  - `prefix`: first 16 generated token ids
  - `len`: generated token count
- Reason: DeepSeek V4 direct decode is greedy and the bench seed defaults to `42`, but generated token ids still matter for hash routing and EPLB/expert balance. Future TPOT comparisons should first check token traces match; otherwise a TPOT delta may be caused by a different expert-load sequence rather than the code change under test.
- Updated this document's benchmark commands to pass `--seed 42` explicitly.
- Verification:
  - local `cargo fmt --check` passed
  - local `git diff --check` passed
  - local `cargo check --release -p pegainfer-server --features deepseek-v4` passed
  - 5090 `bench_serving request --prompt-len 1 --output-len 32 --warmup 1 --iters 1 --seed 42` emitted `generated_token_traces`
- Trace from the verification run:
  - hash: `5f6c64b667f2abf5`
  - prefix: `[303, 1207, 1724, 993, 15238, 303, 36428, 58828, 303, 86532, 18048, 11301, 303, 75379, 1927, 5746]`

### Step 6: Fuse attention FP32 all-reduce tail into HC post
- Added `deepseek_hc_post_f32_branch_cuda` and `all_reduce_hidden_fp32_hc_post`.
- Kept attention output and NCCL all-reduce semantics unchanged: BF16 attention output is converted to F32, then NCCL sums F32.
- Replaced the old attention-tail `deepseek_f32_to_bf16_cuda` plus `deepseek_hc_post_cuda` pair with an attention-specific HC post kernel that reads the F32 all-reduce scratch.
- Preserved the old rounding boundary by converting each F32 reduced branch value to BF16 and back to F32 inside the fused HC post kernel before multiplying by `post`.
- FFN/MoE HC post stays unchanged.
- Verification:
  - local `cargo fmt --check` passed
  - local `git diff --check` passed
  - local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed
  - 5090 full exact E2E passed: `All 20 DeepSeek V4 exact cases passed`
- Bench result:
  - 1x32 with `--seed 42`: `steady_tpot_ms.avg = 91.84ms`, `p50 = 91.63ms`, `p95 = 96.21ms`, token hash `5f6c64b667f2abf5`
  - 1x160 round 1 with `--seed 42`: `steady_tpot_ms.avg = 110.17ms`, token hash `6346f03343d75a65` for all measured iterations
  - 1x160 round 2 with `--seed 42`: `steady_tpot_ms.avg = 105.99ms`, `p50 = 106.24ms`, `p95 = 115.04ms`, token hash `6346f03343d75a65` for all measured iterations
- Result: keep. The short bench improves over the final-logits-scratch 1x32 result (`93.14ms`), while the longer bench remains in the known `~103-106ms` band on its second run. The first long run shows the existing stability variance rather than a token-sequence difference.

### Step 7: Fuse attention KV RoPE plus no-PE quant
- Added `deepseek_apply_rope_and_fp8_act_quant_nope_bf16_cuda` for attention KV prep.
- Replaced the attention projection KV path from two launches:
  - `deepseek_apply_rope_hidden_cuda`
  - `deepseek_fp8_act_quant_nope_bf16_cuda`
- with one fused launch. Q RoPE stays on the existing kernel, and compressor/indexer paths stay unchanged.
- Reason: KV RoPE writes the rotary slice while no-PE quant writes the non-rotary slice, so the operations are independent within the same KV tensor.
- Local verification:
  - `cargo fmt --check` passed.
  - `git diff --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- Sync pitfall: the first 5090 sync used multi-source `rsync` without `-R/--relative`, which copied basenames into the remote repo root. Removed the misplaced root files and reran `rsync -avR ...` so paths were preserved. Future targeted syncs should use `rsync -avR` or run one rsync per destination path.
- Verification:
  - 5090 full exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- Bench result:
  - 1x32 run 1 with `--seed 42`: `steady_tpot_ms.avg = 97.82ms`, token hash `5f6c64b667f2abf5`.
  - 1x32 run 2 with `--seed 42`: `steady_tpot_ms.avg = 95.21ms`, token hash `5f6c64b667f2abf5`.
- Result: reverted the code. Even with identical generated-token hash, fusing these two tiny KV-prep kernels regressed the short decode bench versus the previous `91.84ms` result. The likely cause is worse block scheduling/occupancy from combining a tiny RoPE block with quant blocks; the saved launch did not offset that cost.

### Step 8: Profiling interpretation guardrails
- Nsight Systems CUDA kernel wall time for NCCL kernels is not pure link-transfer time. NCCL kernels include synchronization/rank-arrival waiting, so a large NCCL total can reflect upstream rank skew, load imbalance, or barrier placement rather than raw communication cost.
- Profiling should therefore use NCCL rows as synchronization-window evidence, not as a direct instruction to optimize transport first.
- For decode kernel work, prioritize:
  - token hash equality before comparing TPOT,
  - non-NCCL compute/launch clusters that are repeated per layer,
  - rank-skew sources before treating collectives as bandwidth-bound.
- Profiling pitfall: a direct `target/release/bench_serving` nsys run after rsync can accidentally profile a stale binary. Use `cargo build --release -p pegainfer-server --features deepseek-v4` or `cargo run --release ...` after syncing code before trusting profile artifacts.

### Step 9: Current nsys non-NCCL decode snapshot
- Rebuilt 5090 `pegainfer-server` after reverting the KV-prep fusion, then collected a short nsys CUDA profile:

```bash
nsys profile --force-overwrite=true --stats=false --sample=none --trace=cuda,nvtx,cublas --delay=35 --duration=12 -o /tmp/dsv4_current_decode_profile target/release/bench_serving --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 32 --warmup 1 --iters 1 --seed 42
nsys stats --report cuda_gpu_kern_sum --format csv --output /tmp/dsv4_current_decode_kern_sum /tmp/dsv4_current_decode_profile.nsys-rep
```

- The top NCCL rows were intentionally ignored as optimization targets because they include synchronization wait.
- Top non-NCCL rows in the captured window:
  - `deepseek_tilelang_fp4_grouped_gemm_n2048_k4096_kernel`: `437.83ms`, `10298` instances, `42.52us` avg. MoE path; defer for now.
  - `deepseek_hc_pre_norm_from_mixes_kernel`: `252.25ms`, `10144` instances, `24.87us` avg.
  - `deepseek_tilelang_fp8_gemm_n2048_k4096_kernel`: `205.44ms`, `10298` instances, `19.95us` avg. Shared/MoE FFN path; defer for now.
  - `deepseek_tilelang_act_quant_k4096_kernel`: `129.93ms`, `30891` instances, `4.21us` avg.
  - `deepseek_tilelang_fp4_grouped_gemm_n4096_k2048_kernel`: `124.99ms`, `5150` instances, `24.27us` avg. MoE path; defer for now.
  - cuBLAS small GEMV for HC mixes: `109.64ms`, `4713` instances, `23.26us` avg.
  - `deepseek_compressor_decode_project_kernel`: `82.98ms`, `7421` instances, `11.18us` avg.
  - `deepseek_tilelang_sparse_attn_local_h16_d512_kernel`: `63.26ms`, `5146` instances, `12.29us` avg.
  - `deepseek_indexer_scores_decode_serial_kernel`: `33.73ms`, `2180` instances, `15.47us` avg.
- Takeaway: after excluding MoE and treating NCCL as synchronization evidence, the next non-MoE candidates are HC pre/mixes structure, compressor decode project, sparse attention, and indexer scoring. Tiny KV-prep kernels are not promising by themselves; the failed Step 7 confirms launch-count reduction alone is not enough.

### Step 10: Try computing HC RMS scale during BF16->F32 conversion
- Changed `deepseek_hc_mixes_cuda` experimentally so the BF16->F32 conversion kernel also computes the RMS scale for each token using the same reduction shape as `deepseek_hc_scale_mixes_block_kernel`.
- After cuBLAS computes raw mixes, a tiny `deepseek_hc_apply_mix_scales_kernel` multiplies the `seq_len * mix_hc` mixes by the precomputed scale.
- Goal: avoid rereading the full `hc_dim = hc * dim` BF16 input in `deepseek_hc_scale_mixes_block_kernel`.
- Local verification:
  - `cargo fmt --check` passed.
  - `git diff --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- Verification:
  - 5090 full exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- Bench result:
  - 1x32 with `--seed 42`: `steady_tpot_ms.avg = 110.71ms`, `p50 = 109.78ms`, `p95 = 115.26ms`, token hash `5f6c64b667f2abf5`.
- Result: reverted the code. The token hash matched the baseline, but TPOT regressed heavily versus `91.84ms`. Avoid folding the RMS-scale reduction into the conversion kernel; the extra reduction work and altered kernel shape cost more than rereading the HC input in the existing scale kernel.

### Step 11: Try parallel indexer decode score kernel
- Changed `deepseek_indexer_scores_decode_cuda` experimentally from the serial one-thread-per-compressed-entry kernel to the existing parallel one-block-per-compressed-entry reduction kernel.
- Reason: `deepseek_indexer_scores_decode_serial_kernel` appears in the non-NCCL profile, but the serial shape may still be better for small compressed lengths; this experiment measures rather than guessing.
- Local verification:
  - `cargo fmt --check` passed.
  - `git diff --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- Verification:
  - 5090 full exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- Bench result:
  - 1x32 run 1 with `--seed 42`: `steady_tpot_ms.avg = 89.60ms`, token hash `5f6c64b667f2abf5`.
  - 1x32 run 2 with `--seed 42`: `steady_tpot_ms.avg = 102.59ms`, token hash `5f6c64b667f2abf5`.
  - 1x160 run 1 with `--seed 42`: `steady_tpot_ms.avg = 95.42ms`, token hash `6346f03343d75a65`.
  - 1x160 run 2 with `--seed 42`: `steady_tpot_ms.avg = 112.10ms`, token hash `6346f03343d75a65`.
- Result: reverted the code. The token hashes matched within each shape, but TPOT was not reproducible. The average of the two short runs was worse than the current `91.84ms` short baseline, and the two long runs bracketed the existing noisy range rather than proving a stable improvement.

### Step 12: Share FP8 activation quantization for attention `wq_a` and `wkv`
- Added `deepseek_fp8_two_linear_cuda`, a TileLang-only wrapper that quantizes one BF16 input once and runs two FP8 GEMMs with separate weights/scales/outputs.
- Replaced attention projection's independent `wq_a(input)` and `wkv(input)` calls with one two-linear call. `wq_b(q_norm)` remains separate because it consumes a different input.
- Reason: the nsys snapshot shows many `deepseek_tilelang_act_quant_k4096_kernel` launches, and attention projection was quantizing the same layer input twice.
- Local verification:
  - `cargo fmt --check` passed.
  - `git diff --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- Verification:
  - 5090 full exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- Bench result:
  - 1x32 with `--seed 42`: `steady_tpot_ms.avg = 101.90ms`, token hash `5f6c64b667f2abf5`.
- Result: reverted the code. Sharing the activation quantization across attention `wq_a` and `wkv` reduced one small quant launch per layer, but changed the GEMM scheduling/quant scratch lock window enough that the short decode bench regressed versus `91.84ms`.

### Step 13: Post-revert bench sanity
- After reverting the Step 12 code and syncing source back to 5090, rebuilt through `cargo run --release` and reran the short bench:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 32 --warmup 1 --iters 1 --seed 42
```

- Result: `steady_tpot_ms.avg = 107.25ms`, token hash `5f6c64b667f2abf5`.
- Takeaway: single 1x32 runs are currently too noisy to use as the only decision gate. Keep using token hash equality, repeated runs, and the longer `1x160 warmup=2 iters=3` shape when a candidate looks promising.

### Step 14: Add h8 sparse attention TileLang shape
- Added a `local_h8_d512` TileLang sparse attention shape.
- Changed `deepseek_indexed_attention_prefill_cuda` to call the h8 sparse attention symbol directly for the current MP8 `local_heads=8, head_dim=512` shape.
- Reason: the previous wrapper only had `local_h16_d512`, so every sparse attention call padded Q and sink from h8 to h16, ran the h16 kernel, then copied output back to h8.
- Result: reverted the code before remote testing. Local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` failed during TileLang generation: `M must be divisible by 16, but got 8`. The current TileLang GEMM lowering used by sparse attention requires the head dimension of the GEMM tile to be a multiple of 16, so a direct h8 sparse attention shape is not viable without a different kernel formulation.

## Debrief

- **Outcome**: Kept the changes that survived exact E2E and decode bench review: final logits scratch reuse, generated-token trace reporting, and attention all-reduce tail fusion into HC post.
- **Rejected experiments**: final HC head plus RMSNorm fusion, shared decode window top-k indices, KV RoPE plus no-PE quant fusion, HC RMS-scale folding into BF16->F32 conversion, parallel indexer decode scoring, shared FP8 quantization for attention `wq_a/wkv`, and direct h8 TileLang sparse attention.
- **Benchmark discipline**: token hash equality is required before comparing TPOT because token ids affect hash routing, EPLB/expert balance, and rank skew. Single 1x32 runs on 5090 became too noisy late in the audit; promising candidates need repeated runs and the longer `1x160 warmup=2 iters=3` shape.
- **Profiling discipline**: NCCL kernel wall time is synchronization-window evidence, not pure transport time. Non-NCCL repeated clusters after the kept changes are mostly MoE/grouped GEMM, HC pre/mixes, compressor decode, sparse attention, and indexer scoring.
- **Follow-ups**: Further meaningful gains likely need a more structural MoE route/tile scheduling change, a different sparse-attention kernel formulation that naturally supports h8, or rank-skew analysis around collective arrival timing. Small local fusion attempts are now documented and should not be retried without a new profile or a materially different kernel design.
