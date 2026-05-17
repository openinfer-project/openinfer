# DeepSeek MoE TileLang Review

**Created**: 2026-05-09
**Status**: complete

## Current TL;DR

- Official DeepSeek `tile_kernels/moe` is routing/layout infrastructure, not a drop-in fused FP4 expert MLP. The useful pieces are `top2_sum_gate`, TP masking, group/count helpers, `get_fused_mapping`, `expand_to_fused`, and `reduce_fused`.
- The first local packed-layout experiment preserved exact DeepSeek V4 output, but it did not fix decode TPOT. Non-nsys `prompt-len=1`, `output-len=32` measured `171.11ms/token`; nsys measured `268.51ms/token` with profiler overhead.
- The NCCL bucket was over-interpreted: grouped by 8-rank logical collective, the trace shows most apparent NCCL duration is rank arrival skew/waiting, not pure communication. CPU rank dispatch was the first large cause; after persistent rank workers, the remaining skew mostly comes from uneven rank-local MoE decode work before each all-reduce.
- A scoped rank-thread decode slice improved 1x32 steady TPOT to `113.54ms/token`. Replacing it with 8 persistent Qwen3-style rank workers improved the same bench to `94.84ms/token`; replacing the decode-only packed MoE path with direct top-k expert execution improved it again to `80.49ms/token`. Full exact E2E passed all 20 cases after each effective optimization.
- Persistent-worker nsys shows about `107` f32 all-reduce collectives per steady decode token. Post-arrival NCCL tail is only about `0.019ms/collective`, while arrival skew averages about `0.83ms/collective`; per-token skew is therefore the Amdahl-sized bucket. The direct proof is in `dsv4_rank_stage_proof.sqlite`: for `token=12`, `layer=17`, `moe_ar`, rank0's NCCL kernel lasts `1.220ms` only because rank4 reaches the same collective `1.200ms` later after running local MoE and shared-expert work. The root cause is not a single 60ms NCCL transfer, and EP8 routed expert imbalance alone does not explain 100ms TPOT. The stronger explanation is repeated rank phase skew before collectives, amplified by runtime/allocator/host-loop overhead and then serialized across 43 layers.
- Failed optimizations to avoid repeating: replacing hot zeroed allocations with uninitialized allocation regressed 1x32 TPOT to `107.52ms/token`; sharing W1/W3 FP4 activation quantization regressed it to `119.63ms/token`; CPU-pinning persistent workers regressed to `105.66ms/token`; moving routed all-reduce before shared expert regressed to `85.67ms/token`; rank0-only route broadcast regressed to `86.98ms/token`; moving shared expert before routed expert regressed to `87.58ms/token`; BF16 tensor-op score gate regressed to `85.03ms/token`; BF16 attention all-reduce regressed to `86.51ms/token`. These were reverted.
- Current-pass conclusion: `80.49ms/token` is close to the small-patch ceiling for this architecture. The next real drop needs to reduce routed expert imbalance, fuse/group the decode routed expert path, make route/expert scheduling GPU-resident, or reduce the number of f32 collective barriers. Polishing standalone memset, gate, dtype, CPU affinity, or local operation ordering is not enough unless it moves the per-token skew bucket.
- Defensible TPOT ceiling estimate: `70-80ms/token` is the current structure's near ceiling; `55-65ms/token` likely needs grouped/fused routed expert decode and less runtime/host scheduling; `45-55ms/token` likely needs fewer collective barriers or a different communication/dataflow design; below `45ms/token` is not credible from the present MP8 decode structure.
- Keep this profile as the decode composition baseline for the next refactor; do not judge future MoE changes only by exact e2e text pass.

## Preparation

- **Read**:
  - `docs/index.md` - identified DeepSeek V4 support, DeepSeek kernel paths, and kernel technology reference as the relevant routing docs.
  - `docs/models/deepseek-v4/support.md` - confirmed the current DeepSeek V4 path has native MP8 runtime, TileLang build-time kernels, and a handwritten CUDA MoE path; it also notes MoE route-index D2H synchronization as a higher-risk remaining target.
  - `docs/models/deepseek-v4/kernel-paths.md` - confirmed DeepSeek CUDA sources now live under `pegainfer-kernels/csrc/deepseek_v4/` and TileLang generators live under `pegainfer-kernels/tools/tilelang/deepseek_v4/`.
- **Relevant history**:
  - `docs/models/deepseek-v4/support.md` records that the current local TileLang generator emits quantized linear, sparse attention, and HC kernels, while `deepseek_moe.cu` owns routing, SwiGLU, and expert accumulation.
  - `docs/models/deepseek-v4/kernel-paths.md` records that the DeepSeek kernel routing table was recently organized, so this review should compare against those paths instead of rediscovering ownership from scratch.
- **Plan**:
  1. Inspect the official DeepSeek TileKernels `tile_kernels/moe` directory from `https://github.com/deepseek-ai/TileKernels/tree/main/tile_kernels/moe`, including file names, exported kernels, and expected tensor layouts.
  2. Inspect local MoE code paths: `pegainfer-kernels/csrc/deepseek_v4/deepseek_moe.cu`, related FFI declarations, and `pegainfer-deepseek-v4/src/runtime/` callers.
  3. Compare official kernels against local behavior along routing layout, expert grouping, quantization format, activation, accumulation dtype, and dispatch/combine boundaries.
  4. Summarize what official TileLang operators exist, what they appear to solve, and which local MoE issue they most likely explain or do not explain.
  5. If the gap is clear and small, propose the first implementation slice; otherwise stop with a focused diagnostic checklist.
- **Risks / open questions**:
  - The official TileKernels repository may have changed after the current local implementation was written, so source URLs and commit state need to be cited.
  - Official kernels may target a different model/checkpoint or deployment topology than the current DeepSeek V4 Flash MP8 path.

## Execution Log

### Step 10: Hard stop for current optimization pass
- 2026-05-10 01:18 CST: the user set a hard stop at 2026-05-10 04:30 CST. If no additional effective optimization is found by then, stop with the current evidence, a ceiling estimate, clean worktree, and a final conclusion instead of spending more tokens on low-Amdahl experiments.
- Working rule after the direct top-k MoE commit: only keep changes that move the measured 1x32 steady TPOT below the committed `80.49ms/token` baseline and then pass full exact DeepSeek V4 E2E. Failed experiments should be recorded and reverted.
- 2026-05-10 01:43 CST conclusion: stopped early because the remaining candidates that fit a small safe patch all failed the Amdahl test. The failing experiments consistently regressed to `85-107ms/token`, while the committed direct top-k MoE path remains the best exact-E2E-validated result at `80.49ms/token`.

### Step 1: Inspect official DeepSeek TileKernels MoE directory
- Cloned the official repository for local inspection:
  - `git clone --depth 1 https://github.com/deepseek-ai/TileKernels.git /tmp/deepseek-tilekernels`
  - inspected commit `36d9e45d38e204ebb87e6f6e833821eee0482fe5`.
- Official `tile_kernels/moe/` exports:
  - `top2_sum_gate` and `topk_gate` for routing selection;
  - `normalize_weight` for row-wise top-k weight normalization;
  - `mask_indices_by_tp` for TP-local expert masking/remapping;
  - `group_count`, `aux_fi`, and `inplace_unique_group_indices` for counting/deduplicating expert/group assignments;
  - `get_fused_mapping`, `expand_to_fused`, `expand_to_fused_with_sf`, and `reduce_fused` for expert-major fused execution layout and token-layout reduction.
- Important source observations:
  - `top2_sum_gate_kernel.py` supports `sqrtsoftplus`, `sigmoid`, and `softmax` scoring, applies gate bias for selection, uses un-biased scores for weights, normalizes by top-k sum, applies `routed_scaling_factor`, supports optional fixed routing, maps logical to physical experts, and masks/remaps indices for EP/TP.
  - `get_fused_mapping_kernel.py` builds `pos_to_expert`, `pos_to_token`, `pos_to_token_topk`, `token_topk_to_pos`, `expert_start`, `expert_end`, and `num_tokens_per_expert`. This is the routing metadata needed to execute only packed token slices per expert.
  - `expand_to_fused_kernel.py` copies token activations into the expanded expert-major layout, with an optional scale-factor path for quantized activations.
  - `reduce_fused_kernel.py` gathers expert outputs back to token order and applies top-k weights.
- The checked model config at `/data/DeepSeek-V4-Flash/config.json` has:
  - `n_routed_experts=256`
  - `n_shared_experts=1`
  - `num_experts_per_tok=6`
  - `scoring_func="sqrtsoftplus"`
  - `topk_method="noaux_tc"`
  - `norm_topk_prob=true`
  - `routed_scaling_factor=1.5`

Result: official MoE TileLang is primarily a routing, mapping, packing, and reduction toolkit. It does not appear to provide a single fused FP4 expert MLP kernel in `tile_kernels/moe/`; the fused expert GEMM path is implied by the expert-major layout and the quantized helpers used around it.

### Step 2: Compare local MoE implementation
- Local score routing in `pegainfer-kernels/csrc/deepseek_v4/deepseek_moe.cu` broadly matches the model config's simple scoring semantics:
  - BF16 gate scores are converted to F32 and multiplied through cuBLAS;
  - selection score is `sqrt(softplus(dot)) + gate_bias`;
  - route weight is the original `sqrt(softplus(dot))`;
  - selected weights are normalized and multiplied by `routed_scaling_factor`.
- Local execution differs substantially from the official fused-layout path:
  - `pegainfer-deepseek-v4/src/runtime/moe.rs` copies `routed.indices` from device to host and synchronizes in both `routed_local_experts_forward_bf16_hidden` and `routed_local_experts_forward_f32_hidden`.
  - The CPU then builds `active_local` and loops over local experts.
  - For each active local expert, `local_expert_forward_*` runs W1, W3, SwiGLU, and W2 over the full input batch, then masks/weights the result by route index.
  - Official TileKernels instead keeps routing metadata on GPU, creates expert-major packed token ranges, expands inputs once, executes expert work over packed ranges, then reduces back with `token_topk_to_pos`.
- Local score routing also lacks the official `top2_sum_gate` EP/TP semantics:
  - no logical-to-physical expert map;
  - no TP-local masking/remapping in the gate kernel;
  - no optional shared-as-routed path;
  - no group top-k path, though this may be irrelevant for this `topk_method="noaux_tc"` config unless later checkpoints use grouped routing.

Result: the most likely MoE problem is not the `sqrtsoftplus` math itself for non-hash layers. The larger mismatch is the execution/layout model: local code does per-active-expert full-batch work with a host synchronization, while official TileKernels provide GPU-resident fused mapping, expand, and reduce operators for sparse expert execution.

### Step 3: First implementation slice recommendation
- Do not start by replacing the gate kernel with `top2_sum_gate`. Local gate math should first be parity-checked against official `top2_sum_gate` for `scoring_func="sqrtsoftplus"`, `num_groups=0`, `num_topk_groups=0`, `use_shared_as_routed=false`, `num_shared_experts=1`, `routed_scaling_factor=1.5`, and the current MP8 world layout.
- First low-risk runtime slice:
  1. Add CUDA or TileLang-generated equivalents of `mask_indices_by_tp` and `group_count` for the current local `i32` route-index layout.
  2. Replace the D2H `route_indices` active-expert detection with GPU-side per-local-expert counts and a fixed host policy that can avoid the route-index copy. If avoiding all host control is too large, one small `local_expert_count` D2H vector is still better than copying all token-topk indices and synchronizing on every rank.
  3. Add an operator test comparing local routing indices/weights and counts against a small official/PyTorch reference for this config.
- Larger performance/correctness slice:
  1. Add `get_fused_mapping` and `expand_to_fused` equivalents for local `i32` indices.
  2. Teach the FP4 expert linear path to run a specific expert over `expert_start..expert_end` packed token ranges instead of full `seq_len`.
  3. Replace `deepseek_weighted_expert_accum_*` with a `reduce_fused`-style token reduction.

### Unexpected
- The official MoE directory contains no obvious one-call fused FP4 routed expert MLP. The valuable part for this codebase is the routing/mapping layout, not a drop-in expert GEMM replacement.
- Official `get_fused_mapping` can still host-sync if `num_expanded_tokens=0` and `force_no_sync=false`; for runtime serving, preallocating a worst-case expanded buffer is probably the right shape.

### Step 4: Refactor plan after accepting larger scope
- The current runtime cost is structural: sparse MoE is executed as repeated full-token expert MLPs. A meaningful refactor should change the dataflow, not only tune individual kernels.
- Proposed staged architecture:
  1. **GPU-resident route metadata**: introduce a `MoeRoutePlan` beside `RoutedExperts` with route weights/indices plus local expert counts or active flags computed on GPU. This removes full `route_indices` D2H copies from `routed_local_experts_forward_*`.
  2. **Expert-major mapping**: add official-style mapping tensors for local execution:
     - `pos_to_expert`
     - `pos_to_token`
     - `pos_to_token_topk`
     - `token_topk_to_pos`
     - `expert_start`
     - `expert_end`
     - `num_tokens_per_expert`
  3. **Expand/reduce dataflow**: add `expand_to_fused` and `reduce_fused` equivalents for the local Rust/CUDA ABI. This lets expert kernels consume packed token ranges and produce packed outputs.
  4. **Ranged expert FP4 linear**: extend `deepseek_fp4_linear_cuda` or add a new wrapper so W1/W3/W2 can run for one expert over `expert_start..expert_end` packed tokens instead of the full `seq_len`.
  5. **Shared expert remains dense**: keep `shared_expert_forward_bf16_hidden` unchanged initially. It is dense by design and uses FP8, so it is not the same bottleneck as routed experts.
- Recommended first code slice:
  1. Add `deepseek_moe_local_expert_counts_cuda` and tests. It accepts route indices, `global_start`, and `local_experts`, then writes per-local-expert counts on GPU.
  2. Replace full route-index D2H with a tiny counts D2H in `routed_local_experts_forward_f32_hidden`. This is not the final design, but it immediately removes the worst synchronization volume and gives a correctness-preserving stepping stone.
  3. Add `get_fused_mapping`/`expand_to_fused` after that, because those require more ABI and memory-planning changes.
- Risk:
  - Going straight to expert-major FP4 execution touches routing, kernel generation, quantized GEMM wrappers, and NCCL accumulation at once. The first slice should preserve outputs exactly while shrinking the CPU synchronization boundary.

### Step 5: Decode profile after first packed-layout experiment
- A first local packed-layout experiment was implemented and exact text still passed on all 20 DeepSeek V4 cases, but the user-observed TPOT remained around the same high range. That result made a profile more valuable than further guessing.
- This is now the baseline profile for the MoE refactor. It explains the observed roughly `180-200ms` TPOT range better than per-kernel intuition: decode is split across collective communication, many tiny routed FP4 expert GEMMs, dense FP8 projections, quantization, and model-specific compressor/HC work.
- Non-nsys synthetic decode-heavy command:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- \
  --model-path /data/DeepSeek-V4-Flash --format json \
  request --prompt-len 1 --output-len 32 --warmup 1 --iters 1
```

- Result without nsys:
  - load: `35.28s`
  - steady TPOT avg: `171.11ms/token`
  - e2e: `5.47s`
  - decode throughput: `5.85 tok/s`
- nsys command used after compiling the binary once:

```bash
nsys profile --stats=false --force-overwrite=true \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  --delay=34 --duration=14 \
  -o target/profiling/dsv4_moe_packed_decode_1x32_direct \
  target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash --format json \
  request --prompt-len 1 --output-len 32 --warmup 1 --iters 1
```

- Trace artifacts:
  - `target/profiling/dsv4_moe_packed_decode_1x32_direct.nsys-rep`
  - `target/profiling/dsv4_moe_packed_decode_1x32_direct.sqlite`
- Result under nsys:
  - steady TPOT avg: `268.51ms/token`
  - p50: `250.31ms`
  - p95: `330.18ms`
  - e2e: `8.62s`
- The profile is nsys-inflated in absolute terms, but useful for composition. Approximate per-token breakdown from the 30 steady-token window:

| Bucket | Total in trace | Approx per token | Observation |
| --- | ---: | ---: | --- |
| NCCL f32 all-reduce | `1996ms` | `~66ms/token` | Largest single bucket; `38272` all-reduce kernels in the trace. |
| Routed FP4 expert GEMM | `1972ms` | `~66ms/token` | W1/W3 `deepseek_tilelang_fp4_gemm_n2048_k4096` plus W2 `n4096_k2048`; packed layout did not reduce decode GEMM launch count enough. |
| FP8 GEMM | `~1654ms` | `~55ms/token` | Attention/shared/compressor projections remain large. |
| Activation/FP4 quantization | `~672ms` | `~22ms/token` | TileLang act quant kernels and FP4 quant helper. |
| Compressor/HC/indexer misc | `~650ms+` | `~20ms/token` | Compressor decode, Hadamard, HC sinkhorn/mixes, indexer scoring. |
| Score gate | `~245ms` | `~8ms/token` | Gate selection plus BF16->F32 conversion and cuBLAS gate GEMV. |
| Sparse attention | `196ms` | `~6.5ms/token` | `deepseek_tilelang_sparse_attn_local_h16_d512_kernel`. |
| Packed MoE layout kernels | `~148ms` | `~5ms/token` | Mapping/expand/reduce/clear. Not the main bucket, but added work on decode. |
| RMSNorm | `114ms` | `~4ms/token` | FlashInfer RMSNorm kernels. |

- Top kernel totals from the trace:
  - `ncclDevKernel_AllReduce_Sum_f32_RING_LL`: `1996ms`, `38272` launches, avg `52us`
  - `deepseek_tilelang_fp4_gemm_n2048_k4096_kernel`: `1565ms`, `23676` launches, avg `66us`
  - `deepseek_tilelang_fp4_gemm_n4096_k2048_kernel`: `408ms`, `11838` launches, avg `34us`
  - `deepseek_tilelang_fp8_gemm_n2048_k4096_kernel`: `604ms`, `31568` launches, avg `19us`
  - `deepseek_tilelang_act_quant_k4096_kernel`: `372ms`, `86828` launches, avg `4.3us`
  - `deepseek_score_gate_select_kernel`: `173ms`, `14680` launches, avg `11.8us`
  - `deepseek_moe_expand_to_fused_kernel`: `35.9ms`, `15784` launches, avg `2.3us`
  - `deepseek_moe_reduce_fused_f32_kernel`: `18.9ms`, `15784` launches, avg `1.2us`
  - `deepseek_moe_local_mapping_kernel`: `17.5ms`, `15784` launches, avg `1.1us`
- API/allocator signal under nsys:
  - `cudaLaunchKernel_v7000`: `1,123,549` calls
  - `cuMemsetD8Async`: `804,217` calls
  - `cuMemAllocAsync`: `803,569` calls
  - `cuMemFreeAsync`: `803,448` calls
  - `cuMemcpyDtoHAsync_v2`: `31,613` calls
- Interpretation:
  - The first packed-layout attempt did not solve decode because `seq_len=1` still executes many tiny per-expert W1/W3/W2 FP4 GEMMs. Expert-major layout is more promising for prefill than single-token decode unless the expert GEMMs are grouped or fused.
  - Decode TPOT is dominated by two roughly equal buckets: f32 NCCL all-reduce and routed FP4 expert GEMM. FP8 projections are the next large bucket.
  - The next MoE optimization should not spend much more time on standalone `expand/reduce` polish. It needs a decode-specific grouped/fused routed expert path, and the runtime also needs all-reduce count reduction or fusion.
  - Allocation/memset churn remains severe. Some of it is general runtime scratch behavior, but newly introduced packed tensors should avoid `alloc_zeros` where kernels fully overwrite the output.

### Step 6: Preserve profile implications for the next refactor
- The observed TPOT should be treated as a multi-bucket decode problem, not a pure MoE routing problem.
- Immediate implications:
  - **Routed expert GEMM**: packing helps prefill-style batches, but decode still needs grouped/fused expert execution because top-k routes produce many small W1/W3/W2 launches.
  - **All-reduce**: raw NCCL kernel-duration sums are misleading because the 8 rank kernels overlap and include rank waiting. Group f32 all-reduce by logical 8-rank collective before treating it as a communication bucket.
  - **Allocator/memset churn**: `cuMemAllocAsync`, `cuMemFreeAsync`, and `cuMemsetD8Async` counts are high enough that new MoE buffers should use reusable scratch and uninitialized allocation when kernels fully overwrite their outputs.
  - **Validation**: exact e2e pass is necessary but insufficient. Every major MoE dataflow change should be followed by the same `prompt-len=1`, `output-len=32` bench and, when needed, an nsys composition check.
- No additional profile was run while another compile was in progress.

### Step 7: Rank-thread decode CPU dispatch slice
- Static code review showed the direct DeepSeek decode loop dispatched rank-local work serially from one CPU thread. That matched the nsys shape: f32 NCCL kernels often started rank-by-rank, ended nearly together, and therefore included spin-wait in the earlier ranks.
- Recomputed the existing `dsv4_moe_packed_decode_1x32_direct.sqlite` f32 NCCL bucket by grouping consecutive 8 rank kernels:
  - `38272` f32 NCCL kernels become `4784` logical collectives.
  - Raw per-kernel duration sum: `1996ms`.
  - Logical collective wall sum: `358.69ms`.
  - Arrival skew sum: `267.07ms`.
  - Tail after final rank arrival: `91.62ms`.
  - Interpretation: most apparent NCCL time was rank arrival skew, not link transfer.
- Implemented a first decode CPU-dispatch slice:
  - `run_direct_decode_logits` now calls `block_decode_group_rank_threads_bf16_hidden`.
  - Each decode layer starts one scoped CPU lane per rank. A lane runs rank-local attention, ratio-4 indexer score all-reduce, attention output all-reduce, FFN/MoE local work, MoE routed all-reduce, and HC post for that rank.
  - Added single-rank helpers `all_reduce_hidden_fp32_in_place` and `all_reduce_f32_hidden_in_place` so each rank lane can enter NCCL directly instead of returning to a main-thread grouped caller.
  - Added explicit `Send` justification for `RankGpuContext` and a scoped `RankComm` pointer wrapper; this is a stepping stone until communicators are moved into persistent rank workers.
- Validation:

```bash
cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4
```

  - Passed with the existing unreachable-pub warnings in `runtime/core.rs` and `runtime/state.rs`.

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- \
  --model-path /data/DeepSeek-V4-Flash --format json \
  request --prompt-len 1 --output-len 32 --warmup 1 --iters 1
```

  - load: `35.21s`
  - TTFT: `171.46ms`
  - first decode step: `107.70ms`
  - steady TPOT avg: `113.54ms/token`
  - p50: `112.77ms`
  - p95: `117.04ms`
  - e2e: `3.69s`

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- \
  --model-path /data/DeepSeek-V4-Flash \
  --ground-truth test_data/deepseek-v4-ground-truth.json \
  --offset 0 --limit 1 --max-new-tokens 64
```

  - Passed exact case 0.
- Full exact validation required before commit:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- \
  --model-path /data/DeepSeek-V4-Flash \
  --ground-truth test_data/deepseek-v4-ground-truth.json \
  --max-new-tokens 64
```

  - Passed all 20 DeepSeek V4 exact cases.
  - Representative long-output cases after rank-thread decode:
    - case 4: `21` tokens, `127.99ms` TPOT
    - case 9: `21` tokens, `128.71ms` TPOT
    - case 12: `15` tokens, `131.45ms` TPOT
    - case 18: `14` tokens, `131.35ms` TPOT
- Interpretation:
  - The CPU rank-dispatch hypothesis was correct: a minimal rank-thread decode slice cut steady TPOT from the prior `171ms/token` range to `113.54ms/token`.
  - This slice is not the final architecture because it creates scoped threads every decode layer. The next implementation should mirror Qwen3 TP more closely with 8 persistent rank workers that own `RankGpuContext`, `RankWeightView`, `Comm`, caches, and RoPE state.

### Step 8: Persistent rank workers for direct decode
- Captured a follow-up nsys trace for the scoped rank-thread path:

```bash
nsys profile --stats=false --force-overwrite=true \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  --delay=34 --duration=12 \
  -o target/profiling/dsv4_rank_thread_decode_1x32_direct \
  target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash --format json \
  request --prompt-len 1 --output-len 32 --warmup 1 --iters 1
```

  - Trace artifacts:
    - `target/profiling/dsv4_rank_thread_decode_1x32_direct.nsys-rep`
    - `target/profiling/dsv4_rank_thread_decode_1x32_direct.sqlite`
  - The nsys run was heavily distorted (`138.86ms/token`) and ended with the existing NCCL abort-on-exit panic after benchmark JSON had been emitted.
  - Logical NCCL recomputation for f32 all-reduce kernels:
    - `6527` f32 NCCL kernels, `815` complete 8-rank groups in the trace window.
    - Raw per-kernel duration sum: `42165.87ms`.
    - Logical wall sum: `10448.35ms`.
    - Arrival skew sum: `10432.51ms`.
    - Tail after final rank arrival: `15.84ms`.
    - Interpretation: under nsys, raw NCCL time is almost entirely rank arrival skew / profiling distortion. The post-arrival communication tail averages about `19us` per collective.
- Implemented persistent direct-decode rank workers:
  - `load_full_direct_runtime` now starts `deepseek-v4-rank-{0..7}` workers during model load.
  - Each worker owns a cloned rank CUDA context handle, `RankWeightView`, a decode NCCL communicator, and per-rank RoPE cache. The worker thread binds the CUDA context and initializes thread-local cuBLAS once at startup.
  - Decode dispatch sends one command per rank per token. Each worker runs embedding, embedding all-reduce, HC expand, all layers, and rank-local final logits inside the persistent lane.
  - The main direct thread now only moves per-rank decode cache ownership to/from workers and performs the final logits all-gather/argmax.
  - Prefill remains on the existing group path with its own NCCL communicator. CUDA graph remains unsupported for DeepSeek V4 direct decode.
- Validation:

```bash
cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4
```

  - Passed with the existing unreachable-pub warnings in `runtime/core.rs` and `runtime/state.rs`.

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- \
  --model-path /data/DeepSeek-V4-Flash --format json \
  request --prompt-len 1 --output-len 32 --warmup 1 --iters 1
```

  - load: `35.54s`
  - TTFT: `175.18ms`
  - first decode step: `92.94ms`
  - steady TPOT avg: `94.84ms/token`
  - p50: `94.23ms`
  - p95: `98.23ms`
  - e2e: `3.11s`

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- \
  --model-path /data/DeepSeek-V4-Flash \
  --ground-truth test_data/deepseek-v4-ground-truth.json \
  --max-new-tokens 64
```

  - Passed all 20 DeepSeek V4 exact cases.
  - Representative long-output cases after persistent workers:
    - case 4: `21` tokens, `100.89ms` TPOT
    - case 9: `21` tokens, `102.41ms` TPOT
    - case 12: `15` tokens, `103.56ms` TPOT
    - case 18: `14` tokens, `113.72ms` TPOT
- Interpretation:
  - Persistent rank workers are an effective CPU-dispatch optimization: scoped rank-thread steady TPOT `113.54ms/token` moved to `94.84ms/token`.
  - The remaining decode floor is more likely the many tiny FP4/FP8 GEMMs, activation quantization, allocator/memset churn, and model-specific compressor/indexer work than raw NCCL link time.

### Step 9: Persistent-worker nsys skew root cause
- Captured a follow-up nsys trace for the persistent worker path:

```bash
nsys profile --stats=false --force-overwrite=true \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  --delay=34 --duration=12 \
  -o target/profiling/dsv4_persistent_workers_decode_1x32_direct \
  target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash --format json \
  request --prompt-len 1 --output-len 32 --warmup 1 --iters 1
```

  - Trace artifacts:
    - `target/profiling/dsv4_persistent_workers_decode_1x32_direct.nsys-rep`
    - `target/profiling/dsv4_persistent_workers_decode_1x32_direct.sqlite`
  - The nsys run was distorted (`145.32ms/token`) and ended with the existing NCCL destructor panic after benchmark JSON had been emitted. Use it for composition and ordering, not absolute TPOT.
- Corrected f32 NCCL interpretation:
  - The final logits all-gather groups give `32` decode token boundaries.
  - Steady tokens contain about `107` f32 all-reduce collectives:
    - `43` attention output all-reduces,
    - `43` MoE routed output all-reduces,
    - `21` ratio-4 indexer score all-reduces.
  - Across complete steady-token f32 all-reduce groups:
    - average arrival skew: `0.830ms/collective` (`p50=0.865ms`, `p95=1.449ms`);
    - average post-arrival tail: about `0.019ms/collective`;
    - steady token intervals under nsys average `~88ms/token` of f32 collective arrival skew but only `~2.05ms/token` of post-arrival NCCL tail.
  - Therefore the raw `ncclDevKernel_AllReduce_Sum_f32` total (`~25s` in this trace) is mostly rank wait time. It should not be read as link transfer cost.
- Root-cause signal for the skew:
  - Per-device f32 collective lateness is not completely uniform, but no single device explains the bucket. Devices 1 and 3 are late most often; devices 4 and 0 are early most often.
  - The largest-skew collective ordinals have a repeated pre-collective fingerprint: routed MoE FP4 expert work. Example high-skew ordinals show latest ranks running roughly `5-7` FP4 expert kernels before entering the all-reduce while earliest ranks usually run `0`.
  - Full-trace FP4 expert kernel counts are also uneven by device:

| Device | FP4 kernels | FP4 total |
| ---: | ---: | ---: |
| 0 | `5943` | `327.4ms` |
| 1 | `5979` | `337.1ms` |
| 2 | `6057` | `331.8ms` |
| 3 | `6225` | `346.8ms` |
| 4 | `5340` | `294.9ms` |
| 5 | `5448` | `305.0ms` |
| 6 | `6009` | `335.8ms` |
| 7 | `5826` | `324.5ms` |

  - Average FP4 kernel duration is similar across devices (`~55us`), so the imbalance is mostly how many routed local experts each rank executes, not a slow individual GPU kernel.
  - The current packed MoE decode path still does `clone_dtoh(&plan.expert_start)`, `clone_dtoh(&plan.expert_end)`, and `ctx.sync()` inside `local_experts_forward_packed_bf16_hidden`, then uses a host loop to run only active local experts. This makes rank-local route imbalance visible as all-reduce arrival skew.
  - Runtime API counts under nsys confirm the path is still heavily fragmented on rank worker threads: about `1.46M` kernel launches, `1.04M` async allocs/frees, and `40.8K` D2H memcpy calls on rank-worker threads in the captured window. These are important only insofar as they amplify the MoE imbalance; the Amdahl bucket remains arrival skew before collectives.
- Follow-up sanity check after questioning the EP8-only explanation:
  - Current runtime is best described as MP8: tensor-parallel dense/attention plus EP8-style routed expert ownership. Routed experts are not tensor-sliced within each expert; `256` experts are assigned by id across 8 ranks, `32` complete experts per rank.
  - Even in the worst per-token route imbalance, routed FP4 expert compute alone is too small to explain 100ms TPOT. In the persistent-worker trace, per-device FP4 totals are roughly `~10-12ms/token`.
  - Re-mapping the 107 steady-token collectives by code order gives average skew contribution per token:

| Collective class | Count/token | Skew/token |
| --- | ---: | ---: |
| MoE routed all-reduce | `43` | `~43.0ms` |
| Attention output all-reduce | `43` | `~25.5ms` |
| Ratio-4 indexer score all-reduce | `21` | `~20.4ms` |
| Total | `107` | `~88.8ms` |

  - Late-vs-early segment decomposition shows most of the wall skew is runtime/API overhead rather than active GPU compute:

| Collective class | Late wall | Early wall | Late GPU active | Early GPU active | Late runtime/API | Early runtime/API |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Indexer | `1.613ms` | `0.642ms` | `0.170ms` | `0.169ms` | `1.449ms` | `0.552ms` |
| Attention | `0.976ms` | `0.384ms` | `0.085ms` | `0.084ms` | `0.864ms` | `0.325ms` |
| MoE | `1.379ms` | `0.379ms` | `0.402ms` | `0.115ms` | `1.240ms` | `0.329ms` |

  - Interpretation: MoE EP8 route imbalance creates some phase skew, but the 100ms-scale TPOT comes from many small phase skews being paid at every barrier. Attention and indexer collectives also pay large skew despite nearly equal active GPU work, which points at CPU runtime, allocation/free, launch gaps, and host-controlled loops as the amplification mechanism.
  - Next experiments should measure or reduce runtime churn and host-controlled MoE decode scheduling before assuming expert compute or raw NCCL bandwidth is the limiting factor.
- Temporary NVTX proof trace:
  - Added a temporary profiling-only NVTX loader using runtime `dlopen`/`dlsym` for `nvtxRangePushA`, `nvtxRangePop`, and `nvtxMarkA`, gated by `PEGAINFER_DSV4_NVTX=1`. The instrumentation marked rank worker decode stages (`attn_local`, `indexer_ar`, `attention_ar`, `moe_route`, `moe_plan`, `moe_experts`, `moe_reduce`, `shared_expert`, `moe_ar`) plus active local expert counts and per-local-expert ranges. The temporary code was removed after the trace, so it is not part of the hot path.
  - Build and trace commands:

```bash
PEGAINFER_NVCC_JOBS=8 cargo build --release -p pegainfer-server --bin bench_serving --features deepseek-v4

PEGAINFER_DSV4_NVTX=1 nsys profile --stats=false --force-overwrite=true \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  --delay=34 --duration=12 \
  -o target/profiling/dsv4_rank_stage_proof \
  target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash --format json \
  request --prompt-len 1 --output-len 32 --warmup 1 --iters 1

nsys export --type sqlite --force-overwrite=true \
  -o target/profiling/dsv4_rank_stage_proof.sqlite \
  target/profiling/dsv4_rank_stage_proof.nsys-rep
```

  - Trace artifacts:
    - `target/profiling/dsv4_rank_stage_proof.nsys-rep`
    - `target/profiling/dsv4_rank_stage_proof.sqlite`
  - The nsys run was distorted (`149.51ms/token`) and ended with the known NCCL destructor abort after benchmark JSON. The trace is still useful for timeline proof.
  - Clean MoE proof case: `token=12`, `layer=17`, `moe_ar`, skew `1.200ms`.
    - Proof query: open `target/profiling/dsv4_rank_stage_proof.sqlite`; join `NVTX_EVENTS` to `StringIds` for `dsv4 rank=% token=12 layer=17 moe_ar%`, then join `CUPTI_ACTIVITY_KIND_KERNEL` to `StringIds` and `CUPTI_ACTIVITY_KIND_RUNTIME` for `ncclDevKernel_AllReduce_Sum_f32%` in the same timestamp window.
    - The 8 rank NCCL kernels belong to one logical all-reduce because their end timestamps collapse at the same time: rank0 `+0.018..+1.238ms`, rank5 `+0.064..+1.238ms`, rank2 `+0.147..+1.238ms`, rank1 `+0.165..+1.238ms`, rank7 `+0.192..+1.238ms`, rank6 `+0.400..+1.237ms`, rank3 `+0.950..+1.238ms`, rank4 `+1.218..+1.238ms`.
    - `rank0` entered MoE all-reduce at `+0.000ms`; its NCCL kernel lasted `1.220ms`.
    - `rank4` entered at `+1.200ms`; before that it was still in `moe_experts` `+0.176..+1.042ms`, `moe_reduce` `+1.043..+1.080ms`, and `shared_expert` `+1.081..+1.199ms`. Its own NCCL tail was only `0.0198ms`.
    - NVTX expert ranges show `rank4 active_local_experts=3`: global experts `129` (`+0.221..+0.632ms`), `143` (`+0.632..+0.873ms`), and `158` (`+0.875..+1.042ms`).
    - Interpretation: this is a direct proof that the long NCCL duration on early ranks is wait time; the late rank had not reached the all-reduce because it was still running MoE expert/shared work.
  - Extreme MoE phase-skew case: `token=22`, `layer=12`, `moe_ar`, skew `7.174ms`.
    - `rank5` entered at `+0.000ms`; its NCCL kernel ran `+0.018..+7.433ms`.
    - `rank7` entered at `+7.174ms`; it was in `moe_route` until `+6.738ms`, then `moe_plan`, `moe_expand`, `moe_experts` `+6.848..+7.084ms`, and `shared_expert` `+7.100..+7.173ms`.
    - Interpretation: a 7ms NCCL kernel here is still mostly waiting, but the full 7ms should not be attributed to current-layer expert compute. The late rank already carried phase delay before this MoE stage, so skew can propagate across barriers.
  - Indexer proof case: `token=28`, `layer=26`, `indexer_ar`, skew `1.199ms`.
    - `rank1` entered indexer all-reduce at `+0.000ms`; its NCCL kernel ran `+0.019..+1.231ms`.
    - `rank3` entered at `+1.199ms`; it was still inside `attn_local ratio=4` from `+0.644..+1.554ms`.
    - Interpretation: indexer/attention-side collectives also show early ranks waiting for late ranks that are still in local decode work; this is not unique to MoE expert GEMMs.
  - Practical lesson: proof should be taken from a concrete barrier window. A clean `~1ms` skew can be tied to active local expert count or ratio-4 local attention work. Larger multi-ms skew often includes phase delay carried from earlier barriers, so it should be treated as propagation unless the local NVTX window proves otherwise.
- Failed attempts recorded:
  - **Uninitialized hot outputs / memset slice**: changed hot outputs from zeroed allocation to uninitialized allocation in FP4/FP8/MoE/logits paths. `cargo check` passed, but 1x32 bench regressed from `94.84ms/token` to `107.52ms/token`. Reverted. Lesson: memset/zeroing was not the dominant Amdahl bucket, and allocation semantics can perturb scheduling enough to lose.
  - **Shared W1/W3 FP4 activation quantization**: added a paired FP4 linear wrapper to reuse one `act_quant_k4096` for expert W1/W3. `cargo check` passed, but 1x32 bench regressed to `119.63ms/token`. Reverted. Lesson: removing one quant launch per active expert did not address rank arrival skew and worsened wall time.
  - **Decode MoE Rust-side expert scratch reuse**: added write-into-buffer FP4/SwiGLU helpers and thread-local buffers for direct top-k MoE W1/W3/activation/W2 intermediates. `cargo check` passed, but 1x32 bench was `80.99ms/token`, slightly worse than the committed `80.49ms/token` baseline. Reverted. Lesson: reusing only the BF16 expert intermediates does not move the current Amdahl bucket enough; remaining work should target route/expert scheduling sync, broader allocation churn, or collective count/skew.
  - **Persistent worker CPU affinity**: bound rank workers to CPU0-3 for GPU0-3 and CPU32-35 for GPU4-7 based on `nvidia-smi topo -m` NUMA affinity. `cargo check` passed, but 1x32 bench regressed to `105.66ms/token`. Reverted. Lesson: hard pinning to one CPU per worker can worsen host/runtime scheduling; NUMA affinity is not the main remaining bucket.
  - **Move routed MoE all-reduce before shared expert**: routed expert output and shared expert output are mathematically independent, so this tried to enter the routed f32 all-reduce immediately after routed expert work and run shared expert after the barrier. `cargo check` passed, but 1x32 bench regressed to `85.67ms/token`. Reverted. Lesson: shared expert preceding the MoE all-reduce is not the dominant cause of observed skew; changing barrier order can hurt NCCL/stream scheduling even when numerically valid.
  - **Rank0-only route broadcast**: because decode FFN inputs and full gate weights should be identical across ranks, rank0 computed route weights/indices and broadcast the 6 f32 weights plus 6 i32 indices to the other ranks. `cargo check` passed, but 1x32 bench regressed to `86.98ms/token`. Reverted. Lesson: duplicated gate computation is cheaper than adding two tiny NCCL barriers per layer; route broadcast increases collective count and does not address the larger expert-work skew bucket.
  - **Shared expert before routed expert**: moved the dense shared expert branch before route selection and routed expert execution, hoping to put uniform dense work before the route-dependent skew and then enter the routed all-reduce immediately after routed work. `cargo check` passed, but 1x32 bench regressed to `87.58ms/token`. Reverted. Lesson: local operation reordering without reducing collectives or expert-work imbalance is not enough and can perturb stream/cache scheduling negatively.
  - **BF16 tensor-op score gate**: changed non-hash score routing from repeated BF16-to-F32 conversion plus FP32 `cublasSgemv`/pedantic GEMM to direct cuBLAS BF16-input, FP32-output tensor-op GEMM. `cargo check` passed, but 1x32 bench regressed to `85.03ms/token`. Reverted. Lesson: the repeated gate-weight conversion looks inefficient but is not the current Amdahl bucket; route-gate math should not be changed without an exact E2E pass because small top-k differences are possible.
  - **BF16 decode attention all-reduce**: changed decode attention output all-reduce from BF16->F32 scratch, NCCL F32 sum, F32->BF16 back to direct BF16 all-reduce. `cargo check` passed, but 1x32 bench regressed to `86.51ms/token`. Reverted. Lesson: reducing NCCL payload/convert work without reducing rank-arrival waiting does not move TPOT; the nsys post-arrival tail estimate was the right weighting.
- Decode-only top-k MoE path:
  - Replaced the decode rank-lane MoE path only. Instead of building the packed `topk * local_experts = 192` slot layout for `seq_len=1`, the rank copies the 6 route indices to host, runs only the top-k experts owned by that rank directly on the current token, and accumulates each BF16 expert output into the f32 routed output with the device-side route weight. Group/prefill paths still use the packed fused mapping.
  - Validation:

```bash
cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4

PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- \
  --model-path /data/DeepSeek-V4-Flash --format json \
  request --prompt-len 1 --output-len 32 --warmup 1 --iters 1

PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- \
  --model-path /data/DeepSeek-V4-Flash \
  --ground-truth test_data/deepseek-v4-ground-truth.json \
  --max-new-tokens 64
```

  - `cargo check` passed with the existing unreachable-pub warnings in `runtime/core.rs` and `runtime/state.rs`.
  - 1x32 bench result:
    - load: `35.16s`
    - TTFT: `174.70ms`
    - first decode step: `76.26ms`
    - steady TPOT avg: `80.49ms/token`
    - p50: `80.23ms`
    - p95: `84.34ms`
    - e2e: `2.67s`
  - Full exact DeepSeek V4 E2E passed all 20 cases. Representative long-output cases:
    - case 4: `21` tokens, `87.45ms` TPOT
    - case 9: `21` tokens, `87.64ms` TPOT
    - case 12: `15` tokens, `88.76ms` TPOT
    - case 18: `14` tokens, `88.91ms` TPOT
  - Interpretation: this validates the Amdahl analysis. The win is not from making FP4 GEMMs faster; it removes decode-only packed mapping/expand/reduce work and the host loop over 32 local experts, so ranks reach the MoE all-reduce with less phase skew and less runtime fragmentation. The remaining TPOT still pays 43 MoE all-reduces, 43 attention all-reduces, and 21 indexer all-reduces per token.
- Reprofiled the direct top-k MoE decode path:

```bash
nsys profile --stats=false --force-overwrite=true \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  --delay=34 --duration=12 \
  -o target/profiling/dsv4_decode_direct_topk_moe_1x32 \
  target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash --format json \
  request --prompt-len 1 --output-len 32 --warmup 1 --iters 1

nsys export --type sqlite --force-overwrite=true \
  -o target/profiling/dsv4_decode_direct_topk_moe_1x32.sqlite \
  target/profiling/dsv4_decode_direct_topk_moe_1x32.nsys-rep
```

  - Trace artifacts:
    - `target/profiling/dsv4_decode_direct_topk_moe_1x32.nsys-rep`
    - `target/profiling/dsv4_decode_direct_topk_moe_1x32.sqlite`
  - The nsys run was distorted (`122.89ms/token`) and ended with the known NCCL destructor abort after benchmark JSON. Use it for composition and ordering, not absolute TPOT.
  - Direct top-k MoE removed the intended packed-layout decode work. In this trace, `deepseek_moe_local_mapping_kernel`, `deepseek_moe_expand_to_fused_kernel`, and `deepseek_moe_reduce_fused_f32_kernel` each have only `416` launches; the new `deepseek_moe_accumulate_weighted_bf16_to_f32_kernel` has `14,560` launches but only `12.25ms` total in the trace.
  - The remaining f32 all-reduce bucket is still mostly arrival skew, not NCCL transfer:
    - `32` final logits all-gather groups give token boundaries.
    - steady tokens `2..30` each have `107` f32 all-reduce collectives.
    - average f32 collective skew is `0.572ms`; average post-arrival tail is `0.0193ms`.
    - average per steady token skew is `61.17ms/token`; average post-arrival tail is `2.07ms/token`.
    - per-token skew increased in the tail of the nsys window (`~74-80ms` on tokens 27-30), so the trace shows profiler/runtime drift in addition to real skew.
  - Kernel composition after direct top-k MoE:
    - raw f32 NCCL kernel sum: `17.86s` in trace, still mostly wait time.
    - routed FP4 expert GEMM totals remain large: W1/W3 `1.97s`, W2 `0.51s`.
    - packed MoE layout kernels are no longer material (`<1ms` each).
    - FP4 expert work is still uneven by device; rank/device 6 is highest (`4236` W1/W3 kernels, `279.5ms`) while device 4 is lowest (`3452`, `225.6ms`). This supports route-dependent local expert work as a remaining skew source.
  - Runtime/API churn remains large on rank workers:
    - `cudaLaunchKernel_v7000`: `1.22M` calls.
    - `cuMemAllocAsync`: `836.5K` calls; `cuMemFreeAsync`: `836.1K` calls.
    - `cuMemcpyDtoHAsync_v2`: `20.3K` calls; `cuStreamSynchronize`: `19.9K` calls.
    - The direct top-k path still does one route-index D2H sync per rank per layer; removing packed layout lowered work, but did not make routing/expert scheduling GPU-resident.
- Next action:
  - Stop chasing raw kernel-launch and memset counts until a change shows it reduces per-token skew.
  - The next experiment should target route/expert scheduling or allocation churn, not packed MoE layout. Candidate directions: remove the decode MoE route-index D2H/sync, preallocate per-rank MoE scratch for W1/W3/activation/W2 outputs, or reduce/fuse f32 collectives so each remaining rank-local imbalance is paid fewer times.
  - Any BF16/faster all-reduce experiment must be judged by exact E2E plus logical-collective wall analysis. A faster NCCL tail alone cannot move TPOT much while `~80-100ms/token` is arrival skew.

## Debrief

- **Outcome**: Official DeepSeek TileKernels MoE source was inspected at commit `36d9e45d38e204ebb87e6f6e833821eee0482fe5`. The main finding is that official MoE support centers on GPU-resident routing selection, TP/EP masking/remapping, expert-major fused mapping, input expansion, and output reduction. A first packed-layout experiment preserved exact output but did not improve decode. Rank-thread decode showed the largest immediate issue was CPU rank dispatch/rank arrival skew, improving 1x32 steady TPOT to `113.54ms/token`; persistent rank workers pushed it further to `94.84ms/token`. Persistent-worker nsys then showed the remaining large bucket is not raw NCCL transfer. It is repeated rank phase skew before `107` f32 collectives per token; EP8 routed expert imbalance contributes, but runtime/allocator/host-loop amplification is necessary to explain the 100ms-scale TPOT. A decode-only direct top-k MoE path then cut 1x32 steady TPOT to `80.49ms/token`, with all 20 exact cases passing.
- **Pitfalls encountered**:
  - The name `moe` is easy to misread as a fused expert MLP implementation. In this repository it is mostly layout and routing infrastructure.
  - Local exact text passing does not prove the execution path matches official routing topology; it can still be correct enough for text while doing too much full-batch expert work and synchronizing through host route indices.
  - Expert-major packing alone is not sufficient for decode. With `seq_len=1`, packed `topk * local_experts` layout can add more scheduling and mapping overhead than it removes; decode needs a separate path or grouped/fused routed expert execution.
  - Raw NCCL kernel-duration sums can mislead because they include overlapped per-rank kernels and spin-wait for slower-arriving ranks.
  - Optimizing small-looking details without Amdahl weighting can regress: uninitialized hot allocations and W1/W3 shared FP4 quantization both passed compile checks but made 1x32 TPOT worse.
- **Lessons learned**:
  - For DeepSeek V4 MoE, routing indices should become GPU-resident execution metadata. The CPU should not decide active experts from a full D2H route-index copy in the hot path.
  - The next useful kernel import is likely `mask_indices_by_tp` / `group_count` or `get_fused_mapping`, not `topk_gate`, but decode performance ultimately needs grouped/fused routed expert execution or fewer collective barriers.
  - DeepSeek MP8 decode should keep the persistent worker architecture. Worker-owned contexts/comms removed the per-layer thread lifecycle cost and lowered the dispatch skew enough to make smaller GPU buckets visible.
  - After persistent workers and direct top-k MoE, the next performance question is why ranks still reach each collective at different times. Current evidence points to route-dependent MoE expert work count plus runtime/API amplification from allocation/free, launch gaps, D2H synchronization, and host-controlled route/expert scheduling, not NCCL link bandwidth.
- **Follow-ups**:
  - Build the next experiment around reducing remaining MoE route/expert scheduling sync, rank-local imbalance, runtime allocation churn, or the number of f32 collectives.
  - Add a targeted MoE routing parity test against official `top2_sum_gate` semantics for the current config.
  - Decide whether to keep packed layout only for prefill and add a separate decode-specific routed expert kernel.
