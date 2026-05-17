# DeepSeek V4 Decode Performance

**Created**: 2026-05-12
**Status**: active

## TL;DR

DeepSeek V4 fixed long-decode TPOT moved from the `~108-113ms/token` band to retained sub-`30ms/token`; stable sub-`25ms/token` remains open.

Retained runtime changes:

- GPU-resident grouped MoE route mapping, pointer caching, rank-owned scratch, and removal of hot temporary zero-fill.
- Routed FP4 `W13 grouped GEMM -> fused SwiGLU + W2 activation quant -> W2 grouped GEMM`.
- Shared expert fused W1/W3 and fused SwiGLU+W2 FP8 paths.
- MoE all-gather/reduce-scatter overlap with shared expert compute.
- Fused Q/KV RoPE, score-route BF16 direct GEMM cleanup, parallel score-select, grouped FP4 shared-memory downsizing, and benchmark/counter instrumentation.
- Removal of old split MoE/SwiGLU public and FFI entry points so runtime callers stay on the retained path.

Current evidence:

- Exact E2E remains `20/20`.
- Fixed bench generated-token hash remains `6346f03343d75a65`.
- Best retained repeats reached the `26.28-27.31ms/token` band.
- Fresh reviewer 5090 5-run sweep observed aggregate steady TPOT avg `27.55-29.76ms`, with all 15 hashes matching `6346f03343d75a65`.
- Rejected experiments are recorded below with microbench/E2E/fixed-bench evidence; they are not retained production paths.

## Team Lessons

- Compare identical token traces, not only exact E2E summaries.
- Separate NCCL wait-inclusive time from pure transfer time.
- Treat capacity and logical length as different optimization variables.
- Keep MoE semantic zero on device.
- Prove allocation cleanup with application-visible CUDA API counters rather than Nsight attribution alone.

## Current Completion Audit

| Requirement | Evidence | Status |
| --- | --- | --- |
| Main objective: stable sub-`25ms/token` DeepSeek V4 decode without bs=1 or seq_len=1 specialization | Best retained repeats reached the `26.28-27.31ms/token` band; fresh 5-run stability sweep after the latest rejected act_quant probe is `28.29-28.91ms` aggregate steady TPOT while another CPU load was running | Not achieved. Keep the goal active. |
| Fixed bench stable sub-30 with hash `6346f03343d75a65` | `/tmp/dsv4_stability_after_act_quant_revert_{1..5}.json` records 5 consecutive fixed bench runs, aggregate steady TPOT avg `28.291-28.912ms`, and all 15 per-iteration hashes `6346f03343d75a65`; reviewer rerun `/tmp/pegainfer_dev_pr101_bench_{1..5}.json` observed aggregate steady TPOT avg `27.552965-29.755957ms`, again with all 15 hashes `6346f03343d75a65` | Achieved for the retained tree. |
| Exact E2E remains `20/20` | `/tmp/dsv4_fresh_e2e_after_w2_reduce_doc.log` records `All 20 DeepSeek V4 exact cases passed` | Achieved for the retained tree. |
| Public vLLM/SGLang MoE decomposition is replicated first | Runtime uses routed FP4 `W13 grouped GEMM -> fused SwiGLU + W2 activation quant -> W2 grouped GEMM`; old split W1/W3/SwiGLU/W2 public and FFI paths are removed | Achieved. |
| Deeper W13 accumulator -> SwiGLU -> W2-quant path is explored only after microbench/fuzz | TileLang W13 accumulator prototype was compiled after lowering fixes but failed the first active-expert fuzz shape, so it was removed before runtime integration | Explored and rejected; still open as a future true tensor-core epilogue project. |
| Evidence chain records source anchor, decision, microbench, E2E, fixed bench, hash, TPOT band | The roadmap, baseline table, rejection sections, and evidence log below record retained and rejected MoE attempts. Temporary rejected bench sources were deleted after logging unless they remain as diagnostic microbench tools. | Sufficient for current retained/rejected attempts; continue adding entries for new attempts. |
| Project documentation captures reusable team lessons | This document is the active DeepSeek V4 decode performance record and `docs/index.md` routes to it | Achieved. |

Audit conclusion: the goal is not complete because stable sub-`25ms/token` has not been demonstrated. The next accepted code change must either reduce the large absolute attention/HC local cost or make a real grouped FP4 W13/W2 scheduler/epilogue improvement; isolated standalone SwiGLU/quant substitutions have already been rejected.

## vLLM/SGLang MoE Roadmap

The goal is to copy the mature decomposition and validation discipline, not the framework surface. This table is the current "homework ledger" for DeepSeek V4 decode MoE:

| Source idea | vLLM/SGLang anchor | PegaInfer status | Decision |
| --- | --- | --- | --- |
| Experts core decomposition: `W13 grouped GEMM -> activation/quant -> W2 grouped GEMM` | vLLM `docs/design/fused_moe_modular_kernel.md`; vLLM `fused_moe/modular_kernel.py`; SGLang `srt/layers/moe/moe_runner/triton.py` | Retained as routed FP4 W13 grouped launch plus fused SwiGLU+W2 activation quant plus W2 grouped FP4 launch | Adopted. This is the baseline decomposition and the old split W1/W3/SwiGLU/W2 public path is removed. |
| Prepare/finalize can be separate from experts | vLLM `FusedMoEPrepareAndFinalizeModular`; SGLang EP MoE dispatcher/finalizer split | Our AG/RS, route mapping, local experts, partial combine, and reduce-scatter are explicit stages | Adopted selectively. We keep the simpler PegaInfer scheduler/worker structure rather than importing generic dispatch classes. |
| Async prepare/finalize enables shared expert overlap | vLLM `prepare_no_receive`; vLLM modular-kernel doc notes shared expert overlap during communication | MoE hidden/token all-gather and reduce-scatter run on a MoE NCCL stream while shared expert runs on the main compute stream | Adopted. Full shared-compute-stream overlap changed token hash and was rejected. |
| `TopKWeightAndReduce` may live inside experts | vLLM `topk_weight_and_reduce.py`; vLLM `FusedMoEExpertsModular::finalize_weight_and_reduce_impl` | Atomic epilogue-shaped microbench was slower than current deterministic reduce | Rejected for current layout. This needs a token-major or deterministic W2 scheduler, not atomics bolted onto expert-major TileLang W2. |
| W13 layout should match fused SwiGLU convention | vLLM `oracle/mxfp4.py`; vLLM `quantization/utils/flashinfer_utils.py`; SGLang `moe_runner/flashinfer_cutedsl.py` | Pair-interleaved `[up, gate]` standalone SwiGLU+quant was byte-identical but mostly flat and tiny | Rejected as standalone. Keep the note for a true W13 epilogue fusion. |
| FlashInfer/TRTLLM/CuteDSL FP4 MoE backends use specialized weight/scale reorder | vLLM `experts/trtllm_mxfp4_moe.py`; vLLM `oracle/mxfp4.py`; SGLang `quantization/mxfp4.py`; SGLang `moe_runner/flashinfer_trtllm.py` | Not integrated. Current PegaInfer weights are per-expert tensors and TileLang grouped GEMM takes pointer arrays; FlashInfer routes expect different packed/reordered layouts and runner-level metadata | Candidate, but only after a standalone grouped-GEMM microbench proves a real W13/W2 win on our exact shapes. Do not import the framework runner. |
| DeepGEMM-style deeper epilogue fusion | vLLM `experts/deep_gemm_moe.py`; SGLang DeepGEMM benchmarks under `benchmark/kernels/deepseek` | Scalar upper-bound microbench shows exact feasibility but absolute standalone delta is tiny | Candidate only as true tensor-core W13 epilogue work. Standalone SwiGLU/quant substitutions are no longer enough. |
| FP4 quant before communication for high-throughput all-gather | SGLang `srt/layers/moe/utils.py::should_use_flashinfer_cutlass_moe_fp4_allgather` | Not adopted. Our current AG gathers BF16 hidden before routing; changing this means routing/dispatch protocol changes, not a local kernel swap | Future architecture work. Needs correctness design because router consumes hidden before expert dispatch. |
| High-throughput bs>100 packed MoE layout | vLLM/SGLang packed FP4/MXFP4 backends and dispatcher/finalizer layouts | Not part of the current sub-25 latency patch. Current per-expert tensors are good for low-latency iteration but probably not the final throughput layout | Future architecture work. Design W13/W2 weight layout, FP4 scale layout, dispatch row layout, and combine/finalize together; keep conversion offline/load-time and avoid two production hot paths. |

Current rule for new MoE attempts:

1. First identify the source idea and path above.
2. Write a standalone microbench or trace that isolates the claimed section.
3. Reject ideas that only save `~0.001ms/layer` unless they also remove a larger runtime barrier.
4. Enter runtime only after the microbench and layout story match our expert-major AG/RS path.
5. Preserve fixed-bench hash `6346f03343d75a65`; exact E2E alone is not enough.

Scope guard for the active latency work:

- Optimize small-BS decode latency first, because that is where the current DSV4 TPOT gap is measured.
- Do not add `bs == 1`, `seq_len == 1`, or fixed prompt/output shape branches.
- Keep new kernels and scheduling changes batch-general and expert-general, even when the fixed bench uses prompt len `1`.
- Accept improvements that reduce launches, intermediate writes, rank skew, or grouped-GEMM overhead across small batches.
- Keep bs>100 packed-layout work as a separate throughput architecture track until a standalone microbench justifies changing the production MoE layout.

Large-batch note: the current goal is still low-latency decode, but future bs>100 throughput should not be constrained by the current per-expert pointer-array layout. For large batches, expert GEMM throughput, FP4 scale coalescing, dispatch layout, and W2 combine/finalize cost become the main design variables. The intended direction is `checkpoint tensors -> load-time/offline conversion -> one packed MoE production layout`, not long-term runtime branching between old per-expert and new packed layouts.

Concrete source anchors for the large-batch path:

- SGLang `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py::align_fp4_moe_weights_for_flashinfer_trtllm` rewrites W13/W2 weights and scales in place for FlashInfer TRT-LLM FP4 kernels.
- SGLang `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py::_pack_topk_for_flashinfer_routed` packs top-k ids and weights into the routed backend's expected int32 format.
- SGLang `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py::fused_experts_none_to_flashinfer_trtllm_fp4` quantizes hidden states to FP4, interleaves scales, and calls `trtllm_fp4_block_scale_routed_moe`.
- SGLang `sgl-kernel/csrc/common_extension.cc` exposes `prepare_moe_input`, `fp8_blockwise_scaled_grouped_mm`, and `cutlass_w4a8_moe_mm` with `problem_sizes` and expert offsets as first-class tensors.
- vLLM `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py::prepare_static_weights_for_trtllm_fp4_moe` does per-expert W13/W2 row permutations and `nvfp4_block_scale_interleave` for FP4 scales before runtime.
- vLLM `vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py` wraps `flashinfer.trtllm_fp4_block_scale_routed_moe` as an expert backend and lets the modular layer handle routing/finalization.

Decision: these sources confirm that the mature high-throughput path is a packed-weight, packed-scale, packed-topk, problem-size-aware backend. That is not a safe small latency patch on top of the current per-expert TileLang pointer-array path; it should become a separate packed-layout project with a microbench gate before any runtime migration.

## Baseline And Result

Use this fixed decode bench for comparable DeepSeek V4 direct-runtime work:

```bash
target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash \
  --format json \
  request \
  --prompt-len 1 \
  --output-len 160 \
  --warmup 2 \
  --iters 3 \
  --seed 42
```

| Milestone | Fixed long decode | Key change |
| --- | ---: | --- |
| AG/RS grouped MoE baseline | `107.92-112.61ms/token` | GPU AG/RS path existed, but repeated pointer-array setup and per-token allocations remained. |
| Grouped MoE pointer cache | `83.37-89.65ms/token` | Cache grouped FP4 expert weight/scale pointer arrays per rank worker. |
| Rank-worker affinity | `72.88-73.60ms/token` | Reduce rank-arrival skew before f32 collectives. |
| Remove hot temporary zero-fill | `63.35-64.51ms/token` | Fully overwritten hot temporaries allocate uninitialized storage. |
| Rank-owned decode scratch | `34.34-35.36ms/token` forced NUMA, `32.75-33.90ms/token` same-code fast band | Move hot intermediate storage to per-rank scratch and remove grouped FP4 C-side growth-cache workspace. |
| Final PR validation | `35.253ms/token` | After review fixes: dynamic NUMA topology, buffer-derived capacity checks, and `_ptsz` counter separation. |
| Shared W1/W3 act quant | `33.330ms`, repeated `34.289ms` | Decode scratch W1 and W3 reuse one TileLang `act_quant_k4096`; token hash stays `6346f03343d75a65`, exact E2E `20/20`. |
| W13 grouped runtime launch | text `34.22ms`, JSON `31.986ms` | W1 and W3 share one TileLang grouped FP4 launch after shared activation quant; token hash stays `6346f03343d75a65`, exact E2E `20/20`. |
| Routed fused SwiGLU + W2 act quant | `33.416ms`, repeated `31.180ms` | Mirror vLLM/SGLang's activation+quant fusion after materialized W13 output; token hash stays `6346f03343d75a65`, exact E2E `20/20`. |
| Shared expert fused quant + dense W13 | `29.764ms`, repeated `31.592ms` | Shared expert scratch path reuses one FP8 act quant for W1/W3, fuses shared SwiGLU+W2 act quant, and uses one dense FP8 W13 launch; token hash stays `6346f03343d75a65`, exact E2E `20/20`. |
| Clean sub-30 repeat after trace cleanup | `29.944ms`, `29.907ms`, `29.896ms` | Same fixed bench after removing per-layer trace syncs; all three measured iterations keep token hash `6346f03343d75a65`. |
| Fused MoE mapping clear | `29.862ms`, `29.969ms`, `29.874ms` | Merge six local-route mapping clear launches into one kernel; exact E2E `20/20`, token hash stays `6346f03343d75a65`. |
| Small-route MoE mapping | first run `27.608ms`, `27.662ms`, `27.826ms`; repeat `27.698ms`, `27.693ms`, `27.644ms` | Decode route mapping uses one small-batch kernel for `route_elems <= 1024`; exact E2E `20/20`, token hash stays `6346f03343d75a65`. |
| Current retained calibration | `28.940ms`, `28.942ms`, `28.913ms` | Same retained runtime after tool/doc-only commits and 5090 resync; still stable sub-30 with hash `6346f03343d75a65`, but not a new best. |
| Fused Q/KV RoPE projection | run 1 `28.215ms`, `28.256ms`, `28.236ms`; repeat `27.096ms`, `28.565ms`, `28.349ms` | Attention projection uses existing batch-general `deepseek_apply_rope_q_kv_cuda` instead of two separate Q and KV RoPE launches; exact E2E `20/20`, token hash stays `6346f03343d75a65`. |
| Remove old split MoE/SwiGLU entry points | `27.863ms`, `27.845ms`, `27.872ms` | Non-scratch routed/shared expert helpers now call the same W13 + fused SwiGLU-quant + W2 path as decode scratch; generic grouped FP4 linear and standalone SwiGLU clamp FFI/public Rust entry points are gone. Exact E2E `20/20`, token hash stays `6346f03343d75a65`. |
| MoE reduce-scatter/shared overlap | run 1 `26.773ms`, `26.794ms`, `26.805ms`; repeats `27.757-27.783ms` and `27.637-27.985ms` | Decode MoE reduce-scatter uses a dedicated NCCL communicator/stream and CUDA events so shared expert can run on the main compute stream while routed reduce-scatter is in flight. Exact E2E `20/20`, token hash stays `6346f03343d75a65`; improvement is retained but still noisy. |
| Post full-overlap revert calibration | `28.536ms`, `28.387ms`, `28.381ms` | Same retained MoE reduce-scatter/shared overlap after removing the rejected full shared-expert overlap experiment from local and 5090. Exact E2E `20/20`, token hash restored to `6346f03343d75a65`. |
| MoE all-gather/reduce-scatter shared overlap | run 1 `26.924ms`, `26.923ms`, `27.309ms`; repeat `26.280ms`, `26.293ms`, `26.282ms` | Hidden/token all-gather and routed reduce-scatter both use the MoE NCCL stream. The main compute stream runs shared expert while all-gather is in flight, then waits before router/local expert; exact E2E `20/20`, token hash stays `6346f03343d75a65`. |
| Restored all-gather overlap after split probe | `27.195ms`, `27.203ms`, `27.198ms` | Rebuilt 5090 after dropping the shared-W13/shared-W2 split experiment; code is back on the simpler retained all-gather overlap path, hash stays `6346f03343d75a65`. |
| Restored after WQ_A/WKV shared-quant probe | `28.428ms`, `28.453ms`, `28.050ms` | WQ_A/WKV shared input FP8 quant was reverted after regressing to aggregate `28.672ms`; post-revert exact E2E is `20/20`, and the fixed bench hash stays `6346f03343d75a65`. |
| Post MoE stage-trace cleanup | `27.973ms`, `27.992ms`, `26.661ms` | Temporary host-sync MoE trace was removed from local and 5090; aggregate steady TPOT avg `27.542ms`, token hash stays `6346f03343d75a65`. |
| Score-route BF16 direct GEMM | final run `28.843ms`, `28.832ms`, `29.263ms`; repeat `28.637ms`, `28.536ms`, `28.576ms` | `deepseek_score_gate_cuda` now feeds BF16 `x` and BF16 gate weights directly to `cublasGemmEx` with F32 accumulation, removing per-token BF16-to-F32 conversion of `x` and static score-gate weights. Exact E2E remains `20/20`; token hash stays `6346f03343d75a65`. |
| Grouped FP4 shared-memory downsize | run 1 `28.464ms`, `28.467ms`, `28.512ms`; repeat `28.713ms`, `28.684ms`, `28.685ms` | TileLang grouped FP4 W13/W2 wrappers request `32768` dynamic shared bytes instead of the old `98304`. Microbench shows W13/W2 sparse decode shapes speed up while bitwise outputs stay identical; full fixed bench remains exact and stable sub-30. |
| Parallel score-route top-k select | `28.343ms`, `28.468ms`, `28.562ms` | The post-cuBLAS score-select kernel now uses block-level reduction for top-k instead of thread-0 serial scan. Standalone microbench is bitwise and `1.50x` faster for `seq_len={1,8,16,32}`; exact E2E remains `20/20`, and fixed bench hash stays `6346f03343d75a65`. |
| Rejected hand routed W13 act_quant | `29.014ms`, `28.744ms`, `28.647ms` | A hand CUDA BF16->FP8/E8M0 act_quant kernel was `2.0-3.0x` faster than TileLang in microbench, but runtime fixed bench regressed to aggregate `28.802ms`; reverted to TileLang act_quant for routed W13. |
| Restored after hand act_quant probe | `29.265ms`, `29.272ms`, `29.293ms` | Runtime restored to the retained TileLang routed W13 act_quant path; exact E2E remains `20/20`, and repeat fixed bench hashes are all `6346f03343d75a65`. |
| 5-run stability after hand act_quant revert | aggregate `28.912ms`, `28.867ms`, `28.291ms`, `28.375ms`, `28.715ms` | Five consecutive fixed bench runs, 15 measured iterations, all token hashes `6346f03343d75a65`. Another CPU load was running during this sweep, so use it as a conservative stability check rather than a clean best-band measurement. |
| Fresh retained-runtime validation | `28.505ms`, `28.470ms`, `28.466ms` | No new runtime change; this run revalidates the current retained tree after tool/doc-only MoE probes. Exact E2E remains `20/20`, and all three fixed-bench iterations keep hash `6346f03343d75a65`. |

Final PR validation on 5090:

| Metric | Value |
| --- | ---: |
| steady TPOT avg | `35.253ms` |
| steady TPOT p50 | `34.800ms` |
| steady TPOT p95 | `37.335ms` |
| first decode avg | `33.743ms` |
| generated-token hash | `6346f03343d75a65` |
| exact E2E | `20/20` |

## Retained Design

### Grouped MoE pointer cache

Each persistent rank worker builds a `MoeGroupedPtrCache` once after context binding. The cache stores per-layer GPU arrays for local expert weight pointers and scale pointers for W1/W2/W3 grouped FP4 linears. Decode and prefill MoE paths pass this cache to grouped FP4 local expert execution.

This removed repeated host vector construction and H2D pointer-array copies from every grouped FP4 call. The grouped FP4 kernels did not become materially faster in nsys; the improvement showed up as a shorter MoE reduce-scatter synchronization window, which points to lower rank-arrival skew.

### Rank-worker placement

Rank workers are pinned before CUDA work begins. The final PR path resolves topology dynamically:

1. CUDA driver `cuDeviceGetPCIBusId` maps CUDA ordinal to PCI bus id.
2. `/sys/bus/pci/devices/<pci>/numa_node` maps PCI to NUMA node.
3. `/sys/devices/system/node/node<numa>/cpulist` supplies target CPUs.
4. The target list is intersected with the process's allowed cpuset.
5. Missing topology, empty intersection, or `pthread_setaffinity_np` failure panics.

Do not encode ordinal assumptions such as `GPU0..3 -> NUMA0`. A review caught that earlier draft; it matched 5090 but was still a machine-specific fact in runtime logic. Also avoid CUDA runtime topology calls here: `cudaDeviceGetPCIBusId` loaded an incompatible `libcudart` on 5090, while the CUDA driver API path worked.

5090 final pin evidence:

| GPU ordinal | PCI bus | NUMA | pinned CPU |
| --- | --- | ---: | ---: |
| `0` | `0000:16:00.0` | `0` | `0` |
| `1` | `0000:27:00.0` | `0` | `1` |
| `2` | `0000:38:00.0` | `0` | `2` |
| `3` | `0000:5a:00.0` | `0` | `3` |
| `4` | `0000:98:00.0` | `1` | `36` |
| `5` | `0000:a8:00.0` | `1` | `37` |
| `6` | `0000:c8:00.0` | `1` | `38` |
| `7` | `0000:d8:00.0` | `1` | `39` |

### Rank-owned decode scratch

`RankDecodeScratch` is created once per rank worker and reused by decode commands. The current direct scheduler still sends one token per rank command, but the scratch design is capacity-based and should not assume batch size one in API contracts.

| Area | Scratch owner | Note |
| --- | --- | --- |
| Token upload | `RankDecodeScratch::token_ids` | Replaces per-token `clone_htod(&[token_id])` with H2D copy into rank-owned storage. |
| Entry hidden | `DecodeEntryScratch` | Embedding and HC expand outputs are fully overwritten. |
| HC pre/post | `HcPreNormScratch`, `HcPostScratch` | HC pre-state and layer outputs reuse rank-owned buffers; HC post layer output uses ping-pong slots to avoid adjacent-layer aliasing. |
| Attention | `AttentionProjectionScratch`, `AttentionIndexScratch`, `AttentionAuxScratch`, `AttentionOutputScratch` | Active ratio `0` and ratio `4` decode paths use capacity buffers with logical lengths passed separately. |
| Shared expert | `SharedExpertScratch` | Fixed-shape gate/up/out storage plus caller-owned FP8 activation/scale workspace for shared W1/W3 and W2. |
| MoE AG/RS | `MoeAgRsScratch` | Hidden/token all-gather, route buffers, compact maps, expert intermediates, partial routed output, local reduce-scatter output, routed+shared output. |
| Grouped FP4 workspace | `MoeAgRsScratch::{fp4_act_workspace,fp4_act_scale_workspace}` | Caller-owned workspace avoids the C-side grouped FP4 growth-cache/mutex path. |
| Final logits | `FinalLogitsScratch` | HC head, final norm, local logits, and gathered logits are reusable. |

### Capacity and logical length

Reusable scratch must not use mutable `seq_len` as allocation capacity. The final code exposes buffer-derived `seq_capacity()` helpers for `Bf16HiddenStates`, `F32HiddenStates`, and `HcHiddenStates`. Scratch-backed `*_into` operators check capacity from storage length, then set `seq_len` to the logical length for this decode step.

NCCL calls must use logical prefix slices, not whole-capacity buffers:

- BF16 hidden all-gather sends `hidden_dim * local.seq_len` and receives `hidden_dim * gathered_seq_len`.
- F32 reduce-scatter sends `hidden_dim * global.seq_len` and receives `hidden_dim * local_seq_len`.
- U32 token all-gather and ratio-4 indexer score all-reduce slice to logical prefixes.

### MoE dynamic content

MoE route values remain dynamic. Static storage does not mean static route content:

- route weights and indices change per token/layer.
- compact maps and `expert_indptr` depend on the route.
- local expert counters/cursors need semantic initialization.

Storage is capacity-based. Semantic clears remain inside `deepseek_moe_local_mapping_cuda` for counters/cursors/indptr and mapping sentinels.

### Old split MoE/SwiGLU cleanup

After the retained W13 and fused W2 paths, old split entry points became more dangerous than useful. The non-scratch routed local expert helper now allocates temporary gate/up/out and FP4 activation workspaces, then calls the same W13 grouped FP4 + fused SwiGLU-quant W2 helper as decode scratch. The non-scratch shared expert helper likewise uses shared W13 + fused W2 instead of split W1/W3/SwiGLU/W2.

The cleanup removed these Rust public entry points:

- `local_expert_forward_bf16_hidden`
- `swiglu_clamp_bf16_hidden`

It also removed these C/FFI entry points and helper kernels:

- `deepseek_moe_fp4_grouped_linear_cuda`
- `deepseek_moe_fp4_grouped_linear_with_workspace_cuda`
- `deepseek_moe_clear_bf16_cuda`
- `deepseek_moe_accumulate_weighted_bf16_to_f32_cuda`
- `deepseek_swiglu_clamp_cuda`

The cleanup is not a TPOT claim; decode scratch already used the fused path. Its purpose is to make the vLLM/SGLang-style decomposition the only routed/shared expert implementation path in this crate, so prefill or debug callers cannot drift back to split W1/W3/SwiGLU/W2 kernels by accident.

### MoE all-gather/reduce-scatter shared expert overlap

After local routed experts finish, the decode MoE path has two independent pieces of work:

```text
routed expert output -> f32 partial combine -> reduce_scatter
local hidden -> shared expert W13/fused W2
```

The first retained overlap path created a second NCCL communicator on a per-rank nonblocking stream for MoE reduce-scatter. CUDA events preserve dependencies:

1. main stream records `partial_routed` ready after `reduce_moe_fused_output_f32_into`.
2. MoE NCCL stream waits on that event and launches `reduce_scatter`.
3. main stream immediately launches the shared expert path.
4. main stream waits on the reduce-scatter completion event before adding routed and shared outputs.

The later retained path uses the same MoE NCCL stream for the earlier hidden/token all-gather too:

1. main stream records `input_ready` after FFN pre-norm.
2. MoE NCCL stream waits and launches hidden all-gather plus token all-gather for hash layers.
3. main stream launches shared expert while all-gather is in flight.
4. main stream waits on all-gather completion before router, local experts, and routed combine.
5. routed reduce-scatter still uses the MoE NCCL stream and the final add waits on its completion event.

This does not change route math, grouped GEMM shape, or batch/expert generality. It is not copied directly from vLLM/SGLang operator code; it is a PegaInfer scheduling step that becomes available once the vLLM/SGLang-style local expert path is exact and stable. The first reduce-scatter-only fixed bench run moved to `26.77-26.80ms/token`; two repeats landed in the `27.64-27.99ms/token` band; the post full-overlap revert calibration landed at `28.38-28.54ms/token`. Moving MoE all-gather to the same NCCL stream and overlapping shared expert with all-gather produced fresh repeated fixed benches at `26.28-27.31ms/token`, still with token hash `6346f03343d75a65`. Keep decision: retain. This is safer than the rejected full shared-expert overlap because shared expert stays on the main compute stream; only MoE collectives move to the MoE NCCL stream.

### Fused Q/KV RoPE projection

Attention projection used to apply RoPE to Q and KV with two separate calls:

```text
apply_rope_hidden(q)
apply_rope_hidden(kv)
fp8_act_quant_nope(kv)
```

The retained path uses the existing batch-general CUDA kernel:

```text
deepseek_apply_rope_q_kv_cuda(q, kv)
fp8_act_quant_nope(kv)
```

This does not fuse the KV no-PE FP8 quantization and does not specialize for decode `seq_len=1`. The grid covers `seq_len * (local_heads + 1) * rotary_pairs`, so prefill-shaped projection remains valid. The short nsys profile `/tmp/dsv4_qkv_rope_short.nsys-rep` shows `deepseek_apply_rope_q_kv_kernel` in the kernel summary; residual `deepseek_apply_rope_hidden_kernel` instances still come from compressor/indexer/final inverse RoPE paths.

Validation on 5090:

| Check | Result |
| --- | --- |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench run 1 | aggregate steady TPOT avg `28.236ms`, per-iteration `28.215ms`, `28.256ms`, `28.236ms`; all hash `6346f03343d75a65` |
| fixed bench run 2 | aggregate steady TPOT avg `28.003ms`, per-iteration `27.096ms`, `28.565ms`, `28.349ms`; all hash `6346f03343d75a65` |

Keep decision: retain. The win is small and noisy, but the change removes one projection RoPE launch per attention layer, preserves exact long-token trace, and uses an already-existing batch-general kernel rather than adding a new specialized path.

Rejected follow-up: fusing Q/KV RoPE with KV no-PE quant kept exact E2E `20/20` and preserved hash `6346f03343d75a65`, but the fixed bench regressed to aggregate steady TPOT avg `29.948ms`, with per-iteration `29.885ms`, `29.910ms`, and `30.048ms`. The fused kernel used one mixed launch for rope-pair blocks plus quant-reduction blocks. That saved a launch but combined two different task shapes, so it should not replace the retained two-kernel `Q/KV RoPE -> KV no-PE quant` path.

Rejected follow-up: avoiding `concat_topk_indices_into` in ratio-4 decode by writing window indices and compressed indices directly into `topk_idxs` kept exact E2E `20/20` and preserved hash `6346f03343d75a65`, but the fixed bench regressed to aggregate steady TPOT avg `29.541ms`, with per-iteration `29.551ms`, `29.539ms`, and `29.532ms`. Restore the original `window_idxs + compress_idxs -> concat -> topk_idxs` path; the concat launch is too small to matter, and the direct-write variant did not improve the downstream attention schedule.

After reverting the direct-write top-k experiment and rebuilding 5090 release binaries, the fixed bench returned to aggregate steady TPOT avg `28.333ms`, with per-iteration `28.316ms`, `28.336ms`, and `28.346ms`, all hash `6346f03343d75a65`.

Rejected follow-up: sharing the input FP8 activation quantization between attention WQ_A and WKV kept exact E2E `20/20` and preserved hash `6346f03343d75a65`, but the fixed bench regressed to aggregate steady TPOT avg `28.672ms`, with per-iteration `28.414ms`, `28.432ms`, and `29.169ms`. The wrapper was removed from Rust/CUDA/FFI, and the post-revert fixed bench returned to aggregate steady TPOT avg `28.310ms`, with per-iteration `28.428ms`, `28.453ms`, and `28.050ms`. The likely lesson is that WQ_A/WKV already launch efficient TileLang FP8 linears, while sharing quant adds an extra runtime wrapper and does not remove enough downstream work to matter.

## Active MoE Sub-25 Work

### Goal

Drive decode MoE below the current retained sub-`30ms/token` band toward stable sub-`25ms/token`. Optimizations must remain batch-general, keep route/tile scheduling on GPU, and preserve exact E2E `20/20`.

### Attempt: fuse SwiGLU with W2 activation quant

The local expert decode path originally did:

```text
W1 grouped FP4 GEMM -> gate BF16
W3 grouped FP4 GEMM -> up BF16
SwiGLU clamp -> activated BF16
TileLang act_quant(activated) -> FP8 activation + E8M0 scales
W2 grouped FP4 GEMM
```

The attempted branch replaced the decode scratch W2 input path with:

```text
W1 grouped FP4 GEMM -> gate BF16
W3 grouped FP4 GEMM -> up BF16
fused SwiGLU clamp + BF16 rounding + FP8/E8M0 quant
W2 grouped FP4 GEMM
```

The fused quant keeps the old semantic order by rounding the SwiGLU result to BF16 before FP8 quantization. That matters for exact output: skipping the BF16 intermediate rounding would not be the same operator.

The first implementation used one CTA per `(row, 128-column group)`. That was the wrong GPU shape for this workload: with about `64` routed rows and `16` scale groups, it launched about `1024` CTAs per W2 quant, while the TileLang act_quant shape is `ceil(rows / 32) * 16`, about `32` CTAs. This explains why launch-count fusion did not translate into stable TPOT.

Evidence from the row-per-CTA version:

| Run | Result |
| --- | --- |
| Exact E2E on 5090 | `All 20 DeepSeek V4 exact cases passed` |
| Fixed bench run 1 | steady TPOT avg `31.922ms`, p50 `31.417ms`, p95 `33.812ms`, hash `6346f03343d75a65` |
| Fixed bench run 2 | steady TPOT avg `34.939ms`, p50 `34.388ms`, p95 `37.047ms`, hash `6346f03343d75a65` |
| Fixed bench run 3 | steady TPOT avg `34.572ms`, p50 `34.022ms`, p95 `36.767ms`, hash `6346f03343d75a65` |

Keep/drop decision: do not treat the `31.922ms` run as evidence. The repeated runs show the row-per-CTA version is not a stable win. The useful retained lesson is that a fused kernel must preserve the original row-block quantization shape, not merely reduce kernel launches.

Follow-up repair used a 4-warp row-block kernel and restored exact E2E `20/20`, but the fixed long bench token hash changed from `6346f03343d75a65` to `00fd2d772e4b8886` and TPOT regressed to `35.554ms`. Drop decision: do not retain this path. It changes long decode behavior even though the short exact suite passed.

### Retained: share W1/W3 activation quant

The retained local-expert change is deliberately narrower:

```text
Before:
W1: act_quant(expanded_input) + grouped FP4 GEMM
W3: act_quant(expanded_input) + grouped FP4 GEMM

After:
act_quant(expanded_input) once
W1 grouped FP4 GEMM reuses act/scale
W3 grouped FP4 GEMM reuses act/scale
```

This preserves the original TileLang activation quantization bit path and only removes duplicate work on the identical W1/W3 input. W2, SwiGLU, combine, route planning, and collectives are unchanged.

Validation on 5090:

| Check | Result |
| --- | --- |
| `cargo fmt --check` | passed |
| `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench run 1 | steady TPOT avg `33.330ms`, p50 `32.858ms`, p95 `35.274ms`, hash `6346f03343d75a65` |
| fixed bench run 2 | steady TPOT avg `34.289ms`, p50 `33.979ms`, p95 `36.852ms`, hash `6346f03343d75a65` |

Keep decision: retain. The gain is modest and still noisy, but the method is sound: fixed token trace, exact-safe, and it removes one W1/W3 activation quant launch per layer without introducing new quant math. The next MoE step should be W13 grouped GEMM or GPU active tile list, not more W2 quant fusion.

### Retained: W13 grouped FP4 GEMM runtime launch

W13 was first evaluated as a pure operator change before touching runtime. The TileLang generator now emits:

```text
deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096
```

It is generated from the existing `N=2048,K=4096` FP4 grouped GEMM, but launches `grid.x=32` blocks:

```text
blockIdx.x 0..15   -> W1 pointer arrays -> gate output
blockIdx.x 16..31  -> W3 pointer arrays -> up output
```

The C++ tool `pegainfer-kernels/tools/deepseek_v4/w13_grouped_fp4_bench.cu` links the generated TileLang object directly and compares:

```text
baseline: grouped_gemm(W1) + grouped_gemm(W3)
candidate: grouped_w13_gemm(W1, W3)
```

Fuzz uses BF16 random input, TileLang `act_quant_k4096`, random FP4 bytes and bounded E8M0-like scale bytes, expert-major `expert_indptr` with empty experts, and bitwise BF16 output comparison for `gate` and `up`.

Verified compile command shape:

```bash
OUT_DIR=$(find target/release/build/pegainfer-kernels-* -maxdepth 1 -type d -name out -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_120 \
  -I/usr/local/cuda/include \
  pegainfer-kernels/tools/deepseek_v4/w13_grouped_fp4_bench.cu \
  "$OUT_DIR/libkernels_cuda.a" \
  -lcudart \
  -o /tmp/w13_grouped_fp4_bench
```

5090 microbench results:

| Rows | Experts | Fuzz | Baseline two GEMMs | W13 one GEMM | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| `64` | `4` | PASS | `0.126931ms` | `0.063485ms` | `1.999x` |
| `64` | `8` | PASS | `0.126988ms` | `0.122769ms` | `1.034x` |
| `64` | `16` | PASS | `0.300209ms` | `0.236817ms` | `1.268x` |
| `96` | `4` | PASS | `0.126970ms` | `0.063513ms` | `1.999x` |
| `96` | `8` | PASS | `0.126975ms` | `0.122862ms` | `1.033x` |
| `96` | `16` | PASS | `0.300129ms` | `0.238110ms` | `1.260x` |
| `160` | `4` | PASS | `0.127066ms` | `0.122937ms` | `1.034x` |
| `160` | `8` | PASS | `0.128706ms` | `0.124902ms` | `1.030x` |
| `160` | `16` | PASS | `0.303603ms` | `0.239735ms` | `1.266x` |
| `256` | `4` | PASS | `0.127242ms` | `0.124970ms` | `1.018x` |
| `256` | `8` | PASS | `0.246201ms` | `0.184480ms` | `1.335x` |
| `256` | `16` | PASS | `0.314086ms` | `0.251675ms` | `1.248x` |

Runtime integration replaces the two W1/W3 grouped GEMM launches after shared activation quant with the W13 launcher:

```text
act_quant(expanded_input) once
W13 grouped FP4 GEMM writes gate and up outputs
W2 path unchanged
```

The first runtime attempt failed exact E2E with `CUDA_ERROR_NOT_SUPPORTED` at decode layer 0. The cause was launch-wrapper setup inside CUDA graph capture: the new W13 kernel had not been used before capture, so its first `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, 98304)` could return `cudaErrorNotSupported`. The W13 wrapper now treats `cudaErrorNotSupported` like the existing `cudaErrorInvalidValue` tolerance and lets the actual launch result decide correctness.

Runtime validation on 5090:

| Check | Result |
| --- | --- |
| `cargo fmt --check` | passed |
| `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench text run | steady TPOT avg `34.22ms`, p50 `33.77ms`, p95 `36.53ms`, first decode avg `32.94ms` |
| fixed bench JSON run | steady TPOT avg `31.986ms`, p50 `31.458ms`, p95 `34.052ms`, first decode avg `30.544ms`, hash `6346f03343d75a65` |

Interpretation: W13 is exact at the operator level and in runtime, but the speedup depends heavily on routed-row distribution, local expert count, and run-to-run system noise. It is not automatically a `2x` W1/W3 win; some shapes mainly save launch overhead, while others are dominated by the expanded grid and grouped scheduling. Keep the runtime change because it removes one launch per layer and preserves the fixed token trace, but do not count the `31.986ms` run as stable sub-32 evidence until repeated long benches confirm it.

### Roadmap: mirror vLLM and SGLang decode MoE

The next long-running goal is to systematically absorb the mature vLLM/SGLang decode MoE decomposition and push beyond it only after the reproduced path is stable. The performance target is stable sub-`25ms/token` eventually, with stable sub-`30ms/token` as the first gate. This work remains batch-general and expert-general; do not introduce bs=1 or seq_len=1 special cases.

Reference source positions:

| Runtime | Source | Observed decode MoE shape |
| --- | --- | --- |
| vLLM | `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` | `cutlass_fp4_moe_mm(c1, W13)` writes `gate||up`, then `silu_and_mul_scaled_fp4_experts_quant(c1, ...)`, then `cutlass_fp4_moe_mm(W2)`. MXFP4 uses the same split with `silu_and_mul_mxfp4_experts_quant`. |
| vLLM C++ op registry | `/data/code/workspace-rustllm/vllm/csrc/libtorch_stable/torch_bindings.cpp` | Registers `silu_and_mul_scaled_fp4_experts_quant`, `silu_and_mul_mxfp4_experts_quant`, and grouped FP4/MXFP4 MoE GEMMs as separate ops. |
| SGLang | `/data/code/workspace-rustllm/sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` | `grouped_gemm_nt_f8f8bf16_masked` writes `gateup_output`, then `sglang_per_token_group_quant_8bit(..., fuse_silu_and_mul=True)`, then W2 grouped GEMM. |
| SGLang C++ quant | `/data/code/workspace-rustllm/sglang/sgl-kernel/csrc/gemm/per_token_group_quant_8bit_v2.cu` | The `fuse_silu_and_mul` path fuses activation with group quant, including masked expert layout. |

The next reusable lesson is their problem-size representation. vLLM builds `expert_offsets`, `blockscale_offsets`, `problem_sizes1`, and `problem_sizes2` before CUTLASS grouped GEMM. SGLang's masked path passes `masked_m` and `expected_m` into DeepGEMM. Both make the GEMM scheduler aware of per-expert logical M. PegaInfer currently has `expert_indptr`, but the TileLang grouped launch still uses `dim3 grid(out_tiles, ceil(rows / 32), local_experts)` and returns inside the kernel when `blockIdx.y * 32 >= expert_m`. That is correct and GPU-resident, but it still launches empty CTAs for short or empty experts.

The first active-tile design check found a launch-side constraint: a GPU-generated active tile list cannot by itself shrink the next CUDA launch because grid dimensions are chosen on the host. Using a device-side `active_tile_count` would require a D2H count, CUDA dynamic parallelism, or launching the original capacity grid and returning on `tile >= active_count`. The last option preserves correctness but not the desired launch reduction. A better target is the existing `local_count`: decode route mapping already computes the actual number of local routes on GPU, while runtime still carries `num_expanded = routed.seq_len * topk` (`8 * 6 = 48` for MP8 decode) through expand, activation quant, and grouped GEMM. The hard part is exploiting `local_count` without reintroducing route metadata D2H.

Historical PegaInfer path before the retained fused W2 activation-quant work:

```text
act_quant(expanded_input)
W13 grouped FP4 GEMM -> gate BF16 + up BF16
deepseek_swiglu_clamp_cuda -> activated BF16
TileLang act_quant_k2048(activated) inside W2 wrapper
W2 grouped FP4 GEMM
```

First reproduction target:

```text
act_quant(expanded_input)
W13 grouped FP4 GEMM -> gate BF16 + up BF16
fused SwiGLU clamp + BF16 semantic rounding + TileLang-compatible act_quant_k2048
W2 grouped FP4 GEMM using the produced FP8 activation and E8M0 scales
```

This mirrors vLLM/SGLang's proven operator split while preserving our exact semantic order. It still writes `gate/up` BF16 because vLLM and SGLang also materialize `gate||up` before activation+quant. The later, higher-risk path is to push activation+quant into the W13 accumulator epilogue and avoid writing `gate/up`; that requires separate microbench and fuzz evidence before runtime integration.

The original reproduction microbench compared:

```text
baseline: deepseek_swiglu_clamp_cuda + TileLang act_quant_k2048
candidate: 4-warp fused SwiGLU clamp + BF16 rounding + FP8/E8M0 quant
```

The first fused kernel used one CTA to serially process 32 rows. It was exact but too slow: rows `64` measured baseline `0.007064ms` vs fused `0.016471ms`, speedup `0.429x`. That shape was dropped. The retained microbench shape uses one warp per row and one CTA for four rows per 128-column group.

5090 microbench results:

| Rows | Fuzz | Baseline SwiGLU+act_quant | Fused SwiGLU+quant | Speedup |
| ---: | --- | ---: | ---: | ---: |
| `64` | PASS | `0.005612ms` | `0.002789ms` | `2.013x` |
| `96` | PASS | `0.005574ms` | `0.003139ms` | `1.776x` |
| `160` | PASS | `0.006164ms` | `0.004054ms` | `1.520x` |
| `256` | PASS | `0.007793ms` | `0.004073ms` | `1.914x` |

Runtime integration changes only the scratch hot path:

```text
W13 grouped FP4 GEMM -> gate BF16 + up BF16
deepseek_moe_fp4_grouped_w2_swiglu_with_workspace_cuda
  -> fused SwiGLU+act_quant_k2048-compatible FP8 activation
  -> W2 grouped FP4 GEMM
```

The old Rust scratch helper that performed generic grouped W2 activation quantization was removed so the decode path does not accidentally drift back to the split version. The lower C FFI remains for non-scratch compatibility and older callers.

Runtime validation on 5090:

| Check | Result |
| --- | --- |
| `cargo fmt --check` | passed |
| `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON run 1 | steady TPOT avg `33.416ms`, p50 `32.884ms`, p95 `35.510ms`, first decode avg `31.885ms`, hash `6346f03343d75a65` |
| fixed bench JSON run 2 | steady TPOT avg `31.180ms`, p50 `30.675ms`, p95 `33.151ms`, first decode avg `30.020ms`, hash `6346f03343d75a65` |

Keep decision: retain as the vLLM/SGLang reproduction step. It is exact, removes one kernel launch and one BF16 intermediate write for W2 activation input, and the second fixed run matches the faster W13-only band. It is not sufficient for stable sub-`30ms/token`; the next step needs to explain remaining variance and reduce a larger section than activation+quant alone.

### Retained: shared expert fused decode path

The routed expert path was no longer the only split-MoE region. The shared expert decode scratch path still did:

```text
FP8 W1(input) with act_quant_k4096 -> gate BF16
FP8 W3(input) with act_quant_k4096 -> up BF16
SwiGLU clamp -> activated BF16
FP8 W2(activated) with act_quant_k2048 -> shared output
```

The retained shared path now does:

```text
act_quant_k4096(input) once
dense FP8 W13 -> gate BF16 + up BF16
fused SwiGLU clamp + BF16 rounding + FP8/E8M0 quant
FP8 W2 -> shared output
```

Implementation notes:

- `SharedExpertScratch` owns FP8 activation and scale workspaces so the shared scratch path does not use the C-side growth-cache/mutex path.
- `deepseek_fp8_w1_w3_with_workspace_cuda` reuses one activation quant for shared W1/W3 and calls the dense W13 TileLang kernel for the `4096 -> 2048` shared-expert shape.
- `deepseek_fp8_w2_swiglu_with_workspace_cuda` reuses the same fused SwiGLU+quant semantic order as the routed W2 path before calling the dense `4096 x 2048` FP8 W2 GEMM.
- `deepseek_tilelang_fp8_w13_gemm_n2048_k4096` is generated by transforming the existing dense FP8 TileLang GEMM into a two-output W1/W3 launcher. It is shape-specific to the shared expert, not bs=1 or seq_len=1 specific.

5090 validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| 5090 `cargo fmt --check` | passed |
| 5090 `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON run 1 | steady TPOT avg `29.764ms`, p50 `29.296ms`, p95 `31.766ms`, first decode avg `28.575ms`, hash `6346f03343d75a65` |
| fixed bench JSON run 2 | steady TPOT avg `31.592ms`, p50 `31.082ms`, p95 `33.699ms`, first decode avg `30.019ms`, hash `6346f03343d75a65` |
| additional fixed repeats | `32.220ms`, `30.061ms`, `28.159ms`, all hash `6346f03343d75a65` |
| clean fixed bench after trace removal | `29.944ms`, `29.907ms`, `29.896ms`, all hash `6346f03343d75a65` |
| current HEAD exact E2E after clean bench | `All 20 DeepSeek V4 exact cases passed` |

Short nsys composition evidence, collected with `--output-len 32 --warmup 1 --iters 1 --seed 42` and used only for kernel composition, not TPOT:

| Kernel family | Evidence |
| --- | --- |
| Shared W13 | `deepseek_tilelang_fp8_w13_gemm_n2048_k4096_kernel` appears with `10,151` instances in the short full-process profile. |
| Old shared W1/W3 split GEMM | `deepseek_tilelang_fp8_gemm_n2048_k4096_kernel` drops to `1,118` residual instances, consistent with prefill/non-scratch residue rather than the decode scratch hot path. |
| Shared W2 activation quant | `deepseek_tilelang_act_quant_k2048_kernel` drops to `1,118` residual instances after fused shared/routed W2 quant. |
| Old SwiGLU clamp | `deepseek_swiglu_clamp_kernel` drops to `1,118` residual instances after decode scratch fusion. |

Keep decision: retain. This is the first run to cross sub-`30ms/token`, and the kernel composition proves the intended launches moved. It still does not satisfy the goal because repeated fixed runs returned `29.764ms`, `31.592ms`, `32.220ms`, `30.061ms`, and `28.159ms`; the current blocker is run-to-run variance and remaining synchronization windows, not exactness or missing vLLM/SGLang decomposition.

### Retained: fused MoE mapping clear

The local route mapping wrapper originally launched six tiny clear kernels per layer/rank/token:

```text
pos_to_token = -1
pos_to_token_topk = -1
token_topk_to_pos = -1
expert_indptr = 0
expert_cursor = 0
local_count = 0
```

These clears are semantic initialization, not removable allocation noise, but they do not need six launches. `deepseek_moe_clear_mapping_kernel` now clears all six buffers in one pass before the count, prefix, and mapping kernels. This keeps route metadata GPU-resident and preserves the same `expert_indptr` / `token_topk_to_pos` semantics.

5090 validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON | `29.862ms`, `29.969ms`, `29.874ms`, all hash `6346f03343d75a65` |
| short nsys kernel summary | old `deepseek_moe_clear_i32_kernel` gone; `deepseek_moe_clear_mapping_kernel` appears once per mapping call |

Keep decision: retain. The fixed bench movement is small, but the structural change removes five launches from every MoE mapping without changing math or adding synchronization.

### Retained: small-route MoE mapping

After fusing clears, decode still paid four mapping launches per layer/rank/token:

```text
clear mapping buffers
count local expert rows
prefix local expert row counts
fill compact maps
```

For MP8 decode, `route_elems = global_batch * topk`; with the fixed single-request bench this is `8 * 6 = 48`. The retained fast path uses one block when `route_elems <= 1024 && local_experts <= 256`, doing clear, count, prefix, and map fill with block-level barriers. Larger routed batches and prefill still use the existing multi-kernel path, so this is a small-problem route-mapping path rather than a bs=1-only branch.

5090 validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON run 1 | `27.608ms`, `27.662ms`, `27.826ms`, all hash `6346f03343d75a65` |
| fixed bench JSON run 2 | `27.698ms`, `27.693ms`, `27.644ms`, all hash `6346f03343d75a65` |
| short nsys kernel summary | `deepseek_moe_local_mapping_small_kernel` replaces split clear/count/prefix/mapping kernels for the small decode route shape |

Keep decision: retain. This is a real structural win: the same route semantics move from four launches to one launch in decode-sized routed batches, and repeated fixed benches stay in the `27.6-27.8ms/token` band.

### Rejected: route W13 directly from token activations

After small-route mapping, the next tempting idea was to skip `expand_moe_fused_input_into` for W13:

```text
Before:
expand global_hidden -> expanded_input BF16 rows
act_quant(expanded_input)
expert-major W13 grouped FP4 GEMM

Attempt:
act_quant(global_hidden)
route-row W13 grouped FP4 GEMM -> compact gate/up rows
```

This was exact-safe but not performance-safe. The route W13 kernel launched one TileLang W13 tile set per route element, used `route_indices` and `token_topk_to_pos` to pick the local expert and compact output row, and kept W2/reduce unchanged. That removes one BF16 expand launch and quantizes only token rows instead of route rows, but it also destroys the existing expert-major grouped-GEMM shape. For decode-sized routes, the kernel becomes many small route-row GEMM tiles instead of contiguous expert ranges, so tensor-core work is scheduled less favorably than the retained W13 path.

5090 validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `git diff --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| 5090 release build for `bench_serving` and `deepseek_v4_e2e` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON | aggregate steady TPOT avg `33.217ms`, p50 `33.003ms`, p95 `34.584ms`, decode throughput `30.115 tok/s` |
| fixed bench iterations | `33.162ms`, `33.355ms`, `33.135ms`, all hash `6346f03343d75a65` |

Drop decision: do not retain. Correctness and token trace are not enough here; the implementation regresses from the current small-route mapping band of `27.61-27.83ms/token` to `33.14-33.36ms/token`. The reusable lesson is that reducing route-row materialization can lose if it breaks expert-major GEMM locality. Future work should preserve grouped expert ranges or build a real grouped scheduler that consumes per-expert problem sizes, rather than launching one-row route tiles.

### Rejected: fuse expand with W13 activation quant

The next experiment kept the expert-major W13 grouped GEMM shape and only fused the W13 input preparation:

```text
Before:
expand global_hidden -> expanded_input BF16 rows
TileLang act_quant_k4096(expanded_input) -> FP8 activation + E8M0 scales
expert-major W13 grouped FP4 GEMM

Attempt:
expand+act_quant_k4096(global_hidden, pos_to_token) -> FP8 activation + E8M0 scales
expert-major W13 grouped FP4 GEMM
```

This is the safer version of the previous idea because it preserves expert-major compact rows for W13. A temporary C++ microbench compared the baseline `deepseek_moe_expand_to_fused_cuda + deepseek_tilelang_act_quant_k4096` against a fused CUDA kernel using the same BF16, E8M0, and FP8 E4M3 conversion order as the existing exact-safe fused SwiGLU+quant kernel.

5090 microbench results:

| Tokens | Rows | Fuzz | Baseline expand+act_quant | Fused expand+act_quant | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| `8` | `48` | PASS | `0.008197ms` | `0.002575ms` | `3.183x` |
| `16` | `96` | PASS | `0.008199ms` | `0.002642ms` | `3.103x` |
| `32` | `192` | PASS | `0.010240ms` | `0.004074ms` | `2.514x` |

Runtime validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `git diff --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| 5090 release build for `bench_serving` and `deepseek_v4_e2e` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench run 1 | aggregate steady TPOT avg `27.807ms`; iterations `28.080ms`, `28.143ms`, `27.198ms`; all hash `6346f03343d75a65` |
| fixed bench run 2 | aggregate steady TPOT avg `28.565ms`; iterations `28.509ms`, `28.712ms`, `28.474ms`; all hash `6346f03343d75a65` |

Drop decision: do not retain in runtime. The operator microbench is real, but this is too small relative to the full decode step, and repeated fixed benches do not beat the retained small-route mapping band of `27.61-27.83ms/token`. The reusable lesson is methodological: only integrate this kind of local fusion when a profile proves the fused section is still visible at full-runtime scale, or when it is part of a larger fusion that removes a full synchronization/launch cluster.

### Rejected: skip W2 SwiGLU+quant rows after local_count

The route mapping kernel already computes `expert_indptr[local_experts] = local_count` on GPU. The attempted W2 change passed `expert_indptr + local_experts` into `deepseek_swiglu_clamp_act_quant_k2048_kernel` and skipped rows `>= local_count`:

```text
Before:
fused SwiGLU+quant over rows = route_capacity
W2 grouped FP4 GEMM skips empty rows via expert_indptr

Attempt:
fused SwiGLU+quant reads GPU local_count and only computes compact prefix rows
W2 grouped FP4 GEMM unchanged
```

This preserved the no-D2H rule and kept W2 grouped GEMM semantics unchanged. It was still not a win. The likely reason is that the original row-block quant kernel is very regular and small; adding a device-side count read plus row predicate did not reduce a visible full-runtime section and may have made the kernel shape less friendly.

5090 validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `git diff --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| 5090 release build for `bench_serving` and `deepseek_v4_e2e` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON | aggregate steady TPOT avg `31.270ms`, p50 `30.660ms`, p95 `33.575ms` |
| fixed bench iterations | `31.220ms`, `31.342ms`, `31.248ms`, all hash `6346f03343d75a65` |

Drop decision: do not retain. Skipping empty rows inside a tiny regular kernel is not enough, and in this implementation it regressed sharply. Future `local_count` usage needs to remove a larger launch cluster or preserve a fully regular kernel shape; a branch inside W2 quant is the wrong granularity.

### Rejected: shrink grouped GEMM row-tile launch by seq_len

vLLM's CUTLASS path passes logical per-expert `problem_sizes`, while PegaInfer's TileLang grouped FP4 launch uses a host grid of:

```text
grid.x = output tiles
grid.y = ceil(num_expanded / 32)
grid.z = local_experts
```

For fixed MP8 decode, `num_expanded = global_seq_len * topk = 8 * 6 = 48`, but any one expert can receive at most one route per global token, so `expert_m <= global_seq_len = 8`. The attempted change added a `max_expert_rows` argument and launched grouped W13/W2 with:

```text
grid.y = ceil(max_expert_rows / 32)
```

The runtime passed `plan.routed.seq_len` as the upper bound. This is batch-general for top-k routing with unique experts per token; it is not a bs=1 or seq_len=1 special case.

Standalone W13 microbench on the sparse decode-like shape was exact, but it showed the smaller row-tile bound does not move the important cost when `local_experts=32`:

| Shape | Variant | W13 time |
| --- | --- | ---: |
| `rows=48, experts=32, max_expert_rows=8` | optimized row-tile bound | `0.878124ms` |
| `rows=48, experts=32, max_expert_rows=48` | original row-tile bound | `0.882815ms` |
| `rows=8, experts=1, max_expert_rows=8` | active-expert upper-bound shape | `0.063523ms` |
| `rows=16, experts=2, max_expert_rows=8` | active-expert upper-bound shape | `0.063532ms` |
| `rows=24, experts=3, max_expert_rows=8` | active-expert upper-bound shape | `0.124922ms` |

Runtime validation on 5090:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `git diff --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| 5090 `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON | per-iteration steady TPOT avg `28.504ms`, `28.460ms`, `28.735ms`; all hash `6346f03343d75a65` |

Drop decision: do not retain. The exactness proof is useful, but shrinking only `grid.y` does not address the dominant grouped W13/W2 cost; the `local_experts=32` scheduling dimension still launches many expert slots. The next useful prototype must reduce or repack the expert dimension itself, or use a persistent/problem-size-aware grouped scheduler. A host-known row upper bound alone is not enough.

### Diagnostic: route sparsity at fixed token

A temporary hard-coded diagnostic at `start_pos == 80` synchronized the rank stream and copied `expert_indptr` to host once per `(rank, layer)`. This was intentionally not retained because it performs D2H and stream sync in the decode path. The diagnostic ran with:

```bash
target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash \
  --format json \
  request \
  --prompt-len 1 \
  --output-len 96 \
  --warmup 1 \
  --iters 1 \
  --seed 42
```

Route stats from `/tmp/dsv4_moe_route_stats.log` covered `43 layers * 8 ranks = 344` rows:

| Metric | Value |
| --- | ---: |
| route capacity per rank/layer | `48` |
| `local_count` min / avg / p50 / p95 / max | `0 / 6.0 / 8 / 16 / 40` |
| nonempty local experts min / avg / p50 / p95 / max | `0 / 0.75 / 1 / 2 / 5` |
| max rows per local expert min / avg / p50 / p95 / max | `0 / 4.35 / 8 / 8 / 8` |

Per-rank averages were also sparse: rank-local `local_count` ranged from `4.19` to `7.81` rows on average, and average nonempty experts ranged from `0.53` to `0.98` out of `32` local experts.

Interpretation: the fixed decode route is extremely sparse at the rank-local expert level. Most rank/layer pairs have zero or one active local expert, and nonempty experts usually have exactly eight rows because the gathered MP8 token batch follows the same token trace across ranks. This confirms that empty expert/empty CTA work is real. It also explains the failed valid-row experiment: skipping rows inside W2 quant is too small and too late. A useful next MoE scheduler must reduce or reshape expert-level grouped GEMM work while preserving expert-major locality; route-row W13 and in-kernel row predicates are the wrong granularity.

The existing W13 grouped FP4 microbench gives an upper-bound sanity check for this direction:

| Shape | W13 one GEMM | Notes |
| --- | ---: | --- |
| `rows=48, experts=32` | `0.399417ms` | Capacity-like shape with many expert slots. The bench distribution is not as sparse as the real route, so treat as a pessimistic capacity proxy. |
| `rows=8, experts=1` | `0.061476ms` | Typical one-active-expert rank/layer shape from the route diagnostic. |
| `rows=16, experts=2` | `0.061866ms` | Approx p95 active-expert count. |
| `rows=24, experts=3` | `0.061477ms` | Upper tail seen in several layers. |

Interpretation: the next plausible MoE win is not another scalar/row predicate inside the existing capacity launch. The useful prototype should make grouped W13/W2 see active expert problem sizes, ideally using compact active pointer/indptr metadata, while keeping W13 expert-major. The hard production constraint remains host launch sizing: a GPU-only active list cannot directly shrink grid dimensions without D2H, CUDA dynamic parallelism, or a fixed small upper-bound launch. The microbench says the direction is worth prototyping; it does not yet solve the runtime launch-sizing problem.

### Microbench: compact active expert pointer arrays

The W13 grouped FP4 bench now has an active-expert mode:

```bash
/tmp/w13_grouped_fp4_bench \
  --experts 32 \
  --active-experts 3 \
  --rows-per-active 8 \
  --warmup 20 \
  --iters 300 \
  --seed 44
```

It builds two equivalent W13 problems over the same rows and the same first-N expert weights:

```text
capacity: local_experts = 32, expert_indptr has N nonempty experts and the rest empty
compact:  local_experts = N, compact pointer arrays and compact expert_indptr
```

W13 outputs are bitwise compared against the existing two-GEMM baseline. In active mode, the tool also builds an equivalent W2 grouped GEMM problem and bitwise compares capacity W2 against compact W2. This directly tests whether merely shrinking the expert dimension of the launch is enough for either routed grouped GEMM.

The tool also supports arbitrary sparse counts, so active experts do not have to be a prefix:

```bash
/tmp/w13_grouped_fp4_bench \
  --counts 0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0 \
  --capacity-launch-rows 8 \
  --compact-launch-rows 8 \
  --iters 500
```

5090 results:

| Active experts | Rows per active | Capacity W13 | Compact W13 | W13 compact speedup | Capacity W2 | Compact W2 | W2 compact speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `8` | `0.061482ms` | `0.061536ms` | `0.999x` | `0.032759ms` | `0.032764ms` | `1.000x` |
| `3` | `8` | `0.063314ms` | `0.062743ms` | `1.009x` | `0.032761ms` | `0.032767ms` | `1.000x` |
| `6` | `8` | `0.122856ms` | `0.122886ms` | `1.000x` | `0.063469ms` | `0.063469ms` | `1.000x` |

Interpretation: the current TileLang grouped W13/W2 early-return path already makes empty expert slots nearly free for this shape. The performance jumps are tied to how many active expert tiles fit into waves, not to the existence of 32 pointer slots by itself. A useful next prototype must change the actual active GEMM scheduling or fuse across the W13/W2 boundary; compacting pointer arrays alone should not be moved into runtime.

Follow-up probe: the same bench now supports `--compact-launch-rows`, which keeps the compact active expert pointer arrays but also uses a host-known per-expert row upper bound as the grouped GEMM launch `m`. This tests the vLLM/SGLang `problem_sizes` idea more directly:

```bash
/tmp/w13_grouped_fp4_bench \
  --experts 32 \
  --active-experts 8 \
  --rows-per-active 8 \
  --compact-launch-rows 8 \
  --shared-bytes 32768 \
  --warmup 20 \
  --iters 300
```

5090 result log: `/tmp/dsv4_compact_launch_rows_bench.log`.

| Active experts | Rows per active | Compact launch rows | Compact W13 | W13 speedup vs capacity | Compact W2 | W2 speedup vs capacity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8` | `8` | `64` | `0.108671ms` | `0.999x` | `0.057343ms` | `1.000x` |
| `8` | `8` | `8` | `0.081902ms` | `1.325x` | `0.045061ms` | `1.273x` |
| `8` | `16` | `128` | `0.108818ms` | `0.999x` | `0.056283ms` | `1.001x` |
| `8` | `16` | `16` | `0.083988ms` | `1.293x` | `0.045048ms` | `1.248x` |
| `8` | `32` | `256` | `0.112376ms` | `1.002x` | `0.055638ms` | `0.999x` |
| `8` | `32` | `32` | `0.087171ms` | `1.293x` | `0.043066ms` | `1.292x` |

The route-stat realistic probe uses active `1/2/5`, because the fixed decode trace had nonempty local experts avg `<1`, p95 `2`, and max `5`.

5090 result log: `/tmp/dsv4_compact_launch_rows_realistic_bench.log`.

| Active experts | Rows per active | Compact launch rows | Capacity W13 | Compact W13 | W13 speedup | Capacity W2 | Compact W2 | W2 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `8` | `8` | `0.061480ms` | `0.061477ms` | `1.000x` | `0.032767ms` | `0.032770ms` | `1.000x` |
| `2` | `8` | `8` | `0.061642ms` | `0.061556ms` | `1.001x` | `0.032765ms` | `0.032767ms` | `1.000x` |
| `5` | `8` | `8` | `0.079858ms` | `0.063514ms` | `1.257x` | `0.045044ms` | `0.032787ms` | `1.374x` |

Interpretation: compacting active experts is only useful when the launch also receives the per-expert row bound, and even then the gain mainly appears in the route tail with several active local experts. For the common active `1-2` cases, the current capacity launch already uses one row tile and empty expert slots are nearly free. This makes vLLM/SGLang-style problem sizes a real but secondary lever for our current decode trace, not the main path to sub-25 by itself.

Arbitrary-count follow-up: prefix-active experts had hidden a real cost. When active experts are sparse arbitrary ids, compacting the expert dimension and using a per-expert row bound becomes a clear microbench win:

5090 logs: `/tmp/dsv4_arbitrary_counts_grouped_fp4_bench.log` and `/tmp/dsv4_arbitrary_counts_grouped_fp4_bound_bench.log`.

| Shape | Counts summary | Capacity launch rows | Capacity W13 | Compact W13 | W13 speedup | Capacity W2 | Compact W2 | W2 speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arbitrary one | expert `17:8` | `8` | `0.134560ms` | `0.096356ms` | `1.396x` | `0.068437ms` | `0.049164ms` | `1.392x` |
| arbitrary two | experts `3:8,29:8` | `8` | `0.154884ms` | `0.096365ms` | `1.607x` | `0.076596ms` | `0.049193ms` | `1.557x` |
| arbitrary tail five | experts `1:8,7:8,13:8,22:8,31:8` | `8` | `0.177775ms` | `0.098281ms` | `1.809x` | `0.089979ms` | `0.049184ms` | `1.829x` |
| uneven tail five | experts `0:3,6:11,14:5,19:17,30:8` | `17` | `0.177537ms` | `0.102413ms` | `1.734x` | `0.089849ms` | `0.049332ms` | `1.821x` |

Important correction: `--capacity-launch-rows` by itself did not materially improve the full 32-expert launch; the capacity rows above are almost the same as the no-bound run. The useful microbench lever is shrinking `local_experts`/`blockIdx.z` to the active expert list. That is exactly the part the runtime cannot currently do without making GPU route metadata visible to launch setup.

The production blocker is launch ownership: current route metadata is GPU-resident, while CUDA grid dimensions are host-chosen. Runtime adoption would need one of these, in increasing complexity:

1. A host-known conservative row bound that is still small enough to matter, plus a useful host-known active-expert upper bound.
2. A GPU active problem list consumed by a fixed-capacity persistent scheduler kernel, not the current blockIdx.z expert dimension.
3. A D2H active-count path, which is currently rejected for decode hot path unless a future scheduler explicitly budgets it.

Do not move the compact pointer-array microbench into runtime as-is. It proves the target shape, not the scheduling mechanism.

### Retained: grouped FP4 dynamic shared-memory downsize

The active-expert bench changed the next question from pointer compaction to CTA occupancy. The generated TileLang grouped FP4 wrappers used a conservative dynamic shared-memory request:

```text
kSharedBytes = 98304
```

The generated kernels do not need that much dynamic shared memory for the routed W13/W2 per-CTA tiles. Requesting `98304` bytes suppresses CTA residency on sparse decode shapes where the useful work is many tiny expert GEMMs. The W13 bench now has a raw-launch `--shared-bytes` probe that calls the generated kernel directly and still compares bitwise output against the existing wrapper baseline.

5090 shared-memory probe for decode-like sparse rows:

| Shared bytes | Fuzz | W13 rows=8 active=8 | W2 rows=8 active=8 | Decision |
| ---: | --- | ---: | ---: | --- |
| `98304` | PASS | `0.122941ms` | `0.063471ms` | old wrapper baseline |
| `81920` | PASS | `0.122941ms` | `0.063472ms` | no improvement |
| `65536` | PASS | `0.122943ms` | `0.063474ms` | no improvement |
| `49152` | PASS | `0.080711ms` | `0.045077ms` | faster |
| `32768` | PASS | `0.080758ms` | `0.045085ms` | retained safety point |
| `24576` | PASS | `0.081892ms` | `0.045087ms` | slightly slower |
| `16384` | FAIL | illegal memory access | n/a | too small |

Additional `32768` probes stayed bitwise:

| Shape | W13 old/shared probe comparison | W2 old/shared probe comparison |
| --- | ---: | ---: |
| active `16`, rows/active `1` | `0.307263ms -> 0.281701ms` | `0.122875ms -> 0.081957ms` |
| active `32`, rows/active `1` | `0.544289ms -> 0.499836ms` | `0.252737ms -> 0.231244ms` |
| active `8`, rows/active `8` | `0.122960ms -> 0.108667ms` | `0.063493ms -> 0.057330ms` |
| active `16`, rows/active `4` | `0.315041ms -> 0.283986ms` | `0.122991ms -> 0.082393ms` |

Runtime change: grouped FP4 W13 and W2 wrappers generated by `pegainfer-kernels/tools/tilelang/deepseek_v4/generate.py` now request `32768` dynamic shared bytes. Dense FP4/FP8 wrappers keep their existing requests.

5090 validation:

| Check | Result |
| --- | --- |
| release build for `bench_serving` and `deepseek_v4_e2e` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench run 1 | aggregate steady TPOT avg `28.481ms`; per-iteration `28.464ms`, `28.467ms`, `28.512ms`; all hash `6346f03343d75a65` |
| fixed bench repeat | aggregate steady TPOT avg `28.694ms`; per-iteration `28.713ms`, `28.684ms`, `28.685ms`; all hash `6346f03343d75a65` |

Interpretation: this is a real grouped-GEMM microbench improvement and an exact full-runtime cleanup, but it is not the missing sub-25 lever by itself. The runtime gain is muted because routed W13/W2 are only one part of the decode step and the remaining wait-inclusive collectives plus attention local work still dominate. Keep it because it removes an over-conservative launch resource request without changing math or routing.

### Rejected: grouped FP4 launch bounds 2

After dynamic shared memory was lowered to `32768`, the next occupancy hypothesis was to force the grouped FP4 generated kernels from:

```text
__launch_bounds__(128, 1)
```

to:

```text
__launch_bounds__(128, 2)
```

This compiled and stayed exact, but it did not produce a stable runtime improvement.

5090 validation:

| Check | Result |
| --- | --- |
| grouped W13/W2 microbench | bitwise PASS; timings were effectively the same as the `32768` shared-memory retained path |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench | aggregate steady TPOT avg `28.522ms`; per-iteration `28.508ms`, `28.510ms`, `28.548ms`; all hash `6346f03343d75a65` |

Drop decision: do not retain. The compiler already accepts the lower dynamic shared-memory request, and forcing a higher launch-bound does not buy enough to justify a new generator transform. The retained occupancy fix is the shared-memory request, not the launch-bound constraint.

### Rejected: naive grouped FP4 `block_M=16`

The route sparsity trace showed most decode rank/layer local experts have at most `8` rows. That suggested a smaller grouped FP4 M tile could help sparse decode without changing the K accumulation order. The naive experiment changed TileLang FP4 GEMM's internal:

```text
block_M = 32
```

to:

```text
block_M = 16
```

This is not enough. The grouped transform and launch wrappers still contain `32`-row assumptions for early return and `grid.y`, so the experiment only works for small per-expert row counts and breaks larger batch/expert-general cases.

5090 microbench:

| Shape | Result | Timing |
| --- | --- | --- |
| active `8`, rows/active `1` | PASS | W13 `0.080758ms -> 0.066303ms`; W2 `0.045085ms -> 0.032865ms` versus retained `32768` shared-memory path |
| active `8`, rows/active `8` | PASS | W13 `0.108667ms -> 0.092199ms`; W2 `0.057330ms -> 0.043541ms` |
| active `8`, rows/active `16` | PASS | W13 `0.096129ms`; W2 `0.043824ms` |
| active `8`, rows/active `32` | FAIL | gate/up/W2 outputs left at sentinel values for later rows |

Drop decision: do not retain. This would be a decode-sparse specialization by accident and violates the batch-general requirement. The useful lesson is narrower: a smaller M tile has real sparse-expert potential, but production work must propagate `block_M` through `grouped_fp4_gemm_kernel_source`, `grouped_fp4_w13_gemm_kernel_source`, wrapper `grid.y`, and all hard-coded early-return/index replacements, then fuzz broad row distributions before runtime integration.

### Rejected: parameterized grouped FP4 `block_M=16`

The follow-up implemented that propagation correctly as an experiment:

- `fp4_gemm_kernel(..., block_m=16)` for grouped FP4 only.
- grouped W13/W2 transforms changed the expert early-return guard to `blockIdx.y * 16`.
- grouped W13/W2 wrappers changed `grid.y` to `(m + 15) / 16`.
- dense FP4 wrappers remained at the original `block_M=32`.
- the raw-launch path in `w13_grouped_fp4_bench.cu` used the same grouped block size while testing `--shared-bytes`.

Generated CUDA on 5090 showed the intended shape: grouped W13/W2 had `C_local[16]`, `blockIdx.y * 16`, and wrapper `kBlockM = 16`. After rejection, local and 5090 were restored to `C_local[32]`, `blockIdx.y * 32`, and wrapper `(m + 31) / 32`.

5090 fuzz results for the parameterized version:

| Shape | Result | W13 one GEMM | W2 GEMM |
| --- | --- | ---: | ---: |
| active `1`, rows/active `8`, experts `32` | PASS | `0.051488ms` | `0.026671ms` |
| active `8`, rows/active `8`, experts `32` | PASS | `0.092804ms` | `0.043786ms` |
| active `16`, rows/active `8`, experts `32` | PASS | `0.276781ms` | `0.074541ms` |
| active `8`, rows/active `16`, experts `32` | PASS | `0.096531ms` | `0.045072ms` |
| active `8`, rows/active `32`, experts `32` | PASS | `0.188902ms` | `0.080208ms` |
| rows `64`, experts `8` | PASS | `0.090272ms` | n/a |
| rows `128`, experts `16` | PASS | `0.169116ms` | n/a |
| rows `256`, experts `32` | PASS | `0.389752ms` | n/a |

Full-runtime validation:

| Check | Result |
| --- | --- |
| release `cargo check -p pegainfer-deepseek-v4 --features deepseek-v4` | passed locally and on 5090 |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench run 1 | aggregate steady TPOT avg `28.971ms`; per-iteration `28.727ms`, `28.963ms`, `29.224ms`; all hash `6346f03343d75a65` |
| fixed bench repeat | aggregate steady TPOT avg `29.797ms`; per-iteration `29.913ms`, `29.764ms`, `29.713ms`; all hash `6346f03343d75a65` |
| restored retained path fixed bench | aggregate steady TPOT avg `28.736ms`; per-iteration `28.445ms`, `28.998ms`, `28.763ms`; all hash `6346f03343d75a65` |

Drop decision: do not retain. The smaller M tile is now proven batch/expert-general in fuzz, so the earlier correctness objection is resolved, but the runtime objection remains: the extra tile count worsens full decode scheduling enough that the microbench win does not survive. The restored retained runtime keeps grouped FP4 `block_M=32` plus `32768` dynamic shared bytes.

### Microbench: W13 accumulator to SwiGLU quant upper bound

The aggressive path from the goal is:

```text
W13 GEMM accumulator -> SwiGLU -> W2 quant
```

The standalone `swiglu_quant_bench.cu` now includes an upper-bound mode for that epilogue idea. It compares:

```text
materialized:
  FP32 accumulator proxy -> BF16 gate/up materialization
  fused SwiGLU+quant reads BF16 gate/up

direct:
  FP32 accumulator proxy -> BF16 semantic rounding in registers
  fused SwiGLU+quant writes only W2 FP8 activation/scales
```

The direct path is bitwise compared against the materialized path. This does not implement the real W13 tensor-core epilogue, but it answers the first gating question: how much can the gate/up materialization boundary contribute by itself?

5090 results:

| Rows | Materialized accumulator + quant | Direct accumulator quant | Speedup | Absolute delta |
| ---: | ---: | ---: | ---: | ---: |
| `48` | `0.004577ms` | `0.002322ms` | `1.971x` | `0.002255ms` |
| `96` | `0.004669ms` | `0.004080ms` | `1.144x` | `0.000589ms` |
| `192` | `0.006144ms` | `0.004096ms` | `1.500x` | `0.002048ms` |

Interpretation: the direct epilogue idea is exact-feasible at the scalar/quant semantics level, but the isolated absolute savings are too small to justify runtime integration by itself. A real implementation would need to fuse into the TileLang W13 tensor-core epilogue and avoid the existing W13 gate/up global writes without worsening W13 scheduling. Treat this as a prerequisite proof, not a runtime green light.

### Rejected: standalone TileLang SwiGLU quant

Before trying another W13 epilogue fusion, we isolated the middle operator:

```text
gate BF16 + up BF16 -> SwiGLU clamp -> BF16 semantic round -> FP8 activation + E8M0 scales
```

The source idea is still the vLLM/SGLang decomposition where activation and quantization sit between W13 and W2, but this experiment asks a narrower implementation question: can TileLang replace the current C++ fused SwiGLU+quant kernel with byte-identical output and better scheduling?

Implementation notes:

- Added a temporary generated TileLang `deepseek_tilelang_swiglu_quant_k2048` launcher.
- The wrapper signature initially mismatched the generated ABI; fixing argument order made local and 5090 release generation pass.
- The standalone benchmark compared the temporary TileLang kernel against the current C++ fused kernel, not the removed old split `deepseek_swiglu_clamp_cuda` path.
- After the microbench, the temporary TileLang launcher was removed from the generator so there is no extra runtime-facing path.

5090 result log: `/tmp/dsv4_tilelang_swiglu_quant_bench.log`.

| Rows | Fuzz | Current C++ fused | TileLang candidate | Candidate / C++ |
| ---: | --- | ---: | ---: | ---: |
| `48` | PASS | `0.002383ms` | `0.006148ms` | `0.388x` |
| `96` | PASS | `0.002676ms` | `0.004101ms` | `0.653x` |
| `192` | PASS | `0.004091ms` | `0.004099ms` | `0.998x` |

Drop decision: do not retain. Byte exactness is good, but the standalone TileLang form is slower on the small routed decode shapes that matter and only ties at larger row counts. This reinforces the current direction: standalone activation/quant swaps are not enough; the next meaningful attempt must remove the W13 gate/up materialization boundary inside the W13 tensor-core epilogue or improve grouped GEMM scheduling itself.

### Rejected: TileLang W13 accumulator to SwiGLU quant prototype

The next prototype tried to make the aggressive path concrete in TileLang without touching runtime:

```text
grouped W13 FP4 GEMM accumulator
  -> BF16 semantic round for gate/up
  -> SwiGLU clamp
  -> BF16 semantic round for activated
  -> FP8/E8M0 W2 activation quant
```

The source idea is the same "experts can own activation/quant/finalize boundaries" direction from vLLM modular MoE and SGLang FlashInfer CuteDSL MoE:

- `/data/code/workspace-rustllm/vllm/docs/design/fused_moe_modular_kernel.md`
- `/data/code/workspace-rustllm/sglang/python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py`

Implementation notes:

- Added a temporary generated TileLang kernel that used one CTA for the same `128` output-column block of W1 and W3.
- The first generator form failed TileLang lowering because pipeline planning disallowed two writes to the same `B_fp4_shared` buffer and the hand-written per-row amax reduction had a data-race warning.
- Splitting W1/W3 shared buffers and using `T.reduce_absmax` made local and 5090 release generation pass.
- A C++ microbench compared current `grouped_w13_gemm -> C++ fused SwiGLU+FP8 quant` against the temporary generated `grouped_w13_swiglu_quant`.

5090 result:

| Check | Result |
| --- | --- |
| release `cargo check -p pegainfer-deepseek-v4 --features deepseek-v4` | passed locally and on 5090 after generator fixes |
| microbench first fuzz shape | active `1`, rows/active `8`, experts `32` |
| fuzz result | FAIL: FP8 activation and E8M0 scale bytes differed from the current baseline |
| log | `/tmp/dsv4_w13_swiglu_quant_bench.log` |

Drop decision: do not retain. The generator can express the rough shape, but it did not reproduce the current C++ SwiGLU+E8M0/FP8 quant bytes. That would change the fixed long token trace, so this is not a runtime candidate. The temporary generated launcher and C++ bench source were removed locally and on 5090; the retained generator surface is back to grouped FP4 `block_M=32` with `32768` dynamic shared bytes.

Evidence required for each adoption step:

- vLLM/SGLang source location and whether we copied the decomposition, the kernel shape, or only the validation idea.
- standalone microbench with fuzz against the current PegaInfer baseline.
- exact E2E `20/20`.
- fixed JSON bench with token hash `6346f03343d75a65`.
- repeated TPOT range, not a single fast run.

### Score-route BF16 direct GEMM

The MoE stage trace put router at roughly `2.5ms/token`. For score-route layers, the previous hot path converted both decode hidden states and the static score-gate weight matrix from BF16 to F32 on every call, then used F32 cuBLAS GEMV/GEMM. That is a poor steady-state shape because the score-gate weights are static, and the hidden activation is only an input to a small routing projection.

The retained cleanup changes `deepseek_score_gate_cuda` to call `cublasGemmEx` directly with BF16 `x`, BF16 `gate_weight`, F32 output, and `CUBLAS_COMPUTE_32F_PEDANTIC`. It also removes the per-device scratch fields for converted `x_f32` and `gate_f32`. This is not a bs=1 specialization: the call still uses `(n_experts x seq_len) = gate_weight^T * x` and keeps the batch-general `seq_len` dimension.

Validation:

| Check | Result |
| --- | --- |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench final run | aggregate steady TPOT avg `28.979ms`; per-iteration `28.843ms`, `28.832ms`, `29.263ms`; all hash `6346f03343d75a65` |
| fixed bench repeat | aggregate steady TPOT avg `28.583ms`; per-iteration `28.637ms`, `28.536ms`, `28.576ms`; all hash `6346f03343d75a65` |

Earlier exploratory runs of the same BF16 direct shape landed at `26.194ms`, `30.220ms`, `27.533ms`, and `26.902ms`. The retained conclusion is therefore conservative: keep the cleanup because it removes obvious repeated conversion work and preserves exactness, but do not count it as a major sub-25 lever. Router is visible in trace, yet the full decode TPOT is still dominated by attention local, grouped expert GEMMs, and wait-inclusive collectives.

Rejected variant: caching score-gate weights as F32 preserved exact E2E and token hash, but the fixed bench regressed to aggregate steady TPOT avg `29.148ms` with per-iteration `29.152ms`, `29.139ms`, and `29.155ms`. The extra F32 memory footprint and F32 math path were not worth keeping.

Rejected variant: direct CUDA BF16 router projection. SGLang has a `fused_moe_router_cudacore` route in `/data/code/workspace-rustllm/sglang/python/sglang/srt/layers/moe/router.py`, and TileKernels has warp-level top-k/scoring kernels under `/data/code/workspace-rustllm/TileKernels/tile_kernels/moe/`. We tested the analogous PegaInfer idea with a temporary standalone bench: keep the existing select/normalization semantics, but replace the cuBLAS BF16 projection with a direct CUDA dot-product kernel over `(seq_len, n_experts, hidden_dim)`. The bench source was deleted after rejection so it cannot be accidentally wired into runtime.

5090 microbench:

| `seq_len` | cuBLAS projection + select | direct CUDA projection + select | Speedup | Fuzz |
| ---: | ---: | ---: | ---: | --- |
| `1` | `0.016388ms` | `0.018434ms` | `0.889x` | top-k indices identical |
| `8` | `0.036863ms` | `0.022734ms` | `1.621x` | top-k indices identical |
| `16` | `0.036862ms` | `0.030717ms` | `1.200x` | top-k indices identical |
| `32` | `0.378665ms` | `0.043034ms` | `8.799x` | top-k indices identical |
| `64` | `0.413153ms` | `0.069628ms` | `5.934x` | top-k indices identical |
| `128` | `0.412896ms` | `0.120920ms` | `3.415x` | top-k indices identical |
| `256` | `0.417000ms` | `0.223227ms` | `1.868x` | top-k indices identical |

Runtime experiment:

| Check | Result |
| --- | --- |
| release build | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench | aggregate steady TPOT avg `28.340ms`; per-iteration `28.611ms`, `28.586ms`, `27.823ms` |
| token hash | changed from `6346f03343d75a65` to `abc4a0a2160d7963` |
| post-revert fixed bench | aggregate steady TPOT avg `28.477ms`; per-iteration `28.501ms`, `28.477ms`, `28.455ms`; hash restored to `6346f03343d75a65` |

Drop decision: do not retain. Even though exact E2E passed and the fixed bench got slightly faster, the long fixed token trace changed. The direct kernel changes the F32 accumulation order relative to cuBLAS, and the route boundary is sensitive enough to alter later tokens. A future router replacement must either reproduce cuBLAS accumulation/tie behavior strongly enough to keep the fixed hash, or be introduced together with an explicit accuracy-baseline update rather than as a transparent performance cleanup.

### Rejected: W2 epilogue TopK weight/reduce via atomics

vLLM's modular MoE design explicitly allows `TopKWeightAndReduce` to live inside `FusedMoEExpertsModular::apply()` rather than in the prepare/finalize layer. Relevant source/doc anchors:

- `/data/code/workspace-rustllm/vllm/docs/design/fused_moe_modular_kernel.md` — `TopKWeightAndReduce` and `finalize_weight_and_reduce_impl`.
- `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py` — abstract interfaces for prepare/finalize, experts, and top-k weight/reduce.

The idea maps to our current routed W2 boundary as:

```text
current:
  W2 grouped GEMM -> expanded BF16 route rows
  reduce_fused_f32(token -> topk -> pos) -> F32 per-token partial

candidate:
  W2 grouped GEMM epilogue -> route weight -> direct F32 per-token partial
```

The issue is layout. Our W2 grouped GEMM is expert-major and route-row-major, while the current reduce kernel is token-major and deterministic over `topk`. A direct W2 epilogue combine would need either atomics into per-token output or a different output layout/scheduler. A standalone microbench tested the atomic version before touching runtime:

```bash
/tmp/w2_weighted_reduce_bench 1 6 4096 1000 42
/tmp/w2_weighted_reduce_bench 8 6 4096 1000 42
/tmp/w2_weighted_reduce_bench 16 6 4096 1000 42
/tmp/w2_weighted_reduce_bench 32 6 4096 1000 42
```

5090 result log: `/tmp/dsv4_w2_weighted_reduce_bench.log`.

| `seq_len` | Current deterministic reduce | Atomic epilogue-shaped reduce | Atomic / current | Max abs diff |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `0.002633ms` | `0.004956ms` | `0.531x` | `1.19e-7` |
| `8` | `0.002585ms` | `0.004877ms` | `0.530x` | `2.38e-7` |
| `16` | `0.002305ms` | `0.006007ms` | `0.384x` | `2.38e-7` |
| `32` | `0.002743ms` | `0.006146ms` | `0.446x` | `2.38e-7` |

Drop decision: do not integrate. The numerical difference is small, but the atomic shape is slower even before considering interaction with the actual W2 GEMM epilogue. The DSV4 `topk=6` refresh confirms the older `topk=8` conclusion. To use vLLM's `TopKWeightAndReduce inside experts` principle here, we would first need a token-major or grouped-by-token W2 scheduling strategy that preserves deterministic reduction order without atomics. That is a scheduler/layout project, not a small epilogue patch.

### Rejected: pair-interleaved W13 output layout for standalone SwiGLU quant

vLLM and FlashInfer/TRTLLM paths often reshape W13 to match the fused SwiGLU convention. Relevant source anchors:

- `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` — TRTLLM MXFP4 path swaps/interleaves W1/W3 to match its SwiGLU convention.
- `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` — FlashInfer utility comments note the expected `[up; gate]` convention.
- `/data/code/workspace-rustllm/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` — CuteDSL runner has an explicit W13 interleave helper.

Our current retained routed local expert layout is simpler:

```text
W13 grouped GEMM -> gate BF16 buffer + up BF16 buffer
fused SwiGLU + FP8 quant reads the two buffers
W2 grouped GEMM
```

The candidate was to write W13 as one pair-interleaved BF16 buffer, with adjacent `[up, gate]` values, then make the fused SwiGLU+quant kernel read adjacent BF16 pairs. This could become useful only if it materially improved the standalone SwiGLU+quant read pattern and later justified a W13 output-layout change.

Standalone microbench:

```bash
/tmp/swiglu_layout_bench 48 5000 45 7.0
/tmp/swiglu_layout_bench 64 5000 42 7.0
/tmp/swiglu_layout_bench 96 5000 46 7.0
/tmp/swiglu_layout_bench 128 5000 43 7.0
/tmp/swiglu_layout_bench 192 3000 47 7.0
/tmp/swiglu_layout_bench 256 3000 44 7.0
```

5090 results:

| Rows | Separate gate/up | Pair-interleaved `[up, gate]` | Pair / separate | Correctness |
| ---: | ---: | ---: | ---: | --- |
| `48` | `0.002077ms` | `0.002049ms` | `1.014x` | byte-identical output and scales |
| `64` | `0.002341ms` | `0.002199ms` | `1.065x` | byte-identical output and scales |
| `96` | `0.003286ms` | `0.002579ms` | `1.274x` | byte-identical output and scales |
| `128` | `0.004094ms` | `0.004092ms` | `1.001x` | byte-identical output and scales |
| `192` | `0.004095ms` | `0.004095ms` | `1.000x` | byte-identical output and scales |
| `256` | `0.004095ms` | `0.004094ms` | `1.000x` | byte-identical output and scales |

Drop decision: do not integrate. The best case saves about `0.0007ms` for one standalone SwiGLU+quant call, and most row counts are flat. Runtime integration would require changing the W13 grouped GEMM output layout and the Rust scratch/API surface, which could easily erase the tiny read-side gain. This remains a useful layout note for a future true W13 epilogue fusion, but not a standalone patch.

### Temporary MoE stage trace

A temporary hard-coded trace at `start_pos == 80` split `decode_moe_ag_rs_bf16_hidden_with_scratch` into:

```text
hidden/token all-gather
shared expert
wait after gather
router
local route mapping
expand compact input
local expert W13 + SwiGLU-quant + W2
f32 partial combine
reduce-scatter
wait after reduce-scatter
final routed + shared add
```

The first CUDA-event version failed on 5090 with `CUDA_ERROR_INVALID_HANDLE` when timed events were mixed across the compute stream and MoE NCCL stream. The working trace used host-side synchronization after each stage. That intentionally perturbs overlap and should not be compared as TPOT; it is only a stage-ranking diagnostic. The fixed bench still preserved token hash `6346f03343d75a65`.

Trace log: `/tmp/dsv4_moe_stage_trace_bench.log`.

The log contains three `start_pos == 80` occurrences (`warmup=2`, `iters=1`), each with `43 layers * 8 ranks`. Aggregating by taking the slowest rank per layer gives the following approximate per-token sums:

| Stage | Run 0 | Run 1 | Run 2 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| all-gather, overlapped with shared | `8.38ms` | `7.95ms` | `7.83ms` | Includes synchronization effects; do not read as pure transfer. |
| shared expert | `3.37ms` | `3.39ms` | `3.40ms` | Mostly hidden under all-gather in the retained path. |
| local expert W13/SwiGLU/W2 | `6.19ms` | `6.17ms` | `6.15ms` | The largest non-collective MoE block. |
| reduce-scatter | `7.62ms` | `7.57ms` | `7.64ms` | Includes rank-arrival/wait behavior; not pure NCCL bandwidth. |
| router | `2.53ms` | `2.47ms` | `2.56ms` | Score-route layers dominate; hash-route-only host precheck would cover only `3/43` layers. |
| mapping | `0.84ms` | `0.78ms` | `0.87ms` | Small; not the next primary target. |
| expand compact input | `0.77ms` | `0.80ms` | `0.81ms` | Small after earlier fused-route work. |
| partial combine | `0.62ms` | `0.61ms` | `0.59ms` | Small. |
| final add | `0.76ms` | `0.78ms` | `0.83ms` | Small. |

This changes the next-step priority. More scalar cleanup around mapping/expand/add is unlikely to move TPOT. The two plausible MoE levers are:

1. Local expert compute: change the actual W13/W2 grouped scheduler or fuse deeper into the W13 epilogue, not just compact pointer arrays.
2. Collective arrival: reduce the work before reduce-scatter or change overlap structure, while treating NCCL timings as wall/wait windows rather than transfer cost.

## Rejected Patterns

These are worth remembering because they looked plausible:

| Attempt | Result | Lesson |
| --- | --- | --- |
| Route W13 directly from token activations | Exact-safe, hash-stable, but regressed to `33.14-33.36ms/token` | Removing BF16 expand and reducing W13 quant rows is not enough if the GEMM shape loses expert-major locality. |
| Fuse expand with W13 activation quant | Bitwise microbench PASS and `2.5-3.2x` faster locally, but runtime repeated at `27.20-28.71ms/token` | Local microbench wins that remove only a tiny section can disappear in full decode; require full-runtime proof before retaining. |
| Skip W2 SwiGLU+quant rows after GPU `local_count` | Exact-safe and hash-stable, but regressed to `31.22-31.34ms/token` | Adding a device-side count read and row predicate inside a tiny regular kernel is the wrong granularity. |
| Shrink grouped GEMM row-tile bound to `seq_len` | Exact-safe and hash-stable, but regressed to `28.46-28.74ms/token` | Empty row tiles are not the dominant grouped FP4 cost; the expert scheduling dimension remains too coarse. |
| Compact active expert pointer arrays | Prefix-active microbench looked flat, but arbitrary sparse expert ids show compact active z-dimension plus row bound gives W13/W2 `1.39-1.83x` standalone speedup | The useful lever is shrinking `local_experts`/`blockIdx.z` to an active expert list, not merely changing row predicates. Runtime still lacks a no-D2H way to use GPU route metadata as CUDA launch dimensions. |
| Capacity launch row bound without active expert compaction | Bitwise microbench PASS, but `--capacity-launch-rows` stayed almost unchanged for arbitrary sparse counts | Host-known per-expert row bound alone is not enough; the 32-expert z dimension remains the expensive scheduling shape. |
| Direct W13-accumulator SwiGLU quant epilogue | Bitwise microbench PASS and local epilogue speedup up to `1.97x`, but absolute 5090 delta only `0.0006-0.0023ms` | Scalar epilogue feasibility is proven; runtime work needs true W13 tensor-core epilogue fusion, not a standalone quant kernel swap. |
| W2 epilogue TopK weight/reduce via atomics | DSV4 `topk=6` microbench showed atomic direct reduce at `0.0049-0.0061ms` versus current deterministic reduce at `0.0023-0.0027ms` | vLLM's `TopKWeightAndReduce inside experts` principle is valid, but our expert-major W2 layout would need atomics or a scheduler rewrite; do not bolt it onto the current TileLang grouped W2 epilogue. |
| Pair-interleaved W13 output layout for standalone SwiGLU quant | Byte-identical microbench, but mostly flat; best case was only `0.003286ms -> 0.002579ms` for rows `96` | vLLM/FlashInfer W13 layout conventions matter for monolithic kernels, but changing our current W13 output layout just to speed the standalone SwiGLU+quant kernel is too small. |
| HC direct mixes kernel | Microbench was faster for `seq_len<=16` and exact E2E still passed, but fixed long-decode hash changed to `0e73c031774b6142` | HC mix accumulation order is token-trace sensitive; exact short suites are not enough when routing depends on later tokens. |
| HC mixes unified `GemmEx` instead of `seq_len==1` GEMV | Exact E2E passed and fixed-bench hash stayed `6346f03343d75a65`, but TPOT regressed to `29.67-29.88ms/token` | The existing decode GEMV branch is a specialization smell, but replacing it with batch-general cuBLAS GEMM is not a performance win. A future cleanup needs a better batch-general HC-mix backend, not this substitution. |
| Full shared-expert overlap from MoE entry | Exact E2E still passed and TPOT was `27.126ms/token`, but fixed long-decode hashes changed to `877989965c7b859a`, `57230e28c8776f85`, and `da2087343aac2707` | Moving shared expert to an independent compute stream changed the long decode token trace; keep only the reduce-scatter/shared overlap that preserves hash. |
| Split shared W13 before router and shared W2 during reduce-scatter | Exact E2E passed and hash stayed stable, but fixed bench repeated at `26.62ms` then `27.15ms` | The extra scheduling split did not beat the simpler all-gather overlap path, and it made the shared helper surface noisier. |
| Replace hash-layer token-id all-gather with GPU repeat | Exact E2E passed and hash stayed stable, but fixed bench was `27.297ms` | The u32 token collective is too small to justify a new kernel surface; keep the simpler NCCL all-gather until a profile shows this launch matters. |
| Fuse Q/KV RoPE with KV no-PE quant | Exact E2E passed and hash stayed stable, but fixed bench regressed to `29.88-30.05ms/token` | Mixing rope-pair work and quant-reduction work in one launch hurt scheduling enough to erase the launch-count win. |
| Write ratio-4 window/compressed top-k directly into final buffer | Exact E2E passed and hash stayed stable, but fixed bench regressed to `29.53-29.55ms/token` | Removing a tiny concat launch changed the downstream top-k buffer path and did not improve the real attention window. |
| Fuse final HC head plus RMSNorm | Exact-safe but regressed TPOT | Saving small launches can lose to worse reduction/kernel shape. |
| Reuse deterministic window top-k across layers | Exact-safe, no stable long-bench win | Launch-count reduction alone is weak evidence. |
| Fuse KV RoPE plus no-PE quant | Exact-safe, regressed short decode | Combining tiny kernels can hurt scheduling/occupancy. |
| Hand-written decode HC mixes kernel | Exact-safe, slower than cuBLAS path | cuBLAS small GEMV remained better on this shape. |
| Isolated final logits scratch | Correct but noisy/regressive in repeated runs | Isolated storage movement near sampling boundary did not address the dominant per-layer allocation/skew structure. |
| Host-sized active-tile count for grouped MoE | Not used | Pulling active counts D2H would reintroduce hot-path synchronization. |

## Profiling And Benchmark Rules

### Token trace first

Always compare generated-token hashes before comparing TPOT. DeepSeek V4 routing and expert balance depend on token sequence. The bench JSON now records per-iteration timing and generated-token trace.

### Repeated fixed bench before claiming a win

The shared W13 branch showed a wide fixed-bench band with the same token hash: `32.220ms`, `30.061ms`, and `28.159ms` across consecutive 5090 repeats. A single sub-`30ms/token` run is therefore only evidence that the code path can enter that band, not that the optimization goal is achieved. Record multiple full JSON runs and prefer ranges over point estimates.

After the repeat series, idle `nvidia-smi` showed all GPUs back at `180MHz` SM / `405MHz` memory with no active throttle reason and no remaining `bench_serving`/`deepseek_v4`/`nsys` process. A follow-up fixed bench with `nvidia-smi --loop-ms 200` sampling produced steady TPOT `28.784ms` with all token hashes still `6346f03343d75a65`. Active-window clock averages were roughly `2622-2699MHz` SM and `13.7-13.8GHz` memory across ranks, with throttle reason always `0x0000000000000000`. That weakens the simple “slow run equals thermal/power throttle” hypothesis. The next diagnostic should add per-rank decode stage timestamps around attention local, collectives, routed MoE, shared expert, and logits to catch rank-arrival skew directly.

A temporary hard-coded trace for `start_pos == 80` synchronized each rank stream between broad decode stages, then logged per-rank totals. The trace build itself perturbs one steady token, so use it only for attribution. In one fixed bench with trace enabled, TPOT stayed in the fast band at `29.630ms` and all token hashes stayed `6346f03343d75a65`.

Trace summary across the 5 traced requests (`2` warmup + `3` measured), aggregated over all layers for a single steady decode token:

| Stage | Median / avg shape | Cross-rank range observed |
| --- | ---: | ---: |
| `attention_local` | avg `16.756ms` | `15.099-17.846ms` |
| `attention_collective_post` | avg `3.741ms` | `2.990-6.225ms` |
| `moe` | avg `15.112ms` | `14.779-15.401ms` |
| `hc_attn_pre` | avg `2.179ms` | `2.063-2.469ms` |
| `hc_ffn_pre` | avg `2.248ms` | `2.158-2.345ms` |
| `ffn_post` | avg `0.533ms` | `0.463-0.604ms` |
| `final_logits` | avg `0.268ms` | `0.241-0.352ms` |

Interpretation: the current shared/routed MoE path is not the largest source of rank skew in this trace. MoE is still a large absolute cost, but its rank range is only about `0.6ms`; the larger variance comes from attention local and attention collective+HC-post windows. The next optimization pass should not blindly keep fusing MoE kernels before explaining attention-local variability.

A narrower per-layer trace at the same hard-coded `start_pos == 80` logged `43 layers * 8 ranks * 5 requests = 1720` rows in `/tmp/dsv4_layer_trace_bench.log`. The fixed bench stayed on the same token trace (`6346f03343d75a65`) and measured around `30.84ms/token`, but this run included per-layer stream synchronizations and is attribution-only.

| Stage | Avg per layer | Approx 43-layer sum | p95 per layer | Max per layer |
| --- | ---: | ---: | ---: | ---: |
| HC attention pre-norm | `0.052ms` | `2.22ms` | `0.067ms` | `0.105ms` |
| Attention local | `0.399ms` | `17.16ms` | `0.489ms` | `0.724ms` |
| Attention all-reduce + HC post | `0.088ms` | `3.80ms` | `0.218ms` | `0.483ms` |
| HC FFN pre-norm | `0.052ms` | `2.24ms` | `0.061ms` | `0.077ms` |
| MoE, including AG/RS + shared expert | `0.354ms` | `15.23ms` | `0.390ms` | `0.428ms` |
| FFN HC post | `0.012ms` | `0.50ms` | `0.021ms` | `0.040ms` |

The largest single layer/request cross-rank ranges were `0.448ms` in layer `1` attention collective, `0.375ms` in layer `19` attention local, and `0.363ms` in layer `19` attention collective. There was no repeated multi-millisecond rank outlier in this per-layer view. That changes the next bet: stable sub-`30ms/token` is less likely to come from only rank-affinity or one MoE scalar cleanup, and more likely from reducing the largest absolute sections, namely attention local (`~17ms`) and MoE (`~15ms`), while keeping launch count and synchronization windows low.

A later short full-process nsys run on the current retained code used:

```bash
nsys profile --force-overwrite=true -t cuda,nvtx \
  -o /tmp/dsv4_current_short \
  target/release/bench_serving \
    --model-path /data/DeepSeek-V4-Flash \
    --format json \
    request \
    --prompt-len 1 \
    --output-len 32 \
    --warmup 1 \
    --iters 1 \
    --seed 42
```

The profile is attribution-only, but it confirms why the last MoE microbench wins did not become full-runtime wins:

| Kernel bucket | Total GPU time | Share | Notes |
| --- | ---: | ---: | --- |
| f32 all-reduce, wait-inclusive | `2.690s` | `20.59%` | NCCL wall includes rank-arrival wait; do not read as pure transfer. |
| Dense GEMM/GEMV other | `2.199s` | `16.83%` | Mostly non-MoE dense/HC/attention small GEMM/GEMV work. |
| HC scaffold/norm | `1.598s` | `12.23%` | HC pre/post/norm/Hadamard-style kernels remain launch-heavy. |
| f32 reduce-scatter, wait-inclusive | `1.399s` | `10.71%` | MoE RS window, also wait-inclusive. |
| Attention local | `1.009s` | `7.72%` | Compressor/indexer/sparse-attention local pieces. |
| Routed W13 grouped FP4 | `0.949s` | `7.26%` | Main routed expert local GEMM. |
| BF16 all-gather, wait-inclusive | `0.630s` | `4.82%` | MoE AG window, wait-inclusive. |
| Routed W2 grouped FP4 | `0.531s` | `4.06%` | Main routed down projection. |
| Shared W13 FP8 | `0.432s` | `3.31%` | Shared expert front half. |
| Router | `0.335s` | `2.56%` | Gate score/select/normalize. |
| Shared W2 FP8 | `0.252s` | `1.93%` | Shared expert down projection. |
| Routed mapping/expand/reduce | `0.128s` | `0.98%` | Small-route mapping is no longer a large section. |
| Routed SwiGLU+quant | `0.083s` | `0.64%` | Explains why accumulator-direct scalar epilogue cannot close the sub-25 gap alone. |

Interpretation: after the retained W13/SwiGLU/W2 path, the next real sub-25 work should not be another standalone activation/quant kernel swap. Either change the actual W13/W2 grouped GEMM scheduler/epilogue, or move to the larger attention/HC/dense-GEMV sections. NCCL rows stay wait-inclusive and should be paired with rank-arrival evidence before optimizing collectives.

After retaining MoE all-gather/reduce-scatter shared overlap, a fresh short profile `/tmp/dsv4_ag_overlap_short.nsys-rep` showed the new shape. The profile itself perturbed TPOT to `34.49ms/token`, so use only the kernel distribution:

| Kernel bucket | Total GPU time share | Notes |
| --- | ---: | --- |
| f32 all-reduce, wait-inclusive | `27.6%` | Still the largest apparent row; this is mostly attention/HC synchronization window, not pure transfer. |
| f32 reduce-scatter, wait-inclusive | `12.2%` | MoE RS still visible, but shared expert no longer serially follows all routed work. |
| HC pre-norm scaffold | `6.9%` | Launch-heavy HC scaffold remains material. |
| Routed W13 grouped FP4 | `6.5%` | Main routed expert front half. |
| BF16 all-gather, wait-inclusive | `5.4%` | Now overlapped with shared expert on the main compute stream. |
| Routed W2 grouped FP4 | `3.5%` | Main routed down projection. |
| Shared FP8 W13/W2 kernels | about `7.6%` combined | Much of this is hidden behind all-gather, but not behind router/local expert work. |

The follow-up split experiment tried to run shared W13 during all-gather and shared W2 during reduce-scatter. It was exact and hash-stable, but repeated fixed benches were `26.625ms` and `27.145ms`, not better than the simpler retained all-gather overlap. Keep the simpler schedule unless a future profile proves a different stream split is needed.

A post-`32768` grouped-FP4 shared-memory short profile `/tmp/dsv4_shared32768_short.nsys-rep` used the same `--output-len 32 --warmup 1 --iters 1 --seed 42` attribution workflow. The short run's 32-token hash was `5f6c64b667f2abf5`; it shares the same prefix as the fixed 160-token trace, but the hash is naturally different because the sequence length is different. The profile perturbed steady TPOT to `30.64ms/token`, so use the table only for composition:

| Kernel row | Share | Notes |
| --- | ---: | --- |
| f32 all-reduce, wait-inclusive | `19.3%` | Still the largest apparent row; includes rank-arrival wait. |
| f32 reduce-scatter, wait-inclusive | `16.6%` | MoE RS window remains large in wall attribution. |
| Routed W13 grouped FP4 | `8.9%` | Main routed front projection after shared-memory downsize. |
| HC pre-norm scaffold | `6.9%` | Launch-heavy HC section remains visible. |
| BF16 all-gather, wait-inclusive | `4.8%` | MoE AG, overlapped with shared expert in runtime. |
| Score-route BF16 GEMM | `4.7%` | cuBLAS BF16 score projection; direct BF16 cleanup kept correctness but did not erase router cost. |
| Routed W2 grouped FP4 | `4.3%` | Main routed down projection. |
| Shared FP8 W13 | `3.0%` | Shared expert front half. |
| WQ/WKV FP8 GEMMs | `2.9% + 2.9%` | Attention projection remains a broad contributor. |
| `act_quant_k4096` | `2.5%` | Shared across attention, shared expert, and routed W13 inputs. |
| Compressor/indexer local kernels | `~3.3%` visible top rows | Attention local still has many small kernels outside grouped MoE. |
| Routed SwiGLU+quant | `0.6%` | Confirms again that standalone activation/quant work is too small for sub-25. |

Interpretation: shared-memory downsize reduced the per-kernel grouped GEMM cost in microbench, but the full profile still has routed W13/W2 at about `13.2%` combined and the largest rows remain wait-inclusive collectives plus attention/HC local work. The next attempt should either change the grouped GEMM scheduler/epilogue enough to move `W13+W2`, or shift to attention/HC local reductions. Another isolated SwiGLU/quant kernel is unlikely to pay.

A careful decode-mid profile on 2026-05-13 used a fixed-bench calibration first, then an Nsight Systems delayed capture with CUDA event tracing disabled:

```bash
target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash \
  --format json \
  request \
  --prompt-len 1 \
  --output-len 160 \
  --warmup 2 \
  --iters 3 \
  --seed 42

nsys profile --force-overwrite=true \
  --trace=cuda,nvtx,cublas \
  --sample=none \
  --cpuctxsw=none \
  --cuda-event-trace=false \
  --delay=37 \
  --duration=4 \
  -o /tmp/dsv4_profile_decode_mid \
  target/release/bench_serving \
    --model-path /data/DeepSeek-V4-Flash \
    --format json \
    request \
    --prompt-len 1 \
    --output-len 160 \
    --warmup 1 \
    --iters 1 \
    --seed 42
```

The calibration JSON `/tmp/dsv4_profile_calibration_20260513_101406.json` had steady TPOT avg `28.347ms`, p50 `28.122ms`, p95 `29.653ms`, and all three generated-token hashes `6346f03343d75a65`. The delayed profile JSON `/tmp/dsv4_profile_decode_mid.json` kept the same 160-token hash and reported steady TPOT avg `29.222ms`; use the delayed profile for kernel composition only. The first attempted `--trace=nccl` failed because Nsight Systems 2025.1.3 on 5090 does not support `nccl` as a trace selector; NCCL is therefore identified by CUDA kernel names.

Nsight output files:

- `/tmp/dsv4_profile_decode_mid.nsys-rep`
- `/tmp/dsv4_profile_decode_mid.sqlite`
- `/tmp/dsv4_profile_decode_mid_stats_cuda_gpu_kern_sum.csv`
- `/tmp/dsv4_profile_decode_mid_stats_cuda_api_sum.csv`

The W13 instance count implies the delayed window captured about `94` steady decode steps: `32285 / (8 ranks * 43 layers) = 93.85`. The table below is aggregate GPU time across all 8 ranks in that capture window, not per-token wall time:

| Kernel bucket | Aggregate GPU share | Notes |
| --- | ---: | --- |
| FP8 dense/shared GEMM or quant | `19.8%` | Shared expert plus attention/HC dense FP8 work; still needs finer attribution before changing one subpath. |
| f32 reduce-scatter, wait-inclusive | `18.6%` | MoE RS synchronization window. Do not read as pure transfer. |
| HC/indexer local | `14.9%` | HC pre-norm, Hadamard, indexer, dot/reduce scaffold remains large and launch-heavy. |
| Routed W13 grouped FP4 | `10.3%` | Main routed expert front projection. |
| f32 all-reduce, wait-inclusive | `8.8%` | Attention/HC synchronization window; includes arrival skew. |
| MoE score router | `7.5%` | BF16 cuBLAS score projection plus select. Direct CUDA projection changed the long token trace and is rejected. |
| Attention local/MLA | `5.5%` | Compressor projection, sparse attention, residual RoPE/quant pieces. |
| Routed W2 grouped FP4 | `5.0%` | Main routed down projection. |
| BF16 all-gather, wait-inclusive | `3.2%` | MoE AG window, partly overlapped with shared expert. |
| Routed SwiGLU+W2 act quant | `0.7%` | Too small to justify another standalone activation/quant-only rewrite. |
| MoE route mapping | `0.4%` | Small-route mapping is no longer a material bottleneck. |

The same CSV bucketed by subsystem gives a more TPOT-aligned view. `Aggregate GPU ms/token` is the sum across all 8 ranks. `Rank avg ms/token` divides that by 8 and should be read as GPU busy time, not end-to-end wall time:

| Bucket | Share | Aggregate GPU ms/token | Rank avg ms/token | Instances/token across 8 ranks | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| HC/indexer/cuBLAS scaffold | `21.1%` | `42.92` | `5.36` | `5234.2` | `161.5us` |
| Dense/shared FP8/quant GEMM | `20.0%` | `40.73` | `5.09` | `5053.0` | `22.2us` |
| MoE reduce-scatter, wait-inclusive | `18.6%` | `37.77` | `4.72` | `340.3` | `505.7us` |
| Routed W13 grouped FP4 | `10.3%` | `20.93` | `2.62` | `344.0` | `136.6us` |
| Attention/HC all-reduce, wait-inclusive | `8.8%` | `17.80` | `2.23` | `510.3` | `1625.8us` |
| Attention local/MLA scaffold | `6.6%` | `13.48` | `1.69` | `3907.6` | `20.1us` |
| Routed W2 grouped FP4 | `5.0%` | `10.15` | `1.27` | `344.0` | `62.4us` |
| MoE/token all-gather, wait-inclusive | `3.2%` | `6.51` | `0.81` | `372.1` | `1929.4us` |
| MoE router/mapping/reduce local | `2.7%` | `5.58` | `0.70` | `1400.0` | `11.9us` |
| Other elementwise/embedding | `1.0%` | `2.09` | `0.26` | `1834.9` | `2.1us` |
| Routed/shared SwiGLU+FP8 quant | `0.7%` | `1.33` | `0.17` | `688.0` | `4.3us` |

This table is the current optimization budget. The profile-run TPOT was `29.222ms`, while bucketed single-rank GPU busy time sums to about `25.4ms/token`. The gap is expected because the table is GPU kernel duration aggregated from a perturbed nsys window and does not directly include cross-rank wall alignment, CPU launch/API overhead, or CUDA graph/NCCL scheduling boundaries. The key planning consequence is still stable: standalone activation/quant and route-mapping work is too small; HC/indexer, dense/shared FP8/quant, wait-inclusive collectives, and heavy routed W13/W2 are the remaining meaningful regions.

The sqlite was inspected in place on 5090 with Python `sqlite3`; do not move the full `316MB` sqlite file for routine review. The kernel window spans `3.207s`. Device totals were balanced in average kernel duration (`~9.86-10.14us`) but not in captured instance count, because delayed nsys cuts the global wall-time window across different rank/layer phases. Per-token instance counts from this capture are therefore approximate.

Instance-level duration buckets:

| Kernel | Count | Avg | P50 | P90 | P95 | P99 | Max | Important buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Routed W13 grouped FP4 | `32285` | `60.846us` | `72.384us` | `127.392us` | `128.672us` | `130.687us` | `136.608us` | `<5us`: `14127`; `70-120us`: `9281`; `>=120us`: `8830` |
| Routed W2 grouped FP4 | `32286` | `29.507us` | `36.160us` | `58.656us` | `59.232us` | `60.128us` | `62.432us` | `<5us`: `14127`; `35-70us`: `18143` |
| MoE reduce-scatter, wait-inclusive | `31941` | `110.982us` | `89.216us` | `204.161us` | `207.520us` | `234.015us` | `505.665us` | `>=120us`: `14055` |
| Attention/HC all-reduce, wait-inclusive | `47889` | `34.892us` | `21.920us` | `29.344us` | `63.392us` | `416.861us` | `1625.787us` | `>=120us`: `1096` |
| MoE/token all-gather, wait-inclusive | `34925` | `17.506us` | `14.080us` | `27.648us` | `35.840us` | `48.160us` | `1929.410us` | `>=120us`: `32` |
| HC pre-norm from mixes | `63881` | `24.821us` | `24.863us` | `25.184us` | `25.280us` | `25.440us` | `25.920us` | all in `15-35us` |
| Sparse attention local | `32284` | `13.249us` | `12.095us` | `16.160us` | `16.416us` | `16.800us` | `17.600us` | `5-15us`: `19485`; `15-35us`: `12799` |
| Score select | `30028` | `11.436us` | `11.456us` | `11.584us` | `11.616us` | `11.680us` | `11.872us` | all in `5-15us` |

Two device-0 timeline samples explain the routed expert shape:

```text
empty/near-empty local route:
AG 13.2us overlaps shared expert
shared W13 20.1us -> shared SwiGLU+quant 1.6us -> shared W2 11.4us
score GEMM 36.5us -> select 11.3us -> mapping 1.3us -> expand 1.2us -> act_quant 4.1us
routed W13 2.7us -> routed SwiGLU+quant 2.2us -> routed W2 2.7us -> reduce 1.2us
RS 139.0us -> add/post

heavy local route:
AG 13.8us overlaps shared expert
shared W13 20.4us -> shared SwiGLU+quant 1.6us -> shared W2 11.4us
score GEMM 36.5us -> select 11.3us -> mapping 3.3us -> expand 1.2us -> act_quant 4.1us
routed W13 124.6us -> routed SwiGLU+quant 2.2us -> routed W2 57.8us -> reduce 1.4us
RS 14.5us -> add/post
```

The routed W13/W2 launches use the same `grid=(32,2,32), block=128, dynamic_shared=32768` in both samples. The difference is not launch shape; it is the actual active expert work that survives inside the grouped kernel. This is why active-expert empty-launch cleanup alone is not enough. A meaningful MoE change must reduce the heavy W13/W2 path or reduce the synchronization window created after it.

Selected per-captured-token aggregate GPU time across all 8 ranks:

| Kernel | Aggregate GPU ms/token | Instances/token across 8 ranks |
| --- | ---: | ---: |
| f32 reduce-scatter | `37.77` | `340.3` |
| Routed W13 grouped FP4 | `20.93` | `344.0` |
| f32 all-reduce | `17.80` | `510.3` |
| HC pre-norm | `16.89` | `680.7` |
| Score-route cuBLAS | `11.63` | `316.5` |
| Routed W2 grouped FP4 | `10.15` | `344.0` |
| MoE all-gather | `6.52` | `372.1` |
| Compressor projection | `5.50` | `495.9` |
| Sparse attention local | `4.56` | `344.0` |
| Shared/dense W13 | `7.04` | `344.0` |
| Shared/dense W2 | `3.97` | `344.0` |

Device-spread check: local W13/W2 kernels are not showing a persistent single-GPU slowdown. W13 averages were about `56.7-64.5us` by device, W2 averages `27.7-31.1us`. The larger long tails are wait-inclusive collectives: f32 all-reduce max was about `1.62ms`, all-gather max about `1.93ms`, while W13 max stayed about `0.136ms`. This points back to upstream arrival/synchronization and broad local stage cost, not one pathological grouped GEMM launch.

Single-device kernel-sequence inspection on device 0 shows the steady pattern around MoE:

```text
HC pre-norm -> shared act_quant -> MoE all-gather on NCCL stream
main stream shared W13 -> shared SwiGLU+quant -> shared W2
router -> mapping -> expand -> routed act_quant
routed W13 -> routed SwiGLU+quant -> routed W2 -> reduce
MoE reduce-scatter on NCCL stream -> routed+shared add
```

The grouped W13/W2 launches use a fixed grid, typically `grid=(32,2,32), block=128`, even when local routing is empty or nearly empty. The important nuance is that those empty/near-empty launches are already cheap:

| Kernel | Device-0 count | Duration buckets |
| --- | ---: | --- |
| W13 grouped FP4 | `4069` | `<5us`: `1727`; `70-120us`: `1308`; `>=120us`: `1025` |
| W2 grouped FP4 | `4069` | `<5us`: `1727`; `35-70us`: `2339` |
| f32 reduce-scatter | `4026` | `<15us`: `288`; `15-35us`: `667`; `35-70us`: `814`; `70-120us`: `536`; `>=120us`: `1721` |

This refines the earlier active-expert conclusion: empty grouped-GEMM launches are not free, but they are not the dominant cost. A runtime problem-size path still needs to reduce the heavy W13/W2 work or a synchronization window; only skipping already-cheap empty launches will not close the gap.

Interpretation for the next patch: do not chase another isolated SwiGLU/quant or route-mapping microkernel. The remaining plausible sub-25 levers are either a real grouped W13/W2 scheduler/epilogue change that moves the `15.3%` routed expert GEMM bucket, or a batch-general reduction of HC/attention local launch clusters and the arrival skew that feeds the wait-inclusive collective windows.

A temporary hard-coded `start_pos == 80` trace was rerun after retaining MoE all-gather overlap. The trace synchronized rank streams and is attribution-only; fixed bench token hashes stayed `6346f03343d75a65`, but timing was perturbed by logging/synchronization. The log is `/tmp/dsv4_trace_current_bench.log`.

Excluding the first traced layer-0 MoE warmup outlier, the broad per-layer shape was:

| Stage | Avg per layer | Approx 43-layer sum | Notes |
| --- | ---: | ---: | --- |
| Attention local | `0.404ms` | `17.37ms` | Still the largest absolute section. |
| MoE, including AG/RS + shared expert | `0.320ms` | `13.77ms` | Improved versus the earlier `~15.23ms` trace after AG/shared overlap. |
| Attention all-reduce + HC post | `0.076ms` | `3.29ms` | Rank averages were close; NCCL rows in nsys are still wait-inclusive. |
| HC FFN pre-norm | `0.054ms` | `2.33ms` | Launch-heavy scaffold. |
| HC attention pre-norm | `0.051ms` | `2.19ms` | Launch-heavy scaffold. |
| FFN HC post | `0.011ms` | `0.45ms` | Small. |

Per-rank averages in this trace were tightly grouped: attention local ranged from `0.3937-0.4067ms` per layer across ranks, MoE ranged from `0.3443-0.3498ms`, and attention collective+post ranged from `0.0708-0.0829ms`. That points away from a persistent NUMA/rank-affinity skew issue in the current retained code. The next sub-25 bet should either reduce the large absolute attention-local/HC sections or make a real grouped GEMM scheduler/epilogue improvement; tiny MoE launch substitutions have repeatedly failed to move TPOT.

### Rejected: HC direct mixes kernel

The current HC pre-norm path calls `deepseek_hc_mixes_cuda`, which does:

```text
BF16 hidden -> F32 scratch
cuBLAS SGEMV/GemmEx for 24 HC mix values
RMS scaling of the 24 mix values
```

A temporary standalone bench compared that path with a direct CUDA kernel that computes each `(token, mix)` dot product from BF16 input and applies the same RMS scale. The candidate is batch-general over `seq_len`, but it changes the F32 reduction order versus cuBLAS. The bench source was deleted after rejection so only the validated logs and decision remain.

Local microbench after reverting runtime to the retained cuBLAS path:

| `seq_len` | cuBLAS path | Direct kernel | Speedup | Max scaled-mix abs diff |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `0.022172ms` | `0.006154ms` | `3.603x` | `1.55e-6` |
| `8` | `0.073793ms` | `0.008211ms` | `8.987x` | `2.86e-6` |
| `16` | `0.073808ms` | `0.014348ms` | `5.144x` | `2.98e-6` |
| `32` | `0.026653ms` | `0.024604ms` | `1.083x` | `3.81e-6` |
| `64` | `0.026652ms` | `0.045095ms` | `0.591x` | `4.77e-6` |

Runtime experiment result on 5090:

| Check | Result |
| --- | --- |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench | steady TPOT avg `28.555ms`, per-iteration `28.480ms`, `28.496ms`, `28.690ms` |
| generated-token hash | `0e73c031774b6142`, not the retained `6346f03343d75a65` |

Drop decision: do not retain the runtime branch. The direct HC mix kernel is a good diagnostic and shows the cost is reducible, but token hash changed even though the short exact suite passed. After reverting the runtime branch and rebuilding 5090 release binaries, the fixed bench returned to hash `6346f03343d75a65` with per-iteration steady TPOT `28.312ms`, `28.391ms`, and `28.277ms`. Any future HC replacement must either match the cuBLAS accumulation path closely enough to preserve the long token trace, or intentionally regenerate the DeepSeek V4 baseline after a broader numerical-parity decision.

Rejected follow-up: remove the existing `seq_len == 1` cuBLAS GEMV branch in `deepseek_hc_mixes_cuda` and always use the batch-general `cublasGemmEx` path. This was a guardrail-aligned cleanup attempt, not a bs=1 optimization. It kept exactness but was slower:

| Check | Result |
| --- | --- |
| release build | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench | aggregate steady TPOT avg `29.745ms`; per-iteration `29.881ms`, `29.668ms`, `29.686ms` |
| token hash | all `6346f03343d75a65` |

Drop decision: do not retain. The branch is still undesirable from a cleanliness perspective, but the batch-general cuBLAS GEMM replacement regresses the fixed decode bench. The viable future path is a better HC-mix implementation that is both batch-general and performance-positive, not this substitution.

Rejected follow-up: pre-scale HC input before cuBLAS. The temporary patch changed `deepseek_hc_mixes_cuda` from:

```text
BF16 -> F32 scratch
cuBLAS dot
scale mixes by RMS(x)
```

to:

```text
BF16 -> scaled F32 scratch
cuBLAS dot
```

This was batch-general and kept the cuBLAS dot backend, but it changed the F32 multiplication order from `dot(x, w) * scale` to `dot(x * scale, w)`. Exact E2E still passed, but the fixed long token trace changed:

| Check | Result |
| --- | --- |
| release build | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench | aggregate steady TPOT avg `28.404ms`; per-iteration `28.392ms`, `28.402ms`, `28.419ms` |
| token hash | all `3ed2ed5c96344f69`, not the retained `6346f03343d75a65` |

Drop decision: do not retain. This confirms the HC mixes path is sensitive not only to cuBLAS reduction order but also to where the RMS scale multiplication occurs. After reverting and rebuilding 5090 release binaries, the fixed bench returned to hash `6346f03343d75a65` with aggregate steady TPOT avg `29.181ms`, per-iteration `29.260ms`, `29.456ms`, and `28.828ms`.

Rejected follow-up: apply the grouped-FP4 dynamic shared-memory downsize to dense FP8 GEMM wrappers. The temporary patch changed the FP8 GEMM and shared W13 launcher request from `98304` to `32768` bytes while leaving the TileLang kernels and math unchanged. Unlike grouped FP4, dense FP8 kernels need the larger dynamic shared-memory window: the exact E2E failed immediately at prefill layer 0 with `CUDA_ERROR_ILLEGAL_ADDRESS`.

| Check | Result |
| --- | --- |
| release build | passed |
| release `deepseek_v4_e2e` | failed case 0, `prefill layer 0 rank 0: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")` |
| fixed bench | not run for the failed branch |

Drop decision: do not retain. Keep dense FP8/shared expert launchers at `98304` bytes. After reverting and rebuilding 5090 release binaries, exact E2E returned to `All 20 DeepSeek V4 exact cases passed`, and the fixed bench returned to hash `6346f03343d75a65` with aggregate steady TPOT avg `28.739ms`, per-iteration `28.552ms`, `28.899ms`, and `28.766ms`.

### Retained: parallel score-route top-k select

The score-router path now keeps the exact BF16 `cublasGemmEx` score projection and only changes the post-GEMM selection kernel. The old `deepseek_score_gate_select_kernel` used all threads to compute `sqrt(softplus(raw_score)) + bias`, then let `threadIdx.x == 0` serially scan `n_experts * topk` to pick the selected experts. The retained version uses a block-level max reduction for each top-k slot, with the same tie rule as the serial scan: larger score wins, and equal score keeps the lower expert id. For DSV4 `topk == 6`, it also preserves the existing selected-sum order:

```text
(w0 + w4 + w2) + (w1 + w5 + w3)
```

This is not a vLLM/SGLang architecture import; it is a local cleanup of a small visible kernel. The important process rule is that small kernels are decided by microbench first, then full E2E/bench only as safety gates.

Standalone tool:

```bash
/usr/local/cuda-12.9/bin/nvcc \
  -O3 \
  -std=c++17 \
  -arch=sm_120 \
  pegainfer-kernels/tools/deepseek_v4/score_select_bench.cu \
  -o /tmp/dsv4_score_select_bench

/tmp/dsv4_score_select_bench
```

Microbench result on 5090:

| `seq_len` | Serial select | Parallel select | Speedup | Mismatch |
| ---: | ---: | ---: | ---: | --- |
| `1` | `0.012283ms` | `0.008188ms` | `1.500x` | indices `0`, weights `0`, max diff `0` |
| `8` | `0.012294ms` | `0.008188ms` | `1.501x` | indices `0`, weights `0`, max diff `0` |
| `16` | `0.012294ms` | `0.008191ms` | `1.501x` | indices `0`, weights `0`, max diff `0` |
| `32` | `0.012294ms` | `0.008196ms` | `1.500x` | indices `0`, weights `0`, max diff `0` |

Runtime validation:

| Check | Result |
| --- | --- |
| release build | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench | aggregate steady TPOT avg `28.458ms`; per-iteration `28.343ms`, `28.468ms`, `28.562ms` |
| token hash | all `6346f03343d75a65` |

Keep decision: retain. The full-runtime gain is naturally small because score select is only a small part of the score-router bucket, but the microbench is clean, bitwise, and batch-general. Do not use this as evidence that the larger score projection should move away from cuBLAS; direct score projection already changed the long token hash and remains rejected.

### Rejected: hand routed W13 act_quant

Following the small-kernel rule, routed W13 activation quantization was tested in a temporary standalone microbench before deciding whether to keep a runtime patch. The benchmark compared the generated TileLang `act_quant_k4096/k2048` launch against a hand CUDA kernel with one warp per `(row, 128-column group)` and the same BF16, FP8 E4M3, and E8M0 scale conversion semantics. The temporary source was deleted after rejection so it cannot be wired into runtime later.

Microbench result on 5090:

| Hidden dim | Rows | TileLang | Hand CUDA | Speedup | Mismatch |
| ---: | ---: | ---: | ---: | ---: | --- |
| `4096` | `8` | `0.006144ms` | `0.002053ms` | `2.992x` | output `0`, scale `0` |
| `2048` | `8` | `0.006142ms` | `0.002048ms` | `2.999x` | output `0`, scale `0` |
| `4096` | `48` | `0.004100ms` | `0.002050ms` | `2.000x` | output `0`, scale `0` |
| `2048` | `48` | `0.004098ms` | `0.002049ms` | `2.000x` | output `0`, scale `0` |
| `4096` | `96` | `0.004096ms` | `0.002048ms` | `2.000x` | output `0`, scale `0` |
| `2048` | `96` | `0.004095ms` | `0.002000ms` | `2.047x` | output `0`, scale `0` |

Runtime patch tested:

```text
routed W13:
  TileLang act_quant_k4096(expanded_input)
  -> grouped FP4 W13 GEMM

temporary variant:
  hand CUDA act_quant_k4096(expanded_input)
  -> grouped FP4 W13 GEMM
```

Runtime validation:

| Check | Result |
| --- | --- |
| release build | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench | aggregate steady TPOT avg `28.802ms`; per-iteration `29.014ms`, `28.744ms`, `28.647ms` |
| token hash | all `6346f03343d75a65` |

Drop decision: do not retain the runtime patch. The microbench win is real and bitwise, but it saves about `0.002-0.004ms` per launch in isolation and did not improve the end-to-end decode schedule. After reverting to TileLang act_quant and rebuilding 5090 release binaries, exact E2E returned to `20/20`; the repeat fixed bench was aggregate steady TPOT avg `29.277ms`, per-iteration `29.265ms`, `29.272ms`, and `29.293ms`, with all hashes `6346f03343d75a65`. One immediate post-revert run had a single iteration hash `a278a8140c25b812` while the repeat returned to the expected hash; treat that as a bench-determinism warning and keep checking all per-iteration hashes, not only aggregate TPOT.

### NCCL wall is wait-inclusive

Nsight Systems NCCL kernel wall time includes rank-arrival waiting. Treat NCCL rows as synchronization-window evidence unless rank-arrival skew and post-arrival tail have been separated. The rank-affinity work was selected because corrected f32 all-reduce grouping showed attention hidden all-reduce dominated by arrival skew, not post-arrival NCCL tail.

### Allocation proof

Full-process nsys attribution was not reliable enough for allocation proof:

- nsys-only `cuMemAllocAsync` attribution did not reconcile with application-visible symbols.
- CUDA event tracing can distort API counts.
- NCCL wall can dominate profile views while reflecting upstream skew.

The retained allocation evidence combines source-level inventory with `tools/cuda_api_counter.c`, an `LD_PRELOAD` counter that covers directly linked runtime/driver symbols and CUDA driver function-table lookup via `cuGetProcAddress`.

| API group | Baseline | Current |
| --- | ---: | ---: |
| `cudaMalloc` calls | `12944` | `136` |
| `cudaFree` calls | `12848` | `32` |
| `cuMemAllocAsync/cuMemFreeAsync/cuMemsetD8Async` | noisy nsys-only attribution | `0/0/0` in counter |
| `cudaMallocAsync/cudaFreeAsync/cudaMemsetAsync` | not used | `0/0/0` |
| `cuGetProcAddress` replacements | not covered | `0` |

The current retained tree was rechecked on 5090 with a short `--output-len 64 --warmup 1 --iters 1 --seed 42` counter run:

| API group | Counter result |
| --- | ---: |
| `cuMemAllocAsync/cuMemFreeAsync/cuMemsetD8Async` | `0/0/0` |
| `cudaMallocAsync/cudaFreeAsync/cudaMemsetAsync` | `0/0/0` |
| `cuMemcpyHtoDAsync_v2/cuMemcpyDtoHAsync_v2` | `0/0` |
| `cudaMalloc/cudaFree` | `96/8` |
| `cudaLaunchKernel` | `2074256` |

This confirms the nsys decode-mid `cuMemAllocAsync/cuMemFreeAsync` attribution is not an application-visible hot allocation signal. For DSV4 decode, treat nsys API rows as a hint only; an allocation claim needs the `LD_PRELOAD` counter or a source-level owner. The useful signal in this run is that the hot path is launch/synchronization dominated, not async allocation dominated.

The counter exports base and `_ptsz` wrappers separately for `cuMemAllocAsync`, `cuMemFreeAsync`, and `cuMemsetD8Async`. Do not share one stored real function pointer across base and `_ptsz` variants.

## Remote Workflow Notes

Remote test syncs should use touched-file `rsync -azR`. A full repository rsync with delete/excludes stalled for about 10 minutes during this work. A repeated mistake here was running multi-source `rsync` without `-R`, which copied `index.md`, `decode-performance.md`, `core.rs`, `moe.rs`, `state.rs`, `deepseek_quant.cu`, `ffi.rs`, and `swiglu_quant_bench.cu` into the remote repository root as basename files. Clean those accidental root files immediately and resend with `-R` so paths are preserved. Also, `cargo check` does not rebuild already-built release binaries; rebuild `deepseek_v4_e2e` and `bench_serving` before trusting remote validation.

Verified command set for this PR:

```bash
cargo fmt --check
cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4
cargo check --release -p pegainfer-server --features deepseek-v4
gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl
```

## Validation

Local:

- `cargo fmt --check`
- `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`
- `cargo check --release -p pegainfer-server --features deepseek-v4`
- `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl`
- `nm -D /tmp/cuda_api_counter.so` confirmed base and `_ptsz` wrappers
- `git diff --check`
- pre-commit hooks on commit, including clippy

5090:

- `cargo fmt --check`
- `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`
- `cargo check --release -p pegainfer-server --features deepseek-v4`
- release `deepseek_v4_e2e`: `All 20 DeepSeek V4 exact cases passed`
- release fixed bench log `/tmp/dsv4_pr_driver_numa_bench.log`: steady TPOT avg `35.253ms`, p50 `34.800ms`, p95 `37.335ms`, first decode avg `33.743ms`, hash `6346f03343d75a65`
- current clean fixed bench log `/tmp/dsv4_clean_tpot_now.log`: per-iteration steady TPOT avg `29.944ms`, `29.907ms`, `29.896ms`, all hash `6346f03343d75a65`
- current exact E2E log `/tmp/dsv4_e2e_current.log`: `All 20 DeepSeek V4 exact cases passed`
- fused-clear exact E2E log `/tmp/dsv4_clear_fused_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- fused-clear fixed bench log `/tmp/dsv4_clear_fused_bench.log`: per-iteration steady TPOT avg `29.862ms`, `29.969ms`, `29.874ms`, all hash `6346f03343d75a65`
- fused-clear short nsys log `/tmp/dsv4_clear_fused_short_profile.txt`: `deepseek_moe_clear_mapping_kernel` replaces the old repeated clear kernel
- small-route exact E2E log `/tmp/dsv4_small_mapping_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- small-route fixed bench logs `/tmp/dsv4_small_mapping_bench.log` and `/tmp/dsv4_small_mapping_bench_repeat.log`: per-iteration steady TPOT avg `27.608ms`, `27.662ms`, `27.826ms`, then `27.698ms`, `27.693ms`, `27.644ms`; all hash `6346f03343d75a65`
- small-route short nsys log `/tmp/dsv4_small_mapping_short_profile.txt`: `deepseek_moe_local_mapping_small_kernel` is the decode-sized route mapping path
- rejected route-W13 exact E2E log `/tmp/dsv4_route_w13_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected route-W13 fixed bench log `/tmp/dsv4_route_w13_bench.log`: aggregate steady TPOT avg `33.217ms`, per-iteration `33.162ms`, `33.355ms`, `33.135ms`; all hash `6346f03343d75a65`
- rejected expand+act_quant exact E2E log `/tmp/dsv4_expand_act_quant_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected expand+act_quant fixed bench logs `/tmp/dsv4_expand_act_quant_bench.log` and `/tmp/dsv4_expand_act_quant_bench_repeat.log`: per-iteration steady TPOT avg `28.080ms`, `28.143ms`, `27.198ms`, then `28.509ms`, `28.712ms`, `28.474ms`; all hash `6346f03343d75a65`
- rejected W2 valid-row exact E2E log `/tmp/dsv4_valid_rows_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected W2 valid-row fixed bench log `/tmp/dsv4_valid_rows_bench.log`: aggregate steady TPOT avg `31.270ms`, per-iteration `31.220ms`, `31.342ms`, `31.248ms`; all hash `6346f03343d75a65`
- rejected grouped GEMM row-tile upper-bound exact E2E log `/tmp/dsv4_max_expert_rows_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected grouped GEMM row-tile upper-bound fixed bench log `/tmp/dsv4_max_expert_rows_bench.log`: per-iteration steady TPOT avg `28.504ms`, `28.460ms`, `28.735ms`; all hash `6346f03343d75a65`
- active-expert W13/W2 compact-pointer microbench on 5090: `/tmp/w13_grouped_fp4_bench --experts 32 --active-experts {1,3,6} --rows-per-active 8`; bitwise PASS, W13 compact speedup `0.999x`, `1.009x`, `1.000x`, W2 compact speedup `1.000x`, `1.000x`, `1.000x`
- accumulator-direct SwiGLU quant upper-bound microbench on 5090: `/tmp/swiglu_quant_bench --rows {48,96,192}`; bitwise PASS, materialized/direct deltas `0.002255ms`, `0.000589ms`, `0.002048ms`
- rejected standalone TileLang SwiGLU quant log `/tmp/dsv4_tilelang_swiglu_quant_bench.log`: byte-identical to current C++ fused SwiGLU+FP8 quant, but rows `48/96/192` measured `0.388x`, `0.653x`, and `0.998x` of the current C++ fused kernel. The generated launcher was removed after the probe.
- rejected TileLang W13 accumulator-to-SwiGLU-quant prototype log `/tmp/dsv4_w13_swiglu_quant_bench.log`: the temporary generated kernel compiled after fixing shared-buffer pipeline planning and amax reduction shape, but the first active-expert fuzz shape failed byte comparison against current `grouped_w13_gemm -> C++ fused SwiGLU+FP8 quant`; the generated launcher and C++ bench were removed.
- current retained fixed bench log `/tmp/dsv4_current_retained_bench.log`: per-iteration steady TPOT avg `28.940ms`, `28.942ms`, `28.913ms`; all hash `6346f03343d75a65`
- current retained short profile `/tmp/dsv4_current_short.nsys-rep` and kernel summary `/tmp/dsv4_current_short_kernels_cuda_gpu_kern_sum.csv`: routed SwiGLU+quant only `0.64%`, routed W13/W2 grouped FP4 together `11.32%`, and wait-inclusive NCCL rows remain the largest apparent rows.
- rejected HC direct mixes exact E2E log `/tmp/dsv4_hc_direct_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected HC direct mixes fixed bench log `/tmp/dsv4_hc_direct_bench.log`: aggregate steady TPOT avg `28.555ms`, per-iteration `28.480ms`, `28.496ms`, `28.690ms`; hash changed to `0e73c031774b6142`
- post-revert HC fixed bench log `/tmp/dsv4_hc_reverted_bench.log`: aggregate steady TPOT avg `28.327ms`, per-iteration `28.312ms`, `28.391ms`, `28.277ms`; all hash `6346f03343d75a65`
- rejected HC mixes unified `GemmEx` exact E2E log `/tmp/dsv4_hc_gemmex_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected HC mixes unified `GemmEx` fixed bench log `/tmp/dsv4_hc_gemmex_bench.json`: aggregate steady TPOT avg `29.745ms`, per-iteration `29.881ms`, `29.668ms`, `29.686ms`; all hash `6346f03343d75a65`. The runtime branch was reverted and 5090 release binaries were rebuilt back to the retained HC GEMV/GemmEx split.
- rejected HC pre-scaled input exact E2E log `/tmp/dsv4_hc_prescale_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected HC pre-scaled input fixed bench log `/tmp/dsv4_hc_prescale_bench.json`: aggregate steady TPOT avg `28.404ms`, per-iteration `28.392ms`, `28.402ms`, `28.419ms`; all generated-token hashes changed to `3ed2ed5c96344f69`
- post-revert HC pre-scaled input fixed bench log `/tmp/dsv4_hc_prescale_reverted_bench.json`: aggregate steady TPOT avg `29.181ms`, per-iteration `29.260ms`, `29.456ms`, `28.828ms`; all hashes restored to `6346f03343d75a65`
- rejected dense FP8 shared-memory downsize exact E2E log `/tmp/dsv4_fp8_shared32768_e2e.log`: failed case 0 with `CUDA_ERROR_ILLEGAL_ADDRESS` at `prefill layer 0 rank 0`
- post-revert dense FP8 shared-memory downsize exact E2E log `/tmp/dsv4_fp8_shared32768_reverted_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- post-revert dense FP8 shared-memory downsize fixed bench log `/tmp/dsv4_fp8_shared32768_reverted_bench.json`: aggregate steady TPOT avg `28.739ms`, per-iteration `28.552ms`, `28.899ms`, `28.766ms`; all hashes restored to `6346f03343d75a65`
- retained parallel score-select microbench log `/tmp/dsv4_score_select_bench.log`: serial vs parallel top-k select is bitwise identical for `seq_len={1,8,16,32}` with speedup about `1.50x`
- retained parallel score-select exact E2E log `/tmp/dsv4_score_select_parallel_retained_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- retained parallel score-select fixed bench log `/tmp/dsv4_score_select_parallel_retained_bench.json`: aggregate steady TPOT avg `28.458ms`, per-iteration `28.343ms`, `28.468ms`, `28.562ms`; all hashes `6346f03343d75a65`
- rejected hand routed W13 act_quant microbench log `/tmp/dsv4_act_quant_bench.log`: hand CUDA act_quant was bitwise-identical to TileLang for `hidden_dim={4096,2048}` and `rows={8,48,96}`, with `2.0-3.0x` standalone speedup.
- rejected hand routed W13 act_quant exact E2E log `/tmp/dsv4_hand_act_quant_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected hand routed W13 act_quant fixed bench log `/tmp/dsv4_hand_act_quant_bench.json`: aggregate steady TPOT avg `28.802ms`, per-iteration `29.014ms`, `28.744ms`, `28.647ms`; all hashes `6346f03343d75a65`
- post-revert hand act_quant exact E2E log `/tmp/dsv4_act_quant_restored_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- post-revert hand act_quant fixed bench logs `/tmp/dsv4_act_quant_restored_bench.json` and `/tmp/dsv4_act_quant_restored_bench_repeat.json`: first run aggregate steady TPOT avg `28.249ms` but one iteration hash changed to `a278a8140c25b812`; repeat aggregate steady TPOT avg `29.277ms`, per-iteration `29.265ms`, `29.272ms`, `29.293ms`, with all hashes restored to `6346f03343d75a65`
- post-revert hand act_quant 5-run stability logs `/tmp/dsv4_stability_after_act_quant_revert_{1..5}.json`: aggregate steady TPOT avg `28.912ms`, `28.867ms`, `28.291ms`, `28.375ms`, and `28.715ms`; all 15 per-iteration hashes were `6346f03343d75a65`. Another CPU load was running during this sweep, so the result is a conservative sub-30 stability check rather than a clean machine best-band.
- reviewer 5090 5-run stability rerun `/tmp/pegainfer_dev_pr101_bench_{1..5}.json`: aggregate steady TPOT avg `28.505793ms`, `28.087102ms`, `29.755957ms`, `27.552965ms`, and `29.371630ms`; all 15 per-iteration hashes were `6346f03343d75a65`. One run wrote the complete JSON report and logged scheduler exit, then segfaulted in NCCL shutdown; treat that as the existing shutdown cleanup issue, not decode TPOT or token-correctness evidence.
- fused Q/KV RoPE exact E2E log `/tmp/dsv4_qkv_rope_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- fused Q/KV RoPE fixed bench logs `/tmp/dsv4_qkv_rope_bench.log` and `/tmp/dsv4_qkv_rope_bench_repeat.log`: per-iteration steady TPOT avg `28.215ms`, `28.256ms`, `28.236ms`, then `27.096ms`, `28.565ms`, `28.349ms`; all hash `6346f03343d75a65`
- fused Q/KV RoPE short profile `/tmp/dsv4_qkv_rope_short.nsys-rep` and `/tmp/dsv4_qkv_rope_short_kernels_cuda_gpu_kern_sum.csv`: `deepseek_apply_rope_q_kv_kernel` appears in the kernel summary; residual hidden-RoPE kernels are from non-projection paths.
- rejected Q/KV RoPE+KV quant exact E2E log `/tmp/dsv4_qkv_rope_quant_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected Q/KV RoPE+KV quant fixed bench log `/tmp/dsv4_qkv_rope_quant_bench.log`: aggregate steady TPOT avg `29.948ms`, per-iteration `29.885ms`, `29.910ms`, `30.048ms`; all hash `6346f03343d75a65`
- rejected ratio-4 top-k concat removal exact E2E log `/tmp/dsv4_topk_no_concat_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected ratio-4 top-k concat removal fixed bench log `/tmp/dsv4_topk_no_concat_bench.log`: aggregate steady TPOT avg `29.541ms`, per-iteration `29.551ms`, `29.539ms`, `29.532ms`; all hash `6346f03343d75a65`
- post-revert ratio-4 top-k fixed bench log `/tmp/dsv4_revert_topk_bench.log`: aggregate steady TPOT avg `28.333ms`, per-iteration `28.316ms`, `28.336ms`, `28.346ms`; all hash `6346f03343d75a65`
- old split MoE/SwiGLU cleanup: local `cargo fmt --check`, local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`, local `git diff --check`, 5090 `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`, and 5090 release builds for `bench_serving` and `deepseek_v4_e2e` passed after removing stale public/FFI exports.
- old split MoE/SwiGLU cleanup exact E2E log `/tmp/dsv4_fused_cleanup_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- old split MoE/SwiGLU cleanup fixed bench log `/tmp/dsv4_fused_cleanup_bench.log`: aggregate steady TPOT avg `27.860ms`, per-iteration `27.863ms`, `27.845ms`, `27.872ms`; all hash `6346f03343d75a65`
- MoE reduce-scatter/shared overlap exact E2E log `/tmp/dsv4_moe_rs_overlap_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- MoE reduce-scatter/shared overlap fixed bench log `/tmp/dsv4_moe_rs_overlap_bench.log`: aggregate steady TPOT avg `26.791ms`, per-iteration `26.773ms`, `26.794ms`, `26.805ms`; all hash `6346f03343d75a65`
- MoE reduce-scatter/shared overlap fixed bench repeats `/tmp/dsv4_moe_rs_overlap_bench_repeat.log` and `/tmp/dsv4_moe_rs_overlap_bench_repeat2.log`: aggregate steady TPOT avg `27.771ms` and `27.776ms`; per-iteration range `27.637-27.985ms`; all hash `6346f03343d75a65`
- rejected full shared-expert overlap exact E2E log `/tmp/dsv4_shared_full_overlap_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected full shared-expert overlap fixed bench log `/tmp/dsv4_shared_full_overlap_bench.log`: aggregate steady TPOT avg `27.126ms`, but generated-token hashes changed to `877989965c7b859a`, `57230e28c8776f85`, and `da2087343aac2707`
- post-revert MoE reduce-scatter/shared overlap exact E2E log `/tmp/dsv4_moe_rs_overlap_revert_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- post-revert MoE reduce-scatter/shared overlap fixed bench log `/tmp/dsv4_moe_rs_overlap_revert_bench.log`: aggregate steady TPOT avg `28.435ms`, per-iteration `28.536ms`, `28.387ms`, `28.381ms`; all hash `6346f03343d75a65`
- MoE all-gather/reduce-scatter shared-overlap exact E2E log `/tmp/dsv4_moe_ag_overlap_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- MoE all-gather/reduce-scatter shared-overlap fixed bench logs `/tmp/dsv4_moe_ag_overlap_bench.log` and `/tmp/dsv4_moe_ag_overlap_bench_repeat.log`: aggregate steady TPOT avg `27.052ms` and `26.285ms`; per-iteration range `26.280-27.309ms`; all hash `6346f03343d75a65`
- MoE all-gather/reduce-scatter shared-overlap short profile `/tmp/dsv4_ag_overlap_short.nsys-rep` and `/tmp/dsv4_ag_overlap_short_kernels_cuda_gpu_kern_sum.csv`: f32 all-reduce remains the largest wait-inclusive row at `27.6%`, routed W13/W2 are `6.5%`/`3.5%`, and BF16 all-gather is `5.4%`.
- post-`32768` grouped-FP4 shared-memory short profile `/tmp/dsv4_shared32768_short.nsys-rep` and `/tmp/dsv4_shared32768_short_kernels_cuda_gpu_kern_sum.csv`: f32 all-reduce `19.3%`, f32 reduce-scatter `16.6%`, routed W13/W2 `8.9%`/`4.3%`, HC pre-norm `6.9%`, BF16 all-gather `4.8%`, score-route BF16 GEMM `4.7%`; use for composition only because nsys perturbed short-run TPOT to `30.64ms`.
- compact active expert plus compact launch rows microbench log `/tmp/dsv4_compact_launch_rows_bench.log`: for active `8` and rows/active `8/16/32`, reducing compact launch rows from total rows to per-expert rows moved W13 from about `0.109-0.112ms` to `0.082-0.087ms` and W2 from about `0.056-0.057ms` to `0.043-0.045ms`; rejected for runtime until the scheduler can drive launch dimensions without route metadata D2H.
- route-stat realistic compact launch rows log `/tmp/dsv4_compact_launch_rows_realistic_bench.log`: active `1/2` stayed flat at W13 about `0.0615ms` and W2 about `0.0328ms`; active `5` improved W13 `0.079858ms -> 0.063514ms` and W2 `0.045044ms -> 0.032787ms`. This makes problem sizes a tail optimization, not the main sub-25 lever.
- arbitrary sparse counts grouped FP4 logs `/tmp/dsv4_arbitrary_counts_grouped_fp4_bench.log` and `/tmp/dsv4_arbitrary_counts_grouped_fp4_bound_bench.log`: the bench now supports `--counts` plus selected compact expert pointers. Sparse active ids show compact active expert z-dimension plus row bound speeds W13/W2 by `1.39-1.83x`, while `--capacity-launch-rows` without active z compaction is flat. This corrects the earlier prefix-active microbench interpretation and keeps the runtime blocker focused on no-D2H launch sizing.
- rejected direct CUDA score-route microbench logs `/tmp/dsv4_score_route_direct_microbench.log` and `/tmp/dsv4_score_route_direct_bench_large.log`: direct projection beat cuBLAS for `seq_len>=8` in isolation with identical top-k indices on random fuzz, but runtime fixed bench changed generated-token hash.
- rejected direct CUDA score-route runtime logs `/tmp/dsv4_score_route_direct_e2e.log` and `/tmp/dsv4_score_route_direct_bench.log`: exact E2E `20/20`, aggregate steady TPOT avg `28.340ms`, but fixed-bench hash changed to `abc4a0a2160d7963`; not retained.
- post-revert direct CUDA score-route fixed bench log `/tmp/dsv4_score_route_direct_reverted_bench.log`: aggregate steady TPOT avg `28.477ms`, per-iteration `28.501ms`, `28.477ms`, `28.455ms`; hash restored to `6346f03343d75a65`.
- rejected shared split-overlap exact E2E log `/tmp/dsv4_shared_split_overlap_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected shared split-overlap fixed bench logs `/tmp/dsv4_shared_split_overlap_bench.log` and `/tmp/dsv4_shared_split_overlap_bench_repeat.log`: aggregate steady TPOT avg `26.625ms` and `27.145ms`; all hash `6346f03343d75a65`
- restored MoE all-gather/reduce-scatter shared-overlap fixed bench log `/tmp/dsv4_moe_ag_overlap_restored_bench.log`: aggregate steady TPOT avg `27.198ms`, per-iteration `27.195ms`, `27.203ms`, `27.198ms`; all hash `6346f03343d75a65`
- rejected hash token-id repeat exact E2E log `/tmp/dsv4_repeat_tokens_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected hash token-id repeat fixed bench log `/tmp/dsv4_repeat_tokens_bench.log`: aggregate steady TPOT avg `27.297ms`, per-iteration `27.285ms`, `27.333ms`, `27.273ms`; all hash `6346f03343d75a65`
- temporary current-stage trace log `/tmp/dsv4_trace_current_bench.log`: token hash stayed `6346f03343d75a65`; after excluding the first layer-0 MoE warmup outlier, approximate 43-layer sums were attention local `17.37ms`, MoE `13.77ms`, attention collective+post `3.29ms`, HC FFN pre `2.33ms`, HC attention pre `2.19ms`, FFN post `0.45ms`. Trace code was removed after collection.
- W2 TopK weight/reduce atomic microbench used a temporary standalone source, compiled on 5090 with `/usr/local/cuda-12.9/bin/nvcc -O3 -std=c++17 -arch=sm_120`; the source was deleted after rejection to avoid leaving a stale integration path.
- W2 TopK weight/reduce atomic microbench log `/tmp/dsv4_w2_weighted_reduce_bench.log`: for `seq_len={1,8,16,32}` and DSV4 `topk=6`, current deterministic reduce was `0.0023-0.0027ms`, atomic epilogue-shaped reduce was `0.0049-0.0061ms`, max abs diff stayed under `2.4e-7`; rejected before runtime integration.
- fresh retained-runtime validation logs `/tmp/dsv4_fresh_e2e_after_w2_reduce_doc.log` and `/tmp/dsv4_fresh_bench_after_w2_reduce_doc.json`: exact E2E `All 20 DeepSeek V4 exact cases passed`; fixed bench aggregate steady TPOT avg `28.480ms`, per-iteration `28.505ms`, `28.470ms`, `28.466ms`, all hash `6346f03343d75a65`.
- W13/SwiGLU layout microbench used a temporary standalone source, compiled on 5090 with `/usr/local/cuda-12.9/bin/nvcc -O3 -std=c++17 -arch=sm_120`; the source was deleted after rejection to avoid implying a retained layout path.
- W13/SwiGLU layout microbench logs `/tmp/dsv4_swiglu_layout_bench.log` and `/tmp/dsv4_swiglu_layout_bench_route_rows.log`: pair-interleaved `[up, gate]` input was byte-identical to separate gate/up buffers, but only rows `96` showed a visible standalone gain (`0.003286ms -> 0.002579ms`); rejected before runtime integration.
- grouped FP4 shared-memory probe logs `/tmp/dsv4_w13_shared_81920.log`, `/tmp/dsv4_w13_shared_65536.log`, `/tmp/dsv4_w13_shared_49152.log`, `/tmp/dsv4_w13_shared_32768.log`, `/tmp/dsv4_w13_shared_24576.log`, `/tmp/dsv4_w13_shared_16384.log`, and `/tmp/dsv4_w13_shared_32768_shapes.log`: `32768` bytes is bitwise-safe across tested W13/W2 sparse decode shapes, while `16384` bytes trips illegal memory access.
- grouped FP4 shared-memory runtime validation logs `/tmp/dsv4_grouped_shared32768_e2e.log`, `/tmp/dsv4_grouped_shared32768_bench.log`, and `/tmp/dsv4_grouped_shared32768_bench_repeat.log`: exact E2E `20/20`; fixed bench aggregate steady TPOT avg `28.481ms` then `28.694ms`; all generated-token hashes `6346f03343d75a65`.
- careful decode-mid profile calibration `/tmp/dsv4_profile_calibration_20260513_101406.json`: fixed bench aggregate steady TPOT avg `28.347ms`, p50 `28.122ms`, p95 `29.653ms`, all generated-token hashes `6346f03343d75a65`.
- careful decode-mid profile `/tmp/dsv4_profile_decode_mid.nsys-rep`, `/tmp/dsv4_profile_decode_mid.sqlite`, `/tmp/dsv4_profile_decode_mid_stats_cuda_gpu_kern_sum.csv`, and `/tmp/dsv4_profile_decode_mid_stats_cuda_api_sum.csv`: delayed `--duration=4` capture with `--cuda-event-trace=false`; profile-run TPOT avg `29.222ms`, hash `6346f03343d75a65`. Aggregate GPU share: FP8 dense/shared `19.8%`, f32 reduce-scatter wait-inclusive `18.6%`, HC/indexer `14.9%`, routed W13 `10.3%`, f32 all-reduce wait-inclusive `8.8%`, score router `7.5%`, attention local `5.5%`, routed W2 `5.0%`, MoE all-gather wait-inclusive `3.2%`, routed SwiGLU+quant `0.7%`.
- careful decode-mid sqlite drilldown: device-0 W13/W2 grouped launches usually use fixed `grid=(32,2,32), block=128`; `1727/4069` W13 and W2 launches were `<5us`, while heavy W13 launches occupied the `70us+` buckets and f32 reduce-scatter had `1721/4026` launches `>=120us`. Empty/near-empty grouped launches are therefore not the main remaining lever by themselves.
- current allocation counter rerun `/tmp/dsv4_cuda_api_counter_now.json` and `/tmp/dsv4_cuda_api_counter_now.err`: short `--output-len 64 --warmup 1 --iters 1 --seed 42` run reported steady TPOT avg `29.355ms`; `cuMemAllocAsync/cuMemFreeAsync/cuMemsetD8Async`, `cudaMallocAsync/cudaFreeAsync/cudaMemsetAsync`, and async H2D/D2H counters were all `0`, while `cudaLaunchKernel` was `2074256`. This is the application-visible check that overrides noisy nsys async-allocation attribution.
- rejected grouped FP4 launch-bounds=2 logs `/tmp/dsv4_w13_launch_bounds2_microbench.log`, `/tmp/dsv4_launch_bounds2_e2e.log`, and `/tmp/dsv4_launch_bounds2_bench.log`: exact E2E `20/20`; fixed bench aggregate steady TPOT avg `28.522ms`; all generated-token hashes `6346f03343d75a65`; not retained because it did not beat the retained `32768` shared-memory path.
- rejected naive grouped FP4 `block_M=16` logs `/tmp/dsv4_w13_block_m16_bench.log` and `/tmp/dsv4_w13_block_m16_large_rows_bench.log`: decode-like small rows sped up, but rows/active `32` failed fuzz because grouped transforms/wrappers still have hard-coded `32`-row assumptions; not retained.
- rejected parameterized grouped FP4 `block_M=16` logs `/tmp/dsv4_w13_block_m16_param_fuzz.log`, `/tmp/dsv4_grouped_block_m16_e2e.log`, `/tmp/dsv4_grouped_block_m16_bench.log`, and `/tmp/dsv4_grouped_block_m16_bench_repeat.log`: broad fuzz and exact E2E passed, token hash stayed `6346f03343d75a65`, but fixed bench regressed to aggregate steady TPOT avg `28.971ms` then `29.797ms`; local and 5090 were restored to grouped FP4 `block_M=32`.
- post-restore grouped FP4 fixed bench log `/tmp/dsv4_grouped_block_m16_restored_bench.log`: aggregate steady TPOT avg `28.736ms`, per-iteration `28.445ms`, `28.998ms`, `28.763ms`; all hash `6346f03343d75a65`.
- completion audit and cleanup: local `git diff --check`, `cargo fmt --check`, and `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed after documenting the sub-25 gap and deleting untracked rejected bench sources. The retained tool sources are `score_select_bench.cu`, `swiglu_quant_bench.cu`, and `w13_grouped_fp4_bench.cu`.
- vLLM/SGLang large-batch gap audit: source inspection confirmed the mature FP4 MoE throughput path combines static W13/W2 weight reorder, FP4 scale interleave, packed routed top-k, and problem-size-aware grouped backends. This supports keeping packed MoE layout as a separate bs>100 architecture project rather than mixing it into the current sub-25 latency patch.
- `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl`
- `nm -D /tmp/cuda_api_counter.so` confirmed base and `_ptsz` wrappers

The benchmark process still prints the existing NCCL communicator abort panic during shutdown after JSON output and scheduler exit. Track that as shutdown cleanup, not decode TPOT evidence.

## Follow-ups

- Fix NCCL communicator shutdown.
- Move DeepSeek V4 off the temporary direct runtime into the scheduler/executor shape used by the rest of the engine.
- Revisit CUDA graph capture after pointer stability is broad enough.
- Keep MoE active-expert/tile-list work separate from allocation scratch. After the decode-mid sqlite drilldown, do not expect empty-launch skipping alone to close the gap; empty W13/W2 launches are already mostly `<5us`. The MoE path needs either a real heavy W13/W2 scheduler/epilogue improvement or a synchronization-window reduction.
- For the small-BS latency track, avoid repeating the rejected HC routes: direct HC mixes changed the long token hash, and unified `GemmEx` was hash-stable but slower. The next HC attempt needs a batch-general backend that preserves cuBLAS-compatible accumulation where the long trace is sensitive.
- For bs>100 throughput, design a packed MoE layout document that covers W13/W2 weights, FP4 scales, dispatch row order, and W2 combine/finalize together before changing runtime weights.
