# DeepSeek V4 Decode Performance

**Created**: 2026-05-12
**Status**: complete

## TL;DR

This document consolidates the DeepSeek V4 decode work that moved fixed long decode from the `~108-113ms/token` band to the current `35.253ms/token` PR validation, with prior same-code fast runs at `32.75-33.90ms/token`. The retained changes are grouped MoE pointer caching, rank-worker placement, removal of hot temporary zero-fill, rank-owned decode scratch, caller-owned grouped FP4 workspace, and benchmark/counter instrumentation. Exact E2E remains `20/20`, and the fixed bench token hash remains `6346f03343d75a65`.

The retained team lessons are more important than the discarded attempt logs: compare identical token traces, separate NCCL wait from transfer, treat capacity and logical length separately, keep MoE semantic zero on device, and prove allocation cleanup with application-visible CUDA API counters rather than nsys attribution alone.

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
| Shared expert | `SharedExpertScratch` | Fixed-shape gate/up/activation/out storage. |
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

## Rejected Patterns

These are worth remembering because they looked plausible:

| Attempt | Result | Lesson |
| --- | --- | --- |
| Fuse final HC head plus RMSNorm | Exact-safe but regressed TPOT | Saving small launches can lose to worse reduction/kernel shape. |
| Reuse deterministic window top-k across layers | Exact-safe, no stable long-bench win | Launch-count reduction alone is weak evidence. |
| Fuse KV RoPE plus no-PE quant | Exact-safe, regressed short decode | Combining tiny kernels can hurt scheduling/occupancy. |
| Hand-written decode HC mixes kernel | Exact-safe, slower than cuBLAS path | cuBLAS small GEMV remained better on this shape. |
| Isolated final logits scratch | Correct but noisy/regressive in repeated runs | Isolated storage movement near sampling boundary did not address the dominant per-layer allocation/skew structure. |
| Host-sized active-tile count for grouped MoE | Not used | Pulling active counts D2H would reintroduce hot-path synchronization. |

## Profiling And Benchmark Rules

### Token trace first

Always compare generated-token hashes before comparing TPOT. DeepSeek V4 routing and expert balance depend on token sequence. The bench JSON now records per-iteration timing and generated-token trace.

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

The counter exports base and `_ptsz` wrappers separately for `cuMemAllocAsync`, `cuMemFreeAsync`, and `cuMemsetD8Async`. Do not share one stored real function pointer across base and `_ptsz` variants.

## Remote Workflow Notes

Remote test syncs should use touched-file `rsync -azR`. A full repository rsync with delete/excludes stalled for about 10 minutes during this work. Also, `cargo check` does not rebuild already-built release binaries; rebuild `deepseek_v4_e2e` and `bench_serving` before trusting remote validation.

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
- `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl`
- `nm -D /tmp/cuda_api_counter.so` confirmed base and `_ptsz` wrappers

The benchmark process still prints the existing NCCL communicator abort panic during shutdown after JSON output and scheduler exit. Track that as shutdown cleanup, not decode TPOT evidence.

## Follow-ups

- Fix NCCL communicator shutdown.
- Move DeepSeek V4 off the temporary direct runtime into the scheduler/executor shape used by the rest of the engine.
- Revisit CUDA graph capture after pointer stability is broad enough.
- Keep MoE active-expert/tile-list work separate from allocation scratch; the next MoE win is likely reducing empty CTA/kernel work, not more host allocation cleanup.
