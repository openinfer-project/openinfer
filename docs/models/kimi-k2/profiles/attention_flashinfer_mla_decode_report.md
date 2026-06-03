# Attention FlashInfer MLA Decode Report

> **TL;DR:** `decode.attention.flashinfer_mla_decode` calls FlashInfer `BatchDecodeWithPagedKVCacheDispatchedMLA` through `kimi_flashinfer_batch_decode_mla_rt`. At the anchor `bs=8/rank,ctx=1` it costs `624.6us/step` (`10.24us/call`) across 61 attention layers, but it is the long-context cliff: `ctx=8192` costs `103.50ms/step` (`1.697ms/call`). Baseline NCU for `ctx=8192` shows only `32` CTAs (`0.41` waves/SM), `254` registers/thread, and no DRAM saturation. A bench-scoped planned `partition_kv` probe with `--mla-decode-partition-pages 128` improved H20 synthetic `ctx=4096/8192` from `51.98/103.34ms` to `26.66/52.53ms` per decode step (`~1.95x/1.97x`) by raising the decode launch to `128` CTAs plus a merge kernel. Under the current production decode cap (`2048` tokens/request), graph-safe fixed-grid p32 (`max_pages_per_request=128`, `block_valid_mask`) improved H20 `ctx=1024/2048` from `13.37/26.24ms` to `7.31/13.85ms` with synthetic output-equivalence passing (`max_abs=0.001953`). Production PPLX wiring was rejected: global bs64 runtime gates failed before the partition branch could be validated, including `kv_len=513` long-prefill `NaN` and prompt_len1 decode `-inf` failures, and even the minimal global bs64 `kv_len=2` trace failed under the WIP. The unused partition kernel/FFI/bench code has been removed from the live codebase.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `sources/prs/sglang/PR-3987.md` (`pr-sglang-3987`) and `sources/prs/sglang/PR-4012.md` (`pr-sglang-4012`) | SGLang added a fast decode plan for FlashInfer MLA to avoid CPU/GPU indptr transfer and CUDA Graph replay hangs. | Plan construction and graph-safe metadata are part of the performance surface. Pegainfer already uses CUDA graph decode, so changes here must preserve graph replay. |
| `sources/prs/vllm/PR-21078.md` (`pr-vllm-21078`) | vLLM integrated FlashInfer MLA decode as a dedicated backend and benchmarked it at serving scale. | Confirms FlashInfer MLA is a strong baseline, not a toy fallback. Any custom path must beat this baseline on H20 and keep paged-cache semantics. |
| `sources/prs/flashinfer/PR-2530.md` (`pr-flashinfer-2530`) | FlashInfer's auto backend choice for `BatchDecodeWithPagedKVCacheWrapper` regressed on Hopper, and the PR chose FA2 for non-FP8 workloads. | Backend selection matters on SM90/H20. Before changing code, profile the exact dispatched MLA kernel and verify which backend variant is running. |
| `sources/prs/vllm/PR-34597.md` (`pr-vllm-34597`) | FP8 KV cache in MLA decode reduces KV bandwidth by dequantizing on load. | Kimi's current MLA cache path is BF16. FP8 KV is the main bandwidth-saving direction if accuracy and cache format changes are acceptable. |
| `sources/prs/sglang/PR-18442.md` (`pr-sglang-18442`) | FA4 SM90 paged-KV decode support was added because decode requires paged KV support on Hopper. | Directional only: a future FlashInfer upgrade/backend experiment should compare paged MLA decode kernels on H20, but this repo currently calls the FlashInfer submodule wrapper. |
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | For non-persistent kernels, ensure grid size is much larger than SM count; split long reductions or use persistent scheduling when CTAs underfill the device. | Directly applies to the `ctx=8192` NCU pass: `32` CTAs on `78` H20 SMs leaves most SMs idle. |

Practical conclusion: for `ctx=1`, the row is launch/control overhead. For `ctx>=1024`, the bench payload model is bandwidth-shaped, but the first NCU metrics pass says the current launch is under-filled before DRAM is saturated. The long-context fork is therefore: first increase work partitioning or switch backend, then recheck whether the new path is truly memory-bound.

## NCU Conclusion

Fresh production NCU started working intermittently for this row. Earlier attempts failed because `ncu --version` timed out; on 2026-06-01 `/usr/local/cuda-12.9/bin/ncu --version` returned `Version 2025.2.0.0`.

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# 2026-06-01: returned Version 2025.2.0.0 once; h20-100 filesystem access later timed out while retrieving the report.
```

The first full NCU collection completed on `h20-100` with this kernel:

```text
flashinfer::BatchDecodeWithPagedKVCacheKernelMLA<
  2, 16, 2, 32, 8, 1, 2,
  flashinfer::DefaultAttention<false, false, false, false>,
  flashinfer::BatchDecodeParamsMLA<bf16, bf16, bf16, int>
>
```

Full NCU collection completed and wrote this remote artifact:

```text
/root/develop/xingming/pegainfer/profile/kimi-flashinfer-mla-decode-ctx8192-h20/reports/ctx8192_full.ncu-rep
```

The remote `.ncu-rep` still needs to be retrieved and parsed; repeated listing/copy attempts against the remote profile directory timed out. A source-counter collection was attempted with `--set source --section SourceCounters`, but it did not finish after several minutes and was killed from the local SSH side.

To avoid waiting on the remote artifact, a narrower stdout metrics pass was collected and parsed locally under `profile/kimi-flashinfer-mla-decode-ctx8192-h20/analysis/ctx8192_metrics_summary.json`:

| Metric | Value |
|---|---:|
| Kernel | `BatchDecodeWithPagedKVCacheKernelMLA<2,16,2,32,8,1,2,...>` |
| Block size | `(32, 8, 1)` = `256` threads |
| Grid size | `(8, 4, 1)` = `32` CTAs |
| H20 SMs implied by waves | `78` |
| `launch__waves_per_multiprocessor` | `0.41` |
| `launch__registers_per_thread` | `254` |
| Dynamic shared memory / block | `22,528 B` |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | `12.50%` |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | `70.23%` |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | `28.77%` |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | `22.74%` |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | `0.87%` |
| `dram__bytes_read.sum.per_second` | `41.75 GB/s` |
| `lts__t_sector_hit_rate.pct` | `48.22%` |

Interpretation:

- The selected NCU metrics do not support the claim that this exact launch is already at the H20 HBM limit. The bench payload model reports `~2.85 TB/s`, but NCU's raw DRAM metric is far lower and the grid is smaller than the SM count.
- The first visible bottleneck is launch geometry: `padded_batch_size=8`, `gdy=ceil(64 / (8 * 2))=4`, so FlashInfer launches `8 * 4 = 32` CTAs.
- Register pressure is also severe: `254` registers/thread limits active blocks and active warps.
- The full `.ncu-rep` and a shorter source-counter run are still needed for stall attribution, but a split-K / partition-KV experiment is now justified by evidence.

The partitioned probe has its own selected NCU artifact at `target/kernel_reports/kimi-k2/kimi_mla_p128_ncu_ctx8192_metrics.csv`. For `ctx=8192`, `partition_pages=128`, the first two profiled kernels are:

| Kernel | Grid | Block | Waves/SM | Regs/thread | Dynamic smem | SM throughput | Compute-memory throughput | DRAM throughput | DRAM read | L2 hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `BatchDecodeWithPagedKVCacheKernelMLA<...>` | `(32,4,1)` = `128` CTAs | `(32,8,1)` | `1.64` | `254` | `22,528 B` | `57.29%` | `45.32%` | `3.44%` | `165.4 GB/s` | `35.51%` |
| `PersistentVariableLengthMergeStatesKernel<...>` | `(546,1,1)` = `546` CTAs | `(32,4,1)` | `0.70` | `48` | `16,896 B` | `39.85%` | `36.89%` | `7.29%` | `355.2 GB/s` | `37.34%` |

Interpretation:

- The partitioned path validates the small-grid diagnosis: the decode kernel moves from `32` CTAs / `0.41` waves/SM to `128` CTAs / `1.64` waves/SM.
- The new path is not a pure drop-in single-kernel improvement. It adds `PersistentVariableLengthMergeStatesKernel`, so the bench payload roofline model reports impossible `>100%` HBM percentages. For partitioned rows, use latency and NCU geometry/counters rather than the old payload `%peak`.
- Even after partitioning, NCU still does not show physical HBM saturation. The next bottleneck is likely register pressure in the decode kernel plus merge overhead/tile scheduling, not raw DRAM bandwidth.

The next completed NCU run should isolate:

| Question | Why it matters |
|---|---|
| Which FlashInfer MLA kernel/backend is dispatched? | KernelWiki records backend selection regressions on Hopper; the source call alone is not enough. |
| DRAM read throughput and L2 hit rate at `ctx=8192` | The bench-derived payload is `~2.85 TB/s`, but NCU must confirm actual bytes and cache behavior. |
| Wave count, occupancy, and scheduler stalls at `ctx=1` | Short context is only `10.24us/call`; it may be launch/control limited rather than bandwidth limited. |
| Split-K / partition-KV behavior | Bench path now proves higher parallelism; production path still needs correctness, graph-safe metadata ownership, and merge overhead analysis. |

## Bench Evidence

Runtime path:

| Item | Value |
|---|---|
| Rust wrapper | `pegainfer-kernels::ops::kimi_k2::mla::kimi_flashinfer_batch_decode_mla` |
| CUDA entry | `kimi_flashinfer_batch_decode_mla_cuda` in `pegainfer-kernels/csrc/kimi_k2/kimi_mla.cu` |
| FlashInfer call | `BatchDecodeWithPagedKVCacheDispatchedMLA<HEAD_DIM_CKV=512, HEAD_DIM_KPE=64>` |
| Current flags | `partition_kv=false`, `enable_pdl=false`, no temporary split-K buffers |
| Shape | `q_abs_nope=[8,64,512]`, `q_pe=[8,64,64]`, paged BF16 compressed KV, output `[8,64,512]` |
| Calls per decode step | `61` attention layers |

H20 artifact: `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json`.

`active_rows` does not change this attention row because the bench uses the fixed decode arena (`arena_rows=8`) for attention/final rows; the MoE rows use active rows separately.

| Per-rank active rows | ctx | Step latency | Per call | Payload GB/s | TFLOP/s | H20 HBM pct |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | `0.640 ms` | `10.49us` | `206.1` | `0.106` | `4.3%` |
| 1 | 8192 | `103.34 ms` | `1.694ms` | `2853.2` | `5.388` | `59.4%` |
| 2 | 8192 | `103.52 ms` | `1.697ms` | `2848.2` | `5.378` | `59.3%` |
| 4 | 8192 | `103.74 ms` | `1.701ms` | `2842.2` | `5.367` | `59.2%` |
| 8 | 1 | `0.625 ms` | `10.24us` | `211.2` | `0.109` | `4.4%` |
| 8 | 128 | `2.132 ms` | `34.96us` | `2204.6` | `4.079` | `45.9%` |
| 8 | 1024 | `13.400 ms` | `219.67us` | `2756.7` | `5.194` | `57.4%` |
| 8 | 4096 | `52.230 ms` | `856.24us` | `2823.4` | `5.330` | `58.8%` |
| 8 | 8192 | `103.50 ms` | `1.697ms` | `2848.8` | `5.379` | `59.4%` |

Roofline interpretation:

- Long-context arithmetic intensity from the bench model is about `1.89 flop/byte`, below the H20 ridge point recorded in the master table (`30.83 flop/byte`), so long context is memory-shaped by the payload model.
- `ctx=1` has too little KV work to saturate bandwidth; the row is dominated by FlashInfer launch/control and metadata overhead.
- The event-level roofline row is a payload model, not a proof that physical DRAM is saturated. The selected NCU metrics show `32` CTAs and low DRAM throughput, so the next optimization should test whether FlashInfer's partitioned plan raises parallelism before assuming the only remaining option is reducing bytes.

Bench-scoped partition probe artifacts:

| Artifact | Meaning |
|---|---|
| `target/kernel_reports/kimi-k2/kimi_mla_baseline_h20.json` | H20 baseline, per-rank `active_rows=8`, `ctx=1024,4096,8192`, `iters=8`. |
| `target/kernel_reports/kimi-k2/kimi_mla_p256_h20.json` | Same sweep with `--mla-decode-partition-pages 256`. |
| `target/kernel_reports/kimi-k2/kimi_mla_p128_h20.json` | Same sweep with `--mla-decode-partition-pages 128`. |
| `target/kernel_reports/kimi-k2/kimi_mla_p64_h20.json` | Same sweep with `--mla-decode-partition-pages 64`. |
| `target/kernel_reports/kimi-k2/kimi_mla_p128_confirm_h20.json` | H20 confirm run for p128 at `ctx=4096,8192`, `iters=16`. |
| `target/kernel_reports/kimi-k2/kimi_mla_p128_ncu_ctx8192_metrics.csv` | Selected NCU stdout metrics for p128 at `ctx=8192`. |
| `target/kernel_reports/kimi-k2/kimi_mla_p128_check4096_h20.json` | H20 synthetic output-equivalence check for p128 at `ctx=4096`. |
| `target/kernel_reports/kimi-k2/kimi_mla_p128_check8192_h20.json` | H20 synthetic output-equivalence check for p128 at `ctx=8192`. |
| `target/kernel_reports/kimi-k2/kimi_mla_p32_fixed_ctx2048_h20.json` | H20 production-cap p32 with fixed graph-size metadata: `max_pages_per_request=128`, `block_valid_mask`, `iters=8`. |

Production decode capacity caveat:

- The current runner arena in `pegainfer-kimi-k2/src/runner/worker.rs` fixes `KIMI_DECODE_PAGES_PER_REQUEST=128` and `KIMI_DECODE_PAGE_SIZE=16`, so production decode currently represents up to `2048` cached tokens per request.
- The `ctx=4096/8192` partition numbers above are valid H20 kernel-bench evidence, but they are synthetic long-context shapes outside the current runner arena.
- The production-cap H20 sweep now proves a `ctx<=2048` event-timing win with smaller chunks. `partition_pages=32` is the best current candidate because it improves both `ctx=1024` and `ctx=2048`; `p64` is slightly better at `ctx=2048` alone but regresses `ctx=1024`.
- The graph-safe fixed-grid p32 bench uses `padded_batch_size = batch * ceil(128 / 32)` plus `block_valid_mask`, matching FlashInfer's CUDA-graph plan shape. It is only `~0.3%` slower than real-padded p32 at `ctx=2048`.
- A production `partition_kv` change was attempted and then removed from the runner WIP. The bench artifacts remain useful evidence, but the unused kernel/FFI/bench code has been removed until the base TP1/DP8/PPLX runtime trace gate is healthy at global bs64.

H20 latency comparison:

| ctx | Baseline step | p256 step / speedup | p128 step / speedup | p64 step / speedup |
|---:|---:|---:|---:|---:|
| 1024 | `13.379 ms` | `13.567 ms` / `0.99x` | `13.566 ms` / `0.99x` | `13.567 ms` / `0.99x` |
| 4096 | `51.975 ms` | `52.226 ms` / `1.00x` | `26.631 ms` / `1.95x` | `26.807 ms` / `1.94x` |
| 8192 | `103.269 ms` | `52.355 ms` / `1.97x` | `52.517 ms` / `1.97x` | `52.851 ms` / `1.95x` |

The `iters=16` confirmation for p128 reproduced the result: baseline `51.983/103.336ms` at `ctx=4096/8192`, p128 `26.664/52.535ms`.

H20 production-cap latency comparison:

| ctx | Baseline step | p64 real / speedup | p32 real / speedup | p32 fixed-grid / speedup | p16 real / speedup |
|---:|---:|---:|---:|---:|---:|
| 1024 | `13.371 ms` | `13.572 ms` / `0.99x` | `7.284 ms` / `1.84x` | `7.309 ms` / `1.83x` | `7.424 ms` / `1.80x` |
| 2048 | `26.236 ms` | `13.747 ms` / `1.91x` | `13.837 ms` / `1.90x` | `13.854 ms` / `1.89x` | `14.123 ms` / `1.86x` |

This makes `partition_pages=32` the current production-cap candidate: it halves valid long-context latency at `ctx=2048` and also helps `ctx=1024`, while `p64` only wins the `ctx=2048` micro-point by `~0.09ms`. The fixed-grid variant keeps almost all of the p32 gain while preserving the launch shape needed by CUDA Graph replay.

Synthetic output-equivalence checks compare the non-partition and p128 partition outputs on deterministic BF16 q/cache data:

| ctx | Compared elems | Max abs diff | Mean abs diff |
|---:|---:|---:|---:|
| 4096 | `262144` | `0.001953` | `0.000170` |
| 8192 | `262144` | `0.001953` | `0.000101` |

Production-cap output-equivalence checks:

| Partition pages | ctx | Compared elems | Max abs diff | Mean abs diff |
|---:|---:|---:|---:|---:|
| 64 | 1024 | `262144` | `0` | `0` |
| 64 | 2048 | `262144` | `0.001953` | `0.000161` |
| 32 | 1024 | `262144` | `0.001953` | `0.000360` |
| 32 | 2048 | `262144` | `0.001953` | `0.000245` |
| 16 | 1024 | `262144` | `0.001953` | `0.000222` |
| 16 | 2048 | `262144` | `0.001953` | `0.000250` |

FlashInfer source notes:

| Source | Observation |
|---|---|
| `include/flashinfer/attention/decode.cuh` | `BatchDecodeWithPagedKVCacheDispatchedMLA` sets `params.partition_kv=false` whenever `tmp_v == nullptr`; otherwise it writes partial outputs to `tmp_v/tmp_s` and calls `VariableLengthMergeStates`. |
| `include/flashinfer/attention/scheduler.cuh` | `BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA` computes `max_grid_size`, `kv_chunk_size`, `new_batch_size`, `request_indices`, `kv_tile_indices`, `o_indptr`, and temp buffer offsets. |
| `csrc/batch_decode_mla_run.cu` | The FlashInfer TVM-FFI path wires the plan fields into `params` and only enables split-K when `plan_info.split_kv` is true. |
| `pegainfer-kernels/csrc/kimi_k2/kimi_mla.cu` | The current Kimi wrapper sets `params.padded_batch_size=batch_size`, `params.o_indptr=nullptr`, and passes `tmp_v/tmp_s=nullptr`, so it cannot use the FlashInfer partition-KV branch. |

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current FlashInfer MLA decode path | Baseline recorded across `ctx=1..8192`; long-context `ctx=8192` is `103.50ms/step`, `2.85 TB/s` payload-equivalent. | Current baseline. |
| Fresh production NCU full report | Full `--set full` capture completed and wrote `ctx8192_full.ncu-rep` remotely, but retrieval/parsing still times out. SourceCounters run did not finish after several minutes. | Keep trying narrower parse/source runs; do not claim final stall attribution yet. |
| Selected NCU stdout metrics | Parsed raw metrics show `32` CTAs, `0.41` waves/SM, `254` registers/thread, `12.5%` active warps, `0.87%` DRAM throughput. | Treat the current launch as low-grid/register-limited until the full report proves otherwise. |
| Flip `partition_kv` / split-K | Source inspection complete. It is not a one-flag change: the Kimi wrapper must provide planned `request_indices`, `kv_tile_indices`, `o_indptr`, `kv_chunk_size`, `tmp_v`, `tmp_s`, and possibly `block_valid_mask`. | Next code experiment should wire a FlashInfer-style plan for the bench path first, then measure `ctx=1024..8192`. |
| Bench-scoped `partition_kv` probe | Historical local WIP behind `--mla-decode-partition-pages`; default path unchanged. H20 per-rank `active_rows=8,iters=8`: p128 improves synthetic `ctx=4096/8192` from `51.975/103.269ms` to `26.631/52.517ms` per step; p256 only helps at `ctx=8192`; p64 is slightly slower than p128. Selected NCU shows decode grid `32x4=128` CTAs plus a merge kernel. Synthetic H20 output-equivalence vs baseline passes at ctx4096/8192 with max abs diff `0.001953`. Production-cap H20 sweep identifies p32 as the best current candidate: real-padded `ctx=1024/2048` improves from `13.371/26.236ms` to `7.284/13.837ms`; graph-safe fixed-grid p32 is `7.309/13.854ms`, also with max abs diff `0.001953`. | Strong candidate, not adopted: production validation failed first, so the unused kernel/FFI/bench code was removed from live code. |
| Production PPLX forward WIP | Fixed-grid p32 arena metadata/temp buffers and a `.partitioned_actual` trace marker were wired into PPLX non-graph decode, then tested on H20. Compile passed locally and on H20, but runtime gates failed before validating the partition branch. | Rejected and removed from the runner WIP. Do not commit this path. |
| Global bs64 runtime trace | Correct command shape is `kimi_kernel_report trace --batch-size 64 --kv-len 513 --tp-world 1 --dp-world 8 --ep-backend pplx`, where `--batch-size` is global request count. H20 results: original long-prefill trace failed on rank7 with non-finite top logit `NaN`; a trace-only decode-growth workaround failed at prompt_len1 decode with non-finite top logit `-inf`; the minimal global bs64 `kv_len=2` PPLX trace also failed with prompt_len1 decode `-inf`. | Failed gate. This is not an SSH artifact and not a partition-kernel win; stop production adoption and move to the next kernel. |
| FP8 KV cache | Not attempted. KernelWiki/vLLM direction is promising for bandwidth, but it changes cache dtype and correctness envelope. | Candidate only with an explicit accuracy gate and cache-layout plan. |
| FlashInfer backend/version swap | Not attempted. KernelWiki shows Hopper backend selection can matter. | Candidate only after identifying the exact dispatched kernel/backend in NCU. |

## Final Conclusion

Keep the current FlashInfer MLA decode path as the accepted production baseline for now. The bench-scoped `partition_kv` path had real synthetic long-context and production-cap microbench speedup signals, but production adoption is stopped for this pass because the TP1/DP8/PPLX runtime gate fails before the partition branch can be validated. The attempted runner wiring and unused probe kernel code were removed from the WIP and must not be committed as an optimization.

Adoption bar for any future change:

| Direction | Required proof |
|---|---|
| Backend/plan selection | NCU identifies the current backend and shows a replacement improves `ctx=8192` H20 latency by `>3%` without hurting `ctx=1`. |
| `partition_kv` / split-K | Reopen only after the base TP1/DP8/PPLX global-bs64 runtime trace has a passing token gate. Then reintroduce fixed-grid p32 metadata and verify full decode/token output plus latency at `ctx=1024/2048`. |
| FP8 KV cache | Token/logit correctness gate plus H20 bench improvement from lower KV bytes; report must record cache dtype/layout changes. |

No `opt(...)` commit is appropriate from this report alone. The next kernel should proceed while this row stays on the accepted baseline.
