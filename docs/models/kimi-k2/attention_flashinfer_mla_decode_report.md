# Attention FlashInfer MLA Decode Report

> **TL;DR:** `decode.attention.flashinfer_mla_decode` calls FlashInfer `BatchDecodeWithPagedKVCacheDispatchedMLA` through `kimi_flashinfer_batch_decode_mla_rt`. At the anchor `bs=8/rank,ctx=1` it costs `624.6us/step` (`10.24us/call`) across 61 attention layers, but it is the long-context cliff: `ctx=8192` costs `103.50ms/step` (`1.697ms/call`) and reaches about `2.85 TB/s` payload-equivalent bandwidth (`~59%` of the H20 HBM roofline model). A selected NCU metrics pass for `ctx=8192` does not confirm HBM saturation: it launches only `32` CTAs (`0.41` waves/SM), uses `254` registers/thread, reaches `12.5%` active warps, and reports `0.87%` DRAM throughput / `22.74%` memory throughput. The current Kimi wrapper passes no FlashInfer split-K temporary buffers, so FlashInfer forces `partition_kv=false`; the next real experiment is to wire a planned `partition_kv` path or prove another backend is better, not to tune the current fixed-grid launch blindly.
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

The next completed NCU run should isolate:

| Question | Why it matters |
|---|---|
| Which FlashInfer MLA kernel/backend is dispatched? | KernelWiki records backend selection regressions on Hopper; the source call alone is not enough. |
| DRAM read throughput and L2 hit rate at `ctx=8192` | The bench-derived payload is `~2.85 TB/s`, but NCU must confirm actual bytes and cache behavior. |
| Wave count, occupancy, and scheduler stalls at `ctx=1` | Short context is only `10.24us/call`; it may be launch/control limited rather than bandwidth limited. |
| Split-K / partition-KV behavior | Current wrapper passes `partition_kv=false`; NCU must show whether long-context work fills H20 well enough without partitioning. |

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

- Long-context arithmetic intensity from the bench model is about `3.78 flop/byte`, below the H20 ridge point recorded in the master table (`30.83 flop/byte`), so long context is memory-bound.
- `ctx=1` has too little KV work to saturate bandwidth; the row is dominated by FlashInfer launch/control and metadata overhead.
- The event-level roofline row is a payload model, not a proof that physical DRAM is saturated. The selected NCU metrics show `32` CTAs and low DRAM throughput, so the next optimization should test whether FlashInfer's partitioned plan raises parallelism before assuming the only remaining option is reducing bytes.

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
| FP8 KV cache | Not attempted. KernelWiki/vLLM direction is promising for bandwidth, but it changes cache dtype and correctness envelope. | Candidate only with an explicit accuracy gate and cache-layout plan. |
| FlashInfer backend/version swap | Not attempted. KernelWiki shows Hopper backend selection can matter. | Candidate only after identifying the exact dispatched kernel/backend in NCU. |

## Final Conclusion

Keep the current FlashInfer MLA decode path as the baseline for now. This row is not stopped as an optimization target: it is the highest-priority attention row for long context, and the first evidence-backed code experiment is a planned `partition_kv` path that increases CTAs for long context.

Adoption bar for any future change:

| Direction | Required proof |
|---|---|
| Backend/plan selection | NCU identifies the current backend and shows a replacement improves `ctx=8192` H20 latency by `>3%` without hurting `ctx=1`. |
| `partition_kv` / split-K | Wire all FlashInfer plan metadata and temp buffers; full bench improves at `ctx>=1024` and short context does not regress materially. |
| FP8 KV cache | Token/logit correctness gate plus H20 bench improvement from lower KV bytes; report must record cache dtype/layout changes. |

No `opt(...)` commit is appropriate from this report alone. Reopen for code once the planned `partition_kv` bench path or another backend has a reproducible H20 speedup.
