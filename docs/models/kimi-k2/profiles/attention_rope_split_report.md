# Attention RoPE Split Report

> **TL;DR:** `decode.attention.rope_split` is the Kimi MLA decode helper `rope_split_decode_kernel`: it splits `q_proj` into `q_nope` / `q_pe`, applies RoPE to `q_pe`, and produces `append_kpe` for the MLA KV cache. At TP1 PPLX `bs=8,ctx=1`, H20 event timing is `441.8us` per 61-layer step or `7.24us/call`; selected NCU confirms it is not HBM- or compute-bound (`10.51%` SM, `1.27%` DRAM, `61.96GB/s`, `77.03%` no-eligible, `0.62` waves/SM). Stop standalone tuning and only revisit as a launch-removing fusion around MLA cache prep.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Low SM utilization can come from tail effect, load imbalance, static scheduling, or a grid too small for the GPU; persistent scheduling only helps when there is enough work to reschedule. | Selected NCU matches this pattern: the rule engine reports only `0.6` full waves across SMs, and scheduler no-eligible is `77.03%`. |
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | Memory-bound diagnosis needs measured high DRAM throughput; low arithmetic intensity alone is not enough. | The bench reports only `~54GB/s` payload-equivalent throughput, far from H20 HBM peak, so a standalone bandwidth rewrite is not evidence-backed. |
| `wiki/patterns/tail-effect.md` (`pattern-tail-effect`) | Moderate tile counts can lose time to wave quantization when the last wave is underfilled. | The selected NCU report gives `384` CTAs and `0.62` waves/SM under the occupancy model; this is wave-quantized control work, not a saturated HBM row. |
| `sources/prs/flashinfer/PR-3014.md` (`pr-flashinfer-3014`) | Small-batch decode helper kernels can benefit from removing helper overhead, but the PR is MoE-helper specific. | Directional only: the useful direction is launch removal or fusion, not retuning this helper in isolation. |

Practical conclusion: this is an elementwise MLA preparation helper with tiny per-call payload. The current evidence does not support a standalone rewrite. The plausible future work is to remove the launch by combining `append_kpe` handling with nearby MLA cache prep, while preserving the `q_nope` and `q_pe` consumers.

## NCU Conclusion

Selected H20 NCU was collected in `profile/kimi-attention-rope-split-h20/` with the bench-scoped target binary:

```bash
/usr/local/cuda/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section SchedulerStats --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --launch-skip 3 --launch-count 1 \
  -k regex:rope_split_decode_kernel \
  -o /dev/shm/kimi-rope-split-ncu/reports/rope_split_selected \
  --force-overwrite /dev/shm/pegainfer-kimi-partition-target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.attention.rope_split \
  --out /dev/shm/kimi-rope-split-ncu/rope_split_ncu.json
```

Key counters from `analysis/rope_split_details.csv`:

| Metric | Value |
|---|---:|
| NCU kernel duration | `3.26us` |
| Grid / block | `384` CTAs / `256` threads |
| Waves per SM | `0.62` |
| Registers/thread | `22` |
| Dynamic shared memory/block | `0 B` |
| Compute throughput | `10.51%` |
| DRAM throughput | `1.27%` |
| Memory throughput | `61.96 GB/s` |
| L1 / L2 hit rate | `55.71%` / `58.94%` |
| Achieved occupancy | `48.67%` |
| No eligible | `77.03%` |
| Eligible / active warps per scheduler | `0.64` / `8.53` |

The NCU rule engine reports that the grid is too small to fill H20 and only forms `0.6` full waves across SMs. It also estimates a local `77.03%` issue-slot utilization opportunity because most cycles have no eligible warp. This does not mean a standalone rewrite is automatically worthwhile: event timing is `7.24us/call`, and the row contributes `441.8us/step` before launch removal. The realistic path is to delete or absorb this helper launch, not to chase HBM or tensor throughput.

## Bench Evidence

Runtime path:

| Item | Value |
|---|---|
| Runtime call | `kimi_mla_rope_split_decode_rt` in `pegainfer-kimi-k2/src/runner/worker/forward.rs` |
| Bench op | `kimi_mla_rope_split_decode_rt` / `decode.attention.rope_split` |
| CUDA entry | `kimi_mla_rope_split_decode_cuda` in `pegainfer-kernels/csrc/kimi_k2/kimi_mla.cu` |
| Kernel | `rope_split_decode_kernel` |
| Launch at target shape | `384` CTAs x `256` threads for `batch_size=8, local_heads=64, q_head_dim=192` |
| Calls per decode step | `61` |

The kernel computes:

| Tensor | Shape / dtype |
|---|---|
| `q_proj` input | `[8,64,192]`, BF16 |
| `k_rope` input | `[8,64]`, BF16 |
| `cos/sin` cache | current-position RoPE cache, BF16 |
| `positions` | `[8]`, I32 |
| `q_nope` output | `[8,64,128]`, BF16 |
| `q_pe` output | `[8,64,64]`, BF16 |
| `append_kpe` output | `[8,64]`, BF16 |

H20 TP1 PPLX bench evidence from `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json`:

| active rows | ctx | Step latency | Per call | TFLOP/s | Payload GB/s |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | `441.76us` | `7.24us` | `0.027` | `54.44` |
| 8 | 128 | `421.33us` | `6.91us` | `0.029` | `57.08` |
| 8 | 1024 | `437.19us` | `7.17us` | `0.027` | `55.01` |
| 8 | 4096 | `543.21us` | `8.91us` | `0.022` | `44.28` |
| 8 | 8192 | `537.84us` | `8.82us` | `0.022` | `44.72` |

The row does not depend on `ctx_len` except for the position used to index the RoPE cache, so the ctx variation above should be treated as cache/noise sensitivity. Selected NCU confirms the target `ctx=1` row is far below both H20 HBM and SM limits. That is a control/elementwise signature, not a saturated memory kernel.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current `rope_split_decode_kernel` | `7.24us/call`, `441.8us/step` at `bs=8,ctx=1`; selected NCU: `10.51%` SM, `1.27%` DRAM, `77.03%` no-eligible. | Current baseline. |
| Standalone bandwidth rewrite | Not attempted. NCU shows only `61.96GB/s` memory throughput and `1.27%` DRAM, so this is not a bandwidth ceiling. | Reject. |
| More CTA parallelism | Not attempted. The current target shape already launches `384` CTAs, and NCU's small-grid finding is about insufficient useful work per launch rather than missing a simple split dimension. | Reject without a launch-removing design. |
| Fuse `append_kpe` with paged KV append or nearby MLA prep | Not implemented. This could remove a launch or an intermediate write/read if it preserves `q_nope`, `q_pe`, and cache append semantics. | Future-only direction with full TP1 PPLX bench and correctness gate. |

## Final Conclusion

Keep the current `rope_split_decode_kernel` and stop standalone `decode.attention.rope_split` tuning. Keep the master row as `control/elementwise`: the payload is too small to use H20 bandwidth, and selected NCU confirms low SM/DRAM utilization plus scheduler no-eligible cycles.

Reopen this row only for a fusion that removes a launch around MLA cache prep, with a realistic path to `>3%` full decode improvement at `bs=8/rank`, global `bs~=64`.
