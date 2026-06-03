# Attention Post-Attention Add Norm Report

> **TL;DR:** `decode.attention.post_attn_add_norm` is the Kimi exact-preserving fused add + RMSNorm round kernel after attention: `hidden = bf16(hidden + projected)`, then RMSNorm into the next layer input. At TP1 PPLX `bs=8,ctx=1`, H20 event timing is `527.7-530.0us` per 61-layer step, or `8.65-8.69us/call`, with only `~79GB/s` payload-equivalent throughput. H20 NCU confirms the row is launch/control limited: `8` CTAs, `0.05` waves/SM, `2.29%` SM throughput, `1.11%` DRAM throughput, and `64.78%` no-eligible scheduler cycles. Standalone tuning is stopped; only a launch-removing downstream prologue fusion is worth reopening.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Low SM utilization can come from tiny grids, tail effects, static scheduling, or load imbalance; profile before choosing a scheduling fix. | This row launches one CTA per decode arena row (`8` CTAs total), so standalone tuning is dominated by tiny-grid/control behavior unless NCU proves otherwise. |
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | Memory-bound diagnosis needs high measured DRAM throughput. Low arithmetic intensity alone is not enough. | The event model reports only `~79GB/s`, around `1.6%` of H20 HBM, so this is not a useful HBM-bound target. |
| `sources/prs/flashinfer/PR-3014.md` (`pr-flashinfer-3014`) | Small-batch decode helper overhead should be attacked by removing helper work or launch overhead. | Directional only: this row already fuses add, BF16 rounding, and RMSNorm. Further useful work would have to remove the launch through a downstream prologue, not split it apart. |

Practical conclusion: the current kernel is already the necessary local fusion for the post-attention boundary. The remaining upside is launch deletion, but downstream consumers differ by layer type, so any prologue fusion needs a full TP1 PPLX correctness and bench gate.

## NCU Conclusion

Fresh H20 NCU was collected on `h20-100` for `FusedAddRMSNormRoundKernel`:

```bash
/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -k regex:FusedAddRMSNormRoundKernel \
  -o /dev/shm/kimi-post-attn-add-norm-ncu/reports/post_attn_add_norm_full \
  --force-overwrite /dev/shm/pegainfer-kimi-partition-target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.attention.post_attn_add_norm \
  --out /dev/shm/kimi-post-attn-add-norm-ncu/post_attn_add_norm_ncu.json
```

The report path is `/dev/shm/kimi-post-attn-add-norm-ncu/reports/post_attn_add_norm_full.ncu-rep`. NCU profiling overhead made the bench wall-time row unusable for latency; use the normal event artifacts for latency and the NCU report for counters.

| Source fact | Value |
|---|---:|
| CUDA entry | `fused_add_rms_norm_round_batched_cuda` |
| Kernel | `FusedAddRMSNormRoundKernel<VEC_SIZE,bf16>` in `pegainfer-kernels/csrc/flashinfer_norm.cu` |
| Target `batch_size` | `8` |
| Hidden dim `d` | `7168` |
| BF16 vector size | `8` elements |
| Threads per CTA | `32 x 28 = 896` |
| CTAs | `8` |
| Dynamic shared memory | `(28 + 7168) * 4 = 28,784 B` |

Key NCU counters:

| Metric | Value |
|---|---:|
| Duration | `4.77 us` |
| Grid / block | `8` CTAs / `896` threads |
| Waves per SM | `0.05` |
| Registers/thread | `32` |
| Dynamic shared memory/block | `28.78 KiB` |
| Compute throughput | `2.29%` |
| DRAM throughput | `1.11%` |
| Memory throughput | `54.01 GB/s` |
| L1/TEX throughput | `31.91%` |
| L2 hit rate | `57.11%` |
| SM busy / issue slots busy | `34.13% / 34.13%` |
| No eligible | `64.78%` |
| Eligible / active warps per scheduler | `0.98 / 6.62` |
| NCU launch warning | Grid has only `8` blocks on `78` SMs; `0.1` full waves |

Interpretation:

- The kernel is not HBM-bound: DRAM throughput is only `1.11%`.
- It is not compute-bound either: SM throughput is `2.29%`, and the grid has only `0.05` waves/SM.
- NCU reports shared-memory load/store conflicts with estimated local speedup around `15%`, but the row costs only `8.65-8.69us/call`; even a perfect standalone fix is too small and still leaves a separate launch. Treat this as a launch-removal/fusion candidate, not a standalone kernel target.

## Bench Evidence

Runtime path:

| Item | Value |
|---|---|
| Runtime call | `typed_ops::fused_add_rms_norm_round_into` in `pegainfer-kimi-k2/src/runner/worker/forward.rs` |
| Bench op | `fused_add_rms_norm_round_batch` / `decode.attention.post_attn_add_norm` |
| CUDA entry | `fused_add_rms_norm_round_batched_cuda` in `pegainfer-kernels/csrc/flashinfer_norm.cu` |
| Shape | `rows=8, hidden=7168`, BF16 hidden/residual/output plus BF16 weight |
| Calls per decode step | `61` |

H20 event evidence:

| Artifact | Step latency | Per call | TFLOP/s | Payload GB/s | H20 HBM pct |
|---|---:|---:|---:|---:|---:|
| `tp1-pplx-decode-bench-h20-100.json` | `530.03us` | `8.69us` | `0.046` | `79.20` | not emitted in old label artifact |
| `tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` | `527.74us` | `8.65us` | `0.046` | `79.54` | `1.66%` |
| `tp1-pplx-decode-bench-mla-cublaslt-bs8.json` | `541.62us` | `8.88us` | `0.045` | `77.50` | `1.61%` |

This row exists because Kimi token correctness depends on the BF16 rounding boundary after `hidden + residual`. Replacing it with FlashInfer's ordinary fused add RMSNorm is not an optimization unless it preserves that rounding behavior.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current exact-preserving fused add + RMSNorm round | `8.65-8.69us/call` at `bs=8,ctx=1`; only `~1.6%` H20 HBM by payload model. | Current baseline. |
| Ordinary FlashInfer fused add RMSNorm | Not attempted here. Existing source comments say it keeps the pre-BF16-round add value for the RMS reduction, while Kimi correctness needs the rounded BF16 sum. | Reject unless correctness gate proves equivalence. |
| H20 NCU full profile | NCU confirms `8` CTAs, `0.05` waves/SM, `2.29%` SM throughput, `1.11%` DRAM throughput, and `64.78%` no-eligible scheduler cycles. Shared-memory conflicts show a local `~15%` hint, but the end-to-end impact is too small without launch removal. | Stop standalone direction. |
| Standalone kernel retune | Not attempted after NCU. | Rejected: the profile points to tiny-grid/launch removal, not a profitable local retune. |
| Fuse into downstream GEMM/router prologue | Not implemented. Downstream consumers differ between dense layer and MoE layers, so this is a larger custom path. | Future-only direction with token correctness and full TP1 PPLX bench proof. |

## Final Conclusion

Keep the current exact-preserving `FusedAddRMSNormRoundKernel` and classify `decode.attention.post_attn_add_norm` as `control/tiny-grid`. Do not treat the low HBM percentage as a bandwidth problem. Standalone tuning is stopped; reopen only for a downstream prologue fusion that preserves the BF16 rounding boundary and clears the `>3%` full-bench bar.
