# Attention Post-Attention Add Norm Report

> **TL;DR:** `decode.attention.post_attn_add_norm` is the Kimi exact-preserving fused add + RMSNorm round kernel after attention: `hidden = bf16(hidden + projected)`, then RMSNorm into the next layer input. At TP1 PPLX `bs=8,ctx=1`, H20 event timing is `527.7-530.0us` per 61-layer step, or `8.65-8.69us/call`, with only `~79GB/s` payload-equivalent throughput. Source launch geometry is `8` CTAs x `896` threads with about `28KB` dynamic shared memory per CTA, so classify this as `control/tiny-grid` and stop standalone tuning until production NCU is available.
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

Fresh production NCU is currently unavailable on `h20-100`:

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# exits 124 with no output
```

So this report does not claim measured warp stalls or DRAM percentages for `FusedAddRMSNormRoundKernel`. The control/tiny-grid classification is based on source launch geometry plus H20 event timing:

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

If reopened, collect:

```bash
/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -k regex:FusedAddRMSNormRoundKernel \
  -o profile/kimi-attention-post-attn-add-norm-h20/reports/post_attn_add_norm_full \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.attention.post_attn_add_norm \
  --out profile/kimi-attention-post-attn-add-norm-h20/post_attn_add_norm_ncu.json
```

Required profile questions:

| Question | Why it matters |
|---|---|
| Dynamic shared memory occupancy | The kernel stores the rounded BF16 sum in shared memory before normalization; this may cap active CTAs even beyond the tiny grid. |
| Scheduler no eligible / issue slot utilization | Distinguish launch/control overhead from math or shared-memory stalls. |
| DRAM throughput | Confirm the event payload model's `~79GB/s` is not hiding unmodeled traffic. |

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
| Standalone kernel retune | Not attempted. Source grid is only `8` CTAs, and NCU is unavailable. | Stop standalone direction. |
| Fuse into downstream GEMM/router prologue | Not implemented. Downstream consumers differ between dense layer and MoE layers, so this is a larger custom path. | Future-only direction with token correctness and full TP1 PPLX bench proof. |

## Final Conclusion

Keep the current exact-preserving `FusedAddRMSNormRoundKernel` and classify `decode.attention.post_attn_add_norm` as `control/tiny-grid`. Do not treat the low HBM percentage as a bandwidth problem. Reopen only for production NCU evidence or a downstream prologue fusion that preserves the BF16 rounding boundary and clears the `>3%` full-bench bar.
