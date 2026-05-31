# Final LM Head Report

> **TL;DR:** `decode.final.lm_head` is the final TP1 full-vocab BF16 GEMM, `W[163840,7168] x X[7168,8] -> logits[163840,8]`, through `gemm_graphsafe`. H20 TP1 PPLX `bs=8,ctx=1` measures `542.7us`, `34.63 TF/s`, and `4.33 TB/s`, or `90.3%` of the H20 HBM roofline used by the bench. This is already very close to the practical memory limit for a weight-streaming GEMM; stop standalone LM-head tuning unless a future quantized/FP8 output path changes the bandwidth equation.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | If a kernel has enough work, low utilization usually comes from tail effects, scheduling, or grid size; persistent/tile scheduling are only worth it when profile shows the issue. | The LM head is not a tiny helper; the bench already reports `~4.33 TB/s`. Do not assume a custom schedule can beat cuBLAS without NCU proof. |
| `sources/prs/sglang/PR-20755.md` (`pr-sglang-20755`) | FlashInfer `tinygemm_bf16` was useful for small SM90+ GEMMs in GPT-OSS MoE routing. | Directional only and mostly not applicable: LM head has huge `N=163840`, not a small router GEMM. |
| `sources/prs/flashinfer/PR-2131.md` (`pr-flashinfer-2131`) | FlashInfer exposes DeepGEMM swapAB for SM90 FP8 block-scale GEMM. | A future quantized LM head could be revisited, but the current row is BF16 and already memory-roofline limited. |

Practical conclusion: for the current BF16 full-vocab LM head, the optimization question is weight bandwidth, not compute. The current cuBLAS path is a strong baseline.

## NCU Conclusion

Fresh production NCU is currently unavailable on `h20-100`:

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# exits 124 with no output
```

So this report does not claim a lower-level NCU kernel name or scheduler breakdown. The CUDA-event roofline evidence is still enough to stop standalone replacement work for now because the row is already at `~90%` of the H20 HBM roofline in the TP1 PPLX bench. A future NCU pass should only be used to validate a serious alternative, not to launch another broad GEMM search.

If reopened, collect:

```bash
/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -k regex:.*gemm.* \
  -o profile/kimi-final-lm-head-h20/reports/lm_head_full \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.final.lm_head \
  --out profile/kimi-final-lm-head-h20/lm_head_ncu.json
```

Required profile questions:

| Question | Why it matters |
|---|---|
| DRAM read throughput | Confirm the bench-derived `4.33 TB/s` is real weight-streaming bandwidth, not bytes-model optimism. |
| L2 hit rate | LM head weights are too large to reuse meaningfully at `bs=8`; low L2 reuse is expected. |
| Grid/wave count | Only relevant if an alternative schedule claims better H20 occupancy without extra memory traffic. |

## Bench Evidence

Runtime path:

| Item | Value |
|---|---|
| Runtime call | `typed_ops::gemm_graphsafe_into` in `pegainfer-kimi-k2/src/runner/worker/forward.rs` |
| Bench op | `gemm_graphsafe` / `decode.final.lm_head` |
| CUDA entry | `gemm_graphsafe_cuda` in `pegainfer-kernels/csrc/linear.cu` |
| Shape | `rows=8, out=163840, in=7168`, BF16 weights/activations/output |
| Calls per decode step | `1` |

The final rows use the fixed decode arena (`arena_rows=8`), so the `active_rows=1,2,4,8` H20 baseline entries all measure the same `rows=8` LM-head shape.

| Artifact | Active rows | ctx | Latency | TFLOP/s | Payload GB/s | H20 HBM pct |
|---|---:|---:|---:|---:|---:|---:|
| `tp1-pplx-decode-bench-h20-100.json` | 1 | 1 | `542.36us` | `34.65` | `4335.8` | `90.3%` inferred |
| `tp1-pplx-decode-bench-h20-100.json` | 2 | 1 | `542.30us` | `34.65` | `4336.3` | `90.3%` inferred |
| `tp1-pplx-decode-bench-h20-100.json` | 4 | 1 | `541.95us` | `34.67` | `4339.1` | `90.4%` inferred |
| `tp1-pplx-decode-bench-h20-100.json` | 8 | 1 | `542.69us` | `34.62` | `4333.1` | `90.3%` inferred |
| `tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` | 8 | 1 | `542.68us` | `34.63` | `4333.2` | `90.28%` |
| `tp1-pplx-decode-bench-cublaslt-bs3-bs8.json` | 8 | 1 | `546.44us` | `34.39` | `4303.4` | `89.65%` |

The arithmetic intensity is about `8 flop/byte`, below the H20 ridge point recorded in the master table (`30.83 flop/byte`), so the row is memory-bound. The dominant bytes are the BF16 LM-head weight matrix, roughly `2.35 GB` per call.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current `gemm_graphsafe` / cuBLAS path | `542.7us`, `4.33 TB/s`, `90.3%` H20 HBM roofline at `bs=8,ctx=1`. | Current baseline and practical stop point. |
| FlashInfer tinygemm-style small GEMM | Not attempted. KernelWiki lead targets small SM90 GEMMs; this row has huge vocab output and already reaches high HBM bandwidth. | Reject for this shape. |
| cuBLASLt exact-shape swap | Not attempted. Other Kimi skinny GEMM cuBLASLt swaps helped when cuBLAS left obvious bandwidth on the table; here the current path is already `~90%` HBM. | Do not spend a broad sweep without an NCU-backed hypothesis. |
| Quantized / FP8 LM head | Not attempted. Would reduce weight bytes, but changes numeric and weight-format contracts. | Future-only direction with correctness gate and explicit weight/cache format work. |

## Final Conclusion

Stop standalone `decode.final.lm_head` optimization for the current BF16 path. It is memory-bound and already close enough to H20's measured HBM ceiling that replacing cuBLAS is unlikely to clear the `>3%` reproducible improvement bar without changing the data format.

Reopen only if one of these is true:

| Trigger | Required proof |
|---|---|
| NCU shows actual DRAM bandwidth far below the bench model | Add the NCU artifact and identify the real bottleneck before coding. |
| Quantized/FP8 LM head becomes an accepted model-format change | Include accuracy/token gates and a full TP1 PPLX bench. |
| A library upgrade provides a stronger exact-shape GEMM | Compare against the existing `542.7us` H20 baseline and require `>3%` improvement. |

No `opt(...)` commit is warranted.
