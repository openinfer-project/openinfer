# Dense Gate-Up Report

> **TL;DR:** `decode.dense.gate_up` is the single dense-layer MLP gate/up BF16 skinny GEMM, `W[36864,7168] x X[7168,8] -> gate_up[36864,8]`, through `gemm_dm_typed_to_hs_graphsafe`. H20 TP1 PPLX `bs=8,ctx=1` measures `147.96us`, `28.57 TF/s`, and `3.58 TB/s`, or about `74.5%` of the H20 HBM roofline used by the bench. It is memory-bound, called once per decode step, and too small in total TPOT share to justify standalone replacement without NCU-backed evidence.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | For low-arithmetic-intensity GEMV/small-batch decode work, optimize bandwidth first and do not spend effort on compute unless profile contradicts the roofline. | This row streams a large BF16 weight matrix for only `8` rows and reaches `~3.58 TB/s` by the bench model. The optimization question is weight bandwidth. |
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Low utilization can be a grid/tail scheduling issue, but profile is required before choosing persistent scheduling or tile scheduling. | A custom skinny GEMM is not justified without NCU showing cuBLAS leaves a concrete scheduling gap. |
| `wiki/techniques/epilogue-fusion.md` (`technique-epilogue-fusion`) | GEMM epilogues can fuse bias/activation/quantization to avoid extra launches and memory traffic; SwiGLU is a listed epilogue-family use. | The adjacent dense SwiGLU is a plausible fusion target in principle, but this is only layer0 and the current gate/up GEMM is already a strong bandwidth baseline. |

Practical conclusion: keep the current generic graph-safe cuBLAS GEMM for the dense gate/up row. Future work should only revisit it as a full dense MLP gated-dual path (`gate_up + SwiGLU`) if the combined row 2/3/4 sequence can beat the current baseline in the full TP1 PPLX bench.

## NCU Conclusion

Fresh production NCU is currently unavailable on `h20-100`:

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# exits 124 with no output
```

So this report does not claim a cuBLAS kernel name, occupancy, or stall breakdown for this exact dense layer0 GEMM. The current stop decision is based on H20 CUDA-event roofline evidence plus path contribution: `147.96us` is below `1%` of the `bs=8,ctx=1` local measured decode subtotal, and the row already reaches about three quarters of H20 HBM by the bench byte model.

If reopened, collect:

```bash
/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -k regex:.*gemm.* \
  -o profile/kimi-dense-gate-up-h20/reports/dense_gate_up_full \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.dense.gate_up \
  --out profile/kimi-dense-gate-up-h20/dense_gate_up_ncu.json
```

Required profile questions:

| Question | Why it matters |
|---|---|
| DRAM read throughput | Confirm the event-derived `3.58 TB/s` is real weight streaming. |
| Grid/wave count | Decide whether an exact-shape cuBLASLt or custom skinny GEMM could improve H20 occupancy. |
| Split-K or reduce launches | Check whether generic cuBLAS is adding avoidable helper kernels for this one-off shape. |

## Bench Evidence

Runtime path:

| Item | Value |
|---|---|
| Runtime call | `typed_ops::gemm_dm_typed_to_hs_graphsafe` in `forward_dense_mlp_decode_normed_into` |
| Bench op | `gemm_dm_typed_to_hs_graphsafe` / `decode.dense.gate_up` |
| Shape | `rows=8, out=36864, in=7168`, BF16 |
| Calls per decode step | `1` |

H20 TP1 PPLX bench evidence:

| Artifact | Step latency | TFLOP/s | Payload GB/s | H20 HBM pct |
|---|---:|---:|---:|---:|
| `tp1-pplx-decode-bench-h20-100.json` | `147.95us` | `28.58` | `3576.9` | inferred `74.5%` |
| `tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` | `147.96us` | `28.57` | `3576.5` | `74.51%` |

The arithmetic intensity is around `8 flop/byte`, below the H20 ridge point in the master table (`30.83 flop/byte`), so the row is memory-bound by the bench roofline.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current graph-safe cuBLAS path | `147.96us`, `3.58 TB/s`, `74.5%` H20 HBM payload model. | Current baseline. |
| Standalone cuBLASLt exact-shape swap | Not attempted. The row is one call per decode step and already high-bandwidth; any win is unlikely to clear the full-path `>3%` bar alone. | Reject broad sweep without NCU. |
| Dense gate/up + SwiGLU fusion | Not implemented. KernelWiki supports epilogue fusion directionally, but Kimi already uses a fused gate/up matrix and this layer appears only once. | Future-only dense MLP combined-kernel direction. |

## Final Conclusion

Stop standalone `decode.dense.gate_up` tuning for the current BF16 path. Reopen only if production NCU shows a concrete cuBLAS scheduling issue, or if a combined dense MLP fusion beats the row 2/3/4 sequence in the full TP1 PPLX bench while preserving token correctness.
