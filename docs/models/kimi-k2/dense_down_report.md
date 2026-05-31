# Dense Down Report

> **TL;DR:** `decode.dense.down` is the single dense-layer MLP down BF16 skinny GEMM, `W[7168,18432] x X[18432,8] -> hidden[7168,8]`, through `gemm_dm_hs_to_typed_graphsafe`. H20 TP1 PPLX `bs=8,ctx=1` measures `85.48us`, `24.73 TF/s`, and `3.10 TB/s`, or about `64.5%` of the H20 HBM roofline used by the bench. It is memory-bound and only one call per decode step, so standalone tuning is stopped unless NCU identifies a specific library-scheduling gap.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | Small-batch decode GEMM/GEMV-like kernels are usually bandwidth-bound when arithmetic intensity is below the ridge point. | This down GEMM streams BF16 weights for `M=8` and reaches `~3.10 TB/s` by the bench model. |
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Grid/tail effects are plausible for moderate tile counts, but persistent/tile scheduling should be driven by NCU evidence. | A custom down GEMM is not justified from event timing alone. |
| `wiki/techniques/epilogue-fusion.md` (`technique-epilogue-fusion`) | Residual add can be fused into a GEMM epilogue to remove a follow-up launch and round trip. | The adjacent dense residual add is tiny; a future down+residual epilogue could remove it, but the row is too small to justify a standalone custom path now. |

Practical conclusion: keep the current graph-safe cuBLAS down GEMM. The only plausible future direction is a combined dense down epilogue that preserves Kimi's BF16 rounding semantics and is measured on the full TP1 PPLX bench.

## NCU Conclusion

Fresh production NCU is currently unavailable on `h20-100`:

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# exits 124 with no output
```

So this report does not claim a lower-level cuBLAS kernel name or stall breakdown for this exact dense down row. The current classification is from the H20 event roofline: arithmetic intensity is below the H20 ridge point and achieved payload bandwidth is `~3.10 TB/s`.

If reopened, collect:

```bash
/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -k regex:.*gemm.* \
  -o profile/kimi-dense-down-h20/reports/dense_down_full \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.dense.down \
  --out profile/kimi-dense-down-h20/dense_down_ncu.json
```

Required profile questions:

| Question | Why it matters |
|---|---|
| DRAM read throughput | Validate the `3.10 TB/s` payload model against real H20 counters. |
| Grid/wave count | Determine whether generic cuBLAS has a small-grid/tail issue on this exact `M=8,N=7168,K=18432` shape. |
| Epilogue feasibility | Only relevant if down+residual fusion can preserve BF16 rounding and avoid slowing the GEMM. |

## Bench Evidence

Runtime path:

| Item | Value |
|---|---|
| Runtime call | `typed_ops::gemm_dm_hs_to_typed_graphsafe` in `forward_dense_mlp_decode_normed_into` |
| Bench op | `gemm_dm_hs_to_typed_graphsafe` / `decode.dense.down` |
| Shape | `rows=8, out=7168, in=18432`, BF16 |
| Calls per decode step | `1` |

H20 TP1 PPLX bench evidence:

| Artifact | Step latency | TFLOP/s | Payload GB/s | H20 HBM pct |
|---|---:|---:|---:|---:|
| `tp1-pplx-decode-bench-h20-100.json` | `85.34us` | `24.77` | `3101.0` | inferred `64.6%` |
| `tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` | `85.48us` | `24.73` | `3096.2` | `64.50%` |

The row is smaller than the major attention/shared/PPLX GEMMs and is executed only once per decode step. Even a large standalone improvement would have limited effect on global TPOT compared with routed expert and attention rows.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current graph-safe cuBLAS path | `85.48us`, `3.10 TB/s`, `64.5%` H20 HBM payload model. | Current baseline. |
| Standalone cuBLASLt exact-shape swap | Not attempted. Similar Kimi skinny rows were only accepted when they were repeated across 60/61 layers or had clear NCU evidence. | Reject broad sweep without NCU. |
| Down GEMM + residual add epilogue | Not implemented. It could remove a `~7us` helper, but must preserve BF16 rounding semantics and not slow the GEMM. | Future-only direction. |

## Final Conclusion

Stop standalone `decode.dense.down` tuning for the current BF16 path. Reopen only with production NCU evidence of a concrete cuBLAS scheduling gap, or as part of a measured dense down+residual fusion that clears the full-bench `>3%` bar.
