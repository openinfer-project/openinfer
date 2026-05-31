# Dense Residual Add Report

> **TL;DR:** `decode.dense.residual_add` is the single dense-layer BF16 residual add `add_batch`, shape `rows=8, hidden=7168`. H20 TP1 PPLX `bs=8,ctx=1` measures `6.81-7.51us` and only `45.8-50.5GB/s` payload-equivalent throughput. Source launch geometry is `224` CTAs x `256` threads; this is a control/elementwise helper, not a standalone bandwidth target. Stop standalone tuning.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | Low arithmetic intensity does not by itself prove a bandwidth-bound optimization target; measured DRAM throughput matters. | The row is around `1%` H20 HBM by the bench payload model. |
| `wiki/techniques/epilogue-fusion.md` (`technique-epilogue-fusion`) | Residual add can be fused into GEMM epilogues to remove a follow-up launch and memory round trip. | If this row ever moves, it should be by down-GEMM epilogue fusion, not by rewriting `add_kernel` alone. |

## NCU Conclusion

Fresh production NCU is currently unavailable on `h20-100`:

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# exits 124 with no output
```

Source geometry:

| Source fact | Value |
|---|---:|
| CUDA entry | `add_cuda` |
| Kernel | `add_kernel` in `pegainfer-kernels/csrc/elementwise.cu` |
| Elements | `57344` |
| Launch | `224` CTAs x `256` threads |

No NCU counters are claimed. The row is one tiny helper launch and cannot justify standalone work at the full decode-path level.

## Bench Evidence

| Artifact | Step latency | Payload GB/s | H20 HBM pct |
|---|---:|---:|---:|
| `tp1-pplx-decode-bench-h20-100.json` | `6.81us` | `50.52` | inferred `1.05%` |
| `tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` | `7.51us` | `45.81` | `0.95%` |

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current `add_kernel` | `6.8-7.5us` per decode step. | Current baseline. |
| Standalone add rewrite | Not attempted. The row is a tiny helper and not bandwidth-saturating. | Stop standalone direction. |
| Down GEMM + residual epilogue | Not implemented. This is the only plausible future direction, and must preserve BF16 rounding. | Future-only direction. |

## Final Conclusion

Keep the current `add_batch` provider and classify `decode.dense.residual_add` as `control/elementwise`. Reopen only if a dense down epilogue fusion removes the launch without slowing the GEMM and passes the full TP1 PPLX gate.
