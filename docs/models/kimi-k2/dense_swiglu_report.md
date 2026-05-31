# Dense SwiGLU Report

> **TL;DR:** `decode.dense.swiglu` is the single dense-layer activation helper `silu_mul_hs_fused_into` / `silu_mul_fused_kernel`, consuming `gate_up[36864,8]` and writing `activated[18432,8]`. H20 TP1 PPLX `bs=8,ctx=1` measures `7.79us`, `113.6GB/s` payload-equivalent throughput, and about `2.4%` H20 HBM. It is an elementwise activation row with `576` CTAs and one call per decode step; stop standalone tuning and only revisit as a dense gate/up+SwiGLU or SwiGLU+down fusion.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | A memory-bound label needs high measured DRAM throughput. | The row moves little payload and reports only `~113GB/s`; it is not an H20 bandwidth target. |
| `wiki/techniques/epilogue-fusion.md` (`technique-epilogue-fusion`) | SwiGLU gate is an epilogue-family fusion candidate that can avoid a separate activation launch and global round trip. | The only plausible future work is dense MLP fusion around rows 2/3/4, not a standalone activation rewrite. |
| `sources/prs/flashinfer/PR-3014.md` (`pr-flashinfer-3014`) | Small-batch decode helper overhead should be reduced by removing helper work or launch overhead when possible. | Directional only: this row is one helper launch, so launch removal through fusion is more plausible than tuning the helper alone. |

## NCU Conclusion

Fresh production NCU is currently unavailable on `h20-100`:

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# exits 124 with no output
```

Source geometry:

| Source fact | Value |
|---|---:|
| CUDA entry | `silu_mul_fused_cuda` |
| Kernel | `silu_mul_fused_kernel` in `pegainfer-kernels/csrc/fused_proj.cu` |
| Intermediate | `18432` |
| Batch | `8` |
| Elements | `147456` |
| Launch | `576` CTAs x `256` threads |

The shared-expert SwiGLU row has same kernel code but a different much smaller shape and its own NCU report. This dense row still needs production NCU if someone wants to distinguish SFU/math latency from memory effects. Until then, the event data supports stopping standalone work.

## Bench Evidence

| Artifact | Step latency | Payload GB/s | H20 HBM pct |
|---|---:|---:|---:|
| `tp1-pplx-decode-bench-h20-100.json` | `7.80us` | `113.41` | inferred `2.36%` |
| `tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` | `7.79us` | `113.63` | `2.37%` |

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current fused SiLU-mul helper | `7.79us` per decode step. | Current baseline. |
| Standalone activation rewrite | Not attempted. One call and low payload bandwidth make it low leverage. | Stop standalone direction. |
| Dense gate/up + SwiGLU or SwiGLU + down fusion | Not implemented. Needs a combined dense MLP path and full TP1 PPLX correctness/bench proof. | Future-only direction. |

## Final Conclusion

Keep `silu_mul_fused_kernel` for `decode.dense.swiglu` and classify the row as `control/elementwise`. Reopen only as part of a dense MLP fusion that clears the full-bench gate.
