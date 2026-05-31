# Shared SwiGLU Report

> **TL;DR:** `decode.moe.shared_swiglu` is the shared-expert activation kernel `silu_mul_hs_fused_into` / `silu_mul_fused_kernel`, shape `gate_up=[4096,8] -> out=[2048,8]` BF16. TP1 PPLX `bs=8,ctx=1` bench artifacts put it at `410.2-473.3us` per 60 MoE-layer decode step (`6.8-7.9us/call`), while the isolated H20 harness reports `202.2us/step` (`3.37us/call`) and NCU reports `2.88us` for one launch. NCU says `64` CTAs on `78` SMs, `0.10` waves/SM, `0.51%` DRAM read, `2.53%` SM throughput, and `93.39%` scheduler no eligible. Stop standalone SwiGLU tuning; future work should only be a full row21 -> row22 gated-dual-GEMM or row22 -> row23 activation-prologue fusion that wins in the full TP1 PPLX bench.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/kernels/gated-dual-gemm.md` (`kernel-gated-dual-gemm`) | Gate/up + SwiGLU fusion removes intermediate global-memory roundtrips by keeping both GEMM accumulators live until the activation epilogue. | This is the real row21 -> row22 opportunity, but it requires a gated-dual-GEMM schedule, not a standalone elementwise rewrite. |
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Low SM utilization can come from a grid smaller than the SM count; non-persistent kernels should generally launch many more CTAs than SMs. | Row 22 launches only `64` CTAs on H20's `78` SMs and NCU reports `0.10` waves/SM, so it is tiny-grid/latency limited. |
| `wiki/techniques/epilogue-fusion.md` (`technique-epilogue-fusion`) | Epilogue fusion is useful when post-MMA work consumes an accumulator before global store. | Fits row21 -> row22 gated-dual GEMM. It does not directly express row22 -> row23, which would need an activation transform in the next GEMM input path. |
| `sources/prs/flashinfer/PR-1396.md` (`pr-flashinfer-1396`) | FlashInfer uses CUTLASS MoE paths with SwiGLU-style activation for SM90/SM100 variants. | Directional only: the implementation family is fused MoE/GEMM builders, but the exact Kimi BF16 shared-expert decode shape still has to beat cuBLASLt on H20. |

Practical conclusion: standalone row 22 has no useful HBM or tensor-core roofline target. The useful work is eliminating this launch and intermediate activation traffic as part of a stronger neighboring GEMM path.

## NCU Conclusion

Workload: Kimi K2 TP1 DP8 EP8 + PPLX decode, per-rank `bs=8`, global `bs~=64`, `ctx=1`.

Runtime path:

| Item | Value |
|---|---|
| Rust API | `pegainfer-kernels::typed_ops::silu_mul_hs_fused_into` |
| CUDA entry | `silu_mul_fused_cuda` |
| CUDA kernel | `silu_mul_fused_kernel` in `pegainfer-kernels/csrc/fused_proj.cu` |
| Shape | `gate_up=[4096,8] -> out=[2048,8]`, BF16 |
| Payload bytes | `98,304 B/call`, `5.90 MB/step` for 60 calls |

Bench evidence:

| Artifact | Step latency | Per call | Payload bandwidth | H20 peak note |
|---|---:|---:|---:|---|
| Phase 1 master artifact `tp1-pplx-decode-bench-h20-100.json` | `410.2us` | `6.84us` | `14.38 GB/s` | Not bandwidth-saturating. |
| Current optimized artifact `tp1-pplx-decode-bench-cublaslt-bs3-bs8.json` | `473.3us` | `7.89us` | `12.46 GB/s` | High run-to-run noise at this tiny launch size. |
| Isolated H20 harness in `profile/kimi-shared-swiglu-h20-baseline/` | `202.2us` | `3.37us` | `29.2 GB/s` | Use as NCU isolation, not full-path latency. |

NCU command family:

```bash
profile/kimi-shared-swiglu-h20-baseline/harness/build_command.sh
profile/kimi-shared-swiglu-h20-baseline/harness/shared_swiglu_harness --iters 200

/usr/local/cuda-12.9/bin/ncu --set full \
  --section PmSampling --section PmSampling_WarpStates \
  -k regex:shared_swiglu_kernel -s 1200 -c 1 \
  -o profile/kimi-shared-swiglu-h20-baseline/reports/full_bs8 \
  profile/kimi-shared-swiglu-h20-baseline/harness/shared_swiglu_harness --profile-one
```

Parsed with `ncu-report-skill` helper:

```bash
PYTHONPATH=/opt/nvidia/nsight-compute/2026.2.0/extras/python \
uv run --no-project python \
  /data/code/dev-skills/kernel-design-agents/skills/ncu-report-skill/helpers/analyze_reports.py \
  --run-dir profile/kimi-shared-swiglu-h20-baseline \
  --report profile/kimi-shared-swiglu-h20-baseline/reports/full_bs8.ncu-rep \
  --tag shared_swiglu_bs8
```

Key H20 metrics:

| Metric | Value |
|---|---:|
| NCU kernel duration | `2.88us` |
| Grid / block | `64` CTAs x `256` threads |
| H20 SMs | `78` |
| Waves / SM | `0.10` |
| Achieved occupancy | `11.08%` |
| SM throughput | `2.53%` |
| Compute memory throughput | `0.92%` |
| DRAM read throughput | `24.71 GB/s`, `0.51%` peak |
| Tensor pipe | `0%` |
| L1 hit rate | `15.17%` |
| L2 hit rate | `53.44%` |
| Eligible warps / scheduler | `0.074` |
| Scheduler no eligible | `93.39%` |
| Top stall ratio | long scoreboard `9.91` per issue-active |

Diagnosis: this is a tiny elementwise launch with poor latency hiding. It is not compute-bound, and it is not close to H20 HBM bandwidth. The low payload bandwidth is a symptom of launch geometry and dependency latency, not a standalone memory kernel waiting for bandwidth tuning.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current `silu_mul_fused_kernel` | Full TP1 PPLX bench: `410.2-473.3us/step`; isolated harness: `202.2us/step`; NCU: `64` CTAs, `0.10` waves/SM, `93.39%` no eligible. | Current baseline. |
| Standalone row-22 rewrite | Not attempted. There is too little work per launch and no HBM/compute saturation target. | Stop standalone direction. |
| Row21 -> row22 stock CUTLASS gated-dual GEMM | Tried in `profile/kimi-shared-gated-dual-gemm-h20-prototype/`; best tested tile `16x32x64` was `68.7us/call`, versus `21.1-21.3us/call` for current cuBLASLt shared_gate_up plus current SwiGLU in the same standalone setup. | Rejected; stock example schedule is not shaped for Kimi decode `batch_size=8`. |
| Row22 -> row23 activation prologue | Not implemented. Standard cuBLASLt epilogues do not express "read gate/up, apply SwiGLU, feed down GEMM" as the next GEMM input transform. | Future custom-only direction; must beat full TP1 PPLX bench and preserve BF16 rounding. |

## Final Conclusion

Stop `decode.moe.shared_swiglu` as a standalone Phase 3 target. Reclassify it in the master table as `control/tiny-grid`: it has memory traffic, but the H20 profile is dominated by small launch geometry and scheduler starvation rather than HBM bandwidth.

Reopen only for one of these custom paths:

| Direction | Adoption bar |
|---|---|
| row21 -> row22 gated-dual GEMM | Must beat the current `kimi_shared_gate_up_cublaslt + silu_mul_fused_kernel` full TP1 PPLX bench by more than noise. |
| row22 -> row23 activation-prologue GEMM | Must preserve the exact BF16 rounding contract and beat the current `shared_swiglu + shared_down` full TP1 PPLX bench. |

No docs-only stop report should create an `opt(...)` commit; accepted code commits still require reproducible `>3%` improvement at `bs=8/rank`, global `bs~=64`, H20.
