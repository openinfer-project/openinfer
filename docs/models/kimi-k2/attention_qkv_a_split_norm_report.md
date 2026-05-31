# Attention QKV-A Split Norm Report

> **TL;DR:** `decode.attention.qkv_a_split_norm` is a tiny custom Kimi MLA helper that splits `qkv_a`, RMS-normalizes `q_lora` and compressed KV, and preserves `k_rope`. At the TP1 PPLX target shape it costs `8.217us/call` or `501.2us` per 61-layer decode step, but H20 NCU shows `8` CTAs, `0.01` waves/SM, `0.19-0.20%` DRAM, `0.37-0.39%` compute, and `93.4-93.7%` scheduler no eligible. Stop standalone row-8 tuning; future work should only revisit it as a row8 -> q_b custom prologue/fusion that keeps the required `ckv_normed` and `k_rope` outputs.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Low SM utilization can come from tail effect, static scheduling, or a grid smaller than the SM count; non-persistent kernels need grid size much larger than SM count. | Row 8 launches `8` CTAs on H20's `78` SMs and NCU reports `0.01` waves/SM, so standalone tuning is dominated by launch/grid utilization. |
| `sources/prs/sglang/PR-20755.md` (`pr-sglang-20755`) | Small decode GEMMs sometimes need specialized small-GEMM kernels instead of generic library paths. | Directional only: the adjacent row 9 q_b GEMM is the real baseline any fused prologue must beat. |
| `sources/prs/flashinfer/PR-3014.md` (`pr-flashinfer-3014`) | Small-batch decode helper kernels benefit from removing unnecessary helper work and early exits. | Directional only: row 8 has useful work for both q_b and MLA cache, but the profile supports deleting/absorbing the helper launch rather than retuning it alone. |

Practical conclusion: row 8 is a helper-kernel launch problem. The only high-leverage direction is a custom q_b path that consumes `q_a` with RMSNorm in its prologue while still materializing `ckv_normed` and `k_rope` for the MLA cache path.

## NCU Conclusion

Workload: Kimi K2 TP1 DP8 EP8 + PPLX decode, per-rank `bs=8`, global `bs~=64`, `ctx=1`.

Runtime path: `kimi_mla_split_qkv_a_norm` consumes `qkv_a [batch,2112]`, emits `q_a_normed [batch,1536]`, `ckv_normed [batch,512]`, and `k_rope [batch,64]`. The q branch immediately feeds `decode.attention.q_b`; the compressed-KV and rope outputs feed MLA cache append/decode, so row 8 cannot be dropped unless those outputs remain correct.

Existing H20 run:

```bash
cargo run --release -p pegainfer-kimi-k2 --features kernel-report \
  --bin kimi_tp1_pplx_decode_bench -- \
  --active-rows 8 --ctx-lens 1 --iters 64 --format json \
  --labels decode.attention.qkv_a_split_norm,decode.attention.q_b \
  --out profile/kimi-attention-row8-row9-h20-baseline/row8_row9_event.json

/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -k regex:split_qkv_a_norm_kernel -c 4 \
  -o profile/kimi-attention-row8-row9-h20-baseline/reports/row8_full \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.attention.qkv_a_split_norm \
  --out profile/kimi-attention-row8-row9-h20-baseline/row8_ncu.json
```

Evidence:

| Metric | Value |
|---|---:|
| Provider | `split_qkv_a_norm_kernel` via `kimi_mla_split_qkv_a_norm` |
| Shape | `qkv_a=[8,2112] -> q_a_normed=[8,1536], ckv_normed=[8,512], k_rope=[8,64]`, BF16 |
| Event timing | `8.217us/call`, `501.2us/step` for 61 layers |
| Bench throughput | `12.2GB/s`, `0.25%` of H20 HBM roofline |
| NCU duration | `4.77-5.18us` |
| Grid / block | `8` CTAs x `192` threads |
| Waves / SM | `0.01` |
| DRAM throughput | `0.19-0.20%` |
| Compute throughput | `0.37-0.39%` |
| Achieved occupancy | `7.5-8.1%` |
| Scheduler no eligible | `93.4-93.7%` |
| L2 hit rate | `61-91%` |
| NCU top rule | launch geometry; grid smaller than the GPU |

Diagnosis: row 8 is not HBM-bound in the useful optimization sense. It moves little data and launches only one CTA per active arena row. The low achieved throughput is a symptom of tiny-grid control overhead, not a sign that a standalone memory bandwidth rewrite can recover hundreds of microseconds.

The adjacent q_b row sets the adoption bar:

| Row | H20 evidence |
|---|---|
| `decode.attention.q_b` | `17.255us/call`, `1.053ms/step`; cuBLAS kernel launches `64` CTAs, reaches `59-61%` DRAM, and is memory/skinny-grid limited. |

Any row8 -> q_b fused path must keep q_b near or below `17.255us/call` while removing the `8.217us/call` row-8 launch. It must also preserve the non-q_b outputs: `ckv_normed` and `k_rope`.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Standalone row-8 rewrite | Not attempted. NCU says the row is tiny-grid/launch limited with `<0.2%` DRAM. | Stop standalone direction. |
| Increase row-8 parallelism | Not attempted. Splitting per-row RMSNorm across more CTAs would require cross-CTA reductions for two separate RMSNorms and still leaves output materialization for three consumers. | Reject as low leverage. |
| q_b cuBLASLt exact-shape baseline | Tried and rejected in `q_b_proj_report.md`: target `batch_size=8` improved only `8.899us -> 8.746us` (`1.0175x`). | Do not swap q_b to cuBLASLt as a prerequisite. |
| Fuse row 8 into q_b prologue | Not implemented in Phase 2. This is the remaining plausible path, but it needs a custom skinny GEMM/prologue that preserves MLA cache outputs. | Keep for future custom work only; full TP1 PPLX bench and token correctness are required. |

## Final Conclusion

Stop `decode.attention.qkv_a_split_norm` as a standalone Phase 3 target. Keep the current Kimi helper kernel. Reopen only for a custom row8 -> q_b prologue/fusion that preserves `q_a_normed`, `ckv_normed`, and `k_rope` semantics, beats the current q_b cuBLAS path by enough to absorb implementation risk, and shows `>3%` full-bench improvement at `bs=8/rank`, global `bs~=64`.
