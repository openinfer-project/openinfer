# Kimi-K2 TP1 DP8 EP8 Decode Fusion Scan

> **TL;DR:** Phase 2 fusion scan has NCU evidence for three adjacent pairs. `shared_gate_up -> shared_swiglu` still needs a custom gated-dual-GEMM; `attention input RMSNorm -> qkv_a` is small-grid plus skinny-GEMM limited (`RMSNormKernel<8,bf16>` launches only `8` blocks; qkv_a cuBLAS main GEMM launches `72` blocks and reaches `51-53%` DRAM); `qkv_a_split_norm -> q_b` is split between a tiny-grid norm kernel (`8` blocks, `~0.2%` DRAM) and a cuBLAS skinny GEMM (`64` blocks, `59-61%` DRAM). qkv_a cuBLASLt was tried and rejected (`0.8-1.7%` bench-provider gain). No fusion is accepted yet.
>
> **Last touched:** 2026-05

## Inputs

- Master ledger: `docs/models/kimi-k2/tp1-dp8-ep8-decode-optimization-master.md`
- Current H20 optimized bench artifact: `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-cublaslt-bs3-bs8.json`
- Baseline H20 artifact: `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json`
- NCU run: `profile/kimi-shared-swiglu-h20-baseline/` on `h20-100`
- NCU run: `profile/kimi-attention-row6-row7-h20-baseline/` on `h20-100`
- NCU run: `profile/kimi-attention-row8-row9-h20-baseline/` on `h20-100`
- Rejected attempt: `profile/kimi-qkv-a-cublaslt-h20-baseline/` on `h20-100`

Current row values at `bs=8,ctx=1`:

| Row | Op | Provider | Step latency | Notes |
|---:|---|---|---:|---|
| 6 | attention input RMSNorm | `rms_norm_batch` | `476.3 us` | 61 calls before qkv_a GEMM. |
| 7 | attention qkv_a GEMM | `gemm_graphsafe` | `1.256 ms` | 61 calls, skinny GEMM. |
| 8 | qkv_a split/norm | `kimi_mla_split_qkv_a_norm` | `501.1 us` | 61 calls, already partially fused split plus norm. |
| 9 | q_b GEMM | `gemm_dm_typed_to_hs_graphsafe` | `1.052 ms` | 61 calls. |
| 21 | shared gate/up | `kimi_shared_gate_up_cublaslt` | `1.505 ms` | 60 calls after accepted cuBLASLt optimization. |
| 22 | shared SwiGLU | `silu_mul_hs_fused_into` | `473.3 us` | 60 calls in current optimized bench artifact; standalone harness below measures `202.2 us`. |
| 23 | shared down | `gemm_dm_hs_to_typed_graphsafe` | `902.2 us` | 60 calls. |

The current optimized bench artifact reports `shared_gate_up + shared_swiglu = 1.978 ms` per decode step. The standalone harness for the same row-22 shape reports `202.2 us` per 60 calls; that spread means any future accepted row-21/22 fusion must be measured in the full TP1 PPLX bench, not only in a standalone microbench.

## KernelWiki Conclusion

Relevant KernelWiki pages:

| Page | Relevant conclusion | How it applies here |
|---|---|---|
| `wiki/kernels/gated-dual-gemm.md` (`kernel-gated-dual-gemm`) | Gate/up + SwiGLU fusion removes intermediate global-memory roundtrips and reuses the input tile for both GEMMs. | This is the exact row 21/22 mathematical pattern, but the reference direction is dual GEMM with two accumulators plus a fused epilogue, not a cuBLASLt single-output epilogue. |
| `wiki/kernels/fused-moe.md` (`kernel-fused-moe`) | MoE fusion reduces launches and activation traffic, but small expert/token counts are sensitive to load imbalance and thin-GEMM efficiency. | The shared expert is not EP-routed, but it has the same low-batch decode shape. A custom fused GEMM must beat the current cuBLASLt skinny GEMM before it is worth adopting. |
| `sources/prs/flashinfer/PR-1396.md` (`pr-flashinfer-1396`) | FlashInfer added SM90 BF16 x MXFP4 CUTLASS MoE with `SwigluBias` activation. | Confirms the viable implementation family is CUTLASS-style fused MoE/gated builders, not a standard cuBLASLt epilogue. It is not direct evidence for BF16 dense shared expert on H20. |
| `sources/contests/gpu-mode-nvfp4/problem-3-gated-dual-gemm.md` (`contest-gpumode-p3`) | Fused dual GEMM keeps both gate and up accumulators live and applies SiLU/multiply in the epilogue. | Useful design pattern; much of the page is Blackwell/NVFP4-specific and cannot be copied directly to H20 BF16. |
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Low SM utilization can come from tail effect, load imbalance, static scheduling, or a grid that is smaller than the SM count; for non-persistent kernels, grid size should be much larger than SM count. | Row 6 launches `8` blocks on `78` SMs; row 7's main cuBLAS GEMM launches `72` blocks and the split-K reduce launches `66` blocks. This matches a shape/grid utilization limit, not an H20 peak-compute limit. |
| `sources/prs/sglang/PR-20755.md` (`pr-sglang-20755`) | FlashInfer `tinygemm_bf16` is used for a small SM90+ GEMM in the GPT-OSS MoE router. | Directional only: it supports checking small-GEMM library alternatives for row 9, but it is not direct evidence for Kimi `M=8,N=12288,K=1536`. |
| `sources/prs/flashinfer/PR-1668.md` (`pr-flashinfer-1668`) | TGV GEMM targets minimum-latency BF16 small GEMM on B200 and reports wins over cuBLAS for small problem sizes. | Directional only for H20: the implementation relies on SM100-specific features, but it confirms that low-batch decode GEMM needs a small-GEMM baseline rather than peak-compute intuition. |

Practical conclusion: row 21/22 fusion is plausible, but only as a gated-dual-GEMM kernel. cuBLASLt can provide strong GEMM baselines, but it cannot express `out = SiLU(Y[0:2048]) * Y[2048:4096]` as an epilogue over one 4096-column output.

For row 6/7, the KernelWiki low-SM-utilization pattern applies directly. The useful fusion is not a standalone RMSNorm rewrite; it is removing the RMSNorm launch plus normalized-hidden write/read while preserving or improving the skinny qkv_a GEMM tiling.

For row 8/9, the same low-SM-utilization pattern applies to row 8 and the small-GEMM direction applies to row 9. A useful fusion has to remove or hide the `split_qkv_a_norm` launch without making the row-9 GEMM slower, and it must preserve both consumers of row 8: `q_a_normed` for q_b and `ckv_normed`/`k_rope` for MLA cache.

## NCU Conclusion

Standalone H20 profile for row 22 shape:

```bash
profile/kimi-shared-swiglu-h20-baseline/harness/build_command.sh
profile/kimi-shared-swiglu-h20-baseline/harness/shared_swiglu_harness --iters 200

/usr/local/cuda-12.9/bin/ncu --set full \
  --section PmSampling --section PmSampling_WarpStates \
  -k regex:shared_swiglu_kernel -s 1200 -c 1 \
  -o profile/kimi-shared-swiglu-h20-baseline/reports/full_bs8 \
  profile/kimi-shared-swiglu-h20-baseline/harness/shared_swiglu_harness --profile-one

/usr/local/cuda-12.9/bin/ncu --set source --section SourceCounters \
  -k regex:shared_swiglu_kernel -s 1200 -c 1 \
  -o profile/kimi-shared-swiglu-h20-baseline/reports/source_bs8 \
  profile/kimi-shared-swiglu-h20-baseline/harness/shared_swiglu_harness --profile-one
```

Key H20 numbers:

| Metric | Value |
|---|---:|
| Shape | `gate_up=[4096,8] -> out=[2048,8]`, BF16 |
| Event timing | `202.2 us` per 60 calls, `3.37 us/call` |
| NCU kernel duration | `2.88 us` |
| Grid / block | `64` blocks x `256` threads |
| H20 SMs | `78` |
| Waves per SM | `0.10` |
| Achieved occupancy | `11.08%` |
| Memory throughput | `24.71 GB/s`, `0.92%` memory throughput |
| DRAM throughput | `0.51%` |
| L1/TEX hit rate | `15.17%` |
| L2 hit rate | `53.44%` |
| Compute throughput | `2.53%` |
| Scheduler no eligible | `93.39%` |
| NCU top rule | `42.11%` estimated speedup from L1TEX scoreboard stalls |

Diagnosis: row 22 alone is not HBM-bandwidth-bound. It is a very small memory/SFU elementwise kernel with low waves, low eligible warps, and enough scoreboard waiting that tuning the elementwise kernel in isolation is low leverage. Fusion matters because it can remove the standalone launch and avoid writing/reading the 4096 BF16 gate/up buffer, not because this kernel can be pushed near HBM peak.

### Row 6/7: attention input_norm -> qkv_a

H20 profile for row 6/7:

```bash
cargo run --release -p pegainfer-kimi-k2 --features kernel-report \
  --bin kimi_tp1_pplx_decode_bench -- \
  --active-rows 8 --ctx-lens 1 --iters 64 --format json \
  --labels decode.attention.input_norm,decode.attention.qkv_a \
  --out profile/kimi-attention-row6-row7-h20-baseline/row6_row7_event.json

/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -c 10 -o profile/kimi-attention-row6-row7-h20-baseline/reports/discover_row6 \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.attention.input_norm \
  --out profile/kimi-attention-row6-row7-h20-baseline/row6_ncu_discover.json

/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -c 6 -o profile/kimi-attention-row6-row7-h20-baseline/reports/discover_row7 \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.attention.qkv_a \
  --out profile/kimi-attention-row6-row7-h20-baseline/row7_ncu_discover.json
```

Event timing:

| Row | Label | Calls | Mean/call | Step latency | Bench roofline |
|---:|---|---:|---:|---:|---|
| 6 | `decode.attention.input_norm` | 61 | `8.008us` | `488.5us` | memory, `57.3GB/s`, `1.19%` HBM |
| 7 | `decode.attention.qkv_a` | 61 | `20.407us` | `1.245ms` | memory, `11.87TF/s`, `1.491TB/s`, `31.1%` HBM |

NCU details:

| Row | Kernel | Duration | Grid / block | Waves/SM | DRAM | Compute | Main diagnosis |
|---:|---|---:|---:|---:|---:|---:|---|
| 6 | `RMSNormKernel<8,bf16>` | `3.97-4.22us` | `8 x 896` | `0.05` | `0.70-0.74%` | `2.4-2.5%` | tiny-grid/launch-latency limited, not HBM-bound |
| 7 | `nvjet_tst_128x8_64x12_2x1_v_bz_splitK_TNT` | `11.84-12.22us` | `72 x 384` | `0.92` | `51-53%` | `15-16%` | skinny GEMM with low waves, low L2 hit, and scoreboard stalls |
| 7 | `cublasLt::splitKreduce_kernel<...>` | `3.04-3.14us` | `66 x 512` | `0.21` | `1.8-1.9%` | `5-6%` | extra reduce launch with very small grid |

Diagnosis: row 6/7 together are not at the H20 limit. The pair is dominated by launch count, intermediate traffic, and skinny-GEMM wave quantization. A fusion has about `488.5us/step` gross upside from deleting row 6, but a custom prologue GEMM must not lose more than about `8us/call` versus cuBLAS qkv_a or the win disappears.

Rejected attempt: qkv_a cuBLASLt exact-shape provider. Standalone contiguous-loop timing improved from `15.119us` to `14.052-14.179us` per GEMM, but the temporary `kimi_tp1_pplx_decode_bench` provider measured only `20.407us -> 20.070us` at 64 iters and `20.407us -> 20.242us` at 256 iters. That is below the `>3%` adoption threshold, so no qkv_a cuBLASLt code is kept.

### Row 8/9: qkv_a_split_norm -> q_b

H20 profile for row 8/9:

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

/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -c 6 \
  -o profile/kimi-attention-row8-row9-h20-baseline/reports/row9_full \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.attention.q_b \
  --out profile/kimi-attention-row8-row9-h20-baseline/row9_ncu.json
```

Event timing:

| Row | Label | Calls | Mean/call | Step latency | Bench roofline |
|---:|---|---:|---:|---:|---|
| 8 | `decode.attention.qkv_a_split_norm` | 61 | `8.217us` | `501.2us` | memory, `12.2GB/s`, `0.25%` HBM |
| 9 | `decode.attention.q_b` | 61 | `17.255us` | `1.053ms` | memory, `17.50TF/s`, `2.201TB/s`, `45.8%` HBM |

NCU details:

| Row | Kernel | Duration | Grid / block | Waves/SM | DRAM | Compute | Main diagnosis |
|---:|---|---:|---:|---:|---:|---:|---|
| 8 | `split_qkv_a_norm_kernel` | `4.77-5.18us` | `8 x 192` | `0.01` | `0.19-0.20%` | `0.37-0.39%` | tiny-grid/launch-latency limited; NCU `LaunchConfiguration` estimates `89.74%` local speedup because only `8` blocks run on `78` SMs |
| 9 | `nvjet_tst_192x8_64x8_2x1_v_bz_TNT` | `12.93-13.34us` | `64 x 384` | `0.82` | `59-61%` | `17%` | memory-bound skinny GEMM; grid is still smaller than SM count and L2 hit is only `2.7-2.8%` |

Diagnosis: row 8 alone is not HBM-bound; it is a tiny launch with too few CTAs to fill H20. Row 9 is the real bandwidth user and is already stronger than qkv_a in DRAM percentage, but it is still constrained by low-batch wave quantization and low reuse. The row 8/9 fusion opportunity is therefore not "make row 8 reach bandwidth peak"; it is to delete or absorb the row-8 launch/intermediate traffic while keeping row 9 at least as fast as the current cuBLAS kernel.

## Candidate Scan

| Candidate | Rows | Feasibility | Evidence | Decision |
|---|---|---|---|---|
| Attention RMSNorm -> qkv_a GEMM prologue | 6-7 | Requires custom GEMM prologue or CUTLASS-style visitor; standard cuBLASLt does not encode full RMSNorm over the input row. | NCU done: row 6 is `8` blocks / `0.05` waves/SM; row 7 is cuBLAS split-K, main GEMM `72` blocks / `0.92` waves/SM / `51-53%` DRAM plus `~3us` reduce. qkv_a cuBLASLt provider was rejected: `0.8-1.7%` bench-provider gain only. | Keep in scan queue, but do not tune RMSNorm alone and do not adopt qkv_a cuBLASLt. Only a true RMSNorm-prologue/custom GEMM path remains worth trying here. |
| qkv_a split/norm cleanup | 8-9 | Possible only inside Kimi MLA custom kernels; row 8 already fuses split plus two norms and feeds both q_b and MLA cache. | NCU done: row 8 is `8` blocks / `0.01` waves/SM / `~0.2%` DRAM, while row 9 is cuBLAS `64` blocks / `0.82` waves/SM / `59-61%` DRAM. | Keep in scan queue only as a true row 8/9 fusion or q_b custom-prologue path. Do not tune row 8 as a standalone bandwidth kernel. |
| shared_gate_up -> shared_swiglu | 21-22 | Requires gated-dual-GEMM custom/CUTLASS/CuTe kernel; cuBLASLt cannot express SwiGLU over the two halves of a single output. | KernelWiki gated-dual-GEMM pattern applies. NCU says row 22 is tiny-grid/latency-bound, not bandwidth-bound. Current cuBLASLt row 21 is strong at `1.505 ms`. | Do not replace cuBLASLt without a full TP1 PPLX bench win. Prototype only if it targets gated-dual-GEMM, not elementwise-only tuning. |
| shared_swiglu -> shared_down prologue | 22-23 | Would require activation transform in the GEMM input path; standard cuBLASLt has no arbitrary BF16 input prologue. | Row 23 is `902.2 us`; row 22 standalone is small, but full bench row is `473.3 us`. | Lower priority than row 21/22; revisit after gated-dual-GEMM result. |
| Dequant into routed Marlin GEMM | 25/27 | Not scan-ready because routed PPLX Marlin rows are estimate-only in the current single-rank bench. | Master ledger marks PPLX W13/W2 as estimate-only; EP comm is excluded but local routed compute still needs an all-rank harness. | Blocked on all-rank PPLX local-compute timing, not on communication optimization. |

## Next Action

Phase 2 is not complete yet. The next useful proof is one of:

- build a true row 6/7 RMSNorm-prologue GEMM prototype only if it can preserve the current qkv_a cuBLAS speed; the cuBLASLt-only qkv_a swap was already rejected, or
- build a row 8/9 custom-prologue q_b prototype only if it deletes the `split_qkv_a_norm` launch/intermediate traffic while preserving `ckv_normed` and `k_rope` outputs for MLA cache, or
- build a small CUTLASS/CuTe gated-dual-GEMM prototype for row 21/22 and compare it against `shared_gate_up + shared_swiglu = 1.978 ms` in the full TP1 PPLX bench.

Do not accept a fusion based on standalone microbench speed alone. The adopted threshold stays the project rule: reproducible H20 improvement above noise, preferably `>3%`, with code plus this scan/master table update in one commit.
