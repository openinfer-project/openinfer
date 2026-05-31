# Kimi-K2 TP1 DP8 EP8 Decode Optimization Master

> **TL;DR:** Master ledger for Kimi-K2 TP1/DP8/EP8 decode optimization on H20. Phase 1 baseline is anchored at per-DP-rank `bs=8`, global `bs~=64`, `ctx=1`: every decode-path operator is listed below with shape, latency, H20 roofline class, and peak gap. Accepted optimizations so far are `shared_gate_up` cuBLASLt (`1.818ms -> 1.505ms`), attention `o_proj` cuBLASLt (`2.715ms -> 2.374ms`), MLA `absorb_q_nope` / `v_up` cuBLASLt strided-batched GEMM (`973.6us -> 748.5us`, `781.0us -> 738.5us`), final argmax split-vocab reduction (`125.3us -> 12.7us`), and router post-GEMM score/topk fusion (`3.655ms -> 3.514ms`). Router fast tensor-op logits path was rejected because TP1 DP8 bs64/o5 token traces changed (`30/64` mismatches). qkv_a standalone cuBLASLt was rejected (`20.407us -> 20.242us`, `0.8%` in TP1 bench), q_b standalone cuBLASLt was rejected (`8.899us -> 8.746us`, `1.0175x` at `batch_size=8`), and shared_down standalone cuBLASLt was rejected (`11.000us -> 10.995us`, `~1.0005x`). PPLX routed local compute rows are still measured with a synthetic expected-local-route provider (`64` routes/rank, `400` padded rows/rank, `recv_capacity=848`): W13 `26.19ms`, W2 `14.21ms` per step. A H20 varied-prompt runtime trace now records near-target route histograms (`active_rows=7/8`, `504` routes/wave): active8 rows show p50/p95/max padded rows per rank of `80/216/336`, so the synthetic `400` row is conservative and must not be treated as real route balance. Phase 2 fusion scan has H20 NCU evidence for row 21/22, row 6/7, and row 8/9; no fusion is accepted yet. EP communication rows are included for path coverage but excluded from optimization.
>
> **Last touched:** 2026-06

## Scope

Target workload:

| Item | Value |
|---|---|
| Model | Kimi-K2 |
| Parallelism | TP1 DP8 EP8 |
| Backend | PPLX EP backend |
| Hardware | H20 |
| Optimization path | Decode |
| Anchor load | per-DP-rank `bs=8`, global `bs~=64` |
| Excluded | EP communication operators: all-to-all / dispatch / combine transport kernels |

Roofline convention:

| Field | Current baseline value |
|---|---:|
| H20 BF16 peak used by report | `148 TFLOP/s` |
| H20 HBM peak used by report | `4.8 TB/s` |
| Ridge point | `30.83 flop/byte` |

The table uses the H20 TP1 PPLX bench report's `%peak` and roofline labels. Per-kernel NCU reports may refine a row's peak numbers after profiling; when they do, update this master table in the same commit as the accepted optimization.

## Workflow Contract

1. Phase 1: establish the complete decode-path baseline table, then commit the baseline milestone.
2. Phase 2: scan adjacent operators for fusion first. Candidate directions: RMSNorm to GEMM prologue, GEMM plus bias/activation epilogue, MoE gate/up plus SwiGLU, and dequant fused into GEMM.
3. Phase 3: optimize individual kernels after fusion scan. Each non-communication kernel gets its own `<kernel_name>_report.md` with KernelWiki findings, NCU findings, attempts, and final stop/adopt conclusion.
4. Commit only reproducible wins above noise, preferably `>3%`. One commit contains exactly one accepted optimization plus its report and this master table update.
5. If profile evidence conflicts with hints or intuition, profile wins.

Commit message template:

```text
opt(<kernel>): <what changed> | <before> -> <after> (<speedup>) | bound=<mem|compute>

- KernelWiki: <one-line conclusion>
- NCU: <one-line bottleneck>
- Verification: bs=8/rank, global=64, H20
```

## Baseline Evidence

Baseline raw report:

```bash
cargo run --release -p pegainfer-kimi-k2 --features kernel-report \
  --bin kimi_tp1_pplx_decode_bench -- \
  --iters 32 --format json \
  --out target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json
```

H20 check:

```bash
cargo check --release -p pegainfer-kimi-k2 \
  --features kernel-report --bin kimi_tp1_pplx_decode_bench
```

Result: passed on `h20-100`, `sm_90`. The anchor table below uses `active_rows=8`, `ctx_len=1`, `arena_rows=8` from the pre-optimization baseline artifact. Longer-context rows are tracked by the bench artifact and should be promoted into this master table when optimizing attention decode/cache traffic.

PPLX routed local compute rows still use the synthetic expected-local-route provider for latency until a clean steady-state all-rank provider replaces it: for `bs=8/rank`, global `bs~=64`, each EP rank is modeled with `64` expected local routes, `400` expected padded rows, and `recv_capacity=848`. The runtime trace path records `kimi_pplx_route_histogram` rows after `dispatch_recv`, including `recv_tokens_per_expert`, active local experts, max per-expert count, padded rows, and routing block size. The first H20 varied-prompt trace is `target/kernel_reports/kimi-k2/tp1-dp8-pplx-route-hist-bs64-kv2-varied.json`: it includes two `active_rows=1` admission waves and two near-target waves where rank0 has `active_rows=7` and ranks1-7 have `active_rows=8` (`504` routes/wave). EP dispatch/combine transport remains excluded from optimization.

Accepted optimizations:

| Kernel | Report | Baseline | Current | Speedup | Bound | Verification |
|---|---|---:|---:|---:|---|---|
| `decode.moe.shared_gate_up` | `shared_gate_up_report.md` | `1.818 ms` | `1.505 ms` | `1.21x` | memory | `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-cublaslt-bs3-bs8.json`, `bs=8,ctx=1` |
| `decode.attention.o_proj` | `attention_o_proj_report.md` | `2.715 ms` | `2.374 ms` | `1.14x` | memory | `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json`, `profile/kimi-attention-o-proj-h20-cublaslt/`, `bs=8,ctx=1` |
| `decode.attention.absorb_q_nope` | `attention_absorb_q_nope_report.md` | `973.6 us` | `748.5 us` | `1.30x` | memory | `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-mla-cublaslt-bs8-lazy128.json`, `profile/kimi-mla-cublaslt-h20/`, `bs=8,ctx=1` |
| `decode.attention.v_up` | `attention_v_up_report.md` | `781.0 us` | `738.5 us` | `1.06x` | memory | `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-mla-cublaslt-bs8-lazy128.json`, `profile/kimi-mla-cublaslt-h20/`, `bs=8,ctx=1` |
| `decode.final.argmax` | `final_argmax_report.md` | `125.3 us` | `12.7 us` | `9.85x` | memory | `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-argmax-split-bs8.json`, `profile/kimi-final-argmax-h20-baseline/`, TP1 DP8 bs64/o5 token A/B `0/64` mismatch |
| `decode.moe.router` | `kimi_router_noaux_tc_report.md` | `3.655 ms` | `3.514 ms` | `1.04x` | control | `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-router-fused-bs8.json`, `profile/kimi-router-fused-h20/`, TP1 DP8 bs64/o5 token A/B `0/64` mismatch |

## Commit Queue

| Order | Commit kind | Must include | Must not include |
|---:|---|---|---|
| 1 | Phase 1 baseline milestone | TP1 PPLX decode bench binary/provider infrastructure, this master table, `tp1-pplx-decode-bench.md`, and `docs/index.md` entry. | cuBLASLt shared_gate_up production switch or any other accepted optimization. |
| 2 | `opt(shared_gate_up)` | Kimi-specific cuBLASLt code path, `shared_gate_up_report.md`, and this master table's accepted-result update. | unrelated bench scaffolding or other kernel changes. |
| 3 | `opt(attention_o_proj)` | Kimi-specific cuBLASLt o_proj code path, `attention_o_proj_report.md`, and this master table's accepted-result update. | fusion-scan experiments or other kernel changes. |
| 4 | `opt(mla_batched_gemm)` | Kimi TP1 cuBLASLt strided-batched path for `absorb_q_nope` and `v_up`, both reports, token A/B note, and this master table's accepted-result update. | router math-mode changes, fusion experiments, or PPLX comm changes. |
| 5 | `opt(final_argmax)` | Split-vocab CUDA argmax path, `final_argmax_report.md`, and this master table's accepted-result update. | q_b rejection docs or other kernel experiments. |
| 6 | `opt(router_post_gemm)` | Fused post-GEMM router score/topk selector, `kimi_router_noaux_tc_report.md`, and this master table's accepted-result update. | logits GEMM math-mode changes or other MoE kernels. |

The baseline milestone is already split from the first optimization. Future entries should follow the same pattern: one measured win, one code/report/master-table commit.

## Decode Path Master Table

Columns:

- **Step latency** is the aggregate time per decode step for that row's `calls`.
- **Per call** is `step latency / calls`.
- **Peak gap** is against the row's active roofline limit. For memory-bound rows this is HBM peak gap; for compute-bound rows it is BF16 peak gap.
- **Scope** is `measured`, `measured synthetic`, `estimate-only`, or `excluded EP comm`.

| # | Stage | Op | Kernel/provider | Calls | Shape / dtype | Step latency | Per call | Throughput | H20 roofline | Peak gap | Scope |
|---:|---|---|---|---:|---|---:|---:|---:|---|---|---|
| 1 | embedding | `decode.embedding` | `embedding_batch_vocab_shard` | 1 | vocab=163840, hidden=7168, rows=8, BF16 | 7.2 us | 7.2 us | 31.7 GB/s | memory | 0.7% / gap 99.3% | measured |
| 2 | dense | `decode.dense.gate_up` | `gemm_dm_typed_to_hs_graphsafe` | 1 | rows=8, out=36864, in=7168, BF16 | 147.9 us | 147.9 us | 28.58 TF/s | memory | 74.5% / gap 25.5% | measured |
| 3 | dense | `decode.dense.swiglu` | `silu_mul_hs_fused_into` | 1 | hidden=18432, batch=8, BF16 | 7.8 us | 7.8 us | 113.4 GB/s | memory | 2.4% / gap 97.6% | measured |
| 4 | dense | `decode.dense.down` | `gemm_dm_hs_to_typed_graphsafe` | 1 | rows=8, out=7168, in=18432, BF16 | 85.3 us | 85.3 us | 24.77 TF/s | memory | 64.6% / gap 35.4% | measured |
| 5 | dense | `decode.dense.residual_add` | `add_batch` | 1 | hidden=7168, batch=8, BF16 | 6.8 us | 6.8 us | 50.5 GB/s | memory | 1.1% / gap 98.9% | measured |
| 6 | attention | `decode.attention.input_norm` | `rms_norm_batch` | 61 | elems=57344, BF16 | 476.3 us | 7.8 us | 0.04 TF/s | memory | 1.2% / gap 98.8% | measured |
| 7 | attention | `decode.attention.qkv_a` | `gemm_graphsafe` | 61 | rows=8, out=2112, in=7168, BF16 | 1.256 ms | 20.6 us | 11.77 TF/s | memory | 30.8% / gap 69.2% | measured |
| 8 | attention | `decode.attention.qkv_a_split_norm` | `kimi_mla_split_qkv_a_norm` | 61 | elems=16896, BF16 | 501.1 us | 8.2 us | 0.01 TF/s | memory | 0.3% / gap 99.7% | measured |
| 9 | attention | `decode.attention.q_b` | `gemm_dm_typed_to_hs_graphsafe` | 61 | rows=8, out=12288, in=1536, BF16 | 1.052 ms | 17.2 us | 17.51 TF/s | memory | 45.9% / gap 54.1% | measured |
| 10 | attention | `decode.attention.rope_split` | `kimi_mla_rope_split_decode_rt` | 61 | elems=98816, BF16 | 441.8 us | 7.2 us | 0.03 TF/s | memory | 1.1% / gap 98.9% | measured |
| 11 | attention | `decode.attention.absorb_q_nope` | `kimi_mla_absorb_q_nope_rt` cuBLASLt TP1 path | 61 | rows=8, out=32768, in=128, BF16 | 748.5 us | 12.3 us | 5.47 TF/s | memory | 15.1% / gap 84.9% | measured |
| 12 | attention | `decode.attention.paged_kv_append` | `kimi_mla_paged_kv_append` | 61 | elems=4608, BF16 | - | - | - | estimate-only | - | estimate-only |
| 13 | attention | `decode.attention.flashinfer_mla_decode` | `kimi_flashinfer_batch_decode_mla_rt` | 61 | elems=512, BF16 | 624.6 us | 10.2 us | 0.11 TF/s | memory | 4.4% / gap 95.6% | measured |
| 14 | attention | `decode.attention.v_up` | `kimi_mla_v_up_rt` cuBLASLt TP1 path | 61 | rows=8, out=8192, in=512, BF16 | 738.5 us | 12.1 us | 5.54 TF/s | memory | 14.7% / gap 85.3% | measured |
| 15 | attention | `decode.attention.o_proj` | `kimi_o_proj_cublaslt` | 61 | rows=8, out=7168, in=8192, BF16 | 2.374 ms | 38.9 us | 24.15 TF/s | memory | 63.0% / gap 37.0% | measured |
| 16 | attention | `decode.attention.post_attn_add_norm` | `fused_add_rms_norm_round_batch` | 61 | elems=57344, BF16 | 530.0 us | 8.7 us | 0.05 TF/s | memory | 1.6% / gap 98.4% | measured |
| 17 | final | `decode.final.norm` | `rms_norm_batch` | 1 | elems=57344, BF16 | 8.1 us | 8.1 us | 0.04 TF/s | memory | 1.2% / gap 98.8% | measured |
| 18 | final | `decode.final.lm_head` | `gemm_graphsafe` | 1 | rows=8, out=163840, in=7168, BF16 | 542.7 us | 542.7 us | 34.62 TF/s | memory | 90.3% / gap 9.7% | measured |
| 19 | final | `decode.final.argmax` | `argmax_batch_bf16_split` | 1 | elems=1310720, BF16, tile=4096 | 12.7 us | 12.7 us | 206.0 GB/s | memory | 4.3% / gap 95.7% | measured |
| 20 | moe_router | `decode.moe.router` | `kimi_router_noaux_tc` fused selector | 60 | rows=8, experts=384, topk=8, BF16/F32 | 3.514 ms | 58.6 us | 96.6 GB/s | control | - | measured |
| 21 | moe_shared | `decode.moe.shared_gate_up` | `kimi_shared_gate_up_cublaslt` | 60 | rows=8, out=4096, in=7168, BF16 | 1.505 ms | 25.1 us | 18.72 TF/s | memory | 48.9% / gap 51.1% | measured |
| 22 | moe_shared | `decode.moe.shared_swiglu` | `silu_mul_hs_fused_into` | 60 | rows=8, gate_up=4096, out=2048, BF16 | 410.2 us | 6.8 us | 14.4 GB/s | memory | 0.3% / gap 99.7% | measured |
| 23 | moe_shared | `decode.moe.shared_down` | `gemm_dm_hs_to_typed_graphsafe` | 60 | rows=8, out=7168, in=2048, BF16 | 897.1 us | 15.0 us | 15.71 TF/s | memory | 41.1% / gap 58.9% | measured |
| 24 | moe_pplx_compute | `decode.moe.pplx_build_marlin_routing` | `kimi_pplx_build_marlin_routing_on_stream` | 60 | recv_capacity=848, expected local routes=64, local_experts=48 | 569.3 us | 9.5 us | 0.42 GB/s | control | - | measured synthetic |
| 25 | moe_pplx_compute | `decode.moe.pplx_marlin_w13` | `kimi_marlin_wna16_pplx_w13_gemm` | 60 | work rows=400, recv_capacity=848, out=4096, in=7168, WNA16 INT4 weights + BF16 activations | 26.186 ms | 436.4 us | 53.82 TF/s / 1.84 TB/s | memory | 38.3% / gap 61.7% | measured synthetic |
| 26 | moe_pplx_compute | `decode.moe.pplx_swiglu` | `kimi_marlin_w13_swiglu_pplx` | 60 | work rows=400, recv_capacity=848, gate_up=4096, out=2048, BF16 | 848.1 us | 14.1 us | 347.7 GB/s | memory | 7.2% / gap 92.8% | measured synthetic |
| 27 | moe_pplx_compute | `decode.moe.pplx_marlin_w2` | `kimi_marlin_wna16_pplx_w2_gemm` | 60 | work rows=400, recv_capacity=848, out=7168, in=2048, WNA16 INT4 weights + BF16 activations | 14.208 ms | 236.8 us | 49.60 TF/s / 1.70 TB/s | memory | 35.5% / gap 64.5% | measured synthetic |
| 28 | moe_pplx_compute | `decode.moe.residual_add_scaled` | `kimi_residual_add_scaled_f32` | 60 | rows=8, hidden=7168, BF16 + F32 routed | 408.3 us | 6.8 us | 84.3 GB/s | memory | 1.8% / gap 98.2% | measured |
| 29 | moe_pplx_comm | `decode.moe.pplx.dispatch_send` | `dispatch_send` | 60 | elems=458752, BF16 payload + route metadata | - | - | - | comm | - | excluded EP comm |
| 30 | moe_pplx_comm | `decode.moe.pplx.dispatch_recv` | `dispatch_recv` | 60 | elems=6078464, BF16 payload + route metadata | - | - | - | comm | - | excluded EP comm |
| 31 | moe_pplx_comm | `decode.moe.pplx.combine_send` | `combine_send` | 60 | elems=6078464, BF16 expert output | - | - | - | comm | - | excluded EP comm |
| 32 | moe_pplx_comm | `decode.moe.pplx.combine_recv` | `combine_recv` | 60 | elems=516096, BF16 rows to F32 routed output | - | - | - | comm | - | excluded EP comm |

## Current Bottleneck Order

Measured rows sorted by step latency at `bs=8,ctx=1`:

| Rank | Op | Step latency | Current direction |
|---:|---|---:|---|
| 1 | PPLX `decode.moe.pplx_marlin_w13` | 26.186 ms | `pplx_marlin_compute_report.md`: current focus. Synthetic expected-local-route baseline says memory-bound at `38.3%` HBM; NCU Marlin kernel shows mixed pressure (`234` CTAs, `1.0` waves/SM, `58.7%` SM, `34.7%` DRAM). H20 varied-prompt trace says active8 padded rows/rank are p50 `80`, p95 `216`, max `336`, below the synthetic `400`; next step is a clean steady full-rank trace/provider before changing Marlin parameters. |
| 2 | PPLX `decode.moe.pplx_marlin_w2` | 14.208 ms | `pplx_marlin_compute_report.md`: current focus. Synthetic expected-local-route baseline says memory-bound at `35.5%` HBM; NCU Marlin kernel shows mixed pressure (`234` CTAs, `1.0` waves/SM, `56.8%` SM, `32.6%` DRAM). Same route-hist caveat as W13: current latency row is conservative synthetic local compute, not confirmed serving shape. |
| 3 | `decode.moe.router` | 3.514 ms | Post-GEMM score/topk fusion accepted; NCU still says small-grid/control limited (`8` CTAs, `0.03` waves/SM). Fast `CUBLAS_COMPUTE_32F` logits GEMM gave `3.655ms -> 1.687ms`, but remains rejected due `30/64` TP1 DP8 bs64/o5 token-trace mismatches. |
| 4 | attention `o_proj` (`kimi_o_proj_cublaslt`) | 2.374 ms | cuBLASLt accepted; NCU still shows memory/skinny-grid limit (`56` CTAs, `73.9-75.9%` DRAM). Treat this as the new baseline for any future fusion/custom GEMM work. |
| 5 | `decode.moe.shared_gate_up` | 1.519 ms | cuBLASLt accepted; next work should treat this as the baseline for row 21/22 fusion scans. |
| 6 | attention `decode.attention.qkv_a` (`gemm_graphsafe`) | 1.262 ms | NCU done: cuBLAS split-K skinny GEMM, main kernel `72` blocks/`0.92` waves/SM/`51-53%` DRAM plus `~3us` split-K reduce. cuBLASLt exact-shape provider was rejected because TP1 bench gain was only `0.8-1.7%`. |
| 7 | attention `q_b_proj` | 1.057 ms | `q_b_proj_report.md`: NCU says cuBLAS skinny GEMM is memory/low-wave limited (`64` blocks/`0.82` waves/SM/`59-61%` DRAM, low L2 hit). Standalone cuBLASLt exact-shape sweep was rejected at target `batch_size=8` (`8.899us -> 8.746us`, `1.0175x`); row 8/9 fusion would need a custom prologue while preserving MLA cache outputs. |
| 8 | `decode.moe.shared_down` | 897.1 us | `shared_down_report.md`: NCU says memory-bound skinny GEMM (`56` CTAs, `0.93` waves/SM, `55.94%` DRAM, `82.37%` no eligible). Standalone cuBLASLt exact-shape sweep was rejected (`11.000us -> 10.995us`); only a real `shared_swiglu -> shared_down` fusion remains plausible. |
| 9 | PPLX `decode.moe.pplx_swiglu` | 848.1 us | NCU says `swiglu_w13_pplx_kernel` is not the dominant PPLX cost (`10.62us`, `6784` CTAs, `55.4%` SM, `6.3%` DRAM). |
| 10 | attention `absorb_q_nope` (`kimi_mla_absorb_q_nope_rt` cuBLASLt TP1 path) | 748.5 us | cuBLASLt accepted for `local_heads=64,batch_size<=8`; NCU still shows low-wave memory limit (`78` CTAs, `1.00` waves/SM, `28.4%` DRAM read peak). |

Remaining estimate-only local compute is KV append. PPLX routed local compute now has a synthetic provider for kernel isolation, but an all-rank harness is still required before turning those rows into route-imbalance or serving-throughput claims. EP communication rows remain excluded from optimization but must stay visible in this table.

## Fusion Scan Queue

Start Phase 2 from the rows above:

| Candidate | Rows touched | Why it is plausible | First proof required |
|---|---|---|---|
| Attention RMSNorm -> qkv_a GEMM prologue | rows 6-7 | Norm is launch/memory heavy and immediately feeds a skinny GEMM. | NCU done in `tp1-dp8-ep8-fusion-scan.md`: row 6 is tiny-grid (`8` blocks, `0.05` waves/SM), row 7 is cuBLAS split-K skinny GEMM. qkv_a cuBLASLt exact-shape swap was measured and rejected; remaining path is custom RMSNorm-prologue GEMM only if it wins full TP1 PPLX bench. |
| qkv_a split/norm cleanup | row 8 with row 9 input | Split qkv_a and normalize q_lora/ckv before q_b/MLA cache. | NCU done in `tp1-dp8-ep8-fusion-scan.md`: row 8 is tiny-grid (`8` blocks, `0.01` waves/SM, `~0.2%` DRAM), row 9 is a memory-bound cuBLAS skinny GEMM (`64` blocks, `0.82` waves/SM, `59-61%` DRAM). Only a true fusion/custom q_b prologue is worth trying; standalone row-8 bandwidth tuning is not. |
| MoE shared gate_up + SwiGLU | rows 21-22 | Gate/up output is consumed only by SwiGLU; avoids writing/reading 4096 BF16 per row per layer. | Initial NCU done in `tp1-dp8-ep8-fusion-scan.md`: row 22 is tiny-grid/latency-bound; accepted fusion requires gated-dual-GEMM beating cuBLASLt in the full TP1 PPLX bench. |
| Shared SwiGLU + down prologue | rows 22-23 | Activation output is consumed only by down GEMM. | cuBLASLt epilogue/prologue support or CUTLASS prototype; correctness gate on BF16 rounding. |
| Dequant into routed Marlin GEMM | rows 25 and 27 | INT4/WNA16 path should not materialize dequant separately. | Provider/all-rank harness first; then NCU. |

No fusion is accepted without H20 measured improvement over noise and unchanged correctness envelope.

## Kernel Report Queue

Each report file must live under `docs/models/kimi-k2/` and use this structure:

```markdown
# <kernel_name> Report

> **TL;DR:** ...
>
> **Last touched:** YYYY-MM

## KernelWiki Conclusion
## NCU Conclusion
## Attempts
## Final Conclusion
```

Initial report targets:

| Priority | Kernel/report | Reason |
|---:|---|---|
| 1 | `kimi_router_noaux_tc_report.md` | Exists; fast tensor-op logits GEMM was rejected by TP1 DP8 token-trace mismatch, so future work must preserve pedantic router accuracy. |
| 2 | `attention_o_proj_report.md` | Exists for the accepted cuBLASLt optimization; revisit only if a fused/custom row beats cuBLASLt in full TP1 PPLX bench. |
| 3 | `shared_gate_up_report.md` | Exists for the accepted cuBLASLt optimization; revisit only if row 21/22 fusion or a stronger custom kernel beats cuBLASLt. |
| 4 | `qkv_a_proj_report.md` | Exists; standalone cuBLASLt exact-shape provider was rejected because the full TP1 bench improved only `0.8-1.7%`. Future work should be true row 6/7 RMSNorm-prologue/custom GEMM fusion. |
| 5 | `q_b_proj_report.md` | Exists; standalone cuBLASLt exact-shape sweep was rejected because `batch_size=8` improved only `1.0175x`. Future work should be true row 8/9 fusion/custom prologue. |
| 6 | `shared_down_report.md` | Exists; standalone cuBLASLt exact-shape sweep was rejected because `batch_size=8` improved only `~1.0005x`. Future work should be true row 22/23 fusion. |
| 7 | `pplx_marlin_compute_report.md` | Exists; synthetic expected-local-route provider measures PPLX W13/SwiGLU/W2/routing without EP transport. Runtime route histogram trace is available; the next artifact should replace synthetic rows with H20 all-rank recv histograms before optimization. |
| 8 | `attention_absorb_q_nope_report.md` | Exists for the accepted cuBLASLt strided-batched MLA optimization. |
| 9 | `attention_v_up_report.md` | Exists for the accepted cuBLASLt strided-batched MLA optimization. |
| 10 | `final_argmax_report.md` | Exists for the accepted split-vocab CUDA argmax optimization. |
