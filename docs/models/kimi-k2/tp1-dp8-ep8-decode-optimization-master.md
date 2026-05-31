# Kimi-K2 TP1 DP8 EP8 Decode Optimization Master

> **TL;DR:** Master ledger for Kimi-K2 TP1/DP8/EP8 decode optimization on H20. Phase 1 baseline is anchored at per-DP-rank `bs=8`, global `bs~=64`, `ctx=1`: every decode-path operator is listed below with shape, latency, H20 roofline class, and peak gap. The first accepted optimization is `shared_gate_up` cuBLASLt: `1.818ms -> 1.505ms` per 60 MoE layers. Phase 2 fusion scan is tracked in `tp1-dp8-ep8-fusion-scan.md`; EP communication rows are included for path coverage but excluded from optimization.
>
> **Last touched:** 2026-05

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

Accepted optimizations:

| Kernel | Report | Baseline | Current | Speedup | Bound | Verification |
|---|---|---:|---:|---:|---|---|
| `decode.moe.shared_gate_up` | `shared_gate_up_report.md` | `1.818 ms` | `1.505 ms` | `1.21x` | memory | `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-cublaslt-bs3-bs8.json`, `bs=8,ctx=1` |

## Commit Queue

| Order | Commit kind | Must include | Must not include |
|---:|---|---|---|
| 1 | Phase 1 baseline milestone | TP1 PPLX decode bench binary/provider infrastructure, this master table, `tp1-pplx-decode-bench.md`, and `docs/index.md` entry. | cuBLASLt shared_gate_up production switch or any other accepted optimization. |
| 2 | `opt(shared_gate_up)` | Kimi-specific cuBLASLt code path, `shared_gate_up_report.md`, and this master table's accepted-result update. | unrelated bench scaffolding or other kernel changes. |

The baseline milestone is already split from the first optimization. Future entries should follow the same pattern: one measured win, one code/report/master-table commit.

## Decode Path Master Table

Columns:

- **Step latency** is the aggregate time per decode step for that row's `calls`.
- **Per call** is `step latency / calls`.
- **Peak gap** is against the row's active roofline limit. For memory-bound rows this is HBM peak gap; for compute-bound rows it is BF16 peak gap.
- **Scope** is `measured`, `estimate-only`, or `excluded EP comm`.

| # | Stage | Op | Kernel/provider | Calls | Shape / dtype | Step latency | Per call | Throughput | H20 roofline | Peak gap | Scope |
|---:|---|---|---|---:|---|---:|---:|---:|---|---|---|
| 1 | embedding | `decode.embedding` | `embedding_batch_vocab_shard` | 1 | vocab=163840, hidden=7168, rows=8, BF16 | 7.2 us | 7.2 us | 31.7 GB/s | memory | 0.7% / gap 99.3% | measured |
| 2 | dense | `decode.dense.gate_up` | `gemm_dm_typed_to_hs_graphsafe` | 1 | rows=8, out=36864, in=7168, BF16 | 147.9 us | 147.9 us | 28.58 TF/s | memory | 74.5% / gap 25.5% | measured |
| 3 | dense | `decode.dense.swiglu` | `silu_mul_hs_fused_into` | 1 | hidden=18432, batch=8, BF16 | 7.8 us | 7.8 us | 113.4 GB/s | memory | 2.4% / gap 97.6% | measured |
| 4 | dense | `decode.dense.down` | `gemm_dm_hs_to_typed_graphsafe` | 1 | rows=8, out=7168, in=18432, BF16 | 85.3 us | 85.3 us | 24.77 TF/s | memory | 64.6% / gap 35.4% | measured |
| 5 | dense | `decode.dense.residual_add` | `add_batch` | 1 | hidden=7168, batch=8, BF16 | 6.8 us | 6.8 us | 50.5 GB/s | memory | 1.1% / gap 98.9% | measured |
| 6 | attention | `rms_norm_batch` | `rms_norm_batch` | 61 | elems=57344, BF16 | 476.3 us | 7.8 us | 0.04 TF/s | memory | 1.2% / gap 98.8% | measured |
| 7 | attention | `gemm_graphsafe` | `gemm_graphsafe` | 61 | rows=8, out=2112, in=7168, BF16 | 1.256 ms | 20.6 us | 11.77 TF/s | memory | 30.8% / gap 69.2% | measured |
| 8 | attention | `kimi_mla_split_qkv_a_norm` | `kimi_mla_split_qkv_a_norm` | 61 | elems=16896, BF16 | 501.1 us | 8.2 us | 0.01 TF/s | memory | 0.3% / gap 99.7% | measured |
| 9 | attention | `gemm_dm_typed_to_hs_graphsafe` | `gemm_dm_typed_to_hs_graphsafe` | 61 | rows=8, out=12288, in=1536, BF16 | 1.052 ms | 17.2 us | 17.51 TF/s | memory | 45.9% / gap 54.1% | measured |
| 10 | attention | `kimi_mla_rope_split_decode_rt` | `kimi_mla_rope_split_decode_rt` | 61 | elems=98816, BF16 | 441.8 us | 7.2 us | 0.03 TF/s | memory | 1.1% / gap 98.9% | measured |
| 11 | attention | `kimi_mla_absorb_q_nope_rt` | `kimi_mla_absorb_q_nope_rt` | 61 | rows=8, out=32768, in=128, BF16 | 955.0 us | 15.7 us | 4.29 TF/s | memory | 11.9% / gap 88.1% | measured |
| 12 | attention | `kimi_mla_paged_kv_append` | `kimi_mla_paged_kv_append` | 61 | elems=4608, BF16 | - | - | - | estimate-only | - | estimate-only |
| 13 | attention | `kimi_flashinfer_batch_decode_mla_rt` | `kimi_flashinfer_batch_decode_mla_rt` | 61 | elems=512, BF16 | 624.6 us | 10.2 us | 0.11 TF/s | memory | 4.4% / gap 95.6% | measured |
| 14 | attention | `kimi_mla_v_up_rt` | `kimi_mla_v_up_rt` | 61 | rows=8, out=8192, in=512, BF16 | 771.3 us | 12.6 us | 5.31 TF/s | memory | 14.1% / gap 85.9% | measured |
| 15 | attention | `gemm_dm_hs_to_typed_graphsafe` | `gemm_dm_hs_to_typed_graphsafe` | 61 | rows=8, out=7168, in=8192, BF16 | 2.715 ms | 44.5 us | 21.11 TF/s | memory | 55.1% / gap 44.9% | measured |
| 16 | attention | `fused_add_rms_norm_round_batch` | `fused_add_rms_norm_round_batch` | 61 | elems=57344, BF16 | 530.0 us | 8.7 us | 0.05 TF/s | memory | 1.6% / gap 98.4% | measured |
| 17 | final | `rms_norm_batch` | `rms_norm_batch` | 1 | elems=57344, BF16 | 8.1 us | 8.1 us | 0.04 TF/s | memory | 1.2% / gap 98.8% | measured |
| 18 | final | `gemm_graphsafe` | `gemm_graphsafe` | 1 | rows=8, out=163840, in=7168, BF16 | 542.7 us | 542.7 us | 34.62 TF/s | memory | 90.3% / gap 9.7% | measured |
| 19 | final | `argmax_batch_bf16` | `argmax_batch_bf16` | 1 | elems=1310720, BF16 | 125.3 us | 125.3 us | 20.9 GB/s | memory | 0.4% / gap 99.6% | measured |
| 20 | moe_router | `decode.moe.router` | `kimi_router_noaux_tc` | 60 | rows=8, experts=384, topk=8, BF16/F32 | 3.687 ms | 61.4 us | 92.1 GB/s | control | - | measured |
| 21 | moe_shared | `decode.moe.shared_gate_up` | `kimi_shared_gate_up_cublaslt` | 60 | rows=8, out=4096, in=7168, BF16 | 1.505 ms | 25.1 us | 18.72 TF/s | memory | 48.9% / gap 51.1% | measured |
| 22 | moe_shared | `decode.moe.shared_swiglu` | `silu_mul_hs_fused_into` | 60 | rows=8, gate_up=4096, out=2048, BF16 | 410.2 us | 6.8 us | 14.4 GB/s | memory | 0.3% / gap 99.7% | measured |
| 23 | moe_shared | `decode.moe.shared_down` | `gemm_dm_hs_to_typed_graphsafe` | 60 | rows=8, out=7168, in=2048, BF16 | 904.1 us | 15.1 us | 15.59 TF/s | memory | 40.8% / gap 59.2% | measured |
| 24 | moe_pplx_compute | `decode.moe.pplx_build_marlin_routing` | `kimi_pplx_build_marlin_routing_on_stream` | 60 | recv_capacity=848, local_experts=48 | - | - | - | control | - | estimate-only |
| 25 | moe_pplx_compute | `decode.moe.pplx_marlin_w13` | `kimi_marlin_wna16_pplx_w13_gemm` | 60 | rows=848, out=4096, in=7168, WNA16 INT4 weights + BF16 activations | - | - | - | estimate-only | - | estimate-only |
| 26 | moe_pplx_compute | `decode.moe.pplx_swiglu` | `kimi_marlin_w13_swiglu_pplx` | 60 | rows=848, gate_up=4096, out=2048, BF16 | - | - | - | estimate-only | - | estimate-only |
| 27 | moe_pplx_compute | `decode.moe.pplx_marlin_w2` | `kimi_marlin_wna16_pplx_w2_gemm` | 60 | rows=848, out=7168, in=2048, WNA16 INT4 weights + BF16 activations | - | - | - | estimate-only | - | estimate-only |
| 28 | moe_pplx_compute | `decode.moe.residual_add_scaled` | `kimi_residual_add_scaled_f32` | 60 | rows=8, hidden=7168, BF16 + F32 routed | 408.3 us | 6.8 us | 84.3 GB/s | memory | 1.8% / gap 98.2% | measured |
| 29 | moe_pplx_comm | `decode.moe.pplx.dispatch_send` | `dispatch_send` | 60 | elems=458752, BF16 payload + route metadata | - | - | - | comm | - | excluded EP comm |
| 30 | moe_pplx_comm | `decode.moe.pplx.dispatch_recv` | `dispatch_recv` | 60 | elems=6078464, BF16 payload + route metadata | - | - | - | comm | - | excluded EP comm |
| 31 | moe_pplx_comm | `decode.moe.pplx.combine_send` | `combine_send` | 60 | elems=6078464, BF16 expert output | - | - | - | comm | - | excluded EP comm |
| 32 | moe_pplx_comm | `decode.moe.pplx.combine_recv` | `combine_recv` | 60 | elems=516096, BF16 rows to F32 routed output | - | - | - | comm | - | excluded EP comm |

## Current Bottleneck Order

Measured rows sorted by step latency at `bs=8,ctx=1`:

| Rank | Op | Step latency | Current direction |
|---:|---|---:|---|
| 1 | `decode.moe.router` | 3.687 ms | Profile first: likely control/launch plus router scoring; correctness-sensitive. |
| 2 | attention `o_proj` (`gemm_dm_hs_to_typed_graphsafe`) | 2.715 ms | cuBLASLt skinny GEMM baseline; candidate for RMSNorm/GEMM sequence review only after profile. |
| 3 | `decode.moe.shared_gate_up` | 1.505 ms | cuBLASLt accepted; next work should treat this as the baseline for row 21/22 fusion scans. |
| 4 | attention `fused_qkv_a_proj` (`gemm_graphsafe`) | 1.256 ms | cuBLASLt baseline and prologue fusion candidate. |
| 5 | attention `q_b_proj` | 1.052 ms | cuBLASLt baseline and skinny GEMM profile. |
| 6 | attention `absorb_q_nope` | 955.0 us | Small-K GEMM, profile wave/grid utilization. |
| 7 | `decode.moe.shared_down` | 904.1 us | cuBLASLt baseline; possible shared_gate/SwiGLU/down sequence review. |
| 8 | attention `v_up` | 771.3 us | Small-K GEMM, profile wave/grid utilization. |
| 9 | MLA decode | 624.6 us at `ctx=1` | Context-sensitive; use longer ctx rows for real cache-traffic optimization. |
| 10 | `lm_head` | 542.7 us | Already near memory limit by report peak; likely lower priority unless fused top1/logits path changes. |

Estimate-only local compute rows (`pplx_marlin_w13`, `pplx_swiglu`, `pplx_marlin_w2`, PPLX routing, KV append) need providers or all-rank harness coverage before they can be ranked honestly. EP communication rows remain excluded from optimization but must stay visible in this table.

## Fusion Scan Queue

Start Phase 2 from the rows above:

| Candidate | Rows touched | Why it is plausible | First proof required |
|---|---|---|---|
| Attention RMSNorm -> qkv_a GEMM prologue | rows 6-7 | Norm is launch/memory heavy and immediately feeds a skinny GEMM. | NCU on both kernels plus cuBLASLt/prologue feasibility check. |
| qkv_a split/norm cleanup | row 8 with row 9 input | Split qkv_a and normalize q_lora/ckv before q_b/MLA cache. | Confirm memory traffic and launch overhead dominate. |
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
| 1 | `kimi_router_noaux_tc_report.md` | Largest measured row and currently classified as control. |
| 2 | `attention_o_proj_report.md` | Largest measured GEMM row; skinny GEMM on H20 should be checked against cuBLASLt. |
| 3 | `shared_gate_up_report.md` | Exists for the accepted cuBLASLt optimization; revisit only if row 21/22 fusion or a stronger custom kernel beats cuBLASLt. |
| 4 | `qkv_a_proj_report.md` | Large per-layer GEMM and fusion candidate with preceding norm. |
| 5 | `q_b_proj_report.md` | Skinny GEMM with nontrivial total cost. |
| 6 | `pplx_marlin_compute_report.md` | Estimate-only rows block honest MoE ranking; needs provider/harness work, not EP comm optimization. |
