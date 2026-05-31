# Attention Paged KV Append Report

> **TL;DR:** `decode.attention.paged_kv_append` now has a TP1 PPLX measured provider instead of an estimate-only row. The provider uses the production decode arena page shape (`page_size=16`, `128` pages/request, page base `request_idx * 128`) and reports `BoundKind::Control`, so the bench no longer fabricates an HBM peak percentage for this tiny append. H20 production-metadata bench is `7.07us/call`, `431.0us` per 61 attention layers at `ctx=1`, `achieved_gbps=2.63`, `roofline_bound=control`, and `roofline_peak_pct=null`; ctx sweep through the arena capacity (`1/128/1024/2048`) stays `7.07-7.36us/call`. The earlier H20 run used compact page metadata and remains directional NCU evidence only: it measured `7.34us/call` and NCU showed `78 x 256`, `0.12` waves/SM, `0.09%` DRAM, `11.70%` achieved occupancy, `97.90%` no eligible. Production-metadata NCU is still pending because `ncu --version` currently times out on `h20-100`.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `sources/prs/flashinfer/PR-888.md` | FlashInfer added MLA paged-KV cache append support with `ckv_dim=512,kpe_dim=64`; the PR is correctness/API focused and does not provide a H20 perf target. | Confirms this is the intended upstream kernel family for Kimi MLA cache append, not a local ad hoc path. |
| `sources/prs/flashinfer/PR-2037.md` | FlashInfer has a fused RoPE + quantize + append-cache direction, but it does not map directly to this BF16 non-quantized Kimi decode append. | A future useful optimization would be launch removal or fusion, not retuning this standalone append in isolation. |
| `wiki/patterns/low-sm-utilization.md` | Low SM utilization often comes from a grid/workload too small for the GPU, tail effects, or scheduling overhead; persistent scheduling only helps if there is enough useful work to schedule. | The pre-fix NCU profile matches this pattern: `78` CTAs on `78` SMs but only `0.12` waves/SM and very low useful throughput. |
| `wiki/patterns/memory-bound.md` | Memory-bound diagnosis requires measured bandwidth evidence, not just a low arithmetic-intensity formula. | The row moves BF16 cache data, but the measured DRAM percentage is too low to call it HBM-bound. |

## NCU Conclusion

Workload: Kimi K2 TP1 DP8 EP8 + PPLX decode, per-rank `arena_rows=8`, global `bs~=64`, `ctx=1`.

Local production-metadata smoke:

```bash
cargo run --release -p pegainfer-kimi-k2 --features kernel-report \
  --bin kimi_tp1_pplx_decode_bench -- \
  --active-rows 8 --ctx-lens 1 \
  --labels decode.attention.paged_kv_append --iters 2 --format json \
  --out target/kernel_reports/kimi-k2/tp1-pplx-decode-kv-append-local-production-metadata-smoke.json
```

Result: `6.256us/call`, `381.62us/step`, `achieved_gbps=2.97`, `roofline_bound=control`, `roofline_peak_pct=null`.

H20 production-metadata bench:

```bash
/root/develop/xingming/pegainfer-workspace/pegainfer/target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1,128,1024,2048 \
  --labels decode.attention.paged_kv_append --iters 128 --format json \
  --out /tmp/kimi_kv_h20_prod.json
```

`ctx=4096/8192` is intentionally excluded from the promoted numbers for this row: the production decode arena represented by this provider is `128` pages/request with `page_size=16`, so one request has `2048` token capacity. Values above that return `supported=false` with an explicit capacity reason in the default bench output instead of silently measuring an invalid page table.

| ctx | H20 per call | H20 per 61-layer step | Payload-equivalent throughput | Roofline |
|---:|---:|---:|---:|---|
| 1 | `7.066us` | `431.03us` | `2.63GB/s` | control, no `%peak` |
| 128 | `7.233us` | `441.20us` | `2.57GB/s` | control, no `%peak` |
| 1024 | `7.245us` | `441.93us` | `2.56GB/s` | control, no `%peak` |
| 2048 | `7.358us` | `448.87us` | `2.52GB/s` | control, no `%peak` |

Pre-fix H20 compact-metadata bench, kept only as directional evidence:

```bash
cargo run --release -p pegainfer-kimi-k2 --features kimi-k2,kernel-report \
  --bin kimi_tp1_pplx_decode_bench -- \
  --active-rows 1,2,4,8 --ctx-lens 1,128,1024,4096,8192 \
  --labels decode.attention.paged_kv_append --iters 128 --format json \
  --out target/kernel_reports/kimi-k2/tp1-pplx-decode-kv-append-h20.json
```

Pre-fix H20 compact-metadata NCU full report:

```bash
/usr/local/cuda/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -k regex:AppendPagedKVMlaCacheKernel -c 1 \
  -o profile/kimi-mla-paged-kv-append-h20/reports/kv_append_full \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 \
  --labels decode.attention.paged_kv_append --iters 1 --format text \
  --out profile/kimi-mla-paged-kv-append-h20/kv_append_ncu.json
```

| Metric | Value |
|---|---:|
| Event timing | `7.342us/call`, `447.9us/step` |
| Context sweep | `7.34-7.44us/call` for `ctx=1,128,1024,4096,8192` |
| Effective bench bandwidth | `2.53GB/s` payload-equivalent; not promoted as HBM peak percentage |
| NCU kernel | `AppendPagedKVMlaCacheKernel<512,64,2,bf16,int>` |
| NCU duration | `3.46us` |
| Grid / block | `78` CTAs x `256` threads |
| Waves / SM | `0.12` |
| Memory throughput | `4.44GB/s`, `0.30%` memory throughput |
| DRAM throughput | `0.09%` |
| Compute throughput | `1.34%` |
| Achieved occupancy | `11.70%` |
| L1/TEX / L2 hit rate | `26.92%` / `62.71%` |
| Scheduler no eligible | `97.90%` |
| Top rule | Grid/workload too small; issue slot utilization local speedup estimate `97.9%` |

Diagnosis: the kernel writes only one compressed KV row plus KPE row per arena slot. The provider uses `nnz=8`, so useful payload is tiny even though FlashInfer launches a grid sized to the SM count. The timing is mostly launch/control/scheduler overhead and metadata work; neither H20 tensor compute nor HBM bandwidth is expected to be the active ceiling.

2026-06-01 H20 rerun status: `h20-100` eventually completed the production-metadata bench after rebuilding the target binary with `cargo +nightly build`; `cargo check` also passed on the intended workspace. Nsight Compute is still unavailable on this runner: `/usr/local/cuda/bin/ncu --version` times out, and `--set full` collection did not launch a usable report. Keep the compact-metadata NCU table above as directional support for the control/tiny-grid diagnosis, but do not call it a production-metadata NCU report.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Add measured provider for `kimi_mla_paged_kv_append` | Local production-metadata smoke passed. The provider appends fixed `arena_rows=8`; `active_rows` does not change `nnz` for this row. H20 production bench at `ctx=1` is `7.066us/call`. | Adopted. |
| Fix bound classification | Manifest now marks the row `BoundKind::Control`, so JSON/text output has `roofline_bound=control` and no HBM `%peak`. | Adopted. |
| Fix page metadata | Provider now mirrors production decode arena page stride with `128` pages/request instead of compact page ids. | Adopted; invalidates the earlier compact-metadata H20 run as a production baseline. |
| Standalone kernel rewrite | Not attempted. The profile says the row is not saturating memory or compute, and the aggregate cost is below the existing top-10 local compute rows. | Reject as current priority. |
| Fusion direction | Candidate only if adjacent MLA cache preparation can remove this launch without slowing qkv split/RoPE work. | Keep as a future fusion note, not a Phase 3 standalone target. |

## Final Conclusion

Keep FlashInfer `kimi_mla_paged_kv_append` as the production provider. Do not claim the row has reached any H20 memory peak; it is modeled as control/tiny-grid work. The master table can use the production-metadata H20 latency above. The next required action is a production-metadata NCU rerun after Nsight Compute on `h20-100` recovers.
