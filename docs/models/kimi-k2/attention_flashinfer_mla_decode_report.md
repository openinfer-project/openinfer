# Attention FlashInfer MLA Decode Report

> **TL;DR:** `decode.attention.flashinfer_mla_decode` calls FlashInfer `BatchDecodeWithPagedKVCacheDispatchedMLA` through `kimi_flashinfer_batch_decode_mla_rt`. At the anchor `bs=8/rank,ctx=1` it costs `624.6us/step` (`10.24us/call`) across 61 attention layers, but it is the long-context cliff: `ctx=8192` costs `103.50ms/step` (`1.697ms/call`) and reaches about `2.85 TB/s` payload-equivalent bandwidth (`~59%` of the H20 HBM roofline). This row is memory-bound for long context and launch/control-heavy for `ctx=1`. No code change is adopted yet because fresh production NCU currently cannot be collected on `h20-100` (`ncu --version` times out); next work must profile this row before choosing FlashInfer backend/plan, FP8 KV, split-K/partition-KV, or custom MLA decode changes.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `sources/prs/sglang/PR-3987.md` (`pr-sglang-3987`) and `sources/prs/sglang/PR-4012.md` (`pr-sglang-4012`) | SGLang added a fast decode plan for FlashInfer MLA to avoid CPU/GPU indptr transfer and CUDA Graph replay hangs. | Plan construction and graph-safe metadata are part of the performance surface. Pegainfer already uses CUDA graph decode, so changes here must preserve graph replay. |
| `sources/prs/vllm/PR-21078.md` (`pr-vllm-21078`) | vLLM integrated FlashInfer MLA decode as a dedicated backend and benchmarked it at serving scale. | Confirms FlashInfer MLA is a strong baseline, not a toy fallback. Any custom path must beat this baseline on H20 and keep paged-cache semantics. |
| `sources/prs/flashinfer/PR-2530.md` (`pr-flashinfer-2530`) | FlashInfer's auto backend choice for `BatchDecodeWithPagedKVCacheWrapper` regressed on Hopper, and the PR chose FA2 for non-FP8 workloads. | Backend selection matters on SM90/H20. Before changing code, profile the exact dispatched MLA kernel and verify which backend variant is running. |
| `sources/prs/vllm/PR-34597.md` (`pr-vllm-34597`) | FP8 KV cache in MLA decode reduces KV bandwidth by dequantizing on load. | Kimi's current MLA cache path is BF16. FP8 KV is the main bandwidth-saving direction if accuracy and cache format changes are acceptable. |
| `sources/prs/sglang/PR-18442.md` (`pr-sglang-18442`) | FA4 SM90 paged-KV decode support was added because decode requires paged KV support on Hopper. | Directional only: a future FlashInfer upgrade/backend experiment should compare paged MLA decode kernels on H20, but this repo currently calls the FlashInfer submodule wrapper. |

Practical conclusion: for `ctx=1`, the row is launch/control overhead. For `ctx>=1024`, it is KV-cache bandwidth dominated. The optimization fork is therefore context-dependent: short-context tuning must remove overhead; long-context tuning must reduce bytes moved or improve the chosen FlashInfer backend.

## NCU Conclusion

Fresh production NCU is currently unavailable on `h20-100`:

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# exits 124 with no output
```

The NCU conclusion for this row is therefore intentionally incomplete: do not adopt a code optimization from CUDA-event timing alone. The next NCU run should isolate:

```bash
cargo run --release -p pegainfer-kimi-k2 --features kernel-report \
  --bin kimi_tp1_pplx_decode_bench -- \
  --active-rows 8 --ctx-lens 1,128,1024,4096,8192 \
  --labels decode.attention.flashinfer_mla_decode \
  --iters 16 --format json \
  --out target/kernel_reports/kimi-k2/tp1-pplx-mla-decode-filter-h20.json

/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -k regex:.*BatchDecode.*MLA.* \
  -o profile/kimi-flashinfer-mla-decode-h20/reports/ctx8192_full \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 8192 --iters 1 --format text \
  --labels decode.attention.flashinfer_mla_decode \
  --out profile/kimi-flashinfer-mla-decode-h20/ctx8192_ncu.json
```

Required profile questions:

| Question | Why it matters |
|---|---|
| Which FlashInfer MLA kernel/backend is dispatched? | KernelWiki records backend selection regressions on Hopper; the source call alone is not enough. |
| DRAM read throughput and L2 hit rate at `ctx=8192` | The bench-derived payload is `~2.85 TB/s`, but NCU must confirm actual bytes and cache behavior. |
| Wave count, occupancy, and scheduler stalls at `ctx=1` | Short context is only `10.24us/call`; it may be launch/control limited rather than bandwidth limited. |
| Split-K / partition-KV behavior | Current wrapper passes `partition_kv=false`; NCU must show whether long-context work fills H20 well enough without partitioning. |

## Bench Evidence

Runtime path:

| Item | Value |
|---|---|
| Rust wrapper | `pegainfer-kernels::ops::kimi_k2::mla::kimi_flashinfer_batch_decode_mla` |
| CUDA entry | `kimi_flashinfer_batch_decode_mla_cuda` in `pegainfer-kernels/csrc/kimi_k2/kimi_mla.cu` |
| FlashInfer call | `BatchDecodeWithPagedKVCacheDispatchedMLA<HEAD_DIM_CKV=512, HEAD_DIM_KPE=64>` |
| Current flags | `partition_kv=false`, `enable_pdl=false`, no temporary split-K buffers |
| Shape | `q_abs_nope=[8,64,512]`, `q_pe=[8,64,64]`, paged BF16 compressed KV, output `[8,64,512]` |
| Calls per decode step | `61` attention layers |

H20 artifact: `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json`.

`active_rows` does not change this attention row because the bench uses the fixed decode arena (`arena_rows=8`) for attention/final rows; the MoE rows use active rows separately.

| Per-rank active rows | ctx | Step latency | Per call | Payload GB/s | TFLOP/s | H20 HBM pct |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | `0.640 ms` | `10.49us` | `206.1` | `0.106` | `4.3%` |
| 1 | 8192 | `103.34 ms` | `1.694ms` | `2853.2` | `5.388` | `59.4%` |
| 2 | 8192 | `103.52 ms` | `1.697ms` | `2848.2` | `5.378` | `59.3%` |
| 4 | 8192 | `103.74 ms` | `1.701ms` | `2842.2` | `5.367` | `59.2%` |
| 8 | 1 | `0.625 ms` | `10.24us` | `211.2` | `0.109` | `4.4%` |
| 8 | 128 | `2.132 ms` | `34.96us` | `2204.6` | `4.079` | `45.9%` |
| 8 | 1024 | `13.400 ms` | `219.67us` | `2756.7` | `5.194` | `57.4%` |
| 8 | 4096 | `52.230 ms` | `856.24us` | `2823.4` | `5.330` | `58.8%` |
| 8 | 8192 | `103.50 ms` | `1.697ms` | `2848.8` | `5.379` | `59.4%` |

Roofline interpretation:

- Long-context arithmetic intensity from the bench model is about `3.78 flop/byte`, below the H20 ridge point recorded in the master table (`30.83 flop/byte`), so long context is memory-bound.
- `ctx=1` has too little KV work to saturate bandwidth; the row is dominated by FlashInfer launch/control and metadata overhead.
- Because payload-equivalent bandwidth is already around `59%` of the H20 HBM peak at long context, an optimization must either reduce bytes moved, improve the actual FlashInfer backend/cache behavior, or change cache dtype/layout. Small launch-level tuning will not solve the `ctx=8192` cliff.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current FlashInfer MLA decode path | Baseline recorded across `ctx=1..8192`; long-context `ctx=8192` is `103.50ms/step`, `2.85 TB/s` payload-equivalent. | Current baseline. |
| Fresh production NCU | Attempted `ncu --version` on `h20-100`; it timed out. | Do not optimize from event timing alone. |
| Flip `partition_kv` / split-K | Not attempted. Current wrapper passes `partition_kv=false` and no temp buffers. | Candidate only after NCU shows under-filled SMs or scheduler/tail limits at long context. |
| FP8 KV cache | Not attempted. KernelWiki/vLLM direction is promising for bandwidth, but it changes cache dtype and correctness envelope. | Candidate only with an explicit accuracy gate and cache-layout plan. |
| FlashInfer backend/version swap | Not attempted. KernelWiki shows Hopper backend selection can matter. | Candidate only after identifying the exact dispatched kernel/backend in NCU. |

## Final Conclusion

Keep the current FlashInfer MLA decode path as the baseline for now. This row is not stopped as an optimization target: it is the highest-priority attention row for long context, but the next move must be profile-driven.

Adoption bar for any future change:

| Direction | Required proof |
|---|---|
| Backend/plan selection | NCU identifies the current backend and shows a replacement improves `ctx=8192` H20 latency by `>3%` without hurting `ctx=1`. |
| `partition_kv` / split-K | NCU shows long-context SM underfill/tail effects; full bench improves at `ctx>=1024` and short context does not regress materially. |
| FP8 KV cache | Token/logit correctness gate plus H20 bench improvement from lower KV bytes; report must record cache dtype/layout changes. |

No `opt(...)` commit is appropriate from this report alone. Reopen when Nsight Compute works on `h20-100` or when a comparable H20 NCU artifact is available.
