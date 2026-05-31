# Attention RoPE Split Report

> **TL;DR:** `decode.attention.rope_split` is the Kimi MLA decode helper `rope_split_decode_kernel`: it splits `q_proj` into `q_nope` / `q_pe`, applies RoPE to `q_pe`, and produces `append_kpe` for the MLA KV cache. At TP1 PPLX `bs=8,ctx=1`, H20 event timing is `441.8us` per 61-layer step or `7.24us/call`; payload-equivalent throughput is only `54.4GB/s`, so this is not HBM-bound in the useful roofline sense. Production NCU is pending because `ncu` currently times out on `h20-100`; stop standalone tuning for now and only revisit as a launch-removing fusion around MLA cache prep.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Low SM utilization can come from tail effect, load imbalance, static scheduling, or a grid too small for the GPU; persistent scheduling only helps when there is enough work to reschedule. | This row launches `384` CTAs at the target shape, so it is not the same tiny-grid pattern as row 8. NCU is required before claiming a scheduler-specific fix. |
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | Memory-bound diagnosis needs measured high DRAM throughput; low arithmetic intensity alone is not enough. | The bench reports only `~54GB/s` payload-equivalent throughput, far from H20 HBM peak, so a standalone bandwidth rewrite is not evidence-backed. |
| `wiki/patterns/tail-effect.md` (`pattern-tail-effect`) | Moderate tile counts can lose time to wave quantization when the last wave is underfilled. | `384` CTAs on H20's `78` SMs is about `4.9` CTA waves. Tail effect is plausible but not proven without NCU/PM sampling. |
| `sources/prs/flashinfer/PR-3014.md` (`pr-flashinfer-3014`) | Small-batch decode helper kernels can benefit from removing helper overhead, but the PR is MoE-helper specific. | Directional only: the useful direction is launch removal or fusion, not retuning this helper in isolation. |

Practical conclusion: this is an elementwise MLA preparation helper with tiny per-call payload. The current evidence does not support a standalone rewrite. The plausible future work is to remove the launch by combining `append_kpe` handling with nearby MLA cache prep, while preserving the `q_nope` and `q_pe` consumers.

## NCU Conclusion

Fresh production NCU is currently unavailable on `h20-100`:

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# exits 124 with no output
```

So this report does not claim measured DRAM throughput, warp stalls, or source-line bottlenecks for `rope_split_decode_kernel`. The stop decision is based on the H20 CUDA-event roofline and source launch geometry: the row is small enough that a standalone retune is unlikely to produce a reliable `>3%` full-bench win without deleting a launch.

If reopened, collect:

```bash
/usr/local/cuda-12.9/bin/ncu --target-processes all \
  --kernel-name-base demangled --print-kernel-base demangled --set full \
  -k regex:rope_split_decode_kernel \
  -o profile/kimi-attention-rope-split-h20/reports/rope_split_full \
  --force-overwrite target/release/kimi_tp1_pplx_decode_bench \
  --active-rows 8 --ctx-lens 1 --iters 1 --format text \
  --labels decode.attention.rope_split \
  --out profile/kimi-attention-rope-split-h20/rope_split_ncu.json
```

Required profile questions:

| Question | Why it matters |
|---|---|
| Scheduler stalls and issue slot utilization | Decide whether the `7us/call` is launch/control overhead or instruction latency inside the RoPE math. |
| DRAM/L2 throughput | Confirm the row is not actually limited by unmodeled cos/sin or q-pair traffic. |
| PM sampling timeline | Check whether the `384` CTA launch has a visible tail wave or just uniformly low useful work. |

## Bench Evidence

Runtime path:

| Item | Value |
|---|---|
| Runtime call | `kimi_mla_rope_split_decode_rt` in `pegainfer-kimi-k2/src/runner/worker/forward.rs` |
| Bench op | `kimi_mla_rope_split_decode_rt` / `decode.attention.rope_split` |
| CUDA entry | `kimi_mla_rope_split_decode_cuda` in `pegainfer-kernels/csrc/kimi_k2/kimi_mla.cu` |
| Kernel | `rope_split_decode_kernel` |
| Launch at target shape | `384` CTAs x `256` threads for `batch_size=8, local_heads=64, q_head_dim=192` |
| Calls per decode step | `61` |

The kernel computes:

| Tensor | Shape / dtype |
|---|---|
| `q_proj` input | `[8,64,192]`, BF16 |
| `k_rope` input | `[8,64]`, BF16 |
| `cos/sin` cache | current-position RoPE cache, BF16 |
| `positions` | `[8]`, I32 |
| `q_nope` output | `[8,64,128]`, BF16 |
| `q_pe` output | `[8,64,64]`, BF16 |
| `append_kpe` output | `[8,64]`, BF16 |

H20 TP1 PPLX bench evidence from `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json`:

| active rows | ctx | Step latency | Per call | TFLOP/s | Payload GB/s |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | `441.76us` | `7.24us` | `0.027` | `54.44` |
| 8 | 128 | `421.33us` | `6.91us` | `0.029` | `57.08` |
| 8 | 1024 | `437.19us` | `7.17us` | `0.027` | `55.01` |
| 8 | 4096 | `543.21us` | `8.91us` | `0.022` | `44.28` |
| 8 | 8192 | `537.84us` | `8.82us` | `0.022` | `44.72` |

The row does not depend on `ctx_len` except for the position used to index the RoPE cache, so the ctx variation above should be treated as cache/noise sensitivity until NCU says otherwise. The target `ctx=1` row is only about `1.1%` of H20 HBM on the bench payload model and far below compute peak. That is a control/elementwise signature, not a saturated memory kernel.

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current `rope_split_decode_kernel` | `7.24us/call`, `441.8us/step` at `bs=8,ctx=1`; about `54.4GB/s` payload-equivalent. | Current baseline. |
| Standalone bandwidth rewrite | Not attempted. The event roofline does not show a memory ceiling, and NCU is unavailable. | Reject for now. |
| More CTA parallelism | Not attempted. The current target shape already launches `384` CTAs; changing indexing would add complexity before proving a grid issue. | Reject without NCU evidence. |
| Fuse `append_kpe` with paged KV append or nearby MLA prep | Not implemented. This could remove a launch or an intermediate write/read if it preserves `q_nope`, `q_pe`, and cache append semantics. | Future-only direction with full TP1 PPLX bench and correctness gate. |

## Final Conclusion

Keep the current `rope_split_decode_kernel` and stop standalone `decode.attention.rope_split` tuning. Reclassify the master row as `control/elementwise` rather than memory-bound: the payload is too small to use H20 bandwidth, and there is no production NCU report proving a lower-level bottleneck.

Reopen this row only for a fusion that removes a launch around MLA cache prep, or after production NCU shows a concrete bottleneck with a realistic path to `>3%` full decode improvement at `bs=8/rank`, global `bs~=64`.
