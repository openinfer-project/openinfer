# Kimi-K2 MoE EP: PPLX → DeepEP migration

> **TL;DR:** Implemented — Kimi-K2's MoE EP backend is now DeepEP (elastic API, AOT-instantiated, no torch/NVRTC/NVSHMEM); PPLX is fully deleted from the kimi path (`moe_pplx.rs` gone, kimi crate no longer depends on `pegainfer-comm`). Decode = `do_expand=true` + `do_cpu_sync=false`: fixed worst-case buffers, zero host syncs/allocs per step → CUDA-graph-capturable in principle (#227, capture still disabled). Prefill = `do_cpu_sync=true` with host spin on pinned counters. Marlin consumes the DeepEP recv buffer **in place** (expert_alignment 8 == Marlin block size; identity routing + sentinels). Toxic-reviewed READY; numerics gates (vllm_golden_gate + det contract) and serving bench still pending on 8×H200.
>
> **Last touched:** 2026-06

## Architecture as built

```
pegainfer-kernels/
  third_party/DeepEP            # submodule d4f41e4 (2026-05-26)
  csrc/deepep/deepep_shim.cu    # AOT template instantiation (Kimi config baked:
                                #   384 experts / 48 local, topk 8, hidden 7168, 8 ranks)
                                # + torch-free host launch over cudaLaunchKernelExC
  src/ffi/deepep.rs             # repr(C) DeepEpInfo + extern decls
  src/ops/deepep.rs             # DeepEp wrapper: decode_dispatch/decode_combine (no sync),
                                #   prefill_dispatch_send/wait_counts/recv + prefill_combine
pegainfer-kimi-k2/
  src/runner/moe_deepep.rs      # the MoE layer:
                                #   forward_moe_layer_decode_deepep_normed  (host-quiet)
                                #   forward_moe_layer_prefill_deepep       (cpu-sync)
```

Build needs `PEGAINFER_NCCL_ROOT` pointing at NCCL ≥ 2.30 (device API: `ncclDevComm`,
windows, GIN). The binary links `libnccl.so.2` via `LD_LIBRARY_PATH` at runtime.
Local dev: `PEGAINFER_NCCL_ROOT=/data/opt/nccl-2.30.4`.

Backend selection: TP1/DP8 **requires** `--ep-backend=deepep` (default), TP8/DP1
requires `nccl` — both enforced with `ensure!` in `runner/bringup.rs`. There is no
PPLX fallback by design ("我们并不是很喜欢 pplx ep"). `pegainfer-comm`/PPLX survive
only for the deepseek crates, which use their own `pegainfer_comm::EpBackend` type.

## The contracts the integration stands on (verified in upstream source)

**psum_expert post-epilogue semantics** (`dispatch.cuh:209,251-254`,
`dispatch_copy_epilogue.cuh:117`): dispatch writes the *exclusive* prefix sum of
*aligned* per-local-expert counts (49 entries); the copy epilogue then `atomicAdd`s
+1 per real slot, so afterwards `psum[i] = aligned_start_i + real_count_i` and
`psum[48]` = total aligned expanded rows, untouched. The Marlin routing kernel
(`kimi_deepep_build_marlin_routing_kernel`, `<<<1,64>>>`) reconstructs
`start_i = round_up(psum[i-1], 8)`, `count_i = psum[i] − start_i`, and reads
`num_tokens_post_padded` straight from `psum[48]` — no counts-conversion kernel.

**Marlin in-place consumption**: DeepEP `expert_alignment=8` equals Marlin's block
size, so the expanded recv buffer (expert-major, per-expert 8-aligned segments) *is*
Marlin's "sorted+padded" format. Routing = identity `sorted_token_ids` with sentinel
(=capacity) on pad rows + per-block `expert_ids` + device-resident
`num_tokens_post_padded`. Pad rows compute garbage that combine never reads
(row-independent GEMMs, sentinel-guarded epilogue). Marlin core untouched.

**Weights/scale flow**: dispatch carries per-slot f32 router weights
(`recv_topk_weight`, same row space as `recv_x`); W2 applies them
(`mul_topk_weights=true, top_k=1`); combine sums **unweighted**
(`topk_weights=nullptr`, asserted upstream); `KIMI_K2_ROUTER_SCALE` is applied at
the residual via `kimi_residual_add_scaled_bf16` — same convention as the NCCL
backend. Net numerics delta vs PPLX: combine output is bf16 where PPLX returned
f32, i.e. the routed sum rounds to bf16 one step earlier. Must be re-gated on
hardware (vllm_golden_gate + det contract), expected within the bf16 ULP floor.

**Decode is host-quiet**: all buffers persistent and worst-case-sized at
`enable_deepep` (8528 expanded rows ≈ 350 MB/rank); per step there are zero
allocations, zero D2H syncs, zero `seq_len` mutations. `decode_combine` passes the
worst-case 1024 as `num_reduced_tokens` — that is the upstream *sentinel*
(`combine.cuh:43-44`): the kernel reloads the actual count from `psum_rank` on
device. Graph capture stays disabled for now because the capture closure doesn't
thread EP state — enabling it would silently fall back to NCCL-MoE local-experts
(#227 tracks capture).

**Prefill lock-step**: all 8 DP ranks call the layer fn simultaneously
(synchronized prefill pads idle ranks with 1 dummy token). Order per layer:
enqueue everything GPU-side (shared expert, router on aux joined via event,
`prefill_dispatch_send`) **before** the host spins on pinned counters
(`prefill_wait_counts`, 120 s throwing timeout). Capacity:
`recv_capacity = ep_max_seq_len + 7` (only the owner sends real tokens; 7 dummies),
`expanded_capacity = (recv_capacity·8 + 48·7).next_multiple_of(8)` — tight bound
(≤ 8 distinct local experts per token, ≤ 7 pad per expert); counts are `ensure!`d
against capacity before recv (crash early).

## Known costs / next levers

- Prefill allocates ~5 device buffers per MoE layer (carried verbatim from the
  PPLX prefill) — obvious TTFT cleanup, not a regression.
- Decode swiglu/W13 launch at worst-case grid (8528 rows) and early-exit past
  `num_tokens_post_padded[0]` — same as PPLX; revisit only if it shows in a profile.

## Pending hardware gates (8×H200, jzh200-42)

1. `vllm_golden_gate` accuracy + det contract (token-exact + 0.25 nat) — the bf16
   combine rounding is the one numerics change to watch.
2. Serving bench vs the PPLX baseline (bs64 TPOT p50 ~30 ms on jzh200-15);
   dispatch+combine µs at decode shapes.
3. #227: prototype graph capture of dispatch→GEMM→combine on H200.
