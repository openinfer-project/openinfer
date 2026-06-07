# Kimi-K2 MoE EP: PPLX → DeepEP migration research

> **TL;DR:** DeepEP is vendored as a submodule (`pegainfer-kernels/third_party/DeepEP`, commit `d4f41e4`, 2026-05-26). The integration target is the new **elastic** API (`deep_ep/buffers/elastic.py` — `legacy.py` is the old LL/HT split). For the decode path the right call shape is **fresh dispatch per step with `do_expand=True` + `do_cpu_sync=False`**: fixed worst-case output shapes (CUDA-graph-capturable, no D2H stall) and per-expert aligned contiguous segments that feed a grouped GEMM directly. The Marlin side is an **adapter, not a rewrite** — the PPLX campaign already moved Marlin to GPU-resident counts (sentinel routing + device-read `num_tokens_post_padded`), and DeepEP's psum layout translates into that format with one small kernel. The real integration risks are the host layer, not the math: `at::cuda::CUDAStream` in every launch signature, mandatory NVRTC JIT (we need AOT instantiation), and NVSHMEM as a build-time link dependency even for single-node.
>
> **Last touched:** 2026-06

## Why DeepEP (what PPLX cannot give us)

- **CUDA Graph (#227):** the PPLX path architecturally cannot be captured — it relies on a fabric worker thread per rank (`enable_pplx()` force-disables graphs, `runner/worker/state.rs:5-10`). DeepEP elastic dispatch/combine are stream-ordered kernels over the NCCL device API (`ncclDevComm_t`/`ncclWindow_t` in the kernel signatures, `csrc/.../dispatch.hpp:214-238`); with `do_cpu_sync=False` all shapes are static, so the whole MoE section becomes capturable in principle (needs an H200 prototype to confirm GIN ops capture cleanly).
- **Overlap (#228):** elastic API has first-class event plumbing (`previous_event`, `previous_event_before_epilogue`, `async_with_compute_stream`) designed for comm/compute overlap.
- **Determinism:** `ElasticBuffer(deterministic=...)` exists and `tests/elastic/test_ep.py:460-463` asserts combine output is **bitwise-identical** to an NCCL reference — directly supports our token-exact det contract.
- After the #204 kernel-pick campaign, collectives/MoE is the only structural decode lever left (bs64 TPOT unchanged within ±1ms after five GEMM picks).

## Elastic API facts that drive the design (all verified in source)

Dispatch (`elastic.py:708-855`):

- `do_expand=True` → "one slot per (expert, token)" layout. Per-local-expert segments are contiguous and **aligned to `expert_alignment`**; `psum_num_recv_tokens_per_expert` (GPU, `[num_local_experts]`) encodes both real counts and offsets: `psum[i] − align(psum[i−1], A)` = expert *i*'s real count, `align(psum[i], A)` = expert *i+1*'s start (`elastic.py:41-47`). Set `expert_alignment` = Marlin block size and the layout is exactly what the grouped GEMM wants.
- `do_cpu_sync=False` → no D2H sync; CPU never learns counts; `recv_x` is allocated at fixed worst-case capacity (exact formula lives in csrc buffer sizing — verify before sizing memory). Default is `True`, so this must be passed explicitly.
- In expand mode `recv_topk_idx` is `None` but **`recv_topk_weights` is returned per-slot** (`test_ep.py:151-155, 384-393` indexes it with slot indices) — the expert rank has router weights available, so the current "W2 GEMM applies topk weight" design maps 1:1.
- Cached `EPHandle` re-dispatch is for *unchanged routing* (`topk_idx` must be `None`, reused from handle) — not applicable to decode, where every step has fresh router output. Decode = fresh dispatch each step.
- FP8 dispatch is available (`x` as `(fp8_e4m3, scales)` tuple, per-128-element scales); bf16 is the simple first target.

Combine (`elastic.py:868-928`):

- Takes bf16 `[tokens, hidden]` + handle. **Never multiplies router weights into `x`** in either mode — `combined_topk_weights` is just routed back and asserted equal to the originals (`test_ep.py:464-465`). Weighting is always the caller's job, applied to expert outputs *before* combine. Expand-mode combine ("reduced combine") sums the slots of each source token.
- `bias` (up to 2 tensors) is added in the epilogue — a free fusion point for the shared-expert path if we ever want it.

Answering the design question directly: **`do_expand=True`, `do_cpu_sync=False` is correct for decode.** Expand's cost (token duplicated per selected expert → topk× traffic into the GEMM) is noise at decode batch sizes; what it buys is contiguous aligned per-expert segments with GPU-only counts and fixed shapes — the exact preconditions for grouped GEMM + CUDA graph. `do_cpu_sync=True` would reintroduce a per-layer D2H sync (the thing that makes a step ~unGraphable and stalls the pipeline 61 times per token).

## Marlin interface: adapter, not rewrite

Current PPLX decode flow (`runner/moe_pplx.rs:256-465`):

```
dispatch_recv → recv_tokens_per_expert (GPU, i32/expert)
  → kimi_pplx_build_marlin_routing_on_stream  (<<<1,64>>> prefix sum, kimi_experts.cu:308-364)
  → sorted_token_ids (+sentinel for padding) / expert_ids / num_tokens_post_padded[0]  (all GPU)
  → Marlin W13 (reads M from device, marlin_template.h:307) → SwiGLU → Marlin W2 (applies topk weights)
  → combine_send/recv (F32 out) → ×KIMI_K2_ROUTER_SCALE → residual
```

Marlin already consumes GPU-resident M and guards invalid rows in the epilogue (`block_num_valid_tokens`, `marlin_template.h:509-514, 1640, 1697`). DeepEP expand mode produces the same shape of problem: aligned per-expert segments with garbage padding rows between real count and aligned end — *isomorphic to PPLX's padded expert-major layout once you know per-expert counts*.

Required changes, scoped:

1. **Routing adapter kernel** (small): translate DeepEP's `psum_num_recv_tokens_per_expert` → the existing `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded` triple. Same job as `kimi_pplx_build_marlin_routing_on_stream` does for PPLX counts; with `expert_alignment = 8` (current PPLX expert padding) offsets are already aligned and the prefix sum is already done — the adapter mostly emits sentinels for the pad rows. Marlin core untouched.
2. **Per-slot weights plumbing:** W2 currently reads `pplx_recv_topk_weight`; switch the source to DeepEP's expanded `recv_topk_weights` (already per-slot, same indexing space as the activation rows).
3. **Combine-side rework** (the genuinely new part): DeepEP combine takes **bf16** and returns bf16; PPLX `combine_recv` returns F32 and the router scale is applied post-combine (`KIMI_K2_ROUTER_SCALE`, `moe_pplx.rs:457-465`). Options: fold the router scale into the W2 epilogue (per-slot weight × 0.0625 pre-scale), or accept a bf16 combine and upconvert for the residual add. Numerics must be re-gated either way (det contract + vllm_golden_gate).
4. **SwiGLU:** already launches at worst-case rows and tolerates garbage rows that the next GEMM's epilogue masks — unchanged, but re-verify the garbage-row tolerance claim under DeepEP's padding pattern (the #204 lesson: small-N corruption only shows at decode shapes).

What does *not* change: INT4 weight/scale layout (expert-major wna16, group 32, perm64), the W13→SwiGLU→W2 structure, the sentinel mechanism.

## Host-layer integration (the actual hard part)

The kernels are torch-free (raw pointers + NCCL device API; no torch includes in `*.cuh`). Everything above them is not:

- **`at::cuda::CUDAStream` in every launch signature** (20+ functions, `dispatch.hpp:61,238`, `combine.hpp:132`, `jit/launch_runtime.hpp:50-52`) plus `at::cuda::setCurrentCUDAStream` thread-local games in `buffer.hpp:496-555`. Rust FFI needs a thin C++ shim that accepts `cudaStream_t` and constructs the launch context without ATen — or we port the (modest) host orchestration logic into pegainfer-comm and call the kernel stubs directly.
- **NVRTC JIT is mandatory today**: template params (`num_experts`, `hidden`, `num_topk`, `num_qps`, rank counts) are baked per specialization (`jit/compiler.hpp:111-160`). For us they are all static per model config (Kimi-K2: 384 experts / 48 local, topk 8, hidden 7168, 8 ranks) → AOT-instantiate the needed specializations in `pegainfer-kernels/build.rs`, same pattern as the Triton AOT flow.
- **NVSHMEM is a build-time `REQUIRED` link dep** (`CMakeLists.txt:26`) even though single-node runtime traffic goes over the NCCL device path (GIN + symmetric windows, `nccl.cu:94-147`); upstream `setup.py:93` has "TODO: make NVSHMEM and legacy optional". Either carry the dep or patch it out for intranode-only builds.
- **NCCL version floor:** the device API (`ncclDevComm`, windows, GIN) needs a recent NCCL — pin and verify on the H200 image before anything else.
- **Bootstrap:** symmetric memory via `ncclCommWindowRegister` + CUDA IPC fd exchange (`symmetric.hpp`) — pegainfer-comm's existing single-process 8-rank bootstrap can drive this; no torch.distributed anywhere in that layer.

## Open questions (verify on hardware, in rough order)

1. Exact worst-case `recv_x` capacity formula with `do_cpu_sync=False` (read csrc buffer sizing; matters little at decode shapes — order 10⁴ slots × 14KB — but must be right).
2. Empty-rank semantics: all 8 DP ranks must call dispatch/combine each step even with 0 tokens (PPLX supports this today); confirm elastic does.
3. CUDA-graph capture of GIN/symmetric-window kernels on H200 — the entire #227 thesis rests on this; prototype before any Rust work.
4. Decode-shape latency: dispatch+combine µs at bs64/topk8/hidden7168 vs PPLX dispatch_send/recv+combine_send/recv pair (Python-side microbench is enough for go/no-go).
5. `deterministic=True` cost at decode shapes.

## Next steps

1. H200 prototype (Python, no Rust): build DeepEP, run `tests/elastic/test_ep.py` single-node 8-GPU, microbench decode shapes incl. graph capture of a dispatch→dummy-GEMM→combine step. Go/no-go on items 3-4 above.
2. AOT spike: instantiate the elastic dispatch/combine specializations for the Kimi config without NVRTC; measure build cost in `pegainfer-kernels/build.rs`.
3. C++ shim for `cudaStream_t`-based launch + Rust FFI in pegainfer-comm behind an `ep_backend` variant (keep PPLX as the fallback during bring-up).
4. Marlin routing adapter kernel + combine-side rework; gate with vllm_golden_gate + det contract before any perf claims.
