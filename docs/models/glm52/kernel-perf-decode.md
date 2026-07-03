# GLM5.2 decode kernel performance — measured baseline and optimization ladder

**TL;DR**: After #535 hoisted the FlashMLA tile schedule for every path, one GLM5.2 MLA decode layer costs ~268 µs GPU / 278 µs wall at bs=1 on H100 (measured, parity-verified). Two stacked, implemented optimizations cut it to **168 µs GPU / 178 µs wall (−36%)**: **(0a) an MLA-layer allocation arena** removes ~73 µs — the bring-up forward still does ~20 synchronous `cudaMalloc`s per layer per token (#535's persistent workspace covered the MoE chain, not MLA), and they serialize the decode stream, so eliminating them drops GPU time, not just host time; **(0c) capturing the forward into a CUDA Graph** removes ~27 µs more — graph replay launches the ~18 kernels back-to-back and removes the inter-kernel bubbles where the GPU idled between host launches (this is the "PR5c graph target" the #535 doc names). The scratch's schedule handling reuses #535's `Glm52MlaSchedMetadata` (one plan type, no duplicate). Next design lever assessment unchanged: partial+combine fusion is deprioritized by measurement. Everything below is from `glm52_kernel_bench` on a real H100, parity-verified against the bring-up forward.

Last touched: 2026-07

## Measured baseline (`glm52_kernel_bench`, bs=1, synthetic weights, iters=64)

| stage (ctx 2048, on top of #535) | gpu | wall | cumulative saved |
|---|---|---|---|
| as-is forward (incl. #535's hoisted schedule) | 267.6 µs | 278.0 µs | — |
| 0a MLA arena (`Glm52MlaDecodeScratch`) | 194.7 µs | 205.1 µs | −73 µs |
| **0c + CUDA Graph** | **167.9 µs** | **178.3 µs** | **−100 µs (−36%)** |

ctx 512 tracks it (graph 168.6 / 178.8 µs). Parity-verified bitwise against the bring-up forward. History: pre-#535 this ladder read 288 → 218 (arena) → 190 (schedule hoist) → 166 µs (−42%); #535 landed the schedule hoist for every path (as-is dropped 288 → 268), so this branch's remaining contribution is the arena + the graph.

ctx 512 tracks it (graph 165.6 / 175.8 µs). All three are parity-verified against the alloc-heavy forward. Projected 75-MoE-layer attention share: 22.3 ms/token → **~13.1 ms/token**.

Per stage (ctx 2048, alloc chain included in the projections):

| stage | wall | notes |
|---|---|---|
| o_proj `fp8_linear` | 67.7 µs | [1,16384]·[16384,6144] fp8 — the widest projection |
| kv_a / q_a / q_b | 28.6 / 28.7 / 24.8 µs | quant → TMA-relayout → blockscale GEMM, 4 mallocs each |
| flashmla sparse decode | 48.2 µs | metadata + split-KV partial + combine (3 kernels) |
| assembly family | 8.4 µs | query-assemble + kv quant + cache-pack (buffers reused) |

Context length barely moves the total (286 → 291 µs from 512 → 2048) because sparse top-k = 2048 caps the attended set.

**Measurement provenance**: built from `feat/glm52-kernel-bench` with the CUDA 12.9 toolkit; run on an R535 host (driver 12.2) via a 3-symbol `cuLibrary*`-enumeration `LD_PRELOAD` shim, since cudarc 0.19 calls the 12.4+ enumeration APIs the old driver lacks. The shim only stubs kernel *enumeration* — dispatch is the real `cuLibraryGetKernel`-by-name, and the bench's `verify_scratch_parity` asserts the scratch forward is bitwise-identical before any timing, so a broken load fails loudly rather than faking numbers. A real serving path (and CUDA Graph capture) needs an R550+ driver.

## Step 0 — implemented and measured (−34%/layer)

**0a — allocation arena (68 µs).** The correctness-first bring-up allocates every intermediate fresh (`alloc_zeros`) per projection per token: whole MLA layer ≈ 20 `cudaMalloc`s. Each is synchronous and serializes against the decode stream, so the cost shows up in *GPU* time (287 → 218 µs), not just host time. `Glm52MlaDecodeScratch` + `glm52_mla_decode_forward_into` pre-allocate all 20 buffers once and reuse them.

**0b — hoist the FlashMLA tile schedule (28 µs).** The sparse-decode `metadata` kernel builds `tile_scheduler_metadata` + `num_splits` from `batch_size` and `num_sm_parts` only — both fixed by the contract, independent of the per-token query/KV. The bring-up re-ran it every layer every token; `Glm52MlaDecodeScratch::new` now computes it once and the decode path reuses it (218 → 190 µs). Correctness is guarded by the bench's `verify_scratch_parity` (bitwise vs the alloc-heavy forward that still recomputes it), so the data-independence claim is checked, not assumed. In real serving the schedule must be re-cached whenever `batch_size` changes (num_sm_parts is a device constant); for bs=1 latency decode it is computed exactly once.

**0c — CUDA Graph capture (23 µs).** With the arena (0a) and schedule hoist (0b), the forward is a pure kernel sequence, so `CudaGraphState::run_or_capture` (openinfer-core) captures it once and replays with one `cuGraphLaunch`. This removes host launch overhead *and* GPU time: graph replay issues the ~18 kernels back-to-back, so the GPU stops idling between them waiting for the next host launch (188 → 165 µs GPU). Graph capture/launch are CUDA 11.x APIs, so this runs on the R535 host (unlike the cudarc module-enumeration path that needs the shim). `measure_forward_graph` in the bench captures against the same scratch and reports 165 µs.

## Measured non-result: `num_sm_parts` tuning doesn't help the graphed forward

`current_sm90_num_sm_parts` fills all SMs (132 on H100). For bs=1 top-k=2048 that over-splits — each split handles ~16 KV entries and the combine reduces a 132-way, 17.3 MB `o_accum`. Sweeping the split count (`measure_flashmla_at`) on the **isolated** flashmla stage shows a real 1.68× at `num_sm_parts=16` (48.1 → 27.8 µs), output bitwise-identical to the default (`flashmla_parts_max_diff(16) = 0`, so it's a pure parallelization knob).

**But it does not carry through to the optimized forward.** Measured end-to-end (`--sm-parts 16`): the arena+hoist+graph forward is 169.1 µs at parts=16 vs 166.6 µs at parts=132 — no gain, marginally worse. Two reasons, both only visible end-to-end:
- The isolated 20 µs was dominated by the **metadata kernel**, which step 0b already hoists out of the per-token path — the decode partial+combine alone differs by only ~1.6 µs between 16 and 132 splits (scratch forward 190.7 vs 192.3 µs).
- 16 splits use 16/132 SMs, so the partial underutilizes the GPU in the graphed pipeline, offsetting the smaller combine.

(The 30 µs the ungraphed as-is forward saves at parts=16 — 290 → 260 µs — is mostly the cheaper `cudaMalloc` of the smaller accum buffers, which the arena already eliminates.) This corrects an earlier estimate that projected ~19 µs from this tuning; the real measurement is the opposite. It also lowers the expected payoff of step 1 below.

## Step 1 — fuse the FlashMLA sparse partial+combine (smaller than it first looked)

`glm52_flashmla_sparse_decode_launch` runs two CUDA kernels (`csrc/glm52/glm52_flashmla_sparse.cu`):

1. **split-KV partial** (`run_flash_splitkv_mla_fp8_sparse_kernel`) — splits the top-k=2048 KV across `num_sm_parts` SMs, each writing a partial `o_accum` + `lse_accum` to HBM.
2. **combine** (`CombineParams`) — reads every partial back, does the log-sum-exp reduction into the final `out_latent` + `lse`.

On H100 `num_sm_parts = multiProcessorCount / kSq / (kHeads/64) = 132` (one split per SM). With `stride_o_accum_split = kSq·kHeads·kVDim = 1·64·512`, `o_accum` is `132 · 32768 · f32 = 17.3 MB`. The partial writes it and the combine reads it: **~34.6 MB round-trip ≈ 10.3 µs at 3.35 TB/s**, plus one kernel launch (~2 µs graphed/ungraphed).

**ThunderMLA transfer**: fuse partial+combine into one persistent kernel driven by a host-side instruction/tile schedule, and do the cross-split reduction through SM90 thread-block clusters / distributed shared memory instead of the HBM `o_accum` round-trip. Precedent: ThunderKittens `mla` branch, ~250 LoC device, 20–35% over FlashMLA. **But the `num_sm_parts` measurement above resets the expectation**: in the graphed forward the partial+combine only costs ~1.6 µs more at 132 splits than at the round-trip-minimizing 16, so the HBM round-trip this fusion removes is already small on the critical path once step 0 is applied. The fusion's remaining lever is the reduced-occupancy problem (running the reduction on-chip lets you use fewer splits *without* idling SMs), not the round-trip per se — a subtler and smaller win than the isolated 48 µs suggested. Worth a design issue only if bs=1 attention becomes the dominant remaining cost after step 0; not a priority now. Not a drop-in — ThunderMLA is dense; this is a port onto the vendored sparse-FP8 kernel.

## Not worth it yet

Whole-layer megakernel fusion (glue the projections + norm + RoPE + cache-pack into one persistent kernel) — real but its instruction-set/counter-sync complexity is not justified before step 0 (arena+graph) and step 1 (sparse-ThunderMLA) are exhausted. See [[../../lessons/megakernels-for-decode-latency]].

## Next

Land step 0's arena (done in `feat/glm52-kernel-bench`) + graph capture (needs R550+ driver). Then open a design issue for the sparse-ThunderMLA port (step 1). Re-baseline after each with `glm52_kernel_bench`.
