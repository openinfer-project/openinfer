# GLM5.2 whole-step decode CUDA graph (PR5c)

> **TL;DR:** Execution record of PR5c: the whole per-rank decode step (embed → 78 layers → lm_head → device argmax) is captured into one CUDA graph and replayed every step. Measured on jz-38 8×H200, single request: **200 → 37.5 ms/step from the graph alone** (byte-identical to the PR5b record), **→ 31.3** after switching every m=1 projection to the weight-only fp8 GEMV (activation quant removed — re-gated via the #499 oracle in the new `--precision gemv` mode), **→ 25.3** after grid-striding the capacity-sized MoE quant/SiLU launches (block *scheduling*, not arithmetic, was their cost), **→ 23.4** after packing gate|up into one GEMV, overlapping the shared expert with the MoE collectives on a second stream (fork/join events inside the graph), and staging the relayout's expert ranges in shared memory, **→ 22.9 single / 22.6 at 8-way (~341 tok/s aggregate)** after fusing each layer's closing add with the next layer's input norm (ping-ponged attn buffers, bit-identical `_round` kernel) and running the DSA indexer concurrently with the MLA front's q_b/kv_a. A step-timing probe puts the inter-step host gap at ~0.05 ms — the whole step lives inside the graph; **vLLM GLM5.2 DP8/EP8 measured on the same node/workload: steady-state 20.0 ms/step (TPOT 19.8)** — the real gap is 2.9 ms, and since vLLM runs per-layer DeepEP collectives too, our dispatch/combine wait share is engineering headroom, not a physics floor (#542 has the revised plan: fp8-dispatch perf isolation — its correctness is fully gated on `wip/glm52-fp8-dispatch` — then a same-node nsys diff against vLLM's DeepEP low-latency kernels). Indexer oracle reference drift is #541 (pre-existing on main).
>
> **Last touched:** 2026-07

## Why a whole-step graph works here

- **Every step has the same shape by construction.** The DP8 lock-step protocol (#537) forwards exactly `GLM52_DECODE_GLOBAL_TOKENS = 8` tokens per global step — idle ranks send padding. Unlike kimi's full-bucket-only replay (partial occupancy falls back to eager), GLM5.2 never has a partial bucket: capture once per rank, replay every step, no eager fallback path.
- **No cross-rank barrier needed for capture.** Stream capture records without executing; in lock-step all 8 ranks hit their first step (and therefore capture) on the same global step, then each rank's first graph launch executes the collectives together. The safety ceiling is the DeepEP device timeout (~100 s) against a capture+instantiate window of tens of ms — same argument as kimi (`openinfer-kimi-k2/src/runner/worker/state.rs:261-271`).
- **The collective path is already graph-clean.** `decode_dispatch`/`decode_combine` are single stream-ordered calls with compile-time worst-case shapes, no count readback (`kCpuSync=false` — the host busy-wait path is prefill-only). The kimi decode graph replays the identical shim launch machinery (cooperative + cluster-dim 2 via `cudaLaunchKernelExC`) on H200 in production benches.

## Capture-blocker inventory (main @ 45e72ab)

The captured region is `Glm52RankModel::decode_step` (`openinfer-glm52/src/model.rs:327-415`) minus its prologue.

**1. Per-step host-varying inputs — stay OUTSIDE the graph (kimi pattern: update device buffers, then launch).**
`model.rs:338-347`: rope cos/sin row `memcpy_dtod` (host-computed slice offset), `slot_mapping`/`seq_lens`/`token_id` `memcpy_htod`. All downstream consumers already read these buffers on-device, so the prologue stays eager and the graph starts after it.

**2. `position` as a kernel scalar inside the region — must read a device pointer instead.**
`glm52_mla_cache_pack_launch(…, position)` (`mla_decode.rs:391`) bakes the cache write slot into the capture. Change the kernel to read the slot from `slot_mapping` (already device-resident, already holds `position` as i64). The host `ensure!` bounds checks inside the layer loop (`mla_decode.rs:340-344`, `model.rs:334-337`) only validate the capture step — hoist them to the prologue so they hold for every replay.

**3. Indexer mid-step D2H readbacks — the only true syncs between embed and logits.**
`indexer.rs:292` (`weights_proj` output → host) and `indexer.rs:320-327` (quant scales → host, host fold `weights · q_scale · sm_scale · h^-1/2`, htod back). 21 full-indexer layers × 2 syncs per step. Replace with one fused 32-element device kernel.

**4. Per-step allocation churn — hoist into a persistent arena.**
Remaining after #535's MoE workspace: MLA attend scratch (`ql_nope/query/ckv_fp8/ckv_scales/latent/lse/lse_accum/o_accum/v`, `mla_decode.rs:353-413`), every `fp8_linear` call (`a_fp8/a_scale_plain/a_scale/out`, `fp8.rs:131-168`), indexer (~15 buffers/full layer, `indexer.rs:273-419`), `fp8_mlp` (`fp8.rs:218-227`), layer norm/residual scratch (`layer.rs:104-224`), router outputs (`moe_decode.rs:329-331`), bookends (`bookend.rs:33/53/79`), and the last MoE alloc (`combined`, `moe_ep8.rs:292`). All shapes are static (bs=1/rank, topk 2048, `bound_rows` 2080), and identical across layers — one shared scratch set, not per-layer.

**5. Egress.** `clone_dtoh(logits)` + host `greedy_argmax` (`model.rs:413-436`). Move argmax on-device (the repo already wraps FlashInfer top-k selection K=1: `csrc/shared/flashinfer_top1.cu`); the graph ends at the sampled-token device scalar, and the per-step D2H shrinks from the full vocab row to 4 bytes, outside the graph.

**Already capture-clean (verified):** DeepGEMM MQA AOT (PDL-only `cudaLaunchKernelExC`), FlashMLA sparse (caller-preallocated scratch, no sync), FlashInfer topk (`dsa_graph_safe=true`), all decode cuBLAS through the workspace-free `g_cublas_handle` (`csrc/shared/linear.cu:323-348`). No host branch depends on runtime state inside the loop — dense-vs-MoE and full-vs-shared-indexer are static per layer index.

## Execution (all measured on jz-38 8×H200, 133-step greedy request)

| stage | ms/step | notes |
|---|---|---|
| PR5b baseline (#537) | ~200 | 8 rank threads × ~4155 launches/step through one driver |
| whole-step graph (arena + host-quiet + capture) | **37.5** | byte-identical to the PR5b/PR5a record; capture on each rank's first step, warm+capture 493 ms |
| + weight-only fp8 GEMV for every m=1 projection | **31.3** | numerics change (activation quant removed) — re-gated via #499 oracle in `--precision gemv` mode |
| + device-bounded MoE recv chain (`expert_offsets[n_local]`) | 30.1 | bit-preserving (pad rows never read); small win — block scheduling still dominated |
| + grid-strided quant/SiLU (row grid capped at 256) | 25.3 | ~100k tiny blocks/launch → ~12k; work AND scheduling now scale with real rows |
| + gate\|up packed GEMV + shared-expert ∥ collectives + smem relayout ranges | 23.4 | per-row math unchanged; overlap = fork/join events inside the capture (kimi pattern) |
| + closing-add fused with next input-norm + indexer ∥ MLA front | **22.9** (8-way 22.6, ~341 tok/s) | bit-identical `_round` fusion; indexer forks after q_resid, joins before attend |

8-way concurrency stays free throughout (same per-step wall as bs=1; ~308 tok/s aggregate at 128-tok requests). All e2e gates green each stage: determinism ×2, 8-way identical, mixed concurrency, disconnect, slot reuse, SIGTERM teardown ≤4 s.

Three latent findings shaken out on the way:

- **DeepEP cooperative+cluster launches replay fine under stream capture** — the kimi precedent held for the GLM-baked shim; no fallback path needed.
- **Block scheduling, not arithmetic, dominated the capacity-sized quant/SiLU launches** (2080×48 ≈ 100k 128-thread blocks ≈ 60 µs/layer): a device-bound early-return recovers nothing because retired blocks still get scheduled — the kernels are now grid-strided (row grid capped at 256, loop bounded by `expert_offsets[n_local]` on device).
- **The HF `glm_moe_dsa` indexer reference is a moving target** (#541): the 5.13.0 release regressed the RoPE-interleave fix (the oracle script now refuses such builds), and `5.13.0.dev0` drifted between snapshots — the indexer oracle gate needs a pinned or vllm-derived reference before its overlap threshold means anything. Not introduced by this branch (main scores the same ~1585/2048 against any freshly generated oracle).

## Where the remaining ~30 ms sits (nsys, node-granularity trace — proportions only)

Per rank per step, inflated ~15-20% by tracing:

| bucket | ms | assessment |
|---|---|---|
| dispatch_impl (75×) | ~6.7 | med 12.5 µs — mostly rank-arrival wait: per-layer straggler jitter (expert-count varies per rank per layer) + inter-step launch stagger; the tail (max 24 ms) sits at request/step boundaries |
| weight-only GEMV (588×) | ~6.6 | near the weight-BW floor (o_proj 100 MB → ~25 µs); little headroom |
| quant chain | ~4.7 → ~1 | grid-stride fix above |
| grouped expert GEMM (150×) | ~3.4 | ~8 hit experts/rank × 37.5 MB weights ≈ physics floor shared with vLLM |
| combine_impl (75×) | ~3.3 | already device-bounded (the vendored kernel's capacity-sentinel reads the device psum) — this is the cooperative-launch + NVLink push latency floor of the 2-kernel combine design |
| FlashMLA + cuBLAS absorb + norms + router + bookends | ~5 | near floor |

Attempted and reverted: **FP8 dispatch payload** (source-rank quant commutes with byte-preserving dispatch → bit-identical; vendored SF-pack support wired through config/shim/wrapper/moe_ep8) — first 8-rank run hit a DeepEP NVLink barrier timeout inside dispatch; full diff preserved on `wip/glm52-fp8-dispatch`, debug notes in #542.

Measured dead ends (all recorded in #542): the inter-step host gap is ~0.05 ms (step-timing probe — device self-feeding/pipelining buys nothing); `kDecodeNumSms` 32→16 exactly flat (combine's 42 µs/layer is protocol/NVLink latency, not SM-count-bound); FlashMLA `num_sm_parts` 32→4 REGRESSES (25.1 — fewer splits make splitkv the long pole); L2-prefetching the next layer's MLA weights on the aux stream REGRESSES (24.7 — contends with the grouped GEMM and thrashes L2); the quant-style capacity-loop lever does not exist for dispatch/combine (already device-bounded via the vendored capacity-sentinel). Remaining ~1.4 ms, in priority order: **fp8 dispatch payload** (the vendored dispatch templates carry SF-pack support; source-rank quant commutes with dispatch → bit-identical, halves NVLink bytes, deletes the recv re-quant — est. 0.4-0.7 ms, shim+wrapper+moe_ep8 rewiring), then collective latency floor and GEMV bandwidth headroom (~73% of peak).

## Next action

Follow-ups: #542 (collective latency floor / GEMV BW / `num_sm_parts`), #541 indexer reference re-pin.
