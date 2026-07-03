# GLM5.2 whole-step decode CUDA graph (PR5c)

> **TL;DR:** Design + execution record for capturing the entire per-rank decode step (embed → 78 layers → lm_head → device argmax) into one CUDA graph, replayed every step. Target: the ~200 ms/step fixed cost measured after PR5b (#537) — prime suspects are the host launch path (8 rank threads × ~4155 kernels/step through one driver) and per-step alloc churn (~5 ms MLA/indexer spine). Three stages, each gated on jz-38 byte-parity vs the PR5b record: (1) persistent decode arena, (2) host-quiet step (kill the 2 indexer D2H readbacks + the `position` kernel-scalar), (3) capture via the shared `CudaGraphState`. DeepEP decode is already verified host-quiet and its cooperative+cluster launches replay under graph in kimi production — the substrate risk is retired by precedent.
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

## Plan

| stage | content | gate |
|---|---|---|
| 1. arena | `Glm52DecodeScratch` owned by the rank model: every §4 buffer allocated once at build; `fp8_linear`/`fp8_mlp` take caller scratch | local build + jz-38 byte-parity vs PR5b record; ms/step A/B (expect ~5 ms back) |
| 2. host-quiet | indexer fold kernel (kills both D2H), cache-pack slot from device pointer, bounds checks hoisted, device top-1 egress | byte-parity again (top-1 tie-break must match host argmax on the record) |
| 3. capture | `CudaGraphState::run_or_capture` around embed→argmax per rank; prologue (input uploads + rope row copy) stays eager | full PR5b gate suite rerun (byte-parity, 8-way concurrent, slot reuse, disconnect, SIGTERM) + ms/step A/B vs the 200 ms baseline |

Stages are commits within one PR (`feat/glm52-decode-graph`); each stage must hold byte-parity on its own so a regression bisects to one stage.

## Open questions / risks

- **Cooperative launch under capture:** proven by kimi on H200 driver-wise, but GLM's shim instantiation differs (parameterized `impl.cuh`); stage 3's first jz-38 run answers it definitively. If capture of the cooperative dispatch fails, fallback = graph the per-layer compute and leave dispatch/combine eager (still removes ~95% of launches).
- **Top-1 tie-break:** host `greedy_argmax` and FlashInfer top-1 selection must agree on ties for byte-parity; verify on the recorded outputs before swapping (near-tie positions exist in these gates' history).
- **Capture-step inputs:** the first step per rank may be padding `(0,0)` — the graph shape is input-independent so this is fine, but the capture must not be conditional on "real request" anywhere.

## Next action

Stage 1 (arena) in progress on `feat/glm52-decode-graph`.
