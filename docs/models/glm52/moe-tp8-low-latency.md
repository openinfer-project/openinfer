# GLM5.2 decode MoE: EP8 → TP8-sharded persistent kernel + LL allreduce

> **TL;DR:** Plan (no code yet) to replace the bucket-1 decode MoE path — the DeepEP EP8
> dispatch/grouped-GEMM/combine chain, at ~103 µs/layer kernel-resident / ~145 µs/layer
> wait-inclusive the single largest bucket of the 19.5 ms solo step — with one TP8-sharded
> cooperative persistent kernel per layer plus a low-latency (LL) packet reduce-scatter.
> A standalone prototype on the same 8×H200 node measures the TP form at **~36 µs/layer**;
> target is solo **19.5 → ~14–15 ms/step**. EP8 is *not* deleted: expert sharding is chosen
> at load time (weights are repacked during H2D anyway), so EP8 remains the high-throughput
> launch configuration while TP8 becomes the low-latency one. Inspired by the latency-first
> executor design of TileRT (persistent kernels, communication as graph nodes, no NCCL in
> the hot path); everything here is measured and implemented independently in this repo.
>
> **Last touched:** 2026-07

## Why topology, not more kernel work

The EP8 wait structure has been mined out (evidence in
`whole-step-decode-graph.md`):

- The MoE block (collectives + quant/SiLU glue + masked GEMM) is ~8.3 ms of the
  19.5 ms solo step. Dispatch/combine alone are 4.75 ms kernel-median /
  7.73 ms wait-inclusive.
- **Dispatch is rank-arrival-wait bound, not byte bound**: shrinking the payload to
  fp8 measured perf-neutral.
- Fused-MoE via DeepGEMM MegaMoE was measured and rejected: ~200 µs/layer structural
  floor at decode payloads, ~2× slower than the current chain.
- `kDecodeNumSms` tuning is flat.

What's left is the topology itself. EP8 partitions *experts* across ranks, so tokens
must travel to their experts and every layer waits for the slowest rank's arrival.
TP8 instead stripes *every* expert across all ranks (each rank holds a 1/8
intermediate-dim slice of all 257 experts): every rank does identical, uniform work
per layer and partials are merged with a reduce-scatter. **Per-rank expert bytes read
are exactly equal between the two shardings** — the entire gap is wait structure,
kernel boundaries, and small-op floors. A structural bonus: the per-layer combine
straggler spread (~44 µs/layer under EP8) disappears by construction.

## What changes, what doesn't

Only the **routed-expert weight sharding** changes, and only for the bucket-1 decode
graph at first:

| Plane | Today (DP8×EP8×TP1) | After this slice |
|---|---|---|
| Request/KV data plane | DP8 lock-step, one graph per bucket | unchanged |
| Non-expert stack (attention/MLA/indexer/dense/lm_head, ~19.6 GiB) | replicated per rank | unchanged |
| Routed experts (~85.5 GiB/rank) | 32 whole experts per rank (`weights.rs:275-290`) | 1/8 row-slice of all 257 experts (bucket-1 graph; pilot = 8 layers) |
| Shared expert | aux-stream fork/join around the routed path | folded into the persistent kernel (its I=2048 sliced 256/rank the same way) |
| Buckets 2/4/8 | EP8 chain | unchanged until the M3 decision |

Per-layer target form (replaces the 9-kernel EP8 chain + aux-stream shared expert at
`model/mod.rs:972-1012`); lock-step bucket-1 has `global_tokens = 8` (7 pads,
`model/mod.rs:890`), so the TP kernel processes all 8 global tokens (m=8) with a
uniform shape on every rank:

```text
① LL allgather: each rank pushes its token's post-attn hidden (12 KB bf16) to all peers
   (pad tokens pushed too — uniform shape, no owner branches)
② one cooperative persistent TP-MoE kernel (m=8):
   phase A: 8× rmsnorm + router (noaux_tc semantics: sigmoid+bias top-8, renorm ×2.5,
            computed redundantly on every rank)
   phase B: gate|up GEMV over the 256-row I-slice for 8 tokens — each weight load
            serves 8 rows (8× arithmetic intensity) — + SiLU
   phase C: down GEMV (m=8) producing 8 partials
   epilogue: LL reduce-scatter — token j's partial goes only to rank j;
             rank j sums 8 contributions + residual
```

## Evidence (standalone prototype, same node, jz-38 8×H200)

Measured with a standalone microbench prototype using the exact GLM5.2 shapes
(H=6144, E=256+1 shared, per-rank I-slice 2048/8=256, per-128 fp8 scale semantics).
The prototype kernels land in-tree with the M0/M1 PRs.

- One TP-shard MoE layer as a single cooperative persistent kernel: **30.5 µs cold-L2**
  (vs 38.1 for the best fused-3-kernel graph variant, vs 52 for an 8-kernel chain);
  CPU golden comparison max rel err 3.6e-3.
- LL packet one-shot allreduce with the epoch tag embedded in each 128-bit packet
  (zero fences, zero separate flags): radix-8 marginal cost **5.8 µs/layer**
  (radix 2/4: 1.5/2.5). Epoch advance is a device-side monotonic counter with
  odd/even payload double-buffering, so graph replay advances it without changing
  kernel parameters; 8-rank lock-step numeric identity checks pass.
- Total ≈ **36 µs/layer** (m=1 form) vs the EP8 chain's 103–145 µs/layer.

Arithmetic expectation: 75 MoE layers × (65–105 µs) ≈ **−5 to −8 ms/step** solo.
Per this repo's own calibration discipline, all claims settle on e2e A/B via
`openinfer-server/src/bin/glm52_step_bench.rs`, never on trace proportions
(`whole-step-decode-graph.md`, calibration section).

## Constraints and standing decisions

1. **No dual layout inside one process.** The expert slab is ~85.5 GiB/rank; two
   layouts cannot coexist in 141 GiB HBM (this is why the loader packs experts at
   H2D time, `weights.rs:31-37`). Consequences:
   - The **pilot** dual-resides only N=8 layers (~+9.7 GiB/rank — fits).
   - Full migration makes sharding a **launch-time configuration** (e.g.
     `--moe-topo tp8|ep8`): the loader picks the slicing during H2D repack. EP8 is
     retained as the high-throughput configuration — at large buckets its wait cost
     amortizes per token while TP's thin per-rank GEMM (I-slice 256) loses efficiency,
     so EP8 plausibly stays the right choice there. The M3 measurement decides the
     default, not a code deletion.
   - Cost accepted: two MoE backends means two numeric gates (EP8 oracle exists; TP
     oracle is new in M1) and dual e2e coverage until/unless M3 retires one.
2. **TP slices align with fp8 scale blocks.** I-slice 256 = 2×128, so the W13 row-slice
   and W2 column-slice each cover whole 128-blocks — no scale-boundary straddling.
3. **Numeric contract is a tolerance gate, not bit-parity.** TP partial-sum order
   differs from EP8 by construction. Precedent: the batch-4/8 mma GEMV was accepted on
   `max_rel < 2e-2` + ≥90% bf16-exact + e2e greedy coherence + DSpark accept-rate
   parity (`whole-step-decode-graph.md`). TP-MoE follows the same recipe, plus a TP
   layer oracle gate modeled on `oracle/layer_ep8.rs`. Near-tie router flips need the
   same bounded-allowance mechanism the EP8 oracle already has (2 known outliers).
4. **Cooperative launch under stream capture has no in-repo precedent** but is
   verified on the same node/driver (the prototype's graph timing captures
   `cudaLaunchCooperativeKernel` via `cudaStreamBeginCapture`), and the DeepEP shim
   already replays cooperative+cluster launches inside the production graph. Residual
   risk is plumbing the cudarc/driver-API path, not feasibility.
5. **Two iron rules for the LL protocol** (validated in the prototype; do not relax):
   the epoch only ever advances device-side (graph parameters are frozen), and
   payloads are double-buffered by epoch parity (a fast rank may lead by at most one
   iteration). Spins get an upper bound + `__trap` — crash early instead of letting a
   half-paired collective ride the ~100 s DeepEP-style device timeout.
6. **Comm/stage buffers follow the caller-owned resident-arena convention**
   (`glm52_moe_gemv.cu:281-287` style): pointer-stable across capture/replay.

## Work breakdown (each step has an independent accept/kill)

| # | Content | Accept / kill |
|---|---|---|
| **P0** | Mechanism probes, before any kernel work — these are the only kill-level unknowns: (a) LL buffer `cudaDeviceEnablePeerAccess` coexisting with NCCL ≥2.30 symmetric-window registration in the real process shape (single process, 8 threads, primary contexts); (b) cooperative launch under cudarc stream capture + graph replay | Both probes pass on jz-38. **Kill: either fails → redesign before spending on kernels**. **(a) PASSED 2026-07-07** — standalone probe (`p0_probe_peer_nccl.cu`, lands with the M1 PR), 8×H200 / NCCL 2.30.7: symmetric window + allreduce, peer-access LL mailboxes, and 50 interleaved NCCL+LL rounds all verify in **both** init orders (NCCL-first and peer-first). Fidelity caveat: probe drives the NCCL *host* API over the window; the DeepEP shim's device API (ncclDevComm/GIN) coexistence is re-checked implicitly in M1 when both run in-process. **(b) PASSED 2026-07-07** — `p0_probe_coop_graph.cu`: cooperative launch via `cudaLaunchKernelExC`+attr (the DeepEP-shim launch shape) captured with `cudaStreamBeginCapture`, instantiated, **50 replays enqueued back-to-back with zero host involvement** — epoch advances via a device-side counter, parity double-buffered LL packets across all 8 GPUs, kernel self-verifies every round; NCCL window allreduce healthy before capture and after the storm. Kernel-design lesson baked into the probe: **no block barrier may sit in a thread-divergent branch of a kernel that also calls `grid.sync()`** — threads parked at the grid barrier never release a `__syncthreads`, deadlocking the block (the probe's first version hung on every coop variant this way; isolation matrix + rerunning the R4 ground truth pinned it) |
| **M0** | Kernel extension m=1 → m=8 in the standalone prototype (no pegainfer changes) | **DONE 2026-07-07, PASSED**: solo **55.2 µs/layer** (1.54× the m=1 anchor's 35.7 — under the 2× kill line; **6.9 µs/token = 5.2× per-token amortization**), diverse (E_active=22) 74.8 µs; CPU golden 0.0030/0.0044 < 5e-3 both configs; grid 264. Final form differs from the plan sketch in two load-bearing ways: (1) the compute engine is the **batch-8 m16n8k16 mma port** from `glm52_moe_gemv.cu` (σ-permutation, fp8→bf16 lossless, f32 tensor-core accum) — the scalar 8-token-reuse form measured 171 µs (instruction wall + occupancy); (2) activations stay in **global memory (L2-hot), not smem** — 96 KB smem capped occupancy and the mma B-fragment reads pointers anyway. Phases run as a global warp-job pool (expert × proj × 16-row tile × k-slice) with f32 partial scratch and fixed-order epilogues (deterministic). Two portable traps recorded: mma k-slices MUST be multiples of 128 (scale-fold contract, now static_asserted), and per-token top-8 + union compaction must not serialize on one block (was 41.5 µs of grid.sync parking; block-parallel top-8 + ballot-scan → 8 µs). Remaining headroom: phase B 20.1 µs vs ~7 µs weight roofline. Details: skeleton_bench `M0_实验记录.md` |
| **M1** | Pilot integration, N=8 layers dual-resident: loader TP-slice path (pilot layers only), `Glm52LayerMlp::MoeTp` arm, LL buffers + epoch counters in a resident arena, cooperative-launch FFI, TP layer oracle gate | `glm52_step_bench` bucket-1 A/B: **pilot 8 layers ≥ −0.4 ms total** (~−55 µs/layer, generous margin); oracle + determinism ×2 green. **Kill: < −0.2 ms** → attribute before proceeding |
| **M2** | All 75 layers behind a launch-time sharding switch; pad semantics (pads computed, outputs dropped) | solo **≤ 15.5 ms/step**; all e2e gates (determinism, 8-way, disconnect, teardown) green; launch-ahead unaffected (feed kernel untouched) |
| **M3** | Bucket decision by measurement: masked grouped GEMM on TP slices at m=16/32/64 (per-rank N thins to 512 gate|up) vs staying EP8 for buckets 2/4/8. Within one process the layouts are exclusive (constraint 1), so this sets the *default* of the launch-time switch and decides whether EP8 ever retires | Decision record written from M2 data + the masked-GEMM-on-slice microbench |
| **M4** | Follow-on (next phase): attention projections TP + o_proj allreduce reusing the same LL machinery; MTP verify path | separate doc when reached |

Benefit claims are reported **per workload**: solo (pads share routing, E_active≈9,
maximum win) and diverse bucket-1 (E_active≈57, per-rank ~268 MB/layer expert reads —
same bytes as EP8, but the persistent kernel's tile loop lengthens and dilutes the win).

## Risks, ranked

1. **m=8 E_active dilution** (above) — report solo/diverse separately; no blended claim.
2. ~~**Peer access × NCCL window registration interference**~~ — cleared by P0 probe (a), 2026-07-07 (host-API scope; shim device-API coexistence lands with M1).
3. **Cooperative kernel occupancy vs existing aux-stream overlap** — the shared expert
   folds into the kernel (no loss by construction), and indexer∥MLA overlap lives
   outside the MoE segment; confirm with an nsys graph trace anyway.
4. **Grid size** — re-sweep at m=8 after integration.
5. **Near-tie routing flips under the tolerance gate** — bounded allowance, as EP8.

## Code map (verified against main @ 9c169f9)

| Topic | Location |
|---|---|
| EP8 MoE chain (replacement target) | `openinfer-glm52/src/moe_ep8.rs:193` (`glm52_moe_ep8_routed_forward`); call site `model/mod.rs:972-1012` |
| Router semantics | `moe_decode.rs:396-419` (`run_router_into` → `glm52_router_noaux_tc_launch`) |
| Shared expert (fold target) | `moe_decode.rs:169` (`forward_into`); aux-stream fork/join `model/mod.rs:980-1004` |
| Weight loader (gains the TP-slice path) | `weights.rs:93` (`expert_placement`), `weights.rs:275-290` (rank slicing), `weights/load.rs` |
| Per-layer MLP enum (new arm) | `layer.rs:59` (`Glm52LayerMlp`) |
| Graph capture infra | `openinfer-core/src/cuda_graph.rs`; per-bucket state `model/mod.rs:285-297` |
| New-kernel plumbing | drop `.cu` under `openinfer-kernels/csrc/glm52/` (auto-collected, `build.rs` `is_glm52_source`); FFI `src/ffi/glm52.rs`; wrappers `src/ops/glm52/`; register in `KERNELS.md` |
| GEMV conventions to match (dot/dequant/scale) | `csrc/glm52/glm52_moe_gemv.cu:75-148` |
| Multi-GPU contexts / comm buffers | single process, 8 threads (`runner.rs`), per-rank primary context (`weights/context.rs`); LL buffers = per-device `cudaMalloc` + peer access + pointer table |
| A/B anchor | `openinfer-server/src/bin/glm52_step_bench.rs` (bucket-1 solo = the 19.5 ms reference) |
| Numeric gates | `openinfer-glm52/src/oracle/layer_ep8.rs` (EP8 layer gate; TP twin to be added); e2e gates per `serving-status.md` |

## Next action

P0 fully green (both probes passed 2026-07-07). Next: M0 — extend the prototype
TP-MoE kernel from m=1 to m=8 (8-token smem residency, phase-B weight reuse,
per-token top-8, reduce-scatter exit), re-sweep grid size, CPU-golden gate.
