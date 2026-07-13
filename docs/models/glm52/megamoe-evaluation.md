# MegaMoE evaluation — sole MoE backend candidate for Blackwell EP

> **TL;DR:** **Verdict: adopt (perf gate PASSED; accuracy gate pending).** Phase-1a (2026-07-13, GB300 EP4, nsys under graph replay): our chain's replaceable slice at bucket-8 is **334µs/layer measured** (DeepEP dispatch/combine 92 + WO expert chain 242) vs MegaMoE fp8×fp4's clock-locked **203µs** → **−131µs/layer ≈ −9.8ms/step, b8 40.8→~31ms (−24%)**. Bucket-1 is parity (180 vs 169µs). Prefill (2048 tok/rank) 712µs/layer at 1.8 PFLOPS, NVLink fully overlapped. Remaining doors: W4A8 requant accuracy (oracle), graph capture, EP-width sweep. Ruling: Blackwell keeps ONE EP backend — adoption deletes the DeepEP v2 path from the Blackwell build (Hopper keeps it).

Last touched: 2026-07

## What MegaMoE is

One persistent SM100 kernel per MoE layer fusing dispatch → L1 grouped GEMM
(gate+up) → SwiGLU → L2 grouped GEMM (down) → top-k-weighted combine. Ranks
read/write each other's symmetric buffers directly over NVLink loads/stores
(no network stack), and a fraction of SMs stream communication tiles while
the rest run tcgen05 GEMMs — communication hides behind compute at tile
granularity. Requires every rank in one NVLink domain: NVL72 racks qualify at
any EP width; IB cross-node (our Hopper EP line) can never run it.

Source: our fork `openinfer-project/DeepGEMM` @ ecbbe74 (= upstream #364 +
our `DG_NO_TORCH` commits) already contains the SM100 kernels
(`sm100_bf16_mega_moe.cuh`, `sm100_fp8_fp4_mega_moe.cuh`).

Official dtypes: `bf16xbf16` and `fp8xfp4` (W4A8: fp8 activations × fp4
weights, per-32 UE8M0 scales). `fp8xfp8` exists only as open community PR
deepseek-ai/DeepGEMM#371 — ruled out (we don't take unmerged community code).

## Why we looked (and the prior that had to be overcome)

- D6 (2026-07-05) rejected the **SM90 port** of MegaMoE: ~200µs/call
  structural floor on H200, 2× slower than our chain. That verdict does not
  transfer: the SM100 original is DeepSeek's native design, and their own
  benchmark (PR #316) shows the **best** speedup at bs=1 (V4-Flash: 56.5µs,
  1.96× vs their legacy chain).
- Our decode floor anatomy showed the per-layer MoE chain
  (dispatch → psum → tiles → W13 mma → silu → W2 mma → combine, 7+ kernels)
  is real serial execution time — exactly the kind of cost fusion pays for
  (unlike launch slots, which graph replay already pipelines; see the
  fused-finale select negative result).
- Qualitative extras a fused kernel buys that our graph cannot: in-kernel
  dynamic tile scheduling absorbs per-rank token skew, and the per-step
  cross-rank rendezvous wall disappears (prime suspect for the unexplained
  bucket-8 p99 ~2× bimodal tail seen at every EP width ≥ 8).

## Phase-0: upstream microbench at GLM5.2 shapes

### Methodology

- Harness: upstream `tests/test_mega_moe.py` (unmodified except a one-line
  baseline guard, below), 4 processes = EP4 on one GB300 tray.
- Shapes: `--num-experts 256 --num-topk 8 --hidden 6144
  --intermediate-hidden 2048` — GLM5.2's exact MoE geometry (both dims
  512-aligned, no alignment relaxation needed).
- Timing: the test's `bench_kineto` (30 iterations, 8GB L2 flush between
  calls, `dist.barrier()` per iteration). Reported numbers are rank 0.
- **Clock locking is mandatory**: unlocked single-run numbers swing ±2×
  (fp8×fp4 tok-8 measured 203-384µs across runs). All decode-verdict numbers
  below are at `nvidia-smi -lgc 1965` (max 2070), reset afterwards. Same
  lesson as the router-select microbench.
- Buffer sizing: `--num-max-tokens-per-rank 64` for the decode ladder
  (bucket-cap regime), 2048 for prefill points.

Run (inside the dev container, weights not needed — synthetic data):

```bash
python3 tests/test_mega_moe.py --mma-type fp8xfp4 --num-processes 4 \
  --num-experts 256 --num-topk 8 --hidden 6144 --intermediate-hidden 2048 \
  --num-tokens 8 --num-max-tokens-per-rank 64
```

### Bring-up pitfalls (all hit, all solved)

1. Container ships a slim CUDA toolkit — torch headers need
   `CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include`
   (cusparse.h etc. live in the pip cu13 bundle).
2. `pip install --force-reinstall <wheel>` contacts PyPI for dependency
   metadata and hangs forever behind a stalled proxy —
   `--no-deps --no-index` installs the local wheel without any network.
3. The test's baseline import guard only catches `ImportError`; an old
   `deep_ep` without `ElasticBuffer` crashes it. One-line fix: extend the
   guarded import to `import deep_ep; assert hasattr(deep_ep, "ElasticBuffer")`.
   Consequence: the fused-vs-legacy bitwise check did NOT run — Phase-0 is
   perf-only, numerics are Phase-1's job (against our own oracle anyway).
4. sm_103 (GB300) JIT-compiles and loads fine despite DeepGEMM's arch
   detection mapping 10.x → "100a" — the feared arch-load failure never fired.

### Results — decode ladder (fp8×fp4, clocks locked at 1965MHz, max-tokens 64)

| tok/rank | global tokens | time/layer | HBM |
|---|---|---|---|
| 1 | 4 | 180µs | 632 GB/s |
| 2 | 8 | 203µs | 1.0 TB/s |
| 4 | 16 | 206µs | 1.7 TB/s |
| 8 | 32 | 203µs | 3.7 TB/s |
| 16 | 64 | 387µs | 2.6 TB/s |
| 32 | 128 | 452µs | 2.6 TB/s |
| 64 | 256 | 366µs | 3.3 TB/s |

Readings:

- **1-8 tok/rank is a flat ~200µs floor.** Weight reads at bs=1 are only
  ~25µs of it (≈8 activated experts/rank × 37.7M params × 0.5B fp4) — the
  floor is fixed overhead (in-kernel barriers, buffer sweep, tile
  scheduling), not a bandwidth wall. Software-attackable.
- The kernel **skips non-activated experts** (bs=1 HBM 632GB/s disproves the
  "always streams all weights" hypothesis from the first unlocked sweep —
  that run's tokens=8 point genuinely activates ~all 256 experts, so the
  full-weight read there was correct behavior, not waste).
- 16-64 tok/rank still noisy run-to-run; needs steadier harness in Phase-1.

### Results — prefill points (unlocked clocks, max-tokens 2048)

| tok/rank | bf16×bf16 | fp8×fp4 |
|---|---|---|
| 512 | 931µs / 326 TF | 509µs / 595 TF |
| 2048 | 1360µs / 904 TF | 712µs / **1727 TF** (NVL 443GB/s overlapped) |

75 layers × 712µs ≈ 53ms for all MoE compute of an 8192-global-token prefill
chunk. Strong — but under the P/D architecture prefill currently belongs to
vLLM, so this is a bonus, not the adoption driver.

### Reference: DeepSeek's own numbers (PR #316, EP8, their hardware)

| model | bs=1 | bs=512 | speedup vs legacy @bs=1 |
|---|---|---|---|
| V4-Flash (4096/2048, 256E top6) | 56.5µs | 146.5µs | **1.96×** |
| V4-Pro (7168/3072, 384E top6) | 108.1µs | 369.6µs | 1.61× |

GLM5.2 (6144/2048, 256E top8) interpolates to ~80-110µs on their setup; our
180-206µs suggests 2-3× tuning headroom (SM-split heuristics never tuned for
sm_103/EP4) — or methodology differences to be resolved in Phase-1.

## Our chain's baseline (what MegaMoE would replace)

Bucket-1 (from `ep4-gb300.md`, nsys eager): replaceable slice = WO expert
chain 107µs + DeepEP dispatch/combine 62µs ≈ **169µs/layer**.

Bucket-8 (Phase-1a, 2026-07-13): nsys **node-mode over live graph replay**
(`--cuda-graph-trace=node -s none --cuda-flush-interval 1000` — the flush
interval is mandatory on the GB300 CUPTI stack or zero events are collected;
this is the first time node mode worked here) on
`glm52_step_bench --moe-topo ep4 --buckets 8 --steps 40 --warmup-steps 16`,
averaging 32,400 per-layer kernel instances across 4 ranks:

| component | µs/layer @b8 |
|---|---|
| `deep_ep::elastic` dispatch+combine (5 kernels; combine_impl alone 52.3) | 92.0 |
| `moe_ep_wo_masked_mma` × 2 | 226.0 |
| tiles + silu | 15.6 |
| **replaceable slice** | **~334** |

Cross-check: total GPU busy from the same trace = 523µs/layer, consistent
with the measured b8 step (40.8ms / 75 = 544µs) — the trace is trustworthy.

## Phase-1 verdict

MegaMoE 203µs vs our 334µs at bucket-8 = **−131µs/layer → −9.8ms/step
(−24% b8 ITL)**, before pricing skew absorption and rendezvous-wall removal.
Bucket-1 parity. Perf gate: **PASSED — adopt, pending the accuracy door.**

## Decision framework

| Criterion | Status |
|---|---|
| b1 decode µs | parity (180 vs 169) — fused floor is software, tunable |
| b8 decode µs | projected win 3-7ms/step; exact measurement pending |
| Skew / rendezvous wall / p99 tail | qualitative win, unpriced |
| Accuracy (W4A8 requant, per-32 UE8M0) | untested — oracle gate is the Phase-2 door |
| Weight VRAM | fp4 halves expert weights (2.4→1.2GB/rank at EP4) |
| Graph capture | untested; single kernel + DeepEP-elastic-style in-kernel sync — same contract our shim already satisfies |
| EP width 8..64 | untested; NVL72 全域 NVLink, design says yes |
| Integration | DG_NO_TORCH AOT precedent (paged MQA logits came in this way); symmetric buffer binding is the new work |

**Ruling (user, 2026-07-13): Blackwell keeps exactly one EP backend.** If
MegaMoE passes Phase-1/2, the DeepEP v2 shim path is deleted from the
Blackwell build (stays for Hopper/IB). No dual-backend maintenance.

## Next steps (Phase-2 — adoption)

1. **Accuracy door first**: W4A8 weight requant (checkpoint fp8 128×128
   block scales → fp4 e2m1 + per-32 UE8M0) + activation ue8m0 quant (kernel
   exists: `glm52_fp8_per_token_group_quant_bf16_ue8m0`); gate = layer
   oracle (28/28 precedent from the D6 SM90 trial) then full-model oracle +
   greedy text.
2. Graph-capture proof: capture one fused call in a CUDA graph, replay
   determinism check (single kernel + in-kernel NVLink sync — same contract
   the DeepEP elastic shim already satisfies in our whole-step graphs).
3. Integration: AOT the kernel via the `DG_NO_TORCH` precedent (paged MQA
   logits), bind the symmetric-buffer allocation (the new work — DeepGEMM's
   mega buffer layout, cuMulticast under the hood), route topk_idx/weights
   from our router directly.
4. EP-width sweep 8..64 (borrow free trays, release after — same protocol as
   the EP32 bring-up).
5. Delete the Blackwell DeepEP v2 path (single-backend ruling); Hopper/IB
   builds keep it.
6. Phase-1b (parallel, informs tuning): probe the ~200µs decode floor's
   composition (SM split heuristics, barrier count) — DeepSeek's own table
   says 2-3× headroom exists.
