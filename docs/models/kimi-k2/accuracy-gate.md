# Kimi-K2 accuracy gate (vLLM-golden)

**TL;DR**: `pegainfer-kimi-k2/tests/vllm_golden_gate.rs` + `test_data/kimi-k2.6-vllm-golden.safetensors` give Kimi-K2 its first accuracy gate reproducible from a fresh clone (#223). Reference is vLLM (same INT4 quantized model, marlin kernels), not HF. Two passes through the public serving path: teacher-forced argmax sweep (prefill numerics, regret rule) and free-greedy decode parity (decode kernels, divergence-classified). Needs 8 GPUs + K2.6 weights; fails loudly when prerequisites are missing.

Last touched: 2026-06

## Why vLLM is the reference

Kimi-K2.6 is INT4 (compressed-tensors, pack-quantized). The general methodology
(`docs/subsystems/correctness/logits-golden-gate.md`) uses HF bf16 as golden —
for Kimi that is the wrong precision regime: HF decompresses INT4 to bf16 and
runs dense GEMMs, while both vLLM and pegainfer execute the quantized model
through marlin-style INT4 kernels. vLLM is the closest equal-precision
reference, and the same box that runs the gate can regenerate the fixture
(vLLM 0.22.0 serves K2.6 out of the box).

The H20 dev-box wipe of 2026-06-05 is the cautionary tale: the previous evidence
(PR #204's prefill logits A/B harness and fixtures) lived outside git and is
gone. Everything this gate needs is committed.

## What the gate asserts

The kimi engine exposes no logprobs (#236), so the gate cannot bound a
|Δlogprob| distribution like the Qwen gates. It asserts what the public
`EngineHandle` surface can express — which has the side benefit of testing the
*real serving path* (DP coordinator → PPLX EP → MLA kernels, TP1/DP8/EP8):

1. **Teacher-forced argmax sweep** (prefill numerics): for every tail position
   `i`, prefill `prompt + vllm_tail[..i]` with `max_tokens=1`. pegainfer's
   pick must satisfy the flatness-scaled regret rule (see Tolerances): the
   allowed distance below vLLM's argmax *in vLLM's own logprobs* grows with
   vLLM's own uncertainty at that position. An aggregate exact-match floor
   (`EXACT_FLOOR`) catches "many small in-bound flips" drift. Teacher-forcing
   keeps every position independently comparable — one tie-flip cannot
   cascade.
2. **Free-greedy decode parity** (decode kernels): generate the full tail,
   compare token-by-token; the first divergence is classified by the same
   regret rule (benign tie vs real bug) and ends that sequence's
   comparison. An aggregate coverage floor (`COVERAGE_FLOOR`) prevents mass
   early divergence from passing vacuously. Runs bs=1, concurrent (DP8
   routing), plus a determinism double-run.

## Running it

```bash
# Generate the fixture (once per vLLM/weights revision; ~20 min, 8 GPUs):
.venv/bin/python tools/accuracy/dump_kimi_k2_vllm_golden.py \
  --model-path /data/models/Kimi-K2.6 \
  --out test_data/kimi-k2.6-vllm-golden.safetensors

# Run the gate (8 GPUs; vLLM must be stopped first — both need the full node):
PEGAINFER_TEST_MODEL_PATH=/data/models/Kimi-K2.6 \
cargo test -p pegainfer-kimi-k2 --features kimi-k2 --release \
  --test vllm_golden_gate -- --nocapture
```

Build env on an H200/H20 node: `PATH` must include `/root/.cargo/bin` and
`/usr/local/cuda/bin`, plus `PEGAINFER_CUDA_SM=90a` and
`PEGAINFER_TRITON_PYTHON` (see `docs/models/kimi-k2/tp1-dp8-ep8-performance.md`).

There is no silent skip: missing `PEGAINFER_TEST_MODEL_PATH` or a missing
fixture panics. (The qwen35 gate's env-gated skip silently reported
"ok 0.00s" — this gate deliberately does not.)

Green run on an 8×H200 node (2026-06-06, 252 s): teacher-forced 375/384 exact
(97.7%, max in-bound flip 1.00); greedy bs=1 286/292 exact over 76.0%
coverage; greedy concurrent 288/294 exact over 76.6% coverage; determinism
double-run identical. Engine bringup (weights 127 s + PPLX install)
dominates the wall time.

## Tolerances

The per-position rule is **flatness-scaled regret**:

```
regret ≤ REGRET_BASE + REGRET_FLATNESS_SLOPE × (−vllm_top1_logprob)
       =      0.30   +        0.35           × (−vllm_top1_logprob)
```

where regret = how far pegainfer's pick sits below vLLM's argmax in vLLM's
own logprobs. At a confident position (top-1 ≈ 90%) the bound is ≈ 0.34 nat
— near-exact agreement; at a flat multi-modal position (top-1 ≈ 11%) it
reaches ≈ 1.07, because there is no single correct token for cross-engine
noise to deviate from. The bound depends only on the committed vLLM fixture,
so pegainfer cannot influence its own tolerance.

Calibration (three 8×H200 runs, 2026-06-05/06, vLLM 0.22.0 fixture):
~98% of positions match exactly in every pass; every cross-engine
disagreement beyond 0.25 nat occurred at a low-confidence position and
scaled with flatness — 0.375 @ lp −1.50, 0.50 @ −0.85, 0.625 @ −1.42,
1.00 @ −2.20 (each pick a top-4 vLLM token; the worst is "invent the next
fictional project name", vLLM's top-8 bunched within 1.8 nat). Fixed
thresholds were tried first and failed: each run surfaced a new
boundary-straddler at a different position, and widening a step function to
cover them goes slack exactly where vLLM is confident. The linear rule
keeps ≤ ~2 grid notches (vLLM logprobs are 1/16-quantized) of headroom over
every observed point; its binding fit is the 1.00 @ −2.20 flip
((1.00 − 0.30)/2.20 = 0.318 → slope 0.35).

| Constant | Value | Basis |
|---|---|---|
| `REGRET_BASE` | 0.30 | measured confident-position flip ceiling 0.25 × 1.2 |
| `REGRET_FLATNESS_SLOPE` | 0.35 | binding observation (1.00 nat @ lp −2.20 → 0.318), rounded up one notch |
| `EXACT_FLOOR` | 0.95 | exact-match rate per pass, measured 97.7–98.4%; catches "many small in-bound flips" systematic drift |
| `COVERAGE_FLOOR` | 0.70 | greedy parity must compare ≥70% of positions before benign divergences cut sequences short (measured 72–81%) |

Cross-engine INT4 noise (different marlin tiles, TP8 vs TP1/DP8 parallel
split, different MoE accumulation order) is larger than the same-engine bf16
ULP noise the Qwen gates absorb — the flatness allowance exists at all only
because the reference is a *different* engine. The exact-match floor is the
guard on the other flank: a real numerical bug produces many small flips
long before one large one, and 0.95 trips on a doubling of the measured
flip rate.

## Known limits / next steps

- No |Δlogprob| mean/p99 bound until kimi exposes logprobs (#236); when it
  does, extend the gate to the full methodology and tighten tolerances.
- CUDA-graph decode path is not exercised (`enable_cuda_graph: false`) — the
  PPLX path has no graph support yet (#227); add a graph pass when it lands.
- Fixture regeneration is manual; bump `meta.vllm_version` discipline applies
  (regenerate when vLLM, weights revision, or prompt set changes).
