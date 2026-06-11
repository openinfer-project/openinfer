# Mixed-Load ITL — long prompts arriving into steady-state decode (Qwen3-4B)

**Created**: 2026-06-08
**TL;DR**: A long prompt admitted mid-decode freezes **every** active decode for
the whole prefill (the stalled inter-token gap ≈ prefill wall-time: 4k ≈490ms,
8k ≈1180ms, 12k ≈2230ms); gaps with no prefill in flight stay at baseline TPOT
(~14ms). Whether that reaches headline **ITL p99** is a *frequency* question — it
only does once stalls exceed ~1% of decode gaps, a fraction that grows with both
**QPS and prompt length**. Swept qps×warm×prompt on RTX 5070 Ti: low QPS +
moderate prompt keeps p99 at baseline (~15ms); ≥8k prompts or ≥1 req/s blow it to
0.5–3.8s; **any** prefix-reuse (warm) collapses it back to baseline.
**Decision: chunked prefill is a conditional no-go** — justify only behind a hard
ITL-p99 SLA in a sustained-arrival or long-prompt, cold-prefix regime.

Reproduce the full cube with [`scripts/sweep_mixed_itl.sh`](../../scripts/sweep_mixed_itl.sh).
The canonical (0.5 req/s, 4k, cold) point is a tracked profile in the Qwen3-4B
snapshot — `bench_snapshots/{gpu}/qwen3-4b.json` under `mixed_itl`, refreshed by
`bench_serving snapshot` alongside the prefill/decode profiles (reported by
`compare`, but not regression-gated — see below).

## Why this measurement exists

[Issue #244](https://github.com/xiaguan/openinfer/issues/244). Qwen3-4B's
scheduler admits a pending prefill into a **unified step** when decodes are active
(`pending && active → Unified{pending}`,
[plan.rs:55-68](../../openinfer-qwen3-4b/src/scheduler/plan.rs#L55-L68)) with no
per-step token budget: the long prefill and all active decode rows run in one
forward pass, so that step's wall-time balloons to ~the prefill time and every
decoding request stalls for that one inter-token gap. The scheduler doc carried
an anecdotal "+38% ITL p99 tail vs vLLM (291 vs 211ms)"
([scheduler.md](../subsystems/scheduler/scheduler.md)) from a single QPS=2
random-length run; the maintainer stance is **chunked prefill is not automatically
the fix — measure first**. This characterises the tail across the QPS × prompt-length
× prefix-reuse space. Chunked prefill itself is out of scope for #244.

## Method

In-process, deterministic, single-GPU. `bench_serving mixed` is an **open-loop** driver:

- **Background (decode-heavy steady state):** N long-lived greedy decode streams
  (`ignore_eos`), kept alive for the whole run, timestamping every token.
- **Injector:** one thread submits a long prompt every `1/qps` s (greedy,
  `output_len=1` → prefill-dominated), draining each; `[submit, last-token]` is an
  *in-flight-prefill window*.
- **Metric:** background inter-token gaps, each tagged **stall** (overlaps a prefill
  window) or **steady**; reported as `mixed_itl.{all,steady,stall}` p50/p95/p99/max
  vs a decode-only **baseline**.
- **`--inj-warm-frac`** picks the cold/warm mix: cold = a distinct prompt per
  injection (real prefill; default-on prefix cache #216 would otherwise serve
  repeats as ~37ms hits and *hide* the stall); warm = a shared prompt (cache hit
  after the first). Evenly interleaved, so e.g. 0.5 = every other.

**Sizing (16 GB) for RTX5070Ti.** Admission reserves the **full** `prompt + max_tokens` KV per
request; the pool here is ~2332 blocks ≈ 18.6k tokens. So `bg_conc × (bg_prompt +
bg_out) + inj_prompt` must fit, and `bg_out` must outlast the run. **16k prompts
OOM** (prefill *activation* scratch, not KV) — **12k is the feasible ceiling** on
this card. The sweep holds background at **4-way / 512-prompt / 1024-out** so one
baseline covers every cell and 4k/8k/12k all fit.

**Thermal.** A 10k+ prefill is a heavy compute burst; a back-to-back sweep
throttles the GPU and *fabricates* saturation (12k prefill inflated 2235→4400ms).
The sweep script inserts inter-cell cooldowns — the throttle-check table below
(cold prefill ≈ constant across QPS) confirms the numbers are clock-clean.

## Results — sweep (RTX 5070 Ti, Qwen3-4B, greedy, 4-way background)

**Baseline (decode-only): p50 13.7 / p99 14.7 ms.** Cells are **ITL p99 (ms)**;
`*` = saturated (prefill > 1/qps, prefills overlap, decode starves).

**qps = 0.25**
| prompt | cold | warm½ | warm |
|--------|-----:|------:|-----:|
| 4k  | 14.9 | 16.1 | 15.2 |
| 8k  | 15.1 | 14.5 | 14.8 |
| 12k | 19.4 | 19.4 | 14.9 |

**qps = 0.5**
| prompt | cold | warm½ | warm |
|--------|-----:|------:|-----:|
| 4k  | 16.7 | 14.9 | 15.2 |
| 8k  | **1161** | **28.8** | 14.6 |
| 12k | **3270\*** | **3812\*** | 19.5† |

**qps = 1.0**
| prompt | cold | warm½ | warm |
|--------|-----:|------:|-----:|
| 4k  | **482** | **482** | 24.7 |
| 8k  | **1175\*** | **1166\*** | 28.7† |
| 12k | **3796\*** | **2302\*** | 34.5† |

(warm½ = `--inj-warm-frac 0.5`; warm = 1.0. † = the only saturation is the first
injection's one-time cold cache-fill — the rest are hits, so p99 is clean.)

**Throttle-check — cold prefill median (ms), should be ≈ constant across QPS:**
| prompt | qps 0.25 | 0.5 | 1.0 |
|--------|---------:|----:|----:|
| 4k  | 494 | 497 | 493 |
| 8k  | 1213 | 1170 | 1167 |
| 12k | 2192 | 2180 | 2304 |

## Reading the cube

Two independent knobs explain every cell:

1. **Severity = the prefill wall-time** (throttle-check row): the one stalled gap ≈
   the entire prefill. Scales ~linearly with prompt (4k→8k→12k ≈ 0.5→1.2→2.2s).
   `steady` gaps and **p50 stay at baseline in every cell** — purely a tail effect.

2. **Frequency decides if it reaches p99** = stall-gap fraction
   `≈ qps / (qps + (1−qps·prefill_s)/TPOT)`. It rises with *both* QPS and prompt
   length (a long stall eats decode time, inflating its own share). p99 ≈ baseline
   while frac < ~1%, and climbs toward the per-event stall above it. So the
   **p99-break frontier moves left (lower QPS) as prompts grow**:
   - **4k** stays clean until **1 req/s**.
   - **8k** breaks by **0.5 req/s** (1161ms).
   - **12k** saturates by **0.5 req/s** (3.3s).
   - **qps 0.25 is clean at every length** — even a 12k stall only hits `max`.

3. **Prefix reuse defeats it universally** (`warm` column: 14.6–34.5ms everywhere) —
   a cache hit isn't a prefill. **warm½** only helps when halving the cold rate
   drops below the knee: rescues 8k@0.5 (1161→**29**) but not 12k@0.5 (the cold
   half alone saturates → 3.8s).

4. **Saturation** (`*`): when `qps·prefill_s ≳ 1`, prefills run back-to-back and
   decode never recovers (stall% → ~50–60%, even p50 rises). This is a throughput
   wall, not just a tail — chunked prefill can't add prefill FLOPs; needs
   rate-limit / bigger card.

## Decision — chunked prefill: conditional no-go

**Don't implement it for #244's profile.** Moderate prompts at genuinely low QPS
into a decode-heavy batch (the green cells: all of qps 0.25, 4k@0.5, plus any
warm) keep ITL p99 at **baseline order (~15ms)**; the documented stall lives only
at p99.9/max, tolerable for most decode SLAs. The "varied-length workloads break
the waves naturally" stance holds at p99.

**Implement it only** behind a **hard ITL-p99 SLA** *and* a regime in the **bold
cells** — sustained arrival (≥1 req/s) **or** routinely long, **cold** prompts
(≥8k), where p99 is **0.5–3.8s (30–250× baseline)** and no per-token SLA survives.
Chunked prefill (bounding per-step prefill tokens so decode rows are serviced
between chunks) is the right fix there. In the saturated cells it's necessary but
not sufficient (also rate-limit).

This replaces the single-point anecdote with a severity-vs-frequency model and the
p50/p99/max numbers #244's acceptance boundary requires.

## Reproduce

```bash
# Build (RTX 5070 Ti / Arch WSL2; SM 120, absolute Triton python):
CUDA_HOME=/opt/cuda NVCC_PREPEND_FLAGS="-ccbin g++-13" \
  LIBRARY_PATH=/usr/lib/wsl/lib:/opt/cuda/lib64 \
  OPENINFER_CUDA_SM=120 OPENINFER_TRITON_PYTHON=/abs/.venv/bin/python \
  cargo build -r -p openinfer-server --bin bench_serving

# Quick sweep — the severity axis in two cells (~1 min, no thermal control).
# 4k stays at baseline order, 8k blows the p99 to ~1s: the whole finding in two runs.
# (Cold, qps=0.5. Omits 12k — it sits at the 16 GB activation ceiling and OOMs run-to-run;
#  the full cube below adds the qps × warm axes, cooldowns, and the throttle-check.)
for p in 4096 8192; do
  ./target/release/bench_serving --model-path models/Qwen3-4B \
    --format json --out /tmp/itl_$p.json \
    mixed --bg-prompt-len 512 --bg-concurrency 4 --bg-output-len 1024 \
          --inj-prompt-len $p --inj-output-len 1 --qps 0.5 \
          --num-injections 3 --warmup 3 --inj-warm-frac 0.0 --skip-baseline >/dev/null 2>&1
  echo "${p}: p99=$(python3 -c "import json;print(f\"{json.load(open('/tmp/itl_$p.json'))['mixed_itl']['all']['p99_ms']:.1f}\")")ms"
done
# → 4096: p99≈20ms   8192: p99≈1290ms

# Full qps × warm × prompt cube (with cooldowns + throttle-check):
bash scripts/sweep_mixed_itl.sh

# Canonical cell — refreshed into the tracked snapshot alongside prefill/decode:
CUDA_HOME=/opt/cuda LIBRARY_PATH=/usr/lib/wsl/lib:/opt/cuda/lib64 \
  ./target/release/bench_serving --model-path models/Qwen3-4B snapshot --warmup 5 --iters 20
# → writes bench_snapshots/{gpu}/qwen3-4b.json (mixed_itl = 4k cold @ 0.5 req/s, 4-way bg)
```

The canonical cell is folded into the `snapshot` subcommand as the `mixed_itl`
profile, so it refreshes with the prefill/decode profiles and its history lives in
git ([bench-regression.md](../conventions/bench-regression.md)). The full
qps×warm×prompt sweep above stays a **standalone diagnostic** — too noisy/expensive
for the tracked set. `compare` prints the ITL delta for context but does **not**
gate on it: the stall tail is thermally and run-to-run noisy (see Caveats).

### Caveats
- **16k+ prompts OOM** on 16 GB (prefill activation scratch); 12k is the ceiling.
- **Knee cells are run-to-run noisy**: where stall-fraction sits right at ~1%
  (e.g. 4k@0.5, 4k@1.0) p99 wobbles ~15–19ms or catches/misses a stall. Trends are
  solid; borderline single p99s aren't.
- **Cooldowns are mandatory** for a sweep — without them heavy prefills self-throttle
  and fabricate saturation (see Thermal).
- **Cold prompts are the worst case**; if your real long prompts share prefixes the
  tail is milder (warm column). The harness salts cold prompts so the cache can't
  hide the stall.

## Next

If a deployment surfaces a hard ITL-p99 SLA under sustained/long-cold-prompt load,
re-run the sweep at its QPS/prompt mix and compare mixed vs baseline p99 to
re-open the chunked-prefill decision.
