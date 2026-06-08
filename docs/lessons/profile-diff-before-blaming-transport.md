# Diff Kernels Before Blaming Transport

> **TL;DR:** When two profiles of "the same workload" differ in wall-time, the **first** check is: did the GPU **kernel** times change, or only the **host gaps**? Transport / serving-bridge / scheduler cost cannot change GPU kernel time — so if `cuda_gpu_kern_sum` differs between the two traces, the cause is compute/data and you stop looking at the host immediately. In the #225 "+51% HTTP overhead" hunt I nsys'd both the in-process (~26 ms) and HTTP (~34 ms) paths and **still missed it**, because I never diffed the two GPU-kernel sums — a +15.6% delta (the Marlin expert GEMM doubling) was sitting in plain view the whole time. Three process failures, all reusable.

Sibling to [moe-bench-prompt-diversity.md](moe-bench-prompt-diversity.md), which has the MoE-specific finding. This doc is the **profiling discipline** that generalizes to any perf-attribution.

## The incident

Symptom: Kimi-K2 decode read ~26 ms in-process and ~34 ms over `vllm bench serve`. Conclusion drawn (wrong): "+51% HTTP serving-bridge overhead." Both paths were profiled with nsys. The real cause — diverse prompts over HTTP route across more experts, doubling the Marlin grouped-GEMM per launch — was visible in both traces but never extracted. It took a controlled in-process `--distinct-prompts` sweep + a kernel diff to find, *after* the wrong conclusion had already been written into the roadmap.

## Failure 1 — never diffed the GPU kernel sums

A serving bridge, an IPC socket, an SSE stream, a scheduler — none of them execute GPU kernels. So **transport can only ever show up as host-side gaps between kernels, never as a change in kernel execution time.** The decisive test for "host vs device" is one command run on both traces:

```
nsys stats --report cuda_gpu_kern_sum:base <trace>.sqlite
```

Diff the totals. If the GPU kernel sum is the same and only the wall-clock differs → the host is on the critical path, *now* look at the bridge. If the GPU kernel sum **differs** → it is compute/data, and no amount of transport work could have caused it. In #225 the in-process and HTTP traces differed by **+15.6%** of GPU kernel time, entirely in one kernel (Marlin, 45.7 → 89.0 µs per launch). That single diff would have killed the transport hypothesis on day one. I looked at each trace in isolation and never put them side by side.

## Failure 2 — compared mismatched metrics

The "26 vs 34" gap that launched the investigation was **26 = identical-prompt *first-decode-step*** vs **34 = diverse-prompt *steady-TPOT***. Two different metrics on two different workloads. A clean same-metric sweep later showed the honest splits are first-step 26→28 (+7%) and steady 28.7→32.6 (+14%) — most of the original "gap" was the metric mismatch plus decode context growth, not the thing being blamed. **Pin one metric across both configs before quoting a delta.** A number is only a comparison if both sides measure the same thing.

## Failure 3 — annotated a tail instead of chasing it

The profile showed `dispatch_impl` with p99/p50 ≈ 21× — median 14.7 µs, max **15 ms** — and it got written down as "one-off rank skew." A 15 ms spike in a 26 ms step budget is not a footnote; it is the workload telling you something is wrong with arrival/routing. **A tail that large is a lead, not a caveat.** (It turned out to be benign rank-arrival skew here, but that was confirmed by chasing it, not assumed by annotating it.)

## The checklist

When attributing a wall-time gap between two configs:

1. **Diff the GPU kernel sums first.** Kernels differ → compute/data, stop looking at the host. Only host gaps differ → host is on the critical path.
2. **Same metric, both sides.** first-step vs first-step, steady vs steady. Never quote a delta built from two different metrics.
3. **Instrument the host orchestrator's own per-step clock** to confirm whether the host is even on the critical path — don't infer it from a profiler. (For Kimi the DP coordinator clock showed host = 0.1 ms/step, ranks balanced 8/8 — the bridge was exonerated by direct measurement, not by a trace.)
4. **Hold exactly one variable.** The in-process `--distinct-prompts` sweep changed routing with transport fixed; that isolation is what made the result evidence instead of suspicion.
5. **Chase the tail, don't annotate it.** max/p50 > 2× is a lead to pull, not a sentence to write.

And the meta-rule the whole episode is really about: **a root cause you have not put a number on is a hypothesis, not a finding.** Write the suspicion down as a suspicion; do not let it harden into a roadmap claim until a controlled measurement backs it.
