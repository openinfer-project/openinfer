# MoE Decode Imbalance And DPLB Lessons

> **TL;DR:** DPLB gives us a vocabulary for future PegaFlow/WiDeep MoE+EP decode imbalance, not an algorithm to implement immediately.
>
> - **Want:** raw engine progress that lets an external router approximate decode-worker load.
> - **Avoid:** putting router state, safe-margin math, or BR-0/BR-H decisions inside the engine.
> - **Why:** sticky KV placement turns transient imbalance into long-lived imbalance, and the slowest decode worker sets step latency.
>
> **Status:** Discussion record. No implementation, schema, or connector API is committed here.

## Scope

This note records cross-model MoE and expert-parallel serving lessons (companion: `docs/lessons/moe-zero-prefill-long-prefill.md`). It records what we learned from "Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving: Practical Online Routing at Scale" ([arXiv:2605.06113](https://arxiv.org/abs/2605.06113)) and how that should shape future PegaFlow/WiDeep planning.

The goal is a reusable decision record, not an implementation plan. Specific APIs, telemetry formats, and scheduler changes are outside this document because they depend on the future WiDeep/PegaFlow runtime boundary.

## Paper-Fact: Problem

DPLB's core problem is decode-side data-parallel imbalance. In a serving cluster, each decode step is gated by a synchronization barrier across data-parallel workers. The step finishes when the slowest worker finishes, so lighter workers spend time idle when worker loads diverge.

The load is sticky because decode requests carry KV state. Once a request is assigned to a decode worker, moving it requires moving its KV cache, so the router should treat placement as a long-lived state decision rather than a cheap per-step queue choice. The load also grows over time because every generated token increases the request's KV footprint.

## Paper-Fact: Measurements

The paper's vocabulary is worth reusing:

- `L_g(k)`: worker `g`'s decode load at step `k`, defined as the sum of per-step workload across active requests on that worker.
- `M(k)`: the maximum worker load at step `k`.
- `m_g = M(k) - L_g(k)`: the worker's safe margin before it overtakes the current heaviest worker.
- `I(k) = G * M(k) - sum_g L_g(k)`: total imbalance at the step.
- `s_i`: the request's admission-time load, approximated by prompt/prefill length.

`L_g(k)` is not simply "KV token count". KV tokens are a useful proxy when per-token decode cost is roughly uniform. MoE and EP can break that approximation because expert routing and collective arrival skew add work variance that token count alone does not capture.

## Paper-Fact: Boundary

The paper's deployed system keeps the routing state outside the engine. It uses a stateful proxy in front of vllm-ascend workers, maintains the cross-worker snapshot in that proxy, and recovers token progress from streamed decode output. The engine binary is not modified. In the rest of this document, that proxy-like owner of routing state is called the external state layer.

## Derivation: What We Want

For future PegaFlow/WiDeep MoE+EP serving, the risk has the same shape as DPLB: decode placement and MoE/EP routing can produce persistent worker imbalance, and that imbalance shows up as barrier idle and tail TPOT.

The paper's engine boundary matters more than the specific BR-0 or BR-H algorithm for us right now.

We want the engine to provide raw progress that lets an external state layer approximate decode-worker load. The external layer should own the worker snapshot, routing quantities, capacity policy, and admission decisions.

This keeps the engine useful to multiple routing policies. A future router may use DPLB-style safe margins, a different heuristic, or a learned policy; all of them need raw progress, but none of them require the engine to own router state.

## Derivation: What We Avoid

We should avoid placing derived routing quantities in the engine. Safe margin, imbalance, F-scores, and BR-0/BR-H decisions depend on fleet state and policy choices, so they belong outside the engine.

We should also avoid treating KV footprint as the full truth. KV footprint is the natural first proxy for decode load, but WiDeep-style MoE+EP serving can make actual step cost diverge from that proxy. Future observability should make that gap explainable.

## Derivation: Deferred Decisions

These are intentionally not decided here:

- A `LoadPlan` or connector API.
- Exact telemetry shape or stability promises.
- Where telemetry code should live.
- BR-0/BR-H implementation.
- Termination prediction, lookahead, or horizon tuning.

Those choices are outside this document because they depend on the future WiDeep/PegaFlow runtime boundary, not because they are forgotten.

## Derivation: Future Reuse

When PegaFlow starts supporting WiDeep with MoE+EP, use this note as the entry point for the first design pass:

1. Identify the decode workers and the sticky assignment boundary.
2. Define the raw engine progress needed for an external state layer to approximate `L_g(k)`, starting with a KV-footprint proxy while allowing MoE/EP load skew to be explained separately.
3. Keep derived routing state outside the engine.
4. Treat KV footprint as a proxy, then add MoE/EP detail only where it explains proxy error.
5. Evaluate routing algorithms only after the snapshot and admission boundaries exist.
