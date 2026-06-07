# MoE+TP Greedy-Parity And Reporting Lessons (Kimi-K2 bring-up)

> **TL;DR:** Three reusable lessons from Kimi-K2 TP8/EP8 bring-up that apply to any MoE + tensor-parallel decode engine where greedy token-id parity is the keep/revert gate.
>
> 1. **Reduce hidden states in F32, not BF16.** A BF16 bulk NCCL all-reduce changes per-row reduction rounding and silently breaks greedy decoding. Keep a `BF16 → F32 → all-reduce → BF16` bridge.
> 2. **Don't merge the shared-expert reduce into the routed-expert reduce.** Folding the shared-expert contribution into the routed F32 buffer to save one collective per layer corrupts cold-batch greedy output. The two reductions have different rounding boundaries; keep them separate until a stronger-than-token gate proves the merge.
> 3. **Report p50 and p99, never just mean.** On MoE+EP+TP decode the tail (rank-arrival skew, allocator churn, API drain) dominates the user-visible cost and is invisible in the mean. A keep/revert decision needs the full percentile set.
>
> These are observed-mechanism records, not vendor attributions. The numbers below are from 8×H20 Kimi-K2.5; the mechanism transfers to other models.

## Scope

Kimi-K2 is MLA + MoE with 60 routed-expert layers, served on 8 GPUs with tensor parallelism on the dense/attention/shared path and expert parallelism on the routed path. During bring-up the keep/revert gate was **exact greedy token-id parity** against a vLLM fixture. Several "obvious" collective optimizations passed a short output gate, then diverged on a longer or colder batch. The pattern is general: greedy decoding amplifies sub-ULP per-layer differences across dozens of layers until the argmax flips.

## Lesson 1: Hidden-state reductions must stay F32

### Mechanism

Each decode step reduces hidden states across TP ranks: embedding, every attention `o_proj`, the dense MLP down-proj, and every shared-expert down-proj — on Kimi this is ~123 logical all-reduces per token. The natural implementation is a single BF16 NCCL all-reduce per reduction.

A ring/tree all-reduce sums partial contributions in an implementation-defined order, and the order differs across contiguous row offsets and across ranks. In BF16 each intermediate sum is rounded to ~8 mantissa bits, so a different summation order produces a different rounded result. The difference is ~1 ULP per reduction, but it is **systematic, not noise**: it lands on the same rows every step, propagates through the residual stream, and across 61 layers grows until the greedy argmax picks a different token.

### Evidence

Switching the decode TP hidden reductions from a `BF16 → F32 → NCCL F32 → BF16` bridge to a BF16 bulk NCCL all-reduce (covering embedding, `o_proj`, dense/shared down-proj) built and ran, but on a fixed 4-concurrency fixture one row's greedy prefix changed from `[1008,2742,2531,...]` to `[1008,2742,924,6454,...]` at output position 3, and reproduced at output 64. There was no throughput win to justify the risk. Reverted to the F32 bulk bridge.

A separate first-diff pass confirmed the boundary: with the F32 bridge in place, `mla_projected_allreduce` showed an identical row diff on **all 8 ranks** (not a single rank), with no local `mla_projected` diff — the signature of a reduction-order rounding difference, not a per-rank state bug or a stream race.

### Takeaway

For greedy-parity-gated MoE+TP decode, do hidden reductions in F32. Pay the two casts. Do not "optimize" back to a BF16 bulk collective without a numeric gate stronger than a short token match — the regression is invisible until the batch is long or cold.

## Lesson 2: Keep shared-expert and routed-expert reductions separate

### Mechanism

In a MoE layer with both a shared expert and routed experts, the layer output is `shared(x) + scale * sum_k(routed_k(x)) + residual`. The shared path is dense BF16 (TP-reduced); the routed path is INT4-quantized expert GEMMs reduced in F32 across EP ranks. The tempting optimization: accumulate the shared-expert local BF16 output into the routed F32 buffer and do a single F32 all-reduce per layer, saving one BF16 collective (and, on the old path, one CPU barrier) per MoE layer.

This changes the rounding boundary. Originally the shared contribution is rounded to BF16, reduced, and added; the routed contribution is summed in F32 then scaled. Merging makes a `shared_bf16 + routed_f32 → single F32 reduce → BF16` chain whose intermediate rounding differs from the split version. Same class of sub-ULP systematic error as Lesson 1, but at the per-layer join instead of the TP reduce.

### Evidence

The merged version gave a small short-output win (`max_tokens=16` ~`81 → 85.7/87.1 tok/s`) but broke cold-batch greedy: a fresh 4-concurrency `max_tokens=22` batch diverged to the `[1008,2742,924,6454,...]` signature, `max_tokens=64` produced four inconsistent rows, **and it contaminated the state of later requests on the same server**. Reverted. The same trap recurred when fusing `allreduce_f32 → f32_to_bf16 → add` into a single `add_f32_bf16_to_bf16` kernel: the new kernel's `F32 contribution + BF16 residual → BF16` boundary is not equivalent to `(F32 contribution → BF16) then BF16 add`, and tokens diverged from position 3. A later kernel that explicitly rounds the contribution to BF16 *first*, then does the BF16 residual add (`kimi_scaled_add_f32_bf16_to_bf16`), was parity-safe and kept.

### Takeaway

Reducing collective count is a correct direction, but a per-layer reduce/add fusion is a numeric change, not a free launch-count win. Either keep the reductions separate, or write a fusion kernel that reproduces the exact original rounding order, and gate it on a long/cold greedy run — not a warm short one. The real lever for fewer collectives is a proper EP dispatch/combine (PPLX-style), not folding reductions together.

## Lesson 3: Report the full percentile set, lead with p99

### Mechanism

On MoE+EP+TP decode, per-step latency is set by the slowest rank crossing each collective barrier. Rank-arrival skew, allocator/workspace churn, and host-side stream drains produce a long right tail that the mean and p50 hide. Two changes can have the same mean and p50 while one has a 3× worse p99 — and p99 is what a user on a synchronized decode step actually feels.

### Evidence

8×H20 nsys decode trace, Kimi bs4:

- BF16 all-reduce: `p50 = 74.7us` but `p99 = 780us`, `max = 2.98ms` (p99/p50 ≈ 10×, max/p50 ≈ 40×).
- F32 all-reduce: `p50 = 64.8us`, `p99 = 385us`, `max = 886us`.
- Marlin WNA16 expert GEMM: `p50 = 14.3us`, `p99 = 154us`, `max = 187us`.
- `cuStreamSynchronize`: only 22 calls, `p50 = 28.3us`, but `p99 = max = 9.87ms` — a rare host drain that eats the end-to-end tail.

A kernel with a tiny p50 and a flying p99/max (like the Marlin expert GEMM here) must be analyzed by route/expert load and rank skew, not dismissed by its p50. A `magma_sgemm` count divided over output tokens was an earlier mis-attribution: the count was prefill shared-expert GEMM across requests, not steady decode — average-only accounting hid it.

### Takeaway

Every decode perf report must carry `count / total / mean / std / p50 / p95 / p99 / max` (and `p99-p50`, `max-p50`). A keep/revert decision on mean or p50 alone is not supportable on a barrier-synchronized MoE+EP decode step. When in doubt, lead with p99.
