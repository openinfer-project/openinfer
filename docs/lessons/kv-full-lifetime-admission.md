# Full-Lifetime KV Admission

> **TL;DR:** Without preemption, admitting a request means reserving its peak physical KV footprint across the block manager's full request lifecycle, not only its prompt or final committed token count. Temporarily over-budget requests wait; requests that can never fit are rejected explicitly; every terminal path releases request-owned state. Validate both pressure behavior and a post-pressure completion, because a server can remain reachable while generation is permanently wedged.
>
> **Last touched:** 2026-07

This lesson was extracted from the Qwen3 issue #85 KV-pressure hang, but the invariant applies to any paged-KV scheduler without preemption.

## The invariant

A scheduler must not admit more potential KV growth than the pool can satisfy:

```text
reserved_peak_footprint(active requests)
+ peak_kv_footprint(new request)
<= usable_pool_capacity
```

Prefill-only admission is unsafe. Several requests can all fit their prompts, enter decode, cross page boundaries together, and then fail to allocate. If failure cleanup is incomplete, pages remain pinned and later requests wait forever.

Full-lifetime reservation is conservative: a request may stop early and use less than its reservation. Until the scheduler supports preemption or another recoverable overcommit policy, that lost concurrency is the cost of guaranteeing progress.

## Measure the block-manager peak

Derive the budget from both the model's state transition and the block manager's allocation lifecycle. Do not assume every sampled token is immediately present in KV, or that the final committed-token count is the peak number of physical pages held.

For Qwen3, prefill writes the prompt and returns the first sampled output. The request commits at most `P + N - 1` KV positions for prompt length `P` and completion limit `N`, but kvbm's `schedule_decode` can provision the next decode block before the final input token is applied. A multi-token request can therefore hold `ceil((P + N) / block_size)` blocks at its peak; a one-token completion never schedules decode and only needs the prompt footprint. The scheduler's boundary tests compare this reservation against the real block-pool peak and cover cases where the older `P + N - 1` formula is short by one page.

This allocator-specific peak is not universal. Qwen3.5's current state machine and KV pool reserve from `P + N - 1`; other models may preallocate, append, or retire pages at different points. Encode the formula beside the owning scheduler and test it against the actual allocator lifecycle.

Round the resulting token count through the actual page geometry. Boundary tests should pin cases just below, exactly at, and just above a page transition; otherwise an off-by-one can hide behind page rounding.

## Three admission outcomes

1. **Admit:** the request's worst-case lifetime fits after active reservations.
2. **Defer:** it fits in an empty instance but not beside current work. Keep it waiting and retry after capacity is released.
3. **Reject:** its worst-case lifetime exceeds the instance's total usable capacity. Return an explicit request error so it cannot sit at the head of the queue forever.

Do not turn rejection into an empty successful response. The frontend must preserve the engine's error semantics and message.

## Cleanup is part of admission correctness

KV pages are usually returned through ownership/RAII only after all request state is dropped. Audit every terminal edge, not only the successful finish:

- normal length/EOS completion;
- client or receiver disconnect;
- prefill/decode/unified execution error;
- explicit rejection or cancellation;
- scheduler shutdown and worker failure.

A useful owner API is a single `drop_request(request_id)` path that removes executor state and releases the final page references. Error handling should report the terminal event and invoke the same owner cleanup.

## Verification pattern

Use layers of evidence:

- Unit-test admission with a fake executor and a small page pool. Cover impossible rejection, temporary deferral followed by admission, page boundaries, execution errors, and disconnect cleanup.
- Run a real serving workload that creates KV pressure. Assert every request completes or fails explicitly within a deadline; throughput is a separate claim.
- Immediately send a small post-pressure generation request. Health/model-list endpoints are insufficient because the original failure mode can leave the process alive while completions hang.
- Keep a deadline in concurrent tests. A deadlock without a deadline only wedges CI.

The decisive property is recovery: after pressure and failures, capacity becomes reusable and unrelated requests can still make progress.
