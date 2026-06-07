# KV Admission Is a Full-Lifetime Reservation (Until Preemption Exists)

**TL;DR**: A scheduler without preemption must admit requests on their full-lifetime KV budget, not their prefill footprint — and must explicitly reject requests that can never fit, release KV on every exit path, and prove recovery with a post-pressure request. Learned from the Qwen3-4B issue #85 pressure hang (fixed in PR #131); the same admission rule was reused for Kimi-K2 paged KV (#239).

## The failure mode

Admitting on prefill-only capacity lets many active requests grow into new KV pages together until a decode step cannot allocate. The server then enters a half-alive state: `/v1/models` answers, completions hang forever. The observed symptom (issue #85: `vllm bench serve` QPS=2 over Qwen3-4B) looked like a deadlock but was an admission-accounting bug plus leaked request state.

## The rules

1. **Reserve the full lifetime at admission.** Active requests reserve the remaining pages they may need until `max_tokens`. A pending request is admitted only if its prompt plus maximum generated-token KV footprint fits *after* those reservations. This is deliberately conservative — it defers earlier than a preemption-capable scheduler would — but it makes decode-time allocation failure impossible without implementing preemption.
2. **Defer the temporarily-over-budget; reject the impossible.** A request that fits the model instance but not the current free pool stays in the waiting queue. A request larger than the instance's total usable KV capacity must be rejected explicitly (as a request *error*, not an empty success) — otherwise it waits forever and blocks the queue head.
3. **Count tokens actually written to KV, not tokens returned to the client.** In a prefill/decode split, the sampled token does not occupy KV until it is fed back as the next decode input, so a request returning `N` completion tokens occupies at most `prompt_len + N - 1` KV tokens. Review bots will confidently tell you the formula is `prompt_len + max_tokens`; check what the kernels write before "fixing" it.
4. **Every exit path must release request state.** KV pages are RAII-returned only when request state drops, so client disconnect, execution error, and send-failure paths all need to route through the owner `drop_request` — finishing normally is the only path that happens for free.
5. **Pressure-test evidence needs a post-pressure probe.** Because the failure mode keeps the health endpoints alive, "the benchmark completed" is not enough: the gate is pressure-client success *plus* a fresh completion returning afterwards.
