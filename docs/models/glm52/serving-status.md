# GLM5.2 Serving Status & Remaining Work

> **TL;DR:** Decode serving is feature-complete for its scope (whole-step graph buckets, DSpark speculation, paged KV + prefix cache, VRAM-derived max_model_len); sampling surface frozen at `temperature/top_p/top_k/min_p/seed`. Next up: split the 2.3k-line `scheduler.rs`.
>
> **Last touched:** 2026-07

## Sampling surface (ruled 2026-07-06: frozen, sufficient)

Supported per-request, honor-or-reject, on both the plain path (#586) and the speculative path (#589): `temperature`, `top_p`, `top_k`, `min_p`, `seed`.

Deliberately **not** supported — audited, ruled out:

- `stop` strings / `stop_token_ids` / `min_tokens` / `logprobs` — rejected or ignored at scheduler admission.
- Penalty trio (`presence`/`frequency`/`repetition`) and `n > 1`.

Known limitation: HTTP `seed` is stripped to `None` by the shared vLLM frontend (`wire.rs`), a qwen3-era gap (#284). Engine-level seed works — the #589 determinism gates drive it through `EngineHandle` directly.

## Remaining work (ordered)

1. **`scheduler.rs` split** — 2332 lines, past the 1k-line ceiling; debt flagged in the #588 review. Next up.
2. **#548 bench-client hang** — re-run the Python `vllm bench serve` client against current main; the Rust `vllm-bench` client already completes 8/8 at c8. If Python passes too, close.
3. **P/D disaggregation** — standing decision is prefill-by-vLLM, no dedicated prefill path. Prerequisite: integrate pegaflow first (in-process KV offload seam: numeric layer id + block hash); KV injection work starts after that.
4. **#590 DSpark × prefix caching** — currently mutually exclusive (draft aux hidden + draft KV are not cacheable alongside 656 B/token MLA KV). Plan: draft cold-starts at the cache-hit boundary; first probe the suffix-only-context accept loss with the shadow-slot infra before committing.
5. **Perf backlog** — accept parity with the vLLM production reference is reached, so the first TPOT lever is round cost: #582 draft-round graph (external PR #591 under review), #559 bucket-4/8 step premium, adaptive span (5% of rounds accept all 7 drafts), #542 collective wait structure, #569 PDL weight prefetch, cache-aware placement (admission picks a rank before the prefix match — worst-case hit rate ÷8 under concurrency).

## Shelved / background

- **#551 one-off silent request drop** — never reproduced (>3500-request soaks plus a 40-round instrumented soak on jz-38); kept open as a background watch, off the active queue.
- **#587 observability** (batch occupancy + `EngineHandle` `with_kv_capacity`/`load_watch`/`kv_events` wiring) — deferred pending discussion.
- **#541 indexer oracle reference drift** — HF `glm_moe_dsa` is a moving target; the gate stays excluded on main.
- **#584 empty completion echoes the prompt** — shared-frontend bug, pre-existing.
