# GLM5.2 Serving Status & Remaining Work

> **TL;DR:** Decode serving is feature-complete for its scope (whole-step graph buckets, DSpark speculation, paged KV + prefix cache, VRAM-derived max_model_len, pegaflow host-tier offload behind `--kv-offload`); sampling surface frozen at `temperature/top_p/top_k/min_p/seed`. Low-latency arc: `--moe-topo tp8` (#609) + span MTP (#610) + attention-TP with replicated activations (`feat/glm52-attn-tp`, solo 13.75 ms / MTP code 221 tok/s — see `moe-tp8-low-latency.md`). Next up: pegaflow M2 cross-engine P/D.
>
> **Last touched:** 2026-07

## Sampling surface (ruled 2026-07-06: frozen, sufficient)

Supported per-request, honor-or-reject, on both the plain path (#586) and the speculative path (#589): `temperature`, `top_p`, `top_k`, `min_p`, `seed`.

Deliberately **not** supported — audited, ruled out:

- `stop` strings / `stop_token_ids` / `min_tokens` / `logprobs` — rejected or ignored at scheduler admission.
- Penalty trio (`presence`/`frequency`/`repetition`) and `n > 1`.

Known limitation: HTTP `seed` is stripped to `None` by the shared vLLM frontend (`wire.rs`), a qwen3-era gap (#284). Engine-level seed works — the #589 determinism gates drive it through `EngineHandle` directly.

## Remaining work (ordered)

1. **pegaflow M1: host-tier KV offload** — implemented behind `--kv-offload` (shared host pool, 8 rank instances, one namespace, hugepage option); jz-38 gates ALL PASS: evict-restore byte parity, warm TTFT 5371→157.6 ms (~34×), 16-way mixed restores zero-error. Design + measured details in `pegaflow-offload-pd.md`.
2. **pegaflow M2: cross-engine P/D** (vLLM prefill → openinfer decode) — blocked on M1; hash compat via the #540 pattern. Device-side KV layout is already verified byte-identical to vLLM's at source level (our kernels are ports; see `pegaflow-offload-pd.md`), so the remaining gap is block hashing plus a byte-dump drift gate.
3. **#590 DSpark × prefix caching** — currently mutually exclusive (draft aux hidden + draft KV are not cacheable alongside 656 B/token MLA KV). Plan: draft cold-starts at the cache-hit boundary; first probe the suffix-only-context accept loss with the shadow-slot infra before committing.
4. **Perf backlog** — accept parity with the vLLM production reference is reached, so the first TPOT lever is round cost: #582 draft-round graph (external PR #591: −4.9% draft round re-measured on jz-38, Request-Changes for three capture bugs, waiting on the author), #559 bucket-4/8 step premium, adaptive span (5% of rounds accept all 7 drafts), #542 collective wait structure, #569 PDL weight prefetch, cache-aware placement (admission picks a rank before the prefix match — worst-case hit rate ÷8 under concurrency).

Done since the 2026-07-06 ruling: `scheduler.rs` split (#594) and the coordinator phase decomposition (#596); #548 closed — the Python `vllm bench serve` c8 hang no longer reproduces on main (64/64, 216.9 tok/s, TPOT p50 30.8 / p99 34.2 ms).

## Shelved / background

- **#551 one-off silent request drop** — never reproduced (>3500-request soaks plus a 40-round instrumented soak on jz-38); kept open as a background watch, off the active queue.
- **#587 observability** (batch occupancy + `EngineHandle` `with_kv_capacity`/`load_watch`/`kv_events` wiring) — deferred pending discussion.
- **#541 indexer oracle reference drift** — HF `glm_moe_dsa` is a moving target; the gate stays excluded on main.
- **#584 empty completion echoes the prompt** — shared-frontend bug, pre-existing.
