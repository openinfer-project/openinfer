# GLM5.2 Serving Status & Remaining Work

> **TL;DR:** Decode serving is feature-complete for its scope (whole-step graph buckets, DSpark speculation, paged KV + prefix cache, VRAM-derived max_model_len, pegaflow host-tier offload behind `--kv-offload`); sampling surface frozen at `temperature/top_p/top_k/min_p/seed`. Low-latency arc: `--moe-topo tp8` (#609) + span MTP (#610) + attention-TP with replicated activations (`feat/glm52-attn-tp`, solo 13.75 ms / MTP code 221 tok/s — see `moe-tp8-low-latency.md`). Cross-tray EP-N on GB300 NVL72 shipped on `feat/glm52-rank-host`: `--rank-hosts` remote ranks over framed TCP, EP widths {4..64} instantiated, 2-tray EP8 solo p50 23.61 / p99 24.00 ms (see `cross-node-scaling.md`). GLM5.2's next P/D step is design input only; implementation remains Later under `roadmap-2026-h2.md`.
>
> **Last touched:** 2026-07

## Sampling surface (ruled 2026-07-06: frozen, sufficient)

Supported per-request, honor-or-reject, on both the plain path (#586) and the speculative path (#589): `temperature`, `top_p`, `top_k`, `min_p`, `seed`.

Deliberately **not** supported — audited, ruled out:

- `stop` strings / `stop_token_ids` / `min_tokens` / `logprobs` — rejected or ignored at scheduler admission.
- Penalty trio (`presence`/`frequency`/`repetition`) and `n > 1`.

Known limitation: HTTP `seed` is stripped to `None` by the shared vLLM frontend (`wire.rs`), a qwen3-era gap (#284). Engine-level seed works — the #589 determinism gates drive it through `EngineHandle` directly.

## Remaining model-line work

1. **pegaflow M2 design: cross-engine P/D** (vLLM prefill → openinfer decode) — input to the roadmap's single P/D + NIXL design issue, not an implementation PR yet. Target-only handoff needs hash compat via the #540 pattern plus the byte-dump drift gate. Preserving any model-based speculator with its own KV is additional scope: vLLM transfers target + draft KV, while OpenInfer's 99 arenas/rank cover target state only.
2. **#590 DSpark × prefix caching/P-D** — currently mutually exclusive. Compatibility path: after restoring sealed target blocks, locally prefill the 1–64-token uncached suffix and cold-start the drafter from only those aux-hidden captures; never expose absent draft pages as valid. Measure first-round and steady-state acceptance before paying for vLLM-style draft-KV transfer (80 KiB/token, making target+draft state about 2.52× target-only). See `vllm-speculative-pd-audit.md`.
3. **Perf backlog** — accept parity with the vLLM production reference is reached, so the first TPOT lever is round cost: #582 draft-round graph (external PR #591: −4.9% draft round re-measured on the reference host, Request-Changes for three capture bugs, waiting on the author), #559 bucket-4/8 step premium, adaptive span (5% of rounds accept all 7 drafts), #542 collective wait structure, #569 PDL weight prefetch, cache-aware placement (admission picks a rank before the prefix match — worst-case hit rate ÷8 under concurrency).

Done since the 2026-07-06 ruling: pegaflow M1 host-tier offload (#600), `scheduler.rs` split (#594), and coordinator phase decomposition (#596); #548 closed — the Python `vllm bench serve` c8 hang no longer reproduces on main (64/64, 216.9 tok/s, TPOT p50 30.8 / p99 34.2 ms).

## Shelved / background

- **#551 one-off silent request drop** — never reproduced (>3500-request soaks plus a 40-round instrumented soak on jz-38); kept open as a background watch, off the active queue.
- **#587 observability** (batch occupancy + `EngineHandle` `with_kv_capacity`/`load_watch`/`kv_events` wiring) — deferred pending discussion.
- **#541 indexer oracle reference drift** — HF `glm_moe_dsa` is a moving target; the gate stays excluded on main.
- **#584 empty completion echoes the prompt** — shared-frontend bug, pre-existing.
