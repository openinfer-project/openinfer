# GLM5.2 DP8 scheduler (PR5b)

> **TL;DR:** The bs=1 rejecting/greedy coordinator is replaced by a DP8 lock-step scheduler: every rank holds the full non-expert stack (replicated at load, ~19.6 GiB/rank on top of its 85 GiB expert slab) and serves **one request per rank**; every global step all 8 ranks run the full 78-layer forward simultaneously, each dispatching exactly one token (real or padding) into the per-MoE-layer DeepEP collectives (`GLM52_DECODE_GLOBAL_TOKENS` 1 → 8). DP1 is now just the `active_ranks = 1` special case of the same protocol. Per-request decisions live in `Glm52SlotState` as a pure state machine (unit-tested, no fakes); the coordinator is a thin channel shell. **All 8 jz-38 e2e gates green — single- and 8-way-concurrent outputs byte-identical to the PR5a record.** Bring-up shook three latent bugs out of the DeepGEMM MQA JIT (thread-unsafe globals, per-context kernel handles, per-launch codegen cost) — resolved by **AOT-instantiating both kernels at build time; the runtime JIT is retired**. Known cost: single-request latency is ~200 ms/step vs PR5a's 46–50 (the step cost is fixed — 8-way concurrency serves 8× tokens at the same wall) — the PR5c whole-step CUDA graph is the designed fix.
>
> **Last touched:** 2026-07

## Why this shape

- The DeepEP shim's capacity formula was always baked for all 8 ranks dispatching (`decode_worst_expanded_tokens`); the PR4/PR5a path used it at `g=1` with rank 0 as the only dispatcher. Writing the scheduler directly in the DP8 shape avoids a second scheduler rewrite when concurrency arrives — the kimi `DpCoordinator` precedent, minus its BlockPool/prefix-cache/batching machinery that GLM5.2 bring-up doesn't have yet.
- Non-expert replication costs ~19.6 GiB × 7 extra ranks at load time and ~105 GiB/rank resident (fits H200 141 GB, proven by the PP8-era loads which had the same per-GPU expert-layer count).

## Protocol

One global step = all 8 ranks each forward exactly one token through the full model:

- **Active rank, prompt not fully fed**: feed `prompt[fed]` at position `fed` (prefill rides decode); the model output is discarded except for the last prompt token's step, which yields the first generated token. Different ranks' prefill/decode advance concurrently in the same lock-step.
- **Active rank, decoding**: feed `last_token` at `prompt_len + completion − 1`.
- **Idle rank**: feed the padding input (token 0, position 0). Its KV/index-cache writes land in the idle rank's own dead cache slots and are overwritten when a request is admitted there; its MoE dispatch adds rows to expert segments but every chain kernel is row-independent and combine addresses slots per source token, so real tokens are unaffected.

`GLM52_DECODE_GLOBAL_TOKENS = GLM52_EP_RANKS` stays the single protocol definition; `bound_rows` becomes 2080 (vs 512 at `g=1`, capacity 10240), and the grouped-GEMM metadata kernel still device-traps if a real segment ends past it.

## What changed

| area | change |
|---|---|
| `weights.rs` | every rank's manifest includes the non-expert names (`loads_non_expert` field deleted — always true); coverage validation unchanged (set-union) |
| `model.rs` | `Glm52Rank0Model` → `Glm52RankModel`, `Glm52ExpertRankModel` deleted (every rank runs `decode_step`) |
| `runner.rs` | `Rank0Step`/`ExpertStep` → one `Step { token, position }` command; every rank builds the full model |
| `scheduler.rs` (new) | `Glm52SlotState` pure state machine (`next_input`/`advance`) + `run_dp8_coordinator` lock-step shell; fast-reject at intake, FIFO queue, one request per free rank |
| `lib.rs` / server | launch contract `--dp-size` 1 → 8 (or omitted); coordinator swap |

Semantics preserved from the bs=1 coordinator: EOS suppressed but counted, EOS outranks the length cap, greedy-only/no-logprobs/no-echo/no-LoRA rejections, `prompt + max_tokens − 1 ≤ max_model_len` cap (decided at launch: VRAM-derived per rank, `--max-model-len` overrides), fatal teardown on any step error (a failed step permanently desyncs the EP8 collective group — see `fail_step`). New behavior: a client disconnect frees the rank (send failure during decode, `TokenSink::is_closed` probe during prefill — the bs=1 coordinator decoded to completion into a dead channel). On a failed step the coordinator joins ALL ranks and logs every rank's error before tearing down — the first rank to answer often reports the ~100 s DeepEP device-timeout trap, not the root cause (toxic-review finding).

## Verification

- `cargo test -p openinfer-glm52 --features glm52 --lib` — 9 pass (6 new scheduler state-machine tests; local run needs `LD_LIBRARY_PATH=/data/opt/nccl-2.30.4/lib` and `OPENINFER_NCCL_ROOT=/data/opt/nccl-2.30.4`).
- clippy clean (no new lints) with and without the `glm52` feature; dead-code warning count identical to main (15, all pre-existing EP1-path).
- toxic-review pass done: no fatal findings; the root-cause-masking hard finding (sequential recv dropped the failing rank's error) and the prefill-disconnect zombie are fixed; padding-row bit-parity flagged as a stronger claim than PR3 row-isolation → proven by the gates below.
- **jz-38 8×H200 e2e gates (all green; script `glm52_pr5b_gates.sh` on the node, logs `glm52_pr5b_gates*.log`):**

| gate | result |
|---|---|
| 1/1b. byte-parity vs PR5a record, 24-tok + 128-tok outputs, ×2 determinism | PASS — byte-identical (padding rows never change real tokens' bits) |
| 2. 8 identical prompts concurrent | PASS — all 8 byte-identical to the solo reference; **5.3 s wall for 8 requests ≈ 189 ms/step, same as a single request** |
| 3. ms/step ×5 (133 steps) | 197–220 ms/step (see the perf note below) |
| 4. slot reuse after concurrency | PASS |
| 5. disconnect mid-decode (streamed, killed at 2 s) → 8-way afterwards | PASS |
| 6. mixed lengths concurrent vs solo runs | PASS — byte-identical each |
| 7. invalid requests (non-greedy / over-cap) mixed into live traffic | engine isolation PASS; the HTTP surface returns a generic 500 instead of the rejection message — pre-existing frontend gap (engine-core protocol has no request-error channel, #294 / upstream vllm#45286), same behavior as the bs=1 coordinator |
| 8. SIGTERM mid-decode | PASS — exit in 3 s, no 100 s DeepEP hang |

## The DeepGEMM JIT bring-up saga (3 gate runs)

DP8's 8 concurrent rank threads were the first to run the indexer's DeepGEMM path multi-threaded, and shook out three latent bugs:

1. **Run 1**: the JIT's globals are thread-unsafe end to end — include-parser visited set, compiler cache map, a shared static launch-attrs array. Concurrent first-step builds corrupt them ("Circular include may occur" → `CUDA_ERROR_LAUNCH_FAILED` on ranks 0–2 at layer 0). Fenced with a process-wide mutex over generate→build→launch.
2. **Run 2**: the default driver-API branch caches a **per-context** `CUfunction` (`cuKernelGetFunction` resolves in the building thread's context) — the other 7 ranks get `CUDA_ERROR_INVALID_HANDLE`.
3. **Run 3**: correct but slow — per-launch codegen + code hashing inside the mutex cost ~130 ms/step (8 ranks × 42 serialized JIT calls): 174 ms/step.

**Resolution: AOT-instantiate both kernels at build time** (all codegen parameters are compile-time constants for the decode indexer) and launch via `cudaLaunchKernelExC` — the runtime JIT is retired, along with the `OPENINFER_DEEPGEMM_ROOT`/`CUDA_HOME` runtime env requirements. The TU compiles for sm_90a only (wgmma); other targets get NOT_SUPPORTED stubs. Byte-parity vs the JIT-built PR5a record proves the AOT kernels numerically identical.

## Perf note: fixed step cost, PR5c is the fix

Single-request ms/step regressed 46–50 → ~200 (AOT run; the JIT-mutex run was 174 — the JIT was not the dominant term). GATE2 shows the step cost is **fixed**: 8 concurrent requests decode at the same per-step wall (189 ms), so full occupancy serves 8× tokens for free (aggregate ~36 tok/s vs PR5a's serial ~17.6). Prime suspect: the host launch path — 8 rank threads × ~4155 kernels/step submitted through one CUDA driver process, plus per-MoE-layer collectives waiting for the slowest rank (straggler jitter × 75 layers). Not yet nsys-isolated; the PR5c whole-step CUDA graph (one graph launch per rank per step) removes both terms by construction and should be measured against this 200 ms baseline. `bound_rows` 512→2080 contributes only a minor share (the PR5a-bounded kernels scale ~4× from a small base).

## Next

- jz-38 e2e gate above, then PR.
- PR5c: whole-step decode CUDA-graph capture on top of this stable per-rank shape (kills the ~46% launch-gap wall + residual MLA/indexer `alloc_zeros`); coordinate with PR #533's zero-alloc MLA scratch after its rebase.
