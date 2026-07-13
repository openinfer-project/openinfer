# Qwen3.5 TP Implementation Record

> **TL;DR:** Qwen3.5 TP Phase 1 is implemented as correctness-first eager dense TP: TP2 worker/scheduler execution, short/long HF logits gates, scheduler e2e, and real OpenAI-compatible HTTP serving smoke pass. The branch is rebased onto current `main` with the newer engine, sampling, config, and golden-fixture contracts; remaining TP work is tracked as follow-up, not a Phase 1 claim.
>
> **Last touched:** 2026-07

## Scope

This is the implementation record for Qwen3.5 tensor parallelism. The stable architecture contract lives in `docs/models/qwen35/tp-design.md`; this file records what actually landed, what was verified, and what should carry into later phases.

Keep Phase 1 and Phase 2 in this file until the Phase 2 implementation becomes large enough to split. The same state ownership risks continue across phases, so keeping the history together is useful: Phase 1 proves dense TP and worker-owned request state, while Phase 2 builds on that boundary for mixed prefill/decode and sharded recurrent state.

Out of scope for this file:

- local machine paths, NCCL symlink details, and temporary environment setup
- raw command transcripts unless they are part of retained evidence
- benchmark/performance claims

## Phase 1 Outcome

Phase 1 is complete as a correctness/runtime milestone.

Implemented:

- TP config validation for rank/world size, dense divisibility, and `TP > 1 && CUDA Graph` fail-closed startup.
- Dense TP weight loading for full-attention projections, full-attention KV heads, and MLP projections.
- Rank-local worker executor with worker-owned model shards, KV state, recurrent/conv state, CUDA context, cuBLAS, and NCCL comms.
- Eager TP prefill, chunked prefill, and eager decode.
- Scheduler TP backend that routes chunked prefill and eager decode through TP workers while keeping logical request/page accounting in the scheduler.
- Public multi-device Qwen3.5 engine path and server launch path for `tp_size > 1` with CUDA Graph disabled.
- Real HTTP TP2 serving smoke through the vLLM/OpenAI-compatible frontend.

Not implemented in Phase 1:

- TP CUDA Graph capture/replay.
- TP `RunUnifiedStep` mixed prefill+decode execution.
- Sharded linear-attention/GDR weights, kernels, conv state, or recurrent state.
- Vocab-parallel embedding or `lm_head`.
- Prefix-cache or recurrent-state snapshot support.
- Performance claims.

## Important Fixes

### Gated q projection layout

The major numeric blocker was the full-attention gated `q_proj` TP shard layout.

The wrong assumption was that `q_proj.weight` rows were physically arranged as:

```text
[all q rows][all gate rows]
```

The actual Qwen3.5 kernel contract is per-head interleaved:

```text
[head0 q][head0 gate][head1 q][head1 gate]...
```

For TP2, the fixed loader preserves contiguous head-interleaved ranges:

- rank 0 loads rows `0..4096`
- rank 1 loads rows `4096..8192`

The old loader gathered local q rows and local gate rows separately, then rebuilt a `[q][gate]` fused matrix. That corrupted the first full-attention contribution and failed the TP2 HF gate from prefill position `0`.

### Per-device Triton AOT handles

Real TP2 prefill exposed that Qwen3.5 GDR Triton AOT C stubs could not cache `CUmodule` / `CUfunction` in process-global state. With two CUDA devices, the rank that loaded a GDR kernel first could leave the other rank with an invalid function handle.

The generated stubs now cache module/function handles per CUDA device ordinal. This is an implementation constraint worth remembering for future multi-GPU users of generated Triton C stubs.

Follow-up review tightened this path: the generated stubs now fail closed before indexing the fixed per-device handle tables if `cuCtxGetDevice` returns an ordinal outside the table size. This preserves the Phase 1 static-table implementation while avoiding out-of-bounds writes on high CUDA ordinals.

### Worker-local NCCL setup

NCCL comms are initialized inside rank worker threads after each worker binds its CUDA context and initializes thread-local cuBLAS. Creating comms on the controller thread and moving them into workers led to invalid-handle symptoms and hangs.

This matches the design contract: TP workers own rank-local CUDA/NCCL execution resources.

### Current-main API compatibility

Rebasing Phase 1 onto current `main` required preserving the TP execution boundary while adopting newer shared contracts:

- Hybrid batch decode now builds `Vec<&mut RecurrentState>` from graph-owned slots before entering the common linear-attention helper. This keeps request state in place while satisfying the helper's mutable-reference slice contract.
- `openinfer_sample::select_batch` now requires request-local sampling steps. Phase 1 TP still samples one row at a time and has no request-local sampling counter, so it passes step `0` and retains its existing per-row `sample_seed` offset. Do not substitute batch row indices for request-local steps: that would make seeded output depend on batch composition.
- Qwen3.5 launch and tests use the current `EngineLoadOptions` surface; the removed `enable_prefill_profile` field is no longer supplied.
- TP scheduler tests explicitly set the newer `GenerateRequest::data_parallel_rank` field to `None` because Phase 1 is TP-only, not DP.
- Synthetic TP config/loader fixtures include `tie_word_embeddings`, matching the current `Config35` contract without changing production config loading.
- TP2 short/long HF gates use `Golden::load_for(model_path, long)` and pass the complete `Golden` to metadata validation, matching the model-selected fixture flow used by TP1.

These are compatibility changes, not extensions of Phase 1 scope. In particular, full seeded-sampling replay under TP should add a request-local completion counter rather than overloading batch position.

## Validation Evidence

Phase 1 acceptance coverage:

- TP2 short HF logits gate passes:
  - sequential eager: `108` positions, mean `0.0258`, p99 `0.0801`, max `0.1298`
  - batched eager: `72` positions, mean `0.0257`, p99 `0.0809`, max `0.1298`
- TP2 long HF logits gate passes:
  - prompts `4097` and `8192`, sequential eager: `18` positions, mean `0.0232`, p99 `0.0792`, max `0.1035`
- TP2 scheduler e2e passes and covers:
  - context-window rejection
  - greedy/logprobs paths
  - sequential requests
  - repeated request reuse
  - concurrent mixed greedy/sampling requests
  - consumer drop
  - post-drop scheduler health
- TP2 HTTP serving smoke passes through `openinfer_vllm_frontend::serve`:
  - `/v1/models`
  - non-streaming `/v1/completions`
  - streaming `/v1/completions`
  - concurrent completions
  - finite logprobs
  - chunked prefill forced with `max_prefill_tokens=1`
  - `TP2 + CUDA Graph` fail-closed startup
- TP1 regression gates pass after the TP2 additions:
  - TP1 short/long HF logits gates
  - TP1 scheduler e2e
- Current-main rebase verification passes:
  - formatting check
  - Qwen3.5 release compilation for all test targets
  - `openinfer-server` release compilation with only the `qwen35-4b` model feature

Known validation constraints:

- TP2 tests remain ignored by default because they require two CUDA devices, NCCL, and real Qwen3.5 weights.
- Long TP2 HF replay is GPU-memory-sensitive; choose a sufficiently free device pair.
- Qwen3.5 HF golden integration tests should run serially on memory-constrained hosts to avoid unrelated KV-capacity failures from concurrent model loads.

Stable test knobs:

- `OPENINFER_TEST_MODEL_PATH`: real Qwen3.5 weights path for HF, scheduler, and serving tests.
- `OPENINFER_TEST_TP_DEVICES`: comma-separated TP2 CUDA ordinals. Defaults to `0,1`; examples: `1,2`, `2,3`. TP2 tests require exactly two distinct ordinals.
- `OPENINFER_TEST_FRONTEND_MODEL_PATH`: optional tokenizer/config metadata path for HTTP serving tests. Defaults to `OPENINFER_TEST_MODEL_PATH` when unset.

## Follow-Up Work

The exact Phase 2 split is not decided yet. The items below are retained as follow-up work that should be scoped in the design branch before implementation.

### TP mixed-step unified execution

Implement `RunUnifiedStep` under TP while keeping Phase 1's replicated linear-attention/GDR state unless the design branch decides otherwise.

Goals:

- Support mixed prefill+decode scheduler steps under TP.
- Preserve deterministic collective ordering across ranks.
- Return mixed prefill/decode artifacts from the primary rank.
- Validate finish/drop/client-disconnect cleanup under mixed-step execution.
- Keep TP CUDA Graph disabled unless a separate graph design is completed.

Why this should be separated from GDR sharding:

- Mixed-step scheduling is an execution-protocol problem.
- Sharded linear-attention/GDR is a model-state-shape problem.
- Combining them would make failures hard to attribute.

### Sharded linear-attention/GDR state

Shard the Qwen3.5 linear-attention/GDR path after the mixed-step and state-lifecycle contract is clear.

Expected work:

- shard linear-attention projection weights
- shard conv state and GDR recurrent state by local value/key heads
- adapt or regenerate GDR kernels for local state shapes
- keep recurrent/conv state rank-local and request-local
- all-reduce only after local linear-attention `out_proj`

Non-negotiable invariant:

- Never all-reduce GDR recurrent state or conv state. These states are owned by rank-local request state.

## Follow-Ups

- Promote any stable contract changes discovered here back into `tp-design.md` through the design-doc branch.
- Decide whether Qwen3.5 server CLI should accept arbitrary TP device ordinals instead of only `0..tp_size`.
- Consider lifting the per-device Triton AOT handle lesson into a kernels or runtime subsystem doc if another model hits the same issue.
