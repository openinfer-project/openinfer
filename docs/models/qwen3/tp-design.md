# Qwen3 Tensor Parallelism

> **TL;DR:** The TP runtime is implemented, not a plan. Qwen3-4B runs `TP=2` end-to-end (TP=8 smoke-tested on 8×4090) through a controller/worker broadcast model: every rank — including rank 0 — executes on a `RankWorker` thread under a coarse-grained `StepCommand` protocol, and the scheduler loop is plan → execute → resolve → apply (`scheduler/{plan,resolve,effects}.rs`). The real open remainder: zero automated TP correctness coverage, replicated (not vocab-parallel) embedding/lm_head, and no TP CUDA-graph path.
>
> **Last touched:** 2026-06

## Open items

These are the three real gaps, in priority order (sequenced in `roadmap.md`):

1. **TP correctness coverage.** Every test in the crate runs `device_ordinals: vec![0]`. A reduction-order or shard-offset bug is invisible to every gate. The step is running the existing HF golden gate over `device_ordinals [0,1]` (skip when <2 GPUs), then a systematic TP=8 pass — TP=8 today is only "loads, serves, non-degenerate text" on an 8×4090 host.
2. **Vocab-parallel embedding / lm_head.** Both are replicated per rank by first-pass design. Fine for 4B; becomes the memory bottleneck for larger dense models.
3. **TP CUDA-graph.** Decode graph capture exists only on the single-GPU path; TP decode runs eager. Deferred deliberately until the runtime shape stabilized — it has.

## Execution model (as implemented, `executor.rs`)

One controller decides each step; all ranks execute it under an ordered broadcast:

- `Qwen3Executor` owns a `primary: RankWorker` plus `workers: Vec<RankWorker>`. Rank 0 is not special-cased onto the scheduler thread — it executes on the primary worker thread under the same protocol.
- `StepCommand` is coarse-grained and step-oriented: `Prefill { requests, kv_views, echo }`, `Decode { requests, kv_views }`, `Unified { ... }`. KV mutation details stay inside a step; there are no low-level `EnsureCapacity`/`Advance`-style protocol messages.
- Requests are identified by `RequestId(u64)` from a monotone counter — never by slot indices or parallel-vector alignment.
- Barrier semantics: no worker starts command `N+1` until all workers finished `N`.
- Result flow is asymmetric: non-primary workers return ack/failure only; the primary worker returns the step artifacts.
- Sampling policy (params, RNG inputs) is controller-owned and travels with step items; GPU sampling and logprob extraction execute worker-side.
- Each worker owns its rank-local state: model shard, decode buffers, scratch. The executor owns the KV manager and per-request KV; the scheduler owns request lifecycle (streaming handles, finish bookkeeping, admission).

The rejected alternative — scheduler-owned rank-local mutable state with worker threads borrowing `&mut` into it via pointer wrappers — was the bring-up shape and was deliberately removed. TP is a replicated-local-state problem, not a shared-mutable-state problem.

### Scheduler boundaries (as implemented, `scheduler/`)

The scheduler loop is structured around three step-scoped boundary types:

| Boundary | File | Role |
| --- | --- | --- |
| `ExecutionPlan` | `scheduler/plan.rs` | what runs this step (kind + participating requests) |
| `ExecutionArtifacts` | `scheduler/plan.rs` | raw executor products, before lifecycle interpretation |
| `StepEffects` | `scheduler/effects.rs` | lifecycle transitions + token events, applied to scheduler state |

`resolve_step` (`scheduler/resolve.rs`) turns artifacts into effects; `apply_effects` mutates scheduler-owned state. The split isolates the three independent change vectors: batching/admission policy → scheduler, parallel execution strategy → executor, sampling/logprobs/finish semantics → resolver.

## Partitioning spec (reference)

Standard dense-model TP layout (vLLM/SGLang-style): attention partitioned by head, MLP by intermediate dim, one all-reduce after attention output projection and one after MLP down projection per layer. Embedding and tied lm_head replicated (open item 2).

Qwen3-4B at `TP=2` (`hidden=2560`, `q_heads=32`, `kv_heads=8`, `head_dim=128`, `intermediate=9728`):

| Tensor | Global | Local per rank |
| --- | --- | --- |
| fused `qkv_proj` | `[6144, 2560]` | `[3072, 2560]` (16 q heads + 4 kv heads, head-aligned slices) |
| `o_proj` | row-parallel | partial hidden, all-reduced |
| fused `gate_up_proj` | `[19456, 2560]` | `[9728, 2560]` (intermediate 4864) |
| `down_proj` | row-parallel | partial hidden, all-reduced |

Divisibility (`q_heads % tp == 0`, `kv_heads % tp == 0`, `intermediate % tp == 0`) is a hard requirement, not an accident.

## Bring-up hazards (fixed, kept for the next model-parallel bring-up)

- **cuBLAS handles/workspaces must be thread-local, and every TP worker thread must bind both the CUDA runtime device and the driver context** before touching cuBLAS/FlashInfer/NCCL. The original symptoms were an illegal memory access surfacing later in `paged_kv_scatter_cuda` and intermittent hangs after successful requests — runtime state management, not the scheduler boundary. (Generalized in `docs/lessons/exact-match-gate-thread-cublas.md`.)
- **Request-scoped worker-thread cuBLAS resources need explicit teardown**, or repeated TP requests accumulate unstable per-thread state.
- **Decode KV writes must use the same paged scatter path as prefill.** A decode-only specialized append path silently drifted from the generic scatter semantics, so decode-built KV state diverged from a fresh prefill of the same prefix. The fix was deleting the special path, not patching it.
