# Qwen3.5 Tensor Parallelism Design

> **TL;DR:** Qwen3.5 TP2 should reuse Qwen3's controller/worker TP runtime. Phase 1 shards full-attention and MLP while keeping linear attention/GDR replicated, so the dense TP path can be validated first; Phase 2 then shards linear attention, conv state, GDR recurrent state, and the GDR kernels using vLLM's Qwen3Next/GDN TP contract as the main reference.
>
> **Last touched:** 2026-06

## Goal

Add single-node `TP=2` support for `Qwen3.5-4B`.

The first goal is correctness and architectural integration, not peak performance.

Concrete goals:

- keep `TP=1` healthy
- add a fail-closed `TP=2` path
- reuse Qwen3's existing TP runtime
- shard the dense-compatible Qwen3.5 pieces first
- defer true linear-attention/GDR TP until the dense TP path is correct
- gate the work with Qwen3.5 HF logits and scheduler tests

## Alignment With Qwen3 TP

Qwen3 already has the main tensor-parallel runtime skeleton. Qwen3.5 should not invent a second parallel runtime.

Reuse from Qwen3:

- controller/worker broadcast execution model
- `RequestId` as request identity
- coarse-grained step protocol
- rank-local worker-owned model state
- rank-local CUDA context, cuBLAS, and graph resources
- NCCL hidden all-reduce wrapper
- tensor-parallel config validation pattern
- full-attention head sharding pattern
- MLP intermediate sharding pattern
- first-pass replicated embedding/lm_head simplification

The Qwen3.5 design should focus on what differs from Qwen3:

- which Qwen3.5 components can directly reuse Qwen3 TP
- which components cannot because of hybrid/recurrent state
- why the work is split into phases
- what each phase must prove before the next phase starts

## Non-Goals

Global non-goals:

- do not redesign Qwen3's TP runtime
- do not merge Qwen3 and Qwen3.5 into a generic scheduler
- do not support multi-node TP
- do not support `TP > 2`
- do not add pipeline parallelism
- do not add data parallelism
- do not add vocab-parallel embedding/lm_head
- do not promise first-version performance parity with vLLM
- do not solve Qwen3.5 prefix-cache/recurrent-state snapshotting here

Phase 1 non-goals:

- do not shard linear attention
- do not shard GDR recurrent state
- do not change the GDR Triton AOT kernel shape
- do not change linear-attention conv state layout
- do not optimize away replicated linear-attention compute
- do not introduce a new GDR backend

Phase 2 non-goals:

- do not change GDR math
- do not all-reduce recurrent state
- do not move linear recurrent state ownership back into the scheduler
- do not require TP1 and TP2 bitwise identity; use the HF logits tolerance/regret gate

## Why Two Phases

Qwen3.5 TP has two different complexity classes.

The first class is dense TP that Qwen3 already solved:

- full-attention q/k/v/o sharding
- full-attention local-KV heads
- MLP gate/up/down sharding
- hidden all-reduce
- worker-thread CUDA/NCCL runtime
- request step broadcast

These are engineering integration tasks for Qwen3.5.

The second class is Qwen3.5-specific linear attention and GDR state:

- recurrent state is long-lived request state, not a temporary tensor
- conv state must follow request identity and decode-slot movement
- GDR state must not be all-reduced
- current GDR AOT kernels are built for global value-head shape
- slot compaction, `DropRequest`, and CUDA graph padding must all work with sharded recurrent state

Doing both classes in one change would make failures hard to localize. A bad output could come from the TP runtime, full-attention sharding, MLP sharding, GDR kernel shape, recurrent state movement, or slot compaction.

The two-phase plan narrows the debugging surface:

- Phase 1 proves Qwen3.5 can run correctly under the TP2 runtime with dense TP and replicated linear attention.
- Phase 2 starts from that stable TP2 base and only changes the linear-attention/GDR ownership contract.

## Qwen3.5 Architecture Summary

Qwen3.5-4B is a hybrid decoder:

- 32 layers
- 24 linear-attention layers
- 8 full-attention layers
- full-attention layer indices: `3, 7, 11, 15, 19, 23, 27, 31`
- `hidden_size = 2560`
- `intermediate_size = 9216`
- tied embedding/lm_head
- `vocab_size = 248320`

Full attention:

- `num_attention_heads = 16`
- `num_key_value_heads = 4`
- `head_dim = 256`
- `q_dim = 4096`
- `kv_dim = 1024`
- q projection includes an output gate, so q projection output dim is `8192`

Linear attention:

- `linear_num_key_heads = 16`
- `linear_key_head_dim = 128`
- `linear_num_value_heads = 32`
- `linear_value_head_dim = 128`
- `qkv_dim = 8192`
- `z_dim = 4096`
- recurrent state per linear layer: `[32, 128, 128] f32`
- conv state per linear layer: `8192 * (conv_kernel_dim - 1)` bf16

## Phase 1: Dense TP With Replicated Linear Attention

### Scope

Phase 1 reuses the Qwen3 TP runtime and shards only the Qwen3.5 pieces that match standard dense TP.

Shard:

- full-attention `q_proj`
- full-attention `k_proj`
- full-attention `v_proj`
- full-attention `o_proj`
- full-attention KV cache
- MLP `gate_proj`
- MLP `up_proj`
- MLP `down_proj`

Replicate:

- embedding
- lm_head / tied `embed_tokens`
- all linear-attention weights
- all linear-attention conv state
- all GDR recurrent state
- linear-attention GDR kernels and scratch shape

### Partition Contract

Full attention TP2:

- global q heads: `16`
- local q heads: `8`
- global KV heads: `4`
- local KV heads: `2`
- global q dim: `4096`
- local q dim: `2048`
- global KV dim: `1024`
- local KV dim: `512`
- global gated q projection dim: `8192`
- local gated q projection dim: `4096`

MLP TP2:

- global intermediate: `9216`
- local intermediate: `4608`
- local gate/up rows: `4608` each
- local fused gate_up rows: `9216`
- local down input cols: `4608`

### Execution Semantics

For full-attention layers:

1. each rank computes local q/k/v
2. each rank writes local KV heads
3. each rank runs local attention
4. each rank runs local `o_proj`
5. hidden partials are summed with all-reduce

For MLP:

1. each rank computes local gate/up
2. each rank computes local activation
3. each rank runs local `down_proj`
4. hidden partials are summed with all-reduce

For linear-attention layers:

1. every rank runs the full linear-attention layer
2. every rank updates a full local copy of recurrent state
3. every rank gets the same full hidden output
4. no linear-attention all-reduce is performed

Important invariant:

- replicated linear-attention output must not be accidentally summed across ranks

### Implementation Tasks

- Add a Qwen3.5 `TensorParallelConfig`.
- Validate TP2 divisibility for full-attention heads, KV heads, and MLP intermediate.
- Let Qwen3.5 engine accept `tp_size=2`.
- Reuse Qwen3's controller/worker execution model.
- Load rank-local Qwen3.5 models.
- Add full-attention row/column shard loading.
- Add MLP row/column shard loading.
- Allocate full-attention KV pools with local KV heads.
- Add all-reduce after full-attention `o_proj`.
- Add all-reduce after MLP `down_proj`.
- Keep linear-attention loader and forward path replicated.
- Preserve `TP=1` behavior.

### Acceptance Criteria

Functional:

- `TP=1` behavior is unchanged
- `TP=2` starts successfully on two local GPUs
- Qwen3.5 HF logits gate passes under TP2
- Qwen3.5 scheduler e2e passes under TP2
- a basic HTTP serving smoke test passes under TP2
- active request finish/drop does not leak worker-local state

Correctness:

- TP2 logits match HF golden within existing Qwen3.5 tolerance
- TP2 handles sequential replay
- TP2 handles graph-padded decode bucket replay
- TP2 handles slot compaction replay
- TP2 long-prompt replay passes when the long fixture is available

Operational:

- no deadlock on repeated requests
- no CUDA context mismatch
- no cuBLAS handle cross-device issue
- no NCCL hang on normal shutdown
- unsupported TP sizes fail closed

Out of scope for acceptance:

- TP2 being faster than TP1
- TP2 matching vLLM throughput
- memory optimality from sharding linear attention

## Phase 2: Linear Attention / GDR TP

### Scope

Phase 2 turns Qwen3.5 linear attention from replicated execution into true tensor-parallel execution.

Shard:

- `in_proj_qkv`
- `in_proj_z`
- `in_proj_b`
- `in_proj_a`
- `dt_bias`
- `A_log`
- conv state
- GDR recurrent state
- linear-attention `out_proj`

### vLLM Reference

vLLM's `Qwen3NextForCausalLM` and `QwenGatedDeltaNetAttention` are the primary external reference for Phase 2.

The relevant contract to mirror, not copy mechanically:

- GDN state shape depends on `tp_size`.
- q/k/v/z projections are tensor-parallel column projections.
- `out_proj` is row-parallel and reduces back to full hidden.
- `dt_bias` and `A_log` are sharded over local value heads.
- b/a projections are local-value-head aware; some quantized paths may replicate a small projection and slice locally.
- GDR prefill/decode kernels consume local head/state shapes.

OpenInfer should translate this into its Rust/CUDA execution model:

- worker-owned rank-local recurrent state
- explicit `RequestId` lifecycle
- local GDR state movement during slot compaction
- fail-closed kernel shape validation

The reference does not remove the need for OpenInfer-specific correctness gates. It only proves the partition contract is a known working direction.

### Partition Contract

TP2 local linear-attention dims:

- local key heads: `8`
- local value heads: `16`
- local q dim: `1024`
- local k dim: `1024`
- local v dim: `2048`
- local qkv dim: `4096`
- local z dim: `2048`
- local recurrent state: `[16, 128, 128] f32`
- local conv state: `4096 * (conv_kernel_dim - 1)` bf16

### Execution Semantics

For linear-attention layers:

1. each rank computes local q/k/v/z/b/a
2. each rank updates only local conv state
3. each rank updates only local GDR recurrent state
4. each rank runs local gated RMSNorm/output-gate path
5. each rank runs local `out_proj`
6. hidden partials are summed with all-reduce

Important invariants:

- GDR recurrent state is sharded, not replicated
- GDR recurrent state is never all-reduced
- conv state is sharded by local qkv channels
- all-reduce happens after `out_proj`, not before
- request identity is `RequestId`, not a long-lived batch slot

### Implementation Tasks

- Add local linear dim helpers to Qwen3.5 config.
- Add sharded loading for `in_proj_qkv`.
- Add sharded loading for `in_proj_z`.
- Add sharded loading for `in_proj_b` / `in_proj_a`.
- Add sharded loading for `dt_bias` / `A_log`.
- Decide and document `norm_weight` behavior.
- Allocate per-rank local recurrent state.
- Allocate per-rank local conv state.
- Update prefill GDR scratch sizes to local heads.
- Add or generalize GDR prefill kernels for local `num_value_heads`.
- Add or generalize GDR decode kernels for local `num_value_heads`.
- Update slot compaction to move local recurrent/conv state.
- Update `DropRequest` cleanup on all ranks.
- Insert all-reduce after linear-attention `out_proj`.

### Acceptance Criteria

Functional:

- Phase 1 TP2 gates still pass
- linear attention is no longer replicated in TP2
- per-rank recurrent state size is local, not global
- per-rank conv state size is local, not global
- GDR kernels run with local value heads

Correctness:

- HF logits gate passes under TP2
- long HF logits replay passes
- slot compaction replay passes
- prefill chunking and decode recurrence remain consistent
- TP2 greedy output passes existing regret/logprob tolerance
- TP1 remains unchanged

Operational:

- no recurrent state leak after request finish/drop
- no stale state after slot reuse
- no CUDA graph padding corruption
- no GDR scratch OOM regression
- no kernel shape mismatch hidden behind unsafe casts

Performance sanity:

- TP2 memory footprint should drop relative to Phase 1 because linear state and some weights are sharded
- TP2 should not catastrophically regress versus Phase 1 on basic decode
- exact throughput target is deferred until correctness is stable

## Validation Commands

Phase 1 minimum:

```bash
OPENINFER_CUDA_SM=120 \
OPENINFER_TEST_MODEL_PATH=/abs/path/to/Qwen3.5-4B \
cargo test --release --features qwen35-4b \
  -p openinfer-qwen35-4b --test hf_golden_gate -- --nocapture
```

```bash
OPENINFER_CUDA_SM=120 \
OPENINFER_TEST_MODEL_PATH=/abs/path/to/Qwen3.5-4B \
cargo test --release --features qwen35-4b \
  -p openinfer-qwen35-4b --test e2e_scheduler -- --nocapture
```

Add TP2 variants once the CLI/API exists.

## References

- `docs/models/qwen3/tp-design.md`
- `openinfer-qwen3-4b/src/config.rs`
- `openinfer-qwen3-4b/src/executor.rs`
- `openinfer-qwen35-4b/src/config.rs`
- `openinfer-qwen35-4b/src/weights.rs`
- `openinfer-qwen35-4b/src/recurrent_state.rs`
- `openinfer-qwen35-4b/src/batch_decode.rs`
- vLLM `Qwen3NextForCausalLM`
- vLLM `QwenGatedDeltaNetAttention`

