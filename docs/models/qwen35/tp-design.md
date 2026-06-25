# Qwen3.5 Tensor Parallelism Design

> **TL;DR:** Qwen3.5 tensor parallelism should reuse Qwen3's controller/worker TP runtime and stay degree-parametric. Validate `TP=2` first, fail closed on indivisible degrees, shard dense full-attention/MLP before tackling linear-attention GDR state.
>
> **Last touched:** 2026-06

## Goal

Add tensor-parallel support for `Qwen3.5-4B` by reusing the Qwen3 TP runtime instead of designing a second parallel execution stack.

The implementation should be degree-parametric where the model dimensions divide cleanly. `TP=2` is the first validation target, not an architectural limit. Unsupported or indivisible degrees must fail closed before model load.

## Qwen3 Runtime Reuse

Reuse the Qwen3 TP shape:

- controller/worker broadcast execution model
- `RequestId` request identity
- coarse-grained prefill/decode/unified/drop step protocol
- rank-local worker-owned model state
- rank-local CUDA context, cuBLAS, graph, and NCCL resources
- hidden all-reduce after row-parallel projections
- replicated embedding/lm_head as the first-pass simplification

Qwen3.5-specific design work should stay focused on model geometry and state ownership: hybrid layer layout, gated q projection, linear-attention conv state, and GDR recurrent state.

## Boundaries

This design does not cover multi-node TP, data parallelism, pipeline parallelism, vocab-parallel embedding/lm_head, or Qwen3.5 prefix-cache/recurrent-state snapshots.

Phase 1 does not shard linear attention or change GDR kernel shapes. Phase 2 does not change GDR math, does not all-reduce recurrent state, and does not move recurrent state ownership back into the scheduler.

## Why Dense First, GDR Second

Qwen3.5 has two separable TP problems.

The dense part is already proven by Qwen3: full-attention head sharding, local KV heads, MLP intermediate sharding, all-reduce after row-parallel projections, and worker-thread CUDA/NCCL execution.

The linear-attention part is Qwen3.5-specific: conv state and GDR recurrent state are long-lived request state, current GDR AOT kernels are built for the global value-head shape, and slot compaction / graph padding / `DropRequest` must all preserve rank-local recurrent state. If dense TP and GDR TP land together, failures are hard to attribute. Phase 1 narrows correctness debugging to runtime + dense sharding; Phase 2 then isolates the GDR/recurrent contract.

## Architecture Summary

Qwen3.5-4B:

- 32 layers: 24 linear attention + 8 full attention
- full-attention layers: `3, 7, 11, 15, 19, 23, 27, 31`
- `hidden_size = 2560`
- `intermediate_size = 9216`
- tied embedding/lm_head
- `vocab_size = 248320`

Full attention:

- `num_attention_heads = 16`
- `num_key_value_heads = 4`
- `head_dim = 256`
- `q_dim = num_attention_heads * head_dim = 4096`
- `kv_dim = num_key_value_heads * head_dim = 1024`
- q projection includes an output gate, so gated q projection output dim is `2 * q_dim = 8192`

Linear attention:

- `linear_num_key_heads = 16`
- `linear_key_head_dim = 128`
- `linear_num_value_heads = 32`
- `linear_value_head_dim = 128`
- `linear_q_dim = linear_num_key_heads * linear_key_head_dim = 2048`
- `linear_k_dim = linear_q_dim`
- `linear_v_dim = linear_num_value_heads * linear_value_head_dim = 4096`
- `linear_qkv_dim = linear_q_dim + linear_k_dim + linear_v_dim = 8192`
- `linear_z_dim = linear_v_dim = 4096`
- recurrent state per linear layer: `[linear_num_value_heads, linear_key_head_dim, linear_value_head_dim] f32`
- conv state per linear layer: `linear_qkv_dim * (conv_kernel_dim - 1)` bf16

## Partition Contract

For any candidate `tp`, require:

- `num_attention_heads % tp == 0`
- `num_key_value_heads % tp == 0`
- `intermediate_size % tp == 0`
- Phase 2 additionally requires `linear_num_key_heads % tp == 0` and `linear_num_value_heads % tp == 0`

Full attention local dimensions:

- `local_q_heads = num_attention_heads / tp`
- `local_kv_heads = num_key_value_heads / tp`
- `local_q_dim = local_q_heads * head_dim`
- `local_kv_dim = local_kv_heads * head_dim`
- `local_gated_q_dim = 2 * local_q_dim`

Qwen3.5 full-attention `q_proj` must be sharded by head-local q/gate pairs. Each rank owns a contiguous query-head range, and for each owned head it must receive both that head's q rows and that head's gate rows. Do not reuse a naive contiguous row shard if the physical layout can split q rows from their gate rows.

MLP local dimensions:

- `local_intermediate = intermediate_size / tp`
- local fused `gate_up_proj` rows: `2 * local_intermediate`
- local `down_proj` input cols: `local_intermediate`

Linear-attention local dimensions for Phase 2:

- `local_linear_key_heads = linear_num_key_heads / tp`
- `local_linear_value_heads = linear_num_value_heads / tp`
- `local_linear_q_dim = local_linear_key_heads * linear_key_head_dim`
- `local_linear_k_dim = local_linear_q_dim`
- `local_linear_v_dim = local_linear_value_heads * linear_value_head_dim`
- `local_linear_qkv_dim = local_linear_q_dim + local_linear_k_dim + local_linear_v_dim`
- `local_linear_z_dim = local_linear_v_dim`
- local recurrent state: `[local_linear_value_heads, linear_key_head_dim, linear_value_head_dim] f32`
- local conv state: `local_linear_qkv_dim * (conv_kernel_dim - 1)` bf16

## Phase 1: Dense TP, Replicated Linear Attention

Shard:

- full-attention `q_proj`, `k_proj`, `v_proj`, `o_proj`
- full-attention KV cache over local KV heads
- MLP `gate_proj`, `up_proj`, `down_proj`

Replicate:

- embedding and tied lm_head
- all linear-attention weights
- all linear-attention conv state
- all GDR recurrent state
- existing GDR kernels and scratch shapes

Execution:

- full-attention: local q/k/v + local attention + local `o_proj`, then all-reduce hidden
- MLP: local gate/up + local activation + local `down_proj`, then all-reduce hidden
- linear attention: every rank runs the full layer and updates a full local recurrent-state copy; do not all-reduce replicated linear-attention output

Validation scope:

- first validated degree: `TP=2`
- Qwen3.5 HF logits gate
- Qwen3.5 scheduler e2e
- basic TP2 serving smoke
- startup fails closed for unsupported or indivisible degrees

## Phase 2: Sharded Linear Attention / GDR

Phase 2 converts linear attention from replicated execution to true TP execution.

Shard:

- `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`
- `dt_bias`, `A_log`
- conv state
- GDR recurrent state
- linear-attention `out_proj`

Execution:

- each rank computes local q/k/v/z/b/a
- each rank updates only local conv state and local GDR recurrent state
- each rank runs local gated RMSNorm/output-gate work
- each rank runs local `out_proj`
- all-reduce happens after `out_proj`

Never all-reduce GDR recurrent state or conv state. Their ownership is rank-local and request-local.

### vLLM Reference

Use vLLM's `Qwen3NextForCausalLM` / `QwenGatedDeltaNetAttention` as the reference contract, not as code to copy mechanically:

- GDN state shape depends on `tp_size`
- q/k/v/z projections are tensor-parallel column projections
- `out_proj` is row-parallel and reduces back to full hidden
- `dt_bias` and `A_log` are sharded over local value heads
- b/a projections are local-value-head aware; some quantized paths may replicate small projections and slice locally
- GDR prefill/decode kernels consume local head/state shapes

OpenInfer-specific work remains: worker-owned rank-local recurrent state, `RequestId` lifecycle, local-state slot compaction, `DropRequest` cleanup, and fail-closed kernel-shape validation.

Validation scope:

- Phase 1 gates still pass
- long HF logits replay under the validated degree
- slot compaction replay
- recurrent-state cleanup on finish/drop
- no stale local recurrent state after slot reuse

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
