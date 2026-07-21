# Qwen3 Tensor Parallelism

> **TL;DR:** Qwen3 tensor parallelism is implemented as one rank-local model lane per GPU, coordinated by a controller that broadcasts step commands. Attention and MLP projections are sharded and their row-parallel outputs are combined with NCCL all-reduce. TP decode CUDA Graphs are pre-captured for every reachable bucket at startup. The path has a two-GPU concurrent-completion/deadlock test, but it still lacks a TP=1 versus TP=2 logits correctness gate; embedding and `lm_head` remain replicated.
>
> **Last touched:** 2026-07

## Runtime shape

`Qwen3LaunchOptions::tp_size > 1` maps ranks to CUDA devices `0..tp_size`. During executor construction:

1. Each rank loads a `Qwen3Model` with its own `TensorParallelConfig { rank, world_size }` and CUDA device context.
2. Each rank is profiled independently; the shared logical KV budget uses the minimum available block count across ranks.
3. Each rank gets a rank-local KV buffer and NCCL communicator on the same compute stream used by decode.
4. A `RankWorker` thread owns each `LocalQwen3Lane`, including rank 0. Thread startup binds the CUDA context and thread-local GPU resources before it accepts work.
5. The controller keeps logical request/KV ownership and sends the same `StepCommand` to every rank through `WorkerCommand::RunStep`.
6. All ranks execute the step. The primary rank returns sampling/logit results; peer ranks acknowledge completion without duplicating the user-visible result.

`StepCommand` represents the execution unit, not an individual kernel. Its current variants cover prefill, decode, unified prefill+decode, Green Context split-concurrent execution, and speculative verify/draft. This coarse broadcast boundary keeps ranks in the same collective order.

## Partitioning

`TensorParallelConfig::validate_for` rejects zero world size, out-of-range ranks, and model dimensions that cannot be divided by the requested degree. The Qwen3 dense blocks use the standard Megatron-style split:

| Component | Partition | Communication |
| --- | --- | --- |
| Q/K/V projections | Column parallel over local attention/KV heads | None after projection |
| Attention output projection | Row parallel | Hidden-state all-reduce |
| MLP gate/up projections | Column parallel over intermediate features | None after projection |
| MLP down projection | Row parallel | Hidden-state all-reduce |
| KV cache | Rank-local head shards | None |
| Token embedding and `lm_head` | Replicated | None |

The all-reduces occur after the attention output projection and MLP down projection in prefill, decode, and unified execution. With `TP=1`, the communicator is absent and the same helper is a no-op.

## Controller and worker contract

The controller owns scheduling decisions, request identity, admission, and result application. A lane owns its model weights, device buffers, graph cache, communicator, and CUDA state. Cross-rank mutable GPU state is not shared.

For every collective-bearing step, the controller must fan the command out to all ranks before waiting for responses. A rank error or command-order divergence is fail-loud because NCCL collectives have no useful recovery path once peers execute different sequences.

Model lifecycle commands such as LoRA load/unload are also broadcast so rank-local state stays aligned. Destruction shuts down and joins every worker.

## CUDA Graph startup

TP decode does not capture graphs lazily during serving. Executor construction pre-captures every reachable `(batch bucket, attention path)` graph in lock-step using four phases:

1. `Warmup`: run eager all-reduces for every bucket message size so NCCL establishes its size-selected algorithms before capture.
2. `Capture`: every rank records, instantiates, and uploads one bucket graph.
3. `Launch`: only after all ranks finish capture, every rank launches that bucket so collective peers are present.
4. `Finalize`: verify all expected graphs exist and mark each lane ready for replay.

Capture and first launch are separate barriers. Launching a captured collective while another rank is still inside capture/instantiate/upload can deadlock on process-wide CUDA driver locks. A startup watchdog aborts if the sweep wedges instead of leaving a half-ready server.

Decode steps then replay the captured graph. Mixed prefill+decode unified steps remain eager. Any change to batch buckets, attention paths, NCCL message sizes, or graph ownership must update the pre-capture sweep as one invariant.

## Supported boundary

The implemented path supports dense Qwen3 tensor parallelism on one host. Current launch policy deliberately rejects combinations whose state or ordering contract has not been implemented, including DFlash speculative decoding, KV offload, decode overlap, and batch-invariant mode with TP.

Embedding and `lm_head` replication is a correctness-first tradeoff. It avoids vocab-parallel sampling/communication but keeps a full copy of vocab-facing weights on every rank.

## Coverage and remaining work

`openinfer-qwen3/tests/tp_concurrent_decode.rs` launches TP=2 with CUDA Graph enabled, optionally validates the exported graph, submits 16 concurrent requests, and enforces a deadline so a collective deadlock fails the test. It self-skips without a model checkpoint or two GPUs.

That test proves startup/lifecycle/concurrent completion, not numerical equivalence. The remaining qualification work is:

1. Run the HF logits golden gate, or an equivalent teacher-forced oracle, over TP=2 and compare against the TP=1 tolerance envelope. This is the highest-priority gap because shard-offset or reduction-order bugs can still produce plausible text.
2. Decide when model size justifies vocab-parallel embedding and `lm_head`; doing so also requires defining sampling and output ownership.
3. Continue reducing TP-only runtime branches and keep CUDA context/cuBLAS/NCCL setup and teardown explicit as worker abstractions evolve.

Historical TP=8 smoke results show the sharding shape is not hard-coded to two ranks, but TP=8 is not a systematically qualified support claim.
