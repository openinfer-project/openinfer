# PegaInfer Per-Model Engine And Kernel Boundary

**Created**: 2026-05-03
**Status**: complete
**TL;DR**: pegainfer should evolve as reusable frontend/data-plane infrastructure plus per-model engines, not as one universal model abstraction. The first concrete step is extracting a kernels crate; kernels then become first-class assets through an index, ledger, simulator, and request tracing. PegaFlow remains the KV data plane instead of being folded into model internals.

## Preparation

- **Read**:
  - `docs/index.md` - showed an existing index entry for this boundary doc, but the file was missing locally.
  - `docs/playbooks/model-optimization-pipeline.md` - defined per-model optimization docs, DAG expansion, and profiling-driven optimization.
  - Local source in `src/model_executor.rs`, `src/scheduler_qwen35.rs`, `src/model/qwen35/`, the DSV3 worktree, `vllm_frontend.rs`, and `../pegaflow` - confirmed that model execution, KV/state layout, communication, and kernel needs diverge materially by model family.
- **Relevant history**:
  - Earlier shared-runtime work (consolidated in `docs/subsystems/runtime/runtime.md`) pursued shared model execution boundaries.
  - `docs/models/qwen3/tp-design.md` moved Qwen3 toward a controller/worker executor model.
  - The DSV3 branch demonstrates that FP8, MLA compressed KV, MoE, DeepEP, and EP scheduling form a different model engine shape from dense full attention.
- **Plan**:
  1. Record the architectural decision and its reasoning.
  2. Define which parts remain shared and which parts become per-model.
  3. Define the kernel ledger, simulator, and tracing direction.
  4. Clarify how PegaFlow integrates without becoming model-specific execution logic.
- **Risks / open questions**:
  - The main risk is over-splitting before shared contracts are stable. The boundary should start with a kernel crate and a lightweight routing index; machine-readable model manifests should wait until a model crate owns the DAG and can generate or validate them.

## Decision

PegaInfer should not become a single deep abstraction that forces dense full-attention models, hybrid linear-attention models, MLA/MoE models, multimodal models, and future RL/disaggregated variants through one execution model.

The project should instead use this shape:

```
vLLM Rust frontend
  -> scheduler/control plane
  -> per-model engine
  -> model-owned kernel DAG
  -> shared kernel/runtime/data-plane libraries
```

The reusable layer should be intentionally narrow:

- vLLM Rust frontend bridge: HTTP, chat/completions, tokenizer, OpenAI compatibility, request protocol.
- Runtime primitives: CUDA device/context utilities, cuBLAS/NCCL helpers, tensor wrappers, safetensors loading helpers, logging/tracing utilities.
- Data plane: PegaFlow for content-hash KV blocks, pinned memory, SSD/RDMA transfer, prefix cache, and offload.
- Kernel catalog and measurement tooling: operator metadata, correctness baselines, benchmark snapshots, and profiling records.

Each model engine should own:

- config parsing and architecture interpretation;
- weight loading and sharding policy;
- model state layout, including KV cache, recurrent state, MLA compressed KV, graph slots, expert state, or multimodal state;
- scheduler/executor shape when the model requires different batching, TP/PP/EP, or communication semantics;
- the ordered kernel DAG used for prefill, decode, unified steps, and any model-specific fast paths.

## Why

The local codebase already shows that model families diverge in ways that are not accidental.

Qwen3 is a dense full-attention model. Its current direction is a step-oriented `ModelExecutor` with rank workers, request IDs, and TP collectives. The model code can express prefill/decode/unified full-attention DAGs directly.

Qwen3.5 is a hybrid model with 24 linear-attention layers and 8 full-attention layers. It needs recurrent state, graph slots, and slot compaction in ways Qwen3 should not know about. The separate Qwen3.5 scheduler is not just duplication; it reflects real state-shape differences.

DSV3.2 is a separate execution world: FP8 block-scale GEMM, MLA compressed KV, FlashMLA, MoE routing, DeepEP, EP weight placement, and collective dispatch/combine all belong to that model engine. Forcing this into the Qwen3 abstraction would make the abstraction the main source of complexity.

LLM coding is cheap enough that maintaining clean model-local context can be more productive than compressing every model into a generic framework. The constraint should move from "share all code" to "share the parts whose contracts are genuinely stable."

## Kernel Boundary

Kernel performance and correctness dominate model quality. The first reusable code artifact should be a kernels crate, not a larger model trait. The first reusable knowledge artifact on top of that crate should be a compact kernel index and then a kernel ledger.

A kernel ledger should track:

| Record | Purpose |
| --- | --- |
| `KernelSpec` | Kernel name, backend, supported SMs, dtype/layout constraints, shape constraints, numerical tolerance. |
| `OpInstance` | A concrete model DAG node: model, layer, phase, shape, layout, batch/context parameters. |
| `PerfRecord` | Measured latency, bandwidth, FLOPs, occupancy or NCU counters, GPU, CUDA version, commit, build flags. |
| `CorrectnessRecord` | Reference source, input fixture, max/mean error, greedy-token impact, regression status. |

The initial human/LLM index should live beside the kernels crate so an engineer can jump from a model op to the Rust wrapper, FFI symbol, source file, backend, and shape/layout constraints. Machine-readable model DAG manifests should live with each model crate, not in the generic kernels crate, because only the model crate owns the ordering, layer selection, and phase-specific shape instances. The ledger only earns a richer Rust API once multiple model engines are reading it.

## Simulator

Given a model config and a selected kernel set, pegainfer should be able to build an offline performance estimate:

- prefill TTFT for fixed prompt lengths, e.g. 1k, 2k, 10k;
- decode TPOT for bs1 and high-batch cases;
- per-operator contribution to total time;
- likely bottleneck classification: compute, memory bandwidth, launch overhead, communication, or IO.

The simulator does not need to be perfect. Its first job is to make performance explainable:

- if estimated 1k prefill is 100ms and production sees 500ms, the request needs tracing;
- if TPOT is 20ms, the output should say which kernels account for that 20ms;
- if a load curve has a sweet spot, the simulator should expose where batching stops helping.

## Tracing And Online Profiling

Request tracing should share the same `OpInstance` identity as the simulator and kernel ledger.

The low-overhead path should record:

- queue wait;
- tokenize/render time from the vLLM Rust frontend;
- scheduler admission and step selection;
- prefill/decode/unified step timing;
- batch size, prompt/decode token counts, KV pages, and cache-hit status;
- PegaFlow save/load/prefetch events when used.

The debug path can sample CUDA events or CUPTI activity for a specific request. Whole-process `nsys` remains the deep offline tool, but online traces should answer "what happened to this request" without running a heavyweight profiler continuously.

The current vLLM frontend bridge already has protocol fields for `trace_headers`, `prefill_stats`, and logprob payloads. Filling those with pegainfer scheduler/runtime data is the natural integration point.

## PegaFlow Boundary

PegaFlow should be integrated as the KV data plane, not as model execution logic.

The model engine should expose KV block descriptors:

- namespace/model identity;
- block hash;
- layer or state namespace;
- TP/EP/PP slot identity;
- device pointer and block layout;
- segment layout for K/V split, recurrent state, or MLA compressed KV.

PegaFlow should own:

- content-addressed block storage;
- pinned-memory allocation;
- SSD tiering;
- RDMA fetch/transfer;
- prefix-hit query and prefetch;
- transfer observability.

This keeps Qwen3, Qwen3.5, and DSV3 free to use different state layouts while still sharing one storage and transfer layer.

## Near-Term Implications

1. Extract a kernels crate first, starting with the Qwen3-4B dense full-attention path and preserving Qwen3.5 compatibility symbols.
2. Put kernel source, build ownership, Rust wrappers, FFI, tensor/runtime helpers, and the Qwen3-4B human/LLM kernel index in that crate.
3. Treat `ModelForward` as a legacy/simple path, not the long-term universal model interface.
4. Keep `ModelExecutor` step-oriented for Qwen3, but avoid forcing Qwen3.5 and DSV3 into it until their state contracts are explicit.
5. Use `vllm_frontend.rs` as the northbound stability layer and improve its bridge payloads before rebuilding frontend behavior locally.
6. Integrate PegaFlow first through block descriptors and prefix/offload events, not by embedding its storage policies in model code.

## Debrief

- **Outcome**: Captured the decision that pegainfer's next architecture should be per-model engines backed by shared frontend, runtime, kernel measurement, tracing, and PegaFlow data-plane layers.
- **Pitfalls encountered**:
  - `docs/index.md` already referenced this file, but the file was absent locally. This doc fills that routing gap.
- **Lessons learned**:
  - The existing source is already moving away from a single universal model abstraction; the documentation should make that direction explicit so future refactors do not fight the codebase.
- **Follow-ups**:
  - Extract the Qwen3-4B model crate next. That crate should own config parsing, weight loading, state layout, and the prefill/decode/unified kernel DAG.
  - If a TOML/JSON kernel manifest is still useful, put it in the Qwen3-4B model crate and make it generated or validated from the Rust DAG. Do not hand-maintain model manifests in `pegainfer-kernels`.
  - Turn the kernel ledger into a concrete artifact format only after at least one model crate and tracing path need it.
  - Add request trace IDs and scheduler step spans through the vLLM frontend bridge.
  - Define the first PegaFlow KV block descriptor for Qwen3 paged KV.
