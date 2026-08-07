# Frontend architecture: pegainfer-frontend and the engine boundary

**TL;DR:** One crate — `pegainfer-frontend` — now owns everything north of the model schedulers: the engine request/event contract (formerly `pegainfer-engine`), the vLLM protocol stack (formerly `pegainfer-vllm-frontend`, now the `vllm` module), and the `ModelLine` dispatch trait. The old crates and both `pegainfer-dynamo-*` workspaces are deleted. The trait is not consumed yet; **next step: onboard qwen3 as the first `ModelLine` implementation**, then evaluate a `dynamo` module as a second protocol stack.

Last touched: 2026-08

## The boundary, in one sentence

An engine is a function `launch(model_path, options) -> EngineHandle` whose semantics are: *give me a channel I can push `(GenerateRequest, KvPrefix)` into, and I guarantee every request's `TokenSink` receives a well-formed `Scheduled … terminal` event sequence.* Tokenizer, chat templates, HTTP protocol, metrics, LoRA routing all live north of the channel; KV, batching, CUDA all live south. The contract contains no CUDA types — `KvPrefix`'s anti-eviction hold is a `Box<dyn Any + Send>` for exactly this reason.

## Crate layout

```
pegainfer-frontend
├── engine.rs          # the contract: EngineHandle, GenerateRequest, TokenEvent,
│                      #   TokenSink, KvPrefix, KvCapacity, LoadSnapshot, KvBlockEvent
├── sampler.rs         # SamplingParams
├── parallel.rs        # ParallelConfig
├── tracing_state.rs   # global tracing on/off flag (frontend + schedulers both read it)
├── model_line.rs      # ModelLine trait + ModelLineRegistry (dispatch seam, unused yet)
└── vllm/              # protocol stack #1: vLLM EngineCore impersonation over ZMQ
    ├── mod.rs         #   serve_* entry points, vllm_server::Config assembly
    ├── bridge.rs      #   LocalEngineBridge: handshake, intake, burst demux, stats
    ├── wire.rs        #   EngineCoreSamplingParams <-> SamplingParams translation
    ├── lora.rs        #   /v1/{load,unload}_lora_adapter + adapter-name rewrite layer
    └── request_contract.rs  # GLM5.2 prefill-only route guard
```

Dependency direction: `pegainfer-frontend ← pegainfer-core ← model crates ← pegainfer-server` (thin bin). The frontend never depends on core or on model crates — that is what keeps the contract CUDA-free and lets the server binary hold the only model dispatch. Every model crate imports the contract as `pegainfer_frontend::engine` (the old `pegainfer_core::engine` re-export shims are gone).

Trade-off accepted knowingly: model crates now pull the vllm-server/axum/zeromq tree into their build graph. That is the price of "one frontend crate"; if model-crate compile times become painful, the escape hatch is splitting the contract back out, not feature-gating the stacks.

## How a request crosses the boundary

**Downstream (request in).** The protocol stack calls `EngineHandle::submit` / `submit_resolved`. The handle is a router plus metadata bag: it picks a scheduler partition (a resolved `KvPrefix` binds the request to the rank holding its blocks) and pushes into that partition's unbounded submit channel. **The model crate creates this channel in `launch`**, keeps the receiver in its scheduler thread, and hands the senders to `EngineHandle::new_with_join_handles`. Submission never blocks and never fails for capacity reasons — admission control is the scheduler's job, expressed as a `Rejected` event, not a submit error.

**Upstream (tokens out).** Direction of ownership flips: **the protocol stack creates the event channel** — one shared `TokenStreamReceiver` for all requests — and wraps a per-request `TokenSink` (tag + shared sender + abort flag) into each `GenerateRequest`. The stack demuxes by `RequestTag` (vllm: `dispatch_burst` folds each scheduler step into one ZMQ message).

**Cancellation** is not channel teardown: the stack flips the sink's `RequestAbortReason` (`AtomicU8`), the scheduler polls `is_cancelled()` and retires the request on its next step.

### Event-sequence contract (currently by convention, not by type)

Per request, the scheduler must emit:

1. `Scheduled` first — carries queued/scheduled timestamps and `cached_tokens`; the metrics path depends on it arriving before any token.
2. `PromptTokens` (echo only), then `Token`\*.
3. Exactly one terminal event: `Finished` | `Error` | `Rejected`. Nothing after it.

Both existing translators (vllm `wire.rs`; and dynamo `convert.rs`, now deleted — see below) fold streams assuming this order. It is enforced by hand at every `Finished` call site today; if a scheduler double-terminates, the failure shows up as protocol corruption in the frontend, not at the source. A `debug_assert` in `TokenSink` (terminal-then-anything panics) is the cheap hardening when it next bites.

## ModelLine: what a new model provides

`model_line.rs` defines the dispatch seam. A model crate implements:

- `name()` — family name for logs/`--help`.
- `probe(config_json)` — claim or reject the model directory; exactly one registered line must claim it.
- `augment_cli(cmd)` — the line's own CLI section. The registry diffs the command to learn which arg ids belong to the line, so consumed-args validation needs no separate table.
- `scheduler_partition_count(matches)` — partition count derivable from CLI alone, because the HTTP frontend registers one engine identity per partition *while the engine is still loading* (checked post-launch against `EngineHandle::scheduler_partition_count`).
- `launch(ctx, matches)` — spawn scheduler threads, attach handle metadata (`with_kv_capacity`, `with_load_watch(es)`, `with_kv_events`), return the handle.

The server binary holds a feature-gated `ModelLineRegistry` and does pure dispatch. This replaces today's four hand-edited sites in `pegainfer-server` (`ModelType` enum + `detect_model_type` + `load_engine` match + `consumed_args` table) and un-leaks model option types (`Glm52MoeTopo`, `Qwen3OffloadOptions`, …) from `config.rs`.

**Status: the trait is defined but nothing implements it yet.** `pegainfer-server` still uses the old match-based dispatch. Onboarding qwen3 is the pilot: implement `ModelLine` in `pegainfer-qwen3`, move its args out of `config.rs`, convert its `load_engine` arm, and let the remaining models follow the proven pattern.

## Protocol stacks

**`vllm` (current default, fleet-proven).** Impersonates a vLLM EngineCore process over in-process ZMQ/msgpack because upstream `vllm-server` assumes the engine is a separate process. HTTP routes, OpenAI types, tokenizer, chat templates, and Prometheus all live in the external `vllm-server`/`vllm-metrics`/`vllm-text` crates. The per-step msgpack round-trip and the impersonation handshake are pure overhead for our single-process deployment — tolerated because the stack is validated at EP16/EP32 scale.

**`dynamo` (planned second stack).** dynamo's `lib/llm` has an in-process path that removes the wire protocol entirely: `EngineConfig::InProcessTokens` + `run_input(drt, Input::Http, …)` gives axum → preprocessor (chat template + tokenize) → *your engine as a function call* (`AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, _>`) → detokenize → streaming tool-call/reasoning parsing → SSE. `DistributedConfig::process_local()` runs it with no etcd/NATS. The deleted `pegainfer-dynamo-backend/src/convert.rs` (see git history of this branch's parent) already contained the full `PreprocessedRequest → GenerateRequest` / `TokenEvent → LLMEngineOutput` translation — resurrect it as the adapter core. Known risks before committing: the `InProcessTokens` rename is recent (crates.io releases may still say `StaticCore`), the in-process path is not dynamo's flagship (maintained via tests + the python echo example), and the dep tree compiles fat (unconditional tonic/swagger/object_store; needs libzmq + protoc). Decision gate: prototype, run vllm-bench A/B against the vllm stack, let TTFT/step-overhead numbers pick the default.

## What was deleted, and the heirs

| Deleted | Heir |
| --- | --- |
| `pegainfer-engine`, `pegainfer-vllm-frontend` | merged into `pegainfer-frontend` (git mv, history preserved) |
| `pegainfer-core` re-export shims (`engine`/`sampler`/`parallel`), `pegainfer-server` shims (`scheduler`/`sampler`/`vllm_frontend`, incl. the `SchedulerHandle` alias) | direct `pegainfer_frontend::` imports everywhere |
| `pegainfer-dynamo-frontend`, `pegainfer-dynamo-backend` | future `dynamo` module in `pegainfer-frontend`; `convert.rs` translation logic recoverable from git |
| `bench_serving` bin (3.1k lines, in-process, bypassed the serving path) | HTTP-level benching: `scripts/bench_http_serving.py` + external vllm-bench; see [bench-regression](../../conventions/bench-regression.md) for the retired snapshot gate |
| `glm52_step_bench` bin, `scripts/run_snapshot_benchmark.sh`, `scripts/sweep_mb8.sh` | none — step-level microbenching lives in `pegainfer-glm52/benches/` (its `kernel_lab` docstring still mentions the old bin; harmless) |

## Next step

Onboard **qwen3** as the first `ModelLine` implementation (pilot for killing the `load_engine` match and the `consumed_args` table), then prototype the `dynamo` module and bench it against `vllm`.
