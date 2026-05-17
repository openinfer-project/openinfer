# Execution

> **TL;DR:** Current state and immediate next steps. No timeline — entries move through `In progress → Next → Open` (or get deleted) as work lands. Vision lives in `direction.md`; this file only tracks what's actually being worked on.

## Cross-model infrastructure

These are the shared layers — frontend, runtime, kernels, ledger/simulator/tracing. Each model crate consumes them.

### In progress

- **Model-owned kernel plans.** Qwen3 already carries a light `kernel_plan` mapping prefill/decode/unified phases → Rust wrappers, FFI symbols, and CUDA/Triton/cuBLAS backends. Extend the same shape to Qwen3.5 and DeepSeek V4 so each model crate is self-describing.
- **Frontend polish.** `vllm-frontend-rs` is the default OpenAI surface, talking to pegainfer via a local engine-core IPC bridge. Outstanding: logprobs / prompt-logprobs translation, usage accounting, and a deliberate decision on whether the served-model-id should decouple from the tokenizer path.

### Next

- **Cross-model kernel ledger.** Per-kernel record: dtype/layout/SM/shape support, measured latency/bandwidth (GPU + CUDA + commit pinned), correctness reference and tolerance, and DAG position in each consuming model. Machine-readable so LLMs and humans both navigate from `config.json` to kernel without guessing.
- **Simulator MVP.** Given a kernel DAG plus the ledger, predict TTFT/TPOT/throughput offline. Primary value is *explaining* a measured number, not predicting it exactly. Pick one model (Qwen3-4B) as the first target.
- **Request tracing.** Low-overhead spans from frontend → scheduler → forward steps. Bridges to the simulator: when measured ≠ predicted, tracing shows which span owns the gap. OTLP export so it plugs into existing collectors.

### Open

- Whether kernel ledger lives entirely per-model (denormalized) or also gets a cross-model index. Probably both, but the index format is not designed yet.

## Models

Each model crate owns its own scheduler, kernels, accuracy story, and benchmarks. The boundary with shared infrastructure is `pegainfer-core` (runtime contract) + `pegainfer-kernels` (op layer).

### DeepSeek V4

**Goal:** bring V4 to a serving-ready state on the GPUs we run on.

**Done:** initial native MP8 engine, TileLang build-time kernels, exact E2E, OpenAI-compatible HTTP path. Decode MoE moved onto GPU AG/RS with GPU-side route compaction and grouped TileLang FP4 local experts — no route/expert D2H on the hot path.

**In progress:**
- Decode-side EP — extend the existing MoE decode path to expert-parallel rank workers.

**Next:**
- Hopper performance. Current optimization work is on consumer Blackwell (sm_120); Hopper is the next target architecture.
- P/D and KV cache design — to be captured separately when scoped.

### Qwen3.5-4B

**Status:** at parity with vLLM 0.18 (TTFT 234ms vs 229ms, TPOT 11.77ms vs 11.67ms). Accuracy: 11/13 exact HF match, two small-logit-drift cases remaining.

**In progress:**
- Prefill full-paged migration. Decode is fully paged through the scheduler; prefill still scatters from contiguous HND staging into paged KV before attention. Mirrors Qwen3's step-2 migration but with HD256 kernels and partial RoPE (`rotary_dim=64` of `head_dim=256`) wrinkles.

**Next:**
- Close the last two HF parity cases (small-logit drift).
- Bring Qwen3.5 into the kernel-plan shape Qwen3 uses, so the ledger has both models covered.

**Open:**
- TP support for Qwen3.5 (no current driver — the hybrid linear/full state layout adds complexity Qwen3 didn't have).

### Qwen3-4B

**Goal:** stay ahead of vLLM on serving experience while expanding the parallel-strategy and scheduling surface.

**Done:** single-request and continuous batching ahead of vLLM (decode TPOT wins at all concurrencies; QPS=2 within 2% throughput while leading TTFT, TPOT, and latency stability). TP=2 brought up end-to-end on one machine; TP=8 smoke-tested on 8×4090. Issue #85 KV-pressure hang at QPS=2 is fixed.

**Next:**
- Explore pipeline parallelism (PP) as a complement to TP — particularly for larger model fits and multi-node layouts.
- Chunked prefill — closes the remaining ITL p99 tail under mixed prefill+decode load (pure decode already beats vLLM).
- Further perf tuning surfaces as they appear: batch decode per-row sampling redesign (recorded in memory), TP=8 systematic correctness pass, scheduler-level cleanup against the controller/worker model in `models/qwen3/tp-design.md`.
