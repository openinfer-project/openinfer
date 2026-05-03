<p align="center">
  <img src="logo.png" width="200" alt="pegainfer logo">
</p>

<h1 align="center">pegainfer</h1>

<p align="center">
  Pure Rust + CUDA LLM inference engine. No PyTorch. No frameworks. Just metal.
</p>

<p align="center">
  <a href="#performance">Performance</a> &middot;
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#supported-models">Models</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="#architecture">Architecture</a>
</p>

---

pegainfer is a from-scratch LLM inference engine written in **~9.6K lines of Rust**, **~2.6K lines of CUDA**, and **~1.4K lines of Triton GPU kernels**. No PyTorch, no ONNX, no frameworks — just Rust + raw CUDA/Triton.

The goal is to understand every layer of the inference stack by building it from the ground up, and to explore what a Rust-native inference engine can look like.

## Performance

Measured on **RTX 5070 Ti** (16 GB), BF16, CUDA Graph enabled, single request:

| Metric | Qwen3-4B | Qwen3.5-4B |
|--------|----------|-------------|
| TTFT (short prompt) | ~14 ms | ~22 ms |
| TPOT (decode) | ~11 ms/tok | ~11.8 ms/tok |
| Throughput | **~91 tok/s** | **~85 tok/s** |

<details>
<summary>What do these metrics mean?</summary>

- **TTFT** (Time To First Token): latency from receiving the prompt to generating the first output token. Includes tokenization, embedding, and the full prefill pass.
- **TPOT** (Time Per Output Token): average time to generate each subsequent token during the decode phase.
- **Throughput**: 1000 / TPOT, i.e. tokens generated per second during decode.

</details>

## Quickstart

### Prerequisites

- Rust (2024 edition), CUDA Toolkit (nvcc, cuBLAS), CUDA-capable GPU
- Python 3 + Triton (build-time only — no Python at runtime)

### Build & Run

```bash
# One-time Python setup (for Triton AOT kernel compilation)
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Download a model
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

# Build & start server on port 8000
export CUDA_HOME=/usr/local/cuda
export PEGAINFER_TRITON_PYTHON=.venv/bin/python
cargo run --release
```

```bash
# Try it
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

# Streaming
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku about Rust:", "max_tokens": 64, "stream": true}'
```

> Always use `--release`. Debug builds are extremely slow for GPU/CUDA code.

<details>
<summary>More options</summary>

```bash
# Different model
cargo run --release -- --model-path models/Qwen3.5-4B

# Disable CUDA Graph (useful for debugging)
cargo run --release -- --cuda-graph=false
```

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `CUDA_HOME` | CUDA Toolkit path (default: `/usr/local/cuda`) |
| `PEGAINFER_TRITON_PYTHON` | Python with Triton for build-time AOT compilation |
| `PEGAINFER_CUDA_SM` | GPU SM target override when `nvidia-smi` unavailable (e.g. `120`) |

</details>

<details>
<summary>Windows</summary>

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
uv venv .venv --python 3.12
uv pip install "triton-windows<3.7"
$env:PEGAINFER_TRITON_PYTHON = ".venv\Scripts\python.exe"

cargo build --release
cargo run --release --bin pegainfer -- --model-path models/Qwen3-4B
```

</details>

## Supported Models

| Model | Architecture | Params | Status |
|-------|-------------|--------|--------|
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Full attention (GQA) | 4B | Greedy + sampling |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Full attention (GQA) | 8B | Greedy + sampling |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Hybrid (24 linear + 8 full attention) | 4B | Greedy + sampling |

Model type is auto-detected from `config.json` — just point `--model-path` at any supported model directory.

## API

OpenAI-compatible `/v1/completions` endpoint.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | (required) | Input text |
| `max_tokens` | int | 128 | Maximum tokens to generate |
| `temperature` | float | 0.0 | Sampling temperature (0 = greedy) |
| `top_k` | int | 50 | Top-k sampling |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `stream` | bool | false | Enable SSE streaming |

## Architecture

```
HTTP / vLLM frontend → EngineHandle → per-model engine crate
                                  │
                    ┌─────────────┴─────────────┐
        pegainfer-qwen3-4b                 root Qwen3.5 engine
          (full attention)              (24 linear + 8 full attn)
                    │                             │
                    └────────────┬────────────────┘
                                 │
                 pegainfer-core runtime + pegainfer-kernels
                                 │
                       CUDA / cuBLAS / Triton / FlashInfer
```

**Key design decisions:**

- **All computation on GPU** — no CPU fallback, no hybrid execution
- **Custom GPU kernels** — CUDA for decode-critical paths (GEMV, fused MLP, GDR recurrence), Triton AOT for Qwen3.5 compatibility kernels, FlashInfer for paged attention/sampling, and cuBLAS for matrix multiplication
- **Fused operators** — attention and MLP are each a single kernel launch
- **BF16 storage, FP32 accumulation** — numerical stability without memory overhead
- **CUDA Graph** on decode path — eliminates kernel launch overhead
- **Per-model crate boundary** — Qwen3-4B owns its config, weights, scheduler/executor, tests, benches, and kernel plan in `crates/pegainfer-qwen3-4b`

**Model details:**

- **Qwen3**: 32 Q heads, 8 KV heads (GQA 4:1), head_dim=128
- **Qwen3.5**: hybrid — 24 linear attention layers (Gated Delta Rule) + 8 full attention layers, head_dim=256

### What's not (yet) implemented

- Quantization (INT8/INT4)
- Qwen3.5 extraction into a standalone model crate

## Development

### Tests

```bash
# Unit tests
cargo test --release

# E2E greedy regression (needs GPU + model weights)
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test e2e
cargo test --release --test e2e_qwen35
```

### Triton AOT

Triton compiles the Qwen3.5 compatibility AOT kernels at build time. Qwen3-4B dense full-attention kernels are CUDA/cuBLAS/FlashInfer C++ wrappers. Runtime has no Python dependency — Triton is build-time only.

See `crates/pegainfer-kernels/tools/triton/README.md` for setup and troubleshooting.

### Source Layout

<details>
<summary>Expand</summary>

```
src/
├── main.rs                # CLI + vLLM/OpenAI server startup
├── vllm_frontend.rs       # vLLM engine-core bridge into a generic EngineHandle
├── server_engine.rs       # Model detection and compatibility re-exports
├── scheduler.rs           # Compatibility re-export of core engine request/event types
├── scheduler_qwen35.rs    # Qwen3.5 scheduler
├── model.rs               # Root-local Qwen3.5 model
├── model/
│   └── qwen35/            # Qwen3.5: config, weights, forward, prefill, decode, recurrent_state
├── ops.rs                 # Root compatibility dispatch + Qwen3.5 recurrent wrapper
├── ops/
│   ├── recurrent.rs       # Conv1d, Gated Delta Rule (Qwen3.5)
│   └── tests.rs           # Operator tests
├── tensor.rs              # Re-export of pegainfer-kernels tensor types
├── ffi.rs                 # Re-export of pegainfer-kernels FFI bindings
├── weight_loader.rs       # Safetensors loading + RoPE precomputation
├── sampler.rs             # Temperature, top-k, top-p sampling
└── trace_reporter.rs      # Archived fastrace JSON reporter, not wired into CLI

crates/pegainfer-core/             # Shared runtime API for model crates
├── src/engine.rs                  # EngineHandle, GenerateRequest, TokenEvent
├── src/kv_pool.rs                 # Paged KV pool and request state
├── src/ops.rs                     # Shared op wrappers over pegainfer-kernels
└── src/weight_loader.rs           # Safetensors helpers shared by model crates

crates/pegainfer-kernels/          # Shared GPU kernel/runtime crate
├── KERNELS.md                     # LLM routing index for model op -> wrapper -> FFI -> source
├── src/                           # GPU tensor types, FFI, paged KV layout, Rust ops
├── csrc/                          # Hand-written CUDA / FlashInfer C++ wrappers
└── tools/triton/                  # Triton AOT kernels (build-time compiled)

crates/pegainfer-qwen3-4b/         # Qwen3-4B model-owned engine crate
├── src/                           # Config, weights, prefill/decode/unified, scheduler/executor
├── tests/                         # Qwen3 e2e, paged attention, regression data generation
├── benches/                       # Qwen3 model-level benchmarks
└── src/kernel_plan.rs             # Model DAG phase -> kernel routing index
```

</details>

## License

MIT
