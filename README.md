<p align="center">
  <img src="logo.png" width="200" alt="pegainfer logo">
</p>

<h1 align="center">pegainfer</h1>

<p align="center">
  Pure Rust + CUDA LLM inference engine. No PyTorch. No model framework runtime.
</p>

<p align="center">
  <a href="#performance">Performance</a> &middot;
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#supported-models">Models</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="#architecture">Architecture</a>
</p>

---

pegainfer is a from-scratch LLM inference engine written in **~9.6K lines of Rust**, **~2.6K lines of CUDA**, and **~1.4K lines of Triton GPU kernels**. No PyTorch, no ONNX, no model framework runtime — just Rust plus CUDA, Triton AOT, and generated compatibility kernels.

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
- TileLang for `deepseek-v4` feature builds (build-time only)

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

> **Note**: The server CLI is in `pegainfer-server`. Model crates such as `pegainfer-qwen3-4b`, `pegainfer-qwen35-4b`, and `pegainfer-deepseek-v4` contain model logic and diagnostics but are not server entrypoints. Use `cargo run --release` from the workspace root, or `cargo run --release -p pegainfer-server -- --model-path <path>`.

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

# DeepSeek V4 Flash requires the feature-gated MP8 path and TileLang at build time
uv pip install "tilelang==0.1.9"
export PEGAINFER_TILELANG_PYTHON=.venv/bin/python
cargo run --release --features deepseek-v4 -- --model-path models/DeepSeek-V4-Flash

# Disable CUDA Graph (useful for debugging)
cargo run --release -- --cuda-graph=false
```

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `CUDA_HOME` | CUDA Toolkit path (default: `/usr/local/cuda`) |
| `PEGAINFER_TRITON_PYTHON` | Python with Triton for build-time AOT compilation |
| `PEGAINFER_TILELANG_PYTHON` | Python with TileLang for `deepseek-v4` build-time kernel generation |
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
cargo run --release -p pegainfer-server -- --model-path models/Qwen3-4B
```

</details>

## Supported Models

| Model | Architecture | Params | Status |
|-------|-------------|--------|--------|
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Full attention (GQA) | 4B | Greedy + sampling |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Full attention (GQA) | 8B | Greedy + sampling |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Hybrid (24 linear + 8 full attention) | 4B | Greedy + sampling |
| [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | MoE + EP | 15.7B total / 2.4B active | Feature-gated, `--features deepseek-v2-lite`, 2-GPU path |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | MoE + sparse attention, MP8 checkpoint | 671B total / 37B active | Initial greedy, feature-gated, 8-GPU MP8 |
| [Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct) | MLA + MoE + Marlin INT4 | 1T total / 32B active | Feature-gated, `--features kimi-k2`, 8-GPU EP path |

Model type is auto-detected from `config.json` — just point `--model-path` at any supported model directory. Feature-gated model lines require rebuilding `pegainfer-server` with the matching `--features ...` flag before launch.

DeepSeek V4 support is intentionally narrower than the Qwen paths in the initial PR: it requires `--features deepseek-v4`, uses CUDA devices `0..7`, serves greedy requests only, terminates unsupported logprobs and non-greedy sampling requests with an explicit `stop_reason`, and does not use CUDA Graph yet.

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

Sampling and logprob support is model-dependent. Qwen models support the sampling controls above; the initial DeepSeek V4 path accepts greedy requests only and reports unsupported parameters through `stop_reason`.

## Architecture

```
HTTP / vLLM frontend → EngineHandle → per-model engine crate
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
pegainfer-qwen3-4b      pegainfer-qwen35-4b     pegainfer-deepseek-v4
  (full attention)      (24 linear + 8 full)    (MP8 MoE + sparse attn)
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                  pegainfer-core runtime + pegainfer-kernels
                                  │
                  CUDA / cuBLAS / Triton / TileLang / FlashInfer
```

**Key design decisions:**

- **GPU-first runtime** — model execution stays in native Rust/CUDA paths; initial DeepSeek V4 still performs host-side greedy token selection from rank0 logits
- **Custom GPU kernels** — CUDA for decode-critical paths, Triton AOT for Qwen3.5 compatibility kernels, TileLang-generated CUDA for DeepSeek V4 compatibility kernels, FlashInfer for paged attention/sampling, NCCL for multi-GPU reductions, and cuBLAS for matrix multiplication
- **Fused operators where mature** — Qwen decode paths use fused attention/MLP kernels; DeepSeek V4 is currently a multi-stage MP8 path with TileLang kernels, NCCL reductions, and CUDA glue
- **BF16 storage, FP32 accumulation** — numerical stability without memory overhead
- **CUDA Graph** on Qwen decode paths — eliminates kernel launch overhead where enabled
- **Per-model crate boundary** — Qwen3-4B owns its config, weights, scheduler/executor, tests, benches, and kernel plan in `pegainfer-qwen3-4b`

**Model details:**

- **Qwen3**: 32 Q heads, 8 KV heads (GQA 4:1), head_dim=128
- **Qwen3.5**: hybrid — 24 linear attention layers (Gated Delta Rule) + 8 full attention layers, head_dim=256
- **DeepSeek V4 Flash**: feature-gated 8-way MP8 checkpoint with MoE routing, sparse attention, FP8/FP4 TileLang kernels, and OpenAI-compatible greedy serving

### What's not (yet) implemented

- Additional quantization modes such as INT8/INT4

## Development

### Tests

```bash
# Unit tests
cargo test --release --workspace --lib

# Accuracy and integration tests (need GPU + model weights)
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test hf_golden_gate
PEGAINFER_TEST_MODEL_PATH=models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35-4b --test hf_golden_gate
PEGAINFER_TEST_MODEL_PATH=models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35-4b --test e2e_scheduler
PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V4-Flash cargo test --release -p pegainfer-deepseek-v4 --features deepseek-v4 --test e2e
```

### Triton AOT

Triton compiles the Qwen3.5 compatibility AOT kernels at build time. Qwen3-4B dense full-attention kernels are CUDA/cuBLAS/FlashInfer C++ wrappers. Runtime has no Python dependency — Triton is build-time only.

See `pegainfer-kernels/tools/triton/README.md` for setup and troubleshooting.

### Source Layout

<details>
<summary>Expand</summary>

```
Cargo.toml                         # Virtual workspace root

pegainfer-server/                  # Product package: CLI, vLLM frontend, benchmarks
├── src/main.rs                    # CLI + vLLM/OpenAI server startup
├── src/vllm_frontend.rs           # vLLM engine-core bridge into a generic EngineHandle
├── src/server_engine.rs           # Model detection and shared server helpers
├── src/scheduler.rs               # Compatibility re-export of core engine request/event types
├── src/ops.rs                     # Compatibility re-export of shared GPU ops
├── src/ops/tests.rs               # Server package operator coverage tests
├── src/tensor.rs                  # Re-export of pegainfer-kernels tensor types
├── src/sampler.rs                 # Temperature, top-k, top-p sampling
└── src/logging.rs                 # Runtime logging setup

pegainfer-core/                    # Shared runtime API for model crates
├── src/engine.rs                  # EngineHandle, GenerateRequest, TokenEvent
├── src/kv_pool.rs                 # Paged KV pool and request state
├── src/ops.rs                     # Shared op wrappers over pegainfer-kernels
└── src/weight_loader.rs           # Safetensors helpers shared by model crates

pegainfer-kernels/                 # Shared GPU kernel/runtime crate
├── KERNELS.md                     # LLM routing index for model op -> wrapper -> FFI -> source
├── src/                           # GPU tensor types, FFI, paged KV layout, Rust ops
├── csrc/                          # Hand-written CUDA / FlashInfer C++ wrappers
└── tools/triton/                  # Triton AOT kernels (build-time compiled)

pegainfer-qwen3-4b/                # Qwen3-4B model-owned engine crate
├── src/                           # Config, weights, prefill/decode/unified, scheduler/executor
├── tests/                         # Qwen3 HF logits gate and integration coverage
├── benches/                       # Qwen3 model-level benchmarks
└── src/kernel_plan.rs             # Model DAG phase -> kernel routing index

pegainfer-qwen35-4b/               # Qwen3.5-4B model-owned engine crate
├── src/                           # Config, weights, prefill/decode/unified, recurrent state, scheduler
├── tests/                         # Qwen3.5 HF logits gate and scheduler integration
└── benches/                       # Qwen3.5 recurrent/norm operator benchmarks
```

</details>

## License

MIT
