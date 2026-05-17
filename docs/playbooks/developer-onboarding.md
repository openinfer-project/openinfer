# Developer Onboarding: Setting Up the pegainfer Dev Environment from Scratch

**Status**: Complete
**TL;DR**: Full new-developer onboarding — toolchain check, unified venv, build, tests, benchmark smoke test.

---

## Prerequisites

- GPU machine with CUDA toolkit installed (`/usr/local/cuda`)
- Model files under `models/` (at least one of the supported models below)

## 1. Verify Toolchain

```bash
rustc --version   # need 1.91+ (Rust 2024 edition)
uv --version      # Python package manager
/usr/local/cuda/bin/nvcc --version  # CUDA compiler
```

## 2. Create Unified Python venv

The project uses a single `.venv` for everything: triton (build dependency) and torch/transformers (reference scripts).

```bash
cd pegainfer/
uv venv
uv pip install triton torch transformers accelerate pytest
```

Verify:

```bash
.venv/bin/python -c "import triton; print(triton.__version__)"
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> build.rs auto-detects `.venv/bin/python` for Triton AOT compilation. Override with `PEGAINFER_TRITON_PYTHON` if needed.

## 3. Build

```bash
cargo build --release
```

First build takes ~30s. Compiles CUDA kernels (`pegainfer-kernels/csrc/*.cu`) and Triton AOT kernels (`pegainfer-kernels/tools/triton/*.py`).

## 4. Run Tests

```bash
cargo test -r --workspace --lib   # unit tests (~9s)
cargo test -r -p pegainfer-qwen3-4b --test e2e     # Qwen3 e2e (~50s, needs GPU + model)
```

> **Always use `--release`**. Debug builds are extremely slow for GPU code and will timeout.

Tests requiring Qwen3-8B are marked `#[ignore]` and won't affect the default flow.

## 5. Benchmark Smoke Test

```bash
cargo run -r --bin bench_serving -- request --output-len 32 --iters 3 --warmup 1
```

Expected output (ballpark):

```
ttft_ms       ~14ms
steady_tpot   ~10.5ms
decode_tok_s  ~95 tok/s
```

If you see numbers in this range, the environment is working.

## 6. Start the HTTP Server

```bash
RUST_LOG=info cargo run --release -- --port 8000
```

Test the API:

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":16}' | python3 -m json.tool
```

## Supported Models

All commands default to `models/Qwen3-4B`. Use `--model-path` to switch.

| Model | Path | Notes |
| --- | --- | --- |
| Qwen3-4B | `models/Qwen3-4B` | Default. Tied embeddings (no separate lm_head). |
| Qwen3.5-4B | `models/Qwen3.5-4B` | Hybrid attention (mixed full + sliding window layers). |

### Running a Different Model

Server:

```bash
RUST_LOG=info cargo run -r -- --model-path models/Qwen3.5-4B --port 8000
```

Benchmark:

```bash
cargo run -r --bin bench_serving -- --model-path models/Qwen3.5-4B request
```

E2E tests - Qwen3 and Qwen3.5 live in separate model crates:

```bash
cargo test -r -p pegainfer-qwen3-4b --test e2e     # Qwen3-4B exact greedy regression
cargo test -r -p pegainfer-qwen35-4b --test e2e    # Qwen3.5-4B exact greedy regression
```

### Regenerating Test Data

After kernel changes that affect numerical output, regenerate greedy reference data:

```bash
cargo test -r -p pegainfer-qwen3-4b --test regen_test_data -- --ignored    # writes test_data/Qwen3-4B.json
cargo test -r -p pegainfer-qwen35-4b --test regen_test_data -- --ignored   # writes test_data/Qwen3.5-4B.json
```

Then re-run the e2e tests to confirm the new baselines pass.

## Next Steps

- `docs/playbooks/profiling-guide.md` — profiling toolchain (nsys, ncu, fastrace, Perfetto)
- `docs/playbooks/bench-vs-vllm.md` — comparative benchmarking against vLLM
- `CLAUDE.md` (workspace + project level) — architecture and dev conventions
