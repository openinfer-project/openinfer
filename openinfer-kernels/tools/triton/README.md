# Triton AOT Integration

`openinfer` currently uses Triton AOT for the Qwen3.5 HD256 prefill kernel and the
Qwen3.5 GDR chunkwise prefill kernels.

## What this covers

- Build-time generation of Triton AOT cubins for:
  - `flash_attention_prefill_hd256_kernel.py`
  - `gated_delta_rule_chunkwise_kernels.py`
- Generated C wrappers linked into the normal Rust build
- Native CUDA now covers basic ops (`add`, `silu_mul`, `embedding`) and decode-critical paths

`build.rs` now skips compiling retired legacy translation units that are no longer present in the active kernel set.

## Prerequisites

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Bootstrap a repo-local Triton Python once:

```bash
uv venv .venv
uv pip install -p .venv/bin/python triton
```

Then either point the build to that interpreter explicitly:

```bash
export OPENINFER_TRITON_PYTHON=$PWD/.venv/bin/python
```

or let `build.rs` auto-probe `.venv/bin/python` before trying `python3` / `python`.

If `nvidia-smi` is unavailable where you build, also set the target SM manually.

```bash
export OPENINFER_CUDA_SM=120
```

`OPENINFER_CUDA_SM` also drives the explicit Triton AOT compile target, so it is the default escape hatch when the build environment cannot query a live GPU.

### Windows

Official Triton does not ship Windows wheels. Use [`triton-windows`](https://github.com/woct0rdho/triton-windows) instead:

```powershell
uv venv .venv --python 3.12
uv pip install "triton-windows<3.7"
$env:OPENINFER_TRITON_PYTHON = ".venv\Scripts\python.exe"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
```

Requires CUDA 12+, Python 3.9–3.12, and an NVIDIA GPU with compute capability 7.5+ (GTX 16xx or newer).

## Build

```bash
cargo build --release
```

Generated Triton artifacts are written to Cargo `OUT_DIR`, typically under:

```text
target/release/build/openinfer-kernels-*/out/triton_aot/
```

## Validation

Run the focused GPU tests for the active Triton-backed paths:

```bash
cargo test --release -p openinfer-qwen35-4b recurrent::tests::conv1d_prefill_handoff_matches_single_prefill -- --nocapture
OPENINFER_TEST_MODEL_PATH=/path/to/Qwen3.5-4B cargo test --release -p openinfer-qwen35-4b --test e2e_scheduler -- --nocapture
```

## Common failures

- `Could not find a Python interpreter with Triton installed`
  - Set `OPENINFER_TRITON_PYTHON`, or bootstrap `.venv` with `uv`.
- `GPU detection failed`
  - Set `OPENINFER_CUDA_SM` explicitly if `nvidia-smi` is not available during build.
- `Triton AOT generator failed`
  - Re-run the build and inspect the generator stderr printed by `build.rs`; the generator accepts an explicit `cuda:<sm>:32` target derived from `OPENINFER_CUDA_SM`.
- `CUDA_ERROR_NO_BINARY_FOR_GPU` or similar runtime load errors
  - Rebuild on the target GPU environment; the generated Triton cubin is target-specific.
