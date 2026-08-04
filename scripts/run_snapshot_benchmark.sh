#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  HF_TOKEN=... ./scripts/run_snapshot_benchmark.sh

Optional environment variables:
  MODEL_REPO    Hugging Face repo to download (default: Qwen/Qwen3-4B)
  MODEL_DIR     Local model directory (default: models/Qwen3-4B)
  WARMUP        Snapshot warmup iterations (default: 5)
  ITERS         Snapshot measured iterations (default: 20)
  CUDA_HOME     CUDA toolkit path (default: /usr/local/cuda)
EOF
  exit 0
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required" >&2
  exit 1
fi

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-4B}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-4B}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

need_apt=0
for cmd in curl git python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    need_apt=1
  fi
done
if ! command -v pip3 >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1 || ! command -v uv >/dev/null 2>&1; then
  need_apt=1
fi

if [[ "$need_apt" -eq 1 ]]; then
  export DEBIAN_FRONTEND=noninteractive
  run_root apt-get update
  run_root apt-get install -y build-essential pkg-config libssl-dev curl ca-certificates git python3-pip python3-venv
fi

if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
fi
source "$HOME/.cargo/env"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [[ -f .gitmodules ]] && [[ ! -f pegainfer-kernels/third_party/flashinfer/include/flashinfer/norm.cuh ]]; then
  git submodule update --init --recursive
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv
fi

if ! .venv/bin/python -c "import triton, huggingface_hub" >/dev/null 2>&1; then
  uv pip install -p .venv/bin/python triton "huggingface_hub[cli]"
fi

mkdir -p "$(dirname "$MODEL_DIR")"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  .venv/bin/hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" --token "$HF_TOKEN"
fi

export CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PEGAINFER_TRITON_PYTHON="$PWD/.venv/bin/python"
if [[ -z "${PEGAINFER_CUDA_SM:-}" ]]; then
  PEGAINFER_CUDA_SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
  export PEGAINFER_CUDA_SM
fi

cargo build --release --bin bench_serving
cargo run --release --bin bench_serving -- --model-path "$MODEL_DIR" snapshot --warmup "$WARMUP" --iters "$ITERS"
