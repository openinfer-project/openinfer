#!/usr/bin/env bash
# One-shot serving benchmark: launches a server (openinfer or vLLM), runs a QPS
# sweep (and optional DSpark concurrency sweep, openinfer only), then summarizes.
#
# The script launches the server, waits for readiness, runs all sweeps, and
# kills the server on exit (trap). Results land in RESULT_DIR as JSON + a
# summary table on stdout.
#
# Usage:
#   MODEL=/data/Qwen3-4B tools/bench/run_serving_bench.sh
#
# Optional env:
#   MODEL            model path (required)
#   ENGINE           openinfer | vllm [default: openinfer]
#   DRAFT_MODEL      DSpark/DFlash draft model path (openinfer only, skip spec sweep if omitted)
#   GPU              CUDA device ordinal [default: 0]
#   PORT             server port [default: 8000]
#   RESULT_DIR       output directory [default: ./bench-results]
#   DATASET          vllm-bench dataset: random | sharegpt | sonnet | speed-bench [default: random]
#   QPS_LIST         space-separated QPS values [default: "1 2 4 8 10 12 16"]
#   CONCURRENCY_LIST space-separated concurrency values for spec sweep [default: "1 4 8"]
#   INPUT_LEN        input length [default: 1024]
#   OUTPUT_LEN       output length [default: 128]
#   SEED             random seed [default: 42]
#   SECONDS_PER_RUN  seconds per QPS run [default: 60]
#   BENCH            path to vllm-bench binary [default: vllm-bench on PATH]
#   VLLM             path to vllm binary for ENGINE=vllm [default: vllm on PATH]
#   VLLM_EXTRA_ARGS  extra args passed to `vllm serve` [default: "--max-model-len 8192"]
#   LABEL            engine label for result filenames [default: $ENGINE]
#
# Examples:
#   # openinfer Qwen3-4B QPS sweep
#   MODEL=/data/Qwen3-4B GPU=7 tools/bench/run_serving_bench.sh
#
#   # openinfer Qwen3-4B + DSpark concurrency sweep
#   MODEL=/data/Qwen3-4B DRAFT_MODEL=/data/dspark_qwen3_4b_block7 GPU=7 \
#     QPS_LIST="" CONCURRENCY_LIST="1 4 8" tools/bench/run_serving_bench.sh
#
#   # vLLM Qwen3-4B QPS sweep
#   ENGINE=vllm MODEL=/data/Qwen3-4B GPU=7 \
#     VLLM=~/develop/xingming/.venv/bin/vllm tools/bench/run_serving_bench.sh
set -euo pipefail

MODEL=${MODEL:?MODEL (model path) is required}
ENGINE=${ENGINE:-openinfer}
DRAFT_MODEL=${DRAFT_MODEL:-}
GPU=${GPU:-0}
PORT=${PORT:-8000}
RESULT_DIR=${RESULT_DIR:-./bench-results}
DATASET=${DATASET:-random}
QPS_LIST=${QPS_LIST-"1 2 4 8 10 12 16"}
CONCURRENCY_LIST=${CONCURRENCY_LIST:-"1 4 8"}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
SEED=${SEED:-42}
SECONDS_PER_RUN=${SECONDS_PER_RUN:-60}
BENCH=${BENCH:-vllm-bench}
LABEL=${LABEL:-$ENGINE}
SKIP_BUILD=${SKIP_BUILD:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_LABEL=$(basename "$MODEL")

mkdir -p "$RESULT_DIR"

# ---- launch server ----------------------------------------------------------
case "$ENGINE" in
  openinfer)
    BINARY="$REPO_ROOT/target/release/openinfer"
    if [[ "$SKIP_BUILD" != "1" ]]; then
      echo "=== building openinfer (SKIP_BUILD=1 to skip) ==="
      (cd "$REPO_ROOT" && CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} cargo build --release -p openinfer-server)
    fi
    SERVER_EXTRA_ARGS=()
    if [[ -n "$DRAFT_MODEL" ]]; then
      SERVER_EXTRA_ARGS+=(--dflash-draft-model-path "$DRAFT_MODEL")
      MODEL_LABEL="${MODEL_LABEL}-dspark"
    fi
    echo "=== launching openinfer: model=$MODEL gpu=$GPU port=$PORT draft=${DRAFT_MODEL:-none} ==="
    CUDA_VISIBLE_DEVICES=$GPU "$BINARY" \
      --model-path "$MODEL" \
      --port "$PORT" \
      --served-model-name "$MODEL" \
      "${SERVER_EXTRA_ARGS[@]}" \
      > "$RESULT_DIR/server-${ENGINE}-${MODEL_LABEL}.log" 2>&1 &
    SERVER_PID=$!
    READY_TIMEOUT=120
    ;;
  vllm)
    VLLM=${VLLM:-vllm}
    VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-"--max-model-len 8192"}
    if [[ -n "$DRAFT_MODEL" ]]; then
      echo "WARN: DRAFT_MODEL is ignored for ENGINE=vllm" >&2
    fi
    echo "=== launching vLLM: model=$MODEL gpu=$GPU port=$PORT ==="
    CUDA_VISIBLE_DEVICES=$GPU "$VLLM" serve "$MODEL" \
      --port "$PORT" \
      --served-model-name "$MODEL" \
      --trust-remote-code \
      $VLLM_EXTRA_ARGS \
      > "$RESULT_DIR/server-${ENGINE}-${MODEL_LABEL}.log" 2>&1 &
    SERVER_PID=$!
    # vLLM cold start (torch.compile) can take 70+ seconds
    READY_TIMEOUT=300
    ;;
  *)
    echo "FATAL: ENGINE must be 'openinfer' or 'vllm', got '$ENGINE'" >&2
    exit 1
    ;;
esac

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "=== shutting down server (pid $SERVER_PID) ==="
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ---- wait for readiness -----------------------------------------------------
echo "=== waiting for server readiness (timeout ${READY_TIMEOUT}s) ==="
for i in $(seq 1 "$READY_TIMEOUT"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FATAL: server process died. Server log:" >&2
    cat "$RESULT_DIR/server-${ENGINE}-${MODEL_LABEL}.log" >&2
    exit 1
  fi
  if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "=== server ready (after ${i}s) ==="
    break
  fi
  sleep 1
done

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "FATAL: server process died during readiness wait" >&2
  cat "$RESULT_DIR/server-${ENGINE}-${MODEL_LABEL}.log" >&2
  exit 1
fi

if ! curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
  echo "FATAL: server not ready after ${READY_TIMEOUT}s" >&2
  cat "$RESULT_DIR/server-${ENGINE}-${MODEL_LABEL}.log" >&2
  exit 1
fi

# ---- QPS sweep --------------------------------------------------------------
if [[ -n "${QPS_LIST// /}" ]]; then
  echo "=== QPS sweep: qps=[$QPS_LIST] dataset=$DATASET ==="
  DATASET_ARGS=(--dataset-name "$DATASET")
  if [[ "$DATASET" == "random" ]]; then
    DATASET_ARGS+=(--random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN")
  fi
  for QPS in $QPS_LIST; do
    NUM_PROMPTS=$(python3 -c "print(int($QPS * $SECONDS_PER_RUN))")
    echo "--- $LABEL $MODEL_LABEL qps=$QPS num_prompts=$NUM_PROMPTS dataset=$DATASET ---"
    "$BENCH" \
      --backend openai --model "$MODEL" --port "$PORT" \
      --base-url "http://localhost:$PORT" \
      "${DATASET_ARGS[@]}" \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate "$QPS" \
      --seed "$SEED" \
      --ignore-eos --temperature 0 \
      --tokenizer "$MODEL" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir "$RESULT_DIR" \
      --result-filename "${LABEL}-${MODEL_LABEL}-${DATASET}-qps${QPS}-seed${SEED}.json"
  done
else
  echo "=== QPS sweep skipped (QPS_LIST is empty) ==="
fi

# ---- Concurrency sweep (openinfer only) ------------------------------------
if [[ "${ENGINE}" == "openinfer" && -n "${CONCURRENCY_LIST// /}" ]]; then
  echo "=== spec concurrency sweep: c=[$CONCURRENCY_LIST] dataset=$DATASET ==="
  DATASET_ARGS=(--dataset-name "$DATASET")
  if [[ "$DATASET" == "random" ]]; then
    DATASET_ARGS+=(--random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN")
  fi
  for C in $CONCURRENCY_LIST; do
    NUM_PROMPTS=$(python3 -c "print(int($C * $SECONDS_PER_RUN))")
    echo "--- $LABEL $MODEL_LABEL c=$C num_prompts=$NUM_PROMPTS dataset=$DATASET ---"
    "$BENCH" \
      --backend openai --model "$MODEL" --port "$PORT" \
      --base-url "http://localhost:$PORT" \
      "${DATASET_ARGS[@]}" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$C" \
      --seed "$SEED" \
      --ignore-eos --temperature 0 \
      --tokenizer "$MODEL" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir "$RESULT_DIR" \
      --result-filename "${LABEL}-${MODEL_LABEL}-${DATASET}-c${C}-seed${SEED}.json"
  done
fi

# ---- summary ---------------------------------------------------------------
echo ""
echo "=== results summary ==="
"$SCRIPT_DIR/summarize_qps_sweep.py" "$RESULT_DIR"/${LABEL}-${MODEL_LABEL}-*.json
echo ""
echo "results saved to $RESULT_DIR"
