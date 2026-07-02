#!/usr/bin/env bash
# One-shot serving benchmark: builds and launches openinfer, runs a QPS sweep
# (and optional DSpark concurrency sweep), then summarizes results.
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
#   DRAFT_MODEL      DSpark/DFlash draft model path (omit to skip spec sweep)
#   GPU              CUDA device ordinal [default: 0]
#   PORT             server port [default: 8000]
#   RESULT_DIR       output directory [default: ./bench-results]
#   QPS_LIST         space-separated QPS values [default: "1 2 4 8 10 12 16"]
#   CONCURRENCY_LIST space-separated concurrency values for spec sweep [default: "1 4 8"]
#   INPUT_LEN        input length [default: 1024]
#   OUTPUT_LEN       output length [default: 128]
#   SEED             random seed [default: 42]
#   SECONDS_PER_RUN  seconds per QPS run [default: 60]
#   BENCH            path to vllm-bench binary [default: vllm-bench on PATH]
#   ENGINE           engine label for result filenames [default: openinfer]
#
# Examples:
#   # Qwen3-4B QPS sweep only
#   MODEL=/data/Qwen3-4B GPU=7 tools/bench/run_serving_bench.sh
#
#   # Qwen3-4B QPS sweep + DSpark concurrency sweep
#   MODEL=/data/Qwen3-4B DRAFT_MODEL=/data/dspark_qwen3_4b_block7 GPU=7 \
#     tools/bench/run_serving_bench.sh
#
#   # Qwen3-8B QPS sweep
#   MODEL=/data/Qwen3-8B GPU=7 QPS_LIST="1 2 4 8 12" \
#     tools/bench/run_serving_bench.sh
set -euo pipefail

MODEL=${MODEL:?MODEL (model path) is required}
DRAFT_MODEL=${DRAFT_MODEL:-}
GPU=${GPU:-0}
PORT=${PORT:-8000}
RESULT_DIR=${RESULT_DIR:-./bench-results}
QPS_LIST=${QPS_LIST:-"1 2 4 8 10 12 16"}
CONCURRENCY_LIST=${CONCURRENCY_LIST:-"1 4 8"}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
SEED=${SEED:-42}
SECONDS_PER_RUN=${SECONDS_PER_RUN:-60}
BENCH=${BENCH:-vllm-bench}
ENGINE=${ENGINE:-openinfer}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/target/release/openinfer"
LABEL=$(basename "$MODEL")

mkdir -p "$RESULT_DIR"

# ---- launch server ----------------------------------------------------------
SERVER_EXTRA_ARGS=()
if [[ -n "$DRAFT_MODEL" ]]; then
  SERVER_EXTRA_ARGS+=(--dflash-draft-model-path "$DRAFT_MODEL")
  LABEL="${LABEL}-dspark"
fi

echo "=== launching openinfer: model=$MODEL gpu=$GPU port=$PORT draft=${DRAFT_MODEL:-none} ==="
CUDA_VISIBLE_DEVICES=$GPU "$BINARY" \
  --model-path "$MODEL" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  "${SERVER_EXTRA_ARGS[@]}" \
  > "$RESULT_DIR/server-$LABEL.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "=== shutting down server (pid $SERVER_PID) ==="
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ---- wait for readiness -----------------------------------------------------
echo "=== waiting for server readiness ==="
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "=== server ready (after ${i}s) ==="
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FATAL: server process died. Server log:" >&2
    cat "$RESULT_DIR/server-$LABEL.log" >&2
    exit 1
  fi
  sleep 1
done

if ! curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
  echo "FATAL: server not ready after 120s" >&2
  cat "$RESULT_DIR/server-$LABEL.log" >&2
  exit 1
fi

# ---- QPS sweep --------------------------------------------------------------
if [[ -n "${QPS_LIST// /}" ]]; then
  echo "=== QPS sweep: qps=[$QPS_LIST] in=$INPUT_LEN out=$OUTPUT_LEN ==="
for QPS in $QPS_LIST; do
  NUM_PROMPTS=$(python3 -c "print(int($QPS * $SECONDS_PER_RUN))")
  echo "--- $LABEL qps=$QPS num_prompts=$NUM_PROMPTS ---"
  "$BENCH" \
    --backend openai --model "$MODEL" --port "$PORT" \
    --base-url "http://localhost:$PORT" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$QPS" \
    --seed "$SEED" \
    --ignore-eos --temperature 0 \
    --tokenizer "$MODEL" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$RESULT_DIR" \
    --result-filename "${ENGINE}-${LABEL}-in${INPUT_LEN}-out${OUTPUT_LEN}-qps${QPS}-seed${SEED}.json"
  done
else
  echo "=== QPS sweep skipped (QPS_LIST is empty) ==="
fi

# ---- DSpark/DFlash concurrency sweep ---------------------------------------
if [[ -n "$DRAFT_MODEL" ]]; then
  echo "=== spec concurrency sweep: c=[$CONCURRENCY_LIST] in=$INPUT_LEN out=$OUTPUT_LEN ==="
  for C in $CONCURRENCY_LIST; do
    NUM_PROMPTS=$(python3 -c "print(int($C * $SECONDS_PER_RUN))")
    echo "--- $LABEL c=$C num_prompts=$NUM_PROMPTS ---"
    "$BENCH" \
      --backend openai --model "$MODEL" --port "$PORT" \
      --base-url "http://localhost:$PORT" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$C" \
      --seed "$SEED" \
      --ignore-eos --temperature 0 \
      --tokenizer "$MODEL" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir "$RESULT_DIR" \
      --result-filename "${ENGINE}-${LABEL}-in${INPUT_LEN}-out${OUTPUT_LEN}-c${C}-seed${SEED}.json"
  done
fi

# ---- summary ---------------------------------------------------------------
echo ""
echo "=== results summary ==="
"$SCRIPT_DIR/summarize_qps_sweep.py" "$RESULT_DIR"/${ENGINE}-${LABEL}-*.json
echo ""
echo "results saved to $RESULT_DIR"
