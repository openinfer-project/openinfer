#!/usr/bin/env bash
# Re-run the issue #470 mixed-load matrix with the legal capacity
# `--max-batch 8 --bg-concurrency 4` (physical bucket == admission == 8),
# as suggested by the maintainer. 3 qps x 4 prompt x 2 chunk = 24 valid cells.
# Prereq: build bench_serving (--features qwen35) and run from the repo root.
set -u
BIN=target/release/bench_serving
DATA=datasets/mixed-load-itl-470-data-mb8
mkdir -p "$DATA"
export OPENINFER_ITL_DEBUG=1 RUST_LOG=info

echo "=== sweep start $(date +%T)  (--max-batch 8 --bg-concurrency 4) ==="
i=0
for q in 0.25 0.5 1.0; do
  for p in 4096 8192 12288 16384; do
    for chunk in on off; do
      i=$((i + 1))
      qtag=$(echo "q$q" | tr '.' 'p')
      name="sweep_${qtag}_p${p}_${chunk}"
      extra=""
      [ "$chunk" = off ] && extra="--max-prefill-tokens 99999999"
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      echo "[$(date +%T)] cell $i/24: $name  (gpu used ${used}MiB)"
      # shellcheck disable=SC2086
      "$BIN" --model-path models/Qwen3.5-4B --max-batch 8 $extra \
        --format json --out "$DATA/${name}.json" mixed \
        --bg-concurrency 4 --bg-prompt-len 512 --bg-output-len 4096 \
        --inj-prompt-len "$p" --inj-output-len 1 --qps "$q" --num-injections 10 \
        --inj-warm-frac 0.0 --warmup 5 > "$DATA/${name}.log" 2>&1
      rc=$?
      gate=$(python3 scripts/itl_step_agg.py "$DATA/${name}.log" 2>/dev/null | grep -E 'decode_n=4' | head -1)
      echo "  exit=$rc  valid-gate: ${gate:-<none>}"
      sleep 25
    done
  done
done
echo "=== SWEEP_DONE $(date +%T) ==="
