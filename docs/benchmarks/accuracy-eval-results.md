# Accuracy Evaluation Results — Phase 1 (GSM8K)

## Summary

| Model | Backend | GSM8K 8-shot (strict-match) | GSM8K 8-shot (flexible-extract) | Delta vs HF | Status |
|-------|---------|----------------------------:|--------------------------------:|:-----------:|:------:|
| Qwen3-4B | HF transformers | 85.82% | 85.82% | — | baseline |
| Qwen3-4B | pegainfer | 85.37% | 85.44% | -0.45% / -0.38% | PASS |
| Qwen3.5-4B | HF transformers | 79.45% | 79.45% | — | baseline |
| Qwen3.5-4B | pegainfer | 1.97% | 10.61% | -77.48% | FAIL |

**Pass criteria:** delta < 1%.

## Qwen3-4B: PASS

Pegainfer and HF transformers produce near-identical results. The 0.45% delta is well within the 1% threshold and consistent with expected bf16 tie-sensitive rounding differences (2/13 token-level mismatches observed in prior token-level validation).

## Qwen3.5-4B: FAIL — Long-Prompt Prefill Quality Divergence

### Symptoms

Pegainfer scored 10.61% (flexible) vs HF's 79.45% on GSM8K 8-shot.

### Root Cause

Qwen3.5-4B produces divergent outputs in pegainfer vs HF transformers when processing long prompts (8-shot few-shot prefix, ~1771 input tokens):

- **0-shot (41 tokens):** pegainfer and HF output match — both generate `<think>\n\n</think>` followed by a correct answer.
- **8-shot (1771 tokens):** outputs diverge completely.
  - HF: ` Natalia sold 48 / 2 = <<48/2=24>>24` (correct format, correct answer)
  - pegainfer: ` 168\n\nQuestion: Question: Question:...` (wrong number, degenerate repetition)

The first generated token already differs, indicating the prefill logits diverge for long sequences. This does not affect Qwen3-4B (which uses a standard transformer architecture), only Qwen3.5-4B (which uses a hybrid Mamba-attention architecture with different prefill kernels).

### Next Steps

- File a separate issue for Qwen3.5-4B long-prompt prefill accuracy investigation
- Phase 3 MMLU/HellaSwag/ARC (loglikelihood tasks) may also be affected for Qwen3.5-4B

## Reproducible Commands

### Environment

```
lm-eval: 0.4.11
transformers: 5.4.0
torch: 2.11.0+cu128
GPU: NVIDIA GeForce RTX 5070 Ti (16GB)
pegainfer: commit 280e457 (main)
```

### HF Baselines

```bash
# From the repo root (where .venv and models/ are)

# Qwen3-4B
.venv/bin/lm_eval run --model hf \
  --model_args pretrained=models/Qwen3-4B,dtype=bfloat16 \
  --tasks gsm8k --num_fewshot 8 \
  --output_path results/hf-qwen3-4b

# Qwen3.5-4B
.venv/bin/lm_eval run --model hf \
  --model_args pretrained=models/Qwen3.5-4B,dtype=bfloat16 \
  --tasks gsm8k --num_fewshot 8 \
  --output_path results/hf-qwen35-4b
```

### Pegainfer Eval

```bash
# Start server (one model at a time, single GPU)
PEGAINFER_TRITON_PYTHON=.venv/bin/python \
  cargo run --release -- --model-path models/Qwen3-4B --port 8000 --cuda-graph=false

# Run eval (separate terminal, from repo root)
.venv/bin/lm_eval run --model local-completions \
  --model_args "model=Qwen3-4B,base_url=http://localhost:8000/v1/completions,tokenizer_backend=huggingface,tokenizer=models/Qwen3-4B,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 8 --batch_size 1 \
  --output_path results/pegainfer-qwen3-4b
```

**Note:** `local-completions` requires `tokenized_requests=False` and `base_url` pointing to the full `/v1/completions` endpoint.

## Timing

| Run | Duration |
|-----|----------|
| HF Qwen3-4B | ~1h43m |
| HF Qwen3.5-4B | ~2h11m |
| pegainfer Qwen3-4B | ~1h20m |
| pegainfer Qwen3.5-4B | ~1h16m |
