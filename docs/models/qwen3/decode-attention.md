# Qwen3-4B Decode Attention Path Selection

**TL;DR:** Decode picks between two paged-attention kernels — `NonPartition` (1 CTA per request×kv-head) and `SplitKv` (KV split into 64-token chunks across SMs). The choice is driven by **batch (CTA count vs SM count), not context length**. The old `max_seq_len >= 1024` gate was a tuning artifact that left bs=1 mid-context decode on the SM-starved `NonPartition` kernel, producing a tpot hump that peaked ~ctx800 and dropped off a cliff exactly at ctx1024. Removing it flattens bs=1 tpot across the whole context range (5090 −16% @ctx800, 5070 Ti −7.5%) with no accuracy regression. Kept the `padded_bs <= 32` cap.

Last touched: 2026-06

## The two kernels

`batch_decode_buffers.rs::attention_path()` selects per step:

- **`NonPartition`** — issues one CTA per `(request × kv-head)`. Qwen3-4B GQA has 8 kv-heads, so bs=1 launches only **8 CTAs**. A 5090 has ~170 SMs and a 5070 Ti ~70 — 8 CTAs leave the GPU almost idle, and it gets worse as context grows (each CTA walks a longer KV).
- **`SplitKv`** — splits each request's KV into 64-token chunks and runs them in parallel, then merges. This manufactures enough CTAs to fill the SMs when batch is small.

## Why batch, not context

SplitKv's value is **filling the SMs**. Whether that's needed depends on the launched CTA count (`bs × kv-heads`) versus the SM count — **context length is irrelevant**. At bs=1 the 8 CTAs underfill the GPU at *any* seq_len, so SplitKv should always be used there. The previous `max_seq_len >= 1024` condition mistook "GPU not full" for "context not long enough"; those are different things, and the result was a latency hump.

`SPLIT_KV_MAX_BATCH_SIZE = 32` is the real gate: a coarse proxy for "CTAs already saturate the SMs". Past it, `NonPartition` is fine and SplitKv's merge step is pure overhead. The transition is hardware-dependent (5090 ~bs16 ≈ 128 CTAs ≈ 170 SMs; 5070 Ti ~bs9), so the cap is deliberately not a hardcoded per-GPU number — 32 is a safe upper bound that stays net-positive on both cards.

## bs=1: the tpot hump (root cause)

bs=1 single-stream decode tpot "jittered with the dataset" — but it wasn't prefix cache or noise, it was prompt length landing on different points of the hump. `vllm bench serve --dataset-name random --random-range-ratio 0 -c1`, no prefix cache, output-len 64:

**5090, bs=1, tpot (ms):**

| input-len | 128 | 300 | 500 | 800 | 1000 | 1024 | 1100 |
|---|---|---|---|---|---|---|---|
| gate=1024 (NonPartition <1024) | 5.95 | 6.15 | 6.44 | **6.95** | 6.43 | 5.98 | 5.99 |
| no gate (SplitKv) | 5.78 | 5.78 | 5.80 | **5.84** | 5.86 | 5.86 | 5.87 |

The hump climbs to a peak at ~ctx800 then drops off a cliff exactly at ctx1024 (where the old gate flipped to SplitKv). For reference, vLLM's FlashInfer decode is a flat ~6.0ms across this range — the fix puts openinfer below it everywhere.

**5070 Ti, bs=1, tpot (ms):**

| input-len | 228 | 300 | 500 | 800 | 1000 | 1024 | 1100 |
|---|---|---|---|---|---|---|---|
| gate=1024 | 10.93 | 11.05 | 11.32 | **11.72** | 11.28 | 10.88 | 10.89 |
| no gate | 10.73 | 10.76 | 10.79 | **10.84** | 10.87 | 10.88 | 10.89 |

Same shape, smaller magnitude (−7.5% @ctx800 vs −16% on the 5090) because fewer SMs means `NonPartition`'s underutilization costs less in absolute terms. Small-context cases also improve — there is no merge-overhead penalty for using SplitKv early.

## bs>1: where the cap matters

5090, `vllm bench serve` with `--max-concurrency C`, output-len 64. This is the data behind the `padded_bs <= 32` cap and the honest accounting of its one downside:

| ctx | bs | NonPartition tpot | SplitKv tpot | Δ tpot | NonPart thr | SplitKv thr |
|---|---|---|---|---|---|---|
| 300 | 4  | 7.02  | 6.85  | **−2.4%** | 519 | 530 |
| 300 | 8  | 7.55  | 7.46  | −1.2% | 981 | 993 |
| 300 | 16 | 8.95  | 8.99  | +0.4% | 1642 | 1635 |
| 300 | 32 | 12.16 | 12.23 | +0.6% | 2392 | 2387 |
| 800 | 4  | 8.14  | 7.45  | **−8.5%** | 447 | 484 |
| 800 | 8  | 8.69  | 8.36  | −3.8% | 848 | 880 |
| 800 | 16 | 10.81 | 10.77 | −0.4% | 1347 | 1353 |
| 800 | 32 | 15.51 | 15.65 | **+0.9%** | 1860 | 1848 |

The transition lands exactly where CTA count crosses SM count: bs≤8 (≤64 CTAs) wins big, bs=16 (128 CTAs) is even, bs=32 (256 CTAs) is a **<1% loss** because `NonPartition` already fills the SMs and SplitKv pays a merge it doesn't need. That bs=32 corner is the only regression, it's at the noise floor (~0.1ms), it's in the throughput-saturated regime where latency is least sensitive, and in practice a saturated batch usually also has long context (>1024) where the old gate already chose SplitKv. Net: a large win across the latency-sensitive low-concurrency range for a sub-1% cost at the saturation corner.

## CUDA graph capture with SplitKv

The grid must be fixed for graph replay, but SplitKv's chunk count varies with context. Resolved by **fixed-upper-bound grid + out-of-graph metadata**:

- One captured graph per `(batch bucket, attention_path)` — `graph_index = bucket_idx × 2 + path.graph_slot()`. bs=1 has a `NonPartition` slot and a `SplitKv` slot; first use of a combination captures, later steps replay.
- SplitKv workspace is pre-allocated at the fixed upper bound `bs × 64` chunks (`SPLIT_KV_MAX_CHUNKS_PER_REQUEST`), so buffer pointers are stable and the kernel always launches the same grid regardless of context.
- Per-step context differences go through `memcpy_htod` in `sync_split_kv_meta` **before** `run_or_capture` (outside the graph): chunk_size, valid-chunk count, `valid_mask`, `o_indptr`. Chunks beyond the real count are masked off (`valid_mask = 0`, those CTAs early-exit).

So ctx=300 (5 chunks) and ctx=1024 (16 chunks) share the same SplitKv graph — only the metadata buffer contents differ. This is why dropping the context gate is safe: capture-time context never determined the grid. The accuracy gate's CUDA-graph replay over small-context sequences passes, confirming it.

## The fix

`attention_path()` now gates on `padded_bs <= SPLIT_KV_MAX_BATCH_SIZE` only; the `SPLIT_KV_MIN_SEQ_LEN` constant is gone. Verified with `tests/hf_golden_gate.rs` (small-context sequences now run entirely on SplitKv): head delta stays at bf16 noise level (mean ~0.03, p99 ~0.11), no regression.

Reproduce a curve:

```bash
OPENINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p openinfer-qwen3-4b --test hf_golden_gate
# tpot sweep against a running server:
vllm bench serve --backend openai --base-url http://localhost:8001 --endpoint /v1/completions \
  --model models/Qwen3-4B --dataset-name random --random-input-len 800 --random-output-len 64 \
  --random-range-ratio 0 --num-prompts 10 --max-concurrency 1 --ignore-eos
```
