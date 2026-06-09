# DeepSeek V4 Online Throughput Baseline

Last touched: 2026-05-18

TL;DR: latest main has a clean default `deepseek-v4` quality gate on the 8x5090
validation run, but online throughput is still service-time limited. HTTP c1-c8
fixed-shape load stays around 1.3 req/s with stable hashes because the path is
still a single-request scheduler turn; mixed online load reports separate input
and output token rates. Decode has partial bs>1 primitives, prefill does not have
a native multi-request stack, and DSV4 does not use CUDA Graph yet.

## Run Identity

| Field | Value |
| --- | --- |
| Task | task #43, updated by task #47/#48 for HTTP error detection and short-prompt prefill |
| Code | latest main `019de7f` plus task #48 overlap-compressor short-prompt fix |
| Model | `DeepSeek-V4-Flash` |
| Feature path | default `deepseek-v4`; no EP/PPLX |
| Hardware class | internal 8x RTX 5090 validation host |
| Quality policy | record token/hash/text drift; precision conclusions require second review |

## Harness Delta

`scripts/bench_http_serving.py` now accepts comma-separated `--prompt-words` and
`--max-tokens`, cycles the cartesian product across requests, and reports:

- `input_tokens_total` / `input_tokens_per_s`
- `output_tokens_total` / `output_tokens_per_s`
- per-request `prompt_words` and `max_tokens`
- `mixed_shapes` counts in the workload block

`scripts/bench_http_sweep.py` carries those input/output token rates into each
sweep row. When server trace has exact token counts, the report uses them. When
trace lines are unavailable, prompt tokens fall back to `prompt_words`, and
`ignore_eos=true` output tokens fall back to requested `max_tokens` instead of
stream chunk count.

Task #47 also makes the harness fail a request when the SSE stream reports
`error` / `finish_reason=error`, the server log reports a matched stream error,
or a traced request has `completion_tokens=0` while output tokens were requested.
The older task #43 HTTP table undercounted these server-side generation errors.

## Direct Baseline

| Workload | TTFT / E2E | TPOT | Hash | Notes |
| --- | ---: | ---: | --- | --- |
| synthetic 16 prompt tokens, 160 output tokens, warmup 1, iters 3 | TTFT avg `172.07ms`, E2E avg `4707.33ms` | steady avg `28.53ms/token`, p50 `28.23ms` | `702784c0e5e56d17` across 3/3 | direct decode service-time baseline |
| synthetic 10580 prompt tokens, 1 output token, iters 1 | TTFT `3971.31ms`, E2E `3971.40ms` | n/a | `39a863e299d2b187` | 10k direct baseline |
| same 10k direct under nsys | TTFT `4066.74ms`, E2E `4066.78ms` | n/a | `39a863e299d2b187` | profiling overhead included |

## HTTP Fixed-Shape Sweep

Workload: streaming `/v1/completions`, warmup `2`, requests `8`, prompt words
`16`, max tokens `16`, repeats `3`, `ignore_eos=true`.

| Concurrency | Correctness | QPS avg | TTFT avg | TPOT avg | Combined hash |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | failed `0`, timeout `0`, per-request hashes stable | `1.33` | `275.67ms` | `31.81ms` | `4c7d24746f19ff5b` |
| 2 | failed `0`, timeout `0`, per-request hashes stable | `1.32` | `954.91ms` | `31.94ms` | `4c7d24746f19ff5b` |
| 4 | failed `0`, timeout `0`, per-request hashes stable | `1.34` | `2049.55ms` | `31.33ms` | `4c7d24746f19ff5b` |
| 8 | failed `0`, timeout `0`, per-request hashes stable | `1.34` | `3337.98ms` | `31.70ms` | `4c7d24746f19ff5b` |

Interpretation: QPS and TPOT barely move from c2 to c8 because the current HTTP
scheduler admits one request turn at a time. Concurrency mostly changes queue
shape and TTFT tail, not true active-set throughput.

## HTTP Active-Set Gate

Task #45 restores the existing `DIRECT_BATCH_DECODE_CAPACITY=2` active-set path
after the task #47/#48 baseline fixes. This gate is a serving-path comparison,
not a CUDA Graph enablement.

Workload: streaming `/v1/completions`, warmup `0`, requests `8`, prompt words
`16`, max tokens `16`, `ignore_eos=true`.

| Concurrency | Correctness | Active-set trace | QPS | TTFT avg | TPOT avg | Combined hash |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | failed `0`, timeout `0` | active set `1`, decode batch `1` | `1.35` | `278.05ms` | `31.00ms` | `4c7d24746f19ff5b` |
| 2 | failed `0`, timeout `0` | active set `2`, decode batch `2` | `1.97` | `460.56ms` | `36.80ms` | `3c7a6939ee07b4e5` |
| 4 | failed `0`, timeout `0` | active set `2`, decode batch `2` | `1.99` | `1208.86ms` | `36.67ms` | `3c7a6939ee07b4e5` |
| 8 | failed `0`, timeout `0` | active set `2`, decode batch `2` | `1.98` | `1972.79ms` | `36.67ms` | `5100f393ca65314e` |

Interpretation: active-set serving reaches the batch decode path and improves
fixed-shape output throughput by roughly half versus the single-turn baseline,
but TPOT worsens because current batch decode work is not yet as efficient as
the single-row path. c2/c4/c8 output hashes differ from the single-turn baseline;
this needs second-review quality acceptance before treating the active-set path
as the default serving behavior.

## HTTP Mixed Online Workload

Workload: streaming `/v1/completions`, warmup `2`, requests `12`, concurrency
`4`, prompt words `16,512,2048`, max tokens `16,64`, two requests per shape.

| Metric | Value |
| --- | ---: |
| Completed / failed / timeout | `12 / 0 / 0` |
| QPS | `4.35` |
| Input tokens | `10304` total, `3733.99 tok/s` |
| Output tokens | `90` total, `32.61 tok/s` |
| TTFT avg | `141.68ms` |
| TPOT avg | `28.28ms` |
| ITL avg | `28.31ms` |
| Combined output hash | `a0c1b5679448f2ee` |

Interpretation: input tok/s is high for this small mixed run because long prompts
are queued behind single-request turns and produce long prompt-token payloads;
output tok/s remains close to direct TPOT limits. This is a baseline, not a
serving-optimization result.

## 10k Kernel Buckets

Nsight Systems 10k direct, sorted by CUDA kernel total time:

| Bucket | Total | Calls | Share | Actionability |
| --- | ---: | ---: | ---: | --- |
| F32 NCCL all-reduce | `7281.70ms` | `856` | `28.2%` | collective placement/count problem; not a single CUDA replacement |
| non-overlap compressor weighted prefill | `7144.52ms` | `160` | `27.7%` | biggest standalone CUDA bucket; accumulation-order risk likely mirrors overlap compressor |
| CuTeDSL exact indexer score prefill | `1812.05ms` | `168` | `7.0%` | already default; remaining work should target shape/kernel efficiency, not feature gating |
| TileLang FP4 grouped W13 GEMM | `1383.57ms` | `344` | `5.4%` | MoE grouped GEMM bucket |
| indexer top-k prefill | `1192.36ms` | `168` | `4.6%` | already rewritten once; lower priority than non-overlap compressor/all-reduce |
| TileLang FP4 grouped W2 GEMM | `885.47ms` | `344` | `3.4%` | MoE grouped GEMM bucket |
| HC post | `733.05ms` | `688` | `2.8%` | repeated per attention/FFN branch |
| TileLang sparse indexed attention | `729.67ms` | `344` | `2.8%` | attention compute is not the current largest bucket |

## bs>1 Operator Coverage Map

| Area | Current coverage | Throughput implication |
| --- | --- | --- |
| HTTP admission | `scheduler.rs` creates `wave = vec![req]`; the `wave.len()==1` path is always taken. `handle_request_wave` exists but is not reached from serving. | c2/c4/c8 mainly queue; task #45 must make active set size > 1 before bs>1 decode kernels affect online output tok/s. |
| Decode scheduler/runtime | `DIRECT_BATCH_DECODE_CAPACITY = 2`; `run_direct_decode_batch_logits` and `block_decode_rank_lane_bf16_hidden_batch_with_scratch` exist. | partial bs>1 decode stack exists, capped at 2, but serving does not drive it. |
| Decode attention ratio 0 | batch projection/RoPE/cache scatter/attention path exists through batch block decode. | usable once active-set serving reaches batch path. |
| Decode attention ratio 4 | batch RoPE, indexed score, score all-reduce, top-k, indexed attention, and output projection exist; compressor state update still loops per row for main/indexer compressor and cache copy. | leading operator gap for Pacer after task #44: add clean `_batch_` decode compressor path or replace row-loop hot kernels. |
| Decode non-overlap compressor | no native `_batch_` compressor FFI; current batch path iterates rows around single-row compressor/update semantics. | good decode-side operator candidate because it targets bs>1 online output rather than single-request service time. |
| Decode MoE | `dispatch_decode_moe_step` accepts `input.seq_len`; local routing and grouped GEMM operate over seq length. | not first candidate until active-set path proves MoE is a top online bucket. |
| Prefill request batching | prefill starts one request into one KV slot; no multi-request prefill active set. | input throughput is mostly long-seq single-request kernel efficiency today; true batch prefill needs a larger scheduler/runtime shape change. |
| Prefill attention/compressor | prefill kernels are sequence-parallel for one request; no native multi-request DSV4 prefill stack. | Pacer prefill replacements should target high-share single-request buckets first, especially non-overlap compressor, while preserving the chosen quality policy. |
| CUDA Graph | `openinfer-server` passes `enable_cuda_graph=false` for DeepSeek V4; direct engine warns that DSV4 does not use CUDA Graph yet. | graph work starts after active-set shapes stabilize; blockers are dynamic seq/compressed lengths, collectives, stream/event ownership, allocator/scratch lifetimes, and batch capacity. |

## Next Work Selection

| Task | Owner | Entry |
| --- | --- | --- |
| task #45 HTTP active-set batching + CUDA Graph serving gate | @OpenInfer-Dev | Make serving trace show active set size > 1 under c2/c4/c8, then measure output tok/s/TPOT against this baseline. |
| task #46 decode operator replacement | @Pacer | Prefer decode compressor `_batch_` path from task #44 coverage; fallback is decode indexer top-k batch if compressor exactness blocks. |
| task #46 prefill operator replacement | @Pacer | Prefer non-overlap compressor only when local microbench/correctness and precision review show meaningful input-throughput gain; skip low-yield patches. |
