# GLM5.2 TP4 prefill-only

> **TL;DR:** `--glm52-prefill-only` runs native eager prefill on one 4-GPU
> TP4 host. It requires prefix caching, accepts multiple requests, emits one
> token per request, and never enters decode.

**Last touched:** 2026-07

## Contract

- Launch with `--tp-size=4 --moe-topo=tp4 --glm52-prefill-only`.
- The native prefill kernels require an SM100-class GPU.
- Every request must set `max_tokens=1`.
- Prefix caching is required.
- DSpark, KV offload/external P/D, remote rank hosts, and decode graphs are
  rejected.
- The predicted token is returned without being fed back, so it has no KV
  entry.

The coordinator shares each `--glm52-prefill-chunk-size` budget across active
requests. The default is 16,384 token rows and longer prompts span multiple
chunks; it is not a model-length limit. Attention runs in 512-row tiles.

## Executor

Each tile runs embedding, all 78 decoder layers, final RMSNorm, a
vocabulary-sharded LM head, and global greedy argmax. Dense projections and
the router use cuBLAS or the existing large-M FP8 path. The DSA indexer
gathers paged index keys through a compact per-request table, then runs the
upstream DeepGEMM SM100 unpaged MQA kernel. Attention reuses the BF16 sparse
FlashMLA prefill kernel.

TP4 MoE keeps a quarter of every expert's intermediate dimension on each
rank. Routed experts use FlashInfer's TRT-LLM fused-MoE runner and checked-in
SM100 cubins; the shared expert uses the existing large-M FP8 projections.
Their sum uses the fixed-order four-rank reduction. The reduction buffer has
publish/consume handshakes so a faster rank cannot overwrite data still being
read by a peer.

## Capacity and prefix cache

When `--max-model-len` is omitted, startup derives it from the minimum free
VRAM across the four ranks after loading weights. An explicit value remains a
checked cap. The prefill reservation is:

```text
2 GiB fixed + 160 KiB × prefill_chunk_size
```

The default 16K chunk reserves 4,831,838,208 bytes per rank. This is a
capacity ledger, not one hidden allocation. The 1M-context gathered index-K
workspace is about 132 MiB per rank and fits inside this reservation; the
cap-scaled KV and index caches are accounted separately.

The existing content-hashed 64-token paged KV pool remains authoritative.
Admission matches sealed full blocks, computes only the suffix, and registers
new full blocks after a successful chunk. Multiple requests share the pool and
the coordinator row budget.

## Prefill performance

The comparison used one 4×GB300 host, the same checkpoint, TP4, FP8 E4M3 KV,
concurrency one, one output token, and a 16K chunk budget in both engines.
Each server received an uncounted warm-up request before measurement, so the
table excludes model startup, first-request compilation, and cold page-cache
effects.

| Input tokens | OpenInfer median TTFT | vLLM median TTFT | OpenInfer / vLLM |
| ---: | ---: | ---: | ---: |
| 1K | 170 ms | 306 ms | 0.56× |
| 4K | 459 ms | 304 ms | 1.51× |
| 16K | 1.745 s | 0.898 s | 1.94× |
| 64K | 7.186 s | 3.773 s | 1.90× |
| 128K | 10.312 s | 8.071 s | 1.28× |

The 1K–16K rows contain five requests per engine. OpenInfer's 64K and 128K
rows contain three requests; vLLM's contain five. These results establish the
current long-context gap; they are not a throughput claim.

### vLLM TP4 versus TP4+EP4

A separate vLLM-only sweep checked whether expert parallelism improves
single-request prefill and whether the all-to-all backend changes that result.
It used vLLM `0.1.dev18497+g634ec28cd` on a single 4×GB300 host with the same
GLM-5.2-FP8 checkpoint, FP8 E4M3 KV cache, a 16K chunk budget, concurrency
one, and one greedy output token. Each row is the median of six requests after
server initialization and backend warm-up.

| vLLM configuration | 1K TTFT | 4K TTFT | 16K TTFT | 16K / TP4 |
| --- | ---: | ---: | ---: | ---: |
| TP4 | 211 ms | 226 ms | 1.445 s | 1.00× |
| TP4+EP4, DeepEP high-throughput | 254 ms | 260 ms | 1.464 s | 1.013× |
| TP4+EP4, FlashInfer NVLink two-sided | 295 ms | 253 ms | 1.467 s | 1.015× |
| TP4+EP4, allgather/reduce-scatter | 267 ms | 274 ms | 1.465 s | 1.014× |

All three requested backends initialized without a fallback warning. In this
single-request sweep, EP4 did not beat TP4: the 16K difference among A2A
backends was within 3 ms, while EP4 added a clearer fixed cost at 1K and 4K.
This does not rank the backends for batched or throughput-oriented workloads;
those need a separate concurrency sweep.

The serving benchmark sends a probe before its measured requests. An earlier
TP4 run was not sufficiently warmed, and its first measured request also hit
the probe's prefix-cache entry. Neither that run nor the tool's aggregate from
it is included above. The table uses the subsequent stable sweep and retains
the per-request timings so this boundary remains auditable.

#### Concurrency sweep

The follow-up throughput sweep used unique random token-ID sequences so the
enabled prefix cache remained in the production configuration without any
measured prefix hits. Each cell contains four full concurrency waves, with a
minimum of eight requests, after a distinct warm-up prompt. The tables report
aggregate input-token throughput; concurrency is the maximum number of
simultaneous requests.

| 1K input | C1 | C2 | C4 | C8 | C16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 | 5,249 | 5,282 | 4,383 | 5,711 | **7,305** |
| TP4+EP4, DeepEP high-throughput | 3,723 | 3,763 | 3,935 | 5,363 | **6,905** |
| TP4+EP4, FlashInfer NVLink two-sided | 3,543 | 3,362 | 3,563 | 5,145 | **6,810** |
| TP4+EP4, allgather/reduce-scatter | 3,557 | 3,695 | 4,144 | 5,391 | **6,931** |

| 4K input | C1 | C2 | C4 | C8 | C16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 | 17,039 | **17,444** | 8,441 | 10,319 | 9,632 |
| TP4+EP4, DeepEP high-throughput | 14,666 | **16,654** | 8,310 | 10,010 | 9,350 |
| TP4+EP4, FlashInfer NVLink two-sided | **15,135** | 14,686 | 8,234 | 10,019 | 9,381 |
| TP4+EP4, allgather/reduce-scatter | 15,952 | **17,066** | 7,945 | 10,025 | 9,402 |

| 16K input | C1 | C2 | C4 | C8 | C16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 | 10,276 | **10,359** | 9,745 | 9,442 | 9,297 |
| TP4+EP4, DeepEP high-throughput | 9,979 | **10,095** | 9,457 | 9,163 | 9,022 |
| TP4+EP4, FlashInfer NVLink two-sided | 10,006 | **10,155** | 9,513 | 9,218 | 9,076 |
| TP4+EP4, allgather/reduce-scatter | 10,041 | **10,113** | 9,514 | 9,218 | 9,077 |

No EP4 configuration exceeded TP4 at any measured length and concurrency.
The closest cell was 16K at concurrency two, where the fastest EP4 result was
10,155 input tok/s versus TP4's 10,359 input tok/s, a 2.0% deficit. Increasing
concurrency helped 1K throughput, but 16K was already saturated at concurrency
one or two because the 16K scheduler token budget admits one complete 16K
prefill chunk at a time.

The first 1K C1/C2 pass contained an apparent allgather/reduce-scatter win but
did not reproduce. Three additional prompt sets placed TP4 at 5,235–5,293
input tok/s and that EP4 backend at 3,439–3,868 input tok/s; the tables use
the medians of those confirmation runs for these four cells.

## Validation

On a single 4×GB300 host with the GLM-5.2-FP8 checkpoint:

- `--max-model-len 1000000` with the default 16K chunk reached HTTP-ready;
- all 78 layers passed the four-rank startup preflight;
- a real HTTP request returned `Paris` with one output token and no decode;
- five fixed greedy prompts matched vLLM TP4's first token exactly;
- four concurrent 3K-token requests with a shared prefix completed;
- repeating a 3K-token prompt reported 2,944 cached tokens and computed only
  the suffix;
- the GLM52 library suite passed 78 tests with 17 GPU/oracle tests ignored;
- a four-GPU reduction test passed three consecutive buffer-reuse rounds.

The optimized four-rank preflight completes in 2.72 seconds. Full HTTP-ready
time still includes checkpoint loading and is intentionally separate from the
warm TTFT table.
