# GLM5.2 TP4 prefill-only

> **TL;DR:** `--glm52-prefill-only` runs native layer-outer prefill on one
> 4-GPU TP4 host with a cubin-free kernel stack (FlashInfer CUTLASS grouped
> MoE GEMM, DeepGEMM unpaged MQA indexer, NCCL bf16 all-reduces). 4×GB300
> 16K TTFT 409 s → 1.36 s vs the 32-row bring-up path; leads a same-day
> vLLM 0.25 rerun at every length except 4K.

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
- One coordinator chunk spans at most 128 requests (the per-request indexer
  gather tables are bounded; exceeding it fails the chunk explicitly).

The coordinator shares each `--glm52-prefill-chunk-size` budget across active
requests. The default is 16,384 token rows and longer prompts span multiple
chunks; it is not a model-length limit.

## Executor

The executor is layer-outer: each of the 78 layers processes the whole
coordinator chunk before the next layer starts, so every fp8 GEMM runs at
chunk M and each MoE layer reads its expert bank once per chunk instead of
once per tile. Only two stages sub-tile — the DSA indexer logits/top-k and
the FlashMLA sparse attention run in 512-row slices (per request segment for
the indexer). Packing the whole chunk's KV before attention is safe because
the indexer masks each query to `position + 1` keys.

Kernel stack (all compiled from source at build time — no checked-in or
downloaded cubins):

- Dense projections and the router use cuBLAS and the existing FlashInfer
  CUTLASS groupwise fp8 GEMM.
- TP4 MoE routes on device (`glm52_prefill_moe_route.cu`, hand-written
  metadata/gather/combine glue), quantizes the chunk once, gathers routes in
  fp8, and runs both expert projections as ONE FlashInfer
  `CutlassFP8GroupwiseScaledGroupGEMMSM100` grouped GEMM over the rank's 256
  routed-expert slices — fp8 weights + f32 block scales in checkpoint layout,
  no UE8M0 requant. The shared expert takes every row as a dense large-M
  chain, and a deterministic f32 combine folds `shared + Σ w_j · routed_j`
  (fixed per-row route order, no atomics).
- The DSA indexer runs its projections at chunk M, gathers the paged fp8
  index-K cache into a compact unpaged buffer PER REQUEST SEGMENT (sized by
  the per-request context cap, so requests sharing cached prefix pages can
  never overflow it), and uses the DeepGEMM SM100 *unpaged* MQA logits
  kernel (`fp8_mqa_logits` — the same kernel vLLM's DSv3.2 indexer prefill
  uses), AOT-instantiated at build time. Top-k slots come back through a
  gather-time position→slot LUT.
- Attention reuses the BF16 sparse FlashMLA prefill kernel.
- TP reductions ride NCCL bf16 all-reduces (`cudarc::nccl`), two per layer at
  chunk scale. Bring-up defaults `NCCL_MIN_NCHANNELS=16`/`NCCL_MAX_NCHANNELS=32`
  when unset: the default 2-channel ring measured ~46 GB/s (8.7 ms per
  16K-row all-reduce); 16–32 channels restore ~1.6 ms.

`OPENINFER_GLM52_PREFILL_PROFILE=1` logs a per-chunk CUDA-event section
profile (the numbers below).

## Capacity and prefix cache

When `--max-model-len` is omitted, startup derives it from the minimum free
VRAM across the four ranks after loading weights. An explicit value remains a
checked cap. The prefill reservation is:

```text
3 GiB fixed + 160 KiB × prefill_chunk_size
```

The fixed part covers the row-block-bounded MoE scratch (gathered fp8 routes
and the grouped W2 output, 8,192 rows per pass), the attention/dense
sub-tile buffers, the gathered index-K/logits buffers, and the unpacked bf16
KV pool; the per-row part covers the chunk-scale activations, MLA front and
query buffers, and the indexer top-k carry. This is a capacity ledger, not
one hidden allocation.

The existing content-hashed 64-token paged KV pool remains authoritative.
Admission matches sealed full blocks, computes only the suffix, and registers
new full blocks after a successful chunk. Multiple requests share the pool and
the coordinator row budget.

## Prefill performance

One 4×GB300 host, GLM-5.2-FP8, TP4, concurrency one, one output token,
16K chunk budget, `vllm bench serve` random prompts (5 per length after a
warm-up), medians. The vLLM column is a same-day, same-host, same-config
rerun (stock 0.25.1 container, its default Blackwell stack including the
trtllm-gen cubin MoE, FP8 E4M3 KV, 16,384-token chunked prefill).

| Input tokens | This path | Old 32-row path (#754) | vLLM 0.25 (same day) | vs vLLM | #759 cubin PR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 128 ms | 24.11 s | 220 ms | 0.58× | 170 ms |
| 4K | 357 ms | 95.17 s | 243 ms | 1.47× | 459 ms |
| 16K | 1.364 s | 409.1 s | 1.840 s | 0.74× | 1.745 s |
| 64K | 5.793 s | — | 7.551 s | 0.77× | 7.186 s |
| 128K | 12.363 s | — | 15.625 s | 0.79× | 10.312 s |

The old-path column is the pre-rework `main` (32-row tiles, host-grouped
per-expert MoE, spin-handshake all-reduce) measured with the same harness on
the same host — 188×/267×/300× at 1K/4K/16K; 64K+ was impractical to sweep.
The #759 record also carries older vLLM measurements (0.898 s at 16K /
3.773 s at 64K) from a differently-configured sweep; against those retained
numbers this path is ~1.5× at 16K+. Against the same-day rerun it leads
everywhere except 4K, where the MoE is expert-weight-read-bound (a 16K
chunk amortizes the 2.4 GiB/rank per-layer expert traffic; a 4K one does
not).

Greedy first-token parity vs that vLLM server over ten fixed prompts
(5–17K tokens): 8/10 identical; the two divergences are near-tie argmax
flips consistent with the different numerics (vLLM runs FP8 KV attention,
this path bf16).

16K-chunk CUDA-event section profile (rows=16384, one rank): mla_front 128,
pack 27, indexer 162, sparse attention 420, o_proj 65, attention AR+norm 85,
MoE GEMMs/routing 426, MoE AR 72, dense/misc ~40 — total ≈ 1.43 s.

Known follow-ups (issue #755): the FlashMLA bf16 prefill kernel runs 64
padded query heads for 16 real heads/rank (~4× attention waste — vLLM's
sm103 route through the fp8 sparse decode kernel would cut it), and the
grouped MoE GEMM has not been shape-tuned (MmaSM=1 only).

## Validation

On a single 4×GB300 host with the GLM-5.2-FP8 checkpoint:

- kernels-crate GPU gates: grouped fp8 GEMM vs an exact hand-computed
  reference (empty group included) and the route/gather/combine roundtrip;
- in-process 4-rank NCCL bf16 all-reduce smoke;
- `--max-model-len 131072` reached HTTP-ready; greedy smoke returned
  `Paris`; concurrent requests and a repeated prompt (prefix-cache
  `cached_tokens` reported) completed;
- full TTFT sweep 1K–128K completed 5/5 per length, twice consecutively;
- shared-prefix gate: two 36K-token requests sharing a 36K prefix — the
  second reports `cached_tokens=36096` and completes suffix-only in 0.07 s;
- the GLM52 library suite and workspace `--lib` tests pass.
