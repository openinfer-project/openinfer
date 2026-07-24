# GLM5.2 TP4 prefill-only

> **TL;DR:** `--glm52-prefill-only` runs native eager prefill on one 4-GPU
> TP4 host. It requires prefix caching, accepts multiple requests, emits one
> token per request, and never enters decode.

**Last touched:** 2026-07

## Contract

- Launch with `--tp-size=4 --moe-topo=tp4 --glm52-prefill-only`.
- Every request must set `max_tokens=1`.
- Prefix caching is required.
- DSpark, KV offload/external P/D, remote rank hosts, and decode graphs are
  rejected.
- The predicted token is returned without being fed back, so it has no KV
  entry.

The coordinator shares each `--glm52-prefill-chunk-size` budget across active
requests. The default is 16,384 token rows and longer prompts span multiple
chunks; it is not a model-length limit. The executor tiles each coordinator
batch into 32-row kernel calls because the reused SM103 DeepGEMM DSA kernel
supports at most 32 requests.

## Executor

Each tile runs embedding, all 78 decoder layers, final RMSNorm, a
vocabulary-sharded LM head, and global greedy argmax. Dense projections and
the router use cuBLAS or the existing large-M FP8 CUTLASS path. DSA reuses the
SM103 DeepGEMM metadata/logits kernels, and attention reuses the BF16 sparse
FlashMLA prefill kernel.

TP4 MoE keeps a quarter of every expert's intermediate dimension on each
rank. The correctness-first implementation groups routes on the host, runs
each used expert through the existing FP8 GEMMs, scatters weighted outputs,
and uses a fixed-order four-rank reduction. The reduction buffer has
publish/consume handshakes so a faster rank cannot overwrite data still being
read by a peer. Grouped expert dispatch remains the main throughput follow-up.

## Capacity and prefix cache

When `--max-model-len` is omitted, startup derives it from the minimum free
VRAM across the four ranks after loading weights. An explicit value remains a
checked cap. The prefill reservation is:

```text
256 MiB fixed + 160 KiB × prefill_chunk_size
```

The default 16K chunk reserves 2,952,790,016 bytes per rank. This is a
capacity ledger, not one hidden allocation.

The existing content-hashed 64-token paged KV pool remains authoritative.
Admission matches sealed full blocks, computes only the suffix, and registers
new full blocks after a successful chunk. Multiple requests share the pool and
the coordinator row budget.

## Validation

On a single 4×GB300 host with the GLM-5.2-FP8 checkpoint:

- `--max-model-len 1000000` with the default 16K chunk reached HTTP-ready;
- all 78 layers passed the four-rank startup preflight;
- a real HTTP request returned `Paris` with one output token and no decode;
- concurrent 5-token and 119-token requests both completed;
- repeating the 119-token prompt exercised the cached-prefix suffix path;
- the GLM52 library suite passed 78 tests with 17 GPU/oracle tests ignored;
- a four-GPU reduction test passed three consecutive buffer-reuse rounds.

The final warm-page-cache startup took 60.6 seconds, dominated by the
current per-rank host gather of 75 expert slice banks. The short HTTP smoke
completed in 0.85 seconds. These are bring-up measurements, not a
throughput claim.
