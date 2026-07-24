# GLM5.2 TP4 prefill-only

> **TL;DR:** PR1 establishes a fail-closed TP4 prefill-only service contract:
> one active request per four-rank group, prefix caching required,
> `max_tokens=1`, a page-aligned 16K default future native-prefill chunk, and
> VRAM-derived `max_model_len` after reserving that chunk's scratch envelope.
> It does not add the large-M executor: prompt rows temporarily ride the
> existing small-row compatibility path and the prompt tail returns the first
> predicted token without scheduling decode.

**Last touched:** 2026-07

## Boundary

`--glm52-prefill-only` is deliberately narrower than normal GLM5.2 serving:

- topology is TP4 (`--moe-topo=tp4 --tp-size=4`);
- prefix caching stays enabled;
- one request is active at a time;
- every request must set `max_tokens=1`;
- the prompt tail's logits produce that one token and the request finishes;
- DSpark, KV offload/external P/D, and decode-graph export are rejected.

The existing content-hashed 64-token paged KV pool remains the source of truth.
Admission performs the normal longest-full-block prefix match, feeds only the
suffix, and leaves sealed prompt blocks reusable after the request. The
predicted token is returned but never fed back, so it has no KV entry.

PR1 does not claim a native prefill kernel or prefill throughput. Its
compatibility executor still advances prompt spans through the existing
small-row whole-step implementation, but executes those steps eagerly:
prefill-only startup neither captures nor replays the decode graphs. The
native large-M executor replaces that implementation later and remains eager.

## Capacity contract

`--glm52-prefill-chunk-size` defaults to 16,384 tokens and must be a positive
multiple of the 64-token KV page size. It is a compute tile, not a prompt
length limit. The future executor may process a longer prompt as several
chunks while preserving the same cache block chain.

When `--max-model-len` is omitted, startup keeps the existing fleet-minimum
VRAM probe and exact KV-arena ledger, then also reserves a conservative
large-M scratch envelope:

```text
256 MiB fixed + 72 KiB × prefill_chunk_size
```

The default 16K chunk therefore reserves 1,476,395,008 bytes per rank before
the binary search derives `max_model_len`. An explicit `--max-model-len`
continues to be a checked cap: it must fit after the same reservation, and the
chunk itself must not exceed it. The startup log prints the selected chunk and
reservation beside the existing arena/reserve budget.

This is a reservation, not a hidden allocation. The PR that introduces native
large-M scratch must fit inside this ledger or update the formula and repeat
the capacity gates.

## Gates

- CLI accepts only the TP4/prefix-cache-compatible combination.
- Non-page-aligned chunks and inert chunk flags are rejected.
- The HTTP boundary returns `400 invalid_request_error` for any
  `max_tokens != 1`; scheduler intake repeats the check for non-HTTP callers.
- The slot state finishes with `Length` on the prompt-tail output.
- Prefill-only admission keeps at most one active request in the TP group.
- The coordinator fails closed if an active prefill-only request ever reaches
  a decode state.
- Scratch reservation lowers auto-derived capacity and is covered by the same
  pure, CUDA-free budget tests as the existing `max_model_len` policy.

## Next

Replace the compatibility executor in independent steps: large-M TP4
projection/MoE, tiled causal DSA top-k, then eager causal sparse-MLA prefill.
The service contract, prefix-cache ownership, and capacity policy stay fixed
while those kernels land.
