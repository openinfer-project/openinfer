# Gemma 4 KV cache contract

**TL;DR:** Gemma 4 caches at two head dims — 256 for sliding layers, 512 for full-attention layers —
which the single-geometry `KvCacheManager` facade cannot hold, though heterogeneous buffers over one
pool is an existing pattern here. Two attention backends must exist before the model serves
correctly: a sliding-window instantiation, entered as `sliding_window - 1` because the two sides
count the window differently, and paged attention at head_dim 512. After those, one pool is correct
as long as nothing is reclaimed; reclaiming sliding pages needs two pools, because a request holds
one assignment list and a block is held or released for both groups at once. Reclamation also splits
`positions` into an absolute RoPE position and a view-relative append slot. `attention_k_eq_v` means
full-attention layers ship no `v_proj`; K and V still diverge after the shared projection and both
must be cached.

Last touched: 2026-07

## Two geometries, two lifetimes

`KvLayout` applies one `num_kv_heads` and one `head_dim` to every layer, and `KvCacheManager` pairs
one such buffer with one `BlockPool` (`openinfer-kv-cache`), so Gemma 4 cannot use that facade.
Owning buffers outside it is routine: `BlockPool` holds no GPU memory, and glm52's `kv_arenas`
registers one arena width per layer plus a narrower one on a layer subset. Copy that shape. What is
new is that the two groups want different lifetimes — the sliding group to release pages the
full-attention group still reads.

Per-token cost is `layers × kv_heads × head_dim × 2 (K and V) × 2 (bf16)`, from the published
configs:

| | 12B | 26B-A4B | 31B |
| --- | --- | --- | --- |
| layers, sliding / full | 40 / 8 | 25 / 5 | 50 / 10 |
| sliding, KV heads × head_dim | 8 × 256 | 8 × 256 | 16 × 256 |
| full, KV heads × head_dim | 1 × 512 | 2 × 512 | 4 × 512 |
| sliding, per token | 320 KiB | 200 KiB | 800 KiB |
| full, per token | 16 KiB | 20 KiB | 80 KiB |
| sliding, one 1024-token window | 320 MiB | 200 MiB | 800 MiB |
| full, at 262144 positions | 4 GiB | 5 GiB | 20 GiB |

The sliding group is 20× the full group per token at 12B. That ratio, not the head dims, drives every
sizing decision here. Full-attention layers sit at indices 5, 11, 17, …, which `probe_layer_types`
in `openinfer-gemma4` expresses as `i % 6 == 5 || i == last_index`; all three published layer counts
are divisible by 6, so the last-layer clause never fires on them.

## Two attention backends are prerequisites

**A sliding-window instantiation.** Sliding layers must not attend past 1024 positions, so this is
required for correct output above the window regardless of memory. The shared paged-attention
translation unit in `openinfer-kernels` compiles its attention variant with the window disabled and
passes `window_left = -1` at every call site. FlashInfer supports the window, so what Gemma 4 needs
is an additional template instantiation beside the existing variant, not an edit to it — instantiating
alongside leaves other model lines' codegen untouched.

The two sides count the window differently, so the conversion is part of this contract:
**`flashinfer_window_left = config.sliding_window - 1`**, i.e. 1023. Gemma's mask keeps
`kv_idx > q_idx - sliding_window`, so 1024 means the current token plus 1023 predecessors, 1024 in
total; FlashInfer's `window_left` is an inclusive distance, and its own
`tests/attention/test_sliding_window.py` asserts that a run with `window_left = W` equals one over
the last `W + 1` KV entries. Passing the config value straight through attends to one token too
many. The backend must distinguish a 1024-token
window from a 1025-token one, and that distinction needs its own gate — an off-by-one here changes
output without failing anything.

**Paged attention at head_dim 512.** `head_dim` is a runtime stride argument, but each kernel is
instantiated at a compile-time constant, and the shared TU's families are built for other widths
with runtime rejects on the paths never widened. Gemma 4 requires an HD512 paged-attention backend
covering single prefill, paged batch prefill and paged batch decode; until one exists the
full-attention group's pages are unreadable, which makes it a correctness prerequisite of the same
rank as the mask.

One property of that backend is load-bearing here. FlashInfer's decode dispatcher instantiates a
fixed set of GQA groups, and at TP1 the 12B full-attention group is 16 query heads over a single KV
head, outside that set, so it cannot use the decode kernel while the sliding group can — the two
groups take different decode paths, and anything assuming one decode entry point per model is wrong.
That ratio is **per rank, not per config**: query heads shard with the world size
(`Config::local_num_attention_heads`) while a single KV head can only be replicated, so the same
group is 16 at TP1, 8 at TP2, 4 at TP4. Backend selection, GQA validation and graph capture must all
key off the resolved per-rank mapping.

## Capacity: what reclaiming the window buys

All figures below are per rank at TP1; see the tensor-parallelism section for how they scale.

With both backends and no reclamation, pages stay resident and the sliding group costs its full
320 KiB per token. That is correct, and budget-limited rather than window-limited: one 12B request
costs 336 KiB per token across both groups, so a 20 GiB budget reaches roughly 62k positions for one
request, or 7.8k each at eight concurrent. The declared context is what it cannot reach — at 262144
positions one unreclaimed request wants **84 GiB** (55 GiB at 26B, 220 GiB at 31B), while capping the
sliding group at its residency drops it to about 4.6 GiB, so the same 20 GiB serves four. The trade
is "cannot serve one at the declared maximum" versus "serves four", not "1024 versus 262144": a
mask-only engine is shippable with a budget-derived context ceiling.

**Sliding residency is a window plus a shared prefill burst.** Prefill is append-then-attend, so
while a request forwards a chunk of `C` tokens the span it needs resident is `C + window`, not
`window`. But `max_prefill_tokens` caps the **step's total** forwarded tokens across all requests,
not each request's — `take_prefill_chunks` in `openinfer-qwen3` spends one budget down across the
set. So the burst is bought once for the whole step, and the pool decomposes:

```
sliding pages = concurrency × (ceil(window / page_size) + 1)   steady residency, per request
              + ceil(max_prefill_tokens / page_size)           in-flight burst, shared by the step
              + (concurrency - 1)                             per-row chunk rounding
```

The third term is not optional: rounding the *sum* of the chunks to pages is not the same as summing
each row's rounded chunk, because every active row rounds its own frontier. With page size 16, a
1024 window and eight rows taking `[1003, 3 × 7]` — exactly the 1024 budget — the rows need
`[128, 66 × 7] = 590` pages while the first two terms give 584. Six pages, 30 MiB at 12B, and the
failure mode is admission accepting a request the second pool then cannot schedule. Summing each
row's real offset and chunk is exact; `(concurrency - 1)` is the cheap upper bound, and it uses
concurrency because sizing happens before the live row count is known.

At 12B, page size 16, `max_prefill_tokens` 1024: 129 pages for one request and 591 for eight
concurrent, against 1032 if the burst were charged per request — the shared form is still 1.75×
tighter. Reserving per request is a legitimate simplification to declare, not the capacity the design
requires. `max_prefill_tokens` enters the sum once rather than `concurrency` times, and still has to
be bounded.

Reclamation granularity is one page, which does **not** require the page size to divide the window:
the window's left edge lands mid-page at almost every position regardless — at 5000 it is 3977. The
mechanism is to retain the frontier page whole and let the mask exclude the expired tokens inside it,
which is what `window_left` does and what FlashInfer's page-count arithmetic assumes. The cost is at
most `page_size - 1` over-retained tokens per request, which the `+ 1` above carries.

## What reclamation breaks

**`positions` serves two meanings that reclamation splits.** One array goes both to the fused
norm+RoPE kernel, where it is documented as the single source of truth for each token's absolute
position, and to the KV append, where `AppendPagedKVCacheKernel` computes the destination slot as
`indptr[b] × page_size + positions[i]` — view-relative. They coincide only while every page row
starts at position 0. Once the sliding row is a window-length suffix, a request at position 5000
indexes page 312 of a 65-entry slice: an out-of-bounds **write**, through the append path's
unprotected `get_k_ptr` rather than the read path's `protective_get_k_ptr`. The sliding group needs a
second, cache-relative slot array; RoPE keeps the absolute one. Relatedly, `paged_kv_t` has no
first-page offset, so whole-page front eviction is representable and token-granular is not.

**Page slot index is absolute position** in every list the pool builds:
`RequestKv::current_page_indices` and `RequestKv::step_page_indices` both truncate the assignment
list from the *end*. Dropping from the front is not expressible.

**A request holds one assignment list.** Its lifecycle is `RequestKv::release` plus
`BlockPool::evict_inactive` for blocks no live request holds; the only partial drop kvbm exposes is
`SchedulableSequence::drop_unassigned`, LIFO and Idle-only. There is no per-geometry release.

What is **not** broken: `KvView::new`'s assertion is internally self-consistent and knows nothing
about position 0, so a truncated row plus its resident length passes; the read path needs no absolute
position either, since RoPE is baked into K before the page write. The view constructor does not
change and no other model line is affected.

Reclamation happens between forward passes, never inside one: a pass needs its whole `C + window`
span resident, and pages older than the next pass's span are released afterwards. But chunked prefill
and decode are separate apply paths, so it has two call sites, and they release different amounts —
up to a chunk's worth after a prefill step, one page every `page_size` tokens during decode. Wiring
only the decode path leaves long prompts holding everything.

## One pool or two

One pool whose block ids address a page in each buffer is correct **as long as nothing is
reclaimed** — the glm52 arrangement, and the right shape for a mask-only engine.

It cannot survive reclamation, and not because of a race. A request holds one assignment list. To
release the sliding group's oldest block the request drops it, and the block is gone for the
full-attention group too; to avoid that it keeps the block and reclaims nothing. "Held for one
geometry, released for the other" is not a state the pool can represent. Two pools are therefore the
precondition for independent reclamation, and adopting them earlier is a scheduling choice rather
than a correctness requirement for short contexts. The sizes want to diverge anyway: at 12B with page
size 16 a sliding page is 5 MiB against a full-attention page's 256 KiB, so one shared block count
has to be priced at the larger page.

Logical positions stay aligned across the two pools by two things only: the shared tokens-per-page,
and the two groups' `kv_position` being advanced together. Physical block ids and allocation history
are independent by design, and under reclamation they necessarily differ — the sliding pool's set
rotates while the full-attention pool's grows. Nothing here asks the two pools to stay in physical
lockstep, and an implementation that forces it has misread this.

What does have to be coordinated is the prefix decision. Registration cannot be disabled —
`BlockPool::build` always installs an LRU backend — only matching can, and two pools with independent
residency and independent LRU eviction can return different hit lengths for the same prompt, which
desynchronises the two `kv_position` values from the first step. So the match must be decided once
and imposed on both pools, or matching must be off for this line.

## K and V share a projection weight, not a cache slot

`attention_k_eq_v` is true at all three sizes and means something structural: full-attention layers
ship no `v_proj` of any form, and that single tensor is the entire difference between a sliding
layer's tensor set and a full layer's. Sliding layers do carry `v_proj`, so this concerns the
full-attention group only.

It is not a cache saving. The architecture forks after the shared projection — K takes `k_norm` then
RoPE, V takes the weightless V norm and no RoPE — so two distinct tensors are materialised per token
and both must be cached. That is architectural, not something to re-open with a measurement: a
numerical coincidence at some position would not establish function equivalence, and position 0 is
especially misleading because RoPE is the identity there.

Worth stating because the wrong version is one line away: `make_paged_kv` takes `k_offset_elems` and
`v_offset_elems` as separate caller-supplied offsets into the same buffer, over identical strides.
Passing the same offset twice aliases them, with no kernel change and no error — wrong output rather
than a crash. So the cache stores both and `KvLayout`'s `layer_stride = 2 × kv_block_len` stays.

## Scheduling and budget across two pools

Admission does not allocate — `admit_deferred_requests` decrements a scalar budget and blocks are
taken later by `RequestKv::schedule_prefill` / `schedule_decode` — so there is no partial grant to
roll back there, and where allocation does happen `RequestKv::revert_schedule` already returns the
step's blocks LIFO. What is new is that a step can succeed on one pool and fail on the other:
**schedule both or revert both; a request whose `apply` succeeded on one pool and failed on the other
is fatal**, because the two `kv_position` values can never be reconciled.

The reservation formula is what breaks. Admission reserves whole-lifetime capacity —
`RequestKv::lifetime_blocks` is `(input + max_output).div_ceil(block_size)` and `active_future_blocks`
models occupancy as monotonically growing. For a reclaiming sliding group occupancy is bounded and
non-monotonic, and lifetime reservation would demand 16384 pages for a request declaring the maximum
length — rejecting exactly the requests reclamation exists to serve. Its per-request reservation is
the steady cap alone, `min(lifetime_blocks, ceil(window / page_size) + 1)`. The prefill burst is
**not** part of it: it is one step-wide allowance the scheduler holds out of the pool, matching the
pool formula above. Folding it into each request's reservation re-charges it per request.

Pool sizing has two regimes. Without reclamation both pools need the same page count, and the sliding
one takes 95% of the bytes and sets the context ceiling. With reclamation the sliding pool follows
the three-term formula above and is independent of context — 591 pages, 2.89 GiB, at 12B with eight
concurrent and page size 16 — while the full-attention pool takes the remainder and sets the ceiling.
`servable_len` and the load feed's single `kv_cache_usage` ratio have to state which pool they mean.

## Tensor parallelism

Qwen3's `TensorParallelConfig::validate_for` refuses a world size that does not divide
`num_key_value_heads`, and shards by integer division. That policy is private to that crate and
Gemma 4 cannot reuse it, because the full-attention group has fewer KV heads than a plausible world
size: 1 at 12B, 2 at 26B, 4 at 31B.

Integer division alone is not a contract — it silently drops heads. With `P = world_size`, `Q`,
`Kv` and `G` the attention, sliding-KV and global-KV head counts, a legal `P` satisfies all three:

```
Q  % P == 0
Kv % P == 0
G  % P == 0   ||   P % G == 0
```

Without the third, 12B at `P = 6` yields `16/6 = 2` local query heads and `8/6 = 1` local sliding KV
head, covering 12 of 16 and 6 of 8 — a silent loss, not an error. The third condition splits the
full-attention group into two regimes, and the rank mapping differs by regime:

- **`G >= P`** — each rank owns a contiguous run of `G / P` KV heads. The group shards, so its bytes
  divide by `P` like the sliding group's.
- **`G < P`** — each KV head is replicated to `P / G` contiguous ranks, leaving exactly one head per
  rank. The group no longer divides: from here on it costs the same per rank at every world size.

**Every capacity figure in this document is per rank**, and the earlier tables are TP1. The floor
matters more than the replication: at 12B `G = 1`, so the full group is already at its floor at TP1
and never shrinks, which is why only the sliding half moves below.

| 12B, one request at 262144, per rank | TP1 | TP2 | TP4 |
| --- | --- | --- | --- |
| sliding / full attention, KiB per token | 320 / 16 | 160 / 16 | 80 / 16 |
| total | 84 GiB | 44 GiB | 24 GiB |

TP therefore buys less than a naive division suggests: the group that stops shrinking is the one that
scales with context. It cuts the other way for kernels — `local_q_heads` shrinks, so the per-rank GQA
group is 16 at TP1, 8 at TP2, 4 at TP4, and 12B's full-attention group lands back inside the compiled
decode set above TP1.

## CUDA graph capture

Decode is captured with pre-allocated buffers for pointer stability, and two groups impose two sets
of page-table contents with independent CSR offsets, two attention geometries (different head dims,
hence different plan tiling), and two base pointers with their own per-layer offsets. A shrinking
page row is compatible with replay: sequence length, page count and last-page length are derived
device-side, never host-baked.

The layout is not settled here, and one question precedes it. For GQA groups the decode kernel
cannot instantiate, `openinfer-qwen3`'s executor reroutes decode to the eager unified path and skips
capture; at TP1 12B's full-attention group is exactly such a group. Whether it is captured at all
therefore differs per TP degree, and that comes before workspaces or graph keys.

## RoPE precompute

The two groups need different caches — theta 10000 with full rotation at 256 for sliding layers,
theta 1000000 with proportional partial rotation for full-attention layers. `precompute_rope`
allocates a cos and a sin buffer of `max_seq_len × head_dim` bf16 elements each, where the same
`head_dim` sets both the buffer stride and the frequency denominator. That coupling is what makes
the full-attention cost depend on whether the signature is fixed first:

| cos + sin at 262144 positions | sliding | full attention |
| --- | --- | --- |
| as the signature stands (stride = head dim) | 256 MiB @ 256 | **512 MiB @ 512** |
| with stride and denominator separated | 256 MiB | 128 MiB @ rotary 128, denominator 512 |

Full attention rotates 128 dimensions but derives frequencies from the full 512, so on the current
signature passing 128 would buy the smaller buffer at the price of wrong frequencies. A correct
implementation on it therefore carries 768 MiB, not 384; separating the two parameters is the
proportional-RoPE work. These need no new accounting either way — the KV budget is measured after
weight load and these are allocated during it, so they are already subtracted — but a KV example
that assumes 384 MiB is off by that difference.

Their length is a configuration input, not something to derive from the budget: `servable_len` is
computed *from* the KV budget, which is measured *after* these caches are allocated, so "size them to
the servable context" is circular and resolving it would need two-phase allocation or a fixed point.
The contract is to size both from an explicit configured serving limit, capped at the checkpoint's
`max_position_embeddings`.

## What the loader must validate before building the layouts

A `KvLayout` pair is built from six numbers — per group, the layer count, the KV head count and the
head dim — taken from `layer_types` plus `num_key_value_heads`/`head_dim` for sliding and
`num_global_key_value_heads`/`global_head_dim` for full attention. The loader must reject, rather
than round or default, when any of these fails:

- `layer_types.len() == num_hidden_layers`, and the two group counts sum to it.
- `num_attention_heads` is divisible by both KV head counts — a non-integral GQA group has no kernel.
- Both head dims have a compiled attention backend, and the sliding group's has one with the window
  variant.
- The world size is legal by the three conditions above. This is separate from the GQA divisibility
  on the previous line, and its absence is what makes integer division drop heads silently.
- Each group's **per-rank** GQA group, after query-head sharding and any KV-head replication, has a
  compiled decode kernel or an accepted fallback. This is the resolved mapping, not the config ratio.

This belongs in the Gemma 4 config loader. Whether the detection probe should also pin these values
is a separate question about what the model line claims to support, not part of this contract; the
probe pins the shape fields but not the counts.

## Excluded

- **Prefix matching.** Registration cannot be disabled, only matching, and with two pools an unforced
  match can return different hit lengths per group. This model line must run with matching disabled
  until the two pools' hit length is unified.
- **KV offload and the block-event feed.** Both assume one block id space — `OffloadEngine`'s arena
  contract states it outright — which two pools break. Both must fail closed at startup for this
  model line.
- **Quantised KV.** The NVFP4 checkpoints declare an FP8 KV cache and this engine has no KV
  quantisation concept. An unsupported declared KV dtype must fail closed, naming the scheme, rather
  than silently serving bf16.
- **Cross-layer KV sharing.** `num_kv_shared_layers` is 0 at all three supported sizes.

## Next step

Order is forced. The two attention backends come first and are independent of everything here.
One-pool plumbing with no reclamation is next and yields a correct engine with a budget-derived
context ceiling. Two pools and reclamation are last, and reclamation needs three things, not one:

1. **The second `positions` array** — cache-relative for the append, absolute for RoPE.
2. **Resident origin and length as first-class request state.** `RequestKv::prefill_view` and
   `decode_view` derive `KvView`'s `seq_len` from the absolute `kv_position`, so a front-trimmed page
   row trips the exact-cover assertion on the next step even though the constructor itself is
   position-relative. Gemma 4's request state has to carry absolute position and resident
   origin/length separately and build its views from the latter.
3. **An ownership path for releasing a live request's oldest assigned blocks.** Nothing exposes one:
   the assignment list truncates from the end and the only partial drop is LIFO and Idle-only. This
   is the piece with no existing analogue, and it is what decides whether reclamation is confined to
   Gemma 4's own state or needs a shared-pool API.

Items 1 and 2 are local to this model line. Item 3 may not be.
