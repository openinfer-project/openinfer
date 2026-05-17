# DeepSeek V4 Prefix/Paged KV and P-D Handoff Design Contract

**Created**: 2026-05-14
**Status**: design contract
**Canonical task**: task #19

## Purpose

This document defines how the current DeepSeek V4 direct KV ownership model
should evolve toward prefix cache, paged KV allocation, and prefill/decode
handoff. It is a design contract, not a runtime implementation.

The goal is to keep ownership visible at the KV boundary while the runtime moves
from fixed per-request slots to reusable page/block ownership. Scheduler logic
should consume explicit handles and telemetry; it must not own page internals or
transport details.

## Evidence Layers

`code-fact`: facts observed in the current repository at or after PR #108.

`user-input`: task #19 asks for one executable design contract covering current
slot lease evolution, prefix/paged KV allocator interfaces, and P-D handoff
handle ownership/cancellation/cleanup semantics.

`derivation`: interface and rollout proposals inferred from the code facts and
task #19 target. These are not current runtime behavior until a later PR lands
them.

## Current Code Facts

### Direct KV Ownership

`code-fact`: `pegainfer-deepseek-v4/src/direct/scheduler.rs` owns
`DirectKvCacheManager` and `DirectKvCacheLease`.

Current lifecycle:

1. `reserve` / `reserve_in_slot` records one active request lease.
2. `ensure_direct_decode_caches` or `ensure_direct_decode_batch_caches` allocates
   rank-local decode cache capacity.
3. `attach_prepared` marks the lease attached and increments reset/allocation or
   reuse telemetry.
4. Request state carries `Option<DirectKvCacheLease>`.
5. `release_greedy_request` releases the matching lease and clears logits.
6. Error paths in single-step and batch-step decode release active leases before
   returning errors.

The manager exposes counters through `DirectKvCacheSnapshot`: capacity,
allocated length, request slots, active count, active lease summary, reservation
/ release / rejection / allocation / reset / reuse counters, and the last reject
reason.

### Slot Layout

`code-fact`: `LayerDecodeCache::zeros_with_max_seq_and_slots` builds one fixed
cache layout per layer and per request slot.

For each request slot:

- base KV storage is `sliding_window + compressed_slots` rows;
- compressed layers add compressor state;
- ratio-4 layers add indexer KV and indexer compressor state;
- runtime batch metadata passes `slot_id`, `window_base`, `compressed_base`, and
  `compressed_len` to decode kernels.

This slot model is the bridge from single-request direct cache to block/page
ownership. It already proves that row ownership must be explicit per request; it
is not enough for the scheduler to interleave requests without per-request cache
identity.

### Scheduler Boundary

`code-fact`: PR #107 added scheduler wave batching for up to two ready requests,
with slot-owned leases and independent release. It still uses fixed slots and a
small batch capacity; it does not implement paged KV, prefix reuse, eviction,
P-D handoff, or transport-level handles.

### Communication Boundary

`code-fact`: `pegainfer-comm` currently provides EP all-to-all public surface and
opaque operation handles. It does not yet provide KV transfer or ownership
handoff primitives.

## Capability Target

The P5 target is a design that lets the engine answer these questions before
runtime code is written:

- Which object owns each KV page/block at every point in the request lifecycle?
- Which pages are reusable prefix pages, request-private pages, pinned for
decode, or in transfer for P-D handoff?
- How does allocation fail closed when capacity is insufficient?
- Which telemetry proves capacity, reuse, eviction, and cleanup behavior?
- What handle can a prefill worker export, what handle can a decode worker
import, and who cancels or cleans up in-flight handoff work?

## Ownership Model

`derivation`: replace the direct slot lease with a model-family-neutral KV lease
that owns logical token ranges while hiding physical layout details from the
scheduler.

```rust
pub struct KvLease {
    request_id: RequestId,
    epoch: u64,
    model_layout: KvLayoutId,
    logical_tokens: Range<u32>,
    request_pages: KvPageSet,
    prefix_pins: Vec<PrefixPin>,
    transfer_pins: Vec<KvTransferPin>,
    state: KvLeaseState,
}

pub enum KvLeaseState {
    Reserved,
    Attached,
    SealedPrefix,
    DecodePinned,
    Exporting,
    ImportStaging,
    Imported,
    Released,
    Cancelled,
}
```

Required invariants:

- A page has exactly one owner or is on exactly one free list.
- A `KvLease` separates page classes by cleanup semantics:
  `request_pages` are mutable request-owned pages, `prefix_pins` are borrowed
  read-only prefix pages, and `transfer_pins` are pages held by an export/import
  operation.
- `request_pages` are freed or sealed by request cleanup.
- `prefix_pins` are unpinned by request cleanup and never freed directly by the
  request.
- `transfer_pins` are resolved by transfer cleanup; request cleanup can cancel
  them but cannot free the underlying pages directly.
- Borrowed prefix pages are read-only for the borrower.
- Mutable decode pages cannot be shared across requests.
- Page ownership changes only through allocator methods that update telemetry.
- A lease release must be idempotent from the request cleanup path, but stale
  handles must still be rejected by epoch.

This split is required because a single `KvPageSet` cannot tell whether cleanup
should free pages, unpin a prefix entry, or wait for transfer cleanup. Any
implementation that collapses these resource classes back into one collection
reintroduces ambiguous ownership.

## Page Layout Contract

`derivation`: physical pages should be allocated per rank and per model layout,
not per scheduler request.

A `KvLayoutId` must encode compatibility facts that make pages safe to reuse:

- model family and revision;
- tensor-parallel / rank layout;
- layer count and per-layer compression ratios;
- head dimension, sliding window, indexer dimensions;
- dtype and cache format version;
- RoPE / position semantics relevant to the cache.

A `KvPage` is a logical token block plus model-layout metadata. The allocator is
responsible for mapping that block into the per-layer storage required by DSV4,
including sliding-window rows, compressed rows, compressor state, indexer KV, and
indexer compressor state. The scheduler must not compute `window_base` or
`compressed_base` directly.

```rust
pub struct KvPage {
    id: KvPageId,
    layout: KvLayoutId,
    token_range: Range<u32>,
    owner: KvPageOwner,
    residency: KvResidency,
}

pub enum KvPageOwner {
    Free,
    Request(RequestId),
    Prefix(PrefixEntryId),
    Transfer(KvTransferId),
}
```

Lease-side references are typed by cleanup owner:

```rust
pub struct PrefixPin {
    entry: PrefixEntryId,
    pages: KvPageSet,
    epoch: u64,
}

pub struct KvTransferPin {
    transfer: KvTransferId,
    pages: KvPageSet,
    direction: KvTransferDirection,
    epoch: u64,
}

pub enum KvTransferDirection {
    Export,
    Import,
}
```

The allocator owns the mapping from these logical sets to physical per-layer
storage. A scheduler or request state can carry these handles, but it cannot
interpret them as rows or free them by hand.

## Prefix Cache Contract

`derivation`: prefix cache entries are sealed page sets keyed by exact token and
layout identity. They are not a scheduler shortcut and not a text-level cache.

Minimal key:

```rust
pub struct PrefixKey {
    model_layout: KvLayoutId,
    token_hash: [u8; 32],
    token_len: u32,
    rank: u32,
}
```

Minimal lifecycle:

1. `lookup(PrefixKey)` returns a sealed entry or a miss.
2. `pin(entry)` increments a refcount and returns borrowed read-only pages.
3. `reserve_suffix(request, prefix, suffix_len)` allocates mutable pages for the
   request's suffix/decode tokens.
4. `seal(request_range)` may publish pages into a new prefix entry after prefill
   completion.
5. `unpin(entry)` decrements refcount and makes the entry evictable when zero.
6. `evict(entry)` can only run when refcount is zero and no transfer owns the
   pages.

Reject conditions must be explicit:

- layout mismatch;
- prefix entry not sealed;
- prefix entry currently evicting;
- capacity insufficient for suffix reservation;
- request asks for unsupported mixed prefix/source state.

## Paged Allocator Contract

`derivation`: allocator API should be small and stateful. It should expose
reservation, attachment, release, prefix pinning, and snapshots, but not kernel
row math.

```rust
pub trait KvPageAllocator {
    fn reserve_request(&mut self, req: KvReserveRequest) -> Result<KvLease>;
    fn attach_prefill(&mut self, lease: &KvLease) -> Result<KvAttachToken>;
    fn seal_prefix(&mut self, lease: &mut KvLease, key: PrefixKey) -> Result<PrefixEntryId>;
    fn pin_prefix(&mut self, key: &PrefixKey) -> Result<PrefixPin>;
    fn export_request_pages(&mut self, lease: &mut KvLease) -> Result<KvExportHandle>;
    fn release_request(&mut self, lease: &mut KvLease) -> Result<()>;
    fn cancel_request(&mut self, lease: &mut KvLease, reason: CancelReason) -> Result<()>;
    fn snapshot(&self) -> KvAllocatorSnapshot;
}
```

Allocation must be fail-closed:

- reserve all pages required by the next stage before mutating request state;
- on partial failure, roll back newly reserved pages;
- reject capacity pressure before overwriting active or pinned pages;
- never evict pages that are pinned by decode or in a transfer state.

The first runtime feature PR should keep DSV4 physical storage behind an adapter
that can still emit today's `DecodeBatchMeta`. The scheduler should receive a
`KvLease`/`KvDecodeView`, not raw slot ids.

### Cleanup Semantics By Operation

`derivation`: the allocator methods must define cleanup by resource class.

`release_request(lease)`:

- frees `lease.request_pages` unless they were already sealed into a prefix or
  moved into a transfer pin;
- unpins every `lease.prefix_pins` entry and decrements prefix refcounts;
- rejects stale transfer pins by epoch, cancels pending transfer pins owned by
  this request, and leaves completed transfer cleanup to the transfer manager;
- clears the lease state to `Released` only after all three resource classes are
  empty or delegated to their cleanup owner.

`cancel_request(lease, reason)`:

- performs the same resource-class cleanup as `release_request`;
- records cancellation counters and reason-specific telemetry;
- may mark private pages poisoned before returning them to the free list if a
  kernel or transfer may have partially written them;
- must be safe to call after partial admission, partial prefill, or batch decode
  failure.

`seal_prefix(lease, key)`:

- can only move a contiguous, immutable subset of `lease.request_pages` into a
  sealed prefix entry;
- removes those pages from `lease.request_pages` and records the resulting
  `PrefixEntryId`;
- does not consume `lease.prefix_pins`; borrowed prefix pages stay borrowed and
  are unpinned by request cleanup;
- fails if any page in the sealed range is mutable, transferring, poisoned, or
  layout-incompatible.

`export_request_pages(lease)`:

- moves the exported subset from `lease.request_pages` into a `KvTransferPin`;
- records `KvPageOwner::Transfer(id)` for those pages until the export resolves;
- prevents `release_request` from freeing exported pages directly while the
  transfer is pending;
- on export cancellation, returns pages to `request_pages` or releases them via
  transfer cleanup, with exactly one owner recorded in telemetry.

## Eviction And Reject Policy

`derivation`: eviction should be explicit and observable before it becomes
performance-sensitive.

Minimum policy for the first implementation:

- no eviction of request-private decode pages;
- no eviction of pinned prefix pages;
- no eviction of pages in `Exporting` / `Imported` transfer states;
- LRU eviction among unpinned sealed prefix entries is acceptable;
- capacity reject is preferred over implicit reset when the safe victim set is
  empty.

Reject reasons should be stable enum values, not free-form strings:

```rust
pub enum KvRejectReason {
    CapacityExceeded,
    NoEvictablePrefix,
    LayoutMismatch,
    PrefixNotSealed,
    PrefixEvicting,
    TransferInFlight,
    StaleHandle,
    UnsupportedMode,
}
```

## Telemetry Contract

`derivation`: telemetry should let review answer ownership and capacity questions
without reading scheduler internals.

Minimum snapshot fields:

- total bytes/pages by state: free, reserved, attached, prefix, pinned,
  transferring, evicting;
- active request count and active lease count;
- prefix entries: sealed, pinned, evictable, evicting;
- reserve/release/cancel/evict counters;
- prefix hit/miss counters;
- transfer export/import/cancel/fail counters;
- last reject reason with requested tokens/pages and available pages;
- high-water marks for pages and bytes.

Telemetry names should stay allocator-centric. Do not encode scheduler policy in
field names.

## P-D Handoff Handle Contract

`derivation`: P-D handoff requires ownership handles, not transport objects. The
handle describes who owns cleanup and which observable signal proves transfer
completion or cancellation. It does not choose RDMA, IPC, serialization, or a
specific `pegainfer-comm` operation.

### Export Side

A prefill worker can export only sealed or explicitly transfer-pinned pages.
Export moves the lease pages into a transfer-owned state until completion,
cancellation, or timeout.

```rust
pub struct KvExportHandle {
    id: KvTransferId,
    source_request: RequestId,
    layout: KvLayoutId,
    pages: KvPageSet,
    epoch: u64,
}

pub enum KvExportPoll {
    Pending,
    Ready(KvTransferDescriptor),
    Cancelled,
    Failed(KvTransferError),
}
```

Export invariants:

- source request cannot release exported pages while export is pending;
- cancellation returns pages to source ownership or marks them released, never
  both;
- stale export handles are rejected by epoch;
- export completion produces a descriptor, not decode ownership;
- `KvTransferDescriptor` is single-consume: exactly one import attempt can claim
  it;
- descriptor lifetime must have an explicit TTL or equivalent expiry signal;
- timeout is reported as `Failed(KvTransferError::Timeout)`, not a separate
  terminal state.

### Import Side

A decode worker imports a transfer descriptor and receives a decode-owned lease
only after import completion is observed.

```rust
pub struct KvImportHandle {
    id: KvTransferId,
    target_request: RequestId,
    layout: KvLayoutId,
    epoch: u64,
}

pub enum KvImportPoll {
    Pending,
    Ready(KvLease),
    Cancelled,
    Failed(KvTransferError),
}
```

Import invariants:

- imported pages are not visible to decode until import reports `Ready`;
- target-side staging pages are represented by `ImportStaging` and are cleaned
  by the target allocator until the import becomes `Ready`;
- cancellation before `Ready` cleans local staging pages and releases the
  transfer claim;
- cancellation after `Ready` is ordinary request lease cleanup;
- `Failed` and `Cancelled` have the same cleanup obligation for target staging
  pages; `Failed` additionally records the failure reason and any poisoned pages
  before returning them to the allocator;
- the allocator must record whether cleanup is owned by source, target, or
  transfer manager;
- failed imports do not retry through the same descriptor. A retry requires a new
  export descriptor or an explicit source-side re-export.

### Owner And Observable Signal Checklist

For every P-D handoff operation the design must identify:

| Phase | Owner | Observable signal | Cleanup owner |
| --- | --- | --- | --- |
| Export reserved | source allocator | export handle created | source allocator |
| Export in flight | transfer manager | `KvExportPoll::Pending` | transfer manager |
| Export ready | transfer manager | `KvExportPoll::Ready` descriptor | source or transfer manager per descriptor lifetime |
| Source cancel during export pending | transfer manager | `KvExportPoll::Cancelled` or `Failed(Timeout)` | source allocator after terminal poll |
| Descriptor abandoned | transfer manager | descriptor expiry / TTL elapsed | source allocator |
| Import in flight | target allocator + transfer manager | `KvImportPoll::Pending` | target allocator |
| Import ready | target allocator | `KvImportPoll::Ready(KvLease)` | target request lease |
| Cancel before ready | transfer manager | `Cancelled` | phase owner named above |
| Fail before ready | transfer manager | `Failed(KvTransferError)` | phase owner named above; poisoned staging pages recorded first |
| Request drop after ready | target request | release/cancel lease | target allocator |

This checklist is the only P-D transport-facing content in this document. The
transport implementation can later map these signals to RDMA, CUDA IPC, or a
serialized path, but scheduler code should only see allocator handles and poll
results.

## Scheduler Boundary

`derivation`: scheduler admission should depend on allocator decisions, not on
cache layout.

Allowed scheduler knowledge:

- request has prefix hit/miss;
- reservation accepted/rejected with stable reason;
- decode view is ready for this request;
- transfer handle is pending/ready/failed/cancelled;
- release/cancel succeeded or failed.

Disallowed scheduler knowledge:

- physical page row offsets;
- DSV4 compression/indexer row math;
- transport registration or remote-write state;
- eviction victim selection internals;
- prefix entry mutation outside allocator methods.

## First Feature PR Boundary

`derivation`: the next implementation PR should be one coherent step, not a full
P-D runtime.

Recommended first feature PR:

**`feat(dsv4): introduce paged KV allocator contracts and telemetry`**

In scope:

- add DSV4-local `KvLayoutId`, `KvPageId`, `KvLease`, `KvPageAllocator`, and
  snapshot/reject types;
- implement a single-node fixed-size page allocator behind the current direct
  runtime, initially adapting pages back into today's slot-compatible
  `DecodeBatchMeta`;
- preserve bs=1 and bs=2 exact/hash/HTTP benchmark gates;
- expose allocator snapshot in a debug/test-only path;
- include tests for reserve/attach/release/cancel, prefix pin/unpin, fail-closed
  capacity reject, and stale-handle reject.

Out of scope:

- cross-node or cross-process P-D transport;
- prefix eviction performance tuning;
- production prefix cache policy;
- changing HTTP benchmark semantics;
- replacing `pegainfer-comm` or adding KV transfer to it.

Merge criteria:

- no scheduler code computes physical KV row offsets;
- every request-owned page has exactly one owner until release;
- capacity pressure rejects or evicts only unpinned sealed prefixes;
- telemetry can prove active, free, pinned, evicting, and rejected states;
- current direct single-request and scheduler bs=2 behavior do not drift.

## Open Follow-Ups

- Choose page size after measuring DSV4 layer/cache memory pressure. This design
  deliberately does not commit to a value.
- Decide whether prefix entries are rank-local only in v1 or require a
  multi-rank consistency object.
- Define a future `pegainfer-comm` KV-transfer extension only after allocator
  handles and cleanup semantics are proven locally.
