# openinfer-comm

Skeleton comm-backend surface for **OpenInfer**: a narrow, hardware-free
trait that OpenInfer's request scheduler will use to drive cross-rank
data movement (EP all-to-all first; future data-movement surfaces later).

**Status: skeleton.** This crate currently exposes the *shape* of the
initial public API. The default-feature build compiles and the type
surface is reviewable, but there is **no usable backend yet** — see
[Status](#status) below. The crate is in the workspace so its contract
can be reviewed before the hardware adapter is wired in.

The crate is structured so that the default-feature build does not
require any hardware-class system header (CUDA SDK, GDRCopy, RDMA
Verbs). This lets OpenInfer's main CI lane `cargo check -p openinfer-comm`
on a barebones development machine. Hardware backends live behind
feature flags and only compile in when the matching feature is on.

## Status

This is a **skeleton PR**. Concretely:

- The default-feature build compiles and the public types exist.
- `EpBackendBuilder::build` is **fail-closed in both feature modes**:
  - default-off: returns `Error::BackendUnavailable` (no hardware
    backend feature active).
  - `hw-rdma`: returns `Error::Unimplemented` (the `RdmaBackend`
    adapter exists as a private type but its wiring is not landed
    yet).
- As a result, no caller can obtain an `EpBackend` whose trait methods
  would panic. The `EpAllToAll` trait methods on `RdmaBackend` are
  `todo!()` placeholders that are not reachable through the public
  builder.
- Trait method signatures, plan / handle / buffer field shape, and
  `EpTopology` field set are marked `#[non_exhaustive]` and may evolve
  in follow-up PRs. Treat this crate as a contract *shape under
  review*, not as a frozen API.

The wiring PR that turns this into a working backend will:

1. Remove the `Error::Unimplemented` branch in `EpBackendBuilder::build`
   and replace it with real `RdmaBackend` construction.
2. Replace the `todo!()` bodies on `RdmaBackend`'s `EpAllToAll` impl
   with translation onto the wrapper-crate API.
3. Add an integration test that exercises the public trait surface.

## Public type surface (skeleton shape)

| Type                                  | Purpose                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| `EpAllToAll` (trait, object-safe)     | Per-call dispatch / combine / poll / release entry points.   |
| `EpBackend`, `EpBackendBuilder`       | Builder for the future active backend (fail-closed today).   |
| `EpTopology`                          | World size + rank + expert/dim/token sizing.                 |
| `DispatchPlan`, `CombinePlan`         | Per-call routing descriptors.                                |
| `SendBuf<'a>`, `RecvBuf<'a>`          | Opaque views over caller-owned device buffers.               |
| `DispatchHandle`, `CombineHandle`     | In-flight op tokens; finalized through `EpAllToAll::poll`.   |
| `AnyHandle`, `Poll`                   | Polling surface.                                             |
| `Error`, `Result`                     | Public error type; backend errors erased via `Box<dyn ...>`. |

All non-exhaustive types are marked `#[non_exhaustive]`; adding fields
is a non-breaking change. Method signatures and field sets are
explicitly subject to revision in the wiring PR.

## Feature flags

- `default = []` — pure-Rust surface. Pulls only `thiserror`. No
  wrapper-crate dependency, no `*-sys` link probe, no CUDA / GDRCopy /
  Verbs header lookup.
- `hw-rdma` — compiles the `crate::backend::rdma` module. Enabling
  `hw-rdma` **transitively activates the CUDA subsystem** on the
  underlying wrapper crates (`cuda-lib`, `torch-lib`, `a2a-kernels`)
  because the EP all-to-all path needs both CUDA kernels and RDMA
  Verbs. There is no `hw-rdma`-only path that omits CUDA. In this
  skeleton PR, even with `hw-rdma` on, `EpBackendBuilder::build` still
  returns `Error::Unimplemented`; the feature exists so the wrapper
  crates' build chain is exercised, not so a usable backend is
  produced.

The `HW_RDMA_ENABLED` constant is a diagnostic / build-system signal,
not part of the stable API. Code that needs to react to backend
availability should call `EpBackendBuilder::build` and dispatch on the
returned `Result`.

## Public-surface invariants for this skeleton

These are the properties the skeleton aims to preserve as the contract
matures. They are written in to constrain follow-up PRs:

1. **No wrapper-crate types in the default surface.** The `EpAllToAll`
   trait and the plan / handle / buffer / error / builder types must
   not reference any type from a wrapper crate (`p2p-all-to-all`,
   `fabric-lib`, `cuda-lib`, `torch-lib`, `a2a-kernels`, `cuda-sys`,
   `cudart-sys`, `gdrapi-sys`, `libibverbs-sys`). Backend-specific
   errors are erased through `Error::Backend { source: Box<dyn
   std::error::Error + Send + Sync> }`.
2. **Backends are not re-exported.** Implementation modules live under
   `crate::backend::*` and are only compiled when the matching feature
   is on. They must not be re-exported through this crate's public
   namespace; the only way to obtain a backend is through
   `EpBackendBuilder::build`.
3. **Diagnostic markers are not stable API.** `HW_RDMA_ENABLED` and any
   similar `*_ENABLED` constants exist for runtime / build-system
   introspection only.
4. **Fail-closed before wiring.** While the public surface is in
   skeleton form, `EpBackendBuilder::build` returns `Err` in all
   feature modes; no caller can obtain a backend whose trait methods
   would panic.

## Wrapper crates are *not* OpenInfer's public API

This repository contains several upstream-derived wrapper crates
(`p2p-all-to-all`, `fabric-lib`, `cuda-lib`, `torch-lib`, `a2a-kernels`,
`python-ext`, plus their `*-sys` siblings). They are hardware
implementation packages reached only through `openinfer-comm`
adapters. Their names, types, and feature flags are **not** part of
OpenInfer's API contract and may evolve as the upstream and adapter
layers change.

OpenInfer code (and any code outside this crate) should depend on
`openinfer-comm` only. Direct use of wrapper-crate types from outside
this crate's `backend::*` adapter modules is unsupported.

## Usage sketch

Today, every call to `build` returns an error. The fail-closed result is
intentional — callers must dispatch on the `Result`. **Do not call
`build().unwrap()`** in this PR or in any caller during the skeleton
window:

```rust
use openinfer_comm::{EpBackendBuilder, EpTopology};

// `EpTopology` is `#[non_exhaustive]`; outside this crate the constructor
// is the only stable way to obtain one.
let topology = EpTopology::new(
    /* world_size     */ 1,
    /* rank           */ 0,
    /* num_experts    */ 0,
    /* hidden_dim     */ 0,
    /* max_num_tokens */ 0,
);

match EpBackendBuilder::new().topology(topology).build() {
    Ok(_backend) => {
        // Reached once the wiring PR lands. Currently unreachable;
        // do NOT unwrap build() here — the fail-closed Err is intended.
    }
    Err(e) => {
        // BackendUnavailable on default-off, Unimplemented on hw-rdma
        // until the wiring PR lands.
        eprintln!("EP backend not yet available: {e}");
    }
}
```

## License & provenance

See the top-level `LICENSE` and `NOTICE.md`. The hardware backend is
being adapted from upstream `pplx-garden`; this crate adds the
OpenInfer-facing public surface skeleton and the feature-gating that
keeps the default build hardware-free.
