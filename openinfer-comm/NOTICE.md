# NOTICE

This project is a downstream port of Perplexity AI's `pplx-garden` Rust + CUDA
all-to-all communication library.

## Upstream

* Project: https://github.com/perplexityai/pplx-garden
* Upstream commit pinned for this port: `f84bc412ef651e5f1440d521f14577107082b087` (2025-12-24)
* License: MIT (Copyright (c) 2025 Perplexity AI). The original `LICENSE` file
  is preserved at the root of this tree.
* Research paper: ["RDMA Point-to-Point Communication for LLM Systems"](https://arxiv.org/abs/2510.27656)
  (Nandor Licker, Kevin Hu, Vladimir Zaytsev, Lequn Chen, 2025; arXiv:2510.27656)

## Scope of this port

This port narrows `pplx-garden` to a **Verbs-only** RDMA transport backend, to
be reused by OpenInfer for EP all-to-all on NVLink + InfiniBand / RoCE
machines. The libfabric / non-Verbs RDMA provider is removed in its entirety.

## Modifications applied during the port

Relative to the upstream commit listed above:

* **Removed components**
  * `fabric-lib/src/efa/` (entire EFA provider, ~2k lines of Rust + libfabric
    callouts)
  * `fabric-lib/libfabric-sys/` (bindgen crate for libfabric)
  * `fabric-lib/Cargo.toml`: `libfabric-sys` workspace dependency
  * `fabric-lib/src/lib.rs`: `mod efa;`
  * `fabric-lib/src/provider_dispatch.rs`: `DomainInfo::Efa(EfaDomainInfo)`
    variant + its match arms
  * `fabric-lib/src/topo.rs`: `EfaDomainInfo` → `PciAddress` conversion;
    EFA-first branch in `get_visible_domains`; EFA arm in
    `From<&DomainInfo>`
  * `fabric-lib/src/worker.rs`: EFA dispatch arms (1 / 2 / 4 domains per GPU);
    "Cannot mix EFA and Verbs" check
  * `fabric-lib/src/error.rs`: `FabricLibError::Libfabric` variant +
    `LibfabricError` struct + the `fi_strerror` callout
  * `docker/dev.Dockerfile`: AWS EFA installer block + `NCCL_SOCKET_IFNAME`
    `veth_def_agent` reference
  * `README.md`: AWS EFA support claims, the libfabric system-requirement
    line, and the upstream pplx-EFA vs pplx-CX7 performance table
  * `pyproject.toml`: `fabric: marks tests which require libfabric` marker
    text (updated to "requires an RDMA verbs device")

* **Renamed symbols** (semantic generalization — these names previously
  identified the non-NVLink token path, not specifically the EFA provider)
  * Rust: `num_recv_efa_tokens` → `num_recv_fabric_tokens`
    (`p2p-all-to-all/src/a2a_worker.rs`)
  * CUDA kernels: `num_efa_tokens` → `num_fabric_tokens`,
    `num_local_efa_tokens` → `num_local_fabric_tokens`
    (`p2p-all-to-all/a2a-kernels/src/a2a/a2a_dispatch_recv.cu`,
    `a2a_combine_send.cu`)
  * Comments in `a2a_dispatch_recv.cu`, `a2a_combine_recv.cu`,
    `a2a_worker.rs`: "EFA" → "fabric"
  * `provider_dispatch.rs`, `worker.rs`: short comments noting that the
    non-Verbs provider was removed during the port

* **Replaced**
  * `docker/dev.Dockerfile`: now installs `libibverbs-dev`, `rdma-core`, and
    `ibverbs-providers` from the distro instead of the AWS EFA installer

## Attribution policy

The upstream project name `pplx-garden` is preserved here for attribution
purposes. The upstream commit hash, license, and modification list above are
the authoritative attribution record for this vendored import; they must be
retained in any redistribution.

The vendored Rust workspace crates have been renamed under the
`openinfer-comm-*` prefix (for example `openinfer-comm-p2p-all-to-all`,
`openinfer-comm-fabric-lib`) so the workspace can coexist with OpenInfer's
existing crate namespace and so the public surface is OpenInfer-owned:

* **OpenInfer-owned in this tree** — the Rust workspace crate names, the
  top-level `openinfer-comm` crate, and the public OpenInfer-facing API it
  exposes (`EpAllToAll`, `EpBackend`, `EpBackendBuilder`, `EpTopology`,
  `DispatchPlan` / `CombinePlan`, `SendBuf` / `RecvBuf`, the handle and error
  types). These names are this tree's, not upstream's.
* **Preserved from upstream for attribution / interop where intentionally
  retained** — the Python distribution name `pplx_garden` (Python package /
  wheel name) and the upstream symbol-level identifiers reachable through it
  (`AllToAllContext`, `TransferEngine`, `TransferEngineBuilder`, ...). These
  are kept on the upstream-derived path so the vendored test suite and
  upstream documentation references stay recognisable.
* **Internal layout names** — internal module / file names inside the renamed
  Rust crates may continue to mirror upstream layout when that helps compare
  against the vendored tests or upstream source. These names are not
  OpenInfer public API.

When this tree is included in further redistributions, this NOTICE plus the
original MIT `LICENSE` must be carried alongside the code per the MIT terms.

## Provenance

The original `LICENSE` file is preserved verbatim at the root of this tree
(MIT, Copyright (c) 2025 Perplexity AI). Per the MIT terms, this notice and
the original copyright and permission notice must be retained in any
redistribution.
