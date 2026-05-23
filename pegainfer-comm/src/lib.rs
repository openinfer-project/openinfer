//! PegaInfer comm-backend public surface.
//!
//! [`EpBackend`] wraps the upstream `pplx-garden` NVLink + RDMA all-to-all
//! context with a thin Rust surface tailored for PegaInfer's MoE call sites:
//! `dispatch_send / dispatch_recv / combine_send / combine_recv`, kept
//! separate so callers can overlap host-side compute between send and recv.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

mod error;
pub use error::{Error, Result};

mod ep_backend;
pub use ep_backend::{
    EpBackend, EpBackendParams, EpDtypes, EpRankBuffers, EpTopology, ScalarType,
};

/// Single-process intra-node bootstrap for the pplx-garden EP backend.
pub mod bootstrap;

/// Re-exports of the underlying `pplx-garden` building blocks. Available
/// so PegaInfer-side bootstrap code can build `EpBackendParams` without
/// taking direct dependencies on the vendored crates.
pub mod raw {
    pub use cuda_lib;
    pub use fabric_lib;
    pub use p2p_all_to_all;
}
