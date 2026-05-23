//! Single-process intra-node bootstrap for the pplx-garden EP backend.
//!
//! Builds one `Vec<EpBackend>` (length = world_size) ready for use. Because
//! all ranks live in one process, this skips the per-process pickle/FD
//! rendezvous that the upstream Python bootstrap needs and instead shares
//! CUMem allocation handles by `Arc` clone, mapping each peer's send/recv/sync
//! buffer directly into every other rank's device VA.

mod pplx;

pub use pplx::{
    EpModelShape, PplxBootstrapParams, PplxRankResources, build_intra_node_backends,
    build_intra_node_backends_for_devices,
};
