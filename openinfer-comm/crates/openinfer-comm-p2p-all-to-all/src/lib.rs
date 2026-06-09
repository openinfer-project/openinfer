//! EP all-to-all upstream-derived crate (from `pplx-garden`).
//!
//! When the `hw-rdma` feature is enabled, the dispatch/combine context, rank
//! handles, and worker that drive the CUDA + RDMA Verbs all-to-all are built.
//! Default-off, this crate exposes only `HW_RDMA_ENABLED` diagnostic marker.

/// Whether the `hw-rdma` feature is active. Diagnostic only. (When on, this
/// crate transitively activates `hw-cuda` on `cuda-lib` / `a2a-kernels`,
/// since the all-to-all path needs both CUDA kernels and RDMA Verbs.)
pub const HW_RDMA_ENABLED: bool = cfg!(feature = "hw-rdma");

#[cfg(feature = "hw-rdma")]
mod a2a_context;
#[cfg(feature = "hw-rdma")]
mod a2a_handles;
#[cfg(feature = "hw-rdma")]
mod a2a_worker;

#[cfg(feature = "hw-rdma")]
pub use a2a_context::AllToAllContext;
#[cfg(feature = "hw-rdma")]
pub use a2a_handles::AllToAllRankHandle;
#[cfg(feature = "hw-rdma")]
pub use a2a_kernels::ScalarType;
