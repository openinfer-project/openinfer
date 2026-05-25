//! CUDA all-to-all dispatch/combine kernels (upstream-derived from `pplx-garden`).
//!
//! When the `hw-cuda` feature is enabled, the `nvcc`-compiled kernels in
//! `src/a2a/*.cu` are linked in and the cxx bridge `ffi` module exposes the
//! kernel launch functions. When the feature is disabled (the default),
//! `build.rs` is a no-op and no kernel sources are touched; this crate
//! exports only `HW_CUDA_ENABLED`.

/// Whether the `hw-cuda` feature is active in this build. Diagnostic only.
pub const HW_CUDA_ENABLED: bool = cfg!(feature = "hw-cuda");

#[cfg(feature = "hw-cuda")]
mod hw_cuda_impl;

#[cfg(feature = "hw-cuda")]
pub use hw_cuda_impl::{
    ScalarType, a2a_combine_recv, a2a_combine_send, a2a_dispatch_recv,
    a2a_dispatch_recv_counts, a2a_dispatch_send, a2a_dispatch_send_route_only,
};
