//! PyTorch + CUDA glue (upstream-derived from `pplx-garden`).
//!
//! When the `hw-cuda` feature is enabled, this crate exposes the cxx bridge
//! into LibTorch (`from_blob`, `current_stream`, `torch_profile_range`,
//! `ScalarType`, ...) and pulls in `cuda-lib/hw-cuda` + LibTorch headers via
//! `build.rs`. When the feature is disabled (the default), the crate compiles
//! to a near-empty shell exposing only the `HW_CUDA_ENABLED` diagnostic
//! marker — no PyTorch / CUDA dependency is probed by `build.rs`.

/// Whether the `hw-cuda` feature is active in this build. Diagnostic only.
pub const HW_CUDA_ENABLED: bool = cfg!(feature = "hw-cuda");

#[cfg(feature = "hw-cuda")]
mod hw_cuda_impl;

#[cfg(feature = "hw-cuda")]
pub use hw_cuda_impl::{
    ScalarType, TorchProfilerGuard, current_stream, from_blob, torch_profile_range,
};
