//! CUDA wrapper crate (upstream-derived from `pplx-garden`).
//!
//! When the `hw-cuda` feature is enabled, this crate exposes the CUDA driver
//! and runtime wrappers (`driver`, `rt`, `event`, `gdr`, `mem`, `cumem`,
//! `device`) and re-exports the underlying `cuda-sys`, `cudart-sys`,
//! `gdrapi-sys` FFI crates.
//!
//! When the feature is disabled (the default), the crate compiles to a
//! near-empty shell that only exposes the `HW_CUDA_ENABLED` diagnostic
//! marker. This crate is a hardware implementation layer, not a public
//! abstract API; the OpenInfer-facing trait/plan/error/handle surface lives
//! in the (future) top-level `openinfer-comm` crate.
#![allow(non_snake_case)]

/// Whether the `hw-cuda` feature is active in this build. Diagnostic only.
pub const HW_CUDA_ENABLED: bool = cfg!(feature = "hw-cuda");

#[cfg(feature = "hw-cuda")]
pub use cuda_sys;
#[cfg(feature = "hw-cuda")]
pub use cudart_sys;
#[cfg(feature = "hw-cuda")]
pub use gdrapi_sys;

#[cfg(feature = "hw-cuda")]
pub mod driver;
#[cfg(feature = "hw-cuda")]
pub mod event;
#[cfg(feature = "hw-cuda")]
pub mod gdr;
#[cfg(feature = "hw-cuda")]
pub mod rt;

#[cfg(feature = "hw-cuda")]
pub mod cumem;
#[cfg(feature = "hw-cuda")]
mod error;
#[cfg(feature = "hw-cuda")]
mod mem;
#[cfg(feature = "hw-cuda")]
pub use error::{CudaError, CudaResult};
#[cfg(feature = "hw-cuda")]
pub use mem::{CudaDeviceMemory, CudaHostMemory};
#[cfg(feature = "hw-cuda")]
mod device;
#[cfg(feature = "hw-cuda")]
pub use device::{CudaDeviceId, Device};

#[cfg(all(test, feature = "hw-cuda"))]
mod test_driver;

#[cfg(all(test, feature = "hw-cuda"))]
mod test_gdr;
