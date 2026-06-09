//! Raw FFI bindings for the CUDA driver API.
//!
//! When the sys-crate-internal `system-bindings` feature is enabled, bindgen
//! generates the full FFI surface against the system `cuda.h` and the crate
//! emits link directives for `libcuda`. When the feature is disabled (the
//! default), the crate is empty except for the `SYSTEM_BINDINGS_ENABLED`
//! diagnostic marker. No fake FFI types, no stub safe wrappers — code that
//! depends on the bindings must itself be gated behind a downstream `hw-cuda`
//! feature.
#![allow(warnings)]

/// Whether the `system-bindings` feature is active in this build. Diagnostic
/// only — not part of the FFI surface.
pub const SYSTEM_BINDINGS_ENABLED: bool = cfg!(feature = "system-bindings");

#[cfg(feature = "system-bindings")]
include!(concat!(env!("OUT_DIR"), "/cuda-bindings.rs"));

#[cfg(feature = "system-bindings")]
pub unsafe fn cuMemAlloc(dptr: *mut u64, bytesize: usize) -> CUresult {
    cuMemAlloc_v2(dptr, bytesize)
}

#[cfg(feature = "system-bindings")]
pub unsafe fn cuMemFree(dptr: u64) -> CUresult {
    cuMemFree_v2(dptr)
}
