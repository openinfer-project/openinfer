//! Raw FFI bindings for the CUDA runtime API.
//!
//! See `cuda-sys` for the same feature-gating contract.
#![allow(warnings)]

pub const SYSTEM_BINDINGS_ENABLED: bool = cfg!(feature = "system-bindings");

#[cfg(feature = "system-bindings")]
include!(concat!(env!("OUT_DIR"), "/cudart-bindings.rs"));
