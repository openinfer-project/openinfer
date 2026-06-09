//! Raw FFI bindings for GDRCopy (`gdrapi`).
//!
//! See `cuda-sys` for the same feature-gating contract.
#![allow(warnings)]

pub const SYSTEM_BINDINGS_ENABLED: bool = cfg!(feature = "system-bindings");

#[cfg(feature = "system-bindings")]
include!(concat!(env!("OUT_DIR"), "/gdrapi-bindings.rs"));
