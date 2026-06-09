//! Raw FFI bindings for libibverbs (RDMA Verbs).
//!
//! See `cuda-sys` for the same feature-gating contract.
#![allow(warnings)]

pub const SYSTEM_BINDINGS_ENABLED: bool = cfg!(feature = "system-bindings");

#[cfg(feature = "system-bindings")]
include!(concat!(env!("OUT_DIR"), "/libibverbs-bindings.rs"));
