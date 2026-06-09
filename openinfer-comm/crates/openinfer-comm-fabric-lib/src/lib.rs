//! RDMA Verbs fabric library (upstream-derived from `pplx-garden`).
//!
//! When the `hw-rdma` feature is enabled, this crate exposes the Verbs-based
//! transport implementation (`FabricEngine`, `Worker`, `TransferEngine`, etc.)
//! and pulls in `libibverbs-sys/system-bindings` + `cuda-lib/hw-cuda`.
//!
//! When the feature is disabled (the default), the crate compiles to a
//! near-empty shell with only the `HW_RDMA_ENABLED` diagnostic marker. This
//! crate is a hardware implementation layer, not a public abstract API.

/// Whether the `hw-rdma` feature is active in this build. Diagnostic only.
pub const HW_RDMA_ENABLED: bool = cfg!(feature = "hw-rdma");

#[cfg(feature = "hw-rdma")]
pub mod api;
#[cfg(feature = "hw-rdma")]
mod domain_group;
#[cfg(feature = "hw-rdma")]
mod error;
#[cfg(feature = "hw-rdma")]
mod fabric_engine;
#[cfg(feature = "hw-rdma")]
mod host_buffer;
#[cfg(feature = "hw-rdma")]
mod imm_count;
#[cfg(feature = "hw-rdma")]
mod interface;
#[cfg(feature = "hw-rdma")]
mod mr;
#[cfg(feature = "hw-rdma")]
mod provider;
#[cfg(feature = "hw-rdma")]
mod provider_dispatch;
#[cfg(feature = "hw-rdma")]
mod rdma_op;
#[cfg(feature = "hw-rdma")]
mod topo;
#[cfg(feature = "hw-rdma")]
mod transfer_engine;
#[cfg(feature = "hw-rdma")]
mod transfer_engine_builder;
#[cfg(feature = "hw-rdma")]
mod utils;
#[cfg(feature = "hw-rdma")]
mod verbs;
#[cfg(feature = "hw-rdma")]
mod worker;

#[cfg(feature = "hw-rdma")]
pub use domain_group::DomainGroup;
#[cfg(feature = "hw-rdma")]
pub use error::*;
#[cfg(feature = "hw-rdma")]
pub use fabric_engine::FabricEngine;
#[cfg(feature = "hw-rdma")]
pub use host_buffer::{HostBuffer, HostBufferAllocator};
#[cfg(feature = "hw-rdma")]
pub use interface::{
    AsyncTransferEngine, BouncingErrorCallback, BouncingRecvCallback, ErrorCallback,
    RdmaEngine, RecvCallback, SendBuffer, SendCallback, SendRecvEngine,
};
#[cfg(feature = "hw-rdma")]
pub use provider::{RdmaDomain, RdmaDomainInfo};
#[cfg(feature = "hw-rdma")]
pub use provider_dispatch::DomainInfo;
#[cfg(feature = "hw-rdma")]
pub use topo::{TopologyGroup, detect_topology};
#[cfg(feature = "hw-rdma")]
pub use transfer_engine::{
    ImmCountCallback, TransferCallback, TransferEngine, UvmWatcherCallback,
};
#[cfg(feature = "hw-rdma")]
pub use transfer_engine_builder::TransferEngineBuilder;
#[cfg(feature = "hw-rdma")]
pub use worker::{InitializingWorker, Worker, WorkerHandle};

#[cfg(feature = "hw-rdma")]
pub use interface::MockTestTransferEngine;
