//! In-process KV cache offload bridge between openinfer and pegaflow.
//!
//! openinfer owns the GPU paged-KV (`openinfer-kv-cache::KvBuffer`, page-first
//! layout) and the logical prefix cache (kvbm `BlockPool`). pegaflow owns the
//! deeper tiers (host pinned memory, SSD, RDMA). [`OffloadEngine`] is the
//! connector "brain" that moves blocks between them and decides when.
//!
//! Dense-attention v1 (Qwen3-4B): the GPU prefix hit stays native to kvbm's
//! `BlockPool`; this connector covers the CPU tier and stacks a CPU-hit prefix
//! on top of the GPU-hit prefix (both anchor at prefix 0, so the combined hit
//! is one contiguous prefix split at a single point — GPU→CPU→GPU interleaving
//! is deliberately excluded). Save is best-effort fire-and-forget; load is on
//! the critical path, strongly ordered, but never blocks admission — a request
//! polls its [`LoadHandle`] each scheduler tick.

mod engine;

pub use engine::{
    KvArena, LoadHandle, OffloadConfig, OffloadEngine, P2pConfig, QueryHit, QueryOutcome,
};

// Re-exported so callers name pegaflow's engine types through this bridge.
pub use pegaflow_core::{EngineError, PegaEngine, QueryLeaseId};
