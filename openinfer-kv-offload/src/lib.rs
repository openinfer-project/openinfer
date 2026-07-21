//! KV cache offload client between OpenInfer and an external PegaFlow server.
//!
//! openinfer owns the GPU paged-KV (`openinfer-kv-cache::KvBuffer`, page-first
//! layout) and the logical prefix cache (kvbm `BlockPool`). pegaflow owns the
//! deeper tiers (host pinned memory, SSD, RDMA). [`OffloadEngine`] registers
//! the GPU allocation over CUDA IPC and issues save/query/load RPCs.
//!
//! Dense-attention v1 (Qwen3-4B): the GPU prefix hit stays native to kvbm's
//! `BlockPool`; this connector covers the CPU tier and stacks a CPU-hit prefix
//! on top of the GPU-hit prefix (both anchor at prefix 0, so the combined hit
//! is one contiguous prefix split at a single point — GPU→CPU→GPU interleaving
//! is deliberately excluded). Save is best-effort fire-and-forget; load is on
//! the critical path, strongly ordered, but never blocks admission — a request
//! polls its [`LoadHandle`] each scheduler tick.

mod engine;
mod external;
mod vllm_hash;

pub use engine::{
    KvArena, LoadHandle, OffloadConfig, OffloadEngine, OffloadHost, QueryHit, QueryLeaseId,
    QueryOutcome,
};
pub use vllm_hash::{VLLM_HASH_BYTES, VllmBlockHasher};

pub use pegaflow_core::EngineError;
